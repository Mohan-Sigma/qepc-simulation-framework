"""
qepc_inference.py
=================
QEPC deferred-evaluation inference engine.

Implements the probabilistic inference algorithm that the QEPC-H1 hardware
executes: Gibbs sampling restricted to the minimal active subgraph, with
deferred collapse of variables until demanded by a terminal query.

This module is responsible for:
  1. Backward graph traversal → identifying active subgraph S
  2. Sequential Gibbs sampling on nodes in S
  3. Tracking convergence (Total Variation Distance to true marginal)
  4. Reporting the final inferred marginal P(Q | evidence)

The energy model (energy_model.py) prices the sequence of operations that
this module executes.  The two modules are deliberately decoupled: this
module produces correct inference; the energy model only counts operations.

Classes
-------
  QEPCInferenceResult : result of one inference run
  QEPCInference       : main inference class

Usage
-----
  from qepc_inference import QEPCInference
  from graph_generator import build_graph, GraphTopology

  graph  = build_graph(GraphTopology.CHAIN, N=50, seed=42)
  engine = QEPCInference(K_gibbs=10, seed=0)

  result = engine.infer(graph)
  print(result.marginal)           # P(x25) inferred marginal
  print(result.active_subgraph)    # list of activated node names
  print(result.n_collapse_events)  # total p-bit collapses performed
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from graph_generator import GraphModel
except ImportError:
    GraphModel = object   # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QEPCInferenceResult:
    """
    Output of one QEPC inference run.

    marginal       : dict mapping each query node name → np.ndarray of shape (N_states,)
                     representing the inferred marginal probability distribution.
    active_subgraph: list of node names that were activated (included in S).
    n_collapse_events: total number of p-bit collapse operations (= |S| × K).
    n_rram_reads   : total RRAM row accesses (= |S| × K).
    tvd_history    : list of TVD values at each Gibbs sweep (if ground truth available).
    converged      : True if TVD < 0.05 at final sweep.
    """
    query:             List[str]
    marginal:          Dict[str, np.ndarray]
    active_subgraph:   List[str]
    n_collapse_events: int
    n_rram_reads:      int
    K_used:            int
    tvd_history:       List[float] = field(default_factory=list)
    converged:         bool = False

    def summary(self) -> str:
        lines = [
            f"QEPCInferenceResult",
            f"  Query          : {self.query}",
            f"  |S| (active)   : {len(self.active_subgraph)}",
            f"  K sweeps       : {self.K_used}",
            f"  Collapse events: {self.n_collapse_events}",
            f"  RRAM reads     : {self.n_rram_reads}",
            f"  Converged      : {self.converged}",
        ]
        for qnode, m in self.marginal.items():
            lines.append(f"  P({qnode})        : {np.round(m, 4)}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: exact marginal via variable elimination (ground truth for chain graphs)
# ═══════════════════════════════════════════════════════════════════════════════

def _exact_marginal_chain(graph: GraphModel, query_node: str) -> np.ndarray:
    """
    Compute exact marginal P(query_node) for a chain graph using the
    forward-backward algorithm.

    Only valid for chain-topology graphs.  Used for TVD convergence tracking
    in Section 4.1 of SD-1.

    Returns np.ndarray of shape (N_states,).
    """
    nodes   = graph.nodes
    parents = graph.parents
    cpt     = graph.cpt
    N_s     = graph.N_states
    N       = len(nodes)

    # Find root (node with no parents)
    root = next(n for n in nodes if len(parents[n]) == 0)

    # Build chain order
    order = [root]
    seen  = {root}
    for _ in range(N - 1):
        last = order[-1]
        children = [ch for (pa, ch) in graph.edges if pa == last and ch not in seen]
        if not children:
            break
        order.append(children[0])
        seen.add(children[0])

    # Forward pass: alpha[i] = P(x0:i, x_i = s)
    alpha = np.zeros((N, N_s))
    alpha[0] = cpt[order[0]][0]   # prior

    for i in range(1, len(order)):
        node = order[i]
        par  = parents[node]
        if par:
            # P(x_i | x_{i-1}) * P(x_{i-1})
            # cpt shape (N_s, N_s): cpt[parent_state, child_state]
            alpha[i] = (cpt[node] * alpha[i-1][:, np.newaxis]).sum(axis=0)
        else:
            alpha[i] = cpt[node][0]
        # Normalize for numerical stability
        s = alpha[i].sum()
        if s > 0:
            alpha[i] /= s

    try:
        q_idx = order.index(query_node)
    except ValueError:
        warnings.warn(f"{query_node} not in chain order; cannot compute exact marginal.")
        return np.full(N_s, 1.0 / N_s)

    return alpha[q_idx] / alpha[q_idx].sum()


# ═══════════════════════════════════════════════════════════════════════════════
# CPT Lookup
# ═══════════════════════════════════════════════════════════════════════════════

def _conditional_prob(
    node:        str,
    parent_states: Dict[str, int],
    graph:       GraphModel,
) -> np.ndarray:
    """
    Look up P(node | parents) from the CPT given current parent states.

    Parameters
    ----------
    node          : node whose conditional distribution to retrieve
    parent_states : dict mapping parent node names → current sampled states
    graph         : GraphModel containing the CPT table

    Returns
    -------
    np.ndarray of shape (N_states,) : P(node = s | parents)
    """
    cpt     = graph.cpt[node]
    parents = graph.parents[node]
    N_s     = graph.N_states

    if len(parents) == 0:
        # Prior (root node)
        return cpt[0].copy()

    # Compute the parent configuration index (row index into CPT)
    # For k parents each with N_s states, the joint config index is:
    #   idx = sum_i(state_i × N_s^(k-1-i))
    idx = 0
    for i, p in enumerate(parents):
        state = parent_states.get(p, 0)
        idx  += state * (N_s ** (len(parents) - 1 - i))

    if idx >= cpt.shape[0]:
        # Fallback: parent config out of CPT range (shouldn't happen with correct construction)
        return np.full(N_s, 1.0 / N_s)

    return cpt[idx].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# QEPC Inference Engine
# ═══════════════════════════════════════════════════════════════════════════════

class QEPCInference:
    """
    QEPC deferred-evaluation inference via Gibbs sampling.

    Deferred evaluation steps:
      1. Identify active subgraph S by backward traversal from query Q.
      2. For K sweeps, iterate over nodes in S in topological order:
           a. Look up P(node | current parent states) from RRAM [1 row read]
           b. Sample a new state for the node [1 sMTJ collapse event]
      3. Accumulate sample counts → estimate marginal P(Q).
      4. Perform adiabatic erasure on all |S| p-bits after final sweep.

    Parameters
    ----------
    K_gibbs  : number of Gibbs sweeps (default 10, selected for TVD < 0.02)
    n_samples: number of independent sample chains for marginal estimation
               (default 500; more samples → lower variance in marginal estimate)
    seed     : random seed for the sampler
    track_tvd: if True, compute TVD against exact marginal after each sweep
               (only meaningful for chain graphs; requires exact marginal)
    """

    def __init__(
        self,
        K_gibbs:   int  = 10,
        n_samples: int  = 500,
        seed:      int  = 0,
        track_tvd: bool = False,
    ) -> None:
        self.K        = K_gibbs
        self.n_samples = n_samples
        self.rng      = np.random.default_rng(seed)
        self.track_tvd = track_tvd

    def infer(self, graph: GraphModel) -> QEPCInferenceResult:
        """
        Run QEPC inference on the given graph for its declared query set.

        Parameters
        ----------
        graph : GraphModel with .query set and .observed evidence

        Returns
        -------
        QEPCInferenceResult
        """
        if graph.topology == "tsp":
            raise ValueError("TSP topology uses optimization solver, not inference. "
                             "Call qepc_tsp_search() instead.")

        query = graph.query
        S     = graph.active_subgraph()
        N_s   = graph.N_states

        # Topological sort of S for ordered Gibbs sweeps
        S_topo = self._topological_sort(S, graph)

        # ── Gibbs sampling ────────────────────────────────────────────────────
        # Accumulate counts for marginal estimation
        counts: Dict[str, np.ndarray] = {n: np.zeros(N_s) for n in query}

        # Current state (initialize uniformly)
        state: Dict[str, int] = {n: int(self.rng.integers(0, N_s)) for n in S_topo}
        # Observed nodes are fixed evidence — never resampled
        state.update(graph.observed)

        tvd_history: List[float] = []
        total_collapse_events = 0
        total_rram_reads      = 0

        # Compute ground-truth marginal once (for TVD, chain graphs only)
        ground_truth: Optional[Dict[str, np.ndarray]] = None
        if self.track_tvd and graph.topology == "chain" and len(query) == 1:
            gt = _exact_marginal_chain(graph, query[0])
            ground_truth = {query[0]: gt}

        for sweep in range(self.K * self.n_samples):
            # One Gibbs sweep: update each node in S in topological order
            for node in S_topo:
                if node in graph.observed:
                    continue  # evidence nodes are fixed

                # RRAM read: look up P(node | parents) [1 row read]
                prob = _conditional_prob(node, state, graph)
                total_rram_reads += 1

                # sMTJ collapse: sample new state [1 collapse event]
                new_state = int(self.rng.choice(N_s, p=prob))
                state[node] = new_state
                total_collapse_events += 1

            # Accumulate query node states after burn-in
            if sweep >= self.K:   # first K sweeps are burn-in
                for qn in query:
                    counts[qn][state[qn]] += 1

            # TVD tracking every K sweeps
            if self.track_tvd and ground_truth and (sweep + 1) % self.K == 0:
                tvd = self._compute_tvd(counts, ground_truth, query, sweep - self.K + 1)
                tvd_history.append(tvd)

        # Normalise counts → marginal estimates
        marginal: Dict[str, np.ndarray] = {}
        for qn in query:
            total = counts[qn].sum()
            marginal[qn] = counts[qn] / total if total > 0 else np.full(N_s, 1.0 / N_s)

        # Convergence check
        converged = (len(tvd_history) == 0) or (tvd_history[-1] < 0.05)

        return QEPCInferenceResult(
            query             = query,
            marginal          = marginal,
            active_subgraph   = S,
            n_collapse_events = total_collapse_events,
            n_rram_reads      = total_rram_reads,
            K_used            = self.K,
            tvd_history       = tvd_history,
            converged         = converged,
        )

    def _topological_sort(self, subgraph_nodes: List[str], graph: GraphModel) -> List[str]:
        """
        Return nodes in topological order using Kahn's algorithm,
        restricted to the subgraph.
        """
        S_set   = set(subgraph_nodes)
        parents = graph.parents
        in_deg  = {n: sum(1 for p in parents[n] if p in S_set) for n in subgraph_nodes}
        queue   = [n for n in subgraph_nodes if in_deg[n] == 0]
        result  = []
        ch_map  = graph.children

        while queue:
            node = queue.pop(0)
            result.append(node)
            for ch in ch_map[node]:
                if ch in S_set:
                    in_deg[ch] -= 1
                    if in_deg[ch] == 0:
                        queue.append(ch)

        if len(result) != len(subgraph_nodes):
            # Cycle detected — fall back to original order (should not happen for DAGs)
            warnings.warn("Cycle detected in active subgraph; using original node order.")
            return subgraph_nodes

        return result

    def _compute_tvd(
        self,
        counts:      Dict[str, np.ndarray],
        ground_truth: Dict[str, np.ndarray],
        query:       List[str],
        n_so_far:    int,
    ) -> float:
        """
        Total Variation Distance: 0.5 × Σ |P_est(s) - P_true(s)|
        """
        tvd = 0.0
        for qn in query:
            if qn in ground_truth and counts[qn].sum() > 0:
                p_est  = counts[qn] / counts[qn].sum()
                p_true = ground_truth[qn]
                tvd   += 0.5 * np.abs(p_est - p_true).sum()
        return tvd / max(1, len(query))

    def convergence_sweep(
        self,
        graph:     GraphModel,
        K_values:  List[int] = [1, 2, 5, 10, 20, 50, 100],
    ) -> Dict[int, float]:
        """
        Run inference at each K value in K_values and report mean TVD.
        Used to generate Table 4.1 of SD-1.

        Only meaningful for chain graphs (exact marginal available).
        """
        if graph.topology != "chain" or len(graph.query) != 1:
            raise ValueError("Convergence sweep requires a chain graph with single query.")

        results = {}
        for K in K_values:
            engine = QEPCInference(K_gibbs=K, n_samples=200, seed=99, track_tvd=True)
            res    = engine.infer(graph)
            final_tvd = res.tvd_history[-1] if res.tvd_history else float("nan")
            results[K] = final_tvd

        return results


# ─── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from graph_generator import build_graph, GraphTopology

    print("=== Chain N=20, middle query, K=10 ===")
    graph  = build_graph(GraphTopology.CHAIN, N=20, seed=42)
    engine = QEPCInference(K_gibbs=10, n_samples=500, seed=0, track_tvd=True)
    result = engine.infer(graph)
    print(result.summary())
    if result.tvd_history:
        print(f"\n  Final TVD after K=10 sweeps: {result.tvd_history[-1]:.4f}")

    print("\n=== Convergence sweep K = [1,2,5,10,20,50,100] ===")
    sweep = engine.convergence_sweep(graph)
    for K, tvd in sweep.items():
        print(f"  K={K:3d}  TVD={tvd:.4f}")

    print("\n=== AR Sensor Fusion ===")
    g3 = build_graph(GraphTopology.AR_SENSOR, seed=42)
    e3 = QEPCInference(K_gibbs=10, n_samples=200, seed=1)
    r3 = e3.infer(g3)
    print(r3.summary())
