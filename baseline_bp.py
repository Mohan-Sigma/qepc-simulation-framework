"""
baseline_bp.py
==============
Baseline inference algorithms for comparison against QEPC.

Two baselines:
  1. TargetedBP   : Sum-product belief propagation restricted to the
                    query-relevant active subgraph S.  Used for all
                    PGM inference benchmarks (Tables IV in the article).

  2. SimulatedAnnealing : Standard SA for the TSP optimization benchmark.
                    Used for the TSP negative result in Table IV (last row).

Design choice: TargetedBP uses the SAME hardware energy parameters as QEPC
(see energy_model.py).  Restricting BP to the same subgraph S isolates the
energy savings attributable to deferred collapse vs. eager message-passing,
rather than to subgraph selection.  This makes it an intentionally favorable
(conservative) baseline — exactly as described in SD-1, Section 5.

Classes
-------
  BPResult     : result of one belief propagation run
  TargetedBP   : sum-product BP inference engine
  SAResult     : result of one simulated annealing run
  SimulatedAnnealing : TSP optimizer
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
# Belief Propagation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BPResult:
    """
    Output of one targeted belief propagation run.
    """
    query:             List[str]
    marginal:          Dict[str, np.ndarray]
    active_subgraph:   List[str]
    n_rram_reads:      int     # total RRAM row accesses (N_states per node per round)
    n_sram_writes:     int     # total SRAM marginal write operations
    K_used:            int
    converged:         bool
    residuals:         List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "BPResult [TargetedBP]",
            f"  Query       : {self.query}",
            f"  |S|         : {len(self.active_subgraph)}",
            f"  K rounds    : {self.K_used}",
            f"  RRAM reads  : {self.n_rram_reads}",
            f"  SRAM writes : {self.n_sram_writes}",
            f"  Converged   : {self.converged}",
        ]
        for qn, m in self.marginal.items():
            lines.append(f"  P({qn})    : {np.round(m, 4)}")
        return "\n".join(lines)


class TargetedBP:
    """
    Sum-product (loopy) belief propagation restricted to the query-relevant
    active subgraph S of the given graph.

    Algorithm
    ---------
    1. Initialize all messages µ_{i→j}(x_j) = uniform distribution.
    2. For K rounds:
         For each node i in S (in topological order):
           For each neighbor j of i in S:
             Compute outgoing message:
               µ_{i→j}(x_j) = Σ_{x_i} P(x_i | pa(i)) × Π_{k ≠ j} µ_{k→i}(x_i)
               (RRAM CPT lookup: N_states rows = full CPT matrix)
             Write result to SRAM (1 write per neighbor)
    3. Compute beliefs at query nodes:
         b(x_i) ∝ P(x_i | pa(i)) × Π_k µ_{k→i}(x_i)

    Convergence
    -----------
    Halts when max message residual < tol (default 1e-4) or K rounds reached.
    For loopy graphs, convergence is not guaranteed; K is treated as an upper
    bound matching the QEPC's K_gibbs for fair energy comparison.

    Parameters
    ----------
    K_rounds : number of message-passing rounds (must equal QEPC K_gibbs for
               fair energy comparison)
    tol      : convergence tolerance on max message change
    """

    def __init__(self, K_rounds: int = 10, tol: float = 1e-4) -> None:
        self.K   = K_rounds
        self.tol = tol

    def infer(self, graph: GraphModel) -> BPResult:
        """
        Run targeted BP on graph.query over the active subgraph.
        """
        if graph.topology == "tsp":
            raise ValueError("Use SimulatedAnnealing for TSP topology.")

        query = graph.query
        S     = graph.active_subgraph()
        N_s   = graph.N_states
        S_set = set(S)

        # Topological sort
        S_topo = self._topological_sort(S, graph)
        parents  = graph.parents
        children = graph.children

        # ── Message initialization ────────────────────────────────────────────
        # µ[(i,j)] = message from node i to node j, shape (N_s,)
        messages: Dict[Tuple[str, str], np.ndarray] = {}
        for n in S:
            for p in parents[n]:
                if p in S_set:
                    messages[(p, n)] = np.full(N_s, 1.0 / N_s)
            for c in children[n]:
                if c in S_set:
                    messages[(n, c)] = np.full(N_s, 1.0 / N_s)

        n_rram_reads  = 0
        n_sram_writes = 0
        residuals     = []

        # ── Message passing rounds ────────────────────────────────────────────
        for rnd in range(self.K):
            old_messages = {k: v.copy() for k, v in messages.items()}
            max_residual = 0.0

            for node in S_topo:
                if node in graph.observed:
                    # Evidence node: send delta message (certain state)
                    obs_state = graph.observed[node]
                    msg_out   = np.zeros(N_s)
                    msg_out[obs_state] = 1.0
                    for c in children[node]:
                        if c in S_set:
                            messages[(node, c)] = msg_out.copy()
                    continue

                # Collect incoming messages from all neighbors in S
                incoming: Dict[str, np.ndarray] = {}
                for p in parents[node]:
                    if p in S_set:
                        incoming[p] = messages.get((p, node), np.full(N_s, 1.0/N_s))
                for c in children[node]:
                    if c in S_set:
                        incoming[c] = messages.get((c, node), np.full(N_s, 1.0/N_s))

                # ── Send messages to children (factor → variable) ─────────────
                for c in children[node]:
                    if c not in S_set:
                        continue

                    # Read FULL CPT matrix (N_states rows) from RRAM
                    # — this is the key operation count difference vs. QEPC
                    cpt_node_c = graph.cpt[c]   # shape (N_s^n_parents, N_s)
                    n_rram_reads += N_s          # N_states rows per CPT lookup

                    # Product of incoming messages (exclude message from c)
                    inprod = np.ones(N_s)
                    for src, msg in incoming.items():
                        if src != c:
                            inprod *= msg

                    # Sum over node states weighted by cpt and incoming products
                    # µ_{node→c}(x_c) = Σ_{x_node} P(x_node | pa) × inprod(x_node)
                    # Simplified: assume node's parents are captured in prior
                    prior = graph.cpt[node][0] if len(parents[node]) == 0 else \
                            self._marginal_from_cpt(graph, node, {})

                    new_msg = np.zeros(N_s)
                    for s_node in range(N_s):
                        # Get P(c = x_c | node = s_node) averaged over parent configs
                        # For simplicity: use the CPT row corresponding to s_node only
                        # (single-parent case; for multi-parent, this is an approximation)
                        row_idx = s_node % cpt_node_c.shape[0]
                        new_msg += prior[s_node] * inprod[s_node] * cpt_node_c[row_idx]

                    s = new_msg.sum()
                    new_msg = new_msg / s if s > 0 else np.full(N_s, 1.0/N_s)
                    messages[(node, c)] = new_msg
                    n_sram_writes += 1   # write updated message to SRAM

                    # Track residual
                    old = old_messages.get((node, c), np.full(N_s, 1.0/N_s))
                    max_residual = max(max_residual, np.abs(new_msg - old).max())

                # ── Send messages to parents (variable → factor) ──────────────
                for p in parents[node]:
                    if p not in S_set:
                        continue

                    # Product of incoming messages from children and other parents (exclude p)
                    inprod = np.ones(N_s)
                    for src, msg in incoming.items():
                        if src != p:
                            inprod *= msg

                    # P(x_node | evidence from children)
                    new_msg = inprod.copy()
                    s = new_msg.sum()
                    new_msg = new_msg / s if s > 0 else np.full(N_s, 1.0/N_s)
                    messages[(node, p)] = new_msg
                    n_sram_writes += 1

                    old = old_messages.get((node, p), np.full(N_s, 1.0/N_s))
                    max_residual = max(max_residual, np.abs(new_msg - old).max())

            residuals.append(max_residual)

        # ── Compute beliefs at query nodes ────────────────────────────────────
        marginal: Dict[str, np.ndarray] = {}
        for qn in query:
            if qn not in S_set:
                marginal[qn] = np.full(N_s, 1.0 / N_s)
                continue

            # Belief = prior × product of all incoming messages
            prior = graph.cpt[qn][0] if len(parents[qn]) == 0 else \
                    self._marginal_from_cpt(graph, qn, {})

            belief = prior.copy()
            for p in parents[qn]:
                if p in S_set:
                    belief *= messages.get((p, qn), np.full(N_s, 1.0/N_s))
            for c in children[qn]:
                if c in S_set:
                    belief *= messages.get((c, qn), np.full(N_s, 1.0/N_s))

            s = belief.sum()
            marginal[qn] = belief / s if s > 0 else np.full(N_s, 1.0/N_s)

        converged = len(residuals) > 0 and residuals[-1] < self.tol

        return BPResult(
            query           = query,
            marginal        = marginal,
            active_subgraph = S,
            n_rram_reads    = n_rram_reads,
            n_sram_writes   = n_sram_writes,
            K_used          = self.K,
            converged       = converged,
            residuals       = residuals,
        )

    def _topological_sort(self, subgraph_nodes: List[str], graph: GraphModel) -> List[str]:
        S_set  = set(subgraph_nodes)
        in_deg = {n: sum(1 for p in graph.parents[n] if p in S_set) for n in subgraph_nodes}
        queue  = [n for n in subgraph_nodes if in_deg[n] == 0]
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for ch in graph.children[node]:
                if ch in S_set:
                    in_deg[ch] -= 1
                    if in_deg[ch] == 0:
                        queue.append(ch)
        return result if len(result) == len(subgraph_nodes) else subgraph_nodes

    def _marginal_from_cpt(self, graph: GraphModel, node: str, evidence: Dict) -> np.ndarray:
        """Simple prior approximation: average CPT rows for unknown parents."""
        cpt = graph.cpt[node]
        return cpt.mean(axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# TSP Simulated Annealing Baseline
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SAResult:
    """Result of one simulated annealing TSP run."""
    N:              int
    best_tour:      List[int]
    best_length:    float
    initial_length: float
    optimal_length: Optional[float]
    gap_pct:        Optional[float]          # % above optimal (None if optimal unknown)
    n_iterations:   int
    n_accepted:     int
    temp_history:   List[float] = field(default_factory=list)
    length_history: List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "SAResult [SimulatedAnnealing / TSP]",
            f"  N cities     : {self.N}",
            f"  Best length  : {self.best_length:.4f}",
            f"  Initial      : {self.initial_length:.4f}",
        ]
        if self.optimal_length is not None:
            lines.append(f"  Optimal      : {self.optimal_length:.4f}")
            lines.append(f"  Gap          : {self.gap_pct:.1f}%")
        lines.append(f"  Iterations   : {self.n_iterations:,}")
        lines.append(f"  Accepted     : {self.n_accepted:,}")
        return "\n".join(lines)


class SimulatedAnnealing:
    """
    Standard simulated annealing for the TSP.

    Used as the baseline for the optimization negative result in Table IV.

    Algorithm
    ---------
    1. Start with a random tour.
    2. For n_iter iterations:
         a. Propose a random 2-opt swap (reverse a sub-segment of the tour).
         b. Compute ΔL = new_length - current_length.
         c. Accept if ΔL < 0, or with probability exp(-ΔL / T) otherwise.
         d. Cool: T ← T × alpha.
    3. Return the best tour found.

    Parameters
    ----------
    n_iter   : total iterations (default 50,000 — equivalent computational
               budget to QEPC erasure search for N=10 TSP)
    T0       : initial temperature (default 100.0)
    alpha    : cooling rate (default 0.995)
    seed     : random seed
    """

    def __init__(
        self,
        n_iter: int   = 50_000,
        T0:     float = 100.0,
        alpha:  float = 0.995,
        seed:   int   = 0,
    ) -> None:
        self.n_iter = n_iter
        self.T0     = T0
        self.alpha  = alpha
        self.rng    = np.random.default_rng(seed)

    def solve(self, graph: GraphModel) -> SAResult:
        """
        Run SA on a TSP GraphModel.
        """
        if graph.topology != "tsp":
            raise ValueError("SimulatedAnnealing requires TSP topology.")

        D   = graph.meta["cost_matrix"]
        N   = graph.meta["N"]
        opt = graph.meta.get("optimal_length")

        # Random initial tour
        tour = list(range(N))
        self.rng.shuffle(tour)
        current_length = self._tour_length(tour, D)
        initial_length = current_length
        best_tour      = tour[:]
        best_length    = current_length

        T = self.T0
        n_accepted     = 0
        temp_history   = []
        length_history = []

        for it in range(self.n_iter):
            # Propose 2-opt swap
            i, j = sorted(self.rng.choice(N, size=2, replace=False))
            new_tour   = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
            new_length = self._tour_length(new_tour, D)

            delta = new_length - current_length
            if delta < 0 or self.rng.random() < np.exp(-delta / T):
                tour           = new_tour
                current_length = new_length
                n_accepted    += 1
                if current_length < best_length:
                    best_tour   = tour[:]
                    best_length = current_length

            T *= self.alpha

            if it % (self.n_iter // 200) == 0:
                temp_history.append(T)
                length_history.append(best_length)

        gap = ((best_length - opt) / opt * 100.0) if opt else None

        return SAResult(
            N              = N,
            best_tour      = best_tour,
            best_length    = best_length,
            initial_length = initial_length,
            optimal_length = opt,
            gap_pct        = gap,
            n_iterations   = self.n_iter,
            n_accepted     = n_accepted,
            temp_history   = temp_history,
            length_history = length_history,
        )

    def _tour_length(self, tour: List[int], D: np.ndarray) -> float:
        return sum(D[tour[k], tour[(k + 1) % len(tour)]] for k in range(len(tour)))


# ─── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from graph_generator import build_graph, GraphTopology

    print("=== TargetedBP on Chain N=20 ===")
    graph = build_graph(GraphTopology.CHAIN, N=20, seed=42)
    bp    = TargetedBP(K_rounds=10)
    res   = bp.infer(graph)
    print(res.summary())
    print(f"  Residuals: {[f'{r:.5f}' for r in res.residuals]}")

    print("\n=== SA on TSP N=10 ===")
    tsp   = build_graph(GraphTopology.TSP, N=10, seed=42)
    sa    = SimulatedAnnealing(n_iter=50_000, seed=7)
    tspres = sa.solve(tsp)
    print(tspres.summary())
