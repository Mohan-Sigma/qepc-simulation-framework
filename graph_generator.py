"""
graph_generator.py
==================
Constructs probabilistic graphical model (PGM) topologies used in the
QEPC simulation benchmarks.

Supported topologies
--------------------
  - chain          : Linear Bayesian network (N nodes, sequential dependencies)
  - erdos_renyi    : Dense random DAG (Erdős–Rényi, edge probability p)
  - ar_sensor_fusion : AR-glasses sensor fusion (4 observed + 8 latent nodes)
  - tsp            : Travelling Salesman Problem cost graph (N cities)

All graphs are returned as a GraphModel dataclass containing:
  - nodes     : list of node names
  - edges     : list of (parent, child) tuples
  - cpt       : dict mapping node name → CPT array  shape (N_states^N_parents, N_states)
  - observed  : dict mapping observed node name → observed state index (or None)
  - query     : list of node names to infer (the query set Q)
  - N_states  : number of discrete states per node (default 4)

Usage
-----
  from graph_generator import build_graph, GraphTopology

  graph = build_graph(GraphTopology.CHAIN, N=50, seed=42)
  print(graph.nodes)          # ['x0', 'x1', ..., 'x49']
  print(graph.query)          # ['x25']
  print(graph.cpt['x1'].shape)  # (4, 4)  → P(x1 | x0)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx


# ─── Types ───────────────────────────────────────────────────────────────────

class GraphTopology(enum.Enum):
    CHAIN          = "chain"
    ERDOS_RENYI    = "erdos_renyi"
    AR_SENSOR      = "ar_sensor_fusion"
    TSP            = "tsp"


@dataclass
class GraphModel:
    """Complete specification of a PGM instance."""
    topology:  str
    nodes:     List[str]
    edges:     List[Tuple[str, str]]
    cpt:       Dict[str, np.ndarray]
    observed:  Dict[str, int]          # node_name → observed state (leaf evidence)
    query:     List[str]               # which nodes to infer
    N_states:  int = 4
    meta:      Dict = field(default_factory=dict)   # topology-specific extras

    # Derived convenience properties
    @property
    def parents(self) -> Dict[str, List[str]]:
        """Return mapping node → list of parents."""
        p: Dict[str, List[str]] = {n: [] for n in self.nodes}
        for (pa, ch) in self.edges:
            p[ch].append(pa)
        return p

    @property
    def children(self) -> Dict[str, List[str]]:
        """Return mapping node → list of children."""
        c: Dict[str, List[str]] = {n: [] for n in self.nodes}
        for (pa, ch) in self.edges:
            c[pa].append(ch)
        return c

    def active_subgraph(self) -> List[str]:
        """
        Return the minimal active subgraph S: all ancestor nodes of the
        query set, reached by backward (parent) traversal from query nodes.
        Observed nodes are included as evidence but are not 'collapsed'.
        """
        visited = set()
        stack = list(self.query)
        par = self.parents
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for p in par[node]:
                if p not in visited:
                    stack.append(p)
        return [n for n in self.nodes if n in visited]


# ─── CPT Utilities ───────────────────────────────────────────────────────────

def _dirichlet_cpt(rng: np.random.Generator,
                   n_parents: int,
                   N_states: int,
                   alpha: float = 2.0) -> np.ndarray:
    """
    Draw a CPT from a symmetric Dirichlet distribution.

    Parameters
    ----------
    n_parents : number of parent nodes
    N_states  : number of discrete states per variable
    alpha     : Dirichlet concentration parameter (2.0 gives non-uniform
                but not degenerate distributions; higher = more uniform)

    Returns
    -------
    ndarray of shape (N_states**n_parents, N_states)
        Each row is a valid probability distribution over child states
        conditioned on one joint parent configuration.
    """
    n_parent_configs = N_states ** n_parents
    cpt = np.zeros((n_parent_configs, N_states))
    alpha_vec = np.full(N_states, alpha)
    for i in range(n_parent_configs):
        cpt[i] = rng.dirichlet(alpha_vec)
    return cpt


def _prior_cpt(rng: np.random.Generator, N_states: int, alpha: float = 2.0) -> np.ndarray:
    """
    Draw a prior distribution (no parents) — shape (1, N_states).
    """
    return rng.dirichlet(np.full(N_states, alpha)).reshape(1, N_states)


# ─── Chain Graph ─────────────────────────────────────────────────────────────

def _build_chain(N: int, N_states: int, seed: int, query_position: str = "middle") -> GraphModel:
    """
    Build a directed chain graph x0 → x1 → ... → x_{N-1}.

    Query
    -----
    query_position : 'middle'  → query = {x_{N//2}}   (default, SD-1 benchmark)
                     'local'   → query = {x_{N//10}}  (short traversal, best-case)
                     'deep'    → query = {x_{N-2}}    (near-full traversal, worst-case)
                     'full'    → query = {x_{N-1}}    (full chain)
    """
    rng   = np.random.default_rng(seed)
    nodes = [f"x{i}" for i in range(N)]
    edges = [(f"x{i}", f"x{i+1}") for i in range(N - 1)]

    cpt: Dict[str, np.ndarray] = {}
    # Root node x0 has no parents → prior
    cpt["x0"] = _prior_cpt(rng, N_states)
    for i in range(1, N):
        # Each node has exactly 1 parent → CPT shape (N_states, N_states)
        cpt[f"x{i}"] = _dirichlet_cpt(rng, n_parents=1, N_states=N_states)

    if query_position == "middle":
        query_idx = N // 2
    elif query_position == "local":
        query_idx = max(1, N // 10)
    elif query_position == "deep":
        query_idx = N - 2
    elif query_position == "full":
        query_idx = N - 1
    else:
        raise ValueError(f"Unknown query_position: {query_position}")

    return GraphModel(
        topology  = "chain",
        nodes     = nodes,
        edges     = edges,
        cpt       = cpt,
        observed  = {},
        query     = [f"x{query_idx}"],
        N_states  = N_states,
        meta      = {"N": N, "query_position": query_position, "query_idx": query_idx},
    )


# ─── Erdős–Rényi Random DAG ───────────────────────────────────────────────────

def _build_erdos_renyi(N: int, p: float, N_states: int, seed: int) -> GraphModel:
    """
    Build a random DAG from the Erdős–Rényi G(N, p) model.

    Construction
    ------------
    1. Generate a random undirected ER graph.
    2. Orient each edge from lower-index to higher-index node (guaranteed DAG).
    3. Remove any back-edges that would create cycles (none in this scheme since
       we use index ordering).
    4. Draw CPTs from Dirichlet(alpha=2) for each node given its parents.

    Query
    -----
    A single random non-root node (chosen by seed).
    """
    rng = np.random.default_rng(seed)
    G   = nx.erdos_renyi_graph(N, p, seed=int(rng.integers(1e9)), directed=False)

    nodes = [f"x{i}" for i in range(N)]

    # Orient edges lower-index → higher-index to guarantee DAG
    edges: List[Tuple[str, str]] = []
    for (u, v) in G.edges():
        if u < v:
            edges.append((f"x{u}", f"x{v}"))
        else:
            edges.append((f"x{v}", f"x{u}"))

    # Build parent map
    parents: Dict[str, List[str]] = {n: [] for n in nodes}
    for (pa, ch) in edges:
        parents[ch].append(pa)

    # Draw CPTs
    cpt: Dict[str, np.ndarray] = {}
    for node in nodes:
        n_par = len(parents[node])
        if n_par == 0:
            cpt[node] = _prior_cpt(rng, N_states)
        else:
            cpt[node] = _dirichlet_cpt(rng, n_par, N_states)

    # Query: a randomly chosen non-root node
    non_roots = [n for n in nodes if len(parents[n]) > 0]
    query_node = rng.choice(non_roots)

    return GraphModel(
        topology  = "erdos_renyi",
        nodes     = nodes,
        edges     = edges,
        cpt       = cpt,
        observed  = {},
        query     = [query_node],
        N_states  = N_states,
        meta      = {"N": N, "p": p},
    )


# ─── AR Sensor Fusion ─────────────────────────────────────────────────────────

def _build_ar_sensor_fusion(N_states: int, seed: int) -> GraphModel:
    """
    AR glasses sensor fusion graph.

    Structure
    ---------
    Physical sensor nodes (observed, leaves):
        s_imu, s_cam, s_eye, s_light

    Latent state nodes (hidden, to be inferred):
        z1 (head pose)        : parents = [s_imu, z2]
        z2 (motion state)     : parents = [s_imu]
        z3 (scene content)    : parents = [s_cam, z4]
        z4 (visual saliency)  : parents = [s_cam]
        z5 (gaze direction)   : parents = [s_eye, z6]
        z6 (attention)        : parents = [s_eye, z3]
        z7 (ambient context)  : parents = [s_light]
        z8 (user intent)      : parents = [z1, z5, z7]

    Query
    -----
    Q = {z8} : "Is the user looking at a screen with intent to interact?"
    This is a deep query requiring backward traversal through z1, z5, z7
    and their sensor parents — activating 11 of 12 nodes.

    Alternatively, Q = {z1} activates only 3 nodes (local query, ~7× savings).
    The benchmark in SD-1 uses Q = {z1}.
    """
    rng = np.random.default_rng(seed)

    sensor_nodes = ["s_imu", "s_cam", "s_eye", "s_light"]
    latent_nodes = ["z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8"]
    nodes = sensor_nodes + latent_nodes

    edges = [
        # Sensor → latent
        ("s_imu",   "z1"),
        ("s_imu",   "z2"),
        ("s_cam",   "z3"),
        ("s_cam",   "z4"),
        ("s_eye",   "z5"),
        ("s_eye",   "z6"),
        ("s_light", "z7"),
        # Latent → latent
        ("z2",      "z1"),
        ("z4",      "z3"),
        ("z6",      "z5"),
        ("z3",      "z6"),
        ("z1",      "z8"),
        ("z5",      "z8"),
        ("z7",      "z8"),
    ]

    # Build parent map
    parents: Dict[str, List[str]] = {n: [] for n in nodes}
    for (pa, ch) in edges:
        parents[ch].append(pa)

    # Draw CPTs
    cpt: Dict[str, np.ndarray] = {}
    for node in nodes:
        n_par = len(parents[node])
        if n_par == 0:
            cpt[node] = _prior_cpt(rng, N_states)
        else:
            cpt[node] = _dirichlet_cpt(rng, n_par, N_states)

    # Observed sensors: fix each to a random observed state
    obs_state = {s: int(rng.integers(0, N_states)) for s in sensor_nodes}

    return GraphModel(
        topology  = "ar_sensor_fusion",
        nodes     = nodes,
        edges     = edges,
        cpt       = cpt,
        observed  = obs_state,
        query     = ["z1"],     # local query → ~7× energy savings (SD-1 Table IV)
        N_states  = N_states,
        meta      = {"sensor_nodes": sensor_nodes, "latent_nodes": latent_nodes},
    )


# ─── TSP Graph ────────────────────────────────────────────────────────────────

def _build_tsp(N: int, seed: int) -> GraphModel:
    """
    Travelling Salesman Problem instance with N cities.

    The 'graph' here is a complete cost matrix rather than a Bayesian network.
    City positions are drawn uniformly from [0, 1)^2. The cost matrix D[i][j]
    is the Euclidean distance between city i and city j.

    This topology is used only for the optimization benchmark (Section 4.4 of
    SD-1), not for probabilistic inference. The GraphModel.cpt field holds the
    cost matrix in a compatible format; inference modules should check
    graph.topology == 'tsp' and use the optimization solver instead.

    The GraphModel.meta['cost_matrix'] contains the N×N distance matrix.
    The GraphModel.meta['optimal_tour_length'] is computed by brute force
    for N <= 10 (feasible), otherwise set to None.
    """
    rng       = np.random.default_rng(seed)
    positions = rng.random((N, 2))

    # Euclidean distance matrix
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.sqrt(((positions[i] - positions[j]) ** 2).sum())

    nodes = [f"c{i}" for i in range(N)]
    # No directed edges in TSP — represent as fully connected undirected
    edges = [(f"c{i}", f"c{j}") for i in range(N) for j in range(i + 1, N)]

    # Brute-force optimal tour for N <= 10
    optimal_length = None
    if N <= 10:
        from itertools import permutations
        best = float("inf")
        for perm in permutations(range(1, N)):
            tour = [0] + list(perm) + [0]
            length = sum(D[tour[k], tour[k + 1]] for k in range(len(tour) - 1))
            if length < best:
                best = length
        optimal_length = best

    return GraphModel(
        topology  = "tsp",
        nodes     = nodes,
        edges     = edges,
        cpt       = {},          # not used for TSP
        observed  = {},
        query     = [],          # not applicable for TSP
        N_states  = N,           # repurpose: number of cities
        meta      = {
            "N":               N,
            "positions":       positions,
            "cost_matrix":     D,
            "optimal_length":  optimal_length,
        },
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def build_graph(
    topology:       GraphTopology,
    N:              int   = 50,
    N_states:       int   = 4,
    seed:           int   = 42,
    er_p:           float = 0.3,
    chain_query:    str   = "middle",
) -> GraphModel:
    """
    Build and return a GraphModel for the requested topology.

    Parameters
    ----------
    topology     : GraphTopology enum value
    N            : number of nodes (chain, ER, TSP) — ignored for AR sensor
    N_states     : number of discrete states per variable (default 4)
    seed         : random seed for reproducibility
    er_p         : edge probability for Erdős–Rényi topology
    chain_query  : query position for chain topology
                   ('middle' | 'local' | 'deep' | 'full')

    Returns
    -------
    GraphModel
    """
    if topology == GraphTopology.CHAIN:
        return _build_chain(N, N_states, seed, query_position=chain_query)
    elif topology == GraphTopology.ERDOS_RENYI:
        return _build_erdos_renyi(N, er_p, N_states, seed)
    elif topology == GraphTopology.AR_SENSOR:
        return _build_ar_sensor_fusion(N_states, seed)
    elif topology == GraphTopology.TSP:
        return _build_tsp(N, seed)
    else:
        raise ValueError(f"Unknown topology: {topology}")


def build_ensemble(
    topology:    GraphTopology,
    n_instances: int = 100,
    base_seed:   int = 42,
    **kwargs,
) -> List[GraphModel]:
    """
    Build an ensemble of n_instances graph instances, each with a different
    seed derived from base_seed + instance_index. Used by benchmark.py to
    compute mean and standard deviation of energy ratios.
    """
    return [
        build_graph(topology, seed=base_seed + i, **kwargs)
        for i in range(n_instances)
    ]


# ─── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Chain N=50 ===")
    g = build_graph(GraphTopology.CHAIN, N=50, seed=42)
    print(f"  Nodes : {len(g.nodes)}")
    print(f"  Edges : {len(g.edges)}")
    print(f"  Query : {g.query}")
    print(f"  |S|   : {len(g.active_subgraph())}")
    print(f"  CPT x1 shape: {g.cpt['x1'].shape}")
    print(f"  CPT x1 row0 sums to 1: {g.cpt['x1'][0].sum():.6f}")

    print("\n=== ER N=50 p=0.3 ===")
    g2 = build_graph(GraphTopology.ERDOS_RENYI, N=50, er_p=0.3, seed=42)
    print(f"  Edges: {len(g2.edges)}")
    print(f"  Query: {g2.query}")
    print(f"  |S|  : {len(g2.active_subgraph())}")

    print("\n=== AR Sensor Fusion ===")
    g3 = build_graph(GraphTopology.AR_SENSOR, seed=42)
    print(f"  Nodes: {g3.nodes}")
    print(f"  Query: {g3.query}")
    print(f"  |S|  : {len(g3.active_subgraph())}")

    print("\n=== TSP N=10 ===")
    g4 = build_graph(GraphTopology.TSP, N=10, seed=42)
    print(f"  Cities: {len(g4.nodes)}")
    print(f"  Optimal tour: {g4.meta['optimal_length']:.4f}")
