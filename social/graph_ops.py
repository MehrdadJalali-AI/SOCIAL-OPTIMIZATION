"""Graph operations for SOCIAL optimizer: WS graph, centrality, rewiring."""

import numpy as np
import networkx as nx
from typing import Dict, Optional, Tuple, List
from scipy.stats import entropy


def create_watts_strogatz_graph(n: int, k: int, p: float, seed: Optional[int] = None) -> nx.Graph:
    """
    Create Watts-Strogatz small-world graph.
    
    Args:
        n: Number of nodes
        k: Each node connected to k nearest neighbors (must be even)
        p: Rewiring probability
        seed: Random seed
        
    Returns:
        NetworkX graph
    """
    if seed is not None:
        np.random.seed(seed)
    return nx.watts_strogatz_graph(n, k=k, p=p, seed=seed)


def compute_centrality(G: nx.Graph, mode: str = "betweenness", 
                       cache: Optional[Dict] = None) -> Dict[int, float]:
    """
    Compute centrality measures for graph nodes.
    
    Args:
        G: NetworkX graph
        mode: Centrality type (betweenness, degree, closeness, pagerank, eigenvector)
        cache: Optional cache dictionary to store results
        
    Returns:
        Dictionary mapping node -> centrality value
    """
    if cache is not None and mode in cache:
        return cache[mode]
    
    if mode == "betweenness":
        # Betweenness centrality (SOCIAL paper default)
        # Use approximate for large graphs
        if len(G) > 500:
            cent = nx.betweenness_centrality(G, k=min(100, len(G)//10))
        else:
            cent = nx.betweenness_centrality(G)
    elif mode == "degree":
        cent = nx.degree_centrality(G)
    elif mode == "closeness":
        cent = nx.closeness_centrality(G)
    elif mode == "pagerank":
        cent = nx.pagerank(G, alpha=0.85)
    elif mode == "eigenvector":
        try:
            cent = nx.eigenvector_centrality(G, max_iter=1500)
        except nx.NetworkXError:
            # Fallback to degree if eigenvector fails
            cent = nx.degree_centrality(G)
    else:
        raise ValueError(f"Unknown centrality mode: {mode}")
    
    if cache is not None:
        cache[mode] = cent
    
    return cent


def compute_fitness_influence(fitness_values: np.ndarray, mode: str = "rank") -> np.ndarray:
    """
    Compute relative fitness influence for minimization problems.
    
    Args:
        fitness_values: Array of fitness values (lower is better for minimization)
        mode: Influence computation mode ("rank" or "minmax")
        
    Returns:
        Array of influence values in [0, 1] where higher influence = better fitness
        
    Note:
        Monotonicity guarantee: if f_i < f_j (better), then influence_i >= influence_j
    """
    n = len(fitness_values)
    if n == 0:
        return np.array([])
    
    if n == 1:
        return np.array([1.0])
    
    if mode == "rank":
        # Rank-based influence: best fitness gets rank 0, worst gets rank N-1
        # influence = 1 - rank/(N-1)
        sorted_indices = np.argsort(fitness_values)  # Ascending order (best first)
        ranks = np.zeros(n, dtype=float)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank
        
        influence = 1.0 - ranks / (n - 1)
        return np.clip(influence, 0.0, 1.0)
    
    elif mode == "minmax":
        # Min-max normalization: influence = (f_max - f) / (f_max - f_min + eps)
        f_min = np.min(fitness_values)
        f_max = np.max(fitness_values)
        eps = 1e-10
        
        if f_max - f_min < eps:
            # All fitnesses are equal
            return np.ones_like(fitness_values)
        
        influence = (f_max - fitness_values) / (f_max - f_min + eps)
        return np.clip(influence, 0.0, 1.0)
    
    else:
        raise ValueError(f"Unknown influence mode: {mode}. Must be 'rank' or 'minmax'")


def rewire_graph(G: nx.Graph, mode: str = "periodic", 
                 iteration: int = 0, rewire_interval: int = 75,
                 stagnation_count: int = 0, stagnation_threshold: int = 50,
                 diversity_threshold: float = 0.3, rng: Optional[np.random.Generator] = None,
                 rewire_prob: float = 0.05) -> nx.Graph:
    """
    Rewire graph edges based on strategy.
    
    Args:
        G: NetworkX graph
        mode: Rewiring strategy (none, periodic, stagnation, diversity)
        iteration: Current iteration
        rewire_interval: Interval for periodic rewiring
        stagnation_count: Count of iterations without improvement
        stagnation_threshold: Threshold for stagnation rewiring
        diversity_threshold: Threshold for diversity-based rewiring
        rng: Random number generator
        
    Returns:
        Modified graph
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if mode == "none":
        return G
    
    elif mode == "periodic":
        if iteration > 0 and iteration % rewire_interval == 0:
            # Rewire with configurable probability (default 5% of edges)
            nswap = max(1, int(rewire_prob * G.number_of_edges()))
            try:
                nx.double_edge_swap(G, nswap=nswap, max_tries=1000, seed=rng)
            except nx.NetworkXError:
                pass
    
    elif mode == "stagnation":
        if stagnation_count >= stagnation_threshold:
            nswap = max(1, int(0.15 * G.number_of_edges()))
            try:
                nx.double_edge_swap(G, nswap=nswap, max_tries=1000, seed=rng)
            except nx.NetworkXError:
                pass
    
    elif mode == "diversity":
        # Compute population diversity (variance of positions)
        positions = np.array([G.nodes[n].get('position', np.zeros(1)) for n in G.nodes])
        if len(positions.shape) == 2:
            mean_pos = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - mean_pos, axis=1)
            diversity = np.std(distances)
            
            if diversity < diversity_threshold:
                nswap = max(1, int(0.1 * G.number_of_edges()))
                try:
                    nx.double_edge_swap(G, nswap=nswap, max_tries=1000, seed=rng)
                except nx.NetworkXError:
                    pass
    
    return G


def compute_graph_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Compute graph topology metrics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Average path length
    if nx.is_connected(G):
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
    else:
        # For disconnected graphs, compute for largest component
        largest_cc = max(nx.connected_components(G), key=len)
        if len(largest_cc) > 1:
            subgraph = G.subgraph(largest_cc)
            metrics['avg_path_length'] = nx.average_shortest_path_length(subgraph)
        else:
            metrics['avg_path_length'] = 0.0
    
    # Clustering coefficient
    metrics['clustering_coeff'] = nx.average_clustering(G)
    
    # Algebraic connectivity (second smallest eigenvalue of Laplacian)
    try:
        laplacian = nx.normalized_laplacian_matrix(G)
        eigenvals = np.linalg.eigvals(laplacian.toarray())
        eigenvals = np.sort(np.real(eigenvals))
        if len(eigenvals) > 1:
            metrics['algebraic_connectivity'] = float(eigenvals[1])
            if len(eigenvals) > 2:
                metrics['spectral_gap'] = float(eigenvals[2] - eigenvals[1])
            else:
                metrics['spectral_gap'] = 0.0
        else:
            metrics['algebraic_connectivity'] = 0.0
            metrics['spectral_gap'] = 0.0
    except:
        metrics['algebraic_connectivity'] = 0.0
        metrics['spectral_gap'] = 0.0
    
    # Degree statistics
    degrees = [d for _, d in G.degree()]
    metrics['mean_degree'] = np.mean(degrees)
    metrics['std_degree'] = np.std(degrees)
    
    return metrics


def compute_population_entropy(G: nx.Graph) -> float:
    """
    Compute population entropy based on degree distribution.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Entropy value
    """
    if not G.nodes or G.number_of_edges() == 0:
        return 0.0
    
    deg_vals = np.array([d for _, d in G.degree()])
    if deg_vals.sum() == 0:
        return 0.0
    
    p = deg_vals / deg_vals.sum()
    return entropy(p + 1e-12)

