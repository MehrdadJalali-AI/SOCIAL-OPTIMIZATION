"""SOCIAL optimizer matching paper equations."""

import numpy as np
import networkx as nx
import time
from typing import Dict, Optional, Tuple, List, Callable
from .config import Config
from .budget import BudgetedObjective
from .graph_ops import (
    create_watts_strogatz_graph,
    compute_centrality,
    compute_fitness_influence,
    rewire_graph,
    compute_graph_metrics,
    compute_population_entropy
)


class SOCIALOptimizer:
    """SOCIAL optimizer matching paper Algorithm 1."""
    
    def __init__(self, config: Config, rng: Optional[np.random.Generator] = None):
        """
        Initialize SOCIAL optimizer.
        
        Args:
            config: Configuration object
            rng: Random number generator (for reproducibility)
        """
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Runtime profiling
        self.runtime_stats = {
            't_eval': [],
            't_centrality': [],
            't_update': [],
            't_mut': [],
            't_rewire': [],
            't_total': []
        }
        
        # Centrality cache
        self._centrality_cache = {}
        self._last_centrality_iter = -1
        
        # Stagnation tracking
        self._stagnation_count = 0
        self._last_gbest = None
    
    def compute_schedule_weights(self, t: float, mode: str = "linear") -> Tuple[float, float, float, float]:
        """
        Compute scheduled weights αt, βt, γt, δt (SOCIAL paper Eq. 8-11).
        
        Args:
            t: Normalized iteration (0 to 1)
            mode: Schedule mode (linear, exp, cosine, piecewise)
            
        Returns:
            Tuple of (alpha_t, beta_t, gamma_t, delta_t)
        """
        if mode == "linear":
            # Linear schedule (paper default)
            alpha_t = self.config.ALPHA_INIT * (1 - t) + 0.1 * t
            beta_t = self.config.BETA_INIT * (1 - t) + 0.1 * t
            gamma_t = self.config.GAMMA * t
            delta_t = self.config.DELTA * t
            
        elif mode == "exp":
            # Exponential schedule
            alpha_t = self.config.ALPHA_INIT * np.exp(-5 * t) + 0.1
            beta_t = self.config.BETA_INIT * np.exp(-5 * t) + 0.1
            gamma_t = self.config.GAMMA * (1 - np.exp(-5 * t))
            delta_t = self.config.DELTA * (1 - np.exp(-5 * t))
            
        elif mode == "cosine":
            # Cosine schedule
            alpha_t = self.config.ALPHA_INIT * (0.5 + 0.5 * np.cos(np.pi * t)) + 0.1 * (1 - np.cos(np.pi * t))
            beta_t = self.config.BETA_INIT * (0.5 + 0.5 * np.cos(np.pi * t)) + 0.1 * (1 - np.cos(np.pi * t))
            gamma_t = self.config.GAMMA * (1 - np.cos(np.pi * t / 2))
            delta_t = self.config.DELTA * (1 - np.cos(np.pi * t / 2))
            
        elif mode == "piecewise":
            # Piecewise linear
            if t < 0.5:
                alpha_t = self.config.ALPHA_INIT * (1 - 2*t)
                beta_t = self.config.BETA_INIT * (1 - 2*t)
                gamma_t = 0.0
                delta_t = 0.0
            else:
                alpha_t = 0.1
                beta_t = 0.1
                gamma_t = self.config.GAMMA * (2*t - 1)
                delta_t = self.config.DELTA * (2*t - 1)
        else:
            raise ValueError(f"Unknown schedule mode: {mode}")
        
        # Normalize to ensure sum <= 1
        total = alpha_t + beta_t + gamma_t + delta_t
        if total > 1.0:
            scale = 1.0 / total
            alpha_t *= scale
            beta_t *= scale
            gamma_t *= scale
            delta_t *= scale
        
        return alpha_t, beta_t, gamma_t, delta_t
    
    def initialize_population(self, dim: int, bounds: List[float], seed: Optional[int] = None) -> nx.Graph:
        """
        Initialize population on Watts-Strogatz graph.
        
        Args:
            dim: Problem dimension
            bounds: [lower, upper] bounds
            seed: Random seed for graph generation
            
        Returns:
            NetworkX graph with initialized positions
        """
        G = create_watts_strogatz_graph(
            self.config.NUM_NODES,
            self.config.K,
            self.config.P_BASE,
            seed=seed
        )
        
        # Initialize positions uniformly
        for node in G.nodes:
            G.nodes[node]['position'] = self.rng.uniform(bounds[0], bounds[1], dim)
            G.nodes[node]['fitness'] = None
        
        return G
    
    def evaluate_population(self, G: nx.Graph, obj_func: BudgetedObjective) -> None:
        """
        Evaluate fitness for all nodes in graph.
        
        Args:
            G: NetworkX graph
            obj_func: Budgeted objective function
        """
        t_start = time.perf_counter()
        for node in G.nodes:
            if obj_func.exhausted():
                break
            pos = G.nodes[node]['position']
            G.nodes[node]['fitness'] = obj_func(pos)
        t_eval = time.perf_counter() - t_start
        self.runtime_stats['t_eval'].append(t_eval)
    
    def diffuse(self, G: nx.Graph, gbest_pos: np.ndarray, elite_pos: np.ndarray,
                bounds: List[float], iteration: int, max_iterations: int,
                obj_func: BudgetedObjective) -> nx.Graph:
        """
        Diffusion step (SOCIAL paper Eq. 13-16).
        
        Args:
            G: NetworkX graph
            gbest_pos: Global best position
            elite_pos: Elite position
            bounds: [lower, upper] bounds
            iteration: Current iteration
            max_iterations: Maximum iterations
            obj_func: Budgeted objective function
            
        Returns:
            Modified graph
        """
        t_start_total = time.perf_counter()
        
        if not G.nodes:
            raise ValueError("Graph is empty in diffuse")
        
        t = iteration / max_iterations if max_iterations > 0 else 0.0
        
        # Compute schedule weights
        alpha_t, beta_t, gamma_t, delta_t = self.compute_schedule_weights(t, self.config.SCHEDULE_MODE)
        
        # Compute centrality (with caching)
        t_cent_start = time.perf_counter()
        should_recompute = False
        
        if self.config.CENTRALITY_RECOMPUTE == "always":
            should_recompute = True
        elif self.config.CENTRALITY_RECOMPUTE == "interval":
            if iteration == 0 or (iteration - self._last_centrality_iter) >= self.config.BC_INTERVAL:
                should_recompute = True
        elif self.config.CENTRALITY_RECOMPUTE == "stagnation":
            if self._stagnation_count >= self.config.STAGNATION_THRESHOLD:
                should_recompute = True
        
        if should_recompute:
            self._centrality_cache = {}
            self._last_centrality_iter = iteration
        
        centrality = compute_centrality(G, self.config.CENTRALITY_MODE, self._centrality_cache)
        t_cent = time.perf_counter() - t_cent_start
        self.runtime_stats['t_centrality'].append(t_cent)
        
        # Compute fitness influence
        fitness_values = np.array([G.nodes[n]['fitness'] for n in G.nodes])
        influence = compute_fitness_influence(fitness_values)
        
        # Compute population mean for synchronization
        positions = np.array([G.nodes[n]['position'] for n in G.nodes])
        dimension_mean = np.mean(positions, axis=0)
        
        # Update positions (SOCIAL paper Eq. 13-16)
        t_update_start = time.perf_counter()
        new_positions = {}
        
        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                new_positions[node] = G.nodes[node]['position'].copy()
                continue
            
            # Compute neighbor weights
            if self.config.NEIGHBOR_MODE == "centrality_weighted":
                # Weight by centrality and influence
                neighbor_centralities = np.array([centrality[n] for n in neighbors])
                neighbor_influences = np.array([influence[list(G.nodes).index(n)] for n in neighbors])
                
                weights = alpha_t * neighbor_centralities + beta_t * neighbor_influences
                weights = np.maximum(weights, 1e-10)  # Avoid zeros
                weights = weights / weights.sum()
            else:  # uniform
                weights = np.ones(len(neighbors)) / len(neighbors)
            
            # Aggregate neighbor positions
            neighbor_positions = np.array([G.nodes[n]['position'] for n in neighbors])
            neighbor_contribution = np.average(neighbor_positions, weights=weights, axis=0)
            
            # Update equation (SOCIAL paper Eq. 13)
            current_pos = G.nodes[node]['position']
            self_weight = 1.0 - (alpha_t + beta_t + gamma_t + delta_t)
            
            new_pos = (self_weight * current_pos +
                      (alpha_t + beta_t) * neighbor_contribution +
                      gamma_t * gbest_pos +
                      delta_t * elite_pos)
            
            # Periodic synchronization (every SYNC_INTERVAL iterations)
            if self.config.ENABLE_SYNC and iteration % self.config.SYNC_INTERVAL == 0:
                sync_weight = self.config.SYNC_WEIGHT_INIT * (1 - t)
                new_pos = (1 - sync_weight) * new_pos + sync_weight * dimension_mean
            
            # Enforce bounds
            new_pos = np.clip(new_pos, bounds[0], bounds[1])
            new_positions[node] = new_pos
        
        t_update = time.perf_counter() - t_update_start
        self.runtime_stats['t_update'].append(t_update)
        
        # Apply mutations
        t_mut_start = time.perf_counter()
        if self.config.ENABLE_MUTATION:
            mutation_rate_t = self.config.MUTATION_RATE_INIT * (1 - t) + 0.01
            mutation_strength = self.config.MUTATION_STRENGTH_BASE * (1 - t)
            median_fitness = np.median(fitness_values)
            
            for node in G.nodes:
                pos = new_positions[node].copy()
                
                # Mutation for worse-than-median nodes
                if G.nodes[node]['fitness'] > median_fitness:
                    if self.rng.random() < mutation_rate_t:
                        node_cent = centrality.get(node, 0.0)
                        sigma_i = mutation_strength * (1 - node_cent**1.5) + self.config.MUTATION_STRENGTH_MIN
                        perturb = self.rng.uniform(
                            -sigma_i * (bounds[1] - bounds[0]),
                            sigma_i * (bounds[1] - bounds[0]),
                            len(pos)
                        )
                        pos = np.clip(pos + perturb, bounds[0], bounds[1])
                
                # Periodic perturbation (every 10 iterations)
                if iteration % 10 == 0 and self.rng.random() < 0.05:
                    pos += self.rng.uniform(-1, 1, len(pos)) * 0.5
                    pos = np.clip(pos, bounds[0], bounds[1])
                
                new_positions[node] = pos
        
        # Update positions
        for node, pos in new_positions.items():
            G.nodes[node]['position'] = pos
        
        t_mut = time.perf_counter() - t_mut_start
        self.runtime_stats['t_mut'].append(t_mut)
        
        # Rewiring
        t_rewire_start = time.perf_counter()
        G = rewire_graph(
            G,
            mode=self.config.REWIRE_MODE,
            iteration=iteration,
            rewire_interval=self.config.REWIRE_INTERVAL,
            stagnation_count=self._stagnation_count,
            stagnation_threshold=self.config.STAGNATION_THRESHOLD,
            rng=self.rng
        )
        t_rewire = time.perf_counter() - t_rewire_start
        self.runtime_stats['t_rewire'].append(t_rewire)
        
        # Clear centrality cache if rewired
        if self.config.REWIRE_MODE != "none" and iteration % self.config.REWIRE_INTERVAL == 0:
            self._centrality_cache = {}
        
        t_total = time.perf_counter() - t_start_total
        self.runtime_stats['t_total'].append(t_total)
        
        return G
    
    def optimize(self, obj_func: BudgetedObjective, bounds: List[float],
                 dim: int, seed: Optional[int] = None) -> Dict:
        """
        Run SOCIAL optimization.
        
        Args:
            obj_func: Budgeted objective function
            bounds: [lower, upper] bounds
            dim: Problem dimension
            seed: Random seed
            
        Returns:
            Dictionary with results and history
        """
        # Reset runtime stats
        for key in self.runtime_stats:
            self.runtime_stats[key] = []
        
        # Initialize
        G = self.initialize_population(dim, bounds, seed=seed)
        self.evaluate_population(G, obj_func)
        
        fitness_values = np.array([G.nodes[n]['fitness'] for n in G.nodes])
        gbest_idx = np.argmin(fitness_values)
        gbest_pos = G.nodes[list(G.nodes)[gbest_idx]]['position'].copy()
        gbest_fitness = fitness_values[gbest_idx]
        
        elite_pos = gbest_pos.copy()
        elite_fitness = gbest_fitness
        
        # History tracking
        convergence_history = [gbest_fitness]
        avg_fitness_history = [np.mean(fitness_values)]
        best_fitness_archive = [gbest_fitness]
        
        iteration = 0
        max_iterations = obj_func.max_evals // self.config.NUM_NODES
        
        while not obj_func.exhausted() and iteration < max_iterations:
            # Evaluate population
            self.evaluate_population(G, obj_func)
            
            if obj_func.exhausted():
                break
            
            fitness_values = np.array([G.nodes[n]['fitness'] for n in G.nodes])
            
            # Update gbest
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < gbest_fitness:
                gbest_pos = G.nodes[list(G.nodes)[min_idx]]['position'].copy()
                gbest_fitness = fitness_values[min_idx]
                self._stagnation_count = 0
            else:
                self._stagnation_count += 1
            
            # Update elite (if enabled)
            if self.config.ENABLE_ELITE_MEMORY:
                if gbest_fitness < elite_fitness:
                    elite_pos = gbest_pos.copy()
                    elite_fitness = gbest_fitness
            else:
                elite_pos = gbest_pos.copy()
                elite_fitness = gbest_fitness
            
            # Diffusion step
            G = self.diffuse(G, gbest_pos, elite_pos, bounds, iteration, max_iterations, obj_func)
            
            # Track history
            convergence_history.append(gbest_fitness)
            avg_fitness_history.append(np.mean(fitness_values))
            best_fitness_archive.append(elite_fitness)
            
            iteration += 1
        
        return {
            'best_position': elite_pos,
            'best_fitness': elite_fitness,
            'convergence_history': convergence_history,
            'avg_fitness_history': avg_fitness_history,
            'best_fitness_archive': best_fitness_archive,
            'final_population': G,
            'iterations': iteration,
            'evals_used': obj_func.evals_used,
            'runtime_stats': self.runtime_stats
        }

