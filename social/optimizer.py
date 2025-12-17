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
        
        # SOCIAL++ tracking
        self._de_accepted_count = []  # Track DE acceptances per iteration
    
    def compute_learning_rate(self, t: float) -> float:
        """
        Compute learning rate ETA_t for stabilizing updates.
        
        Args:
            t: Normalized iteration (0 to 1)
            
        Returns:
            Learning rate in [ETA_MIN, ETA_INIT]
        """
        if self.config.ETA_SCHEDULE == "linear":
            eta_t = self.config.ETA_INIT * (1 - t) + self.config.ETA_MIN * t
        elif self.config.ETA_SCHEDULE == "exp":
            eta_t = self.config.ETA_INIT * np.exp(-3 * t) + self.config.ETA_MIN * (1 - np.exp(-3 * t))
        elif self.config.ETA_SCHEDULE == "cosine":
            eta_t = self.config.ETA_MIN + (self.config.ETA_INIT - self.config.ETA_MIN) * (0.5 + 0.5 * np.cos(np.pi * t))
        else:
            eta_t = self.config.ETA_INIT * (1 - t) + self.config.ETA_MIN * t
        
        return np.clip(eta_t, self.config.ETA_MIN, self.config.ETA_INIT)
    
    def repair_boundary(self, x: np.ndarray, bounds: List[float]) -> np.ndarray:
        """
        Repair boundary violations using clip or reflect mode.
        
        Args:
            x: Position vector
            bounds: [lower, upper] bounds
            
        Returns:
            Repaired position vector
        """
        if self.config.BOUNDARY_MODE == "clip":
            return np.clip(x, bounds[0], bounds[1])
        elif self.config.BOUNDARY_MODE == "reflect":
            # Reflect back into bounds using modulo arithmetic
            lower, upper = bounds[0], bounds[1]
            range_size = upper - lower
            
            # Reflect each dimension independently
            repaired = x.copy()
            for i in range(len(x)):
                if repaired[i] < lower:
                    # Reflect: distance from lower bound
                    excess = lower - repaired[i]
                    repaired[i] = lower + (excess % (2 * range_size))
                    if repaired[i] > upper:
                        repaired[i] = 2 * upper - repaired[i]
                elif repaired[i] > upper:
                    # Reflect: distance from upper bound
                    excess = repaired[i] - upper
                    repaired[i] = upper - (excess % (2 * range_size))
                    if repaired[i] < lower:
                        repaired[i] = 2 * lower - repaired[i]
            
            # Final clip to ensure within bounds (safety check)
            return np.clip(repaired, lower, upper)
        else:
            return np.clip(x, bounds[0], bounds[1])
    
    def compute_schedule_weights(self, t: float, mode: str = "linear") -> Tuple[float, float, float, float]:
        """
        Compute scheduled weights αt, βt, γt, δt (SOCIAL paper Eq. 8-11).
        
        Args:
            t: Normalized iteration (0 to 1)
            mode: Schedule mode (linear, exp, cosine, piecewise)
            
        Returns:
            Tuple of (alpha_t, beta_t, gamma_t, delta_t)
        """
        # Use INIT/FINAL if available, otherwise fallback to legacy
        alpha_init = self.config.ALPHA_INIT
        alpha_final = getattr(self.config, 'ALPHA_FINAL', 0.1)
        beta_init = self.config.BETA_INIT
        beta_final = getattr(self.config, 'BETA_FINAL', 0.1)
        gamma_init = getattr(self.config, 'GAMMA_INIT', 0.0)
        gamma_final = getattr(self.config, 'GAMMA_FINAL', self.config.GAMMA)
        delta_init = getattr(self.config, 'DELTA_INIT', 0.0)
        delta_final = getattr(self.config, 'DELTA_FINAL', self.config.DELTA)
        
        if mode == "linear":
            # Linear schedule (paper default) - strengthened exploitation
            alpha_t = alpha_init * (1 - t) + alpha_final * t
            beta_t = beta_init * (1 - t) + beta_final * t
            gamma_t = gamma_init * (1 - t) + gamma_final * t
            delta_t = delta_init * (1 - t) + delta_final * t
            
        elif mode == "exp":
            # Exponential schedule
            alpha_t = alpha_init * np.exp(-5 * t) + alpha_final * (1 - np.exp(-5 * t))
            beta_t = beta_init * np.exp(-5 * t) + beta_final * (1 - np.exp(-5 * t))
            gamma_t = gamma_init * np.exp(-5 * t) + gamma_final * (1 - np.exp(-5 * t))
            delta_t = delta_init * np.exp(-5 * t) + delta_final * (1 - np.exp(-5 * t))
            
        elif mode == "cosine":
            # Cosine schedule
            alpha_t = alpha_init * (0.5 + 0.5 * np.cos(np.pi * t)) + alpha_final * (1 - np.cos(np.pi * t))
            beta_t = beta_init * (0.5 + 0.5 * np.cos(np.pi * t)) + beta_final * (1 - np.cos(np.pi * t))
            gamma_t = gamma_init * (1 - np.cos(np.pi * t / 2)) + gamma_final * (1 - np.cos(np.pi * t / 2))
            delta_t = delta_init * (1 - np.cos(np.pi * t / 2)) + delta_final * (1 - np.cos(np.pi * t / 2))
            
        elif mode == "piecewise":
            # Piecewise linear
            if t < 0.5:
                alpha_t = alpha_init * (1 - 2*t)
                beta_t = beta_init * (1 - 2*t)
                gamma_t = gamma_init
                delta_t = delta_init
            else:
                alpha_t = alpha_final
                beta_t = beta_final
                gamma_t = gamma_init + (gamma_final - gamma_init) * (2*t - 1)
                delta_t = delta_init + (delta_final - delta_init) * (2*t - 1)
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
        
        # Compute fitness influence (with stable node-to-index mapping)
        node_list = list(G.nodes)
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        fitness_values = np.array([G.nodes[n]['fitness'] for n in node_list])
        influence = compute_fitness_influence(fitness_values, mode=self.config.INFLUENCE_MODE)
        
        # Compute population mean for synchronization
        positions = np.array([G.nodes[n]['position'] for n in G.nodes])
        dimension_mean = np.mean(positions, axis=0)
        
        # Compute learning rate
        eta_t = self.compute_learning_rate(t)
        
        # Update positions (SOCIAL paper Eq. 13-16, improved)
        t_update_start = time.perf_counter()
        new_positions = {}
        
        for node in G.nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                new_positions[node] = G.nodes[node]['position'].copy()
                continue
            
            # Compute two separate neighbor aggregates
            neighbor_positions = np.array([G.nodes[n]['position'] for n in neighbors])
            
            if self.config.NEIGHBOR_MODE == "centrality_weighted":
                # Centrality-weighted aggregate
                neighbor_centralities = np.array([centrality[n] for n in neighbors])
                cent_weights = neighbor_centralities + 1e-10  # Avoid zeros
                cent_weights = cent_weights / cent_weights.sum()
                x_cent = np.average(neighbor_positions, weights=cent_weights, axis=0)
                
                # Influence-weighted aggregate
                neighbor_influences = np.array([influence[node_to_idx[n]] for n in neighbors])
                inf_weights = neighbor_influences + 1e-10  # Avoid zeros
                inf_weights = inf_weights / inf_weights.sum()
                x_inf = np.average(neighbor_positions, weights=inf_weights, axis=0)
            else:  # uniform
                # Use equal weights for both aggregates
                x_cent = np.mean(neighbor_positions, axis=0)
                x_inf = np.mean(neighbor_positions, axis=0)
            
            # Update equation with separated aggregates
            current_pos = G.nodes[node]['position']
            self_weight = 1.0 - (alpha_t + beta_t + gamma_t + delta_t)
            
            target = (self_weight * current_pos +
                     alpha_t * x_cent +
                     beta_t * x_inf +
                     gamma_t * gbest_pos +
                     delta_t * elite_pos)
            
            # Apply learning rate stabilization
            new_pos = (1 - eta_t) * current_pos + eta_t * target
            
            # Periodic synchronization (every SYNC_INTERVAL iterations)
            if self.config.ENABLE_SYNC and iteration % self.config.SYNC_INTERVAL == 0:
                sync_weight = self.config.SYNC_WEIGHT_INIT * (1 - t)
                new_pos = (1 - sync_weight) * new_pos + sync_weight * dimension_mean
            
            # Enforce bounds using configured mode
            new_pos = self.repair_boundary(new_pos, bounds)
            new_positions[node] = new_pos
        
        t_update = time.perf_counter() - t_update_start
        self.runtime_stats['t_update'].append(t_update)
        
        # Apply mutations and SOCIAL++ hybrid exploitation
        t_mut_start = time.perf_counter()
        de_accepted = 0
        
        if self.config.ENABLE_MUTATION:
            # Use new mutation schedule if available
            mutation_init = self.config.MUTATION_RATE_INIT
            mutation_final = getattr(self.config, 'MUTATION_RATE_FINAL', 0.01)
            mutation_schedule = getattr(self.config, 'MUTATION_SCHEDULE', 'linear')
            
            if mutation_schedule == "exp":
                mutation_rate_t = mutation_init * np.exp(-5 * t) + mutation_final * (1 - np.exp(-5 * t))
            elif mutation_schedule == "cosine":
                mutation_rate_t = mutation_final + (mutation_init - mutation_final) * (0.5 + 0.5 * np.cos(np.pi * t))
            else:  # linear
                mutation_rate_t = mutation_init * (1 - t) + mutation_final * t
            mutation_strength = self.config.MUTATION_STRENGTH_BASE * (1 - t)
            median_fitness = np.median(fitness_values)
            
            # SOCIAL++: Identify worst nodes for DE exploitation
            if self.config.SOCIALPP_MODE:
                # Compute DE probability (decays over time)
                de_p_t = self.config.DE_P_INIT * (1 - t) + self.config.DE_P_MIN * t
                
                # Sort nodes by fitness (worst first)
                node_fitness_pairs = [(n, G.nodes[n]['fitness']) for n in node_list]
                node_fitness_pairs.sort(key=lambda x: x[1], reverse=True)  # Worst first
                
                # Select worst DE_Q fraction
                n_de = max(1, int(self.config.DE_Q * len(node_list)))
                worst_nodes = [n for n, _ in node_fitness_pairs[:n_de]]
                
                for node in worst_nodes:
                    if self.rng.random() < de_p_t:
                        # DE/current-to-best/1/bin
                        # Select distinct random indices
                        other_nodes = [n for n in node_list if n != node]
                        if len(other_nodes) < 2:
                            continue
                        
                        r1 = self.rng.choice(other_nodes)
                        r2_candidates = [n for n in other_nodes if n != r1]
                        if not r2_candidates:
                            continue
                        r2 = self.rng.choice(r2_candidates)
                        
                        # DE mutation: v = x_i + F*(x_gbest - x_i) + F*(x_r1 - x_r2)
                        x_i = new_positions[node]
                        x_r1 = G.nodes[r1]['position']
                        x_r2 = G.nodes[r2]['position']
                        
                        v = x_i + self.config.DE_F * (gbest_pos - x_i) + self.config.DE_F * (x_r1 - x_r2)
                        
                        # Binomial crossover
                        u = x_i.copy()
                        j_rand = self.rng.integers(len(x_i))
                        for j in range(len(x_i)):
                            if self.rng.random() < self.config.DE_CR or j == j_rand:
                                u[j] = v[j]
                        
                        # Repair bounds
                        u = self.repair_boundary(u, bounds)
                        
                        # Evaluate and accept if better
                        if not obj_func.exhausted():
                            f_u = obj_func(u)
                            f_i = G.nodes[node]['fitness']
                            
                            if f_u < f_i:  # Better fitness (minimization)
                                new_positions[node] = u
                                G.nodes[node]['fitness'] = f_u
                                de_accepted += 1
            
            # Standard mutation for worse-than-median nodes
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
                        pos = pos + perturb
                        pos = self.repair_boundary(pos, bounds)
                
                # Periodic perturbation (every 10 iterations)
                if iteration % 10 == 0 and self.rng.random() < 0.05:
                    pos += self.rng.uniform(-1, 1, len(pos)) * 0.5
                    pos = self.repair_boundary(pos, bounds)
                
                new_positions[node] = pos
        
        # Update positions
        for node, pos in new_positions.items():
            G.nodes[node]['position'] = pos
        
        t_mut = time.perf_counter() - t_mut_start
        self.runtime_stats['t_mut'].append(t_mut)
        self._de_accepted_count.append(de_accepted)
        
        # Rewiring (compute interval based on max_iterations if needed)
        t_rewire_start = time.perf_counter()
        # Update rewire interval if needed (15% of max iterations)
        rewire_interval = self.config.REWIRE_INTERVAL
        if max_iterations > 0:
            computed_interval = max(1, int(0.15 * max_iterations))
            # Use computed interval if it's significantly different from default
            if abs(computed_interval - rewire_interval) > 10:
                rewire_interval = computed_interval
        
        # Get rewire probability from config
        rewire_prob = getattr(self.config, 'REWIRE_PROB', 0.05)
        
        G = rewire_graph(
            G,
            mode=self.config.REWIRE_MODE,
            iteration=iteration,
            rewire_interval=rewire_interval,
            stagnation_count=self._stagnation_count,
            stagnation_threshold=self.config.STAGNATION_THRESHOLD,
            rng=self.rng,
            rewire_prob=rewire_prob
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
        
        # Reset SOCIAL++ tracking
        self._de_accepted_count = []
        
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
        
        # Progress reporting setup
        start_time = time.time()
        progress_interval = getattr(self.config, 'PROGRESS_INTERVAL', 5)
        show_progress = getattr(self.config, 'SHOW_PROGRESS', True)
        last_progress_iter = -1
        last_progress_evals = 0
        progress_eval_interval = max(1, int(0.05 * obj_func.max_evals))  # 5% of budget
        
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
            
            # Stagnation response: rewiring + reseeding
            if self._stagnation_count >= self.config.STAGNATION_THRESHOLD:
                # Trigger graph rewiring
                G = rewire_graph(
                    G,
                    mode="stagnation",
                    iteration=iteration,
                    rewire_interval=self.config.REWIRE_INTERVAL,
                    stagnation_count=self._stagnation_count,
                    stagnation_threshold=self.config.STAGNATION_THRESHOLD,
                    rng=self.rng
                )
                
                # Reseed worst nodes
                node_list = list(G.nodes)
                node_fitness_pairs = [(n, G.nodes[n]['fitness']) for n in node_list]
                node_fitness_pairs.sort(key=lambda x: x[1], reverse=True)  # Worst first
                
                # Keep top ELITE_KEEP_FRAC nodes unchanged
                n_elite = max(1, int(self.config.ELITE_KEEP_FRAC * len(node_list)))
                elite_nodes = {n for n, _ in node_fitness_pairs[-n_elite:]}
                
                # Reseed worst RESEED_FRAC nodes
                n_reseed = max(1, int(self.config.RESEED_FRAC * len(node_list)))
                worst_nodes = [n for n, _ in node_fitness_pairs[:n_reseed] if n not in elite_nodes]
                
                # Reseeding mixture: 50% uniform, 25% opposition, 25% gaussian
                n_uniform = max(1, int(0.5 * len(worst_nodes)))
                n_opposition = max(1, int(0.25 * len(worst_nodes)))
                n_gaussian = len(worst_nodes) - n_uniform - n_opposition
                
                # Shuffle to randomize assignment
                shuffled_nodes = worst_nodes.copy()
                self.rng.shuffle(shuffled_nodes)
                
                for i, node in enumerate(shuffled_nodes):
                    if not obj_func.exhausted():
                        if i < n_uniform:
                            # 50% uniform random
                            G.nodes[node]['position'] = self.rng.uniform(bounds[0], bounds[1], len(elite_pos))
                        elif i < n_uniform + n_opposition:
                            # 25% opposition points: L + U - x_elite
                            x_opp = bounds[0] + bounds[1] - elite_pos
                            G.nodes[node]['position'] = self.repair_boundary(x_opp, bounds)
                        else:
                            # 25% gaussian around elite (sigma decays with iteration)
                            sigma_t = 0.1 * (bounds[1] - bounds[0]) * (1 - iteration / max_iterations)
                            x_gauss = elite_pos + self.rng.normal(0, sigma_t, len(elite_pos))
                            G.nodes[node]['position'] = self.repair_boundary(x_gauss, bounds)
                        
                        # Re-evaluate reseeded node
                        G.nodes[node]['fitness'] = obj_func(G.nodes[node]['position'])
                
                # Clear centrality cache after rewiring
                self._centrality_cache = {}
                self._last_centrality_iter = -1
                
                # Reset stagnation count
                self._stagnation_count = 0
            
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
            
            # Progress reporting (only if enabled and mode is iteration-level)
            progress_mode = getattr(self.config, 'PROGRESS_MODE', 'iteration')
            if show_progress and progress_mode == "iteration":
                current_evals = obj_func.evals_used
                should_report = False
                
                # Report every N iterations or every 5% of budget
                if (iteration - last_progress_iter >= progress_interval) or \
                   (current_evals - last_progress_evals >= progress_eval_interval):
                    should_report = True
                    last_progress_iter = iteration
                    last_progress_evals = current_evals
                
                if should_report:
                    elapsed = time.time() - start_time
                    completed_fraction = current_evals / obj_func.max_evals if obj_func.max_evals > 0 else 0
                    
                    if completed_fraction > 0:
                        eta_seconds = elapsed / completed_fraction * (1 - completed_fraction)
                        eta_str = f"{int(eta_seconds // 60)}m{int(eta_seconds % 60)}s"
                    else:
                        eta_str = "N/A"
                    
                    elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
                    
                    print(f"[PROGRESS] Iter={iteration}/{max_iterations} | "
                          f"Evals={current_evals}/{obj_func.max_evals} | "
                          f"Best={gbest_fitness:.2e} | "
                          f"Elapsed={elapsed_str} | ETA={eta_str}")
            
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

