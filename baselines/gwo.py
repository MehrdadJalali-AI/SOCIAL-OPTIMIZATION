"""Grey Wolf Optimizer (GWO) baseline."""

import numpy as np
from typing import List, Optional
from social.budget import BudgetedObjective


class GreyWolfOptimizer:
    """Grey Wolf Optimizer."""
    
    def __init__(self, npop: int = 50, rng: Optional[np.random.Generator] = None):
        """
        Initialize GWO.
        
        Args:
            npop: Population size (should be >= 4)
            rng: Random number generator
        """
        self.npop = max(4, npop)  # Need at least 4 wolves
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def optimize(self, obj_func: BudgetedObjective, bounds: List[float],
                dim: int, seed: Optional[int] = None) -> dict:
        """
        Run GWO optimization.
        
        Args:
            obj_func: Budgeted objective function
            bounds: [lower, upper] bounds
            dim: Problem dimension
            seed: Random seed
            
        Returns:
            Dictionary with results
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Initialize population
        pop = self.rng.uniform(bounds[0], bounds[1], (self.npop, dim))
        fitness = np.array([obj_func(ind) for ind in pop])
        
        # Sort wolves: alpha (best), beta (2nd), delta (3rd), omega (rest)
        sorted_indices = np.argsort(fitness)
        alpha_idx = sorted_indices[0]
        beta_idx = sorted_indices[1]
        delta_idx = sorted_indices[2]
        
        alpha_pos = pop[alpha_idx].copy()
        beta_pos = pop[beta_idx].copy()
        delta_pos = pop[delta_idx].copy()
        
        alpha_fitness = fitness[alpha_idx]
        
        convergence_history = [alpha_fitness]
        
        iteration = 0
        max_iterations = obj_func.max_evals // self.npop
        
        while not obj_func.exhausted() and iteration < max_iterations:
            # Update a (decreases linearly from 2 to 0)
            a = 2.0 * (1 - iteration / max_iterations)
            
            for i in range(self.npop):
                if obj_func.exhausted():
                    break
                
                # Update position based on alpha, beta, delta
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = np.abs(C1 * alpha_pos - pop[i])
                X1 = alpha_pos - A1 * D_alpha
                
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = np.abs(C2 * beta_pos - pop[i])
                X2 = beta_pos - A2 * D_beta
                
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = np.abs(C3 * delta_pos - pop[i])
                X3 = delta_pos - A3 * D_delta
                
                # New position is average of three
                pop[i] = (X1 + X2 + X3) / 3.0
                pop[i] = np.clip(pop[i], bounds[0], bounds[1])
                
                # Evaluate
                fitness[i] = obj_func(pop[i])
            
            # Update alpha, beta, delta
            sorted_indices = np.argsort(fitness)
            alpha_idx = sorted_indices[0]
            beta_idx = sorted_indices[1]
            delta_idx = sorted_indices[2]
            
            if fitness[alpha_idx] < alpha_fitness:
                alpha_pos = pop[alpha_idx].copy()
                alpha_fitness = fitness[alpha_idx]
            
            beta_pos = pop[beta_idx].copy()
            delta_pos = pop[delta_idx].copy()
            
            convergence_history.append(alpha_fitness)
            iteration += 1
        
        return {
            'best_position': alpha_pos,
            'best_fitness': alpha_fitness,
            'convergence_history': convergence_history,
            'iterations': iteration,
            'evals_used': obj_func.evals_used
        }

