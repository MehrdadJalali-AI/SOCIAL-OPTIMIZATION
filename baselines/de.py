"""Differential Evolution (DE/rand/1/bin) baseline."""

import numpy as np
from typing import List, Optional
from social.budget import BudgetedObjective


class DifferentialEvolution:
    """Classic DE/rand/1/bin algorithm."""
    
    def __init__(self, npop: int = 50, F: float = 0.5, CR: float = 0.9,
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize DE.
        
        Args:
            npop: Population size
            F: Scaling factor
            CR: Crossover rate
            rng: Random number generator
        """
        self.npop = npop
        self.F = F
        self.CR = CR
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def optimize(self, obj_func: BudgetedObjective, bounds: List[float],
                dim: int, seed: Optional[int] = None) -> dict:
        """
        Run DE optimization.
        
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
        
        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_history = [best_fitness]
        
        iteration = 0
        max_iterations = obj_func.max_evals // self.npop
        
        while not obj_func.exhausted() and iteration < max_iterations:
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            
            for i in range(self.npop):
                if obj_func.exhausted():
                    break
                
                # Select three distinct random individuals
                candidates = [j for j in range(self.npop) if j != i]
                a, b, c = self.rng.choice(candidates, size=3, replace=False)
                
                # Mutation: v = x_a + F * (x_b - x_c)
                v = pop[a] + self.F * (pop[b] - pop[c])
                v = np.clip(v, bounds[0], bounds[1])
                
                # Crossover: binomial
                j_rand = self.rng.integers(0, dim)
                u = pop[i].copy()
                for j in range(dim):
                    if self.rng.random() < self.CR or j == j_rand:
                        u[j] = v[j]
                
                # Selection
                u_fitness = obj_func(u)
                if u_fitness < fitness[i]:
                    new_pop[i] = u
                    new_fitness[i] = u_fitness
            
            pop = new_pop
            fitness = new_fitness
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_pos = pop[best_idx].copy()
                best_fitness = fitness[best_idx]
            
            convergence_history.append(best_fitness)
            iteration += 1
        
        return {
            'best_position': best_pos,
            'best_fitness': best_fitness,
            'convergence_history': convergence_history,
            'iterations': iteration,
            'evals_used': obj_func.evals_used
        }

