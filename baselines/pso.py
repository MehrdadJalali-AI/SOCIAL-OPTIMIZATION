"""Particle Swarm Optimization (PSO) baseline."""

import numpy as np
from typing import List, Optional
from social.budget import BudgetedObjective


class ParticleSwarmOptimization:
    """PSO with inertia weight."""
    
    def __init__(self, npop: int = 50, w: float = 0.7, c1: float = 1.5,
                 c2: float = 1.5, rng: Optional[np.random.Generator] = None):
        """
        Initialize PSO.
        
        Args:
            npop: Population size
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            rng: Random number generator
        """
        self.npop = npop
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def optimize(self, obj_func: BudgetedObjective, bounds: List[float],
                dim: int, seed: Optional[int] = None) -> dict:
        """
        Run PSO optimization.
        
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
        velocity = self.rng.uniform(-1, 1, (self.npop, dim))
        fitness = np.array([obj_func(ind) for ind in pop])
        
        pbest = pop.copy()
        pbest_fitness = fitness.copy()
        
        gbest_idx = np.argmin(fitness)
        gbest_pos = pop[gbest_idx].copy()
        gbest_fitness = fitness[gbest_idx]
        
        convergence_history = [gbest_fitness]
        
        iteration = 0
        max_iterations = obj_func.max_evals // self.npop
        
        while not obj_func.exhausted() and iteration < max_iterations:
            for i in range(self.npop):
                if obj_func.exhausted():
                    break
                
                # Update velocity
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                velocity[i] = (self.w * velocity[i] +
                              self.c1 * r1 * (pbest[i] - pop[i]) +
                              self.c2 * r2 * (gbest_pos - pop[i]))
                
                # Update position
                pop[i] = pop[i] + velocity[i]
                pop[i] = np.clip(pop[i], bounds[0], bounds[1])
                
                # Evaluate
                fitness[i] = obj_func(pop[i])
                
                # Update pbest
                if fitness[i] < pbest_fitness[i]:
                    pbest[i] = pop[i].copy()
                    pbest_fitness[i] = fitness[i]
                    
                    # Update gbest
                    if fitness[i] < gbest_fitness:
                        gbest_pos = pop[i].copy()
                        gbest_fitness = fitness[i]
            
            convergence_history.append(gbest_fitness)
            iteration += 1
        
        return {
            'best_position': gbest_pos,
            'best_fitness': gbest_fitness,
            'convergence_history': convergence_history,
            'iterations': iteration,
            'evals_used': obj_func.evals_used
        }

