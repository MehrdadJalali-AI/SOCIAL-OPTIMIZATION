"""Budget management for strict evaluation counting."""

import numpy as np
from typing import Callable, Tuple, Optional


class BudgetedObjective:
    """Wrapper for objective function that counts evaluations."""
    
    def __init__(self, func: Callable, max_evals: int, name: str = "objective"):
        """
        Initialize budgeted objective.
        
        Args:
            func: Objective function to wrap
            max_evals: Maximum number of evaluations
            name: Name for logging
        """
        self.func = func
        self.max_evals = max_evals
        self.evals_used = 0
        self.name = name
        self.history = []  # Store (x, f(x)) pairs
        
    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate function and count evaluation.
        
        Args:
            x: Input vector
            
        Returns:
            Function value
            
        Raises:
            RuntimeError: If budget exceeded
        """
        if self.evals_used >= self.max_evals:
            raise RuntimeError(f"Budget exceeded: {self.evals_used}/{self.max_evals} evaluations used")
        
        result = self.func(x)
        self.evals_used += 1
        self.history.append((x.copy(), result))
        return result
    
    def reset(self):
        """Reset evaluation counter."""
        self.evals_used = 0
        self.history = []
    
    def remaining(self) -> int:
        """Get remaining evaluations."""
        return max(0, self.max_evals - self.evals_used)
    
    def exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self.evals_used >= self.max_evals
    
    def get_stats(self) -> dict:
        """Get statistics about evaluations."""
        if not self.history:
            return {
                "evals_used": self.evals_used,
                "max_evals": self.max_evals,
                "best_fitness": None,
                "worst_fitness": None,
                "mean_fitness": None
            }
        
        fitnesses = [f for _, f in self.history]
        return {
            "evals_used": self.evals_used,
            "max_evals": self.max_evals,
            "best_fitness": min(fitnesses),
            "worst_fitness": max(fitnesses),
            "mean_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses)
        }

