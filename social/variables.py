"""Variable type handling: real, integer, categorical."""

import numpy as np
from typing import List, Union, Optional
from dataclasses import dataclass


@dataclass
class VariableSpec:
    """Specification for a variable."""
    var_type: str  # "real", "int", "cat"
    bounds: Optional[List[float]] = None  # [lower, upper] for real/int
    catalog_values: Optional[List[Union[float, int, str]]] = None  # For categorical
    
    def __post_init__(self):
        """Validate specification."""
        if self.var_type == "cat" and self.catalog_values is None:
            raise ValueError("Categorical variables must have catalog_values")
        if self.var_type in ["real", "int"] and self.bounds is None:
            raise ValueError(f"{self.var_type} variables must have bounds")


def repair_variable(x: np.ndarray, specs: List[VariableSpec],
                   mode: str = "round") -> np.ndarray:
    """
    Repair variables according to their specifications.
    
    Args:
        x: Variable vector
        specs: List of VariableSpec for each variable
        mode: Repair mode (round, stochastic_round, integer_ops)
        
    Returns:
        Repaired variable vector
    """
    repaired = x.copy()
    
    for i, spec in enumerate(specs):
        if spec.var_type == "real":
            # Clip to bounds
            if spec.bounds:
                repaired[i] = np.clip(repaired[i], spec.bounds[0], spec.bounds[1])
        
        elif spec.var_type == "int":
            if mode == "round":
                repaired[i] = np.round(repaired[i])
            elif mode == "stochastic_round":
                # Stochastic rounding: round up with probability = fractional part
                floor_val = np.floor(repaired[i])
                frac = repaired[i] - floor_val
                if np.random.random() < frac:
                    repaired[i] = floor_val + 1
                else:
                    repaired[i] = floor_val
            elif mode == "integer_ops":
                # Integer operations: ensure integer values
                repaired[i] = np.round(repaired[i])
            
            # Clip to bounds
            if spec.bounds:
                repaired[i] = np.clip(repaired[i], spec.bounds[0], spec.bounds[1])
                repaired[i] = int(repaired[i])
        
        elif spec.var_type == "cat":
            # Map to nearest catalog value
            if spec.catalog_values:
                distances = [abs(repaired[i] - float(cv)) for cv in spec.catalog_values]
                nearest_idx = np.argmin(distances)
                repaired[i] = spec.catalog_values[nearest_idx]
    
    return repaired


def integer_mutation(x: np.ndarray, specs: List[VariableSpec],
                    mutation_strength: float = 1.0,
                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Integer-aware mutation (jumps by ±1..±m).
    
    Args:
        x: Variable vector
        specs: Variable specifications
        mutation_strength: Mutation strength (max jump)
        rng: Random number generator
        
    Returns:
        Mutated vector
    """
    if rng is None:
        rng = np.random.default_rng()
    
    mutated = x.copy()
    
    for i, spec in enumerate(specs):
        if spec.var_type == "int":
            # Integer jump mutation
            jump = rng.integers(1, int(mutation_strength) + 1)
            direction = rng.choice([-1, 1])
            mutated[i] = int(x[i]) + direction * jump
            
            # Clip to bounds
            if spec.bounds:
                mutated[i] = np.clip(mutated[i], int(spec.bounds[0]), int(spec.bounds[1]))
    
    return mutated

