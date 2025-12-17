"""Configuration module for SOCIAL optimizer."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


# Function-aware population sizing: NUM_NODES by function category
# This is an experimental design choice, NOT adaptive population dynamics
FUNCTION_CATEGORIES = {
    # Unimodal functions → NUM_NODES = 60
    "unimodal": {
        "Sphere": 60,
        "Schwefel_2_22": 60,
        "Schwefel_1_2": 60,
        "Schwefel_2_21": 60,
        "Rosenbrock": 60,
        "Quartic": 60,
    },
    # Discontinuous (step-like) → NUM_NODES = 150
    "discontinuous": {
        "Step": 150,
    },
    # Multimodal functions → NUM_NODES = 150
    "multimodal": {
        "Schwefel_2_26": 150,
        "Rastrigin": 150,
        "Ackley": 150,
        "Griewank": 150,
        "Foxholes": 150,
        "Kowalik": 150,
        "Camel-Back": 150,
        "Branin": 150,
        "Goldstein-Price": 150,
        "Hartman": 150,
        "Shekel1": 150,  # Maps to hartman6
        "Shekel2": 150,  # Maps to shekel5
        "Shekel3": 150,  # Maps to shekel7
        "Shekel4": 150,  # Maps to shekel10
    },
    # Penalized/constrained → NUM_NODES = 200
    "penalized": {
        "Penalized": 200,
        "Penalized2": 200,
    },
    # Graph-structured (non-classical) → NUM_NODES = 120
    "graph": {
        "Graph-Laplacian": 120,
    }
}

# NUM_NODES by category (for reference)
NUM_NODES_BY_CATEGORY = {
    "unimodal": 60,
    "discontinuous": 150,
    "multimodal": 150,
    "penalized": 200,
    "graph": 120,
}


# Preset configurations
PRESETS = {
    "SOCIAL_balanced": {
        # Default preset: robust and multimodal performance
        "NUM_NODES": 60,
        "K": 5,
        "P_BASE": 0.1,
        "MUTATION_RATE_INIT": 0.1,
        "MUTATION_STRENGTH_BASE": 0.05,
        "MUTATION_STRENGTH_MIN": 0.005,  # Refined: reduced final mutation strength
        "SYNC_WEIGHT_INIT": 0.03,
        "LOTUS_R0": 2.0,
        "LOTUS_BETA_DROPS": 3,
        "LOTUS_PIT_CONST": 13,
        "LOTUS_LOCAL_FREQ": 10,
        "SUCCESS_THRESHOLD": 1e-6,
        # Refined: stronger initial self-weight (INITIAL_WEIGHT = 0.6)
        # This is achieved by adjusting schedule weights so self_weight = 1 - (alpha+beta+gamma+delta) ≈ 0.6 initially
        "ALPHA_INIT": 0.2,  # Reduced to allow higher self-weight
        "BETA_INIT": 0.1,   # Reduced to allow higher self-weight
        "GAMMA_INIT": 0.05,
        "DELTA_INIT": 0.05,
    },
    "SOCIAL_fast": {
        # Fast preset: quick convergence for unimodal-heavy problems
        "NUM_NODES": 120,
        "K": 4,
        "P_BASE": 0.07,
        "MUTATION_RATE_INIT": 0.08,
        "MUTATION_STRENGTH_BASE": 0.04,
        "MUTATION_STRENGTH_MIN": 0.008,
        "SYNC_WEIGHT_INIT": 0.05,
        "LOTUS_R0": 1.6,
        "LOTUS_BETA_DROPS": 2,
        "LOTUS_PIT_CONST": 10,
        "LOTUS_LOCAL_FREQ": 6,
        "SUCCESS_THRESHOLD": 1e-7,
    }
}


@dataclass
class Config:
    """Configuration for SOCIAL optimizer."""
    
    # Preset selection
    PRESET: str = "SOCIAL_balanced"  # SOCIAL_balanced, SOCIAL_fast
    
    # Population and dimensions
    DIM: int = 30
    NUM_NODES: int = 60  # Reduced from 300 for better performance
    MAX_EVALS: int = 105000  # NUM_NODES * ITERATIONS default
    
    # Graph parameters (Watts-Strogatz)
    K: int = 5  # Each node connected to K nearest neighbors
    P_BASE: float = 0.1  # Rewiring probability
    
    # Schedule weights (SOCIAL paper Eq. 8-11) - strengthened exploitation
    ALPHA_INIT: float = 0.6   # Initial neighbor weight (decreases)
    ALPHA_FINAL: float = 0.15  # Final neighbor weight
    BETA_INIT: float = 0.25   # Initial influence weight (decreases)
    BETA_FINAL: float = 0.15  # Final influence weight
    GAMMA_INIT: float = 0.05  # Initial gbest weight (increases)
    GAMMA_FINAL: float = 0.45  # Final gbest weight
    DELTA_INIT: float = 0.05  # Initial elite weight (increases)
    DELTA_FINAL: float = 0.25  # Final elite weight
    # Legacy support (will be computed from INIT/FINAL)
    GAMMA: float = 0.2       # Deprecated, use GAMMA_INIT/GAMMA_FINAL
    DELTA: float = 0.2       # Deprecated, use DELTA_INIT/DELTA_FINAL
    
    # Mutation parameters - reduced late-stage randomness
    MUTATION_RATE_INIT: float = 0.1  # Initial mutation rate (from preset)
    MUTATION_RATE_FINAL: float = 0.02  # Final mutation rate
    MUTATION_SCHEDULE: str = "exp"  # exp, linear, cosine
    MUTATION_STRENGTH_BASE: float = 0.05  # Base mutation strength (from preset)
    MUTATION_STRENGTH_MIN: float = 0.005  # Minimum mutation strength (refined from preset)
    
    # Synchronization
    SYNC_WEIGHT_INIT: float = 0.03
    SYNC_INTERVAL: int = 10  # Periodic sync every N iterations
    
    # Success threshold - from preset
    SUCCESS_THRESHOLD: float = 1e-6  # From preset (was 1e-8)
    
    # Tracking
    TRACKED_DIM: int = 0
    
    # LOTUS (optional, for ablation) - values from preset
    LOTUS_R0: float = 2.0  # From preset
    LOTUS_BETA_DROPS: int = 3  # From preset
    LOTUS_PIT_CONST: float = 13  # From preset (was 40)
    LOTUS_LOCAL_FREQ: int = 10  # From preset
    
    # Seed set size
    SEED_SET_SIZE: Optional[int] = None  # Will be set to 0.1 * NUM_NODES if None
    
    # Centrality computation - stabilized graph dynamics
    CENTRALITY_MODE: str = "betweenness"  # betweenness, degree, closeness, pagerank, eigenvector
    CENTRALITY_RECOMPUTE: str = "interval"  # always, interval, stagnation
    CENTRALITY_INTERVAL: int = 5  # Recompute centrality every N iterations
    BC_INTERVAL: int = 5  # Alias for CENTRALITY_INTERVAL (for backward compatibility)
    
    # Rewiring - stabilized graph dynamics
    REWIRE_MODE: str = "periodic"  # none, periodic, stagnation, diversity
    REWIRE_INTERVAL: int = 75  # Will be computed as int(0.15 * MAX_ITERS) if needed
    REWIRE_PROB: float = 0.05  # Rewiring probability (reduced from default)
    STAGNATION_THRESHOLD: int = 50  # Iterations without improvement
    
    # Schedule mode
    SCHEDULE_MODE: str = "linear"  # linear, exp, cosine, piecewise, bandit_ucb
    
    # Algorithm toggles (for ablation)
    ENABLE_ELITE_MEMORY: bool = True
    ENABLE_MUTATION: bool = True
    ENABLE_SYNC: bool = True
    NEIGHBOR_MODE: str = "centrality_weighted"  # centrality_weighted, uniform
    HYBRID_LOTUS: str = "off"  # off, on
    
    # Fitness influence mode
    INFLUENCE_MODE: str = "rank"  # rank, minmax
    
    # Learning rate schedule (for stabilizing updates)
    ETA_INIT: float = 0.9  # Initial learning rate
    ETA_MIN: float = 0.2   # Minimum learning rate
    ETA_SCHEDULE: str = "linear"  # linear, exp, cosine
    
    # Boundary handling
    BOUNDARY_MODE: str = "reflect"  # clip, reflect
    
    # Hybrid exploitation (SOCIAL++)
    SOCIALPP_MODE: bool = False  # Enable DE/current-to-best exploitation
    DE_Q: float = 0.5  # Fraction of worst nodes to consider
    DE_F: float = 0.5  # Differential evolution scaling factor
    DE_CR: float = 0.9  # Crossover probability
    DE_P_INIT: float = 0.35  # Initial probability of DE step
    DE_P_MIN: float = 0.05  # Minimum probability of DE step
    
    # Stagnation response
    RESEED_FRAC: float = 0.2  # Fraction of nodes to reseed
    ELITE_KEEP_FRAC: float = 0.1  # Fraction of best nodes to keep unchanged
    OPPOSITION_FRAC: float = 0.25  # Fraction of reseeded nodes using opposition
    
    # Progress reporting
    PROGRESS_INTERVAL: int = 5  # Report progress every N iterations or 5% of budget
    SHOW_PROGRESS: bool = True  # Enable progress reporting
    PROGRESS_MODE: str = "function"  # "function" or "iteration" - function-level only for experiments
    
    # Statistical tests
    NUM_RUNS: int = 30
    SEED_LIST: Optional[list] = None  # Will be [0..29] if None
    
    # ========================================================================
    # SOCIAL REFINEMENTS (Pure Social Design Improvements)
    # ========================================================================
    
    # A) Adaptive Social Influence
    ENABLE_ADAPTIVE_SOCIAL: bool = True  # Enable time-dependent social influence
    SOCIAL_INFLUENCE_INIT: float = 0.3  # Initial social influence (weak, for exploration)
    SOCIAL_INFLUENCE_FINAL: float = 0.7  # Final social influence (strong, for exploitation)
    SOCIAL_INFLUENCE_SCHEDULE: str = "sigmoid"  # linear, sigmoid, exp
    
    # B) Elite-Consensus Guidance
    ENABLE_ELITE_CONSENSUS: bool = True  # Enable elite centroid guidance
    ELITE_FRAC: float = 0.15  # Fraction of top agents (10-20% of population)
    ELITE_CONSENSUS_WEIGHT_INIT: float = 0.02  # Small initial weight
    ELITE_CONSENSUS_WEIGHT_FINAL: float = 0.08  # Small final weight (non-dominant)
    
    # C) Degree-Centrality Normalization (already implemented, but add safeguard)
    CENTRALITY_NORMALIZATION_MODE: str = "softmax"  # softmax, l2_norm, none
    CENTRALITY_TEMPERATURE: float = 1.0  # Temperature for softmax normalization
    
    # D) Diversity-Aware Micro-Perturbation
    ENABLE_DIVERSITY_PERTURBATION: bool = True  # Enable diversity monitoring
    DIVERSITY_THRESHOLD: float = 0.01  # Threshold for population diversity (relative to search range)
    DIVERSITY_PERTURBATION_SCALE: float = 0.005  # Scale of Gaussian noise (very small)
    DIVERSITY_CHECK_INTERVAL: int = 5  # Check diversity every N iterations
    
    # E) Boundary-Handling Refinement
    BOUNDARY_DAMPING_FACTOR: float = 0.5  # Damping factor for boundary reflection (0.5 = half reflection)
    ENABLE_DAMPED_BOUNDARY: bool = True  # Use damped reflection instead of hard clipping
    
    # Staged Execution
    RUN_FULL_BENCHMARK: bool = False  # If False, run validation stage only (Sphere, F3, F6, F12)
    VALIDATION_FUNCTIONS: Optional[List[str]] = None  # Will be set to ['Sphere', 'Schwefel_1_2', 'Step', 'Penalized'] if None
    
    @staticmethod
    def get_num_nodes_for_function(function_name: str) -> int:
        """
        Get NUM_NODES for a given function based on its category.
        
        FUNCTION-AWARE POPULATION SIZING:
        This selects NUM_NODES BEFORE optimization based on function category.
        NUM_NODES remains fixed throughout the entire run.
        
        Args:
            function_name: Name of the benchmark function
            
        Returns:
            NUM_NODES value for this function category
        """
        # Search through all categories
        for category, functions in FUNCTION_CATEGORIES.items():
            if function_name in functions:
                return functions[function_name]
        
        # Default fallback (should not happen for known functions)
        print(f"Warning: Function '{function_name}' not found in categories. Using default NUM_NODES=150")
        return 150
    
    @staticmethod
    def get_function_category(function_name: str) -> str:
        """
        Get the category of a function.
        
        Args:
            function_name: Name of the benchmark function
            
        Returns:
            Category name ("unimodal", "multimodal", "penalized", "discontinuous", "graph")
        """
        for category, functions in FUNCTION_CATEGORIES.items():
            if function_name in functions:
                return category
        return "unknown"
    
    def __post_init__(self):
        """Set derived parameters and apply preset."""
        # Apply preset if specified
        if self.PRESET in PRESETS:
            preset_values = PRESETS[self.PRESET]
            for key, value in preset_values.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        if self.SEED_SET_SIZE is None:
            self.SEED_SET_SIZE = int(0.1 * self.NUM_NODES)
        if self.SEED_LIST is None:
            self.SEED_LIST = list(range(30))
        if self.MAX_EVALS is None:
            # Default: NUM_NODES evaluations per iteration
            # For preset: NUM_NODES=200, ITERATIONS=2500 → MAX_EVALS = 200 * 2500 = 500000
            # But we'll keep default as NUM_NODES * 3500 for backward compatibility
            self.MAX_EVALS = self.NUM_NODES * 3500  # Approximate iterations
        
        # Sync BC_INTERVAL with CENTRALITY_INTERVAL
        if not hasattr(self, 'BC_INTERVAL') or self.BC_INTERVAL != self.CENTRALITY_INTERVAL:
            self.BC_INTERVAL = self.CENTRALITY_INTERVAL
        
        # Set validation functions if not specified
        if self.VALIDATION_FUNCTIONS is None:
            self.VALIDATION_FUNCTIONS = ['Sphere', 'Schwefel_1_2', 'Step', 'Penalized']
        
        # Compute REWIRE_INTERVAL if needed (will be updated in optimizer based on max_iterations)
        # Default is set above, but can be recomputed as int(0.15 * MAX_ITERS) at runtime

