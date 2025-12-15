"""Configuration module for SOCIAL optimizer."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for SOCIAL optimizer."""
    
    # Population and dimensions
    DIM: int = 30
    NUM_NODES: int = 300
    MAX_EVALS: int = 105000  # NUM_NODES * ITERATIONS default
    
    # Graph parameters (Watts-Strogatz)
    K: int = 5  # Each node connected to K nearest neighbors
    P_BASE: float = 0.1  # Rewiring probability
    
    # Schedule weights (SOCIAL paper Eq. 8-11)
    ALPHA_INIT: float = 0.5  # Initial neighbor weight (decreases)
    BETA_INIT: float = 0.5   # Initial influence weight (decreases)
    GAMMA: float = 0.2       # gbest weight (increases)
    DELTA: float = 0.2       # elite weight (increases)
    
    # Mutation parameters
    MUTATION_RATE_INIT: float = 0.1
    MUTATION_STRENGTH_BASE: float = 0.05
    MUTATION_STRENGTH_MIN: float = 0.01
    
    # Synchronization
    SYNC_WEIGHT_INIT: float = 0.03
    SYNC_INTERVAL: int = 10  # Periodic sync every N iterations
    
    # Success threshold
    SUCCESS_THRESHOLD: float = 1e-8
    
    # Tracking
    TRACKED_DIM: int = 0
    
    # LOTUS (optional, for ablation)
    LOTUS_R0: float = 2.0
    LOTUS_BETA_DROPS: int = 3
    LOTUS_PIT_CONST: float = 40
    LOTUS_LOCAL_FREQ: int = 10
    
    # Seed set size
    SEED_SET_SIZE: Optional[int] = None  # Will be set to 0.1 * NUM_NODES if None
    
    # Centrality computation
    CENTRALITY_MODE: str = "betweenness"  # betweenness, degree, closeness, pagerank, eigenvector
    CENTRALITY_RECOMPUTE: str = "interval"  # always, interval, stagnation
    BC_INTERVAL: int = 5  # Recompute betweenness every N iterations
    
    # Rewiring
    REWIRE_MODE: str = "periodic"  # none, periodic, stagnation, diversity
    REWIRE_INTERVAL: int = 75
    STAGNATION_THRESHOLD: int = 50  # Iterations without improvement
    
    # Schedule mode
    SCHEDULE_MODE: str = "linear"  # linear, exp, cosine, piecewise, bandit_ucb
    
    # Algorithm toggles (for ablation)
    ENABLE_ELITE_MEMORY: bool = True
    ENABLE_MUTATION: bool = True
    ENABLE_SYNC: bool = True
    NEIGHBOR_MODE: str = "centrality_weighted"  # centrality_weighted, uniform
    HYBRID_LOTUS: str = "off"  # off, on
    
    # Statistical tests
    NUM_RUNS: int = 30
    SEED_LIST: Optional[list] = None  # Will be [0..29] if None
    
    def __post_init__(self):
        """Set derived parameters."""
        if self.SEED_SET_SIZE is None:
            self.SEED_SET_SIZE = int(0.1 * self.NUM_NODES)
        if self.SEED_LIST is None:
            self.SEED_LIST = list(range(30))
        if self.MAX_EVALS is None:
            # Default: NUM_NODES evaluations per iteration
            self.MAX_EVALS = self.NUM_NODES * 3500  # Approximate iterations

