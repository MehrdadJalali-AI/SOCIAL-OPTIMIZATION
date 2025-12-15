# Plotting Guide

This guide explains how to generate all required figures for the SOCIAL paper.

## Overview

All plotting functions are in `social/plotting.py`. Each function:
- Generates a matplotlib figure
- Saves as both PNG (300 DPI) and PDF (vector)
- Exports underlying data as CSV
- Uses consistent styling and formatting

## Figure Generation

### Quick Start

Generate all figures with example data:
```bash
python experiments/generate_figures.py
```

Generate figures from real experiment data:
```bash
python experiments/generate_figures.py --use_real_data --data_dir outputs
```

### Individual Figures

You can also generate individual figures programmatically:

```python
from social.plotting import *

# Runtime breakdown
runtime_data = {
    100: {'t_eval': 0.01, 't_centrality': 0.005, ...},
    200: {'t_eval': 0.02, 't_centrality': 0.01, ...},
    ...
}
fig_runtime_breakdown(runtime_data)

# Centrality comparison
centrality_results = {
    'betweenness': {'avg_rank': 1.5, 'avg_runtime': 2.5},
    'degree': {'avg_rank': 2.8, 'avg_runtime': 0.5},
    ...
}
fig_centrality_rank(centrality_results)
fig_centrality_runtime(centrality_results)
```

## Required Figures

### 1. Runtime & Centrality Cost

**fig_runtime_breakdown**
- Stacked bar chart showing time components (eval, centrality, update, mutation, rewiring)
- X-axis: Population sizes N={100,200,500,1000}
- Y-axis: Average time per iteration (seconds)

**fig_bc_cadence_tradeoff**
- Two subplots: performance vs bc_interval and runtime vs bc_interval
- Shows trade-off between recomputation frequency and performance

### 2. Centrality Comparison

**fig_centrality_rank**
- Bar chart of average rank per centrality measure
- Centralities: betweenness, degree, closeness, pagerank, eigenvector

**fig_centrality_runtime**
- Bar chart of runtime per centrality measure
- Shows computational cost comparison

### 3. Watts-Strogatz (K,p) Sensitivity

**fig_kp_heatmap_<function>**
- Heatmap of mean best fitness for (K,p) combinations
- K values: {2,4,6,8,10}
- p values: {0.0,0.1,0.3,0.5,0.8}
- Generate for at least 3 benchmark functions + 1 engineering problem

**fig_kp_heatmap_runtime**
- Optional heatmap of runtime for (K,p) combinations

### 4. Rewiring Robustness

**fig_rewiring_convergence**
- Best-so-far vs evaluations for different rewiring modes
- Modes: none, periodic, stagnation, diversity

**fig_rewiring_diversity**
- Population diversity vs evaluations for rewiring modes

### 5. Schedule Analysis

**fig_schedule_curves**
- Best-so-far vs evaluations for different schedules
- Schedules: linear, exp, cosine, piecewise, bandit

**fig_phase_diversity**
- Two subplots: diversity and improvement over time
- Vertical lines mark synchronization events

### 6. Ablation

**fig_ablation_bars**
- Horizontal bar chart of mean best fitness (or avg rank) per variant
- Variants: full, no_elite, no_mutation, no_sync, uniform_neighbors, etc.

**fig_ablation_anytime**
- Best-so-far vs evaluations for key ablation variants

### 7. Engineering Feasibility

**fig_feasibility_rate**
- Grouped bar chart of feasibility rate per problem and algorithm
- Problems: Pressure_Vessel, Welded_Beam, Gear_Train, etc.
- Algorithms: SOCIAL, DE, PSO

**fig_violation_box**
- Boxplot of constraint violation distribution per problem

### 8. Discrete Handling

**fig_discrete_modes**
- Two subplots: objective comparison and feasibility comparison
- Modes: round, stochastic_round, integer_ops

**fig_integer_histograms**
- Histograms of selected integer variables
- Shows uniform distribution (no rounding bias)

### 9. Basin Coverage & Topology

**fig_basins_over_time**
- Line plot of number of basins discovered vs evaluations
- Shows exploration capability

**fig_topology_correlation**
- Two scatter plots: λ₂ vs basin count and λ₂ vs performance
- Color-coded by performance/basin count

### 10. Low-Budget + Surrogate

**fig_low_budget_anytime**
- Best-so-far vs evaluations at different budgets
- Budgets: {1k, 2k, 5k, 10k}

**fig_sample_efficiency**
- Final best fitness vs budget (log-log scale)
- Shows sample efficiency

## Data Format

All figures expect data in specific formats:

### Runtime Breakdown
```python
{
    N: {
        't_eval': float,
        't_centrality': float,
        't_update': float,
        't_mut': float,
        't_rewire': float
    }
}
```

### Centrality Results
```python
{
    'centrality_name': {
        'avg_rank': float,
        'avg_runtime': float
    }
}
```

### Convergence Data
```python
{
    'mode_name': [
        (evaluation, fitness),
        ...
    ]
}
```

### (K,p) Data
```python
pd.DataFrame({
    'K': [2, 2, 4, ...],
    'p': [0.0, 0.1, 0.0, ...],
    'Mean': [fitness, ...]
})
```

## Output Structure

```
outputs/
├── figures/
│   ├── fig_runtime_breakdown.png
│   ├── fig_runtime_breakdown.pdf
│   ├── fig_centrality_rank.png
│   ├── fig_centrality_rank.pdf
│   └── ...
└── data/
    ├── fig_runtime_breakdown_data.csv
    ├── fig_centrality_rank_data.csv
    └── ...
```

## Integration with Experiments

To generate figures from actual experiment results:

1. Run experiments (benchmarks, ablation, etc.)
2. Collect results into appropriate data structures
3. Call plotting functions with real data
4. Figures are automatically saved with data

Example integration:
```python
# After running centrality comparison experiment
results = load_centrality_results('outputs/centrality_comparison.csv')
centrality_results = process_for_plotting(results)
fig_centrality_rank(centrality_results)
```

## Styling

All figures use consistent styling:
- Font size: 10pt (base), 11pt (labels), 12pt (titles)
- DPI: 300 for PNG, vector for PDF
- Colors: Colorblind-friendly palettes
- Grid: Light gray, alpha=0.3
- Legends: Upper left or best position
- Labels: Bold, with units where applicable

## Notes

- All figures are saved with `bbox_inches='tight'` for proper margins
- Data is always exported as CSV for reproducibility
- Figures use log scales where appropriate (fitness, budgets)
- Heatmaps use viridis/plasma colormaps (colorblind-friendly)

