# Figures Summary

This document lists all required figures and their status.

## All Required Figures

### ✅ Runtime & Centrality Cost
- [x] `fig_runtime_breakdown` - Stacked bar chart (time components vs N)
- [x] `fig_bc_cadence_tradeoff` - Performance/runtime vs bc_interval

### ✅ Centrality Comparison  
- [x] `fig_centrality_rank` - Bar chart (average rank)
- [x] `fig_centrality_runtime` - Bar chart (runtime)

### ✅ Watts-Strogatz (K,p) Sensitivity
- [x] `fig_kp_heatmap_<function>` - Heatmaps for 3+ benchmarks + 1 engineering
- [x] `fig_kp_heatmap_runtime` - Optional runtime heatmap

### ✅ Rewiring Robustness
- [x] `fig_rewiring_convergence` - Best-so-far vs evals
- [x] `fig_rewiring_diversity` - Diversity vs evals

### ✅ Schedule Analysis
- [x] `fig_schedule_curves` - Best-so-far vs evals (multiple schedules)
- [x] `fig_phase_diversity` - Diversity + improvement with sync markers

### ✅ Ablation
- [x] `fig_ablation_bars` - Mean best fitness bars
- [x] `fig_ablation_anytime` - Best-so-far vs evals

### ✅ Engineering Feasibility
- [x] `fig_feasibility_rate` - Feasibility rate bar chart
- [x] `fig_violation_box` - Constraint violation boxplot

### ✅ Discrete Handling
- [x] `fig_discrete_modes` - Rounding modes comparison
- [x] `fig_integer_histograms` - Integer variable histograms

### ✅ Basin Coverage & Topology
- [x] `fig_basins_over_time` - Basins discovered vs evals
- [x] `fig_topology_correlation` - λ₂ vs basin/performance scatter

### ✅ Low-Budget + Surrogate
- [x] `fig_low_budget_anytime` - Best-so-far vs evals (multiple budgets)
- [x] `fig_sample_efficiency` - Final best vs budget

## Implementation Status

**All 20+ figures are implemented** in `social/plotting.py` with:
- ✅ Matplotlib only (no seaborn)
- ✅ PNG (300 DPI) and PDF (vector) output
- ✅ CSV data export
- ✅ Proper labels, units, legends
- ✅ Consistent styling

## Usage

Generate all figures:
```bash
python experiments/generate_figures.py
```

Generate from real data:
```bash
python experiments/generate_figures.py --use_real_data
```

## Output Locations

- **Figures**: `outputs/figures/` (PNG + PDF)
- **Data**: `outputs/data/` (CSV)

## Figure Naming

All figures follow the naming convention:
- `fig_<category>_<type>` (e.g., `fig_runtime_breakdown`)
- Heatmaps: `fig_kp_heatmap_<function_name>` (e.g., `fig_kp_heatmap_Sphere`)

## Integration

To use with real experiment data:

1. Run experiments (benchmarks, ablation, etc.)
2. Load results into appropriate data structures
3. Call plotting functions:
   ```python
   from social.plotting import fig_centrality_rank
   fig_centrality_rank(your_data)
   ```

All figures automatically:
- Save to `outputs/figures/` as PNG and PDF
- Export data to `outputs/data/` as CSV
- Use consistent styling and formatting

