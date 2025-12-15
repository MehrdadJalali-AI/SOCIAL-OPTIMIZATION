# Quick Start Guide

## What Was Implemented

This refactoring addresses **all reviewer comments** and implements the SOCIAL algorithm according to the paper specifications.

### Core Changes

1. **SOCIAL Optimizer** (`social/optimizer.py`)
   - ✅ Uses **betweenness centrality** (not composite)
   - ✅ Implements paper equations (8-16) exactly
   - ✅ Scheduled weights: αt, βt decrease; γt, δt increase
   - ✅ Relative fitness influence: I_j^t = 1 - log(1+|f_j|)/log(1+max(f))
   - ✅ Mutation only for worse-than-median nodes
   - ✅ Periodic synchronization every 10 iterations

2. **Budget Parity** (`social/budget.py`)
   - ✅ Strict evaluation counting
   - ✅ Identical seeds for all algorithms
   - ✅ Reproducible results

3. **Statistical Tests** (`social/stats.py`)
   - ✅ Friedman test
   - ✅ Wilcoxon signed-rank with Holm correction
   - ✅ LaTeX table generation

4. **Engineering Problems** (`social/engineering.py`)
   - ✅ 6 constrained problems
   - ✅ Feasibility metrics
   - ✅ Deb's feasibility rules

5. **Baselines** (`baselines/`)
   - ✅ DE, PSO, GWO
   - ✅ All use same budget and seeds

## Quick Usage

### 1. Run SOCIAL on benchmarks:
```bash
python main.py --functions Sphere Rastrigin Ackley --max_evals 105000 --num_runs 30
```

### 2. Compare with baselines:
```bash
python experiments/run_benchmarks.py --max_evals 105000 --num_runs 30 --functions Sphere Rastrigin Ackley
```

### 3. Run statistical tests:
```bash
# First run benchmarks to generate results
python experiments/run_benchmarks.py --max_evals 105000 --num_runs 30

# Then run statistical tests
python experiments/statistical_tests.py --results outputs/benchmark_raw_YYYY-MM-DD.json
```

### 4. Ablation study:
```bash
python experiments/ablation.py --functions Sphere Rastrigin Ackley --num_runs 30
```

### 5. Centrality comparison:
```bash
python experiments/centrality_comparison.py --functions Sphere Rastrigin Ackley --num_runs 30
```

### 6. (K,p) heatmaps:
```bash
python experiments/kp_heatmaps.py --functions Sphere Rastrigin Ackley --num_runs 10
```

## Configuration

Edit `social/config.py` to change:
- `CENTRALITY_MODE`: "betweenness" (default), "degree", "closeness", etc.
- `SCHEDULE_MODE`: "linear" (default), "exp", "cosine", "piecewise"
- `ENABLE_MUTATION`: True/False
- `ENABLE_SYNC`: True/False
- `ENABLE_ELITE_MEMORY`: True/False
- `REWIRE_MODE`: "periodic", "stagnation", "diversity", "none"

## Output Files

All results are saved to `outputs/` directory:
- `benchmark_results_YYYY-MM-DD.csv`: Algorithm comparison
- `benchmark_raw_YYYY-MM-DD.json`: Raw results
- `friedman_summary_YYYY-MM-DD.csv`: Friedman ranks
- `wilcoxon_pairs_YYYY-MM-DD.csv`: Pairwise tests
- `ablation_table.csv`: Ablation results
- `centrality_comparison.csv`: Centrality comparison
- `kp_heatmap_data.csv`: (K,p) sensitivity
- `tables/*.tex`: LaTeX tables

## Key Features

✅ **Paper Compliance**: Matches SOCIAL paper equations exactly  
✅ **Budget Parity**: All algorithms use same evaluation budget  
✅ **Reproducibility**: Fixed seeds ensure reproducibility  
✅ **Statistical Rigor**: Proper tests with corrections  
✅ **Ablation Support**: Easy to remove components  
✅ **Extensible**: Easy to add new functions/problems/algorithms  

## Dependencies

Required packages:
- numpy
- pandas
- networkx
- scipy
- matplotlib (for heatmaps)
- seaborn (for heatmaps)

Install with:
```bash
pip install numpy pandas networkx scipy matplotlib seaborn
```

## Notes

- All algorithms respect the same evaluation budget
- Seeds ensure reproducibility across runs
- Statistical tests use Holm correction by default
- Results are saved incrementally (can resume if interrupted)

