# SOCIAL Experiments Guide

This document describes how to run the various experiments for the SOCIAL optimizer.

## Quick Start: Run Everything

**To run all experiments and generate all results/figures with one command:**

```bash
python run_all_experiments.py
```

Or use the bash wrapper:
```bash
./run_all.sh
```

For a quick test run (fewer iterations):
```bash
python run_all_experiments.py --quick
```

See `RUN_ALL.md` for detailed usage and options.

## Package Structure

```
SOCAIL-Code-V1/
├── social/              # Core SOCIAL package
│   ├── config.py       # Configuration
│   ├── budget.py       # Budget management
│   ├── functions.py    # Benchmark functions
│   ├── engineering.py  # Engineering problems
│   ├── graph_ops.py    # Graph operations
│   ├── optimizer.py    # SOCIAL optimizer
│   ├── stats.py        # Statistical tests
│   └── variables.py    # Variable type handling
├── baselines/          # Baseline algorithms
│   ├── de.py          # Differential Evolution
│   ├── pso.py         # Particle Swarm Optimization
│   └── gwo.py         # Grey Wolf Optimizer
├── experiments/        # Experiment scripts
│   ├── run_benchmarks.py
│   └── statistical_tests.py
└── main.py            # Main entry point
```

## Running Experiments

### 1. Benchmark Functions

Run SOCIAL on benchmark functions:

```bash
python main.py --functions Sphere Rastrigin Ackley --max_evals 105000 --num_runs 30
```

Or run all functions:
```bash
python main.py --max_evals 105000 --num_runs 30
```

### 2. Benchmark Comparison (SOCIAL vs Baselines)

Run comparison experiments:

```bash
python experiments/run_benchmarks.py --max_evals 105000 --num_runs 30 --functions Sphere Rastrigin Ackley
```

This will:
- Run SOCIAL, DE, PSO, and GWO on specified functions
- Save results to `outputs/benchmark_results_YYYY-MM-DD.csv`
- Save raw results to `outputs/benchmark_raw_YYYY-MM-DD.json`

### 3. Statistical Tests

After running benchmarks, run statistical tests:

```bash
python experiments/statistical_tests.py --results outputs/benchmark_raw_YYYY-MM-DD.json
```

This generates:
- `outputs/friedman_summary_YYYY-MM-DD.csv`
- `outputs/wilcoxon_pairs_YYYY-MM-DD.csv`
- `outputs/tables/statistical_tables_YYYY-MM-DD.tex`

## Configuration

Key parameters in `social/config.py`:

- `NUM_NODES`: Population size (default: 300)
- `K`: Watts-Strogatz graph parameter (default: 5)
- `P_BASE`: Rewiring probability (default: 0.1)
- `CENTRALITY_MODE`: "betweenness" (paper default), "degree", "closeness", etc.
- `SCHEDULE_MODE`: "linear" (paper default), "exp", "cosine", "piecewise"
- `ENABLE_MUTATION`: Enable mutation (default: True)
- `ENABLE_SYNC`: Enable synchronization (default: True)
- `ENABLE_ELITE_MEMORY`: Enable elite memory (default: True)

## Budget Parity

All algorithms use `BudgetedObjective` wrapper to ensure:
- Strict evaluation counting
- Identical budgets across algorithms
- Reproducible results with seeds

## Output Files

### Benchmark Results
- `qualitative_results/<Function>/metrics.csv`: Per-function metrics
- `qualitative_results/<Function>/convergence.csv`: Convergence history
- `social_results_YYYY-MM-DD.csv`: Summary table

### Comparison Results
- `outputs/benchmark_results_YYYY-MM-DD.csv`: Algorithm comparison
- `outputs/benchmark_raw_YYYY-MM-DD.json`: Raw results for statistical tests

### Statistical Tests
- `outputs/friedman_summary_YYYY-MM-DD.csv`: Friedman test ranks
- `outputs/wilcoxon_pairs_YYYY-MM-DD.csv`: Pairwise Wilcoxon tests
- `outputs/tables/statistical_tables_YYYY-MM-DD.tex`: LaTeX tables

## Reproducibility

All experiments use:
- Fixed seed lists: `[0, 1, 2, ..., num_runs-1]`
- `numpy.random.Generator` for all random operations
- Budgeted objectives for fair comparison

## Engineering Problems

To run engineering problems (coming soon):

```bash
python experiments/run_engineering.py --problems Pressure_Vessel Gear_Train
```

## Additional Experiments

Other experiment scripts (to be implemented):
- `experiments/ablation.py`: Ablation studies
- `experiments/centrality_comparison.py`: Compare centrality measures
- `experiments/kp_heatmaps.py`: (K,p) sensitivity analysis
- `experiments/rewiring_ablation.py`: Rewiring strategies
- `experiments/schedule_comparison.py`: Schedule alternatives
- `experiments/low_budget_surrogate.py`: Surrogate-assisted SOCIAL
- `experiments/basin_metrics.py`: Basin crossing analysis

## Notes

- Results are saved incrementally (checkpointing)
- All algorithms respect the same evaluation budget
- Seeds ensure reproducibility across runs
- Statistical tests use Holm correction by default

