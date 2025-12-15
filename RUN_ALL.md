# Run All Experiments - Quick Guide

## Single Command to Run Everything

Run all experiments and generate all results/figures with one command:

```bash
python run_all_experiments.py
```

This will:
1. ✅ Run benchmark experiments (SOCIAL vs DE/PSO/GWO)
2. ✅ Run statistical tests (Friedman + Wilcoxon)
3. ✅ Run ablation studies
4. ✅ Run centrality comparisons
5. ✅ Generate (K,p) sensitivity heatmaps
6. ✅ Generate all required figures

## Options

### Quick Mode (Faster, Smaller Scale)
```bash
python run_all_experiments.py --quick
```
- Uses fewer runs (10 instead of 30)
- Tests subset of functions (Sphere, Rastrigin, Ackley)
- Faster for testing/debugging

### Skip Figure Generation
```bash
python run_all_experiments.py --skip_figures
```
- Only runs experiments, doesn't generate figures
- Useful if you want to generate figures separately later

### Custom Parameters
```bash
python run_all_experiments.py --num_runs 20 --max_evals 50000 --functions Sphere Rastrigin Ackley
```

### Custom Output Directory
```bash
python run_all_experiments.py --outdir my_results
```

## What Gets Generated

### Results Files
- `outputs/benchmark_results_YYYY-MM-DD.csv` - Algorithm comparison
- `outputs/benchmark_raw_YYYY-MM-DD.json` - Raw results for stats
- `outputs/friedman_summary_YYYY-MM-DD.csv` - Friedman test ranks
- `outputs/wilcoxon_pairs_YYYY-MM-DD.csv` - Pairwise Wilcoxon tests
- `outputs/ablation_table.csv` - Ablation results
- `outputs/centrality_comparison.csv` - Centrality comparison
- `outputs/kp_heatmap_data.csv` - (K,p) sensitivity data

### Figures (PNG + PDF)
- `outputs/figures/fig_runtime_breakdown.*`
- `outputs/figures/fig_centrality_rank.*`
- `outputs/figures/fig_kp_heatmap_*.png/pdf`
- `outputs/figures/fig_rewiring_convergence.*`
- `outputs/figures/fig_schedule_curves.*`
- `outputs/figures/fig_ablation_bars.*`
- And all other required figures...

### Data Files
- `outputs/data/fig_*_data.csv` - Underlying data for each figure

### LaTeX Tables
- `outputs/tables/statistical_tables_YYYY-MM-DD.tex`
- `outputs/tables/ablation_table.tex`

## Estimated Runtime

- **Full run** (30 runs, all functions): ~2-4 hours (depending on hardware)
- **Quick run** (10 runs, 3 functions): ~30-60 minutes

## Step-by-Step (If Needed)

If you want to run steps individually:

```bash
# 1. Benchmarks
python experiments/run_benchmarks.py --max_evals 105000 --num_runs 30

# 2. Statistical tests
python experiments/statistical_tests.py --results outputs/benchmark_raw_YYYY-MM-DD.json

# 3. Ablation
python experiments/ablation.py --num_runs 30

# 4. Centrality comparison
python experiments/centrality_comparison.py --num_runs 30

# 5. (K,p) heatmaps
python experiments/kp_heatmaps.py --num_runs 10

# 6. Generate figures
python experiments/generate_figures.py
```

## Troubleshooting

### If a step fails:
- Check the error message
- You can re-run individual steps
- Results are saved incrementally

### If you want to resume:
- The script saves results as it goes
- You can skip completed steps by running individual scripts

### Memory issues:
- Use `--quick` mode
- Reduce `--num_runs`
- Test with fewer functions first

## Output Structure

```
outputs/
├── benchmark_results_YYYY-MM-DD.csv
├── benchmark_raw_YYYY-MM-DD.json
├── friedman_summary_YYYY-MM-DD.csv
├── wilcoxon_pairs_YYYY-MM-DD.csv
├── ablation_table.csv
├── centrality_comparison.csv
├── kp_heatmap_data.csv
├── figures/
│   ├── fig_*.png
│   └── fig_*.pdf
├── data/
│   └── fig_*_data.csv
└── tables/
    └── *.tex
```

## Notes

- All experiments use the same seeds for reproducibility
- Budget parity is enforced across all algorithms
- Results are timestamped for tracking
- Figures are saved as both PNG (300 DPI) and PDF (vector)

