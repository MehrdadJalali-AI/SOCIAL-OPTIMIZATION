# Quick Command Reference

## 🚀 Run Everything (One Command)

```bash
python run_all_experiments.py
```

This single command will:
1. Run all benchmark experiments (SOCIAL vs baselines)
2. Run statistical tests
3. Run ablation studies
4. Run centrality comparisons
5. Generate (K,p) heatmaps
6. Generate all required figures

## 📊 What You Get

After running, you'll have:

### Results
- `outputs/benchmark_results_*.csv` - Algorithm comparison
- `outputs/friedman_summary_*.csv` - Statistical test results
- `outputs/ablation_table.csv` - Ablation study
- `outputs/centrality_comparison.csv` - Centrality comparison
- `outputs/kp_heatmap_data.csv` - (K,p) sensitivity

### Figures (PNG + PDF)
- All 20+ required figures in `outputs/figures/`
- Runtime breakdown, centrality comparison, heatmaps, etc.

### Data
- Underlying CSV data for each figure in `outputs/data/`

### LaTeX Tables
- Ready-to-use tables in `outputs/tables/`

## ⚡ Quick Options

### Fast Test Run
```bash
python run_all_experiments.py --quick
```
- Fewer runs (10 instead of 30)
- Smaller function subset
- ~30-60 minutes instead of 2-4 hours

### Skip Figures (Experiments Only)
```bash
python run_all_experiments.py --skip_figures
```

### Custom Parameters
```bash
python run_all_experiments.py --num_runs 20 --max_evals 50000
```

## 📝 Alternative: Bash Script

```bash
./run_all.sh
```

## ⏱️ Estimated Time

- **Full run**: 2-4 hours (30 runs, all functions)
- **Quick run**: 30-60 minutes (10 runs, 3 functions)

## 📁 Output Location

All results go to `outputs/` directory:
- `outputs/figures/` - All figures
- `outputs/data/` - Figure data
- `outputs/tables/` - LaTeX tables
- `outputs/*.csv` - Results tables

## 🔍 Check Progress

The script prints progress for each step. If a step fails, you can:
1. Check the error message
2. Re-run individual experiments if needed
3. Results are saved incrementally

## 📚 More Details

- See `RUN_ALL.md` for detailed usage
- See `README_experiments.md` for individual experiment guides
- See `PLOTTING_GUIDE.md` for figure generation details

