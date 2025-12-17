# SOCIAL Optimizer

SOCIAL (Social Optimization with Centrality and Influence Adaptive Learning) is a graph-based population optimization algorithm.

## Quick options

Fast test run (fewer iterations):
python run_all_experiments.py --quick
python run_all_experiments.py --quick

Skip figures (experiments only):
python run_all_experiments.py --skip_figures
python run_all_experiments.py --skip_figures

Custom parameters:
python run_all_experiments.py --num_runs 20 --max_evals 50000 --functions Sphere Rastrigin Ackley
python run_all_experiments.py --num_runs 20 --max_evals 50000 --functions Sphere Rastrigin Ackley

## Installation

Install required dependencies:
```bash
pip install numpy pandas networkx scipy matplotlib seaborn
```

## Usage

See `QUICK_START.md` for detailed usage instructions and `README_experiments.md` for experiment documentation.

## Key Features

- Graph-based optimization with centrality measures
- Adaptive learning with scheduled weights
- Multiple benchmark functions (23 functions)
- Statistical testing and analysis
- Ablation study support

## Configuration

Edit `social/config.py` to customize algorithm parameters.

## Output

Results are saved to `outputs/` directory:
- `benchmark_results_YYYY-MM-DD.csv`: Consolidated results with Function, Algorithm, Mean, Std columns
- `benchmark_raw_YYYY-MM-DD.json`: Raw results for statistical tests
- `outputs/figures/`: Generated plots (PNG + PDF)

