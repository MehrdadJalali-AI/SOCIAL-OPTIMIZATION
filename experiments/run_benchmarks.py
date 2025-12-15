"""Run benchmark functions with SOCIAL and baselines."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from datetime import datetime
from social.config import Config
from social.functions import BenchmarkFunctions
from social.budget import BudgetedObjective
from social.optimizer import SOCIALOptimizer
from baselines.de import DifferentialEvolution
from baselines.pso import ParticleSwarmOptimization
from baselines.gwo import GreyWolfOptimizer


def run_experiment(function_names: list = None, max_evals: int = 105000,
                  num_runs: int = 30, seeds: list = None, outdir: str = "outputs"):
    """
    Run benchmark experiments with SOCIAL and baselines.
    
    Args:
        function_names: List of function names (None = all)
        max_evals: Maximum evaluations
        num_runs: Number of runs
        seeds: List of seeds (None = [0..num_runs-1])
        outdir: Output directory
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Configuration
    config = Config()
    config.MAX_EVALS = max_evals
    config.NUM_RUNS = num_runs
    if seeds is None:
        seeds = list(range(num_runs))
    config.SEED_LIST = seeds
    
    # Functions
    functions = BenchmarkFunctions()
    if function_names is None:
        function_names = list(functions.functions.keys())
    
    # Algorithms
    algorithms = {
        'SOCIAL': SOCIALOptimizer(config),
        'DE': DifferentialEvolution(npop=50),
        'PSO': ParticleSwarmOptimization(npop=50),
        'GWO': GreyWolfOptimizer(npop=50)
    }
    
    # Results storage
    all_results = {alg: {func: [] for func in function_names} for alg in algorithms.keys()}
    
    print(f"Running experiments on {len(function_names)} functions with {num_runs} runs each")
    print(f"Max evaluations: {max_evals}")
    
    for func_name in function_names:
        print(f"\n{'='*60}")
        print(f"Function: {func_name}")
        print(f"{'='*60}")
        
        func, bounds, optimum = functions.functions[func_name]
        dim = config.DIM
        
        for alg_name, optimizer in algorithms.items():
            print(f"\n  Algorithm: {alg_name}")
            
            for run_idx in range(num_runs):
                seed = seeds[run_idx]
                rng = np.random.default_rng(seed)
                
                # Create budgeted objective
                obj_func = BudgetedObjective(func, max_evals, name=f"{func_name}_{alg_name}_{run_idx}")
                
                # Run optimizer
                if alg_name == 'SOCIAL':
                    optimizer.rng = rng
                    result = optimizer.optimize(obj_func, bounds, dim, seed=seed)
                else:
                    optimizer.rng = rng
                    result = optimizer.optimize(obj_func, bounds, dim, seed=seed)
                
                best_fitness = result['best_fitness']
                all_results[alg_name][func_name].append(best_fitness)
                
                if (run_idx + 1) % 10 == 0:
                    print(f"    Run {run_idx+1}/{num_runs}: Best = {best_fitness:.6e}")
            
            # Summary
            fitnesses = all_results[alg_name][func_name]
            mean_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)
            best_fit = np.min(fitnesses)
            print(f"    Summary: Mean = {mean_fit:.6e}, Std = {std_fit:.6e}, Best = {best_fit:.6e}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d")
    results_file = os.path.join(outdir, f"benchmark_results_{timestamp}.csv")
    
    rows = []
    for func_name in function_names:
        for alg_name in algorithms.keys():
            fitnesses = all_results[alg_name][func_name]
            rows.append({
                'Function': func_name,
                'Algorithm': alg_name,
                'Best': np.min(fitnesses),
                'Mean': np.mean(fitnesses),
                'Std': np.std(fitnesses),
                'Worst': np.max(fitnesses)
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Save raw results for statistical tests
    raw_file = os.path.join(outdir, f"benchmark_raw_{timestamp}.json")
    with open(raw_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to {raw_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmark experiments")
    parser.add_argument("--max_evals", type=int, default=105000, help="Maximum evaluations")
    parser.add_argument("--num_runs", type=int, default=30, help="Number of runs")
    parser.add_argument("--functions", nargs="+", default=None, help="Function names (default: all)")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    run_experiment(
        function_names=args.functions,
        max_evals=args.max_evals,
        num_runs=args.num_runs,
        outdir=args.outdir
    )

