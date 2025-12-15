"""Compare different centrality measures."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from social.config import Config
from social.functions import BenchmarkFunctions
from social.budget import BudgetedObjective
from social.optimizer import SOCIALOptimizer


def run_centrality_comparison(function_names=None, max_evals=105000, num_runs=30,
                              seeds=None, outdir="outputs"):
    """
    Compare SOCIAL with different centrality measures.
    
    Args:
        function_names: List of function names (None = subset)
        max_evals: Maximum evaluations
        num_runs: Number of runs
        seeds: List of seeds
        outdir: Output directory
    """
    os.makedirs(outdir, exist_ok=True)
    
    if function_names is None:
        function_names = ['Sphere', 'Rastrigin', 'Ackley', 'Rosenbrock', 'Griewank']
    
    if seeds is None:
        seeds = list(range(num_runs))
    
    centrality_modes = ['betweenness', 'degree', 'closeness', 'pagerank', 'eigenvector']
    
    functions = BenchmarkFunctions()
    results = {mode: {func: [] for func in function_names} for mode in centrality_modes}
    runtime_stats = {mode: {func: [] for func in function_names} for mode in centrality_modes}
    
    print(f"Comparing centrality measures on {len(function_names)} functions")
    print(f"Centrality modes: {centrality_modes}")
    
    for func_name in function_names:
        print(f"\n{'='*60}")
        print(f"Function: {func_name}")
        print(f"{'='*60}")
        
        func, bounds, optimum = functions.functions[func_name]
        dim = Config.DIM
        
        for centrality_mode in centrality_modes:
            print(f"\n  Centrality: {centrality_mode}")
            
            config = Config()
            config.MAX_EVALS = max_evals
            config.CENTRALITY_MODE = centrality_mode
            
            for run_idx in range(num_runs):
                seed = seeds[run_idx]
                rng = np.random.default_rng(seed)
                
                obj_func = BudgetedObjective(func, max_evals, name=f"{func_name}_{centrality_mode}_{run_idx}")
                optimizer = SOCIALOptimizer(config, rng=rng)
                
                result = optimizer.optimize(obj_func, bounds, dim, seed=seed)
                results[centrality_mode][func_name].append(result['best_fitness'])
                
                # Extract runtime for centrality computation
                if result['runtime_stats']['t_centrality']:
                    avg_centrality_time = np.mean(result['runtime_stats']['t_centrality'])
                    runtime_stats[centrality_mode][func_name].append(avg_centrality_time)
            
            fitnesses = results[centrality_mode][func_name]
            mean_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)
            print(f"    Mean = {mean_fit:.6e}, Std = {std_fit:.6e}")
    
    # Compute ranks
    comparison_data = []
    for func_name in function_names:
        func_results = {mode: results[mode][func_name] for mode in centrality_modes}
        
        means = {mode: np.mean(func_results[mode]) for mode in centrality_modes}
        sorted_modes = sorted(means.items(), key=lambda x: x[1])
        ranks = {mode: rank+1 for rank, (mode, _) in enumerate(sorted_modes)}
        
        for mode in centrality_modes:
            avg_runtime = np.mean(runtime_stats[mode][func_name]) if runtime_stats[mode][func_name] else 0.0
            comparison_data.append({
                'Function': func_name,
                'Centrality': mode,
                'Mean': means[mode],
                'Std': np.std(func_results[mode]),
                'Rank': ranks[mode],
                'Avg_Centrality_Time': avg_runtime
            })
    
    df = pd.DataFrame(comparison_data)
    
    comparison_file = os.path.join(outdir, "centrality_comparison.csv")
    df.to_csv(comparison_file, index=False)
    print(f"\nCentrality comparison saved to {comparison_file}")
    
    # Summary
    summary = df.groupby('Centrality').agg({
        'Mean': 'mean',
        'Rank': 'mean',
        'Avg_Centrality_Time': 'mean'
    }).reset_index()
    summary.columns = ['Centrality', 'Avg_Mean', 'Avg_Rank', 'Avg_Centrality_Time']
    summary = summary.sort_values('Avg_Rank')
    
    summary_file = os.path.join(outdir, "centrality_summary.csv")
    summary.to_csv(summary_file, index=False)
    print(f"Centrality summary saved to {summary_file}")
    
    print("\nSummary:")
    print(summary.to_string(index=False))
    
    return df, summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare centrality measures")
    parser.add_argument("--functions", nargs="+", default=None, help="Function names")
    parser.add_argument("--max_evals", type=int, default=105000, help="Maximum evaluations")
    parser.add_argument("--num_runs", type=int, default=30, help="Number of runs")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    run_centrality_comparison(
        function_names=args.functions,
        max_evals=args.max_evals,
        num_runs=args.num_runs,
        outdir=args.outdir
    )

