"""Generate (K,p) sensitivity heatmaps."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from social.config import Config
from social.functions import BenchmarkFunctions
from social.budget import BudgetedObjective
from social.optimizer import SOCIALOptimizer


def run_kp_heatmap(function_names=None, max_evals=105000, num_runs=10,
                  seeds=None, outdir="outputs"):
    """
    Generate (K,p) sensitivity heatmaps.
    
    Args:
        function_names: List of function names (None = subset)
        max_evals: Maximum evaluations
        num_runs: Number of runs per (K,p) combination
        seeds: List of seeds
        outdir: Output directory
    """
    os.makedirs(outdir, exist_ok=True)
    
    if function_names is None:
        function_names = ['Sphere', 'Rastrigin', 'Ackley']
    
    if seeds is None:
        seeds = list(range(num_runs))
    
    K_values = [2, 4, 6, 8, 10]
    p_values = [0.0, 0.1, 0.3, 0.5, 0.8]
    
    functions = BenchmarkFunctions()
    
    print(f"Generating (K,p) heatmaps for {len(function_names)} functions")
    print(f"K values: {K_values}")
    print(f"p values: {p_values}")
    
    all_data = []
    
    for func_name in function_names:
        print(f"\n{'='*60}")
        print(f"Function: {func_name}")
        print(f"{'='*60}")
        
        func, bounds, optimum = functions.functions[func_name]
        dim = Config.DIM
        
        for K in K_values:
            for p in p_values:
                print(f"  K={K}, p={p:.1f}")
                
                config = Config()
                config.MAX_EVALS = max_evals
                config.K = K
                config.P_BASE = p
                
                best_fits = []
                
                for run_idx in range(num_runs):
                    seed = seeds[run_idx]
                    rng = np.random.default_rng(seed)
                    
                    obj_func = BudgetedObjective(func, max_evals, name=f"{func_name}_K{K}_p{p}_{run_idx}")
                    optimizer = SOCIALOptimizer(config, rng=rng)
                    
                    result = optimizer.optimize(obj_func, bounds, dim, seed=seed)
                    best_fits.append(result['best_fitness'])
                
                mean_fit = np.mean(best_fits)
                all_data.append({
                    'Function': func_name,
                    'K': K,
                    'p': p,
                    'Mean': mean_fit,
                    'Std': np.std(best_fits),
                    'Best': np.min(best_fits)
                })
    
    df = pd.DataFrame(all_data)
    
    # Save data
    data_file = os.path.join(outdir, "kp_heatmap_data.csv")
    df.to_csv(data_file, index=False)
    print(f"\n(K,p) data saved to {data_file}")
    
    # Generate heatmaps
    os.makedirs(os.path.join(outdir, "heatmaps"), exist_ok=True)
    
    for func_name in function_names:
        func_data = df[df['Function'] == func_name]
        
        # Pivot for heatmap
        pivot = func_data.pivot(index='K', columns='p', values='Mean')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.2e', cmap='viridis_r', cbar_kws={'label': 'Mean Fitness'})
        plt.title(f'{func_name} - (K,p) Sensitivity')
        plt.xlabel('Rewiring Probability (p)')
        plt.ylabel('Neighbors (K)')
        plt.tight_layout()
        
        heatmap_file = os.path.join(outdir, "heatmaps", f"kp_heatmap_{func_name}.png")
        plt.savefig(heatmap_file, dpi=300)
        plt.close()
        
        print(f"Heatmap saved to {heatmap_file}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate (K,p) heatmaps")
    parser.add_argument("--functions", nargs="+", default=None, help="Function names")
    parser.add_argument("--max_evals", type=int, default=105000, help="Maximum evaluations")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs per (K,p)")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    run_kp_heatmap(
        function_names=args.functions,
        max_evals=args.max_evals,
        num_runs=args.num_runs,
        outdir=args.outdir
    )

