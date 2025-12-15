"""Ablation study: remove components one by one."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from social.config import Config
from social.functions import BenchmarkFunctions
from social.budget import BudgetedObjective
from social.optimizer import SOCIALOptimizer


def run_ablation_study(function_names=None, max_evals=105000, num_runs=30, 
                       seeds=None, outdir="outputs"):
    """
    Run ablation study by removing components.
    
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
    
    # Ablation variants
    variants = {
        'SOCIAL_full': {
            'ENABLE_ELITE_MEMORY': True,
            'ENABLE_MUTATION': True,
            'ENABLE_SYNC': True,
            'NEIGHBOR_MODE': 'centrality_weighted',
            'HYBRID_LOTUS': 'off'
        },
        'no_elite': {
            'ENABLE_ELITE_MEMORY': False,
            'ENABLE_MUTATION': True,
            'ENABLE_SYNC': True,
            'NEIGHBOR_MODE': 'centrality_weighted',
            'HYBRID_LOTUS': 'off'
        },
        'no_mutation': {
            'ENABLE_ELITE_MEMORY': True,
            'ENABLE_MUTATION': False,
            'ENABLE_SYNC': True,
            'NEIGHBOR_MODE': 'centrality_weighted',
            'HYBRID_LOTUS': 'off'
        },
        'no_sync': {
            'ENABLE_ELITE_MEMORY': True,
            'ENABLE_MUTATION': True,
            'ENABLE_SYNC': False,
            'NEIGHBOR_MODE': 'centrality_weighted',
            'HYBRID_LOTUS': 'off'
        },
        'uniform_neighbors': {
            'ENABLE_ELITE_MEMORY': True,
            'ENABLE_MUTATION': True,
            'ENABLE_SYNC': True,
            'NEIGHBOR_MODE': 'uniform',
            'HYBRID_LOTUS': 'off'
        },
        'no_rewiring': {
            'ENABLE_ELITE_MEMORY': True,
            'ENABLE_MUTATION': True,
            'ENABLE_SYNC': True,
            'NEIGHBOR_MODE': 'centrality_weighted',
            'REWIRE_MODE': 'none',
            'HYBRID_LOTUS': 'off'
        },
        'fixed_weights': {
            'ENABLE_ELITE_MEMORY': True,
            'ENABLE_MUTATION': True,
            'ENABLE_SYNC': True,
            'NEIGHBOR_MODE': 'centrality_weighted',
            'SCHEDULE_MODE': 'linear',  # Keep linear but could set fixed
            'HYBRID_LOTUS': 'off'
        }
    }
    
    functions = BenchmarkFunctions()
    results = {variant: {func: [] for func in function_names} for variant in variants.keys()}
    
    print(f"Running ablation study on {len(function_names)} functions")
    print(f"Variants: {list(variants.keys())}")
    
    for func_name in function_names:
        print(f"\n{'='*60}")
        print(f"Function: {func_name}")
        print(f"{'='*60}")
        
        func, bounds, optimum = functions.functions[func_name]
        dim = Config.DIM
        
        for variant_name, variant_config in variants.items():
            print(f"\n  Variant: {variant_name}")
            
            config = Config()
            config.MAX_EVALS = max_evals
            for key, value in variant_config.items():
                setattr(config, key, value)
            
            for run_idx in range(num_runs):
                seed = seeds[run_idx]
                rng = np.random.default_rng(seed)
                
                obj_func = BudgetedObjective(func, max_evals, name=f"{func_name}_{variant_name}_{run_idx}")
                optimizer = SOCIALOptimizer(config, rng=rng)
                
                result = optimizer.optimize(obj_func, bounds, dim, seed=seed)
                results[variant_name][func_name].append(result['best_fitness'])
            
            fitnesses = results[variant_name][func_name]
            mean_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)
            print(f"    Mean = {mean_fit:.6e}, Std = {std_fit:.6e}")
    
    # Compute ranks and save
    ablation_data = []
    for func_name in function_names:
        func_results = {variant: results[variant][func_name] for variant in variants.keys()}
        
        # Compute ranks
        means = {variant: np.mean(func_results[variant]) for variant in variants.keys()}
        sorted_variants = sorted(means.items(), key=lambda x: x[1])
        ranks = {variant: rank+1 for rank, (variant, _) in enumerate(sorted_variants)}
        
        for variant in variants.keys():
            ablation_data.append({
                'Function': func_name,
                'Variant': variant,
                'Mean': means[variant],
                'Std': np.std(func_results[variant]),
                'Rank': ranks[variant]
            })
    
    df = pd.DataFrame(ablation_data)
    
    # Summary table
    summary = df.groupby('Variant').agg({
        'Mean': 'mean',
        'Std': 'mean',
        'Rank': 'mean'
    }).reset_index()
    summary.columns = ['Variant', 'Avg_Mean', 'Avg_Std', 'Avg_Rank']
    summary = summary.sort_values('Avg_Rank')
    
    ablation_file = os.path.join(outdir, "ablation_table.csv")
    df.to_csv(ablation_file, index=False)
    print(f"\nAblation results saved to {ablation_file}")
    
    summary_file = os.path.join(outdir, "ablation_summary.csv")
    summary.to_csv(summary_file, index=False)
    print(f"Ablation summary saved to {summary_file}")
    
    # LaTeX table
    latex_file = os.path.join(outdir, "tables", "ablation_table.tex")
    os.makedirs(os.path.dirname(latex_file), exist_ok=True)
    
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation Study Results}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\hline\n")
        f.write("Variant & Average Mean & Average Std & Average Rank \\\\\n")
        f.write("\\hline\n")
        for _, row in summary.iterrows():
            f.write(f"{row['Variant']} & {row['Avg_Mean']:.6e} & {row['Avg_Std']:.6e} & {row['Avg_Rank']:.2f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {latex_file}")
    
    return df, summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--functions", nargs="+", default=None, help="Function names")
    parser.add_argument("--max_evals", type=int, default=105000, help="Maximum evaluations")
    parser.add_argument("--num_runs", type=int, default=30, help="Number of runs")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    run_ablation_study(
        function_names=args.functions,
        max_evals=args.max_evals,
        num_runs=args.num_runs,
        outdir=args.outdir
    )

