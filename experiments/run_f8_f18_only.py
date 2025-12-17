"""Run ONLY F8 and F18 benchmark functions with SOCIAL and baselines."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
from social.config import Config
from social.functions import BenchmarkFunctions
from social.budget import BudgetedObjective
from social.optimizer import SOCIALOptimizer
from baselines.de import DifferentialEvolution
from baselines.pso import ParticleSwarmOptimization
from baselines.gwo import GreyWolfOptimizer


def run_f8_f18_experiment(max_evals: int = 105000, num_runs: int = 30, 
                          seeds: list = None, outdir: str = "outputs"):
    """
    Run experiments ONLY for F8 (Schwefel_2_26) and F18 (Goldstein-Price).
    
    Args:
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
    
    # ONLY F8 and F18
    TARGET_FUNCTIONS = ['Schwefel_2_26', 'Goldstein-Price']
    
    # Functions
    functions = BenchmarkFunctions()
    
    # Verify functions exist
    for func_name in TARGET_FUNCTIONS:
        if func_name not in functions.functions:
            raise ValueError(f"Function {func_name} not found in BenchmarkFunctions")
    
    # Algorithms
    algorithms = {
        'SOCIAL': SOCIALOptimizer(config),
        'DE': DifferentialEvolution(npop=50),
        'PSO': ParticleSwarmOptimization(npop=50),
        'GWO': GreyWolfOptimizer(npop=50)
    }
    
    # Results storage
    all_results = {alg: {func: [] for func in TARGET_FUNCTIONS} for alg in algorithms.keys()}
    
    print(f"Running experiments on {len(TARGET_FUNCTIONS)} functions: {TARGET_FUNCTIONS}")
    print(f"Max evaluations: {max_evals}, Number of runs: {num_runs}")
    
    for func_name in TARGET_FUNCTIONS:
        print(f"\n{'='*60}")
        print(f"Function: {func_name}")
        print(f"{'='*60}")
        
        func, bounds, optimum = functions.functions[func_name]
        dim = config.DIM
        
        # Sanity check: verify function returns correct sign
        test_x = np.random.uniform(bounds[0], bounds[1], dim)
        test_val = func(test_x)
        if func_name == 'Schwefel_2_26':
            print(f"  Sanity check: Test value = {test_val:.2f} (should be negative for good solutions)")
        elif func_name == 'Goldstein-Price':
            print(f"  Sanity check: Test value = {test_val:.2f} (should be positive)")
        
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
            worst_fit = np.max(fitnesses)
            print(f"    Summary: Mean = {mean_fit:.6e}, Std = {std_fit:.6e}, Best = {best_fit:.6e}, Worst = {worst_fit:.6e}")
            
            # Verify sign
            if func_name == 'Schwefel_2_26':
                if best_fit > 0:
                    print(f"    ⚠ WARNING: Best fitness is POSITIVE (should be negative)")
                else:
                    print(f"    ✓ Best fitness is NEGATIVE (correct)")
            elif func_name == 'Goldstein-Price':
                if best_fit < 0:
                    print(f"    ⚠ WARNING: Best fitness is NEGATIVE (should be positive)")
                else:
                    print(f"    ✓ Best fitness is POSITIVE (correct)")
    
    # Save results to NEW CSV file
    results_file = os.path.join(outdir, "OnlyTwo_F8_F18_results.csv")
    
    rows = []
    for func_name in TARGET_FUNCTIONS:
        for alg_name in algorithms.keys():
            fitnesses = all_results[alg_name][func_name]
            rows.append({
                'Function': func_name,
                'Algorithm': alg_name,
                'Best': np.min(fitnesses),
                'Mean': np.mean(fitnesses),
                'Std': np.std(fitnesses),
                'Worst': np.max(fitnesses),
                'NumRuns': len(fitnesses)
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(results_file, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to {results_file}")
    print(f"{'='*60}")
    
    # Print final verification
    print("\nFinal Verification:")
    print("="*70)
    for func_name in TARGET_FUNCTIONS:
        print(f"\n{func_name}:")
        for alg_name in algorithms.keys():
            best = np.min(all_results[alg_name][func_name])
            if func_name == 'Schwefel_2_26':
                status = "✓" if best < 0 else "✗"
                print(f"  {alg_name:8s}: Best = {best:15.2f} {status} (should be NEGATIVE)")
            elif func_name == 'Goldstein-Price':
                status = "✓" if best > 0 else "✗"
                print(f"  {alg_name:8s}: Best = {best:15.6f} {status} (should be POSITIVE ≈ 3.0)")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run F8 and F18 experiments only")
    parser.add_argument("--max_evals", type=int, default=105000, help="Maximum evaluations")
    parser.add_argument("--num_runs", type=int, default=30, help="Number of runs")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    run_f8_f18_experiment(
        max_evals=args.max_evals,
        num_runs=args.num_runs,
        outdir=args.outdir
    )

