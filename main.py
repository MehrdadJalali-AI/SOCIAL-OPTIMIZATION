"""
Main entry point for SOCIAL optimizer.
Maintains backward compatibility with original code while using new package structure.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

# Import from new package structure
from social.config import Config
from social.functions import BenchmarkFunctions
from social.budget import BudgetedObjective
from social.optimizer import SOCIALOptimizer

# For backward compatibility, also import old classes if needed
try:
    from social.graph_ops import create_watts_strogatz_graph, compute_centrality
except ImportError:
    pass


def run_social_benchmark(function_names=None, max_evals=105000, num_runs=30, 
                        seeds=None, outdir="qualitative_results"):
    """
    Run SOCIAL optimizer on benchmark functions.
    
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
    
    print(f"Running SOCIAL on {len(function_names)} functions")
    print(f"Max evaluations: {max_evals}, Runs: {num_runs}")
    
    results = []
    
    for func_name in function_names:
        print(f"\n{'='*60}")
        print(f"Function: {func_name}")
        print(f"{'='*60}")
        
        func, bounds, optimum = functions.functions[func_name]
        dim = config.DIM
        
        # Create output directory for this function
        func_dir = os.path.join(outdir, func_name)
        os.makedirs(func_dir, exist_ok=True)
        
        best_fits = []
        all_histories = []
        
        for run_idx in range(num_runs):
            seed = seeds[run_idx]
            rng = np.random.default_rng(seed)
            
            # Create budgeted objective
            obj_func = BudgetedObjective(func, max_evals, name=f"{func_name}_{run_idx}")
            
            # Create optimizer
            optimizer = SOCIALOptimizer(config, rng=rng)
            
            # Run optimization
            result = optimizer.optimize(obj_func, bounds, dim, seed=seed)
            
            best_fit = result['best_fitness']
            best_fits.append(best_fit)
            all_histories.append(result['convergence_history'])
            
            # Save first run's detailed data
            if run_idx == 0:
                # Search history (first dimension of all positions)
                search_history = []
                for node in result['final_population'].nodes:
                    pos = result['final_population'].nodes[node]['position']
                    search_history.append(pos[config.TRACKED_DIM])
                
                search_df = pd.DataFrame({
                    f'Agent_{i}': [search_history[i]] for i in range(len(search_history))
                })
                search_df.to_csv(os.path.join(func_dir, 'search_history.csv'), index=False)
                
                # First agent trajectory
                first_agent_traj = [result['final_population'].nodes[0]['position'][config.TRACKED_DIM]]
                trajectory_df = pd.DataFrame({
                    'Iteration': range(len(result['convergence_history'])),
                    'Position': first_agent_traj * len(result['convergence_history'])
                })
                trajectory_df.to_csv(os.path.join(func_dir, 'first_agent_trajectory.csv'), index=False)
                
                # Average fitness
                avg_fitness_df = pd.DataFrame({
                    'Iteration': range(len(result['avg_fitness_history'])),
                    'Average_Fitness': result['avg_fitness_history']
                })
                avg_fitness_df.to_csv(os.path.join(func_dir, 'avg_fitness.csv'), index=False)
                
                # Convergence
                convergence_df = pd.DataFrame({
                    'Iteration': range(len(result['convergence_history'])),
                    'Best_Fitness': result['convergence_history']
                })
                convergence_df.to_csv(os.path.join(func_dir, 'convergence.csv'), index=False)
                
                # Best fitness archive
                best_fitness_df = pd.DataFrame({
                    'Iteration': range(len(result['best_fitness_archive'])),
                    'Best_Fitness': result['best_fitness_archive']
                })
                best_fitness_df.to_csv(os.path.join(func_dir, 'best_fitness_archive.csv'), index=False)
            
            if (run_idx + 1) % 10 == 0:
                print(f"  Run {run_idx+1}/{num_runs}: Best = {best_fit:.6e}")
        
        # Compute metrics
        best_fits = np.array(best_fits)
        mean_fit = np.mean(best_fits)
        std_fit = np.std(best_fits)
        robustness = std_fit**2
        
        # Diversity (simplified)
        diversity = np.mean([np.std([result['final_population'].nodes[n]['position'][i] 
                                    for n in result['final_population'].nodes])
                            for i in range(dim)])
        
        convergence_speed = next((i for i, fit in enumerate(all_histories[0]) 
                                  if fit < mean_fit + std_fit), len(all_histories[0]))
        success_rate = np.mean([1 if abs(bf - optimum) < config.SUCCESS_THRESHOLD else 0 
                               for bf in best_fits])
        
        result_dict = {
            "Function": func_name,
            "Best Fitness": np.min(best_fits),
            "Worst Fitness": np.max(best_fits),
            "Mean Fitness": mean_fit,
            "Std Dev": std_fit,
            "Robustness": robustness,
            "Diversity": diversity,
            "Conv. Speed": convergence_speed,
            "SR": success_rate
        }
        
        # Save metrics
        metrics_file = os.path.join(func_dir, 'metrics.csv')
        pd.DataFrame([result_dict]).to_csv(metrics_file, index=False)
        
        results.append(result_dict)
        
        print(f"\n  Summary: Best = {np.min(best_fits):.6e}, Mean = {mean_fit:.6e}, SR = {success_rate:.2f}")
    
    # Save overall results
    timestamp = datetime.now().strftime("%Y-%m-%d")
    results_file = f'social_results_{timestamp}.csv'
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to '{results_file}'")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SOCIAL Optimizer")
    parser.add_argument("--functions", nargs="+", default=None, help="Function names (default: all)")
    parser.add_argument("--max_evals", type=int, default=105000, help="Maximum evaluations")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of runs")
    parser.add_argument("--outdir", type=str, default="qualitative_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Run with specified or all functions
    run_social_benchmark(
        function_names=args.functions,
        max_evals=args.max_evals,
        num_runs=args.num_runs,
        outdir=args.outdir
    )
