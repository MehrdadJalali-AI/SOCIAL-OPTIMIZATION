"""Run benchmark functions with SOCIAL only (no baselines)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
import time
from collections import defaultdict
from datetime import datetime
from social.config import Config
from social.functions import BenchmarkFunctions
from social.budget import BudgetedObjective
from social.optimizer import SOCIALOptimizer


def format_time(seconds):
    """Format seconds as MMs or HHhMMm."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes}m"


def run_social_experiment(function_names: list = None, max_evals: int = 105000,
                         num_runs: int = 30, seeds: list = None, outdir: str = "outputs",
                         num_nodes: int = None, sweep_nodes: bool = False, preset: str = None):
    """
    Run benchmark experiments with SOCIAL only.
    
    Args:
        function_names: List of function names (None = all)
        max_evals: Maximum evaluations
        num_runs: Number of runs
        seeds: List of seeds (None = [0..num_runs-1])
        outdir: Output directory
        num_nodes: Number of nodes (None = use config default)
        sweep_nodes: If True, sweep NUM_NODES ∈ {60, 100, 150, 300}
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Determine NUM_NODES values - ensure consistent int type
    if sweep_nodes:
        node_values = [60, 100, 150, 300]
    else:
        if num_nodes is not None:
            node_values = [int(num_nodes)]
        else:
            node_values = [None]
    
    # Functions
    functions = BenchmarkFunctions()
    if function_names is None:
        function_names = list(functions.functions.keys())
    
    # Results storage: {num_nodes: {func: [fitnesses]}} - use defaultdict for safety
    all_results = defaultdict(lambda: defaultdict(list))
    
    experiment_start_time = time.time()
    
    for node_idx, num_nodes_val in enumerate(node_values):
        # Configuration - set preset first so it's applied in __post_init__
        preset_to_use = preset if preset is not None else "SOCIAL_balanced"
        config = Config(PRESET=preset_to_use)
        config.MAX_EVALS = max_evals
        config.NUM_RUNS = num_runs
        
        # Ensure num_nodes_val is int (or None) for consistent key type
        if num_nodes_val is not None:
            num_nodes_val = int(num_nodes_val)
            config.NUM_NODES = num_nodes_val
        else:
            # Use config default and ensure it's int
            num_nodes_val = int(config.NUM_NODES)
            config.NUM_NODES = num_nodes_val
        
        # Disable iteration-level progress, only show function-level
        config.SHOW_PROGRESS = False  # Disable optimizer-level progress
        config.PROGRESS_MODE = "function"
        
        if seeds is None:
            seeds = list(range(num_runs))
        config.SEED_LIST = seeds
        
        # Initialize results storage for this NUM_NODES (defaultdict handles this, but explicit for clarity)
        for func_name in function_names:
            if func_name not in all_results[num_nodes_val]:
                all_results[num_nodes_val][func_name] = []
        
        for func_idx, func_name in enumerate(function_names):
            func_start_time = time.time()
            
            func, bounds, optimum = functions.functions[func_name]
            dim = config.DIM
            
            # Track run times for ETA estimation
            run_times = []
            
            for run_idx in range(num_runs):
                run_start_time = time.time()
                seed = seeds[run_idx]
                rng = np.random.default_rng(seed)
                
                # Create optimizer
                optimizer = SOCIALOptimizer(config)
                optimizer.rng = rng
                
                # Create budgeted objective
                obj_func = BudgetedObjective(func, max_evals, name=f"{func_name}_SOCIAL_{run_idx}")
                
                # Run optimizer (no per-run or per-iteration output)
                result = optimizer.optimize(obj_func, bounds, dim, seed=seed)
                
                best_fitness = result['best_fitness']
                all_results[num_nodes_val][func_name].append(best_fitness)
                
                run_time = time.time() - run_start_time
                run_times.append(run_time)
            
            # Function-level summary (only output when function completes)
            func_time = time.time() - func_start_time
            avg_run_time = np.mean(run_times) if run_times else 0
            
            # Safe access to results with error checking
            fitnesses = all_results.get(num_nodes_val, {}).get(func_name, [])
            if len(fitnesses) == 0:
                print(f"[WARN] No results stored for nodes={num_nodes_val}, function={func_name}")
                continue
            
            mean_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)
            best_fit = np.min(fitnesses)
            
            # Estimate remaining time
            remaining_functions = len(function_names) - (func_idx + 1)
            remaining_runs = remaining_functions * num_runs
            if len(run_times) > 0:
                estimated_remaining = avg_run_time * remaining_runs
                remaining_str = format_time(estimated_remaining)
            else:
                remaining_str = "N/A"
            
            # Function-level progress output (only when function finishes)
            print(f"[FUNCTION DONE] {func_name} | runs={num_runs} | best={best_fit:.2e} | "
                  f"avg_time/run={format_time(avg_run_time)} | elapsed={format_time(func_time)}")
        
        # Save intermediate results for sweep (silently)
        if sweep_nodes:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            sweep_file = os.path.join(outdir, f"social_num_nodes_sweep_{timestamp}.csv")
            
            rows = []
            for nodes_val in all_results.keys():
                # Ensure nodes_val is int
                nodes_val = int(nodes_val)
                for func_name in function_names:
                    # Safe access with error checking
                    fitnesses = all_results.get(nodes_val, {}).get(func_name, [])
                    if len(fitnesses) == 0:
                        print(f"[WARN] Missing results for nodes={nodes_val}, function={func_name}")
                        continue
                    
                    rows.append({
                        'NUM_NODES': nodes_val,
                        'Function': func_name,
                        'Mean': np.mean(fitnesses),
                        'Std': np.std(fitnesses),
                        'Best': np.min(fitnesses),
                        'Worst': np.max(fitnesses)
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(sweep_file, index=False)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y-%m-%d")
    
    if sweep_nodes:
        # Already saved above
        results_file = os.path.join(outdir, f"social_num_nodes_sweep_{timestamp}.csv")
    else:
        results_file = os.path.join(outdir, f"social_only_results_{timestamp}.csv")
        
        rows = []
        # Determine nodes_val - ensure it's int
        if node_values[0] is not None:
            nodes_val = int(node_values[0])
        else:
            nodes_val = int(config.NUM_NODES)
        
        for func_name in function_names:
            # Safe access with error checking
            fitnesses = all_results.get(nodes_val, {}).get(func_name, [])
            if len(fitnesses) == 0:
                print(f"[WARN] Missing results for nodes={nodes_val}, function={func_name}")
                continue
            
            rows.append({
                'Function': func_name,
                'Mean': np.mean(fitnesses),
                'Std': np.std(fitnesses),
                'Best': np.min(fitnesses),
                'Worst': np.max(fitnesses)
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(results_file, index=False)
    
    total_time = time.time() - experiment_start_time
    num_functions = len(function_names)
    print(f"\n[ALL DONE] {num_functions} functions completed | total time = {format_time(total_time)}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SOCIAL-only benchmark experiments")
    parser.add_argument("--max_evals", type=int, default=105000, help="Maximum evaluations")
    parser.add_argument("--num_runs", type=int, default=30, help="Number of runs")
    parser.add_argument("--functions", nargs="+", default=None, help="Function names (default: all)")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--num_nodes", type=int, default=None, help="Number of nodes (overrides config)")
    parser.add_argument("--sweep_nodes", action="store_true", help="Sweep NUM_NODES ∈ {60, 100, 150, 300}")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer runs and evals")
    parser.add_argument("--preset", type=str, default=None, 
                       choices=["SOCIAL_balanced", "SOCIAL_fast"],
                       help="Configuration preset (default: SOCIAL_balanced)")
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        if args.num_runs == 30:  # Only override if not explicitly set
            args.num_runs = 5
        if args.max_evals == 105000:  # Only override if not explicitly set
            args.max_evals = 5000
        if args.functions is None:
            args.functions = ["Sphere", "Rastrigin", "Ackley"]
    
    # Default preset
    if args.preset is None:
        args.preset = "SOCIAL_balanced"
    
    run_social_experiment(
        function_names=args.functions,
        max_evals=args.max_evals,
        num_runs=args.num_runs,
        outdir=args.outdir,
        num_nodes=args.num_nodes,
        sweep_nodes=args.sweep_nodes,
        preset=args.preset
    )

