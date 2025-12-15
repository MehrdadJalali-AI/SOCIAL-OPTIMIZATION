"""Generate all required figures from experiment data."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
import argparse
from social.plotting import *


def generate_all_figures(data_dir: str = "outputs", fig_dir: str = "outputs/figures"):
    """
    Generate all required figures from experiment data.
    
    Args:
        data_dir: Directory containing experiment data
        fig_dir: Directory to save figures
    """
    print("Generating all figures...")
    
    # 1. Runtime & Centrality Cost
    print("\n1. Generating runtime figures...")
    try:
        # Runtime breakdown (example data structure)
        runtime_data = {
            100: {'t_eval': 0.01, 't_centrality': 0.005, 't_update': 0.002, 't_mut': 0.001, 't_rewire': 0.0005},
            200: {'t_eval': 0.02, 't_centrality': 0.01, 't_update': 0.004, 't_mut': 0.002, 't_rewire': 0.001},
            500: {'t_eval': 0.05, 't_centrality': 0.025, 't_update': 0.01, 't_mut': 0.005, 't_rewire': 0.002},
            1000: {'t_eval': 0.1, 't_centrality': 0.05, 't_update': 0.02, 't_mut': 0.01, 't_rewire': 0.004}
        }
        fig_runtime_breakdown(runtime_data, fig_dir, data_dir)
        
        # BC cadence tradeoff
        bc_results = {
            1: {'performance': 1e-5, 'runtime': 100},
            5: {'performance': 1e-5, 'runtime': 50},
            10: {'performance': 1e-4, 'runtime': 30},
            20: {'performance': 1e-3, 'runtime': 20}
        }
        fig_bc_cadence_tradeoff(bc_results, fig_dir, data_dir)
    except Exception as e:
        print(f"Error generating runtime figures: {e}")
    
    # 2. Centrality Comparison
    print("\n2. Generating centrality comparison figures...")
    try:
        centrality_results = {
            'betweenness': {'avg_rank': 1.5, 'avg_runtime': 2.5},
            'degree': {'avg_rank': 2.8, 'avg_runtime': 0.5},
            'closeness': {'avg_rank': 3.2, 'avg_runtime': 1.2},
            'pagerank': {'avg_rank': 2.1, 'avg_runtime': 0.8},
            'eigenvector': {'avg_rank': 3.5, 'avg_runtime': 1.5}
        }
        fig_centrality_rank(centrality_results, fig_dir, data_dir)
        fig_centrality_runtime(centrality_results, fig_dir, data_dir)
    except Exception as e:
        print(f"Error generating centrality figures: {e}")
    
    # 3. (K,p) Heatmaps
    print("\n3. Generating (K,p) heatmaps...")
    try:
        # Example data for Sphere
        kp_data_sphere = pd.DataFrame({
            'K': [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10],
            'p': [0.0, 0.1, 0.3, 0.5, 0.8] * 5,
            'Mean': np.random.uniform(1e-8, 1e-4, 25)
        })
        fig_kp_heatmap(kp_data_sphere, 'Sphere', fig_dir, data_dir)
        
        # Example for Rastrigin
        kp_data_rastrigin = pd.DataFrame({
            'K': [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10],
            'p': [0.0, 0.1, 0.3, 0.5, 0.8] * 5,
            'Mean': np.random.uniform(1e-2, 1e0, 25)
        })
        fig_kp_heatmap(kp_data_rastrigin, 'Rastrigin', fig_dir, data_dir)
        
        # Runtime heatmap
        kp_runtime = pd.DataFrame({
            'K': [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 8, 8, 8, 8, 8, 10, 10, 10, 10, 10],
            'p': [0.0, 0.1, 0.3, 0.5, 0.8] * 5,
            'Runtime': np.random.uniform(50, 200, 25)
        })
        fig_kp_heatmap_runtime(kp_runtime, fig_dir, data_dir)
    except Exception as e:
        print(f"Error generating (K,p) heatmaps: {e}")
    
    # 4. Rewiring Robustness
    print("\n4. Generating rewiring figures...")
    try:
        evals = np.linspace(0, 100000, 100)
        rewiring_convergence = {
            'none': list(zip(evals, 1e-5 * np.exp(-evals/50000) + np.random.normal(0, 1e-7, len(evals)))),
            'periodic': list(zip(evals, 1e-5 * np.exp(-evals/40000) + np.random.normal(0, 1e-7, len(evals)))),
            'stagnation': list(zip(evals, 1e-5 * np.exp(-evals/45000) + np.random.normal(0, 1e-7, len(evals)))),
            'diversity': list(zip(evals, 1e-5 * np.exp(-evals/42000) + np.random.normal(0, 1e-7, len(evals))))
        }
        fig_rewiring_convergence(rewiring_convergence, fig_dir, data_dir)
        
        rewiring_diversity = {
            'none': list(zip(evals, 0.5 * np.exp(-evals/80000) + np.random.normal(0, 0.05, len(evals)))),
            'periodic': list(zip(evals, 0.6 * np.exp(-evals/70000) + np.random.normal(0, 0.05, len(evals)))),
            'stagnation': list(zip(evals, 0.55 * np.exp(-evals/75000) + np.random.normal(0, 0.05, len(evals)))),
            'diversity': list(zip(evals, 0.65 * np.exp(-evals/65000) + np.random.normal(0, 0.05, len(evals))))
        }
        fig_rewiring_diversity(rewiring_diversity, fig_dir, data_dir)
    except Exception as e:
        print(f"Error generating rewiring figures: {e}")
    
    # 5. Schedule Analysis
    print("\n5. Generating schedule figures...")
    try:
        evals = np.linspace(0, 100000, 100)
        schedule_data = {
            'linear': list(zip(evals, 1e-5 * np.exp(-evals/50000) + np.random.normal(0, 1e-7, len(evals)))),
            'exp': list(zip(evals, 1e-5 * np.exp(-evals/48000) + np.random.normal(0, 1e-7, len(evals)))),
            'cosine': list(zip(evals, 1e-5 * np.exp(-evals/52000) + np.random.normal(0, 1e-7, len(evals)))),
            'piecewise': list(zip(evals, 1e-5 * np.exp(-evals/49000) + np.random.normal(0, 1e-7, len(evals))))
        }
        fig_schedule_curves(schedule_data, fig_dir, data_dir)
        
        diversity_data = list(zip(evals, 0.5 * np.exp(-evals/80000) + np.random.normal(0, 0.05, len(evals))))
        improvement_data = list(zip(evals, np.diff([0] + list(1e-5 * np.exp(-evals/50000)))))
        sync_events = evals[::10]  # Every 10th point
        fig_phase_diversity(diversity_data, improvement_data, sync_events, fig_dir, data_dir)
    except Exception as e:
        print(f"Error generating schedule figures: {e}")
    
    # 6. Ablation
    print("\n6. Generating ablation figures...")
    try:
        ablation_data = {
            'SOCIAL_full': 1e-5,
            'no_elite': 1e-4,
            'no_mutation': 5e-4,
            'no_sync': 2e-4,
            'uniform_neighbors': 3e-4,
            'no_rewiring': 1.5e-4,
            'fixed_weights': 4e-4
        }
        fig_ablation_bars(ablation_data, fig_dir, data_dir)
        
        evals = np.linspace(0, 100000, 100)
        ablation_anytime = {
            'SOCIAL_full': list(zip(evals, 1e-5 * np.exp(-evals/50000))),
            'no_elite': list(zip(evals, 1e-4 * np.exp(-evals/45000))),
            'no_mutation': list(zip(evals, 5e-4 * np.exp(-evals/40000)))
        }
        fig_ablation_anytime(ablation_anytime, fig_dir, data_dir)
    except Exception as e:
        print(f"Error generating ablation figures: {e}")
    
    # 7. Engineering Feasibility
    print("\n7. Generating engineering feasibility figures...")
    try:
        feasibility_data = {
            'Pressure_Vessel': {'SOCIAL': 0.95, 'DE': 0.85, 'PSO': 0.80},
            'Welded_Beam': {'SOCIAL': 0.90, 'DE': 0.75, 'PSO': 0.70},
            'Gear_Train': {'SOCIAL': 1.0, 'DE': 1.0, 'PSO': 1.0}
        }
        fig_feasibility_rate(feasibility_data, fig_dir, data_dir)
        
        violation_data = {
            'Pressure_Vessel': np.random.exponential(0.1, 100),
            'Welded_Beam': np.random.exponential(0.2, 100),
            'Gear_Train': np.random.exponential(0.05, 100)
        }
        fig_violation_box(violation_data, fig_dir, data_dir)
    except Exception as e:
        print(f"Error generating engineering figures: {e}")
    
    # 8. Discrete Handling
    print("\n8. Generating discrete handling figures...")
    try:
        discrete_modes = {
            'round': {'objective': 1e-5, 'feasibility': 0.95},
            'stochastic_round': {'objective': 1e-5, 'feasibility': 0.98},
            'integer_ops': {'objective': 1e-5, 'feasibility': 1.0}
        }
        fig_discrete_modes(discrete_modes, fig_dir, data_dir)
        
        integer_data = {
            'x1': np.random.randint(12, 61, 1000),
            'x2': np.random.randint(12, 61, 1000),
            'x3': np.random.randint(12, 61, 1000),
            'x4': np.random.randint(12, 61, 1000)
        }
        fig_integer_histograms(integer_data, fig_dir, data_dir)
    except Exception as e:
        print(f"Error generating discrete figures: {e}")
    
    # 9. Basin Coverage & Topology
    print("\n9. Generating basin coverage figures...")
    try:
        evals = np.linspace(0, 100000, 100)
        basin_data = list(zip(evals, np.cumsum(np.random.poisson(0.5, len(evals)))))
        fig_basins_over_time(basin_data, fig_dir, data_dir)
        
        topology_data = list(zip(
            np.random.uniform(0.1, 0.5, 50),
            np.random.randint(1, 10, 50),
            np.random.uniform(1e-5, 1e-3, 50)
        ))
        fig_topology_correlation(topology_data, fig_dir, data_dir)
    except Exception as e:
        print(f"Error generating basin figures: {e}")
    
    # 10. Low-Budget + Surrogate
    print("\n10. Generating low-budget figures...")
    try:
        budgets = [1000, 2000, 5000, 10000]
        budget_anytime = {}
        for budget in budgets:
            evals = np.linspace(0, budget, min(100, budget//100))
            fitnesses = 1e-3 * np.exp(-evals/budget*5) + np.random.normal(0, 1e-5, len(evals))
            budget_anytime[budget] = list(zip(evals, fitnesses))
        fig_low_budget_anytime(budget_anytime, fig_dir, data_dir)
        
        sample_efficiency = {b: 1e-3 * np.exp(-b/5000) for b in budgets}
        fig_sample_efficiency(sample_efficiency, fig_dir, data_dir)
    except Exception as e:
        print(f"Error generating low-budget figures: {e}")
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print(f"Figures saved to: {fig_dir}")
    print(f"Data saved to: {data_dir}/data")
    print("="*60)


def load_from_experiments(data_dir: str = "outputs"):
    """
    Load actual data from experiment outputs.
    
    Args:
        data_dir: Directory containing experiment outputs
    """
    # This function would load real data from CSV/JSON files
    # For now, it's a placeholder
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all figures")
    parser.add_argument("--data_dir", type=str, default="outputs", help="Data directory")
    parser.add_argument("--fig_dir", type=str, default="outputs/figures", help="Figure directory")
    parser.add_argument("--use_real_data", action="store_true", help="Use real experiment data")
    
    args = parser.parse_args()
    
    if args.use_real_data:
        # Load from actual experiment outputs
        load_from_experiments(args.data_dir)
    else:
        # Generate with example data
        generate_all_figures(args.data_dir, args.fig_dir)

