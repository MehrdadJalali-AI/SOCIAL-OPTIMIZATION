"""Plotting utilities for SOCIAL experiments - matplotlib only."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os
from typing import Dict, List, Tuple, Optional


# Set style
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    try:
        plt.style.use('seaborn-paper')
    except:
        plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def save_figure(fig, name: str, fig_dir: str = "outputs/figures", 
                data_dir: str = "outputs/data", data: Optional[pd.DataFrame] = None):
    """
    Save figure as both PNG and PDF, and save underlying data as CSV.
    
    Args:
        fig: matplotlib figure
        name: Figure name (without extension)
        fig_dir: Directory for figures
        data_dir: Directory for data
        data: Optional DataFrame with underlying data
    """
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Save PNG
    png_path = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    
    # Save PDF
    pdf_path = os.path.join(fig_dir, f"{name}.pdf")
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    # Save data if provided
    if data is not None:
        csv_path = os.path.join(data_dir, f"{name}_data.csv")
        data.to_csv(csv_path, index=False)
    
    plt.close(fig)


def fig_runtime_breakdown(runtime_data: Dict[int, Dict[str, float]], 
                          fig_dir: str = "outputs/figures", 
                          data_dir: str = "outputs/data"):
    """
    Stacked bar chart of runtime breakdown for different population sizes.
    
    Args:
        runtime_data: Dict mapping N -> {component: avg_time}
        fig_dir: Figure directory
        data_dir: Data directory
    """
    N_values = sorted(runtime_data.keys())
    components = ['t_eval', 't_centrality', 't_update', 't_mut', 't_rewire']
    component_labels = ['Evaluation', 'Centrality', 'Update', 'Mutation', 'Rewiring']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Prepare data
    data_rows = []
    bottom = np.zeros(len(N_values))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, (comp, label, color) in enumerate(zip(components, component_labels, colors)):
        values = [runtime_data[N].get(comp, 0) for N in N_values]
        ax.bar(range(len(N_values)), values, bottom=bottom, 
               label=label, color=color, alpha=0.8)
        
        for j, (N, val) in enumerate(zip(N_values, values)):
            data_rows.append({
                'N': N,
                'Component': label,
                'Time_ms': val * 1000  # Convert to ms
            })
        
        bottom += values
    
    ax.set_xlabel('Population Size (N)', fontweight='bold')
    ax.set_ylabel('Average Time per Iteration (s)', fontweight='bold')
    ax.set_title('Runtime Breakdown by Component', fontweight='bold')
    ax.set_xticks(range(len(N_values)))
    ax.set_xticklabels([f'N={N}' for N in N_values])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save
    data_df = pd.DataFrame(data_rows)
    save_figure(fig, 'fig_runtime_breakdown', fig_dir, data_dir, data_df)


def fig_bc_cadence_tradeoff(results: Dict[int, Dict[str, float]],
                           fig_dir: str = "outputs/figures",
                           data_dir: str = "outputs/data"):
    """
    Performance vs bc_interval and runtime vs bc_interval.
    
    Args:
        results: Dict mapping bc_interval -> {'performance': float, 'runtime': float}
        fig_dir: Figure directory
        data_dir: Data directory
    """
    intervals = sorted(results.keys())
    performances = [results[i]['performance'] for i in intervals]
    runtimes = [results[i]['runtime'] for i in intervals]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Performance plot
    ax1.plot(intervals, performances, 'o-', linewidth=2, markersize=8, color='#2ca02c')
    ax1.set_xlabel('Betweenness Centrality Interval', fontweight='bold')
    ax1.set_ylabel('Best Fitness', fontweight='bold')
    ax1.set_title('Performance vs BC Recompute Interval', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Runtime plot
    ax2.plot(intervals, runtimes, 's-', linewidth=2, markersize=8, color='#d62728')
    ax2.set_xlabel('Betweenness Centrality Interval', fontweight='bold')
    ax2.set_ylabel('Total Runtime (s)', fontweight='bold')
    ax2.set_title('Runtime vs BC Recompute Interval', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    plt.tight_layout()
    
    # Save data
    data_df = pd.DataFrame({
        'bc_interval': intervals,
        'performance': performances,
        'runtime_s': runtimes
    })
    save_figure(fig, 'fig_bc_cadence_tradeoff', fig_dir, data_dir, data_df)


def fig_centrality_rank(centrality_results: Dict[str, Dict[str, float]],
                       fig_dir: str = "outputs/figures",
                       data_dir: str = "outputs/data"):
    """
    Bar chart of average rank per centrality measure.
    
    Args:
        centrality_results: Dict mapping centrality -> {'avg_rank': float, ...}
        fig_dir: Figure directory
        data_dir: Data directory
    """
    centralities = list(centrality_results.keys())
    avg_ranks = [centrality_results[c]['avg_rank'] for c in centralities]
    
    # Sort by rank
    sorted_data = sorted(zip(centralities, avg_ranks), key=lambda x: x[1])
    centralities, avg_ranks = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(centralities)))
    bars = ax.bar(range(len(centralities)), avg_ranks, color=colors, alpha=0.8)
    
    ax.set_xlabel('Centrality Measure', fontweight='bold')
    ax.set_ylabel('Average Rank', fontweight='bold')
    ax.set_title('Average Rank by Centrality Measure', fontweight='bold')
    ax.set_xticks(range(len(centralities)))
    ax.set_xticklabels([c.capitalize() for c in centralities], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rank in zip(bars, avg_ranks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rank:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save data
    data_df = pd.DataFrame({
        'centrality': centralities,
        'avg_rank': avg_ranks
    })
    save_figure(fig, 'fig_centrality_rank', fig_dir, data_dir, data_df)


def fig_centrality_runtime(centrality_results: Dict[str, Dict[str, float]],
                          fig_dir: str = "outputs/figures",
                          data_dir: str = "outputs/data"):
    """
    Bar chart of runtime per centrality measure.
    
    Args:
        centrality_results: Dict mapping centrality -> {'avg_runtime': float, ...}
        fig_dir: Figure directory
        data_dir: Data directory
    """
    centralities = list(centrality_results.keys())
    avg_runtimes = [centrality_results[c]['avg_runtime'] for c in centralities]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0, 1, len(centralities)))
    bars = ax.bar(range(len(centralities)), avg_runtimes, color=colors, alpha=0.8)
    
    ax.set_xlabel('Centrality Measure', fontweight='bold')
    ax.set_ylabel('Average Runtime (s)', fontweight='bold')
    ax.set_title('Runtime by Centrality Measure', fontweight='bold')
    ax.set_xticks(range(len(centralities)))
    ax.set_xticklabels([c.capitalize() for c in centralities], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rt in zip(bars, avg_runtimes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rt:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save data
    data_df = pd.DataFrame({
        'centrality': centralities,
        'avg_runtime_s': avg_runtimes
    })
    save_figure(fig, 'fig_centrality_runtime', fig_dir, data_dir, data_df)


def fig_kp_heatmap(kp_data: pd.DataFrame, function_name: str,
                  fig_dir: str = "outputs/figures",
                  data_dir: str = "outputs/data"):
    """
    Heatmap of (K,p) sensitivity for a function.
    
    Args:
        kp_data: DataFrame with columns ['K', 'p', 'Mean']
        function_name: Name of function
        fig_dir: Figure directory
        data_dir: Data directory
    """
    # Pivot data
    pivot = kp_data.pivot(index='K', columns='p', values='Mean')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap='viridis_r', aspect='auto', 
                   interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f'{p:.1f}' for p in pivot.columns])
    ax.set_yticklabels([f'{k}' for k in pivot.index])
    
    # Labels
    ax.set_xlabel('Rewiring Probability (p)', fontweight='bold')
    ax.set_ylabel('Neighbors (K)', fontweight='bold')
    ax.set_title(f'{function_name} - (K,p) Sensitivity', fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Best Fitness', fontweight='bold')
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(j, i, f'{pivot.values[i, j]:.2e}',
                          ha="center", va="center", color="white", fontsize=7)
    
    plt.tight_layout()
    
    # Save
    save_figure(fig, f'fig_kp_heatmap_{function_name}', fig_dir, data_dir, kp_data)


def fig_kp_heatmap_runtime(kp_data: pd.DataFrame,
                           fig_dir: str = "outputs/figures",
                           data_dir: str = "outputs/data"):
    """
    Heatmap of runtime for (K,p) combinations.
    
    Args:
        kp_data: DataFrame with columns ['K', 'p', 'Runtime']
        fig_dir: Figure directory
        data_dir: Data directory
    """
    pivot = kp_data.pivot(index='K', columns='p', values='Runtime')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(pivot.values, cmap='plasma', aspect='auto',
                   interpolation='nearest')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f'{p:.1f}' for p in pivot.columns])
    ax.set_yticklabels([f'{k}' for k in pivot.index])
    
    ax.set_xlabel('Rewiring Probability (p)', fontweight='bold')
    ax.set_ylabel('Neighbors (K)', fontweight='bold')
    ax.set_title('Runtime - (K,p) Sensitivity', fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Runtime (s)', fontweight='bold')
    
    plt.tight_layout()
    
    save_figure(fig, 'fig_kp_heatmap_runtime', fig_dir, data_dir, kp_data)


def fig_rewiring_convergence(rewiring_data: Dict[str, List[Tuple[float, float]]],
                             fig_dir: str = "outputs/figures",
                             data_dir: str = "outputs/data"):
    """
    Best-so-far vs evaluations for different rewiring modes.
    
    Args:
        rewiring_data: Dict mapping mode -> [(eval, fitness), ...]
        fig_dir: Figure directory
        data_dir: Data directory
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(rewiring_data)))
    
    data_rows = []
    
    for (mode, data), color in zip(rewiring_data.items(), colors):
        evals, fitnesses = zip(*data)
        ax.plot(evals, fitnesses, label=mode.capitalize(), linewidth=2, color=color)
        
        for e, f in zip(evals, fitnesses):
            data_rows.append({
                'mode': mode,
                'evaluations': e,
                'best_fitness': f
            })
    
    ax.set_xlabel('Evaluations', fontweight='bold')
    ax.set_ylabel('Best Fitness', fontweight='bold')
    ax.set_title('Convergence: Rewiring Modes Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    data_df = pd.DataFrame(data_rows)
    save_figure(fig, 'fig_rewiring_convergence', fig_dir, data_dir, data_df)


def fig_rewiring_diversity(rewiring_data: Dict[str, List[Tuple[float, float]]],
                          fig_dir: str = "outputs/figures",
                          data_dir: str = "outputs/data"):
    """
    Diversity vs evaluations for different rewiring modes.
    
    Args:
        rewiring_data: Dict mapping mode -> [(eval, diversity), ...]
        fig_dir: Figure directory
        data_dir: Data directory
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(rewiring_data)))
    
    data_rows = []
    
    for (mode, data), color in zip(rewiring_data.items(), colors):
        evals, diversities = zip(*data)
        ax.plot(evals, diversities, label=mode.capitalize(), linewidth=2, color=color)
        
        for e, d in zip(evals, diversities):
            data_rows.append({
                'mode': mode,
                'evaluations': e,
                'diversity': d
            })
    
    ax.set_xlabel('Evaluations', fontweight='bold')
    ax.set_ylabel('Population Diversity', fontweight='bold')
    ax.set_title('Diversity: Rewiring Modes Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    data_df = pd.DataFrame(data_rows)
    save_figure(fig, 'fig_rewiring_diversity', fig_dir, data_dir, data_df)


def fig_schedule_curves(schedule_data: Dict[str, List[Tuple[float, float]]],
                       fig_dir: str = "outputs/figures",
                       data_dir: str = "outputs/data"):
    """
    Best-so-far vs evaluations for different schedules.
    
    Args:
        schedule_data: Dict mapping schedule -> [(eval, fitness), ...]
        fig_dir: Figure directory
        data_dir: Data directory
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    data_rows = []
    
    for (schedule, data), color in zip(schedule_data.items(), colors[:len(schedule_data)]):
        evals, fitnesses = zip(*data)
        ax.plot(evals, fitnesses, label=schedule.capitalize(), linewidth=2, color=color)
        
        for e, f in zip(evals, fitnesses):
            data_rows.append({
                'schedule': schedule,
                'evaluations': e,
                'best_fitness': f
            })
    
    ax.set_xlabel('Evaluations', fontweight='bold')
    ax.set_ylabel('Best Fitness', fontweight='bold')
    ax.set_title('Convergence: Schedule Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    data_df = pd.DataFrame(data_rows)
    save_figure(fig, 'fig_schedule_curves', fig_dir, data_dir, data_df)


def fig_phase_diversity(diversity_data: List[Tuple[float, float]],
                        improvement_data: List[Tuple[float, float]],
                        sync_events: List[float],
                        fig_dir: str = "outputs/figures",
                        data_dir: str = "outputs/data"):
    """
    Diversity + improvement curves with phase markers and sync events.
    
    Args:
        diversity_data: [(eval, diversity), ...]
        improvement_data: [(eval, improvement), ...]
        sync_events: List of evaluation numbers where sync occurred
        fig_dir: Figure directory
        data_dir: Data directory
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    evals_div, diversities = zip(*diversity_data)
    evals_imp, improvements = zip(*improvement_data)
    
    # Diversity plot
    ax1.plot(evals_div, diversities, 'b-', linewidth=2, label='Diversity')
    for sync in sync_events:
        ax1.axvline(x=sync, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_ylabel('Population Diversity', fontweight='bold')
    ax1.set_title('Diversity and Improvement with Synchronization Events', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Improvement plot
    ax2.plot(evals_imp, improvements, 'g-', linewidth=2, label='Improvement')
    for sync in sync_events:
        ax2.axvline(x=sync, color='r', linestyle='--', alpha=0.5, linewidth=1, label='Sync' if sync == sync_events[0] else '')
    ax2.set_xlabel('Evaluations', fontweight='bold')
    ax2.set_ylabel('Fitness Improvement', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save data
    data_df = pd.DataFrame({
        'evaluations': list(evals_div) + list(evals_imp),
        'diversity': list(diversities) + [np.nan] * len(improvements),
        'improvement': [np.nan] * len(diversities) + list(improvements),
        'sync_event': [1 if e in sync_events else 0 for e in list(evals_div) + list(evals_imp)]
    })
    save_figure(fig, 'fig_phase_diversity', fig_dir, data_dir, data_df)


def fig_ablation_bars(ablation_data: Dict[str, float],
                     fig_dir: str = "outputs/figures",
                     data_dir: str = "outputs/data"):
    """
    Bar chart of mean best fitness (or avg rank) across ablations.
    
    Args:
        ablation_data: Dict mapping variant -> mean_fitness or avg_rank
        fig_dir: Figure directory
        data_dir: Data directory
    """
    variants = list(ablation_data.keys())
    values = [ablation_data[v] for v in variants]
    
    # Sort by value
    sorted_data = sorted(zip(variants, values), key=lambda x: x[1])
    variants, values = zip(*sorted_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(variants)))
    bars = ax.barh(range(len(variants)), values, color=colors, alpha=0.8)
    
    ax.set_xlabel('Mean Best Fitness', fontweight='bold')
    ax.set_ylabel('Variant', fontweight='bold')
    ax.set_title('Ablation Study: Mean Best Fitness by Variant', fontweight='bold')
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([v.replace('_', ' ').title() for v in variants])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f'{val:.2e}', ha='left', va='center')
    
    plt.tight_layout()
    
    data_df = pd.DataFrame({
        'variant': variants,
        'mean_best_fitness': values
    })
    save_figure(fig, 'fig_ablation_bars', fig_dir, data_dir, data_df)


def fig_ablation_anytime(ablation_data: Dict[str, List[Tuple[float, float]]],
                        fig_dir: str = "outputs/figures",
                        data_dir: str = "outputs/data"):
    """
    Best-so-far vs evaluations for key ablations.
    
    Args:
        ablation_data: Dict mapping variant -> [(eval, fitness), ...]
        fig_dir: Figure directory
        data_dir: Data directory
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(ablation_data)))
    
    data_rows = []
    
    for (variant, data), color in zip(ablation_data.items(), colors):
        evals, fitnesses = zip(*data)
        label = variant.replace('_', ' ').title()
        ax.plot(evals, fitnesses, label=label, linewidth=2, color=color)
        
        for e, f in zip(evals, fitnesses):
            data_rows.append({
                'variant': variant,
                'evaluations': e,
                'best_fitness': f
            })
    
    ax.set_xlabel('Evaluations', fontweight='bold')
    ax.set_ylabel('Best Fitness', fontweight='bold')
    ax.set_title('Anytime Performance: Ablation Study', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    data_df = pd.DataFrame(data_rows)
    save_figure(fig, 'fig_ablation_anytime', fig_dir, data_dir, data_df)


def fig_feasibility_rate(feasibility_data: Dict[str, Dict[str, float]],
                         fig_dir: str = "outputs/figures",
                         data_dir: str = "outputs/data"):
    """
    Bar chart of feasibility rate per engineering problem.
    
    Args:
        feasibility_data: Dict mapping problem -> {algorithm: feasibility_rate}
        fig_dir: Figure directory
        data_dir: Data directory
    """
    problems = list(feasibility_data.keys())
    algorithms = list(list(feasibility_data.values())[0].keys())
    
    x = np.arange(len(problems))
    width = 0.8 / len(algorithms)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
    
    data_rows = []
    
    for i, alg in enumerate(algorithms):
        rates = [feasibility_data[prob][alg] for prob in problems]
        bars = ax.bar(x + i * width, rates, width, label=alg, color=colors[i], alpha=0.8)
        
        for prob, rate in zip(problems, rates):
            data_rows.append({
                'problem': prob,
                'algorithm': alg,
                'feasibility_rate': rate
            })
    
    ax.set_xlabel('Engineering Problem', fontweight='bold')
    ax.set_ylabel('Feasibility Rate', fontweight='bold')
    ax.set_title('Feasibility Rate by Problem and Algorithm', fontweight='bold')
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels([p.replace('_', ' ') for p in problems], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    data_df = pd.DataFrame(data_rows)
    save_figure(fig, 'fig_feasibility_rate', fig_dir, data_dir, data_df)


def fig_violation_box(violation_data: Dict[str, List[float]],
                     fig_dir: str = "outputs/figures",
                     data_dir: str = "outputs/data"):
    """
    Boxplot of constraint violation distribution.
    
    Args:
        violation_data: Dict mapping problem -> [violations, ...]
        fig_dir: Figure directory
        data_dir: Data directory
    """
    problems = list(violation_data.keys())
    data_to_plot = [violation_data[p] for p in problems]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bp = ax.boxplot(data_to_plot, labels=[p.replace('_', ' ') for p in problems],
                   patch_artist=True)
    
    # Color boxes
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(problems)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Engineering Problem', fontweight='bold')
    ax.set_ylabel('Constraint Violation', fontweight='bold')
    ax.set_title('Constraint Violation Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save data
    data_rows = []
    for prob, violations in violation_data.items():
        for v in violations:
            data_rows.append({'problem': prob, 'violation': v})
    data_df = pd.DataFrame(data_rows)
    save_figure(fig, 'fig_violation_box', fig_dir, data_dir, data_df)


def fig_discrete_modes(modes_data: Dict[str, Dict[str, float]],
                      fig_dir: str = "outputs/figures",
                      data_dir: str = "outputs/data"):
    """
    Compare rounding modes (objective + feasibility).
    
    Args:
        modes_data: Dict mapping mode -> {'objective': float, 'feasibility': float}
        fig_dir: Figure directory
        data_dir: Data directory
    """
    modes = list(modes_data.keys())
    objectives = [modes_data[m]['objective'] for m in modes]
    feasibilities = [modes_data[m]['feasibility'] for m in modes]
    
    x = np.arange(len(modes))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Objective plot
    bars1 = ax1.bar(x - width/2, objectives, width, label='Objective', color='#1f77b4', alpha=0.8)
    ax1.set_xlabel('Rounding Mode', fontweight='bold')
    ax1.set_ylabel('Best Objective Value', fontweight='bold')
    ax1.set_title('Objective Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in modes], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Feasibility plot
    bars2 = ax2.bar(x - width/2, feasibilities, width, label='Feasibility Rate', color='#2ca02c', alpha=0.8)
    ax2.set_xlabel('Rounding Mode', fontweight='bold')
    ax2.set_ylabel('Feasibility Rate', fontweight='bold')
    ax2.set_title('Feasibility Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', ' ').title() for m in modes], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    data_df = pd.DataFrame({
        'mode': modes,
        'objective': objectives,
        'feasibility_rate': feasibilities
    })
    save_figure(fig, 'fig_discrete_modes', fig_dir, data_dir, data_df)


def fig_integer_histograms(integer_data: Dict[str, List[int]],
                          fig_dir: str = "outputs/figures",
                          data_dir: str = "outputs/data"):
    """
    Histograms of selected integer variables to show no rounding bias.
    
    Args:
        integer_data: Dict mapping variable_name -> [values, ...]
        fig_dir: Figure directory
        data_dir: Data directory
    """
    n_vars = len(integer_data)
    n_cols = 2
    n_rows = (n_vars + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    if n_vars == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    data_rows = []
    
    for i, (var_name, values) in enumerate(integer_data.items()):
        ax = axes[i]
        ax.hist(values, bins=20, alpha=0.7, color='#9467bd', edgecolor='black')
        ax.set_xlabel('Integer Value', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{var_name}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for val in values:
            data_rows.append({'variable': var_name, 'value': val})
    
    # Hide unused subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    data_df = pd.DataFrame(data_rows)
    save_figure(fig, 'fig_integer_histograms', fig_dir, data_dir, data_df)


def fig_basins_over_time(basin_data: List[Tuple[float, int]],
                        fig_dir: str = "outputs/figures",
                        data_dir: str = "outputs/data"):
    """
    Number of basins discovered vs iterations/evals.
    
    Args:
        basin_data: [(eval, n_basins), ...]
        fig_dir: Figure directory
        data_dir: Data directory
    """
    evals, n_basins = zip(*basin_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(evals, n_basins, 'o-', linewidth=2, markersize=4, color='#2ca02c')
    ax.fill_between(evals, n_basins, alpha=0.3, color='#2ca02c')
    
    ax.set_xlabel('Evaluations', fontweight='bold')
    ax.set_ylabel('Number of Basins Discovered', fontweight='bold')
    ax.set_title('Basin Discovery Over Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    data_df = pd.DataFrame({
        'evaluations': evals,
        'n_basins': n_basins
    })
    save_figure(fig, 'fig_basins_over_time', fig_dir, data_dir, data_df)


def fig_topology_correlation(topology_data: List[Tuple[float, float, float]],
                            fig_dir: str = "outputs/figures",
                            data_dir: str = "outputs/data"):
    """
    Scatter plot: λ2 vs basin count / performance.
    
    Args:
        topology_data: [(lambda2, basin_count, performance), ...]
        fig_dir: Figure directory
        data_dir: Data directory
    """
    lambda2, basin_counts, performances = zip(*topology_data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # λ2 vs basin count
    scatter1 = ax1.scatter(lambda2, basin_counts, alpha=0.6, s=50, c=performances, cmap='viridis')
    ax1.set_xlabel('Algebraic Connectivity (λ₂)', fontweight='bold')
    ax1.set_ylabel('Basin Count', fontweight='bold')
    ax1.set_title('Topology vs Basin Coverage', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Performance', fontweight='bold')
    
    # λ2 vs performance
    scatter2 = ax2.scatter(lambda2, performances, alpha=0.6, s=50, c=basin_counts, cmap='plasma')
    ax2.set_xlabel('Algebraic Connectivity (λ₂)', fontweight='bold')
    ax2.set_ylabel('Performance', fontweight='bold')
    ax2.set_title('Topology vs Performance', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Basin Count', fontweight='bold')
    
    plt.tight_layout()
    
    data_df = pd.DataFrame({
        'lambda2': lambda2,
        'basin_count': basin_counts,
        'performance': performances
    })
    save_figure(fig, 'fig_topology_correlation', fig_dir, data_dir, data_df)


def fig_low_budget_anytime(budget_data: Dict[int, List[Tuple[float, float]]],
                          fig_dir: str = "outputs/figures",
                          data_dir: str = "outputs/data"):
    """
    Best-so-far vs evals at different budgets.
    
    Args:
        budget_data: Dict mapping budget -> [(eval, fitness), ...]
        fig_dir: Figure directory
        data_dir: Data directory
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(budget_data)))
    
    data_rows = []
    
    for (budget, data), color in zip(sorted(budget_data.items()), colors):
        evals, fitnesses = zip(*data)
        ax.plot(evals, fitnesses, label=f'Budget={budget}', linewidth=2, color=color)
        
        for e, f in zip(evals, fitnesses):
            data_rows.append({
                'budget': budget,
                'evaluations': e,
                'best_fitness': f
            })
    
    ax.set_xlabel('Evaluations', fontweight='bold')
    ax.set_ylabel('Best Fitness', fontweight='bold')
    ax.set_title('Low-Budget Performance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    data_df = pd.DataFrame(data_rows)
    save_figure(fig, 'fig_low_budget_anytime', fig_dir, data_dir, data_df)


def fig_sample_efficiency(budget_data: Dict[int, float],
                         fig_dir: str = "outputs/figures",
                         data_dir: str = "outputs/data"):
    """
    Final best vs budget (lines for different algorithms/methods).
    
    Args:
        budget_data: Dict mapping budget -> final_best_fitness
        fig_dir: Figure directory
        data_dir: Data directory
    """
    budgets = sorted(budget_data.keys())
    final_bests = [budget_data[b] for b in budgets]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(budgets, final_bests, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    
    ax.set_xlabel('Budget (Evaluations)', fontweight='bold')
    ax.set_ylabel('Final Best Fitness', fontweight='bold')
    ax.set_title('Sample Efficiency', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    data_df = pd.DataFrame({
        'budget': budgets,
        'final_best_fitness': final_bests
    })
    save_figure(fig, 'fig_sample_efficiency', fig_dir, data_dir, data_df)

