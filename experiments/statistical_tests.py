"""Run statistical tests on results."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import argparse
from social.stats import friedman_test, pairwise_wilcoxon_tests, generate_latex_table


def run_statistical_tests(results_file: str, outdir: str = "outputs"):
    """
    Run Friedman and Wilcoxon tests on results.
    
    Args:
        results_file: JSON file with raw results
        outdir: Output directory
    """
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "tables"), exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Get function names
    function_names = list(list(all_results.values())[0].keys())
    algorithm_names = list(all_results.keys())
    
    print(f"Running statistical tests on {len(function_names)} functions")
    print(f"Algorithms: {algorithm_names}")
    
    # Prepare data for Friedman test (per function)
    friedman_results_per_function = {}
    
    for func_name in function_names:
        # Extract results for this function
        func_results = {alg: all_results[alg][func_name] for alg in algorithm_names}
        
        # Friedman test
        friedman_result = friedman_test(func_results)
        friedman_results_per_function[func_name] = friedman_result
        
        print(f"\n{func_name}:")
        print(f"  Friedman statistic: {friedman_result['statistic']:.4f}")
        print(f"  p-value: {friedman_result['p_value']:.4e}")
        print("  Average ranks:")
        for alg, rank in zip(friedman_result['algorithms'], friedman_result['avg_ranks']):
            print(f"    {alg}: {rank:.3f}")
    
    # Overall Friedman test (across all functions)
    # Aggregate results: for each algorithm, collect all fitness values
    overall_results = {}
    for alg in algorithm_names:
        overall_results[alg] = []
        for func_name in function_names:
            overall_results[alg].extend(all_results[alg][func_name])
    
    overall_friedman = friedman_test(overall_results)
    print(f"\n{'='*60}")
    print("Overall Friedman Test (across all functions):")
    print(f"  Statistic: {overall_friedman['statistic']:.4f}")
    print(f"  p-value: {overall_friedman['p_value']:.4e}")
    print("  Average ranks:")
    for alg, rank in zip(overall_friedman['algorithms'], overall_friedman['avg_ranks']):
        print(f"    {alg}: {rank:.3f}")
    
    # Pairwise Wilcoxon tests (SOCIAL vs others)
    wilcoxon_df = pairwise_wilcoxon_tests(
        overall_results,
        reference='SOCIAL',
        correction='holm',
        alpha=0.05
    )
    
    print(f"\n{'='*60}")
    print("Pairwise Wilcoxon Signed-Rank Tests (SOCIAL vs others):")
    print(wilcoxon_df.to_string(index=False))
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    # Friedman summary
    friedman_summary = []
    for func_name, result in friedman_results_per_function.items():
        for alg, rank in zip(result['algorithms'], result['avg_ranks']):
            friedman_summary.append({
                'Function': func_name,
                'Algorithm': alg,
                'Average_Rank': rank
            })
    friedman_summary.append({
        'Function': 'Overall',
        'Algorithm': 'SOCIAL',
        'Average_Rank': overall_friedman['avg_ranks'][0]
    })
    
    friedman_df = pd.DataFrame(friedman_summary)
    friedman_file = os.path.join(outdir, f"friedman_summary_{timestamp}.csv")
    friedman_df.to_csv(friedman_file, index=False)
    print(f"\nFriedman summary saved to {friedman_file}")
    
    # Wilcoxon pairs
    wilcoxon_file = os.path.join(outdir, f"wilcoxon_pairs_{timestamp}.csv")
    wilcoxon_df.to_csv(wilcoxon_file, index=False)
    print(f"Wilcoxon pairs saved to {wilcoxon_file}")
    
    # LaTeX tables
    latex_file = os.path.join(outdir, "tables", f"statistical_tables_{timestamp}.tex")
    latex_str = generate_latex_table(overall_friedman, wilcoxon_df, filename=latex_file)
    print(f"LaTeX tables saved to {latex_file}")
    
    return friedman_results_per_function, overall_friedman, wilcoxon_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistical tests")
    parser.add_argument("--results", type=str, required=True, help="JSON file with raw results")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    run_statistical_tests(args.results, args.outdir)

