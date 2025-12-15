"""Statistical tests: Friedman, Wilcoxon signed-rank, corrections."""

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from typing import Dict, List, Tuple
import itertools


def friedman_test(results: Dict[str, List[float]]) -> Dict:
    """
    Perform Friedman test across algorithms.
    
    Args:
        results: Dictionary mapping algorithm name -> list of fitness values
        
    Returns:
        Dictionary with test results
    """
    algorithms = list(results.keys())
    n_algorithms = len(algorithms)
    
    if n_algorithms < 2:
        raise ValueError("Need at least 2 algorithms for Friedman test")
    
    # Get number of functions/problems
    n_functions = len(results[algorithms[0]])
    
    # Check all algorithms have same number of results
    for alg in algorithms:
        if len(results[alg]) != n_functions:
            raise ValueError(f"Algorithm {alg} has {len(results[alg])} results, expected {n_functions}")
    
    # Prepare data matrix: rows = functions, columns = algorithms
    data_matrix = np.array([results[alg] for alg in algorithms]).T
    
    # Perform Friedman test
    statistic, p_value = friedmanchisquare(*data_matrix.T)
    
    # Compute average ranks
    ranks = np.zeros((n_functions, n_algorithms))
    for i in range(n_functions):
        row = data_matrix[i, :]
        # Rank from best (lowest) to worst (highest)
        sorted_indices = np.argsort(row)
        for rank, idx in enumerate(sorted_indices):
            ranks[i, idx] = rank + 1
    
    avg_ranks = np.mean(ranks, axis=0)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'algorithms': algorithms,
        'avg_ranks': avg_ranks,
        'ranks': ranks,
        'n_functions': n_functions,
        'n_algorithms': n_algorithms
    }


def wilcoxon_signed_rank_test(results1: List[float], results2: List[float],
                              alternative: str = 'two-sided') -> Dict:
    """
    Perform Wilcoxon signed-rank test between two algorithms.
    
    Args:
        results1: Fitness values for algorithm 1
        results2: Fitness values for algorithm 2
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Dictionary with test results
    """
    if len(results1) != len(results2):
        raise ValueError("Results must have same length")
    
    # Compute differences
    differences = np.array(results1) - np.array(results2)
    
    # Perform test
    statistic, p_value = wilcoxon(differences, alternative=alternative)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'mean_diff': np.mean(differences),
        'n': len(results1)
    }


def holm_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[int, float, float, bool]]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        List of tuples (index, original_p, corrected_p, significant)
    """
    n = len(p_values)
    indexed_p = [(i, p) for i, p in enumerate(p_values)]
    indexed_p.sort(key=lambda x: x[1])  # Sort by p-value
    
    corrected = []
    for k, (i, p) in enumerate(indexed_p):
        corrected_alpha = alpha / (n - k)
        corrected_p = min(1.0, p * (n - k))
        significant = corrected_p < alpha
        corrected.append((i, p, corrected_p, significant))
    
    # Sort back by original index
    corrected.sort(key=lambda x: x[0])
    return corrected


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[Tuple[int, float, float, bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        List of tuples (index, original_p, corrected_p, significant)
    """
    n = len(p_values)
    corrected_alpha = alpha / n
    
    corrected = []
    for i, p in enumerate(p_values):
        corrected_p = min(1.0, p * n)
        significant = corrected_p < alpha
        corrected.append((i, p, corrected_p, significant))
    
    return corrected


def pairwise_wilcoxon_tests(results: Dict[str, List[float]],
                           reference: str = None,
                           correction: str = 'holm',
                           alpha: float = 0.05) -> pd.DataFrame:
    """
    Perform pairwise Wilcoxon signed-rank tests with correction.
    
    Args:
        results: Dictionary mapping algorithm name -> list of fitness values
        reference: Reference algorithm (if None, compare all pairs)
        correction: Correction method ('holm', 'bonferroni', or 'none')
        alpha: Significance level
        
    Returns:
        DataFrame with test results
    """
    algorithms = list(results.keys())
    
    if reference is None:
        # Compare all pairs
        pairs = list(itertools.combinations(algorithms, 2))
    else:
        # Compare reference against all others
        if reference not in algorithms:
            raise ValueError(f"Reference algorithm {reference} not found")
        pairs = [(reference, alg) for alg in algorithms if alg != reference]
    
    test_results = []
    p_values = []
    
    for alg1, alg2 in pairs:
        res1 = results[alg1]
        res2 = results[alg2]
        
        test = wilcoxon_signed_rank_test(res1, res2)
        
        test_results.append({
            'Algorithm_1': alg1,
            'Algorithm_2': alg2,
            'Statistic': test['statistic'],
            'p_value': test['p_value'],
            'Mean_Diff': test['mean_diff'],
            'n': test['n']
        })
        p_values.append(test['p_value'])
    
    df = pd.DataFrame(test_results)
    
    # Apply correction
    if correction == 'holm':
        corrected = holm_correction(p_values, alpha)
        df['p_value_corrected'] = [c[2] for c in corrected]
        df['Significant'] = [c[3] for c in corrected]
    elif correction == 'bonferroni':
        corrected = bonferroni_correction(p_values, alpha)
        df['p_value_corrected'] = [c[2] for c in corrected]
        df['Significant'] = [c[3] for c in corrected]
    else:
        df['p_value_corrected'] = df['p_value']
        df['Significant'] = df['p_value'] < alpha
    
    return df


def generate_latex_table(friedman_result: Dict, wilcoxon_df: pd.DataFrame,
                        filename: str = None) -> str:
    """
    Generate LaTeX table from statistical test results.
    
    Args:
        friedman_result: Result from friedman_test()
        wilcoxon_df: DataFrame from pairwise_wilcoxon_tests()
        filename: Optional filename to save table
        
    Returns:
        LaTeX table string
    """
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Friedman Test Results}")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\hline")
    latex.append("Algorithm & Average Rank & p-value \\\\")
    latex.append("\\hline")
    
    algorithms = friedman_result['algorithms']
    avg_ranks = friedman_result['avg_ranks']
    
    for alg, rank in zip(algorithms, avg_ranks):
        latex.append(f"{alg} & {rank:.3f} & {friedman_result['p_value']:.4f} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    latex.append("\n\n")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Pairwise Wilcoxon Signed-Rank Tests}")
    latex.append("\\begin{tabular}{lccccc}")
    latex.append("\\hline")
    latex.append("Algorithm 1 & Algorithm 2 & Statistic & p-value & Corrected p-value & Significant \\\\")
    latex.append("\\hline")
    
    for _, row in wilcoxon_df.iterrows():
        sig = "Yes" if row['Significant'] else "No"
        latex.append(f"{row['Algorithm_1']} & {row['Algorithm_2']} & "
                    f"{row['Statistic']:.3f} & {row['p_value']:.4f} & "
                    f"{row['p_value_corrected']:.4f} & {sig} \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    latex_str = "\n".join(latex)
    
    if filename:
        with open(filename, 'w') as f:
            f.write(latex_str)
    
    return latex_str

