#!/usr/bin/env python3
"""
Master script to run all SOCIAL experiments and generate all results and figures.

This script:
1. Runs benchmark experiments (SOCIAL + baselines)
2. Runs statistical tests
3. Runs ablation studies
4. Runs centrality comparisons
5. Runs (K,p) sensitivity analysis
6. Generates all required figures

Usage:
    python run_all_experiments.py [--quick] [--skip_figures]
"""

import sys
import os
import argparse
import subprocess
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed!")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"ERROR: {description} failed with exception: {e}")
        return False


def run_python_script(script_path, args, description):
    """Run a Python script with arguments."""
    cmd = [sys.executable, script_path] + args
    return run_command(cmd, description)


def main():
    parser = argparse.ArgumentParser(
        description="Run all SOCIAL experiments and generate results/figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything (full experiments)
  python run_all_experiments.py
  
  # Quick run (fewer runs, smaller subset)
  python run_all_experiments.py --quick
  
  # Skip figure generation (only run experiments)
  python run_all_experiments.py --skip_figures
  
  # Custom number of runs
  python run_all_experiments.py --num_runs 10
        """
    )
    
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: fewer runs and smaller function subset")
    parser.add_argument("--skip_figures", action="store_true",
                       help="Skip figure generation (only run experiments)")
    parser.add_argument("--num_runs", type=int, default=None,
                       help="Number of runs per experiment (default: 30 for full, 10 for quick)")
    parser.add_argument("--max_evals", type=int, default=105000,
                       help="Maximum evaluations per run")
    parser.add_argument("--functions", nargs="+", default=None,
                       help="Specific functions to test (default: all or subset)")
    parser.add_argument("--outdir", type=str, default="outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Determine parameters
    if args.quick:
        num_runs = args.num_runs or 10
        if args.functions is None:
            test_functions = ["Sphere", "Rastrigin", "Ackley"]
        else:
            test_functions = args.functions
        print("\n" + "="*70)
        print("QUICK MODE: Running with reduced parameters")
        print("="*70)
    else:
        num_runs = args.num_runs or 30
        test_functions = args.functions
    
    # Create output directories
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(f"{args.outdir}/figures", exist_ok=True)
    os.makedirs(f"{args.outdir}/data", exist_ok=True)
    os.makedirs(f"{args.outdir}/tables", exist_ok=True)
    
    print("\n" + "="*70)
    print("SOCIAL EXPERIMENTS - MASTER RUNNER")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.outdir}")
    print(f"Number of runs: {num_runs}")
    print(f"Max evaluations: {args.max_evals}")
    if test_functions:
        print(f"Functions: {', '.join(test_functions)}")
    print("="*70)
    
    success_count = 0
    total_steps = 0
    
    # Step 1: Run benchmark experiments (SOCIAL + baselines)
    print("\n" + "="*70)
    print("STEP 1: Benchmark Experiments (SOCIAL vs Baselines)")
    print("="*70)
    
    if test_functions:
        func_args = ["--functions"] + test_functions
    else:
        func_args = []
    
    if run_python_script(
        "experiments/run_benchmarks.py",
        [
            "--max_evals", str(args.max_evals),
            "--num_runs", str(num_runs),
            "--outdir", args.outdir
        ] + func_args,
        "Benchmark comparison experiments"
    ):
        success_count += 1
    total_steps += 1
    
    # Step 2: Statistical tests
    print("\n" + "="*70)
    print("STEP 2: Statistical Tests")
    print("="*70)
    
    # Find the most recent benchmark raw results file
    import glob
    raw_files = glob.glob(f"{args.outdir}/benchmark_raw_*.json")
    if raw_files:
        latest_raw = max(raw_files, key=os.path.getctime)
        if run_python_script(
            "experiments/statistical_tests.py",
            ["--results", latest_raw, "--outdir", args.outdir],
            "Statistical tests (Friedman + Wilcoxon)"
        ):
            success_count += 1
    else:
        print("WARNING: No benchmark raw results found. Skipping statistical tests.")
    total_steps += 1
    
    # Step 3: Ablation study
    print("\n" + "="*70)
    print("STEP 3: Ablation Study")
    print("="*70)
    
    if test_functions:
        func_args = ["--functions"] + test_functions[:3]  # Use subset for ablation
    else:
        func_args = []
    
    if run_python_script(
        "experiments/ablation.py",
        [
            "--max_evals", str(args.max_evals),
            "--num_runs", str(num_runs),
            "--outdir", args.outdir
        ] + func_args,
        "Ablation study"
    ):
        success_count += 1
    total_steps += 1
    
    # Step 4: Centrality comparison
    print("\n" + "="*70)
    print("STEP 4: Centrality Comparison")
    print("="*70)
    
    if test_functions:
        func_args = ["--functions"] + test_functions[:3]
    else:
        func_args = []
    
    if run_python_script(
        "experiments/centrality_comparison.py",
        [
            "--max_evals", str(args.max_evals),
            "--num_runs", str(num_runs),
            "--outdir", args.outdir
        ] + func_args,
        "Centrality comparison"
    ):
        success_count += 1
    total_steps += 1
    
    # Step 5: (K,p) heatmaps
    print("\n" + "="*70)
    print("STEP 5: (K,p) Sensitivity Heatmaps")
    print("="*70)
    
    if test_functions:
        func_args = ["--functions"] + test_functions[:3]
    else:
        func_args = []
    
    # Use fewer runs for heatmaps (they're expensive)
    heatmap_runs = max(5, num_runs // 3)
    
    if run_python_script(
        "experiments/kp_heatmaps.py",
        [
            "--max_evals", str(args.max_evals),
            "--num_runs", str(heatmap_runs),
            "--outdir", args.outdir
        ] + func_args,
        "(K,p) sensitivity heatmaps"
    ):
        success_count += 1
    total_steps += 1
    
    # Step 6: Generate all figures
    if not args.skip_figures:
        print("\n" + "="*70)
        print("STEP 6: Generate All Figures")
        print("="*70)
        
        if run_python_script(
            "experiments/generate_figures.py",
            ["--data_dir", args.outdir, "--fig_dir", f"{args.outdir}/figures"],
            "Figure generation"
        ):
            success_count += 1
        total_steps += 1
    else:
        print("\n" + "="*70)
        print("STEP 6: Skipping figure generation (--skip_figures)")
        print("="*70)
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Completed: {success_count}/{total_steps} steps")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {args.outdir}/")
    print(f"Figures saved to: {args.outdir}/figures/")
    print(f"Data saved to: {args.outdir}/data/")
    print(f"Tables saved to: {args.outdir}/tables/")
    print("="*70)
    
    if success_count == total_steps:
        print("\n✓ All experiments completed successfully!")
        return 0
    else:
        print(f"\n⚠ Some experiments failed ({success_count}/{total_steps} succeeded)")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

