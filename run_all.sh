#!/bin/bash
# Simple bash wrapper for run_all_experiments.py

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run the Python script with all arguments passed through
python3 run_all_experiments.py "$@"

