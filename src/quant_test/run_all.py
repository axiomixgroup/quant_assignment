#!/usr/bin/env python3
"""
Script to run all quant test questions in sequence.
"""

import os
import sys
import time
import subprocess
import shutil
import importlib

def print_header(text):
    """Print a header with the given text."""
    print("\n" + "=" * 80)
    print(f"Running {text}")
    print("=" * 80 + "\n")

def run_module(module_name):
    """Run a Python module and return the exit code."""
    start_time = time.time()
    
    try:
        # Import the module
        module = importlib.import_module(module_name)
        # Run the main function
        result = module.main()
        returncode = 0 if result is None else result
    except Exception as e:
        print(f"Error running {module_name}: {e}")
        returncode = 1
    
    end_time = time.time()
    
    if returncode == 0:
        print(f"\nModule completed successfully in {end_time - start_time:.2f} seconds.")
    else:
        print(f"\nModule failed with exit code {returncode} after {end_time - start_time:.2f} seconds.")
    
    return returncode

def main():
    """Run all quant test questions."""
    print("Quant Test Assignment Runner")
    print("===========================\n")
    
    # Ensure plots directory exists and clean it
    plots_dir = os.path.join(os.getcwd(), 'plots')
    if os.path.exists(plots_dir):
        print(f"Cleaning plots directory: {plots_dir}")
        for file in os.listdir(plots_dir):
            file_path = os.path.join(plots_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        print(f"Creating plots directory: {plots_dir}")
        os.makedirs(plots_dir)
    
    # List of modules to run
    modules = [
        "quant_test.question1_starter",
        "quant_test.question2_starter",
        "quant_test.question3_starter"
    ]
    
    # Run each module
    exit_codes = []
    for module in modules:
        print_header(module)
        exit_code = run_module(module)
        exit_codes.append(exit_code)
    
    # Print summary
    print("\nSummary:")
    print("========")
    for module, exit_code in zip(modules, exit_codes):
        status = "SUCCESS" if exit_code == 0 else f"FAILED (code {exit_code})"
        print(f"{module}: {status}")
    
    # Return non-zero exit code if any module failed
    return 1 if any(exit_codes) else 0

if __name__ == "__main__":
    sys.exit(main()) 