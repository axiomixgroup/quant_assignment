#!/usr/bin/env python3
"""
Command-line interface for the quant_test package.
"""

import argparse
import sys
from importlib.metadata import version
from quant_test import run_all

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Quant Test - A package for quantitative finance analysis."
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run analysis")
    run_parser.add_argument(
        "--all", action="store_true", help="Run all questions"
    )
    run_parser.add_argument(
        "--question", type=int, choices=[1, 2, 3], help="Run a specific question (1, 2, or 3)"
    )
    
    args = parser.parse_args()
    
    if args.version:
        try:
            print(version("quant-test"))
        except:
            from quant_test import __version__
            print(__version__)
        return 0
    
    if args.command == "run":
        if args.all:
            return run_all.main()
        elif args.question:
            if args.question == 1:
                from quant_test import question1_starter
                return question1_starter.main()
            elif args.question == 2:
                from quant_test import question2_starter
                return question2_starter.main()
            elif args.question == 3:
                from quant_test import question3_starter
                return question3_starter.main()
        else:
            print("Please specify --all or --question")
            return 1
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 