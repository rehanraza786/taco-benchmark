"""
Convenience entrypoint for the TACO benchmark.

Usage:
    python run_all.py --data_dir data --results_dir results

This simply forwards to tabular_c.cli.main().
"""
from tabular_c.cli import main

if __name__ == "__main__":
    main()
