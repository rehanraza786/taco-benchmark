"""
TACO: Tabular Corruptions Benchmark (tabular_c package).

This package exposes:
- tabular_c.benchmark: core benchmarking loops
- tabular_c.datasets: dataset loaders
- tabular_c.corruptions: corruption functions and CORRUPTION_FUNCS
- tabular_c.models: model builders and evaluators
- tabular_c.api: public programmatic entrypoints (run_benchmark)
"""
# Import config FIRST to ensure threading environment variables
# (OMP_NUM_THREADS, etc.) are set before numpy/pandas/torch initialize in submodules.
from . import config
from .api import run_benchmark

__all__ = [
    "run_benchmark",
    "config",
    "benchmark",
    "datasets",
    "corruptions",
    "models",
    "plots",
    "utils",
    "cli",
    "api"
]

__version__ = "0.1.0"