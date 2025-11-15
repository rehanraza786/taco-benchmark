"""
TACO: Tabular Corruptions Benchmark (tabular_c package).

This package exposes:
- tabular_c.benchmark: core benchmarking loops
- tabular_c.datasets: dataset loaders
- tabular_c.corruptions: corruption functions and CORRUPTION_FUNCS
- tabular_c.models: model builders and evaluators
- tabular_c.api: public programmatic entrypoints (run_benchmark)
"""
__all__ = ["benchmark", "datasets", "corruptions", "models", "plots", "utils", "cli", "api"]
__version__ = "0.1.0"

from .api import run_benchmark
