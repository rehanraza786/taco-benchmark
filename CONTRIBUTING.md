# Contributing to TACO: Tabular Corruptions Benchmark

Thanks for your interest in contributing to the TACO benchmark! This project
aims to provide an ImageNet-C–style robustness benchmark for tabular models.

## Code of Conduct

Be respectful and constructive. Assume good intent, keep feedback specific
and actionable, and avoid personal attacks. This is a student / research
project, so kindness and patience are appreciated.

## How to Get Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone git@github.com:<your-username>/taco-benchmark.git
   cd taco-benchmark
   ```
3. Install in editable mode (recommended):
   ```bash
   pip install -e .
   ```
4. Make sure basic checks pass:
   ```bash
   python -m compileall tabular_c
   ```

## Project Layout

- `tabular_c/` — core package
  - `datasets.py` — dataset loaders
  - `corruptions.py` — corruption functions and `CORRUPTION_FUNCS`
  - `benchmark.py` — evaluation loops
  - `models.py` — model builders and evaluators
  - `api.py` — public `run_benchmark(...)` function
  - `cli.py` — command-line interface
- `scripts/` — helper scripts (e.g., `make_paper_plots.py`)
- `paper/` — NeurIPS-style writeup (LaTeX)
- `SPEC.md` — benchmark specification
- `README.md` — project overview and usage

## Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-change
   ```
2. Make your edits to the appropriate files.
3. Make sure the package still installs and imports:
   ```bash
   pip install -e .
   python -c "import tabular_c; print(tabular_c.__version__)"
   ```
4. If you modify public APIs or metrics output, please update:
   - `SPEC.md` (to reflect the new behavior)
   - `README.md` (usage examples)
   - Any relevant docstrings.

## Adding Datasets or Corruptions

- **New datasets**:
  - Add a loader to `tabular_c/datasets.py` returning `(X_train, X_test, y_train, y_test)`.
  - Update the CLI in `tabular_c/cli.py` to optionally run on the new dataset.
  - Document any download / preprocessing steps in the README.

- **New corruptions**:
  - Add a function to `tabular_c/corruptions.py` with signature
    `fn(X, y, severity, rng=None) -> (X_corrupted, y_corrupted)`.
  - Register it in `CORRUPTION_FUNCS` so it participates in the benchmark.
  - If you introduce new severity semantics, briefly document them in `SPEC.md`.

## Tests and Continuous Integration

A simple GitHub Actions workflow runs on every push and pull request:
- Installs the package
- Compiles `tabular_c` to bytecode (`python -m compileall tabular_c`)

This ensures at least that syntax and imports are valid. If you add tests
in the future (e.g., using `pytest`), feel free to extend the workflow.

## Submitting a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/my-change
   ```
2. Open a pull request on GitHub against the `main` branch of the upstream repo.
3. In the PR description, briefly explain:
   - What you changed
   - Why you changed it
   - Any backward-incompatible changes or caveats

Thanks again for helping improve the TACO benchmark!
