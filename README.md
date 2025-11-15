# TACO: Tabular Corruptions Benchmark

![CI](/../../actions/workflows/ci.yml/badge.svg)

<p align="center">
  <img src="assets/logo.png" alt="TACO Benchmark Logo" width="320">
</p>

## Install

```bash
pip install -r requirements.txt
```

## Datasets

Place the required CSVs under `./data/`:

- `adult.csv` (UCI Adult)
- `diabetes_130_us_hospitals.csv` (UCI)
- `porto_seguro.csv` (Kaggle)
- `ieee_cis_fraud.csv` (Kaggle merged)
- `nyc_property_sales.csv` (NYC Open Data / Kaggle)

## Running the benchmark

**Option 1: via CLI module**

```bash
python -m tabular_c.cli --data_dir data --results_dir results
```

**Option 2: via helper script**

```bash
python run_all.py --data_dir data --results_dir results
```

Use `--quick` to run a faster sanity check with a single severity level (0.2):

```bash
python run_all.py --quick
```

Or limit to a single dataset, e.g. Adult only:

```bash
python run_all.py --only adult
```

## Package structure

```text
taco-benchmark/
├─ tabular_c/
│  ├─ __init__.py
│  ├─ benchmark.py     # core evaluation loops
│  ├─ datasets.py      # dataset loaders
│  ├─ corruptions.py   # corruption functions + CORRUPTION_FUNCS
│  ├─ models.py        # model builders and evaluators
│  ├─ plots.py         # plotting helpers
│  ├─ utils.py         # small utilities (ensure_dir, etc.)
│  └─ cli.py           # command-line interface
├─ run_all.py          # convenience wrapper: python run_all.py
├─ requirements.txt
└─ README.md
```

## License

This code is released under the MIT License (see `LICENSE`). If you use it
in academic work, please credit the original author:

> TACO: Tabular Corruptions Benchmark — Rehan Azam (CS229 Project)
## Installation as a package

With `pyproject.toml` included, you can install this repo in editable mode:

```bash
pip install -e .
```

This exposes a console script:

```bash
taco-benchmark --data_dir data --results_dir results
```

which is equivalent to:

```bash
python -m tabular_c.cli --data_dir data --results_dir results
```

and you can also use the programmatic API:

```python
from tabular_c import run_benchmark

run_benchmark(
    dataset_name="adult",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    results_dir="results",
)
```

## Generating plots

After running the benchmark and producing metrics CSVs under `results/`,
you can generate example figures with:

```bash
python scripts/make_plots.py
```