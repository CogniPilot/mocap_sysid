# Contributing Methods

Benchmark contributions use a two-phase process so public pull requests can add code without requiring trusted benchmark execution.

## Phase 1: Method Pull Request

A contributor submits a pull request that adds or updates method code under:

```text
methods/plugins/<method_name>/
```

The PR should include:

- `method.json`
- `method.py`
- `README.md`
- any small helper files needed by the method
- a short explanation of expected training data and observation assumptions

The PR should not include regenerated benchmark results, paper figures, or website data. In particular, contributors should avoid editing:

```text
results/
latex/tables/
latex/fig/
latex/generated/
latex/fig/generated_*
site/public/data/
```

GitHub-hosted CI only runs lightweight checks:

- Python syntax checks for benchmark API files
- plugin metadata validation
- plugin import validation
- smoke validation through `python3 -m benchmark.smoke_plugin`

These checks are intentionally CPU-only and do not run the full benchmark suite.

## Phase 2: Maintainer Benchmark Commit

After the method PR is reviewed and merged, a maintainer runs the full benchmark on trusted hardware, typically a local NVIDIA machine.

Recommended local command sequence:

```bash
git pull
.venv/bin/python -m models.aircraft6dof.comparison_suite \
  --datasets work/data/aircraft_6dof_open_loop work/data/aircraft_6dof_sine_sweep \
    work/data/aircraft_6dof_aggressive work/data/aircraft_6dof_trim_grid \
    data/sportcub_mocap_4_17_26_train.npz data/sportcub_mocap_5_22_26_train.npz \
  --results-dir results --fig-dir latex/fig --table-dir latex/tables
./results.py latex-assets
./results.py web-data
python3 latex/paper.py build
```

Then review the changed result files, figures, website data, and paper output. If the results are acceptable, commit them separately:

```bash
git add results latex/tables latex/fig latex/generated site/public/data latex/main.pdf
git commit -m "Update benchmark results for <method-name>"
git push
```

The normal GitHub Pages workflow publishes the updated static site after this trusted results commit reaches `main`.

## Why Results Are Separate

Full benchmarks execute contributed method code, may use GPU resources, and can take significant time. Keeping the result-generation commit separate provides:

- a clear review boundary between method code and benchmark artifacts,
- protection for local/self-hosted GPU machines,
- reproducible maintainer-controlled benchmark settings,
- a clean paper and website update history.
