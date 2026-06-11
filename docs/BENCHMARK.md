# Benchmark Workspace

This repository contains the benchmark framework, the 6DOF method
implementations, compact datasets, and generated paper/site artifacts.

## Layout

- `benchmark/` orchestration, method registry, schema, website export, and plugin API.
- `models/aircraft6dof/` canonical 6DOF dynamics, synthetic dataset generation, the comparison suite, grey-box OEM/EKF/UDE method code, and the flight-explorer export.
- `dataset_tools/` dataset manifests, validation, and the real-flight conversion pipelines (`sportcub_mocap_4_17_26`, `sportcub_mocap_5_22_26`).
- `methods/plugins/` contributed method plugins.
- `data/` compact committed datasets only.
- `work/` ignored local raw/generated data cache.
- `latex/fig/` generated paper figures.
- `latex/tables/` generated CSV/LaTeX tables.
- `results/` raw metrics and metadata.
- `site/` static benchmark website, including the flight explorer.

## Current Commands

Generate the synthetic 6DOF datasets, run the suite on them, and refresh all
paper/site assets in one step:

```bash
./results.py all-6dof
```

Synthetic benchmark datasets are generated into ignored `work/data/`
directories before methods run. They use the same canonical NPZ keys as real
datasets, but are not committed because the simulator and nonlinear aero model
are expected to change. To generate the dataset family without running
methods:

```bash
./results.py simulate-6dof --dataset-modes open_loop sine_sweep aggressive trim_grid
```

The full review-oriented comparison runs the suite over the synthetic family
and both real Sport Cub mocap datasets:

```bash
python3 -m models.aircraft6dof.comparison_suite \
  --datasets work/data/aircraft_6dof_open_loop work/data/aircraft_6dof_sine_sweep \
    work/data/aircraft_6dof_aggressive work/data/aircraft_6dof_trim_grid \
    data/sportcub_mocap_4_17_26_train.npz data/sportcub_mocap_5_22_26_train.npz \
  --results-dir results --fig-dir latex/fig --table-dir latex/tables
```

This writes `results/aircraft6dof_method_comparison.csv`, updates
`latex/tables/aircraft6dof_method_comparison.tex`, creates the
`latex/fig/aircraft6dof_*` figures, and writes per-dataset method traces. The
torch-based 6DOF-UDE-NN row only runs when torch is importable (use
`.venv/bin/python`, set up via `./setup_env.sh`). Local linear,
model-stitching, subspace, and frequency-stitching methods train on the
trim-grid split; global residual, surrogate, symbolic, SINDy, and
output-error-style methods train on the aggressive split; on real datasets
every method trains on that dataset's train split. All fitted models are
validated open-loop on each requested validation dataset, and the validation
trajectory score is a normalized aggregate error, so lower is better.

To refresh only the LaTeX-ready and website assets from existing results:

```bash
./results.py latex-assets
./results.py web-data
```

Fast repository health checks and the paper build:

```bash
./results.py check-setup
./results.py build
```

## Real-Flight Datasets

The real Sport Cub mocap datasets are registered as `sportcub_mocap_4_17_26`
and `sportcub_mocap_5_22_26`. Large raw data is not stored in git; the
manifests under `work/data/<dataset_id>/` record the cloud source. To work
with one locally:

```bash
./results.py fetch-dataset sportcub_mocap_5_22_26
./results.py process-dataset sportcub_mocap_5_22_26
./results.py canonicalize-dataset sportcub_mocap_5_22_26
./results.py check-data sportcub_mocap_5_22_26
```

The reusable Sport Cub 6DOF grey-box model lives in
`models.aircraft6dof.greybox`, with the OEM/EKF fitting machinery in
`models.aircraft6dof.greybox_oem_fit`.

The browser flight explorer consumes a per-dataset JSON payload exported by:

```bash
python3 -m models.aircraft6dof.flight_explorer_export            # 5/22 full flights
python3 -m models.aircraft6dof.flight_explorer_export --dataset 4_17  # 4/17 maneuver windows
```

## Benchmark Policy

Every method consumes the same train/validation splits for a given dataset and
is scored by the same open-loop validation rollout. There is a single
observation policy: methods receive motion-capture position/attitude, with
states derived by differentiation; any further smoothing or state estimation
is part of the method under test. Methods are
registered in `benchmark/registry.py`; contributed methods plug in via
`methods/plugins/<name>/method.json` and are smoke-tested in CI.
