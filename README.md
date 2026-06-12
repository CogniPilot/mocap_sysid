# Motion Capture based System Identifiation

Website: https://cognipilot.github.io/mocap_sysid/

Per-method implementation details and design justifications: the
"Implementation Details and Design Justifications" appendix of the paper
(latex/main.tex).

This project compares the latest machine-learning-based and classical
system-identification methods on fixed-wing aircraft problems. The goal is to
make method comparisons reproducible across shared simulated and real-flight
datasets, with common data formats, common metrics, generated paper artifacts,
and a static benchmark browser published through GitHub Pages.

## Repository Layout

- `benchmark/`: orchestration, registries, schema, website export, and plugin API.
- `models/`: canonical 6DOF aircraft dynamics and synthetic dataset helpers.
- `methods/`: method implementations and plugin examples.
- `dataset_tools/`: contributed dataset manifests, validation, and preprocessing code.
- `data/`: compact committed real datasets only, as flat NPZ split files.
- `work/`: ignored raw downloads, intermediate processing, and generated simulator data.
- `results/`: benchmark CSVs and metadata.
- `latex/fig/` and `latex/tables/`: generated paper figures and tables.
- `site/`: static GitHub Pages application.

## Quick Local Test

Run the setup smoke check:

```bash
./results.py check-setup
```

This compiles the benchmark modules, validates contributed dataset manifests and
compact NPZ files, discovers method plugins, regenerates website JSON, and runs a
small deterministic 6DOF model smoke simulation.

Serve the benchmark website locally:

```bash
./results.py serve-site
```

Open the printed `http://127.0.0.1:<port>` URL.

## Data Format

Committed real datasets use flat split files:

```text
data/<dataset_id>_train.npz
data/<dataset_id>_validation.npz
```

Each NPZ stores the canonical ragged time-series arrays plus scalar
`dataset_id` and `split_name` fields. Validate committed datasets with:

```bash
./results.py check-data
```

The current real datasets are `sportcub_mocap_4_17_26` (maneuver windows) and
`sportcub_mocap_5_22_26` (full flights); their raw Sport Cub data is not
committed. The dataset manifests and conversion code live under
`dataset_tools/<dataset_id>/`.

See [docs/DATASET_CONTRACT.md](docs/DATASET_CONTRACT.md) for the required NPZ
schema and [docs/BENCHMARK.md](docs/BENCHMARK.md) for the full workflow.

## 6DOF Benchmarks

Generate the standard 6DOF synthetic dataset family into ignored `work/data/`
directories:

```bash
./results.py simulate-6dof --dataset-modes open_loop sine_sweep aggressive trim_grid
```

Run the full local 6DOF baseline workflow:

```bash
./results.py all-6dof
```

This generates synthetic train/validation data, runs the implemented 6DOF
baselines with explicit method-specific training splits, exports website JSON,
and refreshes LaTeX-ready figures/tables. Synthetic simulator outputs are not
committed because the 6DOF aero model is expected to change.

The current 6DOF dynamics are a nonlinear small-aircraft benchmark with
body-axis aerodynamic forces and moments, smooth stall onset, post-stall drag
rise, control-effectiveness loss, actuator lag, direct-state measurements, and
mocap-style position/quaternion measurements.

## Sport Cub Real Data

The Sport Cub motion-capture datasets are registered as
`sportcub_mocap_4_17_26` and `sportcub_mocap_5_22_26`.

```bash
./results.py process-dataset sportcub_mocap_5_22_26
./results.py canonicalize-dataset sportcub_mocap_5_22_26
./results.py check-data sportcub_mocap_5_22_26
```

`canonicalize-dataset` writes the flat committed NPZ files under `data/`. The
comparison suite consumes those splits directly (see below), and the browser
flight explorer payloads are exported by
`python3 -m models.aircraft6dof.flight_explorer_export`.

## Method Contributions

Method contributions are split from benchmark-result generation:

1. A method PR adds plugin code and docs under `methods/plugins/<method_name>/`.
2. CI validates imports, plugin metadata, website data, and compact datasets.
3. A maintainer runs full benchmarks on trusted hardware and commits regenerated
   results separately.

See [docs/CONTRIBUTING_METHODS.md](docs/CONTRIBUTING_METHODS.md).

## Website And Release

The static site reads JSON from `site/public/data/`. `./results.py web-data`
refreshes that bundle locally. GitHub Actions runs CI on PRs, `main`, and
release tags matching `v*`; the Pages workflow deploys only after a successful
CI workflow run.

## Full Local Benchmark

For a full local run over the synthetic family and both real datasets (use the
project `.venv` so the torch-based UDE-NN row is included):

```bash
.venv/bin/python -m models.aircraft6dof.comparison_suite \
  --datasets work/data/aircraft_6dof_open_loop work/data/aircraft_6dof_sine_sweep \
    work/data/aircraft_6dof_aggressive work/data/aircraft_6dof_trim_grid \
    data/sportcub_mocap_4_17_26_train.npz data/sportcub_mocap_5_22_26_train.npz \
  --results-dir results --fig-dir latex/fig --table-dir latex/tables
./results.py latex-assets
./results.py web-data
python3 latex/paper.py build
```

See [docs/SELF_HOSTED_GPU.md](docs/SELF_HOSTED_GPU.md) for the self-hosted
runner setup.
