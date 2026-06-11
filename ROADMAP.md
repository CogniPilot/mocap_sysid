# System Identification Benchmark Roadmap

This roadmap tracks the repository's evolution into a public, reproducible
benchmark platform with a static GitHub Pages site, a contribution path for
new methods, and a nonlinear 6DOF aircraft model family evaluated on both
synthetic and real motion-capture flight data.

## Goals

- Publish current benchmark results as an interactive public website.
- Make benchmark data machine-readable so figures, tables, and the website use the same source of truth.
- Let contributors add new identification methods through a small, documented plugin interface.
- Center the benchmark on real indoor motion-capture flight data, with synthetic 6DOF maneuver families for controlled comparisons.
- Preserve the paper workflow so generated figures and tables remain reproducible.

## Architecture

```text
system_identification/
  benchmark/            # schema, method API, registry, web export
  models/
    aircraft6dof/       # dynamics, dataset generator, comparison suite, grey-box methods
  methods/plugins/
    <method_name>/
      method.py
      method.json
      README.md
      test_smoke.py
  dataset_tools/        # manifests, validation, real-flight converters
  data/                 # compact committed datasets
  results/              # raw metrics and metadata
  site/                 # static website incl. flight explorer
  latex/                # paper sources and generated assets
  .github/workflows/
```

The benchmark runner remains Python-native. The website is static and
data-driven; client-side prediction free-runs are for exploration, not for
trusted benchmark execution.

## Data Contract

Every published dataset and result bundle should identify:

- `model_family`: `aircraft6dof`
- `scenario`: trim grid, open-loop, sine sweep, aggressive, or a real-flight dataset id
- `controller`: off, or the stock SAFE stabilization recorded in the real flights
- `inputs`: pilot commands used for validation rollouts
- `states`: true states when available (synthetic only)
- `observations`: measured channels exposed to methods (pose; states derived by the suite)
- `metrics`: validation NRMSE, per-state errors, training time, validation rollout time, failure status, and run metadata
- `provenance`: git SHA, command line, package versions, data-generation seed, and benchmark version

There is a single observation policy: methods receive motion-capture
position/attitude, and any further smoothing or state estimation is part of
the method under test.

Large contributed datasets are indexed in git but stored out-of-band. During
early review a dataset may use a provisional Google Drive, SharePoint, Dropbox,
or similar source URL in `dataset_tools/<dataset_id>/dataset.json`; merged
manifests must record status, source URL, expected size/checksum when known,
license, contact, and date accessed. Maintainers should later mirror accepted
datasets to a durable archive such as Zenodo, Purdue, OSF, Dataverse, or a
project-owned object store and update the manifest from `provisional` to
`archived`.

Processed benchmark data should be compact enough to commit under `data/` when
practical. Raw ROS 2 bags and large exported CSV trees remain external; dataset
processors convert asynchronous topics onto a documented time grid and write the
canonical binary format described in `docs/DATASET_CONTRACT.md`.

## Completed Milestones

- Static GitHub Pages site with leaderboard, cost-error tradeoff, 3D playback, and the flight explorer (segmentation, browser free-runs, model inspector, data-splits view).
- Method plugin API with metadata schema, registry bridge, and CI smoke checks.
- Nonlinear 6DOF model with stall/residual aerodynamics and four synthetic maneuver families.
- Two real Sport Cub indoor mocap datasets (2026-04-17 maneuver windows, 2026-05-22 full flights) converted to the canonical format.
- Grey-box output-error fit, filter-error EKF, and torch ODE-in-the-loop UDE on the 6DOF tier.
- Closed-loop SAFE-mode model identified from the real flights.
- Legacy 3DOF longitudinal tier retired after porting its methods to 6DOF (full implementation remains in git history).
- Single mocap observation policy; the former direct/mocap dual-run was removed.

## Current Roadmap

1. **Frequency-domain maturity.** Replace the Frequency-Welch/Stitching placeholder rows with a CIFER-style multi-window coherence-weighted identification.
2. **Gap-aware canonical format.** Represent mocap dropouts as gaps in timestamps rather than interpolated samples, with segmenters breaking at timing discontinuities.
3. **Method placeholders.** Promote Variational-Mocap, PINN-Closure, and NN-Surrogate from placeholder implementations to genuine baselines.
4. **Durable dataset archiving.** Mirror the Sport Cub datasets from provisional cloud links to a durable archive and flip manifests to `archived`.
5. **Additional model families.** A possible F-16 family based on Stevens--Lewis table-lookup aerodynamics would test high-performance aircraft identification without replacing the small-RC, mocap-motivated benchmark.

## Security Policy

Untrusted pull requests should run formatting, API validation, and small CPU smoke tests only. Full benchmarks, GPU runs, and self-hosted runner execution should require maintainer approval or run only after merge to a trusted branch.

## Contribution Process

Method contributions use a two-phase process:

1. A contributor opens a method PR with plugin code and documentation only. CI validates metadata, imports, and smoke checks on GitHub-hosted CPU runners.
2. After review and merge, a maintainer runs the full benchmark on trusted local GPU hardware and commits regenerated result artifacts separately.

This keeps untrusted code away from self-hosted GPU machines and separates method review from benchmark-result review.
