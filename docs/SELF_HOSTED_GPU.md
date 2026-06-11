# Self-Hosted NVIDIA GPU Runner

Use a self-hosted runner for full benchmark execution on a local NVIDIA machine. The GitHub Pages workflow should remain lightweight and should only publish already-generated static data. Method pull requests should be validated separately with CPU-only smoke checks; full GPU benchmark results should be produced in a separate maintainer commit.

## Recommended Security Model

- Do not run untrusted pull-request code on your local GPU runner.
- Use `workflow_dispatch` or trusted-branch triggers only.
- Run the runner as a dedicated low-privilege OS user.
- Give the runner a specific label such as `gpu`; do not use it for generic CI jobs.
- Review method-contribution PRs with CPU smoke tests first, then run full GPU benchmarks after approval or after merge.
- Commit regenerated benchmark artifacts separately from the method-code PR.

## One-Time Machine Setup

Install NVIDIA drivers and verify CUDA visibility:

```bash
nvidia-smi
```

Install system packages:

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip
```

In GitHub, go to:

```text
Repository -> Settings -> Actions -> Runners -> New self-hosted runner
```

Follow GitHub's commands for Linux x64. When configuring labels, include:

```text
self-hosted, linux, x64, gpu
```

For a persistent service, run GitHub's generated service commands, usually:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

## Benchmark Workflow

The `.github/workflows/benchmark-self-hosted.yml` workflow runs only when manually dispatched. It:

- checks out the repository,
- verifies `nvidia-smi`,
- creates or refreshes `.venv`,
- installs the project with the `gpu` extra from `pyproject.toml`, using the CUDA PyTorch wheel index,
- optionally regenerates datasets,
- runs the 6-DOF comparison suite over the synthetic family and both Sport Cub datasets,
- regenerates LaTeX and website data assets,
- uploads benchmark artifacts.

The default workflow settings use:

```text
workers=30
```

That matches the current local workstation assumption of high CPU parallelism for validation rollouts; torch-based rows use the GPU when available.

## Local Dry Run

Before using GitHub Actions, run this directly:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[gpu]" --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
.venv/bin/python -m models.aircraft6dof.comparison_suite \
  --datasets work/data/aircraft_6dof_open_loop work/data/aircraft_6dof_sine_sweep \
    work/data/aircraft_6dof_aggressive work/data/aircraft_6dof_trim_grid \
    data/sportcub_mocap_4_17_26_train.npz data/sportcub_mocap_5_22_26_train.npz \
  --results-dir results --fig-dir latex/fig --table-dir latex/tables
./results.py latex-assets
./results.py web-data
```

If GPU use is not visible in `nvidia-smi`, check:

```bash
.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Two-Phase Results Update

1. Merge or check out the trusted method-code change.
2. Run the full benchmark locally or through the manual self-hosted workflow.
3. Review regenerated CSV files, figures, LaTeX assets, website JSON, and paper output.
4. Commit only the trusted generated results in a separate commit.

## Updating the Public Site

After a trusted self-hosted benchmark run updates `results` and `site/public/data`, commit those changes and push to `main`. The normal Pages workflow then publishes the static site.
