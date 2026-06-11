#!/usr/bin/env python3
"""Generate benchmark results and LaTeX-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from benchmark.export import export_web_data
from benchmark.paths import (
    DATASET_TOOLS,
    LATEX,
    LATEX_FIG,
    LATEX_GENERATED,
    LATEX_TABLES,
    METHOD_CODE,
    RESULTS as METHOD_RESULTS,
    ROOT,
    SPORTCUB_DATASET_ID,
    WORK,
    WORK_DATA,
)
from benchmark.scenarios import (
    SIX_DOF_DATASET_MODES,
    SIX_DOF_DATASET_OUTPUTS,
    SIX_DOF_DATASET_TITLES,
)


METHOD_FIG = LATEX_FIG
METHOD_TABLES = LATEX_TABLES

SIX_DOF_FIGURE_EXPORTS = {
    "aircraft6dof_validation_score_comparison.svg": "generated_aircraft6dof_validation_score_comparison.svg",
    "aircraft6dof_validation_trajectory_overlay.svg": "generated_aircraft6dof_validation_trajectory_overlay.svg",
    "aircraft6dof_train_time_accuracy_tradeoff.svg": "generated_aircraft6dof_train_time_accuracy_tradeoff.svg",
    "aircraft6dof_method_score_heatmap.svg": "generated_aircraft6dof_method_score_heatmap.svg",
}
LATEX_GENERATED_PATTERNS = ("*.tex",)
LATEX_GENERATED_FIGURE_PATTERNS = ("generated_*",)
TRADEOFF_FAILURE_THRESHOLD = 1.0


def methods_python() -> str:
    venv_python = ROOT / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.exists() else Path(sys.executable))


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_with_env(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def relative_or_absolute(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def worker_env(threads_per_worker: int = 1) -> dict[str, str]:
    env = os.environ.copy()
    value = str(max(1, int(threads_per_worker)))
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[key] = value
    return env


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def ffloat(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    if number == 0.0:
        return "0"
    if abs(number) < 1e-2 or abs(number) >= 1e3:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}g}"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing required results file: {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def remove_matching(directory: Path, patterns: tuple[str, ...]) -> int:
    if not directory.exists():
        return 0
    removed = 0
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
    return removed


def clean_latex_generated_assets() -> None:
    removed_tables = remove_matching(LATEX_GENERATED, LATEX_GENERATED_PATTERNS)
    removed_figures = remove_matching(LATEX_FIG, LATEX_GENERATED_FIGURE_PATTERNS)
    if removed_tables or removed_figures:
        print(f"Removed {removed_tables} generated LaTeX tables and {removed_figures} generated LaTeX figure files")


def copy_figure(source: Path, target: Path) -> None:
    # The paper includes generated figures via \includegraphics (no svg
    # package), so the rasterized sibling must travel with every .svg copy.
    shutil.copy2(source, target)
    png_source = source.with_suffix(".png")
    if source.suffix == ".svg" and png_source.exists():
        shutil.copy2(png_source, target.with_suffix(".png"))


def copy_figures() -> None:
    for source_name, target_name in SIX_DOF_FIGURE_EXPORTS.items():
        source = METHOD_FIG / source_name
        if not source.exists():
            continue
        copy_figure(source, LATEX_FIG / target_name)

def latex_assets(_args: argparse.Namespace) -> None:
    LATEX_GENERATED.mkdir(parents=True, exist_ok=True)
    LATEX_FIG.mkdir(parents=True, exist_ok=True)
    clean_latex_generated_assets()
    six_dof_table = METHOD_TABLES / "aircraft6dof_method_comparison.tex"
    if six_dof_table.exists():
        shutil.copy2(six_dof_table, LATEX_GENERATED / "aircraft6dof_method_comparison_table.tex")
    copy_figures()
    print(f"Wrote LaTeX tables to {LATEX_GENERATED}")
    print(f"Copied generated figures to {LATEX_FIG}")


# Unstabilized high-excitation maneuvers use short flight-test records: with
# the realistic (lightly damped phugoid) dynamics, long open-loop aggressive
# or sweep trials leave the flight envelope, exactly like a real aircraft
# flown hands-off. Stabilized (*_safe / autopilot) modes keep full duration.
SHORT_RECORD_DURATIONS = {"aggressive": 10.0, "sine_sweep": 10.0}


def simulate_6dof(args: argparse.Namespace) -> None:
    modes = list(getattr(args, "dataset_modes", None) or [args.dataset_mode])
    for mode in modes:
        output = args.output if len(modes) == 1 and getattr(args, "output", None) is not None else SIX_DOF_DATASET_OUTPUTS[mode]
        command = [
            sys.executable,
            "-m",
            "models.aircraft6dof.generate_dataset",
            "--output",
            str(output),
            "--train-trials",
            str(args.train_trials),
            "--validation-trials",
            str(args.validation_trials),
            "--duration",
            str(args.duration),
            "--dt",
            str(args.dt),
            "--seed",
            str(args.seed),
            "--dataset-mode",
            mode,
        ]
        if args.no_plot:
            command.append("--no-plot")
        run(command, cwd=ROOT)


def suite_6dof(args: argparse.Namespace) -> None:
    dataset_modes = list(getattr(args, "dataset_modes", None) or [])
    command = [
        sys.executable,
        "-m",
        "models.aircraft6dof.comparison_suite",
        "--ridge",
        str(args.ridge),
        "--workers",
        str(args.workers),
        "--results-dir",
        str(METHOD_RESULTS),
        "--fig-dir",
        str(METHOD_FIG),
        "--table-dir",
        str(METHOD_TABLES),
    ]
    if dataset_modes:
        command.append("--datasets")
        command.extend(str(SIX_DOF_DATASET_OUTPUTS[mode]) for mode in dataset_modes)
    else:
        dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
        command.extend(["--dataset", str(dataset)])
    if args.no_plot:
        command.append("--no-plot")
    run(command, cwd=ROOT)


def all_6dof(args: argparse.Namespace) -> None:
    simulate_6dof(args)
    dataset_modes = list(getattr(args, "dataset_modes", None) or [])
    dataset = args.output if getattr(args, "output", None) is not None else SIX_DOF_DATASET_OUTPUTS[args.dataset_mode]
    suite_6dof(
        argparse.Namespace(
            dataset=dataset,
            dataset_modes=dataset_modes,
            ridge=args.ridge,
            workers=args.workers,
            no_plot=args.no_plot,
        )
    )
    latex_assets(args)
    web_data(argparse.Namespace(output=ROOT / "site" / "public" / "data"))
    if args.build:
        build_pdf(args)


def fetch_dataset(args: argparse.Namespace) -> None:
    command = [sys.executable, "-m", "dataset_tools.fetch", args.dataset_id]
    if args.output_dir is not None:
        command.extend(["--output-dir", str(args.output_dir)])
    if args.url is not None:
        command.extend(["--url", args.url])
    run(command)


def process_dataset(args: argparse.Namespace) -> None:
    if args.dataset_id != SPORTCUB_DATASET_ID:
        raise SystemExit(f"unknown dataset processor: {args.dataset_id}")
    command = [
        sys.executable,
        "-m",
        "dataset_tools.sportcub_mocap_4_17_26.process",
        "--data-root",
        str(args.data_root),
        "--steps",
        args.steps,
    ]
    if args.only_cases:
        command.extend(["--only-cases", args.only_cases])
    if args.no_plots:
        command.append("--no-plots")
    run(command)


def canonicalize_dataset(args: argparse.Namespace) -> None:
    if args.dataset_id != SPORTCUB_DATASET_ID:
        raise SystemExit(f"unknown dataset canonicalizer: {args.dataset_id}")
    command = [
        sys.executable,
        "-m",
        "dataset_tools.sportcub_mocap_4_17_26.canonicalize",
        "--data-root",
        str(args.data_root),
        "--output",
        str(args.output),
    ]
    run(command)


def check_data(args: argparse.Namespace) -> None:
    command = [sys.executable, "-m", "dataset_tools.validate_format"]
    if args.dataset:
        command.extend(args.dataset)
    if args.allow_empty:
        command.append("--allow-empty")
    run(command)


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def archive_split_source_outputs(mode: str, source_outputs: dict[str, list[tuple[Path, Path, Path]]]) -> None:
    if not source_outputs:
        return
    for filename in ["shared_method_comparison.csv", "shared_uq_diagnostics.csv"]:
        rows: list[dict[str, str]] = []
        table_rows: list[dict[str, str]] = []
        for source in ("direct", "mocap"):
            path_list = source_outputs.get(source, [])
            if not path_list:
                continue
            for results_dir, table_dir, _fig_dir in path_list:
                result_file = results_dir / filename
                table_file = table_dir / filename
                if result_file.exists():
                    rows.extend(read_csv(result_file))
                if table_file.exists():
                    table_rows.extend(read_csv(table_file))
        write_csv_rows(archived_path(mode, filename, METHOD_RESULTS), rows)
        write_csv_rows(archived_path(mode, filename, METHOD_TABLES), table_rows or rows)

    trace_rows: list[dict[str, object]] = []
    for source in ("direct", "mocap"):
        for results_dir, _table_dir, _fig_dir in source_outputs.get(source, []):
            trace_file = results_dir / "shared_method_traces.json"
            if not trace_file.exists():
                continue
            payload = json.loads(trace_file.read_text())
            rows = payload.get("traces", payload if isinstance(payload, list) else [])
            trace_rows.extend(row for row in rows if isinstance(row, dict))
    if trace_rows:
        trace_output = archived_path(mode, "shared_method_traces.json", METHOD_RESULTS)
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        trace_output.write_text(json.dumps({"traces": trace_rows}, indent=2, sort_keys=True) + "\n")

    for filename in ["shared_frequency_summary.csv", "shared_sindy_coefficients.csv", "shared_symbolic_coefficients.csv"]:
        for source in ("mocap", "direct"):
            for paths in source_outputs.get(source, []):
                if (paths[0] / filename).exists():
                    shutil.copy2(paths[0] / filename, archived_path(mode, filename, METHOD_RESULTS))
                    break
            else:
                continue
            break

    for filename in ARCHIVED_FIGURES:
        for source in ("mocap", "direct"):
            for paths in source_outputs.get(source, []):
                fig_source = paths[2] / filename
                if fig_source.exists():
                    shutil.copy2(fig_source, METHOD_FIG / f"{mode}_{filename}")
                    png = fig_source.with_suffix(".png")
                    if png.exists():
                        shutil.copy2(png, METHOD_FIG / f"{mode}_{Path(filename).with_suffix('.png').name}")
                    break
            else:
                continue
            break
    print(f"archived {mode} from split source workers", flush=True)


def restore_shared_outputs(mode: str) -> None:
    if mode not in available_archived_modes():
        return
    for filename in ARCHIVED_RESULTS:
        archived = archived_path(mode, filename, METHOD_RESULTS)
        if archived.exists():
            shutil.copy2(archived, METHOD_RESULTS / filename)
        table_archived = archived_path(mode, filename, METHOD_TABLES)
        if table_archived.exists():
            shutil.copy2(table_archived, METHOD_TABLES / filename)
    for filename in ARCHIVED_FIGURES:
        archived = METHOD_FIG / f"{mode}_{filename}"
        if archived.exists():
            shutil.copy2(archived, METHOD_FIG / filename)
            png = archived.with_suffix(".png")
            if png.exists():
                shutil.copy2(png, METHOD_FIG / Path(filename).with_suffix(".png").name)


def build_pdf(_args: argparse.Namespace) -> None:
    run([sys.executable, str(LATEX / "paper.py"), "build"], cwd=LATEX)


def web_data(args: argparse.Namespace) -> None:
    manifest = export_web_data(
        root=ROOT,
        output_dir=args.output,
        results_dir=METHOD_RESULTS,
    )
    print(f"Wrote web benchmark data to {args.output}")
    print(f"Exported {len(manifest['scenarios'])} scenarios at schema {manifest['schema_version']}")


def _plugin_dirs() -> list[Path]:
    plugin_root = METHOD_CODE / "plugins"
    if not plugin_root.exists():
        return []
    return sorted(path.parent for path in plugin_root.glob("*/method.json"))


def check_setup(_args: argparse.Namespace) -> None:
    """Run fast local checks for the website/plugin benchmark setup."""

    from benchmark.registry import all_method_metadata

    py_files = [
        "benchmark/export.py",
        "benchmark/method_api.py",
        "benchmark/registry.py",
        "benchmark/smoke_plugin.py",
        "dataset_tools/fetch.py",
        "dataset_tools/registry.py",
        "dataset_tools/validate_dataset.py",
        "dataset_tools/validate_format.py",
        "dataset_tools/sportcub_mocap_4_17_26/canonicalize.py",
        "dataset_tools/sportcub_mocap_4_17_26/process.py",
        "models/aircraft6dof/model.py",
        "models/aircraft6dof/comparison_suite.py",
        "models/aircraft6dof/smoke.py",
        "results.py",
    ]
    run([sys.executable, "-m", "py_compile", *py_files])
    run([sys.executable, "-m", "dataset_tools.validate_dataset", str(DATASET_TOOLS / SPORTCUB_DATASET_ID)])
    run([sys.executable, "-m", "dataset_tools.validate_format", "--allow-empty"])
    registered_methods = all_method_metadata(METHOD_CODE / "plugins")
    if not registered_methods:
        raise SystemExit("method registry is empty")
    print(f"Registered {len(registered_methods)} methods")
    for plugin_dir in _plugin_dirs():
        run([sys.executable, "-m", "benchmark.smoke_plugin", str(plugin_dir)])
    web_data(argparse.Namespace(output=ROOT / "site" / "public" / "data"))
    manifest_path = ROOT / "site" / "public" / "data" / "manifest.json"
    method_results_path = ROOT / "site" / "public" / "data" / "method_results.json"
    manifest = json.loads(manifest_path.read_text())
    method_results = json.loads(method_results_path.read_text())
    if not manifest.get("scenarios"):
        raise SystemExit("site manifest has no scenarios")
    if not method_results:
        raise SystemExit("site method_results.json has no method rows")
    run([sys.executable, "-m", "models.aircraft6dof.smoke"], cwd=ROOT)
    print("Setup check passed.")
    print(f"Site data: {len(manifest['scenarios'])} scenarios, {len(method_results)} method result rows")


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _choose_port(start: int) -> int:
    port = int(start)
    for candidate in range(port, port + 100):
        if _port_available(candidate):
            return candidate
    raise SystemExit(f"could not find an available port from {port} to {port + 99}")


def serve_site(args: argparse.Namespace) -> None:
    """Serve the static benchmark site locally."""

    web_data(argparse.Namespace(output=ROOT / "site" / "public" / "data"))
    port = _choose_port(args.port)
    print(f"Serving benchmark site at http://127.0.0.1:{port}")
    print("Press Ctrl-C to stop.")
    run([sys.executable, str(ROOT / "site" / "serve.py"), "--port", str(port), "--bind", "127.0.0.1", "--directory", str(ROOT / "site")])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim6 = sub.add_parser("simulate-6dof", help="generate the 6DOF train/validation dataset")
    p_sim6.add_argument("--output", type=Path, default=None, help="single-mode output directory; ignored when multiple --dataset-modes are selected")
    p_sim6.add_argument("--train-trials", type=int, default=32)
    p_sim6.add_argument("--validation-trials", type=int, default=8)
    p_sim6.add_argument("--duration", type=float, default=12.0)
    p_sim6.add_argument("--dt", type=float, default=0.02)
    p_sim6.add_argument("--seed", type=int, default=17)
    p_sim6.add_argument("--dataset-mode", choices=list(SIX_DOF_DATASET_MODES), default="aggressive", help="single 6DOF mode used when --dataset-modes is omitted")
    p_sim6.add_argument("--dataset-modes", nargs="+", choices=list(SIX_DOF_DATASET_MODES), default=None, help="generate several 6DOF modes into their standard data directories")
    p_sim6.add_argument("--no-plot", action="store_true")
    p_sim6.set_defaults(func=simulate_6dof)

    p_suite6 = sub.add_parser("suite-6dof", help="run baseline methods on the 6DOF train/validation dataset")
    p_suite6.add_argument("--dataset", type=Path, default=SIX_DOF_DATASET_OUTPUTS["aggressive"])
    p_suite6.add_argument("--dataset-modes", nargs="+", choices=list(SIX_DOF_DATASET_MODES), default=None, help="standard generated 6DOF datasets to aggregate; omit to use --dataset")
    p_suite6.add_argument("--ridge", type=float, default=1e-5)
    p_suite6.add_argument("--workers", type=int, default=max(1, min(30, (os.cpu_count() or 2) - 2)))
    p_suite6.add_argument("--no-plot", action="store_true")
    p_suite6.set_defaults(func=suite_6dof)

    p_all6 = sub.add_parser("all-6dof", help="generate 6DOF data, run baseline methods, export LaTeX/site assets, and optionally build")
    p_all6.add_argument("--output", type=Path, default=None)
    p_all6.add_argument("--train-trials", type=int, default=256)
    p_all6.add_argument("--validation-trials", type=int, default=64)
    p_all6.add_argument("--duration", type=float, default=20.0)
    p_all6.add_argument("--dt", type=float, default=0.02)
    p_all6.add_argument("--seed", type=int, default=17)
    p_all6.add_argument("--dataset-mode", choices=list(SIX_DOF_DATASET_MODES), default="aggressive")
    p_all6.add_argument("--dataset-modes", nargs="+", choices=list(SIX_DOF_DATASET_MODES), default=list(SIX_DOF_DATASET_MODES))
    p_all6.add_argument("--ridge", type=float, default=1e-5)
    p_all6.add_argument("--workers", type=int, default=max(1, min(30, (os.cpu_count() or 2) - 2)))
    p_all6.add_argument("--no-plot", action="store_true")
    p_all6.add_argument("--build", action="store_true")
    p_all6.set_defaults(func=all_6dof)

    p_fetch_dataset = sub.add_parser("fetch-dataset", help="download a contributed dataset payload into work/data")
    p_fetch_dataset.add_argument("dataset_id")
    p_fetch_dataset.add_argument("--output-dir", type=Path, default=None)
    p_fetch_dataset.add_argument("--url", default=None, help="override the manifest URL")
    p_fetch_dataset.set_defaults(func=fetch_dataset)

    p_process_dataset = sub.add_parser("process-dataset", help="run a contributed dataset's raw-data processing pipeline")
    p_process_dataset.add_argument("dataset_id")
    p_process_dataset.add_argument("--data-root", type=Path, default=WORK_DATA / SPORTCUB_DATASET_ID / "raw")
    p_process_dataset.add_argument("--steps", default="1,2,3")
    p_process_dataset.add_argument("--only-cases", default=None)
    p_process_dataset.add_argument("--no-plots", action="store_true")
    p_process_dataset.set_defaults(func=process_dataset)

    p_canonicalize_dataset = sub.add_parser("canonicalize-dataset", help="convert processed dataset segments to flat compact data/<dataset_id>_<split>.npz arrays")
    p_canonicalize_dataset.add_argument("dataset_id")
    p_canonicalize_dataset.add_argument("--data-root", type=Path, default=WORK_DATA / SPORTCUB_DATASET_ID / "raw")
    p_canonicalize_dataset.add_argument("--output", type=Path, default=ROOT / "data")
    p_canonicalize_dataset.set_defaults(func=canonicalize_dataset)

    p_check_data = sub.add_parser("check-data", help="validate committed compact datasets under data/")
    p_check_data.add_argument("dataset", nargs="*")
    p_check_data.add_argument("--allow-empty", action="store_true")
    p_check_data.set_defaults(func=check_data)

    p_assets = sub.add_parser("latex-assets", help="export current method results into latex/generated and latex/fig")
    p_assets.set_defaults(func=latex_assets)

    p_web = sub.add_parser("web-data", help="export current method results into site/public/data JSON")
    p_web.add_argument("--output", type=Path, default=ROOT / "site" / "public" / "data")
    p_web.set_defaults(func=web_data)

    p_check = sub.add_parser("check-setup", help="run fast checks for the plugin, website, and model setup")
    p_check.set_defaults(func=check_setup)

    p_serve = sub.add_parser("serve-site", help="serve the static benchmark website locally")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=serve_site)

    p_build = sub.add_parser("build", help="build latex/main.pdf")
    p_build.set_defaults(func=build_pdf)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
