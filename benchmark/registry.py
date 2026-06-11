"""Registry bridge for built-in and plugin benchmark methods."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from .method_api import MethodMetadata, load_method_metadata


@dataclass(frozen=True)
class MethodSpec:
    name: str
    model_family: str
    training_scenario: str
    observation_types: tuple[str, ...]
    requires_gpu: bool = False
    heavy: bool = False
    entry_point: str = "comparison_suite:builtin"
    description: str = "Built-in method currently dispatched by comparison_suite.py."


BUILTIN_3DOF_METHOD_SPECS: tuple[MethodSpec, ...] = (
)

BUILTIN_METHOD_TRAINING = {method.name: method.training_scenario for method in BUILTIN_3DOF_METHOD_SPECS}

GPU_BUILTINS = {method.name for method in BUILTIN_3DOF_METHOD_SPECS if method.requires_gpu}
HEAVY_BUILTINS = {method.name for method in BUILTIN_3DOF_METHOD_SPECS if method.heavy}

MOCAP_ONLY_BUILTINS = {method.name for method in BUILTIN_3DOF_METHOD_SPECS if method.observation_types == ("mocap",)}

SIX_DOF_BUILTINS = {
    "6DOF-NominalGreyBox": ("mocap",),
    "6DOF-LinearSS": ("mocap",),
    "6DOF-Model-Stitching": ("mocap",),
    "6DOF-Subspace-Hankel": ("mocap",),
    "6DOF-Frequency-Welch": ("mocap",),
    "6DOF-Frequency-Stitching": ("mocap",),
    "6DOF-Koopman-EDMD": ("mocap",),
    "6DOF-EquationError-LS": ("mocap",),
    "6DOF-EKF-ParamID": ("mocap",),
    "6DOF-Fisher-UQ": ("mocap",),
    "6DOF-RidgeResidual": ("mocap",),
    "6DOF-Variational-Mocap": ("mocap",),
    "6DOF-SINDy": ("mocap",),
    "6DOF-Symbolic-Stepwise": ("mocap",),
    "6DOF-GP-RBF": ("mocap",),
    "6DOF-UDE-Residual": ("mocap",),
    "6DOF-PINN-Closure": ("mocap",),
    "6DOF-NN-Surrogate": ("mocap",),
    "6DOF-GreyBoxOEM-EEMInit": ("mocap",),
}


def builtin_method_metadata() -> list[MethodMetadata]:
    """Return plugin-style metadata for methods still implemented in comparison_suite.py."""

    methods = []
    for method in BUILTIN_3DOF_METHOD_SPECS:
        methods.append(
            MethodMetadata(
                name=method.name,
                entry_point=method.entry_point,
                model_families=(method.model_family,),
                observation_types=method.observation_types,
                training_scenarios=(method.training_scenario,),
                requires_gpu=method.requires_gpu,
                heavy=method.heavy,
                description=method.description,
            )
        )
    for name, observation_types in SIX_DOF_BUILTINS.items():
        requires_gpu = any(token in name for token in ("UDE", "PINN", "NN"))
        heavy = requires_gpu or any(token in name for token in ("GP", "Koopman", "Symbolic"))
        methods.append(
            MethodMetadata(
                name=name,
                entry_point=(
                    "models.aircraft6dof.greybox:main"
                    if name == "6DOF-GreyBoxOEM-EEMInit"
                    else "models.aircraft6dof.comparison_suite:builtin"
                ),
                model_families=("aircraft6dof",),
                observation_types=observation_types,
                training_scenarios=(
                    ("sportcub_mocap_4_17_26",)
                    if name == "6DOF-GreyBoxOEM-EEMInit"
                    else ("aircraft_6dof_open_loop", "aircraft_6dof_sine_sweep", "aircraft_6dof_aggressive", "aircraft_6dof_trim_grid")
                ),
                requires_gpu=requires_gpu,
                heavy=heavy,
                description=(
                    "6DOF grey-box OEM method using the framework-owned Sport Cub model specification."
                    if name == "6DOF-GreyBoxOEM-EEMInit"
                    else "Built-in 6DOF baseline dispatched by models/aircraft6dof/comparison_suite.py."
                ),
            )
        )
    return methods


def worker_method_names() -> tuple[str, ...]:
    """Return the canonical 3DOF built-in worker order."""

    return tuple(method.name for method in BUILTIN_3DOF_METHOD_SPECS)


def heavy_method_names() -> set[str]:
    return set(HEAVY_BUILTINS)


def gpu_method_names() -> set[str]:
    return set(GPU_BUILTINS)


def method_training_modes() -> dict[str, str]:
    return dict(BUILTIN_METHOD_TRAINING)


def discover_plugin_metadata(plugin_root: Path) -> list[MethodMetadata]:
    """Load metadata for contributor plugins with method.json files."""

    if not plugin_root.exists():
        return []
    methods = []
    for path in sorted(plugin_root.glob("*/method.json")):
        methods.append(load_method_metadata(path.parent))
    return methods


def all_method_metadata(plugin_root: Path) -> list[MethodMetadata]:
    """Return built-in method metadata plus contributor plugin metadata."""

    by_name = {method.name: method for method in builtin_method_metadata()}
    for method in discover_plugin_metadata(plugin_root):
        by_name[method.name] = method
    return [by_name[name] for name in sorted(by_name)]


def metadata_to_dict(method: MethodMetadata) -> dict[str, object]:
    return {
        "name": method.name,
        "entry_point": method.entry_point,
        "model_families": list(method.model_families),
        "observation_types": list(method.observation_types),
        "training_scenarios": list(method.training_scenarios),
        "requires_gpu": method.requires_gpu,
        "heavy": method.heavy,
        "description": method.description,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-root", type=Path, default=Path(__file__).resolve().parents[1] / "methods" / "plugins")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = [metadata_to_dict(method) for method in all_method_metadata(args.plugin_root)]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
