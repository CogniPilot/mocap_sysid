"""Canonical benchmark scenario definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import WORK_DATA
from .schema import MODEL_FAMILY_6DOF


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    title: str
    model_family: str
    default_path: Path
    generator: str | None = None
    tags: tuple[str, ...] = ()


SCENARIOS_6DOF: tuple[ScenarioSpec, ...] = (
    ScenarioSpec(
        "aircraft_6dof_open_loop",
        "6-DOF open-loop maneuver",
        MODEL_FAMILY_6DOF,
        WORK_DATA / "aircraft_6dof_open_loop",
    ),
    ScenarioSpec(
        "aircraft_6dof_sine_sweep",
        "6-DOF sine-sweep maneuver",
        MODEL_FAMILY_6DOF,
        WORK_DATA / "aircraft_6dof_sine_sweep",
    ),
    ScenarioSpec(
        "aircraft_6dof_aggressive",
        "6-DOF aggressive nonlinear stall maneuver",
        MODEL_FAMILY_6DOF,
        WORK_DATA / "aircraft_6dof_aggressive",
    ),
    ScenarioSpec(
        "aircraft_6dof_trim_grid",
        "6-DOF local trim-grid small-deviation maneuver",
        MODEL_FAMILY_6DOF,
        WORK_DATA / "aircraft_6dof_trim_grid",
    ),
)


def scenario_map(scenarios: tuple[ScenarioSpec, ...]) -> dict[str, ScenarioSpec]:
    return {scenario.id: scenario for scenario in scenarios}


SCENARIOS_6DOF_BY_ID = scenario_map(SCENARIOS_6DOF)

SIX_DOF_DATASET_MODES = tuple(scenario.id.removeprefix("aircraft_6dof_") for scenario in SCENARIOS_6DOF)
SIX_DOF_DATASET_OUTPUTS = {
    scenario.id.removeprefix("aircraft_6dof_"): scenario.default_path for scenario in SCENARIOS_6DOF
}
SIX_DOF_DATASET_TITLES = {
    scenario.id.removeprefix("aircraft_6dof_"): scenario.title for scenario in SCENARIOS_6DOF
}
SIX_DOF_SCENARIO_TITLES = {scenario.id: scenario.title for scenario in SCENARIOS_6DOF}
