"""Minimal example method plugin."""

from __future__ import annotations

import numpy as np

from benchmark.method_api import FittedMethod, MethodMetadata


class ExampleLinearMethod:
    metadata = MethodMetadata(
        name="ExampleLinear",
        entry_point="method:ExampleLinearMethod",
        model_families=("aircraft3dof",),
        observation_types=("mocap",),
        training_scenarios=("trim_grid",),
        requires_gpu=False,
        description="Minimal contributor example for the benchmark plugin API.",
    )

    def fit(self, train_data, config):
        y = np.asarray(train_data.y_obs)
        return FittedMethod(
            method_name=self.metadata.name,
            model={"initial_observation": y[0].copy()},
            metadata={"example": True, "config": dict(config)},
        )

    def rollout(self, fitted, validation_data, config):
        y0 = np.asarray(fitted.model["initial_observation"])
        horizon = len(validation_data.t)
        return np.repeat(y0[None, :], horizon, axis=0)
