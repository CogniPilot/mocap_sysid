"""Shared benchmark export schema definitions."""

from __future__ import annotations

SCHEMA_VERSION = "0.1.0"
MODEL_FAMILY_6DOF = "aircraft6dof"

METHOD_RESULT_FIELDS = (
    "scenario",
    "scenario_title",
    "model_family",
    "method",
    "description",
    "implementation_status",
    "backend",
    "input_channel",
    "evaluation_mode",
    "training_scenario",
    "validation_scenario",
    "validation_score",
    "train_elapsed_s",
    "train_cpu_s",
    "train_gpu_s",
    "gpu_memory_mb",
    "rollout_elapsed_s",
    "total_elapsed_s",
    "train_loss_final",
    "decision_variables",
    "train_samples",
    "rmse_V",
    "rmse_alpha",
    "rmse_gamma",
    "rmse_Q",
    "rmse_position_m",
    "rmse_velocity_mps",
    "rmse_quaternion",
    "rmse_rates_rad_s",
    "rmse_mocap_position_m",
    "rmse_mocap_quaternion",
    "mocap_rmse_x_pos",
    "mocap_rmse_z_pos",
    "mocap_rmse_theta",
    "coeff_residual_rmse_C_L",
    "coeff_residual_rmse_C_D",
    "coeff_residual_rmse_C_M",
    "notes",
)
