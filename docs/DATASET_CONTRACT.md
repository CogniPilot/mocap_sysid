# Dataset Contract

Benchmark contributors should not submit raw ROS 2 bags, Foxglove exports, or
wide intermediate CSV trees as the benchmark dataset. Those files are useful for
provenance, but they are too large and too inconsistent for routine CI and
website use.

The committed dataset artifact should be a compact binary time-series package:

```text
data/<dataset_id>_train.npz
data/<dataset_id>_validation.npz
```

Use `np.savez_compressed` for the first implementation. It is dependency-light,
already used by the synthetic generators, and compact enough for many processed
flight datasets. If we later need chunked access or columnar analytics, we can
add an optional Zarr or Parquet export without changing the required NPZ
contract.

Datasets must be split into flat `data/<dataset_id>_train.npz` and
`data/<dataset_id>_validation.npz` files. The training split is the only data
methods should use for fitting. Validation is held out for scoring and website
comparisons. Each NPZ must include matching scalar `dataset_id` and `split_name`
fields, where `split_name` is `train` or `validation`.

## ROS 2 / Asynchronous Logs

Raw flight logs often contain asynchronous topics: motion-capture pose, onboard
pose estimates, controller outputs, pilot commands, IMU, airdata, and
mode/status messages may arrive at different timestamps and rates. The dataset
processor must convert the best available pose and control records onto a
declared sample grid before writing the compact artifact.

Required policy:

- Resample each segment onto a fixed-period sample grid and record the sample
  period in each split NPZ.
- Use zero-order hold for command/status channels.
- Use interpolation only for continuously sampled pose/sensor channels.
- Preserve raw message timestamps in the raw archive or processing provenance.
- Record interpolation method, sample rate, filters, and dropped-sample rules.
- Add `valid_mask` so padded/ragged segments and rejected samples are explicit.
- Never silently bridge large time gaps; split segments or mark samples invalid.

Current compact format: `sysid.timeseries.ragged.v1`.

Each split NPZ must include:

- `time_s`: `[segment, sample]`, seconds from segment start.
- `valid_mask`: `[segment, sample]`, boolean.
- `control_meas`: `[segment, sample, control_channel]`, commanded controls.
- `pose_meas`: `[segment, sample, pose_channel]`, best available pose.
- `segment_names`, `control_names`, `pose_names`, `system_dof`,
  `format_version`.
- `dataset_id`, `split_name`, `sample_period_s`, `truth_available`.

For `system_dof = 6`, `pose_names` is fixed as
`[x_e, y_n, z_u, q_w, q_x, q_y, q_z]`. Position uses east/north/up in meters.
Quaternions are scalar-first, normalized, and represent body attitude in the ENU
inertial frame.

For `system_dof = 3`, `pose_names` is fixed as `[x_e, z_u, theta]`, where
`theta` is pitch attitude in radians. This is the compact pose-only layout for
longitudinal datasets.

`control_names` is fixed as `[thrust, aileron, elevator, rudder]`. Use physical
units when available and document units/scaling in the dataset contract or
processing code. Normalized RC command channels are acceptable for early
datasets, but the meaning must be documented. If a dataset has no independent
thrust command, fill thrust from the best available throttle/thrust command and
describe the scaling.

The pose channel should be the best available ground-truth-like pose for the
dataset. In a motion-capture dataset, this is usually cleaned mocap. In a field
flight dataset, it may be a surveyed GNSS/INS reference solution if that is the
best available reference. The source and cleanup steps belong in documented
processing code, not as raw data committed beside the NPZ files.

An optional direct-state channel records the converter's canonical state
estimate:

- 6DOF `direct_state_meas` uses
  `direct_state_names = [x_n, y_e, z_d, u, v, w, q_w, q_x, q_y, q_z, p, q, r]`.

Direct state is separate from pose because the benchmark scores methods from
the pose channel only; `direct_state_meas` supports tooling such as the flight
explorer and dataset validation, not a separate scored observation condition.

## Time Base

Canonical splits are fixed-rate within each segment. `time_s[i, valid_mask[i]]`
must be monotonically increasing and uniformly spaced to within timestamp
tolerance. Use one `sample_period_s` for the dataset when possible. If a dataset
really needs different rates for different segments, store `sample_period_s` as
one value per segment and keep each segment uniform.

Do not store asynchronous raw topic timestamps in the compact benchmark arrays.
The canonicalization script should perform the alignment:

- pick the pose timeline or an explicitly configured sample period,
- interpolate pose to the grid,
- zero-order hold controls to the grid,
- split or invalidate samples across large gaps,
- record the interpolation and gap thresholds in documented processing code.

Optional measured sensor groups use the same leading shape. They are not part of
the required benchmark record and may also be used only inside the
canonicalization script to clean/interpolate the final pose before storage:

- `accel_meas` with `accel_names = [a_x_body, a_y_body, a_z_body]`.
- `gyro_meas` with `gyro_names = [p, q, r]`.
- `mag_meas` with `mag_names = [m_x_body, m_y_body, m_z_body]`.

These are body-frame channels. Accelerometer values are specific force in
`m/s^2`, gyro values are body rates in `rad/s`, and magnetometer values are a
unitless normalized magnetic-field vector unless metadata says otherwise.

Optional onboard-estimator groups are separate from direct measurements:

- `onboard_pose_est` with `onboard_pose_names`, usually
  `[x_e, y_n, z_u, q_w, q_x, q_y, q_z]`.

Do not add a vague catch-all onboard state array. If a future dataset has
onboard velocity, rates, or covariance, add explicitly named groups with
documented channel names.

Absent channels should be omitted rather than filled with dummy data. If a
channel is present but partially invalid, keep the array and mark invalid samples
through `valid_mask` or a future per-channel mask recorded in metadata.

For real datasets, truth may be unavailable. Use `truth_available = False` and
make methods score against held-out measured outputs rather than synthetic true
state.
