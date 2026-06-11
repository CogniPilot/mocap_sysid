# Benchmark Data

This folder is for compact, reviewable datasets that are small enough to live in
git after raw logs have been converted into the canonical benchmark format.
Synthetic simulator outputs are generated into ignored `work/data/`
directories before benchmark runs and should not be committed here.

Raw exports from labs, flight logs, bag files, Foxglove CSV dumps, and large
intermediate products should not be committed here. Keep those under
`work/data/<dataset_id>/raw/` locally or in a provisional external source,
then convert them to:

```text
data/<dataset_id>_train.npz
data/<dataset_id>_validation.npz
```

CI should validate every committed dataset with:

```bash
./results.py check-data
```

The current compact array contract is `sysid.timeseries.ragged.v1`: segment
arrays are padded to the longest segment and `valid_mask` marks real samples.
Required arrays are `time_s`, `valid_mask`, `control_meas`, and `pose_meas`,
plus companion name arrays, `dataset_id`, `split_name`, and `system_dof`.

For `system_dof=6`, `pose_meas` is fixed to
`[x_e, y_n, z_u, q_w, q_x, q_y, q_z]`. For `system_dof=3`, `pose_meas` is fixed
to `[x_e, z_u, theta]`. Both should contain the best available
ground-truth-like pose after dataset-specific cleanup. `control_meas` is fixed
to `[thrust, aileron, elevator, rudder]`.

Datasets may include `direct_state_meas` carrying the converter's canonical
state estimate; the benchmark itself scores methods from the pose channel.

Each segment should be fixed-rate in `time_s`. Raw asynchronous logs are aligned
by the canonicalization script before these NPZ files are committed.

Optional groups such as `accel_meas`, `gyro_meas`, `mag_meas`, and
`onboard_pose_est` may be added with matching `*_names` arrays when a dataset
has those channels. `onboard_pose_est` is always
`[x_e, y_n, z_u, q_w, q_x, q_y, q_z]`.
