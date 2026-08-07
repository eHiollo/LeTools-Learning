# RL100 Point-Cloud and Inference Performance Design

## Problem

The RL100 checkpoint uses ten-step DDIM inference and emits four actions at
10 Hz. Isolated, warmed-up inference takes about 0.42 seconds. In the live
observation loop it takes 2.0–3.6 seconds because every control tick
back-projects and transforms roughly 916,000 depth pixels before reducing the
result to 1,024 points. On Jetson Orin this saturates CPU and shared memory,
increases RAM and swap pressure, and starves GPU inference.

## Scope

This change optimizes depth-to-point-cloud construction without changing the
RL100 observation contract, checkpoint, action scheduler, safety gates, camera
timestamp semantics, or final point-cloud shape. CM training and longer action
horizons are follow-up work because the current checkpoint declares
`use_cm: false` and `n_action_steps: 4`.

## Design

Add deterministic, bounded pixel candidate selection before depth
back-projection. Each camera contributes candidates distributed across its
full image. Invalid depths are removed, candidates are transformed into the
base frame, the three camera clouds are fused and workspace-cropped, and the
existing final resampling still produces exactly `(1024, 3)` float32 points.

The candidate budget is configurable in the shared collection/deployment
point-cloud configuration and defaults to 16,384 pixels per camera for the
RL100 upper-camera configuration. A non-positive value preserves the existing
full-resolution path for compatibility. Collection and deployment use the same
setting so their point-cloud semantics cannot drift.

The implementation must avoid allocating full-image `meshgrid` arrays when a
candidate budget is active. Candidate indices are deterministic for a given
image shape and budget, making tests and offline bag conversion reproducible.

## Instrumentation

Record point-cloud construction latency separately from policy inference
latency. Shadow/live audit records expose the latest point-cloud latency so a
slow observation path cannot be mistaken for a slow model.

## Safety and Data Quality

- Camera freshness, header skew, receive skew, and TF-at-image-stamp checks are
  unchanged.
- Workspace filtering remains after transformation into the base frame.
- `fail_on_empty_pointcloud` and `min_workspace_points` remain hard gates.
- The final cloud remains finite float32 with shape `(1024, 3)`.
- No robot commands are published during performance verification.

## Verification

Automated tests cover deterministic candidate selection, full-resolution
compatibility, geometric correctness, configuration loading, final shape, and
empty-workspace behavior.

On the running robot, compare identical 60-second shadow runs before and after
the change. Acceptance criteria are:

- point-cloud construction p95 below 50 ms;
- warmed policy inference p95 below 500 ms;
- no growth in swap during the measured shadow interval;
- camera header and receive skew remain within the current strict limits;
- no terminal observation or inference fault;
- final point cloud remains `(1024, 3)` with at least the configured minimum
  workspace points.

The four-action horizon covers only 0.4 seconds, so a result near 0.42 seconds
is still marginal. If this optimization restores isolated-model performance
but cannot meet the 500 ms target reliably, the next supported solution is a
trained one-step CM checkpoint or a checkpoint with a longer action horizon,
not silently forcing `use_cm=true` on the current checkpoint.
