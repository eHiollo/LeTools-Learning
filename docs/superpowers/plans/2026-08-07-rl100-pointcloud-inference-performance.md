# RL100 Point-Cloud and Inference Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce RL100 real-time point-cloud CPU and memory pressure so warmed DDIM inference remains close to its isolated latency while preserving the `(1024, 3)` observation contract.

**Architecture:** Deterministically select a bounded set of depth pixels before back-projection, then retain the existing transform, fusion, workspace crop, and final 1,024-point resampling stages. Configure the candidate budget through the shared collection/deployment point-cloud YAML and expose point-cloud generation latency in audit records for on-robot A/B verification.

**Tech Stack:** Python 3.10, NumPy, ROS Noetic/rospy, pytest, RL100 DP3/DDIM, Jetson `tegrastats`.

## Global Constraints

- Final point clouds remain finite `float32` arrays with shape `(1024, 3)`.
- Camera freshness, header skew, receive skew, TF-at-image-stamp, workspace, and minimum-point gates remain unchanged.
- Candidate selection is deterministic for an image shape and budget.
- A candidate budget `<= 0` preserves the full-resolution compatibility path.
- Collection and deployment consume the same point-cloud YAML value.
- The current checkpoint remains DDIM with `use_cm: false`; this plan must not force CM inference.
- Robot verification uses `shadow` only and publishes no commands.

---

### Task 1: Bounded depth-pixel back-projection

**Files:**
- Modify: `kuavo_rl/rl100_zarr/pointcloud.py`
- Test: `kuavo_rl/tests/test_rl100_zarr.py`

**Interfaces:**
- Consumes: depth image `(H, W)`, `(fx, fy, cx, cy)`, optional `T_base_cam`.
- Produces: `depth_to_point_cloud(..., candidate_count: int = 0) -> np.ndarray` where a positive budget limits pixels before allocation/back-projection.

- [ ] **Step 1: Write failing geometry and determinism tests**

```python
def test_depth_to_point_cloud_candidate_budget_is_bounded_and_deterministic():
    depth = np.ones((100, 200), dtype=np.float32)
    first = depth_to_point_cloud(depth, (100, 100, 100, 50), candidate_count=4096)
    second = depth_to_point_cloud(depth, (100, 100, 100, 50), candidate_count=4096)
    assert first.shape == (4096, 3)
    np.testing.assert_array_equal(first, second)


def test_depth_to_point_cloud_nonpositive_budget_preserves_full_resolution():
    depth = np.ones((10, 20), dtype=np.float32)
    points = depth_to_point_cloud(depth, (10, 10, 10, 5), candidate_count=0)
    assert points.shape == (200, 3)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate letools-rl
pytest -q kuavo_rl/tests/test_rl100_zarr.py -k 'candidate_budget or nonpositive_budget'
```

Expected: FAIL because `depth_to_point_cloud()` does not accept `candidate_count`.

- [ ] **Step 3: Implement deterministic pre-back-projection selection**

Add `candidate_count: int = 0`. For a positive budget smaller than `H*W`, compute deterministic flat indices with `np.linspace(0, H*W - 1, candidate_count, dtype=np.int64)`, derive `u = indices % width` and `v = indices // width`, and read only `depth.reshape(-1)[indices]`. Keep the existing full-resolution path for non-positive or non-limiting budgets. Apply finite/depth masks after selection and preserve the existing transform math and float32 return type.

- [ ] **Step 4: Run focused and point-cloud tests**

Run:

```bash
pytest -q kuavo_rl/tests/test_rl100_zarr.py -k 'depth_to_point_cloud or fuse_three_cams or candidate_budget or nonpositive_budget'
```

Expected: all selected tests PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add kuavo_rl/rl100_zarr/pointcloud.py kuavo_rl/tests/test_rl100_zarr.py
git commit -m "perf: bound RL100 depth back-projection"
```

---

### Task 2: Shared candidate-budget configuration and hub integration

**Files:**
- Modify: `kuavo_rl/rl100_zarr/config.py`
- Modify: `kuavo_rl/rl100_zarr/ros_depth.py`
- Modify: `configs/rl/rl100_zarr_collect_upper_cams.yaml`
- Test: `kuavo_rl/tests/test_rl100_zarr.py`
- Test: `kuavo_rl/tests/test_rl100_camera_sync.py`

**Interfaces:**
- Consumes: YAML key `pointcloud_candidate_pixels_per_camera`.
- Produces: `RL100CollectConfig.pointcloud_candidate_pixels_per_camera: int`; `DepthPointCloudHub` passes it to every `depth_to_point_cloud()` call.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_upper_camera_config_bounds_pointcloud_candidates():
    cfg = load_rl100_collect_config("configs/rl/rl100_zarr_collect_upper_cams.yaml")
    assert cfg.pointcloud_candidate_pixels_per_camera == 16384


def test_candidate_budget_rejects_negative_values():
    cfg = RL100CollectConfig(pointcloud_candidate_pixels_per_camera=-1)
    with pytest.raises(ValueError, match="pointcloud_candidate_pixels_per_camera"):
        cfg.validate_contract()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q kuavo_rl/tests/test_rl100_zarr.py -k 'bounds_pointcloud_candidates or candidate_budget_rejects'
```

Expected: FAIL because the configuration field does not exist.

- [ ] **Step 3: Add and load the shared configuration field**

Add this dataclass field:

```python
pointcloud_candidate_pixels_per_camera: int = 0
```

Load it with:

```python
pointcloud_candidate_pixels_per_camera=int(
    raw.get("pointcloud_candidate_pixels_per_camera", 0)
)
```

Validate that it is non-negative. Add `pointcloud_candidate_pixels_per_camera: 16384` beside `num_points` in the upper-camera YAML.

- [ ] **Step 4: Wire the budget into fusion**

Change the hub call to:

```python
depth_to_point_cloud(
    frame.depth_m,
    frame.intrinsics,
    transform,
    candidate_count=self.config.pointcloud_candidate_pixels_per_camera,
)
```

- [ ] **Step 5: Run configuration, synchronization, and point-cloud tests**

Run:

```bash
pytest -q kuavo_rl/tests/test_rl100_zarr.py kuavo_rl/tests/test_rl100_camera_sync.py
```

Expected: all tests PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add kuavo_rl/rl100_zarr/config.py kuavo_rl/rl100_zarr/ros_depth.py configs/rl/rl100_zarr_collect_upper_cams.yaml kuavo_rl/tests/test_rl100_zarr.py kuavo_rl/tests/test_rl100_camera_sync.py
git commit -m "perf: configure bounded RL100 point clouds"
```

---

### Task 3: Point-cloud latency audit and robot verification

**Files:**
- Modify: `kuavo_rl/rl100_zarr/ros_depth.py`
- Modify: `kuavo_rl/rl100_real_runner.py`
- Modify: `kuavo_rl/tests/test_rl100_topic_native.py`
- Modify: `docs/RL100_REAL_DEPLOY_TROUBLESHOOTING.md`

**Interfaces:**
- Consumes: monotonic start/end times around `_fuse_bundle()`.
- Produces: `PointCloudSample.generation_s: float`; audit field `pointcloud_generation_s`.

- [ ] **Step 1: Write a failing audit test**

Extend the topic-native runner test cloud fixture with `generation_s=0.012`, execute one tick, and assert:

```python
assert result.record["pointcloud_generation_s"] == pytest.approx(0.012)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q kuavo_rl/tests/test_rl100_topic_native.py -k pointcloud_generation
```

Expected: FAIL because `PointCloudSample` and the audit record do not expose generation latency.

- [ ] **Step 3: Add generation timing**

Add `generation_s: float = 0.0` to `PointCloudSample`. Capture `generation_started = time.monotonic()` at the beginning of `_fuse_bundle()` and set `generation_s=max(0.0, generated_monotonic-generation_started)` in its result. Add this top-level field to successful runner records:

```python
"pointcloud_generation_s": cloud.generation_s,
```

- [ ] **Step 4: Run all related automated tests**

Run:

```bash
pytest -q kuavo_rl/tests/test_rl100_camera_sync.py kuavo_rl/tests/test_rl100_zarr.py kuavo_rl/tests/test_rl100_topic_executor.py kuavo_rl/tests/test_rl100_topic_native.py
git diff --check
```

Expected: all tests PASS and `git diff --check` emits no output.

- [ ] **Step 5: Run isolated model baseline**

Run:

```bash
bash scripts/rl/run_rl100_real.sh inspect-checkpoint \
  --config configs/rl/rl100_real_deploy.yaml
```

Record all three warmup latencies; the second and third runs are the warmed baseline.

- [ ] **Step 6: Run 60-second robot shadow verification**

Run:

```bash
bash scripts/rl/run_rl100_real.sh shadow \
  --config configs/rl/rl100_real_deploy.yaml \
  --max-steps 200 --duration-s 60 --preflight-timeout-s 10
```

Parse the generated JSONL and report p50/p95/max for `pointcloud_generation_s`, unique completed `inference_s` values, camera header/receive skew, source faults, and terminal state. Acceptance requires point-cloud p95 `< 0.050`, inference p95 `< 0.500`, no terminal fault, and strict camera sync limits.

- [ ] **Step 7: Check Jetson memory pressure during shadow**

Run `tegrastats --interval 1000` concurrently with the shadow command. Compare RAM and SWAP at the start/end; acceptance requires no increasing SWAP over the measured interval.

- [ ] **Step 8: Document measured results**

Update `docs/RL100_REAL_DEPLOY_TROUBLESHOOTING.md` with the before/after point-cloud and inference p95/max values, candidate budget, test command, and any unmet acceptance criterion. Do not claim the 0.4-second action horizon is sustainable if warmed inference remains above it.

- [ ] **Step 9: Commit Task 3**

```bash
git add kuavo_rl/rl100_zarr/ros_depth.py kuavo_rl/rl100_real_runner.py kuavo_rl/tests/test_rl100_topic_native.py docs/RL100_REAL_DEPLOY_TROUBLESHOOTING.md
git commit -m "perf: audit RL100 observation latency"
```
