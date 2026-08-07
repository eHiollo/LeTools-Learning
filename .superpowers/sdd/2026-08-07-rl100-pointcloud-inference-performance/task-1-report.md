# Task 1 Report: Bounded depth-pixel back-projection

## Changed files

- `kuavo_rl/rl100_zarr/pointcloud.py`
  - Added `candidate_count: int = 0` to `depth_to_point_cloud`.
  - Added deterministic `np.linspace` flat-index selection for positive limiting budgets.
  - Preserved the full-resolution path for nonpositive or non-limiting budgets.
  - Applied finite/depth validity masks after candidate selection; retained existing transform math and `float32` output.
- `kuavo_rl/tests/test_rl100_zarr.py`
  - Added bounded/deterministic candidate-budget coverage.
  - Added nonpositive-budget full-resolution coverage.
- `.superpowers/sdd/2026-08-07-rl100-pointcloud-inference-performance/task-1-report.md`
  - Added this report.

## TDD evidence

### RED

Command:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate letools-rl
pytest -q kuavo_rl/tests/test_rl100_zarr.py -k 'candidate_budget or nonpositive_budget'
```

Output:

```text
FF                                                                       [100%]
2 failed, 17 deselected in 0.35s
```

Both tests failed with the expected `TypeError`: `depth_to_point_cloud()` did not accept the `candidate_count` keyword.

### GREEN

Focused command:

```bash
pytest -q kuavo_rl/tests/test_rl100_zarr.py -k 'depth_to_point_cloud or fuse_three_cams or candidate_budget or nonpositive_budget'
```

Output:

```text
...                                                                      [100%]
3 passed, 16 deselected in 0.24s
```

Full point-cloud test-file command:

```bash
pytest -q kuavo_rl/tests/test_rl100_zarr.py
```

Output:

```text
...................                                                      [100%]
19 passed in 0.91s
```

## Self-review

- `git diff --check` passed with no whitespace errors.
- The limiting path reads only selected flattened depth pixels and computes only their coordinates.
- Selection is deterministic and uses the required `np.linspace(0, H*W - 1, candidate_count, dtype=np.int64)` behavior.
- Nonpositive and non-limiting budgets preserve full-resolution behavior.
- Existing validity filtering, camera-pose validation/transform, and `float32` return behavior remain intact.
- No unrelated source files were modified.

## Commit

- Implementation commit: `4ceafafddfe47e3af40430e18909c26f6d45ee1b` (`perf: bound RL100 depth back-projection`)

## Concerns

None.
