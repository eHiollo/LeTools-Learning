# Task 3 report — point-cloud latency audit and robot verification

## Status

Implemented latency instrumentation, passed the scoped suite, and completed a shadow-only robot measurement. No commands were published.

## RED / GREEN

The focused audit test initially failed because `PointCloudSample` had no `generation_s` and successful runner records had no `pointcloud_generation_s`. After implementation:

```text
pytest -q kuavo_rl/tests/test_rl100_camera_sync.py kuavo_rl/tests/test_rl100_zarr.py kuavo_rl/tests/test_rl100_topic_executor.py kuavo_rl/tests/test_rl100_topic_native.py
42 passed in 1.70s
```

`git diff --check` produced no output.

## Robot evidence

`inspect-checkpoint` reported DDIM, `use_cm: false`, 70.8233M parameters, and latencies `[4.563162, 0.415520, 0.417831]` seconds.

The 60-second shadow run produced 586 ticks and 45 completed inferences. Point-cloud generation p50/p95/max was 34.2/58.6/82.6 ms. Inference p50/p95/max was 1.249/1.482/1.499 s. Header skew was 14.3/22.0/22.0 ms and receive skew was 31.5/41.2/43.4 ms. All records were SHADOW, fault NONE, and `published=false`.

During the measured window tegrastats showed RAM growth from roughly 4.9 GB to 6.4 GB and SWAP growth from 1111 MB to roughly 1149 MB. Thus point-cloud p95 and inference p95 miss their acceptance thresholds and swap did grow; synchronization and safety acceptance passed.

## Changed files

- `kuavo_rl/rl100_zarr/ros_depth.py`
- `kuavo_rl/rl100_real_runner.py`
- `kuavo_rl/tests/test_rl100_topic_native.py`
- `docs/RL100_REAL_DEPLOY_TROUBLESHOOTING.md`

## Conclusion

Candidate-bounded back-projection removes the dominant point-cloud allocation path, but the warmed isolated 0.417 s model becomes roughly 1.25 s under full concurrent load. A one-step CM checkpoint produced by corrected after-offline distillation is the appropriate next training target; increasing timeout alone cannot make the 0.4-second action horizon sustainable.
