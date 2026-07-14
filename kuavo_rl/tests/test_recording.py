import json

import numpy as np

from kuavo_rl.recording import HILReplayWriter, TransitionRecord


def test_hil_replay_writer_stores_transition_state_and_frames(tmp_path):
    writer = HILReplayWriter(tmp_path, "hil")
    record = TransitionRecord(
        experiment_id="hil",
        episode_id="episode-1",
        step_id=1,
        timestamp=1.0,
        action=[0.0] * 16,
        reward=1.0,
        reward_source="manual_success",
        terminated=True,
        truncated=False,
        fault_code="NONE",
        is_intervention=True,
        action_clipped=False,
    )
    obs = {
        "observation.state": np.zeros(16, dtype=np.float32),
        "observation.images.head_cam_h": np.zeros((3, 4, 5), dtype=np.uint8),
    }
    next_obs = {
        "observation.state": np.ones(16, dtype=np.float32),
        "observation.images.head_cam_h": np.ones((3, 4, 5), dtype=np.uint8),
    }
    writer.log_transition(record, observation=obs, next_observation=next_obs)
    writer.close()

    root = tmp_path / "hil" / "replay"
    row = json.loads((root / "episodes" / "episode-1" / "transitions.jsonl").read_text())
    assert row["is_intervention"] is True
    assert np.allclose(np.load(root / row["observation"]["state"]), 0.0)
    assert np.allclose(np.load(root / row["next_observation"]["state"]), 1.0)
    image = root / row["observation"]["images"]["observation.images.head_cam_h"]
    assert image.exists()
    assert json.loads((root / "schema.json").read_text())["format"] == "kuavo-hil-replay-v1"
