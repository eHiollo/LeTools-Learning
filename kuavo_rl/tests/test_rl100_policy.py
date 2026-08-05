from __future__ import annotations

import pytest

from kuavo_rl.rl100_policy import (
    normalize_module_prefix,
    select_state_dict_key,
    validate_shape_meta,
)


def _shape_meta():
    return {
        "obs": {
            "point_cloud": {"shape": [1024, 3]},
            "agent_pos": {"shape": [32]},
        },
        "action": {"shape": [26]},
    }


def test_validate_real_robot_shape_meta():
    assert validate_shape_meta(_shape_meta()) == (1024, 3, 32, 26)
    bad = _shape_meta()
    bad["obs"]["agent_pos"]["shape"] = [14]
    with pytest.raises(ValueError, match="agent_pos"):
        validate_shape_meta(bad)


def test_select_ema_and_explicit_missing_error():
    cfg = {"training": {"use_ema": True}}
    states = {"model": {"a": 1}, "ema_model": {"a": 2}}
    assert select_state_dict_key(states, cfg) == "ema_model"
    assert select_state_dict_key(states, cfg, "model") == "model"
    with pytest.raises(ValueError, match="EMA"):
        select_state_dict_key({"model": {"a": 1}}, cfg, "ema_model")


def test_ddp_prefix_is_all_or_nothing():
    stripped, did_strip = normalize_module_prefix({"module.a": 1, "module.b": 2})
    assert did_strip and stripped == {"a": 1, "b": 2}
    untouched, did_strip = normalize_module_prefix({"module.a": 1, "b": 2})
    assert not did_strip and untouched == {"module.a": 1, "b": 2}
