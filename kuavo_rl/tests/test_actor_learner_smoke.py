import pytest


def test_hilserl_imports_optional():
    pytest.importorskip("grpc")
    try:
        from lerobot.rl.train_rl import TrainRLServerPipelineConfig  # noqa: F401
    except Exception as exc:
        pytest.skip(f"lerobot hilserl not available: {exc}")
