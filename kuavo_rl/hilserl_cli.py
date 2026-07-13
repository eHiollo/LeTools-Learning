"""CLI wrappers: apply kuavo_rl patches then run upstream learner/actor."""

from __future__ import annotations


def _boot(role: str) -> None:
    from kuavo_rl.lerobot_patches import apply_hilserl_patches

    apply_hilserl_patches()
    if role == "learner":
        from lerobot.rl.learner import train_cli

        train_cli()
    elif role == "actor":
        from lerobot.rl.actor import actor_cli

        actor_cli()
    else:
        raise SystemExit(f"unknown role: {role}")


def main_learner() -> None:
    _boot("learner")


def main_actor() -> None:
    _boot("actor")


if __name__ == "__main__":
    import sys

    role = sys.argv[1] if len(sys.argv) > 1 else ""
    # Allow: python -m kuavo_rl.hilserl_cli learner --config_path ...
    if role in {"learner", "actor"}:
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        _boot(role)
    else:
        raise SystemExit("usage: python -m kuavo_rl.hilserl_cli {learner|actor} [lerobot args...]")
