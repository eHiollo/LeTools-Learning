from __future__ import annotations

from collections.abc import Callable


class ArmControlModeSession:
    """Track a live arm-control mode change and perform one safe restore."""

    def __init__(
        self,
        set_mode: Callable[[int], bool],
        *,
        external_mode: int = 2,
        restore_mode: int = 0,
    ) -> None:
        self._set_mode = set_mode
        self._external_mode = int(external_mode)
        self._restore_mode = int(restore_mode)
        self._active = False

    def enter(self) -> None:
        if self._active:
            return
        if not self._set_mode(self._external_mode):
            raise RuntimeError("failed to enter external arm control mode")
        self._active = True

    def restore(self) -> None:
        if not self._active:
            return
        self._active = False
        if not self._set_mode(self._restore_mode):
            raise RuntimeError("failed to restore arm control mode")
