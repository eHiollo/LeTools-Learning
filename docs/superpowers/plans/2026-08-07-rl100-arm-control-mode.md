# RL100 实机手臂控制模式生命周期 Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with verification after each task. Keep the existing YAML change and untracked `third_party/kuavo-ros-control/` out of all commits.

**Goal:** 让 RL100 `live` 部署自动进入 Kuavo 轮臂外部控制 mode 2，并在退出时可靠恢复 mode 0。

**Architecture:** 新增一个无 ROS 依赖的生命周期对象，只负责记录 mode 2 是否成功进入以及恢复只执行一次；部署 CLI 提供一个延迟导入 ROS 的服务适配函数，并在 `_run_realtime` 的 live 生命周期中调用。服务失败不进入发布循环，恢复失败只记录并继续清理资源。

**Tech Stack:** Python 3、pytest、rospy、`kuavo_msgs.srv.changeArmCtrlMode`。

## Global Constraints

- 只修改 `kuavo_rl/arm_control_mode.py`、`scripts/rl/run_rl100_real.py` 和对应测试。
- 不修改 `third_party/kuavo-ros-control`、策略模型、观测/动作契约和安全门。
- 只在 `live` 模式调用 `/wheel_arm_change_arm_ctrl_mode`。
- mode `2` 为外部控制，退出恢复 mode `0` 保持当前位置。
- 不新增命令行参数。

### Task 1: 生命周期对象的失败优先测试

**Files:**
- Create: `kuavo_rl/tests/test_arm_control_mode.py`
- Create later: `kuavo_rl/arm_control_mode.py`

**Interfaces:**
- Test the future `ArmControlModeSession(set_mode, external_mode=2, restore_mode=0)` interface.
- `enter()` returns `None` and raises `RuntimeError` if mode 2 is rejected.
- `restore()` returns immediately when the session is inactive; after successful entry it marks the session inactive before calling mode 0, raises `RuntimeError` if mode 0 is rejected, and never calls the service again on repeated cleanup.

- [ ] **Step 1: Write the failing tests**

```python
def test_successful_entry_restores_mode_zero_once():
    calls = []
    session = ArmControlModeSession(lambda mode: calls.append(mode) or True)
    session.enter()
    session.restore()
    session.restore()
    assert calls == [2, 0]

def test_rejected_entry_never_attempts_restore():
    calls = []
    session = ArmControlModeSession(lambda mode: calls.append(mode) and False)
    with pytest.raises(RuntimeError, match="external"):
        session.enter()
    session.restore()
    assert calls == [2]

def test_failed_restore_is_not_retried():
    calls = []
    session = ArmControlModeSession(
        lambda mode: calls.append(mode) and mode == 2
    )
    session.enter()
    with pytest.raises(RuntimeError, match="restore"):
        session.restore()
    session.restore()
    assert calls == [2, 0]
```

- [ ] **Step 2: Run the focused test and verify it fails for the missing production module**

Run: `pytest -q kuavo_rl/tests/test_arm_control_mode.py`

Expected: collection fails because `kuavo_rl.arm_control_mode` does not exist yet.

### Task 2: Implement the minimal lifecycle object

**Files:**
- Create: `kuavo_rl/arm_control_mode.py`
- Test: `kuavo_rl/tests/test_arm_control_mode.py`

**Interfaces:**
- `ArmControlModeSession.enter()` calls the injected setter with `external_mode` once and marks the session active only after a truthy result.
- `ArmControlModeSession.restore()` calls the injected setter with `restore_mode` once after successful entry; it clears active state before the call and raises on a false result.

- [ ] **Step 1: Implement only the class needed by the failing tests**
- [ ] **Step 2: Run `pytest -q kuavo_rl/tests/test_arm_control_mode.py` and verify all focused tests pass**
- [ ] **Step 3: Run `python -m py_compile kuavo_rl/arm_control_mode.py`**

### Task 3: Connect the ROS service to the live deployment lifecycle

**Files:**
- Modify: `scripts/rl/run_rl100_real.py` near the CLI helpers and `_run_realtime`
- Test: `kuavo_rl/tests/test_arm_control_mode.py`

**Interfaces:**
- Add `_set_wheel_arm_control_mode(mode: int) -> bool`; import `rospy` and `kuavo_msgs` only when called, wait up to 5 seconds for `/wheel_arm_change_arm_ctrl_mode`, send `changeArmCtrlModeRequest.control_mode`, and return the response `result`.
- Create an `ArmControlModeSession` only for `live`.
- Call `session.enter()` after `runner.preflight()` and immediately before `runner.arm_live()`.
- In the existing `finally`, close the runner first, restore mode 0 in a guarded cleanup block, then close ROS components. A restore error must be logged to stderr and must not replace the original run error.

- [ ] **Step 1: Add a test that a false setter makes `enter()` fail before live arming**
- [ ] **Step 2: Run the focused tests and verify the new test fails before the integration code exists**
- [ ] **Step 3: Implement the lazy ROS service adapter and lifecycle calls**
- [ ] **Step 4: Run the focused tests and verify they pass**
- [ ] **Step 5: Run `python -m py_compile scripts/rl/run_rl100_real.py`**

### Task 4: Full verification and change-scope review

**Files:**
- Verify only: `kuavo_rl/arm_control_mode.py`, `scripts/rl/run_rl100_real.py`, `kuavo_rl/tests/test_arm_control_mode.py`

- [ ] **Step 1: Run `pytest -q kuavo_rl/tests/test_arm_control_mode.py kuavo_rl/tests/test_rl100_topic_native.py`**
- [ ] **Step 2: Run `git diff --check`**
- [ ] **Step 3: Inspect `git diff` and confirm the existing YAML and untracked third-party tree are not staged or modified by this feature**
- [ ] **Step 4: Report that live ROS service invocation still needs the user's robot-side smoke test**
