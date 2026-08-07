# RL100 实机手臂控制模式生命周期设计

## 目标

让 RL100 实机部署脚本自动管理 Kuavo 轮臂的手臂控制模式：只有真正发布策略动作的 `live` 模式进入外部控制，程序退出时恢复为保持当前位置；其它检查模式不触碰机器人控制模式。

## 范围与约束

- 不修改 `third_party/kuavo-ros-control`。
- 不修改 RL100 策略、观测、动作或安全门逻辑。
- 不增加命令行参数；控制模式生命周期属于部署脚本固定的安全流程。
- 使用与 `src/demo/test_kuavo_wheel_real/cmd_arm_joint_test.py` 相同的 ROS 服务 `/wheel_arm_change_arm_ctrl_mode`。
- mode `2` 表示外部控制；退出恢复 mode `0`，即保持当前控制位置，不主动把手臂移动到初始目标。

## 方案

在部署侧增加一个最小的控制模式适配层：

1. 通过 `kuavo_msgs.srv.changeArmCtrlMode` 调用轮臂控制模式服务，等待服务最多 5 秒，并检查返回的 `result`。
2. 在 `scripts/rl/run_rl100_real.py::_run_realtime` 中，完成 ROS 预检、live 安全确认、策略加载和 runner 预检之后，才调用 mode `2`；调用失败则不进入发布循环。
3. 用一次性生命周期对象记录“是否成功进入外部控制”。进入失败不执行恢复；进入成功后，无论正常结束、Ctrl-C、ROS 异常还是推理/发布异常，都在关闭 runner、停止后续动作发布后调用 mode `0`。
4. 恢复失败只记录错误，不覆盖原始运行异常，也不阻止其它 ROS 资源清理。
5. `inspect-checkpoint`、`offline-replay`、`ros-preflight` 和 `shadow` 不调用该服务。

## 生命周期

```text
live: ROS preflight -> safety checks -> load policy -> runner preflight
      -> set mode 2 -> arm_live -> tick/publish
      -> stop runner -> set mode 0 -> close ROS components

failure before mode 2: no mode restore
failure after mode 2: best-effort mode 0 restore
non-live commands: no mode change
```

## 测试

- 生命周期单元测试验证：成功进入后只恢复一次；进入失败不恢复；恢复失败不重复调用。
- 运行现有 RL100 测试集，确保 topic-native runner 和其它部署命令不受影响。
- 对部署脚本做 Python 编译检查；不在本机伪造 ROS 服务调用，实机再做一次 `live` 前的服务可用性验证。
