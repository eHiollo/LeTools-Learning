# RL100 Follow-up README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a Chinese, copy-paste-ready follow-up workflow for RL100 data conversion, CM retraining, checkpoint inspection, shadow acceptance, and guarded live deployment.

**Architecture:** The LeTools root README owns the robot-side end-to-end runbook and links to detailed troubleshooting. The independent RL-100 README owns the training-side CM distillation recipe and checkpoint acceptance contract. Each repository is committed and pushed independently.

**Tech Stack:** Markdown, Bash, ROS Noetic, RL-100 DP3/DDIM/CM, Git.

## Global Constraints

- Never instruct operators to start with `live`; `inspect-checkpoint`, preflight, and shadow must pass first.
- Preserve the final `(1024, 3)` point-cloud contract and shared conversion/deployment candidate budget.
- Require a one-step CM checkpoint and measured shadow p95 below the 0.4-second action horizon before live acceptance.
- Do not add user-owned untracked files to either commit.

---

### Task 1: Main repository runbook

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: existing RL100 collection/deployment scripts and configs.
- Produces: Chinese operator workflow from data rebuild through guarded live deployment.

- [ ] Add the measured current status and why the old DDIM checkpoint is not live-ready.
- [ ] Add copy-paste commands for conversion checks, checkpoint inspection, ROS preflight, shadow, metric gates, and guarded live.
- [ ] Link the detailed troubleshooting and raw-rosbag documentation.
- [ ] Verify all referenced local paths exist and inspect the Markdown diff.

### Task 2: Independent RL-100 training runbook

**Files:**
- Modify: `third_party/RL-100/README.md` in the independent RL-100 repository.

**Interfaces:**
- Consumes: corrected offline training launchers and LeTools-produced zarr data.
- Produces: Chinese CM retraining and checkpoint handoff workflow.

- [ ] Explain the corrected `distill_phase=after_offline` contract and one-step CM target.
- [ ] Document dataset placement, launcher/config checks, checkpoint acceptance fields, and handoff back to LeTools.
- [ ] Run exact typo scan and `bash -n` for affected launchers.
- [ ] Commit with a Chinese message and push RL-100 `main`.

### Task 3: Verify and publish LeTools documentation

**Files:**
- Modify: `README.md`
- Add: `docs/superpowers/plans/2026-08-07-rl100-follow-up-readme.md`

**Interfaces:**
- Consumes: both completed README edits.
- Produces: pushed `dev/rl100_record` documentation commit.

- [ ] Run the 43-test RL100 scoped suite and `git diff --check`.
- [ ] Confirm only intended tracked files are staged.
- [ ] Commit with a Chinese message and push `dev/rl100_record`.
