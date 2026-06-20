# Unitree G1 VLA Policies

Public Cybernetic Physics Behavior CI showcase for robot policy pull requests.

This repository demonstrates the product shape we want robot teams to see in
GitHub:

- a policy PR changes a Unitree G1-compatible VLA artifact,
- Cybernetic Physics runs an obstacle-shift behavior check,
- the PR gets a pass/fail comment with metrics,
- the artifact bundle includes a static report and Isaac Sim replay videos.

Demo policy story:

- Broken PR: `finetune: update weld-approach VLA for shifted obstacle task`
- Broken policy: `policies/g1_weld_approach_v18.pt`
- Fixed policy: `policies/g1_weld_approach_v19.pt`
- World: `tabletop_welding_obstacle_shift_v1`
- Camera: `/World/Cameras/BehaviorCI_PassFailCamera`

The visible `.pt` files are small JSON policy manifests for this showcase. They
resolve to the honest `scripted-vla-shim` backend until a real VLA/GR00T policy
runner is wired in. The replay videos used by CI are separate Isaac Sim session
captures, not generated placeholder clips.

## Replay Evidence Contract

Behavior CI expects these files before the GitHub Action can produce final
evidence:

```text
assets/isaac-replays/replay-failed.mp4
assets/isaac-replays/replay-passed.mp4
```

Those MP4s must be captured from an Isaac Sim session camera equivalent to:

```text
/World/Cameras/BehaviorCI_PassFailCamera
```

The public workflow runs with `--require-isaac-replays`, so missing or invalid
MP4 inputs are treated as infrastructure failures instead of silently generating
placeholder replay evidence.

## Local Commands

Run the broken policy locally:

```bash
python3 scripts/cybernetic_behavior_ci.py \
  --robot unitree-g1-or-selected-humanoid \
  --policy-ref policies/g1_weld_approach_v18.pt \
  --task configs/tasks/tabletop_welding.yaml \
  --eval evals/g1_weld_obstacle_shift.yaml \
  --scene-env behavior-ci-tabletop-welding \
  --camera /World/Cameras/BehaviorCI_PassFailCamera \
  --out artifacts \
  --replay-source-dir assets/isaac-replays \
  --require-isaac-replays
```

The v18 policy exits `1` after producing artifacts because the robot behavior
regressed. Add `--no-fail-on-result` when generating artifacts for local
inspection without failing the shell command.

Run the fixed policy:

```bash
python3 scripts/cybernetic_behavior_ci.py \
  --robot unitree-g1-or-selected-humanoid \
  --policy-ref policies/g1_weld_approach_v19.pt \
  --task configs/tasks/tabletop_welding.yaml \
  --eval evals/g1_weld_obstacle_shift.yaml \
  --scene-env behavior-ci-tabletop-welding \
  --camera /World/Cameras/BehaviorCI_PassFailCamera \
  --out artifacts \
  --replay-source-dir assets/isaac-replays \
  --require-isaac-replays
```

For runner-only development, omit `--require-isaac-replays` to allow the local
fallback renderer. Do not use that fallback for final showcase evidence.
