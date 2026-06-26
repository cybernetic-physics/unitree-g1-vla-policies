# Unitree G1 VLA Policies — Cybernetic Physics Behavior CI

**CodeRabbit reviews whether the code looks right. Cybernetic Physics reviews whether the robot still works.**

This is a public, runnable showcase of **Behavior CI**: a GitHub-native check that
runs a changed robot policy through a pinned simulation eval and returns a
red/green verdict, metrics, and replay evidence on the pull request — the way a
code linter reviews a diff, but for *robot behavior*.

A developer opens a PR that changes a Unitree G1 weld-approach policy. Cybernetic
Physics runs the obstacle-shift suite, and the PR gets:

- a **pass/fail check** (red if the robot regressed, green if it works),
- a **sticky PR comment** with per-check results and metrics,
- an **artifact bundle**: a static HTML report + replay video from a fixed
  pass/fail camera.

## The demo story

| Policy | Change | Result |
|---|---|---|
| `policies/g1_weld_approach_v18.pt` | clearance margin too small, no online replan | ❌ **fails** 3/8 trials — restricted-zone intrusion, obstacle collision, timeout |
| `policies/g1_weld_approach_v19.pt` | obstacle-shift augmentation + replan restored | ✅ **passes** 8/8 trials |

The difference is a **readable controller parameter** (`clearance_margin_cm`: 6 → 14),
not a hidden flag. A trial fails its stressed check exactly when the controller's
clearance margin is smaller than that scenario's `required_clearance_cm`
(see `evals/g1_weld_obstacle_shift.yaml`).

## Run it

Install the SDK (provides the `cybernetics behavior-ci` runner):

```bash
pip install "cybernetic-physics[behavior-ci] @ git+https://github.com/cybernetic-physics/cybernetic.git@main"
```

Run the regressed policy (exits non-zero — behavior regression):

```bash
cybernetics behavior-ci run \
  --config cybernetic-behavior-ci.yaml \
  --policy-ref policies/g1_weld_approach_v18.pt \
  --eval obstacle_shift \
  --out artifacts/behavior-ci
```

Run the fixed policy (exits zero):

```bash
cybernetics behavior-ci run \
  --config cybernetic-behavior-ci.yaml \
  --policy-ref policies/g1_weld_approach_v19.pt \
  --eval obstacle_shift \
  --out artifacts/behavior-ci
```

Open `artifacts/behavior-ci/report/index.html` for the full report.

## Two backends

| Adapter | What runs | Needs | Config |
|---|---|---|---|
| `isaac-session` (the CI gate) | A **real hosted Cybernetic Physics Isaac Sim session**: boots from the saved **`cicd`** environment (the **real Unitree G1** + welding scene), uploads `isaac/behavior_ci_env.py`, drives the weld-approach in physics for each scenario, measures the metrics off the robot, and captures replay video from the pass/fail camera. | API key only | `cybernetic-behavior-ci.hosted.yaml` |
| `fixture` (local dev only) | Deterministic model from readable controller params — fast offline check while iterating. Not the CI behavior gate. | nothing | `cybernetic-behavior-ci.yaml` |

The PR check (`.github/workflows/cybernetic-behavior-ci.yml`) runs the **real
`isaac-session` validation**: it boots a hosted Isaac session, runs the changed
policy on the G1, and turns the check red/green from the **measured** result.
It needs just one secret — `CYBERNETICS_API_KEY` (in the `behavior-ci`
Environment); base/MCP URLs default to hosted production, and the scene is loaded
from the saved `cicd` environment pinned in `cybernetic-behavior-ci.hosted.yaml`.
Present on this org's PRs; **fork PRs without the key skip the hosted job with a
notice** (they don't fake a green behavior result). An offline `contract` job
always runs to validate config/SDK wiring (not robot behavior).
Provenance is always explicit in `result.json` / `provenance.json`:

- `simulator_adapter`: `fixture` | `isaac-session`
- `replay_source`: `fixture-generated` | `checked-in-demo-evidence` | `isaac-sim-session-video`
- `policy_backend_real_vla`: `false` for the scripted demo controller

> **Honesty:** the visible `.pt` files are JSON policy manifests for this showcase,
> resolved by the `scripted-vla-shim` backend. This is a behavior-CI *workflow*
> demo on a welding-themed scene — not a real learned VLA and not a
> process-accurate welding simulation. Wiring a real VLA/GR00T checkpoint is a
> documented next step (the `PolicyBackend` interface already exists in the SDK).

## Bring your own behavior

To point Behavior CI at *your* robot and task, you provide:

1. **a policy / checkpoint** — a manifest under `policies/` (and, later, a real
   backend that loads your weights),
2. **a scene module** — an `isaac/behavior_ci_env.py` with `setup_scene()` that
   builds your scene + a fixed pass/fail camera (or a pre-published `env_id` to
   warm-start from),
3. **success metrics** — the checks in an eval YAML (`evals/*.yaml`),
4. **replay requirements** — which camera, and whether real Isaac capture is
   required.

Everything is declared in `cybernetic-behavior-ci.yaml`; the SDK and a thin
GitHub workflow do the rest.

### Persona-style pilot

Give us **one** non-sensitive behavior (or a simplified analogue) — a policy, a
scene, and what "still works" means — and we return a PR check with metrics and
replay evidence on that behavior. You can run it three ways: build it internally
on the SDK, run it on our hosted platform, or co-develop a pilot.

## Layout

```
cybernetic-behavior-ci.hosted.yaml   hosted Isaac config (the CI gate)
cybernetic-behavior-ci.yaml          fixture config (local dev only)
policies/                            v18 (regressed) / v19 (fixed) manifests
evals/g1_weld_obstacle_shift.yaml    checks + per-trial obstacle-shift scenarios
isaac/behavior_ci_env.py             in-session entrypoint run on the real G1
configs/tasks/tabletop_welding.yaml  scene/task description
assets/isaac-replays/                real Unitree G1 Isaac replay clips
.github/workflows/                    real hosted-Isaac PR gate + offline contract job
tests/test_behavior_ci.py            golden v18-fails / v19-passes contract
```
