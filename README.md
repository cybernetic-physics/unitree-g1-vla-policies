# Unitree G1 VLA Policies — Cybernetic Physics Behavior CI

**CodeRabbit reviews whether the code looks right. Cybernetic Physics reviews whether the robot still works — in a way you can't game.**

This is a public, runnable showcase of **Behavior CI**: a GitHub-native check that runs a
changed robot policy through a *pinned* simulation eval and returns a red/green verdict,
metrics, and replay evidence on the pull request — a linter, but for *robot behavior*.

The point of this repo is the **trust boundary**. When you let an agent iterate on a policy
to make CI green, the eval has to be something the agent *cannot* simply edit or self-report
its way past. Here it isn't.

## The boundary: a policy changes only its checkpoint

A policy is a `policies/*.pt` file — a closed **v2 manifest** that carries only an opaque
`checkpoint` and the name of a pinned **task**:

```json
{ "schema_version": "behavior-ci-policy/v2", "policy_id": "g1_weld_approach_v21",
  "behavior": "g1_weld_approach", "backend": "scripted-vla-shim",
  "task": "g1_weld_approach",
  "checkpoint": { "detour_mode": "relative", "detour_gain": 1.0, "clearance_margin_cm": 12.0,
                  "top_halfwidth_cm": 30.0, "approach_speed_mps": 0.12 } }
```

Everything that *decides pass/fail* — the eval thresholds, the scenario geometry, the
action/observation contract, the outcome measurement, the in-session grader, the held-out
perturbation bank, and the saved Isaac scene `env_id` — lives in a **Task Pack inside the
installed SDK** (`cybernetics/behavior_ci/tasks/g1_weld_approach/`), pinned by commit SHA.
The candidate repo cannot reach those bytes. The `evals/` and `isaac/` files here are
**verified read-only copies** (their sha256 is pinned in the SDK lock); editing them is
rejected, not honored.

## Reward Hacking?

The policy **emits a trajectory**; the environment **measures it independently**. There is no
number the grader trusts. So the obvious cheats all turn the check **red**:

| Gaming attempt | Result |
|---|---|
| Inflate / self-report a "clearance" number | nothing reads it as truth; the trajectory is measured (and a too-big detour *fails* — see below) |
| Smuggle a `session_entrypoint` (pick your own grader) | ❌ **exit 2** — closed v2 schema rejects unknown keys |
| **Crank the detour** to "be safer" | ❌ **exit 1** — measurement is **non-monotone**: too small collides/intrudes, too large busts tilt/timeout and hits a ceiling zone |
| Memorize the 8 visible scenarios | ❌ — the **held-out perturbation bank** (shipped only in the SDK) fails a non-obstacle-relative policy |
| Lower a threshold in `evals/…yaml` | ❌ **exit 4** — sha256 pin mismatch; the edit is also *inert* (grading uses the pinned pack) |
| Rewrite the grader in `isaac/…py` | ❌ **exit 4** — sha256 pin mismatch |
| Tune the checkpoint into an honest obstacle-relative trajectory | ✅ **exit 0** — earned |

The one move that earns a green check is the one we want: a policy that actually routes the
torch around the *observed* obstacle, within the time and stability budget, on every scenario
including the held-out ones.

## Demo story [PR 12](https://github.com/cybernetic-physics/unitree-g1-vla-policies/pull/12)

| Commit | Change | Check |
|---|---|---|
| crank | `detour_gain` 1.0 → 8.0 ("bigger detour must be safer") | ❌ red (timeout + target miss) |
| lower the bar | edit the verified `evals/…yaml` threshold | ❌ red (pin mismatch, exit 4) |
| honest fix | revert tampering, keep the obstacle-relative `v21` checkpoint | ✅ green (16/16: visible + held-out) |

All of this is visible in the **secrets-free `contract` job**, so anyone — including a fork —
can see the gate work without a hosted session.

## Two backends

| Adapter | What runs | Needs |
|---|---|---|
| `fixture` (the offline gate) | Pure, deterministic geometric measurement of the emitted trajectory over the visible + held-out scenarios. Runs on every PR/fork. | nothing |
| `isaac-session` (the hosted gate) | Boots a hosted Isaac session, loads the pinned saved scene, actuates the real G1 along the emitted trajectory, and captures replay video. Grades with the **same** measurement (no drift). | platform API key |

> **Authoritative verdict.** A candidate can edit their own workflow YAML or SDK pin, so the
> offline job is *necessary but not sufficient*. The binding check must be **posted by
> platform infrastructure** (a server-side run the candidate can't forge), required by branch
> protection. See the PR's post-merge steps.

## Honesty

The `.pt` files are JSON manifests resolved by a `scripted-vla-shim` planner — **not** a
learned VLA, and the fixture is a kinematic model, not process-accurate welding. Provenance is
always explicit in `result.json` (`policy_backend_real_vla: false`, `pins_verified`, the task
digest, `simulator_adapter`, `replay_source`). Wiring a real VLA/GR00T checkpoint is a
documented next step: the `act(observation) -> trajectory` + `measure` contract stays the same.

## Run it

```bash
pip install "cybernetic-physics[behavior-ci] @ git+https://github.com/cybernetic-physics/cybernetic.git@<pinned-sha>"

# honest policy -> exit 0
cybernetics behavior-ci run --config cybernetic-behavior-ci.yaml \
  --policy-ref policies/g1_weld_approach_v21.pt --eval obstacle_shift --out artifacts/behavior-ci

# integrity gate (closed schema + pinned eval/grader)
cybernetics behavior-ci verify-task --config cybernetic-behavior-ci.yaml \
  --policy-ref policies/g1_weld_approach_v21.pt
```
