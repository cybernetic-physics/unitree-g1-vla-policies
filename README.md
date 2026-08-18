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

## The suite: every behavior we've built up, run together

One behavior is a demo. The thing a robotics team actually needs is *all of the existing
behaviors we've built up — from the subsystem layer — run with goal conditions, giving a yes
or no answer, while you vary the domain parameters.* That is `behavior-ci-suite.yaml`:

| Behavior | Subsystem | Task pack | Goal |
|---|---|---|---|
| `weld_approach` | manipulation | `g1_weld_approach` | route the torch to the seam around the shifted obstacle |
| `base_traverse` | locomotion | `g1_base_traverse` | walk the cluttered aisle to the work position |
| `seam_inspect` | perception | `g1_seam_inspect` | aim the head camera so every seam target is inspected |

```bash
cybernetics behavior-ci suite run --suite behavior-ci-suite.yaml \
  --config cybernetic-behavior-ci.yaml --out artifacts/suite
```

```
[PASS] weld_approach (manipulation) 48/48 trials — Route the torch to the weld seam around …
[PASS] base_traverse (locomotion)   48/48 trials — Walk from the start pose to the work position …
[PASS] seam_inspect (perception)    48/48 trials — From a standoff position, aim the head camera …
[PASS] suite 3/3 behaviors passed (grading=geometric-contract, simulator=fixture)
```

Each pack is judged the same way the weld pack always was: 8 visible + 8 held-out scenarios,
built so **no constant answer clears both sets**. The specific numbers are in each
`tasks/<id>/task.py` module docstring under `ANTI-OVERFIT INVARIANT` — e.g. for the traverse,
the held-out racking row forbids a lateral offset ≥ 145 cm while two other held-out rows
require > 150 cm and > 175 cm, and a 1 cm sweep over every constant offset in [60, 400] cm
never does better than 15/16.

## Domain sweeps: does it still work when the world moves?

48 trials, not 16, because each pack declares a `domain_sweep` and **every scenario is graded
once per domain setting**:

| Domain | What varies |
|---|---|
| `nominal` | nothing — the neutral control, numerically identical to a pre-sweep run |
| `obs_noise` | what the policy is allowed to SEE is displaced by a few cm |
| `actuation_degraded` | the achieved speed is scaled down and a start-up latency is charged |

The asymmetry is the point: `apply_domain()` perturbs only the **observation handed to the
policy**, while `measure()` always grades against the **truth** geometry. So a policy cannot
pass by being graded against its own mis-estimate — the same thing that happens on hardware.

That makes the sweep falsifiable, and this repo ships the falsification. `*_nomargin.pt` are
policies that are *geometrically correct with zero reserve*:

| Policy | `nominal` | `obs_noise` | `actuation_degraded` |
|---|---|---|---|
| `g1_base_traverse_v1` (margin 65 cm) | 16/16 | 16/16 | 16/16 |
| `g1_base_traverse_v1_nomargin` (margin 53 cm) | 16/16 | **14/16** ❌ | 16/16 |
| `g1_seam_inspect_v1` (0.8× cone, +20 cm standoff) | 16/16 | 16/16 | 16/16 |
| `g1_seam_inspect_v1_nomargin` (1.0× cone, +0 cm standoff) | 16/16 | **2/16** ❌ | 16/16 |

Both twins are green in the nominal world and red only under observation noise, and each
failure record in `result.json` names the scenario *and* the domain that produced it. A sweep
that couldn't separate those two policies would be decoration.

The sweep values are **per pack**, and deliberately so: the weld pack's shipped policies are
frozen red/green history that must not be re-tuned, and they use 26.8 s of a 30 s budget on
the far-seam rows — so its degraded column is `speed_scale 0.95 / latency_s 0.5`, not the
`0.8 / 1.5` the locomotion pack was designed with. A uniform setting would have been cosmetic;
the reasoning for every number is in the `DOMAIN_SWEEP` comment in each `task.py`.

## Two backends

| Adapter | What runs | Needs |
|---|---|---|
| `fixture` (the offline gate) | Pure, deterministic geometric measurement of the emitted trajectory over the visible + held-out scenarios. Runs on every PR/fork. | nothing |
| `isaac-session` (the hosted gate) | Boots a hosted Isaac session, loads the pinned saved scene, actuates the real G1 along the emitted trajectory, and captures replay video. Grades with the **same** measurement (no drift). | platform API key |

> **Authoritative verdict.** A candidate can edit their own workflow YAML or SDK pin, so the
> offline job is *necessary but not sufficient*. The binding check must be **posted by
> platform infrastructure** (a server-side run the candidate can't forge), required by branch
> protection. See the PR's post-merge steps.

## Use a real learned policy (`learned-mlp`, v24+)

Starting with `policies/g1_weld_approach_v24.pt`, the repo also carries a **real learned
policy**: the checkpoint holds base64-encoded float32 weights for a `[2, 8, 3]` tanh MLP
with a linear skip connection. The SDK's `learned-mlp` backend *decodes* the weights (closed
checkpoint schema: `format`, `arch`, `weights_b64`, `feature_spec`, `output_spec` — nothing
else); the task planner runs the forward pass in pure Python:

```
f   = [obstacle_top_y / 100, seam_x / 300]        # feature_spec weld-geometry/v1
h   = tanh(W1ᵀ f + b1)
out = W2ᵀ h + b2 + Wskipᵀ f + bskip               # output_spec trapezoid/v1
    = [apex_cm, top_halfwidth_cm, speed_mps]
```

The environment still measures the emitted trajectory independently — trained weights get
no more trust than the scripted shim did.

How v24 was made: `scripts/train_weld_mlp.py` (pure numpy, deterministic seed) distills the
obstacle-relative controller (`apex = obstacle_top_y + 12`, halfwidth 30, speed 0.12) into
the net with plain gradient descent over sampled task geometry (offsets 6–70 cm, both +y-face
variants, both seam distances), to near-zero loss. The linear skip keeps extrapolation to
large obstacles linear instead of tanh-saturated — that is what lets it clear the held-out
50/58 cm offsets. `g1_weld_approach_v24_undertrained.pt` is the same architecture stopped
after 5 gradient steps: it fails the gate geometrically, which is the point.

This is a genuinely learned policy but **not** a VLA: provenance reads
`policy backend: learned-mlp (real VLA: false)`. Only a future `real-vla` backend may claim
otherwise.

## Honesty

The two new behaviors (`g1_base_traverse_v1`, `g1_seam_inspect_v1`) ship **scripted**
checkpoints, not learned ones. Their `task.py` planners implement the `learned-mlp` checkpoint
shape as well (`feature_spec traverse-geometry/v1` / `inspect-geometry/v1`, normalized geometry
in, clamped plan parameters out), but the SDK's `learned-mlp` backend currently admits only
`weld-geometry/v1`, so no trained checkpoint can be minted for them yet and none is claimed. A
learned twin for either behavior would ship with its own training script, exactly as
`scripts/train_weld_mlp.py` backs v24.

The v18–v22 `.pt` files are JSON manifests resolved by a `scripted-vla-shim` planner — **not**
learned policies; they remain checked in as the original red/green demo history. From v24 the
manifests carry **real trained MLP weights** (see above) — learned, but still not a VLA, and
the fixture is a kinematic model, not process-accurate welding. Provenance is always explicit
in `result.json` (`policy_backend_real_vla: false`, `pins_verified`, the task digest,
`simulator_adapter`, `replay_source`). Wiring a real VLA/GR00T checkpoint is a documented
next step: the `act(observation) -> trajectory` + `measure` contract stays the same.

## Prerequisites

- Python **>= 3.11** (the SDK and this repo's task pack use 3.11+ features)
- [`uv`](https://docs.astral.sh/uv/) recommended for env + installs (plain `pip` works too)

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  "cybernetic-physics[behavior-ci] @ git+https://github.com/cybernetic-physics/cybernetic.git@<pinned-sha>"
```

(`scripts/train_weld_mlp.py` additionally needs `numpy`; the judged task pack itself is
dependency-free on purpose.)

## Run it

```bash
pip install "cybernetic-physics[behavior-ci] @ git+https://github.com/cybernetic-physics/cybernetic.git@<pinned-sha>"

# honest policy -> exit 0
cybernetics behavior-ci run --config cybernetic-behavior-ci.yaml \
  --policy-ref policies/g1_weld_approach_v21.pt --eval obstacle_shift --out artifacts/behavior-ci

# the real learned MLP policy -> exit 0 (48/48: 16 scenarios x 3 domain settings)
cybernetics behavior-ci run --config cybernetic-behavior-ci.yaml \
  --policy-ref policies/g1_weld_approach_v24.pt --eval obstacle_shift --out artifacts/behavior-ci

# the locomotion + perception behaviors -> exit 0
cybernetics behavior-ci run --config cybernetic-behavior-ci.yaml \
  --policy-ref policies/g1_base_traverse_v1.pt --eval aisle_clutter_shift --out artifacts/behavior-ci
cybernetics behavior-ci run --config cybernetic-behavior-ci.yaml \
  --policy-ref policies/g1_seam_inspect_v1.pt --eval seam_layout_shift --out artifacts/behavior-ci

# a zero-margin twin -> exit 1, red only on the obs_noise domain column
cybernetics behavior-ci run --config cybernetic-behavior-ci.yaml \
  --policy-ref policies/g1_seam_inspect_v1_nomargin.pt --eval seam_layout_shift --out artifacts/behavior-ci

# the whole suite (manipulation + locomotion + perception) in one shot
cybernetics behavior-ci suite run --suite behavior-ci-suite.yaml \
  --config cybernetic-behavior-ci.yaml --out artifacts/suite

# integrity gate (closed schema + pinned eval/grader), per pack
cybernetics behavior-ci verify-task --config cybernetic-behavior-ci.yaml \
  --policy-ref policies/g1_base_traverse_v1.pt
```
