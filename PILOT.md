# Behavior CI Pilot

**One behavior, one Isaac Sim world, one PR gate before hardware.**

That's the whole pilot. Pick a single robot behavior that matters to you (this repo's example:
a Unitree G1 weld approach around a shifting obstacle). We author it as a pinned Task Pack in
one Isaac Sim world, and wire a GitHub check so every policy PR is graded against it —
red/green with replay evidence — before anything touches hardware.

## What's included

- **Task pack authoring.** We work with your team to define the behavior as a pinned task:
  scenario geometry (visible + held-out perturbation bank), the observation/action contract,
  independent trajectory measurement, and pass/fail checks. The judge is sha256-locked
  (`task.lock`); a policy PR cannot edit it, only its own opaque checkpoint.
- **Hosted gate.** The offline fixture check runs on every PR (forks included, no secrets);
  the hosted Isaac Sim run executes the policy in the saved scene per commit.
- **PR comments with replay evidence.** Each commit gets a comment: verdict, per-scenario
  metrics, and the replay GIF captured from the pass/fail camera.
- **Red/green history.** Failing policies stay failing in the record (see
  `g1_weld_approach_v18` and `..._v24_undertrained` here) — the gate's value is that red is
  real.

## What's excluded (for now)

- Fleet capacity SLA — pilot sessions run on shared capacity, best effort.
- Billing/self-serve — the pilot is invoiced manually.
- A GitHub App — the check is wired via workflow + API key, not an installable app.

## Prerequisites

- A GitHub repo where policy changes arrive as PRs.
- Python >= 3.11 in CI (the workflow templates in `.github/workflows/` here are the
  reference wiring).
- A Cybernetic Physics API key (we issue one for the pilot) for the hosted Isaac job; the
  offline contract job needs no secrets.
- A behavior you can describe as: observation in, trajectory out, measurable outcome.

## Honest provenance

Every result states exactly what ran: `simulator_adapter` (fixture vs hosted Isaac),
`replay_source`, `policy_backend` and `policy_backend_real_vla`, and the sha256 digests of
the task that judged it. The learned policies in this repo are real trained MLPs — and the
provenance still says `real VLA: false`, because they aren't VLAs. We do not overclaim what
ran, and the gate does not trust any number a policy reports about itself.
