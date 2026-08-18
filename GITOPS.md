# Behavior CI, in git

Everything that decides whether a robot behavior is good enough to ship is a file in this
repository, and every promotion is a git operation. There is no dashboard you have to trust
and no state that lives somewhere you cannot diff.

## What is a file here

| Thing | Where | Why it is in git |
|---|---|---|
| A policy | `policies/*.pt` | A closed JSON manifest: an id, the task it claims, and an opaque checkpoint. It is a pointer, not weights, so it reviews like a config change and reverts like one. |
| The judge | `tasks/<behavior>/task.py`, `grader_isaac.py` | Scenario geometry, the observation and action contract, the independent measurement, and the thresholds. |
| The judge's fingerprint | `tasks/<behavior>/task.lock` | sha256 of every judge file. A candidate cannot edit the judge for its own run: a mismatch is exit 4. |
| The suite | `behavior-ci-suite.yaml` | Which behaviors exist, which subsystem each belongs to, which policy each pins, and the goal in one sentence. |
| What "still working" means | `baselines/main.json` | The last green result per behavior. A change to what counts as a regression arrives as a reviewable diff. |
| The pipeline | `.github/workflows/behavior-*.yml` | Triggers, tiers, and promotion rules. |

## The three triggers

**Every commit.** `behavior-suite-pr.yml` runs two lanes. The contract lane is cheap, needs no
credentials, runs on forks, and fans out one job per subsystem. The hosted lane spends GPU
time only on behaviors whose policy pointer this pull request actually changed, and posts the
verdict on the pull request.

**Every night.** `behavior-suite-nightly.yml` grades the entire suite against
`baselines/main.json`. This is what catches a behavior nobody edited: per-commit runs only
look at what moved, so without the nightly a skill can rot for weeks between edits. A
regression opens an issue; a green night closes it.

**Every promotion.** `behavior-promote.yml` runs on `main`. If the whole suite passes, it
moves the `promoted` branch to that commit and writes an annotated tag
(`behaviors-YYYY.MM.DD-<sha>`). If anything is red, `promoted` does not move.

## The current branch

`promoted` is an ordinary branch whose tip is always a commit where every behavior passed.

```bash
git fetch origin promoted          # what is blessed right now
git log --oneline promoted         # the promotion history
git diff promoted main -- policies # what is staged to ship but not yet green
git tag --list 'behaviors-*'       # every previously blessed set
```

Anything downstream (a robot image build, a deployment job, an operator asking what is on the
machine) pins to `promoted` or to a tag. Rollback is `git reset --hard <previous tag>` and a
push, not a console click, and because a tag pins the policy pointers, the task packs that
judged them, and the thresholds together, checking out a tag reproduces exactly the state
that was blessed.

## What a policy change looks like

1. Train, export a checkpoint, and update the pointer in `policies/`.
2. Open a pull request. The contract lane answers in about a minute; the hosted lane grades
   the changed behavior on the real robot in simulation and comments with metrics and a
   replay.
3. Red blocks the merge. Green merges to `main`.
4. `main` re-runs the whole suite. If it is green, `promoted` advances and a tag is written.

## Honest provenance

Every verdict this pipeline emits carries a `grading` field. Today it reads
`geometric-contract`: the pass or fail comes from the task's exact geometric measurement of
the emitted trajectory, and the hosted Isaac session renders the replay you see. That is a
deterministic contract check, and it is why the offline and hosted lanes can never disagree.
It is not a physics rollout, and nothing here should be described as one until the grade is
derived from simulated contacts.
