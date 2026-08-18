"""Golden + anti-gaming contract for the public showcase (v2 Task Pack).

These run the installed Cybernetics SDK CLI -- the same path CI uses -- and assert that the
honest red/green story holds AND that gaming attempts are caught. Install the pinned SDK:

    pip install "cybernetic-physics[behavior-ci] @ git+https://github.com/cybernetic-physics/cybernetic.git@<sha>"

Exit codes: 0 pass, 1 behavior regression, 2 invalid/closed-schema input, 4 pin/contract.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

CLI = shutil.which("cybernetics")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = "cybernetic-behavior-ci.yaml"

# Locally, a missing SDK is a skip so the suite stays runnable on a bare checkout. In CI a
# missing SDK is a FAILURE: a silent skip there reports green for a pipeline that never ran,
# which is the one outcome a merge gate must never produce.
if CLI is None and os.environ.get("CI"):
    raise RuntimeError(
        "Cybernetics SDK is not on PATH inside CI. Install it before running the repo tests: "
        "pip install 'cybernetic-physics[behavior-ci] @ git+https://github.com/cybernetic-physics/cybernetic.git@<pin>'"
    )

pytestmark = pytest.mark.skipif(
    CLI is None,
    reason="Cybernetics SDK not installed; pip install 'cybernetic-physics[behavior-ci]'",
)


def _run(policy_ref: str, out: Path, eval_name: str = "obstacle_shift") -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            CLI,
            "behavior-ci",
            "run",
            "--config",
            CONFIG,
            "--policy-ref",
            policy_ref,
            "--eval",
            eval_name,
            "--out",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _verify(policy_ref: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            CLI,
            "behavior-ci",
            "verify-task",
            "--config",
            CONFIG,
            "--policy-ref",
            policy_ref,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


# ----- golden: honest policies -----------------------------------------------------------
def test_v18_regresses(tmp_path: Path):
    proc = _run("policies/g1_weld_approach_v18.pt", tmp_path / "v18")
    assert proc.returncode == 1, proc.stderr
    result = json.loads((tmp_path / "v18" / "result.json").read_text())
    assert result["status"] == "failed"
    # 48 graded: (published 8 + held-out 8) scenarios x the pack's 3 domain settings.
    assert result["summary"]["total_runs"] == 48
    assert result["honesty"]["pins_verified"] is True


@pytest.mark.parametrize("pid", ["g1_weld_approach_v19", "g1_weld_approach_v21"])
def test_honest_policy_passes(pid: str, tmp_path: Path):
    proc = _run(f"policies/{pid}.pt", tmp_path / pid)
    assert proc.returncode == 0, proc.stderr
    assert (
        json.loads((tmp_path / pid / "result.json").read_text())["status"] == "passed"
    )


def test_verify_task_passes_for_clean_repo():
    assert _verify("policies/g1_weld_approach_v21.pt").returncode == 0


# ----- anti-gaming: every cheap cheat turns the check red --------------------------------
def test_crank_the_detour_is_non_monotone_red(tmp_path: Path):
    """Cranking the detour does not "make it safer" -- it busts tilt/timeout. exit 1."""
    base = json.loads((ROOT / "policies/g1_weld_approach_v21.pt").read_text())
    base["checkpoint"] = dict(base["checkpoint"], detour_gain=8.0)
    p = tmp_path / "crank.pt"
    # display_filename must match the on-disk basename or the SDK's spoofing guard rejects
    # the manifest with exit 2 before any behavior is graded -- and this test is about the
    # BEHAVIOR being red, not about the guard.
    base["display_filename"] = p.name
    p.write_text(json.dumps(base))
    assert _run(str(p), tmp_path / "out").returncode == 1


def test_inflate_or_smuggle_key_rejected(tmp_path: Path):
    """A smuggled capability key in the closed v2 manifest -> exit 2."""
    base = json.loads((ROOT / "policies/g1_weld_approach_v21.pt").read_text())
    base["session_entrypoint"] = "evil"
    p = tmp_path / "smuggle.pt"
    base["display_filename"] = p.name
    p.write_text(json.dumps(base))
    assert _verify(str(p)).returncode == 2
    assert _run(str(p), tmp_path / "out").returncode == 2


def test_tampering_task_is_rejected(tmp_path: Path):
    """Lowering the bar by editing the in-repo task (checks/measure) -> lock mismatch, exit 4."""
    task_path = ROOT / "tasks/g1_weld_approach/task.py"
    original = task_path.read_text()
    try:
        # loosen the target_reach threshold in the task's checks()
        task_path.write_text(original.replace('"<=", 2.0', '"<=", 99.0'))
        assert _verify("policies/g1_weld_approach_v18.pt").returncode == 4
    finally:
        task_path.write_text(original)


def test_tampering_grader_is_rejected(tmp_path: Path):
    """Rewriting the in-repo hosted grader -> lock mismatch, exit 4."""
    grader_path = ROOT / "tasks/g1_weld_approach/grader_isaac.py"
    original = grader_path.read_text()
    try:
        grader_path.write_text(original + "\n# tamper\n")
        assert _verify("policies/g1_weld_approach_v18.pt").returncode == 4
    finally:
        grader_path.write_text(original)


# ----- the suite: every behavior, every subsystem, every domain setting -------------------
SUITE = [
    # (policy_id, eval name, expected exit)
    ("g1_weld_approach_v24", "obstacle_shift", 0),
    ("g1_base_traverse_v1", "aisle_clutter_shift", 0),
    ("g1_seam_inspect_v1", "seam_layout_shift", 0),
    ("g1_weld_approach_v24_undertrained", "obstacle_shift", 1),
    ("g1_base_traverse_v1_nomargin", "aisle_clutter_shift", 1),
    ("g1_seam_inspect_v1_nomargin", "seam_layout_shift", 1),
]


@pytest.mark.parametrize("pid,eval_name,expected", SUITE)
def test_suite_behavior_verdicts(pid: str, eval_name: str, expected: int, tmp_path: Path):
    """Each shipped behavior is green and each deliberately-broken twin is red -- and every
    run grades 16 scenarios x 3 domain settings, so 'passes' means passes under domain
    variation, not only in the nominal world."""
    proc = _run(f"policies/{pid}.pt", tmp_path / pid, eval_name)
    assert proc.returncode == expected, proc.stdout + proc.stderr
    result = json.loads((tmp_path / pid / "result.json").read_text())
    assert result["summary"]["total_runs"] == 48


@pytest.mark.parametrize("pid,eval_name", [
    ("g1_base_traverse_v1_nomargin", "aisle_clutter_shift"),
    ("g1_seam_inspect_v1_nomargin", "seam_layout_shift"),
])
def test_zero_margin_policies_only_fail_under_observation_noise(
    pid: str, eval_name: str, tmp_path: Path
):
    """The honesty claim behind the domain sweep, asserted rather than asserted-about: these
    twins are geometrically correct (green on every 'nominal' trial) and go red ONLY on the
    'obs_noise' column. If a sweep can't tell those two apart it is decoration."""
    _run(f"policies/{pid}.pt", tmp_path / pid, eval_name)
    result = json.loads((tmp_path / pid / "result.json").read_text())
    assert result["status"] == "failed"
    domains = {f["domain"] for f in result["failures"]}
    assert domains == {"obs_noise"}, domains
    # and every failure names the scenario it happened on
    assert all(isinstance(f["scenario"], int) for f in result["failures"])


@pytest.mark.parametrize("task_id,policy", [
    ("g1_weld_approach", "g1_weld_approach_v24"),
    ("g1_base_traverse", "g1_base_traverse_v1"),
    ("g1_seam_inspect", "g1_seam_inspect_v1"),
])
def test_every_pack_is_pinned_and_tamper_evident(task_id: str, policy: str):
    """verify-task is green for a clean tree and exit 4 for any edited judge file."""
    assert _verify(f"policies/{policy}.pt").returncode == 0
    for name in ("task.py", "grader_isaac.py"):
        path = ROOT / "tasks" / task_id / name
        original = path.read_text()
        try:
            path.write_text(original + "\n# tamper\n")
            assert _verify(f"policies/{policy}.pt").returncode == 4
        finally:
            path.write_text(original)
    assert _verify(f"policies/{policy}.pt").returncode == 0


def test_suite_runs_all_three_subsystems(tmp_path: Path):
    """`suite run` grades the whole behavior list in one shot: 3 behaviors, 3 subsystems."""
    proc = subprocess.run(
        [
            CLI, "behavior-ci", "suite", "run",
            "--suite", "behavior-ci-suite.yaml",
            "--config", CONFIG,
            "--out", str(tmp_path / "suite"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for subsystem in ("manipulation", "locomotion", "perception"):
        assert subsystem in proc.stdout
