"""Golden + anti-gaming contract for the public showcase (v2 Task Pack).

These run the installed Cybernetics SDK CLI -- the same path CI uses -- and assert that the
honest red/green story holds AND that gaming attempts are caught. Install the pinned SDK:

    pip install "cybernetic-physics[behavior-ci] @ git+https://github.com/cybernetic-physics/cybernetic.git@<sha>"

Exit codes: 0 pass, 1 behavior regression, 2 invalid/closed-schema input, 4 pin/contract.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

CLI = shutil.which("cybernetics")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = "cybernetic-behavior-ci.yaml"

pytestmark = pytest.mark.skipif(
    CLI is None,
    reason="Cybernetics SDK not installed; pip install 'cybernetic-physics[behavior-ci]'",
)


def _run(policy_ref: str, out: Path) -> subprocess.CompletedProcess:
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
            "obstacle_shift",
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
    # 16 graded: the published 8 + the held-out perturbation bank.
    assert result["summary"]["total_runs"] == 16
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
    p.write_text(json.dumps(base))
    assert _run(str(p), tmp_path / "out").returncode == 1


def test_inflate_or_smuggle_key_rejected(tmp_path: Path):
    """A smuggled capability key in the closed v2 manifest -> exit 2."""
    base = json.loads((ROOT / "policies/g1_weld_approach_v21.pt").read_text())
    base["session_entrypoint"] = "evil"
    p = tmp_path / "smuggle.pt"
    p.write_text(json.dumps(base))
    assert _verify(str(p)).returncode == 2
    assert _run(str(p), tmp_path / "out").returncode == 2


def test_tampering_eval_copy_is_rejected(tmp_path: Path):
    """Lowering the bar by editing the verified eval copy -> pin mismatch, exit 4."""
    eval_path = ROOT / "evals/g1_weld_obstacle_shift.yaml"
    original = eval_path.read_text()
    try:
        eval_path.write_text(original.replace("value: 2.0", "value: 99.0"))
        assert _verify("policies/g1_weld_approach_v18.pt").returncode == 4
    finally:
        eval_path.write_text(original)


def test_tampering_grader_copy_is_rejected(tmp_path: Path):
    """Rewriting the verified grader copy -> pin mismatch, exit 4."""
    grader_path = ROOT / "isaac/behavior_ci_env.py"
    original = grader_path.read_text()
    try:
        grader_path.write_text(original + "\n# tamper\n")
        assert _verify("policies/g1_weld_approach_v18.pt").returncode == 4
    finally:
        grader_path.write_text(original)
