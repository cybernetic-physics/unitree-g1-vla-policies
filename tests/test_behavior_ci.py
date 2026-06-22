"""Golden contract for the public showcase: v18 regresses, v19 passes.

These run the installed Cybernetics SDK CLI (`cybernetics behavior-ci run`) over
the repo's config/policies/eval — the same path CI uses — and assert the
red/green story. Install the SDK first:

    pip install "cybernetic-physics[behavior-ci] @ git+https://github.com/cybernetic-physics/cybernetic.git@main"
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

CLI = shutil.which("cybernetics")
ROOT = Path(__file__).resolve().parents[1]

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
            "cybernetic-behavior-ci.yaml",
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


def test_v18_fails_runs_3_5_7(tmp_path: Path) -> None:
    out = tmp_path / "v18"
    proc = _run("policies/g1_weld_approach_v18.pt", out)
    assert proc.returncode == 1, proc.stderr
    result = json.loads((out / "result.json").read_text())
    assert result["schema_version"] == "behavior-ci/v1"
    assert result["status"] == "failed"
    assert result["summary"]["passed_runs"] == 5
    assert [f["run"] for f in result["failures"]] == [3, 5, 7]
    assert [f["code"] for f in result["failures"]] == [
        "SAFETY_ZONE_INTRUSION",
        "OBSTACLE_COLLISION",
        "TARGET_TIMEOUT",
    ]


def test_v19_passes_all_runs(tmp_path: Path) -> None:
    out = tmp_path / "v19"
    proc = _run("policies/g1_weld_approach_v19.pt", out)
    assert proc.returncode == 0, proc.stderr
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "passed"
    assert result["summary"]["passed_runs"] == 8
    assert result["failures"] == []
