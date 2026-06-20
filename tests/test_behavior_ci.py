import json
from pathlib import Path
import subprocess
import sys
import unittest


FIXTURE_ROOT = Path(__file__).resolve().parents[1]
RUNNER = FIXTURE_ROOT / "scripts" / "cybernetic_behavior_ci.py"


class BehaviorCiFixtureTests(unittest.TestCase):
    def run_policy(self, policy: str) -> tuple[int, dict]:
        out_dir = FIXTURE_ROOT / "artifacts" / f"test-{policy}"
        command = [
            sys.executable,
            str(RUNNER),
            "--robot",
            "unitree-g1-or-selected-humanoid",
            "--policy-ref",
            f"policies/{policy}.pt",
            "--task",
            "configs/tasks/tabletop_welding.yaml",
            "--eval",
            "evals/g1_weld_obstacle_shift.yaml",
            "--scene-env",
            "behavior-ci-tabletop-welding",
            "--camera",
            "/World/Cameras/BehaviorCI_PassFailCamera",
            "--out",
            str(out_dir),
            "--no-fail-on-result",
        ]
        completed = subprocess.run(command, check=False, cwd=FIXTURE_ROOT)
        result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        return completed.returncode, result

    def test_v18_fails_expected_runs(self) -> None:
        exit_code, result = self.run_policy("g1_weld_approach_v18")
        self.assertEqual(exit_code, 0)
        self.assertEqual(result["schema_version"], "behavior-ci/v1")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["summary"]["passed_runs"], 5)
        self.assertEqual(result["summary"]["total_runs"], 8)
        self.assertEqual([failure["run"] for failure in result["failures"]], [3, 5, 7])
        self.assertEqual(
            [failure["code"] for failure in result["failures"]],
            ["SAFETY_ZONE_INTRUSION", "OBSTACLE_COLLISION", "TARGET_TIMEOUT"],
        )

    def test_v19_passes_all_runs(self) -> None:
        exit_code, result = self.run_policy("g1_weld_approach_v19")
        self.assertEqual(exit_code, 0)
        self.assertEqual(result["schema_version"], "behavior-ci/v1")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["summary"]["passed_runs"], 8)
        self.assertEqual(result["summary"]["total_runs"], 8)
        self.assertEqual(result["failures"], [])


if __name__ == "__main__":
    unittest.main()
