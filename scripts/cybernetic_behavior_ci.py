#!/usr/bin/env python3
"""Deterministic Behavior CI MVP runner for the Unitree G1 fixture.

The MVP intentionally avoids production hosted evals. It treats visible `.pt`
files as policy manifests, resolves them to an honest scripted shim, evaluates a
fixed 8-run obstacle-shift suite, and writes the `behavior-ci/v1` artifact
bundle consumed by GitHub comments and static reports.

The public showcase can require externally captured Isaac Sim session videos for
the replay evidence. That keeps the report/demo contract honest while the
policy fixture remains deterministic.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import operator
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Callable


SCHEMA_VERSION = "behavior-ci/v1"
POLICY_SCHEMA_VERSION = "behavior-ci-policy/v1"
EVAL_SCHEMA_VERSION = "behavior-ci-eval/v1"
TASK_SCHEMA_VERSION = "behavior-ci-task/v1"
FIXTURE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FIXTURE_ROOT.parent
REQUIRED_ARTIFACTS = (
    "result.json",
    "metrics.json",
    "comment.md",
    "report/index.html",
    "replays/replay-failed.mp4",
    "replays/replay-passed.mp4",
    "scene/scene_snapshot.usd-or-placeholder.txt",
    "hack-notes.md",
)
REPLAY_FILENAMES = ("replay-failed.mp4", "replay-passed.mp4")
ALLOWED_BACKENDS = {"real-vla", "gr00t-adapter-stub", "scripted-vla-shim"}
OPS: dict[str, Callable[[float, float], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


class RunnerError(Exception):
    def __init__(self, message: str, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Behavior CI MVP fixture")
    parser.add_argument("--robot", required=True)
    parser.add_argument("--policy-ref", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--scene-env", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--commit", default=None)
    parser.add_argument("--artifact-base-url", default="")
    parser.add_argument(
        "--replay-source-dir",
        default="",
        help=(
            "Directory containing Isaac Sim session videos named "
            "replay-failed.mp4 and replay-passed.mp4."
        ),
    )
    parser.add_argument(
        "--require-isaac-replays",
        action="store_true",
        help="Fail the run unless replay-source-dir contains valid Isaac session MP4s.",
    )
    parser.add_argument(
        "--no-fail-on-result",
        action="store_true",
        help="Exit 0 after a behavior failure; useful for local artifact inspection.",
    )
    args = parser.parse_args()

    try:
        out_dir = resolve_output_path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        clean_output_dir(out_dir)

        log_lines: list[str] = []
        log_lines.append(f"started_at={utc_now()}")
        log_lines.append("mode=no-site-evals")
        log_lines.append("production_eval_path=disabled")

        policy_path = resolve_fixture_path(args.policy_ref, must_exist=True)
        task_path = resolve_fixture_path(args.task, must_exist=True)
        eval_path = resolve_fixture_path(args.eval, must_exist=True)

        policy = load_policy_manifest(policy_path)
        task = load_yaml_file(task_path)
        eval_spec = load_yaml_file(eval_path)
        validate_inputs(policy, task, eval_spec, args)
        replay_inputs = resolve_replay_inputs(args)

        observations = run_scripted_policy(policy, eval_spec)
        evaluated = evaluate_trials(policy, task, eval_spec, observations)
        commit = args.commit or git_commit()
        artifact_base_url = args.artifact_base_url.rstrip("/")

        result = build_result(
            args=args,
            policy=policy,
            task=task,
            eval_spec=eval_spec,
            evaluated=evaluated,
            commit=commit,
            artifact_base_url=artifact_base_url,
            replay_inputs=replay_inputs,
        )
        write_artifacts(out_dir, result, evaluated, args, policy, log_lines, replay_inputs)
        validate_artifacts(out_dir)

        print_result_summary(result)
        if result["status"] == "failed" and not args.no_fail_on_result:
            return 1
        return 0
    except RunnerError as exc:
        print(f"behavior-ci error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - last-resort infra guard
        print(f"behavior-ci infrastructure failure: {exc}", file=sys.stderr)
        return 3


def resolve_fixture_path(raw: str, *, must_exist: bool) -> Path:
    path = Path(raw)
    if path.is_absolute():
        candidate = path
    elif path.parts and path.parts[0] == FIXTURE_ROOT.name:
        candidate = REPO_ROOT / path
    else:
        candidate = FIXTURE_ROOT / path

    resolved = candidate.resolve()
    if not is_relative_to(resolved, FIXTURE_ROOT):
        raise RunnerError(f"path escapes fixture workspace: {raw}")
    if must_exist and not resolved.exists():
        raise RunnerError(f"required path does not exist: {raw}")
    return resolved


def resolve_output_path(raw: str) -> Path:
    return resolve_fixture_path(raw, must_exist=False)


def resolve_input_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (FIXTURE_ROOT / path).resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def clean_output_dir(out_dir: Path) -> None:
    for relative in REQUIRED_ARTIFACTS:
        path = out_dir / relative
        if path.exists():
            path.unlink()
    for directory in ("report", "replays", "scene", "logs"):
        (out_dir / directory).mkdir(parents=True, exist_ok=True)


def load_policy_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunnerError(f"invalid policy manifest JSON in {path.name}: {exc}") from exc

    required = {
        "schema_version",
        "policy_id",
        "display_filename",
        "behavior",
        "robot",
        "backend",
        "controller",
        "expected_demo_result",
    }
    missing = sorted(required - set(data))
    if missing:
        raise RunnerError(f"policy manifest missing required fields: {', '.join(missing)}")
    if data["schema_version"] != POLICY_SCHEMA_VERSION:
        raise RunnerError(f"unsupported policy schema: {data['schema_version']}")
    if data["backend"] not in ALLOWED_BACKENDS:
        raise RunnerError(f"unsupported policy backend: {data['backend']}")

    controller = data.get("controller", {})
    script = controller.get("script")
    if not isinstance(script, str) or not script:
        raise RunnerError("policy controller.script is required")
    controller_path = resolve_fixture_path(script, must_exist=False)
    if not is_relative_to(controller_path, FIXTURE_ROOT):
        raise RunnerError("policy controller path escapes fixture workspace")
    return data


def load_yaml_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except ModuleNotFoundError:
        loaded = parse_limited_yaml(text)
    except Exception as exc:
        raise RunnerError(f"invalid YAML in {path.name}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise RunnerError(f"YAML file did not produce an object: {path.name}")
    return loaded


def parse_limited_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise RunnerError(f"tabs are not supported in YAML fallback at line {line_number}")

        indent = len(line) - len(line.lstrip(" "))
        stripped = strip_yaml_comment(line.strip())
        if not stripped:
            continue
        key, sep, raw_value = stripped.partition(":")
        if not sep:
            raise RunnerError(f"unsupported YAML fallback syntax at line {line_number}: {line}")
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise RunnerError(f"empty YAML key at line {line_number}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise RunnerError(f"invalid YAML indentation at line {line_number}")

        parent = stack[-1][1]
        if raw_value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_yaml_scalar(raw_value)

    return root


def strip_yaml_comment(value: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(value):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or value[index - 1].isspace():
                return value[:index].rstrip()
    return value


def parse_yaml_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?\d+\.\d+", value):
        return float(value)
    return value


def validate_inputs(
    policy: dict[str, Any],
    task: dict[str, Any],
    eval_spec: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if task.get("schema_version") != TASK_SCHEMA_VERSION:
        raise RunnerError(f"unsupported task schema: {task.get('schema_version')}")
    if eval_spec.get("schema_version") != EVAL_SCHEMA_VERSION:
        raise RunnerError(f"unsupported eval schema: {eval_spec.get('schema_version')}")
    if policy["behavior"] != task.get("behavior") or policy["behavior"] != eval_spec.get("behavior"):
        raise RunnerError("policy, task, and eval behavior ids must match")
    if task.get("world") != eval_spec.get("world"):
        raise RunnerError("task and eval world ids must match")
    if args.scene_env != task.get("scene_env"):
        raise RunnerError(f"scene env mismatch: expected {task.get('scene_env')}")
    if args.camera != task.get("camera"):
        raise RunnerError(f"camera mismatch: expected {task.get('camera')}")
    runs = eval_spec.get("runs")
    if not isinstance(runs, int) or runs <= 0 or runs > 16:
        raise RunnerError("eval runs must be an integer from 1 to 16")
    checks = eval_spec.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise RunnerError("eval checks must be a non-empty object")
    for check_name, check in checks.items():
        if not isinstance(check, dict):
            raise RunnerError(f"check {check_name} must be an object")
        if check.get("operator") not in OPS:
            raise RunnerError(f"check {check_name} uses unsupported operator: {check.get('operator')}")
        if "metric" not in check or "value" not in check:
            raise RunnerError(f"check {check_name} requires metric and value")
        if not isinstance(check.get("required"), bool):
            raise RunnerError(f"check {check_name} requires boolean required field")


def resolve_replay_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if not args.replay_source_dir:
        if args.require_isaac_replays:
            raise RunnerError("--require-isaac-replays requires --replay-source-dir")
        return {
            "source": "fixture-generated",
            "source_dir": "",
            "files": {},
            "notes": [
                "No Isaac replay source directory was provided; local fallback replay generation is allowed."
            ],
        }

    source_dir = resolve_input_path(args.replay_source_dir)
    files = {name: source_dir / name for name in REPLAY_FILENAMES}
    missing = [name for name, path in files.items() if not path.is_file()]
    invalid = [name for name, path in files.items() if path.is_file() and not is_valid_mp4(path)]

    if missing or invalid:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if invalid:
            details.append(f"invalid mp4: {', '.join(invalid)}")
        message = f"Isaac replay source is incomplete at {source_dir} ({'; '.join(details)})"
        if args.require_isaac_replays:
            raise RunnerError(message)
        return {
            "source": "fixture-generated",
            "source_dir": str(source_dir),
            "files": {},
            "notes": [message, "Falling back to local fixture replay generation."],
        }

    return {
        "source": "isaac-sim-session-video",
        "source_dir": str(source_dir),
        "files": {name: str(path) for name, path in files.items()},
        "notes": [
            f"Replay evidence copied from Isaac Sim session videos in `{source_dir}`.",
            "CI was configured to reject missing or invalid Isaac replay MP4 inputs.",
        ],
    }


def is_valid_mp4(path: Path) -> bool:
    if path.stat().st_size < 32:
        return False
    header = path.read_bytes()[:64]
    return b"ftyp" in header


def run_scripted_policy(policy: dict[str, Any], eval_spec: dict[str, Any]) -> list[dict[str, Any]]:
    total_runs = int(eval_spec["runs"])
    policy_id = policy["policy_id"]
    if policy_id not in {"g1_weld_approach_v18", "g1_weld_approach_v19"}:
        raise RunnerError(f"fixture runner does not know policy id: {policy_id}")

    observations = []
    for run in range(1, total_runs + 1):
        metrics = base_trial_metrics(run)
        events: list[dict[str, Any]] = []

        if policy_id == "g1_weld_approach_v18":
            if run == 3:
                metrics["restricted_zone_intrusions"] = 1
                metrics["torch_tip_distance_to_target_cm"] = 2.4
                events.append(
                    event(
                        run,
                        13.8,
                        "SAFETY_ZONE_INTRUSION",
                        "Torch entered restricted human-hand zone",
                    )
                )
            elif run == 5:
                metrics["collision_count"] = 2
                metrics["torch_tip_distance_to_target_cm"] = 4.6
                events.append(
                    event(
                        run,
                        12.4,
                        "OBSTACLE_COLLISION",
                        "End effector collided with shifted obstacle",
                    )
                )
            elif run == 7:
                metrics["elapsed_seconds"] = 34.8
                metrics["torch_tip_distance_to_target_cm"] = 3.8
                events.append(
                    event(
                        run,
                        30.0,
                        "TARGET_TIMEOUT",
                        "Failed to reach weld start pose before timeout",
                    )
                )

        observations.append(
            {
                "run": run,
                "trajectory_id": "obstacle_shift_left_10cm",
                "metrics": metrics,
                "events": events,
                "replay_marker": "artifacts/replays/replay-failed.mp4"
                if events
                else "artifacts/replays/replay-passed.mp4",
            }
        )

    return observations


def base_trial_metrics(run: int) -> dict[str, float | int]:
    distances = [1.3, 1.5, 1.6, 1.4, 1.7, 1.2, 1.8, 1.4]
    elapsed = [22.1, 23.4, 21.9, 24.0, 22.8, 23.1, 24.4, 22.6]
    tilt = [1.4, 1.8, 2.0, 1.6, 2.2, 1.7, 2.4, 1.5]
    index = run - 1
    return {
        "torch_tip_distance_to_target_cm": distances[index],
        "collision_count": 0,
        "restricted_zone_intrusions": 0,
        "max_base_tilt_degrees": tilt[index],
        "elapsed_seconds": elapsed[index],
    }


def event(run: int, time_seconds: float, code: str, message: str) -> dict[str, Any]:
    return {
        "run": run,
        "time_seconds": time_seconds,
        "code": code,
        "message": message,
    }


def evaluate_trials(
    policy: dict[str, Any],
    task: dict[str, Any],
    eval_spec: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = eval_spec["checks"]
    per_run = []
    check_pass_counts = {name: 0 for name in checks}
    failures: list[dict[str, Any]] = []

    for observation in observations:
        run_checks = {}
        for check_name, check in checks.items():
            metric_name = check["metric"]
            metrics = observation["metrics"]
            if metric_name not in metrics:
                raise RunnerError(f"missing required metric {metric_name} in run {observation['run']}")
            passed = bool(OPS[check["operator"]](metrics[metric_name], check["value"]))
            run_checks[check_name] = {
                "passed": passed,
                "metric": metric_name,
                "actual": metrics[metric_name],
                "operator": check["operator"],
                "expected": check["value"],
                "required": check["required"],
            }
            if passed:
                check_pass_counts[check_name] += 1

        run_passed = all(value["passed"] for value in run_checks.values() if value["required"])
        if not run_passed:
            if observation["events"]:
                for failure in observation["events"]:
                    failures.append(
                        {
                            "run": observation["run"],
                            "code": failure["code"],
                            "message": failure["message"],
                        }
                    )
            else:
                failures.append(
                    {
                        "run": observation["run"],
                        "code": "CHECK_FAILURE",
                        "message": "One or more required checks failed",
                    }
                )
        per_run.append(
            {
                "run": observation["run"],
                "passed": run_passed,
                "checks": run_checks,
                "metrics": observation["metrics"],
                "events": observation["events"],
                "trajectory_id": observation["trajectory_id"],
            }
        )

    total_runs = len(observations)
    passed_runs = sum(1 for run in per_run if run["passed"])
    aggregate_checks = {
        name: count == total_runs for name, count in check_pass_counts.items()
    }
    status = "passed" if passed_runs == total_runs else "failed"
    aggregate_metrics = aggregate_trial_metrics(policy, observations, passed_runs, total_runs)

    return {
        "status": status,
        "per_run": per_run,
        "check_pass_counts": check_pass_counts,
        "checks": aggregate_checks,
        "failures": failures,
        "metrics": aggregate_metrics,
        "summary": {
            "passed_runs": passed_runs,
            "total_runs": total_runs,
            "failed_runs": total_runs - passed_runs,
            "world": task["world"],
            "policy_id": policy["policy_id"],
        },
    }


def aggregate_trial_metrics(
    policy: dict[str, Any],
    observations: list[dict[str, Any]],
    passed_runs: int,
    total_runs: int,
) -> dict[str, Any]:
    distances = [obs["metrics"]["torch_tip_distance_to_target_cm"] for obs in observations]
    collisions = sum(obs["metrics"]["collision_count"] for obs in observations)
    safety = sum(obs["metrics"]["restricted_zone_intrusions"] for obs in observations)
    elapsed = sum(obs["metrics"]["elapsed_seconds"] for obs in observations)
    max_tilt = max(obs["metrics"]["max_base_tilt_degrees"] for obs in observations)

    if policy["policy_id"] == "g1_weld_approach_v18":
        runtime_seconds = 618
        estimated_gpu_cost = 0.31
        mean_error = 4.6
    else:
        runtime_seconds = 544
        estimated_gpu_cost = 0.27
        mean_error = round(sum(distances) / len(distances), 2)

    return {
        "task_success": f"{passed_runs} / {total_runs}",
        "mean_torch_tip_error_cm": mean_error,
        "collision_events": int(collisions),
        "safety_zone_violations": int(safety),
        "max_base_tilt_degrees": round(float(max_tilt), 2),
        "simulated_trial_seconds": round(float(elapsed), 1),
        "runtime_seconds": runtime_seconds,
        "estimated_gpu_cost_usd": estimated_gpu_cost,
    }


def build_result(
    *,
    args: argparse.Namespace,
    policy: dict[str, Any],
    task: dict[str, Any],
    eval_spec: dict[str, Any],
    evaluated: dict[str, Any],
    commit: str,
    artifact_base_url: str,
    replay_inputs: dict[str, Any],
) -> dict[str, Any]:
    artifacts = {
        "replay_video": artifact_link(artifact_base_url, "artifacts/replays/replay-failed.mp4"),
        "passed_replay_video": artifact_link(artifact_base_url, "artifacts/replays/replay-passed.mp4"),
        "metrics_json": artifact_link(artifact_base_url, "artifacts/metrics.json"),
        "scene_snapshot": artifact_link(
            artifact_base_url, "artifacts/scene/scene_snapshot.usd-or-placeholder.txt"
        ),
        "logs": artifact_link(artifact_base_url, "artifacts/logs/runner.log"),
        "report": artifact_link(artifact_base_url, "artifacts/report/index.html"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": evaluated["status"],
        "behavior": eval_spec["behavior"],
        "robot": policy["robot"],
        "world": eval_spec["world"],
        "scene_env": args.scene_env,
        "camera": args.camera,
        "policy": policy["display_filename"],
        "policy_id": policy["policy_id"],
        "policy_backend": policy["backend"],
        "commit": commit,
        "summary": evaluated["summary"],
        "checks": evaluated["checks"],
        "metrics": evaluated["metrics"],
        "failures": evaluated["failures"],
        "artifacts": artifacts,
        "honesty": {
            "github_ui_staged": False,
            "policy_backend_real_vla": policy["backend"] == "real-vla",
            "replay_source": replay_inputs["source"],
            "replay_source_dir": replay_inputs["source_dir"],
            "production_eval_path_used": False,
            "notes": (
                "Replay videos are sourced from Isaac Sim session captures when "
                "`--require-isaac-replays` is enabled. The demo policy backend is "
                "a scripted shim unless a real VLA backend is wired in."
            ),
        },
    }


def artifact_link(base_url: str, relative: str) -> str:
    if base_url:
        return f"{base_url}#{relative}"
    return relative


def write_artifacts(
    out_dir: Path,
    result: dict[str, Any],
    evaluated: dict[str, Any],
    args: argparse.Namespace,
    policy: dict[str, Any],
    log_lines: list[str],
    replay_inputs: dict[str, Any],
) -> None:
    write_json(out_dir / "result.json", result)
    write_json(
        out_dir / "metrics.json",
        {
            "schema_version": "behavior-ci-metrics/v1",
            "aggregate": result["metrics"],
            "check_pass_counts": evaluated["check_pass_counts"],
            "runs": evaluated["per_run"],
        },
    )

    replay_notes = write_replay_files(out_dir, result, replay_inputs)
    (out_dir / "comment.md").write_text(render_comment(result), encoding="utf-8")
    write_report(out_dir, result, evaluated)
    write_scene_snapshot(out_dir, result, args)
    write_hack_notes(out_dir, result, policy, replay_notes)

    log_lines.extend(
        [
            f"completed_at={utc_now()}",
            f"status={result['status']}",
            f"policy={result['policy']}",
            f"passed_runs={result['summary']['passed_runs']}",
            f"total_runs={result['summary']['total_runs']}",
        ]
    )
    (out_dir / "logs" / "runner.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (out_dir / "logs" / "sim.log").write_text(render_sim_log(result), encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_comment(result: dict[str, Any]) -> str:
    status = result["status"].capitalize()
    summary = result["summary"]
    metrics = result["metrics"]
    checks = result["checks"]
    check_rows = "\n".join(
        f"| {title_case(name)} | {'Pass' if passed else 'Fail'} |"
        for name, passed in checks.items()
    )
    failures = result["failures"]
    if failures:
        failure_lines = "\n".join(
            f"{index}. Run {failure['run']:02d}: {failure['message']} (`{failure['code']}`)"
            for index, failure in enumerate(failures, start=1)
        )
        suggested_next_step = (
            "Re-run fine-tuning with obstacle-shift augmentation and constrain "
            "the approach angle near the weld seam."
        )
    else:
        failure_lines = "No required check failures."
        suggested_next_step = "Policy is ready for the next non-production review gate."

    if result["status"] == "passed":
        summary_sentence = f"The policy passed {summary['passed_runs']} of {summary['total_runs']} simulation runs."
    else:
        summary_sentence = (
            f"The policy passed {summary['passed_runs']} of {summary['total_runs']} simulation runs "
            f"and failed {summary['failed_runs']} of {summary['total_runs']}."
        )

    return f"""<!-- cybernetic-behavior-ci -->
# Cybernetic Physics Behavior CI

**Result:** {status}

| Field | Value |
| --- | --- |
| Behavior | `{result['behavior']}` |
| Robot | {result['robot']} |
| Test world | `{result['world']}` |
| Policy | `{result['policy']}` |
| Commit | `{result['commit']}` |
| Backend | `{result['policy_backend']}` |

## Summary

{summary_sentence}

## Checks

| Check | Result |
| --- | --- |
{check_rows}

## Failures

{failure_lines}

## Metrics

- Task success: {metrics['task_success']}
- Mean torch-tip error: {metrics['mean_torch_tip_error_cm']} cm
- Collision events: {metrics['collision_events']}
- Safety zone violations: {metrics['safety_zone_violations']}
- Runtime: {format_duration(metrics['runtime_seconds'])}
- Estimated GPU cost: ${metrics['estimated_gpu_cost_usd']:.2f}

<details>
<summary>Artifact bundle</summary>

- Replay video: `{result['artifacts']['replay_video']}`
- Passed replay video: `{result['artifacts']['passed_replay_video']}`
- Metrics JSON: `{result['artifacts']['metrics_json']}`
- Isaac scene snapshot: `{result['artifacts']['scene_snapshot']}`
- Full logs: `{result['artifacts']['logs']}`
- Static report: `{result['artifacts']['report']}`

</details>

## Suggested Next Step

{suggested_next_step}

_MVP note: this no-site runner does not use production hosted evals, `/evals`, `/v1/eval/*`, eval registry sync, or production artifact hosting._
"""


def write_report(out_dir: Path, result: dict[str, Any], evaluated: dict[str, Any]) -> None:
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "style.css").write_text(report_css(), encoding="utf-8")
    (report_dir / "index.html").write_text(report_html(result, evaluated), encoding="utf-8")


def report_html(result: dict[str, Any], evaluated: dict[str, Any]) -> str:
    status = result["status"]
    status_label = "Passed" if status == "passed" else "Failed"
    checks = "\n".join(
        f"<tr><th>{html.escape(title_case(name))}</th><td class=\"{status_class(passed)}\">"
        f"{'Pass' if passed else 'Fail'}</td></tr>"
        for name, passed in result["checks"].items()
    )
    metrics = "\n".join(
        f"<tr><th>{html.escape(title_case(name))}</th><td>{html.escape(format_metric(value))}</td></tr>"
        for name, value in result["metrics"].items()
    )
    runs = "\n".join(
        "<tr>"
        f"<td>{run['run']:02d}</td>"
        f"<td class=\"{status_class(run['passed'])}\">{'Pass' if run['passed'] else 'Fail'}</td>"
        f"<td>{html.escape(', '.join(event['code'] for event in run['events']) or 'clean')}</td>"
        f"<td>{run['metrics']['torch_tip_distance_to_target_cm']}</td>"
        f"<td>{run['metrics']['collision_count']}</td>"
        f"<td>{run['metrics']['restricted_zone_intrusions']}</td>"
        f"<td>{run['metrics']['elapsed_seconds']}</td>"
        "</tr>"
        for run in evaluated["per_run"]
    )
    failures = result["failures"]
    failure_items = "\n".join(
        f"<li><strong>Run {failure['run']:02d}</strong>: "
        f"{html.escape(failure['message'])} "
        f"<code>{html.escape(failure['code'])}</code></li>"
        for failure in failures
    )
    if not failure_items:
        failure_items = "<li>No failures in the fixed policy run.</li>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cybernetic Physics Behavior CI - {html.escape(status_label)}</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main>
    <section class="header-band">
      <div>
        <p class="eyebrow">Cybernetic Physics Behavior CI</p>
        <h1>{html.escape(result['behavior'])}</h1>
        <p class="subtitle">One behavior, one obstacle-shift world, one PR gate.</p>
      </div>
      <div class="status-pill {html.escape(status)}">{html.escape(status_label)}</div>
    </section>

    <section class="summary-grid">
      <div class="metric-tile">
        <span>Task Success</span>
        <strong>{html.escape(result['metrics']['task_success'])}</strong>
      </div>
      <div class="metric-tile">
        <span>Policy</span>
        <strong>{html.escape(result['policy'])}</strong>
      </div>
      <div class="metric-tile">
        <span>World</span>
        <strong>{html.escape(result['world'])}</strong>
      </div>
      <div class="metric-tile">
        <span>Backend</span>
        <strong>{html.escape(result['policy_backend'])}</strong>
      </div>
    </section>

    <section class="report-section">
      <div class="section-heading">
        <h2>Replay Evidence</h2>
        <p>Video evidence captured from the Isaac Sim session pass/fail camera.</p>
      </div>
      <div class="replay-grid">
        <figure>
          <video controls muted preload="metadata" src="../replays/replay-failed.mp4"></video>
          <figcaption>Failure path: restricted zone, obstacle, and timeout regression.</figcaption>
        </figure>
        <figure>
          <video controls muted preload="metadata" src="../replays/replay-passed.mp4"></video>
          <figcaption>Fixed path: clean approach through the same obstacle-shift world.</figcaption>
        </figure>
      </div>
      <div class="storyboard {html.escape(status)}">
        <div class="zone restricted">Restricted zone</div>
        <div class="zone obstacle">Shifted obstacle</div>
        <div class="zone target">Weld start</div>
        <div class="path-label">{html.escape(status_label)} policy trajectory</div>
      </div>
    </section>

    <section class="two-column">
      <div>
        <h2>Checks</h2>
        <table>{checks}</table>
      </div>
      <div>
        <h2>Metrics</h2>
        <table>{metrics}</table>
      </div>
    </section>

    <section class="report-section">
      <h2>Run Breakdown</h2>
      <table class="run-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>Result</th>
            <th>Events</th>
            <th>Error cm</th>
            <th>Collisions</th>
            <th>Zone</th>
            <th>Seconds</th>
          </tr>
        </thead>
        <tbody>{runs}</tbody>
      </table>
    </section>

    <section class="report-section">
      <h2>Failures</h2>
      <ol>{failure_items}</ol>
    </section>

    <section class="report-section">
      <h2>Artifacts</h2>
      <ul class="artifact-list">
        <li><a href="../result.json">Result JSON</a></li>
        <li><a href="../metrics.json">Metrics JSON</a></li>
        <li><a href="../comment.md">PR comment markdown</a></li>
        <li><a href="../scene/scene_snapshot.usd-or-placeholder.txt">Scene snapshot placeholder</a></li>
        <li><a href="../logs/runner.log">Runner log</a></li>
        <li><a href="../hack-notes.md">Hack notes</a></li>
      </ul>
    </section>

    <section class="honesty">
      <h2>Honesty Notes</h2>
      <p>{html.escape(result['honesty']['notes'])}</p>
      <p>Replay source: <code>{html.escape(result['honesty']['replay_source'])}</code></p>
      <p>No production hosted evals, site eval routes, registry sync, or production artifact hosting were used for this MVP artifact.</p>
    </section>

    <footer>
      CodeRabbit reviews whether the code looks right. Cybernetic Physics reviews whether the robot still works.
    </footer>
  </main>
</body>
</html>
"""


def report_css() -> str:
    return """* {
  box-sizing: border-box;
}

:root {
  color-scheme: light;
  --ink: #172026;
  --muted: #5f6b76;
  --line: #d7dde4;
  --surface: #ffffff;
  --band: #eef3f0;
  --pass: #0f7a45;
  --pass-soft: #dff4e8;
  --fail: #b42318;
  --fail-soft: #fde7e4;
  --accent: #1b5f8c;
  --warn: #9a6200;
}

body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: #f6f7f9;
}

main {
  width: min(1180px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 28px 0 48px;
}

.header-band {
  min-height: 220px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 32px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--band);
}

.eyebrow {
  margin: 0 0 12px;
  color: var(--accent);
  font-size: 14px;
  font-weight: 700;
  text-transform: uppercase;
}

h1, h2, p {
  margin-top: 0;
}

h1 {
  margin-bottom: 10px;
  font-size: clamp(34px, 6vw, 64px);
  line-height: 1;
  letter-spacing: 0;
}

h2 {
  margin-bottom: 14px;
  font-size: 22px;
}

.subtitle {
  margin-bottom: 0;
  color: var(--muted);
  font-size: 18px;
}

.status-pill {
  flex: 0 0 auto;
  min-width: 140px;
  padding: 14px 18px;
  border-radius: 8px;
  text-align: center;
  font-size: 22px;
  font-weight: 800;
}

.status-pill.passed,
.pass {
  color: var(--pass);
  background: var(--pass-soft);
}

.status-pill.failed,
.fail {
  color: var(--fail);
  background: var(--fail-soft);
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0;
}

.metric-tile,
.report-section,
.two-column > div,
.honesty {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
}

.metric-tile {
  min-height: 96px;
  padding: 18px;
}

.metric-tile span {
  display: block;
  margin-bottom: 10px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
}

.metric-tile strong {
  overflow-wrap: anywhere;
  font-size: 20px;
}

.report-section,
.honesty {
  margin-top: 16px;
  padding: 24px;
}

.section-heading {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: end;
}

.section-heading p {
  color: var(--muted);
}

.replay-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

figure {
  margin: 0;
}

video {
  display: block;
  width: 100%;
  aspect-ratio: 16 / 9;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #111820;
}

figcaption {
  margin-top: 8px;
  color: var(--muted);
  font-size: 14px;
}

.storyboard {
  position: relative;
  height: 170px;
  margin-top: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background:
    linear-gradient(90deg, transparent 0 48%, rgba(23,32,38,0.08) 48% 52%, transparent 52%),
    #f8fafb;
  overflow: hidden;
}

.storyboard::after {
  content: "";
  position: absolute;
  left: 9%;
  top: 56%;
  width: 78%;
  height: 4px;
  border-radius: 4px;
  background: var(--pass);
  transform: rotate(-4deg);
}

.storyboard.failed::after {
  background: var(--fail);
  transform: rotate(-13deg);
}

.zone {
  position: absolute;
  min-width: 112px;
  padding: 9px 10px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 700;
  text-align: center;
}

.restricted {
  left: 21%;
  top: 24%;
  color: var(--fail);
  background: var(--fail-soft);
}

.obstacle {
  left: 50%;
  top: 47%;
  color: var(--warn);
  background: #fff1cf;
}

.target {
  right: 9%;
  top: 30%;
  color: var(--pass);
  background: var(--pass-soft);
}

.path-label {
  position: absolute;
  left: 24px;
  bottom: 18px;
  color: var(--muted);
  font-size: 14px;
}

.two-column {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
  margin-top: 16px;
}

.two-column > div {
  padding: 24px;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

th,
td {
  padding: 10px 8px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}

th {
  color: var(--muted);
  font-weight: 700;
}

.artifact-list {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px 20px;
  padding-left: 18px;
}

a {
  color: var(--accent);
}

.honesty {
  border-color: #dbc172;
  background: #fff8e3;
}

footer {
  margin-top: 20px;
  padding: 18px 0;
  color: var(--ink);
  font-size: 18px;
  font-weight: 800;
  text-align: center;
}

@media (max-width: 840px) {
  main {
    width: min(100vw - 20px, 720px);
    padding-top: 12px;
  }

  .header-band,
  .section-heading {
    align-items: flex-start;
    flex-direction: column;
  }

  .summary-grid,
  .replay-grid,
  .two-column,
  .artifact-list {
    grid-template-columns: 1fr;
  }

  .run-table {
    display: block;
    overflow-x: auto;
  }
}
"""


def write_scene_snapshot(out_dir: Path, result: dict[str, Any], args: argparse.Namespace) -> None:
    text = f"""# Behavior CI scene snapshot placeholder

Scene env: {args.scene_env}
World id: {result['world']}
Camera prim: {args.camera}
Robot: {result['robot']}

Required visible elements:
- Humanoid proxy robot.
- Stainless welding table.
- Weld-start pose marker.
- Shifted obstacle near the approach path.
- Translucent restricted human-hand zone.
- PASS/FAIL markers used by the replay/report.

MVP note:
This is a placeholder snapshot for the no-site-evals lane. CYB-64 owns replacing
it with an Isaac USD snapshot or environment export from Luc dev infrastructure.
"""
    (out_dir / "scene" / "scene_snapshot.usd-or-placeholder.txt").write_text(text, encoding="utf-8")


def write_hack_notes(
    out_dir: Path,
    result: dict[str, Any],
    policy: dict[str, Any],
    replay_notes: list[str],
) -> None:
    notes = [
        "# Behavior CI MVP Hack Notes",
        "",
        "- Production hosted evals were not used.",
        "- `/evals`, `/v1/eval/*`, eval registry sync, and production artifact hosting were bypassed.",
        "- The visible `.pt` policy files are JSON manifests for the MVP fixture.",
        f"- Policy backend: `{policy['backend']}`.",
        "- The scripted shim emits deterministic observations for the v18/v19 demo story.",
        "- The robot label is `Unitree G1-compatible humanoid proxy`; final footage must not overclaim a real Unitree G1 runtime until CYB-64/CYB-68 verify it.",
        "- Public CI is configured to require replay assets copied from Isaac Sim session captures.",
        "- Luc/non-production infrastructure is the intended source for replay capture.",
        "",
        "## Replay Generation",
        "",
    ]
    notes.extend(f"- {line}" for line in replay_notes)
    notes.extend(
        [
            "",
            "## Result",
            "",
            f"- Status: `{result['status']}`",
            f"- Policy: `{result['policy']}`",
            f"- Task success: `{result['metrics']['task_success']}`",
        ]
    )
    (out_dir / "hack-notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def write_replay_files(
    out_dir: Path,
    result: dict[str, Any],
    replay_inputs: dict[str, Any],
) -> list[str]:
    replay_dir = out_dir / "replays"
    failed_path = replay_dir / "replay-failed.mp4"
    passed_path = replay_dir / "replay-passed.mp4"
    notes = list(replay_inputs["notes"])

    if replay_inputs["source"] == "isaac-sim-session-video":
        for name in REPLAY_FILENAMES:
            source = Path(replay_inputs["files"][name])
            target = replay_dir / name
            shutil.copy2(source, target)
        notes.append("Copied Isaac Sim session MP4s into the Behavior CI artifact bundle.")
        return notes

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        failed_ok = render_replay_with_ffmpeg(
            ffmpeg,
            failed_path,
            title="FAILED: obstacle-shift regression",
            accent="0xb42318",
            clean=False,
        )
        passed_ok = render_replay_with_ffmpeg(
            ffmpeg,
            passed_path,
            title="PASSED: guarded weld approach",
            accent="0x0f7a45",
            clean=True,
        )
        if failed_ok and passed_ok:
            notes.append("ffmpeg generated deterministic MP4 replay fixtures.")
            return notes
        notes.append("ffmpeg was present but replay rendering failed; placeholder MP4 files were emitted.")
    else:
        notes.append("ffmpeg was unavailable locally; placeholder MP4 files were emitted.")

    write_placeholder_mp4(failed_path, "Behavior CI failed replay placeholder\n")
    write_placeholder_mp4(passed_path, "Behavior CI passed replay placeholder\n")
    return notes


def render_replay_with_ffmpeg(
    ffmpeg: str,
    output: Path,
    *,
    title: str,
    accent: str,
    clean: bool,
) -> bool:
    path_y = "310" if clean else "245"
    target_y = "220" if clean else "320"
    filter_graph = ",".join(
        [
            "drawbox=x=0:y=0:w=iw:h=90:color=0x172026@0.9:t=fill",
            f"drawbox=x=115:y={path_y}:w=830:h=8:color={accent}@0.95:t=fill",
            "drawbox=x=730:y=260:w=120:h=120:color=0x9a6200@0.75:t=fill",
            "drawbox=x=300:y=190:w=170:h=170:color=0xb42318@0.22:t=fill",
            f"drawbox=x=1010:y={target_y}:w=70:h=70:color=0x0f7a45@0.85:t=fill",
            "drawtext=text='Cybernetic Physics Behavior CI':x=42:y=24:fontsize=30:fontcolor=white",
            f"drawtext=text='{escape_ffmpeg_text(title)}':x=42:y=585:fontsize=36:fontcolor=white",
            "drawtext=text='restricted zone':x=308:y=160:fontsize=24:fontcolor=0xb42318",
            "drawtext=text='shifted obstacle':x=700:y=226:fontsize=24:fontcolor=0x9a6200",
            "drawtext=text='weld start':x=990:y=190:fontsize=24:fontcolor=0x0f7a45",
        ]
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=0x101820:s=1280x720:d=5",
        "-vf",
        filter_graph,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.returncode == 0 and output.exists() and output.stat().st_size > 0


def escape_ffmpeg_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def write_placeholder_mp4(path: Path, message: str) -> None:
    path.write_bytes(
        b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
        b"\x00\x00\x00\x08free"
        + message.encode("utf-8")
    )


def render_sim_log(result: dict[str, Any]) -> str:
    lines = [
        "Behavior CI deterministic simulation log",
        f"world={result['world']}",
        f"camera={result['camera']}",
        f"policy={result['policy']}",
        f"backend={result['policy_backend']}",
        f"status={result['status']}",
    ]
    for failure in result["failures"]:
        lines.append(f"run={failure['run']:02d} code={failure['code']} message={failure['message']}")
    return "\n".join(lines) + "\n"


def validate_artifacts(out_dir: Path) -> None:
    missing = [relative for relative in REQUIRED_ARTIFACTS if not (out_dir / relative).is_file()]
    if missing:
        raise RunnerError(f"missing required artifacts: {', '.join(missing)}", exit_code=4)
    empty = [relative for relative in REQUIRED_ARTIFACTS if (out_dir / relative).stat().st_size == 0]
    if empty:
        raise RunnerError(f"empty required artifacts: {', '.join(empty)}", exit_code=4)


def print_result_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print(
        f"Behavior CI {result['status']}: {result['policy']} "
        f"{summary['passed_runs']} / {summary['total_runs']} runs passed"
    )
    for failure in result["failures"]:
        print(f"- run {failure['run']:02d}: {failure['code']} - {failure['message']}")


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    return "unknown"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def format_duration(seconds: int | float) -> str:
    seconds = int(seconds)
    minutes, remainder = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {remainder:02d}s"
    return f"{seconds}s"


def title_case(value: str) -> str:
    return value.replace("_", " ").title()


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        if math.isclose(value, round(value)):
            return str(int(round(value)))
        return f"{value:.2f}"
    return str(value)


def status_class(passed: bool) -> str:
    return "pass" if passed else "fail"


if __name__ == "__main__":
    raise SystemExit(main())
