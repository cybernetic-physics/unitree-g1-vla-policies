#!/usr/bin/env python3
"""Map changed policy files to the suite behaviors that pin them.

Used by the pull-request workflow to decide which behaviors deserve GPU time: a behavior
whose policy pointer this PR did not touch is not re-graded per commit, because the nightly
run already grades the whole suite. Reads the changed paths on stdin (one per line, as
``git diff --name-only`` prints them) and writes the matching behavior ids, comma separated.

Kept as a file rather than an inline heredoc so it can be unit tested and so the workflow
does not depend on shell quoting.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml


def changed_behavior_ids(suite: dict, changed: set[str]) -> list[str]:
    ids = []
    for behavior in suite.get("behaviors", []) or []:
        policy = behavior.get("policy")
        if policy and policy in changed:
            ids.append(behavior["id"])
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="behavior-ci-suite.yaml")
    args = parser.parse_args(argv)

    suite_path = pathlib.Path(args.suite)
    if not suite_path.is_file():
        print(f"suite manifest not found: {suite_path}", file=sys.stderr)
        return 2

    suite = yaml.safe_load(suite_path.read_text()) or {}
    changed = {line.strip() for line in sys.stdin.read().splitlines() if line.strip()}
    print(",".join(changed_behavior_ids(suite, changed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
