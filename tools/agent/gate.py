# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Acceptance gate for an agent change, run from the repository root.

usage: python3 tools/agent/gate.py <base_ref> [--lint-base REF] [--scope] [--build]
                                   [--sass DIR] [--tests [PATH ...]] [--junit-base XML]

The lint and scope checks run from the copy of tools/lint at base_ref, so editing them
in the change has no effect. Each selected step must pass:
  lint    halops_lint.py --diff base_ref (working tree, untracked files included); with
          --lint-base, --diff REF as well, e.g. the merge base of a feature branch
  tree    halops_lint.py --all minus the baseline.json of base_ref, so a change cannot
          create findings elsewhere (for example a name clash after removing a namespace)
  scope   check_scope.py base_ref, for lint-fix changes
  build   PYTHON=python3 bash build.sh
  sass    check_sass.py DIR halops; DIR holds the extension built from base_ref
  tests   pytest; with --junit-base, every test that passed there must pass here
Test runs take a lock shared by all worktrees of the clone because some tests use every GPU.
"""

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def step(name, cmd, **kw):
    print(f"== {name}: {' '.join(cmd)}", flush=True)
    code = subprocess.run(cmd, **kw).returncode
    print(f"== {name}: {'ok' if code == 0 else f'FAILED (exit {code})'}", flush=True)
    return code == 0


def outcomes(path):
    out = {}
    for case in ET.parse(path).iter("testcase"):
        tags = {c.tag for c in case}
        state = (
            "failed"
            if tags & {"failure", "error"}
            else "skipped" if "skipped" in tags else "passed"
        )
        out[f"{case.get('classname')}::{case.get('name')}"] = state
    return out


def tests(args, root):
    xml = os.path.join(tempfile.mkdtemp(prefix="halops-gate-"), "pytest.xml")
    lock = open(
        os.path.join(
            git("rev-parse", "--path-format=absolute", "--git-common-dir"),
            "halops-gate-gpu.lock",
        ),
        "w",
    )
    print("== tests: waiting for the GPU lock", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        ok = step(
            "tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={xml}",
                *(args.tests or ["tests"]),
            ],
            cwd=root,
        )
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
    print(f"   junit: {xml}")
    if not args.junit_base or not os.path.exists(xml):
        return ok
    before, after = outcomes(args.junit_base), outcomes(xml)
    lost = sorted(
        k
        for k, v in before.items()
        if v == "passed" and k in after and after[k] != "passed"
    )
    new_fail = sorted(k for k, v in after.items() if v == "failed" and k not in before)
    for k in lost:
        print(f"   regressed: {k} ({after[k]})")
    for k in new_fail:
        print(f"   new test failing: {k}")
    same_fail = sorted(
        k for k, v in after.items() if v == "failed" and before.get(k) == "failed"
    )
    if same_fail:
        print(f"   failing on base as well ({len(same_fail)}): {', '.join(same_fail)}")
    print(
        f"== tests vs {args.junit_base}: {'ok' if not lost and not new_fail else 'FAILED'}",
        flush=True,
    )
    return not lost and not new_fail


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base")
    ap.add_argument("--scope", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--lint-base", metavar="REF")
    ap.add_argument("--jobs", type=int, default=32, help="MAX_JOBS for the build")
    ap.add_argument("--sass", metavar="DIR")
    ap.add_argument("--tests", nargs="*")
    ap.add_argument("--junit-base", metavar="XML")
    args = ap.parse_args()
    root = git("rev-parse", "--show-toplevel")
    tools = tempfile.mkdtemp(prefix="halops-gate-tools-")
    archive = subprocess.run(
        ["git", "-C", root, "archive", args.base, "tools/lint"],
        capture_output=True,
        check=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", tools], input=archive, check=True)
    lint = os.path.join(tools, "tools/lint")
    results = {}
    try:
        results["lint"] = step(
            "lint",
            [sys.executable, "-I", f"{lint}/halops_lint.py", "--diff", args.base],
            cwd=root,
        )
        if args.lint_base:
            results["lint-base"] = step(
                "lint-base",
                [
                    sys.executable,
                    "-I",
                    f"{lint}/halops_lint.py",
                    "--diff",
                    args.lint_base,
                ],
                cwd=root,
            )
        results["tree"] = step(
            "tree",
            [
                sys.executable,
                "-I",
                f"{lint}/halops_lint.py",
                "--all",
                "--baseline",
                f"{lint}/baseline.json",
            ],
            cwd=root,
        )
        if args.scope:
            results["scope"] = step(
                "scope",
                [sys.executable, "-I", f"{lint}/check_scope.py", args.base],
                cwd=root,
            )
        if args.build:
            env = dict(os.environ, PYTHON="python3", MAX_JOBS=str(args.jobs))
            results["build"] = step("build", ["bash", "build.sh"], cwd=root, env=env)
        if args.sass and results.get("build", True):
            results["sass"] = step(
                "sass",
                [
                    sys.executable,
                    "-I",
                    f"{lint}/check_sass.py",
                    args.sass,
                    os.path.join(root, "halops"),
                ],
                cwd=root,
            )
        if args.tests is not None and results.get("build", True):
            results["tests"] = tests(args, root)
    finally:
        shutil.rmtree(tools, ignore_errors=True)
    print(
        "gate: "
        + ", ".join(f"{k} {'ok' if v else 'FAILED'}" for k, v in results.items())
    )
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
