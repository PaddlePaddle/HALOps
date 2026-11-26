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
"""Claude Code hooks for agents that edit HALOps.

usage, from .claude/settings.json or skill frontmatter, with the hook input on stdin:
  python3 -I tools/agent/hooks.py pre [--role writer|brush]   # PreToolUse
  python3 -I tools/agent/hooks.py post                        # PostToolUse
  python3 -I tools/agent/hooks.py stop                        # Stop

pre   blocks (exit 2) edits to the lint tools, hooks, CI or pytest configuration, new
      `halops-lint: allow` exemptions, `git push` and `--no-verify`. In tests/ it blocks
      changed or removed lines with assertions, tolerances or parametrization; the brush
      role may not add such lines either.
post  lints the lines added to the edited file since the base and reports them (exit 2).
stop  lints every HALOps checkout edited in the session and keeps the agent working
      once (exit 2) while error-level findings remain.
The base is $HALOPS_BASE, else the merge base of HEAD and origin/main. The lint used is
the one next to this script, not the one in the checkout being edited.
"""

import argparse
import difflib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LINT = os.path.join(HERE, "..", "lint", "halops_lint.py")
sys.path.insert(0, os.path.join(HERE, "..", "lint"))
import check_scope  # noqa: E402

MARKER = "csrc/common/cpp_extensions.cc"
EXEMPT = "halops-lint: allow"
PROT = "|".join(re.escape(p) for p in check_scope.PROTECTED)
BASH_WRITE = re.compile(
    r"(^|\s)(sed\s.*-i|perl\s.*-i|tee|mv|cp|rm|truncate|install|ln|"
    r"git\s+(checkout|restore|rm|mv|apply|stash|reset))\b"
)
REDIRECT = re.compile(r">>?\s*['\"]?\S*(" + PROT + ")")


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def locate(path):
    """(checkout root, path relative to it) for a path inside a HALOps checkout."""
    d = os.path.dirname(os.path.abspath(path))
    while d != "/" and not os.path.isdir(d):
        d = os.path.dirname(d)
    top = run(["git", "rev-parse", "--show-toplevel"], d).stdout.strip()
    if not top or not os.path.exists(os.path.join(top, MARKER)):
        return None, None
    return top, os.path.relpath(os.path.abspath(path), top)


def base_ref(top):
    if os.environ.get("HALOPS_BASE"):
        return os.environ["HALOPS_BASE"]
    return (
        run(["git", "merge-base", "HEAD", "origin/main"], top).stdout.strip() or "HEAD"
    )


def deny(msg):
    print(msg, file=sys.stderr)
    return 2


def new_text(tool, inp, old):
    if tool == "Write":
        return inp.get("content", "")
    edits = inp.get("edits") or [inp]
    for e in edits:
        a, b = e.get("old_string", ""), e.get("new_string", "")
        old = old.replace(a, b) if e.get("replace_all") else old.replace(a, b, 1)
    return old


def pre(data, role):
    tool, inp = data.get("tool_name", ""), data.get("tool_input") or {}
    if tool == "Bash":
        cmd = inp.get("command", "")
        if re.search(r"--no-verify\b|\bgit\s+push\b", cmd):
            return deny(
                "commit with the hooks enabled and do not push; the reviewer pushes"
            )
        if EXEMPT in cmd:
            return deny(
                "exemptions are granted by maintainers; list the finding and the reason in your comment"
            )
        for seg in re.split(r"&&|\|\||[;|\n]", cmd):
            if (BASH_WRITE.search(seg) and re.search(PROT, seg)) or REDIRECT.search(
                seg
            ):
                return deny(f"read-only for agents: {', '.join(check_scope.PROTECTED)}")
        return 0
    path = inp.get("file_path") or inp.get("notebook_path")
    if not path:
        return 0
    top, rel = locate(path)
    if not top:
        return 0
    if rel.startswith(check_scope.PROTECTED):
        return deny(
            f"{rel} is read-only for agents (lint tools, hooks, CI and pytest configuration)"
        )
    try:
        with open(os.path.join(top, rel), encoding="utf-8", errors="replace") as f:
            old = f.read()
    except OSError:
        old = ""
    diff = list(difflib.ndiff(old.splitlines(), new_text(tool, inp, old).splitlines()))
    added = [x[2:] for x in diff if x.startswith("+ ")]
    removed = [x[2:] for x in diff if x.startswith("- ")]
    if any(EXEMPT in x for x in added):
        return deny(
            "exemptions are granted by maintainers; list the finding and the reason in your comment"
        )
    if rel.startswith("tests/"):
        lines = removed + added if role == "brush" else removed
        bad = [
            x.strip() for x in lines if check_scope.TEST_LOGIC.search(x.split("#")[0])
        ]
        if bad:
            return deny(
                "do not change assertions, tolerances, skips or parametrization of tests: "
                + bad[0][:120]
            )
    return 0


def lint(top, paths=()):
    return run([sys.executable, "-I", LINT, "--diff", base_ref(top), *paths], top)


def post(data):
    path = (data.get("tool_input") or {}).get("file_path")
    top, rel = locate(path) if path else (None, None)
    if not top:
        return 0
    out = lint(top, [rel]).stdout.strip()
    if out:
        print(f"halops-lint on the lines you added to {rel}:\n{out}", file=sys.stderr)
        return 2
    return 0


def edited_checkouts(data):
    tops = set()
    try:
        with open(data.get("transcript_path") or "", encoding="utf-8") as f:
            for line in f:
                for path in re.findall(
                    r'"(?:file_path|notebook_path)"\s*:\s*"([^"]+)"', line
                ):
                    top, _ = locate(path)
                    if top:
                        tops.add(top)
    except OSError:
        pass
    top, _ = locate(os.path.join(data.get("cwd") or os.getcwd(), "x"))
    if top:
        tops.add(top)
    return sorted(tops)


def stop(data):
    if data.get("stop_hook_active"):
        return 0
    msgs = []
    for top in edited_checkouts(data):
        r = lint(top)
        if r.returncode == 1:
            msgs.append(f"{top}:\n{r.stdout.strip()}")
    if not msgs:
        return 0
    return deny(
        "error-level halops-lint findings remain in lines you added. Fix them; if one cannot be "
        "fixed, list it with the reason in your final comment instead of exempting it.\n"
        + "\n".join(msgs)
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("event", choices=("pre", "post", "stop"))
    ap.add_argument("--role", choices=("writer", "brush"), default="writer")
    args = ap.parse_args()
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return 0
    if args.event == "pre":
        return pre(data, args.role)
    return post(data) if args.event == "post" else stop(data)


if __name__ == "__main__":
    sys.exit(main())
