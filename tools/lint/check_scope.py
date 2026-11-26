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
"""Check that a lint-fix change stays inside the lint findings it is fixing.

usage: python3 tools/lint/check_scope.py <base_ref> [<head_ref>]

Without head_ref the working tree, untracked files included, is compared. Fails when the change
  - touches the lint tools, agent hooks, CI or pytest configuration,
  - grows tools/lint/baseline.json or adds a `halops-lint: allow` exemption,
  - edits assertions, tolerances or parametrization in tests/,
  - edits a hunk that is not near a finding reported on base_ref, is not structural
    (include, namespace open/close, blank), is not an addition under csrc/common/,
    is not a templating edit (only adds or removes template heads, template arguments
    that are names, `typename` or `X::` qualifiers), and does not mention a symbol
    named by a finding.
"""

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules  # noqa: E402

PROTECTED = (
    "tools/lint/",
    "tools/agent/",
    ".claude/",
    ".github/",
    ".pre-commit-config.yaml",
    "tests/conftest.py",
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    "3rdparty/",
)
BASELINE = "tools/lint/baseline.json"
TEST_LOGIC = re.compile(
    r"\b(assert\w*|atol|rtol|allclose|assert_close|parametrize|skip\w*|xfail|TOL\w*)\b"
)
STRUCTURAL = re.compile(
    r"^\s*(#include\b.*|namespace\s*[\w:]*\s*\{|\}\s*(//.*)?|using namespace\s+[\w:]+;)?\s*$"
)
TEMPLATING = re.compile(
    r"\btemplate\s*<[^<>]*>|<\s*(?:class\s+|typename\s+)?[A-Za-z_][\w:]*\s*>"
    r"|\btypename\s+|\b[A-Za-z_]\w*::(?=[A-Za-z_])"
)
WINDOW = 8
KEYWORDS = {
    "const",
    "static",
    "inline",
    "constexpr",
    "void",
    "auto",
    "return",
    "template",
    "typename",
    "class",
    "struct",
    "namespace",
    "using",
    "int",
    "bool",
    "float",
    "double",
    "unsigned",
    "uint32_t",
    "uint64_t",
    "int64_t",
    "int32_t",
    "uint8_t",
    "CUTE_DEVICE",
    "CUTLASS_DEVICE",
    "__device__",
    "__host__",
    "__forceinline__",
    "__global__",
    "torch",
    "Tensor",
    "std",
    "cute",
    "halops",
    "nullptr",
    "true",
    "false",
    "self",
    "def",
    "import",
    "from",
}


def templating(removed, added):
    """True when the hunk only adds or removes template syntax and scope qualifiers."""

    def strip(lines):
        out = (re.sub(r"\s+", "", TEMPLATING.sub("", x)) for x in lines)
        return [x for x in out if x]

    return strip(removed) == strip(added)


def git(root, *args, check=True):
    return subprocess.run(
        ["git", "-C", root, *args], capture_output=True, text=True, check=check
    ).stdout


def base_findings(root, base):
    """Run the lint script as it exists on base over a pristine checkout of base."""
    tmp = os.path.join(tempfile.mkdtemp(prefix="halops-scope-"), "base")
    git(root, "worktree", "add", "--detach", tmp, base)
    try:
        out = subprocess.run(
            [
                sys.executable,
                "-I",
                os.path.join(tmp, "tools/lint/halops_lint.py"),
                "--all",
                "--no-baseline",
                "--format",
                "json",
            ],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        return json.loads(out.stdout)
    finally:
        git(root, "worktree", "remove", "--force", tmp, check=False)
        shutil.rmtree(os.path.dirname(tmp), ignore_errors=True)
        git(root, "worktree", "prune", check=False)


def symbols(root, base, findings):
    """Identifiers named by findings: words in the flagged line and the enclosing function name."""
    out, texts = set(), {}
    for f in findings:
        out.update(
            w
            for w in re.findall(r"[A-Za-z_]\w{3,}", f["code"] + " " + f["message"])
            if w not in KEYWORDS
        )
        if f["path"].endswith(rules.CPP):
            text = texts.setdefault(f["path"], git(root, "show", f"{base}:{f['path']}"))
            defs = []
            rules.scope_map(text, defs)
            starts = [d for d in defs if d[0] <= f["line"]]
            if starts:
                m = re.search(
                    r"(\w+)\s*(<[^()]*>)?\s*\([^()]*(\([^()]*\)[^()]*)*\)[^()]*$",
                    starts[-1][2],
                )
                if m:
                    out.add(m.group(1))
    return out


def hunks(root, base, head):
    """[(path, old_start, old_lines, new_start, removed, added)] from git diff -U0."""
    refs = [base] + ([head] if head else [])
    out, path, cur = [], None, None
    for line in git(
        root, "diff", "-U0", "--no-color", "--no-ext-diff", "-M", *refs
    ).splitlines():
        if line.startswith("diff --git"):
            path = None
        elif line.startswith("--- "):
            old_path = line[6:] if line.startswith("--- a/") else None
        elif line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else old_path
        elif line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            cur = [path, int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), [], []]
            out.append(cur)
        elif cur and line[:1] == "-" and not line.startswith("---"):
            cur[4].append(line[1:])
        elif cur and line[:1] == "+" and not line.startswith("+++"):
            cur[5].append(line[1:])
    if not head:
        for p in git(root, "ls-files", "--others", "--exclude-standard").split("\n"):
            if p:
                with open(
                    os.path.join(root, p), encoding="utf-8", errors="replace"
                ) as f:
                    out.append([p, 0, 0, 1, [], f.read().splitlines()])
    return out


def baseline_entries(root, ref):
    try:
        text = (
            git(root, "show", f"{ref}:{BASELINE}")
            if ref
            else open(os.path.join(root, BASELINE)).read()
        )
    except (subprocess.CalledProcessError, OSError):
        return collections.Counter()
    return collections.Counter(
        (e["path"], e["rule"], e["code"]) for e in json.loads(text)["entries"]
    )


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    root = git(os.getcwd(), "rev-parse", "--show-toplevel").strip()
    base, head = sys.argv[1], (sys.argv[2] if len(sys.argv) == 3 else None)
    findings = base_findings(root, base)
    near = collections.defaultdict(set)
    for f in findings:
        near[f["path"]].update(range(f["line"] - WINDOW, f["line"] + WINDOW + 1))
    names = symbols(root, base, findings)
    problems = []

    for path, old, n_old, new, removed, added in hunks(root, base, head):
        where = f"{path}:{new}"
        if path == BASELINE:
            continue
        if path.startswith(PROTECTED):
            problems.append(f"{where}: protected path")
            continue
        if any("halops-lint: allow" in x for x in added):
            problems.append(f"{where}: adds an exemption")
            continue
        if path.startswith("tests/") and any(
            TEST_LOGIC.search(x.split("#")[0]) for x in removed + added
        ):
            problems.append(
                f"{where}: changes an assertion, tolerance or parametrization"
            )
            continue
        if not path.startswith(("csrc/", "halops/", "tests/")):
            problems.append(f"{where}: outside csrc/, halops/, tests/")
            continue
        span = set(range(old, old + max(n_old, 1)))
        if span & near[path]:
            continue
        changed = removed + added
        if all(STRUCTURAL.match(x) for x in changed):
            continue
        if path.startswith(rules.COMMON) and not removed:
            continue
        if templating(removed, added):
            continue
        words = {w for x in changed for w in re.findall(r"[A-Za-z_]\w{3,}", x)}
        if words & names:
            continue
        problems.append(f"{where}: hunk is not near any finding on {base}")

    before, after = baseline_entries(root, base), baseline_entries(root, head)
    grown = after - before
    for path, rule, code in sorted(grown.elements()):
        problems.append(f"{BASELINE}: new entry {path} [{rule}] {code[:80]}")

    for p in problems:
        print(p)
    print(f"check_scope: {len(problems)} problem(s) against {base}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
