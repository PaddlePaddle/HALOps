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
"""HALOps code-shape lint.

usage:
  python3 tools/lint/halops_lint.py --diff origin/main       # lines added since a ref
  python3 tools/lint/halops_lint.py --staged                  # staged lines
  python3 tools/lint/halops_lint.py --pre-commit              # pre-commit hook, see below
  python3 tools/lint/halops_lint.py --all                     # whole tree, minus baseline
  python3 tools/lint/halops_lint.py --all --update-baseline   # rewrite baseline.json

--pre-commit checks PRE_COMMIT_FROM_REF..PRE_COMMIT_TO_REF when pre-commit sets them (CI on a
pull request), the staged lines when something is staged (git commit), and the whole tree
minus the baseline otherwise (pre-commit run --all-files).
Exit status is 1 when an error-level finding is not exempted or baselined.
Exempt one line with a reason, on the line itself or the line above:
  // halops-lint: allow(<rule>) <reason>
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules  # noqa: E402

ROOT = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
).stdout.strip()
BASELINE = os.path.join(ROOT, "tools/lint/baseline.json")
SCAN = ("csrc/", "halops/", "tests/", "setup.py")
SKIP = ("3rdparty/", "tools/lint/")
ALLOW = re.compile(r"halops-lint:\s*allow\(([\w,\s-]+)\)\s*(\S.*)?$")


def git(*args):
    return subprocess.run(
        ["git", "-C", ROOT, *args], capture_output=True, text=True, check=True
    ).stdout


def wanted(path):
    return path.startswith(SCAN) and not path.startswith(SKIP)


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8", errors="replace") as f:
        return f.read()


def tracked(prefix=""):
    return [
        p for p in git("ls-files", "--", prefix or ".").split("\n") if p and wanted(p)
    ]


def diff_lines(args):
    """{path: (added line numbers, is new file)} from git diff -U0."""
    refs = (["--cached"] if args.staged else []) + [
        r for r in (args.diff, args.to_ref) if r
    ]
    new = set(git("diff", "--name-only", "--diff-filter=A", *refs).split())
    out, path = {}, None
    if not args.staged and not args.to_ref:
        for p in git("ls-files", "--others", "--exclude-standard").split("\n"):
            if p and wanted(p):
                out[p] = (set(range(1, len(read(p).splitlines()) + 1)), True)
    for line in git(
        "diff", "-U0", "--no-color", "--no-ext-diff", "--diff-filter=AM", *refs
    ).splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            if path and wanted(path):
                out[path] = (set(), path in new)
            else:
                path = None
        elif line.startswith("@@") and path:
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            start, n = int(m.group(1)), int(m.group(2) or 1)
            out[path][0].update(range(start, start + n))
    return out


def file_text(path, args):
    if args.to_ref:
        return git("show", f"{args.to_ref}:{path}")
    if args.staged:
        return git("show", f":{path}")
    return read(path)


def build_index():
    common = [(p, read(p)) for p in tracked(rules.COMMON) if p.endswith(rules.CPP)]
    names, asms = rules.common_index(lambda _: common)
    all_asms, defs = collections.defaultdict(set), collections.defaultdict(set)
    for p in tracked("csrc"):
        if p.endswith(rules.CPP):
            text = read(p)
            for m in rules.ASM.finditer(text):
                all_asms[rules.asm_key(m.group(2))].add(p)
            if p.endswith(rules.TU):
                for name in rules.definitions(text).values():
                    defs[name].add(p)
    return {"names": names, "asms": asms, "all_asms": all_asms, "defs": defs}


def exemptions(text):
    """{line: set(rules)} and malformed exemption lines."""
    allowed, bad = collections.defaultdict(set), []
    for no, line in enumerate(text.splitlines(), 1):
        m = ALLOW.search(line)
        if not m:
            continue
        ids = {r.strip() for r in m.group(1).split(",") if r.strip()}
        if not m.group(2) or not ids <= set(rules.RULES):
            bad.append(no)
            continue
        target = no if line.split("halops-lint")[0].strip(" \t/#*") else no + 1
        allowed[target] |= ids
    return allowed, bad


def lint(targets, args):
    index = build_index()
    findings = []
    for path, (added, is_new) in sorted(targets.items()):
        try:
            text = file_text(path, args)
        except (OSError, subprocess.CalledProcessError):
            continue
        lines = text.splitlines()
        allowed, bad = exemptions(text)
        for no in bad:
            if no in added:
                findings.append(
                    (
                        path,
                        no,
                        "bad-exemption",
                        "exemption needs a known rule id and a reason",
                    )
                )
        for no, rid, msg in rules.check(path, text, added, is_new, index):
            if rid not in allowed.get(no, ()):
                findings.append((path, no, rid, msg))
        findings = [
            (
                f + (lines[f[1] - 1].strip() if 0 < f[1] <= len(lines) else "",)
                if len(f) == 4
                else f
            )
            for f in findings
        ]
    return findings


def load_baseline(path):
    if not os.path.exists(path):
        return collections.Counter()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return collections.Counter(
        (e["path"], e["rule"], e["code"]) for e in data["entries"]
    )


def write_baseline(findings):
    entries = [{"path": p, "rule": r, "code": c} for p, _, r, _, c in findings]
    with open(BASELINE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "note": "pre-existing findings on main; only shrink this file",
                "entries": entries,
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
        f.write("\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--diff",
        metavar="REF",
        help="check lines added since REF (working tree, untracked included)",
    )
    mode.add_argument("--staged", action="store_true", help="check staged lines")
    mode.add_argument("--all", action="store_true", help="check every tracked file")
    mode.add_argument(
        "--pre-commit",
        action="store_true",
        help="pick the mode from the pre-commit context",
    )
    ap.add_argument(
        "--to-ref", help="with --diff, compare REF..TO_REF instead of the working tree"
    )
    ap.add_argument(
        "--no-baseline",
        action="store_true",
        help="with --all, do not subtract baseline.json",
    )
    ap.add_argument(
        "--baseline",
        default=BASELINE,
        help="with --all, subtract this file instead of baseline.json",
    )
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="with --all, rewrite baseline.json",
    )
    ap.add_argument(
        "--strict", action="store_true", help="warn-level findings also fail"
    )
    ap.add_argument("--format", choices=("text", "json"), default="text")
    ap.add_argument("paths", nargs="*", help="limit to these paths")
    args = ap.parse_args()
    if args.pre_commit:
        frm, to = os.environ.get("PRE_COMMIT_FROM_REF"), os.environ.get(
            "PRE_COMMIT_TO_REF"
        )
        if frm and to:
            args.diff, args.to_ref = git("merge-base", frm, to).strip(), to
        elif git("diff", "--cached", "--name-only").strip():
            args.staged = True
        else:
            args.all = True

    if args.all:
        targets = {}
        for p in tracked():
            n = len(read(p).splitlines())
            targets[p] = (set(range(1, n + 1)), False)
    else:
        targets = diff_lines(args)
    if args.paths:
        keep = tuple(os.path.relpath(os.path.abspath(p), ROOT) for p in args.paths)
        targets = {p: v for p, v in targets.items() if p.startswith(keep)}
    findings = lint(targets, args)

    if args.all and args.update_baseline:
        write_baseline(findings)
        print(f"baseline.json: {len(findings)} entries")
        return 0
    if args.all and not args.no_baseline:
        base, kept = load_baseline(args.baseline), []
        for f in findings:
            key = (f[0], f[2], f[4])
            if base[key] > 0:
                base[key] -= 1
            else:
                kept.append(f)
        findings = kept

    failing = [f for f in findings if args.strict or rules.RULES[f[2]][0] == "error"]
    if args.format == "json":
        json.dump(
            [
                {
                    "path": p,
                    "line": n,
                    "rule": r,
                    "level": rules.RULES[r][0],
                    "message": m,
                    "code": c,
                }
                for p, n, r, m, c in findings
            ],
            sys.stdout,
            ensure_ascii=False,
            indent=1,
        )
        print()
    else:
        for p, n, r, m, _ in findings:
            print(f"{p}:{n}: {rules.RULES[r][0]}: [{r}] {m}")
        if findings:
            by_rule = collections.Counter(f[2] for f in findings)
            print(
                "summary: " + ", ".join(f"{r}={n}" for r, n in by_rule.most_common()),
                file=sys.stderr,
            )
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
