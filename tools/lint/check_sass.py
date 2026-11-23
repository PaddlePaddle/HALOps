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
"""Compare the SASS of two builds of the HALOps extensions.

usage: python3 tools/lint/check_sass.py <before_dir> <after_dir> [--json OUT]

Each directory holds the built halops_kernel*.so / halops_mhc_kernel*.so files.
Kernels are matched by demangled name with namespace qualifiers removed, so moving code
out of an anonymous namespace or between namespaces is not a change; symbol references
inside the instructions are normalized the same way. A kernel whose name changed but
whose instructions are identical counts as renamed. Exit status is 1 when any kernel's
instructions changed or a kernel was added or removed.
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

FUNC = re.compile(r"^\s*Function : (\S+)")
INSN = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;?\s*(/\*.*\*/)?\s*$")
MANGLED = re.compile(r"\b_Z\w+")
QUAL = re.compile(r"\b\w+::")


def demangle(names):
    if not names:
        return {}
    out = subprocess.run(
        ["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True
    ).stdout
    return dict(zip(names, out.splitlines()))


def normalize(name):
    return QUAL.sub("", name.replace("(anonymous namespace)::", ""))


def kernels(so):
    """{normalized kernel name: sorted sha1s of its normalized instruction streams}.

    A template kernel instantiated in several translation units has one copy per cubin.
    """
    dump = subprocess.run(
        ["cuobjdump", "-sass", so], capture_output=True, text=True, check=True
    ).stdout
    bodies = []
    for line in dump.splitlines():
        m = FUNC.match(line)
        if m:
            bodies.append((m.group(1), []))
            continue
        m = INSN.match(line)
        if bodies and m and m.group(1):
            bodies[-1][1].append(m.group(1))
    refs = {raw for raw, _ in bodies}
    for _, body in bodies:
        for ins in body:
            refs.update(MANGLED.findall(ins))
    names = demangle(sorted(refs))
    out = collections.defaultdict(list)
    for raw, body in bodies:
        text = "\n".join(
            MANGLED.sub(lambda m: normalize(names.get(m.group(0), m.group(0))), i)
            for i in body
        )
        out[normalize(names.get(raw, raw))].append(
            hashlib.sha1(text.encode()).hexdigest()
        )
    return {k: tuple(sorted(v)) for k, v in out.items()}


def libs(d):
    found = {}
    for so in glob.glob(os.path.join(d, "**", "halops*kernel*.so"), recursive=True):
        found.setdefault(os.path.basename(so).split(".")[0], so)
    return found


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--json", help="write the per-kernel result here")
    args = ap.parse_args()
    before, after = libs(args.before), libs(args.after)
    if not before or set(before) != set(after):
        print(
            f"extension sets differ: {sorted(before)} vs {sorted(after)}",
            file=sys.stderr,
        )
        return 1
    report, bad = {}, 0
    for lib in sorted(before):
        a, b = kernels(before[lib]), kernels(after[lib])
        same = sorted(k for k in a if k in b and a[k] == b[k])
        changed = sorted(k for k in a if k in b and a[k] != b[k])
        only_a = {k: a[k] for k in a if k not in b}
        only_b = {k: b[k] for k in b if k not in a}
        by_hash = collections.defaultdict(list)
        for k, h in only_b.items():
            by_hash[h].append(k)
        renamed, removed = [], []
        for k, h in sorted(only_a.items()):
            if by_hash.get(h):
                renamed.append((k, by_hash[h].pop()))
            else:
                removed.append(k)
        added = sorted(k for v in by_hash.values() for k in v)
        report[lib] = {
            "identical": len(same),
            "renamed": renamed,
            "changed": changed,
            "removed": removed,
            "added": added,
        }
        bad += len(changed) + len(removed) + len(added)
        print(
            f"{lib}: {len(a)} kernels, identical {len(same)}, renamed {len(renamed)}, "
            f"changed {len(changed)}, removed {len(removed)}, added {len(added)}"
        )
        for k in changed:
            print(f"  changed: {k[:200]}")
        for k in removed:
            print(f"  removed: {k[:200]}")
        for k in added:
            print(f"  added:   {k[:200]}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
