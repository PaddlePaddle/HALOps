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
"""Rule table and per-file checks for halops_lint.py."""

import ast
import re

CPP = (".cu", ".cuh", ".h", ".cc", ".hpp")
COMMON = "csrc/common/"

# rule id -> (level, review topic)
RULES = {
    "anon-namespace": ("error", "namespace"),
    "new-namespace": ("error", "namespace"),
    "ns-static": ("error", "namespace"),
    "ns-collision": ("error", "namespace"),
    "ns-constexpr-in-cu": ("warn", "magic"),
    "reuse-cuda-check": ("error", "reuse"),
    "reuse-launch": ("error", "reuse"),
    "reuse-pdl": ("error", "reuse"),
    "reuse-stream": ("error", "reuse"),
    "reuse-helper": ("error", "reuse"),
    "reuse-asm": ("error", "reuse"),
    "dup-asm": ("warn", "reuse"),
    "getenv": ("error", "debug"),
    "binding-location": ("error", "layout"),
    "torch-zeros": ("warn", "perf"),
    "device-macro": ("error", "alias"),
    "type-alias": ("warn", "alias"),
    "check-budget": ("warn", "defensive"),
    "debug-comment": ("error", "comment"),
    "comment-density": ("error", "comment"),
    "file-too-long": ("error", "layout"),
    "hpp": ("error", "layout"),
    "new-ext": ("error", "layout"),
    "new-extension": ("error", "layout"),
    "py-defensive": ("warn", "defensive"),
    "py-alloc": ("warn", "api"),
    "missing-docstring": ("error", "doc"),
    "test-print": ("error", "test"),
    "test-seed": ("error", "test"),
    "test-tol-literal": ("warn", "test"),
    "bad-exemption": ("error", "meta"),
}

NS_OK = re.compile(
    r"^(halops(::(moe|mla|mhc|gemm|attention|indexer|mxfp8?))?(::detail)?|detail|cute)$"
)
# (rule id, pattern, path prefix where the pattern is allowed, message)
BANNED = [
    (
        "reuse-cuda-check",
        r"\b(C10_CUDA_CHECK|AT_CUDA_CHECK|C10_CUDA_KERNEL_LAUNCH_CHECK)\b",
        None,
        "use HALOPS_CUDA_CHECK (common/cuda_utils.h)",
    ),
    (
        "reuse-launch",
        r"\bcudaLaunchKernelEx\b|\bcudaLaunchAttribute\b",
        COMMON,
        "use LaunchKernelWithPdl (common/pdl.cuh)",
    ),
    (
        "reuse-pdl",
        r"griddepcontrol|cudaGridDependencySynchronize|cudaTriggerProgrammaticLaunchCompletion",
        COMMON,
        "use pdl_wait / pdl_trigger (common/pdl.cuh)",
    ),
    (
        "reuse-stream",
        r"\b(at|c10)::cuda::getCurrentCUDAStream\b",
        COMMON,
        "use GetCurrentCUDAStream (common/cuda_utils.h)",
    ),
    (
        "getenv",
        r"\bgetenv\s*\(",
        None,
        "kernels and launchers do not read environment variables",
    ),
    (
        "binding-location",
        r"\bPYBIND11_MODULE\b",
        "csrc/common/cpp_extensions.cc",
        "bindings live only in csrc/common/cpp_extensions.cc",
    ),
    (
        "torch-zeros",
        r"\b(torch|at)::zeros\b",
        None,
        "use empty unless zero-init is required",
    ),
    (
        "device-macro",
        r"#define\s+\w+\s+(__device__|__forceinline__|__host__|CUTE_DEVICE)",
        None,
        "do not define device qualifier macros; use CUTE_DEVICE",
    ),
    (
        "type-alias",
        r"^\s*using\s+\w+\s*=\s*(__nv_bfloat16|__half|float[24]?|u?int[24]?|half|"
        r"cutlass::bfloat16_t|cutlass::float_e4m3_t|std::u?int\d+_t)\s*;",
        None,
        "do not alias built-in types",
    ),
]
DEBUG_CMT = re.compile(
    r"\b\d+(\.\d+)?\s*(us|ms|ns|GB/s|TB/s|TFLOPS)\b|\bmeasured\b|\bfaster than\b|"
    r"\bslower than\b|\bNCU\b|\bnsys\b|\btried\b|\bpreviously\b|\bused to be\b|"
    r"\bold (version|path|kernel)\b|\bworkaround\b|\bhack\b|\btrick\b|\bTODO\b|\bFIXME\b|"
    r"\bwas \w+ed\b|实测|调试|尝试|之前|旧版|临时"
)
DEV_FN = re.compile(
    r"(CUTE_DEVICE|CUTLASS_DEVICE|__device__|CUTE_HOST_DEVICE)[^;{(=]*?\b(\w+)\s*\("
)
ASM = re.compile(r'asm\s*(volatile)?\s*\(\s*"([^"]{16,})"')
GLOBAL_SEED = re.compile(
    r"\btorch\.(cuda\.)?manual_seed(_all)?\s*\(|\bnp\.random\.seed\s*\(|(?<![\w.])random\.seed\s*\("
)
SPAWN = re.compile(
    r"\bsubprocess\.(run|Popen|check_call|check_output)\b|\bmp\.spawn\b|\bmultiprocessing\b"
)
CHECK_BUDGET = 6
NEW_FILE_MAX_LINES = 600
CMT_DENSITY_MAX = 0.10  # p90 on main is 0.091
LICENSE_LINES = 14


def strip_strings(line):
    return re.sub(r'"(\\.|[^"\\])*"', '""', line)


def scope_map(text, defs=None):
    """Per line: (all enclosing scopes are namespaces, enclosing function start line,
    enclosing namespace path such as "halops::moe"; an anonymous namespace is "").
    When defs is a list, append (line, kind, head, namespace path) for every function or
    type body opened at namespace scope."""
    stack, out, pending, func = [], [], "", None
    for no, raw in enumerate(text.splitlines(), 1):
        line = strip_strings(raw.split("//")[0])
        out.append(
            (
                all(k == "ns" for k, _ in stack),
                func,
                "::".join(n for k, n in stack if k == "ns"),
            )
        )
        for ch in line:
            if ch == "{":
                head = pending.strip()
                name = ""
                m = re.search(r"\bnamespace\b\s*([\w:]*)[^;]*$", head)
                if m:
                    kind, name = "ns", m.group(1)
                elif re.search(r"\b(struct|class|union|enum)\b[^()]*$", head):
                    kind = "type"
                elif head.endswith(")") or re.search(
                    r"\)\s*(const|noexcept|->.*)?\s*$", head
                ):
                    kind = "fn" if all(k in ("ns", "type") for k, _ in stack) else "blk"
                else:
                    kind = "blk"
                if kind == "fn" and func is None:
                    func = no
                if (
                    defs is not None
                    and kind in ("fn", "type")
                    and all(k == "ns" for k, _ in stack)
                ):
                    names = [n for k, n in stack if k == "ns"]
                    defs.append(
                        (no, kind, head, None if "" in names else "::".join(names))
                    )
                stack.append((kind, name))
                pending = ""
            elif ch == "}":
                if stack:
                    k, _ = stack.pop()
                    if k == "fn" and not any(x == "fn" for x, _ in stack):
                        func = None
                pending = ""
            elif ch == ";":
                pending = ""
            else:
                pending += ch
        pending += " "
    return out


def comment_of(path, line):
    if path.endswith(CPP):
        m = re.search(r"//(.*)$", line)
        return (
            m.group(1)
            if m
            else (line.strip() if line.strip().startswith("*") else None)
        )
    m = re.search(r"#(.*)$", line)
    return m.group(1) if m else None


def asm_key(s):
    return re.sub(r"\s+", " ", s).strip()


def definitions(text):
    """{line: qualified name} for functions and types with external linkage defined at
    namespace scope (anonymous namespaces and static functions are skipped)."""
    defs, out = [], {}
    scope_map(text, defs)
    for no, kind, head, ns in defs:
        head = re.sub(r"\s+", " ", head)
        if ns is None or re.search(r"\b(static|extern)\b", head):
            continue
        if kind == "type":
            m = re.search(
                r"\b(?:struct|class|union|enum(?: class)?)\b(?:\s+alignas\([^)]*\))?\s+(\w+(?:<.*>)?)",
                head,
            )
        else:
            depth, i = 0, len(head) - 1
            while i >= 0:
                depth += head[i] == ")"
                depth -= head[i] == "("
                if depth == 0 and head[i] == "(":
                    break
                i -= 1
            m = re.search(r"([\w:~]+)\s*$", head[: max(i, 0)])
        if m and not m.group(1).startswith("__"):
            out[no] = f"{ns}::{m.group(1)}" if ns else m.group(1)
    return out


TU = (".cu", ".cc")


def common_index(read):
    """Device helper names and inline PTX strings defined in csrc/common."""
    names, asms = {}, {}
    for path, text in read(COMMON):
        for m in DEV_FN.finditer(text):
            names.setdefault(m.group(2), path)
        for m in ASM.finditer(text):
            asms.setdefault(asm_key(m.group(2)), path)
    return names, asms


def check_cpp(path, text, added, is_new, index):
    f = []
    scopes = scope_map(text)
    lines = text.splitlines()
    checks_in_fn = {}
    in_common = path.startswith(COMMON)
    defs = definitions(text) if path.endswith(TU) and index else {}
    for no in sorted(added):
        line = lines[no - 1] if no <= len(lines) else ""
        code = strip_strings(line.split("//")[0])
        at_ns, fn, ns = scopes[no - 1] if no <= len(scopes) else (False, None, "")
        if no in defs and index["defs"].get(defs[no], set()) - {path}:
            other = sorted(index["defs"][defs[no]] - {path})[0]
            f.append(
                (
                    no,
                    "ns-collision",
                    f"{defs[no]} is also defined in {other}; rename or move it to a header",
                )
            )
        if re.match(r"^\s*namespace\s*\{", code):
            f.append(
                (
                    no,
                    "anon-namespace",
                    "no anonymous namespace; use the file's named namespace",
                )
            )
        m = re.match(r"^\s*namespace\s+([\w:]+)\s*\{", code)
        full = f"{ns}::{m.group(1)}" if m and ns else (m.group(1) if m else "")
        if m and not NS_OK.match(full):
            f.append((no, "new-namespace", f"do not add namespace {full}"))
        if at_ns and re.match(r"^\s*(template\s*<.*>\s*)?([A-Z_]+\s+)*static\b", code):
            f.append((no, "ns-static", "no static at namespace scope"))
        if (
            at_ns
            and path.endswith(".cu")
            and re.match(r"^\s*(inline\s+)?constexpr\b", code)
        ):
            f.append((no, "ns-constexpr-in-cu", "file-scope constexpr in .cu"))
        for rid, pat, allow, msg in BANNED:
            if re.search(pat, code) and not (allow and path.startswith(allow)):
                f.append((no, rid, msg))
        if fn and re.search(r"\bTORCH_CHECK\b", code):
            checks_in_fn[fn] = checks_in_fn.get(fn, 0) + 1
            if checks_in_fn[fn] == CHECK_BUDGET + 1:
                f.append(
                    (
                        no,
                        "check-budget",
                        f"more than {CHECK_BUDGET} TORCH_CHECK in one function",
                    )
                )
        c = comment_of(path, line)
        if c and DEBUG_CMT.search(c):
            f.append(
                (
                    no,
                    "debug-comment",
                    "comment records debugging, experiments or history",
                )
            )
        if in_common or not index:
            continue
        m = DEV_FN.search(code)
        if m and m.group(2) in index["names"]:
            f.append(
                (
                    no,
                    "reuse-helper",
                    f"{m.group(2)} already exists in {index['names'][m.group(2)]}",
                )
            )
        m = ASM.search(line)
        if m:
            key = asm_key(m.group(2))
            if key in index["asms"]:
                f.append(
                    (
                        no,
                        "reuse-asm",
                        f"same PTX already wrapped in {index['asms'][key]}",
                    )
                )
            elif index["all_asms"].get(key, set()) - {path}:
                other = sorted(index["all_asms"][key] - {path})[0]
                f.append(
                    (no, "dup-asm", f"same PTX also in {other}; move it to csrc/common")
                )
    if is_new:
        body = [s.strip() for s in lines[LICENSE_LINES:] if s.strip()]
        if body:
            dens = sum(s.startswith(("//", "*", "/*")) for s in body) / len(body)
            if dens > CMT_DENSITY_MAX:
                f.append(
                    (
                        1,
                        "comment-density",
                        f"comment density {dens:.2f} > {CMT_DENSITY_MAX}",
                    )
                )
        if len(lines) > NEW_FILE_MAX_LINES and path.endswith(".cu"):
            f.append(
                (
                    1,
                    "file-too-long",
                    f"new .cu has {len(lines)} lines; split kernel and launcher",
                )
            )
        if path.endswith(".hpp"):
            f.append((1, "hpp", "use .h / .cuh, not .hpp"))
        if re.search(r"_ext\.cc$", path):
            f.append((1, "new-ext", "do not add a separate binding file"))
    return f


def worker_lines(text):
    """Lines inside helper functions of a test file that launches worker processes.

    Worker entry points run outside pytest, so they print results and seed themselves.
    """
    if not SPAWN.search(text):
        return set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out = set()
    for node in tree.body:
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) or node.name.startswith("test_"):
            continue
        if any("fixture" in ast.unparse(d) for d in node.decorator_list):
            continue
        out.update(range(node.lineno, node.end_lineno + 1))
    return out


def check_py(path, text, added, is_new):
    f = []
    lines = text.splitlines()
    api = path.startswith("halops/") and "/kernels/" not in path
    test = path.startswith("tests/") and not path.endswith("conftest.py")
    workers = worker_lines(text) if test else set()
    for no in sorted(added):
        line = lines[no - 1] if no <= len(lines) else ""
        code = line.split("#")[0]
        if api and re.match(r"^\s*(raise|assert)\b", code):
            f.append(
                (
                    no,
                    "py-defensive",
                    "the C++ entry already checks; do not repeat it in the API",
                )
            )
        if api and re.search(r"\btorch\.(zeros|empty|full|ones)\b", code):
            f.append((no, "py-alloc", "the Python API does not allocate or initialize"))
        if test and no not in workers and re.match(r"^\s*print\(", code):
            f.append((no, "test-print", "no print in tests"))
        if test and no not in workers and GLOBAL_SEED.search(code):
            f.append(
                (
                    no,
                    "test-seed",
                    "global seeds are set in tests/conftest.py; use a local torch.Generator",
                )
            )
        if test and re.search(r"\b(atol|rtol)\s*=\s*[0-9.]", code):
            f.append((no, "test-tol-literal", "use a named tolerance constant"))
        c = comment_of(path, line)
        if c and DEBUG_CMT.search(c):
            f.append(
                (
                    no,
                    "debug-comment",
                    "comment records debugging, experiments or history",
                )
            )
    if api:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return f
        for node in tree.body:
            if (
                isinstance(node, ast.FunctionDef)
                and not node.name.startswith("_")
                and node.lineno in added
                and ast.get_docstring(node) is None
            ):
                f.append(
                    (node.lineno, "missing-docstring", "public API needs a docstring")
                )
    if path == "setup.py" and len(added) < len(lines):
        for no in added:
            if no <= len(lines) and "CUDAExtension(" in lines[no - 1]:
                f.append((no, "new-extension", "do not add another CUDAExtension"))
    return f


def check(path, text, added, is_new, index=None):
    """Return [(line, rule id, message)] for the given added line numbers."""
    if path.endswith(CPP):
        return check_cpp(path, text, added, is_new, index)
    if path.endswith(".py"):
        return check_py(path, text, added, is_new)
    return []
