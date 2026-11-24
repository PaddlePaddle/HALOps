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
import io
import re
import tokenize

CPP = (".cu", ".cuh", ".h", ".cc", ".hpp")
COMMON = "csrc/common/"

# rule id -> (level, review topic)
RULES = {
    "anon-namespace": ("error", "namespace"),
    "new-namespace": ("error", "namespace"),
    "ns-static": ("error", "namespace"),
    "ns-collision": ("error", "namespace"),
    "model-constexpr": ("error", "magic"),
    "ns-constexpr": ("error", "magic"),
    "local-constexpr": ("error", "magic"),
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
    "comment": ("error", "comment"),
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
MODEL_CAMEL = re.compile(
    r"^k?(\w*?)(Heads?|HeadDim|Dims?|Hidden|Nope|Rope|Latent|Lora|Vocab|Experts?|TopK|Topk|"
    r"Intermediate|PageSize|ScaleGroup|QuantGroup|RowBytes|Sf)(?=[A-Z0-9_]|$)(\w*)"
)
NOT_MODEL = re.compile(
    r"(Tile|Block|Box|Warp|Cta|Mma|Atom|Chunk|Stage|Depth|Smem|Tmem|Thread|Lane|Vec|Pack|"
    r"Per|No|Invalid|Swizzle)$"
)
MODEL_SUFFIX = re.compile(
    r"^(_?(Size|Dims?|Bytes|Rank|Ratio|Groups?|Num|Count|Len|Elems?|Cols|Rows|Total|K|Q|V|"
    r"Kv|Qk|Vo|O|Nope|Rope|Ckv|Kpe|\d+))*$"
)
DESCRIPTOR = re.compile(r"Model|Dims|Shape|Desc")
MODEL_UPPER = re.compile(
    r"(^|_)(HEAD_?DIM|HEAD_SIZE|HEADS|HIDDEN|NOPE|ROPE|LATENT|LORA|VOCAB|TOP_?K|EXPERTS|"
    r"INTERMEDIATE|PAGE_SIZE|SCALE_GROUP|QUANT_GROUP)(_|$)",
    re.I,
)
CONSTEXPR = re.compile(r"^\s*((static|inline)\s+)*(constexpr|const)\b")
DEFINE = re.compile(r"^\s*#\s*define\s+(\w+)\s+(\S.*)$")
IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
LITERAL_OK = re.compile(
    r"^(true|false|nullptr|static_cast|int|unsigned|signed|long|short|char|bool|float|"
    r"double|size_t|u?int\d+_t)$"
)
CONST_NAME = re.compile(r"^([A-Z][A-Z0-9_]*|k[A-Z]\w*)$")
DIRECTIVE = re.compile(
    r"halops-lint:|clang-format\s+(on|off)|NOLINT|IWYU\s+pragma|\bnoqa\b|type:\s*ignore|"
    r"fmt:\s*(on|off|skip)|^pragma\b|pylint:|isort:|pyright:|mypy:|^!/|coding[:=]"
)
LICENSE = re.compile(r"Copyright|Licensed|SPDX-License")
DOC_TYPE = re.compile(
    r"^(template\b|(struct|class|union|enum)\b(?!.*;\s*$)|typedef\b|using\s+\w+\s*=)"
)
NOT_FN = re.compile(
    r"^(#|[{}()]|static_assert\b|return\b|using\s+namespace\b|(public|private|"
    r"protected)\s*:)"
)
BANNER = re.compile(r"[=\-#*~/]{6,}")
ARG_CMT = re.compile(r"^\s*\w+\s*=\s*$")
CHECK_BUDGET = 6
NEW_FILE_MAX_LINES = 600
CMT_DENSITY_MAX = 0.10  # p90 on main is 0.091
LICENSE_LINES = 14


def strip_strings(line):
    return re.sub(r'"(\\.|[^"\\])*"', '""', line)


def scope_map(text, defs=None, decl=None):
    """Per line: (all enclosing scopes are namespaces, enclosing function start line,
    enclosing namespace path such as "halops::moe"; an anonymous namespace is "").
    When defs is a list, append (line, kind, head, namespace path) for every function or
    type body opened at namespace scope. When decl is a list, append per line whether
    every enclosing scope is a namespace or a type body, where declarations live."""
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
        if decl is not None:
            decl.append(all(k in ("ns", "type") for k, _ in stack))
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


def model_name(name):
    """Whether a constant is named after a model or data-format dimension: head count,
    head and latent dims, rope, hidden, experts, top-k, page size, quantization groups,
    row layout. A tile, block, atom ... of such a dim (kTileDim, kSfAtomM, kRopeDepth) is
    tuning or hardware layout, and kNoExpert-style sentinels are not shapes."""
    m = MODEL_CAMEL.match(name)
    if m:
        return not NOT_MODEL.search(m.group(1)) and bool(MODEL_SUFFIX.match(m.group(3)))
    return bool(MODEL_UPPER.search(name))


def parse_decls(body):
    """[(name, initializer)] of a constant variable declaration after its qualifiers;
    [] for a function."""
    parts, depth, cur = [], 0, ""
    for ch in body:
        depth += ch in "([{"
        depth -= ch in ")]}"
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    cuts = [parts[0].find(c) for c in "=({" if c in parts[0]]
    if not cuts or parts[0][min(cuts)] == "(":
        return []
    out = []
    for p in parts:
        m = re.search(r"(\w+)\s*(?:\[[^\]]*\])?\s*(?:=|\{)", p)
        if m:
            out.append((m.group(1), p[m.end() :]))
    return out


def literal(value, known=frozenset()):
    """The initializer is a number or bool written out, possibly combined with sizeof,
    casts to built-in types and the known names."""
    value = re.sub(r"\bsizeof\s*\((?:[^()]|\([^()]*\))*\)", " sizeof ", value)
    names = IDENT.findall(re.sub(r"\b\d[\w.']*", " ", value))
    if not re.search(r"\b\d|\b(true|false)\b", value):
        return False
    return all(n in known or n == "sizeof" or LITERAL_OK.match(n) for n in names)


def constant_decls(text, scopes):
    """[(first line, last line, scope, qualifiers, [(name, initializer)])] for constant
    variables and object-like #define values; scope is "ns", "fn" or "type". A line that
    continues an open parenthesis (parameters, arguments) is not a declaration."""
    lines, decls, depth, opened = text.splitlines(), [], 0, []
    for raw in lines:
        opened.append(depth)
        code = strip_strings(raw.split("//")[0])
        depth = max(0, depth + code.count("(") - code.count(")"))
    i = 0
    while i < len(lines):
        stmt = strip_strings(lines[i].split("//")[0])
        m = DEFINE.match(stmt)
        if m and not m.group(2).rstrip().endswith("\\"):
            decls.append((i + 1, i + 1, "ns", "#define", [(m.group(1), m.group(2))]))
        q = CONSTEXPR.match(stmt)
        if not q or opened[i]:
            i += 1
            continue
        j = i
        while ";" not in stmt and j + 1 < len(lines) and j - i < 20:
            j += 1
            stmt += " " + strip_strings(lines[j].split("//")[0])
        ds = parse_decls(stmt[q.end() :].split(";")[0])
        if not ds:
            i += 1
            continue
        at_ns, fn, _ = scopes[i]
        scope = "ns" if at_ns else ("fn" if fn else "type")
        decls.append((i + 1, j + 1, scope, q.group(0).split(), ds))
        i = j + 1
    return decls


def model_set(text, scopes, decls, types):
    """Namespace-scope constants that carry the model shape: named after a model dim,
    derived from one, or sizeof of a non-template type whose body uses one."""
    lines, bodies = text.splitlines(), []
    for no, h in types:
        m = re.search(r"\b(?:struct|class|union)\b(?:\s+alignas\([^)]*\))?\s+(\w+)", h)
        if m and not re.search(r"\btemplate\s*<", h):
            end = no
            while end < len(lines) and not scopes[end][0]:
                end += 1
            body = " ".join(
                strip_strings(s.split("//")[0]) for s in lines[no - 1 : end]
            )
            bodies.append((m.group(1), set(IDENT.findall(body))))
    model, grew = set(), True
    while grew:
        grew = False
        for _, _, scope, _, ds in decls:
            for n, v in ds:
                if scope == "ns" and n not in model:
                    if model_name(n) or set(IDENT.findall(v)) & model:
                        model.add(n)
                        grew = True
        for n, used in bodies:
            if n not in model and used & model:
                model.add(n)
                grew = True
    return model


MSG_NS = (
    "constant at file scope; put tuning values in the kernel's Config/Traits as static "
    "constexpr, arch constants in csrc/common and schedule constants in the host planner"
)
MSG_MODEL_NS = (
    "model-shape constant at namespace scope; make the model values template parameters "
    "of the kernel config and derive the rest there"
)
MSG_LOCAL = (
    "literal constant inside a function; make it a static constexpr member of the "
    "kernel's Config/Traits"
)


def constant_findings(path, text):
    """{first line: (last line, rule, message)} for constant definitions.

    model-constexpr: a namespace-scope constant in the model set, a literal model value
    inside a function or a primary class template, and a literal model value in a
    non-template type read as Type::kName. A non-template type or explicit specialization
    that is only passed as a template argument describes the model and is allowed.
    ns-constexpr: any other namespace-scope constant or numeric #define.
    local-constexpr: a constexpr or static const local with a literal initializer.
    The last two apply to .cu/.cuh/.h outside csrc/common, which holds arch constants.
    """
    defs = []
    scopes = scope_map(text, defs)
    decls = constant_decls(text, scopes)
    types = [
        (no, re.sub(r"\s+", " ", head)) for no, kind, head, _ in defs if kind == "type"
    ]
    model = model_set(text, scopes, decls, types)
    kernel = path.endswith((".cu", ".cuh", ".h")) and not path.startswith(COMMON)
    out = {}
    for first, last, scope, qual, ds in decls:
        rule = "model-constexpr"
        if scope == "ns":
            hit, msg = [n for n, _ in ds if n in model], MSG_MODEL_NS
            if not hit and kernel:
                rule, msg = "ns-constexpr", MSG_NS
                if qual != "#define":
                    hit = [n for n, _ in ds]
                elif re.search(r"\d", ds[0][1]) and re.fullmatch(
                    r"[\w\s()+\-*/%<>&|^~.]+", ds[0][1]
                ):
                    names = IDENT.findall(ds[0][1])
                    if all(LITERAL_OK.match(n) or CONST_NAME.match(n) for n in names):
                        hit = [ds[0][0]]
        elif scope == "fn":
            msg = "model value fixed in a function; take it from the template config"
            hit = [n for n, v in ds if model_name(n) and literal(v, model)]
            if not hit and kernel:
                rule, msg = "local-constexpr", MSG_LOCAL
                named = "constexpr" in qual or "static" in qual
                hit = [
                    n for n, v in ds if literal(v) and (named or CONST_NAME.match(n))
                ]
        else:
            owner = [h for no, h in types if no <= first]
            m = owner and re.search(
                r"\b(?:struct|class)\b(?:\s+alignas\([^)]*\))?\s+(\w+)", owner[-1]
            )
            if not m or re.search(r"\btemplate\s*<\s*>", owner[-1]):
                continue
            name = m.group(1)
            hit = [n for n, v in ds if model_name(n) and literal(v, model)]
            read = [
                n
                for n in hit
                if re.search(rf"\b{name}\s*(<[^;{{}}]*?>)?\s*::\s*{n}\b", text)
            ]
            if read:
                hit, msg = read, (
                    f"model value read as {name}::; pass {name} to the kernel as a "
                    "template argument instead of naming it"
                )
            elif re.search(r"\btemplate\s*<", owner[-1]) and not DESCRIPTOR.search(
                name
            ):
                msg = f"model value fixed in template {name}; make it a template parameter"
            else:
                continue
        if hit:
            out[first] = (last, rule, f"{', '.join(hit)}: {msg}")
    return out


def cpp_comment_rows(text):
    """Per line: (code outside comments, comment text or None), with string literals and
    block comments tracked across lines."""
    rows, block = [], False
    for raw in text.splitlines():
        code, cmt, i, n = "", [] if block else None, 0, len(raw)
        while i < n:
            if block:
                j = raw.find("*/", i)
                cmt.append(raw[i:] if j < 0 else raw[i:j])
                i, block = (n, True) if j < 0 else (j + 2, False)
                continue
            ch = raw[i]
            if ch == '"' or (ch == "'" and not (i and raw[i - 1].isalnum())):
                j = i + 1
                while j < n and raw[j] != ch:
                    j += 2 if raw[j] == "\\" else 1
                code += raw[i : j + 1]
                i = j + 1
            elif raw.startswith("//", i):
                cmt = (cmt or []) + [raw[i + 2 :]]
                break
            elif raw.startswith("/*", i):
                j = raw.find("*/", i + 2)
                if j > 0 and ARG_CMT.match(raw[i + 2 : j]):
                    code += " "
                    i = j + 2
                    continue
                cmt, block, i = cmt or [], True, i + 2
            else:
                code += ch
                i += 1
        text_ = None if cmt is None else " ".join(cmt).strip(" */")
        rows.append((code.strip(), text_))
    return rows


def py_comment_rows(text):
    """Per line: (code outside the comment, comment text or None) from tokenize."""
    lines = text.splitlines()
    rows = [(s.strip(), None) for s in lines]
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT and tok.start[0] <= len(lines):
                r, c = tok.start
                rows[r - 1] = (lines[r - 1][:c].strip(), tok.string[1:].strip())
    except (tokenize.TokenError, SyntaxError):
        return [(s.strip(), None) for s in lines]
    return rows


MSG_COMMENT = (
    "only docstrings are kept (a comment block directly above a function, type or alias "
    "declaration, or a Python docstring); delete this comment"
)
MSG_TRAILING = "trailing comment; delete it or move the reason into a name"


def doc_head(code):
    """Whether a line at namespace or class scope starts a function, type or alias
    declaration, the declarations a docstring may document."""
    code = re.sub(r"\b(alignas|__align__)\s*\([^()]*\)", "", code).strip()
    if DOC_TYPE.match(code):
        return True
    if NOT_FN.match(code):
        return False
    m = re.search(r"[=(]", code)
    return bool(m and m.group(0) == "(" and re.search(r"\w\s*\($", code[: m.end()]))


def comment_findings(rows, doc_target):
    """[(first line, last line, message)] for comments that are not docstrings.

    rows holds (code, comment text or None) per line, and doc_target(index) tells whether
    the code on that line may carry a docstring. A run of comment-only lines is a
    docstring when the next line is such a declaration. Tool directives, the license
    header and the labels on a namespace's closing brace or an #endif are kept."""
    n, i, out = len(rows), 0, []
    while i < n and not rows[i][0]:
        i += 1
    if not any(c and LICENSE.search(c) for _, c in rows[:i]):
        i = 0
    while i < n:
        code, cmt = rows[i]
        if cmt is None or (code and DIRECTIVE.search(cmt)):
            i += 1
            continue
        if code:
            label = re.match(r"^\};?$", code) and re.search(r"\bnamespace\b", cmt)
            if not label and not re.match(r"^#\s*(endif|else|elif)\b", code):
                out.append((i + 1, i + 1, MSG_TRAILING))
            i += 1
            continue
        j = i
        while j + 1 < n and not rows[j + 1][0] and rows[j + 1][1] is not None:
            j += 1
        prose = [c for _, c in rows[i : j + 1] if not DIRECTIVE.search(c)]
        nxt = j + 1
        doc = nxt < n and rows[nxt][0] and doc_target(nxt)
        if prose and (not doc or any(BANNER.search(c) for c in prose)):
            out.append((i + 1, j + 1, MSG_COMMENT))
        i = j + 1
    return out


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
    for first, (last, rid, msg) in sorted(constant_findings(path, text).items()):
        if any(no in added for no in range(first, last + 1)):
            f.append((first, rid, msg))
    decl = []
    scope_map(text, decl=decl)
    rows = cpp_comment_rows(text)

    def doc_target(i):
        return decl[i] and doc_head(rows[i][0])

    for first, last, msg in comment_findings(rows, doc_target):
        if any(no in added for no in range(first, last + 1)):
            f.append((first, "comment", msg))
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
    for first, last, msg in comment_findings(py_comment_rows(text), lambda i: False):
        if any(no in added for no in range(first, last + 1)):
            f.append((first, "comment", msg))
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
