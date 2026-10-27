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

import ast
import os
import re
import shutil
import sys
import tomllib
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))

project = "HALOps"
author = "PaddlePaddle Authors"
copyright = "2026, PaddlePaddle Authors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "myst_parser",
]

# The API pages only read docstrings; neither the CUDA extension nor torch is
# needed to build them.
autodoc_mock_imports = ["torch", "halops.halops_kernel"]
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_member_order = "bysource"
autosummary_generate = False
add_module_names = False
python_maximum_signature_line_length = 60
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False

copybutton_exclude = ".linenos, .gp, .go"

exclude_patterns = ["_build", "*_CN.md", "code_standards.md"]
# included repository files keep their own heading levels
suppress_warnings = ["myst.xref_missing", "myst.header"]

html_theme = "furo"
html_title = "HALOps"
_FONTS = {
    "font-stack": '"Inter Variable", -apple-system, BlinkMacSystemFont, "Segoe UI", '
    '"PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", Helvetica, Arial, sans-serif',
    "font-stack--monospace": '"JetBrains Mono Variable", "SFMono-Regular", Menlo, '
    "Consolas, monospace",
}
html_logo = "images/halops-logo-light.svg"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        **_FONTS,
        "color-brand-primary": "#2563EB",
        "color-brand-content": "#2563EB",
    },
}
html_static_path = ["images"]
_FONTSOURCE = "https://cdn.jsdelivr.net/npm/@fontsource-variable"
html_css_files = [
    f"{_FONTSOURCE}/inter@5.2.8/index.min.css",
    f"{_FONTSOURCE}/jetbrains-mono@5.2.8/index.min.css",
    "halops.css",
]
pygments_style = "friendly"
toc_object_entries = False

# Written into the build output by setup(), so the repository needs no CSS file.
# The site is light only: the theme toggle is hidden and the theme pinned.
_CSS = """
.theme-toggle-container { display: none !important; }
body { line-height: 1.65; color: #1F2937; }
h1, h2, h3 { font-weight: 700; letter-spacing: -0.015em; color: #0F172A; }
code.literal, p code.literal { border: none; border-radius: 4px; font-size: 0.86em; }
pre, div.highlight { border-radius: 8px; }

/* one section per API: deep-blue heading over its signature card */
section.api-entry { margin-top: 3rem; padding-top: 0.5rem;
  border-top: 1px solid var(--color-background-border); }
section.api-entry > h2 { font-family: var(--font-stack--monospace); font-size: 1.35rem;
  font-weight: 700; color: #1E3A8A; margin-bottom: 0.8rem; }
dl.py.function { margin: 0 0 1rem; }
dl.py.function > dt.sig { margin: 0 0 1rem; padding: 0.6rem 1rem 0.6rem 3em;
  line-height: 1.7; font-size: 0.86rem; background: #F1F5F9;
  border-left: 3px solid var(--color-brand-primary); border-radius: 0 6px 6px 0; }
dl.py.function > dt.sig .sig-name { color: var(--color-brand-primary); }
dl.py.function > dt.sig .sig-param { font-weight: 400; color: #334155; }
dl.py.function > dt.sig .default_value { color: #9333EA; }
dl.py.function > dd { margin-left: 0; }

/* the one-line summary reads as a lead paragraph */
dl.py.function > dd > p:first-child { font-size: 1.05rem; font-weight: 500; color: #0F172A; }

/* "Parameters" / "Returns" / "Example" section labels */
dl.field-list > dt, dl.py.function p.rubric { display: block; margin: 1.4rem 0 0.6rem;
  padding: 0.15rem 0 0.15rem 0.6rem; font-size: 0.95rem; font-weight: 700;
  letter-spacing: 0.02em; text-transform: none; color: #1E40AF;
  border-left: 3px solid #93C5FD; background: none; }
dl.field-list { display: block; }
dl.field-list > dd { margin-left: 0.9rem; }
dl.field-list .colon { display: none; }

/* parameter name / type / description */
dl.field-list > dd > ul > li { margin-bottom: 0.55rem; }
dl.field-list > dd strong { font-family: var(--font-stack--monospace); font-size: 0.9em;
  color: #B45309; }
dl.field-list > dd > ul > li > p > em,
dl.field-list > dd > ul > li > em { color: #64748B; }
dl.field-list > dd ul ul { margin-top: 0.3rem; font-size: 0.93em; color: #475569; }

/* summary table at the top of each API page */
table.autosummary { width: 100%; border: 1px solid var(--color-background-border);
  border-radius: 8px; border-collapse: separate; border-spacing: 0; box-shadow: none; }
table.autosummary td { padding: 0.45rem 0.9rem;
  border-top: 1px solid var(--color-background-border); }
table.autosummary tr:first-child td { border-top: none; }
table.autosummary td:first-child { white-space: nowrap; }
table.autosummary code.literal { background: none; padding: 0; font-weight: 600;
  color: var(--color-brand-primary); }
"""


def _public_names(module: Path) -> list[str]:
    for node in ast.parse(module.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return [ast.literal_eval(e) for e in node.value.elts]
    return []


def _page_title(pkg: Path) -> str:
    """``__title__`` of the subpackage if set, else its name (short names such
    as gemm / mla read as acronyms)."""
    for node in ast.parse((pkg / "__init__.py").read_text("utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__title__" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return pkg.name.upper() if len(pkg.name) <= 4 else pkg.name.capitalize()


def _write_api_pages() -> None:
    """One page per ``halops`` subpackage, listing the ``__all__`` of its
    modules, so new or removed APIs and subpackages need no docs edit."""
    out = DOCS / "api"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    for pkg in sorted(p for p in (ROOT / "halops").iterdir() if p.is_dir()):
        if pkg.name.startswith("_") or not (pkg / "__init__.py").exists():
            continue
        names = [
            name
            for module in sorted(pkg.glob("[!_]*.py"))
            for name in _public_names(module)
        ]
        if not names:
            continue
        lines = [f"# {_page_title(pkg)}", "", "```{eval-rst}"]
        lines += [".. currentmodule:: halops", ""]
        lines += [".. autosummary::", "   :nosignatures:", ""]
        lines += [f"   {name}" for name in names] + ["```", ""]
        for name in names:
            lines += ["```{rst-class} api-entry", "```", ""]
            lines += [f"## {name}", "", "```{eval-rst}", ".. currentmodule:: halops"]
            lines += ["", f".. autofunction:: {name}", "```", ""]
        (out / f"{pkg.name}.md").write_text("\n".join(lines), encoding="utf-8")


# Pages that show part of a repository file, generated into docs/pages/:
# (page, title, source file, start-after, end-before).
_INCLUDE_PAGES = [
    (
        "installation",
        "Installation and Quick Start",
        "README.md",
        "</table>",
        "## Roadmap",
    ),
    (
        "benchmark",
        "Benchmark",
        "benchmark/README.md",
        "## English Quick Reference",
        "The Chinese sections",
    ),
    ("contributing", "Contributing", "CONTRIBUTING.md", "## English", "## 简体中文"),
    ("changelog", None, "CHANGELOG.md", None, None),
    (
        "acknowledgement",
        "Acknowledgement",
        "README.md",
        "## Acknowledgement",
        "## License",
    ),
]
# links to these files from included text point at their page instead
_INCLUDED_PAGES = {src: f"pages/{page}.md" for page, _, src, _, _ in _INCLUDE_PAGES}
_REPO_URL = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"][
    "urls"
]["Repository"]
_LINK = re.compile(r"(\]\(|src=\"|href=\")([^)\"\s]+)")
_NAV_BAR = re.compile(r"^\[[^\]]+\]\([^)]+\)( \| \[[^\]]+\]\([^)]+\))+$", re.M)


def _rewrite_link(link: str, src_dir: Path, to_root: str) -> str:
    if link.startswith(("http://", "https://", "mailto:", "#")):
        return link
    path, _, frag = link.partition("#")
    rel = Path(os.path.relpath((src_dir / path).resolve(), ROOT)).as_posix()
    if rel.startswith("docs/images/"):
        return to_root + "_static/" + rel.removeprefix("docs/images/")
    if rel in _INCLUDED_PAGES:
        return to_root + _INCLUDED_PAGES[rel]
    if rel.startswith("docs/") and rel.endswith(".md") and (ROOT / rel).exists():
        page = rel.removeprefix("docs/")
        if not any(Path(page).match(p) for p in exclude_patterns):
            return to_root + page + (f"#{frag}" if frag else "")
    return f"{_REPO_URL}/blob/main/{rel}" + (f"#{frag}" if frag else "")


def _fix_included_links(app, relative_path, parent_docname, content):
    """Make links in repository files included into docs pages resolve on the
    docs site, and drop their in-page navigation bars."""
    src_dir = (DOCS / relative_path).resolve().parent
    to_root = "../" * app.env.docname.count("/")
    text = _NAV_BAR.sub("", content[0])
    content[0] = _LINK.sub(
        lambda m: m.group(1) + _rewrite_link(m.group(2), src_dir, to_root), text
    )


def _write_include_pages() -> None:
    out = DOCS / "pages"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    for page, title, src, start, end in _INCLUDE_PAGES:
        lines = [f"# {title}", ""] if title else []
        lines.append(f"```{{include}} ../../{src}")
        if start:
            lines.append(f':start-after: "{start}"')
        if end:
            lines.append(f':end-before: "{end}"')
        lines += ["```", ""]
        (out / f"{page}.md").write_text("\n".join(lines), encoding="utf-8")


def _doctest_to_code_block(app, what, name, obj, options, lines):
    """Render ``>>>`` examples as plain Python code blocks."""
    out, i = [], 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if not stripped.startswith(">>>"):
            out.append(lines[i])
            i += 1
            continue
        indent = lines[i][: len(lines[i]) - len(stripped)]
        out += [f"{indent}.. code-block:: python", ""]
        while i < len(lines) and lines[i].lstrip().startswith((">>>", "...")):
            code = lines[i].lstrip()[4:]
            out.append(f"{indent}   {code}" if code.strip() else "")
            i += 1
        out.append("")
    lines[:] = out


def _drop_none_return(app, what, name, obj, options, signature, return_annotation):
    """Skip the "Return type: None" field of in-place operators."""
    annotations = app.env.current_document.autodoc_annotations.get(name, {})
    if annotations.get("return") == "None":
        del annotations["return"]


_write_api_pages()
_write_include_pages()


def _write_css(app, exception):
    if exception is None and app.builder.format == "html":
        (Path(app.outdir) / "_static" / "halops.css").write_text(_CSS, "utf-8")


def setup(app):
    app.add_js_file(None, body='localStorage.setItem("theme", "light");')
    app.connect("build-finished", _write_css)
    app.connect("autodoc-process-docstring", _doctest_to_code_block)
    app.connect("autodoc-process-signature", _drop_none_return)
    app.connect("include-read", _fix_included_links)
