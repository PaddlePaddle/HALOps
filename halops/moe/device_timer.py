"""Device event timer postprocessing: decode, resolve call sites, export.

The device writes one 16-byte record per probe into a per-(CTA, warp) slot
range (``csrc/common/device_timer.hpp``)::

    word[0] = site_id | dir << 32 | warp << 40 | arg << 48
    word[1] = %globaltimer in ns, 0 for an untouched slot

``dir`` is bits 32..33 and bit 34 flags that bits 48..63 hold a runtime ``arg``
(an expert id, a K-loop iteration, ...: each probe picks its own meaning and the
event is shown as ``name(arg)``).  A span's ``begin`` owns the arg; its ``end``
carries none.  ``site_id`` is the CALL SITE, not an entry in an event table::

    site_id = (fold16(fnv1a(basename(__FILE__))) << 16) | (__LINE__ & 0xffff)

so this module recovers ``file:line`` by hashing the basenames of the kernel
sources itself, and takes each probe's display name from the string literal
written at that line.  Nothing on the host or in python mirrors an event enum;
adding a probe is a one-line edit in the kernel.

The export is one Chrome Trace Format file -- one track per (CTA, warp), spans
from ``begin``/``end`` pairs, marks as instants -- loadable in ui.perfetto.dev
or chrome://tracing.  Driven by ``bash profile.sh``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

__all__ = ["SiteMap", "collect_and_write", "write_csv"]

_KERNEL_SOURCE_SUFFIXES = (".cu", ".cuh", ".h", ".hpp")
# A probe line, for the stale-source self-check and for finding the tag.
_PROBE_RE = re.compile(r"KERNEL_TIMER_[A-Z_]+\s*\(")
_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_FNV_OFFSET, _FNV_PRIME = 0x811C9DC5, 0x01000193

# Measured against nvcc 13.2: `__LINE__` inside a function-like macro
# invocation is the line the invocation STARTS on, whether or not the
# arguments wrap.  So a site id resolves to exactly one line -- and a line
# without a probe on it means the sources moved, not that the resolver needs a
# wider net.  Only the tag literal has to be searched forward, for the case
# clang-format puts it on the next line.
_TAG_LOOKAHEAD = 3

DIR_INSTANT, DIR_BEGIN, DIR_END, DIR_TRUNCATED = 0, 1, 2, 3
FLAG_ARG_BIT, ARG_SHIFT = 34, 48
NO_ARG = -1


def _fnv1a32(data: bytes) -> int:
    h = _FNV_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV_PRIME) & 0xFFFFFFFF
    return h


def file_hash16(basename: str) -> int:
    """`TimerFold16(TimerFnv1a(basename))` from device_timer.hpp, verbatim."""
    h = _fnv1a32(basename.encode())
    return ((h >> 16) ^ h) & 0xFFFF


def default_source_roots() -> list[Path]:
    """The trees the MoE probes live in: ../../csrc/{kernel/moe,common}.

    Narrower than all of ``csrc`` on purpose: the other kernels reuse file
    basenames (``utils.cuh``, ``kernel_traits.cuh``), and the site id only
    hashes the basename.
    """
    csrc = Path(__file__).resolve().parents[2] / "csrc"
    return [csrc / "kernel" / "moe", csrc / "common"]


class SiteMap:
    """Resolves a site id back to the probe that wrote it.

    Built by hashing the basename of every kernel source, which is the same
    input the device hashed -- so a run and its sources agree as long as the
    sources have not moved since.  Two self-checks make a disagreement visible
    instead of silently mislabelling the trace:

    * two sources whose basenames collide in 16 bits land in `collisions`;
    * a resolved line that holds no `KERNEL_TIMER_*` probe lands in `stale`,
      which is what an edited (or rebuilt-from-elsewhere) tree looks like.
    """

    def __init__(self, roots: list[Path] | None = None):
        self.collisions: dict[int, list[str]] = {}
        self.stale: list[str] = []
        self.unresolved: set[int] = set()
        self._by_hash: dict[int, list[Path]] = {}
        self._lines: dict[Path, list[str]] = {}
        self._cache: dict[int, dict] = {}
        for root in roots if roots is not None else default_source_roots():
            for path in sorted(Path(root).rglob("*")):
                if path.suffix in _KERNEL_SOURCE_SUFFIXES and path.is_file():
                    self._by_hash.setdefault(file_hash16(path.name), []).append(path)
        self.collisions = {
            h: [str(p) for p in paths]
            for h, paths in self._by_hash.items()
            if len(paths) > 1
        }

    def _source(self, path: Path) -> list[str]:
        if path not in self._lines:
            self._lines[path] = path.read_text(errors="replace").splitlines()
        return self._lines[path]

    @staticmethod
    def _tag(lines: list[str], index: int) -> str | None:
        """First string literal at or just after the probe's opening paren."""
        for text in lines[index : index + _TAG_LOOKAHEAD]:
            found = _STRING_RE.search(text)
            if found:
                return found.group(1)
        return None

    def resolve(self, site_id: int) -> dict:
        """-> {name, file, line, code, resolved}; never raises."""
        site_id = int(site_id)
        if site_id in self._cache:
            return self._cache[site_id]
        line = site_id & 0xFFFF
        paths = self._by_hash.get(site_id >> 16, [])
        info = {
            "name": f"site:{site_id:#010x}",
            "file": None,
            "line": line,
            "code": None,
            "resolved": False,
        }
        for path in paths:
            lines = self._source(path)
            index = line - 1
            if not 0 <= index < len(lines) or not _PROBE_RE.search(lines[index]):
                continue
            info = {
                "name": self._tag(lines, index) or f"{path.name}:{line}",
                "file": path.name,
                "line": line,
                "code": lines[index].strip(),
                "resolved": True,
            }
            break
        else:
            if paths:
                where = f"{paths[0].name}:{line}"
                info["name"], info["file"] = where, paths[0].name
                self.stale.append(where)
            else:
                self.unresolved.add(site_id)
        self._cache[site_id] = info
        return info

    def name(self, site_id: int) -> str:
        return self.resolve(site_id)["name"]


def collect(data: dict):
    """Normalize a host timer export into (records, meta).

    `data` needs `records` (numpy array or torch tensor of int64/uint64 words),
    `ctas`, `warps` and `cap`.  Optional: `warp_roles` (display-only labels
    for the per-warp tracks), `launch` (free-form launch metadata) and
    `sources` (kernel roots to resolve call sites against).
    """
    if not data or data.get("records") is None:
        raise RuntimeError(
            "no timer records: build with ENABLE_KERNEL_TIMER=1 (use "
            "`bash profile.sh`) and run a launch before collecting"
        )
    records = data["records"]
    if hasattr(records, "detach"):
        records = records.detach().cpu().numpy()
    recs = np.asarray(records)
    if recs.dtype not in (np.dtype(np.int64), np.dtype(np.uint64)):
        raise ValueError("timer records must contain int64 or uint64 words")
    meta = {
        "ctas": int(data["ctas"]),
        "warps": int(data["warps"]),
        "cap": int(data["cap"]),
        "warp_roles": {
            int(k): str(v) for k, v in (data.get("warp_roles") or {}).items()
        },
        "launch": dict(data.get("launch") or {}),
    }
    return recs.reshape(-1).copy().view(np.uint64), meta


def decode(recs: np.ndarray, meta: dict):
    """-> (rows, t0).  rows is ordered by (cta, warp, slot), one per record.

    Fields: cta, warp (the track, from the slot's position in the pool), site,
    dir, wrote (the warp id the device stamped -- a cross-check on the layout),
    t, slot, arg (the record's 16-bit tag, or NO_ARG when the probe had none).
    """
    ctas, warps, cap = meta["ctas"], meta["warps"], meta["cap"]
    need = ctas * warps * cap * 2
    if recs.size < need:
        raise ValueError(
            f"timer records hold {recs.size} words, need {need} for "
            f"ctas={ctas} warps={warps} cap={cap}"
        )
    rec = recs[:need].reshape(ctas, warps, cap, 2)
    cta_ax, warp_ax, slot_ax = np.nonzero(rec[..., 1] != 0)
    word = rec[cta_ax, warp_ax, slot_ax, 0]
    has_arg = ((word >> np.uint64(FLAG_ARG_BIT)) & np.uint64(1)).astype(bool)
    arg = np.where(
        has_arg,
        ((word >> np.uint64(ARG_SHIFT)) & np.uint64(0xFFFF)).astype(np.int32),
        np.int32(NO_ARG),
    ).astype(np.int32)
    rows = np.rec.fromarrays(
        [
            cta_ax.astype(np.int32),
            warp_ax.astype(np.int32),
            (word & np.uint64(0xFFFFFFFF)).astype(np.uint32),
            ((word >> np.uint64(32)) & np.uint64(0x3)).astype(np.int8),
            ((word >> np.uint64(40)) & np.uint64(0xFF)).astype(np.int32),
            rec[cta_ax, warp_ax, slot_ax, 1].astype(np.int64),
            slot_ax.astype(np.int32),
            arg,
        ],
        names="cta,warp,site,dir,wrote,t,slot,arg",
    )
    rows.sort(order=("cta", "warp", "slot"))
    return rows, (int(rows["t"].min()) if len(rows) else 0)


def _columns(rows: np.ndarray):
    """Column-wise python lists; ~20x faster to walk than a recarray."""
    return (
        rows["cta"].tolist(),
        rows["warp"].tolist(),
        rows["site"].tolist(),
        rows["dir"].tolist(),
        rows["t"].tolist(),
        rows["arg"].tolist(),
    )


def pair(rows: np.ndarray):
    """Pair begin/end within each (cta, warp) stream, LIFO per site id.

    `begin` hands its own id to `end` on the device, so a pair matches by id
    and nesting: a span whose two ends sit on different lines or in different
    functions still pairs, and two spans of the same name nest correctly.  A
    pair takes its `arg` from the `begin`; an `end` carries none.
    Returns (intervals, stray_ends, unclosed_begins) -- the last two are
    instrumentation bugs (an unbalanced path) and are reported, never hidden.
    """
    intervals: list[dict] = []
    stray: list[tuple] = []
    stacks: dict[tuple, list[tuple]] = {}
    for cta, warp, site, direction, t, arg in zip(*_columns(rows)):
        if direction == DIR_BEGIN:
            stacks.setdefault((cta, warp, site), []).append((t, arg))
        elif direction == DIR_END:
            stack = stacks.get((cta, warp, site))
            if not stack:
                stray.append((cta, warp, site))
                continue
            start, begin_arg = stack.pop()
            intervals.append(
                {
                    "cta": cta,
                    "warp": warp,
                    "site": site,
                    "start": start,
                    "end": t,
                    "dur": t - start,
                    "arg": begin_arg,
                }
            )
    unclosed = [key for key, stack in stacks.items() for _ in stack]
    return intervals, stray, unclosed


def instants(rows: np.ndarray) -> list[tuple]:
    """(cta, warp, site, t, arg) for every mark, in stream order."""
    return [
        (cta, warp, site, t, arg)
        for cta, warp, site, direction, t, arg in zip(*_columns(rows))
        if direction == DIR_INSTANT
    ]


def phase_spans(marks: list[tuple]) -> list[dict]:
    """Gap-free bars between consecutive marks of one warp: "a -> b".

    A mark says when a warp reached a boundary; this says where the time
    between two boundaries went without instrumenting a span for every region.
    Derived from marks only, so it never interleaves with real spans.
    """
    spans: list[dict] = []
    for index in range(len(marks) - 1):
        cta, warp, site, t, arg = marks[index]
        next_cta, next_warp, next_site, next_t, _ = marks[index + 1]
        if cta == next_cta and warp == next_warp and next_t >= t:
            spans.append(
                {
                    "cta": cta,
                    "warp": warp,
                    "site": site,
                    "to_site": next_site,
                    "start": t,
                    "end": next_t,
                    "dur": next_t - t,
                    "arg": arg,
                }
            )
    return spans


def truncated(rows: np.ndarray, sites: SiteMap) -> list[tuple]:
    """(cta, warp, dropped probe) per stream that ran into `cap`.

    The device stamps its last slot the first time it has to drop a record and
    names the probe it dropped, so an overflowing stream says where it
    overflowed rather than just ending early.  Empty means every stream is
    complete.
    """
    hit = rows["dir"] == DIR_TRUNCATED
    return sorted(
        {
            (int(cta), int(warp), sites.name(int(site)))
            for cta, warp, site in zip(
                rows["cta"][hit], rows["warp"][hit], rows["site"][hit]
            )
        }
    )


def _tracks(rows: np.ndarray, meta: dict) -> list[dict]:
    """Chrome-trace metadata so perfetto shows "CTA 7 / warp 8 mma" tracks."""
    roles = meta.get("warp_roles") or {}
    out: list[dict] = []
    for cta in sorted({int(c) for c in rows["cta"].tolist()}):
        out.append(
            {
                "ph": "M",
                "name": "process_name",
                "pid": cta,
                "args": {"name": f"CTA {cta:03d}"},
            }
        )
        out.append(
            {
                "ph": "M",
                "name": "process_sort_index",
                "pid": cta,
                "args": {"sort_index": cta},
            }
        )
    for cta, warp in sorted(
        {(int(c), int(w)) for c, w in zip(rows["cta"].tolist(), rows["warp"].tolist())}
    ):
        role = roles.get(warp)
        out.append(
            {
                "ph": "M",
                "name": "thread_name",
                "pid": cta,
                "tid": warp,
                "args": {"name": f"warp {warp:02d}" + (f" {role}" if role else "")},
            }
        )
        out.append(
            {
                "ph": "M",
                "name": "thread_sort_index",
                "pid": cta,
                "tid": warp,
                "args": {"sort_index": warp},
            }
        )
    return out


def _event_args(info: dict, arg: int) -> dict:
    args = {"at": f"{info['file']}:{info['line']}"}
    if arg != NO_ARG:
        args["arg"] = arg
        args["probe"] = info["name"]
    return args


def _event_name(name: str, arg: int) -> str:
    return name if arg == NO_ARG else f"{name}({arg})"


def to_perfetto(intervals, marks, phases, sites: SiteMap, t0: int) -> list[dict]:
    """Chrome Trace Format events, timestamps relative to t0 (ns).

    A record that carries an arg is shown as "<probe>(<arg>)" with the arg also
    in the event's args; what the number means is the probe's business (an
    expert id, a K-loop iteration, ...).  `args["probe"]` keeps the bare probe
    name so a flat table can still group by probe.
    """
    out: list[dict] = []
    for span in intervals:
        info = sites.resolve(span["site"])
        arg = span.get("arg", NO_ARG)
        out.append(
            {
                "ph": "X",
                "name": _event_name(info["name"], arg),
                "cat": info["file"] or "kernel",
                "pid": span["cta"],
                "tid": span["warp"],
                "ts": span["start"] - t0,
                "dur": max(span["dur"], 1),
                "args": _event_args(info, arg),
            }
        )
    for cta, warp, site, t, arg in marks:
        info = sites.resolve(site)
        out.append(
            {
                "ph": "i",
                "name": _event_name(info["name"], arg),
                "cat": info["file"] or "kernel",
                "pid": cta,
                "tid": warp,
                "ts": t - t0,
                "s": "t",
                "args": _event_args(info, arg),
            }
        )
    for span in phases:
        args = {"derived": "between consecutive marks"}
        if span.get("arg", NO_ARG) != NO_ARG:
            args["arg"] = span["arg"]
        out.append(
            {
                "ph": "X",
                "name": f"{sites.name(span['site'])} -> {sites.name(span['to_site'])}",
                "cat": "phase",
                "pid": span["cta"],
                "tid": span["warp"],
                "ts": span["start"] - t0,
                "dur": max(span["dur"], 1),
                "args": args,
            }
        )
    return out


def _percentiles(values) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.percentile(array, 50)),
        float(np.percentile(array, 90)),
        float(array.max()),
        float(array.sum()),
    )


def _by_name(items, sites: SiteMap, key="site") -> dict[str, list]:
    grouped: dict[str, list] = {}
    for item in items:
        site = item[key] if isinstance(item, dict) else item[2]
        grouped.setdefault(sites.name(site), []).append(item)
    return grouped


def _report_health(rows, sites: SiteMap, meta, stray, unclosed) -> None:
    """Everything that would make the trace a lie, on stderr."""
    mismatched = int(np.count_nonzero(rows["wrote"] != rows["warp"]))
    if mismatched:
        print(
            f"[stats] {mismatched} records whose stamped warp disagrees with "
            f"their slot -- the pool geometry (warps={meta['warps']} "
            f"cap={meta['cap']}) does not match the build",
            file=sys.stderr,
        )
    for h, paths in sites.collisions.items():
        print(
            f"[stats] basename hash {h:#06x} collides: {', '.join(paths)} -- "
            f"probe names from these files may be attributed to the wrong one",
            file=sys.stderr,
        )
    if sites.stale:
        shown = ", ".join(sorted(set(sites.stale))[:5])
        print(
            f"[stats] {len(set(sites.stale))} site ids resolved to a line with "
            f"no KERNEL_TIMER_* probe ({shown}) -- the sources changed since "
            f"this run, so labels are unreliable; re-run `bash profile.sh`",
            file=sys.stderr,
        )
    if sites.unresolved:
        print(
            f"[stats] {len(sites.unresolved)} site ids from a file that is not "
            f"under the scanned kernel roots; they keep their raw ids",
            file=sys.stderr,
        )
    for label, bad in (("unmatched end", stray), ("unclosed begin", unclosed)):
        if bad:
            names = sorted({sites.name(key[2]) for key in bad})
            print(
                f"[stats] {len(bad)} {label} records ({', '.join(names[:5])}) "
                f"-- a span opens and closes on unbalanced paths",
                file=sys.stderr,
            )


def stats(rows, intervals, marks, sites: SiteMap, meta, *, stray=(), unclosed=()):
    """Span durations, per-boundary warp skew, and every health warning."""
    print(
        f"[stats] ctas={meta['ctas']} warps={meta['warps']} cap={meta['cap']} "
        f"records={len(rows)} spans={len(intervals)} marks={len(marks)}"
    )
    _report_health(rows, sites, meta, stray, unclosed)
    hit_cap = truncated(rows, sites)
    if hit_cap:
        dropped = sorted({name for _, _, name in hit_cap})
        print(
            f"[stats] TRUNCATED: {len(hit_cap)} warp streams hit cap="
            f"{meta['cap']} and end early; first dropped probe(s): "
            f"{', '.join(dropped[:5])}",
            file=sys.stderr,
        )

    by_name = _by_name(marks, sites)
    # calibrate() emits two back-to-back reads per CTA, so pair them within a
    # CTA rather than across the run.
    per_cta_calib: dict[int, list[int]] = {}
    for cta, _, _, t, _ in by_name.get("calib_timer", []):
        per_cta_calib.setdefault(cta, []).append(t)
    reads: list[int] = []
    for times in per_cta_calib.values():
        ordered = sorted(times)
        reads += [b - a for a, b in zip(ordered[0::2], ordered[1::2])]
    if reads:
        print(f"[stats] %globaltimer read: {int(np.median(reads))} ns (median)")
    pair_cost = [
        span["dur"] for span in intervals if sites.name(span["site"]) == "calib_pair"
    ]
    if pair_cost:
        print(
            f"[stats] empty begin/end pair: {int(np.median(pair_cost))} ns "
            f"(median) -- subtract it from every span below"
        )

    spans_by_name = _by_name(intervals, sites)
    if spans_by_name:
        print(
            f"[stats] {'span':<28s} {'n':>6s} {'p50(ns)':>9s} {'p90':>9s} "
            f"{'max':>9s} {'sum(us)':>9s}"
        )
        for name in sorted(spans_by_name):
            durations = [span["dur"] for span in spans_by_name[name]]
            p50, p90, top, total = _percentiles(durations)
            print(
                f"[stats] {name:<28s} {len(durations):6d} {p50:9.0f} "
                f"{p90:9.0f} {top:9.0f} {total / 1e3:9.1f}"
            )

    # Warp skew at each boundary: how far apart the warps of one CTA reach the
    # same mark.  This is what a per-warp trace is for -- a large skew is the
    # barrier or handoff that is actually costing time.  A tagged mark (one per
    # expert) is compared only against the same tag, never across tags.
    skews: dict[str, list[int]] = {}
    for name, items in by_name.items():
        per_cta: dict[tuple, list[int]] = {}
        for cta, _, _, t, arg in items:
            per_cta.setdefault((cta, arg), []).append(t)
        for times in per_cta.values():
            if len(times) > 1:
                skews.setdefault(name, []).append(max(times) - min(times))
    if skews:
        print(
            f"[stats] {'mark (warp skew)':<28s} {'groups':>6s} {'p50(ns)':>9s} {'p90':>9s} {'max':>9s}"
        )
        for name in sorted(skews):
            p50, p90, top, _ = _percentiles(skews[name])
            print(
                f"[stats] {name:<28s} {len(skews[name]):6d} {p50:9.0f} {p90:9.0f} {top:9.0f}"
            )


def collect_and_write(
    out_path,
    do_stats: bool = True,
    *,
    data: dict | None = None,
    phases: bool = False,
    sources=None,
):
    """Decode a timer export, resolve its call sites, write the trace JSON.

    `phases=True` adds a derived bar between each pair of consecutive marks in
    a warp's stream, which turns a trace of instants into a gap-free timeline
    at the cost of roughly one extra event per mark.
    """
    recs, meta = collect(data)
    sites = SiteMap(sources if sources is not None else (data or {}).get("sources"))
    rows, t0 = decode(recs, meta)
    intervals, stray, unclosed = pair(rows)
    marks = instants(rows)
    derived = phase_spans(marks) if phases else []
    trace = {
        "traceEvents": _tracks(rows, meta)
        + to_perfetto(intervals, marks, derived, sites, t0),
        "metadata": {
            "ctas": meta["ctas"],
            "warps": meta["warps"],
            "cap": meta["cap"],
            "launch": meta["launch"],
            "t0_ns": t0,
            "clock_domain": "globaltimer_ns",
            # The decoded probe table, so the JSON stays readable without the
            # sources: id -> {name, file, line, code}.
            "sites": {
                f"{site:#010x}": sites.resolve(site)
                for site in sorted({int(s) for s in rows["site"].tolist()})
            },
        },
    }
    with open(out_path, "w") as handle:
        json.dump(trace, handle)
    print(f"[trace] wrote {out_path} ({len(trace['traceEvents'])} events)")
    if do_stats:
        stats(rows, intervals, marks, sites, meta, stray=stray, unclosed=unclosed)
    return trace


def write_csv(trace: dict, path) -> None:
    """Flat span table for pandas: one row per bar in the trace.

    `name` is the bare probe name and `arg` its tag (empty when untagged), so a
    groupby on `name` does not fan out into one group per expert.
    """
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cta", "warp", "name", "cat", "ts_ns", "dur_ns", "at", "arg"])
        for event in trace["traceEvents"]:
            if event.get("ph") == "X":
                args = event.get("args", {})
                writer.writerow(
                    [
                        event["pid"],
                        event["tid"],
                        args.get("probe", event["name"]),
                        event.get("cat", ""),
                        event["ts"],
                        event["dur"],
                        args.get("at", ""),
                        args.get("arg", ""),
                    ]
                )
    print(f"[trace] wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Decode a timer export JSON dump (see profile.sh)."
    )
    parser.add_argument("records", help="npy file of int64/uint64 record words")
    parser.add_argument("--ctas", type=int, required=True)
    parser.add_argument("--warps", type=int, required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--out", default="timer.json")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--phases", action="store_true")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    trace = collect_and_write(
        args.out,
        args.stats,
        data={
            "records": np.load(args.records),
            "ctas": args.ctas,
            "warps": args.warps,
            "cap": args.cap,
        },
        phases=args.phases,
    )
    if args.csv:
        write_csv(trace, args.csv)


if __name__ == "__main__":
    main()
