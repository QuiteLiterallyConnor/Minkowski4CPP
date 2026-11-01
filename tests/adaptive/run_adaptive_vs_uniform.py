#!/usr/bin/env python3
"""Driver for adaptive_vs_uniform_bench: one process per (dataset, depth, method, device) so the
peak-RAM numbers and the process-cold first-call timings belong to a single case.

  python tests/adaptive/run_adaptive_vs_uniform.py --exe build/adaptive_vs_uniform_bench \
      --out results/avu --devices cpu,cuda [--datasets ...] [--methods ...] [--depths 3,4,5] \
      [--extra "--points 100000 --reps 5"]

Writes <out>/cases.jsonl (raw), <out>/cases.csv (flattened) and <out>/summary.md (tables).
--summarize-only rebuilds csv/md from an existing cases.jsonl. Standard library only.
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time

DATASETS = "uniform,clustered,shell,ply,scan"
METHODS = "none,u_coarse,u_fine,u_matched,a_roi,a_dens4,a_dens16,a_dens64,a_var,a_fnorm"


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def run(args):
    os.makedirs(args.out, exist_ok=True)
    jsonl = os.path.join(args.out, "cases.jsonl")
    extra = shlex.split(args.extra)
    for depth in args.depths.split(","):
        for ds in args.datasets.split(","):
            for m in args.methods.split(","):
                devices = ["cpu"] if m == "none" else args.devices.split(",")
                for dev in devices:
                    cmd = [args.exe, "--datasets", ds, "--methods", m, "--devices", dev,
                           "--depth", depth, "--jsonl", jsonl] + extra
                    t0 = time.time()
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    dt = time.time() - t0
                    status = "ok" if r.returncode == 0 else f"exit {r.returncode}"
                    print(f"[{dt:6.1f}s] {status} depth={depth} {ds} {m} {dev}", flush=True)
                    if r.returncode != 0:
                        with open(os.path.join(args.out, "failures.txt"), "a") as f:
                            f.write(" ".join(cmd) + "\n" + r.stdout[-2000:] + r.stderr[-4000:] + "\n")


def load(out):
    rows = []
    with open(os.path.join(out, "cases.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def g(d, path, default=None):
    for p in path.split("."):
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def fmt(v, nd=1):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def ms(d, path):
    s = g(d, path)
    if not isinstance(s, dict):
        return "n/a"
    return f"{fmt(s['med'])} ({fmt(s['min'])}-{fmt(s['max'])})"


def summarize(out):
    rows = load(out)
    # A re-run of a case (same dataset/method/device/depth) replaces the earlier row.
    latest = {}
    for r in rows:
        if r.get("kind") == "case":
            latest[(r["dataset"], r["method"], r["device"], r["depth"])] = r
    cases = list(latest.values())
    flat = [flatten(r) for r in cases]
    keys = []
    for r in flat:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(out, "cases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in flat:
            w.writerow(r)

    by = {}
    floor = {}
    for r in cases:
        if r["method"] == "none":
            floor[(r["dataset"], r["depth"])] = r.get("peak_ram_mb")
            continue
        by.setdefault((r["dataset"], r["depth"]), {}).setdefault(r["method"], {})[r["device"]] = r

    lines = ["# adaptive_vs_uniform_bench summary", ""]
    for (ds, depth), methods in by.items():
        any_case = next(iter(next(iter(methods.values())).values()))
        lines.append(f"## {ds}, depth {depth}, base {any_case['base']} "
                     f"({any_case['train_points']:,} train points)")
        lines.append("")
        lines.append(f"_{any_case.get('dataset_note', '')}_; peak RAM with data only: "
                     f"{fmt(floor.get((ds, depth)))} MB")
        lines.append("")
        lines.append("| method | voxels | levels | feat+coord MB | kmap pairs | nbrs | build ms | "
                     "coverage | nRMSE nearest | RMSE multilinear |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for m, devs in methods.items():
            r = devs.get("cpu") or next(iter(devs.values()))
            mb = (r["feature_bytes"] + r["coord_bytes_adaptive"]) / 2**20
            lines.append("| " + " | ".join([
                m, fmt(r["voxels"]), json.dumps(r["levels"]).replace(" ", ""), fmt(mb, 2),
                fmt(g(r, "adaptive_engine.kmap_pairs")), fmt(g(r, "adaptive_engine.avg_neighbors")),
                ms(r, "build_ms"), fmt(g(r, "accuracy.coverage"), 3),
                fmt(g(r, "accuracy.nrmse_nearest_all"), 3),
                fmt(g(r, "accuracy.rmse_multilinear_all"), 3)]) + " |")
        lines.append("")
        for dev in ("cpu", "cuda"):
            if not any(dev in d for d in methods.values()):
                continue
            lines.append(f"**{dev.upper()}** (ms, median (min-max); adaptive engine for all methods, "
                         "MinkowskiEngine conv for uniform ones)")
            lines.append("")
            lines.append("| method | kmap cold | first conv | fwd | fwd+bwd | mink first | mink fwd | "
                         "mink fwd+bwd | mink kmap est | peak RAM MB | peak VRAM alloc/reserved MB |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for m, devs in methods.items():
                r = devs.get(dev)
                if not r:
                    continue
                vram = "n/a"
                if r.get("peak_vram_alloc_mb") is not None:
                    vram = f"{fmt(r['peak_vram_alloc_mb'])}/{fmt(r['peak_vram_reserved_mb'])}"
                lines.append("| " + " | ".join([
                    m, ms(r, "adaptive_engine.kmap_cold_ms"), fmt(g(r, "adaptive_engine.first_conv_ms")),
                    ms(r, "adaptive_engine.fwd_ms"), ms(r, "adaptive_engine.fwd_bwd_ms"),
                    fmt(g(r, "minkowski_engine.mink_first_conv_ms")), ms(r, "minkowski_engine.mink_fwd_ms"),
                    ms(r, "minkowski_engine.mink_fwd_bwd_ms"), fmt(g(r, "minkowski_engine.mink_kmap_est_ms")),
                    fmt(r.get("peak_ram_mb")), vram]) + " |")
            lines.append("")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}/cases.csv and {out}/summary.md ({len(cases)} cases)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exe", default="adaptive_vs_uniform_bench")
    p.add_argument("--out", required=True)
    p.add_argument("--datasets", default=DATASETS)
    p.add_argument("--methods", default=METHODS)
    p.add_argument("--devices", default="cpu")
    p.add_argument("--depths", default="4")
    p.add_argument("--extra", default="")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    if not args.summarize_only:
        run(args)
    summarize(args.out)


if __name__ == "__main__":
    sys.exit(main())
