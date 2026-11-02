"""Interleaved A/B comparison of the same benchmark executable from two build directories.

    python -I ab_compare.py --a orig=BUILD_A --b opt=BUILD_B --out DIR --name NAME \
        [--rounds 5] [--warmup-rounds 1] [--budget-s 480] [--gpu-sample-ms 200] \
        [--load-gate 25] [--extract key=REGEX] -- EXE [args...]

Runs EXE from BUILD_A and BUILD_B alternately (A B A B ...), each as a fresh process, and records
per process:
  * wall_ms           whole-process elapsed time = cold start (DLL load, CUDA context, first calls)
  * peak_ws_mb        process peak working set (Windows GetProcessMemoryInfo, read after exit)
  * cpu_s / cpu_cores user+kernel CPU time of the process and its mean core usage
  * gpu_util_mean/max device-wide utilization.gpu sampled by nvidia-smi every --gpu-sample-ms
    gpu_mem_max_mb    device-wide memory.used peak (includes other processes on the GPU!)
  * every "key   number" stdout line (keys may contain single spaces) and, if an argument contains
    {json}, every number of the JSON file the benchmark writes there (flattened, list items keyed by
    their name/device/count fields). In-process warm-up/timed repeats are the benchmark's own.
  * --extract key=REGEX: group 1 of the first stdout match, as a number (e.g. epoch time).
{tmp} in an argument is replaced by a fresh per-process scratch directory (deleted afterwards);
the process also runs with that directory as its working directory. EXE ending in .py runs with
the current Python interpreter.

Resumable: NAME.json in --out keeps every run; a rerun appends rounds until --rounds timed rounds
exist. Each invocation first runs --warmup-rounds discarded A B pairs (file cache, driver JIT).
--budget-s stops before a pair that would not fit (exit 3 = incomplete, rerun to continue), so each
invocation can live inside one benchmark-lock hold. Results: NAME.json (raw + summary), NAME.md
(table: median [min-max] per side, % change of B vs A), NAME.<side>.<run>.txt (stdout),
NAME.<side>.<run>.gpu.csv (nvidia-smi samples), NAME.env.txt (load gate, GPU processes per block).
Standard library only.
"""
import argparse
import ctypes
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

LINE = re.compile(r"^\s*([A-Za-z_][\w.:/-]*(?: [\w.:/-]+)*)\s{2,}([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$")
WIN = sys.platform == "win32"


def now():
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- process metrics (Windows)
if WIN:
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    _psapi = ctypes.WinDLL("psapi")
    _k32 = ctypes.WinDLL("kernel32")


def process_stats(popen):
    """Peak working set (MB) and CPU seconds of an exited process (handle still open)."""
    if not WIN:
        return {}
    handle = wintypes.HANDLE(int(popen._handle))
    out = {}
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if _psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
        out["peak_ws_mb"] = pmc.PeakWorkingSetSize / 2**20
    ft = [wintypes.FILETIME() for _ in range(4)]
    if _k32.GetProcessTimes(handle, *[ctypes.byref(f) for f in ft]):
        val = lambda f: ((f.dwHighDateTime << 32) | f.dwLowDateTime) / 1e7
        out["cpu_s"] = val(ft[2]) + val(ft[3])
    return out


# ---------------------------------------------------------------- metric parsing
def flatten(obj, prefix, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            tag = str(i)
            if isinstance(v, dict):
                ids = [str(v[k]) for k in ("name", "device", "count") if k in v]
                if ids:
                    tag = "@".join(ids)
            flatten(v, f"{prefix}[{tag}]", out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and math.isfinite(obj):
        out[prefix] = float(obj)


def parse_stdout(text, extract):
    vals = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if m:
            vals.setdefault(m.group(1).strip().replace(" ", "_"), float(m.group(2)))
    for key, rx in extract:
        m = re.search(rx, text, re.MULTILINE)
        if m:
            vals[key] = float(m.group(1))
    return vals


# ---------------------------------------------------------------- environment probes
def gpu_sampler(path, period_ms):
    if not shutil.which("nvidia-smi"):
        return None
    f = open(path, "w", encoding="utf-8")
    p = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used",
                          "--format=csv,noheader,nounits", f"-lms={period_ms}"],
                         stdout=f, stderr=subprocess.DEVNULL)
    return p, f


def gpu_summary(path):
    util, mem = [], []
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3:
                try:
                    util.append(float(parts[1]))
                    mem.append(float(parts[2]))
                except ValueError:
                    pass
    except OSError:
        pass
    if not util:
        return {}
    return {"gpu_util_mean": statistics.fmean(util), "gpu_util_max": max(util),
            "gpu_mem_max_mb": max(mem), "gpu_samples": float(len(util))}


def cpu_load(seconds=30):
    """Mean total CPU % over `seconds` (Windows typeperf), or None."""
    if not WIN or not shutil.which("typeperf"):
        return None
    n = max(1, seconds // 3)
    try:
        out = subprocess.run(["typeperf", r"\Processor(_Total)\% Processor Time", "-si", "3", "-sc", str(n)],
                             capture_output=True, text=True, timeout=seconds + 30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    vals = []
    for line in out.splitlines():
        parts = line.strip().strip('"').split('","')
        if len(parts) == 2:
            try:
                vals.append(float(parts[1]))
            except ValueError:
                pass
    return statistics.fmean(vals) if vals else None


def env_snapshot(tag):
    lines = [f"===== {now()} {tag}"]
    for cmd in (["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm",
                 "--format=csv"],
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                ["nvidia-smi", "pmon", "-c", "2"]):
        if shutil.which(cmd[0]):
            try:
                lines.append(subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.rstrip())
            except (OSError, subprocess.TimeoutExpired) as e:
                lines.append(f"{cmd[0]} failed: {e}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- one benchmark process
def run_once(exe, args, outdir, name, side, tag, cfg):
    scratch = Path(tempfile.mkdtemp(prefix=f"{name}.{side}.{tag}.", dir=outdir))
    json_path = outdir / f"{name}.{side}.{tag}.out.json"
    argv = [str(exe)] + [a.replace("{json}", str(json_path)).replace("{tmp}", str(scratch)) for a in args]
    if Path(exe).suffix == ".py":  # script benchmarks (and the self-test) run with this interpreter
        argv = [sys.executable, "-I"] + argv
    gpu_csv = outdir / f"{name}.{side}.{tag}.gpu.csv"
    sampler = gpu_sampler(gpu_csv, cfg.gpu_sample_ms) if cfg.gpu_sample_ms > 0 else None
    t0 = time.perf_counter()
    p = subprocess.Popen(argv, cwd=scratch, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        out, _ = p.communicate(timeout=cfg.timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        timed_out = True
    wall = (time.perf_counter() - t0) * 1000.0
    stats = process_stats(p)
    if sampler:
        sampler[0].terminate()
        sampler[0].wait()
        sampler[1].close()
    text = out.decode("utf-8", "replace")
    (outdir / f"{name}.{side}.{tag}.txt").write_text(f"$ {' '.join(argv)}\n# exit {p.returncode} wall_ms {wall:.1f}\n{text}",
                                                     encoding="utf-8")
    rec = {"side": side, "run": tag, "argv": argv, "exit": p.returncode, "timed_out": timed_out,
           "started": now(), "wall_ms": wall}
    metrics = parse_stdout(text, cfg.extract)
    if json_path.exists():
        try:
            flatten(json.loads(json_path.read_text(encoding="utf-8")), "", metrics)
        except (ValueError, OSError) as e:
            rec["json_error"] = str(e)
    metrics["wall_ms"] = wall
    metrics.update(stats)
    if "cpu_s" in stats and wall > 0:
        metrics["cpu_cores"] = stats["cpu_s"] / (wall / 1000.0)
    metrics.update(gpu_summary(gpu_csv))
    rec["metrics"] = metrics
    shutil.rmtree(scratch, ignore_errors=True)
    return rec


# ---------------------------------------------------------------- summary
def summarize(state):
    sides = [s for s, _ in state["sides"]]
    timed = [r for r in state["runs"] if not r["warmup"] and r["exit"] == 0 and not r["timed_out"]]
    summary = {}
    for side in sides:
        rs = [r["metrics"] for r in timed if r["side"] == side]
        keys = sorted({k for m in rs for k in m})
        summary[side] = {}
        for k in keys:
            vs = [m[k] for m in rs if k in m]
            summary[side][k] = {"median": statistics.median(vs), "min": min(vs), "max": max(vs), "n": len(vs),
                                "stdev": statistics.stdev(vs) if len(vs) > 1 else 0.0}
    state["summary"] = summary
    return summary


def fmt(v):
    a = abs(v)
    return f"{v:.0f}" if a >= 1000 else f"{v:.1f}" if a >= 100 else f"{v:.2f}" if a >= 1 else f"{v:.4g}"


def markdown(state, keys_filter=None):
    (na, _), (nb, _) = state["sides"]
    sa, sb = state["summary"].get(na, {}), state["summary"].get(nb, {})
    lines = [f"### {state['name']}", "", f"`{' '.join(state['cmd'])}`", "",
             f"| metric | {na} median [min-max] | {nb} median [min-max] | change | n |", "|---|---|---|---|---|"]
    for k in sorted(set(sa) & set(sb)):
        if k.endswith(("_min_ms", "_max_ms")) or (keys_filter and not re.search(keys_filter, k)):
            continue
        a, b = sa[k], sb[k]
        if a["min"] == a["max"] == b["min"] == b["max"] and not k.startswith(("wall", "peak", "cpu", "gpu")):
            continue  # identical constants (counts, sizes) are listed separately
        pct = (b["median"] - a["median"]) / a["median"] * 100 if a["median"] else float("nan")
        lines.append(f"| {k} | {fmt(a['median'])} [{fmt(a['min'])}-{fmt(a['max'])}] | "
                     f"{fmt(b['median'])} [{fmt(b['min'])}-{fmt(b['max'])}] | {pct:+.1f} % | {a['n']}/{b['n']} |")
    same = [k for k in sorted(set(sa) & set(sb)) if sa[k]["min"] == sa[k]["max"] == sb[k]["min"] == sb[k]["max"]
            and not k.startswith(("wall", "peak", "cpu", "gpu"))]
    diff = [k for k in sorted(set(sa) ^ set(sb))]
    if same:
        lines += ["", "identical on both sides: " + ", ".join(f"{k}={fmt(sa[k]['median'])}" for k in same[:40])]
    if diff:
        lines += ["", "only on one side: " + ", ".join(diff[:40])]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main
def resolve(build, exe):
    p = Path(build) / exe
    if WIN and not p.suffix and not p.is_file():
        p = p.with_suffix(".exe")
    if not p.is_file():
        raise SystemExit(f"missing executable {p}")
    return p.resolve()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="NAME=BUILD_DIR (baseline)")
    ap.add_argument("--b", required=True, help="NAME=BUILD_DIR (candidate)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rounds", type=int, default=5, help="timed A B rounds in total (default 5)")
    ap.add_argument("--warmup-rounds", type=int, default=1, help="discarded A B rounds per invocation")
    ap.add_argument("--budget-s", type=float, default=0, help="stop before a pair that would exceed this")
    ap.add_argument("--gpu-sample-ms", type=int, default=200, help="0 disables all nvidia-smi probes")
    ap.add_argument("--load-gate", type=float, default=0,
                    help="wait (3 x 30 s max) while total CPU %% is above this; 0 = off")
    ap.add_argument("--timeout", type=float, default=900, help="per-process limit in seconds")
    ap.add_argument("--extract", action="append", default=[], help="key=REGEX on stdout (group 1)")
    ap.add_argument("--label", default="", help="free text stored with this invocation (e.g. contention)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("benchmark command required after --")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", a.name):
        ap.error("--name must be a simple file name")
    extract = []
    for e in a.extract:
        k, _, rx = e.partition("=")
        extract.append((k, rx))
    a.extract = extract
    sides = [tuple(s.split("=", 1)) for s in (a.a, a.b)]
    exes = {n: resolve(d, cmd[0]) for n, d in sides}
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / f"{a.name}.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["cmd"] != cmd or [tuple(s) for s in state["sides"]] != sides:
            raise SystemExit(f"{state_path} exists for a different command/sides")
    else:
        state = {"schema": 1, "name": a.name, "cmd": cmd, "sides": sides, "runs": [], "blocks": []}
    state["exe"] = {n: str(p) for n, p in exes.items()}
    envlog = out / f"{a.name}.env.txt"

    def save():
        summarize(state)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
        os.replace(tmp, state_path)
        (out / f"{a.name}.md").write_text(markdown(state), encoding="utf-8")

    block = {"started": now(), "label": a.label, "load_samples": []}
    state["blocks"].append(block)
    if a.load_gate > 0:
        for i in range(3):
            load = cpu_load(30)
            block["load_samples"].append(load)
            if load is None or load < a.load_gate:
                break
        block["gate"] = "open" if block["load_samples"] and block["load_samples"][-1] is not None and \
            block["load_samples"][-1] < a.load_gate else "timeout-contended"
    probe = a.gpu_sample_ms > 0  # --gpu-sample-ms 0 also skips the nvidia-smi snapshots
    with envlog.open("a", encoding="utf-8") as f:
        f.write(env_snapshot(f"block {len(state['blocks'])} before; load {block['load_samples']} "
                             f"gate {block.get('gate')} label {a.label!r}") if probe else
                f"===== {now()} block {len(state['blocks'])} label {a.label!r}\n")
    started = time.perf_counter()
    done = lambda: sum(1 for r in state["runs"] if not r["warmup"] and r["side"] == sides[1][0] and r["exit"] == 0)
    pair_s = 0.0
    plan = ["warmup"] * a.warmup_rounds + ["timed"] * max(0, a.rounds - done())
    rc = 0
    for kind in plan:
        if kind == "timed" and done() >= a.rounds:
            break
        elapsed = time.perf_counter() - started
        if a.budget_s and pair_s and elapsed + pair_s * 1.15 > a.budget_s:
            rc = 3
            break
        t_pair = time.perf_counter()
        for n, _ in sides:
            idx = sum(1 for r in state["runs"] if r["side"] == n)
            tag = f"w{idx}" if kind == "warmup" else f"r{idx}"
            rec = run_once(exes[n], cmd[1:], out, a.name, n, tag, a)
            rec["warmup"] = kind == "warmup"
            rec["block"] = len(state["blocks"])
            state["runs"].append(rec)
            save()
            print(f"{now()} {a.name} {n} {tag} exit {rec['exit']} wall {rec['wall_ms'] / 1000:.1f}s", flush=True)
            if rec["exit"] != 0 or rec["timed_out"]:
                rc = 1
        if rc == 1:
            break
        pair_s = time.perf_counter() - t_pair
    block["ended"] = now()
    block["timed_rounds_total"] = done()
    if probe:
        with envlog.open("a", encoding="utf-8") as f:
            f.write(env_snapshot(f"block {len(state['blocks'])} after"))
    save()
    if rc == 0 and done() < a.rounds:
        rc = 3
    print(f"{now()} {a.name} block done rc={rc} timed rounds {done()}/{a.rounds}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
