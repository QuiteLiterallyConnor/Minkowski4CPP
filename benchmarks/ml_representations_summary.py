"""Aggregate ml_representations JSON runs into Markdown tables (mean +- std over seeds).

    python benchmarks/ml_representations_summary.py <dir-with-run-jsons> [--csv out.csv]
"""
import json
import statistics
import sys
from pathlib import Path

METRICS = [
    ("accuracy", "accuracy", 100.0, 1),
    ("macro_precision", "macro P", 100.0, 1),
    ("macro_recall", "macro R", 100.0, 1),
    ("macro_f1", "macro F1", 100.0, 1),
    ("train_time_s", "train s", 1.0, 1),
    ("inference_latency_ms_median_b1", "latency b1 ms", 1.0, 2),
    ("throughput_samples_per_s", "throughput /s", 1.0, 0),
    ("peak_cuda_alloc_mb_train", "peak VRAM MB", 1.0, 0),
    ("peak_rss_mb", "peak RAM MB", 1.0, 0),
    ("active_elements_mean", "active/sample", 1.0, 0),
    ("conversion_ms_median", "convert ms", 1.0, 2),
    ("parameters", "params", 1.0, 0),
]
ORDER = ["uniform", "adaptive", "gaussian", "adaptive_gaussian"]


def fmt(values, scale, digits):
    v = [x * scale for x in values]
    m = statistics.fmean(v)
    s = statistics.stdev(v) if len(v) > 1 else 0.0
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def main():
    root = Path(sys.argv[1])
    runs = {}
    for p in sorted(root.glob("*.json")):
        try:
            j = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if "method" in j and "accuracy" in j:
            runs.setdefault(j["method"], []).append(j)
    methods = [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER]
    print("| method | seeds | " + " | ".join(h for _, h, _, _ in METRICS) + " |")
    print("|---" * (len(METRICS) + 2) + "|")
    for m in methods:
        rs = runs[m]
        cells = [fmt([r[k] for r in rs], sc, d) for k, _, sc, d in METRICS]
        print(f"| {m} | {','.join(str(r['seed']) for r in rs)} | " + " | ".join(cells) + " |")
    print()
    epochs = max(len(r["curve"]) for rs in runs.values() for r in rs)
    print("Test accuracy (%) per epoch, mean over seeds:")
    print()
    print("| epoch | " + " | ".join(methods) + " |")
    print("|---" * (len(methods) + 1) + "|")
    for e in range(epochs):
        row = []
        for m in methods:
            v = [r["curve"][e]["test_acc"] * 100 for r in runs[m] if len(r["curve"]) > e]
            row.append(f"{statistics.fmean(v):.1f}" if v else "")
        print(f"| {e + 1} | " + " | ".join(row) + " |")
    if "--csv" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--csv") + 1])
        with out.open("w") as f:
            f.write("method,seed," + ",".join(k for k, _, _, _ in METRICS) + "\n")
            for m in methods:
                for r in runs[m]:
                    f.write(f"{m},{r['seed']}," + ",".join(str(r[k]) for k, _, _, _ in METRICS) + "\n")


if __name__ == "__main__":
    main()
