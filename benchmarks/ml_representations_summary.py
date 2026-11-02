"""Aggregate comparable ml_representations runs (mean +- std over matched seeds).

    python benchmarks/ml_representations_summary.py <dir-with-run-jsons> [--csv out.csv]
        [--methods uniform,adaptive,gaussian,adaptive_gaussian] [--seeds 1,2,3]

The directory must contain only complete run JSONs. All four planned methods are
required by default; --methods declares an intentional subset. Methods must use
the same seed cohort and shared controls. --seeds declares the required cohort;
without it this checks observed cohorts, not a seed absent from every method.
Parameter counts/input widths may differ between methods. The legacy *_train
VRAM fields include per-epoch evaluation, so the table labels that measured scope.
"""
import argparse
import csv
import json
import math
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
    ("peak_cuda_alloc_mb_train", "peak VRAM train+eval MB", 1.0, 0),
    ("peak_rss_mb", "peak RAM MB", 1.0, 0),
    ("active_elements_mean", "active/sample", 1.0, 0),
    ("conversion_ms_median", "convert ms", 1.0, 2),
    ("parameters", "params", 1.0, 0),
]
ORDER = ["uniform", "adaptive", "gaussian", "adaptive_gaussian"]
SHARED_CONTROLS = (
    "device", "data_seed", "epochs", "batch_size", "lr", "train_samples",
    "test_samples", "points_per_sample",
)
ACCURACIES = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
MEMORY = ("peak_cuda_alloc_mb_train", "peak_rss_mb")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(record, key, minimum=None):
    value = record.get(key)
    require(type(value) is int, f"{key} must be an integer")
    if minimum is not None:
        require(value >= minimum, f"{key} must be >= {minimum}")
    return value


def number(record, key, minimum=0.0, maximum=None):
    value = record.get(key)
    require(type(value) in (int, float) and math.isfinite(value), f"{key} must be finite numeric data")
    require(value >= minimum, f"{key} must be >= {minimum}")
    if maximum is not None:
        require(value <= maximum, f"{key} must be <= {maximum}")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def validate_run(run):
    require(isinstance(run, dict), "run must be a JSON object")
    require(run.get("method") in ORDER, "method must be uniform, adaptive, gaussian or adaptive_gaussian")
    require(run.get("device") in ("cpu", "cuda"), "device must identify the actual cpu or cuda execution")
    integer(run, "seed")
    integer(run, "data_seed", 0)
    integer(run, "epochs", 1)
    integer(run, "batch_size", 2)
    for key in ("train_samples", "test_samples", "points_per_sample", "input_channels", "parameters"):
        integer(run, key, 1)
    require(number(run, "lr") > 0, "lr must be positive")
    if run["method"] == "adaptive":
        integer(run, "density_threshold", 0)
    for key, _, _, _ in METRICS:
        if key in ACCURACIES:
            number(run, key, maximum=1.0)
        elif key in MEMORY:
            value = number(run, key, minimum=-1.0)
            require(value == -1 or value >= 0, f"{key} must be nonnegative or the unavailable sentinel -1")
        else:
            number(run, key)
    for key in ("inference_latency_ms_median_b1", "throughput_samples_per_s"):
        require(run[key] > 0, f"{key} must be positive (missing measurements must not be reported as zero)")
    curve = run.get("curve")
    require(isinstance(curve, list) and len(curve) == run["epochs"], "curve must contain exactly epochs completed entries")
    for epoch, row in enumerate(curve, 1):
        require(isinstance(row, dict), f"curve epoch {epoch} must be an object")
        require(integer(row, "epoch", 1) == epoch, "curve epochs must be consecutive and start at 1")
        number(row, "train_loss")
        number(row, "train_acc", maximum=1.0)
        number(row, "test_acc", maximum=1.0)
        number(row, "epoch_train_s")


def declared_values(value, label, convert=str):
    parts = [part.strip() for part in value.split(",")]
    require(all(parts), f"--{label} must be a nonempty comma-separated list")
    try:
        values = [convert(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"--{label} contains an invalid value") from exc
    require(len(values) == len(set(values)), f"--{label} contains duplicate values")
    return set(values)


def load_runs(root, expected_methods, expected_seeds):
    require(root.is_dir(), f"run directory does not exist: {root}")
    paths = sorted(root.glob("*.json"))
    require(bool(paths), f"no run JSON files found in {root}")
    runs, identities = {}, {}
    baseline, baseline_path = None, None
    for path in paths:
        try:
            run = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                             parse_constant=invalid_constant)
            validate_run(run)
            identity = (run["method"], run["seed"])
            require(identity not in identities,
                    f"duplicate method/seed {identity}; first file: {identities.get(identity)}")
            if baseline is not None:
                for key in SHARED_CONTROLS:
                    require(run[key] == baseline[key],
                            f"shared control {key} differs from {baseline_path.name}: {run[key]!r} != {baseline[key]!r}")
            else:
                baseline, baseline_path = run, path
            earlier = runs.get(run["method"], [])
            if earlier:
                method_controls = ["parameters", "input_channels"]
                if run["method"] == "adaptive":
                    method_controls.append("density_threshold")
                for key in method_controls:
                    require(run[key] == earlier[0][key],
                            f"{run['method']} control {key} differs across seeds")
                for key in MEMORY:
                    require((run[key] == -1) == (earlier[0][key] == -1),
                            f"{run['method']} metric {key} mixes available and unavailable measurements")
            identities[identity] = path
            runs.setdefault(run["method"], []).append(run)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    require(set(runs) == expected_methods,
            f"method coverage differs: missing {sorted(expected_methods - set(runs))}; "
            f"unexpected {sorted(set(runs) - expected_methods)}")
    expected = expected_seeds if expected_seeds is not None else {
        row["seed"] for row in runs[baseline["method"]]}
    for method, rows in runs.items():
        actual = {row["seed"] for row in rows}
        require(actual == expected,
                f"seed cohort differs for {method}: {sorted(actual)}; expected {sorted(expected)}")
        rows.sort(key=lambda row: row["seed"])
    return runs


def fmt(values, scale, digits):
    v = [x * scale for x in values]
    m = statistics.fmean(v)
    s = statistics.stdev(v) if len(v) > 1 else 0.0
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def render(runs):
    methods = [m for m in ORDER if m in runs]
    lines = ["| method | seeds | " + " | ".join(h for _, h, _, _ in METRICS) + " |",
             "|---" * (len(METRICS) + 2) + "|"]
    for m in methods:
        rs = runs[m]
        cells = ["unavailable" if k in MEMORY and rs[0][k] == -1 else
                 fmt([r[k] for r in rs], sc, d) for k, _, sc, d in METRICS]
        lines.append(f"| {m} | {','.join(str(r['seed']) for r in rs)} | " + " | ".join(cells) + " |")
    epochs = runs[methods[0]][0]["epochs"]
    lines += ["", "Test accuracy (%) per epoch, mean over matched seeds:", "",
              "| epoch | " + " | ".join(methods) + " |", "|---" * (len(methods) + 1) + "|"]
    for e in range(epochs):
        row = [f"{statistics.fmean(r['curve'][e]['test_acc'] * 100 for r in runs[m]):.1f}" for m in methods]
        lines.append(f"| {e + 1} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--methods", default=",".join(ORDER),
                        help="comma-separated expected methods; defaults to all four")
    parser.add_argument("--seeds", help="comma-separated required seeds; otherwise compare observed cohorts")
    args = parser.parse_args()
    try:
        expected_methods = declared_values(args.methods, "methods")
        require(expected_methods <= set(ORDER), f"--methods contains unknown methods: {sorted(expected_methods - set(ORDER))}")
        expected_seeds = declared_values(args.seeds, "seeds", int) if args.seeds is not None else None
        runs = load_runs(args.root, expected_methods, expected_seeds)
        table = render(runs)
        if args.csv:
            require(args.csv.resolve() not in {path.resolve() for path in args.root.glob("*.json")},
                    "CSV output must not overwrite a run JSON file")
            with args.csv.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(["method", "seed"] + [key for key, _, _, _ in METRICS])
                for method in ORDER:
                    for run in runs.get(method, []):
                        writer.writerow([method, run["seed"]] + [run[key] for key, _, _, _ in METRICS])
    except (OSError, ValueError, OverflowError) as exc:
        print(f"ml_representations_summary: {exc}", file=sys.stderr)
        return 1
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
