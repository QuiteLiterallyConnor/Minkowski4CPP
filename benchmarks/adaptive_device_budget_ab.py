"""Balanced original/unlimited/bounded adaptive map benchmark; Python standard library only."""
import argparse
import itertools
import json
from pathlib import Path
import re
import subprocess


def parse_metrics(output):
    metrics = {}
    for line in output.splitlines():
        match = re.fullmatch(r"(\S+)\s+(-?\d+(?:\.\d+)?)", line)
        if match:
            metrics[match[1]] = float(match[2])
    for key in ("kmap_batch_first_epoch_ms", "kmap_batch_later_epochs_ms"):
        if key not in metrics:
            raise ValueError(f"Missing benchmark metric: {key}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-mib", type=float, default=16)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    if args.rounds < 6 or args.rounds % 6:
        parser.error("rounds must be a positive multiple of six for balanced arm order")
    if args.budget_mib < 0:
        parser.error("budget must be nonnegative")
    executables = {"baseline": args.baseline.resolve(),
                   "unlimited": args.candidate.resolve(), "bounded": args.candidate.resolve()}
    if any(not path.is_file() for path in executables.values()):
        parser.error("both executables must exist")
    args.output.mkdir(parents=True, exist_ok=False)
    # Same six permutations as the original measured Windows experiment.
    orders = [("unlimited", "bounded", "baseline"), ("bounded", "baseline", "unlimited"),
              ("baseline", "unlimited", "bounded"), ("baseline", "bounded", "unlimited"),
              ("bounded", "unlimited", "baseline"), ("unlimited", "baseline", "bounded")]
    assert set(orders) == set(itertools.permutations(executables))
    records = []
    for mode in ("center", "volume"):
        for iteration in range(-1, args.rounds + 1):
            tag = "warm1" if iteration == -1 else "warm2" if iteration == 0 else f"r{iteration}"
            for arm in orders[(iteration + 5) % 6]:
                command = [str(executables[arm]), "--batch", "16", "--samples", "64",
                           "--epochs", "4", "--n", "2048", "--depth", "3", "--part-reuse-mb", "512",
                           "--batch-device", "cuda", "--device-compose", "1", "--modes", mode]
                if arm != "baseline":
                    command += ["--part-device-mb", str(args.budget_mib if arm == "bounded" else -1)]
                run = subprocess.run(command, cwd=executables[arm].parent, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                (args.output / f"{mode}_{arm}_{tag}.txt").write_text(run.stdout, encoding="utf-8")
                run.check_returncode()
                metrics = parse_metrics(run.stdout)
                if arm == "bounded" and metrics["part_device_store_mb"] > args.budget_mib + 0.001:
                    raise ValueError("Retained part charge exceeded configured budget")
                records.append({"mode": mode, "arm": arm, "iteration": iteration,
                                "command": command, "metrics": metrics})
                (args.output / "runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
                print(mode, arm, tag, "passed", flush=True)


if __name__ == "__main__":
    main()
