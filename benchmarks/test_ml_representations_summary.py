"""Tiny artifact fixtures for the summary CLI; no model or GPU execution."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("ml_representations_summary.py")


def run_record(method="uniform", seed=1):
    return {
        "method": method, "seed": seed, "data_seed": 2026, "device": "cuda",
        "epochs": 2, "batch_size": 16, "lr": 0.001, "train_samples": 800,
        "test_samples": 400, "points_per_sample": 2048, "density_threshold": 12,
        "input_channels": 4 if method == "uniform" else 12,
        "parameters": 1000 if method == "uniform" else 2000,
        "accuracy": 0.5, "macro_precision": 0.5, "macro_recall": 0.5,
        "macro_f1": 0.5, "train_time_s": 2.0,
        "inference_latency_ms_median_b1": 1.0, "throughput_samples_per_s": 100.0,
        "peak_cuda_alloc_mb_train": 20.0, "peak_rss_mb": 100.0,
        "active_elements_mean": 64.0, "conversion_ms_median": 2.0,
        "curve": [
            {"epoch": 1, "train_loss": 1.2, "train_acc": 0.4, "test_acc": 0.3, "epoch_train_s": 1.0},
            {"epoch": 2, "train_loss": 1.0, "train_acc": 0.6, "test_acc": 0.5, "epoch_train_s": 1.0},
        ],
    }


class SummaryCliTests(unittest.TestCase):
    def invoke(self, records=(), extra_text=None, csv_output=False, methods="observed", seeds=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, record in enumerate(records):
                (root / f"run-{index}.json").write_text(json.dumps(record), encoding="utf-8")
            if extra_text is not None:
                (root / "extra.json").write_text(extra_text, encoding="utf-8")
            output = root / "summary.csv"
            args = [sys.executable, "-I", "-X", "utf8", str(SCRIPT), str(root)]
            if methods == "observed":
                methods = ",".join(sorted({record["method"] for record in records})) if records else None
            if methods is not None:
                args += ["--methods", methods]
            if seeds is not None:
                args += ["--seeds", seeds]
            if csv_output:
                output.write_text("preserved previous CSV", encoding="utf-8")
                args += ["--csv", str(output)]
            result = subprocess.run(args, text=True, encoding="utf-8", capture_output=True, timeout=10)
            return result, output.read_text(encoding="utf-8") if output.exists() else None

    def assert_rejected(self, records=(), *, extra_text=None, diagnostic=None, methods="observed", seeds=None):
        result, csv_text = self.invoke(records, extra_text, csv_output=True, methods=methods, seeds=seeds)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "", "must validate before printing comparison tables")
        self.assertEqual(csv_text, "preserved previous CSV")
        if diagnostic:
            self.assertIn(diagnostic, result.stderr)

    def test_balanced_methods_preserve_architecture_differences_and_csv(self):
        records = [run_record(method, seed) for method in
                   ("uniform", "adaptive", "gaussian", "adaptive_gaussian") for seed in (2, 1)]
        for record in records:
            record["accuracy"] = 0.4 if record["seed"] == 1 else 0.6
        result, csv_text = self.invoke(records, csv_output=True, methods=None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("| uniform | 1,2 | 50.0 ± 14.1", result.stdout)
        self.assertIn("peak VRAM train+eval MB", result.stdout)
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual(len(rows), 8)
        self.assertEqual([row["seed"] for row in rows[:2]], ["1", "2"])
        self.assertEqual({row["parameters"] for row in rows}, {"1000", "2000"})

    def test_single_method_subset_and_unavailable_memory_are_valid(self):
        record = run_record("gaussian")
        record.update(device="cpu", peak_cuda_alloc_mb_train=-1, peak_rss_mb=-1)
        result, _ = self.invoke([record])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("| gaussian | 1 |", result.stdout)
        self.assertIn("unavailable", result.stdout)

    def test_default_requires_all_four_planned_methods(self):
        records = [run_record(method) for method in ("uniform", "adaptive", "gaussian")]
        self.assert_rejected(records, methods=None, diagnostic="method coverage differs")

    def test_declared_seeds_catch_a_seed_missing_from_every_method(self):
        records = [run_record(method, seed) for method in ("uniform", "gaussian") for seed in (1, 2)]
        self.assert_rejected(records, seeds="1,2,3", diagnostic="seed cohort differs")
        result, _ = self.invoke(records, seeds="1,2")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_declared_coverage_is_rejected(self):
        for methods in ("uniform,uniform", "uniform,unknown", ""):
            with self.subTest(methods=methods):
                self.assert_rejected([run_record()], methods=methods, diagnostic="--methods")
        for seeds in ("1,1", "1,invalid", ""):
            with self.subTest(seeds=seeds):
                self.assert_rejected([run_record()], seeds=seeds, diagnostic="--seeds")

    def test_available_memory_is_not_averaged_with_unavailable_sentinel(self):
        candidate = run_record(seed=2)
        candidate["peak_rss_mb"] = -1
        self.assert_rejected([run_record(seed=1), candidate], diagnostic="mixes available and unavailable")

    def test_duplicate_seed_cannot_gain_statistical_weight(self):
        self.assert_rejected([run_record(), run_record()], diagnostic="duplicate method/seed")

    def test_shared_controls_cannot_be_mixed(self):
        changes = {
            "device": "cpu", "data_seed": 2027, "epochs": 3, "batch_size": 8,
            "lr": 0.002, "train_samples": 400, "test_samples": 200,
            "points_per_sample": 1024,
        }
        for key, value in changes.items():
            with self.subTest(control=key):
                candidate = run_record("gaussian")
                candidate[key] = value
                if key == "epochs":
                    candidate["curve"].append({**candidate["curve"][-1], "epoch": 3})
                self.assert_rejected([run_record(), candidate], diagnostic=f"shared control {key}")

    def test_seed_cohort_must_match_between_represented_methods(self):
        self.assert_rejected([run_record(seed=1), run_record(seed=2), run_record("gaussian", 1)],
                             diagnostic="seed cohort differs")

    def test_density_is_controlled_only_where_it_is_used(self):
        candidate = run_record("adaptive", 2)
        candidate["density_threshold"] = 24
        self.assert_rejected([run_record("adaptive", 1), candidate], diagnostic="density_threshold")
        records = [run_record("gaussian", 1), run_record("gaussian", 2)]
        records[1]["density_threshold"] = 999
        result, _ = self.invoke(records)
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [run_record("adaptive", 1), run_record("adaptive", 2)]
        for record in records:
            record["density_threshold"] = 0
        result, _ = self.invoke(records)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_architecture_cannot_change_between_seeds_of_one_method(self):
        for key in ("input_channels", "parameters"):
            with self.subTest(control=key):
                candidate = run_record(seed=2)
                candidate[key] += 1
                self.assert_rejected([run_record(seed=1), candidate], diagnostic=key)

    def test_malformed_json_is_not_silently_ignored(self):
        self.assert_rejected([run_record()], extra_text='{"method":', diagnostic="extra.json")

    def test_duplicate_json_keys_are_rejected(self):
        raw = json.dumps(run_record()).replace('"seed": 1', '"seed": 1, "seed": 2')
        self.assert_rejected(extra_text=raw, diagnostic="duplicate JSON key")

    def test_empty_or_unrelated_json_is_rejected(self):
        self.assert_rejected(diagnostic="no run JSON")
        self.assert_rejected(extra_text='{"note": "not a training result"}', diagnostic="method")

    def test_invalid_metrics_do_not_produce_a_table(self):
        mutations = [
            ("accuracy", 1.1), ("macro_f1", -0.1), ("train_time_s", float("nan")),
            ("train_time_s", float("inf")), ("train_time_s", True),
            ("train_time_s", "2.0"), ("throughput_samples_per_s", 0),
            ("inference_latency_ms_median_b1", 0), ("peak_rss_mb", -0.5),
        ]
        for key, value in mutations:
            with self.subTest(field=key, value=value):
                record = run_record()
                record[key] = value
                self.assert_rejected([record])
        record = run_record()
        del record["macro_precision"]
        self.assert_rejected([record], diagnostic="macro_precision")

    def test_curve_must_be_complete_ordered_and_valid(self):
        mutations = []
        truncated = run_record()
        truncated["curve"].pop()
        mutations.append(truncated)
        unordered = run_record()
        unordered["curve"][1]["epoch"] = 1
        mutations.append(unordered)
        bad_accuracy = run_record()
        bad_accuracy["curve"][0]["test_acc"] = 1.1
        mutations.append(bad_accuracy)
        bad_loss = run_record()
        bad_loss["curve"][0]["train_loss"] = -1
        mutations.append(bad_loss)
        for index, record in enumerate(mutations):
            with self.subTest(case=index):
                self.assert_rejected([record])


if __name__ == "__main__":
    unittest.main()
