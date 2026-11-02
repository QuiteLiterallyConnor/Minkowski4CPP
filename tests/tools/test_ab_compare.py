"""Self-test for benchmarks/ab_compare.py (standard library only; registered in CTest).

Two fake "build directories" hold the same script benchmark; the one in dir B reports half the
latency. Checks interleaving, warm-up exclusion, metric parsing (stdout + {json}), resume, the
budget stop (exit 3), failure handling (exit 1) and the Markdown table.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
import ab_compare  # noqa: E402

FAKE = r'''
import json, sys, time
from pathlib import Path
here = Path(__file__).resolve().parent
scale = float((here / "scale.txt").read_text())
args = sys.argv[1:]
if "--fail" in args and scale < 1:
    sys.exit(7)
if "--sleep" in args:
    time.sleep(float(args[args.index("--sleep") + 1]))
print("leaves                 1000")
print("fwd_ms                 %.3f" % (10.0 * scale))
print("level 0            31")
if "--json" in args:
    out = args[args.index("--json") + 1]
    Path(out).write_text(json.dumps({"results": [{"name": "render", "device": "cpu", "median_ms": 4.0 * scale}]}))
print("Epoch 01 | loss=0.5000 | 2.5s")
'''


class AbCompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.dirs = {}
        for name, scale in (("base", 1.0), ("cand", 0.5)):
            d = root / name
            d.mkdir()
            (d / "bench.py").write_text(FAKE)
            (d / "scale.txt").write_text(str(scale))
            self.dirs[name] = d
        self.out = root / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def run_ab(self, *extra, rounds=3, warmup=1, name="fake"):
        argv = ["--a", f"base={self.dirs['base']}", "--b", f"cand={self.dirs['cand']}", "--out", str(self.out),
                "--name", name, "--rounds", str(rounds), "--warmup-rounds", str(warmup), "--gpu-sample-ms", "0",
                "--extract", r"epoch_s=\| ([0-9.]+)s\s*$"]
        return ab_compare.main(argv + list(extra))

    def state(self, name="fake"):
        return json.loads((self.out / f"{name}.json").read_text())

    def test_interleaved_summary_and_markdown(self):
        self.assertEqual(self.run_ab("--", "bench.py", "--json", "{json}"), 0)
        st = self.state()
        self.assertEqual([r["side"] for r in st["runs"]], ["base", "cand"] * 4)
        self.assertEqual([r["warmup"] for r in st["runs"]], [True, True] + [False] * 6)
        base, cand = st["summary"]["base"], st["summary"]["cand"]
        self.assertEqual(base["fwd_ms"]["n"], 3)
        self.assertAlmostEqual(base["fwd_ms"]["median"], 10.0)
        self.assertAlmostEqual(cand["fwd_ms"]["median"], 5.0)
        self.assertAlmostEqual(cand["results[render@cpu].median_ms"]["median"], 2.0)
        self.assertAlmostEqual(base["level_0"]["median"], 31)
        self.assertAlmostEqual(base["epoch_s"]["median"], 2.5)
        self.assertIn("wall_ms", base)
        md = (self.out / "fake.md").read_text()
        self.assertIn("| fwd_ms | 10.00 [10.00-10.00] | 5.00 [5.00-5.00] | -50.0 % | 3/3 |", md)
        self.assertIn("leaves=1000", md)  # identical constants listed once

    def test_resume_adds_only_missing_rounds(self):
        self.assertEqual(self.run_ab("--", "bench.py", rounds=2), 0)
        self.assertEqual(self.run_ab("--", "bench.py", rounds=2, warmup=0), 0)
        self.assertEqual(len(self.state()["runs"]), 6)  # nothing added
        self.assertEqual(self.run_ab("--", "bench.py", rounds=3, warmup=0), 0)
        st = self.state()
        self.assertEqual(len(st["runs"]), 8)
        self.assertEqual(st["summary"]["cand"]["fwd_ms"]["n"], 3)
        self.assertEqual(len(st["blocks"]), 3)

    def test_budget_stops_with_exit_3(self):
        rc = self.run_ab("--budget-s", "0.3", "--", "bench.py", "--sleep", "0.2", rounds=4, warmup=0)
        self.assertEqual(rc, 3)
        timed = [r for r in self.state()["runs"] if not r["warmup"]]
        self.assertLess(len(timed), 8)
        self.assertEqual(len(timed) % 2, 0)  # never stops between A and B

    def test_failure_exit_1(self):
        self.assertEqual(self.run_ab("--", "bench.py", "--fail", rounds=2, warmup=0), 1)
        runs = self.state()["runs"]
        self.assertEqual(runs[-1]["side"], "cand")
        self.assertNotEqual(runs[-1]["exit"], 0)

    def test_rejects_changed_command(self):
        self.assertEqual(self.run_ab("--", "bench.py", rounds=1, warmup=0), 0)
        with self.assertRaises(SystemExit):
            self.run_ab("--", "bench.py", "--json", "{json}", rounds=1, warmup=0)

    def test_parse_stdout(self):
        vals = ab_compare.parse_stdout("kmap_center_cold_ms   1.5\npoints=20 depth=4\n  level 3   7\nbad 1\n", [])
        self.assertEqual(vals, {"kmap_center_cold_ms": 1.5, "level_3": 7.0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
