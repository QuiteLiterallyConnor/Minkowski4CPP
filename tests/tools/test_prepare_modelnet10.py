"""Offline regression tests; generated fixtures contain no redistributed CAD data."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("modelnet", Path(__file__).resolve().parents[2] / "benchmarks/prepare_modelnet10.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)
TRIANGLE = b"OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n"


class PrepareTests(unittest.TestCase):
    def test_off_header_variants(self):
        expected = prep.sample_points(TRIANGLE, 32, 7)
        for data in (TRIANGLE.replace(b"OFF\n", b"OFF"), TRIANGLE.replace(b"OFF\n", b"OFF "), b"# comment\n" + TRIANGLE):
            self.assertEqual(expected, prep.sample_points(data, 32, 7))

    def test_determinism_and_bounds(self):
        points = prep.sample_points(TRIANGLE, 100, 5)
        self.assertEqual(points, prep.sample_points(TRIANGLE, 100, 5))
        self.assertNotEqual(points, prep.sample_points(TRIANGLE, 100, 6))
        self.assertTrue(all(-0.951 <= x <= 0.951 and math.isfinite(x) for p in points for x in p))
        self.assertTrue(all(p[2] == 0 for p in points))

    def test_area_weighting(self):
        mesh = b"OFF\n6 2 0\n0 0 0\n1 0 0\n0 1 0\n0 0 10\n2 0 10\n0 2 10\n3 0 1 2\n3 3 4 5\n"
        points = prep.sample_points(mesh, 10000, 3)
        large_fraction = sum(p[2] > 0 for p in points) / len(points)
        self.assertAlmostEqual(large_fraction, .8, delta=.02)

    def test_reject_bad_meshes(self):
        bad = [b"", TRIANGLE.replace(b"OFF", b"COFF"), TRIANGLE.replace(b"3 0 1 2", b"3 0 1 3"),
               TRIANGLE.replace(b"1 0 0", b"nan 0 0"), TRIANGLE.replace(b"3 0 1 2", b"3 0 0 0"),
               b"OFF\n0 0 0\n", TRIANGLE.rsplit(b"3 0 1 2", 1)[0]]
        for mesh in bad:
            with self.subTest(mesh=mesh), self.assertRaises((ValueError, IndexError)):
                prep.parse_off(mesh)

    def test_archive_splits_and_binary_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as out:
                for cls in prep.CLASSES:
                    for split in ("train", "test"):
                        for i in range(3):
                            out.writestr(f"ModelNet10/{cls}/{split}/{i}.off", TRIANGLE)
                out.writestr("__MACOSX/ignore.off", b"invalid")
            a = prep.prepare(archive, root / "a", 2, 1, 16, 2026)
            b = prep.prepare(archive, root / "b", 2, 1, 16, 2026)
            self.assertEqual(a, b)
            self.assertFalse(set(a["splits"]["train"]["ids"]) & set(a["splits"]["test"]["ids"]))
            for split, count in (("train", 20), ("test", 10)):
                data = (root / "a" / f"{split}.bin").read_bytes()
                self.assertEqual(data[:8], prep.MAGIC)
                self.assertEqual(struct.unpack_from("<II", data, 8), (count, 16))
                self.assertEqual(len(data), 16 + count * (4 + 16 * 24))
                self.assertEqual(prep.fnv1a64(data), a["splits"][split]["fnv1a64"])
                self.assertEqual(data, (root / "b" / f"{split}.bin").read_bytes())
            with self.assertRaises(ValueError):
                prep.prepare(archive, root / "a", 2, 1, 16, 2026)
            with self.assertRaises(ValueError):
                prep.prepare(archive, root / "c", 4, 1, 16, 2026)


if __name__ == "__main__":
    unittest.main()
