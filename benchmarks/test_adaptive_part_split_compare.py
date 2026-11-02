"""Fail-closed regression checks for the portable paired benchmark runner."""
import unittest

from adaptive_part_split_compare import summarize


def fixture():
    return [dict(workload='control', round=rnd, arm=arm, metrics=dict(
        n=100, layout='clustered', checksum=123, median_ms=10.0 if arm == 0 else 8.0,
        peak_process_mib=50.0)) for rnd in (1, 2)
        for arm in ([0, 1] if rnd == 1 else [1, 0])]


class CompareContract(unittest.TestCase):
    def test_balanced_summary_preserves_pair_identity(self):
        result = summarize(fixture(), 'phase', 2)
        self.assertEqual(result[0]['before'], 10.0)
        self.assertEqual(result[0]['after'], 8.0)
        self.assertAlmostEqual(result[0]['change_percent'], -20.0)
        self.assertEqual(result[0]['after_lower_pairs'], 2)

    def test_incomplete_or_unbalanced_runs_rejected(self):
        for rows in [fixture()[:-1], sorted(fixture(), key=lambda r: (r['round'], r['arm']))]:
            with self.assertRaisesRegex(RuntimeError, 'pairs'):
                summarize(rows, 'phase', 2)

    def test_workload_or_checksum_changes_rejected(self):
        for key, value in [('n', 101), ('layout', 'unique'), ('checksum', 124)]:
            rows = fixture()
            rows[1]['metrics'][key] = value
            with self.assertRaisesRegex(RuntimeError, key):
                summarize(rows, 'phase', 2)

    def test_missing_or_nonfinite_metrics_rejected(self):
        cases = [None, float('nan'), float('inf'), -1.0, 0.0]
        for value in cases:
            rows = fixture()
            if value is None:
                del rows[1]['metrics']['median_ms']
            else:
                rows[1]['metrics']['median_ms'] = value
            with self.assertRaises(RuntimeError):
                summarize(rows, 'phase', 2)

    def test_missing_optional_metric_only_in_one_arm_rejected(self):
        rows = fixture()
        for row in rows:
            row['metrics'] = dict(samples=64, batch=16, pairs_first_batch=4,
                                  leaves_per_sample=2, part_hits=5, part_misses=3,
                                  kmap_batch_first_epoch_ms=3.0,
                                  kmap_batch_later_epochs_ms=2.0, peak_working_set_mb=50.0)
        summarize(rows, 'map', 2)  # CPU runs need no CUDA allocator metrics.
        rows[0]['metrics']['cuda_max_alloc_mb'] = 0.0
        with self.assertRaisesRegex(RuntimeError, 'Incomplete metric'):
            summarize(rows, 'map', 2)

    def test_optional_cache_metadata_changes_rejected(self):
        rows = fixture()
        for row in rows:
            row['metrics'] = dict(samples=64, batch=16, pairs_first_batch=4,
                                  leaves_per_sample=2, part_hits=5, part_misses=3,
                                  part_device_hits=6, kmap_batch_first_epoch_ms=3.0,
                                  kmap_batch_later_epochs_ms=2.0, peak_working_set_mb=50.0)
        rows[1]['metrics']['part_device_hits'] += 1
        with self.assertRaisesRegex(RuntimeError, 'part_device_hits'):
            summarize(rows, 'map', 2)


if __name__ == '__main__':
    unittest.main()
