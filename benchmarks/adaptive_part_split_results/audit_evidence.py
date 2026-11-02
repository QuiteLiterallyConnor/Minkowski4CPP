"""Independently reparse immutable native logs and recompute paired summaries."""
import hashlib
import importlib.util
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path(__file__).resolve().parent
helper_path = Path(r'C:\Users\User\Work\mink-wt\device-compose\benchmarks\adaptive_part_split_compare.py')
spec = importlib.util.spec_from_file_location('portable_split_runner', helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
findings = []
for tag, count in [('phase-cpu', 60), ('map-cpu', 24), ('map-cuda', 24)]:
    receipt = read(root / tag / 'receipt.json')
    assert receipt['status'] == 'passed' and len(receipt['runs']) == count
    assert receipt['source'] == '95093e2938e939ad5dd954c33ae4ea583879fa4d'
    for executable, expected in zip(receipt['executables'], receipt['executable_sha256']):
        assert hashlib.sha256(Path(executable).read_bytes()).hexdigest() == expected
    by_workload = {}
    for row in receipt['runs']:
        assert row['exit'] == 0
        stem = f"{row['workload']}-r{row['round']}-a{row['arm']}"
        raw = (root / tag / (stem + '.txt')).read_text(encoding='utf-8')
        metrics = json.loads(raw) if tag == 'phase-cpu' else {
            m.group(1): float(m.group(2)) for line in raw.splitlines()
            if (m := re.fullmatch(r'([a-zA-Z_0-9]+)\s+([0-9.eE+\-]+)', line))}
        assert metrics == row['metrics'], stem
        if tag == 'phase-cpu':
            assert metrics['arm'] == row['arm']
            assert len(metrics['samples_ms']) == 15
            assert math.isclose(statistics.median(metrics['samples_ms']), metrics['median_ms'], rel_tol=1e-10)
        by_workload.setdefault(row['workload'], []).append(row)
    for workload, rows in by_workload.items():
        assert [(r['round'], r['arm']) for r in rows] == [
            (rnd, arm) for rnd in range(1, 7) for arm in ([0, 1] if rnd % 2 else [1, 0])]
        invariant_keys = (['n', 'layout', 'checksum'] if tag == 'phase-cpu' else
            ['batch_device_cuda', 'device_compose', 'samples', 'batch', 'leaves_per_sample',
             'part_reuse_mb', 'part_device_mb', 'pairs_first_batch', 'part_hits', 'part_misses',
             'part_store_mb', 'part_device_store_mb', 'part_device_peak_mb',
             'part_device_hits', 'part_device_misses', 'part_device_groups'])
        for key in invariant_keys:
            if key in rows[0]['metrics']:
                assert all(r['metrics'][key] == rows[0]['metrics'][key] for r in rows), (tag, workload, key)
    summaries = read(root / tag / 'summary.json')
    portable_summary = [item for rows in by_workload.values()
                        for item in helper.summarize(rows, receipt['kind'], receipt['rounds'])]
    assert len(portable_summary) == len(summaries)
    for portable, historical in zip(portable_summary, summaries):
        assert portable.pop('after_lower_pairs') == historical['faster']
        assert portable == {k: v for k, v in historical.items() if k != 'faster'}
    for summary in summaries:
        rows = by_workload[summary['workload']]
        values = {arm: [r['metrics'][summary['metric']] for r in rows if r['arm'] == arm] for arm in (0, 1)}
        before, after = [statistics.median(values[arm]) for arm in (0, 1)]
        assert summary['before'] == before and summary['after'] == after
        assert math.isclose(summary['change_percent'], 100 * (after / before - 1), abs_tol=1e-12)
        assert summary['faster'] == sum(b < a for a, b in zip(values[0], values[1]))
        for arm, prefix in [(0, 'before'), (1, 'after')]:
            assert summary[prefix + '_range'] == [min(values[arm]), max(values[arm])]
    findings.append({'experiment': tag, 'raw_processes_reparsed': count,
                     'summaries_recomputed': len(summaries), 'binary_hashes_match': True,
                     'paired_order_and_all_recorded_workload_invariants_match': True})
for device in ('cpu', 'cuda'):
    tree = ET.parse(root / f'ctest-{device}.xml').getroot()
    assert tree.attrib['tests'] == '5' and tree.attrib['failures'] == '0'
    assert {x.attrib['name'] for x in tree.findall('testcase')} == {
        'adaptive_tests', 'adaptive_split_tests', 'adaptive_vs_uniform_bench_smoke',
        'gaussian_tests', 'test_framework_contract'}
    findings.append({'registered_ctest_device': device, 'passed': 5, 'failed': 0})
out = {'status': 'passed', 'findings': findings,
       'limits': ['Audit of prior native executions; no new timings or container GPU execution.',
                  'Exact map order, values, gradients are covered by CTest; aggregate logs alone do not establish them.']}
(root / 'evidence-audit.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
print(json.dumps(out, indent=2))
