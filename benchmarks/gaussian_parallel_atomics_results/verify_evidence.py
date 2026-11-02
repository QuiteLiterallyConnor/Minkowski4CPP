"""Verify frozen evidence files and independently recompute raw benchmark summaries.

Run from any directory; this script needs only Python's standard library.
"""
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import statistics
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)

def main():
    manifest = read(ROOT / 'manifest.json')
    expected_files = set()
    for row in manifest['files']:
        path = (ROOT / row['path']).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file()
        assert row['path'] not in expected_files
        expected_files.add(row['path'])
        assert sha(path) == row['sha256'], row['path']
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    assert actual == expected_files, 'Evidence file inventory changed'
    receipt = read(ROOT / 'ab/receipt.json')
    assert receipt['status'] == 'passed' and receipt['head'] == manifest['runtime_source']
    assert len(receipt['runs']) == 80
    identities = {(r['round'], r['arm'], r['pattern']) for r in receipt['runs']}
    assert identities == {(r, a, p) for r in range(1, 11) for a in (0, 1) for p in ('row-groups', 'sparse', 'dense', 'render')}
    for run in receipt['runs']:
        assert run['exit'] == 0
        assert run['arguments'][run['arguments'].index('--parallel-atomics') + 1] == str(run['arm'])
    preserved = read(ROOT / 'ab/pre-resume-hashes.json')
    assert len(preserved) == 60
    for row in preserved:
        path = ROOT / 'ab' / PureWindowsPath(row['Path']).name
        assert sha(path).lower() == row['Hash'].lower()
    results = {}
    for pattern in ('row-groups', 'dense', 'sparse'):
        arms = {key: [[], []] for key in ('gpu_ms', 'wall_ms')}
        for repeat in range(1, 11):
            pair = [read(ROOT / f'ab/{pattern}_a{arm}_r{repeat}.json') for arm in (0, 1)]
            for key in ('pattern', 'tiles', 'per_tile', 'warmup', 'iterations', 'block_scans', 'warp_scans'):
                assert pair[0][key] == pair[1][key]
            assert pair[0]['pattern'] == pattern and pair[0]['tiles'] == 1024 and pair[0]['per_tile'] == 256
            for arm, row in enumerate(pair):
                assert row['warmup'] == 5 and row['iterations'] == 30
                for key in arms:
                    assert len(row[key]) == 30 and all(math.isfinite(v) and v > 0 for v in row[key])
                    arms[key][arm].append(statistics.median(row[key]))
        results.update({(pattern, key): value for key, value in arms.items()})
    for size in (512, 1024):
        arms = {}
        for repeat in range(1, 11):
            files = [read(ROOT / f'ab/render_a{arm}_r{repeat}.json') for arm in (0, 1)]
            assert all(row['scene_gaussians'] == 139410 and row['warmup'] == 3 and row['iterations'] == 10 for row in files)
            pair = [next(row for row in file['records'] if row['size'] == size) for file in files]
            for key in ('device', 'count', 'visible', 'pairs', 'pairs_source', 'mean_alpha'):
                assert pair[0][key] == pair[1][key]
            for arm, row in enumerate(pair):
                for key, value in row.items():
                    if key.endswith('_median_ms') or key.startswith('cuda_peak') or key == 'process_peak_working_set':
                        assert math.isfinite(value) and value > 0
                        if key.endswith('_median_ms'):
                            values = row[key.replace('_median_ms', '_ms')]
                            assert len(values) == 10 and all(math.isfinite(v) and v > 0 for v in values)
                            assert abs(statistics.median(values) - value) < .001
                        arms.setdefault(key, [[], []])[arm].append(value)
        results.update({(str(size), key): value for key, value in arms.items()})
    summary = read(ROOT / 'ab/summary.json')
    assert len(summary) == len(results)
    for row in summary:
        arms = results.pop((row['workload'], row['metric']))
        assert arms == [row['off'], row['on']]
        a, b = map(statistics.median, arms)
        close(row['change_percent'], 100*(b/a-1))
        assert row['lower_pairs'] == sum(y < x for x, y in zip(*arms))
    assert not results
    validation = read(ROOT / 'validation/receipt.json')
    assert validation['status'] == 'passed' and validation['head'] == receipt['head']
    for kind in ('cpu', 'cuda'):
        xml = ET.parse(ROOT / f'validation/ctest-{kind}.xml').getroot()
        assert xml.attrib['tests'] == '4' and xml.attrib['failures'] == '0'
    sanitizer = read(ROOT / 'sanitize/receipt.json')
    assert sanitizer['status'] == 'passed' and sanitizer['exclusions'] == []
    assert {row['tool'] for row in sanitizer['tools']} == {'memcheck', 'racecheck', 'initcheck'}
    assert all(row['exit'] == 0 for row in sanitizer['tools'])
    print(f'PASS: {len(expected_files)} SHA256 files; 80 benchmark trials; all raw summary rows; 60 preserved pre-resume files; CPU/CUDA and sanitizer receipts')

if __name__ == '__main__':
    main()
