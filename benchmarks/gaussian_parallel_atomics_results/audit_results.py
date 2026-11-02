"""Independently audit all raw Gaussian atomics trial identities and invariants."""
import hashlib
import json
import re
import statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'ab'
def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
r = read(OUT / 'receipt.json')
assert r['status'] == 'passed'
assert len(r['runs']) == 80
preserved = read(OUT / 'pre-resume-hashes.json')
assert len(preserved) == 60
for artifact in preserved:
    assert sha(Path(artifact['Path'])).upper() == artifact['Hash'].upper()
seen = set()
for run in r['runs']:
    ident = (run['round'], run['arm'], run['pattern'])
    assert ident not in seen and run['exit'] == 0
    seen.add(ident)
    assert run['arguments'][run['arguments'].index('--parallel-atomics') + 1] == str(run['arm'])
assert seen == {(i, a, p) for i in range(1, 11) for a in range(2) for p in ('row-groups', 'dense', 'sparse', 'render')}
rows = []
for pattern in ('row-groups', 'dense', 'sparse'):
    arms = [[], []]
    for i in range(1, 11):
        ds = [read(OUT / f'{pattern}_a{a}_r{i}.json') for a in range(2)]
        for key in ('pattern', 'tiles', 'per_tile', 'warmup', 'iterations', 'block_scans', 'warp_scans'):
            assert ds[0][key] == ds[1][key], (i, pattern, key)
        for a, d in enumerate(ds):
            assert d['pattern'] == pattern and d['warmup'] == 5 and d['iterations'] == 30
            assert len(d['gpu_ms']) == len(d['wall_ms']) == 30
            assert all(v > 0 for v in d['gpu_ms'] + d['wall_ms'])
            arms[a].append(st.median(d['gpu_ms']))
    rows.append({'workload': pattern, 'metric': 'gpu_ms', 'serial': arms[0], 'parallel': arms[1]})
for size in (512, 1024):
    metrics = {key: [[], []] for key in ('forward_median_ms', 'backward_median_ms', 'forward_backward_median_ms', 'train_median_ms', 'process_peak_working_set', 'cuda_peak_allocated_forward', 'cuda_peak_allocated_backward', 'cuda_peak_reserved')}
    for i in range(1, 11):
        ds = [read(OUT / f'render_a{a}_r{i}.json') for a in range(2)]
        assert all(d['scene_gaussians'] == 139410 and d['warmup'] == 3 and d['iterations'] == 10 for d in ds)
        rs = [next(x for x in d['records'] if x['size'] == size) for d in ds]
        for key in ('device', 'count', 'visible', 'pairs', 'pairs_source', 'mean_alpha'):
            assert rs[0][key] == rs[1][key], (i, size, key)
        for key, arms in metrics.items():
            for a in range(2):
                assert rs[a][key] > 0
                if key.endswith('_median_ms'):
                    samples = rs[a][key.replace('_median_ms', '_ms')]
                    assert len(samples) == 10
                    assert abs(st.median(samples) - rs[a][key]) < 0.001
                arms[a].append(rs[a][key])
    rows.extend({'workload': size, 'metric': key, 'serial': arms[0], 'parallel': arms[1]} for key, arms in metrics.items())
for row in rows:
    a, b = row['serial'], row['parallel']
    row.update(serial_median=st.median(a), parallel_median=st.median(b), aggregate_change_percent=100*(st.median(b)/st.median(a)-1), paired_change_median_percent=st.median([100*(y/x-1) for x,y in zip(a,b)]), lower_pairs=sum(y<x for x,y in zip(a,b)))
sass = (ROOT / 'candidate-sass.txt').read_text(encoding='utf-8-sig')
functions = re.split(r'Function : ', sass)[1:]
static = []
for function in functions:
    if 'raster_backward_kernel' in function:
        static.append({'parallel': 'ILb1E' in function.splitlines()[0], 'shuffle_instructions': len(re.findall(r'\bSHFL\.', function)), 'global_atomic_add_instructions': len(re.findall(r'\bRED\.E\.ADD', function))})
result = {'status': 'passed', 'head': r['head'], 'receipt_sha256': sha(OUT / 'receipt.json'), 'measured_raster_binary_sha256': r['raster_sha256'], 'measured_render_binary_sha256': r['render_sha256'], 'trials': 80, 'invariants': ['60 pre-resume raw output hashes unchanged', 'balanced unique trials', 'explicit CLI mode identity', 'identical synthetic dimensions/scans', 'identical real scene counts/visibility/pairs/alpha', 'all sample counts and medians', 'positive timings and memory'], 'static': static, 'metrics': rows}
(ROOT / 'independent-audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print('AUDIT PASSED: 80 trials and all invariants')
for x in rows:
    if x['metric'] in ('gpu_ms', 'backward_median_ms', 'train_median_ms', 'forward_median_ms'):
        print(x['workload'], x['metric'], f"{x['serial_median']:.5f} -> {x['parallel_median']:.5f}", f"{x['aggregate_change_percent']:+.2f}%", f"paired {x['paired_change_median_percent']:+.2f}%", f"{x['lower_pairs']}/10 lower")
print(static)

