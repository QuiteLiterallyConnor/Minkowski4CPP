import json
import statistics as st
from pathlib import Path

root = Path(__file__).resolve().parent / 'ab'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
receipt = read(root / 'receipt.json')
assert receipt['status'] == 'passed'
assert len(receipt['runs']) == 80 and all(r['exit'] == 0 for r in receipt['runs'])
assert len({(r['round'], r['arm'], r['pattern']) for r in receipt['runs']}) == 80

values = {}
for arm in (0, 1):
    for round in range(1, 11):
        for pattern in ('row-groups', 'dense', 'sparse'):
            data = read(root / f'{pattern}_a{arm}_r{round}.json')
            for kind in ('gpu', 'wall'):
                samples = data[f'{kind}_ms']
                assert len(samples) == 30
                values.setdefault((pattern, f'{kind}_ms'), {0: [], 1: []})[arm].append(st.median(samples))
        data = read(root / f'render_a{arm}_r{round}.json')
        assert data['scene_gaussians'] == 139410
        for row in data['records']:
            for key, value in row.items():
                if key.endswith('_median_ms') or key.startswith('cuda_peak') or key == 'process_peak_working_set':
                    if key.endswith('_median_ms'):
                        assert len(row[key.replace('_median_ms', '_ms')]) == 10
                    values.setdefault((str(row['size']), key), {0: [], 1: []})[arm].append(value)

lines = ['# Parallel Gaussian backward atomics: balanced A/B', '',
         f"Measured source `{receipt['head']}`. Ten paired rounds, order reversed each round. "
         'All other code paths are unchanged in the same binary. GPU event timing below isolates '
         'the synthetic backward launch; real-scene rows use synchronized wall time. '
         'The real scene has 139410 Gaussians. Host-load observations are in hostload.log.', '',
         '| Workload | Metric | Serial atomics: median [min,max] | Parallel atomics: median [min,max] | Change | Lower pairs |',
         '|---|---|---|---|---|---|']
results = []
for (workload, key), arms in sorted(values.items()):
    a, b = arms[0], arms[1]
    assert len(a) == len(b) == 10
    am, bm = st.median(a), st.median(b)
    delta = 100 * (bm / am - 1) if am else 0
    wins = sum(y < x for x, y in zip(a, b))
    scale = 1 / 1048576 if 'peak' in key else 1
    fmt = lambda v: f'{st.median(v)*scale:.4f} [{min(v)*scale:.4f},{max(v)*scale:.4f}]'
    lines.append(f'| {workload} | {key} | {fmt(a)} | {fmt(b)} | {delta:+.2f}% | {wins}/10 |')
    results.append(dict(workload=workload,metric=key,off=a,on=b,change_percent=delta,lower_pairs=wins))
lines += ['', 'Memory columns are MiB. This branch only records render-phase CUDA peaks, not training VRAM. '
          'Working set is cumulative over the complete process. The forward-only measurements are unchanged-path '
          'noise controls. Synthetic tile layouts are stress controls, not observed real-scene coverage. '
          'Training throughput can be derived as 1000/step_ms; no accuracy or convergence improvement is implied.', '']
(root / 'summary.md').write_text('\n'.join(lines),encoding='utf-8')
(root / 'summary.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
print('\n'.join(line for line in lines if 'gpu_ms' in line or 'train_median_ms' in line or 'backward_median_ms' in line))
