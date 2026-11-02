"""Recompute final A/B summaries from committed per-iteration measurements."""
import json
import statistics as st
from pathlib import Path

root = Path(__file__).resolve().parent
runs = json.loads((root / 'runs.json').read_text(encoding='utf-8'))
print('| Workload | Metric | Reference median [min-max] | Fused median [min-max] | Change | Lower pairs |')
print('|---|---|---:|---:|---:|---:|')
for kind in ('render', 'train'):
    for size in (512, 1024):
        records = [[next(x for x in runs[f'{kind}_a{arm}_r{rnd}.json']['records']
                         if x['size'] == size) for rnd in range(1, 11)] for arm in (0, 1)]
        for metric in records[0][0]:
            if not (metric.endswith('_median_ms') or 'peak' in metric):
                continue
            vals = [[x[metric] for x in arm] for arm in records]
            med = [st.median(arm) for arm in vals]
            if med[0] == 0:
                continue
            factor = 1 / 1048576 if 'peak' in metric else 1
            unit = 'MiB' if 'peak' in metric else 'ms'
            cells = [f'{m*factor:.3f} [{min(a)*factor:.3f}-{max(a)*factor:.3f}] {unit}'
                     for m, a in zip(med, vals)]
            wins = sum(b < a for a, b in zip(*vals))
            print(f'| {kind} {size}px | {metric} | {cells[0]} | {cells[1]} | '
                  f'{100*(med[1]/med[0]-1):+.2f}% | {wins}/10 |')
