"""Run balanced process pairs for part splitting or complete cached batch maps.

Build first and run without competing builds/GPU experiments. Use the same
adaptive_split_bench executable for --before and --after in phase mode; arm 0
contains the former algorithm and arm 1 calls the production helpers. Map mode
requires separately compiled original and changed adaptive_kmap_bench binaries.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_workload(rows, kind, rounds):
    """Reject incomplete pairs and silent workload/cache changes before summarizing."""
    expected = [(rnd, arm) for rnd in range(1, rounds + 1)
                for arm in ([0, 1] if rnd % 2 else [1, 0])]
    if [(r['round'], r['arm']) for r in rows] != expected:
        raise RuntimeError('Incomplete or unbalanced process pairs')
    invariants = (['n', 'layout', 'checksum'] if kind == 'phase' else
                  ['batch_device_cuda', 'device_compose', 'samples', 'batch', 'leaves_per_sample',
                   'part_reuse_mb', 'part_device_mb', 'pairs_first_batch', 'part_hits',
                   'part_misses', 'part_store_mb', 'part_device_store_mb', 'part_device_peak_mb',
                   'part_device_hits', 'part_device_misses', 'part_device_groups'])
    required = {'n', 'layout', 'checksum'} if kind == 'phase' else {
        'samples', 'batch', 'pairs_first_batch', 'leaves_per_sample', 'part_hits', 'part_misses'}
    for key in invariants:
        present = [key in r['metrics'] for r in rows]
        if key not in required and not any(present):
            continue
        if not all(present) or any(r['metrics'][key] != rows[0]['metrics'][key] for r in rows):
            raise RuntimeError(f'Changed or missing workload invariant {key}')


def summarize(rows, kind, rounds):
    validate_workload(rows, kind, rounds)
    fields = (['median_ms', 'peak_process_mib'] if kind == 'phase' else
              ['kmap_batch_first_epoch_ms', 'kmap_batch_later_epochs_ms', 'peak_working_set_mb',
               'cuda_max_alloc_mb', 'cuda_max_reserved_mb'])
    result = []
    for field in fields:
        values = {arm: [r['metrics'][field] for r in rows if r['arm'] == arm and field in r['metrics']]
                  for arm in (0, 1)}
        if not values[0] and not values[1] and field.startswith('cuda_'):
            continue
        if not all(len(values[arm]) == rounds for arm in (0, 1)):
            raise RuntimeError(f'Incomplete metric: {field}')
        if not all(math.isfinite(v) and v >= 0 for arm in (0, 1) for v in values[arm]):
            raise RuntimeError(f'Invalid metric: {field}')
        if field.endswith('_ms') and any(v == 0 for arm in (0, 1) for v in values[arm]):
            raise RuntimeError(f'Nonpositive timing: {field}')
        before, after = [statistics.median(values[arm]) for arm in (0, 1)]
        result.append(dict(workload=rows[0]['workload'], metric=field, before=before, after=after,
                           change_percent=100*(after/before-1) if before else None,
                           after_lower_pairs=sum(b < a for a, b in zip(values[0], values[1])),
                           before_range=[min(values[0]), max(values[0])],
                           after_range=[min(values[1]), max(values[1])]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['phase', 'map'], required=True)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--before-revision', required=True)
    parser.add_argument('--after-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--rounds', type=int, default=6)
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2:
        parser.error('Use an even number of paired rounds, at least two')
    if args.kind == 'phase' and args.device != 'cpu':
        parser.error('Part splitting is a CPU phase, including in CUDA builds')
    executables = [args.before.resolve(strict=True), args.after.resolve(strict=True)]
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = [digest(path) for path in executables]
    receipt = dict(status='running', started=time.time(), kind=args.kind,
                   platform=platform.platform(), device=args.device, rounds=args.rounds,
                   revisions=[args.before_revision, args.after_revision],
                   environment={key: os.environ.get(key) for key in
                                ['CUDA_MODULE_LOADING', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
                   executable_sha256=hashes, runs=[])

    def save():
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')

    workloads = ([('clustered16k', ['--layout', 'clustered', '--n', '16384']),
                  ('clustered131k', ['--layout', 'clustered', '--n', '131072']),
                  ('clustered1m', ['--layout', 'clustered', '--n', '1048576']),
                  ('single131k', ['--layout', 'single', '--n', '131072']),
                  ('unique131k', ['--layout', 'unique', '--n', '131072'])]
                 if args.kind == 'phase' else
                 [('center', ['--modes', 'center']), ('volume', ['--modes', 'volume'])])
    save()
    summary = []
    try:
        for name, extra in workloads:
            for rnd in range(1, args.rounds + 1):
                for arm in ([0, 1] if rnd % 2 else [1, 0]):
                    controls = (['--reps', '15', '--warmup', '3', '--arm', str(arm)]
                                if args.kind == 'phase' else
                                ['--n', '2048', '--depth', '3', '--batch', '16', '--samples', '64',
                                 '--epochs', '4', '--part-reuse-mb', '512', '--batch-device', args.device])
                    command = [str(executables[arm]), *controls, *extra]
                    start = time.time()
                    result = subprocess.run(command, cwd=executables[arm].parent, capture_output=True,
                                            text=True, encoding='utf-8')
                    stem = f'{name}-r{rnd}-a{arm}'
                    (args.output / f'{stem}.txt').write_text(result.stdout + result.stderr, encoding='utf-8')
                    row = dict(workload=name, round=rnd, arm=arm, command=command,
                               started=start, elapsed=time.time()-start, exit=result.returncode)
                    receipt['runs'].append(row)
                    if result.returncode:
                        raise RuntimeError(f'{stem} exited with {result.returncode}')
                    metrics = (json.loads(result.stdout) if args.kind == 'phase' else
                               {m.group(1): float(m.group(2)) for line in result.stdout.splitlines()
                                if (m := re.fullmatch(r'([a-zA-Z_0-9]+)\s+([0-9.eE+\-]+)', line))})
                    metric = 'median_ms' if args.kind == 'phase' else 'kmap_batch_later_epochs_ms'
                    if not math.isfinite(metrics[metric]) or metrics[metric] <= 0:
                        raise RuntimeError(f'{stem} has an invalid timing')
                    row['metrics'] = metrics
                    save()
            rows = [r for r in receipt['runs'] if r['workload'] == name]
            summary.extend(summarize(rows, args.kind, args.rounds))
        if [digest(path) for path in executables] != hashes:
            raise RuntimeError('Executable changed during measurement')
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        receipt['status'] = 'passed'
    except BaseException as error:
        receipt['status'] = 'failed'
        receipt['error'] = repr(error)
        raise
    finally:
        receipt['completed'] = time.time()
        save()

    print(json.dumps({'status': receipt['status'], 'runs': len(receipt['runs']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
