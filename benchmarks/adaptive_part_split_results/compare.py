"""Paired round-10 splitting comparisons; invoke under the lab bench lock."""
import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--kind', choices=['phase', 'map'], required=True)
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--rounds', type=int, default=6)
args = parser.parse_args()
build = Path(r'C:\Users\User\Work\mink-build') / ('device-compose-' + args.device)
source = Path(r'C:\Users\User\Work\mink-wt\device-compose')
out = BASE / (args.kind + '-' + args.device)
out.mkdir(exist_ok=True)
head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
if subprocess.check_output(['git', '-C', str(source), 'diff', '--name-only', 'HEAD'], text=True).strip():
    raise RuntimeError('Tracked source is dirty')
bins = [build / 'adaptive_split_bench.exe'] * 2 if args.kind == 'phase' else [
    build / 'adaptive_kmap_bench.split-baseline.exe', build / 'adaptive_kmap_bench.exe']
hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in bins]
receipt = dict(status='running', source=head, baseline_source='647450b00e4c16c7b6b60e95f8823231ebc35214',
               kind=args.kind, device=args.device, rounds=args.rounds, executables=[str(p) for p in bins],
               executable_sha256=hashes, started=time.time(), runs=[])
workloads = [('clustered16k', ['--layout', 'clustered', '--n', '16384']),
             ('clustered131k', ['--layout', 'clustered', '--n', '131072']),
             ('clustered1m', ['--layout', 'clustered', '--n', '1048576']),
             ('single131k', ['--layout', 'single', '--n', '131072']),
             ('unique131k', ['--layout', 'unique', '--n', '131072'])] if args.kind == 'phase' else [
             ('center', ['--modes', 'center']), ('volume', ['--modes', 'volume'])]
def save():
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
save()
try:
    for name, extra in workloads:
        for rnd in range(1, args.rounds + 1):
            for arm in ([0, 1] if rnd % 2 else [1, 0]):
                options = ['--reps', '15', '--warmup', '3', '--arm', str(arm)] if args.kind == 'phase' else [
                    '--n','2048','--depth','3','--batch','16','--samples','64','--epochs','4',
                    '--part-reuse-mb','512','--batch-device',args.device]
                cmd = [str(bins[arm]), *options, *extra]
                start = time.time()
                process = subprocess.run(cmd, cwd=build, text=True, encoding='utf-8', capture_output=True)
                stem = f'{name}-r{rnd}-a{arm}'
                (out / (stem + '.txt')).write_text(process.stdout + process.stderr, encoding='utf-8')
                row = dict(workload=name, round=rnd, arm=arm, command=cmd, exit=process.returncode,
                           started=start, elapsed=time.time()-start)
                if process.returncode:
                    receipt['runs'].append(row)
                    raise RuntimeError(f'{stem} failed: {process.returncode}')
                if args.kind == 'phase':
                    row['metrics'] = json.loads(process.stdout)
                else:
                    row['metrics'] = {m.group(1):float(m.group(2)) for line in process.stdout.splitlines()
                                      if (m:=re.fullmatch(r'([a-zA-Z_0-9]+)\s+([0-9.eE+\-]+)',line))}
                    assert 'kmap_batch_later_epochs_ms' in row['metrics']
                receipt['runs'].append(row)
                save()
        print(name, 'completed', flush=True)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in zip(bins,hashes))
    receipt['status']='passed'
except BaseException as e:
    receipt['status']='failed'
    receipt['error']=repr(e)
    raise
finally:
    receipt['completed']=time.time()
    save()
summary=[]
for name,_ in workloads:
    rows=[r for r in receipt['runs'] if r['workload']==name]
    keys=['median_ms','peak_process_mib'] if args.kind=='phase' else ['kmap_batch_first_epoch_ms','kmap_batch_later_epochs_ms','peak_working_set_mb','cuda_max_alloc_mb','cuda_max_reserved_mb']
    for key in keys:
        values={arm:[r['metrics'][key] for r in rows if r['arm']==arm and key in r['metrics']] for arm in (0,1)}
        if not values[0]: continue
        assert len(values[0])==len(values[1])==args.rounds
        before,after=[statistics.median(values[a]) for a in (0,1)]
        summary.append(dict(workload=name,metric=key,before=before,after=after,
                            change_percent=100*(after/before-1),faster=sum(b<a for a,b in zip(values[0],values[1])),
                            before_range=[min(values[0]),max(values[0])],after_range=[min(values[1]),max(values[1])]))
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary),flush=True)
