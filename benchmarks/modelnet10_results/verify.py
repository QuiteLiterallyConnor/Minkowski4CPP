"""Verify frozen raw evidence without requiring the original data or LibTorch."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
assert len({row['file'] for row in manifest}) == len(manifest), 'Duplicate manifest entries'
for row in manifest:
    path = (root / row['file']).resolve()
    assert path.is_relative_to(root), 'Evidence manifest escapes its directory'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], row['file']
identities = set()
for path in sorted((root / 'raw').glob('*.json')):
    row = json.loads(path.read_text(encoding='utf-8'))
    identity = (row['method'], row['seed'])
    assert identity not in identities
    identities.add(identity)
assert identities == {(method, seed) for method in ('uniform', 'adaptive', 'gaussian', 'adaptive_gaussian')
                      for seed in (1, 2, 3)}, 'Incomplete expected cohort'
print(f'Verified {len(manifest)} evidence SHA256 identities and all 12 expected trials')
