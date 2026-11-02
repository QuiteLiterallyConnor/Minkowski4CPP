"""Independently check final CTest, sanitizer and binary identities."""
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
validation, sanitizer, benchmark = [read(ROOT / p) for p in
    ('validation/receipt.json', 'sanitize/receipt.json', 'ab/receipt.json')]
assert all(r['status'] == 'passed' for r in (validation, sanitizer, benchmark))
assert validation['head'] == benchmark['head'] == 'b96db27aa5ef68e67d8c8bc3b3e6fa3a7548dcac'
configurations = {}
for config in validation['configurations']:
    kind = config['kind']
    assert kind not in configurations and config['build_exit'] == config['ctest_exit'] == 0
    xml = ET.parse(ROOT / f'validation/ctest-{kind}.xml').getroot()
    assert xml.attrib['tests'] == '4' and xml.attrib['failures'] == '0'
    assert len(xml.findall('testcase')) == 4
    binaries = []
    for artifact in config['binaries']:
        path = Path(artifact['Path'])
        assert sha(path).lower() == artifact['Hash'].lower()
        binaries.append(dict(name=path.name, sha256=sha(path)))
    test_log = Path(r'C:\Users\User\Work\mink-build') / ('gauss-cull-' + kind) / 'Testing/Temporary/LastTest.log'
    counts = re.findall(r'(\d+) passed, (\d+) failed, (\d+) skipped', test_log.read_text(encoding='utf-8-sig'))
    assert len(counts) == 2 and all(int(row[1]) == 0 for row in counts)
    configurations[kind] = dict(ctest_suites=4, suite_counts=counts, binaries=binaries)
assert set(configurations) == {'cpu', 'cuda'}
assert len(sanitizer['tools']) == 3 and sanitizer['exclusions'] == []
for tool in sanitizer['tools']:
    name = tool['tool']
    assert tool['exit'] == 0
    log = (ROOT / f'sanitize/{name}.txt').read_text(encoding='utf-8-sig')
    assert '3 passed, 0 failed, 68 skipped' in log
    assert (('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in log)
            if name == 'racecheck' else ('ERROR SUMMARY: 0 errors' in log))
cuda_test = Path(r'C:\Users\User\Work\mink-build\gauss-cull-cuda\gaussian_tests.exe')
assert sha(cuda_test).lower() == sanitizer['sha256'].lower()
for name, key in (('gaussian_raster_backward_bench.exe', 'raster_sha256'), ('gaussian_render_bench.exe', 'render_sha256')):
    assert sha(cuda_test.with_name(name)).lower() == benchmark[key].lower()
result = dict(status='passed', source=validation['head'], configurations=configurations,
              tools=sanitizer['tools'], sanitizer_exclusions=[],
              invariants=['4/4 complete CTests in each build', 'zero suite failures',
                          'live test and benchmark executable SHA256 matches receipts',
                          'all three sanitizer filtered tests pass under every tool',
                          'memcheck/initcheck zero errors; racecheck zero hazards/errors/warnings'])
(ROOT / 'validation-audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print('AUDIT PASSED: native CPU/CUDA CTest4/4 each and all three sanitizer tools')
