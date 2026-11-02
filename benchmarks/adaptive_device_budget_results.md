## SUMMARY

Add an independent, optional byte budget for CUDA copies retained by adaptive kernel-map part reuse. PR #47 batches uploads into shared storage; a host part byte budget alone cannot limit device retention, and dropping one tensor view does not release a slab still owned by another part. The default stays unlimited. This is a memory/throughput control, not a claim that a smaller cache always trains faster.

This branch is stacked on PR #47 (and its #41 dependencies). The new implementation starts at `d8b8e04`; the tested runtime source is `2623c56b933d7cc90d97f337778a49aa3ebb5e40`. Later reporting commits do not alter the tested implementation.

## TECHNICAL DETAILS

- `KernelMapCache::set_part_device_bytes(size_t)` controls retained part uploads across CUDA devices. Zero disables retention; `size_t(-1)` preserves existing unlimited retention.
- Whole upload groups are evicted in least-recently-used order. Weak ownership does not retain host maps. A partially expired group remains conservatively charged at its original size until eviction; counters are not exact remaining allocator bytes in that case.
- Host kernel maps remain cached. Oversized uploads are transient. Concurrent/in-flight compositions hold their own tensor handles.
- Source allocations are recorded on the composition's current CUDA stream before kernel launch, so eviction cannot recycle cross-stream source storage prematurely.
- Repeated references to one stored part share an upload. New stats expose retained/peak group charge, groups, and unique part upload hits/misses.
- Batch benchmark adds `--part-device-mb` plus CUDA allocator peak allocated/reserved measurements.

## TESTING

Final native validation at `2623c56`:

| Backend | CTest | Adaptive cases (pass/fail/skip) | Gaussian cases |
|---|---:|---:|---:|
| CPU | 4/4 pass | 133/0/8 | 36/0/4 |
| CUDA | 4/4 pass | 144/0/0 | 42/0/0 |

CTest also runs the adaptive-versus-uniform benchmark smoke check and nine test-framework contract checks. The five new CUDA budget tests all pass. Compute Sanitizer: memcheck 5/5, zero errors and zero leaked bytes; racecheck cross-stream case 1/1, zero hazards; initcheck 5/5, zero errors. All sanitizer invocations use the existing `--kernel-name-exclude kns=indexing_backward_kernel` workaround for LibTorch indexing backward instrumentation; these are not unfiltered checks of every ATen kernel. Float64 forward and backward checks require exact equality against uncached maps; stream/width tests compare map indices and weights exactly.

The existing three-profile GitHub Docker build matrix (CUDA 124/132/134) is requested on this PR. Its status must be checked separately; native runtime tests are not Docker execution. The workflow compiles containers and does not run GPU tests on the hosted runner.

Five new tests exercise bounded/disabled retention with exact deterministic float64 forward/backward comparisons; whole-group eviction and live output maps; LRU order and repeated references; cross-stream eviction with allocator churn; and mixed 16/32-bit upload groups after partial host eviction. CPU execution preserves the API and skips CUDA-specific cases explicitly.

## BENCHMARKS

Results measured on 2026-11-02. Two warm-up processes per arm and mode, followed by six measured processes using all six arm-order permutations. Each process uses 64 samples, batch 16, 2048 points/sample, depth 3, four epochs, host part budget 512 MiB, CUDA device composition. Table values are medians [minimum, maximum]; lower latency/memory is better. The bounded arm uses 16 MiB. A global benchmark lock excluded team builds and other team GPU jobs, but unrelated host work remained (CPU 25.1% before/37.1% after). Treat small differences as contended-host measurements.

The timer starts after host batch/pool preparation and synchronizes at the end; without a pre-timer CUDA synchronization it can include queued preparation work. This is a synchronized batch-map phase, **not isolated kernel time**. Allocator peaks cover the whole process (batch/pool/cache/temporaries), and retained group charge is neither reserved VRAM nor exact live payload after partial expiration.

### Original versus new default unlimited mode

| Mode | Metric | Before | After | Change | Lower in paired rounds |
|---|---|---:|---:|---:|---:|
| center | First epoch batch phase (ms) | 127.079 [107.797, 159.233] | 121.447 [108.490, 133.207] | -4.43% | 4/6 |
| center | Warm batch phase (ms) | 9.722 [8.567, 10.896] | 10.267 [8.686, 11.079] | +5.61% | 1/6 |
| center | Reciprocal median warm phase (batches/s) | 102.86 | 97.40 | -5.31% | derived, not end-to-end throughput |
| center | Peak CPU working set (MiB) | 1059.438 [1055.078, 1060.988] | 1061.850 [1057.809, 1063.031] | +0.23% | 2/6 |
| volume | First epoch batch phase (ms) | 138.548 [125.715, 154.423] | 146.902 [126.790, 155.916] | +6.03% | 2/6 |
| volume | Warm batch phase (ms) | 11.922 [11.062, 12.188] | 11.223 [9.991, 11.846] | -5.86% | 5/6 |
| volume | Reciprocal median warm phase (batches/s) | 83.88 | 89.10 | +6.23% | derived, not end-to-end throughput |
| volume | Peak CPU working set (MiB) | 1121.668 [1119.789, 1126.715] | 1121.798 [1117.633, 1122.957] | +0.01% | 3/6 |

### New unlimited versus 16 MiB retention cap

| Mode | Metric | Before | After | Change | Lower in paired rounds |
|---|---|---:|---:|---:|---:|
| center | First epoch batch phase (ms) | 121.447 [108.490, 133.207] | 127.876 [119.916, 137.031] | +5.29% | 3/6 |
| center | Warm batch phase (ms) | 10.267 [8.686, 11.079] | 11.537 [10.879, 11.856] | +12.36% | 0/6 |
| center | Reciprocal median warm phase (batches/s) | 97.40 | 86.68 | -11.00% | derived, not end-to-end throughput |
| center | Peak CPU working set (MiB) | 1061.850 [1057.809, 1063.031] | 1058.756 [1055.594, 1063.555] | -0.29% | 5/6 |
| center | Peak CUDA allocated (MiB) | 784.768 [784.768, 784.768] | 781.068 [781.068, 781.068] | -0.47% | 6/6 |
| center | Peak CUDA reserved (MiB) | 818.000 [818.000, 818.000] | 818.000 [818.000, 818.000] | +0.00% | 0/6 |
| center | Final retained group charge (MiB) | 20.058 [20.058, 20.058] | 15.376 [15.376, 15.376] | -23.34% | 6/6 |
| center | Peak retained group charge (MiB) | 20.058 [20.058, 20.058] | 15.986 [15.986, 15.986] | -20.30% | 6/6 |
| center | Unique part upload misses | 96.000 [96.000, 96.000] | 190.000 [190.000, 190.000] | +97.92% | 0/6 |
| volume | First epoch batch phase (ms) | 146.902 [126.790, 155.916] | 130.306 [115.481, 145.489] | -11.30% | 5/6 |
| volume | Warm batch phase (ms) | 11.223 [9.991, 11.846] | 29.965 [24.923, 30.969] | +166.98% | 0/6 |
| volume | Reciprocal median warm phase (batches/s) | 89.10 | 33.37 | -62.54% | derived, not end-to-end throughput |
| volume | Peak CPU working set (MiB) | 1121.798 [1117.633, 1122.957] | 1151.153 [1148.145, 1152.961] | +2.62% | 0/6 |
| volume | Peak CUDA allocated (MiB) | 1319.652 [1319.652, 1319.652] | 1267.333 [1267.333, 1267.333] | -3.96% | 6/6 |
| volume | Peak CUDA reserved (MiB) | 1366.000 [1366.000, 1366.000] | 1324.000 [1324.000, 1324.000] | -3.07% | 6/6 |
| volume | Final retained group charge (MiB) | 60.175 [60.175, 60.175] | 6.942 [6.942, 6.942] | -88.46% | 6/6 |
| volume | Peak retained group charge (MiB) | 60.175 [60.175, 60.175] | 14.415 [14.415, 14.415] | -76.04% | 6/6 |
| volume | Unique part upload misses | 96.000 [96.000, 96.000] | 214.000 [214.000, 214.000] | +122.92% | 0/6 |

The 16 MiB cap cuts final retained charge 23.3% (center) /88.5% (volume), but increases warm phase latency 12.4% /167.0% in all six paired rounds. Volume peak CUDA allocation falls 3.96%; center falls 0.47%. Volume CPU working set increases 2.62%. These are explicit tradeoffs for a memory control, not an unconditional optimization. Default-unlimited overhead is mixed: center warm phase +5.61% (only 1/6 faster), volume -5.86% (5/6 faster), and opposite first-epoch movements. Retain the default unlimited budget unless memory pressure warrants the tradeoff. Baseline lacks new CUDA peak fields; no baseline GPU memory is fabricated. Kernel execution time and GPU utilization over the timed interval were not measured; endpoint GPU utilization is not representative.

Pilot exploration also tested budgets 0/8/16 before choosing 16 for the balanced experiment; those single-process pilot observations are not confirmation trials. The original test-only CUDA compilation failure (`detail` namespace ambiguity) is preserved in the local experiment history and corrected by 2623c56; final suites above were rebuilt and rerun.

## MACHINE LEARNING RESULTS

The feature preserves maps and gradients in automated tests. Full-stack classification comparing unlimited/32 MiB/zero retention is running separately at integration commit `f3f74919a64bede57fab7d5f303e94ca188c931a`; no accuracy, convergence, training speed, or model-quality conclusion is claimed before those results are available. A follow-up comment will preserve the separate integration provenance.

## TRADEOFFS

The cap excludes composed batch maps, allocator reservations, and live operation temporaries; it is not an instantaneous total-VRAM cap. Lower caps may increase uploads and latency. Group eviction may discard several parts when one large group exceeds the budget. Partially expired mixed-width groups are conservatively overcharged. Default unlimited-mode overhead is measured separately. Public stats fields expand the C++ struct: source compatibility is preserved after rebuilding consumers; this is not a binary-ABI compatibility claim.

Initial CUDA compilation found an ambiguous test-only `detail` namespace; `2623c56` qualified it and the final checks above passed after rebuilding. Docker is unavailable on this Windows test host; native tests reuse the existing toolchain matching the repository's CUDA 134 profile. Do not describe these as local container execution.

## REPRODUCTION

Hardware: Ryzen 5 5600X, 16 GiB RAM, RTX 2080 SUPER 8 GiB (SM 75), Windows 11, MSVC 19.44, CUDA 13.4.3, LibTorch 2.15.0+cu134 / CPU, CMake 4.2.3. Baseline `b1a7d94`; baseline static benchmark SHA256 `41BC667DA2A2B323D0AB06F15DD5786A0969D60965F48A8C28B22D706BECEE88`.

Existing-image container recipe (provided for reproduction; not locally executed):

```sh
docker pull acaicia/minkowski4cpp:latest
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace acaicia/minkowski4cpp:latest bash -lc '
  export TORCH_CUDA_ARCH_LIST=7.5
  cmake -S . -B build-budget -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch -DCMAKE_CUDA_ARCHITECTURES=75
  cmake --build build-budget --parallel 4
  ctest --test-dir build-budget --output-on-failure
  ./build-budget/adaptive_kmap_bench --batch 16 --samples 64 --epochs 4 --n 2048 --depth 3 --part-reuse-mb 512 --batch-device cuda --device-compose 1 --modes center --part-device-mb 16
'
```

Use a separate build directory and execute its newly linked benchmark, ensuring the installed image library is not substituted for modified source. Record the image digest and CUDA/LibTorch profile when reproducing. Repeat with volume mode and budgets -1/0/8/16; use balanced process order and exclusive GPU access. Actual native commands, final source/binary hashes and raw results will be linked in the final report.


Actual native build/test commands (existing local toolchain, no host dependencies installed):

```powershell
pwsh -File C:\Users\User\Work\mink-lab\tools\mink-build.ps1 -Src C:\Users\User\Work\mink-wt\device-compose -Build C:\Users\User\Work\mink-build\device-compose-cpu -Cpu -Targets adaptive_tests,adaptive_vs_uniform_bench,gaussian_tests,adaptive_kmap_bench,test_framework_contract -Who adaptive-budget
pwsh -File C:\Users\User\Work\mink-lab\tools\mink-build.ps1 -Src C:\Users\User\Work\mink-wt\device-compose -Build C:\Users\User\Work\mink-build\device-compose-cuda -Targets adaptive_tests,adaptive_vs_uniform_bench,gaussian_tests,adaptive_kmap_bench,test_framework_contract -Who adaptive-budget
ctest --test-dir C:\Users\User\Work\mink-build\device-compose-cpu -V --output-on-failure
ctest --test-dir C:\Users\User\Work\mink-build\device-compose-cuda -V --output-on-failure
```

The native runs used shared GPU/build locks around these commands. Build baseline b1a7d94 in a separate directory, then use the portable stdlib runner from this branch (outside shared GPU workloads):

```sh
python benchmarks/adaptive_device_budget_ab.py --baseline /path/to/baseline/adaptive_kmap_bench --candidate /path/to/candidate/adaptive_kmap_bench --output budget-ab --budget-mib 16 --rounds 6
```

The runner reproduces the original PowerShell arm order and commands, checks process failures and required metrics, and refuses to overwrite an existing result directory. Run output parser verified against all 48 original process outputs; CLI smoke check passed. The runner itself was not used to collect the table above. Benchmark source uses fixed synthetic seeds; no learned dataset participates in this map microbenchmark. Original raw metrics (including warmups), all comparisons, host-load observations and executable SHA256 values are checked in at `benchmarks/adaptive_device_budget_results.json`; source/report at `benchmarks/adaptive_device_budget_results.md`. The final candidate benchmark SHA256 is `9BFFFCE65FA01D4CC272F409F6FFD488DD7DB26D82B687CBE7FB16AB155B01BD`. Full native build/CTest/sanitizer logs remain at `C:\Users\User\Work\mink-lab\results\round9\adaptive`.

Future work: characterize pruning/eviction overhead on a quiet host, consider size-aware admission after profiling, and validate multi-GPU runtime behavior on suitable hardware. Single-GPU tests do not establish multi-GPU correctness.
