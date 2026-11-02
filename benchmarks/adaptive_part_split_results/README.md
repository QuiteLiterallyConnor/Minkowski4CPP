# Compact adaptive part splitting

Measured runtime: `95093e2938e939ad5dd954c33ae4ea583879fa4d`.
Baseline branch: `647450b00e4c16c7b6b60e95f8823231ebc35214` (PR52 documentation
head, runtime `2623c56`). This change is stacked on PR52 and its existing adaptive
dependencies. Native timing and correctness evidence below is distinct from Docker
validation; container receipts will be added separately when that experiment completes.

## Problem and implementation

Profiling in the earlier adaptive experiment identified about 1.11 ms per training
step in splitting/probing. The former implementation sorted one level-zero x value
for every leaf, then binary-searched part boundaries for every input/output leaf.
Immutable topology keys already have a canonical order: level first, then coordinates.
Within a level, level-zero x is nondecreasing. Many refined leaves share one x cell.

The new internal helpers remove repeated x cells within each level before sorting
the compact list. Single-level input needs no additional sort. Leaf assignment uses
a forward cursor within each level, with one binary search when a new level starts.
Part boundaries, integer floor semantics for negative coordinates, leaf order,
weights, cache policy and public APIs are unchanged. This only affects the opt-in
part-reuse path; it introduces no CUDA kernel or numeric reduction change.

If N is the leaf count and U the count of distinct (level, level-zero x) runs,
boundary construction changes from O(N log N) sorting to O(N + U log U). U may
equal N for sparse inputs. Cursor assignment takes O(N + L*P) in the worst case
for L occupied levels and P parts, with a binary search at each level's start.
This is an explicit tradeoff against the former O(N log P) assignment. The unique-x
control below covers a less compact workload; it is not a proof for every layout.
The sorted temporary contains fewer elements on clustered inputs, but whole-process
memory measurements do not establish a material working-set reduction.

## Validation and method

Native Release builds were incrementally rebuilt from the changed source using the
existing development environment. CPU CTest: 5/5 (adaptive 133 passed/8 skipped,
split 6/0, Gaussian 36 passed/4 skipped, plus smoke/framework). CUDA CTest: 5/5
(adaptive 144/0, split 6/0, Gaussian 42/0, plus smoke/framework). Full logs and
JUnit are included. The six added differential tests cover empty/single inputs,
negative floors and level resets, exact radius boundaries, repeated cells,
coordinates near +/-2^52, sparse levels and 250 deterministic randomized cases.
Existing part-cache tests check exact map indices/order/weights and forward/backward
CPU/CUDA behavior, including cache eviction and stream lifetime.

All comparisons use six paired rounds with order A/B then B/A alternating. Phase
runs use three warmups and 15 timed iterations per process, on sorted synthetic keys.
The baseline reference retains the original full sort and per-leaf binary searches;
the changed arm calls the production helpers. Synthetic keys isolate this integer
phase and are not presented as complete valid adaptive scene topologies.

Complete batch-map runs instead use valid adaptive tensors: 64 ellipsoid samples,
2048 points each, refinement depth 3, batch16, four epochs, 512 MiB host part cache,
unlimited device-part retention, and CenterSampled or VolumeWeighted 3x3x3 kernels.
Sample seeds are100+s; epoch shuffle seed is0. Each process reports median first-epoch
batch time and median later-epoch batch time across the batch and its pooled stages.
The first epoch covers initially uncached samples; later epochs measure reuse.
CUDA operations synchronize at the timing end. That phase may include outstanding
preparation work queued before the timer; it is not isolated GPU kernel duration.
CUDA map comparisons explicitly use CUDA_MODULE_LOADING=EAGER in both arms.
The final portable runner was also checked with six fail-closed contract tests:
`python3 benchmarks/test_adaptive_part_split_compare.py`. These reject incomplete
pairs, changed checksums/workloads/cache metadata and missing/nonfinite metrics.

Raw receipts bind process commands, exit status and executable SHA256. Phase checksums
and complete-map leaf/pair/cache-hit/miss counts are identical across all compared
processes. Native source/build identities and old executable hashes are preserved.
The host is shared and CPU load is high; timings are contended observations.

## Split-and-assign phase (milliseconds)

| Workload / metric | Original median [min, max] | Changed median [min, max] | Change | Changed lower |
|---|---|---|---|---|
| clustered16k / median_ms | 0.4099 [0.3255, 0.6116] | 0.1909 [0.1567, 0.2698] | -53.43% | 6/6 |
| clustered16k / peak_process_mib | 5.4551 [5.4414, 5.9141] | 5.4004 [5.3945, 5.4180] | -1.00% | 6/6 |
| clustered131k / median_ms | 3.0564 [2.2140, 3.7021] | 1.3765 [1.1500, 1.7825] | -54.96% | 6/6 |
| clustered131k / peak_process_mib | 11.7441 [11.7383, 11.7500] | 11.7422 [11.7227, 11.7617] | -0.02% | 3/6 |
| clustered1m / median_ms | 23.8377 [17.3551, 27.0397] | 13.4992 [10.3906, 17.9616] | -43.37% | 6/6 |
| clustered1m / peak_process_mib | 54.1426 [54.1328, 54.1484] | 54.1348 [54.1250, 54.2812] | -0.01% | 4/6 |
| single131k / median_ms | 2.3876 [1.8178, 2.6623] | 0.9368 [0.8942, 1.5063] | -60.76% | 6/6 |
| single131k / peak_process_mib | 10.8340 [10.8242, 10.8555] | 10.8242 [10.8242, 10.8594] | -0.09% | 4/6 |
| unique131k / median_ms | 19.8478 [12.9162, 24.8629] | 15.6759 [9.8855, 18.6439] | -21.02% | 6/6 |
| unique131k / peak_process_mib | 20.5762 [20.5312, 20.6172] | 20.5566 [20.5195, 20.6250] | -0.09% | 3/6 |

The peak-process figures include a common correctness precheck that runs both
algorithms. They are not isolated temporary-memory measurements and must not be
used to claim the cache itself saves those amounts of RAM. The latency metric times
construction/assignment; object destruction is outside each sample.

## Complete CPU batch-map phase

| Workload / metric | Original median [min, max] | Changed median [min, max] | Change | Changed lower |
|---|---|---|---|---|
| center / kmap_batch_first_epoch_ms | 148.3785 [143.4990, 177.2560] | 137.4425 [131.1630, 154.3710] | -7.37% | 6/6 |
| center / kmap_batch_later_epochs_ms | 23.7925 [20.9480, 33.5640] | 19.2345 [18.1450, 20.5770] | -19.16% | 6/6 |
| center / peak_working_set_mb | 1894.4845 [1891.5470, 1896.7300] | 1895.1110 [1893.2970, 1897.9960] | +0.03% | 2/6 |
| volume / kmap_batch_first_epoch_ms | 170.6435 [139.9110, 195.0270] | 167.1530 [147.3070, 175.4240] | -2.05% | 4/6 |
| volume / kmap_batch_later_epochs_ms | 25.0485 [20.9920, 29.7160] | 22.0835 [20.8040, 29.1820] | -11.84% | 5/6 |
| volume / peak_working_set_mb | 1935.0780 [1933.1020, 1936.7500] | 1934.7325 [1933.2190, 1938.1990] | -0.02% | 2/6 |

## Complete CUDA batch-map phase

| Workload / metric | Original median [min, max] | Changed median [min, max] | Change | Changed lower |
|---|---|---|---|---|
| center / kmap_batch_first_epoch_ms | 139.7570 [101.6810, 262.1070] | 141.7780 [121.2720, 201.6000] | +1.45% | 4/6 |
| center / kmap_batch_later_epochs_ms | 12.0645 [11.4130, 14.7090] | 10.7460 [9.9730, 11.8010] | -10.93% | 6/6 |
| center / peak_working_set_mb | 3562.7540 [3559.3200, 3567.9610] | 3562.0235 [3558.7070, 3563.5780] | -0.02% | 4/6 |
| center / cuda_max_alloc_mb | 784.7680 [784.7680, 784.7680] | 784.7680 [784.7680, 784.7680] | +0.00% | 0/6 |
| center / cuda_max_reserved_mb | 818.0000 [818.0000, 818.0000] | 818.0000 [818.0000, 818.0000] | +0.00% | 0/6 |
| volume / kmap_batch_first_epoch_ms | 163.7640 [134.5310, 204.0040] | 164.7095 [130.0110, 245.0780] | +0.58% | 3/6 |
| volume / kmap_batch_later_epochs_ms | 13.5580 [12.7910, 16.8650] | 12.2455 [11.6420, 16.9340] | -9.68% | 5/6 |
| volume / peak_working_set_mb | 3604.1915 [3599.9650, 3611.0740] | 3603.7925 [3601.2230, 3608.5620] | -0.01% | 4/6 |
| volume / cuda_max_alloc_mb | 1319.6520 [1319.6520, 1319.6520] | 1319.6520 [1319.6520, 1319.6520] | +0.00% | 0/6 |
| volume / cuda_max_reserved_mb | 1366.0000 [1366.0000, 1366.0000] | 1366.0000 [1366.0000, 1366.0000] | +0.00% | 0/6 |

## Derived map-phase throughput

| Device / later-epoch mode | Original map-phase calls/s | Changed map-phase calls/s | Change |
|---|---:|---:|---:|
| cpu / center | 42.03 | 51.99 | +23.70% |
| cpu / volume | 39.92 | 45.28 | +13.43% |
| cuda / center | 82.89 | 93.06 | +12.27% |
| cuda / volume | 73.76 | 81.66 | +10.72% |

These rates are reciprocals of the reported map-phase medians. They describe that
timed phase only, with identical 16-sample batches, and exclude model compute,
dataset conversion and total process overhead. They are not training throughput.

Memory units are MiB; latency units are milliseconds. CUDA allocated/reserved values
are process allocator peaks, not whole-device VRAM. CPU process peaks include all
dataset construction and execution. No model accuracy, convergence or end-to-end
training speedup is inferred from map timings. No CUDA kernel was changed; isolated
kernel time and GPU utilization improvements are not claimed. Tiny working-set
differences and overlapping timing ranges should be treated cautiously. Every
observed regression is retained in these tables.

## Reproduction

The measured host is Windows11, Ryzen5 5600X (6 cores/12 threads), 16 GiB RAM,
RTX2080SUPER 8 GiB/SM75, driver610.88, MSVC19.44, CMake4.2.3, CUDA13.4.3
(nvcc13.4.92), LibTorch2.15.0. The existing native toolchain was reused; no host
development environment was installed. Original lab runners retain their absolute
paths in this directory. The portable Python runner is
`benchmarks/adaptive_part_split_compare.py`; use fresh result directories.

Build original and changed checkouts in the existing Docker image with separate
build volumes. Record its immutable digest. Inside the image, use `/opt/libtorch`
and build mounted source, so the image's installed old library cannot mask changes:

```bash
cmake -S /workspace/source -B /workspace/build -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/libtorch \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DMINK_BUILD_TESTS=ON
cmake --build /workspace/build --parallel 2
ctest --test-dir /workspace/build --output-on-failure
python3 /workspace/source/benchmarks/adaptive_part_split_compare.py \
  --kind phase --before /workspace/build/adaptive_split_bench \
  --after /workspace/build/adaptive_split_bench \
  --before-revision 647450b --after-revision 95093e2 \
  --output /workspace/results/split-phase
python3 /workspace/source/benchmarks/adaptive_part_split_compare.py \
  --kind map --device cuda --before /workspace/baseline/adaptive_kmap_bench \
  --after /workspace/build/adaptive_kmap_bench \
  --before-revision 647450b --after-revision 95093e2 \
  --output /workspace/results/split-map-cuda
```

Select the target GPU architecture for another machine. Set CUDA_MODULE_LOADING=EAGER
for the CUDA map reproduction and record all load/environment differences. The
portable runner checks workload invariants and writes raw outputs, receipts and
summary tables in JSON. It does not grant hardware exclusivity: serialize timings
against all other builds and benchmarks. Never stop household services to obtain
quieter results. Different hardware/compiler/LibTorch results are new experiments.
