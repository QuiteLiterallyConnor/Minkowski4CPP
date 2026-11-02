# ModelNet10 representation pilot and measurement safeguards

This is a small systems comparison on real CAD objects, not an architecture-matched
study of representation quality. All four methods use the same 400 training objects,
200 held-out test objects, 1024 sampled points per object, six training epochs,
batch size 16, learning rate 0.001, data seed 2026 and model/shuffle seeds 1, 2, 3.
No hyperparameters were selected from these results. There is no validation split;
one must be introduced before any subsequent model or hyperparameter selection.

## Source and dataset identity

The measured runtime is the combined integration commit
`0a4b62a83a98b091497e996773c68b6f89c38a3c`, whose `ml_representations.exe` SHA256 is
`ec62ba12f3e29c4b4c23f53c0bbd51296c2e77e805d970c69f990eeef0c6c095`.
The standalone dataset feature is `0e5c4fe06084e0a71e67ce5f9659e573de46bbe3`.
The subsequent summary guards (`7460c7e`) and portable RSS reporting (`2ceceea`)
do not describe a second set of model training runs. The pilot used the original
Windows RSS implementation; the new Windows helper preserves its API and units.
Later integration builds and their test receipts have separate identities.

The official ModelNet10 ZIP SHA256 is
`9d8679435fc07d1d26f13009878db164a7aa8ea5e7ea3c8880e42794b7307d51`.
The processed dataset manifest SHA256 is
`983109bf609859eccc4ae302e1ae966fb6a5537032d6f8c139022779f5f36e6c`.
Training binary SHA256:
`ab5cf9c3d45fbacc8fa5e1e8ffec86cdab076276b736d4814e50342baafee3c4`.
Test binary SHA256:
`d4d0069e34f607a589af5e067510201a9a03d514d645d016ba1cad5d0825441f`.

Preprocessing preserves the archive's original train/test split and selects 40/20
objects per class using deterministic per-class shuffles. Faces are triangulated
as fans, surface points are sampled proportional to triangle area, and each mesh
is centered on its bounding box and rescaled to longest side 1.9. There is no
augmentation. The loader checks fixed local filenames, binary sizes and headers,
FNV-1a checksums, label histograms, coordinate bounds, and unique nonoverlapping
sample identities. SHA256 identities, all selected source paths and source hashes
are preserved in each raw result's full dataset manifest. The independent source
review found no overlapping IDs or exact mesh hashes across the selected splits;
all 600 source hashes were verified. This does not rule out near-duplicate meshes
or dataset-specific biases.

No CAD meshes or sampled point binaries are redistributed here. Download them
from the official source using `prepare_modelnet10.py --download` as shown below.

## Models and information supplied to them

| Method | Input channels | Trainable parameters | Features and architecture |
|---|---:|---:|---|
| uniform | 2 | 677,930 | Uniform 32-cubed sparse grid; occupancy and scaled log count; sparse convolution network |
| adaptive | 3 | 882,650 | Base cell 0.5, depth 3, density refinement; scaled log count, log count per finest-cell volume, normalized level; adaptive convolution network |
| gaussian | 11 | 110,218 | Per-Gaussian mean, absolute fitted normal, three sigmas, scaled log count and fitting cell size; set MLP with average/max aggregation |
| adaptive_gaussian | 9 | 885,242 | Same fitted Gaussian mixture mapped with Center splats; mass, normal, sigmas, log count and normalized level; adaptive convolution network |

Gaussian fitting starts with 0.25 cells and refines nonplanar cells to 0.0625
when their smallest-eigenvalue fraction exceeds 0.015 and they contain at least
16 points. Cells with fewer than three points are omitted. Covariance has
diagonal regularization `(0.05 * cell_edge)^2`. The combined method uses
`cell_per_sigma=sqrt(12)` and the existing Gaussian/adaptive bridge. These choices
give the methods different input information and different capacities. The
Gaussian set model receives coordinates directly; convolutional models receive
spatial coordinates through their operators. Matched-information and
capacity-matched ablations remain future work.

## Results

Means and sample standard deviations over the three matched seeds are in
[SUMMARY.md](SUMMARY.md), including precision, recall, F1 and all six accuracy
curve points. The following selected metrics come from exactly those 12 runs.

| Method | Accuracy % | Macro F1 % | Train seconds | Cached batch-1 inference ms | Batched evaluation samples/s | Active elements/sample |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 74.5 +/- 2.8 | 74.3 +/- 2.9 | 12.0 +/- 2.8 | 37.36 +/- 5.13 | 446 +/- 40 | 785.81 |
| adaptive | 78.0 +/- 1.7 | 77.6 +/- 1.7 | 6.7 +/- 0.4 | 10.86 +/- 0.36 | 948 +/- 11 | 368.89 |
| gaussian | 79.5 +/- 2.6 | 78.4 +/- 3.5 | 2.6 +/- 0.2 | 2.31 +/- 0.12 | 6738 +/- 225 | 147.02 |
| adaptive_gaussian | 80.8 +/- 1.3 | 80.6 +/- 1.0 | 5.8 +/- 0.3 | 11.85 +/- 1.66 | 1135 +/- 74 | 190.24 |

| Method | Conversion median ms/object | Peak allocated VRAM MiB | Peak reserved VRAM MiB | Lifetime process peak RSS MiB |
|---|---:|---:|---:|---:|
| uniform | 0.22 +/- 0.04 | 57 | 76 | 3777 +/- 24 |
| adaptive | 0.98 +/- 0.23 | 171 | 200 | 4078 +/- 13 |
| gaussian | 6.07 +/- 0.17 | 42 | 58 | 3771 +/- 0 |
| adaptive_gaussian | 5.93 +/- 0.64 | 94 | 120 | 3949 +/- 1 |

VRAM rounded values above are allocator peaks over the whole training loop,
including per-epoch test evaluation. Exact values are in the raw JSONs. Process
RSS includes data loading, conversion, model setup, training, evaluation and
inference. It is a lifetime high-water mark and cannot be subtracted to infer
individual phase peaks. Neither quantity is whole-device VRAM consumption.
Separate conversion/training/evaluation/inference memory experiments remain
unmeasured; the new JSON scope fields state this explicitly.

Adaptive inputs use 53.1% fewer active elements than uniform inputs in this
dataset, but their training-loop allocated VRAM is about three times larger.
Fewer active elements alone do not establish a memory advantage. The combined
method's observed accuracy is 6.3 percentage points above uniform, with the
architecture and information differences above. Three seeds and a 200-object
test subset do not establish statistical significance or general superiority.

Training seconds exclude each epoch's evaluation and representation conversion,
but include the first epoch and its cold initialization. Throughput times batched
evaluation, including batch assembly from precomputed representations. Batch-1
inference includes assembly, forward execution and device synchronization after
ten warmups, then measures up to 50 objects. It excludes points-to-representation
conversion. Conversion medians and inference medians are separate distributions;
adding them does not create a measured end-to-end latency. In particular, the
Gaussian conversion cost exceeds its cached inference cost. End-to-end
conversion-inclusive inference remains a required follow-up.

Trials ran under an exclusive lab benchmark lock, but unrelated applications
continued using the shared Build PC. Sampled CPU utilization was 23.8%-81.9%
and whole-device VRAM usage was about 6.8-7.1 GiB. Within each seed the method
order was fixed: uniform, adaptive, Gaussian, combined. These results are
contended pilot observations, not quiet-machine balanced A/B speedup estimates.
No performance percentage gain is attributed to the dataset loader or the RSS
reporting change, and no isolated kernel time was measured by this ML harness.

## Correctness and measurement changes

The external loader has 22 standalone C++ regression cases. Five Python
preprocessing tests cover OFF header variants, deterministic bounded samples,
area weighting, malformed/degenerate meshes, split isolation and binary roundtrip.
The measured integration commit passed native CPU and CUDA CTest 11/11 each.
Full logs, JUnit, build records and executable hashes are in `native_validation/`.

The summary now rejects incompatible dataset names, full manifests or train/test
SHA256 identities even if point counts, seeds and budgets match. Unidentified
historical procedural cohorts remain supported but cannot mix with explicitly
identified datasets. Twenty summary tests also reject malformed JSON, duplicate
seeds, incomplete curves, mismatched seed cohorts and invalid metrics. All pass.

The portable RSS helper preserves Windows PeakWorkingSetSize and adds
`getrusage(RUSAGE_SELF).ru_maxrss` on Linux (KiB) and Darwin (bytes), converting
each to MiB. Unsupported APIs return -1 instead of a fabricated zero. A
dependency-free test touches 64 MiB, checks plausible RSS growth and ensures
the high-water mark does not fall after release. The corrected native CTest
run passed all four focused suites: 22 loader cases, one RSS allocation test,
20 summary cases and five preprocessing cases. Its Windows RSS measurements
were 4.40625 -> 68.4219 -> 68.4219 MiB. Linux container execution is tracked
separately by the Docker evidence report; no local Linux result is implied here.

`audit.json` recomputes every scalar and per-class confusion metric and all
training-time sums, and verifies data/manifest/executable identities and the
complete cohort. `independent-benchmark-audit.json` is a second independently
implemented audit of all 12 ML runs. The before/after identity regression
accepted different training split hashes on the baseline and rejected them
with the guard. `manifest.json` binds the 46 raw evidence
files by SHA256. Verify those before regenerating the summary.

## Reproduction

Measured hardware: AMD Ryzen 5 5600X, six cores/twelve threads, 16 GiB system RAM,
NVIDIA RTX 2080 SUPER 8 GiB, driver 610.88. Native runtime: Windows 11, MSVC19.44,
CUDA13.4.3, LibTorch2.15.0, Release. Docker is unavailable on the Build PC;
container CPU builds use the existing pinned image on a GitHub Linux runner,
while the CUDA pilot is native supplemental evidence. No container GPU or
three-toolkit compatibility claim follows from these measurements.

Download and prepare the identical subset outside the repository:

```sh
python benchmarks/prepare_modelnet10.py --archive /data/ModelNet10.zip --download \
  --output /data/subset40-20-p1024-s2026 --train-per-class 40 --test-per-class 20 \
  --points 1024 --seed 2026
```

The portable feature branch can be built with
`-DMINK_BUILD_ML_EXPERIMENTS=ON`. For the recorded optimized pilot, checkout
`0a4b62a83a98b091497e996773c68b6f89c38a3c` from the research integration
history and use the existing Docker image or recorded native environment.
The integration-only cache/fusion options below are absent from the standalone
dataset feature's base; omitting them is a different runtime configuration.

```sh
cmake --build /build --parallel 2 --target ml_representations point_dataset_tests
CUDA_MODULE_LOADING=EAGER CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /build/ml_representations --method uniform --dataset-dir /data/subset40-20-p1024-s2026 \
  --device cuda --epochs 6 --seed 1 --batch 16 --kmap-part-reuse-mb 512 \
  --fused-apply 1 --deterministic 0 --output /results/uniform-seed1.json
```

Repeat for seeds 1,2,3 and all four method strings, using fresh processes and
preserving outputs. The original PowerShell commands, exact output paths,
return codes and timestamps are in `receipts/seed*-receipt.json`. Do not run
overlapping GPU benchmarks. Nondeterministic CUDA atomics remain enabled, so
fixed seeds preserve input/order/initialization but do not promise bitwise
cross-run or cross-toolkit reproducibility.

Regenerate the table without model execution:

```sh
python benchmarks/ml_representations_summary.py benchmarks/modelnet10_results/raw --seeds 1,2,3
python benchmarks/modelnet10_results/verify.py
ctest --test-dir /build -R 'point_dataset_tests|benchmark_memory_tests|ml_summary_tests|modelnet_preprocessing_tests' --output-on-failure
```

Container image digest, actual runtime test receipts and any Linux RSS result
belong in the accompanying Docker report. They must be checked before claiming
the whole feature has completed container validation.
