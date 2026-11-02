# Fused CUDA Adam update

`GaussianTrainer` uses one CUDA kernel per contiguous float32 parameter tensor for
the Adam update. CPU tensors, other scalar types, and noncontiguous gradients use
the existing LibTorch path. The optimizer equations, parameter ordering, bias
correction, learning-rate schedules, epsilon placement, and public API are unchanged.
The internal `detail::cuda_fused_adam()` switch and the renderer benchmark's
`--fused-adam 0|1` option retain a same-binary reference path for testing.

## Floating-point behavior

Fusion must preserve the stored float32 intermediates of the separate operations:
the old first and second moments are multiplied by their beta and rounded before
the accumulation. The squared gradient is evaluated before multiplication by
`1 - beta2`. Explicit round-to-nearest multiplication and fused multiply-add make
those boundaries independent of the compiler's contraction decisions.

The scalar reciprocal for second-moment bias correction follows the linked
LibTorch version. Versions before 2.10 compute it in float32; 2.10 and later
compute it in float64 and then convert it to float32. This distinction matters
near cancellation between a parameter and its update. Primary implementation
references:

- [LibTorch 2.9 scalar division](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)
- [LibTorch 2.10 scalar division](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)
- [LibTorch 2.15 pointwise accumulation](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/cuda/DeviceAddCmulCdiv.cuh)

An unfinished implementation at `fbcc72c` used float32 reciprocal calculation
with LibTorch 2.15 and associated the second-moment expression differently. Its
100,003-element test failed after the first step: the maximum error was 1.02609
times `float_epsilon * (abs(parameter) + learning_rate)`. That quantity is a
scale-aware absolute tolerance, **not an IEEE ULP distance**. The worst element
had an ordinary gradient (`-0.0001782548352`) and equal moments; it was not a
denormal-gradient failure. Fixing only the reciprocal passed the existing bound
but left 506 differing parameter elements after six steps. Preserving the moment
rounding boundaries as well produced bit-identical parameters and both moments
at each of the six steps on LibTorch 2.15 / CUDA 13.4 / SM 75. The tests keep a
documented numerical tolerance for other supported builds; bit identity is an
observation on that configuration, not a cross-platform guarantee.

## Correctness coverage

`gaussian_tests gaussian_fused_adam` includes:

- Six updates of 100,003 elements with zero, tiny, ordinary, and huge gradients.
  Parameter error is bounded by one float epsilon of `abs(parameter) + lr` per
  step; moment comparisons are also elementwise so large elements cannot mask
  small-element errors (`rtol=1e-6`, `atol=1e-30` / `1e-40`).
- Cancellation fixtures over 16 steps and `beta2` values 0, 0.9, 0.999, and
  0.999999.
- Empty tensors; rejected CPU, float64, strided, and mismatched-length gradients.
- Execution on the current nondefault CUDA stream and a partial thread block.
- Eight complete training steps on 3,000 Gaussians, comparing all six parameter
  groups and reconstruction loss against the unchanged optimizer path.

The trainer test permits parameter-group relative error below `1e-4` and loss
relative error below `1e-5`, allowing the renderer's existing nondeterministic
gradient reductions. It does not establish long-run convergence or accuracy on
an independent dataset.

## Reproduction

Use the existing image and bind-mounted source, then build the changed source in
the mounted directory. This prevents an image's older installed library from
being selected accidentally. From the repository root on a Docker GPU host:

```sh
docker pull acaicia/minkowski4cpp
docker image inspect acaicia/minkowski4cpp --format '{{json .RepoDigests}}'
docker run --rm --gpus all -v "$PWD:/src" -w /src acaicia/minkowski4cpp \
  bash -lc 'export TORCH_CUDA_ARCH_LIST=7.5; \
    cmake -S . -B build-adam -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/libtorch -DCMAKE_CUDA_ARCHITECTURES=75 && \
    cmake --build build-adam -j4 && \
    ctest --test-dir build-adam --output-on-failure && \
    build-adam/gaussian_tests gaussian_fused_adam'
```

An existing configured build can be reused. Adjust the architecture to the host
GPU and use the image's LibTorch prefix if its CMake configuration requires it.
The commands above are instructions for a Docker host, not evidence of a local
Docker run. The research host had no Docker executable; recorded native builds
used the versions pinned by `docker/cuda-profile.sh` and the PR's Docker CI is
reported separately.

For the end-to-end comparison, run both settings in balanced alternating order
(0,1 in odd rounds; 1,0 in even rounds), with exclusive access to the GPU and no
concurrent compilation. Use at least ten paired rounds:

```sh
build-adam/gaussian_render_bench --input tests/data/test.ply --counts 0 \
  --sizes 512,1024 --devices cuda --warmup 3 --iterations 10 --train \
  --fused-adam 0 --output aten-r1.json
build-adam/gaussian_render_bench --input tests/data/test.ply --counts 0 \
  --sizes 512,1024 --devices cuda --warmup 3 --iterations 10 --train \
  --fused-adam 1 --output fused-r1.json
```

Timers synchronize the device before and after each sample. Report medians of
round medians, range across rounds, paired win count, and host contention.
`cuda_peak_allocated_train` and `cuda_peak_reserved_train` reset peak statistics
before trainer construction and sample them after training. The older forward
and backward peak fields describe rendering and are sampled before training;
they must not be presented as optimizer memory measurements. Reserved memory
can include the allocator cache from the earlier render workload. CPU peak
working set is the process lifetime high-water mark, not a trainer-only measure.

For CUDA memory diagnostics, use Compute Sanitizer with `--error-exitcode 86`
and the test filter above. Report any kernel exclusions explicitly: on the
research host, the known LibTorch `indexing_backward_kernel` issue required an
exclusion in the whole-trainer sanitizer run. The fused Adam kernel remained
instrumented. Kernel-only runtime and allocation profiling should be reported
separately from end-to-end trainer latency.

## Round 9 measured results

Source `977bd71e2dc87ad718e5cf323252e2347dc3d0c3`, based on upstream
`f4ac612` plus the allocator namespace compatibility fix from PR #32. This
standalone experiment does not include the projection, tile-list, or SSIM fusions
from PRs #48–#50. The baseline is the unchanged ATen optimizer path selected with
`--fused-adam 0` in the same executable.

All 20 benchmark processes succeeded. Each rendered the 139,410-Gaussian
`tests/data/test.ply` scene at 512 and 1024 pixels, with three warm-ups and ten
timed samples. Ten paired rounds alternated arm order. An exclusive lock
prevented team compilation or other team GPU work during the experiment.
**The host was contended:** CPU utilization sampled before and after the run
was 40.7% and 54.6%. GPU clocks were not fixed. Values below are medians of
round medians; brackets show the range across rounds.

| Image | Metric | ATen Adam | Fused Adam | Change | Faster pairs |
|---|---|---:|---:|---:|---:|
| 512 | Training step, ms | 25.787 [24.181–32.873] | 24.268 [23.310–27.006] | -5.89% | 9/10 |
| 1024 | Training step, ms | 43.438 [42.367–52.960] | 43.079 [40.860–58.813] | -0.83% | 5/10 |
| 512 | Reciprocal median throughput, steps/s | 38.78 | 41.21 | +6.26% | — |
| 1024 | Reciprocal median throughput, steps/s | 23.02 | 23.21 | +0.83% | — |
| 512 | Training-phase CUDA allocated peak, MiB | 330.728 | 330.765 | +0.01% | — |
| 1024 | Training-phase CUDA allocated peak, MiB | 558.366 | 558.830 | +0.08% | — |
| 512 | Training-phase CUDA reserved peak, MiB | 408 | 400 | -1.96% | — |
| 1024 | Training-phase CUDA reserved peak, MiB | 702 | 716 | +1.99% | — |
| 512 | Process peak working set, MiB | 1153.535 | 1124.943 | -2.48% | — |
| 1024 | Process peak working set, MiB | 1164.625 | 1136.033 | -2.46% | — |

The 1024-pixel timing result is inconclusive, and its reserved-memory increase
is a regression. Even unchanged renderer-only control measurements moved by
about -4% in their pooled medians, with mixed paired wins. These observations
limit attribution of small end-to-end differences on this shared host; no
universal speedup or GPU-memory reduction is claimed. Kernel-only time and
cold-start time were not measured. Whole-run `nvidia-smi` samples included both
arms and initialization (75 samples; utilization median 64%, range 1–92%) and
must not be used as an arm-specific utilization improvement.

The allocation peak includes model cloning, optimizer initialization and
persistent scene/probe tensors. The reservation peak can retain cached blocks
from preceding render phases. Neither is an optimizer-only or whole-device
VRAM metric. The slight allocation differences can include allocator behavior
and renderer/training trajectories; they are not evidence of lower optimizer
storage.

Validation of this implementation:

| Configuration | Adaptive pass/fail/skip | Gaussian pass/fail/skip | CTest |
|---|---|---|---|
| CPU-only | 119/0/1 | 36/0/4 | 4/4 |
| CUDA | 123/0/0 | 47/0/0 | 4/4 |

All five Adam cases also passed Compute Sanitizer with zero errors: memcheck
excluded only the known LibTorch `indexing_backward_kernel` issue; racecheck
instrumented `adam_kernel`; initcheck used full instrumentation. A previous
initcheck run filtered to only Adam failed with uninitialized-source reports
after ATen initialization writes were excluded. The same binary passed the
unfiltered run; the failed filtered experiment is retained in the evidence
history. Sanitizer runs also varied the renderer's nondeterministic trainer
gradients slightly (up to 2.08e-7 relative), within the unchanged tolerance.

Hardware: AMD Ryzen 5 5600X (6 cores/12 threads), 16 GiB-class RAM, NVIDIA
RTX 2080 SUPER (8 GiB, sm_75), driver 610.88, Windows 11 Pro 10.0.26200,
MSVC 19.44, LibTorch 2.15.0+cu134, CUDA toolkit profile 13.4.3
(nvcc 13.4.92), Release. Docker was unavailable locally; Docker CI compilation
is separate from native execution, and older LibTorch runtime parity remains
unverified.

The [machine-readable evidence](benchmarks/gaussian_adam_round9.json) contains
every raw timing sample, arm/round identity, test summaries, hardware,
executable/scene hashes, memory measurements and failure history. It is enough
to recompute this table without relying on the rounded values printed here.
