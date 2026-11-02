# Fused Gaussian attribute gathering

This experiment is stacked on PR #49 (`ba119b2aacceabe046166fcb3a37537129769942`),
which includes the projection and tile-list optimizations. It replaces eight
`index_select` calls plus `stack`/`cat` with one CUDA gather. The adjoint uses an
assignment scatter because the renderer produces a unique permutation subset
from visibility indices and a stable depth sort. CPU rendering and camera-pose
optimization continue through the existing path.

The screen-space mean remains a separate autograd input. Its retained gradient,
used by densification, is preserved. The four requested gradient arrays share
one zero-initialized allocation in disjoint contiguous regions. Frozen inputs
receive no gradient allocation. This is a first-order autograd primitive, like
the existing fused projection and rasterizer; no higher-order support is claimed.

## Experimental history

The first implementation, `d58ac07a23f840916a3f95b638620b1226a28322`, passed the
CUDA Gaussian suite (66/0/0), adaptive suite (123/0/0), CPU Gaussian suite
(47/0/5 skipped), and CPU adaptive suite (119/0/1 skipped). Four new tests
passed memcheck, racecheck and initcheck. The test-only follow-up `fbb3527`
added nondefault-stream/temporary-allocation coverage; CUDA Gaussian 67/0/0,
CPU unchanged, and five gather tests passed all three sanitizers.

Ten balanced A/B rounds on the initial implementation gave the following
medians. The host was contended (17â€“62% sampled CPU load); these are retained
as history, not substituted for the final implementation's measurements.

| Initial experiment | LibTorch gather | Fused gather | Change | Faster pairs |
|---|---:|---:|---:|---:|
| 512px train step | 10.65 ms | 10.16 ms | -4.6% | 8/10 |
| 1024px train step | 23.79 ms | 23.39 ms | -1.6% | 7/10 |
| 512px render forward+backward | 9.98 ms | 9.68 ms | -3.0% | 6/10 |
| 1024px render forward+backward | 14.98 ms | 15.25 ms | +1.9% | 5/10 |

Forward-only results were mixed; no universal latency reduction was established.
512px render backward peak allocation increased 195.18 to 196.52 MiB (+0.7%);
1024px decreased 233.73 to 232.51 MiB (-0.5%). Reserved memory and host peak
working set were essentially unchanged.

A follow-up regression exposed unnecessary retained storage with frozen inputs:
for 257 rows and only the two mean columns trainable, the gradient retained
16,448 bytes instead of the reference's 2,056 bytes. The initial implementation
allocated all four gradient arrays even when only one was requested. The final
implementation sizes the shared allocation from `needs_input_grad` and skips
unrequested scatter outputs. The regression checks means, projection, opacity
and color independently, including exact gradient values and retained storage.

## Final implementation validation and results

Runtime source: `d45b9112ac664688bfefbc79d302738dd7a4aecc`. Report-only commits
do not change the measured implementation. The immutable uninstrumented benchmark
SHA-256 is `33922770cc2a672c231828a7f25f277a826a73d54feac5145edb598c44df43d4`.
All 40 final subprocesses succeeded: render and train, two arms, ten rounds.
The table reports medians of per-process medians, with the range across rounds;
the paired wins compare the same round. Throughput is the reciprocal of latency,
not an independently measured rate. Memory values are in MiB (2^20 bytes).

| Final workload/metric | LibTorch gather | Fused gather | Change | Faster pairs |
|---|---:|---:|---:|---:|
| 512px render forward | 3.97 [3.67–4.60] ms | 3.88 [3.69–4.27] ms | -2.26% | 6/10 |
| 512px render forward+backward | 9.94 [9.06–11.59] ms | 9.48 [8.82–9.88] ms | -4.63% | 7/10 |
| 512px train step | 10.37 [10.06–11.03] ms | 9.95 [9.79–10.42] ms | -4.11% | 9/10 |
| 512px training throughput | 96.39 [90.67–99.37] steps/s | 100.52 [96.01–102.18] steps/s | +4.29% | 9/10 |
| 1024px render forward | 5.16 [4.76–5.67] ms | 5.05 [4.75–5.56] ms | -2.11% | 6/10 |
| 1024px render forward+backward | 14.85 [14.36–15.65] ms | 14.40 [14.35–14.94] ms | -3.01% | 6/10 |
| 1024px train step | 23.78 [23.40–24.17] ms | 23.28 [22.96–23.76] ms | -2.14% | 10/10 |
| 1024px training throughput | 42.04 [41.37–42.74] steps/s | 42.96 [42.09–43.56] steps/s | +2.18% | 10/10 |
| 512px render CUDA peak allocated (forward) | 100.15 [100.15–100.15] MiB | 100.15 [100.15–100.15] MiB | +0.00% | 0/10 |
| 512px render CUDA peak allocated (backward) | 195.18 [195.18–195.18] MiB | 196.52 [196.52–196.52] MiB | +0.69% | 0/10 |
| 512px render CUDA peak reserved | 244.00 [244.00–244.00] MiB | 244.00 [244.00–244.00] MiB | +0.00% | 0/10 |
| 512px render CPU peak working set | 869.53 [869.14–871.05] MiB | 869.53 [869.32–879.82] MiB | +0.00% | 3/10 |
| 512px training-process CPU peak working set | 964.91 [964.62–975.00] MiB | 965.31 [965.09–975.47] MiB | +0.04% | 1/10 |
| 1024px render CUDA peak allocated (forward) | 129.72 [129.72–129.72] MiB | 129.72 [129.72–129.72] MiB | +0.00% | 0/10 |
| 1024px render CUDA peak allocated (backward) | 233.73 [233.73–233.73] MiB | 232.51 [232.51–232.51] MiB | -0.53% | 10/10 |
| 1024px render CUDA peak reserved | 284.00 [284.00–284.00] MiB | 284.00 [284.00–284.00] MiB | +0.00% | 0/10 |
| 1024px render CPU peak working set | 880.58 [880.25–882.16] MiB | 880.64 [880.43–880.95] MiB | +0.01% | 4/10 |
| 1024px training-process CPU peak working set | 975.96 [975.71–976.28] MiB | 976.38 [976.18–976.54] MiB | +0.04% | 0/10 |

Sampled host CPU utilization was 20.7–45.1%, so this remains a contended-host
measurement despite exclusive team benchmark locks and initial settling. The
modest end-to-end gains should be repeated on a quiet system before generalizing.
The 512px backward allocation regression (+0.7%) is retained in the table. The
1024px training-process CPU peak increased about 0.04%. No global memory reduction
is claimed. Raw per-iteration measurements, full suite logs and sanitizer logs are
committed under [results/gaussian-gather](results/gaussian-gather).

| Final suite | CPU passed / failed / skipped | CUDA passed / failed / skipped |
|---|---:|---:|
| Gaussian | 47 / 0 / 5 | 68 / 0 / 0 |
| Adaptive | 119 / 0 / 1 | 123 / 0 / 0 |

Six new CUDA tests cover exact permutations/subsets/empty inputs/noncontiguous
strides, 112 finite-difference components, metadata rejection, current-stream
temporary lifetimes, frozen-input retained storage, and complete renderer
gradients including `means2d`. Isolated gradients match exactly. The renderer
outputs match bit for bit; gradients allow `1e-5 * max_abs_reference + 1e-6`
because the existing rasterizer accumulates atomically. Observed maximum relative
difference was approximately 1.4e-7 in the final suite. All six focused tests pass
Compute Sanitizer memcheck, racecheck and initcheck (zero errors/hazards). The
known LibTorch `indexing_backward_kernel` is excluded from instrumentation in all
three tools; this is not a claim that the excluded kernel was checked. The new
gather and scatter kernels remain instrumented.

For 257 rows with only one input trainable, the final gradient storage equals the
reference: means 2,056 bytes, projection 10,280, opacity 1,028, colors 3,084.
The rejected first implementation and its failure log are preserved.

## Kernel profile

A separate Kineto build recorded five measured training steps at 512px after three
warmups. The benchmark source and original uninstrumented executable were restored
after building the profile binary. These are one instrumented sample per arm;
the summed durations are not wall-clock latency and have no repeat variability.

| Profile metric, per step | LibTorch gather | Fused gather | Change |
|---|---:|---:|---:|
| GPU kernel/copy/memset events | 283 | 244 | -13.8% |
| GPU memset events | 38 | 25 | -34.2% |
| Summed GPU event duration | 9.935 ms | 9.276 ms | -6.6% |
| New fused gather kernel | n/a | 0.204 ms | n/a |
| New fused scatter kernel | n/a | 0.287 ms | n/a |

The event reduction supports the launch-overhead hypothesis. Remaining work
includes visibility/depth sorting and rasterization; further fusion should be
justified by an updated profile, not by extrapolating these modest total gains.

## Compatibility and limitations

The unique, in-range ID contract is internal to the renderer and guaranteed by
`nonzero(visible)` followed by `argsort`. The helper validates metadata but does
not synchronize to check index values. Duplicate or out-of-range IDs are not
supported. There is no change to the public renderer API. The CPU and camera-pose
gradient paths keep the existing implementation. The process-wide diagnostic
toggle is for controlled tests/benchmarks, not concurrent per-request settings.

This fixed-scene fitting benchmark has no reported classification accuracy, F1,
PSNR or long-run convergence result. CPU/GPU mathematical coverage comes from the
existing suites and new reference/finite-difference tests. Native runtime testing
uses LibTorch 2.15.0; other supported versions are checked by Docker build CI,
not by native GPU execution here. Training-phase VRAM is not available from this
branch's benchmark and must not be inferred from its pre-training render peaks.

## Measurement definitions

Each A/B pair uses the same executable with `--fused-gather 0` or `1`.
The existing fused projection and tile-list paths stay enabled in both arms.
The scene is `tests/data/test.ply`, 139,410 Gaussians, SHA-256
`95ba6aeb2825674814543d930c1a7e1c65bbabd57a735a669ae71ca24178e1e5`.
The benchmark uses subset seed 0 and loss-probe seed 1. Every subprocess has
three warmups and ten measured iterations. Ten rounds alternate A/B and B/A.
CUDA synchronizes around each timed region. Cold start is excluded.

CUDA allocation/reservation values are caching-allocator high-water marks
through the render/backward phase; this branch's original benchmark does not
measure the training phase separately. Host memory is process peak working set,
not an allocation delta. Throughput is the reciprocal of median latency.
The training workload is a fixed-budget Gaussian fitting step with densification
disabled and a constant target; it is not a classification-accuracy or long-run
convergence study.

Hardware/toolchain: RTX 2080 SUPER 8 GiB (sm_75), Ryzen 5 5600X 6C/12T,
16 GiB RAM, Windows 11, NVIDIA driver 610.88, MSVC 19.44, CUDA toolkit 13.4
(nvcc 13.4.92), LibTorch 2.15.0+cu134. CPU validation uses LibTorch 2.15.0+cpu.
Docker is unavailable on this host; native runtime validation and upstream
Docker CI build status are reported separately.

## Reproduction

Build this checkout against an existing LibTorch environment:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/libtorch -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --parallel 4 --target gaussian_tests adaptive_tests gaussian_render_bench
./build/gaussian_tests
./build/adaptive_tests
./build/gaussian_render_bench --input tests/data/test.ply --counts 0 \
  --sizes 512,1024 --devices cuda --warmup 3 --iterations 10 \
  --fused-gather 0 --output reference.json
./build/gaussian_render_bench --input tests/data/test.ply --counts 0 \
  --sizes 512,1024 --devices cuda --warmup 3 --iterations 10 \
  --fused-gather 1 --output fused.json
```

Repeat the last two commands for ten rounds, reversing their order on even
rounds and using distinct output filenames. Repeat with `--train` for training
steps. Use `-DCPU_ONLY=ON` and a CPU LibTorch prefix in a separate build directory
for CPU validation. Keep other compiles and GPU jobs out of timed intervals.

The same source/build commands can run in the existing image with a bind mount:

```sh
docker pull acaicia/minkowski4cpp:latest
docker image inspect acaicia/minkowski4cpp:latest --format '{{.Id}}'
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace \
  acaicia/minkowski4cpp:latest bash
```

These local Docker commands are reproduction instructions, not a claim that
this host executed them. Record the image digest and ensure the bind-mounted
checkout is rebuilt; the image's preinstalled library is not this experiment.

Run the focused checks with the final binaries (the kernel exclusion is explicit):

```sh
for tool in memcheck racecheck initcheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 99 --show-backtrace no \
    --kernel-name-exclude kns=indexing_backward_kernel \
    ./build/gaussian_tests gaussian_fused_gather_
done
```

The full native suites were executed as `gaussian_tests` and `adaptive_tests` in
each build directory. Each ended with exit code 0; the committed logs provide
the complete individual outcomes. No upstream main push or automatic merge is
part of this experiment.
