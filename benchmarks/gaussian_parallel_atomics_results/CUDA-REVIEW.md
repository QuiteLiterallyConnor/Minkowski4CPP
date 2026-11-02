# Gaussian parallel-atomics CUDA review

Reviewed runtime `b96db27aa5ef68e67d8c8bc3b3e6fa3a7548dcac` against
`dbead49`. No blocking correctness issue was found in the changed kernel.

- Each block remains 16 by 16 threads, eight complete warps. Threads outside
  partial image tiles participate in all votes, shuffles and barriers with zero
  gradient contributions. There are no thread-level early returns.
- The existing `__any_sync(0xffffffffu, ...)` skip is uniform within each warp.
  Every active lane consequently reaches each of the five shuffle-down stages
  and the new lane-zero broadcast with the same full mask. The template branch
  selects one kernel specialization and introduces no divergence at shuffles.
- Each attribute retains the exact prior shuffle-down reduction tree. The
  resulting lane-zero sum is broadcast, assigned only to its attribute lane,
  and accumulated into the same Gaussian/attribute address as before. Cross-warp
  and cross-block accumulation remains floating-point atomic and nondeterministic.
- Batch Gaussian IDs are loaded before a block barrier. All participating lanes
  read the same valid ID for a loop iteration. The new path introduces no new
  shared-memory write or dependency between atomic lanes.
- Launches still use the current CUDA stream; no new synchronization, transfer,
  allocation or forward-path change was introduced. The process-wide internal
  comparison switch must not be changed concurrently by application threads;
  normal execution leaves its default enabled.
- Existing direct tests exercise 0/1/31/255/256/257/513 contributions per tile,
  19 by 35 partial image edges, nonuniform RGB/depth/alpha loss, nondefault
  stream, zero loss and finite differences of all ten attributes. Existing
  renderer tests cover complete scenes and shared Gaussian contributions across
  tiles. The focused serial/parallel maximum difference was 4.76837e-7.

Static `cuobjdump` evidence binds to object SHA256
`9414aa7cbd1bc007ff60f301129955b76dc24ff0207d5db7155f48569d305b4b`.
Serial/parallel versions use 62/58 registers, 55/65 static shuffle instructions,
10/1 static global atomic-add instructions, zero stack/local spilling and the
same 11,268 bytes of shared memory. These are binary properties, not observed
occupancy or dynamic instruction counters.

The synthetic kernel gain is supported by balanced trials, but measured
real-scene training is approximately flat and the unchanged forward controls
vary materially. This review does not infer a training, accuracy or memory
improvement from static instruction counts or synthetic timing.

Native full CUDA and sanitizer completion, and actual container CPU execution,
must be read from their final receipts before publication. This file records
source review and the already completed focused/benchmark evidence only.

Follow-up validation at 20:41: native CPU/CUDA CTest passed 4/4 each, CUDA
Gaussian tests 71/0/0. Memcheck/initcheck each reported zero errors; racecheck
reported zero hazards/errors/warnings, with all three new tests and no
exclusions. The independent validation audit passed. Container CPU execution
remains pending with the testing specialist.
