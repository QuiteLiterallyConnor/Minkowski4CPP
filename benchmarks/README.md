# Sphere volume benchmark

The total voxel count (1 to 10,000,000, default 1000) is given on the command line:

```powershell
.\build\sphere_volume_standalone\sphere_volume_benchmark.exe 1000
```

Or give a target accuracy (percent, above 0 and below 100) to find how many voxels each
method needs to reach it (searched up to 10,000,000 voxels):

```powershell
.\build\sphere_volume_standalone\sphere_volume_benchmark.exe --accuracy 99
```

The reported count is the first size that reaches the target; uniform accuracy is not
monotonic in voxel count, so a larger grid can occasionally dip below it. Full timing and
stats are then printed at those counts.

To find the best latency-to-accuracy trade-off, sweep both methods across many sizes:

```powershell
.\build\sphere_volume_standalone\sphere_volume_benchmark.exe --optimal
```

It times build plus volume evaluation end to end at each size and prints, per method and
overall, the fastest point, the point with the most voxels per ms, the point with the most accuracy per ms (accuracy measured in
digits, `-log10(relative error)`), and the Pareto frontier (points no faster point matches
in accuracy). Sub-0.1 ms timings are noisy; run it a few times.

It compares a center-sampled voxel estimate of a radius-100 m sphere's volume for:

- a uniform grid with the nearest cubic voxel count (1000 is 10³; 100 becomes 5³ = 125),
- an adaptive octree given the same voxel count as a budget, refining only cells that
  intersect the sphere boundary, coarsest first.

The voxel edge follows from the count (`200 m / cells per axis`) and is printed. Large
counts take longer (8 million voxels takes about a minute); runs above 200,000 voxels time
a single repetition instead of the median of seven.

An octree split replaces one leaf with eight; therefore, its leaf count advances in
increments of seven and finishes slightly below its budget. The executable prints both
target and actual counts. The estimator includes the
full volume of each voxel whose center lies inside a radius-100 m sphere; this simple,
consistent rule makes the error and signed bias visible. The analytic sphere formula is
the unvoxelized 100%-accuracy reference; its evaluation time is reported separately from
voxel construction and measurement.

The output also includes boundary-cell count, adaptive level distribution, construction
time, volume-evaluation time, and end-to-end median time. Timings are the median of
seven runs after one warm-up (single run for large grids); compare results on the same machine and build configuration.
These are geometric-approximation and CPU timing measurements, not claims about all
datasets or Minkowski operation performance.

This is a standalone CMake project. It links against an installed MinkowskiEngine package
and does not build the repository's library sources. Ensure CMake can find both the
installed MinkowskiEngine package and its LibTorch dependency. If the install does not set
`MinkowskiEngine_DIR`, pass the path to its `MinkowskiEngineConfig.cmake` directory:

```powershell
cmake -G Ninja -S benchmarks -B build-sphere-benchmark `
  -DMinkowskiEngine_DIR="C:\Program Files (x86)\MinkowskiEngine\lib\cmake\MinkowskiEngine" `
  -DCMAKE_PREFIX_PATH="C:\path\to\libtorch" `
  -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe" `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
cmake --build build-sphere-benchmark --config Release
.\build-sphere-benchmark\sphere_volume_benchmark.exe
```

Use the CUDA toolkit version that matches the installed MinkowskiEngine build. If the
installed package and LibTorch are already discoverable through CMake's package search
paths, their explicit directory options can be omitted.

# A/B comparison of two builds (`ab_compare.py`)

`benchmarks/ab_compare.py` (Python 3 standard library only) runs the same benchmark executable
from two build directories alternately (A B A B ...), one fresh process per run, so that slow
drift in machine load affects both sides equally. Example, original vs optimized adaptive
convolution on CUDA:

```powershell
python benchmarks/ab_compare.py --a base=build-main --b new=build-branch --out ab --name conv20k `
    --rounds 5 --warmup-rounds 1 -- adaptive_kmap_bench --conv cuda --n 20000 --depth 4 `
    --channels 8 --cout 16 --modes center,volume --warmup 3 --reps 10
```

Per process it records the whole-process wall time (cold start: DLL load, CUDA context, first
calls), peak working set and CPU time (Windows), device-wide GPU utilization and memory sampled
with `nvidia-smi` every 200 ms (these include other processes on the GPU), every `key   value`
line the benchmark prints and every number of the JSON file it writes when an argument contains
`{json}`. `--extract key=REGEX` adds a number from free-form output (for example the epoch time
of `train`). In-process warm-up and timed repeats stay the benchmark's own, so its warm
(steady-state) timings and the process-level cold timing are reported separately.

The run is resumable: `ab/NAME.json` keeps every run, and calling the command again adds rounds
until `--rounds` timed rounds exist. `--budget-s` stops before a pair that would not fit (exit
code 3) so long comparisons can be split across short exclusive-benchmark windows; `--load-gate`
waits (bounded) for total CPU load to drop and records it. Output: `NAME.md` (median [min-max]
per side and % change), raw stdout per run, `nvidia-smi` samples, and GPU process snapshots per
block. The self-test `tests/tools/test_ab_compare.py` runs in CTest (`ab_compare_selftest`) when
CMake finds a Python 3 interpreter.
