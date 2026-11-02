# Gaussian backward parallel atomics: candidate report

Candidate source: b96db27aa5ef68e67d8c8bc3b3e6fa3a7548dcac. Baseline algorithm is selected with --parallel-atomics 0 in the same candidate executable; --parallel-atomics 1 enables the improvement. Parent feature dependency is Gaussian gather PR53 (229f04d), followed by direct-raster benchmark commit dbead49. No upstream main push or merge.

## Scope and decision

Retain a narrow isolated CUDA-kernel improvement. Native correctness and sanitizer gates passed; actual Docker source execution remains pending. Three synthetic direct backward workloads improve; representative real-scene training remains approximately flat. There is no demonstrated ML accuracy, convergence or real-training throughput benefit and no measured memory reduction. CPU mathematics and the forward path are unchanged.

Ten warp-reduced gradient attributes are broadcast from lane zero and assigned to lanes zero through nine for one adjacent atomic instruction, preserving the existing reduction tree. Serial-reference execution remains selectable for reproducibility. The existing atomic accumulation nondeterminism remains; tests allow 2e-5 relative plus2e-6 absolute for serial/parallel gradients, finite differences allow1%relative plus0.002absolute.

## Evidence and measurements

Focused correctness: 3 passed, 0 failed, 68 skipped by the explicit filter. Coverage includes tile lengths 0/1/31/255/256/257/513, partial image edges 19 by 35, nondefault stream, zero upstream gradients and ten finite differences for all projected attributes.

Full native builds and CTest passed on CPU and CUDA, four registered suites each. CPU adaptive: 119 passed, 0 failed, 1 skipped; Gaussian: 47 passed, 0 failed, 8 skipped. CUDA adaptive: 123 passed, 0 failed, 0 skipped; Gaussian: 71 passed, 0 failed, 0 skipped. The adaptive-versus-uniform smoke and nine test-framework cases passed in both builds. CPU skips cover unavailable CUDA paths.

Compute Sanitizer memcheck and initcheck each reported zero errors; racecheck reported zero hazards, errors or warnings. All three added tests ran under each tool, with no exclusions. The initial focused gradient difference was at most 4.76837e-7; sanitizer runs observed at most 9.53674e-7, within documented atomic tolerances. `validation-audit.json` independently verifies final CTest XML, test counts, all tool summaries, source identity and live test/benchmark SHA256 bindings.

Balanced A/B:10paired rounds, reversed order each round,80successful process invocations. Synthetic workloads use1024tilesx256Gaussians,5warmups+30CUDAevent measurements. Real scene tests use139410Gaussians,512/1024images,3warmups+10synchronized wall measurements. Source/input/executable identities are in ab/receipt.json; full raw distributions and summary are in ab/summary.md. independent-audit.json checks all80trial identities and mathematical scene invariants.

| Workload | Metric | Serial ms | Parallel ms | Change | Lower paired processes |
|---|---|---:|---:|---:|---:|
| Row groups synthetic | Backward kernel GPU event |2.02486|1.43217|-29.27%|10/10|
| Sparse synthetic | Backward kernel GPU event |2.36523|2.10533|-10.99%|9/10|
| Dense synthetic | Backward kernel GPU event |9.89513|9.58031|-3.18%|8/10|
| Real512 | Backward synchronized wall |14.48255|14.17745|-2.11%|5/10|
| Real1024 | Backward synchronized wall |26.89975|26.69355|-0.77%|7/10|
| Real512 | Training step |27.39870|27.37945|-0.07%|6/10|
| Real1024 | Training step |66.58185|67.08795|+0.76%|6/10|

Render-phase allocated/reserved VRAM exactly matches between arms:512forward100.1484MiB,backward196.5239MiB,reserved244MiB;1024forward129.7222MiB,backward232.8652MiB,reserved308MiB. Cumulative process peak working set approximately965/976MiB; differences below0.01%are not evidence of memory improvement. Training-phase VRAM is unavailable. Synthetic kernel event time is isolated; real-scene kernel-only timing was not collected.

Static cuobjdump of measured object9414aa7cbd1bc007ff60f301129955b76dc24ff0207d5db7155f48569d305b4b: serial62registers/55shuffle instructions/10global atomic add instructions, parallel58registers/65shuffles/1global atomic add instruction. Both no stack/local spilling and11268shared bytes. This does not measure dynamic occupancy or prove the timing mechanism.

## Limitations and interrupted experiment history

The shared workstation had substantial background CPU load(38%-94%observed), around7GiBGPUuse and variable application/OSactivity. Unchanged forward-only controls vary+3.28%/+2.70%between aggregate medians, so small real-scene differences are not established improvements. Initial round includes large outliers(e.g.1024backward83.9963ms)and is retained. See ab/hostload.log and every raw record.

Initial run stopped after60successful processes due a Windows file-sharing conflict writing receipt.json. ab/receipt-initial-failed.json preserves failure. Resume checked identical source/input/binary hashes, preserved completed outputs, and ran missing20processes only after ModelNetseed2 released the bench lock. A/Brounds8-10therefore span a scheduling gap. The writer now atomically replaces the receipt with bounded retry; original script is preserved. No benchmark error was discarded or reclassified.

The preceding warp-bounds candidate8438c04 was rejected: denseGPU+4.95%,row-groupGPU+2.40%,real512backward+4.11%(0/10better). Its source and raw results remain in siblinggaussian/. This candidate is independent of those rejected bounds.

## Reproduction

Native supplementary environment:Windows11,Ryzen55600X6cores/12threads,16GiBsystemRAM,RTX2080SUPER8GiBsm75,driver610.88,MSVC19.44,CUDA13.4.3,LibTorch2.15.0cu134. Existing Docker environment execution remains pending; do not mark container validation complete based on native results.

Build:pwsh -File C:/Users/User/Work/mink-lab/tools/mink-build.ps1 -Src C:/Users/User/Work/mink-wt/gauss-cull -Build C:/Users/User/Work/mink-build/gauss-cull-cuda -Who gaussian-atomics-cuda

Locked A/B:pwsh -File C:/Users/User/Work/mink-lab/tools/with-lock.ps1 -Kind bench -Who gaussian-atomics-ab -Command "pwsh -File C:/Users/User/Work/mink-lab/results/round10/gaussian-atomics/ab.ps1". Fresh runs need a fresh output directory; existing receipt requires -Resume to prevent overwriting evidence.

Reports:python summarize.py;python audit_results.py. Full validation:pwsh -File finish-validation.ps1. All native resource use is serialized by existing lab locks.

## Publication gates

No feature PR is complete yet. Native full CPU/CUDA CTest and sanitizer gates are complete. Actual Docker CPU compile/test for this source and PR publication remain pending. Root coordinates publication after these gates pass. Docker GPU execution is unavailable on this host and must be disclosed separately. Future work: profile real-scene kernel share with counters on an isolated GPU; verify the synthetic gain on additional architectures before generalizing.
