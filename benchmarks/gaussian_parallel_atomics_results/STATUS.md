# Round 10 Gaussian backward atomics

Root approved follow-up after rejecting per-warp bounds (../gaussian/STATUS.md).
Worktree C:/Users/User/Work/mink-wt/gauss-cull, builds gauss-cull-cuda/cpu.
Branch perf/gaussian-backward-parallel-atomics, runtime candidate b96db27,
baseline harness dbead49, base feature branch PR53 229f04d.

Hypothesis: baseline warp aggregation makes lane0 issue10 scalar atomic instructions.
New candidate preserves the exact existing shuffle-down reduction order, broadcasts
each sum from lane0, and lets lanes0..9 issue the adjacent attribute atomics together.
This adds10 broadcast shuffles, replacing10 serial atomic instructions with one warp
instruction. No changes to forward, projection, tile construction or other pipelines.
Old path retained via cuda_raster_backward_parallel_atomics()/--parallel-atomics.

Candidate build PID29076 started20:19, logs candidate-build.log/err.
after-build.ps1 is detached: waits build, runs three focused tests with GPU lock,
then ab.ps1 requests bench lock for10 balanced pairs of three synthetic direct
raster workloads and real PLY512/1024 render/training. Durable receipt at ab/receipt.json.
Read logs/processes before relaunching. Root coordinates actual Docker environment.

Tests cover per-tile counts0/1/31/255/256/257/513, edge tiles, sparse/nonuniform loss,
nondefault streams, zero gradients and10 finite differences of all raster attributes.
Tests preserve existing atomic nondeterminism tolerances. GPU implementation needs
full CPU+CUDA CTest and memcheck/racecheck/initcheck after measured performance gate.
Scripts prepared but not launched: full-validation.ps1, sanitize.ps1.
summarize.py requires all80 A/B process runs to pass before reporting.

No root integration edits; no service changes; all GPU work uses shared lab locks.
No obsolete generated footers or automatic upstream merges. Commit/PR publishing
only after validation and reporting; PR will stack on existing gather branch.

2026-11-02 20:35 recovery audit:
- Runtime b96db27 remains clean and unchanged. Focused tests passed3/3, max serial/parallel difference4.76837e-7; ten finite differences passed.
- Initial A/B stopped at60/80successful processes due PowerShell receipt file-sharing violation, not benchmark failure. Original failure receipt and all completed-output hashes preserved as ab/receipt-initial-failed.json and ab/pre-resume-hashes.json.
- ab.ps1 now supports identity-checked -Resume, skips completed trials, and writes receipts using atomic replacement+bounded sharing retries. Original script retained as ab.before-resume.ps1. Resume acquired exclusive bench20:32:28, completed20:33:50; all80/80pass. Original60trials were not rerun.
- summarize.py and independent audit_results.py both pass. Independent audit checks CLI arm,80unique expected identities, synthetic dimensions/scans, real-scene visible/pair counts/alpha, sample cardinality/medians. Raw report ab/summary.md, audit independent-audit.json.
- Synthetic GPU medians: row-groups2.02486->1.43217ms(-29.27%,10/10lower), sparse2.36523->2.10533(-10.99%,9/10), dense9.89513->9.58031(-3.18%,8/10). Real backward51214.48255->14.17745(-2.11%,5/10),102426.89975->26.69355(-0.77%,7/10). Real train51227.39870->27.37945(-0.07%);102466.58185->67.08795(+0.76%). No demonstrated real-training gain; unchanged forward controls+3.28%/+2.70%. Host load and initial severe outliers disclosed.
- cuobjdump measured object: serial62registers/55SHFL/10global RED.ADD, parallel58registers/65SHFL/1global RED.ADD; both0local/stack and11268shared bytes. Static profile receipts and SASS saved. This is static inspection, not measured occupancy/counters.
- Root retains narrow isolated-kernel improvement pending full validation; no claim of real-scene training gain or memory reduction. No PR pushed/opened yet.
- finish-validation.ps1 detached PID26884 runs full-validation.ps1 (CPU then CUDA builds/complete CTests) then sanitize.ps1 under GPU lock (memcheck/racecheck/initcheck). Currently queued behind ML seed3 benchmark. Durable logs finish-validation.log/.err, validation/receipt.json, sanitize/receipt.json.
- Testing teammate combined Docker CPU run41511099983 already active without Gaussian. Preserve this run. Stage b96db27 on next Docker candidate after first compile outcome; do not claim Docker validation complete.

2026-11-02 20:41 native validation complete after restart:
- Full CPU and CUDA builds passed. CTest 4/4 in each configuration. CPU adaptive119/0/1 and Gaussian47/0/8; CUDA adaptive123/0/0 and Gaussian71/0/0; framework9cases each.
- memcheck and initcheck report zero errors; racecheck reports zero hazards/errors/warnings. All three new tests pass under each tool, no exclusions. Sanitizer gradient difference up to9.53674e-7 remains within tolerance.
- audit_validation.py independently parsed final XML/counts/tool summaries and verified live test/benchmark executable SHA256 against original receipts. validation-audit.json passed.
- After restart, root discovered run41511099983 ran a different short build/deploy workflow; it is not Docker runtime evidence. docker_testing_recovery now owns the correct next run including Gaussian source. No container success claimed.
- REPORT.md, CUDA-REVIEW.md, PR-BODY.md and PR-COMMENT.md describe narrow synthetic gain and flat real training. Packaging all original raw evidence with portable verify_evidence.py; Docker remains the publication gate.
