[0m
> build · muse-spark-1.3-contributor
[0m
Speedup plan coming — reading your LSGS setup to nail the M rule and proxies.
[0m→ [0mRead .
[0m→ [0mRead lsgs-phase3-muse-verdict.md
[0m→ [0mRead lsgs-phase3-sol-verdict.md
[0m✱ [0mGlob "**/*LSGS*"[90m in . · 0 matches[0m
[0m✱ [0mGlob "**/*LaunchSite*Geometry*"[90m in . · 0 matches[0m
[0m✱ [0mGrep "reqntid|minncta|LaunchSite"[90m in . · 100 matches[0m
[0m→ [0mRead clang/lib/CIR/Dialect/Transforms/OffloadOpt/LaunchGeometrySpecialization.cpp
[0m→ [0mRead clang/lib/CIR/Dialect/Transforms/OffloadOpt/LaunchBoundsPropagation.cpp
I've mapped your exact-geometry pass — now checking how the backend carries the occupancy lever.
[0m✱ [0mGrep "minncta|minBlocksPerMultiprocessor|reqntid"[90m in llvm/lib/Target/NVPTX · 8 matches[0m
[0m→ [0mRead llvm/lib/Target/NVPTX/NVPTXAsmPrinter.cpp[90m [limit=80, offset=940][0m
[0m✱ [0mGrep "MinCTA|MaxNTID|ReqNTID"[90m in llvm/include/llvm/Support · 3 matches[0m
[0m→ [0mRead llvm/include/llvm/Support/NVVMAttributes.h
# LSGS Phase-3 speedup plan: `.minnctapersm` as opt-in forcing lever

V1 facts reused: LSGS proves exact `(Bx,By,Bz)` in `clang/lib/CIR/Dialect/Transforms/OffloadOpt/LaunchGeometrySpecialization.cpp:183-223`, folds `ntid` to `cir.const`, stamps `cir.nvvm.reqntid`. Backend co-emission is legal: `llvm/lib/Target/NVPTX/NVPTXAsmPrinter.cpp:969-985` emits `.reqntid` + `.minnctapersm` unconditionally with no mutual exclusion (only `maxntid+reqntid` is forbidden, which LBP already handles). Attr exists: `llvm/include/llvm/Support/NVVMAttributes.h:25` = `nvvm.minctasm`.

## Q1. Exact M emission rule

Yes, PTX allows `.reqntid Bx,By,Bz` + `.minnctapersm M` together. That is the `__launch_bounds__(B,M)` lowering shape, and NVPTX prints both.

Safe emission reuses all v1 soundness conditions (static `__global__`, all visible launch sites same constant triple, no address-escape/dynamic-parallelism, no pre-existing incompatible `reqntid`/`maxntid`) **plus** an arch validity bound:

```
B = Bx*By*Bz
Arch(R,T,Bmax): regs/SM, threads/SM, blocks/SM
  sm_86/A10: R=65536, T=1536, Bmax=16
  sm_90+:    R=65536, T=2048, Bmax=32  // look up from target, don't hardcode
Valid M: 1 <= M <= min(Bmax, floor(T/B))
Regs cap(M) = floor(R / (M*B))
```

Examples: B=256 on sm_86: `Mfull=floor(1536/256)=6`, `cap(6)=42`, `cap(4)=64`. Your `(256,4)` win was `3 blocks@80 regs -> 4 blocks@64 regs`.

Do **not** derive M from a static formula alone. Do this:

1. LSGS computes `B`, `Mfull`, candidate set `{Mon+1, Mon+2, Mfull}` where `Mon` = natural resident blocks from baseline ptxas regs `R0` (`Mon=min(floor(T/B),Bmax,floor(R/(R0*B)))`).
2. Trial-compile the clone twice **without GPU**: baseline (`reqntid` only) -> `(R0, spill0)` from `ptxas -v`, forced (`reqntid+M`) -> `(R1, spill1)`.
3. Emit `M` only if `R0 > cap(M)` (it actually forces), occupancy strictly increases (`M > Mon`), and spill delta is affordable (see Q2/Q6).

Static heuristic picks candidates; **ptxas feedback decides**. Static-only M selection is unsound because only ptxas knows true `R0`/spill cost.

Implementation: sibling opt-in pass/flag after LSGS, e.g. `--cir-lsgs-minctasm trial` that stamps `cir.nvvm.minctasm="M"` alongside `reqntid`. Never in default v1 path.

## Q2. Compile-time proxies, ranked

No CIR/LLVM-IR proxy sees latency-boundness reliably. Rank:

1. **ptxas-feedback regs/spill delta (only sound gate, needs no GPU).** Baseline vs forced `ptxas -v`: `R0->R1`, `spill stores/loads bytes`. Predictive because it is the actual allocator decision. Everything else is a pre-filter.
2. **Register-capped gap estimate.** From `R0`: `Mon < Mfull` (kernel is register-capped below thread ceiling). If `Mon == Mfull`, forcing cannot add occupancy — reject. Computable after one baseline ptxas query.
3. **Low per-thread MLP: count independent global loads in hot loop.** Walk natural loops in CIR/LLVM: number of `cir.load`/`llvm.load.global` with no address dependency chain, trip-count constant or profile-free SCEV large. 1-2 independent loads + loop-carried dependency = good; 4+ independent or unrolled vector loads = high MLP, forcing won't help.
4. **Latency-bound vs compute-bound: arithmetic intensity.** Ratio `fp/int ops : global mem ops` in hot loop. High ALU per byte, tensor/shared-mem use, `__syncthreads` in hot path = reject. Pure gather/add/store loop = accept.
5. **Block size + sync.** Require `B>=128` (room for `M>B` math to matter) and no hot `bar.sync`. Small `B=32` or block-sync kernels rarely benefit.

Rule: static proxies (3-5) select *candidates to trial-compile*; (1)+(2) make the *emit decision*. Never emit on static score alone.

## Q3. Which kernels first, what magnitude

* (a) PolyBench/HECbench static kernels: expect ~0. Already proven parity — they are register-lean (`R0 <= cap(Mfull)`), so there is nothing to force.
* (b) Grid-stride/reduction/scan: usually shared-mem or barrier bound, not register-cap+latency. Skip.
* (c) Deliberately latency-bound sparse/gather with low MLP: the only regime verified (`+6.35%, CI [6.04,6.50]`, scaling 1.7%@32 -> 8.8%@512 iters). This is the target.

Do (ii): keep the pass general, demonstrate on the synthetic existence-proof as mechanism evidence, **and** hunt one real kernel in the same regime (SpMV/gather, graph walk, hash-join probe, ray-trace indirect load). Do not relabel synthetic as real. Honest magnitude for this lever alone: **2-8%, scaling with loop trip count**, only when register-capped + latency-bound + low-MLP + affordable cut all hold. Anything claiming 20%+ from `minctasm` alone is hype.

## Q4. Minimal measured scope

1. Kernel: the existing latency-bound low-MLP loop, fixed `B=256`, `M=4` (i.e. `.reqntid 256,1,1` + `.minnctapersm 4`).
2. Pipeline: full `cir-offload-merge` with LSGS + opt-in minctasm flag stamping both attrs; check final PTX contains both lines.
3. Evidence: `ptxas -v` before/after (e.g. `80 regs, 0 spill -> 64 regs, ~1KB spill`), theoretical occupancy `3->4 blocks`, SASS `STL/LDL` spill traffic small vs global-load latency hidden.
4. Measurement: same A10, locked clocks, warmup, paired interleaved samples (N>=30 each, alternate baseline/forced per iteration), report median delta + 95% CI (bootstrap or t), plus trip-count sweep to show scaling. No Nsight-required claim; occupancy math + regs + CI suffices.

Ship nothing else in phase-3 scope. One kernel, one `(B,M)`, one CI.

## Q5. PR + paper framing

Do not rewrite v1 PR to claim the speedup as shipped. V1 stays: "exact-geometry propagation (enabling infra), perf-neutral on affine kernels."

Add a short forward section:

> LSGS is infrastructure for occupancy forcing. As existence proof, an opt-in follow-up emitting `.minnctapersm 4` on a latency-bound low-MLP kernel (`B=256`) moves `80->64 regs, 3->4 resident blocks` for a measured `+6.35% [6.04,6.50]` on A10/sm_86. General heuristic gating is future work; v1 ships only the exact-geometry part.

Paper leads with speedup *direction* (the problem: register-cap stalls; the lever: `minctasm` now computable from exact geometry), presents synthetic as mechanism validation, then states v1 contribution honestly. Reviewers accept "enabler + measured instance"; they reject "v1 speeds up everything."

## Q6. Regression safety, default

Unconditional or weakly-gated `M` regresses: spill-heavy kernels (extra local-memory traffic > latency saved), compute-bound/high-MLP kernels (fewer regs hurts ILP, no latency to hide), shared-mem/barrier-bound kernels (occupancy capped elsewhere, spills pure loss — 10-30% slowdowns observed in this class generally). Even `M=1` is unsafe: `cap(1)=R/B` still forces for large `B` (B=1024 -> cap 64).

There is **no safe unconditional M**. Safe default: **never emit `minctasm` by default (NFC-on-by-default)**. Gate behind explicit opt-in flag + trial-compile filter:

* reject if `Mon == Mfull` (no occupancy gain possible),
* reject if spill delta > ~1-2KB or `R0-R1 > ~50%` cut,
* reject if static pre-filter says high MLP / compute-bound / hot sync / `M*B > T`.

Priority order: (1) land opt-in `minctasm` plumbing + arch table, (2) land ptxas-feedback trial gate, (3) publish one measured `(256,4)` instance, (4) only then consider heuristic auto-on.
