# LSGS Phase-3 speedup plan — how to get real runtime wins

CONTEXT. The v1 launch-site geometry specialization (LSGS) is built and validated
(commit f1377581, branch gsoc/lsgs-geometry-spec): it folds `blockDim.{x,y,z}`
(static __global__, single constant block triple) to `cir.const` and emits
`nvvm.reqntid`. Codegen-proven, perf-neutral on affine-index kernels.

The user's PRIORITY is real speedups (runtime wins), not just codegen
simplification. I need a concrete phase-3 plan that produces MEASURABLE runtime
gains on real kernels. Both Sol and Muse earlier confirmed the ONLY forcing lever
is `.minnctapersm` (minBlocksPerMultiprocessor), which v1 deliberately excluded.

ESTABLISHED FACTS (from earlier experiments, A10/sm_86, 65536 regs/SM, 1536 threads/SM):
- Single-arg `.maxntid` is proven INERT (ceiling, never forces; PR25 benchmark parity).
- `.reqntid` alone does NOT force registers/occupancy (it's an exactness contract;
  ptxas budgets vs maxntid worst-case identically; only adds ntid-range info).
- `.minnctapersm M` is the forcing lever, VERIFIED on ptxas: caps regs to
  `65536/(M*block)`; forces via spills. On an 80-reg kernel: M=8 -> 64 regs,
  M=12 -> 40 regs, with ~1KB spill store/load. Gives a real speedup ONLY when
  kernel is (register-capped below thread ceiling + memory-latency-bound + LOW
  per-thread MLP + affordable cut).
- A controlled existence proof (2026-09-03) showed forcing `(256,4)` on a
  latency-bound low-MLP kernel gives +6.35% (CI [6.04,6.50]), scaling with loop
  iterations (1.7% @32 -> 8.8% @512). Real suites (PolyBench/HECbench) were all
  parity because their kernels are register-lean / not in the beneficial regime.
- LSGS now knows the exact block triple => it can COMPUTE the natural register
  ceiling for full occupancy (e.g. block 256 -> 65536/(6*256)=42 regs) and M
  targets, and pair that with `reqntid`.

THE GOAL: extend LSGS (or a sibling pass) to emit `.minnctapersm M` for the
RIGHT kernels, gated by a sound compile-time proxy, to produce real speedups on
real workloads. Also plan how to lead the PR/paper with this speedup direction.

MY QUESTIONS (answer precisely, honest, concrete):
Q1. Given LSGS emits `.reqntid` already, and PTX allows `.reqntid` + `.minnctapersm`
    together (needs verification), how should we extend it to also emit
    `.minnctapersm M`? What is the EXACT safe emission rule for M given the known
    block triple and the target arch's regs/SM, threads/SM? Should M be derived
    from ptxas-feedback (compile the clone twice, compare regs/spills) or a
    static heuristic?

Q2. What is a SOUND COMPILE-TIME proxy for "this kernel is in the beneficial
    regime" (register-capped + latency-bound + low per-thread MLP) that LSGS can
    see in CIR/LLVM IR WITHOUT a GPU profile? Concretely: number of independent
    global loads in the hot loop, register-pressure estimate, block size, no hot
    __syncthreads, arithmetic intensity. Rank these proxies by how predictive
    they are. Is ptxas-feedback trial-compile (which needs no GPU) the only
    sound route, or can a static heuristic gate it?

Q3. On WHICH REAL kernels should we target first, and what realistic speedup
    magnitude? Candidate classes: (a) PolyBench/HECbench static kernels
    (register-lean, likely ~0), (b) a grid-stride/reduction/scan kernel, (c) a
    deliberately latency-bound sparse/gather kernel like the +6.35% one. Since
    real suites are register-lean, should we (i) find/fabricate a real kernel in
    the beneficial regime, (ii) accept that the speedup is demonstrated on a
    synthetic but keep the pass general, or (iii) something else?

Q4. What is the MINIMAL phase-3 scope that yields a measured, honest, reproducible
    runtime speedup on a REAL workload through the full cir-offload-merge
    pipeline? Give the exact plan: which kernel, what M emission, what measurement
    protocol (paired samples, CI), what SASS/regs evidence.

Q5. For the PR + paper: how should we LEAD with the speedup direction while the
    v1 codegen pass is the shipped artifact? Should the PR be rewritten to
    lead with "LSGS is infrastructure for a measured +6.35% speedup" and include
    the synthetic existence proof as evidence of the mechanism's potential, then
    state v1 ships the exact-geometry propagation (the enabling part)?

Q6. RISK check: if we emit `.minnctapersm M` unconditionally or with a weak
    heuristic, what regressions can occur (spill-heavy, compute-bound) and how do
    we guarantee NFC-on-by-default? Is there a safe default that never regresses?

Give a prioritized, decision-usable phase-3 plan. The user values real speedups
most. Be honest about what magnitude is realistic and what is hype.