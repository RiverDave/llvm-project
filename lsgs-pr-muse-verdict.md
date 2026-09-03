[0m
> build · muse-spark-1.3-contributor
[0m
Honest PR framing — checking your spec then answering Q1-Q5 without inflation.
[0m→ [0mRead sol-lsgs-pr-consult.md
# LSGS PR consult — honest answers Q1-Q5

## Q1. Is `reqntid`+fold sound/useful without speedup?

**Yes, as a canonicalization, No as a perf feature.**

Technical truth:

* `.reqntid bx,by,bz` is an **exactness contract**: kernel is UB / launch-fails unless launched with exactly that shape. `.maxntid` is only an upper bound.
* `ptxas` uses both for the same register-budget calculation. `reqntid` does **not** budget tighter than `maxntid == same triple`. It lets `ptxas` replace `ntid` SREG reads with immediates and size shared frames exactly, but without `.minnctapersm` it does **not** force spilling / occupancy target.
* So `reqntid` alone is **not** a register/occupancy lever. Its perf value in v1 comes **only** from (a) CIR->LLVM constant exposure: `blockDim.{x,y,z}` -> `cir.const` -> LLVM `i32 256`, enabling SCEV/range/strength-reduction downstream.

**HONEST PR claim:**

> Exact-geometry specialization for `static __global__` with a single visible constant launch shape: fold dynamic `blockDim` queries to constants and emit exact `nvvm.reqntid`. Compile-time simplification / contract enforcement. No runtime speedup claimed.

That survives Kaylor/Dinos review. Claiming `reqntid` improves occupancy would not.

Correctness argument you must make: `static` + all visible launch sites same triple + no address-escape => safe in-place. Never folding `tid/ctaid/gridDim` is correct. Running before LBP + LBP skipping `reqntid` carriers is **required** — PTX forbids `maxntid+reqntid` together. Good catch, keep it.

## Q2. Where could a measurable win come from?

* `(a) LLVM folds`: only plausible path in v1, and small. `i = bIdx*blockDim.x+tid` becomes `i = bIdx*256+tid` — still dynamic because `tid` remains. Helps range analysis, `mul by const` strength reduction. Only large if kernel does `div/mod by blockDim` (becomes shift/mask) or loops on `blockDim` (unroll/SCEV). PolyBench GEMM/2MM/3MM etc. do not — they do affine index math with dynamic `tid/bid`. Expect noise, 0-2%.
* `(b) reqntid tighter budgeting than maxntid`: **No.** `ptxas` already assumes worst-case `maxntid`. Exactness adds nothing to allocation without forcing lever.
* `(c) Neither without `minctasm``: **Correct.** Systematic speedup needs `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` -> `.minnctapersm`, which forces spill/occupancy tradeoff. You explicitly excluded it, so do not expect runtime wins.

**Verdict:** Frame v1 as **codegen/simplification**, not perf. Any wall-clock delta on GEMM/floydwarshall2/mpc on A10 will be in noise — do not benchmark-chase it.

## Q3. Minimum honest `[PROTOTYPE]` evidence set

1. 3x FileCheck `lit` tests: direct-`ntid` fold+`reqntid`, runtime-shape no-op, multi-shape no-op. All PASS.
2. LaunchBounds regression still PASSES (proves `maxntid:reqntid` mutual exclusion handling).
3. E2E CIR dumps: `lsgs_e2e_static.cu` -> `cir.nvvm.reqntid="256,1,1"`, `__fetch_builtin_*blockDim*` call count `N->0` in kernel body, `cir.const 256` in address math, survives `cir-split`. GEMM -> `"32,8,1"`.
4. One IR-level simplification proof: before/after LLVM-IR or PTX snippet showing `llvm.nvvm.read.ptx.sreg.ntid.x` / `mov.u32 %r, %ntid.x` eliminated. `cuobjdump` insn count or `ptxas -v` registers/smem if you have it — as **codegen fact**, not speedup.
5. NFC proof: unrelated kernels unchanged; full `cir` suite delta = 0 vs baseline. State 11 `abi-lowering` failures pre-exist on `bb385fee` untouched subsystem — show baseline run hash.
6. Compile-time: state `cir-offload-merge` time on GEMM before/after (expect neutral).

**Explicitly DO NOT claim:** wall-clock speedup, occupancy %, register reduction, "enables better occupancy", generic benefit to dynamic/multi-shape kernels.

## Q4. Land as-is or wait for `minctasm`?

**Land as narrow stepping-stone `[PROTOTYPE]`, not as perf feature.**

Standalone value: infrastructure + exact contract + unlocks future clone/dispatch + `minctasm` forcing. Not meaningful alone for runtime.

Conditions to be landable: on-by-default only if strictly NFC when pattern doesn't match; documented limits (`static` only, single constant triple, no clone/dispatch, no `minctasm`, function-pointer / cross-TU launch = no-fold conservative path); reviewer will probe escape via taken address / driver launch — be conservative.

Do not add `minctasm` to v1 to chase a number. That is a separate policy decision (spill vs occupancy) and needs heuristics + benchmarks.

## Q5. PR body structure (no slop)

```
[PROTOTYPE][CIR] LSGS: exact launch-site geometry spec

Summary: what it does, 2 lines.
What it does NOT do: no tid/grid fold, no clone, no minctasm, no runtime claim.
Correctness: static visibility + single-shape check + LBP skip + PTX rule cite.
Codegen evidence: FileCheck counts, e2e reqntid strings, fetch-call N->0,
  PTX/IR snippet, ptxas -v if available.
Testing: new tests 3/3 pass, LB regression pass, full-suite baseline-identical
  (11 pre-existing abi failures on bb385fee), A10/sm_86 fork clang24@f1377581.
Perf: explicitly none claimed; compile-time neutral [numbers].
Limits/Future: dynamic, multi-shape clone/dispatch, minctasm.
Model-assist note: [tool + human verification statement].
```

Numbers appropriate: test counts, `cir.const` values, call-count `->0`, `.reqntid` strings, build config + commit hashes, compile wall-time. **Not appropriate:** GEMM GFLOPS, `% faster`, occupancy graphs from <5 runs.

Decision line for PR: `Prototype codegen simplification; perf work (minctasm + versioning) follows separately.`
