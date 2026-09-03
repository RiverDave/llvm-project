# LSGS PR strategy + honest measurement framing consult

CONTEXT: I built a ClangIR launch-site geometry specialization pass (LSGS) — 
commit f1377581 on branch gsoc/lsgs-geometry-spec (off PR25 bb385fee). In-place
exact-geometry specialization: when every visible launch of a `static __global__`
kernel uses one constant block triple, it folds `blockDim.{x,y,z}` (both direct
ntid intrinsics AND the `__cuda_builtin_blockDim_t::__fetch_builtin_*` call path)
to `cir.const` and emits `cir.nvvm.reqntid = "bx,by,bz"`. Runs before
launch-bounds propagation (PTX forbids maxntid+reqntid together); LBP now skips
reqntid carriers. Never folds tid/ctaid/gridDim. No clone/dispatch.

VALIDATED results (A10/sm_86, fork clang 24 @ f1377581, full cir-offload-merge pipeline):
- FileCheck: direct-ntid substitution + reqntid; runtime-block no-op; multi-shape no-op; all PASS.
- LaunchBounds regression test still PASSES (reqntid suppression works).
- Real `lsgs_e2e_static.cu` (static __global__, 256-thread block): through
  cir-offload-merge emits `cir.nvvm.reqntid = "256,1,1"`; blockDim fetch-builtin
  calls -> 0 in kernel body; address math folds blockDim to cir.const 256; survives split.
- Real PolyBench GEMM (gemm_kernel, static __global__, block dim3(32,8,1)):
  through the FULL pipeline emits `cir.nvvm.reqntid = "32,8,1"` (the exact block).
- 11 pre-existing abi-lowering/ lit failures are unrelated (untouched subsystem).

CANDIDATE WORKLOADS (static __global__ + constant block + blockDim index math):
- PolyBench 15: 2MM 3MM ADI ATAX BICG DOITGEN GEMM GEMVER GESUMMV JACOBI1D
  JACOBI2D LU MVT SYR2K SYRK.
- HeCBench static 5: cc floydwarshall2 graphB+ mpc rsmt (floydwarshall2/mpc/rsmt
  read 9-11 geometry sregs).

MY QUESTIONS (answer precisely, honest — do NOT inflate):
Q1. Is emitting `nvvm.reqntid` + folding blockDim-to-constant a sound, correct,
   PR-reviewable contribution in its own right even WITHOUT a measured runtime
   speedup? Does ptxas actually DO anything useful with `.reqntid` alone (vs
   `.maxntid`) for register/occupancy, or is reqntid primarily a correctness/
   exactness contract whose perf value comes only from downstream LLVM folds?
   What is the HONEST claim we can make in the PR body?

Q2. Where, if anywhere, would LSGS actually produce a MEASURABLE runtime win on
   the candidate class? Is it (a) the blockDim-constant folds enabling better LLVM
   optimization of address math / loop unrolling, (b) reqntid letting ptxas budget
   registers tighter than maxntid, or (c) neither without also emitting the forcing
   minctasm? Given v1 EXCLUDES minctasm, is a runtime speedup on real kernels
   realistic or should the PR be framed as codegen/simplification?

Q3. For the PR to be defensible as a `[PROTOTYPE]`, what is the MINIMUM honest
   evidence set? (e.g. FileCheck tests + the GEMM/synthetic reqntid emission +
   SASS-level or LLVM-IR-level simplification proof, WITHOUT claiming runtime
   speedup). What should I explicitly NOT claim?

Q4. Is this pass worth landing as-is, or is the value only realized once
   combined with the (excluded here) minctasm forcing lever? i.e. is the v1
   reqntid+fold a meaningful standalone CIR contribution or a stepping stone?

Q5. PR body structure: given the user insists on real measurements, no MLM slop,
   and a model-assist note, what sections/facts belong, and what numbers (wall
   time, build time, test counts) are appropriate to include?

Give a honest, decision-usable answer. Do NOT inflate the result. Ensure the PR
would survive an experienced LLVM reviewer (e.g. Dinos/Andy Kaylor) who knows
launch bounds are often perf-neutral.