[0m
> build · gpt-5.6-sol
[0m
I’ll validate the brief against the implementation and tests, then separate guaranteed semantic/codegen effects from speculative performance effects.
[0m$ [0mgit show --stat --oneline f1377581 && git show --format=fuller --no-ext-diff f1377581
f1377581c624 [CIR][offload-merge] LSGS: also fold blockDim_t::__fetch_builtin_* call path
 .../OffloadOpt/LaunchGeometrySpecialization.cpp    | 85 +++++++++++++++++++---
 1 file changed, 75 insertions(+), 10 deletions(-)
commit f1377581c6243e73a345ec9e640ee16f4c99458a
Author:     David Rivera <davidriverg@gmail.com>
AuthorDate: Thu Sep 3 05:10:36 2026 -0400
Commit:     David Rivera <davidriverg@gmail.com>
CommitDate: Thu Sep 3 05:10:36 2026 -0400

    [CIR][offload-merge] LSGS: also fold blockDim_t::__fetch_builtin_* call path
    
    Real device CIR materializes blockDim.{x,y,z} as a call to the always-inline
    __cuda_builtin_blockDim_t::__fetch_builtin_{x,y,z} helper (linkonce_odr,
    shared across kernels), not a direct ntid intrinsic in the kernel body.
    Replace the per-call result with the constant, leaving the shared helper
    body (and any other kernel reading it) untouched.

diff --git a/clang/lib/CIR/Dialect/Transforms/OffloadOpt/LaunchGeometrySpecialization.cpp b/clang/lib/CIR/Dialect/Transforms/OffloadOpt/LaunchGeometrySpecialization.cpp
index 7aad433bdf86..d83a9d2902c6 100644
--- a/clang/lib/CIR/Dialect/Transforms/OffloadOpt/LaunchGeometrySpecialization.cpp
+++ b/clang/lib/CIR/Dialect/Transforms/OffloadOpt/LaunchGeometrySpecialization.cpp
@@ -79,28 +79,93 @@ commonBlockShape(llvm::ArrayRef<cir::LaunchSite> sites) {
   return common ? std::optional<std::array<uint64_t, 3>>(common) : std::nullopt;
 }
 
+// The mangled fresh blockDim helper names. Real device CIR reads `blockDim`
+// through the always-inline `__cuda_builtin_blockDim_t::__fetch_builtin_{x,y,z}`
+// helper, which internally reads the `ntid.{x,y,z}` special register. The
+// helpers are `linkonce_odr` and shared across kernels, so the pass must NOT
+// rewrite their bodies; it replaces the per-call result in the kernel body,
+// leaving the shared helper (and any other kernel that reads it) untouched.
+static StringRef blockDimFetchAxis(cir::CallOp op) {
+  if (auto callee = op.getCalleeAttr(); callee) {
+    StringRef name = callee.getValue();
+    StringRef memb = "__cuda_builtin_blockDim_t17__fetch_builtin_";
+    size_t pos = name.find(memb);
+    if (pos != StringRef::npos) {
+      StringRef rest = name.drop_front(pos + memb.size());
+      if (rest.starts_with("x"))
+        return "x";
+      if (rest.starts_with("y"))
+        return "y";
+      if (rest.starts_with("z"))
+        return "z";
+    }
+  }
+  return "";
+}
+
 // Replace every read of the named geometry special register in `kernel` with a
-// `cir.const` carrying the given value, reusing the intrinsic's result type so
-// any downstream zext/sext/cast in the original IR stays correct.
+// `cir.const` carrying the given value, reusing the result type so any
+// downstream zext/sext/cast in the original IR stays correct. Also replaces the
+// result of a `blockDim_t::__fetch_builtin_{x,y,z}` call, which is how real
+// device CIR materializes `blockDim.{x,y,z}` at the call site.
 static bool substituteGeometryRead(cir::FuncOp kernel, StringRef sreg,
                                    uint64_t value) {
   bool changed = false;
-  kernel.walk([&](cir::LLVMIntrinsicCallOp op) {
-    if (op.getIntrinsicName() != sreg)
-      return;
-    if (op->getNumResults() != 1)
-      return;
-    mlir::Type resTy = op->getResultTypes()[0];
+
+  auto replaceWithConst = [&](mlir::Value result, mlir::Type resTy,
+                              mlir::Operation *op) {
     unsigned width = 32;
     if (auto intTy = mlir::dyn_cast<cir::IntType>(resTy))
       width = intTy.getWidth();
     auto attr = cir::IntAttr::get(resTy, llvm::APInt(width, value));
     mlir::OpBuilder builder(op);
     auto constant = builder.create<cir::ConstantOp>(op->getLoc(), attr);
-    op.getResult().replaceAllUsesWith(constant);
-    op.erase();
+    result.replaceAllUsesWith(constant);
+    op->erase();
     changed = true;
+  };
+
+  kernel.walk([&](cir::LLVMIntrinsicCallOp op) {
+    if (op.getIntrinsicName() != sreg)
+      return;
+    if (op->getNumResults() != 1)
+      return;
+    replaceWithConst(op.getResult(), op->getResultTypes()[0], op);
   });
+
+  // Handle the alias call path: `blockDim.x` is materialized as a call to the
+  // shared `__fetch_builtin_x` helper rather than a direct intrinsic, depending
+  // on the CIRGen / lowering stage. Only replace calls whose callee is the
+  // blockDim_t fetch helper for the matching axis; never rewrite the helper
+  // body, never touch blockIdx/gridDim/threadIdx fetch helpers.
+  if (sreg == "nvvm.read.ptx.sreg.ntid.x") {
+    kernel.walk([&](cir::CallOp op) {
+      if (blockDimFetchAxis(op) != "x")
+        return;
+      if (op->getNumResults() != 1)
+        return;
+      replaceWithConst(op.getResult(), op->getResultTypes()[0], op);
+    });
+  }
+  if (sreg == "nvvm.read.ptx.sreg.ntid.y") {
+    kernel.walk([&](cir::CallOp op) {
+      if (blockDimFetchAxis(op) != "y")
+        return;
+      if (op->getNumResults() != 1)
+        return;
+      replaceWithConst(op.getResult(), op->getResultTypes()[0], op);
+    });
+  }
+  if (sreg == "nvvm.read.ptx.sreg.ntid.z") {
+    kernel.walk([&](cir::CallOp op) {
+      if (blockDimFetchAxis(op) != "z")
+        return;
+      if (op->getNumResults() != 1)
+        return;
+      replaceWithConst(op.getResult(), op->getResultTypes()[0], op);
+    });
+  }
+
   return changed;
 }
[0m
[0m$ [0mgit diff --stat bb385fee..f1377581 && git diff --name-only bb385fee..f1377581
clang/include/clang/CIR/Dialect/Passes.h           |   1 +
 clang/include/clang/CIR/Dialect/Passes.td          |  37 +++
 .../Dialect/Transforms/OffloadOpt/CMakeLists.txt   |   1 +
 .../OffloadOpt/LaunchBoundsPropagation.cpp         |   9 +-
 .../OffloadOpt/LaunchGeometrySpecialization.cpp    | 254 +++++++++++++++++++++
 .../Transforms/launch-geometry-specialization.cir  | 160 +++++++++++++
 .../tools/cir-offload-merge/cir-offload-merge.cpp  |  11 +
 clang/tools/cir-opt/cir-opt.cpp                    |   4 +
 8 files changed, 476 insertions(+), 1 deletion(-)
clang/include/clang/CIR/Dialect/Passes.h
clang/include/clang/CIR/Dialect/Passes.td
clang/lib/CIR/Dialect/Transforms/OffloadOpt/CMakeLists.txt
clang/lib/CIR/Dialect/Transforms/OffloadOpt/LaunchBoundsPropagation.cpp
clang/lib/CIR/Dialect/Transforms/OffloadOpt/LaunchGeometrySpecialization.cpp
clang/test/CIR/Transforms/launch-geometry-specialization.cir
clang/tools/cir-offload-merge/cir-offload-merge.cpp
clang/tools/cir-opt/cir-opt.cpp
[0m
[0m✱ [0mGrep "reqntid|maxntid|minctasm|LaunchGeometrySpecialization"[90m in clang · 78 matches[0m
[0m
[0m$ [0mgit log --oneline --decorate bb385fee..f1377581
f1377581c624 (HEAD -> gsoc/lsgs-geometry-spec, fork/gsoc/lsgs-geometry-spec) [CIR][offload-merge] LSGS: also fold blockDim_t::__fetch_builtin_* call path
6730696b01c7 [CIR][offload-merge] Fix LSGS FileCheck directives (reqntid on func line, consts via CHECK)
cebc3740ca12 [CIR][offload-merge] Fix LSGS test CIR syntax (void ptx_kernel, no cir.binop)
2b31d36710ca [CIR][offload-merge] Launch geometry specialization (in-place ntid substitution + reqntid)
[0m
[0m✱ [0mGrep "reqntid|maxntid|minnctapersm"[90m in llvm · 91 matches[0m
[0m→ [0mRead clang/test/CIR/Transforms/launch-geometry-specialization.cir
[0m→ [0mRead llvm/lib/Target/NVPTX/NVVMIntrRange.cpp[90m [offset=65, limit=100][0m
[0m→ [0mRead llvm/test/CodeGen/NVPTX/reqntid-const-fold.ll
[0m→ [0mRead llvm/lib/Target/NVPTX/NVVMProperties.cpp[90m [offset=220, limit=85][0m
[0m→ [0mRead clang/lib/CIR/Dialect/Transforms/OffloadOpt/LaunchGeometrySpecialization.cpp
[0m✗ [0mRead clang/lib/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h failed
[91m[1mError: [0mFile not found: /Users/davidfeliperiveraguerra/dev/gsoc-combine/clang/lib/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h
[0m→ [0mRead clang/include/clang/CIR/Dialect/Passes.td[90m [offset=380, limit=38][0m
[0m→ [0mRead clang/lib/CIR/Dialect/Transforms/OffloadOpt/LaunchBoundsPropagation.cpp[90m [offset=95, limit=45][0m
[0m✱ [0mGrep "allLaunchSitesVisible|hasUseOutsideSelf"[90m in clang/lib/CIR · 8 matches[0m
[0m✱ [0mGlob "**/KernelBindingTable.{h,cpp}"[90m in clang · 2 matches[0m
[0m→ [0mRead clang/lib/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.cpp[90m [offset=250, limit=75][0m
[0m→ [0mRead clang/include/clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h[90m [offset=1, limit=220][0m
[0m% [0mWebFetch https://docs.nvidia.com/cuda/parallel-thread-execution/#performance-tuning-directives-minnctapersm
[0m% [0mWebFetch https://docs.nvidia.com/cuda/parallel-thread-execution/#performance-tuning-directives-reqntid
[0m% [0mWebFetch https://docs.nvidia.com/cuda/parallel-thread-execution/#performance-tuning-directives-maxntid
The implementation’s strongest standalone effect is already visible: it converts a cross-module launch fact into an exact kernel contract and exposes constants before lowering. The uncertain part is any independent `ptxas` register-allocation benefit; that should be treated as an empirical question, not asserted from `.reqntid` semantics.
[0m✱ [0mGrep "The \.reqntid directive|\.reqntid directive specifies|requested number of threads|must be launched|launch.*fail|exact number"[90m in /Users/davidfeliperiveraguerra/.local/share/opencode/tool-output · 9 matches[0m
[0m✱ [0mGrep "The \.maxntid directive|maximum number of threads|register use|register usage|register.*thread|occupancy"[90m in /Users/davidfeliperiveraguerra/.local/share/opencode/tool-output · 54 matches[0m
[0m✱ [0mGrep "The \.minnctapersm directive|minimum number|register use|register usage|register.*thread|occupancy"[90m in /Users/davidfeliperiveraguerra/.local/share/opencode/tool-output · 51 matches[0m
[0m✱ [0mGrep "reqntid.*maxntid|maxntid.*reqntid|mutually exclusive"[90m in /Users/davidfeliperiveraguerra/.local/share/opencode/tool-output · 27 matches[0m
**Decision**

LSGS v1 is a meaningful standalone prototype, but its defensible value is **exact launch-geometry propagation and early IR simplification**, not demonstrated runtime performance. Post it as `[PROTOTYPE]` or RFC-level work. Do not present it as a performance optimization until enabled/disabled output or runtime measurements show a material difference.

**Q1. Soundness And `.reqntid` Value**

Yes, the transformation is conceptually sound when the visibility proof is complete:

- The host stub has local linkage.
- Every use of that stub is recognized as a launch.
- Every launch has the same constant component-wise block shape.
- The device kernel has no externally visible/device-side use that could launch it differently.
- Existing incompatible `reqntid` and any `maxntid` are rejected.

Those conditions are represented in `KernelBindingTable::allLaunchSitesVisible()` and the device-use checks around `LaunchGeometrySpecialization.cpp:183-223`. Under them, replacing `%ntid.{x,y,z}` with constants is a valid specialization, and adding `.reqntid` preserves that assumption as a runtime-enforced contract.

`.reqntid` has two concrete effects:

1. It requires every launch to use the stated CTA dimensions. A mismatch causes a runtime error or launch failure.
2. It communicates exact dimensions to compiler stages. LLVM’s `NVVMIntrRange` uses `nvvm.reqntid` to give `%ntid` single-value ranges and `%tid` dimension-specific ranges (`llvm/lib/Target/NVPTX/NVVMIntrRange.cpp:82-137`).

What it does **not** guarantee is improved occupancy or a different register allocation. PTX documents resource tradeoffs primarily for `.minnctapersm` combined with `.maxntid` or `.reqntid`. Without `.minnctapersm`, `.reqntid` supplies exact geometry but does not impose a desired number of resident CTAs.

There is therefore no defensible general claim that:

> `.reqntid` budgets registers more tightly than `.maxntid`.

For occupancy-related register budgeting, equal-total-thread `.maxntid` already supplies the relevant upper bound. Exact dimensions may expose additional optimization opportunities, but that is not an occupancy guarantee.

An honest PR claim is:

> This pass propagates an exact block shape from all visible launch sites into an internal device kernel. It replaces `blockDim`/`ntid` reads with constants and emits `nvvm.reqntid`, preserving the specialization as an exact launch contract. This exposes launch geometry to CIR and downstream LLVM optimization. No runtime speedup or occupancy improvement is claimed.

One nuance: because LLVM already derives single-value `%ntid` ranges from `nvvm.reqntid`, some late folding may happen from the attribute alone. Your direct CIR substitution remains useful if it enables CIR-level simplification or removes the CUDA builtin helper path before lowering, but the PR should distinguish that from functionality already available downstream.

**Q2. Where A Runtime Win Could Occur**

The realistic source is **(a), constant propagation through address arithmetic or control flow**, not **(b), register budgeting from `.reqntid` alone**.

Possible wins include:

- Folding expressions such as `blockIdx.x * blockDim.x`.
- Strength-reducing multiplication by a known block dimension.
- Eliminating branches dependent on `blockDim`.
- Resolving loop trip counts and enabling unrolling.
- Simplifying multidimensional linearization.
- Removing repeated builtin-helper materialization if it otherwise survives optimization.
- Supplying tighter `%tid.{x,y,z}` ranges downstream through `reqntid`.

However, ordinary CUDA index calculations are already easy for `ptxas`, and `%ntid` special-register reads are cheap. Many PolyBench kernels will therefore produce identical or nearly identical SASS. Even when SASS changes, memory traffic or arithmetic throughput may dominate enough that runtime remains unchanged.

For `reqntid` alone:

- A register-count change is possible as an implementation consequence.
- An occupancy improvement is possible only if that change crosses an occupancy boundary.
- Neither is guaranteed or even especially likely without `.minnctapersm`.
- `.reqntid` does not “force tighter registers” by itself.

A measurable win is most plausible in kernels where a block dimension controls substantial compile-time-transformable code, not merely one indexing multiply. Candidates with 9-11 geometry reads are useful for code-size experiments, but geometry-read count alone does not predict runtime benefit.

Use an enabled/disabled ablation and compare:

- Optimized LLVM IR.
- PTX instruction sequence.
- SASS instruction count.
- Registers per thread.
- Spill loads/stores.
- Static occupancy limits.
- Kernel-only runtime.

If SASS and resource usage are identical, a runtime speedup is not realistic regardless of CIR simplification.

**Q3. Minimum Evidence**

For a defensible `[PROTOTYPE]`, the minimum evidence is:

1. **Automated positive tests**
   - Direct `ntid.x/y/z` substitution.
   - Real `__cuda_builtin_blockDim_t::__fetch_builtin_{x,y,z}` call-path substitution.
   - Component-wise 2D or 3D geometry.
   - `nvvm.reqntid` emission.

2. **Automated negative tests**
   - Runtime-valued block dimension.
   - Multiple differing shapes.
   - Externally callable/non-local stub.
   - Unrecognized stub use or address escape.
   - Device-side/address-taken kernel use.
   - Existing incompatible `reqntid`.
   - Existing `maxntid`.
   - No launches.

3. **Pipeline interaction**
   - LSGS runs before launch-bounds propagation.
   - LBP does not add `maxntid` to a `reqntid` carrier.
   - The attribute survives offload splitting and LLVM lowering.
   - Resulting PTX contains `.reqntid 32, 8, 1` or the corresponding tested shape.

4. **One real source example**
   - GEMM is sufficient to demonstrate applicability.
   - Show source launch shape, specialized CIR, and final PTX.
   - State that this demonstrates propagation, not speedup.

5. **One downstream-difference table**
   - Compare enabled versus disabled LLVM IR/PTX/SASS.
   - If only CIR changes, say exactly that.
   - If final SASS is identical, report that directly.
   - If registers or instructions change, report raw values without inferring runtime impact.

The current manually validated helper-call example should become an automated lit test before asking to land. The three existing tests are enough to demonstrate the central idea, but they are thin for a transformation that introduces a hard runtime contract.

Explicitly do **not** claim:

- Runtime acceleration without controlled timings.
- Improved occupancy without occupancy/resource evidence.
- Reduced register usage unless `ptxas` reports it.
- That `.reqntid` is equivalent to `.minnctapersm`.
- That eliminating `%ntid` reads necessarily eliminates SASS instructions.
- General applicability to externally visible kernels.
- Whole-program launch visibility beyond the restrictions actually proved.
- A benefit proportional to geometry-register read count.
- That 11 unrelated failures establish general test-suite cleanliness.

**Q4. Landing Value**

It has standalone value, but that value is narrower than a performance feature:

- It demonstrates useful host/device analysis over `cir.offload.container`.
- It propagates information unavailable during ordinary device-only compilation.
- It provides exact geometry to CIR transformations.
- It emits the semantically appropriate PTX contract.
- It creates infrastructure upon which cloning, dispatch, or occupancy-directed policy could later be built.

That makes it more than merely scaffolding for `minctasm`. Still, whether upstream reviewers want a dedicated pass depends on observable consumers. If the final generated code is always identical because LLVM/`ptxas` already recover everything from `.reqntid`, reviewers may reasonably ask why CIR should rewrite the reads itself. The strongest answer is a demonstrated CIR or LLVM optimization unlocked by early substitution.

Recommended landing position:

- **Worth posting now as `[PROTOTYPE]`.**
- **Potentially worth landing as-is** if tests are strengthened and at least one real pipeline example demonstrates a concrete IR/codegen simplification.
- **Do not make landing depend on runtime speedup.**
- **Do not add `minctasm` merely to manufacture a benchmark result.** It changes register-allocation policy and can improve or regress kernels, so it requires separate justification and tuning policy.

If no downstream difference survives beyond prettier CIR, characterize v1 as infrastructure/code-quality work and expect reviewers to question the pass’s cost-benefit ratio.

**Q5. PR Body Structure**

Use this structure:

1. **Problem**
   - Device compilation does not ordinarily know the launch-site block shape.
   - The combined CIR container can prove an exact shape for a restricted kernel class.

2. **Transformation**
   - Define the eligibility conditions precisely.
   - State that specialization is in-place.
   - State that no cloning or dispatch is performed.
   - List exactly what is folded and what is deliberately untouched.

3. **Correctness**
   - Explain local host-stub linkage and complete-use enumeration.
   - Explain rejection of opaque, runtime, and disagreeing launches.
   - Explain device-side-use rejection.
   - Explain that `.reqntid` is a runtime-enforced contract.
   - Explain the `maxntid` incompatibility and pass ordering.

4. **Examples**
   - Small before/after CIR.
   - Real GEMM: launch `(32,8,1)` becomes `cir.nvvm.reqntid = "32,8,1"`.
   - Final PTX contains `.reqntid 32, 8, 1`.

5. **Evaluation**
   - Exact compiler revision, CUDA/PTX toolchain, target `sm_86`, and GPU A10.
   - Enabled/disabled comparison.
   - CIR/LLVM IR/PTX/SASS changes.
   - `ptxas` registers, spills, and code size.
   - Runtime results only if measured correctly.
   - Report neutral results as neutral.

6. **Testing**
   - Name the targeted lit tests and scenarios.
   - Report the launch-bounds regression.
   - Mention the 11 unrelated failures only with baseline comparison showing they occur both with and without the patch.

7. **Limitations**
   - Internal/static kernels only.
   - Constant identical block shapes only.
   - No fallback.
   - No `gridDim`, `threadIdx`, `blockIdx`, cloning, dispatch, or `minctasm`.
   - No general performance claim.

8. **Model Assistance**
   - Short factual disclosure, for example:
   > An AI assistant was used to review wording and explore test cases. The implementation, validation, measurements, and conclusions were checked by the author.

Build wall time is generally irrelevant. Include it only if measuring compile-time overhead with repeated controlled builds. Personal incremental-build duration is not useful review evidence.

Appropriate numbers are:

- Number of newly added targeted tests, with scenarios named.
- Baseline versus patched test results.
- Compile-time delta only from repeated comparable runs.
- PTX/SASS instruction counts.
- Registers and spills from `ptxas`.
- Kernel runtime distributions, not a single timing.
- Median or mean with dispersion, repetition count, warm-up policy, and timer type.

The shortest honest headline is:

> LSGS safely turns a proven exact launch shape into CIR constants and a PTX `.reqntid` contract. It demonstrates real full-pipeline geometry propagation; performance impact is workload-dependent and has not been established.
