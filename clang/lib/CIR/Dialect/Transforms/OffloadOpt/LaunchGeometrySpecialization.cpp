//===- LaunchGeometrySpecialization.cpp - Specialize on exact geometry ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Launch-site geometry specialization. When every visible launch of a kernel
// uses the same compile-time block shape, the device kernel's `ntid.{x,y,z}`
// reads are replaced by that constant and the kernel is annotated with
// `nvvm.reqntid` so ptxas treats the exact (not maximum) CTA shape as a
// contract.
//
// This is the in-place, sound version. It deliberately does not clone kernels
// or dispatch between a specialized and a generic copy: a cloned NVPTX kernel
// would need matching host-stub + CUDA registration rewriting to remain
// launchable by the runtime, which is out of scope for this pass. Instead, an
// exact shape is only stamped when every launch of the kernel that this
// pipeline can see uses that shape, so there is no fallback launch to route.
//
// `nvvm.reqntid` is a hard contract, not a hint: launching the kernel with any
// other block shape is a runtime launch failure. The exact-shape proof below
// is therefore conservative -- any launch that cannot be confidently traced to
// a constant block triple disqualifies the kernel entirely.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/NVVMAttributes.h"

namespace mlir {
#define GEN_PASS_DEF_OFFLOADLAUNCHGEOMETRYSPECIALIZATION
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

// The constant block triple every visible launch agrees on, or nullopt when any
// launch is opaque, runtime-valued, or disagrees on a component. The triple is
// compared component-wise: (32,8,1) and (256,1,1) are different shapes even
// though they have the same valid thread count.
static std::optional<std::array<uint64_t, 3>>
commonBlockShape(llvm::ArrayRef<cir::LaunchSite> sites) {
  std::optional<std::array<uint64_t, 3>> common;
  for (const cir::LaunchSite &site : sites) {
    if (!site.hasGeometry())
      return std::nullopt;
    cir::LaunchSite::Dim3 block = site.getBlockDim();
    if (!block.isFullyConstant())
      return std::nullopt;

    auto asInt = [](mlir::TypedAttr a) -> std::optional<uint64_t> {
      auto i = mlir::dyn_cast_or_null<cir::IntAttr>(a);
      if (!i)
        return std::nullopt;
      return i.getUInt();
    };
    std::optional<uint64_t> x = asInt(block.constX());
    std::optional<uint64_t> y = asInt(block.constY());
    std::optional<uint64_t> z = asInt(block.constZ());
    if (!x || !y || !z)
      return std::nullopt;

    std::array<uint64_t, 3> shape = {*x, *y, *z};
    if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0)
      return std::nullopt;
    if (!common)
      common = shape;
    else if (common != shape)
      return std::nullopt;
  }
  return common ? std::optional<std::array<uint64_t, 3>>(common) : std::nullopt;
}

// The mangled fresh blockDim helper names. Real device CIR reads `blockDim`
// through the always-inline `__cuda_builtin_blockDim_t::__fetch_builtin_{x,y,z}`
// helper, which internally reads the `ntid.{x,y,z}` special register. The
// helpers are `linkonce_odr` and shared across kernels, so the pass must NOT
// rewrite their bodies; it replaces the per-call result in the kernel body,
// leaving the shared helper (and any other kernel that reads it) untouched.
static StringRef blockDimFetchAxis(cir::CallOp op) {
  if (auto callee = op.getCalleeAttr(); callee) {
    StringRef name = callee.getValue();
    StringRef memb = "__cuda_builtin_blockDim_t17__fetch_builtin_";
    size_t pos = name.find(memb);
    if (pos != StringRef::npos) {
      StringRef rest = name.drop_front(pos + memb.size());
      if (rest.starts_with("x"))
        return "x";
      if (rest.starts_with("y"))
        return "y";
      if (rest.starts_with("z"))
        return "z";
    }
  }
  return "";
}

// Replace every read of the named geometry special register in `kernel` with a
// `cir.const` carrying the given value, reusing the result type so any
// downstream zext/sext/cast in the original IR stays correct. Also replaces the
// result of a `blockDim_t::__fetch_builtin_{x,y,z}` call, which is how real
// device CIR materializes `blockDim.{x,y,z}` at the call site.
static bool substituteGeometryRead(cir::FuncOp kernel, StringRef sreg,
                                   uint64_t value) {
  bool changed = false;

  auto replaceWithConst = [&](mlir::Value result, mlir::Type resTy,
                              mlir::Operation *op) {
    unsigned width = 32;
    if (auto intTy = mlir::dyn_cast<cir::IntType>(resTy))
      width = intTy.getWidth();
    auto attr = cir::IntAttr::get(resTy, llvm::APInt(width, value));
    mlir::OpBuilder builder(op);
    auto constant = builder.create<cir::ConstantOp>(op->getLoc(), attr);
    result.replaceAllUsesWith(constant);
    op->erase();
    changed = true;
  };

  kernel.walk([&](cir::LLVMIntrinsicCallOp op) {
    if (op.getIntrinsicName() != sreg)
      return;
    if (op->getNumResults() != 1)
      return;
    replaceWithConst(op.getResult(), op->getResultTypes()[0], op);
  });

  // Handle the alias call path: `blockDim.x` is materialized as a call to the
  // shared `__fetch_builtin_x` helper rather than a direct intrinsic, depending
  // on the CIRGen / lowering stage. Only replace calls whose callee is the
  // blockDim_t fetch helper for the matching axis; never rewrite the helper
  // body, never touch blockIdx/gridDim/threadIdx fetch helpers.
  if (sreg == "nvvm.read.ptx.sreg.ntid.x") {
    kernel.walk([&](cir::CallOp op) {
      if (blockDimFetchAxis(op) != "x")
        return;
      if (op->getNumResults() != 1)
        return;
      replaceWithConst(op.getResult(), op->getResultTypes()[0], op);
    });
  }
  if (sreg == "nvvm.read.ptx.sreg.ntid.y") {
    kernel.walk([&](cir::CallOp op) {
      if (blockDimFetchAxis(op) != "y")
        return;
      if (op->getNumResults() != 1)
        return;
      replaceWithConst(op.getResult(), op->getResultTypes()[0], op);
    });
  }
  if (sreg == "nvvm.read.ptx.sreg.ntid.z") {
    kernel.walk([&](cir::CallOp op) {
      if (blockDimFetchAxis(op) != "z")
        return;
      if (op->getNumResults() != 1)
        return;
      replaceWithConst(op.getResult(), op->getResultTypes()[0], op);
    });
  }

  return changed;
}

struct OffloadLaunchGeometrySpecializationPass
    : public impl::OffloadLaunchGeometrySpecializationBase<
          OffloadLaunchGeometrySpecializationPass> {
  OffloadLaunchGeometrySpecializationPass(unsigned minCtasPerSm)
      : minCtasPerSm(minCtasPerSm) {}
  void runOnOperation() override;

private:
  // Opt-in .minnctapersm target (0 = none). Passed via the factory so the pass
  // body stays option-free (matching CallConvLowering); selecting a profitable
  // M is a ptxas-feedback / runtime decision done outside the pass.
  unsigned minCtasPerSm = 0;
};

void OffloadLaunchGeometrySpecializationPass::runOnOperation() {
  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  mlir::OpBuilder builder(&getContext());
  bool changed = false;

  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;
    llvm::ArrayRef<cir::LaunchSite> sites = binding.launchSites;

    // A kernel launched nowhere in this TU says nothing about its geometry.
    if (sites.empty())
      continue;
    if (!table.allLaunchSitesVisible(entry.first))
      continue;

    std::optional<std::array<uint64_t, 3>> shape = commonBlockShape(sites);
    if (!shape)
      continue;

    for (cir::FuncOp kernel : binding.deviceKernels) {
      if (kernel.isDeclaration())
        continue;
      if (kernel.getCallingConv() != cir::CallingConv::PTXKernel)
        continue;

      // The kernel must not be reachable from device code (dynamic
      // parallelism, address-taken) under a different shape.
      if (cir::hasUseOutsideSelf(kernel, kernel->getParentOfType<mlir::ModuleOp>()))
        continue;

      // Skip if the kernel already carries an incompatible exact-shape or a
      // user-authored maxntid we must not silently drop (PTX forbids maxntid
      // together with reqntid).
      {
        std::string reqName = ("cir." + llvm::NVVMAttr::ReqNTID).str();
        if (auto existing =
                kernel->getAttrOfType<mlir::StringAttr>(reqName)) {
          if (existing.getValue() != llvm::join(llvm::ArrayRef<std::string>{
                   llvm::utostr((*shape)[0]), llvm::utostr((*shape)[1]),
                   llvm::utostr((*shape)[2])}, ","))
            continue;
        }
        std::string maxName = ("cir." + llvm::NVVMAttr::MaxNTID).str();
        if (kernel->hasAttr(maxName))
          continue;
      }

      bool kernelChanged = false;
      kernelChanged |= substituteGeometryRead(
          kernel, "nvvm.read.ptx.sreg.ntid.x", (*shape)[0]);
      kernelChanged |= substituteGeometryRead(
          kernel, "nvvm.read.ptx.sreg.ntid.y", (*shape)[1]);
      kernelChanged |= substituteGeometryRead(
          kernel, "nvvm.read.ptx.sreg.ntid.z", (*shape)[2]);

      if (!kernelChanged)
        continue;

      std::string reqName = ("cir." + llvm::NVVMAttr::ReqNTID).str();
      std::string value = llvm::join(llvm::ArrayRef<std::string>{
          llvm::utostr((*shape)[0]), llvm::utostr((*shape)[1]),
          llvm::utostr((*shape)[2])}, ",");
      kernel->setAttr(reqName, builder.getStringAttr(value));
      changed = true;

      // Opt-in .minnctapersm forcing. Never inferred; only emitted when the
      // pass option requests a specific M. Guard M*B against an obviously
      // infeasible residency (ptxas safely ignores an unattainable M, but a
      // request beyond 1024 threads/SM is never useful on sm_86/sm_90 and
      // would otherwise risk a launch contract ptxas rejects). The choice of a
      // profitable M is a ptxas-feedback / runtime decision done outside the
      // pass; the default (0) emits nothing.
      if (minCtasPerSm > 0) {
        uint64_t threadsPerBlock =
            (*shape)[0] * (*shape)[1] * (*shape)[2];
        if (minCtasPerSm * threadsPerBlock <= 1024) {
          std::string minName = ("cir." + llvm::NVVMAttr::MinCTASm).str();
          kernel->setAttr(minName, builder.getStringAttr(llvm::utostr(minCtasPerSm)));
        }
      }
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
}

} // namespace

std::unique_ptr<Pass>
mlir::createOffloadLaunchGeometrySpecializationPass(unsigned minCtasPerSm) {
  return std::make_unique<OffloadLaunchGeometrySpecializationPass>(
      minCtasPerSm);
}