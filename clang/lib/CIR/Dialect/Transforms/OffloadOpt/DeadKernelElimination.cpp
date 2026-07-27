//===- DeadKernelElimination.cpp - Remove unlaunchable device kernels -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_OFFLOADDEADKERNELELIMINATION
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

// A symbol is used outside its own body if any reference to it lives in an op
// that is not nested within the symbol itself. This matters because a CUDA
// device stub takes its own address to launch, so it always references itself.
static bool hasUseOutsideSelf(cir::FuncOp fn, mlir::Operation *scope) {
  auto uses = mlir::SymbolTable::getSymbolUses(fn, scope);
  if (!uses)
    return true; // Uses could not be enumerated: conservatively keep.
  return llvm::any_of(*uses, [&](const mlir::SymbolTable::SymbolUse &use) {
    return !fn->isProperAncestor(use.getUser());
  });
}

static bool isDeviceKernel(cir::FuncOp fn) {
  cir::CallingConv cc = fn.getCallingConv();
  return cc == cir::CallingConv::PTXKernel ||
         cc == cir::CallingConv::AMDGPUKernel;
}

struct OffloadDeadKernelEliminationPass
    : public impl::OffloadDeadKernelEliminationBase<
          OffloadDeadKernelEliminationPass> {
  void runOnOperation() override;
};

void OffloadDeadKernelEliminationPass::runOnOperation() {
  cir::OffloadContainerOp container = getOperation();
  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  mlir::ModuleOp hostModule = container.getHostModule();

  llvm::SmallVector<cir::FuncOp> toErase;
  llvm::DenseSet<cir::FuncOp> live;

  // Phase 1 (stub-anchored): a host stub with internal linkage and no launch
  // left is dead. External stubs are kept -- another TU may launch through
  // them. Device kernels of a surviving stub are marked live so phase 2 keeps
  // them; those of a dead stub fall through to phase 2's device-side check.
  for (const auto &binding : table) {
    cir::FuncOp stub = binding.second.hostStub;
    bool stubDead = cir::isLocalLinkage(stub.getLinkage()) &&
                    !hasUseOutsideSelf(stub, hostModule);
    if (stubDead) {
      toErase.push_back(stub);
    } else {
      for (cir::FuncOp kernel : binding.second.deviceKernels)
        live.insert(kernel);
    }
  }

  // Phase 2 (device-anchored): a kernel with no live host stub is a
  // static/anonymous kernel never launched in this TU. Delete it unless device
  // code still references it (dynamic parallelism / address-taken).
  for (mlir::ModuleOp deviceMod : container.getDeviceModules()) {
    deviceMod.walk([&](cir::FuncOp fn) {
      if (!isDeviceKernel(fn) || live.contains(fn))
        return;
      if (!hasUseOutsideSelf(fn, deviceMod))
        toErase.push_back(fn);
    });
  }

  // Cache binding table if nothing was deleted.
  if (toErase.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  for (cir::FuncOp fn : toErase)
    fn.erase();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadDeadKernelEliminationPass() {
  return std::make_unique<OffloadDeadKernelEliminationPass>();
}
