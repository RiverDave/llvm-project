//===- DeadKernelElimination.cpp - Remove unlaunchable device kernels -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Remarks.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "cir-offload-dead-kernel-elimination"

namespace mlir {
#define GEN_PASS_DEF_OFFLOADDEADKERNELELIMINATION
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

static constexpr llvm::StringLiteral kRemarkName = "DeadKernelElimination";
static constexpr llvm::StringLiteral kRemarkCategory = "cir-offload-dke";

static void remarkErased(cir::FuncOp fn, llvm::StringRef kernelName,
                         const char *what, const char *reason) {
  remark::passed(fn.getLoc(), remark::RemarkOpts::name(kRemarkName)
                                  .category(kRemarkCategory)
                                  .function(kernelName))
      << remark::add(what) << remark::reason(reason);
}

static void remarkKept(cir::FuncOp fn, llvm::StringRef kernelName,
                       const char *reason) {
  remark::missed(fn.getLoc(), remark::RemarkOpts::name(kRemarkName)
                                  .category(kRemarkCategory)
                                  .function(kernelName))
      << remark::add("kernel preserved") << remark::reason(reason);
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
  LDBG() << "container has " << table.size() << " kernel binding(s)";

  for (const auto &binding : table) {
    cir::FuncOp stub = binding.second.hostStub;
    bool localLinkage = cir::isLocalLinkage(stub.getLinkage());
    bool stubDead =
        localLinkage && !cir::hasUseOutsideSelf(stub, hostModule);
    if (stubDead) {
      LDBG() << "stub for '" << binding.first << "': dead, erasing";
      remarkErased(stub, binding.first, "removed dead host stub",
                   "internal linkage and no launch left");
      toErase.push_back(stub);
      ++numStubsErased;
    } else {
      if (!localLinkage) {
        LDBG() << "stub for '" << binding.first
               << "': kept, external linkage";
        remarkKept(stub, binding.first,
                   "external linkage: another TU may launch through this stub");
      }
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
      if (!cir::hasUseOutsideSelf(fn, deviceMod)) {
        LDBG() << "kernel '" << fn.getSymName() << "': stubless, erasing";
        remarkErased(fn, fn.getSymName(), "removed unlaunchable device kernel",
                     "no host stub and no device-side reference");
        toErase.push_back(fn);
        ++numKernelsErased;
      } else {
        LDBG() << "kernel '" << fn.getSymName()
               << "': kept, referenced by device code";
        remarkKept(fn, fn.getSymName(),
                   "referenced by device code (dynamic parallelism or "
                   "address-taken)");
        ++numKernelsPreserved;
      }
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
