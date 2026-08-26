//===- KernelArgConstantPropagation.cpp - Constant launch arguments -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"

namespace mlir {
#define GEN_PASS_DEF_OFFLOADKERNELARGCONSTANTPROPAGATION
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

// The constant every site passes for argument `argIdx`, or null if they
// disagree or any of them passes a non-constant. Agreement is a property of the
// set of sites, so it lives here rather than on a single site.
static mlir::TypedAttr
commonConstantArg(llvm::ArrayRef<cir::LaunchSite> sites, unsigned argIdx) {
  mlir::TypedAttr common;
  for (const cir::LaunchSite &site : sites) {
    mlir::TypedAttr value = site.getConstArg(argIdx);
    if (!value)
      return {};
    if (!common)
      common = value;
    else if (common != value)
      return {};
  }
  return common;
}

struct OffloadKernelArgConstantPropagationPass
    : public impl::OffloadKernelArgConstantPropagationBase<
          OffloadKernelArgConstantPropagationPass> {
  void runOnOperation() override;
};

void OffloadKernelArgConstantPropagationPass::runOnOperation() {
  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  bool changed = false;

  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;
    cir::FuncOp stub = binding.hostStub;
    llvm::ArrayRef<cir::LaunchSite> sites = binding.launchSites;

    // A kernel launched nowhere in this TU says nothing about its arguments.
    if (sites.empty())
      continue;
    if (!table.allLaunchSitesVisible(entry.first))
      continue;

    for (unsigned i = 0, e = stub.getNumArguments(); i != e; ++i) {
      mlir::TypedAttr value = commonConstantArg(sites, i);
      if (!value)
        continue;

      // The stub argument and the kernel argument at the same index are the
      // same kernel parameter; nothing has rewritten either signature yet.
      for (cir::FuncOp kernel : binding.deviceKernels) {
        if (kernel.isDeclaration() || i >= kernel.getNumArguments())
          continue;
        mlir::BlockArgument arg = kernel.getArgument(i);
        if (arg.use_empty() || arg.getType() != value.getType())
          continue;

        mlir::OpBuilder builder(&kernel.getBody().front(),
                                kernel.getBody().front().begin());
        auto constant =
            cir::ConstantOp::create(builder, kernel.getLoc(), value);
        arg.replaceAllUsesWith(constant);
        changed = true;
      }
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadKernelArgConstantPropagationPass() {
  return std::make_unique<OffloadKernelArgConstantPropagationPass>();
}
