//===- KernelArgConstantPropagation.cpp - Constant launch arguments -------===//
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
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "cir-offload-kernel-arg-const-prop"

namespace mlir {
#define GEN_PASS_DEF_OFFLOADKERNELARGCONSTANTPROPAGATION
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

static constexpr llvm::StringLiteral kRemarkName = "KernelArgConstProp";
static constexpr llvm::StringLiteral kRemarkCategory = "cir-offload-const-prop";

static void remarkSkipped(cir::FuncOp stub, llvm::StringRef kernelName,
                          const char *reason, size_t numLaunchSites) {
  remark::missed(stub.getLoc(), remark::RemarkOpts::name(kRemarkName)
                                   .category(kRemarkCategory)
                                   .function(kernelName))
      << remark::add("kernel arguments not propagated")
      << remark::reason(reason)
      << remark::metric("launchSites", numLaunchSites);
}

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
  llvm::SmallPtrSet<mlir::Operation *, 4> rewrittenKernels;

  LDBG() << "container has " << table.size() << " kernel binding(s)";

  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;
    cir::FuncOp stub = binding.hostStub;
    llvm::ArrayRef<cir::LaunchSite> sites = binding.launchSites;

    LDBG() << "kernel '" << entry.first << "': " << sites.size()
           << " launch site(s), " << binding.deviceKernels.size()
           << " device kernel(s)";

    // A kernel launched nowhere in this TU says nothing about its arguments.
    if (sites.empty()) {
      LDBG() << "  skipped: no launch site in this TU";
      remarkSkipped(stub, entry.first, "no launch site in this translation unit",
                    sites.size());
      ++numKernelsSkipped;
      continue;
    }
    if (!table.allLaunchSitesVisible(entry.first)) {
      LDBG() << "  skipped: launch sites not all visible (stub linkage "
             << stringifyGlobalLinkageKind(stub.getLinkage()) << ")";
      remarkSkipped(stub, entry.first, "launch sites not all visible",
                    sites.size());
      ++numKernelsSkipped;
      continue;
    }

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

        LDBG() << "  '" << kernel.getSymName() << "': replaced arg " << i;

        remark::passed(kernel.getLoc(), remark::RemarkOpts::name(kRemarkName)
                                            .category(kRemarkCategory)
                                            .function(kernel.getSymName()))
            << remark::add("propagated constant launch argument")
            << remark::metric("argIndex", i)
            << remark::metric("kernelArgCount", kernel.getNumArguments())
            << remark::metric("launchSites", sites.size());
        ++numArgsPropagated;
        rewrittenKernels.insert(kernel);
      }
    }
  }

  numKernelsRewritten = rewrittenKernels.size();

  if (!changed)
    markAllAnalysesPreserved();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadKernelArgConstantPropagationPass() {
  return std::make_unique<OffloadKernelArgConstantPropagationPass>();
}
