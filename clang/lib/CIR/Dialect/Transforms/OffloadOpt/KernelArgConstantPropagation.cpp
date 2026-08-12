//===- KernelArgConstantPropagation.cpp - Constant launch arguments -------===//
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

namespace mlir {
#define GEN_PASS_DEF_OFFLOADKERNELARGCONSTANTPROPAGATION
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

// The launch sites recorded for a stub are all of them when no other TU can
// call it and its address never escapes. An escaped stub may be called
// indirectly with arguments this pass cannot see. The stub references itself,
// since it hands its own address to cudaLaunchKernel, so only uses outside its
// own body are considered.
static bool allLaunchSitesVisible(cir::FuncOp stub, mlir::Operation *scope) {
  if (!cir::isLocalLinkage(stub.getLinkage()))
    return false;
  auto uses = mlir::SymbolTable::getSymbolUses(stub, scope);
  if (!uses)
    return false;
  return llvm::all_of(*uses, [&](const mlir::SymbolTable::SymbolUse &use) {
    return stub->isProperAncestor(use.getUser()) ||
           mlir::isa<cir::CallOp>(use.getUser());
  });
}

// The constant every site passes for argument `argIdx`, or null if they
// disagree or any of them passes a non-constant.
static mlir::TypedAttr
commonConstantArg(llvm::ArrayRef<cir::CallOp> sites, unsigned argIdx) {
  mlir::TypedAttr common;
  for (cir::CallOp site : sites) {
    mlir::OperandRange args = site.getArgOperands();
    if (argIdx >= args.size())
      return {};
    auto cst = args[argIdx].getDefiningOp<cir::ConstantOp>();
    if (!cst)
      return {};
    if (!common)
      common = cst.getValue();
    else if (common != cst.getValue())
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
  cir::OffloadContainerOp container = getOperation();
  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  mlir::ModuleOp hostModule = container.getHostModule();
  bool changed = false;

  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;
    cir::FuncOp stub = binding.hostStub;
    llvm::ArrayRef<cir::CallOp> sites = binding.launchSites;

    // A kernel launched nowhere in this TU says nothing about its arguments.
    if (sites.empty())
      continue;
    if (!allLaunchSitesVisible(stub, hostModule))
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
