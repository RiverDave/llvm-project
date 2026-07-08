//===- TestKernelBindings.cpp - Print the kernel binding table ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test pass that constructs and prints a KernelBindingTable, so the analysis
// can be exercised through cir-opt + FileCheck (mirrors MLIR's test-print-*
// analysis passes).
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct PrintKernelBindingsPass
    : public PassWrapper<PrintKernelBindingsPass,
                         OperationPass<cir::OffloadContainerOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintKernelBindingsPass)

  StringRef getArgument() const final { return "test-print-kernel-bindings"; }
  StringRef getDescription() const final {
    return "Print the host<->device kernel bindings of a "
           "cir.offload.container.";
  }
  void runOnOperation() override {
    getAnalysis<cir::KernelBindingTable>().print(llvm::errs());
  }
};
} // namespace

namespace cir {
namespace test {
void registerPrintKernelBindingsPass() {
  PassRegistration<PrintKernelBindingsPass>();
}
} // namespace test
} // namespace cir
