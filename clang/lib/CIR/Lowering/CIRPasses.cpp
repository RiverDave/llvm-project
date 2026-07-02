//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements machinery for any CIR <-> CIR passes used by clang.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace cir {
mlir::LogicalResult
runCIRToCIRPasses(mlir::ModuleOp theModule, mlir::MLIRContext &mlirContext,
                  clang::ASTContext &astContext, cir::LowerModule &lowerModule,
                  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs,
                  bool enableVerifier, bool enableIdiomRecognizer,
                  bool enableCIRSimplify) {

  llvm::TimeTraceScope scope("CIR To CIR Passes");

  mlir::PassManager pm(&mlirContext);
  pm.addPass(mlir::createCIRCanonicalizePass());

  // Snapshot AST-derived facts into CIR attributes while the ASTContext is
  // still live, so later passes (notably LoweringPrepare) can run on
  // serialized CIR without an AST.
  pm.addPass(mlir::createMaterializeASTFactsPass());

  if (enableCIRSimplify)
    pm.addPass(mlir::createCIRSimplifyPass());

  if (enableIdiomRecognizer)
    pm.addPass(mlir::createIdiomRecognizerPass(&astContext));

  pm.addPass(mlir::createTargetLoweringPass());
  pm.addPass(mlir::createCXXABILoweringPass());
  pm.addPass(mlir::createLoweringPreparePass(&lowerModule, std::move(vfs)));

  pm.enableVerifier(enableVerifier);
  (void)mlir::applyPassManagerCLOptions(pm);
  return pm.run(theModule);
}

} // namespace cir

namespace mlir {

void populateCIRPreLoweringPasses(OpPassManager &pm) {
  pm.addPass(createHoistAllocasPass());
  pm.addPass(createCIRFlattenCFGPass());
  pm.addPass(createCIREHABILoweringPass());
  pm.addPass(createGotoSolverPass());
}

} // namespace mlir
