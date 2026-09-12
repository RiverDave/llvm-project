//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Similar to MLIR/LLVM's "opt" tools but also deals with analysis and custom
// arguments. TODO: this is basically a copy from MlirOptMain.cpp, but capable
// of module emission as specified by the user.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/InitAllDialects.h"
#include "clang/CIR/Passes.h"

#ifdef CLANG_INCLUDE_TESTS
namespace cir::test {
void registerTestCIRAliasAnalysisPass();
} // namespace cir::test
#endif

struct CIRToLLVMPipelineOptions
    : public mlir::PassPipelineOptions<CIRToLLVMPipelineOptions> {
  Option<bool> enableOpenMP{
      *this, "enable-openmp",
      llvm::cl::desc("Add OpenMP-specific CIR-to-LLVM lowering passes"),
      llvm::cl::init(false)};
};

namespace cir {
namespace test {
void registerPrintKernelBindingsPass();
} // namespace test
} // namespace cir

namespace {
/// Minimal DialectInlinerInterface for CIR, sufficient to exercise the generic
/// MLIR inliner on ClangIR. Everything is legal to inline; the terminator
/// handler rewrites cir.return into a cir.br to the continuation block.
struct CIRInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    // Launch markers stay calls: the offload pipeline rebuilds its kernel
    // binding table from `cu.kernel_name` stub calls, so inlining them away
    // would erase the launch sites.
    if (call && call->hasAttr("cu.kernel_name"))
      return false;
    // Only single-block bodies that end in `cir.return` are supported: the
    // generic inliner models the call results through that terminator, and
    // multi-block bodies (or bodies whose block ends in another op, e.g. a
    // `cir.scope` whose paths all return) need result plumbing CIR does not
    // have yet.
    auto func = mlir::dyn_cast<cir::FuncOp>(callable);
    if (!func || func.isDeclaration())
      return false;
    mlir::Region &body = func.getBody();
    if (!body.hasOneBlock())
      return false;
    mlir::Block &block = body.front();
    if (block.empty())
      return false;
    return mlir::isa<cir::ReturnOp>(&block.back());
  }
  bool isLegalToInline(mlir::Region *dest, mlir::Region *src, bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(mlir::Operation *op, mlir::Region *dest, bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final {
    return true;
  }
  void handleTerminator(mlir::Operation *op, mlir::Block *newDest) const final {
    auto ret = mlir::dyn_cast<cir::ReturnOp>(op);
    assert(ret && "expected cir.return");
    mlir::OpBuilder builder(op);
    cir::BrOp::create(builder, op->getLoc(), newDest, ret.getInput());
    op->erase();
  }
  void handleTerminator(mlir::Operation *op,
                        mlir::ValueRange valuesToReplace) const final {
    auto ret = mlir::dyn_cast<cir::ReturnOp>(op);
    assert(ret && "expected cir.return");
    for (auto [oldVal, newVal] : llvm::zip(valuesToReplace, ret.getInput()))
      oldVal.replaceAllUsesWith(newVal);
  }
};
} // namespace

int main(int argc, char **argv) {
  // TODO: register needed MLIR passes for CIR?
  mlir::DialectRegistry registry;
  cir::registerAllDialects(registry);

#ifdef CLANG_INCLUDE_TESTS
  cir::test::registerTestCIRAliasAnalysisPass();
#endif
  registry.insert<mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect>();

  // The LLVM dialect promises DialectInlinerInterface; the implementation is
  // shipped as a dialect extension that must be registered explicitly.
  mlir::LLVM::registerInlinerInterface(registry);
  // CIR ships no DialectInlinerInterface yet; attach a minimal one so the
  // generic MLIR inliner can process CIR (experiment hook).
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, cir::CIRDialect *dialect) {
        dialect->addInterfaces<CIRInlinerInterface>();
      });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIRCanonicalizePass();
  });
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIRSimplifyPass();
  });

  mlir::PassPipelineRegistration<CIRToLLVMPipelineOptions> pipeline(
      "cir-to-llvm", "",
      [](mlir::OpPassManager &pm, const CIRToLLVMPipelineOptions &options) {
        cir::direct::populateCIRToLLVMPasses(pm, options.enableOpenMP);
      });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIRFlattenCFGPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCIREHABILoweringPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createHoistAllocasPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCUDARegisterModulePass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createGotoSolverPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCXXABILoweringPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createIdiomRecognizerPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createCallConvLoweringPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createMaterializeASTFactsPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createLoweringPreparePass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createOffloadDeadKernelEliminationPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createOffloadKernelArgConstantPropagationPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::createOffloadLaunchNoaliasPass();
  });

  cir::test::registerPrintKernelBindingsPass();

  mlir::omp::registerOpenMPPasses();
  mlir::registerTransformsPasses();
  mlir::registerInlinerPass();

  return mlir::asMainReturnCode(MlirOptMain(
      argc, argv, "Clang IR analysis and optimization tool\n", registry));
}
