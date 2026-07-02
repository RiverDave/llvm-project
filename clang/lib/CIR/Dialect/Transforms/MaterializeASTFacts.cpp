//====- MaterializeASTFacts.cpp - Snapshot AST facts into CIR attrs ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "clang/AST/Decl.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Interfaces/ASTAttrInterfaces.h"
#include "llvm/Support/TimeProfiler.h"

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_MATERIALIZEASTFACTS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct MaterializeASTFactsPass
    : public impl::MaterializeASTFactsBase<MaterializeASTFactsPass> {
  MaterializeASTFactsPass() = default;
  void runOnOperation() override;
};

// Rebuild the `$ast` attribute of a cir.global as a StaticLocalInfoAttr when
// the current attribute is an AST-backed ASTVarDeclAttr that still holds a
// live VarDecl pointer. StaticLocalInfoAttr instances are left alone.
//
// For now the pass only materializes static-local guarded globals, the
// subset whose facts LoweringPrepare actually queries through the
// ASTVarDeclInterface. Other AST-backed attributes that only exist as IR
// metadata are left untouched so textual IR stays stable.
static void materializeGlobal(cir::GlobalOp globalOp) {
  if (!globalOp.getStaticLocalGuard().has_value())
    return;

  std::optional<cir::ASTVarDeclInterface> iface = globalOp.getAst();
  if (!iface)
    return;

  // Already materialized.
  if (mlir::isa<cir::StaticLocalInfoAttr>(*iface))
    return;

  auto astBacked = mlir::dyn_cast<cir::ASTVarDeclAttr>(*iface);
  if (!astBacked)
    return;

  const clang::VarDecl *vd = astBacked.getAst();
  if (!vd)
    return;

  mlir::MLIRContext *ctx = globalOp.getContext();
  auto snapshot = cir::StaticLocalInfoAttr::get(
      ctx,
      /*is_local_var_decl=*/vd->isLocalVarDecl(),
      /*tls=*/static_cast<uint32_t>(vd->getTLSKind()),
      /*is_inline=*/vd->isInline(),
      /*tsk=*/static_cast<uint32_t>(vd->getTemplateSpecializationKind()));
  globalOp.setAstAttr(snapshot);
}

void MaterializeASTFactsPass::runOnOperation() {
  llvm::TimeTraceScope scope("Materialize AST Facts");
  getOperation()->walk(&materializeGlobal);
}

} // namespace

std::unique_ptr<Pass> mlir::createMaterializeASTFactsPass() {
  return std::make_unique<MaterializeASTFactsPass>();
}
