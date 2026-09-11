//===- CIRBasicAliasAnalysis.h - Basic CIR Alias Analysis -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines CIRBasicAliasAnalysis, a CIR-specific alias analysis
// implementation based on pointer provenance and distinct allocation sites.
// Register with an mlir::AliasAnalysis aggregate via
// addAnalysisImplementation(), or use registerCIRAliasAnalyses() to add the
// full suite of CIR analyses at once.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_CIRBASICALIASANALYSIS_H
#define CLANG_CIR_DIALECT_ANALYSIS_CIRBASICALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace cir {

/// Basic CIR alias analysis based on pointer provenance and distinct allocation
/// sites. Conservative defaults (MayAlias / ModRef) are returned for cases
/// that are not yet handled.
class CIRBasicAliasAnalysis {
  enum class ObjectRelation {
    /// Provably different underlying allocations.
    Distinct,
    /// Same underlying allocation, no offset.
    Identical,
    /// Cannot determine the relationship.
    Unknown,
  };

public:
  CIRBasicAliasAnalysis() = default;
  CIRBasicAliasAnalysis(CIRBasicAliasAnalysis &&) = default;

  /// Return the aliasing behavior between two values.
  ///
  /// Returns MayAlias conservatively unless a more precise result can be
  /// determined from CIR-specific information (e.g. distinct alloca ops,
  /// pointer provenance, restrict attributes).
  mlir::AliasResult alias(mlir::Value lhs, mlir::Value rhs);

  /// Return the modify-reference behavior of `op` on `location`.
  ///
  /// Returns ModRef conservatively. CIR ops that carry explicit memory-effect
  /// attributes or that are known to be pure/read-only can be handled here.
  mlir::ModRefResult getModRef(mlir::Operation *op, mlir::Value location);

private:
  /// Attempt to find the underlying allocation source for `val` by walking
  /// through pointer arithmetic, casts, and other CIR ops. Returns `val` if
  /// no more specific source is found.
  mlir::Value getUnderlyingObject(mlir::Value val);

  /// Classify the relationship between \p lhs and \p rhs.  Returns one of:
  ///   Distinct      – provably different allocations
  ///   Identical     – same allocation, no offset
  ///   Unknown       – cannot determine
  ObjectRelation classifyObjects(mlir::Value lhs, mlir::Value rhs);
};

/// Walk past pointer-preserving operations — pointer casts, zero-strided
/// `cir.ptr_stride`, zero-index member and element accesses — to the
/// underlying object `val` refers to. Stops at block arguments and unknown
/// operations, returning the current value. Shared by CIRBasicAliasAnalysis
/// and by passes that need pointer provenance roots.
mlir::Value getUnderlyingObject(mlir::Value val);

/// Strip pointer-preserving casts only: bitcasts, address-space casts and
/// `array_to_ptrdecay`. Stops at every other operation.
mlir::Value stripPointerCasts(mlir::Value val);

} // namespace cir

#endif // CLANG_CIR_DIALECT_ANALYSIS_CIRBASICALIASANALYSIS_H
