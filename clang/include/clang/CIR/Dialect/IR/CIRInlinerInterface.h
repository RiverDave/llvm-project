//===- CIRInlinerInterface.h - CIR dialect inliner interface ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The generic MLIR inliner only processes calls whose dialect provides a
// `DialectInlinerInterface`. This file attaches the (minimal) CIR one and
// registers the LLVM dialect's inliner-interface extension, which lives in
// this layer so every tool that inlines CIR (cir-opt, cir-offload-merge)
// shares exactly one definition of the legality rules.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRINLINERINTERFACE_H
#define CLANG_CIR_DIALECT_IR_CIRINLINERINTERFACE_H

#include "mlir/Support/LLVM.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace cir {

/// Attach the CIR `DialectInlinerInterface` to the dialect and register the
/// LLVM dialect's inliner-interface extension. Call this before creating an
/// MLIRContext that may run the generic inliner over CIR.
void registerInlinerInterface(mlir::DialectRegistry &registry);

} // namespace cir

#endif // CLANG_CIR_DIALECT_IR_CIRINLINERINTERFACE_H
