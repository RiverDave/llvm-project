// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// CIRGen precomputes the mangled C++20 named-module initializer function
// name and stores it as a module-level attribute so LoweringPrepare can
// consume it without a live ASTContext after split-compilation.

export module A;

int x = 5;

// CIR: module
// CIR-SAME: cir.cxx_module_init_fn_name = "_ZGIW1A"
