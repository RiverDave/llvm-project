// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s \
// RUN:   -o %t.cir 2>&1 | FileCheck %s

// The cir-materialize-ast-facts pass runs inside the default CIR pipeline.
// For static-local variables that carry an ASTVarDecl reference, its effect
// is that the AST-backed #cir.var.decl placeholder on cir.global is
// replaced by a concrete #cir.static_local_info attribute with the cached
// VarDecl facts. The IR dumped immediately before LoweringPrepare must
// therefore use the cached form and no longer reference #cir.var.decl.

struct HasCtor {
  HasCtor();
  int x;
};

int referenced_inside() {
  static HasCtor static_local;
  return static_local.x;
}

// CHECK-NOT: #cir.var.decl
// CHECK: cir.global
// CHECK-SAME: @_ZZ17referenced_insidevE12static_local
// CHECK-SAME: #cir.static_local_info<{{.*}}is_local_var_decl = true
