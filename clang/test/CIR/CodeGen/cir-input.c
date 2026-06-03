// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cir %t.cir -emit-llvm -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cir %t.cir -emit-llvm-bc -o %t.bc
// RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=BC
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cir %t.cir -S -o - | FileCheck %s --check-prefix=ASM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cir %t.cir -emit-obj -o %t.o
// RUN: llvm-objdump -t %t.o | FileCheck %s --check-prefix=OBJ
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -x cir %t.cir -fsyntax-only 2>&1 | FileCheck %s --check-prefix=NO-AST
// RUN: echo "not cir" > %t.invalid.cir
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -x cir %t.invalid.cir -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=INVALID

// REQUIRES: x86-registered-target

int x = 1;

int f(void) {
  return x;
}

// LLVM: @x = {{(dso_local )?}}global i32 1
// LLVM: define{{.*}} i32 @f()

// BC: @x = {{(dso_local )?}}global i32 1
// BC: define{{.*}} i32 @f()

// ASM: x:
// ASM: .long   1

// OBJ: .data
// OBJ-SAME: x

// NO-AST: cannot apply AST actions to CIR file

// INVALID: failed to parse CIR input
