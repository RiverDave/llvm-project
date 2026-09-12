// The first CIRGen stamps the resolved FP-contraction model (Fast for CUDA,
// FastHonorPragmas for HIP) as a module-level cir.fp_contract_mode attribute.
// A later resume cc1 (-x cir) reads it back to reconstruct the backend
// AllowFPOpFusion policy. This pins the emission side of the round-trip.
//
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda \
// RUN:   -fcuda-is-device -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=ATTR

#include "Inputs/cuda.h"

// ATTR: module {{.*}} attributes {{.*}}cir.fp_contract_mode = "fast"
// ATTR: cir.fmul
// ATTR: cir.fadd
extern "C" __global__ void k(float *o, float a, float b, float c) {
  o[0] = a * b + c;
}