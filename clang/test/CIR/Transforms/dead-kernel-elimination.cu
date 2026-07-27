// End-to-end dead kernel elimination: real CIRGen output for host and device
// TUs, merged into a cir.offload.container, then run through the DKE pass.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64-nvidia-cuda \
// RUN:   -target-sdk-version=9.2 -x cuda -I %S/../../CodeGenCUDA/Inputs \
// RUN:   -emit-cir %s -o %t-host.cir
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-sdk-version=9.2 -fcuda-is-device \
// RUN:   -x cuda -I %S/../../CodeGenCUDA/Inputs -emit-cir %s -o %t-dev.cir
// RUN: cir-offload-merge -combine -input=%t-host.cir -input=%t-dev.cir \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,cuda-nvptx64-nvidia-cuda--sm_80 \
// RUN:   -output=%t-combined.cir
// RUN: cir-opt %t-combined.cir \
// RUN:   -pass-pipeline="builtin.module(cir.offload.container(cir-offload-dead-kernel-elimination))" \
// RUN:   | FileCheck %s --implicit-check-not="dead_static" --implicit-check-not="dead_anon"

// A launched kernel and an externally-linked kernel (which another TU may launch
// through its external host stub) survive; the static and anonymous-namespace
// kernels that are never launched here are removed device-side.
// CHECK-DAG: cir.func{{.*}} @_Z4livev() cc(ptx_kernel)
// CHECK-DAG: cir.func{{.*}} @_Z10ext_unusedv() cc(ptx_kernel)

#include "cuda.h"

__global__ void live() {}
static __global__ void dead_static() {}
namespace {
__global__ void dead_anon() {}
} // namespace
__global__ void ext_unused() {}

void host() { live<<<1, 1>>>(); }
