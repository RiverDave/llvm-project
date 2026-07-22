// End-to-end dead kernel elimination for HIP: real CIRGen output for host and
// device TUs, merged into a cir.offload.container by the merge tool, which runs
// the offload-container passes (DKE) by default. Mirrors
// dead-kernel-elimination.cu; device kernels here carry cc(amdgpu_kernel) and
// the host emits a launch-handle global per kernel.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgcn-amd-amdhsa \
// RUN:   -target-sdk-version=9.2 -x hip -I %S/../../CodeGenCUDA/Inputs \
// RUN:   -emit-cir %s -o %t-host.cir
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 -target-sdk-version=9.2 \
// RUN:   -fcuda-is-device -x hip -I %S/../../CodeGenCUDA/Inputs -emit-cir %s -o %t-dev.cir
// RUN: cir-offload-merge -combine -input=%t-host.cir -input=%t-dev.cir \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa--gfx900 \
// RUN:   -output=%t-combined.cir
// RUN: FileCheck %s --input-file=%t-combined.cir \
// RUN:   --implicit-check-not="dead_static" --implicit-check-not="dead_anon"

// A launched kernel and an externally-linked kernel survive; the static and
// anonymous-namespace kernels that are never launched here are removed.
// CHECK-DAG: cir.func{{.*}} @_Z4livev() cc(amdgpu_kernel)
// CHECK-DAG: cir.func{{.*}} @_Z10ext_unusedv() cc(amdgpu_kernel)

#include "cuda.h"

__global__ void live() {}
static __global__ void dead_static() {}
namespace {
__global__ void dead_anon() {}
} // namespace
__global__ void ext_unused() {}

void host() { live<<<1, 1>>>(); }
