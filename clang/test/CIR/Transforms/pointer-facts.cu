// End-to-end launch-derived noalias: real CIRGen output for host and device
// TUs, merged into a cir.offload.container. The merge tool runs the
// offload-container passes by default, so the combined container comes out with
// llvm.noalias on the device kernel's pointer parameters; -no-pointer-facts
// (what -fno-clangir-offload-merge-pointer-facts forwards) turns it off.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64-nvidia-cuda \
// RUN:   -target-sdk-version=9.2 -x cuda -I %S/../CodeGenCUDA/Inputs \
// RUN:   -emit-cir %s -o %t-host.cir
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -target-sdk-version=9.2 \
// RUN:   -fcuda-is-device -x cuda -I %S/../CodeGenCUDA/Inputs \
// RUN:   -emit-cir %s -o %t-device.cir
// RUN: cir-offload-merge -combine -input=%t-host.cir -input=%t-device.cir \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,cuda-nvptx64-nvidia-cuda--sm_80 \
// RUN:   -output=%t-enabled.cir
// RUN: FileCheck %s --check-prefix=ENABLED --input-file=%t-enabled.cir
// RUN: cir-offload-merge -combine -no-pointer-facts \
// RUN:   -input=%t-host.cir -input=%t-device.cir \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,cuda-nvptx64-nvidia-cuda--sm_80 \
// RUN:   -output=%t-disabled.cir
// RUN: FileCheck %s --check-prefix=DISABLED \
// RUN:   --implicit-check-not=llvm.noalias --input-file=%t-disabled.cir

#include "cuda.h"

extern "C" int cudaMalloc(void **ptr, unsigned long size);

// The kernel is internal, so its host stub is internal too: every launch site
// of it is visible in this container. A non-static __global__ would leave the
// stub externally visible and the fact is dropped ("launch sites not all
// visible").
static __global__ void add(float *dst, const float *src) { dst[0] = src[0]; }

void host() {
  float *a, *b;
  cudaMalloc((void **)&a, 16);
  cudaMalloc((void **)&b, 16);
  add<<<1, 32>>>(a, b);
}

// Both pointer parameters resolve to distinct cudaMalloc slots on the only
// launch site, and the kernel body only reads and writes through them.
// ENABLED-LABEL: cir.func{{.*}} @_ZL3addPfPKf(%arg0: !cir.ptr<!cir.float> {llvm.noalias
// ENABLED-SAME: %arg1: !cir.ptr<!cir.float> {llvm.noalias
// ENABLED-SAME: cc(ptx_kernel)

// Same module with the pass disabled: the signature keeps only llvm.noundef.
// DISABLED-LABEL: cir.func{{.*}} @_ZL3addPfPKf(
// DISABLED-SAME: %arg0: !cir.ptr<!cir.float> {llvm.noundef}
// DISABLED-SAME: cc(ptx_kernel)
