// End-to-end CUDA offload optimization pipeline. Real host and device CIRGen
// output is combined before checking the cross-boundary transformations.

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
// RUN: cir-offload-merge -combine -disable-cir-infer-launch-bounds \
// RUN:   -input=%t-host.cir -input=%t-device.cir \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,cuda-nvptx64-nvidia-cuda--sm_80 \
// RUN:   -output=%t-disabled.cir
// RUN: FileCheck %s --check-prefix=DISABLED \
// RUN:   --implicit-check-not=cir.nvvm.maxntid --input-file=%t-disabled.cir

// Split the combined module and resume the host with a device binary: the
// kernel is registered with the runtime.
// RUN: cir-offload-merge -split -input=%t-enabled.cir \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,cuda-nvptx64-nvidia-cuda--sm_80 \
// RUN:   -output=%t-split-host.cir -output=%t-split-device.cir
// RUN: echo -n "GPU binary would be here." > %t.fatbin
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cir -emit-llvm \
// RUN:   -foffload-include-binary %t.fatbin %t-split-host.cir -o - \
// RUN:   | FileCheck %s --check-prefix=REG

#include "cuda.h"

__device__ int result;

static __global__ void optimized(int value) { result = value; }

void host() { optimized<<<1, 32>>>(7); }

// ENABLED-LABEL: cir.func{{.*}} @_ZL9optimizedi(
// ENABLED-SAME: cir.nvvm.maxntid = "32"
// ENABLED: %[[VALUE:.*]] = cir.const #cir.int<7> : !s32i
// ENABLED: cir.store{{.*}} %[[VALUE]],

// DISABLED-LABEL: cir.func{{.*}} @_ZL9optimizedi(
// DISABLED: %[[VALUE:.*]] = cir.const #cir.int<7> : !s32i
// DISABLED: cir.store{{.*}} %[[VALUE]],

// REG: call i32 @__cudaRegisterFunction(ptr %{{.*}}, ptr @_ZL24__device_stub__optimizedi, ptr @{{.*}}_ZL9optimizedi,
