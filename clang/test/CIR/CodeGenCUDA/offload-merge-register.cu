// Verify that CUDA host LLVM IR resumed from the ClangIR offload-merge pipeline
// emits the registration code for the device image and kernels.

// REQUIRES: cir-support

// RUN: %clang -target x86_64-unknown-linux-gnu -x cuda -fclangir \
// RUN:   --cuda-gpu-arch=sm_80 \
// RUN:   --cuda-path=%S/../../Driver/Inputs/CUDA_102/usr/local/cuda \
// RUN:   -nocudainc -nocudalib -Wno-unknown-cuda-version \
// RUN:   --clangir-offload-merge -S -emit-llvm %s -o - \
// RUN: | FileCheck %s

#include "Inputs/cuda.h"

__global__ void kernel() {}

// CHECK: @__cuda_fatbin_str = private constant
// CHECK-SAME: section ".nv_fatbin"
// CHECK: @__cuda_fatbin_wrapper = private constant
// CHECK-SAME: section ".nvFatBinSegment"
// CHECK: @__cuda_gpubin_handle = internal global ptr null
// CHECK: @llvm.global_ctors = appending global
// CHECK-SAME: @__cuda_module_ctor
// CHECK: define internal void @__cuda_register_globals
// CHECK: call{{.*}}@__cudaRegisterFunction
// CHECK: define internal void @__cuda_module_ctor
// CHECK: call{{.*}}@__cudaRegisterFatBinary
// CHECK: call void @__cuda_register_globals
