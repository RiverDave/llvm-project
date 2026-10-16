// -combine warns about kernel arguments whose layout differs between the host
// and a device module.

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgcn-amd-amdhsa \
// RUN:   -fhip-new-launch-api -x hip -I %S/../CodeGenCUDA/Inputs -emit-cir %s -o %t-host.cir
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -fcuda-is-device \
// RUN:   -x hip -I %S/../CodeGenCUDA/Inputs -emit-cir %s -o %t-gfx942.cir
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1030 -fcuda-is-device \
// RUN:   -x hip -I %S/../CodeGenCUDA/Inputs -emit-cir %s -o %t-gfx1030.cir
// RUN: cir-offload-merge -combine -input=%t-host.cir -input=%t-gfx942.cir \
// RUN:   -input=%t-gfx1030.cir \
// RUN:   -targets=host-x86_64-unknown-linux-gnu,hip-amdgcn-amd-amdhsa--gfx942,hip-amdgcn-amd-amdhsa--gfx1030 \
// RUN:   -output=%t-combined.cir 2>&1 | FileCheck %s --implicit-check-not=warning:

#include "cuda.h"

// A member that only the host compilation sees.
struct Params {
  float *out;
#ifndef __HIP_DEVICE_COMPILE__
  int hostOnly;
#endif
  float scale;
};
__global__ void params(Params p) { *p.out = p.scale; }
void launchParams(Params p) { params<<<1, 1>>>(p); }

// CHECK-DAG: warning: argument 0 of kernel '_Z6params6Params' has a different layout on the host and in {{.*}}gfx942: the host has '8:!cir.int<s, 32>' where the device has '8:!cir.float'
// CHECK-DAG: warning: argument 0 of kernel '_Z6params6Params' has a different layout on the host and in {{.*}}gfx1030: the host has '8:!cir.int<s, 32>' where the device has '8:!cir.float'

// A layout that depends on the device arch: only gfx1030 disagrees.
struct Scratch {
#if defined(__gfx1030__)
  int lanes[2];
#else
  int lanes[4];
#endif
  float *out;
};
__global__ void scratch(Scratch s) { *s.out = s.lanes[0]; }
void launchScratch(Scratch s) { scratch<<<1, 1>>>(s); }

// CHECK-DAG: warning: argument 0 of kernel '_Z7scratch7Scratch' has a different layout on the host and in {{.*}}gfx1030: the host has '8:!cir.int<s, 32>' where the device has '8:8 bytes'

// A lambda that captures `scale` only in the host compilation.
template <typename F> __global__ void run(F f) { f(); }
void launchCapture(float *out, float scale) {
  run<<<1, 1>>>([=] __device__ __host__() {
#ifdef __HIP_DEVICE_COMPILE__
    *out = 1.0f;
#else
    *out = scale;
#endif
  });
}

// CHECK-DAG: warning: argument 0 of kernel '{{.*}}launchCapture{{.*}}' has a different layout on the host and in {{.*}}gfx942: the host has '8:!cir.float' where the device has 'size 8'
// CHECK-DAG: warning: argument 0 of kernel '{{.*}}launchCapture{{.*}}' has a different layout on the host and in {{.*}}gfx1030: the host has '8:!cir.float' where the device has 'size 8'

// No warning below this line.

// The same layout, even where the two sides spell their padding differently.
struct Mixed {
  double d;
  int i;
};
__global__ void mixed(Mixed m, int *out) { *out = m.i; }
void launchMixed(Mixed m, int *out) { mixed<<<1, 1>>>(m, out); }

// A host-only lambda shifts the host's numbering of anonymous records, so the
// kernel lambda's record has another name on each side; its layout agrees.
template <typename F> int callOnHost(F f) { return f(); }
void launchShifted(float *out, double d, int i) {
  callOnHost([=] { return (int)d + i; });
  run<<<1, 1>>>([=] __device__() { *out = 2.0f; });
}
