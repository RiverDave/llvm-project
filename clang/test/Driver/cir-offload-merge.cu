// Driver pipeline for a CUDA compilation with ClangIR offload merge enabled
// (single device arch).

// REQUIRES: cir-support

// RUN: %clang -### -target x86_64-unknown-linux-gnu -x cuda -fclangir \
// RUN:   --cuda-gpu-arch=sm_80 -nocudainc -nocudalib \
// RUN:   --clangir-offload-merge -S -emit-llvm %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=MERGE

// Host and device translation units are first lowered to serialized CIR, then
// combined into one cir.offload.container and split back, and finally each
// module resumes the backend from -x cir.
// MERGE: "-cc1" {{.*}}"-fclangir" {{.*}}"-emit-cir"
// MERGE: "-cc1" {{.*}}"-fclangir" {{.*}}"-emit-cir" {{.*}}"-fcuda-is-device"
// MERGE: "{{.*}}cir-offload-merge{{(\.exe)?}}" "-combine" "-targets=host-x86_64-unknown-linux-gnu,cuda-nvptx64-nvidia-cuda-unknown-sm_80"
// MERGE: "{{.*}}cir-offload-merge{{(\.exe)?}}" "-split" "-targets=host-x86_64-unknown-linux-gnu,cuda-nvptx64-nvidia-cuda-unknown-sm_80"
// MERGE: "-cc1" {{.*}}"-fclangir" {{.*}}"-emit-llvm-bc" {{.*}}"-fcuda-is-device" {{.*}}"-x" "cir"
// MERGE: "-cc1" {{.*}}"-fclangir" {{.*}}"-emit-llvm" {{.*}}"-fcuda-include-gpubinary" {{.*}}"-x" "cir"

// Without --clangir-offload-merge the normal pipeline runs: no merge/split.
// RUN: %clang -### -target x86_64-unknown-linux-gnu -x cuda -fclangir \
// RUN:   --cuda-gpu-arch=sm_80 -nocudainc -nocudalib -S -emit-llvm %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=NO-MERGE
// NO-MERGE-NOT: "-combine"
// NO-MERGE-NOT: "-split"
// NO-MERGE-NOT: "-x" "cir"

// -emit-cir stops at CIR; there is nothing to merge/split.
// RUN: %clang -### -target x86_64-unknown-linux-gnu -x cuda -fclangir \
// RUN:   --cuda-gpu-arch=sm_80 -nocudainc -nocudalib --clangir-offload-merge \
// RUN:   -emit-cir %s 2>&1 | FileCheck %s --check-prefix=EMIT-CIR
// EMIT-CIR-NOT: "-combine"
// EMIT-CIR-NOT: "-split"

__global__ void kernel() {}
void host() { kernel<<<1, 1>>>(); }
