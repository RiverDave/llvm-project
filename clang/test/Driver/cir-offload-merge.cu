// Driver pipeline for a CUDA compilation with ClangIR offload merge enabled
// (single device arch), compiled all the way to an object.

// REQUIRES: cir-support

// RUN: %clang -### -target x86_64-unknown-linux-gnu -x cuda -fclangir \
// RUN:   --cuda-gpu-arch=sm_80 -nocudainc -nocudalib \
// RUN:   --clangir-offload-merge -c %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=MERGE

// Host and device translation units are first lowered to serialized CIR, then
// combined into one cir.offload.container and split back. Each module resumes
// the backend from -x cir: the device module is lowered to PTX, assembled by
// ptxas, and the resulting binary is embedded into the host object. The capture
// variables track that every file is routed to the right consumer.

// Host TU -> CIR. The host compile must carry the CUDA aux-triple (i.e. present
// as a CUDA host compile) so it sees __global__, the runtime API, etc.; the
// merge path used to drop this because the host action sits below the split
// barrier and never received the host offload kind.
// MERGE: "-cc1"{{.*}} "-aux-triple" "nvptx64-nvidia-cuda"{{.*}} "-emit-cir"{{.*}} "-o" "[[HOST_CIR:[^"]+\.cir]]"
// Device TU -> CIR.
// MERGE: "-cc1"{{.*}} "-emit-cir"{{.*}} "-fcuda-is-device"{{.*}} "-o" "[[DEV_CIR:[^"]+\.cir]]"
// Both CIR modules are forwarded as inputs to -combine; output is the container.
// MERGE: "{{.*}}cir-offload-merge{{(\.exe)?}}" "-combine" "-targets=host-x86_64-unknown-linux-gnu,cuda-nvptx64-nvidia-cuda-unknown-sm_80" "-output=[[CONTAINER:[^"]+\.cir]]" "-input=[[HOST_CIR]]" "-input=[[DEV_CIR]]"
// The container is split back into one output per target.
// MERGE: "{{.*}}cir-offload-merge{{(\.exe)?}}" "-split" "-targets=host-x86_64-unknown-linux-gnu,cuda-nvptx64-nvidia-cuda-unknown-sm_80" "-input=[[CONTAINER]]" "-output=[[HOST_SPLIT:[^"]+\.cir]]" "-output=[[DEV_SPLIT:[^"]+\.cir]]"
// Device module: CIR -> PTX assembly.
// MERGE: "-cc1"{{.*}} "-S"{{.*}} "-fcuda-is-device"{{.*}} "-o" "[[DEV_PTX:[^"]+\.s]]"{{.*}} "-x" "cir" "[[DEV_SPLIT]]"
// PTX -> cubin.
// MERGE: "{{.*}}ptxas{{(\.exe)?}}"{{.*}} "--output-file" "[[DEV_CUBIN:[^"]+\.o]]"{{.*}} "[[DEV_PTX]]"
// cubin -> CUDA fatbinary.
// MERGE: "{{.*}}fatbinary{{(\.exe)?}}"{{.*}} "--create" "[[DEV_FATBIN:[^"]+\.fatbin]]"{{.*}} "--image3=kind=elf,sm=80,file=[[DEV_CUBIN]]"
// Host module: CIR -> object, embedding the device fatbinary.
// MERGE: "-cc1"{{.*}} "-emit-obj"{{.*}} "-fcuda-include-gpubinary" "[[DEV_FATBIN]]"{{.*}} "-x" "cir" "[[HOST_SPLIT]]"

// Without --clangir-offload-merge the normal pipeline runs: no merge/split.
// RUN: %clang -### -target x86_64-unknown-linux-gnu -x cuda -fclangir \
// RUN:   --cuda-gpu-arch=sm_80 -nocudainc -nocudalib -c %s 2>&1 \
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
