# CIR Offload Merge: Host-Device ClangIR for CUDA and HIP

This branch (`gsoc/combine-cir`) is the final work product of the Google
Summer of Code 2026 project
[Host-Device CIR Combine Pipeline](https://discourse.llvm.org/t/gsoc-2026-host-device-cir-combine-pipeline-project-idea-discussion/89623).

- **Student:** David Rivera ([@RiverDave](https://github.com/RiverDave))
- **Mentors:** Konstantinos Parasyris, Joseph Huber

The rest of this repository is upstream LLVM. The project work lives under
`clang/lib/CIR`, `clang/tools/cir-offload-merge` and the CIR driver code. See
[What's on this branch](#whats-on-this-branch).

## Summary

Clang compiles CUDA and HIP host and device code separately. They only meet
again when the device binary is embedded in the host object, after the device
side is already machine code, so nothing can reason across the boundary.

This project merges the host and device ClangIR (CIR) modules of a translation
unit into one container module. Passes running on the container can see both
the kernel launch sites and the kernel bodies. The container is then split
back into its host and device modules, and each continues through the normal
pipeline.

```
            host cc1 ──► host.cir ──┐
                                    ├─► combine ─► cross-boundary passes ─► split ─┬─► host backend
 device cc1 (per arch) ─► dev.cir ──┘                                              └─► device backend(s)
```

## Usage

With the driver, for CUDA or HIP, single or multi-arch:

```
clang -x cuda -fclangir --clangir-offload-merge --cuda-gpu-arch=sm_80 -c kernel.cu
clang -x hip  -fclangir --clangir-offload-merge --offload-arch=gfx942 -c kernel.hip
```

Or run each step by hand:

```
cir-offload-merge -combine \
  -input=host.cir -input=device-sm80.cir \
  -targets=host-x86_64-unknown-linux-gnu,cuda-nvptx64-nvidia-cuda--sm_80 \
  -output=combined.cir
cir-offload-merge -split \
  -input=combined.cir \
  -targets=cuda-nvptx64-nvidia-cuda--sm_80,host-x86_64-unknown-linux-gnu \
  -output=device-sm80.cir -output=host.cir
```

## What's on this branch

Fork PRs are on [RiverDave/llvm-project](https://github.com/RiverDave/llvm-project/pulls).

**Container and tooling**
- The `cir.offload.container` module, which holds one host module and one
  device module per GPU arch (#2, #4; upstreamed as the attribute in
  [llvm#206576](https://github.com/llvm/llvm-project/pull/206576)).
- The `cir-offload-merge` tool, with `-combine` and `-split` (#3, #4, #31).
- `.cir` inputs accepted by the driver and by cc1 (#5).
- Driver job actions for merge and split, with CUDA and HIP, single and
  multi-arch (#6, #7, #9, #10, #32).
- CUDA registration as its own pass, so a host resumed from `.cir` still
  registers its kernels (#8).

**Analysis**
- `KernelBindingTable` maps each host stub to its device kernel and records
  every launch site, including its grid and block geometry (#11, #15, #17,
  #22, #23).

**Cross-boundary passes**
- Dead kernel elimination: removes device kernels the host never launches
  (#12, #13, #20).
- Kernel argument constant propagation: when every launch passes the same
  constant, folds it into the kernel (#16, #19).
- Launch bounds propagation: derives `__launch_bounds__` from launch sites
  with constant geometry (#21).
- `mem2reg` and `sccp` scheduled on the nested modules (#14, #25).
- Launch-derived `noalias`: marks kernel pointer parameters `noalias` when
  every launch passes distinct `cudaMalloc` buffers and the kernel touches
  memory only through its parameters (#30).
- Kernel cloning helpers (#28).

## Upstream contributions

CIR CUDA/HIP patches merged into llvm/llvm-project during the project:

- Offload container attribute: [#206576](https://github.com/llvm/llvm-project/pull/206576)
- CUDA/HIP codegen:
  [#195381](https://github.com/llvm/llvm-project/pull/195381),
  [#195382](https://github.com/llvm/llvm-project/pull/195382),
  [#195391](https://github.com/llvm/llvm-project/pull/195391),
  [#199270](https://github.com/llvm/llvm-project/pull/199270),
  [#204663](https://github.com/llvm/llvm-project/pull/204663),
  [#215314](https://github.com/llvm/llvm-project/pull/215314),
  [#231390](https://github.com/llvm/llvm-project/pull/231390)
- NVPTX builtins and attributes:
  [#195214](https://github.com/llvm/llvm-project/pull/195214),
  [#195217](https://github.com/llvm/llvm-project/pull/195217),
  [#214708](https://github.com/llvm/llvm-project/pull/214708),
  [#216447](https://github.com/llvm/llvm-project/pull/216447),
  [#221255](https://github.com/llvm/llvm-project/pull/221255),
  [#221261](https://github.com/llvm/llvm-project/pull/221261),
  [#221262](https://github.com/llvm/llvm-project/pull/221262)
- Address spaces:
  [#196868](https://github.com/llvm/llvm-project/pull/196868),
  [#226649](https://github.com/llvm/llvm-project/pull/226649),
  [#226650](https://github.com/llvm/llvm-project/pull/226650)
- Aliasing and FP:
  [#221817](https://github.com/llvm/llvm-project/pull/221817),
  [#221818](https://github.com/llvm/llvm-project/pull/221818),
  [#226334](https://github.com/llvm/llvm-project/pull/226334)
- Tooling:
  [#196245](https://github.com/llvm/llvm-project/pull/196245),
  [#200266](https://github.com/llvm/llvm-project/pull/200266),
  [#229565](https://github.com/llvm/llvm-project/pull/229565),
  [#229575](https://github.com/llvm/llvm-project/pull/229575)

Still open upstream:
[#205009](https://github.com/llvm/llvm-project/pull/205009) (CUDA RDC
linkage) and [#215780](https://github.com/llvm/llvm-project/pull/215780)
(`PromotableRegionOpInterface`, carried here as fork commits).

## Results

TODO: PolyBench numbers on A10 (sm_86), with the commit they were measured at.

Talk: *CUDA and HIP in ClangIR*, LLVM Offload Workshop, Santa Clara,
October 2026. TODO: slides link.

## Limitations

- Non-RDC only. Every launch site and kernel must be in one translation unit.
- A kernel launched through a stub address that escapes (stored in a global,
  passed through a function pointer) is left alone.
- Kernel fusion, listed in the proposal, was not started.

## Building and testing

```
cmake -G Ninja -S llvm -B build -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DCLANG_ENABLE_CIR=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON
ninja -C build clang cir-opt cir-offload-merge
build/bin/llvm-lit -v clang/test/CIR/Transforms clang/test/CIR/Tools \
  clang/test/Driver/cir-offload-merge.cu clang/test/Driver/cir-offload-merge.hip
```

## Acknowledgements

Thanks to Konstantinos Parasyris and Joseph Huber for mentoring. Thanks also to
Steffen Larsen (AMD), who contributed #27, #28, #31 and #32, and to the ClangIR
maintainers for reviews upstream.

---

## Upstream LLVM README

[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/llvm/llvm-project/badge)](https://securityscorecards.dev/viewer/?uri=github.com/llvm/llvm-project)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8273/badge)](https://www.bestpractices.dev/projects/8273)
[![libc++](https://github.com/llvm/llvm-project/actions/workflows/libcxx-pr-conformance-tests.yaml/badge.svg?branch=main&event=schedule)](https://github.com/llvm/llvm-project/actions/workflows/libcxx-pr-conformance-tests.yaml?query=event%3Aschedule)

Welcome to the LLVM project!

This repository contains the source code for LLVM, a toolkit for the
construction of highly optimized compilers, optimizers, and run-time
environments.

The LLVM project has multiple components. The core of the project is
itself called "LLVM". This contains all of the tools, libraries, and header
files needed to process intermediate representations and convert them into
object files. Tools include an assembler, disassembler, bitcode analyzer, and
bitcode optimizer.

C-like languages use the [Clang](https://clang.llvm.org/) frontend. This
component compiles C, C++, Objective-C, and Objective-C++ code into LLVM bitcode
-- and from there into object files, using LLVM.

Other components include:
the [libc++ C++ standard library](https://libcxx.llvm.org),
the [LLD linker](https://lld.llvm.org), and more.

## Getting the Source Code and Building LLVM

Consult the
[Getting Started with LLVM](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
page for information on building and running LLVM.

For information on how to contribute to the LLVM project, please take a look at
the [Contributing to LLVM](https://llvm.org/docs/Contributing.html) guide.

## Getting in touch

Join the [LLVM Discourse forums](https://discourse.llvm.org/), [Discord
chat](https://discord.gg/xS7Z362),
[LLVM Office Hours](https://llvm.org/docs/GettingInvolved.html#office-hours) or
[Regular sync-ups](https://llvm.org/docs/GettingInvolved.html#online-sync-ups).

The LLVM project has adopted a [code of conduct](https://llvm.org/docs/CodeOfConduct.html) for
participants to all modes of communication within the project.
