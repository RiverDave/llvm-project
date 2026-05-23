# GSoC 2026 — Unified Host/Device CIR for CUDA & HIP

This is a development fork of [llvm/llvm-project](https://github.com/llvm/llvm-project)
hosting the GSoC 2026 project **"Combine/Split CIR for CUDA & HIP offloading"**.

- **Mentors:** Konstantinos Parasyris, Joseph Huber
- **Contributor:** David Rivera ([@RiverDave](https://github.com/RiverDave))

## Background

LLVM's offload pipeline keeps host and device code separate until it's too late for
cross-boundary optimization. CIR Combine fixes this with a merge-optimize-split stage:
both modules are lowered to CIR, merged into a single heterogeneous translation unit,
optimized together (constant propagation across launch sites, dead kernel elimination,
launch dimensionality inference), then split back into their respective backend pipelines.


The intended flow looks like (We depict tool invocations in this context):

```mermaid
flowchart LR
    SRC[".cu / .cpp"]

    PRE_H["cc1 (host)<br/>emit pre-lowering CIR"]
    PRE_70["cc1 (sm_70)<br/>emit pre-lowering CIR"]
    PRE_90["cc1 (sm_90)<br/>emit pre-lowering CIR"]

    COMBINE["cir-combine-bundler<br/>--combine"]
    BUNDLE[("combined.cir<br/>cir.offload.container")]
    UNBUNDLE["cir-combine-bundler<br/>--unbundle"]

    POST_H["cc1 (host)<br/>post-lowering"]
    POST_70["cc1 (sm_70)<br/>post-lowering"]
    POST_90["cc1 (sm_90)<br/>post-lowering"]

    OBJ["host.o"]
    F70["fatbin_sm_70"]
    F90["fatbin_sm_90"]

    SRC --> PRE_H & PRE_70 & PRE_90
    PRE_H & PRE_70 & PRE_90 --> COMBINE --> BUNDLE --> UNBUNDLE
    UNBUNDLE --> POST_H --> OBJ
    UNBUNDLE --> POST_70 --> F70
    UNBUNDLE --> POST_90 --> F90

    classDef action fill:#bbdefb,stroke:#1565c0,color:#000;
    classDef artifact fill:#d1c4e9,stroke:#512da8,color:#000;
    classDef tool fill:#fff9c4,stroke:#f57f17,color:#000;
    class PRE_H,PRE_70,PRE_90,POST_H,POST_70,POST_90 action;
    class BUNDLE,OBJ,F70,F90 artifact;
    class COMBINE,UNBUNDLE tool;
```

## Status

_Updated 2026-05-23._ Bootstrapping. RFC draft depicting intended
driver semantics in-progress bundler tool and new driver actions not yet committed.

