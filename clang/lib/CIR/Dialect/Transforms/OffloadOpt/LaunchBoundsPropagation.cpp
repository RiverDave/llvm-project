//===- LaunchBoundsPropagation.cpp - Infer kernel launch bounds -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/NVVMAttributes.h"

namespace mlir {
#define GEN_PASS_DEF_OFFLOADLAUNCHBOUNDSPROPAGATION
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

// Threads per block for one launch, or nullopt when the geometry was not traced
// or is not a compile-time value. Those are different answers -- see
// `LaunchSite::Dim3` -- but neither yields a bound, so both stop here.
static std::optional<uint64_t> constBlockSize(const cir::LaunchSite &site) {
  if (!site.hasGeometry())
    return std::nullopt;
  cir::LaunchSite::Dim3 block = site.getBlockDim();
  if (!block.isFullyConstant())
    return std::nullopt;

  auto asInt = [](mlir::TypedAttr a) -> std::optional<uint64_t> {
    auto i = mlir::dyn_cast_or_null<cir::IntAttr>(a);
    if (!i)
      return std::nullopt;
    return i.getUInt();
  };
  std::optional<uint64_t> x = asInt(block.constX());
  std::optional<uint64_t> y = asInt(block.constY());
  std::optional<uint64_t> z = asInt(block.constZ());
  if (!x || !y || !z)
    return std::nullopt;

  // `getOverallMaxNTID` takes the product of the vector, so a flattened scalar
  // and a 3-vector mean the same thing to the backend.
  return *x * *y * *z;
}

// Largest block any visible launch uses. Null when a site's geometry is opaque:
// an unseen launch may use a larger block, and the bound is a hard contract.
static std::optional<uint64_t>
maxBlockSize(llvm::ArrayRef<cir::LaunchSite> sites) {
  uint64_t max = 0;
  for (const cir::LaunchSite &site : sites) {
    std::optional<uint64_t> size = constBlockSize(site);
    if (!size)
      return std::nullopt;
    max = std::max(max, *size);
  }
  return max ? std::optional<uint64_t>(max) : std::nullopt;
}

// Only NVPTX for now. HIP carries `amdgpu-flat-work-group-size` on every kernel
// already, so propagating there is an overwrite rather than an addition and
// needs its own provenance story.
static bool isNVPTXKernel(cir::FuncOp fn) {
  return fn.getCallingConv() == cir::CallingConv::PTXKernel;
}

// Tighten `cir.nvvm.maxntid`, never loosen it: a user's __launch_bounds__ is a
// promise we may sharpen but must not widen. The attribute name has to match
// what CIRGen writes in Targets/NVPTX.cpp or the two mechanisms drift apart.
static bool setMaxNTID(cir::FuncOp kernel, uint64_t bound,
                       mlir::OpBuilder &builder) {
  std::string name = ("cir." + llvm::NVVMAttr::MaxNTID).str();
  if (auto existing = kernel->getAttrOfType<mlir::StringAttr>(name)) {
    uint64_t prev;
    if (!llvm::to_integer(existing.getValue(), prev, 10))
      return false;
    if (prev <= bound)
      return false;
  }
  kernel->setAttr(name, builder.getStringAttr(llvm::utostr(bound)));
  return true;
}

struct OffloadLaunchBoundsPropagationPass
    : public impl::OffloadLaunchBoundsPropagationBase<
          OffloadLaunchBoundsPropagationPass> {
  void runOnOperation() override;
};

void OffloadLaunchBoundsPropagationPass::runOnOperation() {
  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  mlir::OpBuilder builder(&getContext());
  bool changed = false;

  for (const auto &entry : table) {
    const cir::KernelBinding &binding = entry.second;
    llvm::ArrayRef<cir::LaunchSite> sites = binding.launchSites;

    // A kernel launched nowhere in this TU says nothing about its geometry.
    if (sites.empty())
      continue;
    if (!table.allLaunchSitesVisible(entry.first))
      continue;

    std::optional<uint64_t> bound = maxBlockSize(sites);
    if (!bound)
      continue;

    // The launch configuration is host-side, so one bound applies to every
    // arch's copy of the kernel even when __CUDA_ARCH__ made the bodies differ.
    for (cir::FuncOp kernel : binding.deviceKernels) {
      if (kernel.isDeclaration() || !isNVPTXKernel(kernel))
        continue;
      if (setMaxNTID(kernel, *bound, builder))
        changed = true;
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadLaunchBoundsPropagationPass() {
  return std::make_unique<OffloadLaunchBoundsPropagationPass>();
}
