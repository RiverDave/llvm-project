//===- KernelCloning.cpp - Clone a kernel for a subset of launches --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelCloning.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/ADT/StringSet.h"

using namespace cir;

// Determine unique names for a clone of a kernel and stub. Each clone's name is
// a concatenation of base and suffix. If a name is already taken, the
// candidate name is extended with a number separated by a dot.
static std::tuple<std::string, std::string>
uniqueNames(cir::OffloadContainerOp container, llvm::StringRef baseKernelName,
            llvm::StringRef baseStubName, llvm::StringRef suffix) {
  llvm::StringSet<> taken;
  container->walk([&](mlir::Operation *op) {
    if (auto sym = mlir::dyn_cast<mlir::SymbolOpInterface>(op))
      taken.insert(sym.getName());
  });
  const auto uniqueName = [&](llvm::StringRef base) -> std::string {
    std::string candidate = (base + suffix).str();
    if (!taken.contains(candidate))
      return candidate;
    for (unsigned i = 1;; ++i) {
      std::string next = candidate + "." + std::to_string(i);
      if (!taken.contains(next))
        return next;
    }
  };
  return {uniqueName(baseKernelName), uniqueName(baseStubName)};
}

// The handle global a device stub launches through: CIRGen names it with the
// kernel's mangled name and initialises it with the stub's address, and the
// stub body reads it with a cir.get_global. Returned along with the read, so
// the clone's copy of the read can be pointed at the clone's own handle.
static cir::GetGlobalOp findHandleRead(cir::FuncOp stub,
                                       llvm::StringRef kernelName) {
  cir::GetGlobalOp found;
  stub.walk([&](cir::GetGlobalOp get) {
    if (get.getName() == kernelName)
      found = get;
  });
  return found;
}

std::optional<cir::KernelClone> cir::cloneKernelForSites(
    cir::OffloadContainerOp container, const cir::KernelBinding &binding,
    llvm::StringRef suffix, llvm::ArrayRef<cir::LaunchSite> sites) {
  if (binding.deviceKernels.empty() || !binding.hostStub || sites.empty())
    return std::nullopt;

  cir::FuncOp stub = binding.hostStub;
  mlir::ModuleOp hostModule = container.getHostModule();
  auto kernelNameAttr = stub->getAttrOfType<cir::CUDAKernelNameAttr>(
      cir::CUDAKernelNameAttr::getMnemonic());
  if (!kernelNameAttr)
    return std::nullopt;
  llvm::StringRef oldKernelName = kernelNameAttr.getKernelName();

  // The stub must launch through its handle; without that read there is no way
  // to make the copy launch the copied kernel.
  cir::GetGlobalOp handleRead = findHandleRead(stub, oldKernelName);
  auto handle = mlir::dyn_cast_or_null<cir::GlobalOp>(
      hostModule.lookupSymbol(oldKernelName));
  if (!handleRead || !handle)
    return std::nullopt;

  cir::KernelClone clone;
  std::string newStubName;
  std::tie(clone.kernelName, newStubName) =
      uniqueNames(container, oldKernelName, stub.getSymName(), suffix);

  mlir::MLIRContext *ctx = container.getContext();
  auto newKernelNameAttr = cir::CUDAKernelNameAttr::get(
      ctx, mlir::StringAttr::get(ctx, clone.kernelName));

  // Device side: one copy per module that held the original.
  clone.deviceKernels.reserve(binding.deviceKernels.size());
  for (cir::FuncOp kernel : binding.deviceKernels) {
    auto deviceModule = kernel->getParentOfType<mlir::ModuleOp>();
    if (!deviceModule)
      continue;
    auto copy = mlir::cast<cir::FuncOp>(kernel->clone());
    mlir::SymbolTable::setSymbolName(copy, clone.kernelName);
    mlir::SymbolTable(deviceModule).insert(copy);
    copy->moveAfter(kernel);
    clone.deviceKernels.push_back(copy);
  }
  if (clone.deviceKernels.empty())
    return std::nullopt;

  // Host side: copy the stub, rebind it to the cloned kernel, and give it a
  // handle global of its own. `LoweringPrepare` emits the registration by
  // looking up a GlobalOp named exactly like `cu.kernel_name`, so the handle
  // has to exist under that name or registration will fail on the cast.
  auto newStub = mlir::cast<cir::FuncOp>(stub->clone());
  mlir::SymbolTable::setSymbolName(newStub, newStubName);
  newStub->setAttr(cir::CUDAKernelNameAttr::getMnemonic(), newKernelNameAttr);
  newStub->setAttr(kSpecializationCloneAttr, mlir::UnitAttr::get(ctx));
  mlir::SymbolTable(hostModule).insert(newStub);
  newStub->moveAfter(stub);
  clone.hostStub = newStub;

  // Point the copy's handle read at the copy's handle.
  if (cir::GetGlobalOp copiedRead = findHandleRead(newStub, oldKernelName))
    copiedRead.setName(clone.kernelName);

  mlir::OpBuilder builder(handle);
  auto newHandle = mlir::cast<cir::GlobalOp>(builder.clone(*handle));
  mlir::SymbolTable::setSymbolName(newHandle, clone.kernelName);
  newHandle.setInitialValueAttr(cir::GlobalViewAttr::get(
      newHandle.getSymType(), mlir::FlatSymbolRefAttr::get(ctx, newStubName)));

  // Retarget the requested launches. The callee and the `cu.kernel_name` on
  // the call have to move together: KernelBindingTable reads the attribute to
  // decide what a call launches, so leaving it stale would make the table
  // disagree with the IR.
  for (const cir::LaunchSite &site : sites) {
    cir::CallOp call = site.stubCall;
    call.setCalleeAttr(mlir::FlatSymbolRefAttr::get(ctx, newStubName));
    call->setAttr(cir::CUDAKernelNameAttr::getMnemonic(), newKernelNameAttr);
  }

  return clone;
}

cir::SpecializationTarget cir::getSpecializationTarget(
    cir::OffloadContainerOp container, llvm::StringRef kernelName,
    const cir::KernelBinding &binding, llvm::StringRef suffix,
    llvm::ArrayRef<cir::LaunchSite> sites) {
  if (!binding.hostStub || sites.empty())
    return {};

  // If the stub is already a specialization clone, then we can just use the
  // existing clone.
  if (binding.hostStub->hasAttr(kSpecializationCloneAttr))
    return {binding.hostStub,
            {binding.deviceKernels.begin(), binding.deviceKernels.end()},
            /*cloned=*/false};

  // Otherwise we try to clone the kernel for the requested sites.
  std::optional<cir::KernelClone> clone =
      cloneKernelForSites(container, binding, suffix, sites);
  if (!clone)
    return {};
  return {std::move(clone->hostStub), std::move(clone->deviceKernels),
          /*cloned=*/true};
}
