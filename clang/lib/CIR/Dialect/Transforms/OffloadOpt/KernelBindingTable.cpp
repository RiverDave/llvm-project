//===- KernelBindingTable.cpp - Host<->device kernel bindings -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace cir;

KernelBindingTable::KernelBindingTable(mlir::Operation *container) {
  auto containerOp = mlir::cast<cir::OffloadContainerOp>(container);
  mlir::ModuleOp hostModule = containerOp.getHostModule();

  hostModule.walk([&](cir::FuncOp hostFn) {
    auto kernelNameAttr = hostFn->getAttrOfType<cir::CUDAKernelNameAttr>(
        cir::CUDAKernelNameAttr::getMnemonic());
    if (kernelNameAttr) {
      bindings[kernelNameAttr.getKernelName()].hostStub = hostFn;
    }
  });

  for (mlir::ModuleOp deviceMod : containerOp.getDeviceModules()) {
    for (auto &binding : bindings) {
      if (cir::FuncOp kernel = llvm::dyn_cast_if_present<cir::FuncOp>(
              deviceMod.lookupSymbol(binding.first))) {
        binding.second.deviceKernels.push_back(kernel);
      }
    }
  }
}

const KernelBinding *KernelBindingTable::lookup(llvm::StringRef name) const {
  auto it = bindings.find(name);
  return it == bindings.end() ? nullptr : &it->second;
}

void KernelBindingTable::print(llvm::raw_ostream &os) const {
  os << "// ---- KernelBindingTable ----\n";
  for (const auto &b : bindings) {
    cir::FuncOp stub = b.second.hostStub;
    os << "// - kernel " << b.first << "\n";
    os << "//   host-stub @" << stub.getName() << "\n";
    if (b.second.deviceKernels.empty())
      os << "//   <no device kernel>\n";
    for (cir::FuncOp kernel : b.second.deviceKernels) {
      auto mod = kernel->getParentOfType<mlir::ModuleOp>();
      os << "//   device @" << mod.getSymName().value_or("<unnamed>") << " : @"
         << kernel.getName() << "\n";
    }
    os << "//\n";
  }
  os << "// ----------------------------\n";
}
