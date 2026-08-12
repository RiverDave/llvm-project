//===- KernelBindingTable.h - Host<->device kernel bindings -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Resolves the host device-stub <-> device kernel correspondence inside a
// cir.offload.container, along with the host call sites that launch each
// kernel. The bridge is the `cu.kernel_name` attribute, whose value is the
// device kernel's mangled symbol name; both the host stub and the calls to it
// carry it.
//
// Usable as an MLIR analysis: getAnalysis<KernelBindingTable>() on a pass
// anchored at a cir.offload.container.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_KERNELBINDINGTABLE_H
#define LLVM_CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_KERNELBINDINGTABLE_H

#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace cir {

// Binding for one kernel, keyed by its device mangled name. A name may resolve
// in several device modules (multi-arch); the owning module of each kernel is
// recoverable via k->getParentOfType<mlir::ModuleOp>().
//
// `launchSites` are the host calls to `hostStub`, which carry the same
// `cu.kernel_name` key as the stub itself. A site holds the kernel arguments,
// not the launch configuration: grid and block dimensions are operands of the
// `__cudaPushCallConfiguration` call preceding it.
struct KernelBinding {
  cir::FuncOp hostStub;
  llvm::SmallVector<cir::FuncOp, 2> deviceKernels;
  llvm::SmallVector<cir::CallOp, 2> launchSites;
};

class KernelBindingTable {
public:
  explicit KernelBindingTable(mlir::Operation *container);

  const KernelBinding *lookup(llvm::StringRef kernelName) const;

  // Device kernels bound to a host stub. The stub carries the `cu.kernel_name`
  // key itself, so a pass holding a launch callee can resolve device kernels
  // without re-deriving the mangled name. Empty if the func is not a stub.
  llvm::ArrayRef<cir::FuncOp> getDeviceFuncs(cir::FuncOp hostFn) const;

  // Host calls launching the named kernel. Empty if the name is unbound or the
  // kernel is never launched in this TU.
  llvm::ArrayRef<cir::CallOp> getLaunchSites(llvm::StringRef kernelName) const;

  bool empty() const { return bindings.empty(); }
  size_t size() const { return bindings.size(); }
  auto begin() const { return bindings.begin(); }
  auto end() const { return bindings.end(); }

  void print(llvm::raw_ostream &os) const;

private:
  // Keys are StringRefs into the `cu.kernel_name` attribute storage, so the
  // table's lifetime is tied to the container IR it was built from.
  llvm::MapVector<llvm::StringRef, KernelBinding> bindings;
};

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_TRANSFORMS_OFFLOADOPT_KERNELBINDINGTABLE_H
