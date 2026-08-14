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

// One host-side launch of a kernel: the call to the device stub, which carries
// the kernel arguments, and the `__cudaPushCallConfiguration` /
// `__hipPushCallConfiguration` call preceding it, which carries the geometry.
//
// Only the two anchor calls are stored. Argument values, launch dimensions and
// their constants are read back out of the IR on demand, so a descriptor can
// never report a value the IR has since stopped holding.
struct LaunchSite {
  cir::CallOp stubCall;

  // Null when the geometry could not be traced: the launch was not emitted in
  // the shape CIRGen produces, or the CFG has already been flattened.
  cir::CallOp pushConfigCall;

  llvm::StringRef getKernelName() const;

  unsigned getNumArgs() const;
  mlir::Value getArg(unsigned i) const;
  // The constant passed for argument `i`, or null if it is not a constant.
  mlir::TypedAttr getConstArg(unsigned i) const;

  // A launch dimension, recovered from the dim3 temporary the push call reads.
  // A component is null when it could not be traced.
  struct Dim3 {
    mlir::Value x, y, z;
    mlir::TypedAttr constX() const;
    mlir::TypedAttr constY() const;
    mlir::TypedAttr constZ() const;
    bool isFullyConstant() const;
  };

  Dim3 getGridDim() const;
  Dim3 getBlockDim() const;

  mlir::Value getSharedMemBytes() const;
  mlir::TypedAttr getConstSharedMem() const;

  mlir::Value getStream() const;
  bool isDefaultStream() const;

  bool hasGeometry() const {
    return pushConfigCall != nullptr;
  }
};

// Binding for one kernel, keyed by its device mangled name. A name may resolve
// in several device modules (multi-arch); the owning module of each kernel is
// recoverable via k->getParentOfType<mlir::ModuleOp>().
//
// `launchSites` are the host calls to `hostStub`, which carry the same
// `cu.kernel_name` key as the stub itself.
struct KernelBinding {
  cir::FuncOp hostStub;
  llvm::SmallVector<cir::FuncOp, 2> deviceKernels;
  llvm::SmallVector<LaunchSite, 2> launchSites;
};

// The kernel `op` launches, or a null attribute if it is not a launch site.
// This is what the table records as a launch site, and what a pass must accept
// when checking that it can see every launch of a stub: the two answers come
// from here so they cannot drift apart.
cir::CUDAKernelNameAttr getLaunchedKernel(mlir::Operation *op);

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
  llvm::ArrayRef<cir::LaunchSite>
  getLaunchSites(llvm::StringRef kernelName) const;

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
