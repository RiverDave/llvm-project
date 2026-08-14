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

// The constant `v` holds, or null if it is not a constant.
static mlir::TypedAttr asConst(mlir::Value v) {
  if (!v)
    return {};
  auto cst = v.getDefiningOp<cir::ConstantOp>();
  return cst ? cst.getValue() : mlir::TypedAttr{};
}

static bool isPushCallConfiguration(cir::CallOp call) {
  std::optional<llvm::StringRef> callee = call.getCallee();
  if (!callee)
    return false;
  assert(*callee != "__llvmPushCallConfiguration" &&
         "NYI: new offload driver launch configuration");
  return *callee == "__cudaPushCallConfiguration" ||
         *callee == "__hipPushCallConfiguration";
}

// CIRGen guards a launch with the result of the push-call-configuration call:
//
//   %c = cir.call @__cudaPushCallConfiguration(...)
//   %b = cir.cast int_to_bool %c
//   cir.if %b { } else { cir.call @stub(...) }
//
// Walk that back from the stub call. Returns null for any other shape,
// including a launch whose CFG has already been flattened.
static cir::CallOp tracePushConfiguration(cir::CallOp stubCall) {
  auto ifOp = stubCall->getParentOfType<cir::IfOp>();
  if (!ifOp)
    return {};
  auto cast = ifOp.getCondition().getDefiningOp<cir::CastOp>();
  if (!cast)
    return {};
  auto call = cast.getSrc().getDefiningOp<cir::CallOp>();
  if (!call || !isPushCallConfiguration(call))
    return {};
  return call;
}

// Whether `call` runs a constructor of `recordType`. CIR marks a constructor
// with a cxx_ctor attribute naming the record it builds, so the call is
// identified by that rather than by a mangled name: dim3 has more than one
// constructor symbol, and a name check would also accept an unrelated function
// with a matching signature.
static bool constructsRecord(cir::CallOp call, mlir::Type recordType) {
  mlir::FlatSymbolRefAttr callee = call.getCalleeAttr();
  if (!callee)
    return false;
  auto fn =
      mlir::SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(call, callee);
  if (!fn)
    return false;
  auto ctor = mlir::dyn_cast_if_present<cir::CXXCtorAttr>(fn.getFuncInfoAttr());
  return ctor && ctor.getType() == recordType;
}

// A dim3 argument of the push call is a load of a stack temporary that a
// constructor filled in:
//
//   %t = cir.alloca "agg.tmp0" : !cir.ptr<!rec_dim3>
//   cir.call @_ZN4dim3C1Ejjj(%t, %x, %y, %z)
//   %d = cir.load %t
//
// The constructor is found through the slot, so the components belong to the
// temporary this launch reads rather than to any dim3 in the function.
static cir::LaunchSite::Dim3 traceDim3(mlir::Value dim) {
  auto load = dim.getDefiningOp<cir::LoadOp>();
  if (!load)
    return {};
  mlir::Value slot = load.getAddr();
  auto slotType = mlir::dyn_cast<cir::PointerType>(slot.getType());
  if (!slotType)
    return {};

  cir::CallOp ctor;
  for (mlir::Operation *user : slot.getUsers()) {
    auto call = mlir::dyn_cast<cir::CallOp>(user);
    if (!call || !constructsRecord(call, slotType.getPointee()))
      continue;
    // Users are unordered, so a second construction of the same slot leaves
    // the one reaching the load undecided; report no geometry instead.
    if (ctor)
      return {};
    ctor = call;
  }
  if (!ctor || ctor->getBlock() != load->getBlock() ||
      !ctor->isBeforeInBlock(load))
    return {};

  mlir::OperandRange args = ctor.getArgOperands();
  if (args.size() != 4 || args[0] != slot)
    return {};
  return {args[1], args[2], args[3]};
}

llvm::StringRef cir::LaunchSite::getKernelName() const {
  cir::CallOp call = stubCall;
  return cir::getLaunchedKernel(call).getKernelName();
}

unsigned cir::LaunchSite::getNumArgs() const {
  return cir::CallOp(stubCall).getArgOperands().size();
}

mlir::Value cir::LaunchSite::getArg(unsigned i) const {
  mlir::OperandRange args = cir::CallOp(stubCall).getArgOperands();
  return i < args.size() ? args[i] : mlir::Value{};
}

mlir::TypedAttr cir::LaunchSite::getConstArg(unsigned i) const {
  return asConst(getArg(i));
}

mlir::TypedAttr cir::LaunchSite::Dim3::constX() const { return asConst(x); }
mlir::TypedAttr cir::LaunchSite::Dim3::constY() const { return asConst(y); }
mlir::TypedAttr cir::LaunchSite::Dim3::constZ() const { return asConst(z); }

bool cir::LaunchSite::Dim3::isFullyConstant() const {
  return constX() && constY() && constZ();
}

// Push call operands: grid, block, shared memory, stream.
cir::LaunchSite::Dim3 cir::LaunchSite::getGridDim() const {
  if (!hasGeometry())
    return {};
  return traceDim3(cir::CallOp(pushConfigCall).getArgOperands()[0]);
}

cir::LaunchSite::Dim3 cir::LaunchSite::getBlockDim() const {
  if (!hasGeometry())
    return {};
  return traceDim3(cir::CallOp(pushConfigCall).getArgOperands()[1]);
}

mlir::Value cir::LaunchSite::getSharedMemBytes() const {
  if (!hasGeometry())
    return {};
  return cir::CallOp(pushConfigCall).getArgOperands()[2];
}

mlir::TypedAttr cir::LaunchSite::getConstSharedMem() const {
  return asConst(getSharedMemBytes());
}

mlir::Value cir::LaunchSite::getStream() const {
  if (!hasGeometry())
    return {};
  return cir::CallOp(pushConfigCall).getArgOperands()[3];
}

bool cir::LaunchSite::isDefaultStream() const {
  return mlir::isa_and_present<cir::ConstPtrAttr>(asConst(getStream()));
}

cir::CUDAKernelNameAttr cir::getLaunchedKernel(mlir::Operation *op) {
  if (!mlir::isa<cir::CallOp>(op))
    return {};
  return op->getAttrOfType<cir::CUDAKernelNameAttr>(
      cir::CUDAKernelNameAttr::getMnemonic());
}

KernelBindingTable::KernelBindingTable(mlir::Operation *container) {
  auto containerOp = mlir::cast<cir::OffloadContainerOp>(container);
  mlir::ModuleOp hostModule = containerOp.getHostModule();

  hostModule.walk([&](cir::FuncOp hostFn) {
    auto kernelNameAttr = hostFn->getAttrOfType<cir::CUDAKernelNameAttr>(
        cir::CUDAKernelNameAttr::getMnemonic());
    if (kernelNameAttr) {
      assert(!lookup(kernelNameAttr.getKernelName()) &&
             "Duplicate stub found for Host TU");
      bindings[kernelNameAttr.getKernelName()].hostStub = hostFn;
    }
  });

  // Walked after the stubs, so every binding already has its host stub. A
  // launch site naming a kernel with no stub here launches nothing this table
  // describes, and recording it would leave a binding without a stub.
  hostModule.walk([&](mlir::Operation *op) {
    cir::CUDAKernelNameAttr kernel = cir::getLaunchedKernel(op);
    if (!kernel)
      return;
    auto it = bindings.find(kernel.getKernelName());
    if (it == bindings.end())
      return;
    cir::CallOp stubCall = mlir::cast<cir::CallOp>(op);
    it->second.launchSites.push_back(
        LaunchSite{stubCall, tracePushConfiguration(stubCall)});
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

const KernelBinding *KernelBindingTable::lookup(llvm::StringRef kernelName) const {
  auto it = bindings.find(kernelName);
  return it == bindings.end() ? nullptr : &it->second;
}

llvm::ArrayRef<cir::FuncOp>
KernelBindingTable::getDeviceFuncs(cir::FuncOp hostFn) const {
  auto kernelNameAttr = hostFn->getAttrOfType<cir::CUDAKernelNameAttr>(
      cir::CUDAKernelNameAttr::getMnemonic());
  if (!kernelNameAttr)
    return {};
  const KernelBinding *binding = lookup(kernelNameAttr.getKernelName());
  return binding ? binding->deviceKernels : llvm::ArrayRef<cir::FuncOp>{};
}

llvm::ArrayRef<cir::LaunchSite>
KernelBindingTable::getLaunchSites(llvm::StringRef kernelName) const {
  const KernelBinding *binding = lookup(kernelName);
  return binding ? binding->launchSites : llvm::ArrayRef<cir::LaunchSite>{};
}

void KernelBindingTable::print(llvm::raw_ostream &os) const {
  os << "// ---- KernelBindingTable ----\n";
  os << "// size " << size() << " empty " << (empty() ? "true" : "false")
     << "\n";
  for (const auto &binding : bindings) {
    cir::FuncOp stub = binding.second.hostStub;
    os << "// - kernel " << binding.first << "\n";
    os << "//   host-stub @" << stub.getName() << "\n";
    if (binding.second.deviceKernels.empty())
      os << "//   <no device kernel>\n";
    for (cir::FuncOp kernel : binding.second.deviceKernels) {
      auto mod = kernel->getParentOfType<mlir::ModuleOp>();
      os << "//   device @" << mod.getSymName().value_or("<unnamed>") << " : @"
         << kernel.getName() << "\n";
    }
    if (binding.second.launchSites.empty())
      os << "//   <no launch site>\n";
    for (const cir::LaunchSite &launch : binding.second.launchSites) {
      cir::CallOp stubCall = launch.stubCall;
      auto caller = stubCall->getParentOfType<cir::FuncOp>();
      os << "//   launch in @" << caller.getName() << " : "
         << launch.getNumArgs() << " args";
      if (!launch.hasGeometry()) {
        os << ", <no geometry>\n";
        continue;
      }
      cir::LaunchSite::Dim3 grid = launch.getGridDim();
      cir::LaunchSite::Dim3 block = launch.getBlockDim();
      os << ", grid " << (grid.isFullyConstant() ? "const" : "dynamic")
         << ", block " << (block.isFullyConstant() ? "const" : "dynamic")
         << ", stream " << (launch.isDefaultStream() ? "default" : "explicit")
         << "\n";
    }
    os << "//\n";
  }
  os << "// ----------------------------\n";
}
