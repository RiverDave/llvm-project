//===- CUDARegistration.cpp - Register CUDA/HIP kernels and variables ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builds the host module ctor that registers the device binary, the kernels
// (functions carrying `cu.kernel_name`) and the device variables (globals
// carrying `cu.var_registration`) with the CUDA/HIP runtime.
//
// It runs after LoweringPrepare, both in-process and on a host module resumed
// from .cir: in the offload-merge pipeline the device binary only exists by the
// time the host is resumed.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "TargetLowering/LowerModule.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "clang/Basic/Cuda.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/Dialect/Builder/CIRBaseBuilder.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDataLayout.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VersionTuple.h"

#include <memory>
#include <optional>

using namespace mlir;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CUDAREGISTERMODULE
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

struct CUDARegisterModulePass
    : public impl::CUDARegisterModuleBase<CUDARegisterModulePass> {
  CUDARegisterModulePass() = default;
  // `clonePass()` needs a copy; per-run state starts fresh in the clone.
  CUDARegisterModulePass(const CUDARegisterModulePass &other)
      : impl::CUDARegisterModuleBase<CUDARegisterModulePass>(other) {}

  void runOnOperation() override;

private:
  mlir::ModuleOp mlirModule;
  std::unique_ptr<cir::LowerModule> lowerModule;
  llvm::StringMap<FuncOp> cudaKernelMap;
  llvm::SmallVector<std::pair<cir::GlobalOp, cir::CUDAVarRegistrationInfoAttr>>
      cudaDeviceVars;

  const clang::TargetInfo &getTargetInfo() const {
    return lowerModule->getTarget();
  }
  const clang::LangOptions &getLangOpts() const {
    return lowerModule->getLangOpts();
  }

  /// Platform SDK version recorded on the module by CIRGen. An absent attribute
  /// yields an empty VersionTuple. Returns std::nullopt if the attribute is
  /// present but unparseable; an error has been emitted and the pass marked as
  /// failed, so the caller must bail out.
  std::optional<llvm::VersionTuple> getSDKVersion() {
    auto sdkVersionAttr = mlirModule->getAttrOfType<mlir::StringAttr>(
        CIRDialect::getSDKVersionAttrName());
    if (!sdkVersionAttr)
      return llvm::VersionTuple();

    llvm::VersionTuple sdkVersion;
    if (sdkVersion.tryParse(sdkVersionAttr.getValue())) {
      mlirModule->emitError("cannot parse platform SDK version from ")
          << CIRDialect::getSDKVersionAttrName() << " = '"
          << sdkVersionAttr.getValue() << "'";
      signalPassFailure();
      return std::nullopt;
    }
    return sdkVersion;
  }

  cir::FuncOp buildRuntimeFunction(
      mlir::OpBuilder &builder, llvm::StringRef name, mlir::Location loc,
      cir::FuncType type,
      cir::GlobalLinkageKind linkage = cir::GlobalLinkageKind::ExternalLinkage);

  /// LoweringPrepare already wrote the module's ctor list; the module ctor
  /// goes after the ctors it holds.
  void appendGlobalCtor(llvm::StringRef name);

  /// Build the CUDA module constructor that registers the fat binary
  /// with the CUDA runtime.
  void buildCUDAModuleCtor();
  std::optional<FuncOp> buildCUDAModuleDtor();
  std::optional<FuncOp> buildHIPModuleDtor();
  std::optional<FuncOp> buildCUDARegisterGlobals();
  void buildCUDARegisterVars(cir::CIRBaseBuilderTy &builder,
                             FuncOp regGlobalFunc);
  void buildCUDARegisterGlobalFunctions(cir::CIRBaseBuilderTy &builder,
                                        FuncOp regGlobalFunc);
};

} // namespace

cir::FuncOp CUDARegisterModulePass::buildRuntimeFunction(
    mlir::OpBuilder &builder, llvm::StringRef name, mlir::Location loc,
    cir::FuncType type, cir::GlobalLinkageKind linkage) {
  cir::FuncOp f = dyn_cast_or_null<FuncOp>(SymbolTable::lookupNearestSymbolFrom(
      mlirModule, StringAttr::get(mlirModule->getContext(), name)));
  if (!f) {
    f = cir::FuncOp::create(builder, loc, name, type);
    f.setLinkageAttr(
        cir::GlobalLinkageKindAttr::get(builder.getContext(), linkage));
    mlir::SymbolTable::setSymbolVisibility(
        f, mlir::SymbolTable::Visibility::Private);

    assert(!cir::MissingFeatures::opFuncExtraAttrs());
  }
  return f;
}

void CUDARegisterModulePass::appendGlobalCtor(llvm::StringRef name) {
  llvm::SmallVector<mlir::Attribute> ctors;
  if (auto existing = mlirModule->getAttrOfType<mlir::ArrayAttr>(
          CIRDialect::getGlobalCtorsAttrName()))
    ctors.append(existing.begin(), existing.end());
  ctors.push_back(cir::GlobalCtorAttr::get(
      &getContext(), name, cir::GlobalCtorAttr::getDefaultPriority()));
  mlirModule->setAttr(CIRDialect::getGlobalCtorsAttrName(),
                      mlir::ArrayAttr::get(&getContext(), ctors));
}

static llvm::StringRef getCUDAPrefix(const clang::LangOptions &langOpts) {
  if (langOpts.HIP)
    return "hip";
  return "cuda";
}

static std::string addUnderscoredPrefix(llvm::StringRef prefix,
                                        llvm::StringRef name) {
  return ("__" + prefix + name).str();
}

/// Creates a global constructor function for the module:
///
/// For CUDA:
/// \code
/// void __cuda_module_ctor() {
///     Handle = __cudaRegisterFatBinary(GpuBinaryBlob);
///     __cuda_register_globals(Handle);
/// }
/// \endcode
///
/// For HIP:
/// \code
/// void __hip_module_ctor() {
///     if (__hip_gpubin_handle == 0) {
///         __hip_gpubin_handle  = __hipRegisterFatBinary(GpuBinaryBlob);
///         __hip_register_globals(__hip_gpubin_handle);
///     }
/// }
/// \endcode
void CUDARegisterModulePass::buildCUDAModuleCtor() {
  bool isHIP = getLangOpts().HIP;

  if (getLangOpts().GPURelocatableDeviceCode)
    llvm_unreachable("GPU RDC NYI");

  // For CUDA without -fgpu-rdc, it's safe to stop generating ctor
  // if there's nothing to register.
  if (cudaKernelMap.empty() && cudaDeviceVars.empty())
    return;

  // There's no device-side binary, so no need to proceed for CUDA.
  // HIP has to create an external symbol in this case, which is NYI.
  auto deviceBinaryAttr = mlirModule->getAttrOfType<mlir::StringAttr>(
      CIRDialect::getCUDADeviceBinaryAttrName());
  if (!deviceBinaryAttr) {
    if (isHIP)
      assert(!cir::MissingFeatures::hipModuleCtor());
    return;
  }

  // Set up common types and builder.
  llvm::StringRef cudaPrefix = getCUDAPrefix(getLangOpts());
  mlir::Location loc = mlirModule->getLoc();
  CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointToStart(mlirModule.getBody());

  Type voidTy = builder.getVoidTy();
  PointerType voidPtrTy = builder.getVoidPtrTy();
  PointerType voidPtrPtrTy = builder.getPointerTo(voidPtrTy);
  IntType intTy = builder.getSIntNTy(32);

  // --- Create fatbin globals ---

  // The section names are different for MAC OS X.
  llvm::StringRef fatbinConstName =
      getLangOpts().HIP ? ".hip_fatbin" : ".nv_fatbin";

  llvm::StringRef fatbinSectionName =
      getLangOpts().HIP ? ".hipFatBinSegment" : ".nvFatBinSegment";

  // Create the fatbin string constant with GPU binary contents.
  // The dialect verifier guarantees the attribute is typed as the array.
  auto fatbinType = mlir::cast<ArrayType>(deviceBinaryAttr.getType());
  std::string fatbinStrName = addUnderscoredPrefix(cudaPrefix, "_fatbin_str");
  GlobalOp fatbinStr = GlobalOp::create(builder, loc, fatbinStrName, fatbinType,
                                        /*isConstant=*/true, {},
                                        GlobalLinkageKind::PrivateLinkage);
  if (isHIP) {
    const unsigned HIPCodeObjectAlign = 4096;
    fatbinStr.setAlignment(HIPCodeObjectAlign);
  } else {
    fatbinStr.setAlignment(8);
  }

  fatbinStr.setInitialValueAttr(
      cir::ConstArrayAttr::get(fatbinType, deviceBinaryAttr));
  fatbinStr.setSection(fatbinConstName);
  fatbinStr.setPrivate();

  // Create the fatbin wrapper struct:
  //    struct { int magic; int version; void *fatbin; void *unused; };
  mlir::Type fatbinWrapperMembers[] = {intTy, intTy, voidPtrTy, voidPtrTy};
  auto fatbinWrapperType = cir::StructType::get(
      &getContext(), fatbinWrapperMembers, /*packed=*/false, /*is_class=*/false,
      cir::RecordType::getAllDataKinds(fatbinWrapperMembers));
  std::string fatbinWrapperName =
      addUnderscoredPrefix(cudaPrefix, "_fatbin_wrapper");
  GlobalOp fatbinWrapper = GlobalOp::create(
      builder, loc, fatbinWrapperName, fatbinWrapperType,
      /*isConstant=*/true, {}, GlobalLinkageKind::PrivateLinkage);
  fatbinWrapper.setSection(fatbinSectionName);

  constexpr unsigned cudaFatMagic = 0x466243b1;
  constexpr unsigned hipFatMagic = 0x48495046;
  unsigned fatMagic = isHIP ? hipFatMagic : cudaFatMagic;

  auto magicInit = IntAttr::get(intTy, fatMagic);
  auto versionInit = IntAttr::get(intTy, 1);
  auto fatbinStrSymbol =
      mlir::FlatSymbolRefAttr::get(fatbinStr.getSymNameAttr());
  auto fatbinInit = GlobalViewAttr::get(voidPtrTy, fatbinStrSymbol);
  mlir::TypedAttr unusedInit = builder.getConstNullPtrAttr(voidPtrTy);
  fatbinWrapper.setInitialValueAttr(cir::ConstRecordAttr::get(
      fatbinWrapperType,
      mlir::ArrayAttr::get(&getContext(),
                           {magicInit, versionInit, fatbinInit, unusedInit})));

  // Create the GPU binary handle global variable.
  std::string gpubinHandleName =
      addUnderscoredPrefix(cudaPrefix, "_gpubin_handle");

  GlobalOp gpuBinHandle = GlobalOp::create(
      builder, loc, gpubinHandleName, voidPtrPtrTy,
      /*isConstant=*/false, {}, cir::GlobalLinkageKind::InternalLinkage);
  gpuBinHandle.setInitialValueAttr(builder.getConstNullPtrAttr(voidPtrPtrTy));
  gpuBinHandle.setPrivate();

  // Declare this function:
  //    void **__{cuda|hip}RegisterFatBinary(void *);

  std::string regFuncName =
      addUnderscoredPrefix(cudaPrefix, "RegisterFatBinary");
  FuncType regFuncType = FuncType::get({voidPtrTy}, voidPtrPtrTy);
  cir::FuncOp regFunc =
      buildRuntimeFunction(builder, regFuncName, loc, regFuncType);

  std::string moduleCtorName = addUnderscoredPrefix(cudaPrefix, "_module_ctor");
  cir::FuncOp moduleCtor = buildRuntimeFunction(
      builder, moduleCtorName, loc, FuncType::get({}, voidTy),
      GlobalLinkageKind::InternalLinkage);

  appendGlobalCtor(moduleCtorName);
  builder.setInsertionPointToStart(moduleCtor.addEntryBlock());
  assert(!cir::MissingFeatures::opGlobalCtorPriority());
  if (isHIP) {
    // --- Create HIP CTOR ---
    //   if (__hip_gpubin_handle == nullptr)
    //     __hip_gpubin_handle = __hipRegisterFatBinary(&fatbinWrapper);
    //   __hip_register_globals(__hip_gpubin_handle);
    //   atexit(__hip_module_dtor);
    mlir::Block *entryBlock = builder.getInsertionBlock();
    mlir::Region *parent = entryBlock->getParent();
    mlir::Block *ifBlock = builder.createBlock(parent);
    mlir::Block *exitBlock = builder.createBlock(parent);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(entryBlock);
      mlir::Value handle =
          builder.createLoad(loc, builder.createGetGlobal(gpuBinHandle));
      auto handlePtrTy = mlir::cast<cir::PointerType>(handle.getType());
      mlir::Value nullPtr = builder.getNullPtr(handlePtrTy, loc);
      mlir::Value isNull =
          builder.createCompare(loc, cir::CmpOpKind::eq, handle, nullPtr);
      cir::BrCondOp::create(builder, loc, isNull, ifBlock, exitBlock);
    }
    {
      // Handle is null: load the fatbin and register it.
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(ifBlock);
      mlir::Value wrapper = builder.createGetGlobal(fatbinWrapper);
      mlir::Value fatbinVoidPtr = builder.createBitcast(wrapper, voidPtrTy);
      cir::CallOp gpuBinaryHandleCall =
          builder.createCallOp(loc, regFunc, fatbinVoidPtr);
      mlir::Value gpuBinaryHandle = gpuBinaryHandleCall.getResult();
      // Store the value back to the global `__hip_gpubin_handle`.
      mlir::Value gpuBinaryHandleGlobal = builder.createGetGlobal(gpuBinHandle);
      builder.createStore(loc, gpuBinaryHandle, gpuBinaryHandleGlobal);
      cir::BrOp::create(builder, loc, exitBlock);
    }
    {
      // Exit block: load the (possibly newly-registered) handle, call
      // __hip_register_globals, and register the module dtor with atexit().
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(exitBlock);
      mlir::Value gHandle =
          builder.createLoad(loc, builder.createGetGlobal(gpuBinHandle));

      if (std::optional<FuncOp> regGlobal = buildCUDARegisterGlobals())
        builder.createCallOp(loc, *regGlobal, gHandle);

      if (std::optional<FuncOp> dtor = buildHIPModuleDtor()) {
        cir::CIRBaseBuilderTy globalBuilder(getContext());
        globalBuilder.setInsertionPointToStart(mlirModule.getBody());
        FuncOp atexit = buildRuntimeFunction(
            globalBuilder, "atexit", loc,
            FuncType::get(PointerType::get(dtor->getFunctionType()), intTy));
        mlir::Value dtorFunc = GetGlobalOp::create(
            builder, loc, PointerType::get(dtor->getFunctionType()),
            mlir::FlatSymbolRefAttr::get(dtor->getSymNameAttr()));
        builder.createCallOp(loc, atexit, dtorFunc);
      }
      cir::ReturnOp::create(builder, loc);
    }
    return;
  }
  if (!getLangOpts().GPURelocatableDeviceCode) {

    // --- Create CUDA CTOR-DTOR ---
    // Register binary with CUDA runtime. This is substantially different in
    // default mode vs. separate compilation.
    // Corresponding code:
    //     gpuBinaryHandle = __cudaRegisterFatBinary(&fatbinWrapper);
    mlir::Value wrapper = builder.createGetGlobal(fatbinWrapper);
    mlir::Value fatbinVoidPtr = builder.createBitcast(wrapper, voidPtrTy);
    cir::CallOp gpuBinaryHandleCall =
        builder.createCallOp(loc, regFunc, fatbinVoidPtr);
    mlir::Value gpuBinaryHandle = gpuBinaryHandleCall.getResult();
    // Store the value back to the global `__cuda_gpubin_handle`.
    mlir::Value gpuBinaryHandleGlobal = builder.createGetGlobal(gpuBinHandle);
    builder.createStore(loc, gpuBinaryHandle, gpuBinaryHandleGlobal);

    // --- Generate __cuda_register_globals and call it ---
    if (std::optional<FuncOp> regGlobal = buildCUDARegisterGlobals()) {
      builder.createCallOp(loc, *regGlobal, gpuBinaryHandle);
    }

    // From CUDA 10.1 onwards, we must call this function to end registration:
    //      void __cudaRegisterFatBinaryEnd(void **fatbinHandle);
    // This is CUDA-specific, so no need to use `addUnderscoredPrefix`.
    std::optional<llvm::VersionTuple> sdkVersion = getSDKVersion();
    if (!sdkVersion)
      return;
    if (clang::CudaFeatureEnabled(
            *sdkVersion, clang::CudaFeature::CUDA_USES_FATBIN_REGISTER_END)) {
      cir::CIRBaseBuilderTy globalBuilder(getContext());
      globalBuilder.setInsertionPointToStart(mlirModule.getBody());
      FuncOp endFunc =
          buildRuntimeFunction(globalBuilder, "__cudaRegisterFatBinaryEnd", loc,
                               FuncType::get({voidPtrPtrTy}, voidTy));
      builder.createCallOp(loc, endFunc, gpuBinaryHandle);
    }
  } else
    llvm_unreachable("GPU RDC NYI");

  // Create destructor and register it with atexit() the way NVCC does it. Doing
  // it during regular destructor phase worked in CUDA before 9.2 but results in
  // double-free in 9.2.
  if (std::optional<FuncOp> dtor = buildCUDAModuleDtor()) {

    // extern "C" int atexit(void (*f)(void));
    cir::CIRBaseBuilderTy globalBuilder(getContext());
    globalBuilder.setInsertionPointToStart(mlirModule.getBody());
    FuncOp atexit = buildRuntimeFunction(
        globalBuilder, "atexit", loc,
        FuncType::get(PointerType::get(dtor->getFunctionType()), intTy));
    mlir::Value dtorFunc = GetGlobalOp::create(
        builder, loc, PointerType::get(dtor->getFunctionType()),
        mlir::FlatSymbolRefAttr::get(dtor->getSymNameAttr()));
    builder.createCallOp(loc, atexit, dtorFunc);
  }
  cir::ReturnOp::create(builder, loc);
}

std::optional<FuncOp> CUDARegisterModulePass::buildCUDAModuleDtor() {
  if (!mlirModule->getAttr(CIRDialect::getCUDADeviceBinaryAttrName()))
    return {};

  llvm::StringRef prefix = getCUDAPrefix(getLangOpts());

  VoidType voidTy = VoidType::get(&getContext());
  PointerType voidPtrPtrTy = PointerType::get(PointerType::get(voidTy));

  mlir::Location loc = mlirModule.getLoc();

  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointToStart(mlirModule.getBody());

  // define: void __cudaUnregisterFatBinary(void ** handle);
  std::string unregisterFuncName =
      addUnderscoredPrefix(prefix, "UnregisterFatBinary");
  FuncOp unregisterFunc = buildRuntimeFunction(
      builder, unregisterFuncName, loc, FuncType::get({voidPtrPtrTy}, voidTy));

  // void __cuda_module_dtor();
  // Despite the name, OG doesn't treat it as a destructor, so it shouldn't be
  // put into globalDtorList. If it were a real dtor, then it would cause
  // double free above CUDA 9.2. The way to use it is to manually call
  // atexit() at end of module ctor.
  std::string dtorName = addUnderscoredPrefix(prefix, "_module_dtor");
  FuncOp dtor =
      buildRuntimeFunction(builder, dtorName, loc, FuncType::get({}, voidTy),
                           GlobalLinkageKind::InternalLinkage);

  builder.setInsertionPointToStart(dtor.addEntryBlock());

  // For dtor, we only need to call:
  //    __cudaUnregisterFatBinary(__cuda_gpubin_handle);

  std::string gpubinName = addUnderscoredPrefix(prefix, "_gpubin_handle");
  GlobalOp gpubinGlobal = cast<GlobalOp>(mlirModule.lookupSymbol(gpubinName));
  mlir::Value gpubinAddress = builder.createGetGlobal(gpubinGlobal);
  mlir::Value gpubin = builder.createLoad(loc, gpubinAddress);
  builder.createCallOp(loc, unregisterFunc, gpubin);
  ReturnOp::create(builder, loc);

  return dtor;
}

/// Build the HIP module dtor:
///
///     void __hip_module_dtor() {
///       if (__hip_gpubin_handle != nullptr) {
///         __hipUnregisterFatBinary(__hip_gpubin_handle);
///         __hip_gpubin_handle = nullptr;
///       }
///     }
///
/// Despite the name, OG doesn't treat this as a real destructor: putting it on
/// the dtor list would cause a double-free. It is meant to be registered via
/// atexit() at the end of the module ctor.
std::optional<FuncOp> CUDARegisterModulePass::buildHIPModuleDtor() {
  if (!mlirModule->getAttr(CIRDialect::getCUDADeviceBinaryAttrName()))
    return {};

  llvm::StringRef prefix = getCUDAPrefix(getLangOpts());

  VoidType voidTy = VoidType::get(&getContext());
  PointerType voidPtrPtrTy = PointerType::get(PointerType::get(voidTy));

  mlir::Location loc = mlirModule.getLoc();

  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointToStart(mlirModule.getBody());

  // void __hipUnregisterFatBinary(void ** handle);
  std::string unregisterFuncName =
      addUnderscoredPrefix(prefix, "UnregisterFatBinary");
  FuncOp unregisterFunc = buildRuntimeFunction(
      builder, unregisterFuncName, loc, FuncType::get({voidPtrPtrTy}, voidTy));

  std::string dtorName = addUnderscoredPrefix(prefix, "_module_dtor");
  FuncOp dtor =
      buildRuntimeFunction(builder, dtorName, loc, FuncType::get({}, voidTy),
                           GlobalLinkageKind::InternalLinkage);

  std::string gpubinName = addUnderscoredPrefix(prefix, "_gpubin_handle");
  GlobalOp gpuBinGlobal = cast<GlobalOp>(mlirModule.lookupSymbol(gpubinName));

  mlir::Block *entryBlock = dtor.addEntryBlock();
  mlir::Block *ifBlock = builder.createBlock(&dtor.getBody());
  mlir::Block *exitBlock = builder.createBlock(&dtor.getBody());

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(entryBlock);
  mlir::Value handle =
      builder.createLoad(loc, builder.createGetGlobal(gpuBinGlobal));
  auto handlePtrTy = mlir::cast<cir::PointerType>(handle.getType());
  mlir::Value nullPtr = builder.getNullPtr(handlePtrTy, loc);
  mlir::Value isNotNull =
      builder.createCompare(loc, cir::CmpOpKind::ne, handle, nullPtr);
  cir::BrCondOp::create(builder, loc, isNotNull, ifBlock, exitBlock);

  {
    // Handle is non-null: unregister and clear it.
    mlir::OpBuilder::InsertionGuard ifGuard(builder);
    builder.setInsertionPointToStart(ifBlock);
    builder.createCallOp(loc, unregisterFunc, handle);
    builder.createStore(loc, nullPtr, builder.createGetGlobal(gpuBinGlobal));
    cir::BrOp::create(builder, loc, exitBlock);
  }
  {
    mlir::OpBuilder::InsertionGuard exitGuard(builder);
    builder.setInsertionPointToStart(exitBlock);
    cir::ReturnOp::create(builder, loc);
  }

  return dtor;
}

std::optional<FuncOp> CUDARegisterModulePass::buildCUDARegisterGlobals() {
  if (cudaKernelMap.empty() && cudaDeviceVars.empty())
    return {};

  cir::CIRBaseBuilderTy builder(getContext());
  builder.setInsertionPointToStart(mlirModule.getBody());

  mlir::Location loc = mlirModule.getLoc();
  llvm::StringRef cudaPrefix = getCUDAPrefix(getLangOpts());

  auto voidTy = VoidType::get(&getContext());
  auto voidPtrTy = PointerType::get(voidTy);
  auto voidPtrPtrTy = PointerType::get(voidPtrTy);

  // Create the function:
  //      void __cuda_register_globals(void **fatbinHandle)
  std::string regGlobalFuncName =
      addUnderscoredPrefix(cudaPrefix, "_register_globals");
  auto regGlobalFuncTy = FuncType::get({voidPtrPtrTy}, voidTy);
  FuncOp regGlobalFunc =
      buildRuntimeFunction(builder, regGlobalFuncName, loc, regGlobalFuncTy,
                           /*linkage=*/GlobalLinkageKind::InternalLinkage);
  builder.setInsertionPointToStart(regGlobalFunc.addEntryBlock());

  buildCUDARegisterGlobalFunctions(builder, regGlobalFunc);
  buildCUDARegisterVars(builder, regGlobalFunc);

  ReturnOp::create(builder, loc);
  return regGlobalFunc;
}

void CUDARegisterModulePass::buildCUDARegisterGlobalFunctions(
    cir::CIRBaseBuilderTy &builder, FuncOp regGlobalFunc) {
  mlir::Location loc = mlirModule.getLoc();
  llvm::StringRef cudaPrefix = getCUDAPrefix(getLangOpts());
  cir::CIRDataLayout dataLayout(mlirModule);

  auto voidTy = VoidType::get(&getContext());
  auto voidPtrTy = PointerType::get(voidTy);
  auto voidPtrPtrTy = PointerType::get(voidPtrTy);
  IntType intTy = builder.getSIntNTy(32);
  IntType charTy =
      cir::IntType::get(&getContext(), getTargetInfo().getCharWidth(),
                        /*isSigned=*/false);

  // Extract the GPU binary handle argument.
  mlir::Value fatbinHandle = *regGlobalFunc.args_begin();

  cir::CIRBaseBuilderTy globalBuilder(getContext());
  globalBuilder.setInsertionPointToStart(mlirModule.getBody());

  // Declare CUDA internal functions:
  // int __cudaRegisterFunction(
  //   void **fatbinHandle,
  //   const char *hostFunc,
  //   char *deviceFunc,
  //   const char *deviceName,
  //   int threadLimit,
  //   uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim,
  //   int *wsize
  // )
  // OG doesn't care about the types at all. They're treated as void*.

  FuncOp cudaRegisterFunction = buildRuntimeFunction(
      globalBuilder, addUnderscoredPrefix(cudaPrefix, "RegisterFunction"), loc,
      FuncType::get({voidPtrPtrTy, voidPtrTy, voidPtrTy, voidPtrTy, intTy,
                     voidPtrTy, voidPtrTy, voidPtrTy, voidPtrTy, voidPtrTy},
                    intTy));

  auto makeConstantString = [&](llvm::StringRef str) -> GlobalOp {
    auto strType = ArrayType::get(&getContext(), charTy, 1 + str.size());
    auto tmpString = cir::GlobalOp::create(
        globalBuilder, loc, (".str" + str).str(), strType,
        /*isConstant=*/true, {},
        /*linkage=*/cir::GlobalLinkageKind::PrivateLinkage);

    // We must make the string zero-terminated.
    tmpString.setInitialValueAttr(
        ConstArrayAttr::get(strType, StringAttr::get(str + "\0", strType)));
    tmpString.setPrivate();
    return tmpString;
  };

  cir::ConstantOp cirNullPtr = builder.getNullPtr(voidPtrTy, loc);
  bool isHIP = getLangOpts().HIP;
  for (auto kernelName : cudaKernelMap.keys()) {
    FuncOp deviceStub = cudaKernelMap[kernelName];
    GlobalOp deviceFuncStr = makeConstantString(kernelName);
    mlir::Value deviceFunc = builder.createBitcast(
        builder.createGetGlobal(deviceFuncStr), voidPtrTy);

    mlir::Value hostFunc;
    if (isHIP) {
      // Under HIP, the kernel-handle is a GlobalOp shadow created by CIR
      // codegen and named with the kernel-reference mangled name (e.g.
      // `@_Z2fnv` pointing at the device-stub function
      // `_Z17__device_stub__fnv`). The CUDAKernelNameAttr on the device-stub
      // uses the same name, so we can resolve the shadow by symbol lookup.
      auto funcHandle = cast<GlobalOp>(mlirModule.lookupSymbol(kernelName));
      hostFunc =
          builder.createBitcast(builder.createGetGlobal(funcHandle), voidPtrTy);
    } else {
      hostFunc = builder.createBitcast(
          GetGlobalOp::create(
              builder, loc, PointerType::get(deviceStub.getFunctionType()),
              mlir::FlatSymbolRefAttr::get(deviceStub.getSymNameAttr())),
          voidPtrTy);
    }
    builder.createCallOp(
        loc, cudaRegisterFunction,
        {fatbinHandle, hostFunc, deviceFunc, deviceFunc,
         ConstantOp::create(builder, loc, IntAttr::get(intTy, -1)), cirNullPtr,
         cirNullPtr, cirNullPtr, cirNullPtr, cirNullPtr});
  }
}

// Emit `__{cuda|hip}RegisterVar` calls inside `__{cuda|hip}_register_globals`
// for every device-side shadow that carries a `cu.var_registration` attribute
// (attached by `CIRGenNVCUDARuntime::handleVarRegistration`).
void CUDARegisterModulePass::buildCUDARegisterVars(
    cir::CIRBaseBuilderTy &builder, FuncOp regGlobalFunc) {
  mlir::Location loc = mlirModule.getLoc();
  llvm::StringRef cudaPrefix = getCUDAPrefix(getLangOpts());
  cir::CIRDataLayout dataLayout(mlirModule);

  PointerType voidPtrTy = builder.getVoidPtrTy();
  PointerType voidPtrPtrTy = builder.getPointerTo(voidPtrTy);
  IntType intTy = builder.getSIntNTy(32);
  IntType sizeTy = builder.getUIntNTy(getTargetInfo().getMaxPointerWidth());
  IntType charTy =
      cir::IntType::get(&getContext(), getTargetInfo().getCharWidth(),
                        /*isSigned=*/false);

  if (cudaDeviceVars.empty())
    return;

  cir::CIRBaseBuilderTy globalBuilder(getContext());
  globalBuilder.setInsertionPointToStart(mlirModule.getBody());

  // void __{cuda|hip}RegisterVar(void **fatbinHandle,
  //                              char *hostVar, char *deviceAddress,
  //                              const char *deviceName, int ext,
  //                              size_t size, int constant, int normalized);
  // OG ignores parameter types, treating pointers as void*.
  cir::VoidType voidTy = builder.getVoidTy();
  FuncOp cudaRegisterVar = buildRuntimeFunction(
      globalBuilder, addUnderscoredPrefix(cudaPrefix, "RegisterVar"), loc,
      FuncType::get({voidPtrPtrTy, voidPtrTy, voidPtrTy, voidPtrTy, intTy,
                     sizeTy, intTy, intTy},
                    voidTy));

  auto makeConstantString = [&](llvm::StringRef str) -> GlobalOp {
    auto strType = ArrayType::get(&getContext(), charTy, 1 + str.size());
    auto tmpString = cir::GlobalOp::create(
        globalBuilder, loc, (".str" + str).str(), strType,
        /*isConstant=*/true, {},
        /*linkage=*/cir::GlobalLinkageKind::PrivateLinkage);
    tmpString.setInitialValueAttr(
        ConstArrayAttr::get(strType, StringAttr::get(str + "\0", strType)));
    tmpString.setPrivate();
    return tmpString;
  };

  mlir::Value fatbinHandle = *regGlobalFunc.args_begin();

  for (auto &[global, regAttr] : cudaDeviceVars) {
    switch (regAttr.getKind()) {
    case cir::CUDADeviceVarKind::Variable:
      break;
    case cir::CUDADeviceVarKind::Surface:
      llvm_unreachable("Surface registration NYI");
    case cir::CUDADeviceVarKind::Texture:
      llvm_unreachable("Texture registration NYI");
    }

    if (regAttr.getIsManaged())
      llvm_unreachable("Managed variable registration NYI");

    GlobalOp deviceNameStr = makeConstantString(regAttr.getDeviceSideName());
    mlir::Value deviceName = builder.createBitcast(
        builder.createGetGlobal(deviceNameStr), voidPtrTy);
    mlir::Value hostVar =
        builder.createBitcast(builder.createGetGlobal(global), voidPtrTy);

    auto isExtern = ConstantOp::create(
        builder, loc, IntAttr::get(intTy, regAttr.getIsExtern() ? 1 : 0));
    llvm::TypeSize size = dataLayout.getTypeAllocSize(global.getSymType());
    auto varSize = ConstantOp::create(
        builder, loc, IntAttr::get(sizeTy, size.getFixedValue()));
    auto isConstant = ConstantOp::create(
        builder, loc, IntAttr::get(intTy, regAttr.getIsConstant() ? 1 : 0));
    auto normalized = ConstantOp::create(builder, loc, IntAttr::get(intTy, 0));
    builder.createCallOp(loc, cudaRegisterVar,
                         {fatbinHandle, hostVar, deviceName, deviceName,
                          isExtern, varSize, isConstant, normalized});
  }
}

void CUDARegisterModulePass::runOnOperation() {
  mlirModule = getOperation();
  // Nothing is registered without a device binary, so most modules stop here
  // without building a LowerModule.
  if (!mlirModule->hasAttr(CIRDialect::getCUDADeviceBinaryAttrName()))
    return;
  lowerModule = cir::createLowerModule(mlirModule);
  if (!lowerModule || !getLangOpts().CUDA || getLangOpts().CUDAIsDevice)
    return;

  // Stubs and registered variables are module-level symbols.
  for (cir::FuncOp fnOp : mlirModule.getOps<cir::FuncOp>())
    if (auto kernelNameAttr = fnOp->getAttrOfType<CUDAKernelNameAttr>(
            CUDAKernelNameAttr::getMnemonic()))
      cudaKernelMap[kernelNameAttr.getKernelName()] = fnOp;
  for (cir::GlobalOp glob : mlirModule.getOps<cir::GlobalOp>())
    if (auto regAttr = glob->getAttrOfType<CUDAVarRegistrationInfoAttr>(
            CUDAVarRegistrationInfoAttr::getMnemonic()))
      cudaDeviceVars.emplace_back(glob, regAttr);

  buildCUDAModuleCtor();
  // The fatbin global now references the same attribute; drop the module's
  // reference so an emitted .cir doesn't print the bytes twice. This has to
  // happen out here because the ctor and both dtor builders test the
  // attribute to decide whether a device-side binary exists at all.
  mlirModule->removeAttr(CIRDialect::getCUDADeviceBinaryAttrName());
}

std::unique_ptr<Pass> mlir::createCUDARegisterModulePass() {
  return std::make_unique<CUDARegisterModulePass>();
}
