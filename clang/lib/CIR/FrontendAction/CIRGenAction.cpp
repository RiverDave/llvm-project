//===--- CIRGenAction.cpp - LLVM Code generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/FrontendAction/CIRGenAction.h"
#include "CIRDiagnosticHandler.h"
#include "TargetLowering/LowerModule.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "clang/Basic/DiagnosticCodeGen.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/CIR/CIRToCIRPasses.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/OpenMP/RegisterOpenMPExtensions.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/CodeGen/ModuleLinker.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendOptions.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/Internalize.h"

using namespace cir;
using namespace clang;

namespace cir {

static BackendAction
getBackendActionFromOutputType(CIRGenAction::OutputType Action) {
  switch (Action) {
  case CIRGenAction::OutputType::EmitCIR:
    assert(false &&
           "Unsupported output type for getBackendActionFromOutputType!");
    break; // Unreachable, but fall through to report that
  case CIRGenAction::OutputType::EmitAssembly:
    return BackendAction::Backend_EmitAssembly;
  case CIRGenAction::OutputType::EmitBC:
    return BackendAction::Backend_EmitBC;
  case CIRGenAction::OutputType::EmitLLVM:
    return BackendAction::Backend_EmitLL;
  case CIRGenAction::OutputType::EmitObj:
    return BackendAction::Backend_EmitObj;
  }
  // We should only get here if a non-enum value is passed in or we went through
  // the assert(false) case above
  llvm_unreachable("Unsupported output type!");
}

static std::unique_ptr<llvm::Module> lowerFromCIRToLLVMIR(
    mlir::ModuleOp MLIRModule, llvm::LLVMContext &LLVMCtx,
    bool EnableOpenMP, llvm::StringRef mlirSaveTempsOutFile = {},
    llvm::vfs::FileSystem *fs = nullptr, bool enableOffloadSplit = false,
    llvm::ArrayRef<std::string> offloadArchs = {},
    bool isDeviceCompilation = false, unsigned deviceOptLevel = 2,
    const cir::CIROffloadConfig &offloadConfig = {}) {
  return direct::lowerDirectlyFromCIRToLLVMIR(
      MLIRModule, LLVMCtx, EnableOpenMP, mlirSaveTempsOutFile, fs,
      enableOffloadSplit, offloadArchs, isDeviceCompilation, deviceOptLevel,
      offloadConfig);
}

static void registerDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::BuiltinDialect, cir::CIRDialect,
                  mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect,
                  mlir::DLTIDialect, mlir::omp::OpenMPDialect>();
  cir::omp::registerOpenMPExtensions(registry);
}

static void prepareCIRInputContext(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  registerDialects(registry);
  context.appendDialectRegistry(registry);
  context.loadDialect<cir::CIRDialect, mlir::memref::MemRefDialect,
                      mlir::LLVM::LLVMDialect, mlir::DLTIDialect,
                      mlir::omp::OpenMPDialect>();
}

static void reportError(CompilerInstance &CI, llvm::Twine message) {
  unsigned DiagID =
      CI.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error, "%0");
  CI.getDiagnostics().Report(DiagID) << message.str();
}

static mlir::OwningOpRef<mlir::ModuleOp>
parseCIRInput(CompilerInstance &CI, mlir::MLIRContext &context,
              llvm::MemoryBufferRef input) {
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(input.getBuffer(),
                                           input.getBufferIdentifier()),
      llvm::SMLoc());

  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);
  mlir::ParserConfig parserConfig(&context, /*verifyAfterParse=*/false);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, parserConfig);
  if (!module) {
    reportError(CI, "failed to parse CIR input");
    return {};
  }
  if (mlir::failed(mlir::verify(*module))) {
    reportError(CI, "failed to verify CIR input");
    return {};
  }
  return module;
}

static bool linkInModules(CompilerInstance &CI, CodeGenOptions &CGO,
                          llvm::Module &M,
                          SmallVectorImpl<::clang::LinkModule> &LinkModules) {
  for (auto &LM : LinkModules) {
    assert(LM.Module && "LinkModule does not actually have a module");

    if (LM.PropagateAttrs)
      for (llvm::Function &F : *LM.Module) {
        if (F.isIntrinsic())
          continue;
        clang::CodeGen::mergeDefaultFunctionDefinitionAttributes(
            F, CGO, CI.getLangOpts(), CI.getTargetOpts(), LM.Internalize);
      }

    bool Err;
    if (LM.Internalize) {
      Err = llvm::Linker::linkModules(
          M, std::move(LM.Module), LM.LinkFlags,
          [](llvm::Module &M, const llvm::StringSet<> &GVS) {
            llvm::internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
              return !GV.hasName() || (GVS.count(GV.getName()) == 0);
            });
          });
    } else {
      Err = llvm::Linker::linkModules(M, std::move(LM.Module), LM.LinkFlags);
    }

    if (Err)
      return true;
  }

  LinkModules.clear();
  return false;
}

// Build a LowerModule from the surrounding cc1 invocation. Used both for
// in-process CIRGen (where the same TargetInfo also drives the AST) and for
// the .cir input path, where there is no AST at all.
static std::unique_ptr<cir::LowerModule>
makeLowerModuleFromInvocation(CompilerInstance &CI, mlir::ModuleOp module) {
  // Clone TargetInfo: LowerModule takes ownership and CI keeps its copy.
  auto target = std::unique_ptr<clang::TargetInfo>(
      clang::TargetInfo::CreateTargetInfo(CI.getDiagnostics(),
                                          CI.getInvocation().getTargetOpts()));
  if (!target)
    return nullptr;
  // On the .cir resume path the input language is CIR, so LangOpts.HIP is unset
  // even for HIP. Recover the device flavor from the aux-triple (amdgcn for HIP)
  // so the registration pass emits the HIP runtime calls instead of CUDA's.
  clang::LangOptions langOpts = CI.getLangOpts();
  bool auxIsAMDGPU = !CI.getFrontendOpts().AuxTriple.empty() &&
                     llvm::Triple(CI.getFrontendOpts().AuxTriple).isAMDGPU();
  bool otherAMDGPUOffload =
      langOpts.OpenMP || langOpts.SYCLIsDevice || langOpts.SYCLIsHost;
  assert(!(otherAMDGPUOffload && auxIsAMDGPU) &&
         "non-HIP AMDGPU offload (OpenMP/SYCL) is not supported on the .cir "
         "resume path");
  if (!langOpts.HIP && auxIsAMDGPU)
    langOpts.HIP = true;
  return cir::createLowerModule(module, langOpts, CI.getCodeGenOpts(),
                                std::move(target));
}

// On the .cir resume path the fatbin only exists at this second cc1 invocation
// (it is produced downstream of the serialized host.cir). Stamp its path so the
// registration pass can find it. Returns true if a binary was passed.
static bool stampCUDABinaryHandle(CompilerInstance &CI,
                                  mlir::ModuleOp mlirModule,
                                  mlir::MLIRContext &mlirContext) {
  llvm::StringRef cudaBinaryName = CI.getCodeGenOpts().CudaGpuBinaryFileName;
  if (cudaBinaryName.empty())
    return false;

  mlirModule->setAttr(
      cir::CIRDialect::getCUDABinaryHandleAttrName(),
      cir::CUDABinaryHandleAttr::get(
          &mlirContext, mlir::StringAttr::get(&mlirContext, cudaBinaryName)));
  return true;
}

// LoweringPrepare already ran (pre-serialization) on this module, so run only
// the registration step here, sourcing target facts from a LowerModule built
// off the surrounding invocation.
static bool runCUDARegisterModulePass(CompilerInstance &CI,
                                      mlir::ModuleOp mlirModule,
                                      mlir::MLIRContext &mlirContext) {
  std::unique_ptr<cir::LowerModule> lowerModule =
      makeLowerModuleFromInvocation(CI, mlirModule);
  if (!lowerModule) {
    reportError(CI, "failed to build LowerModule for CUDA registration");
    return true;
  }

  mlir::PassManager pm(&mlirContext);
  pm.addPass(mlir::createCUDARegisterModulePass(lowerModule.get(),
                                                &CI.getVirtualFileSystem()));
  if (mlir::failed(pm.run(mlirModule))) {
    reportError(CI, "failed to run CUDA registration pass");
    return true;
  }
  return false;
}

class CIRGenConsumer : public clang::ASTConsumer {

  virtual void anchor();

  CIRGenAction::OutputType Action;

  CompilerInstance &CI;

  std::unique_ptr<raw_pwrite_stream> OutputStream;

  ASTContext *Context{nullptr};
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
  std::unique_ptr<CIRGenerator> Gen;
  const FrontendOptions &FEOptions;
  CodeGenOptions &CGO;

  llvm::LLVMContext &LLVMCtx;
  SmallVectorImpl<::clang::LinkModule> &LinkModules;

  std::optional<CIRDiagnosticHandler> MLIRDiagHandler;

public:
  CIRGenConsumer(CIRGenAction::OutputType Action, CompilerInstance &CI,
                 CodeGenOptions &CGO, std::unique_ptr<raw_pwrite_stream> OS,
                 llvm::LLVMContext &LLVMCtx,
                 SmallVectorImpl<::clang::LinkModule> &LinkModules)
      : Action(Action), CI(CI), OutputStream(std::move(OS)),
        FS(&CI.getVirtualFileSystem()),
        Gen(std::make_unique<CIRGenerator>(CI.getDiagnostics(), std::move(FS),
                                           CI.getCodeGenOpts())),
        FEOptions(CI.getFrontendOpts()), CGO(CGO), LLVMCtx(LLVMCtx),
        LinkModules(LinkModules) {}

  void Initialize(ASTContext &Ctx) override {
    assert(!Context && "initialized multiple times");
    Context = &Ctx;
    Gen->Initialize(Ctx);
    // Install the MLIR diagnostic handler now that CIRGenerator owns its
    // MLIRContext. Lifetime is tied to this consumer, which spans CIRGen,
    // CIR-to-CIR passes, and CIR-to-LLVM lowering.
    MLIRDiagHandler.emplace(&Gen->getMLIRContext(), CI.getDiagnostics(),
                            CI.getSourceManager(), CI.getFileManager());
  }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    Gen->HandleTopLevelDecl(D);
    return true;
  }

  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *VD) override {
    Gen->HandleCXXStaticMemberVarInstantiation(VD);
  }

  void HandleOpenACCRoutineReference(const FunctionDecl *FD,
                                     const OpenACCRoutineDecl *RD) override {
    Gen->HandleOpenACCRoutineReference(FD, RD);
  }

  void HandleInlineFunctionDefinition(FunctionDecl *D) override {
    Gen->HandleInlineFunctionDefinition(D);
  }

  void HandleTranslationUnit(ASTContext &C) override {
    Gen->HandleTranslationUnit(C);

    // In two-pass offload host-emit-CIR mode the module is intentionally
    // partial: gpu.launch_func ops reference kernel symbols that live in the
    // device cc1's CIR and are merged in by MergeOffloadModules.  Skip the
    // pre-pass verifier here; it runs again after the merge step.
    bool skipVerifier =
        FEOptions.ClangIRDisableCIRVerifier || CGO.ClangIROffloadHostEmitCIR;
    if (!skipVerifier) {
      if (!Gen->verifyModule()) {
        // Verifier output already routed through ClangIRDiagnosticHandler.
        // Only emit the generic fatal if nothing more specific was reported.
        if (!CI.getDiagnostics().hasErrorOccurred())
          CI.getDiagnostics().Report(
              diag::err_cir_verification_failed_pre_passes);
        llvm::report_fatal_error(
            "CIR codegen: module verification error before running CIR passes");
        return;
      }
    }

    mlir::ModuleOp MlirModule = Gen->getModule();
    mlir::MLIRContext &MlirCtx = Gen->getMLIRContext();

    if (!FEOptions.ClangIRDisablePasses) {
      std::string LibOptOptions = FEOptions.ClangIRLibOptOptions;

      // Setup and run CIR pipeline.
      const bool EnableLibOpt =
          FEOptions.ClangIRLibOptEnabled && (CGO.OptimizationLevel > 0);
      std::unique_ptr<cir::LowerModule> lowerModule =
          makeLowerModuleFromInvocation(CI, MlirModule);
      if (!lowerModule) {
        CI.getDiagnostics().Report(diag::err_cir_to_cir_transform_failed);
        return;
      }
      // In two-pass host-emit-CIR mode the module is partial (no device
      // functions yet), so suppress per-pass verification to avoid false
      // failures from gpu.launch_func referencing not-yet-merged symbols.
      bool enableVerifier = !FEOptions.ClangIRDisableCIRVerifier &&
                            !CGO.ClangIROffloadHostEmitCIR;
      if (runCIRToCIRPasses(MlirModule, MlirCtx, C, *lowerModule,
                            &CI.getVirtualFileSystem(), enableVerifier,
                            FEOptions.ClangIREnableIdiomRecognizer,
                            CGO.OptimizationLevel > 0, EnableLibOpt,
                            LibOptOptions, FEOptions.ClangIRCallConvLowering)
              .failed()) {
        // Pass-side errors already routed through ClangIRDiagnosticHandler.
        // Skip the generic catch-all if a specific diagnostic was emitted.
        if (!CI.getDiagnostics().hasErrorOccurred())
          CI.getDiagnostics().Report(diag::err_cir_to_cir_transform_failed);
        return;
      }
    }

    // Collect GPU arch(s) and attach offload.target to the module so it
    // survives serialization to a .cir file.  The -x cir path reads it back
    // when the arch isn't available from cc1 flags.
    llvm::SmallVector<std::string> offloadArchs;
    if (CGO.ClangIROffload) {
      llvm::StringRef primaryCPU = C.getTargetInfo().getTargetOpts().CPU;
      if (!primaryCPU.empty() && !primaryCPU.starts_with("x86") &&
          !primaryCPU.starts_with("generic") && primaryCPU != "x86-64") {
        offloadArchs.push_back(primaryCPU.str()); // device cc1
      } else if (!CGO.ClangIROffloadArch.empty()) {
        offloadArchs.push_back(CGO.ClangIROffloadArch); // host cc1
      }
      if (!offloadArchs.empty()) {
        mlir::Builder b(MlirModule.getContext());
        llvm::SmallVector<mlir::Attribute> archAttrs;
        for (const std::string &arch : offloadArchs)
          archAttrs.push_back(b.getStringAttr(arch));
        // Store arch list as a plain ArrayAttr of StringAttr under
        // "offload.target".
        MlirModule->setAttr("offload.target", b.getArrayAttr(archAttrs));
      }
    }

    // Attach fast-math flags as module attributes so they survive .cir
    // serialization and can be read by the -x cir merge cc1.
    {
      const auto &LO = C.getLangOpts();
      mlir::Builder b(MlirModule.getContext());
      if (LO.FastMath || LO.UnsafeFPMath)
        MlirModule->setAttr("cir.unsafe_fp_math", b.getUnitAttr());
      if (LO.FastMath || (LO.NoHonorNaNs && LO.NoHonorInfs))
        MlirModule->setAttr("cir.finite_math_only", b.getUnitAttr());
      if (LO.getDefaultFPContractMode() == LangOptions::FPM_Fast ||
          LO.getDefaultFPContractMode() == LangOptions::FPM_FastHonorPragmas)
        MlirModule->setAttr("cir.fp_contract_fast", b.getUnitAttr());
    }

    switch (Action) {
    case CIRGenAction::OutputType::EmitCIR:
      if (OutputStream && MlirModule) {
        mlir::OpPrintingFlags Flags;
        Flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
        MlirModule->print(*OutputStream, Flags);
      }
      break;
    case CIRGenAction::OutputType::EmitLLVM:
    case CIRGenAction::OutputType::EmitBC:
    case CIRGenAction::OutputType::EmitObj:
    case CIRGenAction::OutputType::EmitAssembly: {
      StringRef saveTempsPrefix = CGO.SaveTempsFilePrefix;
      std::string cirSaveTempsOutFile, mlirSaveTempsOutFile;
      if (!saveTempsPrefix.empty()) {
        SmallString<128> stem(saveTempsPrefix);
        llvm::sys::path::replace_extension(stem, "cir");
        cirSaveTempsOutFile = std::string(stem);
        llvm::sys::path::replace_extension(stem, "mlir");
        mlirSaveTempsOutFile = std::string(stem);
      }

      if (!cirSaveTempsOutFile.empty()) {
        std::error_code ec;
        llvm::raw_fd_ostream out(cirSaveTempsOutFile, ec);
        if (!ec)
          MlirModule->print(out);
      }

      bool isDeviceCompilation = C.getLangOpts().CUDAIsDevice;
      cir::CIROffloadConfig offloadConfig;
      offloadConfig.tightenLaunchBounds = !CGO.ClangIRNoTightenLaunchBounds;
      offloadConfig.propagateBlockShape = !CGO.ClangIRNoPropagateBlockShape;
      offloadConfig.propagatePointerFacts =
          !CGO.ClangIRNoPropagatePointerFacts;
      offloadConfig.propagateGridCoverage =
          !CGO.ClangIRNoPropagateGridCoverage;
      offloadConfig.specializeSharedMemory =
          !CGO.ClangIRNoSpecializeSharedMemory;
      offloadConfig.specializeScalarArgs =
          !CGO.ClangIRNoSpecializeScalarArgs;
      offloadConfig.multiversionDivisibility =
          !CGO.ClangIRNoMultiversionDivisibility;
      if (!CGO.ClangIRDeadKernelAction.empty()) {
        if (CGO.ClangIRDeadKernelAction == "discard")
          offloadConfig.deadKernelAction = cir::DeadKernelAction::Discard;
        else if (CGO.ClangIRDeadKernelAction == "none")
          offloadConfig.deadKernelAction = cir::DeadKernelAction::None;
      } else if (CGO.ClangIRNoDeadKernelElim) {
        offloadConfig.deadKernelAction = cir::DeadKernelAction::None;
      }
      offloadConfig.deadArgElimination = !CGO.ClangIRNoDeadArgElim;
      offloadConfig.promoteConstantArgs = CGO.ClangIRPromoteConstantArgs;
      offloadConfig.unrollBarrierLoops = !CGO.ClangIRNoUnrollBarrierLoops;
      const auto &LO = C.getLangOpts();
      offloadConfig.unsafeMathOpt = LO.FastMath || LO.UnsafeFPMath;
      offloadConfig.finiteOnly =
          LO.FastMath || (LO.NoHonorNaNs && LO.NoHonorInfs);
      offloadConfig.daz = LO.FastMath || LO.UnsafeFPMath;
      std::unique_ptr<llvm::Module> LLVMModule = lowerFromCIRToLLVMIR(
          MlirModule, LLVMCtx, C.getLangOpts().OpenMP, mlirSaveTempsOutFile,
          &CI.getVirtualFileSystem(), CGO.ClangIROffload, offloadArchs,
          isDeviceCompilation, CGO.OptimizationLevel, offloadConfig);

      if (linkInModules(CI, CGO, *LLVMModule, LinkModules))
        return;

      BackendAction BEAction = getBackendActionFromOutputType(Action);
      emitBackendOutput(
          CI, CI.getCodeGenOpts(), C.getTargetInfo().getDataLayoutString(),
          LLVMModule.get(), BEAction, FS, std::move(OutputStream));
      break;
    }
    }
  }

  void HandleTagDeclDefinition(TagDecl *D) override {
    PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                   Context->getSourceManager(),
                                   "CIR generation of declaration");
    Gen->HandleTagDeclDefinition(D);
  }

  void HandleTagDeclRequiredDefinition(const TagDecl *D) override {
    Gen->HandleTagDeclRequiredDefinition(D);
  }

  void CompleteTentativeDefinition(VarDecl *D) override {
    Gen->CompleteTentativeDefinition(D);
  }

  void HandleVTable(CXXRecordDecl *RD) override { Gen->HandleVTable(RD); }
};
} // namespace cir

void CIRGenConsumer::anchor() {}

CIRGenAction::CIRGenAction(OutputType Act, mlir::MLIRContext *MLIRCtx)
    : MLIRCtx(MLIRCtx ? MLIRCtx : new mlir::MLIRContext),
      Ctx(std::make_unique<llvm::LLVMContext>()), Action(Act) {}

CIRGenAction::~CIRGenAction() { MLIRMod.release(); }

bool CIRGenAction::BeginSourceFileAction(CompilerInstance &CI) {
  if (clang::loadLinkModules(CI, *Ctx, LinkModules))
    return false;
  return ASTFrontendAction::BeginSourceFileAction(CI);
}

static std::unique_ptr<raw_pwrite_stream>
getOutputStream(CompilerInstance &CI, StringRef InFile,
                CIRGenAction::OutputType Action) {
  switch (Action) {
  case CIRGenAction::OutputType::EmitAssembly:
    return CI.createDefaultOutputFile(false, InFile, "s");
  case CIRGenAction::OutputType::EmitCIR:
    return CI.createDefaultOutputFile(false, InFile, "cir");
  case CIRGenAction::OutputType::EmitLLVM:
    return CI.createDefaultOutputFile(false, InFile, "ll");
  case CIRGenAction::OutputType::EmitBC:
    return CI.createDefaultOutputFile(true, InFile, "bc");
  case CIRGenAction::OutputType::EmitObj:
    return CI.createDefaultOutputFile(true, InFile, "o");
  }
  llvm_unreachable("Invalid CIRGenAction::OutputType");
}

bool CIRGenAction::hasCIRSupport() const { return true; }

void CIRGenAction::ExecuteAction() {
  if (getCurrentFileKind().getLanguage() != Language::CIR) {
    ASTFrontendAction::ExecuteAction();
    return;
  }

  CompilerInstance &CI = getCompilerInstance();
  std::unique_ptr<llvm::raw_pwrite_stream> OS = CI.takeOutputStream();
  if (!OS)
    OS = getOutputStream(CI, getCurrentFileOrBufferName(), Action);
  if (!OS)
    return;

  SourceManager &SM = CI.getSourceManager();
  std::optional<llvm::MemoryBufferRef> MainFile =
      SM.getBufferOrNone(SM.getMainFileID());
  if (!MainFile)
    return;

  prepareCIRInputContext(*MLIRCtx);
  MLIRMod = parseCIRInput(CI, *MLIRCtx, *MainFile);
  if (!MLIRMod)
    return;

  // Stamp before the emit-cir early return so `-emit-cir` reflects the handle.
  bool hasCUDABinary = stampCUDABinaryHandle(CI, *MLIRMod, *MLIRCtx);

  if (Action == OutputType::EmitCIR) {
    mlir::OpPrintingFlags Flags;
    Flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);
    MLIRMod->print(*OS, Flags);
    return;
  }

  if (hasCUDABinary && runCUDARegisterModulePass(CI, *MLIRMod, *MLIRCtx))
    return;

  std::string mlirSaveTempsOutFile;
  if (!CI.getCodeGenOpts().SaveTempsFilePrefix.empty()) {
    SmallString<128> stem(CI.getCodeGenOpts().SaveTempsFilePrefix);
    llvm::sys::path::replace_extension(stem, "mlir");
    mlirSaveTempsOutFile = std::string(stem);
  }

  const CodeGenOptions &CGO = CI.getCodeGenOpts();
  llvm::SmallVector<std::string> offloadArchs;
  if (CGO.ClangIROffload) {
    if (!CGO.ClangIROffloadArch.empty())
      offloadArchs.push_back(CGO.ClangIROffloadArch);
    // Fall back to offload.target on the parsed module (stamped during codegen
    // or when the .cir file was written).
    if (offloadArchs.empty())
      if (auto archs = MLIRMod->getAttrOfType<mlir::ArrayAttr>("offload.target"))
        for (auto arch : archs)
          if (auto s = mlir::dyn_cast<mlir::StringAttr>(arch))
            offloadArchs.push_back(s.getValue().str());
  }
  cir::CIROffloadConfig offloadConfig;
  offloadConfig.tightenLaunchBounds = !CGO.ClangIRNoTightenLaunchBounds;
  offloadConfig.propagateBlockShape = !CGO.ClangIRNoPropagateBlockShape;
  offloadConfig.propagatePointerFacts = !CGO.ClangIRNoPropagatePointerFacts;
  offloadConfig.propagateGridCoverage = !CGO.ClangIRNoPropagateGridCoverage;
  offloadConfig.specializeSharedMemory = !CGO.ClangIRNoSpecializeSharedMemory;
  offloadConfig.specializeScalarArgs = !CGO.ClangIRNoSpecializeScalarArgs;
  offloadConfig.multiversionDivisibility =
      !CGO.ClangIRNoMultiversionDivisibility;
  if (!CGO.ClangIRDeadKernelAction.empty()) {
    if (CGO.ClangIRDeadKernelAction == "discard")
      offloadConfig.deadKernelAction = cir::DeadKernelAction::Discard;
    else if (CGO.ClangIRDeadKernelAction == "none")
      offloadConfig.deadKernelAction = cir::DeadKernelAction::None;
  } else if (CGO.ClangIRNoDeadKernelElim) {
    offloadConfig.deadKernelAction = cir::DeadKernelAction::None;
  }
  offloadConfig.deadArgElimination = !CGO.ClangIRNoDeadArgElim;
  offloadConfig.promoteConstantArgs = CGO.ClangIRPromoteConstantArgs;
  offloadConfig.unrollBarrierLoops = !CGO.ClangIRNoUnrollBarrierLoops;
  offloadConfig.unsafeMathOpt = MLIRMod->hasAttr("cir.unsafe_fp_math");
  offloadConfig.finiteOnly = MLIRMod->hasAttr("cir.finite_math_only");
  offloadConfig.daz = offloadConfig.unsafeMathOpt;
  offloadConfig.fpContractFast = MLIRMod->hasAttr("cir.fp_contract_fast");
  std::unique_ptr<llvm::Module> LLVMModule = lowerFromCIRToLLVMIR(
      *MLIRMod, *Ctx, /*EnableOpenMP=*/false, mlirSaveTempsOutFile,
      &CI.getVirtualFileSystem(), CGO.ClangIROffload, offloadArchs,
      /*isDeviceCompilation=*/false, CGO.OptimizationLevel, offloadConfig);
  if (!LLVMModule)
    return;

  if (linkInModules(CI, CI.getCodeGenOpts(), *LLVMModule, LinkModules))
    return;

  BackendAction BEAction = getBackendActionFromOutputType(Action);
  emitBackendOutput(CI, CI.getCodeGenOpts(),
                    CI.getTarget().getDataLayoutString(), LLVMModule.get(),
                    BEAction, CI.getFileManager().getVirtualFileSystemPtr(),
                    std::move(OS));
}

std::unique_ptr<ASTConsumer>
CIRGenAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  // For pre-built CIR input (-x cir), we bypass AST parsing entirely in
  // ExecuteAction(). Return a no-op consumer so we don't construct CIRGenModule
  // (which would fail since there are no CUDA/C++ language opts).
  if (getCurrentFileKind().getLanguage() == clang::Language::CIR)
    return std::make_unique<ASTConsumer>();

  std::unique_ptr<llvm::raw_pwrite_stream> Out = CI.takeOutputStream();

  if (!Out)
    Out = getOutputStream(CI, InFile, Action);

  auto Result = std::make_unique<cir::CIRGenConsumer>(
      Action, CI, CI.getCodeGenOpts(), std::move(Out), *Ctx, LinkModules);

  return Result;
}

void EmitAssemblyAction::anchor() {}
EmitAssemblyAction::EmitAssemblyAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitAssembly, MLIRCtx) {}

void EmitCIRAction::anchor() {}
EmitCIRAction::EmitCIRAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitCIR, MLIRCtx) {}

void EmitLLVMAction::anchor() {}
EmitLLVMAction::EmitLLVMAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitLLVM, MLIRCtx) {}

void EmitBCAction::anchor() {}
EmitBCAction::EmitBCAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitBC, MLIRCtx) {}

void EmitObjAction::anchor() {}
EmitObjAction::EmitObjAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitObj, MLIRCtx) {}
