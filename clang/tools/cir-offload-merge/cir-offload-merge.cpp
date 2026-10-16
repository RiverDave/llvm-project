//===-- cir-offload-merge/cir-offload-merge.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This tool merges target-specific CIR modules into a single top-level module
// marked as an offload container (carrying the `cir.offload.container` unit
// attribute), and can split such a module back into its nested host and device
// modules.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/OpenMP/RegisterOpenMPExtensions.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "clang/Driver/OffloadBundler.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>

namespace {

llvm::cl::OptionCategory CIROffloadMergeCategory("cir-offload-merge options");

llvm::cl::opt<bool> Combine("combine", llvm::cl::desc("Combine CIR inputs"),
                            llvm::cl::cat(CIROffloadMergeCategory));

llvm::cl::opt<bool> Split("split", llvm::cl::desc("Split combined CIR input"),
                          llvm::cl::cat(CIROffloadMergeCategory));

llvm::cl::list<std::string>
    InputFileNames("input",
                   llvm::cl::desc("Input CIR file. Can be specified multiple "
                                  "times for multiple input files."),
                   llvm::cl::cat(CIROffloadMergeCategory));

llvm::cl::list<std::string>
    TargetNames("targets", llvm::cl::CommaSeparated,
                llvm::cl::desc("[<offload kind>-<target triple>,...]"),
                llvm::cl::cat(CIROffloadMergeCategory));

llvm::cl::list<std::string>
    OutputFileNames("output",
                    llvm::cl::desc("Output CIR file. Can be specified "
                                   "multiple times in split mode."),
                    llvm::cl::cat(CIROffloadMergeCategory));

llvm::cl::opt<bool> DisableCirInferLaunchBounds(
    "disable-cir-infer-launch-bounds",
    llvm::cl::desc("Disable launch-bound inference from host launch sites"),
    llvm::cl::cat(CIROffloadMergeCategory));
llvm::cl::opt<bool> DisableCirPropKernelArgs(
    "disable-cir-prop-kernel-args",
    llvm::cl::desc("Disable constant kernel-argument propagation"),
    llvm::cl::cat(CIROffloadMergeCategory));

struct InputTarget {
  std::string Input;
  std::string Target;
  bool IsHost = false;
};

struct TargetOutput {
  std::string Target;
  std::string Output;
  bool Written = false;
};

int reportError(const llvm::Twine &message) {
  llvm::errs() << "error: " << message << '\n';
  return 1;
}

std::string sanitizeModuleName(llvm::StringRef target) {
  std::string name = "device_";
  for (char c : target) {
    if (std::isalnum(static_cast<unsigned char>(c)))
      name.push_back(c);
    else
      name.push_back('_');
  }
  return name;
}

std::string makeUnique(llvm::StringRef name,
                       llvm::StringMap<unsigned> &seenNames) {
  unsigned &count = seenNames[name];
  if (count++ == 0)
    return name.str();
  return (name + "_" + llvm::Twine(count - 1)).str();
}

bool isValidOffloadKind(const clang::OffloadTargetInfo &offloadInfo) {
  // OffloadTargetInfo rejects CUDA today, even though CUDA-shaped bundle IDs
  // are valid input for this CIR combine scaffold.
  return offloadInfo.isOffloadKindValid() || offloadInfo.OffloadKind == "cuda";
}

bool isValidTarget(llvm::StringRef target,
                   const clang::OffloadBundlerConfig &bundlerConfig) {
  if (!clang::checkOffloadBundleID(target))
    return false;

  clang::OffloadTargetInfo offloadInfo(target, bundlerConfig);
  return isValidOffloadKind(offloadInfo) && offloadInfo.isTripleValid();
}

bool isCompatibleTarget(llvm::StringRef bundleID, llvm::StringRef target,
                        const clang::OffloadBundlerConfig &bundlerConfig) {
  clang::OffloadTargetInfo bundleInfo(bundleID, bundlerConfig);
  clang::OffloadTargetInfo targetInfo(target, bundlerConfig);
  if (bundleInfo == targetInfo)
    return true;

  // Match clang-offload-bundler's code-object compatibility policy: exact
  // bundle IDs are not required when the target ID feature sets are compatible.
  if (!bundleInfo.isOffloadKindCompatible(targetInfo.OffloadKind) ||
      !bundleInfo.Triple.isCompatibleWith(targetInfo.Triple))
    return false;

  // Only AMDGPU target IDs carry features (xnack/sramecc); elsewhere the
  // processor has to match exactly.
  if (!bundleInfo.Triple.isAMDGPU())
    return bundleInfo.TargetID == targetInfo.TargetID;

  std::optional<llvm::AMDGPU::TargetID> bundleAMDGPUID =
      llvm::AMDGPU::TargetID::parse(bundleInfo.Triple, bundleInfo.TargetID);
  std::optional<llvm::AMDGPU::TargetID> targetAMDGPUID =
      llvm::AMDGPU::TargetID::parse(targetInfo.Triple, targetInfo.TargetID);
  if (!bundleAMDGPUID || !targetAMDGPUID ||
      bundleAMDGPUID->getGPUKind() != targetAMDGPUID->getGPUKind())
    return false;

  // A feature the bundle leaves unspecified matches anything; one it sets
  // must match the target's setting.
  auto featureCompatible = [](llvm::AMDGPU::TargetIDSetting bundle,
                              llvm::AMDGPU::TargetIDSetting target) {
    return (bundle != llvm::AMDGPU::TargetIDSetting::On &&
            bundle != llvm::AMDGPU::TargetIDSetting::Off) ||
           bundle == target;
  };
  return featureCompatible(bundleAMDGPUID->getXnackSetting(),
                           targetAMDGPUID->getXnackSetting()) &&
         featureCompatible(bundleAMDGPUID->getSramEccSetting(),
                           targetAMDGPUID->getSramEccSetting());
}

int validateCombineCommandLine(
    llvm::SmallVectorImpl<InputTarget> &inputTargets) {
  if (Combine == Split)
    return reportError("expected exactly one of -combine or -split");
  if (InputFileNames.empty())
    return reportError("missing required -input");
  if (TargetNames.empty())
    return reportError("missing required -targets");
  if (OutputFileNames.empty())
    return reportError("missing required -output");
  if (OutputFileNames.size() != 1)
    return reportError("combine mode expects exactly one output");
  if (InputFileNames.size() != TargetNames.size())
    return reportError("number of input files and targets should match in "
                       "combine mode");

  clang::OffloadBundlerConfig bundlerConfig;
  llvm::StringSet<> seenTargets;
  unsigned numHostTargets = 0;
  unsigned numDeviceTargets = 0;

  for (auto [input, target] : llvm::zip_equal(InputFileNames, TargetNames)) {
    if (!seenTargets.insert(target).second)
      return reportError("duplicate target '" + target + "'");

    if (!isValidTarget(target, bundlerConfig))
      return reportError("invalid target '" + target + "'");

    clang::OffloadTargetInfo offloadInfo(target, bundlerConfig);
    bool isHost = offloadInfo.hasHostKind();
    if (isHost)
      ++numHostTargets;
    else
      ++numDeviceTargets;

    inputTargets.push_back({input, target, isHost});
  }

  if (numHostTargets != 1)
    return reportError("expected exactly one host target");
  if (numDeviceTargets == 0)
    return reportError("expected at least one device target");

  return 0;
}

int validateSplitCommandLine(
    llvm::SmallVectorImpl<TargetOutput> &targetOutputs) {
  if (Combine == Split)
    return reportError("expected exactly one of -combine or -split");
  if (InputFileNames.empty())
    return reportError("missing required -input");
  if (InputFileNames.size() != 1)
    return reportError("split mode expects exactly one input");
  if (TargetNames.empty())
    return reportError("missing required -targets");
  if (OutputFileNames.empty())
    return reportError("missing required -output");
  if (OutputFileNames.size() != TargetNames.size())
    return reportError("number of output files and targets should match in "
                       "split mode");

  clang::OffloadBundlerConfig bundlerConfig;
  llvm::StringSet<> seenTargets;
  llvm::StringSet<> seenOutputs;

  for (auto [target, output] : llvm::zip_equal(TargetNames, OutputFileNames)) {
    if (!seenTargets.insert(target).second)
      return reportError("duplicate target '" + target + "'");
    if (!seenOutputs.insert(output).second)
      return reportError("duplicate output '" + output + "'");
    if (!isValidTarget(target, bundlerConfig))
      return reportError("invalid target '" + target + "'");

    targetOutputs.push_back({target, output, false});
  }

  return 0;
}

void registerDialects(mlir::DialectRegistry &registry) {
  // Match cir-opt's parser surface: offload CIR inputs may carry OpenMP/LLVM
  // dialect attrs or ops before this tool wraps them in an offload container.
  registry.insert<mlir::BuiltinDialect, cir::CIRDialect,
                  mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect,
                  mlir::DLTIDialect, mlir::omp::OpenMPDialect>();
  cir::omp::registerOpenMPExtensions(registry);
}

mlir::OwningOpRef<mlir::ModuleOp> parseCIRInput(llvm::StringRef inputFileName,
                                                mlir::MLIRContext &context) {
  mlir::ParserConfig parserConfig(&context);
  return mlir::parseSourceFile<mlir::ModuleOp>(inputFileName, parserConfig);
}

void setOffloadAttrs(mlir::ModuleOp cirModule, llvm::StringRef name,
                     cir::OffloadKind offloadKind, llvm::StringRef bundleID) {
  cirModule.setSymName(name);
  cirModule->setAttr(
      cir::CIRDialect::getOffloadKindAttrName(),
      cir::OffloadKindAttr::get(cirModule.getContext(), offloadKind));
  // This is bundler metadata used to recover output routing during split, not
  // target lowering state for the CIR module.
  cirModule->setAttr(cir::CIRDialect::getOffloadBundleIDAttrName(),
                     mlir::StringAttr::get(cirModule.getContext(), bundleID));
}

mlir::OwningOpRef<mlir::ModuleOp>
combineInputs(llvm::ArrayRef<InputTarget> inputTargets,
              mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> hostModule;
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 4> deviceModules;
  llvm::StringMap<unsigned> deviceNames;

  // Host and device compilations name records independently: the same name
  // can carry a different layout on each side (e.g. a member under
  // __CUDA_ARCH__) or even a different record (CIRGen's "anon.N" counter).
  // Give each device module's records its own names.
  std::string &recordNameSuffix =
      context.getLoadedDialect<cir::CIRDialect>()->recordNameSuffix;
  for (const InputTarget &inputTarget : inputTargets) {
    std::string deviceName;
    if (!inputTarget.IsHost)
      deviceName =
          makeUnique(sanitizeModuleName(inputTarget.Target), deviceNames);
    recordNameSuffix = inputTarget.IsHost ? "" : "." + deviceName;
    mlir::OwningOpRef<mlir::ModuleOp> cirModule =
        parseCIRInput(inputTarget.Input, context);
    recordNameSuffix.clear();
    if (!cirModule)
      return {};

    if (inputTarget.IsHost) {
      setOffloadAttrs(*cirModule, "host", cir::OffloadKind::Host,
                      inputTarget.Target);
      hostModule = std::move(cirModule);
      continue;
    }

    setOffloadAttrs(*cirModule, deviceName, cir::OffloadKind::Device,
                    inputTarget.Target);
    deviceModules.push_back(std::move(cirModule));
  }

  auto loc = mlir::UnknownLoc::get(&context);
  // The combined module is itself the offload container: it carries the
  // `cir.offload.container` unit attribute and holds the host and device
  // modules directly, host first, matching the container verifier.
  mlir::OwningOpRef<mlir::ModuleOp> combinedModule(mlir::ModuleOp::create(loc));
  (*combinedModule)->setAttr(cir::CIRDialect::getOffloadContainerAttrName(),
                             mlir::UnitAttr::get(&context));

  // Preserve the container invariant expected by the verifier: the host module
  // is the first nested op, followed by device modules in input order.
  combinedModule->getBody()->push_back(hostModule.release().getOperation());
  for (mlir::OwningOpRef<mlir::ModuleOp> &deviceModule : deviceModules)
    combinedModule->getBody()->push_back(deviceModule.release().getOperation());

  return combinedModule;
}

// Run the offload-container optimization passes on the freshly combined module.
// This is the pipeline seam where cross-boundary opts (DKE, and other passes)
// run while host and device modules are still joined in one container.
//
// The container is a module, so passes are scheduled over the module; DKE and
// KACP early-return on modules that do not carry the `cir.offload.container`
// unit attribute.
int runOffloadOptPasses(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext(), mlir::ModuleOp::getOperationName());
  mlir::OpPassManager &modulePM = pm.nest<mlir::ModuleOp>();

  // Promote stack slots and propagate constants so that a kernel argument known
  // at compile time reaches the device stub call site as a cir.const. Runs on
  // every nested module, host and device.
  modulePM.addPass(mlir::createMem2Reg());
  modulePM.addPass(mlir::createSCCPPass());

  // Container passes run on the container itself; the nested modules carry no
  // container attribute and would make them early-return.
  pm.addPass(mlir::createOffloadDeadKernelEliminationPass());
  if (!DisableCirPropKernelArgs)
    pm.addPass(mlir::createOffloadKernelArgConstantPropagationPass());
  if (!DisableCirInferLaunchBounds)
    pm.addPass(mlir::createOffloadLaunchBoundsPropagationPass());

  if (mlir::failed(pm.run(module)))
    return reportError("offload-container passes failed");
  return 0;
}

int writeModuleToOutput(mlir::ModuleOp module, llvm::StringRef outputFileName) {
  assert(!module->getParentOp() &&
         "module must be top-level for its type aliases to be printed");
  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      mlir::openOutputFile(outputFileName, &errorMessage);
  if (!outputFile)
    return reportError(errorMessage);

  // Emit the combined CIR module as textual MLIR.
  // TODO: Eventually when bytecode support lands for CIR, we should handle
  // such case here.
  module->print(outputFile->os());
  outputFile->os() << '\n';
  outputFile->keep();
  return 0;
}

// The single top-level offload container in `cirModule`, or null if there is
// not exactly one module carrying the `cir.offload.container` unit attribute.
mlir::ModuleOp findContainer(mlir::ModuleOp cirModule) {
  // -combine writes the container as the module itself, so check there first;
  // the nested scan below keeps the hand-written container fixtures working.
  if (cir::isOffloadContainer(cirModule))
    return cirModule;

  mlir::ModuleOp container;
  for (mlir::ModuleOp candidate : cirModule.getOps<mlir::ModuleOp>()) {
    if (!cir::isOffloadContainer(candidate))
      continue;
    if (container)
      return {};
    container = candidate;
  }
  return container;
}

int splitInput(llvm::StringRef inputFileName,
               llvm::MutableArrayRef<TargetOutput> targetOutputs,
               mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> cirModule =
      parseCIRInput(inputFileName, context);
  if (!cirModule)
    return reportError("failed to parse input file");

  if (mlir::failed(mlir::verify(*cirModule)))
    return reportError("failed to verify input module");

  mlir::ModuleOp container = findContainer(*cirModule);
  if (!container)
    return reportError(
        "expected exactly one direct top-level offload container");

  clang::OffloadBundlerConfig bundlerConfig;
  llvm::StringSet<> seenBundleIDs;

  // Host first, then devices: the host is a legitimate split target too, and
  // getOffloadContainerDeviceModules() deliberately skips it. Written modules
  // are detached from the container, so iterate with an early-inc range.
  for (mlir::ModuleOp nestedModule :
       llvm::make_early_inc_range(container.getOps<mlir::ModuleOp>())) {
    auto bundleIDAttr = nestedModule->getAttrOfType<mlir::StringAttr>(
        cir::CIRDialect::getOffloadBundleIDAttrName());
    if (!bundleIDAttr)
      return reportError("nested module is missing 'cir.offload.bundle_id'");

    llvm::StringRef bundleID = bundleIDAttr.getValue();
    if (!seenBundleIDs.insert(bundleID).second)
      return reportError(llvm::Twine("duplicate module bundle ID '") +
                         bundleID + "'");
    if (!isValidTarget(bundleID, bundlerConfig))
      return reportError(llvm::Twine("invalid module bundle ID '") + bundleID +
                         "'");

    // The requested -targets/-output pairs only form the output map. The
    // stored bundle ID decides where this nested module is written.
    TargetOutput *match = nullptr;
    for (TargetOutput &targetOutput : targetOutputs) {
      if (!isCompatibleTarget(bundleID, targetOutput.Target, bundlerConfig))
        continue;
      if (match)
        return reportError(llvm::Twine("module bundle ID '") + bundleID +
                           "' matches multiple requested targets");
      match = &targetOutput;
    }

    if (!match)
      continue;

    if (match->Written)
      return reportError("multiple modules match target '" + match->Target +
                         "'");
    // Print the module as a top-level operation of its own. MLIR emits type
    // aliases only when printing an operation without a parent, so if nested
    // inside the container, every type would be spelled out inline at each use.
    nestedModule->remove();
    mlir::OwningOpRef<mlir::ModuleOp> detachedModule(nestedModule);
    if (int errorCode = writeModuleToOutput(*detachedModule, match->Output))
      return errorCode;
    match->Written = true;
  }

  for (const TargetOutput &targetOutput : targetOutputs)
    if (!targetOutput.Written)
      return reportError("can't find module for target '" +
                         targetOutput.Target + "'");

  return 0;
}

// Appends the scalar leaves of `ty`, placed at `offset`, as "offset:type"
// strings, so that records compare by layout rather than by name or by how
// their padding is spelled.
void appendLayoutLeaves(mlir::Type ty, uint64_t offset,
                        const mlir::DataLayout &dl,
                        llvm::SmallVectorImpl<std::string> &leaves) {
  if (auto record = mlir::dyn_cast<cir::StructType>(ty)) {
    for (auto [i, member] : llvm::enumerate(record.getMembers()))
      if (record.getMemberKinds()[i] != cir::RecordMemberKind::Pad)
        appendLayoutLeaves(member, offset + record.getElementOffset(dl, i), dl,
                           leaves);
    return;
  }
  if (auto array = mlir::dyn_cast<cir::ArrayType>(ty)) {
    mlir::Type elt = array.getElementType();
    uint64_t stride =
        llvm::alignTo(dl.getTypeSize(elt), dl.getTypeABIAlignment(elt));
    for (uint64_t i = 0; i < array.getSize(); ++i)
      appendLayoutLeaves(elt, offset + i * stride, dl, leaves);
    return;
  }
  std::string leaf;
  llvm::raw_string_ostream os(leaf);
  os << offset << ':';
  // Pointers and unions can spell a record name, which differs between
  // modules; compare them by size only.
  if (mlir::isa<cir::IntType, cir::BoolType, cir::FPTypeInterface>(ty))
    os << ty;
  else
    os << dl.getTypeSize(ty) << " bytes";
  leaves.push_back(leaf);
}

// Warns about kernel arguments whose layout differs between the host and a
// device module: the kernel would read the host's bytes with another layout.
// The host side comes from the HIP kernel handle, whose type keeps the source
// signature. CUDA has no handle and its stub is already ABI-lowered, so CUDA
// kernels are not checked.
void checkKernelArgLayouts(mlir::ModuleOp container) {
  mlir::ModuleOp host = cir::getOffloadContainerHostModule(container);
  mlir::DataLayout hostDL(host);
  for (const auto &[name, binding] : cir::KernelBindingTable(container)) {
    auto handle = host.lookupSymbol<cir::GlobalOp>(name);
    auto handleTy = handle
                        ? mlir::dyn_cast<cir::PointerType>(handle.getSymType())
                        : nullptr;
    auto hostFnTy = handleTy
                        ? mlir::dyn_cast<cir::FuncType>(handleTy.getPointee())
                        : nullptr;
    if (!hostFnTy)
      continue;
    for (cir::FuncOp kernel : binding.deviceKernels) {
      auto device = kernel->getParentOfType<mlir::ModuleOp>();
      mlir::DataLayout deviceDL(device);
      llvm::ArrayRef<mlir::Type> deviceTys =
          kernel.getFunctionType().getInputs();
      if (deviceTys.size() != hostFnTy.getNumInputs())
        continue;
      for (auto [i, hostTy, deviceTy] :
           llvm::enumerate(hostFnTy.getInputs(), deviceTys)) {
        llvm::SmallVector<std::string> hostLeaves, deviceLeaves;
        appendLayoutLeaves(hostTy, 0, hostDL, hostLeaves);
        appendLayoutLeaves(deviceTy, 0, deviceDL, deviceLeaves);
        hostLeaves.push_back("size " +
                             std::to_string(hostDL.getTypeSize(hostTy)));
        deviceLeaves.push_back("size " +
                               std::to_string(deviceDL.getTypeSize(deviceTy)));
        auto [h, d] = std::mismatch(hostLeaves.begin(), hostLeaves.end(),
                                    deviceLeaves.begin(), deviceLeaves.end());
        if (h == hostLeaves.end() && d == deviceLeaves.end())
          continue;
        kernel.emitWarning()
            << "argument " << i << " of kernel '" << name
            << "' has a different layout on the host and in "
            << *device.getSymName() << ": the host has '"
            << (h == hostLeaves.end() ? "nothing" : *h)
            << "' where the device has '"
            << (d == deviceLeaves.end() ? "nothing" : *d) << "'";
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::HideUnrelatedOptions(CIROffloadMergeCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "CIR host-device offload merge\n");

  mlir::DialectRegistry registry;
  registerDialects(registry);
  mlir::MLIRContext context;
  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  context.loadDialect<cir::CIRDialect, mlir::memref::MemRefDialect,
                      mlir::LLVM::LLVMDialect, mlir::DLTIDialect,
                      mlir::omp::OpenMPDialect>();
  context.appendDialectRegistry(registry);

  if (Combine) {
    llvm::SmallVector<InputTarget, 4> inputTargets;
    if (int errorCode = validateCombineCommandLine(inputTargets))
      return errorCode;

    mlir::OwningOpRef<mlir::ModuleOp> combinedModule =
        combineInputs(inputTargets, context);
    if (!combinedModule)
      return reportError("failed to parse input file");

    if (mlir::failed(mlir::verify(*combinedModule)))
      return reportError("failed to verify combined module");

    checkKernelArgLayouts(*combinedModule);

    if (int errorCode = runOffloadOptPasses(*combinedModule))
      return errorCode;

    return writeModuleToOutput(*combinedModule, OutputFileNames.front());
  }

  llvm::SmallVector<TargetOutput, 4> targetOutputs;
  if (int errorCode = validateSplitCommandLine(targetOutputs))
    return errorCode;

  return splitInput(InputFileNames.front(), targetOutputs, context);
}