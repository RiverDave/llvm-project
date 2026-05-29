//===-- cir-offload-merge/cir-offload-merge.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This tool merges CIR modules coming from CUDA/HIP programs
/// into a single top-level modulecontaining a cir.offload.container operation.
///
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/OpenMP/RegisterOpenMPExtensions.h"
#include "clang/Driver/OffloadBundler.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <string>

namespace {

llvm::cl::OptionCategory CIROffloadMergeCategory("cir-offload-merge options");

llvm::cl::opt<bool> Combine("combine", llvm::cl::desc("Combine CIR inputs"),
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

llvm::cl::opt<std::string>
    OutputFileName("output", llvm::cl::desc("Combined CIR output file"),
                   llvm::cl::init(""), llvm::cl::cat(CIROffloadMergeCategory));

struct InputTarget {
  std::string Input;
  std::string Target;
  bool IsHost = false;
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

int validateCommandLine(llvm::SmallVectorImpl<InputTarget> &inputTargets) {
  if (!Combine)
    return reportError("missing required --combine");
  if (InputFileNames.empty())
    return reportError("missing required --input");
  if (TargetNames.empty())
    return reportError("missing required --targets");
  if (OutputFileName.empty())
    return reportError("missing required --output");
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

    if (!clang::checkOffloadBundleID(target))
      return reportError("invalid target '" + target + "'");

    clang::OffloadTargetInfo offloadInfo(target, bundlerConfig);
    if (!isValidOffloadKind(offloadInfo) || !offloadInfo.isTripleValid())
      return reportError("invalid target '" + target + "'");

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

void registerDialects(mlir::DialectRegistry &registry) {
  // Match cir-opt's parser surface: offload CIR inputs may carry OpenMP/LLVM
  // dialect attrs or ops before this tool wraps them in cir.offload.container.
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
                     cir::OffloadKind offloadKind) {
  cirModule.setSymName(name);
  cirModule->setAttr(
      cir::CIRDialect::getOffloadKindAttrName(),
      cir::OffloadKindAttr::get(cirModule.getContext(), offloadKind));
}

mlir::OwningOpRef<mlir::ModuleOp>
combineInputs(llvm::ArrayRef<InputTarget> inputTargets,
              mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> hostModule;
  llvm::SmallVector<mlir::OwningOpRef<mlir::ModuleOp>, 4> deviceModules;
  llvm::StringMap<unsigned> deviceNames;

  for (const InputTarget &inputTarget : inputTargets) {
    mlir::OwningOpRef<mlir::ModuleOp> cirModule =
        parseCIRInput(inputTarget.Input, context);
    if (!cirModule)
      return {};

    if (inputTarget.IsHost) {
      setOffloadAttrs(*cirModule, "host", cir::OffloadKind::Host);
      hostModule = std::move(cirModule);
      continue;
    }

    std::string deviceName =
        makeUnique(sanitizeModuleName(inputTarget.Target), deviceNames);
    setOffloadAttrs(*cirModule, deviceName, cir::OffloadKind::Device);
    deviceModules.push_back(std::move(cirModule));
  }

  auto loc = mlir::UnknownLoc::get(&context);
  mlir::OwningOpRef<mlir::ModuleOp> combinedModule(mlir::ModuleOp::create(loc));
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToStart(combinedModule->getBody());
  auto container = cir::OffloadContainerOp::create(builder, loc);
  mlir::Block &body = container.getBody().emplaceBlock();

  // Preserve the container invariant expected by the verifier: the host module
  // is the first nested op, followed by device modules in input order.
  body.push_back(hostModule.release().getOperation());
  for (mlir::OwningOpRef<mlir::ModuleOp> &deviceModule : deviceModules)
    body.push_back(deviceModule.release().getOperation());

  return combinedModule;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::HideUnrelatedOptions(CIROffloadMergeCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "CIR host-device offload merge\n");

  llvm::SmallVector<InputTarget> inputTargets;
  if (int errorCode = validateCommandLine(inputTargets))
    return errorCode;

  mlir::DialectRegistry registry;
  registerDialects(registry);
  mlir::MLIRContext context;
  context.loadDialect<cir::CIRDialect, mlir::memref::MemRefDialect,
                      mlir::LLVM::LLVMDialect, mlir::DLTIDialect,
                      mlir::omp::OpenMPDialect>();
  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> combinedModule =
      combineInputs(inputTargets, context);
  if (!combinedModule)
    return reportError("failed to parse input file");

  if (mlir::failed(mlir::verify(*combinedModule)))
    return reportError("failed to verify combined module");

  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      mlir::openOutputFile(OutputFileName, &errorMessage);
  if (!outputFile)
    return reportError(errorMessage);

  // Emit the combined CIR module as textual MLIR.
  // TODO: Eventually when bytecode support lands for CIR, we should handle
  // such case here.

  combinedModule->print(outputFile->os());
  outputFile->os() << '\n';
  outputFile->keep();
  return 0;
}
