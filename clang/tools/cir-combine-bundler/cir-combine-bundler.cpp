//===-- cir-combine-bundler/cir-combine-bundler.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

using namespace mlir;

namespace {

llvm::cl::OptionCategory
    CIRCombineBundlerCategory("cir-combine-bundler options");

llvm::cl::opt<bool> Combine("combine", llvm::cl::desc("Combine CIR inputs"),
                            llvm::cl::cat(CIRCombineBundlerCategory));

llvm::cl::list<std::string>
    InputFileNames("input",
                   llvm::cl::desc("Input CIR file. Can be specified multiple "
                                  "times for multiple input files."),
                   llvm::cl::cat(CIRCombineBundlerCategory));

llvm::cl::list<std::string>
    TargetNames("targets", llvm::cl::CommaSeparated,
                llvm::cl::desc("[<offload kind>-<target triple>,...]"),
                llvm::cl::cat(CIRCombineBundlerCategory));

llvm::cl::opt<std::string>
    OutputFileName("output", llvm::cl::desc("Combined CIR output file"),
                   llvm::cl::init(""),
                   llvm::cl::cat(CIRCombineBundlerCategory));

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

std::string makeUnique(StringRef name, llvm::StringMap<unsigned> &seenNames) {
  unsigned &count = seenNames[name];
  if (count++ == 0)
    return name.str();
  return (name + "_" + llvm::Twine(count - 1)).str();
}

bool isValidOffloadKind(const clang::OffloadTargetInfo &offloadInfo) {
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

// Dialect registration considering the shape used
// deserializing cir modules on cir-opt
void registerDialects(DialectRegistry &registry) {
  registry.insert<mlir::BuiltinDialect, cir::CIRDialect,
                  mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect,
                  mlir::DLTIDialect, mlir::omp::OpenMPDialect>();
  cir::omp::registerOpenMPExtensions(registry);
}

OwningOpRef<ModuleOp> parseCIRInput(StringRef inputFileName,
                                    MLIRContext &context) {
  ParserConfig parserConfig(&context);
  return parseSourceFile<ModuleOp>(inputFileName, parserConfig);
}

void setOffloadAttrs(ModuleOp module, StringRef name,
                     cir::OffloadKind offloadKind) {
  module.setSymName(name);
  module->setAttr(cir::CIRDialect::getOffloadKindAttrName(),
                  cir::OffloadKindAttr::get(module.getContext(), offloadKind));
}

OwningOpRef<ModuleOp> combineInputs(ArrayRef<InputTarget> inputTargets,
                                    MLIRContext &context) {
  OwningOpRef<ModuleOp> hostModule;
  SmallVector<OwningOpRef<ModuleOp>, 4> deviceModules;
  llvm::StringMap<unsigned> deviceNames;

  for (const InputTarget &inputTarget : inputTargets) {
    OwningOpRef<ModuleOp> module = parseCIRInput(inputTarget.Input, context);
    if (!module)
      return {};

    if (inputTarget.IsHost) {
      setOffloadAttrs(*module, "host", cir::OffloadKind::Host);
      hostModule = std::move(module);
      continue;
    }

    std::string deviceName =
        makeUnique(sanitizeModuleName(inputTarget.Target), deviceNames);
    setOffloadAttrs(*module, deviceName, cir::OffloadKind::Device);
    deviceModules.push_back(std::move(module));
  }

  auto loc = UnknownLoc::get(&context);
  OwningOpRef<ModuleOp> combinedModule(ModuleOp::create(loc));
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(combinedModule->getBody());
  auto container = cir::OffloadContainerOp::create(builder, loc);
  Block &body = container.getBody().emplaceBlock();

  // Populate combined module
  body.push_back(hostModule.release().getOperation());
  for (OwningOpRef<ModuleOp> &deviceModule : deviceModules)
    body.push_back(deviceModule.release().getOperation());

  return combinedModule;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::HideUnrelatedOptions(CIRCombineBundlerCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "CIR host-device combine bundler\n");

  SmallVector<InputTarget, 4> inputTargets;
  if (int errorCode = validateCommandLine(inputTargets))
    return errorCode;

  DialectRegistry registry;
  registerDialects(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> combinedModule = combineInputs(inputTargets, context);
  if (!combinedModule)
    return reportError("failed to parse input file");

  if (failed(verify(*combinedModule)))
    return reportError("failed to verify combined module");

  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      mlir::openOutputFile(OutputFileName, &errorMessage);
  if (!outputFile)
    return reportError(errorMessage);

  // serialize final output
  combinedModule->print(outputFile->os());
  outputFile->os() << '\n';
  outputFile->keep();
  return 0;
}
