//===- OffloadLaunchNoalias.cpp - Launch-derived noalias on kernel params -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stamps `llvm.noalias` on device-kernel pointer parameters when every visible
// launch of the kernel passes pointers that provably come from distinct
// cudaMalloc slots, the allocation is checked for success on the path to the
// launch, and the kernel body cannot reach the pointer through any other path.
//
// The analysis is conservative and kernel-wide: a repeated root, an unprovable
// argument, or an unprovable body path drops every launch-derived annotation
// for that kernel. It attaches attributes in place, creates no clones, emits no
// diagnostics, and always succeeds.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Remarks.h"
#include "clang/CIR/Dialect/Analysis/CIRBasicAliasAnalysis.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/Passes.h"
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelBindingTable.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "cir-offload-launch-noalias"

namespace mlir {
#define GEN_PASS_DEF_OFFLOADLAUNCHNOALIAS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

static constexpr llvm::StringLiteral kRemarkName = "OffloadLaunchNoalias";
static constexpr llvm::StringLiteral kRemarkCategory = "cir-offload-noalias";

static void remarkSkipped(cir::FuncOp anchor, llvm::StringRef kernelName,
                          llvm::StringRef reason, size_t numLaunchSites) {
  if (!anchor)
    return;
  remark::missed(anchor.getLoc(), remark::RemarkOpts::name(kRemarkName)
                                      .category(kRemarkCategory)
                                      .function(kernelName))
      << remark::add("no launch-derived noalias")
      << remark::reason("{0}", reason)
      << remark::metric("launchSites", numLaunchSites);
}

static void remarkStamped(cir::FuncOp kernel, llvm::StringRef kernelName,
                          size_t numParams, size_t numLaunchSites) {
  remark::passed(kernel.getLoc(), remark::RemarkOpts::name(kRemarkName)
                                      .category(kRemarkCategory)
                                      .function(kernelName))
      << remark::add("stamped llvm.noalias on kernel pointer parameter(s)")
      << remark::metric("pointerParams", numParams)
      << remark::metric("launchSites", numLaunchSites);
}

static bool isZeroConstant(Value v) {
  auto cst = v.getDefiningOp<cir::ConstantOp>();
  if (!cst)
    return false;
  if (auto intAttr = mlir::dyn_cast<cir::IntAttr>(cst.getValue()))
    return intAttr.getValue().isZero();
  return false;
}

// A comparison against the success code: either a literal zero or the value
// loaded from the canonical CUDA/HIP success-code global. v1 only knows these
// spellings; broaden when other forms appear in the corpus.
static bool isSuccessCode(Value v) {
  if (isZeroConstant(v))
    return true;
  auto load = v.getDefiningOp<cir::LoadOp>();
  if (!load)
    return false;
  auto getGlobal = cir::stripPointerCasts(load.getAddr()).getDefiningOp<cir::GetGlobalOp>();
  if (!getGlobal)
    return false;
  return getGlobal.getName().contains("cudaSuccess") ||
         getGlobal.getName().contains("hipSuccess");
}

// True when `a` executes before `b` in the top-level block of the same host
// function, looking through structured regions such as `cir.scope`.
static bool lexicallyBefore(Operation *a, Operation *b) {
  auto fn = a->getParentOfType<cir::FuncOp>();
  if (!fn || b->getParentOfType<cir::FuncOp>() != fn)
    return false;
  Operation *ta = a, *tb = b;
  while (ta->getParentOp() != fn.getOperation())
    ta = ta->getParentOp();
  while (tb->getParentOp() != fn.getOperation())
    tb = tb->getParentOp();
  return ta->getBlock() == tb->getBlock() && ta->isBeforeInBlock(tb);
}

static bool isContainedIn(Operation *op, Region &region) {
  for (Region *r = op->getParentRegion(); r;
       r = r->getParentOp()->getParentRegion())
    if (r == &region)
      return true;
  return false;
}

// The cudaMalloc out-parameter slot a launch argument derives from.
struct SlotRoot {
  Value slot;
  cir::CallOp malloc;
};

static bool resolveValueToRoot(Value v, SlotRoot &out, unsigned depth);

// Whether `v` transitively derives from a pointer parameter of `kernel`. Only
// pointer-typed operands are followed, so comparing a pointer against null and
// using the boolean result does not count as a derivation.
static bool derivesFromPointerParam(Value v, cir::FuncOp kernel) {
  if (!mlir::isa<cir::PointerType>(v.getType()))
    return false;

  llvm::SmallPtrSet<Value, 16> visited;
  llvm::SmallVector<Value, 8> work{v};
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    if (!visited.insert(cur).second)
      continue;
    if (auto arg = mlir::dyn_cast<BlockArgument>(cur)) {
      if (arg.getOwner()->getParentOp() == kernel.getOperation())
        return true;
      // Region arguments are not followed to their parent operands: fine today
      // (CIR loops carry no value region args), but derivations through them
      // would be missed if that changes.
      continue;
    }
    Operation *def = cur.getDefiningOp();
    if (!def)
      continue;
    for (Value operand : def->getOperands())
      if (mlir::isa<cir::PointerType>(operand.getType()))
        work.push_back(operand);
  }
  return false;
}

// The universality obligations from the v1 drop-table that are visible in the
// kernel body: address-taken globals, pointers loaded from memory, pointers
// forwarded to any callee, escaped pointer values (including pointer-to-integer
// laundering), and in-kernel allocation all drop the fact.
static bool bodyAllowsNoalias(cir::FuncOp kernel) {
  if (kernel.isDeclaration() || kernel.getBody().empty())
    return false;

  bool ok = true;
  kernel.walk([&](Operation *op) {
    if (!ok)
      return;

    if (mlir::isa<cir::GetGlobalOp>(op)) {
      ok = false;
      return;
    }

    if (auto call = mlir::dyn_cast<cir::CallOp>(op)) {
      std::optional<llvm::StringRef> callee = call.getCallee();
      if (callee && (callee->contains("malloc") || callee->contains("Malloc") ||
                     callee->contains("free") || callee->contains("Free"))) {
        ok = false;
        return;
      }
      for (Value operand : call.getArgOperands())
        if (derivesFromPointerParam(operand, kernel)) {
          ok = false;
          return;
        }
    }

    if (auto load = mlir::dyn_cast<cir::LoadOp>(op)) {
      if (mlir::isa<cir::PointerType>(load.getResult().getType())) {
        ok = false;
        return;
      }
    }

    if (auto store = mlir::dyn_cast<cir::StoreOp>(op)) {
      if (derivesFromPointerParam(store.getValue(), kernel)) {
        ok = false;
        return;
      }
    }

    if (auto cast = mlir::dyn_cast<cir::CastOp>(op)) {
      if (cast.getKind() == cir::CastKind::ptr_to_int &&
          derivesFromPointerParam(cast.getSrc(), kernel)) {
        ok = false;
        return;
      }
    }
  });
  return ok;
}

// Resolve the storage address of a launch argument to its allocation root. The
// address must be an alloca that either a cudaMalloc wrote through (an
// allocation instance) or that a single store forwards into; the slot and its
// cast family must have no escaping use and no writer after the allocation.
static bool resolveAddressToRoot(Value addr, SlotRoot &out, unsigned depth) {
  if (depth > 8)
    return false;
  addr = cir::getUnderlyingObject(addr);
  if (!addr.getDefiningOp<cir::AllocaOp>())
    return false;

  // Follow alloca-preserving casts so the recognizer sees the slot behind the
  // `ptr<ptr<T>> -> ptr<ptr<void>>` cast cudaMalloc takes.
  llvm::SmallPtrSet<Value, 8> family;
  llvm::SmallVector<Value, 8> work{addr};
  family.insert(addr);
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    for (OpOperand &use : cur.getUses()) {
      auto cast = mlir::dyn_cast<cir::CastOp>(use.getOwner());
      if (cast && cast.getSrc() == cur && cast.isAllocaPreservingCast())
        if (family.insert(cast.getResult()).second)
          work.push_back(cast.getResult());
    }
  }

  llvm::SmallVector<cir::CallOp, 2> mallocs;
  llvm::SmallVector<cir::StoreOp, 2> stores;
  for (Value cur : family) {
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (auto cast = mlir::dyn_cast<cir::CastOp>(user)) {
        if (cast.getSrc() == cur && cast.isAllocaPreservingCast())
          continue;
        return false; // Opaque cast: laundering or escape.
      }
      if (auto load = mlir::dyn_cast<cir::LoadOp>(user)) {
        if (load.getAddr() == cur)
          continue;
        return false;
      }
      if (auto call = mlir::dyn_cast<cir::CallOp>(user)) {
        std::optional<llvm::StringRef> callee = call.getCallee();
        if (callee && *callee == "cudaMalloc" &&
            !call.getArgOperands().empty() &&
            call.getArgOperands().front() == cur) {
          mallocs.push_back(call);
          continue;
        }
        return false; // Slot address handed to another callee: escape.
      }
      if (auto store = mlir::dyn_cast<cir::StoreOp>(user)) {
        if (store.getAddr() == cur) {
          stores.push_back(store);
          continue;
        }
        return false; // Slot value stored elsewhere: escape.
      }
      return false; // Any other use: escape.
    }
  }

  if (mallocs.size() == 1) {
    // A store after the allocation is a second writer / invalidation; only
    // initialization that precedes the allocation is harmless.
    for (cir::StoreOp store : stores)
      if (!lexicallyBefore(store.getOperation(), mallocs.front().getOperation()))
        return false;
    out = {addr, mallocs.front()};
    return true;
  }

  if (mallocs.empty() && stores.size() == 1) {
    // Single-store forwarding: the launch reads the value this store wrote.
    return resolveValueToRoot(stores.front().getValue(), out, depth + 1);
  }

  return false;
}

static bool resolveValueToRoot(Value v, SlotRoot &out, unsigned depth) {
  if (!v || depth > 8)
    return false;
  v = cir::getUnderlyingObject(v);
  if (auto load = v.getDefiningOp<cir::LoadOp>())
    return resolveAddressToRoot(load.getAddr(), out, depth + 1);
  // Block arguments, globals, constants and call results have no v1 root.
  return false;
}

// Whether the launch is only reachable when cudaMalloc succeeded. The
// allocation must execute unconditionally, and the failure arm of a comparison
// against the success code must exit before the launch.
static bool isAllocationSuccessChecked(cir::CallOp mallocCall,
                                       Operation *launch) {
  if (mallocCall.getNumResults() != 1)
    return false;
  Value err = mallocCall.getResult();
  auto fn = launch->getParentOfType<cir::FuncOp>();
  if (!fn || mallocCall->getParentOfType<cir::FuncOp>() != fn)
    return false;

  for (Operation *p = mallocCall->getParentOp(); p && p != fn.getOperation();
       p = p->getParentOp())
    if (mlir::isa<cir::IfOp, cir::SwitchOp, cir::WhileOp, cir::DoWhileOp,
                  cir::ForOp>(p))
      return false;

  for (Operation *user : err.getUsers()) {
    auto cmp = mlir::dyn_cast<cir::CmpOp>(user);
    if (!cmp || (cmp.getLhs() != err && cmp.getRhs() != err))
      continue;
    Value other = cmp.getLhs() == err ? cmp.getRhs() : cmp.getLhs();
    if (!isSuccessCode(other))
      continue;

    if (cmp.getKind() == cir::CmpOpKind::ne) {
      // `if (err != success) exit;` leaves the fall-through path as success.
      for (Operation *cmpUser : cmp.getResult().getUsers()) {
        auto ifOp = mlir::dyn_cast<cir::IfOp>(cmpUser);
        if (!ifOp || ifOp.getCondition() != cmp.getResult())
          continue;
        if (ifOp.getThenRegion().empty() ||
            !ifOp.getThenRegion().hasOneBlock() ||
            !mlir::isa<cir::ReturnOp>(
                ifOp.getThenRegion().front().getTerminator()))
          continue;
        if (isContainedIn(launch, ifOp.getThenRegion()))
          continue;
        if (isContainedIn(launch, ifOp.getElseRegion()) ||
            (!ifOp->isProperAncestor(launch) && lexicallyBefore(ifOp, launch)))
          return true;
      }
    } else if (cmp.getKind() == cir::CmpOpKind::eq) {
      // `if (err == success) { ... launch ... }` is the success arm.
      for (Operation *cmpUser : cmp.getResult().getUsers()) {
        auto ifOp = mlir::dyn_cast<cir::IfOp>(cmpUser);
        if (!ifOp || ifOp.getCondition() != cmp.getResult())
          continue;
        if (isContainedIn(launch, ifOp.getThenRegion()))
          return true;
      }
    }
  }
  return false;
}

// Whether the launched allocation is freed between its allocation and the
// launch: the pointer would then be dangling and the fact unsound. A free after
// the launch is the normal cleanup and does not invalidate the fact.
static bool allocationFreed(cir::CallOp mallocCall, Value slot,
                            Operation *launch) {
  auto hostFn = mallocCall->getParentOfType<cir::FuncOp>();
  if (!hostFn)
    return false;
  Block *entry = mallocCall->getBlock();
  Operation *launchAncestor = launch;
  while (launchAncestor && launchAncestor->getBlock() != entry)
    launchAncestor = launchAncestor->getParentOp();

  bool freed = false;
  hostFn.walk([&](cir::CallOp call) {
    if (freed)
      return;
    std::optional<llvm::StringRef> callee = call.getCallee();
    if (!callee || !callee->contains_insensitive("free"))
      return;
    if (call->getBlock() == entry && launchAncestor &&
        !call->isBeforeInBlock(launchAncestor))
      return;
    for (Value arg : call.getArgOperands()) {
      auto load = cir::stripPointerCasts(arg).getDefiningOp<cir::LoadOp>();
      if (load && cir::stripPointerCasts(load.getAddr()) == slot) {
        freed = true;
        return;
      }
    }
  });
  return freed;
}

struct OffloadLaunchNoaliasPass
    : public impl::OffloadLaunchNoaliasBase<OffloadLaunchNoaliasPass> {
  void runOnOperation() override;
};

void OffloadLaunchNoaliasPass::runOnOperation() {
  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  cir::CIRBasicAliasAnalysis aliasAnalysis;
  bool changed = false;

  auto anchorOf = [](const cir::KernelBinding &binding) {
    return binding.deviceKernels.empty() ? binding.hostStub
                                         : binding.deviceKernels.front();
  };

  LDBG() << "container has " << table.size() << " kernel binding(s)";

  for (const auto &entry : table) {
    llvm::StringRef kernelName = entry.first;
    const cir::KernelBinding &binding = entry.second;

    LDBG() << "kernel '" << kernelName << "': " << binding.launchSites.size()
           << " launch site(s), " << binding.deviceKernels.size()
           << " device kernel(s)";

    // Closed-world gate: reuse the binding table's linkage/visibility predicate
    // so this pass and the other container passes agree on which launches are
    // observable. A kernel launched nowhere yields no launch-derived facts.
    if (binding.launchSites.empty()) {
      LDBG() << "  skipped: no launch site in this translation unit";
      remarkSkipped(anchorOf(binding), kernelName,
                    "no launch site in this translation unit",
                    binding.launchSites.size());
      ++numKernelsSkipped;
      continue;
    }
    if (binding.deviceKernels.empty()) {
      LDBG() << "  skipped: no device kernel bound";
      remarkSkipped(anchorOf(binding), kernelName, "no device kernel bound",
                    binding.launchSites.size());
      ++numKernelsSkipped;
      continue;
    }
    if (!table.allLaunchSitesVisible(kernelName)) {
      LDBG() << "  skipped: launch sites not all visible";
      remarkSkipped(anchorOf(binding), kernelName,
                    "launch sites not all visible",
                    binding.launchSites.size());
      ++numKernelsSkipped;
      continue;
    }

    std::string reason;
    bool kernelOk = true;
    for (cir::FuncOp kernel : binding.deviceKernels)
      if (!bodyAllowsNoalias(kernel)) {
        kernelOk = false;
        break;
      }
    if (!kernelOk) {
      LDBG() << "  skipped: kernel body has a drop-table violation";
      remarkSkipped(anchorOf(binding), kernelName,
                    "kernel body has a drop-table violation",
                    binding.launchSites.size());
      ++numKernelsSkipped;
      continue;
    }

    cir::FuncOp stub = binding.hostStub;
    unsigned numArgs = stub.getNumArguments();
    llvm::SmallVector<bool> isPointer(numArgs, false);
    for (unsigned i = 0; i != numArgs; ++i)
      isPointer[i] = mlir::isa<cir::PointerType>(stub.getArgument(i).getType());

    // Meet across every launch: all pointer arguments must be provable and
    // pairwise distinct at every site, or the kernel gets no annotations.
    for (const cir::LaunchSite &site : binding.launchSites) {
      llvm::SmallVector<SlotRoot> roots(numArgs);
      for (unsigned i = 0; i != numArgs && kernelOk; ++i) {
        if (!isPointer[i])
          continue;
        SlotRoot root;
        cir::CallOp launchCall = site.stubCall;
        if (!resolveValueToRoot(site.getArg(i), root, 0)) {
          kernelOk = false;
          reason = "no allocation root for pointer argument " + std::to_string(i);
          break;
        }
        if (!isAllocationSuccessChecked(root.malloc,
                                        launchCall.getOperation())) {
          kernelOk = false;
          reason = "allocation success not proven for pointer argument " +
                   std::to_string(i);
          break;
        }
        if (allocationFreed(root.malloc, root.slot,
                            launchCall.getOperation())) {
          kernelOk = false;
          reason = "allocation freed before the launch for pointer argument " +
                   std::to_string(i);
          break;
        }
        roots[i] = root;
      }
      if (!kernelOk)
        break;

      for (unsigned i = 0; i != numArgs && kernelOk; ++i) {
        if (!isPointer[i])
          continue;
        for (unsigned j = i + 1; j != numArgs; ++j) {
          if (!isPointer[j])
            continue;
          if (!aliasAnalysis.alias(roots[i].slot, roots[j].slot).isNo()) {
            kernelOk = false;
            reason = "repeated allocation root across pointer arguments";
            break;
          }
        }
      }
    }
    if (!kernelOk) {
      LDBG() << "  skipped: " << reason;
      remarkSkipped(anchorOf(binding), kernelName, reason,
                    binding.launchSites.size());
      ++numKernelsSkipped;
      continue;
    }

    // In-place attachment on the existing kernel parameters; no clones.
    for (cir::FuncOp kernel : binding.deviceKernels) {
      unsigned stamped = 0;
      for (unsigned i = 0; i != numArgs && i < kernel.getNumArguments(); ++i) {
        if (!isPointer[i] ||
            !mlir::isa<cir::PointerType>(kernel.getArgument(i).getType()))
          continue;
        kernel.setArgAttr(i, mlir::LLVM::LLVMDialect::getNoAliasAttrName(),
                          mlir::UnitAttr::get(&getContext()));
        ++stamped;
        changed = true;
      }
      if (stamped) {
        LDBG() << "  '" << kernel.getSymName() << "': stamped " << stamped
               << " pointer parameter(s)";
        remarkStamped(kernel, kernelName, stamped, binding.launchSites.size());
        ++numKernelsAnnotated;
        numParamsAnnotated += stamped;
      }
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadLaunchNoaliasPass() {
  return std::make_unique<OffloadLaunchNoaliasPass>();
}
