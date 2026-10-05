//===- PointerFacts.cpp - Launch-derived pointer facts on kernel params ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stamps `llvm.noalias` on device-kernel pointer parameters when every visible
// launch of the kernel passes pointers that provably come from distinct
// cudaMalloc calls and the kernel body cannot reach the pointer through any
// other path. A launch argument's root is the cudaMalloc whose result reaches
// the launch: it dominates the read of the slot and every other write to the
// slot dominates it. Unchecked allocations are accepted; only a free before the
// launch invalidates a slot.
//
// Separately, `llvm.align` is stamped on a pointer parameter when every launch
// a kernel serves passes a pointer the CUDA runtime allocated, which it aligns
// to at least 256 bytes. That fact depends only on the passed value, so it
// needs neither the body check nor distinct roots.
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
#include "clang/CIR/Dialect/Transforms/OffloadOpt/KernelCloning.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "cir-offload-pointer-facts"

namespace mlir {
#define GEN_PASS_DEF_OFFLOADPOINTERFACTS
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace cir;

namespace {

static constexpr llvm::StringLiteral kRemarkName = "OffloadPointerFacts";
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

// The cudaMalloc out-parameter slot a launch argument derives from.
struct SlotRoot {
  Value slot;
  cir::CallOp malloc;
};

static bool resolveValueToRoot(Value v, SlotRoot &out, DominanceInfo &dom,
                               unsigned depth);

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
    // Pointer values round-trip through local stack slots: CIRGen materializes
    // kernel parameters (and other address-taken values) into allocas. Follow
    // the stores into such a slot so a reloaded pointer still counts as
    // derived.
    if (auto load = mlir::dyn_cast<cir::LoadOp>(def))
      if (auto slot = load.getAddr().getDefiningOp<cir::AllocaOp>()) {
        for (Operation *user : slot->getUsers())
          if (auto store = mlir::dyn_cast<cir::StoreOp>(user))
            if (mlir::isa<cir::PointerType>(store.getValue().getType()))
              work.push_back(store.getValue());
        continue;
      }
    for (Value operand : def->getOperands())
      if (mlir::isa<cir::PointerType>(operand.getType()))
        work.push_back(operand);
  }
  return false;
}

// CUDA builtin variables (__cuda_builtin_blockIdx_t, ...) are compiler-owned
// records in the constant bank; they can never alias device allocations.
static bool isCudaBuiltinVar(cir::GetGlobalOp getGlobal) {
  auto ptrTy = mlir::dyn_cast<cir::PointerType>(getGlobal.getResult().getType());
  if (!ptrTy)
    return false;
  auto rec = mlir::dyn_cast<cir::RecordType>(ptrTy.getPointee());
  return rec && rec.getName().getValue().starts_with("__cuda_builtin_");
}

// A local slot holding a pointer derived from a kernel parameter carries that
// pointer on the stack; forwarding the slot itself to a callee can leak it.
static bool slotHoldsDerivedPointer(Value slot, cir::FuncOp kernel) {
  if (!slot.getDefiningOp<cir::AllocaOp>())
    return false;
  for (Operation *user : slot.getUsers())
    if (auto store = mlir::dyn_cast<cir::StoreOp>(user))
      if (derivesFromPointerParam(store.getValue(), kernel))
        return true;
  return false;
}

// A callee without a body whose memory behavior is known: libdevice math
// (`__nv_*`, a name reserved to the implementation) or a declaration that
// touches no memory besides its arguments. Only consulted for declarations, so
// a user-defined body under one of these names is still walked.
static bool isKnownSafeDeclaration(cir::FuncOp fn, cir::CallOp call) {
  if (fn.getSymName().starts_with("__nv_"))
    return true;
  auto touchesNoOtherMemory = [](std::optional<cir::MemoryEffectsAttr> me) {
    return me && me->getOther() == cir::ModRefInfo::NoModRef;
  };
  return touchesNoOtherMemory(fn.getMemoryEffects()) ||
         touchesNoOtherMemory(call.getMemoryEffects());
}

// Thread-geometry reads and CTA barriers; neither accesses memory through a
// pointer.
static bool isAllowedIntrinsic(llvm::StringRef name) {
  // TODO: Handle AMD intrinsics
  return name.starts_with("nvvm.read.ptx.sreg.") ||
         name.starts_with("nvvm.barrier.cta.");
}

// Checks one function reachable from the kernel and collects the callees whose
// bodies must be checked next. `fn`'s own parameters anchor the derivation
// checks: in a callee they can only carry pointers the caller was allowed to
// pass.
static bool functionAllowsNoalias(cir::FuncOp fn,
                                  llvm::SmallVectorImpl<cir::FuncOp> &callees) {
  bool ok = true;
  fn.walk([&](Operation *op) {
    if (!ok)
      return;

    if (auto getGlobal = mlir::dyn_cast<cir::GetGlobalOp>(op)) {
      if (!isCudaBuiltinVar(getGlobal)) {
        ok = false;
        return;
      }
    }

    if (auto call = mlir::dyn_cast<cir::CallOp>(op)) {
      std::optional<llvm::StringRef> callee = call.getCallee();
      if (callee && (callee->contains("malloc") || callee->contains("Malloc") ||
                     callee->contains("free") || callee->contains("Free"))) {
        ok = false;
        return;
      }
      for (Value operand : call.getArgOperands())
        if (derivesFromPointerParam(operand, fn) ||
            slotHoldsDerivedPointer(operand, fn)) {
          ok = false;
          return;
        }
      // The callee runs during the kernel, so its body is part of the proof.
      // Indirect calls and unknown declarations cannot be checked.
      cir::FuncOp target;
      if (callee)
        target = SymbolTable::lookupNearestSymbolFrom<cir::FuncOp>(
            call, call.getCalleeAttr());
      if (!target) {
        ok = false;
        return;
      }
      if (!target.isDeclaration())
        callees.push_back(target);
      else if (!isKnownSafeDeclaration(target, call))
        ok = false;
      return;
    }

    if (auto intrinsic = mlir::dyn_cast<cir::LLVMIntrinsicCallOp>(op)) {
      if (!isAllowedIntrinsic(intrinsic.getIntrinsicName()))
        ok = false;
      return;
    }

    if (mlir::isa<CallOpInterface>(op)) {
      ok = false;
      return;
    }

    if (auto load = mlir::dyn_cast<cir::LoadOp>(op)) {
      // Reloading a pointer from a local stack slot is the round-trip of a
      // value this body already holds (parameter materialization); pointers
      // loaded from real memory still drop the fact.
      if (mlir::isa<cir::PointerType>(load.getResult().getType()) &&
          !load.getAddr().getDefiningOp<cir::AllocaOp>()) {
        ok = false;
        return;
      }
    }

    if (auto store = mlir::dyn_cast<cir::StoreOp>(op)) {
      // Storing into a local stack slot keeps the value on the stack; escapes
      // through memory or callees are caught by the other obligations (and by
      // slotHoldsDerivedPointer at call sites).
      if (derivesFromPointerParam(store.getValue(), fn) &&
          !store.getAddr().getDefiningOp<cir::AllocaOp>()) {
        ok = false;
        return;
      }
    }

    if (auto cast = mlir::dyn_cast<cir::CastOp>(op)) {
      if (cast.getKind() == cir::CastKind::ptr_to_int &&
          derivesFromPointerParam(cast.getSrc(), fn)) {
        ok = false;
        return;
      }
      // A pointer made from an integer is not based on any parameter, yet it
      // can hold the address of a parameter's buffer.
      if (cast.getKind() == cir::CastKind::int_to_ptr) {
        ok = false;
        return;
      }
    }
  });
  return ok;
}

// The kernel and every function it can call must pass functionAllowsNoalias.
static bool bodyAllowsNoalias(cir::FuncOp kernel) {
  if (kernel.isDeclaration() || kernel.getBody().empty())
    return false;

  llvm::SmallPtrSet<Operation *, 8> visited;
  llvm::SmallVector<cir::FuncOp, 8> worklist{kernel};
  while (!worklist.empty()) {
    cir::FuncOp fn = worklist.pop_back_val();
    if (!visited.insert(fn.getOperation()).second)
      continue;
    if (!functionAllowsNoalias(fn, worklist))
      return false;
  }
  return true;
}

// The runtime declares cudaMalloc as a C function, but the C++ headers also
// provide an inline `template <class T> cudaMalloc(T **, size_t)` shim that
// PolyBench-style code calls without a cast; CIRGen keeps the call to the
// instantiation visible (no inlining runs before this pass), so recognize both
// the plain symbol and its mangled C++ template instantiations.
//TODO: Implement this for HIP symbols!.
static bool isCudaMallocSymbol(llvm::StringRef callee) {
  if (callee == "cudaMalloc")
    return true;
  if (!callee.starts_with("_Z"))
    return false;
  std::string demangled = llvm::demangle(callee.str());
  return llvm::StringRef(demangled).contains("cudaMalloc<");
}

// Sorts every use of a host variable holding a device pointer (`float *A`), and
// of the casts that still point at it.
//   Ignored:   a load of the variable (it only reads the device address out:
//              the launch, cudaMemcpy, `C = A`), a cast that still points at
//              the variable, and lifetime markers (they only bound when A is
//              alive; reading A outside that range is undefined).
//   Collected: cudaMalloc(&A, ...) and a store into A (`A = ...`); these change
//              which device address A holds.
//   Rejected:  anything else, such as passing &A to another callee or storing
//              &A somewhere; whoever holds &A could rewrite A unseen.
// Returns false on the first rejected use.
static bool
classifyDevPtrWriters(const llvm::SmallPtrSetImpl<Value> &family,
                      llvm::SmallVectorImpl<Operation *> &devPtrWriters) {
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
        if (callee && isCudaMallocSymbol(*callee) &&
            !call.getArgOperands().empty() &&
            call.getArgOperands().front() == cur) {
          devPtrWriters.push_back(call);
          continue;
        }
        return false; // Slot address handed to another callee: escape.
      }
      if (auto store = mlir::dyn_cast<cir::StoreOp>(user)) {
        if (store.getAddr() == cur) {
          devPtrWriters.push_back(store);
          continue;
        }
        return false; // Slot value stored elsewhere: escape.
      }
      if (mlir::isa<cir::LifetimeStartOp, cir::LifetimeEndOp>(user))
        continue;
      return false; // Any other use: escape.
    }
  }
  return true;
}

// The variable `addr` and the alloca-preserving casts of it, so the recognizer
// sees the slot behind the `ptr<ptr<T>> -> ptr<ptr<void>>` cast cudaMalloc
// takes.
static void collectSlotFamily(Value addr,
                              llvm::SmallPtrSetImpl<Value> &family) {
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
}

// The writer whose device address `devPtrLoad` reads: it dominates
// `devPtrLoad` and every other writer dominates it, so no other write lands in
// between.
static Operation *reachingWriter(llvm::ArrayRef<Operation *> devPtrWriters,
                                 Operation *devPtrLoad, DominanceInfo &dom) {
  for (Operation *w : devPtrWriters)
    if (dom.properlyDominates(w, devPtrLoad) &&
        llvm::all_of(devPtrWriters, [&](Operation *o) {
          return o == w || dom.properlyDominates(o, w);
        }))
      return w;
  return nullptr;
}

// Resolve the variable `devPtrLoad` reads from to its allocation root. The
// variable must be an alloca whose address never escapes; the writer reaching
// `devPtrLoad` is either a cudaMalloc (the root) or a store whose value is
// resolved in turn. `devPtrLoad` starts as the launch argument's load and, when
// following `C = A`, becomes the load of A feeding that store.
static bool resolveAddressToRoot(Value addr, Operation *devPtrLoad,
                                 SlotRoot &out, DominanceInfo &dom,
                                 unsigned depth) {
  if (depth > 8)
    return false;
  addr = cir::getUnderlyingObject(addr);
  if (!addr.getDefiningOp<cir::AllocaOp>())
    return false;

  llvm::SmallPtrSet<Value, 8> family;
  collectSlotFamily(addr, family);
  llvm::SmallVector<Operation *, 4> devPtrWriters;
  if (!classifyDevPtrWriters(family, devPtrWriters))
    return false;

  Operation *writer = reachingWriter(devPtrWriters, devPtrLoad, dom);
  if (!writer)
    return false;
  if (auto malloc = mlir::dyn_cast<cir::CallOp>(writer)) {
    out = {addr, malloc};
    return true;
  }
  return resolveValueToRoot(mlir::cast<cir::StoreOp>(writer).getValue(), out,
                            dom, depth + 1);
}

static bool resolveValueToRoot(Value v, SlotRoot &out, DominanceInfo &dom,
                               unsigned depth) {
  if (!v || depth > 8)
    return false;
  v = cir::getUnderlyingObject(v);
  if (auto load = v.getDefiningOp<cir::LoadOp>())
    return resolveAddressToRoot(load.getAddr(), load, out, dom, depth + 1);
  return false;
}

// Whether `v` was read from a variable that only ever holds cudaMalloc
// results: every write to it is a cudaMalloc, a null store, or a copy of
// another such variable. The runtime aligns allocations to at least 256 bytes
// and null is aligned, so `v` is aligned whichever write reached it, including
// when a cudaMalloc failed and left an earlier value in place.
static bool holdsMallocAlignedPtr(Value v, unsigned depth) {
  if (!v || depth > 8)
    return false;
  auto load = cir::getUnderlyingObject(v).getDefiningOp<cir::LoadOp>();
  if (!load)
    return false;
  Value addr = cir::getUnderlyingObject(load.getAddr());
  if (!addr.getDefiningOp<cir::AllocaOp>())
    return false;

  llvm::SmallPtrSet<Value, 8> family;
  collectSlotFamily(addr, family);
  llvm::SmallVector<Operation *, 4> devPtrWriters;
  if (!classifyDevPtrWriters(family, devPtrWriters) || devPtrWriters.empty())
    return false;
  return llvm::all_of(devPtrWriters, [&](Operation *w) {
    auto store = mlir::dyn_cast<cir::StoreOp>(w);
    if (!store)
      return true; // cudaMalloc.
    if (auto c = store.getValue().getDefiningOp<cir::ConstantOp>())
      if (auto ptr = mlir::dyn_cast<cir::ConstPtrAttr>(c.getValue()))
        return ptr.isNullValue();
    return holdsMallocAlignedPtr(store.getValue(), depth + 1);
  });
}

// Whether `a` runs after `b`: compares their ancestors in the closest block
// that holds both. Ops in different regions of the same op are unordered.
static bool isAfterInProgramOrder(Operation *a, Operation *b) {
  for (Block *blk = a->getBlock(); blk;
       blk = blk->getParentOp() ? blk->getParentOp()->getBlock() : nullptr) {
    Operation *aa = blk->findAncestorOpInBlock(*a);
    Operation *bb = blk->findAncestorOpInBlock(*b);
    if (aa && bb)
      return aa != bb && bb->isBeforeInBlock(aa);
  }
  return false;
}

// Whether a free can run between the root malloc and the launch. A free that
// always runs before the malloc is replaced by it; one after the launch only
// matters if a loop brings control back to the launch without the malloc.
static bool freeCanReachLaunch(Operation *free, cir::CallOp malloc,
                               Operation *launch, DominanceInfo &dom) {
  if (dom.properlyDominates(free, malloc))
    return false;
  if (!isAfterInProgramOrder(free, launch))
    return true;
  for (auto loop = free->getParentOfType<cir::LoopOpInterface>(); loop;
       loop = loop->getParentOfType<cir::LoopOpInterface>())
    if (loop->isAncestor(launch) && !loop->isAncestor(malloc))
      return true;
  return false;
}

// Whether the launched allocation is freed between its allocation and the
// launch: the pointer would then be dangling and the fact unsound. A free after
// the launch is the normal cleanup and does not invalidate the fact.
static bool allocationFreed(cir::CallOp mallocCall, Value slot,
                            Operation *launch, DominanceInfo &dom) {
  auto hostFn = mallocCall->getParentOfType<cir::FuncOp>();
  if (!hostFn)
    return false;

  bool freed = false;
  hostFn.walk([&](cir::CallOp call) {
    if (freed)
      return;
    std::optional<llvm::StringRef> callee = call.getCallee();
    if (!callee || !callee->contains_insensitive("free"))
      return;
    // The launch itself can match by name (a kernel called `*free*`).
    if (call.getOperation() == launch ||
        !freeCanReachLaunch(call, mallocCall, launch, dom))
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

// Whether one launch passes every pointer argument from its own allocation:
// each resolves to a root malloc that is not freed before the launch, and no
// two share a root. On failure `reason` says why.
static bool launchHasDistinctRoots(const cir::LaunchSite &site,
                                   llvm::ArrayRef<bool> isPointer,
                                   DominanceInfo &dom, std::string &reason) {
  unsigned numArgs = isPointer.size();
  cir::CallOp launchCall = site.stubCall;
  llvm::SmallVector<SlotRoot> roots(numArgs);
  for (unsigned i = 0; i != numArgs; ++i) {
    if (!isPointer[i])
      continue;
    if (!resolveValueToRoot(site.getArg(i), roots[i], dom, 0)) {
      reason = "no allocation root for pointer argument " + std::to_string(i);
      return false;
    }
    if (allocationFreed(roots[i].malloc, roots[i].slot,
                        launchCall.getOperation(), dom)) {
      reason = "allocation freed before the launch for pointer argument " +
               std::to_string(i);
      return false;
    }
  }
  for (unsigned i = 0; i != numArgs; ++i)
    for (unsigned j = i + 1; j != numArgs; ++j)
      if (isPointer[i] && isPointer[j] && roots[i].malloc == roots[j].malloc) {
        reason = "repeated allocation root across pointer arguments";
        return false;
      }
  return true;
}

// Nothing outside this pass knows a clone's name; only the launches it was
// made for may reach it. Host side only: device kernels stay externally
// visible, as clang emits them, since the runtime looks them up by name.
static void makeInternal(cir::FuncOp fn) {
  fn.setLinkage(cir::GlobalLinkageKind::InternalLinkage);
  fn.setSymVisibility("private");
}

// The alignment the CUDA runtime guarantees for its allocations.
static constexpr int64_t kCudaMallocAlign = 256;

// Stamps `llvm.align` on each pointer parameter of `kernel` whose argument is
// malloc-aligned at every one of `sites`. Returns the number stamped.
static unsigned stampAlignment(cir::FuncOp kernel,
                               llvm::ArrayRef<bool> isPointer,
                               llvm::ArrayRef<cir::LaunchSite> sites) {
  llvm::StringRef alignName = mlir::LLVM::LLVMDialect::getAlignAttrName();
  unsigned stamped = 0;
  for (unsigned i = 0; i != isPointer.size() && i < kernel.getNumArguments();
       ++i) {
    if (!isPointer[i] ||
        !mlir::isa<cir::PointerType>(kernel.getArgument(i).getType()))
      continue;
    if (!llvm::all_of(sites, [&](const cir::LaunchSite &site) {
          return holdsMallocAlignedPtr(site.getArg(i), 0);
        }))
      continue;
    if (auto existing =
            kernel.getArgAttrOfType<mlir::IntegerAttr>(i, alignName))
      if (existing.getInt() >= kCudaMallocAlign)
        continue;
    kernel.setArgAttr(
        i, alignName,
        mlir::IntegerAttr::get(mlir::IntegerType::get(kernel.getContext(), 64),
                               kCudaMallocAlign));
    ++stamped;
  }
  if (stamped)
    LDBG() << "  '" << kernel.getSymName() << "': aligned " << stamped
           << " pointer parameter(s)";
  return stamped;
}

struct OffloadPointerFactsPass
    : public impl::OffloadPointerFactsBase<OffloadPointerFactsPass> {
  void runOnOperation() override;
};

void OffloadPointerFactsPass::runOnOperation() {
  mlir::ModuleOp container = getOperation();
  // The pass is anchored on a module, but only an offload container holds the
  // host<->device correspondence it reasons about. Any other module is left
  // alone, matching the other container passes.
  if (!cir::isOffloadContainer(container))
    return;

  cir::KernelBindingTable &table = getAnalysis<cir::KernelBindingTable>();
  DominanceInfo &dom = getAnalysis<DominanceInfo>();
  bool changed = false;

  auto anchorOf = [](const cir::KernelBinding &binding) {
    return binding.deviceKernels.empty() ? binding.hostStub
                                         : binding.deviceKernels.front();
  };

  LDBG() << "container has " << table.size() << " kernel binding(s)";

  // Cloning adds bindings and rewrites launches, so walk a snapshot.
  llvm::SmallVector<std::pair<std::string, cir::KernelBinding>> bindings;
  for (const auto &entry : table)
    bindings.emplace_back(entry.first.str(), entry.second);

  for (const auto &[kernelNameStr, binding] : bindings) {
    llvm::StringRef kernelName = kernelNameStr;

    LDBG() << "kernel '" << kernelName << "': " << binding.launchSites.size()
           << " launch site(s), " << binding.deviceKernels.size()
           << " device kernel(s)";

    // A kernel launched nowhere yields no launch-derived facts.
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

    cir::FuncOp stub = binding.hostStub;
    unsigned numArgs = stub.getNumArguments();
    llvm::SmallVector<bool> isPointer(numArgs, false);
    for (unsigned i = 0; i != numArgs; ++i)
      isPointer[i] = mlir::isa<cir::PointerType>(stub.getArgument(i).getType());

    // Alignment only depends on the passed values, so the original kernel gets
    // it whenever all of its launches are visible, whatever noalias decides.
    // Launches later redirected to a clone only make this stricter.
    if (table.allLaunchSitesVisible(kernelName))
      for (cir::FuncOp kernel : binding.deviceKernels)
        if (unsigned n =
                stampAlignment(kernel, isPointer, binding.launchSites)) {
          numParamsAligned += n;
          changed = true;
        }

    if (!llvm::all_of(binding.deviceKernels, bodyAllowsNoalias)) {
      LDBG() << "  skipped: kernel body has a drop-table violation";
      remarkSkipped(anchorOf(binding), kernelName,
                    "kernel body has a drop-table violation",
                    binding.launchSites.size());
      ++numKernelsSkipped;
      continue;
    }

    // Each launch is decided on its own.
    std::string reason;
    llvm::SmallVector<cir::LaunchSite> provenSites;
    for (const cir::LaunchSite &site : binding.launchSites) {
      std::string siteReason;
      if (launchHasDistinctRoots(site, isPointer, dom, siteReason))
        provenSites.push_back(site);
      else if (reason.empty())
        reason = siteReason;
    }
    if (provenSites.empty()) {
      LDBG() << "  skipped: " << reason;
      remarkSkipped(anchorOf(binding), kernelName, reason,
                    binding.launchSites.size());
      ++numKernelsSkipped;
      continue;
    }

    // When every launch is visible and proven, annotate the kernel itself.
    // Otherwise the original keeps serving the other launches, and the proven
    // ones are redirected to an internal copy that carries the facts.
    llvm::SmallVector<cir::FuncOp, 2> targets(binding.deviceKernels.begin(),
                                              binding.deviceKernels.end());
    if (!table.allLaunchSitesVisible(kernelName) ||
        provenSites.size() != binding.launchSites.size()) {
      cir::SpecializationTarget target = cir::getSpecializationTarget(
          container, binding, "__noalias", provenSites);
      if (!target) {
        LDBG() << "  skipped: kernel cannot be cloned";
        remarkSkipped(anchorOf(binding), kernelName,
                      "kernel cannot be cloned for its proven launches",
                      binding.launchSites.size());
        ++numKernelsSkipped;
        continue;
      }
      if (target.cloned) {
        makeInternal(target.hostStub);
        auto cloneName =
            target.hostStub->getAttrOfType<cir::CUDAKernelNameAttr>(
                cir::CUDAKernelNameAttr::getMnemonic());
        if (auto handle = mlir::dyn_cast_or_null<cir::GlobalOp>(
                cir::getOffloadContainerHostModule(container).lookupSymbol(
                    cloneName.getKernelName()))) {
          handle.setLinkage(cir::GlobalLinkageKind::InternalLinkage);
          handle.setSymVisibility("private");
        }
        LDBG() << "  cloned for " << provenSites.size() << " of "
               << binding.launchSites.size() << " launch site(s)";
        changed = true;
        for (cir::FuncOp kernel : target.deviceKernels)
          numParamsAligned += stampAlignment(kernel, isPointer, provenSites);
      }
      targets.assign(target.deviceKernels.begin(), target.deviceKernels.end());
    }

    for (cir::FuncOp kernel : targets) {
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
        remarkStamped(kernel, kernelName, stamped, provenSites.size());
        ++numKernelsNoalias;
        numParamsNoalias += stamped;
      }
    }
  }

  if (!changed)
    markAllAnalysesPreserved();
}

} // namespace

std::unique_ptr<Pass> mlir::createOffloadPointerFactsPass() {
  return std::make_unique<OffloadPointerFactsPass>();
}
