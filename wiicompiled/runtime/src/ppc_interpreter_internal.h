// What the cached interpreter (ppc_interpreter.cpp) shares with its JIT backend (docs/jit-design.md): the decoded block
// types, the helpers compiled code calls, and the interface a backend implements. Private to src/.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>

#include "abi_bridge.h"
#include "brawl_dirty_memory.h"
#include "guest_float_abi.h"
#include "ppc_interpreter.h"
#include "ppc_runtime.h"

#ifndef MKW_JIT
#define MKW_JIT 0
#endif

namespace PpcInterpreter::detail {

enum class Form : uint8_t { Unknown, Scalar, Paired };

// What a call into a translated function does around the call itself, worked out once per target after the function registry
// is published (nothing in it changes then): the raw dispatch record's entry and guards and the float ABI. Interned per target
// address, immutable, never freed.
struct JitCallTarget {
    uint32_t target;               // the guest address
    uint32_t address;              // the raw record's address (the execution address the call scopes)
    void (*entry)(CpuContext*);    // the raw record's entry: the function's wrapper
    const uint8_t* patchedByte;    // &g_mkwGuestCodePatched[guardIdx] when variants are off, else null
    uint32_t fprMask;              // nonvolatile FPRs the callee may write, as PpcNonvolatileFprGuard takes it
    uint32_t argMask;              // float arguments NormalizeCallFloatArguments checks: bit r for f1..f13
    uint8_t fprFirst, fprCount;    // the mask as one run of registers (first - 14, length), or fprCount 0: not a single run
    bool preserveGprs;             // PpcNonvolatileGprGuard enabled
    bool special;                  // 0x80017618, 0x8002A210, 0x80029438: Call treats them specially
    bool returnsPaired;            // NormalizeCalleeFloatReturn converts f1
    bool lean;                     // fprMask == 0 and no GPR guard: the call needs no guard, only the execution address and the context
    bool fprsOnly;                 // an FPR guard and no GPR guard
    bool fastDirect;               // lean && !returnsPaired && !special: callable directly from JIT
};

// One guest function's run in the interpreter. Run keeps its own copies of what it needs in the interpreter loop; the
// members are kept in memory because compiled code reads and writes them.
constexpr uint32_t kFrameFlagLoopFrame = 1u << 0;   // this frame is BrawlRollback::RunLoop's frame
constexpr uint32_t kFrameFlagDirectEnter = 1u << 1; // entered via jit_enter_direct (in-trampoline Frame)
constexpr uint32_t kFrameFlagSubCall = 1u << 2;     // a frame the JIT made for a function it called (its continuation: JitLiftContinue)

struct Frame {
    Frame(CpuContext* c, uint32_t at) : ctx(c), pc(at), entry(at) {}
    CpuContext* ctx;
    uint32_t pairedMask = 0;          // bit r is 1 when fpr[r] holds a paired value, 0 when it holds a scalar double
    // The lr the function running was entered with, & ~3 (compiled code compares lr & ~3 with it). Every reader masks it with ~3:
    // compiled code sets bit 0 while a function it called with a host call runs in this frame (src/jit/jit_x64_branch.cpp), and
    // keeps this field and pairedMask (one 8-byte word) on the host stack across that call.
    uint32_t returnAddress = 0;
    uint32_t flags = 0;               // kFrameFlag*
    uint32_t pc;
    uint32_t entry;                   // the address the function running was entered at
    uint32_t reserved = 0;
    uint64_t executed = 0;            // guest instructions compiled code ran since the dispatcher last collected them
    std::exception_ptr pending;       // an exception caught by a helper compiled code called
};

// --- FPR forms ------------------------------------------------------------------------------------------------

inline bool IsPaired(const Frame& f, uint32_t r) { return ((f.pairedMask >> (r & 31u)) & 1u) != 0; }
inline double Scalar(Frame& f, uint32_t r) {
    if (IsPaired(f, r)) {
        return PPC_PsToScalarInline(f.ctx->fpr[r].d);
    }
    return f.ctx->fpr[r].d;
}
inline double Paired(Frame& f, uint32_t r) {
    if (!IsPaired(f, r)) {
        return PPC_PsFromScalarInline(f.ctx->fpr[r].d);
    }
    return f.ctx->fpr[r].d;
}
inline void SetScalar(Frame& f, uint32_t r, double v) { f.ctx->fpr[r].d = v; f.pairedMask &= ~(1u << (r & 31u)); }
inline void SetPaired(Frame& f, uint32_t r, double v) { f.ctx->fpr[r].d = v; f.pairedMask |= (1u << (r & 31u)); }
inline void CopyForm(Frame& f, uint32_t dst, uint32_t src) {
    const uint32_t bit = 1u << (dst & 31u);
    if (IsPaired(f, src)) f.pairedMask |= bit;
    else f.pairedMask &= ~bit;
}

// Branch condition (BO/BI) with the ctr decrement.
inline bool BranchTaken(CpuContext* c, uint32_t bo, uint32_t bi) {
    if ((bo & 0x04u) == 0) {
        --c->ctr;
    }
    const bool ctrOk = (bo & 0x04u) != 0 || ((c->ctr != 0) != ((bo & 0x02u) != 0));
    const bool condOk = (bo & 0x10u) != 0 || (PpcCrvBit(c->crv[(bi >> 2) & 7u], bi & 3u) == ((bo & 0x08u) != 0));
    return ctrOk && condOk;
}

// A bl's target, resolved when its block is built once the function registry is published (nothing in it changes
// after that): the translated function, its raw dispatch record and its float ABI, each null when there is none.
struct CallSite {
    const TranslatedFunctionInfo* info;
    const RawDispatchRecord* raw;
    const MkwGuestFloatAbi* abi;
};

// One guest instruction w at pc, with next = pc + 4 on entry: the pc to continue at afterwards. Return when the frame
// is done (a blr or tail call back to the address it was entered with).
enum class Step : uint8_t { Continue, Return };

constexpr uint32_t kMaxBlock = 32;
constexpr uint32_t kMaxSegments = 6;

// What the dispatch loop does after an op.
enum class Act : uint8_t {
    Next,    // the next op
    Exit,    // leave the block for `next`
    Return,  // the frame returned
    Exec,    // run Exec on the op's word (Generic, or a tail call)
};

enum class K : uint8_t {
    Generic,
    // integer: d, a, b are register numbers; imm the immediate as the handler uses it
    Li, Addi, Lis, Addis, Mulli, Ori, Oris, Xori, Xoris, AndiRc, AndisRc,
    Rlwinm, RlwinmRc, Rlwimi, Cmpwi, Cmplwi, Cmpw, Cmplw,
    Add, Subf, Neg, Mullw, Or, OrRc, And, Xor, Andc, Nor, Slw, Srw, Srawi, Extsb, Extsh, Cntlzw,
    Mflr, Mtlr, Mfctr, Mtctr, Mfcr,
    AndRc, Subfic, Rlwnm, RlwnmRc, Mfspr, Mtspr,  // Rlwnm: imm the mask; Mfspr/Mtspr: imm the spr (not lr or ctr)
    // loads and stores, D-form with rA != 0 (imm the displacement) and X-form with rA != 0
    Lwz, Lwzu, Lbz, Lbzu, Lhz, Lhzu, Lha, Stw, Stwu, Stb, Stbu, Sth, Sthu, Lmw, Stmw,
    Lwzx, Lbzx, Lhzx, Stwx, Stbx, Sthx,
    Lfs, Lfsu, Lfd, Lfdu, Stfs, Stfsu, Stfd, Stfdu, Lfsx, Stfsx,
    PsqL, PsqLu, PsqSt, PsqStu,  // b = W << 3 | I
    // floating point, no Rc: imm holds frC
    Fadds, Fsubs, Fmuls, Fdivs, Fmadds, Fmsubs, Fnmsubs, Fnmadds,
    Fadd, Fsub, Fmul, Fdiv, Fmadd, Fmsub, Fneg, Fabs, Fmr, Frsp, Fcmpu,
    PsAdd, PsSub, PsMul, PsMadd, PsMsub, PsNmadd, PsNmsub, PsMuls0, PsMuls1, PsMadds0, PsMadds1, PsSum0, PsSum1,
    PsMerge00, PsMerge01, PsMerge10, PsMerge11, PsMr, PsNeg, PsAbs,
    // bc without LK: d = BO, a = BI, imm = the target; leaves the block when taken
    Bc,
    // a compare (or rlwinm.) and the bc after it, in one dispatch: this op has the compare's operands, the next op is
    // the bc (which the handler runs too and steps over)
    CmpwiBc, CmplwiBc, CmpwBc, CmplwBc, RlwinmRcBc,
    // b to untranslated code, followed at decode: the block goes on at the target (imm)
    Jmp,
    // block ends, each Exec's case for the word with what it can work out once done at decode (imm = the target)
    B,      // b: a jump, or a tail call when the target is translated (then Exec)
    Bl,     // bl: a call
    Blr,    // blr (unconditional): the frame's return, or a jump through lr
    Bclr,   // bclr without LK, conditional (d = BO, a = BI): Blr when taken, else the next op
    Bctr,   // bctr: a jump through ctr, or a tail call when the target is translated (then Exec)
    Bctrl,  // bctrl: a call through ctr
    End,    // after a block's last op (not counted in it): leave for imm, where the block falls through
};

// Kinds before Bc have a handler that only reads and writes registers and memory (the verifier's subjects).
inline bool Plain(K k) { return k != K::Generic && k < K::Bc; }
inline K Unfused(K k) {
    switch (k) {
    case K::CmpwiBc: return K::Cmpwi;
    case K::CmplwiBc: return K::Cmplwi;
    case K::CmpwBc: return K::Cmpw;
    case K::CmplwBc: return K::Cmplw;
    case K::RlwinmRcBc: return K::RlwinmRc;
    default: return k;
    }
}

inline constexpr const char* kKindNames[] = {
    "Generic", "Li", "Addi", "Lis", "Addis", "Mulli", "Ori", "Oris", "Xori", "Xoris", "AndiRc", "AndisRc", "Rlwinm", "RlwinmRc",
    "Rlwimi", "Cmpwi", "Cmplwi", "Cmpw", "Cmplw", "Add", "Subf", "Neg", "Mullw", "Or", "OrRc", "And", "Xor", "Andc", "Nor", "Slw",
    "Srw", "Srawi", "Extsb", "Extsh", "Cntlzw", "Mflr", "Mtlr", "Mfctr", "Mtctr", "Mfcr", "AndRc", "Subfic", "Rlwnm", "RlwnmRc",
    "Mfspr", "Mtspr", "Lwz", "Lwzu", "Lbz", "Lbzu", "Lhz", "Lhzu", "Lha", "Stw", "Stwu", "Stb", "Stbu", "Sth", "Sthu", "Lmw",
    "Stmw", "Lwzx", "Lbzx", "Lhzx", "Stwx", "Stbx", "Sthx", "Lfs", "Lfsu", "Lfd", "Lfdu", "Stfs", "Stfsu", "Stfd", "Stfdu",
    "Lfsx", "Stfsx", "PsqL", "PsqLu", "PsqSt", "PsqStu", "Fadds", "Fsubs", "Fmuls", "Fdivs", "Fmadds", "Fmsubs", "Fnmsubs",
    "Fnmadds", "Fadd", "Fsub", "Fmul", "Fdiv", "Fmadd", "Fmsub", "Fneg", "Fabs", "Fmr", "Frsp", "Fcmpu", "PsAdd", "PsSub",
    "PsMul", "PsMadd", "PsMsub", "PsNmadd", "PsNmsub", "PsMuls0", "PsMuls1", "PsMadds0", "PsMadds1", "PsSum0", "PsSum1",
    "PsMerge00", "PsMerge01", "PsMerge10", "PsMerge11", "PsMr", "PsNeg", "PsAbs", "Bc", "CmpwiBc", "CmplwiBc", "CmpwBc",
    "CmplwBc", "RlwinmRcBc", "Jmp", "B", "Bl", "Blr", "Bclr", "Bctr", "Bctrl", "End",
};
inline constexpr uint32_t kKinds = sizeof(kKindNames) / sizeof(kKindNames[0]);

struct JitLink;  // a link slot, the backend's

struct Op {
    K kind;
    uint8_t d, a, b;
    uint32_t imm;  // Generic: the instruction word
};

// A straight run of guest code: Build follows a b to untranslated code (a Jmp op), so a block can be up to
// kMaxSegments runs of consecutive words.
struct Segment {
    uint32_t start, count;
    uint32_t firstMark, lastMark;  // its dirty marks (BrawlDirtyMemory::leanBlocks), one per 64 guest bytes
};
// The dirty marks a block's code lies under, as one 8-byte load each: the marks from `at` that `mask` keeps. A block
// usually needs one or two (its segments are near each other); unused ones keep no marks.
struct MarkCheck {
    uint64_t mask;
    uint32_t at;
};
struct Block {
    uint32_t start;
    uint32_t count;
    uint32_t end;           // the last op's pc
    uint32_t segments;
    const uint32_t* words;  // the guest words decoded, as FlatRead32 returns them
    const Op* ops;          // count of them, then End
    const uint32_t* pcs;    // each op's pc
    const CallSite* calls;  // its bls' targets (a Bl op's b: index + 1; 0: resolved at the call)
    const Segment* segment;
    uint32_t epoch;         // the BrawlDirtyMemory::markEpoch it last matched guest memory in (0: compare again)
    uint32_t lastNext;      // BRAWL_INTERP_STATS: where it went last time
    Block* link[2];         // the blocks it went on to last (most recent first), tried before the table
    uint32_t checks;        // the mark loads it needs (Current always does the first two)
    bool hook;              // it starts where Run has a hook: entered through Run's loop, never a link
    MarkCheck check[kMaxSegments < 2 ? 2 : kMaxSegments];
    // The JIT's (docs/jit-design.md); Build clears them. The backend owns what they point to.
    uint32_t entries;       // times the dispatcher entered it while it had no code
    uint8_t jitState;       // 0: not tried, 1: compiled, 2: not compilable (the interpreter runs it)
    // Guarded code (BrawlDirtyMemory::guardedWritten, JitGuardedLine): the backend compiled the block without the dirty-mark
    // tests of its guarded lines (on its linked entry and after its calls), as a mark of one of them drops it
    // (BlockCache::GuardedWritten). jitNoGuard: the backend must not (its start was dropped that way too often:
    // BlockCache::unguarded); guardDrops counts those drops, and jitDemote asks the block cache to build the block again
    // (with all its tests) instead of refreshing it.
    bool jitGuarded;
    bool jitNoGuard;
    bool jitDemote;
    uint8_t guardDrops;
    const void* jitEntry;   // compiled code's entry that skips the validity check; null when not compiled
    const void* jitChecked; // the entry that performs all of it (epoch and marks): entries from outside compiled code
    // The entry that tests only the dirty marks, which link slots and inline caches aim at. A slot is aimed at a block only
    // while its epoch is the current one: whatever retires a block (epoch 0) resets the slots aimed at it first, and a new
    // epoch resets every slot (BlockCache::NoteEpoch) before compiled code can follow one (JitNoteEpoch).
    const void* jitLinked;
    JitLink* jitIncoming;   // head of the link slots aimed at this block (the backend's list)
    // The backend's record of the places its code holds this block's epoch (the epoch tests of the entry from outside and of
    // the tests after calls): every write of `epoch` while the block has code goes through BlockCache::SetEpoch, which has
    // JitBackend::setEpoch copy it there. Null when not compiled.
    void* jitEpoch;
};

// --- the backend interface ------------------------------------------------------------------------------------------

enum class JitStatus : uint32_t { Continue, Return, Pending };
struct JitCodeCache;  // the backend's: one per BlockCache, owns its blocks' code
// next: where to go on, for Continue. link: the slot the exit left through, or null.
struct JitExit { JitStatus status; uint32_t next; JitLink* link; };
// L1 direct-mapped cache of compiled, non-hook entry points -> jitChecked, shared with jit_subcall.
struct FastEntrySlot {
    uint32_t pc = 0;
    uint32_t pad = 0;
    const void* jitChecked = nullptr;
};
inline constexpr uint32_t kFastEntrySlots = 2048;
inline uint32_t FastEntryIdx(uint32_t pc) { return ((pc >> 2) * 0x9E3779B1u) >> 21; }

// Per-thread state passed to jit_enter_direct so its fast exit counts instructions without TLS, jit_subcall can probe
// fastEntries without TLS, and its slow exit can continue in RunLoop. First field is insnCount (read by exitReturn).
struct DirectEnterState {
    uint64_t* insnCount;
    void* threadState;
    const FastEntrySlot* fastEntries;
};

// Hooks registered once by GuestCodePatch::Protect (when code variants are off) so JIT-to-patched-JIT calls can
// bypass RunPatched/RunCompiledEntry/jit_enter_direct while preserving redirect/unpatch/Sync checks. GuestCodePatch calls
// BrawlDirtyMemory::MarksCleared whenever a guard byte drops to 0 or lowestRedirect moves: a linked call of a patched function's
// block tests neither (its link's stub does), and that suspends every link before compiled code follows one.
struct PatchedHooks {
    const uint8_t* (*patchedByteFor)(uint32_t address) = nullptr;
    const std::atomic<uint64_t>* pendingFast = nullptr;
    const std::atomic<bool>* anyDirty = nullptr;
    const std::atomic<uint32_t>* lowestRedirect = nullptr;
    void (*sync)() = nullptr;
    // For RunPatchedJit (GuestCodePatch::RunPatched with code variants off): the third flag of RunPatched's Sync test (open
    // edge pages), and RunPatched's own path for a call RunPatchedJit does not make. Both null: RunPatched never hands calls
    // to RunPatchedJit.
    const std::atomic<uint32_t>* openEdgeCount = nullptr;
    void (*runPatched)(CpuContext* ctx, uint32_t address) = nullptr;
};
void RegisterPatchedHooks(const PatchedHooks& hooks) noexcept;
const PatchedHooks& GetPatchedHooks() noexcept;
// GuestCodePatch::RunPatched for a patched function while code variants are off (with the hooks above registered): below the
// redirect bound, an entry the thread's fast-entry table holds is entered as RunCompiledEntry enters it, then RunPatched's Sync
// test; any other call is hooks.runPatched's. RunPatched tail-calls it, so the common call is one C++ function, not three.
void RunPatchedJit(CpuContext* ctx, uint32_t address);

// Direct call to a non-throwing-guard fprsOnly target (no try/catch: unwinds directly through chunk .eh_frame).
void JitDirectFprsCall(CpuContext* cpu, const JitCallTarget* t);
// The same for a target with a GPR guard (a native): ThinGuarded's work, its guards restoring when the callee throws.
void JitDirectGuardedCall(CpuContext* cpu, const JitCallTarget* t);

using EnterDirectFn = void (*)(CpuContext* ctx, uint32_t entry, const void* jitChecked, DirectEnterState* state);

struct JitBackend {
    JitCodeCache* (*createCache)();
    void (*destroyCache)(JitCodeCache* cache);          // frees every block's code in it
    bool (*compile)(JitCodeCache& cache, Block& b);      // sets b.jitEntry / b.jitChecked; false: leave b to the interpreter
    JitExit (*enter)(Frame& f, Block& b);                // runs b from its first op, and every block reached through links, until one exits
    void (*link)(JitLink& slot, Block& to);              // aims slot at `to` (compiled) and adds it to to.jitIncoming
    void (*retire)(Block& b);                            // b is being replaced or dropped: resets every slot aimed at it, clears b.jitIncoming
    size_t (*codeBytes)(const JitCodeCache& cache);      // bytes of code the cache holds (the interpreter drops the cache past a limit); may be null
    EnterDirectFn enterDirect;                           // fast AOT->JIT trampoline (may be null)
    // b no longer matches guest memory as far as the interpreter knows (epoch 0, or an old epoch): every slot aimed at b leaves
    // through its unlinked stub until resume, but stays on b.jitIncoming.
    void (*suspend)(Block& b);
    // b matches guest memory in the current epoch again (a refresh found its words unchanged): every slot on b.jitIncoming is
    // aimed at b again.
    void (*resume)(Block& b);
    // b.epoch changed: the code that tests it gets the new value (b.jitEpoch).
    void (*setEpoch)(Block& b);
};

// Guarded code: whether the dirty mark `mark` (one per 64 guest bytes of the canonical window) of a block's segment starting at
// segmentStart covers a guarded line: its 4 KiB page is registered as executable (RecompMod's write guard) and the store policy
// flags its 64 KiB chunk. Every write that marks such a line marks through BrawlDirtyMemory::Mark or LeanMarkRange (guest stores
// there take the checked path), so a compiled block may leave the line's mark untested once the line is in the guard tables
// (BlockCache::NoteGuardedBlock), whose hook then drops it. Executable ranges are only ever added, so the answer stays true.
bool JitGuardedLine(uint32_t segmentStart, uint32_t mark) noexcept;

// Whether compiled code may run now: not while XER[SO] is set. The JIT's compares store the field value without the SO copy
// PpcCrvCompare makes (BlockEmitter::CrStore), which is exact only while SO is clear; nothing the game runs sets it, so while
// it is set the dispatcher interprets (every entry into compiled code tests this first: RunLoop, RunCompiledEntry), and
// compiled code that may have set it (a call into guest code, a fallback op, mtxer) leaves for the dispatcher.
inline bool JitBlocksMayRun(const CpuContext* c) noexcept { return MKW_LEAN_FLAGS || (c->xer & 0x80000000u) == 0; }

// Called from jit_enter_direct's slow exit when a compiled block leaves without returning (stale, unlinked, hook, or
// pending exception), and for the frame the JIT makes for a function it called with a host call (JitLiftContinue; state null: the
// thread's). May throw (unwinds through the blocks' FDE).
void JitDirectContinue(DirectEnterState* state, Frame* f, uint64_t statusAndNext, JitLink* link);
// The calling thread's fast-entry table (BlockCache::fastEntries), for jit_subcall.
const FastEntrySlot* JitThreadFastEntries() noexcept;
// The backend this build has, or null: JitX64Backend() under MKW_JIT (defined in src/jit/jit_x64.cpp; it may itself
// return null), otherwise none.
const JitBackend* Backend();
#if MKW_JIT
const JitBackend* JitX64Backend();
#endif

// --- what compiled code calls ---------------------------------------------------------------------------------------
// JitFallbackOp lets no exception out: it catches it, stores it in f->pending and reports it.

// Link slots are aimed only at blocks that matched guest memory in the epoch held here (Block::jitLinked): when
// BrawlDirtyMemory::markEpoch moves (guest memory rewritten without marks), every slot is reset before compiled code can
// follow one. The interpreter resets them when it first validates a block in the new epoch; code that returns into a block
// after running C++ or guest code that may have moved the epoch either tests the block's own epoch (EmitRecheck), compares
// the two cells (BlockEmitter::ExitDynamicAfterCall), or calls JitNoteEpoch before returning.
inline uint32_t g_jitLinkEpoch = 0;
inline const uint32_t* JitLinkEpochCell() noexcept { return &g_jitLinkEpoch; }
// A test aid (jit_differential_tests): the link epoch made wrong on purpose so that the tests can show they notice. 1: a new
// epoch suspends no slot; 2: JitNoteEpoch does nothing; 3: ExitDynamicAfterCall takes the inline cache unguarded (read when a
// block is compiled); 4: a block's epoch is not copied into its code (JitBackend::setEpoch); 5: a write under guarded code drops
// nothing (BlockCache::GuardedWritten); 6: a restore's writes are not compared, every block it would drop is kept
// (BlockCache::RestoreWritten). 0 in a real run.
inline int g_jitEpochFault = 0;
// JitNoteEpoch's work when the epochs differ: every slot suspended (JitBackend::suspend), the cell set to markEpoch.
void JitSweepLinks() noexcept;
// For the helpers compiled code calls that may run guest code or write guest memory and return into the middle of a block
// (JitFallbackOp, the store shims): when markEpoch is no longer the epoch the slots were aimed in, every slot is suspended.
inline void JitNoteEpoch() noexcept {
    if (__builtin_expect(BrawlDirtyMemory::markEpoch.load(std::memory_order_relaxed) != g_jitLinkEpoch, 0)) JitSweepLinks();
}

// Runs op `index` of block b exactly as RunBlock<false> does for that op (RunOp, then Exec when RunOp answers Act::Exec;
// a fused compare runs its bc too).
// Returns (uint64_t(what) << 32) | next, what: 0 = go on with the next op (after a fused pair: the op after the bc;
// next is that op's index), 1 = leave the block for `next`, 2 = the frame returned, 3 = an exception is pending in f.
uint64_t JitFallbackOp(Frame* f, const Block* b, uint32_t index) noexcept;

// The branch group's shims (src/jit/jit_x64_branch.cpp). Each answers (uint64_t(what) << 32) | value and lets no exception
// out (it is stored in f->pending, what 3).
//
// JitCallOp: op `index` of b, a Bl or a Bctrl, as RunOp runs it up to the AfterCall test: ctx->pc is stored, then Call.
//   what 0: the call was made (the block goes on after a validity test); 1: the target (value) is code to interpret in
//   this frame, lr is set; 3: pending.
uint64_t JitCallOp(Frame* f, const Block* b, uint32_t index) noexcept;
// JitTailOp: op `index`, a B or a Bctr, as Exec's cases 18 and 528 run it for a non-linking branch, without the
// GuestCodePatch continuation (the caller has ruled it out for a B). what 1: leave the block for value; 2: the frame
// returned; 3: pending; 4 (only Bctr): the target (value) is no translated function, the jump is a plain one.
uint64_t JitTailOp(Frame* f, const Block* b, uint32_t index) noexcept;
// --- thin calls (docs/jit-design.md) -----------------------------------------------------------------------------------------
// The facts for calling or tail calling `target`, when the call can be made as a thin one: the registry is published, the
// target has a translation and a raw dispatch record. Null otherwise. A call (not a tail call) to a target with `special`
// set goes through Call.
const JitCallTarget* JitCallTargetFor(uint32_t target) noexcept;

// What a compiled call site that remembers the last ctr (a bctrl, a bctr) keeps: the key (guest target, or kNoKey), its record, and
// the code the site calls for it (JitFastCallFor).
struct JitTargetCell {
    uint32_t target;
    uint16_t argMask;                // record->argMask and record->entry, beside the key: what the call routines read on every call
    uint8_t wayHits;                 // the site's: hits of this entry out of line, past its inline ways (it goes in them every 32)
    uint8_t hits;                    // the site's: hits of this entry while it is not the first (it moves to the first every 8)
    const JitCallTarget* record;
    const void* fast;
    void (*entry)(CpuContext*);
    static constexpr uint32_t kNoKey = 1;
};
// The code a compiled bctrl site (with `tail`, a bctr site) calls for the target t of a JitTargetCell entry, with the context in the
// first argument register and the entry in rdx: t's entry itself when nothing has to be done around it, else a routine of the
// backend's that does what JitThinCall does for t. Null without a backend that calls it (MKW_JIT off, or not Linux x86-64).
const void* JitFastCallFor(const JitCallTarget* t, bool tail) noexcept;

// How many targets a call site's cell remembers. A cell is that many JitTargetCell entries in a row, the newest first, a probe
// tests them in that order.
inline constexpr uint32_t kCellWays = 8;  // the virtual calls of a task list (gfTask::process) cycle through 5 to 8 targets per site

// A Bl or Bctrl whose callee is `t`, after ctx->pc and lr are set and the rollback loop frame is ruled out: exactly what
// Call does for it (float arguments, the dispatch record's guards and scopes, the call, the float return), nothing looked up.
//   what 0: the call was made; 3: pending.
uint64_t JitThinCall(Frame* f, const JitCallTarget* t) noexcept;
// The same for a call site that has made the float arguments of `t` right itself (JitNormalizeArgs, when one is held paired).
uint64_t JitThinCallPrepared(Frame* f, const JitCallTarget* t) noexcept;
uint64_t JitThinTailPrepared(Frame* f, const JitCallTarget* t) noexcept;
// What Call does to the float arguments of a callee `t` stands for: a register named in t->argMask that is held paired becomes the
// scalar it stands for.
void JitNormalizeArgs(Frame* f, const JitCallTarget* t) noexcept;
// A B or Bctr tail call to `t` (the loop frame ruled out), as TailCall: what 2: the frame returned; 1: go on at value (lr);
// 3: pending.
uint64_t JitThinTail(Frame* f, const JitCallTarget* t) noexcept;
// JitCallOp / JitTailOp for a site that remembers its last target: the same, and `cell` is filled with the target's
// record when a thin call to it can be made.
uint64_t JitCallOpCached(Frame* f, const Block* b, uint32_t index, JitTargetCell* cell) noexcept;
uint64_t JitTailOpCached(Frame* f, const Block* b, uint32_t index, JitTargetCell* cell) noexcept;
// SetScalar(f, 1, Scalar(f, 1)): the normalisation of f1 on a frame's return, for the case where f1 is Paired.
void JitNormalizeF1(Frame* f) noexcept;
// The cell holding the pointer of the rollback loop frame (Call compares the frame with its current value).
const void* const* JitRollbackLoopFrameCell() noexcept;
// Whether GuestCodePatch::ContinuationAt(pc) can ever answer non-null (GuestCodePatch::ContinuationPossibleAt), for a
// block's compile to ask about a b's target. False is permanent for that pc.
bool JitContinuationPossibleAt(uint32_t pc) noexcept;

// The interpreter's own routines, for native code to call or mirror. They may throw (a guest call into code that does).
Step Exec(Frame& f, uint32_t pc, uint32_t w, uint32_t& next, uint32_t returnAddress);
// A call (bl, bcl, blrl, bctrl): true when it was made (translated, native or patched code), false when the target is
// code to interpret in this frame.
bool Call(Frame& f, uint32_t target, uint32_t returnTo, const CallSite* site = nullptr);
bool TailCall(Frame& f, uint32_t target, uint32_t returnAddress);
bool LoopTailCall(Frame& f, uint32_t target, uint32_t& next);
bool TailCallEntry(Frame& f, void (*entry)(CpuContext*), uint32_t returnAddress);
uint32_t ReadSpr(CpuContext* c, uint32_t spr, uint32_t pc, uint32_t w);
void WriteSpr(CpuContext* c, uint32_t spr, uint32_t value);
bool IsHookPc(uint32_t pc);

// Run(ctx, entry) for an entry whose first block has compiled code, current, and is no hook pc, without the general entry's
// setup: true when it ran, false when it did nothing and the caller makes the general call.
// Defined in ppc_interpreter.cpp; GuestCodePatch::RunPatched's last step is its caller, and Run's first.
bool RunCompiledEntry(CpuContext* ctx, uint32_t entry);
// Run without its RunCompiledEntry attempt: for a caller that has just made it.
void RunGeneral(CpuContext* ctx, uint32_t entry);

}  // namespace PpcInterpreter::detail
