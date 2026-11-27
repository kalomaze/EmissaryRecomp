// The x86-64 backend's block emitter (docs/jit-design.md): what the per-group code generators (jit_x64_int.cpp,
// jit_x64_mem.cpp, jit_x64_branch.cpp, jit_x64_fp.cpp, and jit_x64_generic.cpp for the Generic ops) write native code against.
// Private to src/jit.
//
// A group's entry point is called once per op while the core compiles a block, with the emitter positioned where the
// op's code goes. It emits native code for op `index` (and, for a fused compare and bc, the bc after it), or returns 0
// having emitted nothing so the core emits the call to the interpreter's handler (JitFallbackOp) instead. A group may
// also fall back for one case of an op it mostly handles: `return e.EmitFallback(index);`.
//
// What the code around an op guarantees, and what an op's code must leave:
//
//  * Pinned registers (docs/jit-design.md): rbx = CpuContext* + kCtxBias (see kCtxBase), r12 = lean window base,
//    r13 = Frame*, r14 = the address of BrawlDirtyMemory::markEpoch (the block entry's epoch test reads it: kEpochCell),
//    r15 = the executed-instruction count of this enter (kExecCount). rbp is free (callee-saved; the trampoline saves it).
//    rax, rcx, rdx, rsi, rdi, r8-r11 and xmm0-xmm5 are scratch; every one of them is clobbered by a call (CallHelper), on both
//    host ABIs. No register holds a value from one op to the next.
//  * All guest state lives in memory between ops and is reached through the accessors below (Gpr, Crv, Lr, ...), never
//    through rbx plus an offset written by hand.
//  * rsp is 16-byte aligned throughout a block. Native code must not push, pop or move rsp (a call needs the alignment,
//    and on Windows the 32-byte shadow space sits at rsp already).
//  * ctx->pc is not stored at the block entry (BRAWL_JIT_BLOCK_PC=1 stores it): it holds what the dispatcher stored when it
//    entered the chain of linked blocks, or what a call or slow path stored last. Code that calls something needing the current
//    pc stores it first (StorePc).
//  * The whole block's op count has been added to the count in r15 (kExecCount) at its entry; the enter trampoline's epilogue
//    adds r15 to frame.executed (BRAWL_JIT_EXEC_MEM=1: the block adds its count to frame.executed in memory instead). The only
//    reader of frame.executed is the dispatcher, after the enter returns.
//  * Frame::form bytes: BlockEmitter::KnownForm says what a block knows of a register's byte on entry to an op (the analysis and its
//    rules are at the top of jit_x64_forms.cpp). An op that writes a form byte keeps writing it unless the byte is known to hold
//    the value; an op that changes a form with no kind the analysis lists makes it clear what it knows.
//  * Control leaves an op's code by falling through to the next op's code, or by one of the Exit* functions. The code
//    after the last op is the block's own fall-through exit (the core emits it).
//  * Anything a call into the interpreter or guest code can throw must be caught by a noexcept shim (design rule 4):
//    CallHelper only calls functions that do not throw. A shim stores the exception in frame.pending and answers so that
//    the code calls ExitPending.
//  * Code size: one op's code, cold paths included, stays under kMaxBytesPerOp bytes (the buffer is reserved from that;
//    a block that overflows is simply not compiled).
#pragma once

#include <xbyak/xbyak.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>
#include <type_traits>

#include "../ppc_interpreter_internal.h"

namespace PpcInterpreter::detail::x64 {

// --- host ABI ------------------------------------------------------------------------------------------------------

#if defined(_WIN32)
inline constexpr bool kWin64 = true;
#else
inline constexpr bool kWin64 = false;
#endif

// Direct calls from JIT blocks into lean AOT functions (unwound via per-chunk .eh_frame FDEs on Linux x86-64).
#if defined(__linux__) && defined(__x86_64__)
inline constexpr bool kDirectCalls = true;
#else
inline constexpr bool kDirectCalls = false;
#endif

// Integer argument registers of the host's C calling convention (SysV: rdi, rsi, rdx, rcx, r8, r9; Win64: rcx, rdx, r8,
// r9). CallHelper takes at most kMaxCallArgs arguments, the number both ABIs pass in registers.
inline constexpr unsigned kMaxCallArgs = 4;
const Xbyak::Reg64& ArgReg(unsigned i);

// The registers every block runs with.
// rbx holds the CpuContext's address plus kCtxBias, so that the guest GPRs (the first 128 bytes of the context) and the
// registers after them (the eight CR field values, lr, ctr, xer, fpscr, pc and the first four FPRs) are all reached with a
// one-byte displacement (the static_asserts in jit_x64.cpp hold the layout to that). Code reaches context fields only through
// the accessors (Gpr, Crv, ..., CtxField), never with rbx and a hand-written offset.
inline constexpr int kCtxBias = 0x80;
inline const Xbyak::Reg64 kCtxBase = Xbyak::util::rbx;  // CpuContext* + kCtxBias
inline int CtxDisp(size_t offsetInContext) { return static_cast<int>(offsetInContext) - kCtxBias; }  // from kCtxBase
inline const Xbyak::Reg64 kLean = Xbyak::util::r12;     // lean window base: guest address a is at [kLean + (a & kLeanMask)]
inline const Xbyak::Reg64 kFrame = Xbyak::util::r13;    // Frame*
inline const Xbyak::Reg64 kEpochCell = Xbyak::util::r14;  // &BrawlDirtyMemory::markEpoch (a uint32_t): the block entry's epoch test
// The guest instructions the blocks have counted since the enter trampoline started: every block entry adds its op count here,
// and the trampoline's epilogue adds the total to frame.executed (the only reader of that field is the dispatcher, after the
// enter returns). Callee-saved on both host ABIs, so a helper call keeps it. Zeroed by the trampoline.
inline const Xbyak::Reg64 kExecCount = Xbyak::util::r15;

inline constexpr uint32_t kMaxBytesPerOp = 768;

// The groups of op kinds, and each kind's. A kind's group's entry point is asked first; Jmp and End have none (the core handles
// them). A Generic op asks EmitGeneric (jit_x64_generic.h) for native code of its word and is a fallback call when it answers 0.
enum class Group : uint8_t { None, Int, Mem, Branch, Fp };
Group GroupOf(K kind);

// Per-group entry points: native code for op `index` of e.block (e.index == index), or 0 (nothing emitted). Otherwise the
// number of ops consumed: 1, or 2 for a fused compare and bc (CmpwiBc ... RlwinmRcBc) handled whole.
//   Int:    Li ... Mtspr          (integer, compares, spr moves)
//   Mem:    Lwz ... PsqStu        (loads and stores, integer and float, quantized)
//   Branch: Bc, the fused compare and bc kinds, B, Bl, Blr, Bclr, Bctr, Bctrl
//   Fp:     Fadds ... PsAbs       (floating point arithmetic and moves)
class BlockEmitter;
uint32_t EmitInt(BlockEmitter& e, uint32_t index);
uint32_t EmitMem(BlockEmitter& e, uint32_t index);
uint32_t EmitBranch(BlockEmitter& e, uint32_t index);
uint32_t EmitFp(BlockEmitter& e, uint32_t index);

// One argument of a helper call: an immediate (integer or pointer), a register (a 32-bit or smaller register is
// zero-extended), or a memory operand (loaded with its own size, zero-extended). Registers and the base and index of a
// memory operand may be argument registers: CallHelper orders the moves so none is overwritten before it is read.
class CallArg {
public:
    enum class Kind : uint8_t { Imm, Reg, Mem };
    CallArg() : kind(Kind::Imm) {}
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T> || std::is_enum_v<T>>>
    CallArg(T v) : kind(Kind::Imm), imm(static_cast<uint64_t>(v)) {}
    template <typename T>
    CallArg(T* p) : kind(Kind::Imm), imm(reinterpret_cast<uintptr_t>(p)) {}
    CallArg(const Xbyak::Reg& r) : kind(Kind::Reg), reg(r) {}
    CallArg(const Xbyak::Address& m) : kind(Kind::Mem), mem(&m) {}  // m must outlive the CallHelper call
    Kind kind;
    uint64_t imm = 0;
    Xbyak::Reg reg;
    const Xbyak::Address* mem = nullptr;
};

// A block's stale path (the validity test failed on a linked entry) jumps through the `target` of this cell (rcx holds the cell's
// address), data after the block's code. It starts at the counting code shared by all blocks, which counts the failures; past
// the threshold InstallStaleStub, a noexcept helper, makes a stub in the code cache that, when the block's epoch is the current
// one, compares the guest words of the block in place and enters the block when all are unchanged, else leaves as the stale
// path always does (through `plain`, which asks the dispatcher). A block whose dirty marks stay set (a mod variable inside its
// 64 bytes) then no longer goes round the dispatcher on every entry. The stub clears no mark and does not touch the block's
// epoch: it enters exactly when the dispatcher's Refresh, which would find the words unchanged, enters.
struct StaleCell {
    const void* target;    // read by the stale path: first member
    uint32_t failures;     // stale paths taken through the shared counting code
    uint32_t stubbed;      // 1: a stub was made (or making one failed): the cell is final
    const Block* block;
    void* cache;           // the JitCodeCache that owns the block's code
    const void* plain;     // the stale exit through the dispatcher
};
// What the counting code calls at the threshold: makes the block's stub, aims the cell's target at it, and answers the host
// address to go on at (the stub, or the plain exit when no stub could be made).
const void* InstallStaleStub(StaleCell* cell) noexcept;

// What one block's compile makes. Constructed by the core over `size` bytes at `buffer`; the group generators only use
// the members and functions below.
class BlockEmitter {
public:
    BlockEmitter(const Block& b, void* buffer, size_t size);
    ~BlockEmitter();
    BlockEmitter(const BlockEmitter&) = delete;
    BlockEmitter& operator=(const BlockEmitter&) = delete;

    // Forward jumps to undefined labels are near (32-bit) unless T_SHORT is given; emits at the current position; throws
    // Xbyak::Error when the buffer is full (the block is then not compiled).
    Xbyak::CodeGenerator a;
    const Block& block;
    uint32_t index = 0;      // the op being emitted

    const Op& CurrentOp() const { return block.ops[index]; }
    // The guest pc of op i (a block can span several segments: not always start + 4 * i).
    uint32_t PcOf(uint32_t i) const { return block.pcs[i]; }

    // --- guest state: operands in memory (sizes as the guest register holds them) -------------------------------------
    Xbyak::Address Gpr(uint32_t r) const;      // dword
    Xbyak::Address Crv(uint32_t field) const;  // qword: CR field `field` (0 = cr0 ... 7) as CpuContext::crv holds it (isa/ppc_isa_crv.h)
    Xbyak::Address Lr() const;                 // dword
    Xbyak::Address Ctr() const;                // dword
    Xbyak::Address Xer() const;                // dword
    Xbyak::Address Pc() const;                 // dword: ctx->pc
    Xbyak::Address Gqr(uint32_t i) const;      // dword
    Xbyak::Address Fpr(uint32_t r) const;      // qword: the register's 64 raw bits (CpuContext form: ps0 as a double or paired)
    Xbyak::Address FprWord(uint32_t r, uint32_t half) const;  // dword; half 0: the low word (ps1), 1: the high word (ps0)
    Xbyak::Address PairedMask() const;         // dword: Frame::pairedMask (bit r = 1 iff fpr[r] is Paired)
    Xbyak::Address FrameFlags() const;         // dword: Frame::flags (kFrameFlag*)
    Xbyak::Address Executed() const;           // qword: Frame::executed
    // The same registers' addresses by host offset into CpuContext, for state this list lacks (offsetof(CpuContext, ...)).
    Xbyak::Address CtxField(size_t offset) const;
    // Guest memory: the host address of guest address `ea` held in a 32-bit register (already zero-extended in its 64-bit
    // register) is [kLean + (ea & kLeanMask)]; the accessors for it belong to the memory group (stores also mark dirty
    // blocks and test the store policy, see memory_access.h).

    // --- exits: control leaves the block; nothing after an Exit* is reached by falling through ---------------------------
    // To the statically known guest pc: through a link slot (`jmp [slot]`) that the dispatcher aims at the target block's
    // checked entry once the target is compiled, or a plain dispatcher exit when targetPc is a hook pc (IsHookPc). One
    // slot per target per block, shared by every ExitLinked to it. Emits only the jump; the slot's unlinked stub and
    // the slot itself go after the block.
    void ExitLinked(uint32_t targetPc);
    // To the pc in eax, through the dispatcher (no link): jumps through ctr or lr, calls, anything not static.
    void ExitDynamic();
    // To a static pc through the dispatcher, no link (mov eax, pc; then ExitDynamic).
    void ExitTo(uint32_t targetPc);
    // To the pc in eax, through an inline cache of the targets the dispatcher linked this block's dynamic exits to: a
    // direct-mapped table of kCacheEntries[which] entries, selected by (pc >> 2) & (entries - 1); an entry holds a guest pc
    // (the key) and the checked entry of the block compiled for it. A pc that is cached is jumped to directly; any other
    // leaves for the dispatcher, handing it the table's first entry as its link slot, and the dispatcher's link() puts the
    // block in the entry its pc selects (replacing what was there). A key is only ever the start of a block that was compiled
    // and is not a hook block (the dispatcher links no other); retire of that block empties the entry. Because a hit skips the
    // dispatcher's Find, an exit that must make a decision before it goes to a pc (is it a translated function? is it a
    // call's interpret-here answer?) makes that decision first and only then uses ExitDynamicCached; a jump that is probed
    // before the decision (see CacheProbe) uses a cache of its own (`which`) that only decided pcs ever enter. Clobbers rax,
    // rcx, rdx, r8. A block has kCaches caches; they share nothing.
    static constexpr uint32_t kCaches = 2;
    static constexpr uint32_t kCacheEntries[kCaches] = {4, 16};  // powers of two; cache 1 belongs to bctr (jit_x64_branch.cpp)
    void ExitDynamicCached(uint32_t which = 0);
    // The two halves of ExitDynamicCached, for code that decides something on a miss: CacheProbe jumps to the cached
    // target of eax's pc on a hit and falls through on a miss (emits no jump past itself); ExitDynamicMiss then leaves
    // for eax's pc through the dispatcher with the cache's slot (eax must still hold the pc). Several probes and misses
    // of one block share the cache.
    void CacheProbe(uint32_t which = 0);
    void ExitDynamicMiss(uint32_t which = 0);
    // The frame's function returned (a blr to the address it was entered with), or an exception is stored in
    // frame.pending.
    void ExitReturn();
    void ExitPending();

    // --- calls ------------------------------------------------------------------------------------------------------
    // Calls fn with up to kMaxCallArgs integer arguments loaded into the host's argument registers (see CallArg). The result
    // is in rax (xmm0 for a double or float). fn must not throw. Clobbers every scratch register. The stack stays aligned
    // (the trampoline's frame holds the shadow space on Windows).
    template <typename F, typename... Args>
    void CallHelper(F* fn, Args... args) {
        static_assert(sizeof...(Args) <= kMaxCallArgs);
        CallHelperImpl(reinterpret_cast<const void*>(fn), {CallArg(args)...});
    }
    // fn's address is in rax already, or is loaded here: for helpers with floating point arguments, whose xmm registers
    // the caller has set (xmm0 and xmm1 are the first two on both ABIs; on Win64 float arguments share positions with the
    // integer ones, so a helper that takes (double, uint32_t) has the integer in rdx there and rsi on SysV).
    void CallRaw(const void* fn);
    // Calls the shared JIT->patched-JIT subcall trampoline (eax = target, rdx = const JitCallTarget*). Linux x86-64 only.
    void CallSubCall();

    // Op `index` as a call to the interpreter's handler and the handling of its answer (go on, or leave the block); the
    // number of ops it ran (2 for a fused pair). Used by the core for every op without native code.
    uint32_t EmitFallback(uint32_t index);
    // The same call as a slow path of op `index`'s own native code (a NaN result, say): the op stays native, so FallbackCount
    // does not count it.
    uint32_t EmitSlowPath(uint32_t index);
    // How many fallback calls this block has had emitted so far (a group that returns EmitFallback's value counts as one).
    uint32_t FallbackCount() const { return fallbackCalls_; }

    // --- condition register liveness (jit_x64_cr.cpp) -----------------------------------------------------------------
    // Computed by the core once per block (AnalyseCr, from EmitPrologue). CrDead(i, f): CR field f (0 = cr0 ... 7) is
    // overwritten in full by a later op of this block before anything that can read the condition register or leave the
    // block, so nothing observes the value op i leaves in it, on the fall-through path. i is the first op of an op's code
    // (a fused compare and bc: the compare; the answer is for after the pair). An op whose field is dead writes no CR; a
    // fused compare and bc whose field is dead branches on the host flags of its compare and must write the field before
    // it leaves, on the taken path. The classification of the op kinds is at the top of jit_x64_cr.cpp.
    bool CrDead(uint32_t i, uint32_t field) const { return ((crDead_[i] >> field) & 1u) != 0; }
    // The code that makes a field value the way the interpreter does (PpcCrvCompare), so that a field holds the same 64 bits
    // whichever engine wrote it. A compare is: the first operand in rax (a signed compare: sign-extended, an unsigned one: eax
    // loaded, so zero-extended), CrDiffImm or CrDiffGpr (rax = a - b in 64 bits), CrStore. A record form is: the result in eax,
    // movsxd rax, eax, CrStore.
    //   CrDiffImm / CrDiffGpr: b is an immediate or guest register r (extended as the compare's signedness says). Clobbers rcx.
    //     The host flags afterwards describe the difference, as 64-bit signed (l, g, e) for both signednesses, when `flags` is
    //     set (CrDiffImm: a difference of an immediate 0 emits nothing otherwise).
    //   CrStore: rax is the value. Unless MKW_LEAN_FLAGS, a set XER[SO] makes it PpcCrvWithSo's (a call out of line, cold);
    //     then it is stored in the field. Leaves rax as it was unless that call is made; clobbers the flags.
    void CrDiffImm(bool isSigned, uint32_t imm, bool flags);
    void CrDiffGpr(bool isSigned, uint32_t reg);
    void CrStore(uint32_t field);
    // The other whole-register CR operations, which call a noexcept helper (they are rare): eax = the packed CR (mfcr), and
    // crnor ... cror (op as PpcCrvLogic's) on CR bits bt, ba, bb as the instruction numbers them. Clobber every scratch register.
    void CrPack();
    void CrLogic(uint32_t op, uint32_t bt, uint32_t ba, uint32_t bb);
    // mtcrf: the fields crm selects (bit 7 = cr0) from the guest register rs, as PpcCrvEncode makes them. Clobbers rax, rcx, rdx, rsi.
    void CrFromGpr(uint32_t rs, uint32_t crm);
    // Counters the tests read (JitX64GetCrStats).
    enum class CrEvent : uint8_t { WriteSkipped, FusedFlagsOnly, CompactWrite, FusedLive };
    static void NoteCr(CrEvent e);

    // --- FPR form knowledge (jit_x64_forms.cpp) ---------------------------------------------------------------------------
    // Computed by the core once per block (AnalyseForms, from EmitPrologue). KnownForm(i, r): Form::Scalar or Form::Paired when
    // Frame::form[r] is known to hold that value on entry to the code of op i (a fused compare and bc: the compare; the sources
    // of a float op are read with this knowledge, its destination is written after), Form::Unknown when it is not known.
    // Knowledge is about the byte in memory, which every op keeps up to date. The rules are at the top of jit_x64_forms.cpp.
    PpcInterpreter::detail::Form KnownForm(uint32_t i, uint32_t r) const {
        const uint32_t bit = 1u << (r & 31u);
        using F = PpcInterpreter::detail::Form;
        return (formScalar_[i] & bit) != 0 ? F::Scalar : (formPaired_[i] & bit) != 0 ? F::Paired : F::Unknown;
    }
    // Counters the tests read (JitX64GetFormStats): a form test not emitted because the form is known and the source needs no
    // conversion, a conversion emitted without a test because the form is known to differ, a test emitted, a form write not
    // emitted because the byte is known to hold the value, one emitted, a source load shared with an earlier operand of the op.
    enum class FormEvent : uint8_t { TestSkipped, ConversionForced, TestEmitted, WriteSkipped, WriteEmitted, SharedLoad };
    static void NoteForm(FormEvent e);

    // The block's validity test again (what its `checked` entry does): after native code has called something that may have
    // written guest memory or run guest code. If it fails, control leaves for nextPc, through the dispatcher. Falls through
    // when the block still matches guest memory. Clobbers rax, rcx.
    void EmitRecheck(uint32_t nextPc);

    // Stores the current pc of op `i` into ctx->pc (for helpers and handlers that read it).
    void StorePc(uint32_t i);

    // A cell of the block's own data, after its code: kCellWays JitTargetCells (16 bytes each) that start with no key, which a call
    // site fills at run time with its latest targets and what each resolved to (CellProbe reads them). Valid until the block is
    // done (jump or `lea` to it with rip-relative addressing).
    static constexpr uint32_t kMaxCells = 16;
    Xbyak::Label& NewCell();
    // The probe of a cell for the guest target in eax (already & ~3): falls through with rcx = the cell's address and rdx = the
    // record of the entry whose key is eax, or jumps to `miss` (near). A cell holds kCellWays entries (a JitTargetCell each,
    // after the code), the newest first; `NewCell` makes room for all of them.
    void CellProbe(Xbyak::Label& cell, const Xbyak::Label& miss);

    // --- intra-block GPR cache (ebp, r8d, r9d, r10d, r11d) -------------------------------------------------------------
    // Caches hot guest GPRs in 5 host registers within a block (ebp is callee-saved, r8d-r11d are volatile).
    // Flushed before calls, exits, and non-cache-aware ops; preserved across conditional branches' fall-through path
    // (dirty slots are flushed on the cold taken path).
    static constexpr uint32_t kGprCacheSlots = 5;
    struct GprSnapshot {
        int8_t guest[kGprCacheSlots] = {-1, -1, -1, -1, -1};
        uint8_t dirtyMask = 0;
    };
    static const Xbyak::Reg32& GprCacheReg(uint32_t slot);
    static const Xbyak::Reg64& GprCacheReg64(uint32_t slot);
    int FindCachedGpr(uint32_t r) const;
    bool GprReadLater(uint32_t r) const { return !inCold_ && ((gprReadLater_[index] >> r) & 1u) != 0; }
    bool WantCacheGpr(uint32_t r) const { return FindCachedGpr(r) >= 0 || GprReadLater(r); }
    const Xbyak::Reg32& ReadGpr(uint32_t r, uint32_t pinMask = 0);
    const Xbyak::Reg32& WriteGpr(uint32_t r, uint32_t pinMask = 0);
    const Xbyak::Reg32& ModifyGpr(uint32_t r, uint32_t pinMask = 0);
    void DropGpr(uint32_t r);
    void FlushGpr(uint32_t r);
    void FlushDirtyGprs();
    void FlushVolatileGprs();
    void FlushAllGprs();
    GprSnapshot SnapshotGprs() const;
    void EmitFlushSnapshot(const GprSnapshot& snap);
    void EmitReloadSnapshotVolatile(const GprSnapshot& snap);

    // --- cold code ----------------------------------------------------------------------------------------------------
    // A label owned by the emitter (valid until the block is done); jump to it from the main path, define it in Cold code.
    Xbyak::Label& NewLabel();
    // Defers `fn(BlockEmitter&)` to the cold section after the block's main path: out-of-line slow paths. fn emits code that
    // does not fall through its end (it ends in a jump or an Exit*, usually back to a label of the main path). fn must be
    // trivially copyable and at most kColdBytes: capture labels from NewLabel by reference, other values by copy. It runs
    // after the whole block is emitted (so what it captures must outlive the call that registered it: never a local's address).
    static constexpr size_t kColdBytes = 72;
    template <typename F>
    void Cold(const F& fn) {
        static_assert(sizeof(F) <= kColdBytes && std::is_trivially_copyable_v<F> && alignof(F) <= 8,
                      "a cold-path lambda captures a few values (labels by reference, via NewLabel)");
        ColdItem& c = NewCold();
        ::new (static_cast<void*>(c.storage)) F(fn);
        c.run = [](const void* p, BlockEmitter& e) { (*static_cast<const F*>(p))(e); };
    }

    // --- the core's -------------------------------------------------------------------------------------------------
    // The code cache the block's code is made in (the stale stub's, see StaleCell): set before EmitPrologue.
    void SetOwner(void* cache) { owner_ = cache; }
    // Emits the validity test and the block's entry (count, pc).
    void EmitPrologue();
    // After the last op: the cold section, the stale exit, the unlinked stubs, the link slots (data).
    void Finish();
    // Valid after Finish: the addresses of the two entries.
    const uint8_t* Checked() const { return checkedAt_; }
    const uint8_t* Entry() const { return entryAt_; }
    // Bytes of code before the link slots' data (for the perf map and dumps).
    size_t CodeBytes() const { return codeBytes_; }

    static constexpr uint32_t kMaxExits = 64;
    static constexpr uint32_t kMaxLabels = 96;
    static constexpr uint32_t kMaxCold = 64;

private:
    struct ColdItem {
        alignas(8) unsigned char storage[kColdBytes];
        void (*run)(const void*, BlockEmitter&);
    };
    struct Exit {
        uint32_t pc;
        uint32_t patchOffset;  // offset of 5-byte `jmp rel32` in block buffer, or kNoPatch for indirect
        Xbyak::Label stub;
        Xbyak::Label far;      // `jmp [rip + slot]` fallback when target lies outside ±2 GiB
        Xbyak::Label slot;
        static constexpr uint32_t kNoPatch = ~0u;
    };
    ColdItem& NewCold();
    void CallHelperImpl(const void* fn, std::initializer_list<CallArg> args);
    void AnalyseCr();
    void AnalyseForms();
    void AnalyseGprUses();
    Xbyak::Label staleCell_, plainStale_;
    struct GprSlot {
        int8_t guest = -1;
        bool dirty = false;
        uint32_t lastUse = 0;
    };
    uint32_t AllocGprSlot(uint32_t pinMask);

    GprSlot gprSlots_[kGprCacheSlots];
    uint32_t gprReadLater_[kMaxBlock + 1] = {};
    uint32_t gprTick_ = 0;
    bool inCold_ = false;
    void* owner_ = nullptr;
    bool staleStub_ = false;
    void EmitValidityTest(const Xbyak::Label& fail, bool shortJump);
    void EmitMarkTest(const MarkCheck& m, const Xbyak::Label& fail, bool shortJump);
    void JmpAbs(const void* target);

    struct DynCache {
        Xbyak::Label miss;   // the miss path (leaves for the dispatcher with the first entry as its slot)
        Xbyak::Label slots;  // the kCacheEntries entries (JitLink, data after the code)
        bool probed = false, missEmitted = false;
    };
    Xbyak::Label stale_;
    Xbyak::Label fallbackExit_;
    DynCache caches_[kCaches];
    bool fallbackUsed_ = false;
    uint32_t fallbackCalls_ = 0;
    const uint8_t* checkedAt_ = nullptr;
    const uint8_t* entryAt_ = nullptr;
    size_t codeBytes_ = 0;
    uint32_t exitCount_ = 0;
    uint32_t labelCount_ = 0;
    uint32_t coldCount_ = 0;
    uint32_t cellCount_ = 0;
    Xbyak::Label cells_[kMaxCells];
    uint8_t crDead_[kMaxBlock + 2] = {};  // per op: the CR fields dead after it (CrDead)
    uint32_t formScalar_[kMaxBlock + 2] = {}, formPaired_[kMaxBlock + 2] = {};  // per op: the FPRs known Scalar / Paired on entry (KnownForm)
    Exit exits_[kMaxExits];
    Xbyak::Label labels_[kMaxLabels];
    ColdItem cold_[kMaxCold];
};

}  // namespace PpcInterpreter::detail::x64

namespace PpcInterpreter::detail {
// A block's link slot, as the dispatcher sees it through JitLink*: a block's `jmp [slot]` goes to `target`, which is the
// slot's unlinked stub (exits Continue to the slot's pc, handing the dispatcher the slot's address) until link() aims it
// at a block's checked entry.
struct JitLink {
    const void* target;    // read by `jmp [slot]` (indirect / far / inline cache): first member
    const void* unlinked;  // the slot's own stub: what target and patchSite are reset to
    JitLink* next;         // the list of slots aimed at one block (Block::jitIncoming)
    JitLink** prev;        // the pointer that points at this slot (the block's head or the previous slot's next)
    Block* to;             // the block target aims at, or null
    // Inline-cache entries (BlockEmitter::ExitDynamicCached) only: the guest pc that selects this entry, or kNoKey, and,
    // in the first entry of a cache, the number of entries of the table (0 in an ordinary slot and the other entries).
    uint32_t key;
    uint32_t group;
    uint8_t* patchSite;    // for direct static exits: address of 5-byte `jmp rel32` (0xE9); null for indirect/cache
    const void* farTramp;  // for direct static exits: `jmp [rip + slot]` fallback when target is outside ±2 GiB
    static constexpr uint32_t kNoKey = 1;  // no guest pc equals it (pcs are word aligned)
};

// Counters of this process's backend (BRAWL_JIT_STATS prints them; the core's unit test reads them).
struct JitX64Stats {
    uint64_t compiled, compileFailed, codeBytes, links, retires, enters, cachesCreated, cachesDestroyed;
    uint64_t compiledOps;     // guest instructions in the blocks compiled
    uint64_t compileNanos;    // time spent in compile (only while BRAWL_JIT_STATS is on)
    uint64_t cacheHits, cacheMisses;  // inline-cache probes that jumped / that left for the dispatcher (only while BRAWL_JIT_STATS is on)
    uint64_t staleStubs = 0, staleStubFailures = 0;  // stale stubs made (StaleCell) / that could not be made
};
JitX64Stats JitX64GetStats();
// What the condition register analysis did (jit_x64_cr.cpp): blocks analysed, compare and record-form CR writes not emitted,
// fused compare and bc pairs that branch on the host flags and write the field on the taken path only, and CR writes
// emitted (compact form).
struct JitX64CrStats {
    uint64_t blocksAnalysed, writesSkipped, fusedFlagsOnly, compactWrites;
    uint64_t fusedLive = 0;  // fused pairs whose field is live that branch on the compare's flags and write the field on both paths
};
JitX64CrStats JitX64GetCrStats();
// What the FPR form analysis did (jit_x64_forms.cpp): blocks analysed, and the events BlockEmitter::FormEvent lists.
struct JitX64FormStats {
    uint64_t blocksAnalysed, testsSkipped, conversionsForced, testsEmitted, writesSkipped, writesEmitted, sharedLoads;
};
JitX64FormStats JitX64GetFormStats();
// How many ops of `kind` the backend compiled with native code (native) or as a call to the interpreter's handler (the fused
// kinds count once, for the compare's kind): tests use it to show a kind is not silently a fallback.
uint64_t JitX64KindOps(K kind, bool native);
}  // namespace PpcInterpreter::detail
