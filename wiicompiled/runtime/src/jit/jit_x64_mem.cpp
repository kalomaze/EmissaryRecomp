// Native code for the loads and stores (Lwz ... PsqStu) of the x86-64 backend, written against the emitter API in
// jit_x64.h (docs/jit-design.md).
//
// Every op does what its handler in ppc_interpreter.cpp (RunOp) does, in the same order of effects: a load reads, then
// writes its register, then an update form writes rA; a store that leaves the lean window's fast path (the policy byte of
// its 64 KiB chunk is set: MMIO, executable pages, anything outside the guest regions) calls the same routine the
// interpreter's FlatWriteN ends in for such a chunk, before rA is written, so an access violation it throws leaves rA as
// it was. A guest access is [r12 + (ea & kLeanMask)] (memory_access.h): loads are and, load, byte swap; stores are the
// policy test, the swapped bytes, then the dirty marks (LeanTryStore's 16-bit pair).
//
// Registers inside an op: eax holds the effective address, ecx its canonical form (ea & kLeanMask), edx the value (host
// order); esi and edi are temporaries. Slow paths are Cold code that calls a noexcept shim of this file (an exception
// is stored in frame.pending and the code exits through ExitPending) and rejoins the main path.
//
// Store groups (BuildPlan below): D-form integer stores off one base value in a block share one policy test and one
// set of dirty marks, made by the group's first store (its leader) for the bytes all of them write; the others (its
// members) are the bare byte-swapped store. What that relies on:
//  * Between two ops of a block's main path with no call between them, no dirty mark is cleared and the store policy
//    does not change. Marks are cleared by rollback's consume and restore, the interpreter's block refresh and the
//    verifiers, the policy is rebuilt when executable ranges change: all of them run in calls (the dispatcher, hooks,
//    bl/bctrl, the interpreter's handler of a Generic op), from a thread that is not inside a block, or with the guest
//    threads' writers quiescent. The slow paths of loads and stores this group calls (StoreFlagged and the like) only set
//    marks. So a mark the leader set is still set when a member writes, and a policy byte it found clear still is.
//  * A group is a run of stores of the same base value (a register's value at the block's start or at an untracked
//    write, plus the constants addi and the update forms add) with no call, Generic op or mtspr between them, and the
//    bytes they write lie in at most kGroupSpanMax bytes. The leader tests the range policy of the chunk of the range's
//    first byte (that chunk and the next: the range lies in them) and marks the range's blocks; a range that straddles a
//    1 GiB view starts in chunk 0x3FFF, which is flagged, so a clear test also proves every member's canonical address
//    lies in the range the leader computed.
//  * A leader whose test fails for another reason than the next chunk does a precise test; if its own chunk is flagged
//    or the range crosses into the flagged next chunk, it stores through the shim and then either finds the range plain
//    RAM (the checked path would end in a direct store and a mark there: GroupStoreFlagged marks the range and the block
//    goes on) or leaves the block after its own store, so no member writes without a test.
//  * The interpreter marks each store's pair of blocks; a group marks the blocks of the bytes the group writes, which
//    every store's own pair also covers when the whole group runs (a member past a taken exit leaves a mark of a block
//    not written: a mark only ever says a block may have changed).
#include "jit_x64.h"
#include "jit_x64_generic.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <utility>
#include <vector>

#include "brawl_dirty_memory.h"
#include "isa/ppc_isa_float.h"
#include "isa/ppc_isa_quantized.h"
#include "lean_window.h"
#include "memory_access.h"

#if defined(__linux__) && defined(__x86_64__)
#include <signal.h>
#include <ucontext.h>
#define MKW_JIT_FAULT_STORES 1
#else
#define MKW_JIT_FAULT_STORES 0
#endif

namespace PpcInterpreter::detail::x64 {

// --- fault-based stores: the sites, and the handler that sends a faulting store to its slow entry ----------------------------
// With the store view (lean_window.h, g_storeView), a guest store is one store instruction at base + kStoreViewOffset + ea,
// which faults exactly where the store policy would send the store to the checked path (memory.cpp keeps the view's protection
// in step with the policy), or where a store's bytes reach a chunk the policy flags or no region backs. A fault at a registered
// site (the instruction's address in a code chunk's table) resumes at the site's slow entry, in the op's cold code, with every
// register as the store found it; the entry tests the policy as the interpreter's store does and either stores through the
// window (LeanTryStore's fast path) or calls the checked path's routine. The handler also patches the store instruction into a
// jump to the slow entry (every store instruction here is at least 8 bytes: its displacement is a disp32), so a site faults at
// most once per compile. Any other fault goes to the handler that was installed before (the runtime's report, or the default).
//
// A chunk's table is written only by the thread whose code cache owns the chunk, while it compiles (never while it runs that
// chunk's code), and read only by the handler on a thread that faulted in that chunk's code: the same thread.

struct FaultChunk {
    const uint8_t* base;
    size_t size;
    std::vector<std::pair<uint32_t, uint32_t>> sites;  // (store instruction, slow entry) as offsets from base, by store
    uint32_t slot;
};

namespace {

constexpr uint32_t kFaultSlots = 4096;
struct FaultSlot {
    std::atomic<uintptr_t> base{0};  // 0: free, 1: being filled, else the chunk's base
    std::atomic<size_t> size{0};
    std::atomic<FaultChunk*> chunk{nullptr};
};
FaultSlot g_faultSlots[kFaultSlots];
bool g_faultStores = false;
std::atomic<uint64_t> g_faultSites{0}, g_faults{0}, g_faultRepeats{0}, g_faultChunks{0};

#if MKW_JIT_FAULT_STORES
uintptr_t g_storeLo = 0, g_storeSpan = 0;  // where a store through the store view can fault
struct sigaction g_previousSegv;

// The slow entry of the store instruction at rip, or 0 when rip is no registered site.
uintptr_t FindFaultSite(uintptr_t rip) {
    for (FaultSlot& s : g_faultSlots) {
        const uintptr_t base = s.base.load(std::memory_order_acquire);
        if (base <= 1 || rip - base >= s.size.load(std::memory_order_relaxed)) continue;
        const FaultChunk* const c = s.chunk.load(std::memory_order_relaxed);
        const uint32_t at = static_cast<uint32_t>(rip - base);
        const auto it = std::lower_bound(c->sites.begin(), c->sites.end(), std::make_pair(at, uint32_t{0}));
        return it != c->sites.end() && it->first == at ? base + it->second : 0;
    }
    return 0;
}

void OnStoreFault(int sig, siginfo_t* info, void* context) {
    ucontext_t* const uc = static_cast<ucontext_t*>(context);
    const uintptr_t at = reinterpret_cast<uintptr_t>(info != nullptr ? info->si_addr : nullptr);
    if (sig == SIGSEGV && info != nullptr && at - g_storeLo < g_storeSpan) {
        greg_t& rip = uc->uc_mcontext.gregs[REG_RIP];
        const uintptr_t site = static_cast<uintptr_t>(rip);
        if (const uintptr_t slow = FindFaultSite(site)) {
            uint8_t* const code = reinterpret_cast<uint8_t*>(site);
            if (code[0] == 0xE9u) g_faultRepeats.fetch_add(1, std::memory_order_relaxed);
            const int32_t rel = static_cast<int32_t>(static_cast<intptr_t>(slow) - static_cast<intptr_t>(site + 5));
            code[0] = 0xE9u;  // jmp rel32 over the store instruction's first bytes (it is longer; the rest is never reached)
            std::memcpy(code + 1, &rel, sizeof(rel));
            rip = static_cast<greg_t>(slow);
            g_faults.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    // not a store of compiled code: what was there before handles it
    if ((g_previousSegv.sa_flags & SA_SIGINFO) != 0) {
        if (g_previousSegv.sa_sigaction != nullptr) {
            g_previousSegv.sa_sigaction(sig, info, context);
            return;
        }
    } else if (g_previousSegv.sa_handler != SIG_DFL && g_previousSegv.sa_handler != SIG_IGN) {
        g_previousSegv.sa_handler(sig);
        return;
    }
    signal(sig, SIG_DFL);  // the instruction runs again and the default action ends the process
}
#endif

}  // namespace

bool FaultStoresInit() {
    static const bool ok = [] {
        if (!LeanWindow::g_storeView) return true;
#if MKW_JIT_FAULT_STORES
        struct sigaction action {};
        action.sa_sigaction = OnStoreFault;
        action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&action.sa_mask);
        // base + 1 GiB + ea + d: from d = -32 KiB under the cached copy to the tail and the guard after it
        g_storeLo = reinterpret_cast<uintptr_t>(MemoryInline::g_leanBase) + LeanWindow::kStoreViewOffset - 0x8000u;
        g_storeSpan = (uintptr_t{1} << 32) + 0x8000u + 2 * LeanWindow::kAliasTail;
        if (sigaction(SIGSEGV, &action, &g_previousSegv) != 0) return false;
        g_faultStores = true;
        return true;
#else
        return false;
#endif
    }();
    return ok;
}

bool FaultStores() { return FaultStoresInit() && g_faultStores; }

FaultChunk* FaultChunkOpen(const uint8_t* base, size_t size) {
    if (!FaultStores()) return nullptr;
    for (uint32_t i = 0; i < kFaultSlots; ++i) {
        uintptr_t expected = 0;
        if (!g_faultSlots[i].base.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) continue;
        FaultChunk* const c = new (std::nothrow) FaultChunk{base, size, {}, i};
        if (c == nullptr) {
            g_faultSlots[i].base.store(0, std::memory_order_release);
            return nullptr;
        }
        g_faultSlots[i].size.store(size, std::memory_order_relaxed);
        g_faultSlots[i].chunk.store(c, std::memory_order_relaxed);
        g_faultSlots[i].base.store(reinterpret_cast<uintptr_t>(base), std::memory_order_release);
        g_faultChunks.fetch_add(1, std::memory_order_relaxed);
        return c;
    }
    return nullptr;
}

void FaultChunkClose(FaultChunk* chunk) {
    if (chunk == nullptr) return;
    FaultSlot& s = g_faultSlots[chunk->slot];
    s.base.store(1, std::memory_order_release);
    s.chunk.store(nullptr, std::memory_order_relaxed);
    s.size.store(0, std::memory_order_relaxed);
    s.base.store(0, std::memory_order_release);
    g_faultChunks.fetch_sub(1, std::memory_order_relaxed);
    delete chunk;
}

void FaultChunkAdd(FaultChunk* chunk, const uint8_t* code, const BlockEmitter& e) {
    const size_t at = static_cast<size_t>(code - chunk->base);
    for (uint32_t i = 0; i < e.FaultSiteCount(); ++i) {
        const auto site = std::make_pair(static_cast<uint32_t>(at + e.FaultSiteStore(i)), static_cast<uint32_t>(at + e.FaultSiteSlow(i)));
        chunk->sites.insert(std::upper_bound(chunk->sites.begin(), chunk->sites.end(), site), site);
    }
    g_faultSites.fetch_add(e.FaultSiteCount(), std::memory_order_relaxed);
}

namespace {

using namespace Xbyak::util;
using CG = Xbyak::CodeGenerator;

#if !MKW_LEAN_MEMORY
#error "the x86-64 JIT is built for lean memory only"
#endif
static_assert(MemoryInline::kLeanPolicyShift == 16, "the policy byte index below shifts by 16");

constexpr int kPolicyDisp = static_cast<int>(MemoryInline::kLeanPolicyOffset);
constexpr int kRangePolicyDisp = static_cast<int>(MemoryInline::kLeanRangePolicyOffset);
constexpr int kDirtyDisp = static_cast<int>(MemoryInline::kLeanDirtyOffset);
constexpr uint32_t kMask = MemoryInline::kLeanMask;
static_assert(kMask == 0x3FFFFFFFu, "the code below assumes the 1 GiB lean window");
// A store group's bytes: at most three blocks from any start (the pair of marks from the first byte's block covers 65
// bytes from any start, the mark of the last byte's block the third), far less than a policy chunk.
constexpr int kGroupSpanMax = 129;
constexpr int kGroupOnePair = 65;
enum : uint8_t { kRoleSingle = 0, kRoleLeader = 1, kRoleMember = 2 };

// --- the shims the slow paths call -----------------------------------------------------------------------------------------
// None throws: an exception is kept in the frame and the shim answers 1, so the code calls ExitPending.

uint32_t Keep(Frame* f) noexcept {
    f->pending = std::current_exception();
    return 1;
}

// What FlatWrite8/16/32 and the 64-bit float store end in when the chunk's policy byte is set (LeanStoreBody).
template <typename T>
uint32_t StoreFlagged(Frame* f, uint32_t ea, uint64_t value) noexcept {
    try {
        MemoryInline::WriteResolvedFallback<T>(ea, static_cast<T>(value));
        JitNoteEpoch();  // the checked path may have run a handler: the block goes on to its link slots
        return 0;
    } catch (...) {
        return Keep(f);
    }
}

struct StoreGroupCounters {
    std::atomic<uint64_t> leaders{0}, members{0}, slowGoOn{0}, slowLeave{0};
    uint64_t rangeBack = 0;  // incremented by compiled code (not atomically: a count for the tests and stats)
};
StoreGroupCounters g_groups;

// A store group's bytes [start, start + span) (span <= kGroupSpanMax), all in the cached view of MEM1 or MEM2 and in 4 KiB
// pages without guarded code: there the checked path of each store ends in exactly the lean store, the dirty mark and
// the bytes (WriteResolvedFallback's own shortcut; Memory::Write* finds no guard, the bytes are the window's), so the
// group's stores may write directly once the range is marked. It must refuse whatever a store's checked path has to see:
// a new reason to flag a chunk in the store policy (RefreshLeanStorePolicy) that the checked path acts on needs its test
// here too.
bool GroupRangePlain(uint32_t start, uint32_t span) {
    const uint32_t last = start + span - 1u;
    const uint32_t top = start >> 28;
    if (last < start || (last >> 28) != top) return false;
    const uint64_t size = top == 0x8u ? Memory::kMem1BankSize : top == 0x9u ? Memory::kMem2Size : 0;
    if ((last & 0x0FFFFFFFu) >= size) return false;
    constexpr uint32_t kShift = RecompMod::kExecutableWriteGuardPageShift;  // the range is at most two of these pages
    return RecompMod::g_executableWriteGuardPages[start >> kShift].load(std::memory_order_relaxed) == 0 &&
           RecompMod::g_executableWriteGuardPages[last >> kShift].load(std::memory_order_relaxed) == 0;
}

// The slow path of a group's leader (its range's policy said the checked path for a byte of the range): its own store as
// the interpreter makes it (FlatStore: the policy of its own chunk decides), then 0 when the group's other stores may
// write their range directly (GroupRangePlain; the range is then marked), 2 when they may not (the block leaves before
// them), 1 on an exception (kept in the frame). packed = the leader's offset in the range | span << 8 | members << 16
// (members = 0: no member relies on the leader, which then only stores).
template <typename T>
uint32_t GroupStoreFlagged(Frame* f, uint32_t ea, uint64_t value, uint32_t packed) noexcept {
    try {
        MemoryInline::FlatStore<T>(ea, static_cast<T>(value));
        JitNoteEpoch();  // as StoreFlagged: the checked path may have run a handler, and the block may go on to its link slots
    } catch (...) {
        return Keep(f);
    }
    if ((packed >> 16) == 0) return 0;
    const uint32_t start = ea - (packed & 0xFFu), span = (packed >> 8) & 0xFFu;
    if (!GroupRangePlain(start, span)) {
        g_groups.slowLeave.fetch_add(1, std::memory_order_relaxed);
        return 2;
    }
    BrawlDirtyMemory::Mark(start, span);
    g_groups.slowGoOn.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

// Stmw: one FlatWrite32 per register, as the interpreter does (a range of words that touches a flagged chunk).
uint32_t StmwFlagged(Frame* f, uint32_t ea, uint32_t first) noexcept {
    try {
        for (uint32_t k = first; k < 32; ++k, ea += 4) MemoryInline::FlatWrite32(ea, f->ctx->gpr[k]);
        JitNoteEpoch();
        return 0;
    } catch (...) {
        return Keep(f);
    }
}

// lfs of a word that is not a normal single or a zero: the manual's conversion (PpcConvertToDoubleBitsInline).
void LfsOther(Frame* f, uint32_t raw, uint32_t d) noexcept {
    SetScalar(*f, d, PpcBitCastToDoubleInline(PpcConvertToDoubleBitsInline(raw)));
}

// ConvertPpcDoubleToSingleBits for the exponents 874..896.
uint32_t SubnormalSingle(uint64_t bits) noexcept {
    return MemoryInline::ConvertPpcDoubleToSubnormalSingleBits(bits, static_cast<uint32_t>((bits >> 52) & 0x7FFu));
}

// psq_l / psq_st with the quantization the GQR selects at run time. packed = d | (W << 3 | I) << 8.
uint32_t PsqLoad(Frame* f, uint32_t ea, uint32_t packed) noexcept {
    try {
        const uint32_t wi = packed >> 8;
        SetPaired(*f, packed & 0xFFu, PPC_PsqL(ea, wi >> 3, wi & 7u));
        JitNoteEpoch();
        return 0;
    } catch (...) {
        return Keep(f);
    }
}
uint32_t PsqStore(Frame* f, uint32_t ea, uint32_t packed) noexcept {
    try {
        const uint32_t wi = packed >> 8;
        PPC_PsqSt(ea, Paired(*f, packed & 0xFFu), wi >> 3, wi & 7u);
        JitNoteEpoch();
        return 0;
    } catch (...) {
        return Keep(f);
    }
}
// psq_l (W = 0, float GQR) of a pair with a signalling NaN lane: the lanes quieted as the loader does; bits as the
// swapped load leaves them (ps0 in the high word).
void PsqPairQuieted(Frame* f, uint64_t bits, uint32_t d) noexcept {
    SetPaired(*f, d, PpcBitCastToDoubleInline(PpcLoadPairPsqFloatBitsPackedInline(bits)));
}

// --- decoding the kinds ---------------------------------------------------------------------------------------------------

struct Shape {
    uint8_t width;  // bytes
    bool update, indexed, sign;
    bool zeroBase = false;  // rA = 0 in the word: the address is rB, or the displacement, alone (Exec's EaX and Ea)
    bool reverse = false;   // lwbrx, lhbrx, stwbrx, sthbrx: the bytes as they lie in memory, not swapped
};

bool IntLoadShape(K k, Shape& s) {
    switch (k) {
    case K::Lwz: s = {4, false, false, false}; return true;
    case K::Lwzu: s = {4, true, false, false}; return true;
    case K::Lbz: s = {1, false, false, false}; return true;
    case K::Lbzu: s = {1, true, false, false}; return true;
    case K::Lhz: s = {2, false, false, false}; return true;
    case K::Lhzu: s = {2, true, false, false}; return true;
    case K::Lha: s = {2, false, false, true}; return true;
    case K::Lwzx: s = {4, false, true, false}; return true;
    case K::Lbzx: s = {1, false, true, false}; return true;
    case K::Lhzx: s = {2, false, true, false}; return true;
    default: return false;
    }
}

bool IntStoreShape(K k, Shape& s) {
    switch (k) {
    case K::Stw: s = {4, false, false, false}; return true;
    case K::Stwu: s = {4, true, false, false}; return true;
    case K::Stb: s = {1, false, false, false}; return true;
    case K::Stbu: s = {1, true, false, false}; return true;
    case K::Sth: s = {2, false, false, false}; return true;
    case K::Sthu: s = {2, true, false, false}; return true;
    case K::Stwx: s = {4, false, true, false}; return true;
    case K::Stbx: s = {1, false, true, false}; return true;
    case K::Sthx: s = {2, false, true, false}; return true;
    default: return false;
    }
}

// --- emission helpers -----------------------------------------------------------------------------------------------------

// eax = the effective address: r[a] + imm, or r[a] + r[b] (32-bit wrap).
void EmitEa(BlockEmitter& e, const Op& o, bool indexed, bool zeroBase = false) {
    Xbyak::CodeGenerator& a = e.a;
    if (zeroBase) {
        if (indexed) {
            if (e.WantCacheGpr(o.b)) a.mov(eax, e.ReadGpr(o.b));
            else a.mov(eax, e.Gpr(o.b));
        } else {
            a.mov(eax, o.imm);
        }
        return;
    }
    if (e.WantCacheGpr(o.a)) {
        const Xbyak::Reg32& ra = e.ReadGpr(o.a);
        if (indexed) {
            if (e.WantCacheGpr(o.b)) {
                const Xbyak::Reg32& rb = e.ReadGpr(o.b, 1u << o.a);
                a.lea(eax, a.ptr[ra.cvt64() + rb.cvt64()]);
            } else {
                a.mov(eax, ra);
                a.add(eax, e.Gpr(o.b));
            }
        } else if (o.imm != 0) {
            a.lea(eax, a.ptr[ra.cvt64() + static_cast<int32_t>(o.imm)]);
        } else {
            a.mov(eax, ra);
        }
    } else {
        a.mov(eax, e.Gpr(o.a));
        if (indexed) {
            if (e.WantCacheGpr(o.b)) a.add(eax, e.ReadGpr(o.b));
            else a.add(eax, e.Gpr(o.b));
        } else if (o.imm != 0) {
            a.add(eax, o.imm);
        }
    }
}

// ecx = the canonical offset of the address in eax.
void EmitCanonical(BlockEmitter& e) {
    e.a.mov(ecx, eax);
    e.a.and_(ecx, kMask);
}

// ecx = the canonical offset of the effective address, for an op that does not need the address itself afterwards (not an
// update form, and no slow path that passes it on): one register less to copy.
void EmitCanonicalEa(BlockEmitter& e, const Op& o, bool indexed, bool zeroBase = false) {
    Xbyak::CodeGenerator& a = e.a;
    if (zeroBase) {
        if (indexed) {
            if (e.WantCacheGpr(o.b)) a.mov(ecx, e.ReadGpr(o.b));
            else a.mov(ecx, e.Gpr(o.b));
        } else {
            a.mov(ecx, o.imm);
        }
        a.and_(ecx, kMask);
        return;
    }
    if (e.WantCacheGpr(o.a)) {
        const Xbyak::Reg32& ra = e.ReadGpr(o.a);
        if (indexed) {
            if (e.WantCacheGpr(o.b)) {
                const Xbyak::Reg32& rb = e.ReadGpr(o.b, 1u << o.a);
                a.lea(ecx, a.ptr[ra.cvt64() + rb.cvt64()]);
            } else {
                a.mov(ecx, ra);
                a.add(ecx, e.Gpr(o.b));
            }
        } else if (o.imm != 0) {
            a.lea(ecx, a.ptr[ra.cvt64() + static_cast<int32_t>(o.imm)]);
        } else {
            a.mov(ecx, ra);
        }
    } else {
        a.mov(ecx, e.Gpr(o.a));
        if (indexed) {
            if (e.WantCacheGpr(o.b)) a.add(ecx, e.ReadGpr(o.b));
            else a.add(ecx, e.Gpr(o.b));
        } else if (o.imm != 0) {
            a.add(ecx, o.imm);
        }
    }
    a.and_(ecx, kMask);
}

// An update form's write of rA (the address is in eax; a slow path that clobbered it rebuilds it first from memory).
void EmitRebuildEa(Xbyak::CodeGenerator& a, BlockEmitter& e, uint32_t ra, uint32_t imm, bool indexed = false, uint32_t rb = 0) {
    a.mov(eax, e.Gpr(ra));
    if (indexed) a.add(eax, e.Gpr(rb));
    else if (imm != 0) a.add(eax, imm);
}

// The byte-swapped load of `width` bytes at `at` into `dst` (rdx for 8), zero-extended.
void EmitLoadSwappedAt(Xbyak::CodeGenerator& a, unsigned width, const Xbyak::Reg32& dst, const Xbyak::RegExp& at,
                       bool reverse = false) {
    switch (width) {
    case 1: a.movzx(dst, a.byte[at]); break;
    case 2:
        a.movzx(dst, a.word[at]);
        if (!reverse) a.rol(dst.cvt16(), 8);
        break;
    case 4:
        if (reverse) {  // lwbrx: the bytes as memory holds them
            a.mov(dst, a.dword[at]);
            break;
        }
#if defined(__MOVBE__)
        a.movbe(dst, a.dword[at]);
#else
        a.mov(dst, a.dword[at]);
        a.bswap(dst);
#endif
        break;
    default:
#if defined(__MOVBE__)
        a.movbe(dst.cvt64(), a.qword[at]);
#else
        a.mov(dst.cvt64(), a.qword[at]);
        a.bswap(dst.cvt64());
#endif
        break;
    }
}

// The same at [r12 + rcx] (rcx: the canonical address).
void EmitLoadSwappedTo(Xbyak::CodeGenerator& a, unsigned width, const Xbyak::Reg32& dst, bool reverse = false) {
    EmitLoadSwappedAt(a, width, dst, kLean + rcx, reverse);
}

void EmitLoadSwapped(Xbyak::CodeGenerator& a, unsigned width, bool reverse = false) {
    EmitLoadSwappedTo(a, width, edx, reverse);
}

// The host address of a load's or a store's guest bytes without a mask, in the aliased view (lean_window.h,
// g_aliased): [r12 + base + disp] with the base register's 32-bit value zero-extended (a cache slot's 64-bit register
// holds it so) and |disp| < 32 KiB, [r12 + rax] for an address computed in eax (32-bit: indexed forms, and update forms,
// which write eax to rA), [r12 + canonical] for an absolute one. Sets eax for update forms; uses eax for a base read from
// memory. ReadGpr the base before the operand registers that must stay pinned. With the store view the address is one GiB
// up (every copy there reads the same bytes): loads then use the copies the stores use, so that a page the JIT reads and
// writes has one TLB entry for both, as the window's for the code that masks.
Xbyak::RegExp EmitAliasedAddress(BlockEmitter& e, const Op& o, const Shape& s) {
    Xbyak::CodeGenerator& a = e.a;
    const int view = LeanWindow::g_storeView ? static_cast<int>(LeanWindow::kStoreViewOffset) : 0;
    const int32_t disp = static_cast<int32_t>(o.imm);
    if (s.zeroBase) {
        if (!s.indexed) return kLean + static_cast<int>(o.imm & kMask);
        if (e.WantCacheGpr(o.b)) return kLean + e.ReadGpr(o.b).cvt64() + view;
        a.mov(eax, e.Gpr(o.b));
        return kLean + rax + view;
    }
    const Xbyak::Reg32* base = e.WantCacheGpr(o.a) ? &e.ReadGpr(o.a) : nullptr;
    if (s.indexed || s.update) {  // the address computed in 32 bits (a sum of two registers can pass 2^32)
        if (s.indexed) {
            const Xbyak::Reg32* index = e.WantCacheGpr(o.b) ? &e.ReadGpr(o.b, 1u << o.a) : nullptr;
            if (base != nullptr && index != nullptr) {
                a.lea(eax, a.ptr[base->cvt64() + index->cvt64()]);
            } else {
                if (base != nullptr) a.mov(eax, *base);
                else a.mov(eax, e.Gpr(o.a));
                if (index != nullptr) a.add(eax, *index);
                else a.add(eax, e.Gpr(o.b));
            }
        } else if (base != nullptr) {
            a.lea(eax, a.ptr[base->cvt64() + disp]);
        } else {
            a.mov(eax, e.Gpr(o.a));
            if (disp != 0) a.add(eax, disp);
        }
        return kLean + rax + view;
    }
    if (base != nullptr) return kLean + base->cvt64() + (disp + view);
    a.mov(eax, e.Gpr(o.a));
    return kLean + rax + (disp + view);
}

// --- forward jumps to cold code ----------------------------------------------------------------------------------------
// The emitter's label pool (BlockEmitter::kMaxLabels, shared by the groups) is small next to the number of slow paths a
// block of loads and stores has, so this file does not spend labels: a forward jump to cold code is written with a
// zero displacement and remembers where, the cold code patches it when it is emitted (it is emitted into the same,
// writable buffer), and a jump back from cold code goes to an offset noted in the main path. Offsets are from the
// start of the block's code.

struct Patch {
    uint32_t at;  // offset of the 32-bit displacement
};

enum Cc : uint8_t { kB = 0x2, kE = 0x4, kNe = 0x5, kBe = 0x6, kA = 0x7 };

Patch EmitJcc(Xbyak::CodeGenerator& a, Cc cc) {
    a.db(0x0F);
    a.db(0x80 | cc);
    const Patch p{static_cast<uint32_t>(a.getSize())};
    a.dd(0);
    return p;
}

// The jump written at p goes to here.
void Resolve(Xbyak::CodeGenerator& a, Patch p) {
    uint8_t* const base = const_cast<uint8_t*>(a.getCode());
    const int32_t rel = static_cast<int32_t>(a.getSize() - (p.at + 4));
    std::memcpy(base + p.at, &rel, sizeof(rel));
}

uint32_t Here(Xbyak::CodeGenerator& a) { return static_cast<uint32_t>(a.getSize()); }
void JmpTo(Xbyak::CodeGenerator& a, uint32_t offset) { a.jmp(static_cast<const void*>(a.getCode() + offset), CG::T_NEAR); }

// Up to a few jumps to one cold entry.
struct Patches {
    static constexpr unsigned kMax = 8;
    uint32_t at[kMax];
    uint8_t n;
    void Add(Patch p) { at[n++] = p.at; }
    void Resolve(Xbyak::CodeGenerator& a) const {
        for (unsigned i = 0; i < n; ++i) x64::Resolve(a, Patch{at[i]});
    }
};

// After a shim that answers 1 on an exception: leave the block with it pending.
void EmitPendingCheck(BlockEmitter& e) {
    Xbyak::Label ok;
    e.a.test(eax, eax);
    e.a.jz(ok);
    e.ExitPending();
    e.a.L(ok);
}

using StoreShim = uint32_t (*)(Frame*, uint32_t, uint64_t);
StoreShim ShimFor(unsigned width) {
    switch (width) {
    case 1: return &StoreFlagged<uint8_t>;
    case 2: return &StoreFlagged<uint16_t>;
    case 4: return &StoreFlagged<uint32_t>;
    default: return &StoreFlagged<uint64_t>;
    }
}

// The slow path of a store whose address is in eax and value in rdx: the flagged chunk's routine, then back to the
// join point (an update form rebuilds the address first: the call clobbered it).
struct StoreCold {
    Patch slow;
    uint32_t join, pcIndex, imm;
    uint8_t ra, rb, width;
    bool update, indexed, reverse;
    BlockEmitter::GprSnapshot snap;
    bool fault = false;  // the slow entry of fault site `site` (the address in eax, the value in edx)
    uint8_t site = 0;
    void operator()(BlockEmitter& e) const;
};

// The store of `width` bytes of v (host order) at `at`, byte-swapped unless reverse (stwbrx, sthbrx: the register's bytes
// as they are). v keeps its value unless it is edx / rdx and the build has no movbe.
void EmitStoreValue(Xbyak::CodeGenerator& a, unsigned width, const Xbyak::RegExp& at, const Xbyak::Reg32& v,
                    bool reverse = false) {
    switch (width) {
    case 1: a.mov(a.byte[at], v.cvt8()); break;
    case 2:
        if (reverse) {
            a.mov(a.word[at], v.cvt16());
            break;
        }
#if defined(__MOVBE__)
        a.movbe(a.word[at], v.cvt16());
#else
        if (v.getIdx() != edx.getIdx()) a.mov(edx, v);
        a.rol(dx, 8);
        a.mov(a.word[at], dx);
#endif
        break;
    case 4:
        if (reverse) {
            a.mov(a.dword[at], v);
            break;
        }
#if defined(__MOVBE__)
        a.movbe(a.dword[at], v);
#else
        if (v.getIdx() != edx.getIdx()) a.mov(edx, v);
        a.bswap(edx);
        a.mov(a.dword[at], edx);
#endif
        break;
    default:
#if defined(__MOVBE__)
        a.movbe(a.qword[at], v.cvt64());
#else
        if (v.getIdx() != edx.getIdx()) a.mov(rdx, v.cvt64());
        a.bswap(rdx);
        a.mov(a.qword[at], rdx);
#endif
        break;
    }
}

// The dirty marks of the store whose canonical address is in ecx (clobbered): the 16-bit pair from its block.
void EmitMarkPair(Xbyak::CodeGenerator& a) {
#if MKW_LEAN_DIRTY_MARKS
    a.shr(ecx, 6);
    a.mov(a.word[kLean + rcx + kDirtyDisp], 0x0101);
#else
    (void)a;
#endif
}

// The policy test, the swapped store of the value in edx / rdx and the dirty marks, for the address in eax, ecx = its
// canonical form. A set policy byte jumps to the returned patch (the cold path) before anything is written, with eax and
// edx / rdx as they came. Clobbers ecx and esi.
Patch EmitStoreChecked(BlockEmitter& e, unsigned width, bool reverse = false) {
    Xbyak::CodeGenerator& a = e.a;
    a.mov(esi, ecx);
    a.shr(esi, MemoryInline::kLeanPolicyShift);
    a.cmp(a.byte[kLean + rsi + kPolicyDisp], 0);
    const Patch slow = EmitJcc(a, kNe);
    EmitStoreValue(a, width, kLean + rcx, edx, reverse);
    EmitMarkPair(a);  // after the bytes: no call between them, nothing reads the marks in between
    return slow;
}

// --- fault sites: stores through the store view (the handler is at the top of this file) -----------------------------------
// A site is one store instruction at [r12 + reg + disp + kStoreView] (lean_window.h); a store that faults there resumes at the
// site's slow entry in the op's cold code, which begins with EmitSiteFast: the policy test of the store's first byte (what
// LeanTryStore tests), then the store through the window and back to the main path right after the site, or on to the op's
// checked path. A site's marks come after it on the main path: only a store that went through (to a chunk the policy leaves
// clear) or came back from the fast part reaches them.

#if defined(__MOVBE__)
constexpr bool kMovbe = true;
#else
constexpr bool kMovbe = false;
#endif
constexpr int kStoreView = static_cast<int>(LeanWindow::kStoreViewOffset);

// The guest address of a site's store: (the 32-bit value in host register `reg` + disp) mod 2^32.
struct SiteAddr {
    uint8_t reg;
    int32_t disp;
};

void EmitRawStore(CG& a, unsigned width, const Xbyak::RegExp& at, const Xbyak::Reg32& v) {
    switch (width) {
    case 1: a.mov(a.byte[at], v.cvt8()); break;
    case 2: a.mov(a.word[at], v.cvt16()); break;
    case 4: a.mov(a.dword[at], v); break;
    default: a.mov(a.qword[at], v.cvt64()); break;
    }
}

// Byte order the other way in edx / rdx (the value a site without movbe swapped before storing, back to host order).
void EmitUnswap(CG& a, unsigned width) {
    if (width == 2) a.rol(dx, 8);
    else if (width == 4) a.bswap(edx);
    else if (width == 8) a.bswap(rdx);
}

// Whether a site of this width stores edx swapped by the instructions before it (a build without movbe).
bool SiteSwapsFirst(unsigned width, bool reverse) { return !kMovbe && width > 1 && !reverse; }

// A site: the store of `width` bytes of v (host order), byte-swapped unless reverse. Answers the site.
uint32_t EmitSiteStore(BlockEmitter& e, unsigned width, SiteAddr at, const Xbyak::Reg32& v, bool reverse = false) {
    CG& a = e.a;
    const Xbyak::RegExp to = kLean + Xbyak::Reg64(at.reg) + (at.disp + kStoreView);
    if (SiteSwapsFirst(width, reverse)) {  // swapped into edx first: the site is the plain store after it
        if (v.getIdx() != edx.getIdx()) {
            if (width == 8) a.mov(rdx, v.cvt64());
            else a.mov(edx, v);
        }
        EmitUnswap(a, width);
        const uint32_t site = e.AddFaultSite();
        EmitRawStore(a, width, to, edx);
        return site;
    }
    const uint32_t site = e.AddFaultSite();
    EmitStoreValue(a, width, to, v, reverse);  // one instruction (mov or movbe)
    return site;
}

// What the first part of a site's slow entry marks after its store through the window: nothing (a group member: its leader
// marked the group's bytes), the pair of the store's own first byte, or a leader's range (from `offset` bytes below its first
// byte, `span` bytes).
struct SiteMarks {
    uint8_t offset = 0, span = 0;
    bool none = false;
};
constexpr SiteMarks kMarkNone{0, 0, true};
constexpr SiteMarks kMarkOwn{0, 0, false};

// The first part of a site's slow entry (see above): the policy test of the store's first byte (with `both`, of its last byte
// too: psq_st's pair tests both ends, as the interpreter's quantized store does), then the store through the window at the
// canonical address and its marks (`marks`), on to `join` (after the main path's marks, which are made from the unmasked
// address of a store that went through the store view: EmitMarkUnmasked). v: the register the site stored. Answers the jumps to
// the checked path. Clobbers esi and edi (no op holds a value in them at a site).
Patches EmitSiteFast(BlockEmitter& e, uint32_t site, SiteAddr at, unsigned width, const Xbyak::Reg32& v, bool reverse,
                     uint32_t join, bool both, SiteMarks marks) {
    CG& a = e.a;
    e.FaultSiteSlowHere(site);
    Patches checked{};
    a.lea(esi, a.ptr[Xbyak::Reg64(at.reg) + at.disp]);
    a.and_(esi, kMask);
    a.mov(edi, esi);
    a.shr(edi, MemoryInline::kLeanPolicyShift);
    a.cmp(a.byte[kLean + rdi + kPolicyDisp], 0);
    checked.Add(EmitJcc(a, kNe));
    if (both) {
        a.lea(edi, a.ptr[rsi + static_cast<int>(width - 1)]);
        a.and_(edi, kMask);
        a.shr(edi, MemoryInline::kLeanPolicyShift);
        a.cmp(a.byte[kLean + rdi + kPolicyDisp], 0);
        checked.Add(EmitJcc(a, kNe));
    }
    if (SiteSwapsFirst(width, reverse)) EmitRawStore(a, width, kLean + rsi, edx);
    else EmitStoreValue(a, width, kLean + rsi, v, reverse);
#if MKW_LEAN_DIRTY_MARKS
    if (!marks.none) {  // as the main path's marks, from the canonical address (the store need not be in the cached view)
        if (marks.offset != 0) {
            a.sub(esi, marks.offset);
            a.and_(esi, kMask);
        }
        if (marks.span > kGroupOnePair) {
            a.lea(edi, a.ptr[rsi + (marks.span - 1)]);
            a.shr(edi, 6);
            a.mov(a.byte[kLean + rdi + kDirtyDisp], 1);
        }
        a.shr(esi, 6);
        a.mov(a.word[kLean + rsi + kDirtyDisp], 0x0101);
    }
#else
    (void)marks;
#endif
    JmpTo(a, join);
    return checked;
}

// The marks of a store that went through the store view, from its unmasked guest address in r (clobbered): such a store is in
// the cached view (lean_window.h: only that copy is writable), whose address >> 6 is the canonical one's plus 0x02000000, so
// the mark is at a fixed distance below the canonical one with no mask. A group's range may start up to kGroupSpanMax bytes
// below 0x80000000: those marks land in the 64 KiB below the dirty map, which memory.cpp backs with the store view.
constexpr int kMarkUnmaskedDisp = kDirtyDisp - static_cast<int>(0x80000000u >> 6);
void EmitMarkUnmasked(CG& a, const Xbyak::Reg32& r) {
#if MKW_LEAN_DIRTY_MARKS
    a.shr(r, 6);
    a.mov(a.word[kLean + r.cvt64() + kMarkUnmaskedDisp], 0x0101);
#else
    (void)a;
    (void)r;
#endif
}

// The marks of a site's store whose guest address is in eax (after the site).
void EmitMarksOfEax(CG& a) {
#if MKW_LEAN_DIRTY_MARKS
    a.mov(ecx, eax);
    EmitMarkUnmasked(a, ecx);
#else
    (void)a;
#endif
}

void StoreCold::operator()(BlockEmitter& e) const {
    Xbyak::CodeGenerator& a = e.a;
    if (fault) {
        EmitSiteFast(e, site, SiteAddr{static_cast<uint8_t>(rax.getIdx()), 0}, width, edx, reverse, join, false, kMarkOwn).Resolve(a);
        if (SiteSwapsFirst(width, reverse)) EmitUnswap(a, width);
    } else {
        Resolve(a, slow);
    }
    e.EmitFlushSnapshot(snap);
    e.StorePc(pcIndex);
    if (reverse) {  // the shim writes big-endian: hand it the value swapped, so memory gets the register's own byte order
        if (width == 4) a.bswap(edx);
        else if (width == 2) a.rol(dx, 8);
    }
    e.CallHelper(ShimFor(width), kFrame, eax, rdx);
    EmitPendingCheck(e);
    e.EmitReloadSnapshotVolatile(snap);
    if (update) EmitRebuildEa(a, e, ra, imm, indexed, rb);
    JmpTo(a, join);
}

// --- store groups: which stores share a policy test and dirty marks (the rules are at the top of this file) ---------------

bool GroupedStore(K k, uint8_t& width, bool& update) {
    switch (k) {
    case K::Stw: width = 4; update = false; return true;
    case K::Stwu: width = 4; update = true; return true;
    case K::Stb: width = 1; update = false; return true;
    case K::Stbu: width = 1; update = true; return true;
    case K::Sth: width = 2; update = false; return true;
    case K::Sthu: width = 2; update = true; return true;
    default: return false;
    }
}

void BuildPlan(BlockEmitter& e) {
    BlockEmitter::MemPlan& p = e.memPlan;
    p.built = true;
    // Each register's value as a base: a root (its value at the block's start or at a write the analysis does not follow)
    // plus a constant. mmio: a lis of the MMIO window's high half, whose stores take the checked path (no group).
    struct Value {
        uint32_t root;
        int64_t off;
        bool mmio;
    };
    Value val[32];
    uint32_t roots = 0;
    for (uint32_t r = 0; r < 32; ++r) val[r] = Value{roots++, 0, false};
    const auto define = [&](uint32_t r) { val[r] = Value{roots++, 0, false}; };
    struct Group {
        bool live;
        uint32_t root;
        int64_t lo, hi, leaderOff;  // the bytes [lo, hi) and the leader's base, from the root
        uint8_t leader, members;
    };
    constexpr uint32_t kGroups = 8;
    Group groups[kGroups] = {};
    const auto close = [&](Group& g) {
        if (!g.live) return;
        g.live = false;
        if (g.members == 0) return;  // a leader alone tests and marks for itself
        p.role[g.leader] = kRoleLeader;
        p.rangeDisp[g.leader] = static_cast<int32_t>(g.lo - g.leaderOff);
        p.span[g.leader] = static_cast<uint8_t>(g.hi - g.lo);
    };
    const auto reset = [&] {
        for (Group& g : groups) close(g);
        for (uint32_t r = 0; r < 32; ++r) define(r);
    };
    for (uint32_t i = 0; i < e.block.count; ++i) {
        const Op& o = e.block.ops[i];
        const int64_t imm = static_cast<int32_t>(o.imm);
        uint8_t width = 0;
        bool update = false;
        if (GroupedStore(o.kind, width, update)) {
            const Value v = val[o.a];
            p.mmio[i] = v.mmio;
            const int64_t lo = v.off + imm, hi = lo + width;
            Group* g = nullptr;
            for (Group& c : groups)
                if (c.live && c.root == v.root) g = &c;
            if (g != nullptr && !v.mmio && std::max(hi, g->hi) - std::min(lo, g->lo) <= kGroupSpanMax) {
                g->lo = std::min(lo, g->lo);
                g->hi = std::max(hi, g->hi);
                ++g->members;
                p.role[i] = kRoleMember;
                p.leader[i] = g->leader;
            } else if (!v.mmio) {
                if (g == nullptr) {  // a free slot, else the oldest group
                    g = &groups[0];
                    for (Group& c : groups) {
                        if (!c.live) { g = &c; break; }
                        if (c.leader < g->leader) g = &c;
                    }
                }
                close(*g);
                *g = Group{true, v.root, lo, hi, v.off, static_cast<uint8_t>(i), 0};
            }
            if (update) val[o.a].off += imm;
            continue;
        }
        switch (o.kind) {
        case K::Addi: val[o.d] = Value{val[o.a].root, val[o.a].off + imm, val[o.a].mmio}; break;
        case K::Or:  // mr rA, rS
            if (o.d == o.b) val[o.a] = val[o.d];
            else define(o.a);
            break;
        case K::Lis:
            define(o.d);
            val[o.d].mmio = (o.imm & 0xFE000000u) == 0xCC000000u;
            break;
        case K::Lwzu: case K::Lbzu: case K::Lhzu: {  // rD, then rA = the address (the value of rA when rD = rA)
            const Value next{val[o.a].root, val[o.a].off + imm, val[o.a].mmio};
            define(o.d);
            val[o.a] = next;
            break;
        }
        case K::Li: case K::Addis: case K::Mulli: case K::Subfic: case K::Neg: case K::Add: case K::Subf: case K::Mullw:
        case K::Mflr: case K::Mfctr: case K::Mfcr: case K::Mfspr:
        case K::Lwz: case K::Lbz: case K::Lhz: case K::Lha: case K::Lwzx: case K::Lbzx: case K::Lhzx:
            define(o.d);
            break;
        case K::Ori: case K::Oris: case K::Xori: case K::Xoris: case K::AndiRc: case K::AndisRc: case K::Rlwinm:
        case K::RlwinmRc: case K::RlwinmRcBc: case K::Rlwimi: case K::OrRc: case K::And: case K::AndRc: case K::Xor:
        case K::Andc: case K::Nor: case K::Slw: case K::Srw: case K::Srawi: case K::Extsb: case K::Extsh: case K::Cntlzw:
        case K::Rlwnm: case K::RlwnmRc:
        case K::Lfsu: case K::Lfdu: case K::Stfsu: case K::Stfdu: case K::PsqLu: case K::PsqStu:
            define(o.a);
            break;
        case K::Lmw:
            for (uint32_t r = o.d; r < 32; ++r) define(r);
            break;
        case K::Cmpwi: case K::Cmplwi: case K::Cmpw: case K::Cmplw: case K::CmpwiBc: case K::CmplwiBc: case K::CmpwBc:
        case K::CmplwBc: case K::Mtlr: case K::Mtctr: case K::Stwx: case K::Stbx: case K::Sthx: case K::Stmw:
        case K::Lfs: case K::Lfd: case K::Stfs: case K::Stfd: case K::Lfsx: case K::Stfsx: case K::PsqL: case K::PsqSt:
        case K::Fadds: case K::Fsubs: case K::Fmuls: case K::Fdivs: case K::Fmadds: case K::Fmsubs: case K::Fnmsubs:
        case K::Fnmadds: case K::Fadd: case K::Fsub: case K::Fmul: case K::Fdiv: case K::Fmadd: case K::Fmsub: case K::Fneg:
        case K::Fabs: case K::Fmr: case K::Frsp: case K::Fcmpu: case K::PsAdd: case K::PsSub: case K::PsMul: case K::PsMadd:
        case K::PsMsub: case K::PsNmadd: case K::PsNmsub: case K::PsMuls0: case K::PsMuls1: case K::PsMadds0:
        case K::PsMadds1: case K::PsSum0: case K::PsSum1: case K::PsMerge00: case K::PsMerge01: case K::PsMerge10:
        case K::PsMerge11: case K::PsMr: case K::PsNeg: case K::PsAbs: case K::Bc: case K::Bclr: case K::Jmp:
            break;  // no GPR written, nothing called that clears a mark or changes the policy
        default:  // Generic, calls, jumps, mtspr, anything else: a new start
            reset();
            break;
        }
    }
    for (Group& g : groups) close(g);
}

// The slow path of a D-form store that tests for itself or leads a group. The address and the value are taken again from
// the guest registers once the snapshot is flushed (the main path changed none of them before the test). A leader's
// range policy byte is set when its chunk or the next one is flagged: its own chunk's byte and whether the range reaches
// the next chunk decide first (back to the main path when neither). With fault stores the main path tests nothing: this is
// the slow entry of the store's site (EmitSiteFast first, the policy of the store's own first byte; ecx holds the guest
// address of the range's first byte, `offset` below the store's), and the members fault for themselves.
struct StoreDCold {
    Patch slow;
    uint32_t join, back, pcIndex, imm;
    uint8_t ra, rs, width, offset, span;
    bool update, joinNeedsEa, leader;  // joinNeedsEa: the join writes rA from eax (else it adds to rA's register)
    BlockEmitter::GprSnapshot snap;
    bool fault = false;
    uint8_t site = 0, value = 0;  // fault: the site, and the host register it stored
    void operator()(BlockEmitter& e) const {
        Xbyak::CodeGenerator& a = e.a;
        if (fault) {
            const SiteMarks marks{offset, leader ? span : uint8_t{0}, false};
            EmitSiteFast(e, site, SiteAddr{static_cast<uint8_t>(rcx.getIdx()), offset}, width, Xbyak::Reg32(value), false, join,
                         false, marks)
                .Resolve(a);
        } else {
            Resolve(a, slow);
        }
        Patches real{};
        if (leader && !fault) {  // rsi = the range's chunk, cx = the low half of its canonical first byte
            a.cmp(a.byte[kLean + rsi + kPolicyDisp], 0);
            real.Add(EmitJcc(a, kNe));
            a.cmp(cx, 0x10000 - span);
            real.Add(EmitJcc(a, kA));
            a.mov(rdi, reinterpret_cast<uintptr_t>(&g_groups.rangeBack));  // rdi: no op value lives in it
            a.inc(a.qword[rdi]);
            JmpTo(a, back);
            real.Resolve(a);
        }
        e.EmitFlushSnapshot(snap);
        e.StorePc(pcIndex);
        a.mov(edx, e.Gpr(rs));
        EmitRebuildEa(a, e, ra, imm);
        if (leader) {  // its own store as the interpreter's (its own chunk decides), then the members' range
            const bool used = e.memPlan.used[pcIndex];
            const auto shim = width == 1 ? &GroupStoreFlagged<uint8_t> : width == 2 ? &GroupStoreFlagged<uint16_t>
                                                                                   : &GroupStoreFlagged<uint32_t>;
            e.CallHelper(shim, kFrame, eax, rdx,
                         static_cast<uint32_t>(offset) | (static_cast<uint32_t>(span) << 8) | (used ? 1u << 16 : 0u));
            a.cmp(eax, 1);
            const Patch leave = EmitJcc(a, kA), pending = EmitJcc(a, kE);
            e.EmitReloadSnapshotVolatile(snap);
            if (joinNeedsEa) EmitRebuildEa(a, e, ra, imm);
            JmpTo(a, join);
            Resolve(a, pending);
            e.ExitPending();
            Resolve(a, leave);  // the members may not write without a test: go on at the next op through the dispatcher
            if (update) {
                EmitRebuildEa(a, e, ra, imm);
                a.mov(e.Gpr(ra), eax);
            }
            e.ExitTo(e.PcOf(pcIndex + 1));
            return;
        }
        e.CallHelper(ShimFor(width), kFrame, eax, rdx);
        EmitPendingCheck(e);
        e.EmitReloadSnapshotVolatile(snap);
        if (joinNeedsEa) EmitRebuildEa(a, e, ra, imm);
        JmpTo(a, join);
    }
};

// The slow entry of a group member's site (fault stores): EmitSiteFast, else the checked path's routine for its own store as
// the interpreter's (the address and the value taken again from the guest registers), then on after the site (the leader marked
// the group's bytes).
struct MemberCold {
    uint32_t join, pcIndex, imm;
    int32_t disp;  // the site's address: reg + disp
    uint8_t site, reg, value, ra, rs, width;
    bool update;
    BlockEmitter::GprSnapshot snap;
    void operator()(BlockEmitter& e) const {
        Xbyak::CodeGenerator& a = e.a;
        EmitSiteFast(e, site, SiteAddr{reg, disp}, width, Xbyak::Reg32(value), false, join, false, kMarkNone).Resolve(a);
        e.EmitFlushSnapshot(snap);
        e.StorePc(pcIndex);
        a.mov(edx, e.Gpr(rs));
        EmitRebuildEa(a, e, ra, imm);
        e.CallHelper(ShimFor(width), kFrame, eax, rdx);
        EmitPendingCheck(e);
        e.EmitReloadSnapshotVolatile(snap);
        if (update) EmitRebuildEa(a, e, ra, imm);  // the join writes rA from eax
        JmpTo(a, join);
    }
};

// A D-form store with fault stores (role as the plan has it, see EmitIntStoreD): one site, and for a store that is not a member
// the marks after it (of the group's bytes for a leader). The main path tests no policy.
uint32_t EmitIntStoreDFault(BlockEmitter& e, const Op& o, const Shape& s, uint8_t role) {
    Xbyak::CodeGenerator& a = e.a;
    BlockEmitter::MemPlan& p = e.memPlan;
    const uint32_t i = e.index;
    const int32_t disp = static_cast<int32_t>(o.imm);
    const Xbyak::Reg32* base = e.WantCacheGpr(o.a) ? &e.ReadGpr(o.a) : nullptr;
    const Xbyak::Reg32* value = e.WantCacheGpr(o.d) ? &e.ReadGpr(o.d, 1u << o.a) : nullptr;
    if (value == nullptr) {
        a.mov(edx, e.Gpr(o.d));
        value = &edx;
    }
    if (role == kRoleMember) {
        // the address: base + disp, in eax for an update form (rA's new value) or a base read from memory
        SiteAddr at{};
        if (s.update) {
            if (base != nullptr) a.lea(eax, a.ptr[base->cvt64() + disp]);
            else {
                a.mov(eax, e.Gpr(o.a));
                if (disp != 0) a.add(eax, disp);
            }
            at = SiteAddr{static_cast<uint8_t>(rax.getIdx()), 0};
        } else if (base != nullptr) {
            at = SiteAddr{static_cast<uint8_t>(base->getIdx()), disp};
        } else {
            a.mov(eax, e.Gpr(o.a));
            at = SiteAddr{static_cast<uint8_t>(rax.getIdx()), disp};
        }
        const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
        const uint32_t site = EmitSiteStore(e, s.width, at, *value);
        const uint32_t join = Here(a);
        if (s.update) {
            if (e.GprReadLater(o.a)) a.mov(e.WriteGpr(o.a), eax);
            else { e.DropGpr(o.a); a.mov(e.Gpr(o.a), eax); }
        }
        e.Cold(MemberCold{join, i, o.imm, at.disp, static_cast<uint8_t>(site), at.reg, static_cast<uint8_t>(value->getIdx()), o.a,
                          o.d, s.width, s.update, snap});
        return 1;
    }
    const int32_t first = role == kRoleLeader ? p.rangeDisp[i] : disp;  // from the base to the first byte marked
    const int32_t offset = disp - first;
    // an update form whose base is cached and read again adds the displacement to its register at the join (both paths
    // reach it with the old value there); any other keeps the address in eax for rA
    const bool joinAdds = s.update && base != nullptr && e.GprReadLater(o.a);
    // ecx = the guest address of the first byte marked (32 bits: the site adds `offset` in 64, past 2^32 into the tail, which
    // faults); the site stores at ecx + offset
    if (s.update && !joinAdds) {  // eax = the address (rA's new value)
        if (base != nullptr) a.lea(eax, a.ptr[base->cvt64() + disp]);
        else {
            a.mov(eax, e.Gpr(o.a));
            a.add(eax, disp);
        }
        if (first == disp) a.mov(ecx, eax);
        else a.lea(ecx, a.ptr[rax + (first - disp)]);
    } else if (base != nullptr) {
        if (first != 0) a.lea(ecx, a.ptr[base->cvt64() + first]);
        else a.mov(ecx, *base);
    } else {
        a.mov(ecx, e.Gpr(o.a));
        if (first != 0) a.add(ecx, first);
    }
    const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
    const uint32_t site = EmitSiteStore(e, s.width, SiteAddr{static_cast<uint8_t>(rcx.getIdx()), offset}, *value);
#if MKW_LEAN_DIRTY_MARKS
    // as EmitIntStoreD's: the pair from the range's first block and, for a range that may reach two blocks on, its last byte's;
    // from the unmasked address (EmitMarkUnmasked: only a store that went through the store view gets here)
    const bool far = role == kRoleLeader && p.span[i] > kGroupOnePair;
    if (far) a.lea(esi, a.ptr[rcx + (p.span[i] - 1)]);
    EmitMarkUnmasked(a, ecx);
    if (far) {
        a.shr(esi, 6);
        a.mov(a.byte[kLean + rsi + kMarkUnmaskedDisp], 1);
    }
#endif
    p.done[i] = role == kRoleLeader;
    const uint32_t join = Here(a);
    if (joinAdds) {
        a.add(e.WriteGpr(o.a), disp);
    } else if (s.update) {
        if (e.GprReadLater(o.a)) a.mov(e.WriteGpr(o.a), eax);
        else { e.DropGpr(o.a); a.mov(e.Gpr(o.a), eax); }
    }
    StoreDCold cold{Patch{}, join, 0, i, o.imm, o.a, o.d, s.width, static_cast<uint8_t>(offset),
                    role == kRoleLeader ? p.span[i] : uint8_t{0}, s.update, s.update && !joinAdds, role == kRoleLeader, snap};
    cold.fault = true;
    cold.site = static_cast<uint8_t>(site);
    cold.value = static_cast<uint8_t>(value->getIdx());
    e.Cold(cold);
    return 1;
}

// A D-form integer store (rA != 0, not byte-reversed) as the plan has it: alone (the policy test, the store, its marks),
// leading a group (the range policy test and the group's marks), or in a group whose leader was emitted as one (the
// store alone). With fault stores: EmitIntStoreDFault (a base of the MMIO window keeps the checked path).
uint32_t EmitIntStoreD(BlockEmitter& e, const Op& o, const Shape& s) {
    Xbyak::CodeGenerator& a = e.a;
    BlockEmitter::MemPlan& p = e.memPlan;
    const uint32_t i = e.index;
    uint8_t role = p.role[i];
    if (role == kRoleMember) {
        if (p.done[p.leader[i]]) p.used[p.leader[i]] = true;
        else role = kRoleSingle;
    }
    if (role == kRoleMember) g_groups.members.fetch_add(1, std::memory_order_relaxed);
    if (role == kRoleLeader) g_groups.leaders.fetch_add(1, std::memory_order_relaxed);
    if (FaultStores() && !p.mmio[i]) return EmitIntStoreDFault(e, o, s, role);
    if (role == kRoleMember && LeanWindow::g_aliased) {  // the store alone, at its unmasked address
        // A member's bytes lie in the range its leader tested, which no 1 GiB view's end (the 32-bit wrap included) cuts:
        // the address is never below the window, and past 4 GiB it is in the aliased tail.
        const Xbyak::RegExp at = EmitAliasedAddress(e, o, s);
        const Xbyak::Reg32* value = e.WantCacheGpr(o.d) ? &e.ReadGpr(o.d, 1u << o.a) : nullptr;
        if (value == nullptr) {
            a.mov(edx, e.Gpr(o.d));
            value = &edx;
        }
        EmitStoreValue(a, s.width, at, *value);
        if (s.update) {
            if (e.GprReadLater(o.a)) a.mov(e.WriteGpr(o.a), eax);
            else { e.DropGpr(o.a); a.mov(e.Gpr(o.a), eax); }
        }
        return 1;
    }
    const int32_t disp = static_cast<int32_t>(o.imm);
    const int32_t first = role == kRoleLeader ? p.rangeDisp[i] : disp;  // from the base to the first byte tested and marked
    const int32_t offset = disp - first;
    const Xbyak::Reg32* base = e.WantCacheGpr(o.a) ? &e.ReadGpr(o.a) : nullptr;
    const Xbyak::Reg32* value = e.WantCacheGpr(o.d) ? &e.ReadGpr(o.d, 1u << o.a) : nullptr;
    if (value == nullptr) {
        a.mov(edx, e.Gpr(o.d));
        value = &edx;
    }
    // an update form whose base is cached and read again adds the displacement to its register at the join (both paths
    // reach it with the old value there); any other keeps the address in eax for rA
    const bool joinAdds = s.update && base != nullptr && e.GprReadLater(o.a);
    if (s.update && !joinAdds) {  // eax = the address (rA's new value)
        if (base != nullptr) a.lea(eax, a.ptr[base->cvt64() + disp]);
        else {
            a.mov(eax, e.Gpr(o.a));
            a.add(eax, disp);
        }
        if (first == disp) a.mov(ecx, eax);
        else a.lea(ecx, a.ptr[rax + (first - disp)]);
    } else if (base != nullptr) {
        if (first != 0) a.lea(ecx, a.ptr[base->cvt64() + first]);
        else a.mov(ecx, *base);
    } else {
        a.mov(ecx, e.Gpr(o.a));
        if (first != 0) a.add(ecx, first);
    }
    a.and_(ecx, kMask);
    const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
    Patch slow{};
    uint32_t back = 0;
    if (role != kRoleMember) {
        a.mov(esi, ecx);
        a.shr(esi, MemoryInline::kLeanPolicyShift);
        a.cmp(a.byte[kLean + rsi + (role == kRoleLeader ? kRangePolicyDisp : kPolicyDisp)], 0);
        slow = EmitJcc(a, kNe);
        back = Here(a);
    }
    EmitStoreValue(a, s.width, kLean + rcx + offset, *value);
    if (role != kRoleMember) {
#if MKW_LEAN_DIRTY_MARKS
        // the pair from the range's first block (the lowest store's own pair) and, for a range that may reach two blocks on,
        // the block of its last byte (in the highest store's own pair): every block written and none the interpreter does
        // not mark
        const bool far = role == kRoleLeader && p.span[i] > kGroupOnePair;
        if (far) a.lea(esi, a.ptr[rcx + (p.span[i] - 1)]);
        a.shr(ecx, 6);
        a.mov(a.word[kLean + rcx + kDirtyDisp], 0x0101);
        if (far) {
            a.shr(esi, 6);
            a.mov(a.byte[kLean + rsi + kDirtyDisp], 1);
        }
#endif
        p.done[i] = role == kRoleLeader;
    }
    const uint32_t join = Here(a);
    if (joinAdds) {
        a.add(e.WriteGpr(o.a), disp);
    } else if (s.update) {
        if (e.GprReadLater(o.a)) a.mov(e.WriteGpr(o.a), eax);
        else { e.DropGpr(o.a); a.mov(e.Gpr(o.a), eax); }
    }
    if (role != kRoleMember)
        e.Cold(StoreDCold{slow, join, back, i, o.imm, o.a, o.d, s.width, static_cast<uint8_t>(offset),
                          role == kRoleLeader ? p.span[i] : uint8_t{0}, s.update, s.update && !joinAdds,
                          role == kRoleLeader, snap});
    return 1;
}

// --- the integer ops ------------------------------------------------------------------------------------------------------

uint32_t EmitIntLoad(BlockEmitter& e, const Op& o, const Shape& s) {
    Xbyak::CodeGenerator& a = e.a;
    if (LeanWindow::g_aliased) {  // no mask: the byte the masked address names (EmitAliasedAddress)
        const Xbyak::RegExp at = EmitAliasedAddress(e, o, s);
        const uint32_t keep = s.zeroBase ? 0u : 1u << o.a;  // the base stays cached for the ops after
        if (e.GprReadLater(o.d)) {
            const Xbyak::Reg32& dst = e.WriteGpr(o.d, keep);
            EmitLoadSwappedAt(a, s.width, dst, at, s.reverse);
            if (s.sign) a.movsx(dst, dst.cvt16());
        } else {
            e.DropGpr(o.d);
            EmitLoadSwappedAt(a, s.width, edx, at, s.reverse);
            if (s.sign) a.movsx(edx, dx);
            a.mov(e.Gpr(o.d), edx);
        }
        if (s.update) {
            if (e.GprReadLater(o.a)) a.mov(e.WriteGpr(o.a, 1u << o.d), eax);
            else { e.DropGpr(o.a); a.mov(e.Gpr(o.a), eax); }
        }
        return 1;
    }
    if (s.update) {
        EmitEa(e, o, s.indexed);
        EmitCanonical(e);
    } else {
        EmitCanonicalEa(e, o, s.indexed, s.zeroBase);
    }
    if (e.GprReadLater(o.d)) {
        const Xbyak::Reg32& dst = e.WriteGpr(o.d);
        EmitLoadSwappedTo(a, s.width, dst, s.reverse);
        if (s.sign) a.movsx(dst, dst.cvt16());
    } else {
        e.DropGpr(o.d);
        EmitLoadSwappedTo(a, s.width, edx, s.reverse);
        if (s.sign) a.movsx(edx, dx);
        a.mov(e.Gpr(o.d), edx);
    }
    if (s.update) {
        if (e.GprReadLater(o.a)) a.mov(e.WriteGpr(o.a, 1u << o.d), eax);
        else { e.DropGpr(o.a); a.mov(e.Gpr(o.a), eax); }
    }
    return 1;
}

uint32_t EmitIntStore(BlockEmitter& e, const Op& o, const Shape& s, bool fromFpr = false) {
    Xbyak::CodeGenerator& a = e.a;
    if (!s.indexed && !s.zeroBase && !s.reverse && !fromFpr) return EmitIntStoreD(e, o, s);
    EmitEa(e, o, s.indexed, s.zeroBase);
    // after the address: the address does not read rS, an update form's rA is written last. stfiwx stores the low word of the
    // register's bits whatever its form
    if (fromFpr) a.mov(edx, e.FprWord(o.d, 0));
    else if (e.WantCacheGpr(o.d)) a.mov(edx, e.ReadGpr(o.d));
    else a.mov(edx, e.Gpr(o.d));
    if (FaultStores()) {  // the site at eax, then the marks
        const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
        StoreCold cold{Patch{}, 0, e.index, o.imm, o.a, o.b, s.width, s.update, s.indexed, s.reverse, snap};
        cold.fault = true;
        cold.site = static_cast<uint8_t>(EmitSiteStore(e, s.width, SiteAddr{static_cast<uint8_t>(rax.getIdx()), 0}, edx, s.reverse));
                EmitMarksOfEax(a);
        cold.join = Here(a);
        if (s.update) {
            if (e.GprReadLater(o.a)) a.mov(e.WriteGpr(o.a), eax);
            else { e.DropGpr(o.a); a.mov(e.Gpr(o.a), eax); }
        }
        e.Cold(cold);
        return 1;
    }
    EmitCanonical(e);
    const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
    const Patch slow = EmitStoreChecked(e, s.width, s.reverse);
    const uint32_t join = Here(a);
    if (s.update) {
        if (e.GprReadLater(o.a)) a.mov(e.WriteGpr(o.a), eax);
        else { e.DropGpr(o.a); a.mov(e.Gpr(o.a), eax); }
    }
    e.Cold(StoreCold{slow, join, e.index, o.imm, o.a, o.b, s.width, s.update, s.indexed, s.reverse, snap});
    return 1;
}

// lmw: one load per register (unrolled), each as FlatRead32 at ea + 4 * k. The words lie in the lean window one after
// the other (the last 128 bytes below the window's top are unbacked, the first word faults there as in the interpreter).
uint32_t EmitLmw(BlockEmitter& e, const Op& o) {
    Xbyak::CodeGenerator& a = e.a;
    Xbyak::RegExp first = kLean + rcx;
    if (LeanWindow::g_aliased) {  // each word at its own unmasked address (the flush below keeps the base register)
        first = EmitAliasedAddress(e, o, Shape{4, false, false, false});
    } else {
        EmitEa(e, o, false);
        EmitCanonical(e);
    }
    e.FlushAllGprs();
    for (uint32_t k = o.d; k < 32; ++k) {
        const Xbyak::RegExp at = first + static_cast<int>(4 * (k - o.d));
#if defined(__MOVBE__)
        a.movbe(edx, a.dword[at]);
#else
        a.mov(edx, a.dword[at]);
        a.bswap(edx);
#endif
        a.mov(e.Gpr(k), edx);
    }
    return 1;
}

// stmw: the interpreter does one FlatWrite32 per register. The words are 4 bytes apart and cover at most 128 bytes, so
// they start in the chunk of the first word or the one of the last (a policy byte per 64 KiB: test those two) and
// the pairs of dirty marks they set are the union of the pair at the first word's block and the pair at the last word's
// (at most 2 blocks apart). If either chunk is flagged, a shim runs the whole per-word loop as the interpreter does.
struct StmwCold {
    Patches slow;
    uint32_t join, pcIndex, first;
    BlockEmitter::GprSnapshot snap;
    void operator()(BlockEmitter& e) const {
        slow.Resolve(e.a);
        e.StorePc(pcIndex);
        e.CallHelper(&StmwFlagged, kFrame, eax, first);
        EmitPendingCheck(e);
        e.EmitReloadSnapshotVolatile(snap);
        JmpTo(e.a, join);
    }
};

uint32_t EmitStmw(BlockEmitter& e, const Op& o) {
    Xbyak::CodeGenerator& a = e.a;
    const uint32_t n = 32 - o.d;
    StmwCold cold{};
    EmitEa(e, o, false);
    e.FlushDirtyGprs();
    cold.snap = e.SnapshotGprs();
    EmitCanonical(e);
    a.mov(esi, ecx);
    a.shr(esi, MemoryInline::kLeanPolicyShift);
    a.cmp(a.byte[kLean + rsi + kPolicyDisp], 0);
    cold.slow.Add(EmitJcc(a, kNe));
    if (n > 1) {
        a.lea(edx, a.ptr[rax + static_cast<int>(4 * (n - 1))]);
        a.and_(edx, kMask);
        a.mov(esi, edx);
        a.shr(esi, MemoryInline::kLeanPolicyShift);
        a.cmp(a.byte[kLean + rsi + kPolicyDisp], 0);
        cold.slow.Add(EmitJcc(a, kNe));
    }
#if MKW_LEAN_DIRTY_MARKS
    a.mov(esi, ecx);
    a.shr(esi, 6);
    a.mov(a.word[kLean + rsi + kDirtyDisp], 0x0101);
    if (n > 1) {
        a.shr(edx, 6);
        a.mov(a.word[kLean + rdx + kDirtyDisp], 0x0101);
    }
#endif
    for (uint32_t k = o.d; k < 32; ++k) {
        const Xbyak::RegExp at = kLean + rcx + static_cast<int>(4 * (k - o.d));
        a.mov(edx, e.Gpr(k));
#if defined(__MOVBE__)
        a.movbe(a.dword[at], edx);
#else
        a.bswap(edx);
        a.mov(a.dword[at], edx);
#endif
    }
    cold.join = Here(a);
    cold.pcIndex = e.index;
    cold.first = o.d;
    e.Cold(cold);
    return 1;
}

// --- floating point loads and stores --------------------------------------------------------------------------------------
// The FPR side of these ops goes through the FPR cache (jit_x64_fpr.cpp): a load writes the destination's slot (or memory when
// nothing later in the block reads it: BlockEmitter::WriteFpr), a store reads the register's slot when it has one, and the forms
// they give are pending writes (BlockEmitter::SetFprForm). A cold path that writes the destination in memory (a shim) reloads the
// snapshot it stored and the destination's slot.

#if defined(__AVX__)
constexpr bool kFpAvx = true;
#else
constexpr bool kFpAvx = false;
#endif

void FpMovqTo(CG& a, const Xbyak::Xmm& x, const Xbyak::Reg64& r) {
    if (kFpAvx) a.vmovq(x, r);
    else a.movq(x, r);
}
void FpMovqFrom(CG& a, const Xbyak::Reg64& r, const Xbyak::Xmm& x) {
    if (kFpAvx) a.vmovq(r, x);
    else a.movq(r, x);
}
void FpLoadSlot(CG& a, BlockEmitter& e, int slot, uint32_t r) {
    if (slot >= 0) e.LoadFpr(BlockEmitter::FprCacheReg(static_cast<uint32_t>(slot)), r);
}

// rdx = FPR r's raw bits: from its slot when it has one, else from memory.
void FprBitsToRdx(BlockEmitter& e, uint32_t r) {
    const int slot = e.FindCachedFpr(r);
    if (slot >= 0) FpMovqFrom(e.a, rdx, BlockEmitter::FprCacheReg(static_cast<uint32_t>(slot)));
    else e.a.mov(rdx, e.Fpr(r));
}

// rdx (the raw bits of a register held as a pair) becomes Scalar(f, r): ps0 widened (PPC_PsToScalarInline: cvtss2sd of the high
// word). Clobbers xmm0.
void PairBitsToScalar(CG& a) {
    FpMovqTo(a, xmm0, rdx);
    if (kFpAvx) {
        a.vpsrlq(xmm0, xmm0, 32);
        a.vcvtss2sd(xmm0, xmm0, xmm0);
    } else {
        a.psrlq(xmm0, 32);
        a.cvtss2sd(xmm0, xmm0);
    }
    FpMovqFrom(a, rdx, xmm0);
}

// lfs: a normal single or a zero widens exactly with cvtss2sd (PpcConvertToDoubleBitsInline); denormals, infinities and
// NaNs (the word is in edx) go to the manual's algorithm in a shim and rejoin.
struct LfsCold {
    Patch other;
    uint32_t join, imm;
    uint8_t d, ra, rb;
    bool update, indexed;
    int8_t dSlot;
    BlockEmitter::GprSnapshot snap;
    void operator()(BlockEmitter& e) const {
        Resolve(e.a, other);
        e.EmitFlushSnapshot(snap);
        e.CallHelper(&LfsOther, kFrame, edx, static_cast<uint32_t>(d));
        e.EmitReloadSnapshotVolatile(snap);
        FpLoadSlot(e.a, e, dSlot, d);
        if (update) EmitRebuildEa(e.a, e, ra, imm, indexed, rb);
        JmpTo(e.a, join);
    }
};

// The address of a float load's bytes: unmasked in the aliased view (EmitAliasedAddress, which sets eax for an update form), else
// [r12 + rcx] with ecx the canonical address (and eax the address for an update form).
Xbyak::RegExp EmitFloatLoadAddress(BlockEmitter& e, const Op& o, unsigned width, bool update, bool indexed, bool zeroBase) {
    if (LeanWindow::g_aliased) return EmitAliasedAddress(e, o, Shape{static_cast<uint8_t>(width), update, indexed, false, zeroBase});
    if (update) {
        EmitEa(e, o, indexed, zeroBase);
        EmitCanonical(e);
    } else {
        EmitCanonicalEa(e, o, indexed, zeroBase);
    }
    return kLean + rcx;
}

uint32_t EmitLfs(BlockEmitter& e, const Op& o, bool update, bool indexed, bool zeroBase = false) {
    Xbyak::CodeGenerator& a = e.a;
    EmitLoadSwappedAt(a, 4, edx, EmitFloatLoadAddress(e, o, 4, update, indexed, zeroBase));
    const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
    Patch other{};
#if !MKW_FAST_FLOAT
    // Normal or zero: t = the word shifted left once (no sign), t - 2^24 is below 0xFE000000 for a normal; a zero's word (0 or
    // 0x80000000) is put in that range by the cmov, so zeros and normals take the same path with no branch between them (a
    // second branch for zeros, two fewer instructions, mispredicts where zeros and normals mix: measured +80% cycles on lfs).
    a.lea(esi, a.ptr[rdx + rdx - 0x01000000]);
    a.test(edx, 0x7FFFFFFF);
    a.cmovz(esi, edx);
    a.cmp(esi, 0xFE000000u);
    other = EmitJcc(a, static_cast<Cc>(0x3));  // jae: a denormal, an infinity or a NaN
#endif
    const int dSlot = e.WriteFpr(o.d, Form::Scalar, xmm0, 0, [&](const Xbyak::Xmm& x) {
        if (kFpAvx) {
            a.vmovd(x, edx);
            a.vcvtss2sd(x, x, x);
        } else {
            a.movd(x, edx);
            a.cvtss2sd(x, x);
        }
    });
    const uint32_t join = Here(a);
    if (update) a.mov(e.WriteGpr(o.a), eax);
#if !MKW_FAST_FLOAT
    e.Cold(LfsCold{other, join, o.imm, o.d, o.a, o.b, update, indexed, static_cast<int8_t>(dSlot), snap});
#else
    (void)snap;
    (void)dSlot;
#endif
    return 1;
}

uint32_t EmitLfd(BlockEmitter& e, const Op& o, bool update, bool indexed = false, bool zeroBase = false) {
    Xbyak::CodeGenerator& a = e.a;
    // FlatReadFloat64 under lean memory is FlatLoad<uint64_t>
    EmitLoadSwappedAt(a, 8, edx, EmitFloatLoadAddress(e, o, 8, update, indexed, zeroBase));
    if (e.FprReadLater(o.d) && (e.FindCachedFpr(o.d) >= 0 || e.FprSlotFree(0))) {
        FpMovqTo(a, e.ReserveFpr(o.d), rdx);
        e.MarkFprDirty(o.d);
    } else {
        e.DropFpr(o.d);
        a.mov(e.Fpr(o.d), rdx);
    }
    e.SetFprForm(o.d, Form::Scalar);
    if (update) a.mov(e.WriteGpr(o.a), eax);
    return 1;
}

// stfs / stfd: the register read as a scalar (a Paired one's ps0 widened, out of line when the form is not known), stfs's
// narrowing (the subnormal-single window in a shim), then the checked store.
struct StfCold {
    Patch paired, subnormal, slow;
    uint32_t haveValue, haveBits, join, pcIndex, imm;
    uint8_t d, ra, rb, width;
    bool update, hasSubnormal, indexed, havePaired;
    BlockEmitter::GprSnapshot snap;
    // fault stores: the site (its address addrReg + addrDisp), and whether eax must be made for the routine
    bool fault = false, rebuildEa = false;
    uint8_t site = 0, addrReg = 0;
    int32_t addrDisp = 0;
    void operator()(BlockEmitter& e) const {
        Xbyak::CodeGenerator& a = e.a;
        if (havePaired) {
            Resolve(a, paired);
            PairBitsToScalar(a);
            JmpTo(a, haveValue);
        }
        if (hasSubnormal) {
            Resolve(a, subnormal);
            e.EmitFlushSnapshot(snap);
            e.CallHelper(&SubnormalSingle, rdx);
            e.EmitReloadSnapshotVolatile(snap);
            a.mov(edx, eax);
            JmpTo(a, haveBits);
        }
        if (fault) {
            EmitSiteFast(e, site, SiteAddr{addrReg, addrDisp}, width, edx, false, join, false, kMarkOwn).Resolve(a);
            if (SiteSwapsFirst(width, false)) EmitUnswap(a, width);
        } else {
            Resolve(a, slow);
        }
        e.EmitFlushSnapshot(snap);
        e.StorePc(pcIndex);
        if (rebuildEa) EmitRebuildEa(a, e, ra, imm, indexed, rb);
        e.CallHelper(ShimFor(width), kFrame, eax, rdx);
        EmitPendingCheck(e);
        e.EmitReloadSnapshotVolatile(snap);
        if (update) EmitRebuildEa(a, e, ra, imm, indexed, rb);
        JmpTo(a, join);
    }
};

uint32_t EmitStf(BlockEmitter& e, const Op& o, bool single, bool update, bool indexed, bool zeroBase = false) {
    Xbyak::CodeGenerator& a = e.a;
    StfCold cold{};
    // the address's registers in the GPR cache first, so the state is the same on every path from here
    if (!zeroBase) {
        if (e.WantCacheGpr(o.a)) (void)e.ReadGpr(o.a);
        if (indexed && e.WantCacheGpr(o.b)) (void)e.ReadGpr(o.b, 1u << o.a);
    } else if (indexed && e.WantCacheGpr(o.b)) {
        (void)e.ReadGpr(o.b);
    }
    cold.snap = e.SnapshotGprs();
    FprBitsToRdx(e, o.d);
    const Form known = e.KnownForm(e.index, o.d);
    if (known == Form::Paired) {  // no test: ps0 widened, in line
        BlockEmitter::NoteForm(BlockEmitter::FormEvent::ConversionForced);
        PairBitsToScalar(a);
    } else if (known == Form::Scalar) {  // no test: the register holds a double
        BlockEmitter::NoteForm(BlockEmitter::FormEvent::TestSkipped);
    } else {
        BlockEmitter::NoteForm(BlockEmitter::FormEvent::TestEmitted);
        a.test(e.PairedMask(), 1u << (o.d & 31u));
        cold.paired = EmitJcc(a, kNe);
        cold.havePaired = true;
    }
    cold.haveValue = Here(a);
    if (single) {
#if MKW_FAST_FLOAT
        FpMovqTo(a, xmm0, rdx);
        a.cvtsd2ss(xmm0, xmm0);
        a.movd(edx, xmm0);
#else
        // ConvertPpcDoubleToSingleBits: the exponents 874..896 are the subnormal-single arm's (a shim); every other is the
        // narrowing to sign, exponent's top bit and bits 58..29, which is pext with that mask.
        cold.hasSubnormal = true;
#if defined(__BMI2__)
        a.rorx(rcx, rdx, 31);  // ecx = bits 62..31: the exponent in its top 11 bits
        a.sub(ecx, 874u << 21);
        a.cmp(ecx, 23u << 21);
        cold.subnormal = EmitJcc(a, kB);
        a.pext(rdx, rdx, e.Const128(0xC7FFFFFFE0000000ull));
#else
        a.mov(rcx, rdx);
        a.shr(rcx, 52);
        a.and_(ecx, 0x7FF);
        a.sub(ecx, 874);
        a.cmp(ecx, 22);
        cold.subnormal = EmitJcc(a, kBe);
        a.mov(rsi, rdx);
        a.shr(rsi, 32);
        a.and_(esi, 0xC0000000u);
        a.shr(rdx, 29);
        a.and_(edx, 0x3FFFFFFF);
        a.or_(edx, esi);
#endif
        cold.haveBits = Here(a);
#endif
    }
    cold.width = single ? 4 : 8;
    if (FaultStores()) {  // a site, then the marks
        cold.fault = true;
        const int hit = !update && !indexed && !zeroBase ? e.FindCachedGpr(o.a) : -1;
        if (hit >= 0) {  // at base + disp, the marks from ecx: no eax (the routine's call makes it)
            const Xbyak::Reg32& base = BlockEmitter::GprCacheReg(static_cast<uint32_t>(hit));
            cold.addrReg = static_cast<uint8_t>(base.getIdx());
            cold.addrDisp = static_cast<int32_t>(o.imm);
            cold.rebuildEa = true;
            cold.site = static_cast<uint8_t>(EmitSiteStore(e, cold.width, SiteAddr{cold.addrReg, cold.addrDisp}, edx));
            a.lea(ecx, a.ptr[base.cvt64() + cold.addrDisp]);
            EmitMarkUnmasked(a, ecx);
        } else {
            EmitEa(e, o, indexed, zeroBase);
            cold.addrReg = static_cast<uint8_t>(rax.getIdx());
            cold.site = static_cast<uint8_t>(EmitSiteStore(e, cold.width, SiteAddr{cold.addrReg, 0}, edx));
                        EmitMarksOfEax(a);
        }
    } else {
        EmitEa(e, o, indexed, zeroBase);
        EmitCanonical(e);
        cold.slow = EmitStoreChecked(e, cold.width);
    }
    cold.join = Here(a);
    if (update) a.mov(e.WriteGpr(o.a), eax);
    cold.pcIndex = e.index;
    cold.imm = o.imm;
    cold.d = o.d;
    cold.ra = o.a;
    cold.rb = o.b;
    cold.update = update;
    cold.indexed = indexed;
    e.Cold(cold);
    return 1;
}

// --- quantized loads and stores -------------------------------------------------------------------------------------------

// psq_l: the float GQR (the test PPC_PsqLInline makes) is native; any other state calls PPC_PsqL in a shim. Both shims write
// the destination in memory.
struct PsqLCold {
    Patch generic, quiet;
    uint32_t join, pcIndex, imm;
    uint8_t d, b, ra;
    bool update, hasQuiet;
    int8_t dSlot;
    BlockEmitter::GprSnapshot snap;
    void operator()(BlockEmitter& e) const {
        Xbyak::CodeGenerator& a = e.a;
        if (hasQuiet) {
            Resolve(a, quiet);
            e.EmitFlushSnapshot(snap);
            e.CallHelper(&PsqPairQuieted, kFrame, rdx, static_cast<uint32_t>(d));
            e.EmitReloadSnapshotVolatile(snap);
            FpLoadSlot(a, e, dSlot, d);
            if (update) EmitRebuildEa(a, e, ra, imm);
            JmpTo(a, join);
        }
        Resolve(a, generic);
        e.EmitFlushSnapshot(snap);
        e.StorePc(pcIndex);
        e.CallHelper(&PsqLoad, kFrame, eax, static_cast<uint32_t>(d) | (static_cast<uint32_t>(b) << 8));
        EmitPendingCheck(e);
        e.EmitReloadSnapshotVolatile(snap);
        FpLoadSlot(a, e, dSlot, d);
        if (update) EmitRebuildEa(a, e, ra, imm);
        JmpTo(a, join);
    }
};

uint32_t EmitPsqL(BlockEmitter& e, const Op& o, bool update) {
    Xbyak::CodeGenerator& a = e.a;
    const uint32_t w = o.b >> 3, i = o.b & 7u;
    PsqLCold cold{};
    EmitEa(e, o, false);
    cold.snap = e.SnapshotGprs();
    if (w == 0) a.test(e.Gqr(i), 0xFFFF0000u);  // type 0 and load scale 0
    else a.test(e.Gqr(i), 0x00070000u);         // type 0
    cold.generic = EmitJcc(a, kNe);
    // the bytes at eax unmasked in the aliased view (EmitAliasedAddress), else at the canonical address
    Xbyak::RegExp at = kLean + rcx;
    if (LeanWindow::g_aliased) at = kLean + rax + (LeanWindow::g_storeView ? static_cast<int>(LeanWindow::kStoreViewOffset) : 0);
    else EmitCanonical(e);
    if (w == 0) {
        EmitLoadSwappedAt(a, 8, edx, at);  // ps0 in the high word, ps1 in the low
        FpMovqTo(a, xmm0, rdx);
#if !MKW_FAST_FLOAT
        // a NaN lane (a signalling one is quieted by the loader): that is the shim's
        cold.hasQuiet = true;
        if (kFpAvx) {
            a.vcmpunordps(xmm1, xmm0, xmm0);
            a.vmovmskps(esi, xmm1);
        } else {
            a.movaps(xmm1, xmm0);
            a.cmpunordps(xmm1, xmm1);
            a.movmskps(esi, xmm1);
        }
        a.test(esi, 3);
        cold.quiet = EmitJcc(a, kNe);
#endif
    } else {
        // PpcLoadSinglePsqFloatFastInline: ps0 = the word as it is, ps1 = 1.0f
        EmitLoadSwappedAt(a, 4, edx, at);
        a.shl(rdx, 32);
        a.mov(esi, 0x3F800000);
        a.or_(rdx, rsi);
        FpMovqTo(a, xmm0, rdx);
    }
    const int dSlot = e.WriteFpr(o.d, Form::Paired, xmm0, 0, [&](const Xbyak::Xmm& x) {
        if (x.getIdx() == 0) return;
        if (kFpAvx) a.vmovaps(x, xmm0);
        else a.movaps(x, xmm0);
    });
    cold.dSlot = static_cast<int8_t>(dSlot);
    cold.join = Here(a);
    if (update) a.mov(e.WriteGpr(o.a), eax);
    cold.pcIndex = e.index;
    cold.imm = o.imm;
    cold.d = o.d;
    cold.b = o.b;
    cold.ra = o.a;
    cold.update = update;
    e.Cold(cold);
    return 1;
}

// psq_st: native for a paired float store (W = 0, the GQR's store halfword 0) of a register that is not in Scalar form
// and whose lanes need no flushing or quieting, and for no other case: everything else calls PPC_PsqSt in a shim, which
// does what the interpreter does (policy, marks, MMIO, the quantizers).
struct PsqStCold {
    Patches generic;
    uint32_t join, pcIndex, imm;
    uint8_t d, b, ra;
    bool update;
    BlockEmitter::GprSnapshot snap;
    bool fault = false;  // fault stores: the site (at eax, `width` bytes of rdx)
    uint8_t site = 0, width = 0;
    void operator()(BlockEmitter& e) const {
        Xbyak::CodeGenerator& a = e.a;
        if (fault) EmitSiteFast(e, site, SiteAddr{static_cast<uint8_t>(rax.getIdx()), 0}, width, edx, false, join, width == 8, kMarkOwn).Resolve(a);
        generic.Resolve(a);
        e.EmitFlushSnapshot(snap);
        e.StorePc(pcIndex);
        e.CallHelper(&PsqStore, kFrame, eax, static_cast<uint32_t>(d) | (static_cast<uint32_t>(b) << 8));
        EmitPendingCheck(e);
        e.EmitReloadSnapshotVolatile(snap);
        if (update) EmitRebuildEa(a, e, ra, imm);
        JmpTo(a, join);
    }
};

uint32_t EmitPsqSt(BlockEmitter& e, const Op& o, bool update) {
    Xbyak::CodeGenerator& a = e.a;
    const uint32_t w = o.b >> 3, i = o.b & 7u;
    EmitEa(e, o, false);
    PsqStCold cold{};
    cold.snap = e.SnapshotGprs();
    if (w == 0) a.test(e.Gqr(i), 0x0000FFFFu);  // paired float: store type 0 and store scale 0
    else a.test(e.Gqr(i), 0x00000007u);         // single float: store type 0 (ignores store scale)
    cold.generic.Add(EmitJcc(a, kNe));
    // The pair to store: the register's bits when it is held as a pair; a register held as a scalar is Paired(f, r) of it
    // (PPC_PsFromScalarInline: the double narrowed by cvtsd2ss, in both lanes; prologues spill callee-saved FPRs this way),
    // narrowed straight from the register's slot or memory. W = 1 stores ps0 alone: edx.
    const int slot = e.FindCachedFpr(o.d);
    const auto pairBits = [&]() {
        FprBitsToRdx(e, o.d);
        if (w != 0) a.shr(rdx, 32);  // ps0 sits in the high word of the paired FPR
    };
    const auto fromScalar = [&]() {
        if (slot >= 0) {
            const Xbyak::Xmm& x = BlockEmitter::FprCacheReg(static_cast<uint32_t>(slot));
            if (kFpAvx) a.vcvtsd2ss(xmm0, xmm0, x);
            else a.cvtsd2ss(xmm0, x);
        } else if (kFpAvx) {
            a.vcvtsd2ss(xmm0, xmm0, e.Fpr(o.d));
        } else {
            a.cvtsd2ss(xmm0, e.Fpr(o.d));
        }
        if (w == 0) {
            if (kFpAvx) a.vmovsldup(xmm0, xmm0);  // the single in lanes 0 and 1
            else a.pshufd(xmm0, xmm0, 0x00);
            FpMovqFrom(a, rdx, xmm0);
        } else if (kFpAvx) {
            a.vmovd(edx, xmm0);
        } else {
            a.movd(edx, xmm0);
        }
    };
    // The store keeps a lane unless it is a subnormal (flushed to a signed zero) or a NaN (quieted): those, and only those, are
    // the shim's. Per lane, with m = its bits & 0x7FFFFFFF: (m << 1) - 2^24 is at most 0xFE000000 for a normal or an infinity;
    // a zero is the other lane kept (tested out of the way). lane 0 is the low word (edx), lane 1 the high word.
    const auto laneTest = [&](int lane) {
#if !MKW_FAST_FLOAT
        Xbyak::Label keep;
        const Xbyak::Reg32 bits = lane == 0 ? edx : esi;
        if (lane != 0) {
#if defined(__BMI2__)
            a.rorx(rsi, rdx, 32);
#else
            a.mov(rsi, rdx);
            a.shr(rsi, 32);
#endif
        }
        a.lea(edi, a.ptr[bits.cvt64() + bits.cvt64() - 0x01000000]);
        a.cmp(edi, 0xFE000000u);
        a.jbe(keep);
        a.test(bits, 0x7FFFFFFF);
        cold.generic.Add(EmitJcc(a, kNe));
        a.L(keep);
#else
        (void)lane;
#endif
    };
    const Form known = e.KnownForm(e.index, o.d);
    if (known == Form::Paired) {
        BlockEmitter::NoteForm(BlockEmitter::FormEvent::TestSkipped);
        pairBits();
        laneTest(0);
        if (w == 0) laneTest(1);
    } else if (known == Form::Scalar) {
        BlockEmitter::NoteForm(BlockEmitter::FormEvent::ConversionForced);
        fromScalar();
        laneTest(0);  // a pair made of a scalar has two equal lanes
    } else {
        BlockEmitter::NoteForm(BlockEmitter::FormEvent::TestEmitted);
        Xbyak::Label scalar, done;
        a.test(e.PairedMask(), 1u << (o.d & 31u));
        a.jz(scalar, CG::T_NEAR);
        pairBits();
        laneTest(0);
        if (w == 0) laneTest(1);
        a.jmp(done, CG::T_NEAR);
        a.L(scalar);
        fromScalar();
        laneTest(0);
        a.L(done);
    }
    if (FaultStores()) {  // a site (whose slow entry tests both ends of a pair), then the marks
        cold.fault = true;
        cold.width = w == 0 ? 8 : 4;
        cold.site = static_cast<uint8_t>(EmitSiteStore(e, cold.width, SiteAddr{static_cast<uint8_t>(rax.getIdx()), 0}, edx));
                EmitMarksOfEax(a);
        cold.join = Here(a);
        if (update) a.mov(e.WriteGpr(o.a), eax);
        cold.pcIndex = e.index;
        cold.imm = o.imm;
        cold.d = o.d;
        cold.b = o.b;
        cold.ra = o.a;
        cold.update = update;
        e.Cold(cold);
        return 1;
    }
    EmitCanonical(e);
    if (w == 0) {
        // both ends of the 8 bytes: a pair can straddle two chunks (LeanStoreNeedsPolicy(addr) and (addr + 7))
        a.mov(esi, ecx);
        a.shr(esi, MemoryInline::kLeanPolicyShift);
        a.cmp(a.byte[kLean + rsi + kPolicyDisp], 0);
        cold.generic.Add(EmitJcc(a, kNe));
        a.lea(edi, a.ptr[rax + 7]);
        a.and_(edi, kMask);
        a.shr(edi, MemoryInline::kLeanPolicyShift);
        a.cmp(a.byte[kLean + rdi + kPolicyDisp], 0);
        cold.generic.Add(EmitJcc(a, kNe));
#if MKW_LEAN_DIRTY_MARKS
        a.mov(esi, ecx);
        a.shr(esi, 6);
        a.mov(a.word[kLean + rsi + kDirtyDisp], 0x0101);
#endif
        // the packed register byte-swapped: ps0 (the high word) lands at the lower address, big-endian
#if defined(__MOVBE__)
        a.movbe(a.qword[kLean + rcx], rdx);
#else
        a.bswap(rdx);
        a.mov(a.qword[kLean + rcx], rdx);
#endif
    } else {
        cold.generic.Add(EmitStoreChecked(e, 4));
    }
    cold.join = Here(a);
    if (update) a.mov(e.WriteGpr(o.a), eax);
    cold.pcIndex = e.index;
    cold.imm = o.imm;
    cold.d = o.d;
    cold.b = o.b;
    cold.ra = o.a;
    cold.update = update;
    e.Cold(cold);
    return 1;
}

uint32_t EmitMemOp(BlockEmitter& e, uint32_t index) {
    const Op& o = e.block.ops[index];
    if (!e.memPlan.built) BuildPlan(e);
    Shape s;
    if (IntLoadShape(o.kind, s)) return EmitIntLoad(e, o, s);
    if (IntStoreShape(o.kind, s)) return EmitIntStore(e, o, s);
    switch (o.kind) {
    case K::Lmw: return EmitLmw(e, o);
    case K::Stmw: return EmitStmw(e, o);
    case K::Lfs: return EmitLfs(e, o, false, false);
    case K::Lfsu: return EmitLfs(e, o, true, false);
    case K::Lfsx: return EmitLfs(e, o, false, true);
    case K::Lfd: return EmitLfd(e, o, false);
    case K::Lfdu: return EmitLfd(e, o, true);
    case K::Stfs: return EmitStf(e, o, true, false, false);
    case K::Stfsu: return EmitStf(e, o, true, true, false);
    case K::Stfsx: return EmitStf(e, o, true, false, true);
    case K::Stfd: return EmitStf(e, o, false, false, false);
    case K::Stfdu: return EmitStf(e, o, false, true, false);
    case K::PsqL: return EmitPsqL(e, o, false);
    case K::PsqLu: return EmitPsqL(e, o, true);
    case K::PsqSt: return EmitPsqSt(e, o, false);
    case K::PsqStu: return EmitPsqSt(e, o, true);
    default: return 0;
    }
}

}  // namespace

uint32_t EmitMem(BlockEmitter& e, uint32_t index) { return EmitMemOp(e, index); }

// The loads and stores Decode leaves Generic: rA = 0 (an absolute address), the update, byte-reversed, algebraic and float forms
// without a kind of their own, lwarx, stwcx. and stfiwx. Each does what Exec does for the word; 0 for any other word and for a
// form Exec handles in a way that is not worth native code (an update form with rA = 0).
uint32_t EmitMemGeneric(BlockEmitter& e, uint32_t w) {
    if (!e.memPlan.built) BuildPlan(e);
    const uint32_t po = w >> 26, d = (w >> 21) & 31, a = (w >> 16) & 31, b = (w >> 11) & 31, xo = (w >> 1) & 0x3FF;
    Op o{K::Generic, static_cast<uint8_t>(d), static_cast<uint8_t>(a), static_cast<uint8_t>(b),
         static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(w & 0xFFFFu)))};
    Shape s{};
    enum class Kind { None, IntLoad, IntStore, Lfs, Lfd, Stfs, Stfd } kind = Kind::None;
    bool stfiwx = false, stwcx = false, update = false, indexed = false;
    if (po == 31) {
        indexed = true;
        switch (xo) {
        case 20: case 23: s = {4, false, true, false}; kind = Kind::IntLoad; break;       // lwarx, lwzx
        case 55: s = {4, true, true, false}; kind = Kind::IntLoad; break;                 // lwzux
        case 87: s = {1, false, true, false}; kind = Kind::IntLoad; break;                // lbzx
        case 119: s = {1, true, true, false}; kind = Kind::IntLoad; break;                // lbzux
        case 279: s = {2, false, true, false}; kind = Kind::IntLoad; break;               // lhzx
        case 311: s = {2, true, true, false}; kind = Kind::IntLoad; break;                // lhzux
        case 343: s = {2, false, true, true}; kind = Kind::IntLoad; break;                // lhax
        case 375: s = {2, true, true, true}; kind = Kind::IntLoad; break;                 // lhaux
        case 534: s = {4, false, true, false, false, true}; kind = Kind::IntLoad; break;  // lwbrx
        case 790: s = {2, false, true, false, false, true}; kind = Kind::IntLoad; break;  // lhbrx
        case 151: s = {4, false, true, false}; kind = Kind::IntStore; break;              // stwx
        case 183: s = {4, true, true, false}; kind = Kind::IntStore; break;               // stwux
        case 215: s = {1, false, true, false}; kind = Kind::IntStore; break;              // stbx
        case 247: s = {1, true, true, false}; kind = Kind::IntStore; break;               // stbux
        case 407: s = {2, false, true, false}; kind = Kind::IntStore; break;              // sthx
        case 439: s = {2, true, true, false}; kind = Kind::IntStore; break;               // sthux
        case 662: s = {4, false, true, false, false, true}; kind = Kind::IntStore; break; // stwbrx
        case 918: s = {2, false, true, false, false, true}; kind = Kind::IntStore; break; // sthbrx
        case 150: s = {4, false, true, false}; kind = Kind::IntStore; stwcx = true; break;  // stwcx.
        case 983: s = {4, false, true, false}; kind = Kind::IntStore; stfiwx = true; break;
        case 535: kind = Kind::Lfs; break;
        case 567: kind = Kind::Lfs; update = true; break;
        case 599: kind = Kind::Lfd; break;
        case 631: kind = Kind::Lfd; update = true; break;
        case 663: kind = Kind::Stfs; break;
        case 695: kind = Kind::Stfs; update = true; break;
        case 727: kind = Kind::Stfd; break;
        case 759: kind = Kind::Stfd; update = true; break;
        default: return 0;
        }
    } else {
        switch (po) {
        case 32: s = {4, false, false, false}; kind = Kind::IntLoad; break;
        case 33: s = {4, true, false, false}; kind = Kind::IntLoad; break;
        case 34: s = {1, false, false, false}; kind = Kind::IntLoad; break;
        case 35: s = {1, true, false, false}; kind = Kind::IntLoad; break;
        case 40: s = {2, false, false, false}; kind = Kind::IntLoad; break;
        case 41: s = {2, true, false, false}; kind = Kind::IntLoad; break;
        case 42: s = {2, false, false, true}; kind = Kind::IntLoad; break;
        case 43: s = {2, true, false, true}; kind = Kind::IntLoad; break;  // lhau
        case 36: s = {4, false, false, false}; kind = Kind::IntStore; break;
        case 37: s = {4, true, false, false}; kind = Kind::IntStore; break;
        case 38: s = {1, false, false, false}; kind = Kind::IntStore; break;
        case 39: s = {1, true, false, false}; kind = Kind::IntStore; break;
        case 44: s = {2, false, false, false}; kind = Kind::IntStore; break;
        case 45: s = {2, true, false, false}; kind = Kind::IntStore; break;
        case 48: kind = Kind::Lfs; break;
        case 49: kind = Kind::Lfs; update = true; break;
        case 50: kind = Kind::Lfd; break;
        case 51: kind = Kind::Lfd; update = true; break;
        case 52: kind = Kind::Stfs; break;
        case 53: kind = Kind::Stfs; update = true; break;
        case 54: kind = Kind::Stfd; break;
        case 55: kind = Kind::Stfd; update = true; break;
        default: return 0;
        }
    }
    if (kind == Kind::IntLoad || kind == Kind::IntStore) update = s.update;
    if (a == 0) {
        if (update) return 0;  // Exec writes r0 with the address: not worth a native form
        s.zeroBase = true;
    }
    switch (kind) {
    case Kind::IntLoad: return EmitIntLoad(e, o, s);
    case Kind::IntStore: {
        const uint32_t used = EmitIntStore(e, o, s, stfiwx);
        if (stwcx) {  // the store always succeeds: CR0 = EQ, and XER[SO] copied (the values of nibble 2 and 3)
            Xbyak::CodeGenerator& c = e.a;
            c.mov(rax, static_cast<uint64_t>(PpcCrvEncode(0x2u)));
            c.mov(rcx, static_cast<uint64_t>(PpcCrvEncode(0x3u)));
            c.cmp(e.Xer(), 0);
            c.cmovs(rax, rcx);
            c.mov(e.Crv(0), rax);
        }
        return used;
    }
    case Kind::Lfs: return EmitLfs(e, o, update, indexed, a == 0);
    case Kind::Lfd: return EmitLfd(e, o, update, indexed, a == 0);
    case Kind::Stfs: return EmitStf(e, o, true, update, indexed, a == 0);
    case Kind::Stfd: return EmitStf(e, o, false, update, indexed, a == 0);
    case Kind::None: break;
    }
    return 0;
}

}  // namespace PpcInterpreter::detail::x64

namespace PpcInterpreter::detail {
JitX64StoreGroupStats JitX64GetStoreGroupStats() {
    const auto& g = x64::g_groups;
    return JitX64StoreGroupStats{g.leaders.load(std::memory_order_relaxed), g.members.load(std::memory_order_relaxed),
                                 g.rangeBack, g.slowGoOn.load(std::memory_order_relaxed),
                                 g.slowLeave.load(std::memory_order_relaxed)};
}
JitX64FaultStats JitX64GetFaultStats() {
    return JitX64FaultStats{x64::g_faultSites.load(std::memory_order_relaxed), x64::g_faults.load(std::memory_order_relaxed),
                            x64::g_faultRepeats.load(std::memory_order_relaxed), x64::g_faultChunks.load(std::memory_order_relaxed)};
}
}  // namespace PpcInterpreter::detail
