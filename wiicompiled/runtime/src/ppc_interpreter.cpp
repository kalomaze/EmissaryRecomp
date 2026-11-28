// PowerPC (Gekko/Broadway) interpreter for guest code the static translation does not have: code a mod puts in
// memory (Gecko C0/C2 inserts, Syriinge plugins, replaced .rel modules) and any address the translation missed.
// A call to such an address (InvokeIndirectCpu, InvokeDirectCpu, InvokeIndirectJump) runs here instead of failing:
// the function is interpreted until it returns to its caller (blr to the link address it was entered with). Its calls
// into translated, native or patched code go through InvokeIndirectCpu; calls into other untranslated code continue
// in the same loop, as on the processor.
//
// Floating point uses the runtime's own helpers (ppc_isa_float.h, ppc_isa_quantized.h), so results match the
// translated code bit for bit. An FPR holds either a scalar double or a packed pair (PPC_FPR); the translator picks
// one per use, and here each register remembers which form its last write left and is converted at a use of the
// other form, the same PPC_PsFromScalarInline / PPC_PsToScalarInline conversions translated code makes.
#include "ppc_interpreter.h"
#include "ppc_interpreter_internal.h"
#include "ppc_interpreter_diag.h"
#include "brawl_online_core.h"
#include "brawl_rollback.h"
#include "brawl_dirty_memory.h"
#include "brawl_online_ipc.h"

// Guest comparisons must preserve unordered results. P+ disables Dreamland's
// wind with infinite timer bounds: their subtraction produces a NaN, which
// must not compare equal to zero and start the hazard.
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "The PowerPC interpreter requires strict floating-point semantics"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "abi_bridge.h"
#include "brawl_write_watch.h"
#include "guest_code_patch.h"
#include "hz_perf.h"
#include "guest_float_abi.h"
#include "memory.h"
#include "memory_access.h"
#include "ppc_runtime.h"
#include "recomp_mod_loader.h"
#include "runtime_log.h"
#include "system_bridge.h"

#if MKW_INTERP_PROFILE
#include <optional>

#include <pthread.h>

#include "save_states.h"
#endif

namespace PpcInterpreter {
using namespace detail;
namespace {

constexpr uint32_t kXerSo = 0x80000000u;
constexpr uint32_t kXerOv = 0x40000000u;
constexpr uint32_t kXerCa = 0x20000000u;

std::atomic<bool> g_enabled{true};
// Interpreted instructions, for the per-frame log. Each thread adds to its own counter (only the owner writes, so a
// plain add) and InstructionCount sums them: a shared atomic add on every return from Run was measurable.
struct InstructionCounter { std::atomic<uint64_t> count{0}; };
std::mutex g_countersMutex;
std::vector<InstructionCounter*> g_counters;  // never freed: a thread's count outlives it
thread_local InstructionCounter* t_counter = nullptr;
// A thread's first count: its counter is made and registered (out of line: the common call is the two lines below it).
MKW_PPC_NO_INLINE InstructionCounter* NewCounter() {
    InstructionCounter* const c = t_counter = new InstructionCounter;
    std::lock_guard<std::mutex> lock(g_countersMutex);
    g_counters.push_back(c);
    return c;
}
inline void CountInstructions(uint64_t n) {
    InstructionCounter* c = t_counter;
    if (c == nullptr) c = NewCounter();
    c->count.store(c->count.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

// The last addresses interpreted code jumped to on this thread (taken branches, calls, returns), oldest first
// once wrapped: the trail behind a fault in a mod's code.
constexpr uint32_t kRecentBranches = 64;
thread_local uint32_t t_recentBranches[kRecentBranches];
thread_local uint32_t t_recentBranchCount = 0;
thread_local uint32_t t_lastPc = 0;
// The interpreter frame running the rollback-driven main loop body (BrawlRollback::RunLoop), whose calls
// BrawlRollback::LoopCall sees (and, during resimulation, may skip or run inline). Only the game thread sets it; other
// threads' frames never compare equal. Not a thread_local: Call() reads it on every interpreted call, and a
// thread_local access is not a plain load on every target (emulated TLS makes it a runtime call).
std::atomic<const Frame*> g_rollbackLoopFrame{nullptr};

#if MKW_INTERP_PROFILE
// MKW_INTERP_PROFILE=1 builds only (a diagnostic; product builds compile none of this). With BRAWL_INTERP_PROFILE=<prefix>
// set, every interpreter entry address gets its calls, guest instructions and exclusive host ticks (rdtsc while the
// interpreter runs it, the translated, native and patched code it calls excluded), and every guest PC its executed
// instructions. ProfileDump writes them to <prefix>.<n>.<label>.txt and starts the next segment (pad scripts dump at
// each scene change and at EXIT; BRAWL_INTERP_PROFILE_FRAMES=N also dumps every N frames).
namespace prof {
struct Entry {
    uint64_t calls = 0, instructions = 0, ticks = 0;
    uint32_t firstCaller = 0;
};
constexpr uint32_t kMem1Words = 0x01800000u / 4, kMem2Words = Memory::kMem2Size / 4;  // 0x80000000.., 0x90000000..
struct Table {
    std::mutex mutex;  // guards `entries`' structure: inserts here, iteration by the dump
    std::unordered_map<uint32_t, Entry> entries;
    std::vector<uint32_t> pcs = std::vector<uint32_t>(kMem1Words + kMem2Words);
    uint64_t otherInstructions = 0;
    std::string thread;
    struct Slot {
        uint32_t entry = 0;
        Entry* stat = nullptr;
    } cache[4096];
};
std::mutex g_tablesMutex;
std::vector<Table*> g_tables;
const char* Prefix() {
    static const char* const prefix = std::getenv("BRAWL_INTERP_PROFILE");
    return prefix;
}
thread_local Table* t_table = nullptr;
thread_local Entry* t_running = nullptr;
thread_local uint64_t t_since = 0;
inline uint64_t Ticks() { return __builtin_ia32_rdtsc(); }
Table* MyTable() {
    if (t_table == nullptr) {
        t_table = new Table;
        char name[32] = "?";
        pthread_getname_np(pthread_self(), name, sizeof name);
        t_table->thread = name;
        std::lock_guard<std::mutex> lock(g_tablesMutex);
        g_tables.push_back(t_table);
    }
    return t_table;
}
Entry* Find(Table* t, uint32_t entry, uint32_t caller) {
    auto& slot = t->cache[(entry >> 2) & 4095];
    if (slot.stat != nullptr && slot.entry == entry) return slot.stat;
    std::lock_guard<std::mutex> lock(t->mutex);
    auto [it, fresh] = t->entries.try_emplace(entry);
    if (fresh) it->second.firstCaller = caller;
    slot = {entry, &it->second};
    return &it->second;
}
inline void Switch(Entry* next) {
    const uint64_t now = Ticks();
    if (t_running != nullptr) t_running->ticks += now - t_since;
    t_running = next;
    t_since = now;
}
inline void CountPc(Table* t, uint32_t pc) {
    const uint32_t mem1 = (pc - 0x80000000u) >> 2, mem2 = (pc - 0x90000000u) >> 2;
    if (mem1 < kMem1Words) ++t->pcs[mem1];
    else if (mem2 < kMem2Words) ++t->pcs[kMem1Words + mem2];
    else ++t->otherInstructions;
}
// One interpreter frame: its entry's clock runs while it lives (paused by Pause for the code it calls out to).
struct Scope {
    Table* table;
    Entry* previous;
    Entry* mine;
    const uint64_t& executed;
    Scope(uint32_t entry, uint32_t caller, const uint64_t& count)
        : table(MyTable()), previous(t_running), mine(Find(table, entry, caller)), executed(count) {
        ++mine->calls;
        Switch(mine);
    }
    ~Scope() {
        mine->instructions += executed;
        Switch(previous);
    }
};
struct Pause {
    Entry* previous;
    Pause() : previous(t_running) { Switch(nullptr); }
    ~Pause() { Switch(previous); }
};
uint64_t g_frame = 0, g_segmentStart = 0;
uint32_t g_dumps = 0;
}  // namespace prof
#define MKW_INTERP_PROFILE_PAUSE() prof::Pause interpProfilePause
#else
#define MKW_INTERP_PROFILE_PAUSE() ((void)0)
#endif

// BRAWL_TRACE_PC=<hex>,<hex>...: a debugging aid for mods; the registers whenever interpreted code reaches one of
// those addresses (the first 20 times each).
const std::vector<uint32_t>& TracePcs() {
    static const std::vector<uint32_t> pcs = [] {
        std::vector<uint32_t> out;
        for (const char* p = std::getenv("BRAWL_TRACE_PC"); p != nullptr && *p;) {
            char* end = nullptr;
            out.push_back(static_cast<uint32_t>(std::strtoul(p, &end, 16)));
            if (end == p) break;
            p = *end ? end + 1 : end;
        }
        return out;
    }();
    return pcs;
}

void TracePc(const CpuContext* c, uint32_t pc) {
    static std::unordered_map<uint64_t, int> hits;
    static std::mutex mutex;
    // BRAWL_TRACE_UNIQUE=1: once per (pc, lr, r4) instead of the first 20 hits (e.g. every character a print
    // function is asked for, whenever it happens).
    static const bool unique = std::getenv("BRAWL_TRACE_UNIQUE") != nullptr;
    std::lock_guard<std::mutex> lock(mutex);
    if (unique ? hits[(uint64_t{pc} << 32) ^ (uint64_t{c->lr} << 16) ^ c->gpr[4]]++ > 0 : hits[pc]++ >= 20) return;
    std::string regs;
    char item[24];
    for (int r = 0; r < 32; ++r) {
        std::snprintf(item, sizeof(item), " r%d=%08X", r, c->gpr[r]);
        regs += item;
    }
    RT_LOGF(RT_TAG_RUNTIME, "[trace] 0x%08X lr=%08X ctr=%08X%s\n", pc, c->lr, c->ctr, regs.c_str());
}

inline uint32_t Bits(uint32_t w, int from, int count) { return (w >> from) & ((1u << count) - 1u); }
inline uint32_t RD(uint32_t w) { return Bits(w, 21, 5); }
inline uint32_t RA(uint32_t w) { return Bits(w, 16, 5); }
inline uint32_t RB(uint32_t w) { return Bits(w, 11, 5); }
inline uint32_t RC(uint32_t w) { return Bits(w, 6, 5); }
inline int32_t Simm(uint32_t w) { return static_cast<int16_t>(w & 0xFFFFu); }
inline uint32_t Uimm(uint32_t w) { return w & 0xFFFFu; }
inline bool Rc(uint32_t w) { return (w & 1u) != 0; }

inline uint32_t Mask(uint32_t mb, uint32_t me) {
    const uint32_t a = 0xFFFFFFFFu >> mb;
    const uint32_t b = me == 31 ? 0xFFFFFFFFu : ~(0xFFFFFFFFu >> (me + 1));
    return mb <= me ? (a & b) : (a | b);
}

// --- integer helpers -------------------------------------------------------------------------------------------

void SetCr0(CpuContext* c, uint32_t value) {
    c->crv[0] = PpcCrvCompare(static_cast<int32_t>(value), int32_t{0}, c->xer);
}
void SetCr1FromFpscr(CpuContext* c) {
    c->crv[1] = PpcCrvEncode(c->fpscr >> 28);
}
void SetCa(CpuContext* c, bool carry) { c->xer = carry ? (c->xer | kXerCa) : (c->xer & ~kXerCa); }
bool Ca(const CpuContext* c) { return (c->xer & kXerCa) != 0; }
void SetOv(CpuContext* c, bool overflow) {
    c->xer = overflow ? (c->xer | kXerOv | kXerSo) : (c->xer & ~kXerOv);
}
uint32_t Ea(CpuContext* c, uint32_t w) { return (RA(w) ? c->gpr[RA(w)] : 0) + static_cast<uint32_t>(Simm(w)); }
uint32_t EaX(CpuContext* c, uint32_t w) { return (RA(w) ? c->gpr[RA(w)] : 0) + c->gpr[RB(w)]; }

// add with carry in, carry out and overflow: the XO-form arithmetic core.
uint32_t AddCarry(uint32_t a, uint32_t b, uint32_t carryIn, bool& carry, bool& overflow) {
    const uint64_t wide = static_cast<uint64_t>(a) + b + carryIn;
    const uint32_t result = static_cast<uint32_t>(wide);
    carry = (wide >> 32) != 0;
    overflow = ((~(a ^ b) & (a ^ result)) & 0x80000000u) != 0;
    return result;
}

[[noreturn]] void Undefined(CpuContext* c, uint32_t pc, uint32_t w, const char* what) {
    DumpRecentBranches();
    SystemBridge::DumpCpuState(c);  // where the interpreted code came from: lr, the stack's return addresses
    PPC_Undefined(pc, w, what);
}

}  // namespace

namespace detail {

uint32_t ReadSpr(CpuContext* c, uint32_t spr, uint32_t pc, uint32_t w) {
    switch (spr) {
    case 1: return c->xer;
    case 8: return c->lr;
    case 9: return c->ctr;
    case 26: return c->srr0;
    case 27: return c->srr1;
    case 268: return PPC_Mftb();
    case 269: return PPC_Mftbu();
    case 912: case 913: case 914: case 915: case 916: case 917: case 918: case 919: return c->gqr[spr - 912];
    case 920: return c->hid2;
    case 1008: return c->hid0;
    case 1009: return c->hid1;
    default:
        RT_LOGF(RT_TAG_RUNTIME, "interpreter: mfspr %u at 0x%08X reads 0\n", spr, pc);
        (void)w;
        return 0;
    }
}

void WriteSpr(CpuContext* c, uint32_t spr, uint32_t value) {
    switch (spr) {
    case 1: c->xer = value; break;
    case 8: c->lr = value; break;
    case 9: c->ctr = value; break;
    case 26: c->srr0 = value; break;
    case 27: c->srr1 = value; break;
    case 912: case 913: case 914: case 915: case 916: case 917: case 918: case 919: c->gqr[spr - 912] = value; break;
    case 920: c->hid2 = value; break;
    case 1008: c->hid0 = value; break;
    case 1009: c->hid1 = value; break;
    default: break;  // decrementer, performance monitors, BATs: no effect in the runtime
    }
}

}  // namespace detail

namespace {

// Match translated callers: f1 is the scalar floating-point return; retain the
// other registers' representation tags across calls. ABI volatility alone does
// not change their representation. P+ keeps a scalar multiplier in f3 across a
// posture-scale call, then consumes it with ps_muls0. Forgetting its scalar tag
// interprets double 1.0 as packed float 1.875 and changes jump velocity.
void NormalizeCallFloatReturn(Frame& f) {
    f.pairedMask &= ~2u;
}

// The float ABI translated callers assume of a translated callee (guest_float_abi.h); null for anything without an
// entry (natives, mod overrides, interpreted code), which then gets no conversion, as from a translated caller that
// knows no ABI for it.
const MkwGuestFloatAbi* FloatAbiSearch(uint32_t target) {
    const MkwGuestFloatAbi* begin = g_mkwGuestFloatAbi;
    const MkwGuestFloatAbi* end = begin + g_mkwGuestFloatAbiCount;
    const MkwGuestFloatAbi* it = std::lower_bound(begin, end, target,
        [](const MkwGuestFloatAbi& entry, uint32_t address) { return entry.address < address; });
    return it != end && it->address == target ? it : nullptr;
}

// The table is fixed at build time, so the last answers are kept per thread (a direct-mapped memo, as
// TranslatedFunctionRegistry::FindByAddressPtr keeps): every interpreted call asks, mostly for the same few targets.
const MkwGuestFloatAbi* FloatAbiOf(uint32_t target) {
    struct Memo { uint32_t target; const MkwGuestFloatAbi* abi; };
    constexpr uint32_t kMemo = 4096;  // calls reach a few thousand targets: a smaller memo kept missing
    thread_local Memo* memo = nullptr;
    if (memo == nullptr) memo = new Memo[kMemo]();
    Memo& m = memo[((target >> 2) ^ (target >> 14)) & (kMemo - 1u)];
    if (m.target != target || target == 0) m = Memo{target, FloatAbiSearch(target)};
    return m.abi;
}

// Before a call into translated code, as a translated caller does (EmitNormalizeGuestCallFloatArguments): a register the
// callee reads as a scalar float, held here in paired form, becomes the scalar it stands for. Registers the callee does
// not read that way keep their paired form (P+ code keeps paired values in volatile FPRs across calls).
void NormalizeCallFloatArguments(Frame& f, const MkwGuestFloatAbi* abi) {
    if (abi == nullptr) return;
    uint32_t m = f.pairedMask & (abi->scalarFloatArgMask & 0x3FFEu);
    while (m != 0) {
        const uint32_t r = static_cast<uint32_t>(__builtin_ctz(m));
        m &= m - 1u;
        SetScalar(f, r, Scalar(f, r));
    }
}

// After it (EmitNormalizeGuestCallFloatReturn): a callee that returns a paired scalar leaves f1 in paired form.
void NormalizeCalleeFloatReturn(Frame& f, const MkwGuestFloatAbi* abi) {
    if (abi != nullptr && abi->returnsPairedScalarFloat != 0) f.ctx->fpr[1].d = PPC_PsToScalarInline(f.ctx->fpr[1].d);
}

// A bl's target resolved (see CallSite).
inline CallSite ResolveCall(uint32_t target) {
    return CallSite{TranslatedFunctionRegistry::FindByAddressPtr(target),
                    TranslatedFunctionRegistry::FindRawByAddressPtr(target), FloatAbiSearch(target)};
}

}  // namespace

namespace detail {

// A call (bl, bcl, blrl, bctrl). Translated, native and patched code runs through the function registry; other code
// (a mod's) returns false and is interpreted right here, as the processor would run it: code that does not return the
// usual way (bl to read its own address, a hook that branches back) must not leave a host call frame behind.
// site: the call's target resolved at decode, if it was.
bool Call(Frame& f, uint32_t target, uint32_t returnTo, const CallSite* site) {
    f.ctx->lr = returnTo;
    const bool loopCall = &f == g_rollbackLoopFrame.load(std::memory_order_relaxed);
    if (loopCall) {
        switch (BrawlRollback::LoopCall(f.ctx, target)) {
        case BrawlRollback::LoopCallAction::Inline: return false;  // interpreted in this frame: its calls come here too
        case BrawlRollback::LoopCallAction::Skipped: NormalizeCallFloatReturn(f); return true;
        case BrawlRollback::LoopCallAction::Invoke: break;
        }
    }
    const bool translated =
        site != nullptr ? site->info != nullptr : TranslatedFunctionRegistry::FindByAddressPtr(target) != nullptr;
    if (!translated && CanRun(target)) return false;
    const MkwGuestFloatAbi* floatAbi = site != nullptr ? site->abi : FloatAbiOf(target);
    NormalizeCallFloatArguments(f, floatAbi);
    HZ_EXCLUSIVE_PAUSE();
    MKW_INTERP_PROFILE_PAUSE();
    if (target == 0x80017618u)
        BrawlRollback::Run(f.ctx, [](CpuContext* ctx) { InvokeIndirectCpu(0x80017618u, ctx); });
    else if (target == 0x8002A210u)
        BrawlRollback::UpdateSystem(f.ctx, [](CpuContext* ctx) { InvokeIndirectCpu(0x8002A210u, ctx); });
    else {
        // InvokeIndirectCpu tries the raw record first too
        if (site == nullptr || !TryDispatchRawCpuTarget(site->raw, f.ctx)) InvokeIndirectCpu(target, f.ctx);
        if (target == 0x80029438u) BrawlRollback::AnchorPads();
    }
    if (loopCall) BrawlRollback::LoopCallDone(target);
    NormalizeCalleeFloatReturn(f, floatAbi);
    NormalizeCallFloatReturn(f);
    return true;
}

// A tail call (b or bctr to a translated function): the callee returns to lr. True when that is this frame's own
// return address, so the frame is done; false when the tail caller was code interpreted inside this frame (a mod's
// function called from here), which then continues at lr. Returning from the frame there would abandon the rest of
// the function the frame was entered for.
bool TailCall(Frame& f, uint32_t target, uint32_t returnAddress) {
    const uint32_t back = f.ctx->lr & ~3u;
    const MkwGuestFloatAbi* floatAbi = FloatAbiOf(target);
    NormalizeCallFloatArguments(f, floatAbi);
    {
        HZ_EXCLUSIVE_PAUSE();
        MKW_INTERP_PROFILE_PAUSE();
        InvokeIndirectCpu(target, f.ctx);
    }
    if (&f == g_rollbackLoopFrame.load(std::memory_order_relaxed)) BrawlRollback::LoopCallDone(target);
    NormalizeCalleeFloatReturn(f, floatAbi);
    NormalizeCallFloatReturn(f);
    f.ctx->lr = back;
    return back == (returnAddress & ~3u);
}

// A tail call (b/bctr to a translated function) made in the rollback loop frame, as BrawlRollback::LoopCall decides
// (Call() does the same for calls): true when handled here, with `next` the pc to continue at (inline: the callee,
// interpreted in this frame; skipped: the caller's return address, lr). False: make the tail call as usual.
bool LoopTailCall(Frame& f, uint32_t target, uint32_t& next) {
    if (&f != g_rollbackLoopFrame.load(std::memory_order_relaxed)) return false;
    switch (BrawlRollback::LoopCall(f.ctx, target)) {
    case BrawlRollback::LoopCallAction::Inline: next = target; return true;
    case BrawlRollback::LoopCallAction::Skipped: NormalizeCallFloatReturn(f); next = f.ctx->lr & ~3u; return true;
    case BrawlRollback::LoopCallAction::Invoke: return false;
    }
    return false;
}

// TailCall into a code variant entered inside a function (GuestCodePatch::ContinuationAt): a plugin loader's "call the
// original" stub runs the replaced function's first instruction and branches to the second.
bool TailCallEntry(Frame& f, void (*entry)(CpuContext*), uint32_t returnAddress) {
    const uint32_t back = f.ctx->lr & ~3u;
    {
        HZ_EXCLUSIVE_PAUSE();
        MKW_INTERP_PROFILE_PAUSE();
        entry(f.ctx);
    }
    NormalizeCallFloatReturn(f);
    f.ctx->lr = back;
    return back == (returnAddress & ~3u);
}


// --- thin calls ---------------------------------------------------------------------------------------------------------------
// A call from compiled code to a translated function does what Call and TryDispatchRawCpuTarget do, with everything they look
// up decided once per target (JitCallTarget): the rest in this order, as they do it.

static PatchedHooks g_patchedHooks;

void RegisterPatchedHooks(const PatchedHooks& hooks) noexcept { g_patchedHooks = hooks; }
const PatchedHooks& GetPatchedHooks() noexcept { return g_patchedHooks; }

static const JitCallTarget* JitCallTargetForSlow(uint32_t target) noexcept;

const JitCallTarget* JitCallTargetFor(uint32_t target) noexcept {
    // The records never change once the registry is published: the last answers are kept per thread (a null answer too).
    struct Memo { uint32_t target; bool valid; const JitCallTarget* record; };
    constexpr uint32_t kMemo = 2048;
    thread_local Memo memo[kMemo];
    if (!TranslatedFunctionRegistry::IsLookupPublished()) return nullptr;
    Memo& m = memo[((target >> 2) * 0x9E3779B1u) >> 21];
    if (m.valid && m.target == target) return m.record;
    const JitCallTarget* const record = JitCallTargetForSlow(target);
    m = Memo{target, true, record};
    return record;
}

static const JitCallTarget* JitCallTargetForSlow(uint32_t target) noexcept {
    // One record per target, kept for the life of the process; a null entry remembers a target that has none.
    static std::mutex mutex;
    static auto* const table = new std::unordered_map<uint32_t, JitCallTarget*>();
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = table->find(target);
    if (found != table->end()) return found->second;
    JitCallTarget* record = nullptr;
    const TranslatedFunctionInfo* const info = TranslatedFunctionRegistry::FindByAddressPtr(target);
    const RawDispatchRecord* const raw = TranslatedFunctionRegistry::FindRawByAddressPtr(target);
    if (info != nullptr && raw != nullptr && raw->entry != nullptr) {
        record = new JitCallTarget{};
        record->target = target;
        record->address = raw->address;
        record->entry = raw->entry;
        record->patchedByte = g_patchedHooks.patchedByteFor != nullptr ? g_patchedHooks.patchedByteFor(target) : nullptr;
        record->fprMask = raw->nonvolatileFprWriteMask & kPpcAllNonvolatileFprMask;
        record->preserveGprs = raw->preserveNonvolatileGprs;
        if (const uint32_t m = record->fprMask; m != 0) {
            const uint32_t lowest = m & (0u - m), aboveRun = m + lowest;
            if ((aboveRun & (aboveRun - 1u)) == 0u) {  // one run of set bits
                const unsigned first = static_cast<unsigned>(__builtin_ctz(m));
                const unsigned end = aboveRun == 0u ? 32u : static_cast<unsigned>(__builtin_ctz(aboveRun));
                record->fprFirst = static_cast<uint8_t>(first - 14u);
                record->fprCount = static_cast<uint8_t>(end - first);
            }
        }
        record->special = target == 0x80017618u || target == 0x8002A210u || target == 0x80029438u;
        const MkwGuestFloatAbi* const abi = FloatAbiSearch(target);
        record->returnsPaired = abi != nullptr && abi->returnsPairedScalarFloat != 0;
        record->argMask = abi != nullptr ? (abi->scalarFloatArgMask & 0x3FFEu) : 0u;
        record->lean = record->fprMask == 0 && !record->preserveGprs;
        record->fprsOnly = record->fprMask != 0 && !record->preserveGprs;
        record->fastDirect = record->lean && !record->returnsPaired && !record->special;
    }
    (*table)[target] = record;
    return record;
}

namespace {

// NormalizeCallFloatArguments for a mask worked out beforehand (JitCallTarget::argMask, bit r for f1..f13).
inline void NormalizeArgMask(Frame& f, uint32_t mask) {
    uint32_t m = f.pairedMask & mask;
    while (m != 0) {
        const uint32_t r = static_cast<uint32_t>(__builtin_ctz(m));
        m &= m - 1u;
        SetScalar(f, r, Scalar(f, r));
    }
}

inline void NormalizeArgs(Frame& f, const JitCallTarget& t) {
    if ((f.pairedMask & t.argMask) != 0) NormalizeArgMask(f, t.argMask);
}

// The FPRs of `mask` (bits 14-31 as PpcNonvolatileFprGuard keeps them) to and from the record's slots.
inline void SaveFprs(const CpuContext* c, uint32_t mask, PPC_FPR* saved) {
    for (uint32_t m = mask; m != 0; m &= m - 1) {
        const unsigned r = static_cast<unsigned>(__builtin_ctz(m));
        saved[r - 14] = c->fpr[r];
    }
}
inline void RestoreFprs(CpuContext* c, uint32_t mask, const PPC_FPR* saved) {
    for (uint32_t m = mask; m != 0; m &= m - 1) {
        const unsigned r = static_cast<unsigned>(__builtin_ctz(m));
        c->fpr[r] = saved[r - 14];
    }
}

static_assert(sizeof(PPC_FPR) == 8);

// Registers first..31 (first 14-31) to and from a guard's slots (slot r - 14): a jump into a straight run of 8-byte moves. The
// translator measures most masks to such a run (f31 alone the most common, every non-volatile FPR for a callee it knows nothing
// better of and the natives). Eight-byte moves, as compiled and translated code store these registers: each load is forwarded
// from a store still in flight, which a wider load spanning two of them is not.
inline void SaveTopFprs(const CpuContext* c, unsigned first, PPC_FPR* saved) {
    switch (first) {
    case 14: saved[0] = c->fpr[14]; [[fallthrough]];
    case 15: saved[1] = c->fpr[15]; [[fallthrough]];
    case 16: saved[2] = c->fpr[16]; [[fallthrough]];
    case 17: saved[3] = c->fpr[17]; [[fallthrough]];
    case 18: saved[4] = c->fpr[18]; [[fallthrough]];
    case 19: saved[5] = c->fpr[19]; [[fallthrough]];
    case 20: saved[6] = c->fpr[20]; [[fallthrough]];
    case 21: saved[7] = c->fpr[21]; [[fallthrough]];
    case 22: saved[8] = c->fpr[22]; [[fallthrough]];
    case 23: saved[9] = c->fpr[23]; [[fallthrough]];
    case 24: saved[10] = c->fpr[24]; [[fallthrough]];
    case 25: saved[11] = c->fpr[25]; [[fallthrough]];
    case 26: saved[12] = c->fpr[26]; [[fallthrough]];
    case 27: saved[13] = c->fpr[27]; [[fallthrough]];
    case 28: saved[14] = c->fpr[28]; [[fallthrough]];
    case 29: saved[15] = c->fpr[29]; [[fallthrough]];
    case 30: saved[16] = c->fpr[30]; [[fallthrough]];
    case 31: saved[17] = c->fpr[31]; [[fallthrough]];
    default: break;
    }
}
inline void RestoreTopFprs(CpuContext* c, unsigned first, const PPC_FPR* saved) {
    switch (first) {
    case 14: c->fpr[14] = saved[0]; [[fallthrough]];
    case 15: c->fpr[15] = saved[1]; [[fallthrough]];
    case 16: c->fpr[16] = saved[2]; [[fallthrough]];
    case 17: c->fpr[17] = saved[3]; [[fallthrough]];
    case 18: c->fpr[18] = saved[4]; [[fallthrough]];
    case 19: c->fpr[19] = saved[5]; [[fallthrough]];
    case 20: c->fpr[20] = saved[6]; [[fallthrough]];
    case 21: c->fpr[21] = saved[7]; [[fallthrough]];
    case 22: c->fpr[22] = saved[8]; [[fallthrough]];
    case 23: c->fpr[23] = saved[9]; [[fallthrough]];
    case 24: c->fpr[24] = saved[10]; [[fallthrough]];
    case 25: c->fpr[25] = saved[11]; [[fallthrough]];
    case 26: c->fpr[26] = saved[12]; [[fallthrough]];
    case 27: c->fpr[27] = saved[13]; [[fallthrough]];
    case 28: c->fpr[28] = saved[14]; [[fallthrough]];
    case 29: c->fpr[29] = saved[15]; [[fallthrough]];
    case 30: c->fpr[30] = saved[16]; [[fallthrough]];
    case 31: c->fpr[31] = saved[17]; [[fallthrough]];
    default: break;
    }
}

// PpcNonvolatileFprGuard's work for the record `t` stands for, inline: the registers of its mask saved on construction and put
// back on destruction (also when the callee throws), nothing else touched. A run ending at f31 is a straight run of moves
// (SaveTopFprs), any other mask a loop over its registers.
struct ThinFprGuard {
    CpuContext* const cpu;
    const JitCallTarget& t;
    PPC_FPR saved[18];
    ThinFprGuard(CpuContext* c, const JitCallTarget& target) noexcept : cpu(c), t(target) {
        if (t.fprCount != 0 && t.fprFirst + t.fprCount == 18) SaveTopFprs(cpu, 14u + t.fprFirst, saved);
        else SaveFprs(cpu, t.fprMask, saved);
    }
    ~ThinFprGuard() {
        if (t.fprCount != 0 && t.fprFirst + t.fprCount == 18) RestoreTopFprs(cpu, 14u + t.fprFirst, saved);
        else RestoreFprs(cpu, t.fprMask, saved);
    }
    ThinFprGuard(const ThinFprGuard&) = delete;
    ThinFprGuard& operator=(const ThinFprGuard&) = delete;
};

// TryDispatchRawCpuTarget(record, cpu) for the record `t` stands for: the execution address scope, the guards, the context scope
// when the ambient context is another, the call. A callee that throws unwinds the same guards.
// The call of a target whose guards have nothing to save, made in another ambient context than the caller's: out of line, as it
// sets the host float mode and is rare.
MKW_PPC_NO_INLINE void EnterUnderContext(CpuContext* cpu, void (*entry)(CpuContext*)) {
    CpuContextScope scope(cpu);
    entry(cpu);
}

// ThinDispatch for a target with no guard to keep (JitCallTarget::lean): sets the diagnostic address without restoring
// (same as MkwInvokeIndirectCpuSite) and invokes the entry.
inline void ThinLean(CpuContext* cpu, const JitCallTarget& t) {
    if (t.address != 0) RecompMod::g_currentTranslatedExecutionAddress = t.address;
    if (TryGetCpuContext() != cpu) EnterUnderContext(cpu, t.entry);
    else t.entry(cpu);
}

// ThinDispatch for a target with a GPR guard (a native): the execution address scope, the guards, the context test.
MKW_PPC_NO_INLINE void ThinGuarded(CpuContext* cpu, const JitCallTarget& t) {
    RecompMod::ScopedTranslatedExecutionAddress translatedExecution(t.address);
    PpcNonvolatileGprGuard gprGuard(cpu, t.preserveGprs);
    if (t.fprMask == 0) {
        if (TryGetCpuContext() != cpu) {
            CpuContextScope scope(cpu);
            t.entry(cpu);
        } else {
            t.entry(cpu);
        }
        return;
    }
    ThinFprGuard fprGuard(cpu, t);
    if (TryGetCpuContext() != cpu) {
        CpuContextScope scope(cpu);
        t.entry(cpu);
    } else {
        t.entry(cpu);
    }
}

// ThinDispatch for a target with FPRs to keep and no GPR guard: sets diagnostic address without restoring,
// builds the FPR guard, and invokes the entry.
MKW_PPC_NO_INLINE void ThinFprs(CpuContext* cpu, const JitCallTarget& t) {
    if (t.address != 0) RecompMod::g_currentTranslatedExecutionAddress = t.address;
    ThinFprGuard fprGuard(cpu, t);
    if (TryGetCpuContext() != cpu) EnterUnderContext(cpu, t.entry);
    else t.entry(cpu);
}

inline void ThinDispatch(CpuContext* cpu, const JitCallTarget& t) {
    if (t.lean) ThinLean(cpu, t);
    else if (t.fprsOnly) ThinFprs(cpu, t);
    else ThinGuarded(cpu, t);
}

inline void ThinAfterCall(Frame& f, const JitCallTarget& t) {
    if (t.returnsPaired) f.ctx->fpr[1].d = PPC_PsToScalarInline(f.ctx->fpr[1].d);  // NormalizeCalleeFloatReturn
    f.pairedMask &= ~2u;                                                          // NormalizeCallFloatReturn
}

template <bool Normalize>
inline uint64_t ThinCallImpl(Frame* f, const JitCallTarget* t) noexcept {
    try {
        if (Normalize) NormalizeArgs(*f, *t);
        ThinDispatch(f->ctx, *t);
        ThinAfterCall(*f, *t);
        return 0;
    } catch (...) {
        f->pending = std::current_exception();
        return uint64_t{3} << 32;
    }
}

template <bool Normalize>
inline uint64_t ThinTailImpl(Frame* f, const JitCallTarget* t) noexcept {
    try {
        CpuContext* const c = f->ctx;
        const uint32_t back = c->lr & ~3u;  // TailCall
        if (Normalize) NormalizeArgs(*f, *t);
        ThinDispatch(c, *t);
        ThinAfterCall(*f, *t);
        c->lr = back;
        if (back == (f->returnAddress & ~3u)) return uint64_t{2} << 32;
        return (uint64_t{1} << 32) | back;
    } catch (...) {
        f->pending = std::current_exception();
        return uint64_t{3} << 32;
    }
}
}  // namespace

uint64_t JitThinCall(Frame* f, const JitCallTarget* t) noexcept { return ThinCallImpl<true>(f, t); }
uint64_t JitThinCallPrepared(Frame* f, const JitCallTarget* t) noexcept { return ThinCallImpl<false>(f, t); }
uint64_t JitThinTail(Frame* f, const JitCallTarget* t) noexcept { return ThinTailImpl<true>(f, t); }
uint64_t JitThinTailPrepared(Frame* f, const JitCallTarget* t) noexcept { return ThinTailImpl<false>(f, t); }
void JitNormalizeArgs(Frame* f, const JitCallTarget* t) noexcept { NormalizeArgMask(*f, t->argMask); }

namespace {
// Puts `t` at the front of the site's cell, the older entries moving down (the oldest falls out), unless it is there already (a call
// made in the rollback loop frame comes here without having probed).
inline void RememberTarget(JitTargetCell* cell, uint32_t target, const JitCallTarget* t, bool tail) noexcept {
    for (uint32_t i = 0; i < kCellWays; ++i)
        if (cell[i].target == target) return;
    for (uint32_t i = kCellWays - 1; i > 0; --i) cell[i] = cell[i - 1];
    cell[0].record = t;
    cell[0].argMask = static_cast<uint16_t>(t->argMask);  // bits 1-13
    cell[0].hits = 0;
    cell[0].wayHits = 0;
    cell[0].entry = t->entry;
    cell[0].fast = JitFastCallFor(t, tail);
    cell[0].target = target;
}
}  // namespace

#if !MKW_JIT
const void* JitFastCallFor(const JitCallTarget*, bool) noexcept { return nullptr; }
#endif

void JitDirectGuardedCall(CpuContext* cpu, const JitCallTarget* t) { ThinGuarded(cpu, *t); }

void JitDirectFprsCall(CpuContext* cpu, const JitCallTarget* t) {
    if (t->address != 0) RecompMod::g_currentTranslatedExecutionAddress = t->address;
    ThinFprGuard fprGuard(cpu, *t);
    if (__builtin_expect(TryGetCpuContext() != cpu, 0)) EnterUnderContext(cpu, t->entry);
    else t->entry(cpu);
}

uint64_t JitCallOpCached(Frame* f, const Block* b, uint32_t index, JitTargetCell* cell) noexcept {
    const uint32_t target = f->ctx->ctr & ~3u;
    const JitCallTarget* const t = JitCallTargetFor(target);
    if (t != nullptr && !t->special) {
        RememberTarget(cell, target, t, false);
        if (__builtin_expect((f->flags & kFrameFlagLoopFrame) == 0, 1)) {
            return ThinCallImpl<true>(f, t);
        }
    }
    return JitCallOp(f, b, index);
}

uint64_t JitTailOpCached(Frame* f, const Block* b, uint32_t index, JitTargetCell* cell) noexcept {
    const uint32_t target = f->ctx->ctr & ~3u;
    const JitCallTarget* const t = JitCallTargetFor(target);
    if (t != nullptr) {
        RememberTarget(cell, target, t, true);
        if (__builtin_expect(!t->special && (f->flags & kFrameFlagLoopFrame) == 0, 1)) {
            f->ctx->pc = b->pcs[index];
            return ThinTailImpl<true>(f, t);
        }
    }
    return JitTailOp(f, b, index);
}


// One guest instruction w at pc, with next = pc + 4 on entry: the pc to continue at afterwards. Return when the frame
// is done (a blr or tail call back to the address it was entered with).
Step Exec(Frame& f, uint32_t pc, uint32_t w, uint32_t& next, uint32_t returnAddress) {
    CpuContext* const c = f.ctx;
    switch (w >> 26) {
    case 3: break;  // twi: traps are debug asserts
    case 4: {  // paired singles
        const uint32_t xo10 = Bits(w, 1, 10);
        const uint32_t xo6 = Bits(w, 1, 6);
        const uint32_t xo5 = Bits(w, 1, 5);
        const uint32_t d = RD(w), a = RA(w), b = RB(w), cc = RC(w);
        if (xo6 == 6 || xo6 == 7 || xo6 == 38 || xo6 == 39) {  // psq_lx, psq_stx, psq_lux, psq_stux
            const uint32_t wBit = Bits(w, 10, 1), i = Bits(w, 7, 3);
            const uint32_t ea = ((xo6 & 32u) || a) ? c->gpr[a] + c->gpr[b] : c->gpr[b];
            if (xo6 == 6 || xo6 == 38) SetPaired(f, d, PPC_PsqL(ea, wBit, i));
            else PPC_PsqSt(ea, Paired(f, d), wBit, i);
            if (xo6 & 32u) c->gpr[a] = ea;
            break;
        }
        switch (xo10) {
        case 0: case 32: case 64: case 96: {  // ps_cmpu0, ps_cmpo0, ps_cmpu1, ps_cmpo1
            const double pa = Paired(f, a), pb = Paired(f, b);
            const bool lane1 = xo10 >= 64;
            const double x = lane1 ? PpcGetPs1Inline(pa) : PpcGetPs0Inline(pa);
            const double y = lane1 ? PpcGetPs1Inline(pb) : PpcGetPs0Inline(pb);
            c->crv[(static_cast<int>(d >> 2)) & 7] = PpcCrvCompareFloat(x, y);
            break;
        }
        case 40: SetPaired(f, d, PPC_PsNegInline(Paired(f, b))); break;
        case 72: c->fpr[d] = c->fpr[b]; CopyForm(f, d, b); break;
        case 136: SetPaired(f, d, PPC_PsNegInline(PPC_PsAbsInline(Paired(f, b)))); break;
        case 264: SetPaired(f, d, PPC_PsAbsInline(Paired(f, b))); break;
        case 528: SetPaired(f, d, PPC_PsMerge00Inline(Paired(f, a), Paired(f, b))); break;
        case 560: SetPaired(f, d, PPC_PsMerge01Inline(Paired(f, a), Paired(f, b))); break;
        case 592: SetPaired(f, d, PPC_PsMerge10Inline(Paired(f, a), Paired(f, b))); break;
        case 624: SetPaired(f, d, PPC_PsMerge11Inline(Paired(f, a), Paired(f, b))); break;
        case 1014: {  // dcbz_l
            const uint32_t ea = ((a ? c->gpr[a] : 0) + c->gpr[b]) & ~31u;
            for (uint32_t k = 0; k < 32; k += 4) MemoryInline::FlatWrite32(ea + k, 0);
            break;
        }
        default:
            switch (xo5) {
            case 10: SetPaired(f, d, PPC_PsSum0Inline(Paired(f, a), Paired(f, b), Paired(f, cc))); break;
            case 11: SetPaired(f, d, PPC_PsSum1Inline(Paired(f, a), Paired(f, b), Paired(f, cc))); break;
            case 12: SetPaired(f, d, PPC_PsMuls0Inline(Paired(f, a), Paired(f, cc))); break;
            case 13: SetPaired(f, d, PPC_PsMuls1Inline(Paired(f, a), Paired(f, cc))); break;
            case 14: SetPaired(f, d, PPC_PsMadds0Inline(Paired(f, a), Paired(f, cc), Paired(f, b))); break;
            case 15: SetPaired(f, d, PPC_PsMadds1Inline(Paired(f, a), Paired(f, cc), Paired(f, b))); break;
            case 18: SetPaired(f, d, PPC_PsDivInline(Paired(f, a), Paired(f, b))); break;
            case 20: SetPaired(f, d, PPC_PsSubInline(Paired(f, a), Paired(f, b))); break;
            case 21: SetPaired(f, d, PPC_PsAddInline(Paired(f, a), Paired(f, b))); break;
            case 23: SetPaired(f, d, PPC_PsSelInline(Paired(f, cc), Paired(f, a), Paired(f, b))); break;
            case 24: SetPaired(f, d, PPC_PsRes(Paired(f, b))); break;
            case 25: SetPaired(f, d, PPC_PsMulInline(Paired(f, a), Paired(f, cc))); break;
            case 26: SetPaired(f, d, PPC_PsRsqrte(Paired(f, b))); break;
            case 28: SetPaired(f, d, PPC_PsMsubInline(Paired(f, a), Paired(f, cc), Paired(f, b))); break;
            case 29: SetPaired(f, d, PPC_PsMaddInline(Paired(f, a), Paired(f, cc), Paired(f, b))); break;
            case 30: SetPaired(f, d, PPC_PsNmsubInline(Paired(f, a), Paired(f, cc), Paired(f, b))); break;
            case 31: SetPaired(f, d, PPC_PsNmaddInline(Paired(f, a), Paired(f, cc), Paired(f, b))); break;
            default: Undefined(c, pc, w, "interpreter: paired-single opcode");
            }
        }
        if (Rc(w) && xo10 != 1014) SetCr1FromFpscr(c);
        break;
    }
    case 7: c->gpr[RD(w)] = static_cast<uint32_t>(static_cast<int32_t>(c->gpr[RA(w)]) * Simm(w)); break;  // mulli
    case 8: {  // subfic
        bool carry, ov;
        c->gpr[RD(w)] = AddCarry(~c->gpr[RA(w)], static_cast<uint32_t>(Simm(w)), 1, carry, ov);
        SetCa(c, carry);
        break;
    }
    case 10: c->crv[(static_cast<int>(RD(w) >> 2)) & 7] = PpcCrvCompare(c->gpr[RA(w)], Uimm(w), c->xer); break;  // cmpli
    case 11:  // cmpi
        c->crv[(static_cast<int>(RD(w) >> 2)) & 7] = PpcCrvCompare(static_cast<int32_t>(c->gpr[RA(w)]), Simm(w), c->xer);
        break;
    case 12: case 13: {  // addic, addic.
        bool carry, ov;
        c->gpr[RD(w)] = AddCarry(c->gpr[RA(w)], static_cast<uint32_t>(Simm(w)), 0, carry, ov);
        SetCa(c, carry);
        if ((w >> 26) == 13) SetCr0(c, c->gpr[RD(w)]);
        break;
    }
    case 14: c->gpr[RD(w)] = (RA(w) ? c->gpr[RA(w)] : 0) + static_cast<uint32_t>(Simm(w)); break;  // addi
    case 15: c->gpr[RD(w)] = (RA(w) ? c->gpr[RA(w)] : 0) + (Uimm(w) << 16); break;  // addis
    case 16: {  // bc
        const int32_t disp = static_cast<int16_t>(w & 0xFFFCu);
        const uint32_t target = (w & 2u) ? static_cast<uint32_t>(disp) : pc + static_cast<uint32_t>(disp);
        if (BranchTaken(c, RD(w), RA(w))) {
            if (!(w & 1u) || !Call(f, target, pc + 4)) next = target;
        }
        break;
    }
    case 17: RT_LOGF(RT_TAG_RUNTIME, "interpreter: sc at 0x%08X ignored\n", pc); break;
    case 18: {  // b
        int32_t disp = static_cast<int32_t>(w & 0x03FFFFFCu);
        if (disp & 0x02000000) disp -= 0x04000000;
        uint32_t target = (w & 2u) ? static_cast<uint32_t>(disp) : pc + static_cast<uint32_t>(disp);
        // ba/bla to low addresses: mods (Project+) map virtual 0 onto MEM1 with an IBAT, so they reach MEM1 code.
        if (target < 0x01800000u) target |= 0x80000000u;
        if (w & 1u) { if (!Call(f, target, pc + 4)) next = target; break; }
        // A branch to another function's start is a tail call (a Gecko hook ending a function with b 0x800018A8).
        if (TranslatedFunctionRegistry::FindByAddressPtr(target) != nullptr) {
            if (LoopTailCall(f, target, next)) break;
            if (TailCall(f, target, returnAddress)) return Step::Return;
            next = c->lr & ~3u;
            break;
        }
        if (const auto continuation = GuestCodePatch::ContinuationAt(target)) {
            if (TailCallEntry(f, continuation, returnAddress)) return Step::Return;
            next = c->lr & ~3u;
            break;
        }
        next = target;
        break;
    }
    case 19: {
        const uint32_t xo = Bits(w, 1, 10);
        switch (xo) {
        case 0: c->crv[(RD(w) >> 2) & 7u] = c->crv[(RA(w) >> 2) & 7u]; break;  // mcrf
        case 16: {  // bclr
            const uint32_t target = c->lr & ~3u;
            if (BranchTaken(c, RD(w), RA(w))) {
                if (w & 1u) { if (!Call(f, target, pc + 4)) next = target; break; }
                if (target == (returnAddress & ~3u)) {
                    SetScalar(f, 1, Scalar(f, 1));  // the caller takes f1 as the scalar return value
                    return Step::Return;
                }
                next = target;  // a computed jump through lr inside the function
            }
            break;
        }
        case 33: case 129: case 193: case 225: case 257: case 289: case 417: case 449:  // crnor..cror
        {
            const uint32_t op = xo == 33 ? 0 : xo == 129 ? 1 : xo == 193 ? 2 : xo == 225 ? 3 :
                                xo == 257 ? 4 : xo == 289 ? 5 : xo == 417 ? 6 : 7;
            const uint32_t bt = RD(w), ba = RA(w), bb = RB(w);
            c->crv[bt >> 2] = PpcCrvWithBit(c->crv[bt >> 2], bt & 3u,
                                            PpcCrvLogic(op, PpcCrvBit(c->crv[ba >> 2], ba & 3u),
                                                        PpcCrvBit(c->crv[bb >> 2], bb & 3u)));
            break;
        }
        case 50: next = c->srr0; c->msr = c->srr1; break;  // rfi
        case 150: break;  // isync
        case 528: {  // bcctr
            const uint32_t target = c->ctr & ~3u;
            if (BranchTaken(c, RD(w) | 0x04u, RA(w))) {
                if (w & 1u) { if (!Call(f, target, pc + 4)) next = target; break; }
                // bctr: a switch-table jump inside this function, or a tail call to another function.
                if (TranslatedFunctionRegistry::FindByAddressPtr(target) != nullptr) {
                    if (LoopTailCall(f, target, next)) break;
                    if (TailCall(f, target, returnAddress)) return Step::Return;
                    next = c->lr & ~3u;
                    break;
                }
                next = target;
            }
            break;
        }
        default: Undefined(c, pc, w, "interpreter: opcode 19");
        }
        break;
    }
    case 20: {  // rlwimi
        const uint32_t m = Mask(Bits(w, 6, 5), Bits(w, 1, 5));
        const uint32_t r = PpcRotl32Inline(c->gpr[RD(w)], RB(w));
        c->gpr[RA(w)] = (r & m) | (c->gpr[RA(w)] & ~m);
        if (Rc(w)) SetCr0(c, c->gpr[RA(w)]);
        break;
    }
    case 21:  // rlwinm
        c->gpr[RA(w)] = PpcRotl32Inline(c->gpr[RD(w)], RB(w)) & Mask(Bits(w, 6, 5), Bits(w, 1, 5));
        if (Rc(w)) SetCr0(c, c->gpr[RA(w)]);
        break;
    case 23:  // rlwnm
        c->gpr[RA(w)] = PpcRotl32Inline(c->gpr[RD(w)], c->gpr[RB(w)] & 31u) & Mask(Bits(w, 6, 5), Bits(w, 1, 5));
        if (Rc(w)) SetCr0(c, c->gpr[RA(w)]);
        break;
    case 24: c->gpr[RA(w)] = c->gpr[RD(w)] | Uimm(w); break;
    case 25: c->gpr[RA(w)] = c->gpr[RD(w)] | (Uimm(w) << 16); break;
    case 26: c->gpr[RA(w)] = c->gpr[RD(w)] ^ Uimm(w); break;
    case 27: c->gpr[RA(w)] = c->gpr[RD(w)] ^ (Uimm(w) << 16); break;
    case 28: c->gpr[RA(w)] = c->gpr[RD(w)] & Uimm(w); SetCr0(c, c->gpr[RA(w)]); break;
    case 29: c->gpr[RA(w)] = c->gpr[RD(w)] & (Uimm(w) << 16); SetCr0(c, c->gpr[RA(w)]); break;
    case 31: {
        const uint32_t xo = Bits(w, 1, 10);
        const uint32_t d = RD(w), a = RA(w), b = RB(w);
        const bool oe = (w & 0x400u) != 0;
        bool carry = false, ov = false;
        uint32_t result = 0;
        bool writeD = false, writeA = false, carryOut = false;
        switch (xo & 0x1FFu) {  // XO-form arithmetic: OE is bit 10
        case 8: result = AddCarry(~c->gpr[a], c->gpr[b], 1, carry, ov); writeD = carryOut = true; break;    // subfc
        case 10: result = AddCarry(c->gpr[a], c->gpr[b], 0, carry, ov); writeD = carryOut = true; break;    // addc
        case 11: result = static_cast<uint32_t>((static_cast<uint64_t>(c->gpr[a]) * c->gpr[b]) >> 32); writeD = true; break;
        case 40: result = AddCarry(~c->gpr[a], c->gpr[b], 1, carry, ov); writeD = true; break;              // subf
        case 75: result = static_cast<uint32_t>((static_cast<int64_t>(static_cast<int32_t>(c->gpr[a])) *
                                                 static_cast<int32_t>(c->gpr[b])) >> 32); writeD = true; break;  // mulhw
        case 104: result = AddCarry(~c->gpr[a], 0, 1, carry, ov); writeD = true; break;                     // neg
        case 136: result = AddCarry(~c->gpr[a], c->gpr[b], Ca(c), carry, ov); writeD = carryOut = true; break;  // subfe
        case 138: result = AddCarry(c->gpr[a], c->gpr[b], Ca(c), carry, ov); writeD = carryOut = true; break;  // adde
        case 200: result = AddCarry(~c->gpr[a], 0, Ca(c), carry, ov); writeD = carryOut = true; break;         // subfze
        case 202: result = AddCarry(c->gpr[a], 0, Ca(c), carry, ov); writeD = carryOut = true; break;          // addze
        case 232: result = AddCarry(~c->gpr[a], 0xFFFFFFFFu, Ca(c), carry, ov); writeD = carryOut = true; break;  // subfme
        case 234: result = AddCarry(c->gpr[a], 0xFFFFFFFFu, Ca(c), carry, ov); writeD = carryOut = true; break;   // addme
        case 235: {  // mullw
            const int64_t wide = static_cast<int64_t>(static_cast<int32_t>(c->gpr[a])) * static_cast<int32_t>(c->gpr[b]);
            result = static_cast<uint32_t>(wide);
            ov = wide != static_cast<int32_t>(wide);
            writeD = true;
            break;
        }
        case 266: result = AddCarry(c->gpr[a], c->gpr[b], 0, carry, ov); writeD = true; break;  // add
        case 459: result = PPC_Divwu(c->gpr[a], c->gpr[b]); ov = c->gpr[b] == 0; writeD = true; break;  // divwu
        case 491:  // divw
            result = static_cast<uint32_t>(PPC_Divw(static_cast<int32_t>(c->gpr[a]), static_cast<int32_t>(c->gpr[b])));
            ov = c->gpr[b] == 0 || (c->gpr[a] == 0x80000000u && c->gpr[b] == 0xFFFFFFFFu);
            writeD = true;
            break;
        default: break;
        }
        if (writeD) {
            c->gpr[d] = result;
            if (carryOut) SetCa(c, carry);
            if (oe) SetOv(c, ov);
            if (Rc(w)) SetCr0(c, result);
            break;
        }
        switch (xo) {
        case 0: c->crv[(static_cast<int>(d >> 2)) & 7] = PpcCrvCompare(static_cast<int32_t>(c->gpr[a]), static_cast<int32_t>(c->gpr[b]), c->xer); break;
        case 32: c->crv[(static_cast<int>(d >> 2)) & 7] = PpcCrvCompare(c->gpr[a], c->gpr[b], c->xer); break;
        case 4: break;  // tw
        case 19: c->gpr[d] = PpcGetCr(c); break;  // mfcr
        case 20: case 23: c->gpr[d] = MemoryInline::FlatRead32(EaX(c, w)); break;  // lwarx, lwzx
        case 24: result = PPC_Slw(c->gpr[d], c->gpr[b]); writeA = true; break;
        case 26: result = PPC_CntlzwInline(c->gpr[d]); writeA = true; break;
        case 28: result = c->gpr[d] & c->gpr[b]; writeA = true; break;
        case 54: case 86: case 246: case 278: case 470: case 982: case 598: case 854: break;  // cache ops, sync, eieio
        case 55: { const uint32_t ea = EaX(c, w); c->gpr[d] = MemoryInline::FlatRead32(ea); c->gpr[a] = ea; break; }
        case 60: result = c->gpr[d] & ~c->gpr[b]; writeA = true; break;
        case 83: c->gpr[d] = c->msr; break;
        case 87: c->gpr[d] = MemoryInline::FlatRead8(EaX(c, w)); break;
        case 119: { const uint32_t ea = EaX(c, w); c->gpr[d] = MemoryInline::FlatRead8(ea); c->gpr[a] = ea; break; }
        case 124: result = ~(c->gpr[d] | c->gpr[b]); writeA = true; break;
        case 144: {  // mtcrf
            const uint32_t crm = Bits(w, 12, 8);
            for (uint32_t k = 0; k < 8; ++k)
                if (crm & (0x80u >> k)) c->crv[k] = PpcCrvEncode((c->gpr[d] >> (28u - k * 4u)) & 0xFu);
            break;
        }
        case 146: c->msr = c->gpr[d]; break;
        case 150:  // stwcx.: always succeeds (one core, no reservations lost)
            MemoryInline::FlatWrite32(EaX(c, w), c->gpr[d]);
            c->crv[0] = PpcCrvEncode(0x2u | ((c->xer & kXerSo) ? 0x1u : 0u));
            break;
        case 151: MemoryInline::FlatWrite32(EaX(c, w), c->gpr[d]); break;
        case 183: { const uint32_t ea = EaX(c, w); MemoryInline::FlatWrite32(ea, c->gpr[d]); c->gpr[a] = ea; break; }
        case 210: case 242: break;  // mtsr, mtsrin
        case 215: MemoryInline::FlatWrite8(EaX(c, w), static_cast<uint8_t>(c->gpr[d])); break;
        case 247: { const uint32_t ea = EaX(c, w); MemoryInline::FlatWrite8(ea, static_cast<uint8_t>(c->gpr[d])); c->gpr[a] = ea; break; }
        case 279: c->gpr[d] = MemoryInline::FlatRead16(EaX(c, w)); break;
        case 284: result = ~(c->gpr[d] ^ c->gpr[b]); writeA = true; break;
        case 311: { const uint32_t ea = EaX(c, w); c->gpr[d] = MemoryInline::FlatRead16(ea); c->gpr[a] = ea; break; }
        case 316: result = c->gpr[d] ^ c->gpr[b]; writeA = true; break;
        case 339: c->gpr[d] = ReadSpr(c, (b << 5) | a, pc, w); break;
        case 343: c->gpr[d] = static_cast<uint32_t>(static_cast<int16_t>(MemoryInline::FlatRead16(EaX(c, w)))); break;
        case 371: c->gpr[d] = ((b << 5) | a) == 269 ? PPC_Mftbu() : PPC_Mftb(); break;
        case 375: { const uint32_t ea = EaX(c, w); c->gpr[d] = static_cast<uint32_t>(static_cast<int16_t>(MemoryInline::FlatRead16(ea))); c->gpr[a] = ea; break; }
        case 407: MemoryInline::FlatWrite16(EaX(c, w), static_cast<uint16_t>(c->gpr[d])); break;
        case 412: result = c->gpr[d] | ~c->gpr[b]; writeA = true; break;
        case 439: { const uint32_t ea = EaX(c, w); MemoryInline::FlatWrite16(ea, static_cast<uint16_t>(c->gpr[d])); c->gpr[a] = ea; break; }
        case 444: result = c->gpr[d] | c->gpr[b]; writeA = true; break;
        case 467: WriteSpr(c, (b << 5) | a, c->gpr[d]); break;
        case 476: result = ~(c->gpr[d] & c->gpr[b]); writeA = true; break;
        case 512: {  // mcrxr
            c->crv[(d >> 2) & 7u] = PpcCrvEncode(c->xer >> 28);
            c->xer &= 0x0FFFFFFFu;
            break;
        }
        case 534: c->gpr[d] = __builtin_bswap32(MemoryInline::FlatRead32(EaX(c, w))); break;  // lwbrx
        case 535: SetScalar(f, d, MemoryInline::FlatReadFloat32(EaX(c, w))); break;  // lfsx
        case 536: result = PPC_Srw(c->gpr[d], c->gpr[b]); writeA = true; break;
        case 567: { const uint32_t ea = EaX(c, w); SetScalar(f, d, MemoryInline::FlatReadFloat32(ea)); c->gpr[a] = ea; break; }
        case 533: case 597: {  // lswx (XER bytes from rA|0 + rB: a P+ poke loop), lswi
            uint32_t ea = xo == 533 ? EaX(c, w) : a ? c->gpr[a] : 0;
            uint32_t n = xo == 533 ? (c->xer & 0x7Fu) : b ? b : 32, r = d - 1;
            for (uint32_t k = 0; k < n; ++k) {
                if (k % 4 == 0) { r = (r + 1) & 31u; c->gpr[r] = 0; }
                c->gpr[r] |= static_cast<uint32_t>(MemoryInline::FlatRead8(ea++)) << (24 - 8 * (k % 4));
            }
            break;
        }
        case 599: SetScalar(f, d, MemoryInline::FlatReadFloat64(EaX(c, w))); break;  // lfdx
        case 631: { const uint32_t ea = EaX(c, w); SetScalar(f, d, MemoryInline::FlatReadFloat64(ea)); c->gpr[a] = ea; break; }
        case 662: MemoryInline::FlatWrite32(EaX(c, w), __builtin_bswap32(c->gpr[d])); break;  // stwbrx
        case 663: MemoryInline::FlatWriteFloat32(EaX(c, w), Scalar(f, d)); break;  // stfsx
        case 695: { const uint32_t ea = EaX(c, w); MemoryInline::FlatWriteFloat32(ea, Scalar(f, d)); c->gpr[a] = ea; break; }
        case 661: case 725: {  // stswx, stswi
            uint32_t ea = xo == 661 ? EaX(c, w) : a ? c->gpr[a] : 0;
            uint32_t n = xo == 661 ? (c->xer & 0x7Fu) : b ? b : 32, r = d - 1;
            for (uint32_t k = 0; k < n; ++k) {
                if (k % 4 == 0) r = (r + 1) & 31u;
                MemoryInline::FlatWrite8(ea++, static_cast<uint8_t>(c->gpr[r] >> (24 - 8 * (k % 4))));
            }
            break;
        }
        case 727: MemoryInline::FlatWriteFloat64(EaX(c, w), Scalar(f, d)); break;  // stfdx
        case 759: { const uint32_t ea = EaX(c, w); MemoryInline::FlatWriteFloat64(ea, Scalar(f, d)); c->gpr[a] = ea; break; }
        case 790: c->gpr[d] = __builtin_bswap16(MemoryInline::FlatRead16(EaX(c, w))); break;  // lhbrx
        case 792: {  // sraw
            const uint32_t sh = c->gpr[b] & 63u;
            const int32_t s = static_cast<int32_t>(c->gpr[d]);
            result = PPC_Sraw(c->gpr[d], c->gpr[b]);
            SetCa(c, s < 0 && (sh > 31 ? true : (static_cast<uint32_t>(s) & ((1u << sh) - 1u)) != 0));
            writeA = true;
            break;
        }
        case 824: {  // srawi
            const uint32_t sh = b;
            const int32_t s = static_cast<int32_t>(c->gpr[d]);
            result = static_cast<uint32_t>(s >> sh);
            SetCa(c, s < 0 && sh != 0 && (static_cast<uint32_t>(s) & ((1u << sh) - 1u)) != 0);
            writeA = true;
            break;
        }
        case 918: MemoryInline::FlatWrite16(EaX(c, w), __builtin_bswap16(static_cast<uint16_t>(c->gpr[d]))); break;
        case 922: result = static_cast<uint32_t>(static_cast<int16_t>(c->gpr[d])); writeA = true; break;
        case 954: result = static_cast<uint32_t>(static_cast<int8_t>(c->gpr[d])); writeA = true; break;
        case 983: MemoryInline::FlatWrite32(EaX(c, w), static_cast<uint32_t>(c->fpr[d].raw)); break;  // stfiwx
        case 1014: {  // dcbz
            const uint32_t ea = EaX(c, w) & ~31u;
            for (uint32_t k = 0; k < 32; k += 4) MemoryInline::FlatWrite32(ea + k, 0);
            break;
        }
        default: Undefined(c, pc, w, "interpreter: opcode 31");
        }
        if (writeA) {
            c->gpr[a] = result;
            if (Rc(w)) SetCr0(c, result);
        }
        break;
    }
    case 32: c->gpr[RD(w)] = MemoryInline::FlatRead32(Ea(c, w)); break;
    case 33: { const uint32_t ea = Ea(c, w); c->gpr[RD(w)] = MemoryInline::FlatRead32(ea); c->gpr[RA(w)] = ea; break; }
    case 34: c->gpr[RD(w)] = MemoryInline::FlatRead8(Ea(c, w)); break;
    case 35: { const uint32_t ea = Ea(c, w); c->gpr[RD(w)] = MemoryInline::FlatRead8(ea); c->gpr[RA(w)] = ea; break; }
    case 36: MemoryInline::FlatWrite32(Ea(c, w), c->gpr[RD(w)]); break;
    case 37: { const uint32_t ea = Ea(c, w); MemoryInline::FlatWrite32(ea, c->gpr[RD(w)]); c->gpr[RA(w)] = ea; break; }
    case 38: MemoryInline::FlatWrite8(Ea(c, w), static_cast<uint8_t>(c->gpr[RD(w)])); break;
    case 39: { const uint32_t ea = Ea(c, w); MemoryInline::FlatWrite8(ea, static_cast<uint8_t>(c->gpr[RD(w)])); c->gpr[RA(w)] = ea; break; }
    case 40: c->gpr[RD(w)] = MemoryInline::FlatRead16(Ea(c, w)); break;
    case 41: { const uint32_t ea = Ea(c, w); c->gpr[RD(w)] = MemoryInline::FlatRead16(ea); c->gpr[RA(w)] = ea; break; }
    case 42: c->gpr[RD(w)] = static_cast<uint32_t>(static_cast<int16_t>(MemoryInline::FlatRead16(Ea(c, w)))); break;
    case 43: { const uint32_t ea = Ea(c, w); c->gpr[RD(w)] = static_cast<uint32_t>(static_cast<int16_t>(MemoryInline::FlatRead16(ea))); c->gpr[RA(w)] = ea; break; }
    case 44: MemoryInline::FlatWrite16(Ea(c, w), static_cast<uint16_t>(c->gpr[RD(w)])); break;
    case 45: { const uint32_t ea = Ea(c, w); MemoryInline::FlatWrite16(ea, static_cast<uint16_t>(c->gpr[RD(w)])); c->gpr[RA(w)] = ea; break; }
    case 46: {  // lmw
        uint32_t ea = Ea(c, w);
        for (uint32_t r = RD(w); r < 32; ++r, ea += 4) c->gpr[r] = MemoryInline::FlatRead32(ea);
        break;
    }
    case 47: {  // stmw
        uint32_t ea = Ea(c, w);
        for (uint32_t r = RD(w); r < 32; ++r, ea += 4) MemoryInline::FlatWrite32(ea, c->gpr[r]);
        break;
    }
    case 48: SetScalar(f, RD(w), MemoryInline::FlatReadFloat32(Ea(c, w))); break;
    case 49: { const uint32_t ea = Ea(c, w); SetScalar(f, RD(w), MemoryInline::FlatReadFloat32(ea)); c->gpr[RA(w)] = ea; break; }
    case 50: SetScalar(f, RD(w), MemoryInline::FlatReadFloat64(Ea(c, w))); break;
    case 51: { const uint32_t ea = Ea(c, w); SetScalar(f, RD(w), MemoryInline::FlatReadFloat64(ea)); c->gpr[RA(w)] = ea; break; }
    case 52: MemoryInline::FlatWriteFloat32(Ea(c, w), Scalar(f, RD(w))); break;
    case 53: { const uint32_t ea = Ea(c, w); MemoryInline::FlatWriteFloat32(ea, Scalar(f, RD(w))); c->gpr[RA(w)] = ea; break; }
    case 54: MemoryInline::FlatWriteFloat64(Ea(c, w), Scalar(f, RD(w))); break;
    case 55: { const uint32_t ea = Ea(c, w); MemoryInline::FlatWriteFloat64(ea, Scalar(f, RD(w))); c->gpr[RA(w)] = ea; break; }
    case 56: case 57: case 60: case 61: {  // psq_l, psq_lu, psq_st, psq_stu: 12-bit displacement
        int32_t disp = static_cast<int32_t>(w & 0xFFFu);
        if (disp & 0x800) disp -= 0x1000;
        const uint32_t op = w >> 26, a = RA(w);
        const uint32_t ea = ((op == 57 || op == 61) || a ? c->gpr[a] : 0) + static_cast<uint32_t>(disp);
        const uint32_t wBit = Bits(w, 15, 1), i = Bits(w, 12, 3);
        if (op <= 57) SetPaired(f, RD(w), PPC_PsqL(ea, wBit, i));
        else PPC_PsqSt(ea, Paired(f, RD(w)), wBit, i);
        if (op == 57 || op == 61) c->gpr[a] = ea;
        break;
    }
    case 59: {  // single-precision arithmetic
        const uint32_t d = RD(w), a = RA(w), b = RB(w), cc = RC(w);
        switch (Bits(w, 1, 5)) {
        case 18: SetScalar(f, d, PpcForceSingleValueInline(PpcFdivInline(Scalar(f, a), Scalar(f, b)))); break;
        case 20: SetScalar(f, d, PpcForceSingleValueInline(PpcFsubInline(Scalar(f, a), Scalar(f, b)))); break;
        case 21: SetScalar(f, d, PpcForceSingleValueInline(PpcFaddInline(Scalar(f, a), Scalar(f, b)))); break;
        case 24: SetScalar(f, d, PpcForceSingleValueInline(PpcApproximateReciprocalInline(Scalar(f, b)))); break;
        case 25: SetScalar(f, d, PpcFmulsInline(Scalar(f, a), Scalar(f, cc))); break;
        case 28: SetScalar(f, d, PpcFmsubsInline(Scalar(f, a), Scalar(f, cc), Scalar(f, b))); break;
        case 29: SetScalar(f, d, PpcFmaddsInline(Scalar(f, a), Scalar(f, cc), Scalar(f, b))); break;
        case 30: SetScalar(f, d, PpcFnmsubsInline(Scalar(f, a), Scalar(f, cc), Scalar(f, b))); break;
        case 31: SetScalar(f, d, PpcFnmaddsInline(Scalar(f, a), Scalar(f, cc), Scalar(f, b))); break;
        default: Undefined(c, pc, w, "interpreter: opcode 59");
        }
        if (Rc(w)) SetCr1FromFpscr(c);
        break;
    }
    case 63: {  // double-precision arithmetic, moves, compares, FPSCR
        const uint32_t d = RD(w), a = RA(w), b = RB(w), cc = RC(w);
        const uint32_t xo5 = Bits(w, 1, 5);
        if (xo5 >= 18 && xo5 != 22 && xo5 != 24 && xo5 != 26 && xo5 <= 31) {
            switch (xo5) {
            case 18: SetScalar(f, d, PpcFdivInline(Scalar(f, a), Scalar(f, b))); break;
            case 20: SetScalar(f, d, PpcFsubInline(Scalar(f, a), Scalar(f, b))); break;
            case 21: SetScalar(f, d, PpcFaddInline(Scalar(f, a), Scalar(f, b))); break;
            case 23: SetScalar(f, d, Scalar(f, a) >= 0.0 ? Scalar(f, cc) : Scalar(f, b)); break;  // fsel
            case 25: SetScalar(f, d, PpcFmulInline(Scalar(f, a), Scalar(f, cc))); break;
            case 28: SetScalar(f, d, PpcFmsubInline(Scalar(f, a), Scalar(f, cc), Scalar(f, b))); break;
            case 29: SetScalar(f, d, PpcFmaddInline(Scalar(f, a), Scalar(f, cc), Scalar(f, b))); break;
            case 30: SetScalar(f, d, PpcFnmsubInline(Scalar(f, a), Scalar(f, cc), Scalar(f, b))); break;
            case 31: SetScalar(f, d, PpcFnmaddInline(Scalar(f, a), Scalar(f, cc), Scalar(f, b))); break;
            default: Undefined(c, pc, w, "interpreter: opcode 63 A-form");
            }
            if (Rc(w)) SetCr1FromFpscr(c);
            break;
        }
        switch (Bits(w, 1, 10)) {
        case 0: case 32: c->crv[(static_cast<int>(d >> 2)) & 7] = PpcCrvCompareFloat(Scalar(f, a), Scalar(f, b)); break;
        case 12: SetScalar(f, d, PpcForceSingleValueInline(Scalar(f, b))); break;  // frsp
        case 14: SetScalar(f, d, PPC_Fctiw(Scalar(f, b))); break;
        case 15: SetScalar(f, d, PPC_Fctiwz(Scalar(f, b))); break;
        case 26: SetScalar(f, d, PpcApproximateReciprocalSquareRootInline(Scalar(f, b))); break;  // frsqrte
        case 38: PPC_Mtfsb1(d); break;   // mtfsb1
        case 70: PPC_Mtfsb0(d); break; // mtfsb0
        case 40: SetScalar(f, d, -Scalar(f, b)); break;  // fneg
        case 64: PPC_Mcrfs(d >> 2, a >> 2); break;  // writes the field itself
        case 72:  // fmr: copies whichever form the source holds
            c->fpr[d] = c->fpr[b];
            CopyForm(f, d, b);
            break;
        case 134: PPC_Mtfsfi(d >> 2, Bits(w, 12, 4)); break;
        case 136: SetScalar(f, d, -std::fabs(Scalar(f, b))); break;  // fnabs
        case 264: SetScalar(f, d, std::fabs(Scalar(f, b))); break;   // fabs
        case 583: c->fpr[d].raw = c->fpscr; f.pairedMask &= ~(1u << (d & 31u)); break;  // mffs
        case 711: PPC_Mtfsf(Bits(w, 17, 8), c->fpr[b].d); break;
        default: Undefined(c, pc, w, "interpreter: opcode 63");
        }
        if (Rc(w)) SetCr1FromFpscr(c);
        break;
    }
    default:
        Undefined(c, pc, w, "interpreter: primary opcode");
    }
    return Step::Continue;
}

}  // namespace detail

namespace {

// --- the cached interpreter ----------------------------------------------------------------------------------------
//
// Run's default path. Guest code is decoded once into blocks: straight runs of instructions that end at b, a call (bl,
// bcl), bclr, bcctr, rfi or sc, before a pc with a runtime hook (kHookPcs: those are only tested where a block
// starts), at the end of guest memory, or after kMaxBlock instructions. A conditional bc does not end a block: taken,
// it leaves it there. Each instruction becomes an Op: the handler's kind and
// the operands extracted (registers, sign-extended or shifted immediates, rlwinm's mask). The common integer, load,
// store and float forms have handlers of their own; every other instruction is a Generic op that runs Exec, the
// per-instruction interpreter above, on the original word, so the two paths share every rarer semantic.
//
// Plain data and a switch: no code is generated at run time, so it runs wherever the runtime does (no executable
// memory, no JIT permission), and every compiler turns the switch into a jump table. Each op is dispatched once: the
// handlers' switch, then only the block's last op (its branch) or a Generic op goes on to a second.
//
// What costs at this size is memory, not decoding: blocks are packed in build order into 256 KB chunks, found through
// a 256 KB direct-mapped table (a map behind it), and their code check is a few inline word compares. A match
// builds about 2-3 MB of blocks; kMaxCacheBytes caps it.
//
// A block keeps the words it was decoded from and compares them with guest memory whenever it is entered, so code a
// mod writes, a module linked over another, a Gecko patch or a rollback restore is decoded again: nothing about the
// code is assumed. Blocks are per thread (no locks); they are never freed while a Run is active on the thread, and
// the whole cache is dropped once it grows past kMaxCacheBytes at the next outermost Run.
//
// BRAWL_INTERP=classic: decode every instruction as it runs (the previous interpreter). BRAWL_INTERP_VERIFY=1: run
// every register-only handler (all but the stores) against Exec on a copy of the registers, and log differences.

}  // namespace

namespace detail {

constexpr uint32_t kHookPcs[] = {0x80949BC0u, 0x809435F0u, 0x800171B4u, 0x80017508u, 0x800172D0u, 0x8001730Cu,
    0x80017504u, 0x801C9AACu, 0x801C9B54u, 0x801C9BFCu, 0x801C9C70u, 0x801BC684u};

bool IsHookPc(uint32_t pc) {
    for (uint32_t h : kHookPcs) if (pc == h) return true;
    return false;
}

}  // namespace detail

namespace {

constexpr size_t kMaxCacheBytes = 32u << 20;
// The same rule for compiled code: past this much the whole cache is dropped at the next outermost Run (the JIT's code
// goes with its blocks). BRAWL_JIT_CODE_LIMIT=<bytes> overrides it (the tests use a small one).
size_t MaxJitCodeBytes() {
    static const size_t limit = [] {
        const char* text = std::getenv("BRAWL_JIT_CODE_LIMIT");
        return text != nullptr && *text != '\0' ? static_cast<size_t>(std::strtoull(text, nullptr, 10)) : size_t{64} << 20;
    }();
    return limit;
}


bool EndsBlock(uint32_t w) {
    const uint32_t op = w >> 26;
    if (op == 16) return (w & 1u) != 0;  // bcl ends a block; bc becomes a Bc op that leaves it when taken
    if (op == 18) return (w & 1u) == 0;  // a bl's call returns to the next op, which the block goes on with
    if (w == 0x4E800421u) return false;  // bctrl, the same
    if (op == 17) return true;
    if (op == 19) {
        const uint32_t xo = Bits(w, 1, 10);
        if (xo == 16 && (w & 1u) == 0 && (RD(w) & 0x14u) != 0x14u) return false;  // a Bclr op: leaves when taken
        return xo == 16 || xo == 50 || xo == 528;
    }
    return false;
}

Op Decode(uint32_t w, uint32_t pc) {
    Op o{K::Generic, static_cast<uint8_t>(RD(w)), static_cast<uint8_t>(RA(w)), static_cast<uint8_t>(RB(w)), w};
    const uint32_t a = RA(w), b = RB(w), cc = RC(w);
    const auto as = [&](K kind, uint32_t imm) { o.kind = kind; o.imm = imm; return o; };
    const uint32_t simm = static_cast<uint32_t>(Simm(w)), uimm = Uimm(w);
    switch (w >> 26) {
    case 4: {
        const uint32_t xo6 = Bits(w, 1, 6), xo10 = Bits(w, 1, 10), xo5 = Bits(w, 1, 5);
        if (Rc(w) || xo6 == 6 || xo6 == 7 || xo6 == 38 || xo6 == 39) return o;
        switch (xo10) {
        case 40: return as(K::PsNeg, cc);
        case 72: return as(K::PsMr, cc);
        case 264: return as(K::PsAbs, cc);
        case 528: return as(K::PsMerge00, cc);
        case 560: return as(K::PsMerge01, cc);
        case 592: return as(K::PsMerge10, cc);
        case 624: return as(K::PsMerge11, cc);
        case 0: case 32: case 64: case 96: case 136: case 1014: return o;
        default: break;
        }
        switch (xo5) {
        case 10: return as(K::PsSum0, cc);
        case 11: return as(K::PsSum1, cc);
        case 12: return as(K::PsMuls0, cc);
        case 13: return as(K::PsMuls1, cc);
        case 14: return as(K::PsMadds0, cc);
        case 15: return as(K::PsMadds1, cc);
        case 20: return as(K::PsSub, cc);
        case 21: return as(K::PsAdd, cc);
        case 25: return as(K::PsMul, cc);
        case 28: return as(K::PsMsub, cc);
        case 29: return as(K::PsMadd, cc);
        case 30: return as(K::PsNmsub, cc);
        case 31: return as(K::PsNmadd, cc);
        default: return o;
        }
    }
    case 16: {
        if (w & 1u) return o;
        const int32_t disp = static_cast<int16_t>(w & 0xFFFCu);
        return as(K::Bc, (w & 2u) ? static_cast<uint32_t>(disp) : pc + static_cast<uint32_t>(disp));
    }
    case 18: {
        int32_t disp = static_cast<int32_t>(w & 0x03FFFFFCu);
        if (disp & 0x02000000) disp -= 0x04000000;
        uint32_t target = (w & 2u) ? static_cast<uint32_t>(disp) : pc + static_cast<uint32_t>(disp);
        if (target < 0x01800000u) target |= 0x80000000u;  // as Exec: P+ maps virtual 0 onto MEM1
        return as((w & 1u) ? K::Bl : K::B, target);
    }
    case 7: return as(K::Mulli, simm);
    case 8: return as(K::Subfic, simm);
    case 23: return as(Rc(w) ? K::RlwnmRc : K::Rlwnm, Mask(Bits(w, 6, 5), Bits(w, 1, 5)));
    case 10: return as(K::Cmplwi, uimm);
    case 11: return as(K::Cmpwi, simm);
    case 14: return a ? as(K::Addi, simm) : as(K::Li, simm);
    case 15: return a ? as(K::Addis, uimm << 16) : as(K::Lis, uimm << 16);
    case 19:
        if (w == 0x4E800020u) return as(K::Blr, 0);
        if (w == 0x4E800420u) return as(K::Bctr, 0);
        if (w == 0x4E800421u) return as(K::Bctrl, 0);
        if (Bits(w, 1, 10) == 16 && (w & 1u) == 0 && (RD(w) & 0x14u) != 0x14u) return as(K::Bclr, 0);
        return o;
    case 20: return Rc(w) ? o : as(K::Rlwimi, Mask(Bits(w, 6, 5), Bits(w, 1, 5)));
    case 21: return as(Rc(w) ? K::RlwinmRc : K::Rlwinm, Mask(Bits(w, 6, 5), Bits(w, 1, 5)));
    case 24: return as(K::Ori, uimm);
    case 25: return as(K::Oris, uimm << 16);
    case 26: return as(K::Xori, uimm);
    case 27: return as(K::Xoris, uimm << 16);
    case 28: return as(K::AndiRc, uimm);
    case 29: return as(K::AndisRc, uimm << 16);
    case 31: {
        const uint32_t xo = Bits(w, 1, 10);
        const bool oe = (w & 0x400u) != 0;
        switch (xo) {
        case 0: return as(K::Cmpw, 0);
        case 32: return as(K::Cmplw, 0);
        case 19: return as(K::Mfcr, 0);
        default: break;
        }
        if (!oe && !Rc(w)) {
            switch (xo & 0x1FFu) {
            case 266: return as(K::Add, 0);
            case 40: return as(K::Subf, 0);
            case 104: return as(K::Neg, 0);
            case 235: return as(K::Mullw, 0);
            default: break;
            }
        }
        if (xo == 444 && Rc(w)) return as(K::OrRc, 0);
        if (xo == 28 && Rc(w)) return as(K::AndRc, 0);
        if (!Rc(w)) {
            switch (xo) {
            case 444: return as(K::Or, 0);
            case 28: return as(K::And, 0);
            case 316: return as(K::Xor, 0);
            case 60: return as(K::Andc, 0);
            case 124: return as(K::Nor, 0);
            case 24: return as(K::Slw, 0);
            case 536: return as(K::Srw, 0);
            case 824: return as(K::Srawi, 0);
            case 954: return as(K::Extsb, 0);
            case 922: return as(K::Extsh, 0);
            case 26: return as(K::Cntlzw, 0);
            default: break;
            }
        }
        if (xo == 339 || xo == 467) {
            const uint32_t spr = (b << 5) | a;
            if (spr == 8) return as(xo == 339 ? K::Mflr : K::Mtlr, 0);
            if (spr == 9) return as(xo == 339 ? K::Mfctr : K::Mtctr, 0);
            if (spr == 268 || spr == 269) return o;  // the time base: Exec's (the verifier could not compare it)
            return as(xo == 339 ? K::Mfspr : K::Mtspr, spr);
        }
        if (a == 0) return o;
        switch (xo) {
        case 23: return as(K::Lwzx, 0);
        case 87: return as(K::Lbzx, 0);
        case 279: return as(K::Lhzx, 0);
        case 151: return as(K::Stwx, 0);
        case 215: return as(K::Stbx, 0);
        case 407: return as(K::Sthx, 0);
        case 535: return as(K::Lfsx, 0);
        case 663: return as(K::Stfsx, 0);
        default: return o;
        }
    }
    case 59: {
        if (Rc(w)) return o;
        switch (Bits(w, 1, 5)) {
        case 18: return as(K::Fdivs, cc);
        case 20: return as(K::Fsubs, cc);
        case 21: return as(K::Fadds, cc);
        case 25: return as(K::Fmuls, cc);
        case 28: return as(K::Fmsubs, cc);
        case 29: return as(K::Fmadds, cc);
        case 30: return as(K::Fnmsubs, cc);
        case 31: return as(K::Fnmadds, cc);
        default: return o;
        }
    }
    case 63: {
        if (Rc(w)) return o;
        const uint32_t xo5 = Bits(w, 1, 5);
        if (xo5 >= 18 && xo5 != 22 && xo5 != 24 && xo5 != 26 && xo5 <= 31) {
            switch (xo5) {
            case 18: return as(K::Fdiv, cc);
            case 20: return as(K::Fsub, cc);
            case 21: return as(K::Fadd, cc);
            case 25: return as(K::Fmul, cc);
            case 28: return as(K::Fmsub, cc);
            case 29: return as(K::Fmadd, cc);
            default: return o;
            }
        }
        switch (Bits(w, 1, 10)) {
        case 0: case 32: return as(K::Fcmpu, 0);  // fcmpo too: Exec runs both as one
        case 12: return as(K::Frsp, 0);
        case 40: return as(K::Fneg, 0);
        case 72: return as(K::Fmr, 0);
        case 264: return as(K::Fabs, 0);
        default: return o;
        }
    }
    default: break;
    }
    // D-form loads and stores. With rA = 0 the address is absolute and the update forms are invalid: Exec has them.
    if (a == 0) return o;
    switch (w >> 26) {
    case 32: return as(K::Lwz, simm);
    case 33: return as(K::Lwzu, simm);
    case 34: return as(K::Lbz, simm);
    case 35: return as(K::Lbzu, simm);
    case 36: return as(K::Stw, simm);
    case 37: return as(K::Stwu, simm);
    case 38: return as(K::Stb, simm);
    case 39: return as(K::Stbu, simm);
    case 40: return as(K::Lhz, simm);
    case 41: return as(K::Lhzu, simm);
    case 42: return as(K::Lha, simm);
    case 44: return as(K::Sth, simm);
    case 45: return as(K::Sthu, simm);
    case 46: return as(K::Lmw, simm);
    case 47: return as(K::Stmw, simm);
    case 48: return as(K::Lfs, simm);
    case 49: return as(K::Lfsu, simm);
    case 50: return as(K::Lfd, simm);
    case 51: return as(K::Lfdu, simm);
    case 52: return as(K::Stfs, simm);
    case 53: return as(K::Stfsu, simm);
    case 54: return as(K::Stfd, simm);
    case 55: return as(K::Stfdu, simm);
    case 56: case 57: case 60: case 61: {
        int32_t disp = static_cast<int32_t>(w & 0xFFFu);
        if (disp & 0x800) disp -= 0x1000;
        o.b = static_cast<uint8_t>((Bits(w, 15, 1) << 3) | Bits(w, 12, 3));
        const uint32_t op = w >> 26;
        return as(op == 56 ? K::PsqL : op == 57 ? K::PsqLu : op == 60 ? K::PsqSt : K::PsqStu,
                  static_cast<uint32_t>(disp));
    }
    default: return o;
    }
}

bool IsStore(K k) {
    switch (k) {
    case K::Stw: case K::Stwu: case K::Stb: case K::Stbu: case K::Sth: case K::Sthu: case K::Stmw:
    case K::Stwx: case K::Stbx: case K::Sthx: case K::Stfs: case K::Stfsu: case K::Stfd: case K::Stfdu: case K::Stfsx:
    case K::PsqSt: case K::PsqStu:
        return true;
    default: return false;
    }
}

struct BlockCache {
    struct Slot { uint32_t pc; Block* block; };
    // Direct-mapped by a hash of pc that folds in the high bits, so the dol and the modules above it don't collide;
    // 256 KB, small enough to stay in cache.
    static constexpr uint32_t kSlots = 16384;
    static uint32_t SlotOf(uint32_t pc) { return ((pc >> 2) ^ (pc >> 16)) & (kSlots - 1u); }
    Slot slots[kSlots] = {};
    FastEntrySlot fastEntries[kFastEntrySlots] = {};  // L1 cache of compiled, logged, non-hook entries -> jitChecked
    // The entry each slot of fastEntries last held before another took it (a second way, which only the C++ entries probe:
    // jit_subcall reads fastEntries alone). Hot patched functions whose starts share a slot took turns in it, each call of
    // one sending the other to the slow entry (two such pairs ran ~17 times an update each).
    FastEntrySlot fastVictims[kFastEntrySlots] = {};
    // The compiled entry of `pc` in either way, or null.
    MKW_PPC_FORCE_INLINE const void* FastEntryFor(uint32_t pc) const {
        const uint32_t i = FastEntryIdx(pc);
        if (__builtin_expect(fastEntries[i].pc == pc, 1)) return fastEntries[i].jitChecked;
        return fastVictims[i].pc == pc ? fastVictims[i].jitChecked : nullptr;
    }
    void FastEntryPut(uint32_t pc, const void* jitChecked) {
        static const bool victims = [] {  // BRAWL_JIT_FAST_VICTIMS=0: one way, as before
            const char* v = std::getenv("BRAWL_JIT_FAST_VICTIMS");
            return v == nullptr || *v != '0';
        }();
        const uint32_t i = FastEntryIdx(pc);
        if (victims && fastEntries[i].pc != pc && fastEntries[i].jitChecked != nullptr) fastVictims[i] = fastEntries[i];
        if (fastVictims[i].pc == pc) fastVictims[i] = FastEntrySlot{};
        fastEntries[i] = FastEntrySlot{pc, 0, jitChecked};
    }
    std::unordered_map<uint32_t, Block*> blocks;
    // Blocks are packed into chunks in the order they are built, so code that runs together sits together.
    static constexpr size_t kChunkBytes = 256 * 1024;
    std::vector<std::unique_ptr<uint8_t[]>> storage;
    uint8_t* chunkAt = nullptr;
    size_t chunkLeft = 0;
    size_t bytes = 0;
    unsigned depth = 0;  // Runs active on this thread: blocks they may still hold must stay
    bool overLimit = false;  // bytes > kMaxCacheBytes || JitBytes() > MaxJitCodeBytes() (updated only on Build/Compile)
    JitCodeCache* jit = nullptr;  // the backend's code for the blocks, made when the first one is compiled

    uint64_t built = 0, rebuilt = 0, clears = 0, runs = 0, blocksRun = 0, opsRun = 0, genericRun = 0, slotMisses = 0, current = 0;
    uint64_t linked = 0, claims = 0, retired = 0, epochs = 0;
    uint64_t staleTracked = 0, staleOther = 0, partial = 0, linkRefreshes = 0;
    std::unordered_map<uint32_t, uint64_t> stalePcs;  // BRAWL_INTERP_STATS: compares per block start
    uint32_t lastEpoch = 0;

#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
    // The blocks whose code lies under each mark (replaced ones too: their epoch stays 0); those marks as the lines of the
    // mark map they fall in; for the marks of rollback's tracked pages that a consume found set, the last such consume (see
    // BrawlDirtyMemory::consumeSeq); and the blocks in the order built, with the consume count when built.
    std::unordered_map<uint32_t, std::vector<Block*>> owners;
    // A line is 64 marks (one cache line of the mark map, 4 KiB of guest memory, inside one tracked page): its number
    // (mark >> 6) and which of its marks lie under code (a byte of 0xFF each). Ascending; the marks that got their first
    // owner since the last consume wait in newMarks and are merged in by the next one (CodeLinesMerge).
    struct CodeLine {
        uint32_t line;
        uint32_t reserved;
        uint64_t mask[8];
    };
    std::vector<CodeLine> codeLines;
    // codeLines by tracked page (BrawlDirtyMemory::trackedPages: 16 lines a page): a page's number and its lines [begin, end)
    struct CodePage {
        uint32_t page, begin, end;
    };
    std::vector<CodePage> codePages;
    std::vector<uint32_t> newMarks;
    void CodeLinesMerge() {  // with CachesMutex held
        std::sort(newMarks.begin(), newMarks.end());
        std::vector<CodeLine> added;
        for (const uint32_t m : newMarks) {
            if (added.empty() || added.back().line != (m >> 6)) added.push_back(CodeLine{m >> 6, 0, {}});
            uint8_t* const bytes = reinterpret_cast<uint8_t*>(added.back().mask);
            bytes[m & 63u] = 0xFF;  // the marks are bytes: the line's loads are little or big endian alike
        }
        newMarks.clear();
        std::vector<CodeLine> merged;
        merged.reserve(codeLines.size() + added.size());
        size_t i = 0, j = 0;
        while (i < codeLines.size() || j < added.size()) {
            if (j == added.size() || (i < codeLines.size() && codeLines[i].line < added[j].line)) {
                merged.push_back(codeLines[i++]);
            } else if (i == codeLines.size() || added[j].line < codeLines[i].line) {
                merged.push_back(added[j++]);
            } else {
                CodeLine k = codeLines[i++];
                for (uint32_t w = 0; w < 8; ++w) k.mask[w] |= added[j].mask[w];
                ++j;
                merged.push_back(k);
            }
        }
        codeLines.swap(merged);
        codePages.clear();
        for (uint32_t i = 0; i < codeLines.size(); ++i) {
            const uint32_t page = codeLines[i].line >> 4;
            if (codePages.empty() || codePages.back().page != page) codePages.push_back(CodePage{page, i, i});
            codePages.back().end = i + 1;
        }
    }
    // (consume count, mark) in count order: each time a consume found a code mark set (the newest entry of a mark is
    // its last such consume). A restore reads it back from the end, only as far as its snapshot's count.
    std::vector<std::pair<uint32_t, uint32_t>> writtenAt;
    void NoteWritten(uint32_t mark, uint32_t seq) {  // with CachesMutex held
        auto at = writtenAt.end();
        while (at != writtenAt.begin() && std::prev(at)->first > seq) --at;  // counts arrive in order but one thread
        writtenAt.insert(at, {seq, mark});                                   // may take its count before another
        if (writtenAt.size() > writtenCompactAt) CompactWritten();
    }
    // Drop the entries a newer one of the same mark supersedes, keeping the order; next time at twice what is left.
    size_t writtenCompactAt = size_t{1} << 16;
    void CompactWritten() {
        std::unordered_set<uint32_t> seen;
        size_t keep = writtenAt.size();
        for (size_t i = writtenAt.size(); i-- > 0;)
            if (seen.insert(writtenAt[i].second).second) writtenAt[--keep] = writtenAt[i];
        writtenAt.erase(writtenAt.begin(), writtenAt.begin() + static_cast<std::ptrdiff_t>(keep));
        writtenCompactAt = std::max(size_t{1} << 16, 2 * writtenAt.size());
    }
    std::vector<std::pair<uint32_t, Block*>> builtAt;
    // Every thread's cache, for retiring blocks under a mark another thread's cache (or a scanner) clears.
    static std::mutex& CachesMutex() { static std::mutex m; return m; }
    static std::vector<BlockCache*>& Caches() { static std::vector<BlockCache*> v; return v; }
    BlockCache() {
        std::lock_guard<std::mutex> lock(CachesMutex());
        Caches().push_back(this);
        BrawlDirtyMemory::consumingMarks = &ConsumingMarks;
        BrawlDirtyMemory::restoredMarks = &RestoredMarks;
        BrawlDirtyMemory::restoringMarks = &RestoringMarks;
        BrawlDirtyMemory::restoredWritten = &RestoreWritten;
        t_own = this;
        BrawlDirtyMemory::guardedWritten = &GuardedWritten;
    }
    ~BlockCache() {
        {
            std::lock_guard<std::mutex> lock(CachesMutex());
            Caches().erase(std::find(Caches().begin(), Caches().end(), this));
        }
        if (t_own == this) t_own = nullptr;
        DestroyJit();
    }
    BlockCache(const BlockCache&) = delete;
    BlockCache& operator=(const BlockCache&) = delete;
    void Own(Block* b, uint32_t mark) {  // with CachesMutex held
        std::vector<Block*>& v = owners[mark];
        if (v.empty()) newMarks.push_back(mark);
        v.push_back(b);
    }
    // The calling thread's cache (made by its ThreadState, on that thread).
    static inline thread_local BlockCache* t_own = nullptr;
    // The blocks of this cache a restore announced with MarksRestoring would drop, kept as they are until the restore has written
    // (RestoreWritten: with CachesMutex held).
    std::vector<Block*> restorePending;
    // A block the restore may change: dropped now, or (`pending`: a restore that RestoreWritten follows, this thread's cache) once
    // the restore has written and only if its words changed.
    void DropRestored(Block& b, bool pending) {  // with CachesMutex held
        if (pending) restorePending.push_back(&b);
        else DropCounted(b);
    }
    void RetireAll(uint32_t mark, bool pending) {  // with CachesMutex held
        const auto it = owners.find(mark);
        if (it == owners.end()) return;
        for (Block* o : it->second)
            if (o->epoch != 0) DropRestored(*o, pending);
    }
    // A block no longer known to match guest memory (epoch 0): the link slots aimed at it are suspended first, as compiled code
    // follows a slot without testing the target's epoch (Block::jitLinked). A refresh that finds its words unchanged aims them
    // at it again (Resume), so a block rollback's restore only may have written costs one compare, not a relink of every way in.
    // Every write of a block's epoch: its compiled code holds a copy (Block::jitEpoch).
    static void SetEpoch(Block& b, uint32_t epoch) {
        b.epoch = epoch;
        if (b.jitEpoch != nullptr)
            if (const JitBackend* const backend = Backend()) backend->setEpoch(b);
    }
    static void Drop(Block& b) {
        SetEpoch(b, 0);
        if (b.jitIncoming != nullptr)
            if (const JitBackend* const backend = Backend()) backend->suspend(b);
    }
    void DropCounted(Block& b) {
        Drop(b);
        ++retired;
    }
    static void Resume(Block& b) {
        if (b.jitIncoming != nullptr)
            if (const JitBackend* const backend = Backend()) backend->resume(b);
    }
    // The epoch the link slots of every cache were aimed in (g_jitLinkEpoch). When markEpoch has moved since, every slot is
    // suspended: a block that matched guest memory then may not now (memory rewritten without marks), and its slots do not test
    // its epoch. Called before a block is taken as matching in the epoch (Build, Refresh) and by the helpers compiled code
    // calls that may have run guest code (JitNoteEpoch). Takes CachesMutex.
    static void NoteEpoch(uint32_t epoch) {
        if (__builtin_expect(epoch == g_jitLinkEpoch, 1)) return;
        SweepLinks(epoch);
    }
    static MKW_PPC_NO_INLINE void SweepLinks(uint32_t epoch) {
        std::lock_guard<std::mutex> lock(CachesMutex());
        if (const JitBackend* const backend = Backend(); backend != nullptr && g_jitEpochFault != 1)
            for (BlockCache* c : Caches())
                for (const auto& [pc, b] : c->blocks)
                    if (b->jitIncoming != nullptr) backend->suspend(*b);
        g_jitLinkEpoch = epoch;
    }
    // BrawlDirtyMemory::consumingMarks: rollback's WriteWatch is about to clear its pages' marks, after which a
    // write they show would no longer show; retire the code under the set ones now. The marks are read a line of the
    // mark map at a time (CodeLine: one cache line, two vector loads), only the lines that hold code.
    static void ConsumingMarks(uint32_t seq) {
        diag::Add(diag::kConsumes);
        std::lock_guard<std::mutex> lock(CachesMutex());
        const uint8_t* const marks = BrawlDirtyMemory::leanBlocks;
        for (BlockCache* c : Caches()) {
            if (!c->newMarks.empty()) c->CodeLinesMerge();
            const uint32_t curEpoch = BrawlDirtyMemory::markEpoch.load(std::memory_order_acquire);
            const CodeLine* const lines = c->codeLines.data();
            for (const CodePage& g : c->codePages) {
                // only the tracked pages' marks are consumed
                if (!BrawlDirtyMemory::trackedPages[g.page & 0x3FFFu].load(std::memory_order_relaxed)) continue;
                for (uint32_t i = g.begin; i < g.end; ++i) {
                    const CodeLine& k = lines[i];
                    const uint8_t* const at = marks + (size_t{k.line} << 6);
                    uint64_t any = 0;
                    for (uint32_t w = 0; w < 8; ++w) {
                        uint64_t v;
                        std::memcpy(&v, at + 8u * w, 8);
                        any |= v & k.mask[w];
                    }
                    if (__builtin_expect(any == 0, 1)) continue;
                    const uint8_t* const code = reinterpret_cast<const uint8_t*>(k.mask);
                    for (uint32_t j = 0; j < 64; ++j) {
                        if (code[j] == 0 || at[j] == 0) continue;
                        const uint32_t m = (k.line << 6) + j;
                        const auto it = c->owners.find(m);
                        if (it != c->owners.end()) {
                            for (Block* o : it->second)
                                if (o->epoch != 0 && (o->epoch != curEpoch || !HoldsUnder(*o, m))) {
                                    c->DropCounted(*o);
                                    diag::Add(diag::kConsumeRetired);
                                }
                        }
                        c->NoteWritten(m, seq);
                    }
                }
            }
        }
    }
    // BrawlDirtyMemory::restoredMarks: rollback restored its pages to a snapshot taken at consume `snapshotSeq`, so
    // guest memory changed where it was written since, all of which a later consume found (or code built since then
    // was decoded from). Retire that code; the restore counts as a write for any older snapshot.
    // BrawlDirtyMemory::restoringMarks: the same for a restore that calls MarksRestoreWritten once it has written: the blocks of
    // this thread's cache wait for RestoreWritten (nothing runs on this thread in between: the caller restores and returns),
    // those of other threads' caches are dropped now.
    static void RestoredMarks(uint32_t snapshotSeq) { RestoredMarksImpl(snapshotSeq, false); }
    static void RestoringMarks(uint32_t snapshotSeq) {
        static const bool pending = [] {  // BRAWL_JIT_RESTORE_PENDING=0: every block dropped before the restore writes, as before
            const char* v = std::getenv("BRAWL_JIT_RESTORE_PENDING");
            return v == nullptr || *v != '0';
        }();
        RestoredMarksImpl(snapshotSeq, pending);
    }
    static void RestoredMarksImpl(uint32_t snapshotSeq, bool written) {
        diag::Add(diag::kRestores);
        const uint64_t retiredBefore = diag::ChurnOn() ? RetiredTotal() : 0;
        struct Tally {
            uint64_t before;
            ~Tally() { if (diag::ChurnOn()) diag::Add(diag::kRestoreRetired, RetiredTotal() - before); }
        } tally{retiredBefore};
        const uint32_t now = BrawlDirtyMemory::consumeSeq.fetch_add(1, std::memory_order_acq_rel) + 1;
        std::lock_guard<std::mutex> lock(CachesMutex());
        for (BlockCache* c : Caches()) {
            const bool pending = written && c == t_own;
            // the marks last found set after the snapshot; the restore writes them again, now
            size_t first = c->writtenAt.size();
            while (first > 0 && c->writtenAt[first - 1].first > snapshotSeq) --first;
            if (first < c->writtenAt.size()) {
                std::vector<std::pair<uint32_t, uint32_t>> again(c->writtenAt.begin() + static_cast<std::ptrdiff_t>(first),
                                                                 c->writtenAt.end());
                c->writtenAt.resize(first);
                std::sort(again.begin(), again.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
                again.erase(std::unique(again.begin(), again.end(),
                                        [](const auto& a, const auto& b) { return a.second == b.second; }),
                            again.end());
                for (auto& [at, m] : again) {
                    c->RetireAll(m, pending);
                    at = now;
                }
                c->writtenAt.insert(c->writtenAt.end(), again.begin(), again.end());
            }
            for (auto it = c->builtAt.rbegin(); it != c->builtAt.rend() && it->first >= snapshotSeq; ++it)
                if (it->second->epoch != 0) c->DropRestored(*it->second, pending);
        }
    }
    // BrawlDirtyMemory::restoredWritten: the restore MarksRestoring announced has written guest memory. Of the blocks of this
    // thread's cache it would have dropped (restorePending), those whose words it changed are dropped now, before any of them can
    // run; the others (a mod's variable next to them is what the restore rewrote, or code built since the snapshot from words the
    // snapshot holds too) stay as they were: no epoch or link slot in their code is written, and no refresh follows.
    static void RestoreWritten() {
        std::lock_guard<std::mutex> lock(CachesMutex());
        for (BlockCache* c : Caches()) {
            for (Block* b : c->restorePending) {
                if (b->epoch == 0 || g_jitEpochFault == 6) continue;  // dropped since (a consume); 6: a test aid
                if (!Holds(*b)) c->DropCounted(*b);
            }
            c->restorePending.clear();
        }
    }
    // Guarded code (BrawlDirtyMemory::guardedWritten, Block::jitGuarded): the start pcs whose blocks must keep their tests (they
    // were dropped by writes kGuardDemote times), and the hook. A write that marks a block under guarded code drops every
    // guarded block with code there in every cache, as a mark its test would see: before compiled code can follow a slot
    // aimed at it (the slots are suspended; the next refresh compares its words and aims them again). Blocks that test
    // their marks need nothing here.
    std::unordered_set<uint32_t> unguarded;  // with CachesMutex held
    static constexpr uint8_t kGuardDemote = 4;
    static void GuardedWritten(uint32_t canonical, size_t size) {
        if (g_jitEpochFault == 5) return;  // a test aid (g_jitEpochFault)
        std::lock_guard<std::mutex> lock(CachesMutex());
        uint64_t* const lines = BrawlDirtyMemory::guardLines;
        const uint32_t first = canonical >> 6;
        const uint32_t last = static_cast<uint32_t>((uint64_t{canonical} + size - 1) >> 6);
        for (uint32_t m = first; m <= last && m < BrawlDirtyMemory::kLeanDirtyBlocks; ++m) {
            const uint64_t bit = uint64_t{1} << (m & 63u);
            if ((lines[m >> 6] & bit) == 0) continue;
            // The line's bit stays while a guarded block that may run again has code there (a dropped one may be refreshed);
            // with none left (demoted, or replaced: never refreshed) later writes there skip the lookup.
            bool needed = false;
            for (BlockCache* c : Caches()) {
                const auto it = c->owners.find(m);
                if (it == c->owners.end()) continue;
                for (Block* o : it->second) {
                    if (!o->jitGuarded || o->jitDemote) continue;
                    if (o->epoch != 0) {
                        c->DropCounted(*o);
                        diag::Add(diag::kGuardDrops);
                        if (++o->guardDrops >= kGuardDemote) {
                            o->jitDemote = true;
                            c->unguarded.insert(o->start);
                            continue;
                        }
                    }
                    needed = true;
                }
            }
            if (!needed) lines[m >> 6] &= ~bit;
        }
    }
    // A block just compiled as guarded: its marks go into the guard tables (never taken out: a mark there with no guarded
    // block under it only costs the hook a lookup).
    static void NoteGuardedBlock(const Block& b) {
        static uint8_t chunks[1u << 14];
        static uint64_t lines[BrawlDirtyMemory::kLeanDirtyBlocks / 64];
        std::lock_guard<std::mutex> lock(CachesMutex());
        for (uint32_t s = 0; s < b.segments; ++s)
            for (uint32_t m = b.segment[s].firstMark; m <= b.segment[s].lastMark; ++m) {
                if (!JitGuardedLine(b.segment[s].start, m)) continue;  // its mark is tested
                lines[m >> 6] |= uint64_t{1} << (m & 63u);
                chunks[m >> 10] = 1;
            }
        BrawlDirtyMemory::guardLines = lines;
        BrawlDirtyMemory::guardChunks = chunks;
    }
    static uint64_t RetiredTotal() {  // BRAWL_JIT_CHURN: every cache's retired count (takes CachesMutex)
        std::lock_guard<std::mutex> lock(CachesMutex());
        uint64_t n = 0;
        for (const BlockCache* c : Caches()) n += c->retired;
        return n;
    }
    // The mark is about to be cleared: a block under it that matched guest memory in this epoch still does if its
    // words under the mark do; any other compares all its code before it runs again (with CachesMutex held).
    void Recheck(uint32_t mark, const Block* except, uint32_t epoch) {
        const auto it = owners.find(mark);
        if (it == owners.end()) return;
        for (Block* o : it->second)
            if (o != except && o->epoch != 0 && (o->epoch != epoch || !HoldsUnder(*o, mark))) DropCounted(*o);
    }
#endif

    // BRAWL_INTERP_STATS=1: the first stale blocks (code changed since decoding) and the cache's counts.
    static bool Stats() {
        static const bool on = [] { const char* e = std::getenv("BRAWL_INTERP_STATS"); return e && *e && *e != '0'; }();
        return on;
    }
    void NoteStale(const Block& b) {
        if (!Stats()) return;
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) >= 64) return;
        uint32_t i = 0;
        for (uint32_t s = 0; s < b.segments; ++s) {
            for (uint32_t k = 0; k < b.segment[s].count; ++k, ++i) {
                const uint32_t at = b.segment[s].start + 4u * k;
                const uint32_t now = MemoryInline::FlatRead32(at);
                if (now != b.words[i]) {
                    RT_LOGF(RT_TAG_RUNTIME, "interpreter: block 0x%08X (%u words) stale at 0x%08X: %08X now %08X\n",
                            b.start, b.count, at, b.words[i], now);
                    return;
                }
            }
        }
    }
    // BRAWL_INTERP_STATS: what the dispatch loop runs. Kinds and adjacent pairs are counted over whole blocks as
    // entered (a taken bc leaves early, so ops after it are slightly overcounted); exits by the block's last op.
    struct Profile {
        uint64_t kinds[kKinds] = {};
        uint64_t pairs[kKinds][kKinds] = {};
        uint64_t exits[kKinds + 1] = {};  // by last op kind; [kKinds]: frame return
        uint64_t sameNext = 0, otherNext = 0;
        std::unordered_map<uint32_t, uint64_t> generic;  // by primary opcode << 16 | extended opcode
    };
    std::unique_ptr<Profile> profile;
    void Count(const Block& b, Step step, uint32_t next) {
        if (!profile) profile.reset(new Profile);
        Profile& p = *profile;
        for (uint32_t i = 0; i < b.count; ++i) {
            const uint32_t k = static_cast<uint32_t>(b.ops[i].kind);
            ++p.kinds[k];
            if (b.ops[i].kind == K::Generic) {
                const uint32_t w = b.words[i], op = w >> 26;
                ++p.generic[op << 16 | ((op == 4 || op == 19 || op == 31 || op == 59 || op == 63) ? Bits(w, 1, 10) : 0u)];
            }
            if (i + 1 < b.count) ++p.pairs[k][static_cast<uint32_t>(b.ops[i + 1].kind)];
        }
        if (step == Step::Return) { ++p.exits[kKinds]; return; }
        ++p.exits[static_cast<uint32_t>(b.ops[b.count - 1].kind)];
        Block& m = const_cast<Block&>(b);
        if (m.lastNext == next) ++p.sameNext; else ++p.otherNext;
        m.lastNext = next;
    }
    void ReportProfile() {
        if (!profile) return;
        const Profile& p = *profile;
        uint64_t total = 0;
        for (uint64_t k : p.kinds) total += k;
        std::vector<std::pair<uint64_t, uint32_t>> v;
        auto top = [&](const char* what, size_t n) {
            std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
            std::string line;
            for (size_t i = 0; i < v.size() && i < n; ++i) {
                char item[64];
                const uint32_t key = v[i].second;
                if (key >> 16) std::snprintf(item, sizeof(item), " %s+%s %.2f%%", kKindNames[(key >> 8) & 255], kKindNames[key & 255], 100.0 * v[i].first / total);
                else std::snprintf(item, sizeof(item), " %s %.2f%%", key < kKinds ? kKindNames[key] : "return", 100.0 * v[i].first / total);
                line += item;
            }
            RT_LOGF(RT_TAG_RUNTIME, "interpreter profile %s:%s\n", what, line.c_str());
            v.clear();
        };
        for (uint32_t k = 0; k < kKinds; ++k) v.emplace_back(p.kinds[k], k);
        top("kinds", 40);
        for (uint32_t a = 0; a < kKinds; ++a)
            for (uint32_t c = 0; c < kKinds; ++c) v.emplace_back(p.pairs[a][c], (1u << 16) | (a << 8) | c);
        top("pairs", 40);
        uint64_t exits = 0;
        for (uint64_t e : p.exits) exits += e;
        for (uint32_t k = 0; k <= kKinds; ++k) v.emplace_back(p.exits[k] * total / (exits ? exits : 1), k);
        top("exits (share of exits, scaled)", 12);
        {
            std::vector<std::pair<uint64_t, uint32_t>> g;
            for (const auto& [key, n] : p.generic) g.emplace_back(n, key);
            std::sort(g.begin(), g.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
            std::string line;
            for (size_t i = 0; i < g.size() && i < 16; ++i) {
                char item[48];
                std::snprintf(item, sizeof(item), " %u/%u %.2f%%", g[i].second >> 16, g[i].second & 0xFFFFu,
                              100.0 * g[i].first / total);
                line += item;
            }
            RT_LOGF(RT_TAG_RUNTIME, "interpreter profile generic (opcode/xo):%s\n", line.c_str());
        }
        RT_LOGF(RT_TAG_RUNTIME, "interpreter profile: %llu exits, next block same as last time %.1f%%\n",
                static_cast<unsigned long long>(exits), 100.0 * p.sameNext / ((p.sameNext + p.otherNext) ? (p.sameNext + p.otherNext) : 1));
    }

    void Report() {
        if (!Stats()) return;
        ReportProfile();
        RT_LOGF(RT_TAG_RUNTIME, "interpreter cache: %llu built, %llu rebuilt (code changed), %llu clears, %zu blocks, %.1f MB; "
                "%llu runs, %llu blocks run, %llu ops (%llu generic), %llu slot misses, %llu unchanged by marks, %llu linked, "
                "%llu claims, %llu retired, %llu epochs\n",
                static_cast<unsigned long long>(built), static_cast<unsigned long long>(rebuilt),
                static_cast<unsigned long long>(clears), blocks.size(), bytes / 1048576.0,
                static_cast<unsigned long long>(runs), static_cast<unsigned long long>(blocksRun),
                static_cast<unsigned long long>(opsRun), static_cast<unsigned long long>(genericRun),
                static_cast<unsigned long long>(slotMisses), static_cast<unsigned long long>(current),
                static_cast<unsigned long long>(linked), static_cast<unsigned long long>(claims),
                static_cast<unsigned long long>(retired), static_cast<unsigned long long>(epochs));
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
        {
            std::vector<std::pair<uint64_t, uint32_t>> v;
            for (const auto& [at, n] : stalePcs) v.emplace_back(n, at);
            std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
            std::string line;
            for (size_t i = 0; i < v.size() && i < 24; ++i) {
                const Block* const sb = blocks.count(v[i].second) ? blocks.at(v[i].second) : nullptr;
                char item[96];
                std::snprintf(item, sizeof item, " %08X:%llu(m%u-%u)", v[i].second,
                              static_cast<unsigned long long>(v[i].first), sb ? sb->segment[0].firstMark : 0u,
                              sb ? sb->segment[sb->segments - 1].lastMark : 0u);
                line += item;
            }
            RT_LOGF(RT_TAG_RUNTIME, "interpreter stale: %llu tracked, %llu other, %llu partial, %llu link refreshes;%s\n",
                    static_cast<unsigned long long>(staleTracked), static_cast<unsigned long long>(staleOther),
                    static_cast<unsigned long long>(partial), static_cast<unsigned long long>(linkRefreshes),
                    line.c_str());
        }
#endif
    }

    void Clear() {
        ++clears;
        Report();
        for (Slot& s : slots) s = Slot{};
        std::memset(fastEntries, 0, sizeof(fastEntries));
        std::memset(fastVictims, 0, sizeof(fastVictims));
        for (const auto& [pc, b] : blocks) Retire(*b);
        blocks.clear();
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
        {
            std::lock_guard<std::mutex> lock(CachesMutex());
            owners.clear();
            codeLines.clear();
            codePages.clear();
            newMarks.clear();
            writtenAt.clear();
            builtAt.clear();
            restorePending.clear();
        }
#endif
        DestroyJit();  // after the blocks are dropped: all their code goes
        storage.clear();
        chunkAt = nullptr;
        chunkLeft = 0;
        bytes = 0;
        overLimit = false;
    }

    // A block that is being replaced or dropped, before its replacement is published: the backend resets the link slots
    // aimed at it and drops its fast-entry slot. Its own code stays until the cache's goes.
    void Retire(Block& b) {
        FastEntrySlot& fe = fastEntries[FastEntryIdx(b.start)];
        if (fe.pc == b.start) fe = FastEntrySlot{};
        FastEntrySlot& victim = fastVictims[FastEntryIdx(b.start)];
        if (victim.pc == b.start) victim = FastEntrySlot{};
        if (b.jitEntry == nullptr && b.jitIncoming == nullptr) return;
        if (const JitBackend* const backend = Backend()) backend->retire(b);
    }
    void UpdateOverLimit() {
        overLimit = bytes > kMaxCacheBytes || JitBytes() > MaxJitCodeBytes();
    }
    size_t JitBytes() const {
        if (jit == nullptr) return 0;
        const JitBackend* const backend = Backend();
        return backend != nullptr && backend->codeBytes != nullptr ? backend->codeBytes(*jit) : 0;
    }
    void DestroyJit() {
        if (jit == nullptr) return;
        Backend()->destroyCache(jit);
        jit = nullptr;
    }
    // Out-of-line tail for blocks spanning > 2 mark checks (rare: kMaxSegments <= 6), keeping Current tiny and
    // preventing Clang from emitting a massive AVX2 vectorization epilogue in callers.
    static MKW_PPC_NO_INLINE bool CurrentExtraChecks(const Block& b, const uint8_t* marks) {
        uint64_t any = 0;
        for (uint32_t k = 2; k < b.checks; ++k) {
            uint64_t v;
            std::memcpy(&v, marks + b.check[k].at, 8);
            any |= v & b.check[k].mask;
        }
        return any == 0;
    }
    static bool Current(const Block& b, uint32_t epoch) {
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
        if (b.epoch != epoch) return false;
        const uint8_t* const marks = BrawlDirtyMemory::leanBlocks;
        uint64_t v0, v1;
        std::memcpy(&v0, marks + b.check[0].at, 8);
        std::memcpy(&v1, marks + b.check[1].at, 8);
        if (((v0 & b.check[0].mask) | (v1 & b.check[1].mask)) != 0) return false;
        if (__builtin_expect(b.checks > 2, 0)) return CurrentExtraChecks(b, marks);
        return true;
#else
        (void)b;
        (void)epoch;
        return false;
#endif
    }
    // A block Current turned down matches guest memory again if its code compares equal: all of it in a new epoch
    // (or once retired), else only its words under the marks that are set, as the rest has not been written since it
    // last matched. Marks outside rollback's tracked pages (nothing else reads them) are cleared first, so Current
    // holds again until a write marks them; the clear comes before the compare, so a write after it marks the block
    // again and one before it shows in the compare. A mark also covers other blocks' code (blocks overlap, and 64
    // bytes hold several, maybe in another thread's cache), which Recheck compares the same way. Tracked marks stay
    // set until rollback consumes them (starting a new epoch), so until then the block compares under them each run.
    MKW_PPC_NO_INLINE bool Refresh(Block& b, uint32_t epoch) {
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
        NoteEpoch(epoch);  // before b can be taken in this epoch: no slot aimed in an older one may stay
        if (b.jitDemote) {  // guarded code written too often: built again, compiled with its tests (GuardedWritten)
            SetEpoch(b, 0);
            return false;
        }
        const bool whole = b.epoch != epoch;
        bool tracked = false;
        for (uint32_t s = 0; s < b.segments && !tracked; ++s) {
            const Segment& g = b.segment[s];
            tracked = g.firstMark <= g.lastMark &&
                      (BrawlDirtyMemory::trackedPages[(g.firstMark >> 10) & 0x3FFFu].load(std::memory_order_acquire) ||
                       BrawlDirtyMemory::trackedPages[(g.lastMark >> 10) & 0x3FFFu].load(std::memory_order_acquire));
        }
        auto* const marks = reinterpret_cast<std::atomic<uint8_t>*>(BrawlDirtyMemory::leanBlocks);
        for (uint32_t s = 0; s < b.segments; ++s)
            for (uint32_t m = b.segment[s].firstMark; m <= b.segment[s].lastMark; ++m) {
                if (!marks[m].load(std::memory_order_relaxed)) continue;
                if (!tracked) {
                    marks[m].store(0, std::memory_order_relaxed);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    ++claims;
                    std::lock_guard<std::mutex> lock(CachesMutex());
                    for (BlockCache* c : Caches()) c->Recheck(m, &b, epoch);
                }
                if (!whole && !HoldsUnder(b, m)) {
                    SetEpoch(b, 0);  // its marks may be clear now: Current must not take it
                    return false;
                }
            }
        if (whole) {
            if (!Holds(b)) {
                SetEpoch(b, 0);
                return false;
            }
            SetEpoch(b, epoch);
            Resume(b);  // its words are unchanged: the slots suspended when it was dropped (or the epoch moved) aim at it again
        } else {
            ++partial;
        }
        return true;
#else
        if (!Holds(b)) {
            SetEpoch(b, 0);
            return false;
        }
        SetEpoch(b, epoch);
        return true;
#endif
    }

    // BRAWL_INTERP_CHECK=1: compare the code of a block Current says is unchanged anyway, and log any write the marks
    // missed (which would also be a rollback bug).
    static bool CheckMarks() {
        static const bool on = [] { const char* e = std::getenv("BRAWL_INTERP_CHECK"); return e && *e && *e != '0'; }();
        return on;
    }

    // The words that differ (BRAWL_INTERP_CHECK), so the writer the marks missed can be told from what it wrote.
    static void LogChange(const Block& b, uint32_t epoch) {
        RT_LOGF(RT_TAG_RUNTIME, "interpreter: block 0x%08X changed without a dirty mark (epoch %u, now %u)\n", b.start,
                b.epoch, epoch);
        const uint32_t* word = b.words;
        int shown = 0;
        for (uint32_t s = 0; s < b.segments; ++s)
            for (uint32_t k = 0; k < b.segment[s].count; ++k, ++word) {
                const uint32_t at = b.segment[s].start + 4u * k;
                const uint32_t now = MemoryInline::FlatRead32(at);
                if (now != *word && shown++ < 8)
                    RT_LOGF(RT_TAG_RUNTIME, "interpreter:   0x%08X was %08X, now %08X (mark %u)\n", at, *word, now,
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
                            static_cast<unsigned>(BrawlDirtyMemory::leanBlocks[(at & 0x3FFFFFFFu) >> 6]));
#else
                            0u);
#endif
            }
    }

    static bool Holds(const Block& b) {
        uint32_t diff = 0;
#if MKW_LEAN_MEMORY
        // Inline word compares: blocks are short, and a libc memcmp call costs more than the compare itself.
        const uint32_t* raw = b.words + b.count;
        for (uint32_t s = 0; s < b.segments; ++s) {
            const uint8_t* host = MemoryInline::LeanHost(b.segment[s].start);
            for (uint32_t k = 0; k < b.segment[s].count; ++k) {
                uint32_t w;
                std::memcpy(&w, host + 4u * k, 4);
                diff |= w ^ *raw++;
            }
        }
#else
        const uint32_t* word = b.words;
        for (uint32_t s = 0; s < b.segments; ++s)
            for (uint32_t k = 0; k < b.segment[s].count; ++k)
                diff |= MemoryInline::FlatRead32(b.segment[s].start + 4u * k) ^ *word++;
#endif
        return diff == 0;
    }
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
    // Holds for the block's words under one mark.
    static bool HoldsUnder(const Block& b, uint32_t mark) {
        uint32_t diff = 0;
        const uint32_t* raw = b.words + b.count;
        for (uint32_t s = 0; s < b.segments; ++s) {
            const Segment& g = b.segment[s];
            if (mark - g.firstMark <= g.lastMark - g.firstMark) {
                const uint32_t canonical = g.start & BrawlDirtyMemory::kLeanDirtyMask;
                const uint8_t* const host = MemoryInline::LeanHost(g.start);
                for (uint32_t k = 0; k < g.count; ++k) {
                    if (((canonical + 4u * k) >> 6) != mark) continue;
                    uint32_t w;
                    std::memcpy(&w, host + 4u * k, 4);
                    diff |= w ^ raw[k];
                }
            }
            raw += g.count;
        }
        return diff == 0;
    }
#endif

    // A b is followed when its target is untranslated (registry lookups are fixed once published), and not in code
    // the block already has (a loop's back edge ends it).
    static bool Follows(uint32_t target, const Segment* segment, uint32_t segments) {
        if (!TranslatedFunctionRegistry::IsLookupPublished()) return false;
        if (IsHookPc(target) || GuestCodePatch::ContinuationAt(target)) return false;
        if (TranslatedFunctionRegistry::FindByAddressPtr(target) != nullptr) return false;
        for (uint32_t s = 0; s < segments; ++s)
            if (target - segment[s].start <= 4u * segment[s].count) return false;
        return true;
    }

    Block* Build(uint32_t pc) {
        const uint32_t epoch = BrawlDirtyMemory::markEpoch.load(std::memory_order_acquire);
        uint32_t words[kMaxBlock], pcs[kMaxBlock];
        Op decoded[kMaxBlock];
        Segment segment[kMaxSegments];
        uint32_t n = 0, segments = 1, at = pc, end = pc;
        segment[0] = Segment{pc, 0, 0, 0};
        while (n < kMaxBlock) {
            if (n != 0 && (IsHookPc(at) || !Memory::Contains(at, 4))) break;
            const uint32_t w = MemoryInline::FlatRead32(at);
            words[n] = w;
            pcs[n] = at;
            decoded[n] = Decode(w, at);
            ++segment[segments - 1].count;
            end = at;
            ++n;
            if (decoded[n - 1].kind == K::B && segments < kMaxSegments &&
                Follows(decoded[n - 1].imm, segment, segments)) {
                decoded[n - 1].kind = K::Jmp;
                at = decoded[n - 1].imm;
                segment[segments++] = Segment{at, 0, 0, 0};
                continue;
            }
            if (decoded[n - 1].kind == K::B)  // a b that stays a block end: whether it is a tail call, when known
                decoded[n - 1].b = !TranslatedFunctionRegistry::IsLookupPublished()                          ? 0
                                   : TranslatedFunctionRegistry::FindByAddressPtr(decoded[n - 1].imm) != nullptr ? 1
                                                                                                           : 2;
            if (EndsBlock(w)) {
                at += 4;
                break;
            }
            at += 4;
        }
        const uint32_t fallsTo = at;  // after the last op (a followed b's target, when that is the last)
        // A compare (or rlwinm.) right before a bc runs with it.
        for (uint32_t i = 0; i + 1 < n; ++i) {
            if (decoded[i + 1].kind != K::Bc) continue;
            switch (decoded[i].kind) {
            case K::Cmpwi: decoded[i].kind = K::CmpwiBc; break;
            case K::Cmplwi: decoded[i].kind = K::CmplwiBc; break;
            case K::Cmpw: decoded[i].kind = K::CmpwBc; break;
            case K::Cmplw: decoded[i].kind = K::CmplwBc; break;
            case K::RlwinmRc: decoded[i].kind = K::RlwinmRcBc; break;
            default: break;
            }
        }
        // ops and End, then the words as FlatRead32 returns them, then (lean memory) the same words as guest memory
        // holds them, then the pcs, then the segments
        // (Decode leaves a bl's b as the word's rb bits, part of its displacement)
        uint32_t calls = 0;
        const bool published = TranslatedFunctionRegistry::IsLookupPublished();
        for (uint32_t i = 0; i < n; ++i)
            if (decoded[i].kind == K::Bl) decoded[i].b = published ? static_cast<uint8_t>(++calls) : uint8_t{0};
        const size_t size = (sizeof(Block) + (n + 1) * sizeof(Op) + calls * sizeof(CallSite) +
                             n * 3 * sizeof(uint32_t) + segments * sizeof(Segment) + 7u) & ~size_t{7};
        if (size > chunkLeft) {
            storage.emplace_back(new uint8_t[kChunkBytes]);
            chunkAt = storage.back().get();
            chunkLeft = kChunkBytes;
        }
        uint8_t* raw = chunkAt;
        chunkAt += size;
        chunkLeft -= size;
        Block* b = reinterpret_cast<Block*>(raw);
        Op* ops = reinterpret_cast<Op*>(raw + sizeof(Block));
        CallSite* const sites = reinterpret_cast<CallSite*>(raw + sizeof(Block) + (n + 1) * sizeof(Op));
        for (uint32_t i = 0; i < n; ++i)
            if (decoded[i].kind == K::Bl && decoded[i].b != 0) sites[decoded[i].b - 1] = ResolveCall(decoded[i].imm);
        uint32_t* stored = reinterpret_cast<uint32_t*>(sites + calls);
        uint32_t* const opPcs = stored + 2 * n;
        Segment* segs = reinterpret_cast<Segment*>(stored + 3 * n);
        for (uint32_t i = 0; i < n; ++i) {
            ops[i] = decoded[i];
            stored[i] = words[i];
            opPcs[i] = pcs[i];
        }
        ops[n] = Op{K::End, 0, 0, 0, fallsTo};
#if MKW_LEAN_MEMORY
        uint32_t* rawWords = stored + n;
#endif
        for (uint32_t s = 0; s < segments; ++s) {
            if (segment[s].count == 0) {  // a followed b at the block's end: nothing to check
                segment[s].firstMark = 1;
                segment[s].lastMark = 0;
            } else {
                segment[s].firstMark = (segment[s].start & 0x3FFFFFFFu) >> 6;
                segment[s].lastMark = ((segment[s].start + 4u * segment[s].count - 1u) & 0x3FFFFFFFu) >> 6;
            }
            segs[s] = segment[s];
#if MKW_LEAN_MEMORY
            std::memcpy(rawWords, MemoryInline::LeanHost(segment[s].start), 4u * segment[s].count);
            rawWords += segment[s].count;
#endif
        }
        b->start = pc;
        b->count = n;
        b->end = end;
        b->segments = segments;
        b->segment = segs;
        b->epoch = epoch;
        b->lastNext = 0;
        b->link[0] = b->link[1] = nullptr;
        b->hook = IsHookPc(pc);
        b->checks = 0;
        b->entries = 0;
        b->jitState = 0;
        b->jitEntry = nullptr;
        b->jitChecked = nullptr;
        b->jitLinked = nullptr;
        b->jitIncoming = nullptr;
        b->jitEpoch = nullptr;
        b->jitGuarded = false;
        b->jitDemote = false;
        b->guardDrops = 0;
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
        b->jitNoGuard = unguarded.count(pc) != 0;
#else
        b->jitNoGuard = true;
#endif
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
        NoteEpoch(epoch);  // the block is taken as matching in this epoch: no slot aimed in an older one may stay
        {
            // every mark under the block's words, ascending; then the fewest 8-mark windows over them, each starting at
            // the lowest mark not yet covered (a segment spans at most three marks)
            uint32_t marks[kMaxSegments * 3];
            uint32_t n = 0;
            for (uint32_t s = 0; s < segments; ++s)
                for (uint32_t m = segs[s].firstMark; m <= segs[s].lastMark && m - segs[s].firstMark < 3u; ++m) {
                    uint32_t i = n++;
                    for (; i > 0 && marks[i - 1] > m; --i) marks[i] = marks[i - 1];
                    marks[i] = m;
                }
            // the load must stay inside the marks (a code address is far below their end)
            constexpr uint32_t kLast = static_cast<uint32_t>(BrawlDirtyMemory::kLeanDirtyBlocks) - 7u;
            {
                std::lock_guard<std::mutex> lock(CachesMutex());
                for (uint32_t i = 0; i < n; ++i)
                    if (i == 0 || marks[i] != marks[i - 1]) Own(b, marks[i]);
                builtAt.emplace_back(BrawlDirtyMemory::consumeSeq.load(std::memory_order_acquire), b);
            }
            for (uint32_t i = 0; i < n; ++i) {
                if (b->checks == 0 || marks[i] - b->check[b->checks - 1].at > 7u)
                    b->check[b->checks++] = MarkCheck{0, marks[i] < kLast ? marks[i] : kLast};
                MarkCheck& k = b->check[b->checks - 1];
                const uint32_t byte = marks[i] - k.at;  // the marks are bytes: the load is little or big endian
                uint8_t bytes[8];
                std::memcpy(bytes, &k.mask, 8);
                if (byte < 8) bytes[byte] = 0xFF;
                std::memcpy(&k.mask, bytes, 8);
            }
        }
#endif
        for (uint32_t k = b->checks; k < 2; ++k) b->check[k] = MarkCheck{0, 0};
        b->words = stored;
        b->ops = ops;
        b->pcs = opPcs;
        b->calls = sites;
        bytes += size;
        if (bytes > kMaxCacheBytes) overLimit = true;
        if (Stats() && ((built + rebuilt) & 0xFFFFu) == 0) Report();
        blocks[pc] = b;
        return b;
    }

    Block* Find(uint32_t pc) {
        const uint32_t epoch = BrawlDirtyMemory::markEpoch.load(std::memory_order_acquire);
        if (epoch != lastEpoch) { lastEpoch = epoch; ++epochs; }
        diag::Add(diag::kFinds);
        Slot& slot = slots[SlotOf(pc)];
        Block* b = slot.pc == pc ? slot.block : nullptr;
        if (b == nullptr) {
            ++slotMisses;
            const auto it = blocks.find(pc);
            if (it != blocks.end()) b = it->second;
        }
        if (b == nullptr) {
            b = Build(pc);
            ++built;
        } else if (Current(*b, epoch)) {
            ++current;
            if (CheckMarks() && !Holds(*b)) {
                static std::atomic<int> logged{0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 64) LogChange(*b, epoch);
                SetEpoch(*b, 0);
                Retire(*b);
                b->jitDemote = true;
                b = Build(pc);
                ++rebuilt;
            }
        } else {
            if (Stats()) {
                const Segment& g = b->segment[0];
                (BrawlDirtyMemory::trackedPages[(g.firstMark >> 10) & 0x3FFFu].load(std::memory_order_relaxed)
                     ? staleTracked : staleOther)++;
                ++stalePcs[pc];
            }
            if (diag::ChurnOn()) diag::AddKey(diag::kTStale, pc);
            if (!Refresh(*b, epoch)) {
                NoteStale(*b);
                SetEpoch(*b, 0);
                Retire(*b);
                b->jitDemote = true;  // replaced: never refreshed again (and its guarded lines need no hook: GuardedWritten)
                b = Build(pc);
                ++rebuilt;
                if (diag::ChurnOn()) diag::AddKey(diag::kTRebuild, pc);
            }
        }
        slot = Slot{pc, b};
        return b;
    }

    // Find for the common case, with the statistics and BRAWL_INTERP_CHECK off (the caller has settled that): the block the table
    // holds for pc when it is current, which is all Find does then (its counters are the statistics'). Any other case is Find.
    MKW_PPC_FORCE_INLINE Block* FindCurrent(uint32_t pc) {
        const uint32_t epoch = BrawlDirtyMemory::markEpoch.load(std::memory_order_acquire);
        const Slot& slot = slots[SlotOf(pc)];
        if (slot.pc == pc && slot.block != nullptr && Current(*slot.block, epoch)) return slot.block;
        return Find(pc);
    }

    // The block to run after `from` leaves for pc: one of its links when that is still current, else Find's (which
    // then becomes the first link).
    MKW_PPC_FORCE_INLINE Block* Next(Block& from, uint32_t pc) {
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
        const uint32_t epoch = BrawlDirtyMemory::markEpoch.load(std::memory_order_acquire);
        Block* b = from.link[0];
        if (b != nullptr && b->start == pc) {
            if (Current(*b, epoch) || (++linkRefreshes, Refresh(*b, epoch))) {
                ++linked;
                return b;
            }
        } else if ((b = from.link[1]) != nullptr && b->start == pc &&
                   (Current(*b, epoch) || (++linkRefreshes, Refresh(*b, epoch)))) {
            ++linked;
            from.link[1] = from.link[0];
            from.link[0] = b;
            return b;
        }
#endif
        if (diag::ChurnOn()) diag::AddKey(diag::kTFind, (uint64_t{diag::kFromNext} << 32) | pc);
        Block* const found = Find(pc);
        if (from.link[0] != found) {
            from.link[1] = from.link[0];
            from.link[0] = found;
        }
        return found;
    }
};

// Handlers shared by the dispatch loop and the verifier. Each mirrors Exec's case for the same instruction.
inline uint32_t BaseD(const CpuContext* c, const Op& o) { return c->gpr[o.a] + o.imm; }
inline uint32_t BaseX(const CpuContext* c, const Op& o) { return c->gpr[o.a] + c->gpr[o.b]; }

// Where an op runs: its pc and position in the block, and the frame's return address. A handler that steps over the
// next op (a fused bc) advances it.
struct Cursor {
    const Op* op;
    uint32_t pc;  // without a block (the verifier's single op): the op's pc
    uint32_t next;
    uint32_t returnAddress;
    const Block* block;
};

MKW_PPC_FORCE_INLINE uint32_t PcOf(const Cursor& at) {
    return at.block != nullptr ? at.block->pcs[at.op - at.block->ops] : at.pc;
}

// After a call that returned to the op after it: the block goes on while its code is current (the callee may have
// written it, or a rollback restore replaced it), else it leaves for that op, which Find checks.
MKW_PPC_FORCE_INLINE Act AfterCall(Cursor& at) {
    if (BlockCache::Current(*at.block, BrawlDirtyMemory::markEpoch.load(std::memory_order_acquire))) return Act::Next;
    at.next = PcOf(at) + 4;
    return Act::Exit;
}

// A fused compare's bc: the op after it.
MKW_PPC_FORCE_INLINE Act FusedBc(CpuContext* c, Cursor& at) {
    ++at.op;
    if (BranchTaken(c, at.op->d, at.op->a)) {
        at.next = at.op->imm;
        return Act::Exit;
    }
    return Act::Next;
}

#if defined(_MSC_VER) && !defined(__clang__)
#define MKW_INTERP_UNREACHABLE() __assume(0)
#else
#define MKW_INTERP_UNREACHABLE() __builtin_unreachable()
#endif

// Every op in one switch (one dispatch). The plain kinds' handlers mirror Exec's case for the same instruction; the
// branches decide what Exec would, with the target worked out at decode.
// c is f.ctx, which the loop keeps in a register (f escapes to the handlers' calls, so f.ctx would be reloaded each
// op).
MKW_PPC_FORCE_INLINE Act RunOp(Frame& f, CpuContext* const c, Cursor& at) {
    const Op& o = *at.op;
    uint32_t* const r = c->gpr;
    switch (o.kind) {
    case K::Generic: return Act::Exec;
    case K::Bc:
        if (BranchTaken(c, o.d, o.a)) { at.next = o.imm; return Act::Exit; }
        return Act::Next;
    case K::CmpwiBc:
        c->crv[(static_cast<int>(o.d >> 2)) & 7] = PpcCrvCompare(static_cast<int32_t>(r[o.a]), static_cast<int32_t>(o.imm), c->xer);
        return FusedBc(c, at);
    case K::CmplwiBc: c->crv[(static_cast<int>(o.d >> 2)) & 7] = PpcCrvCompare(r[o.a], o.imm, c->xer); return FusedBc(c, at);
    case K::CmpwBc:
        c->crv[(static_cast<int>(o.d >> 2)) & 7] = PpcCrvCompare(static_cast<int32_t>(r[o.a]), static_cast<int32_t>(r[o.b]), c->xer);
        return FusedBc(c, at);
    case K::CmplwBc: c->crv[(static_cast<int>(o.d >> 2)) & 7] = PpcCrvCompare(r[o.a], r[o.b], c->xer); return FusedBc(c, at);
    case K::RlwinmRcBc: r[o.a] = PpcRotl32Inline(r[o.d], o.b) & o.imm; SetCr0(c, r[o.a]); return FusedBc(c, at);
    case K::Jmp: return Act::Next;  // the next op is the target's
    case K::End: at.next = o.imm; return Act::Exit;
    case K::B:  // b = 1: the target is translated (a tail call), 2: it is not, 0: not known at decode
        if (GuestCodePatch::ContinuationAt(o.imm) || o.b == 1 ||
            (o.b == 0 && TranslatedFunctionRegistry::FindByAddressPtr(o.imm) != nullptr)) return Act::Exec;
        at.next = o.imm;
        return Act::Exit;
    case K::Bl:
        c->pc = PcOf(at);
        if (!Call(f, o.imm, c->pc + 4, o.b != 0 && at.block != nullptr ? &at.block->calls[o.b - 1] : nullptr)) {
            at.next = o.imm;
            return Act::Exit;
        }
        return AfterCall(at);
    case K::Blr: {
        const uint32_t target = c->lr & ~3u;
        if (target == (at.returnAddress & ~3u)) {
            SetScalar(f, 1, Scalar(f, 1));  // the caller takes f1 as the scalar return value
            return Act::Return;
        }
        at.next = target;
        return Act::Exit;
    }
    case K::Bclr: {
        const uint32_t target = c->lr & ~3u;
        if (!BranchTaken(c, o.d, o.a)) return Act::Next;
        if (target == (at.returnAddress & ~3u)) {
            SetScalar(f, 1, Scalar(f, 1));  // the caller takes f1 as the scalar return value
            return Act::Return;
        }
        at.next = target;
        return Act::Exit;
    }
    case K::Bctr: {
        const uint32_t target = c->ctr & ~3u;
        if (TranslatedFunctionRegistry::FindByAddressPtr(target) != nullptr) return Act::Exec;  // a tail call
        at.next = target;
        return Act::Exit;
    }
    case K::Bctrl: {
        const uint32_t target = c->ctr & ~3u;
        c->pc = PcOf(at);
        if (!Call(f, target, c->pc + 4)) {
            at.next = target;
            return Act::Exit;
        }
        return AfterCall(at);
    }
    case K::Li: r[o.d] = o.imm; break;
    case K::Addi: r[o.d] = r[o.a] + o.imm; break;
    case K::Lis: r[o.d] = o.imm; break;
    case K::Addis: r[o.d] = r[o.a] + o.imm; break;
    case K::Mulli: r[o.d] = r[o.a] * o.imm; break;  // the low 32 bits of the signed product
    case K::Ori: r[o.a] = r[o.d] | o.imm; break;
    case K::Oris: r[o.a] = r[o.d] | o.imm; break;
    case K::Xori: r[o.a] = r[o.d] ^ o.imm; break;
    case K::Xoris: r[o.a] = r[o.d] ^ o.imm; break;
    case K::AndiRc: r[o.a] = r[o.d] & o.imm; SetCr0(c, r[o.a]); break;
    case K::AndisRc: r[o.a] = r[o.d] & o.imm; SetCr0(c, r[o.a]); break;
    case K::Rlwinm: r[o.a] = PpcRotl32Inline(r[o.d], o.b) & o.imm; break;
    case K::RlwinmRc: r[o.a] = PpcRotl32Inline(r[o.d], o.b) & o.imm; SetCr0(c, r[o.a]); break;
    case K::Rlwimi: r[o.a] = (PpcRotl32Inline(r[o.d], o.b) & o.imm) | (r[o.a] & ~o.imm); break;
    case K::Cmpwi: c->crv[(static_cast<int>(o.d >> 2)) & 7] = PpcCrvCompare(static_cast<int32_t>(r[o.a]), static_cast<int32_t>(o.imm), c->xer); break;
    case K::Cmplwi: c->crv[(static_cast<int>(o.d >> 2)) & 7] = PpcCrvCompare(r[o.a], o.imm, c->xer); break;
    case K::Cmpw: c->crv[(static_cast<int>(o.d >> 2)) & 7] = PpcCrvCompare(static_cast<int32_t>(r[o.a]), static_cast<int32_t>(r[o.b]), c->xer); break;
    case K::Cmplw: c->crv[(static_cast<int>(o.d >> 2)) & 7] = PpcCrvCompare(r[o.a], r[o.b], c->xer); break;
    case K::Add: r[o.d] = r[o.a] + r[o.b]; break;
    case K::Subf: r[o.d] = r[o.b] - r[o.a]; break;
    case K::Neg: r[o.d] = 0u - r[o.a]; break;
    case K::Mullw: r[o.d] = r[o.a] * r[o.b]; break;
    case K::Or: r[o.a] = r[o.d] | r[o.b]; break;
    case K::OrRc: r[o.a] = r[o.d] | r[o.b]; SetCr0(c, r[o.a]); break;
    case K::And: r[o.a] = r[o.d] & r[o.b]; break;
    case K::Xor: r[o.a] = r[o.d] ^ r[o.b]; break;
    case K::Andc: r[o.a] = r[o.d] & ~r[o.b]; break;
    case K::Nor: r[o.a] = ~(r[o.d] | r[o.b]); break;
    case K::Slw: r[o.a] = PPC_Slw(r[o.d], r[o.b]); break;
    case K::Srw: r[o.a] = PPC_Srw(r[o.d], r[o.b]); break;
    case K::Srawi: {
        const uint32_t sh = o.b;
        const int32_t s = static_cast<int32_t>(r[o.d]);
        SetCa(c, s < 0 && sh != 0 && (static_cast<uint32_t>(s) & ((1u << sh) - 1u)) != 0);
        r[o.a] = static_cast<uint32_t>(s >> sh);
        break;
    }
    case K::Extsb: r[o.a] = static_cast<uint32_t>(static_cast<int8_t>(r[o.d])); break;
    case K::Extsh: r[o.a] = static_cast<uint32_t>(static_cast<int16_t>(r[o.d])); break;
    case K::Cntlzw: r[o.a] = PPC_CntlzwInline(r[o.d]); break;
    case K::Mflr: r[o.d] = c->lr; break;
    case K::Mtlr: c->lr = r[o.d]; break;
    case K::Mfctr: r[o.d] = c->ctr; break;
    case K::Mtctr: c->ctr = r[o.d]; break;
    case K::Mfcr: r[o.d] = PpcGetCr(c); break;
    case K::AndRc: r[o.a] = r[o.d] & r[o.b]; SetCr0(c, r[o.a]); break;
    case K::Subfic: {
        bool carry, ov;
        r[o.d] = AddCarry(~r[o.a], o.imm, 1, carry, ov);
        SetCa(c, carry);
        break;
    }
    case K::Rlwnm: r[o.a] = PpcRotl32Inline(r[o.d], r[o.b] & 31u) & o.imm; break;
    case K::RlwnmRc: r[o.a] = PpcRotl32Inline(r[o.d], r[o.b] & 31u) & o.imm; SetCr0(c, r[o.a]); break;
    case K::Mfspr: r[o.d] = ReadSpr(c, o.imm, PcOf(at), 0); break;
    case K::Mtspr: WriteSpr(c, o.imm, r[o.d]); break;
    case K::Lwz: r[o.d] = MemoryInline::FlatRead32(BaseD(c, o)); break;
    case K::Lwzu: { const uint32_t ea = BaseD(c, o); r[o.d] = MemoryInline::FlatRead32(ea); r[o.a] = ea; break; }
    case K::Lbz: r[o.d] = MemoryInline::FlatRead8(BaseD(c, o)); break;
    case K::Lbzu: { const uint32_t ea = BaseD(c, o); r[o.d] = MemoryInline::FlatRead8(ea); r[o.a] = ea; break; }
    case K::Lhz: r[o.d] = MemoryInline::FlatRead16(BaseD(c, o)); break;
    case K::Lhzu: { const uint32_t ea = BaseD(c, o); r[o.d] = MemoryInline::FlatRead16(ea); r[o.a] = ea; break; }
    case K::Lha: r[o.d] = static_cast<uint32_t>(static_cast<int16_t>(MemoryInline::FlatRead16(BaseD(c, o)))); break;
    case K::Stw: MemoryInline::FlatWrite32(BaseD(c, o), r[o.d]); break;
    case K::Stwu: { const uint32_t ea = BaseD(c, o); MemoryInline::FlatWrite32(ea, r[o.d]); r[o.a] = ea; break; }
    case K::Stb: MemoryInline::FlatWrite8(BaseD(c, o), static_cast<uint8_t>(r[o.d])); break;
    case K::Stbu: { const uint32_t ea = BaseD(c, o); MemoryInline::FlatWrite8(ea, static_cast<uint8_t>(r[o.d])); r[o.a] = ea; break; }
    case K::Sth: MemoryInline::FlatWrite16(BaseD(c, o), static_cast<uint16_t>(r[o.d])); break;
    case K::Sthu: { const uint32_t ea = BaseD(c, o); MemoryInline::FlatWrite16(ea, static_cast<uint16_t>(r[o.d])); r[o.a] = ea; break; }
    case K::Lmw: { uint32_t ea = BaseD(c, o); for (uint32_t k = o.d; k < 32; ++k, ea += 4) r[k] = MemoryInline::FlatRead32(ea); break; }
    case K::Stmw: { uint32_t ea = BaseD(c, o); for (uint32_t k = o.d; k < 32; ++k, ea += 4) MemoryInline::FlatWrite32(ea, r[k]); break; }
    case K::Lwzx: r[o.d] = MemoryInline::FlatRead32(BaseX(c, o)); break;
    case K::Lbzx: r[o.d] = MemoryInline::FlatRead8(BaseX(c, o)); break;
    case K::Lhzx: r[o.d] = MemoryInline::FlatRead16(BaseX(c, o)); break;
    case K::Stwx: MemoryInline::FlatWrite32(BaseX(c, o), r[o.d]); break;
    case K::Stbx: MemoryInline::FlatWrite8(BaseX(c, o), static_cast<uint8_t>(r[o.d])); break;
    case K::Sthx: MemoryInline::FlatWrite16(BaseX(c, o), static_cast<uint16_t>(r[o.d])); break;
    case K::Lfs: SetScalar(f, o.d, MemoryInline::FlatReadFloat32(BaseD(c, o))); break;
    case K::Lfsu: { const uint32_t ea = BaseD(c, o); SetScalar(f, o.d, MemoryInline::FlatReadFloat32(ea)); r[o.a] = ea; break; }
    case K::Lfd: SetScalar(f, o.d, MemoryInline::FlatReadFloat64(BaseD(c, o))); break;
    case K::Lfdu: { const uint32_t ea = BaseD(c, o); SetScalar(f, o.d, MemoryInline::FlatReadFloat64(ea)); r[o.a] = ea; break; }
    case K::Stfs: MemoryInline::FlatWriteFloat32(BaseD(c, o), Scalar(f, o.d)); break;
    case K::Stfsu: { const uint32_t ea = BaseD(c, o); MemoryInline::FlatWriteFloat32(ea, Scalar(f, o.d)); r[o.a] = ea; break; }
    case K::Stfd: MemoryInline::FlatWriteFloat64(BaseD(c, o), Scalar(f, o.d)); break;
    case K::Stfdu: { const uint32_t ea = BaseD(c, o); MemoryInline::FlatWriteFloat64(ea, Scalar(f, o.d)); r[o.a] = ea; break; }
    case K::Lfsx: SetScalar(f, o.d, MemoryInline::FlatReadFloat32(BaseX(c, o))); break;
    case K::Stfsx: MemoryInline::FlatWriteFloat32(BaseX(c, o), Scalar(f, o.d)); break;
    case K::PsqL: SetPaired(f, o.d, PPC_PsqL(BaseD(c, o), o.b >> 3, o.b & 7u)); break;
    case K::PsqLu: { const uint32_t ea = BaseD(c, o); SetPaired(f, o.d, PPC_PsqL(ea, o.b >> 3, o.b & 7u)); r[o.a] = ea; break; }
    case K::PsqSt: PPC_PsqSt(BaseD(c, o), Paired(f, o.d), o.b >> 3, o.b & 7u); break;
    case K::PsqStu: { const uint32_t ea = BaseD(c, o); PPC_PsqSt(ea, Paired(f, o.d), o.b >> 3, o.b & 7u); r[o.a] = ea; break; }
    case K::Fadds: SetScalar(f, o.d, PpcForceSingleValueInline(PpcFaddInline(Scalar(f, o.a), Scalar(f, o.b)))); break;
    case K::Fsubs: SetScalar(f, o.d, PpcForceSingleValueInline(PpcFsubInline(Scalar(f, o.a), Scalar(f, o.b)))); break;
    case K::Fmuls: SetScalar(f, o.d, PpcFmulsInline(Scalar(f, o.a), Scalar(f, o.imm))); break;
    case K::Fdivs: SetScalar(f, o.d, PpcForceSingleValueInline(PpcFdivInline(Scalar(f, o.a), Scalar(f, o.b)))); break;
    case K::Fmadds: SetScalar(f, o.d, PpcFmaddsInline(Scalar(f, o.a), Scalar(f, o.imm), Scalar(f, o.b))); break;
    case K::Fmsubs: SetScalar(f, o.d, PpcFmsubsInline(Scalar(f, o.a), Scalar(f, o.imm), Scalar(f, o.b))); break;
    case K::Fnmsubs: SetScalar(f, o.d, PpcFnmsubsInline(Scalar(f, o.a), Scalar(f, o.imm), Scalar(f, o.b))); break;
    case K::Fnmadds: SetScalar(f, o.d, PpcFnmaddsInline(Scalar(f, o.a), Scalar(f, o.imm), Scalar(f, o.b))); break;
    case K::Fadd: SetScalar(f, o.d, PpcFaddInline(Scalar(f, o.a), Scalar(f, o.b))); break;
    case K::Fsub: SetScalar(f, o.d, PpcFsubInline(Scalar(f, o.a), Scalar(f, o.b))); break;
    case K::Fmul: SetScalar(f, o.d, PpcFmulInline(Scalar(f, o.a), Scalar(f, o.imm))); break;
    case K::Fdiv: SetScalar(f, o.d, PpcFdivInline(Scalar(f, o.a), Scalar(f, o.b))); break;
    case K::Fmadd: SetScalar(f, o.d, PpcFmaddInline(Scalar(f, o.a), Scalar(f, o.imm), Scalar(f, o.b))); break;
    case K::Fmsub: SetScalar(f, o.d, PpcFmsubInline(Scalar(f, o.a), Scalar(f, o.imm), Scalar(f, o.b))); break;
    case K::Fneg: SetScalar(f, o.d, -Scalar(f, o.b)); break;
    case K::Fabs: SetScalar(f, o.d, std::fabs(Scalar(f, o.b))); break;
    case K::Fmr: c->fpr[o.d] = c->fpr[o.b]; CopyForm(f, o.d, o.b); break;
    case K::Frsp: SetScalar(f, o.d, PpcForceSingleValueInline(Scalar(f, o.b))); break;
    case K::Fcmpu: c->crv[(static_cast<int>(o.d >> 2)) & 7] = PpcCrvCompareFloat(Scalar(f, o.a), Scalar(f, o.b)); break;
    case K::PsAdd: SetPaired(f, o.d, PPC_PsAddInline(Paired(f, o.a), Paired(f, o.b))); break;
    case K::PsSub: SetPaired(f, o.d, PPC_PsSubInline(Paired(f, o.a), Paired(f, o.b))); break;
    case K::PsMul: SetPaired(f, o.d, PPC_PsMulInline(Paired(f, o.a), Paired(f, o.imm))); break;
    case K::PsMadd: SetPaired(f, o.d, PPC_PsMaddInline(Paired(f, o.a), Paired(f, o.imm), Paired(f, o.b))); break;
    case K::PsMsub: SetPaired(f, o.d, PPC_PsMsubInline(Paired(f, o.a), Paired(f, o.imm), Paired(f, o.b))); break;
    case K::PsNmadd: SetPaired(f, o.d, PPC_PsNmaddInline(Paired(f, o.a), Paired(f, o.imm), Paired(f, o.b))); break;
    case K::PsNmsub: SetPaired(f, o.d, PPC_PsNmsubInline(Paired(f, o.a), Paired(f, o.imm), Paired(f, o.b))); break;
    case K::PsMuls0: SetPaired(f, o.d, PPC_PsMuls0Inline(Paired(f, o.a), Paired(f, o.imm))); break;
    case K::PsMuls1: SetPaired(f, o.d, PPC_PsMuls1Inline(Paired(f, o.a), Paired(f, o.imm))); break;
    case K::PsMadds0: SetPaired(f, o.d, PPC_PsMadds0Inline(Paired(f, o.a), Paired(f, o.imm), Paired(f, o.b))); break;
    case K::PsMadds1: SetPaired(f, o.d, PPC_PsMadds1Inline(Paired(f, o.a), Paired(f, o.imm), Paired(f, o.b))); break;
    case K::PsSum0: SetPaired(f, o.d, PPC_PsSum0Inline(Paired(f, o.a), Paired(f, o.b), Paired(f, o.imm))); break;
    case K::PsSum1: SetPaired(f, o.d, PPC_PsSum1Inline(Paired(f, o.a), Paired(f, o.b), Paired(f, o.imm))); break;
    case K::PsMerge00: SetPaired(f, o.d, PPC_PsMerge00Inline(Paired(f, o.a), Paired(f, o.b))); break;
    case K::PsMerge01: SetPaired(f, o.d, PPC_PsMerge01Inline(Paired(f, o.a), Paired(f, o.b))); break;
    case K::PsMerge10: SetPaired(f, o.d, PPC_PsMerge10Inline(Paired(f, o.a), Paired(f, o.b))); break;
    case K::PsMerge11: SetPaired(f, o.d, PPC_PsMerge11Inline(Paired(f, o.a), Paired(f, o.b))); break;
    case K::PsMr: c->fpr[o.d] = c->fpr[o.b]; CopyForm(f, o.d, o.b); break;
    case K::PsNeg: SetPaired(f, o.d, PPC_PsNegInline(Paired(f, o.b))); break;
    case K::PsAbs: SetPaired(f, o.d, PPC_PsAbsInline(Paired(f, o.b))); break;
    default: MKW_INTERP_UNREACHABLE();  // Build makes every op: no range check in the dispatch
    }
    return Act::Next;
}

// A plain op on its own (the verifier).
inline void RunPlain(Frame& f, const Op& o, uint32_t pc) {
    Cursor at{&o, pc, 0, 0, nullptr};
    (void)RunOp(f, f.ctx, at);
}

// BRAWL_INTERP_VERIFY=1: the handler against Exec on a copy of the registers. Stores are left out (Exec would store
// a second time).
void VerifyOp(Frame& f, const Op& o, uint32_t pc, uint32_t w) {
    if (IsStore(o.kind)) { RunPlain(f, o, pc); return; }
    CpuContext shadow = *f.ctx;
    Frame g(&shadow, pc);
    g.pairedMask = f.pairedMask;
    uint32_t next = pc + 4;
    (void)Exec(g, pc, w, next, 0);
    RunPlain(f, o, pc);
    const CpuContext& c = *f.ctx;
    bool same = std::memcmp(c.gpr, shadow.gpr, sizeof(c.gpr)) == 0 && PpcGetCr(&c) == PpcGetCr(&shadow) && c.lr == shadow.lr &&
                c.ctr == shadow.ctr && c.xer == shadow.xer && f.pairedMask == g.pairedMask;
    for (int r = 0; same && r < 32; ++r) same = c.fpr[r].raw == shadow.fpr[r].raw;
    if (!same) {
        static std::mutex mutex;
        static std::unordered_set<uint32_t> logged;
        std::lock_guard<std::mutex> lock(mutex);
        if (logged.size() < 256 && logged.insert(pc).second)
            RT_LOGF(RT_TAG_RUNTIME, "interpreter verify: handler %u for %08X at 0x%08X differs from Exec\n",
                    static_cast<unsigned>(o.kind), w, pc);
    }
}

// Runs block b. next: where to continue; Return when the frame is done.
template <bool Verify>
MKW_PPC_FORCE_INLINE Step RunBlock(Frame& f, const Block& b, uint32_t& next, uint32_t returnAddress) {
    // Ends at the block's End op at the latest: no count to check.
    Cursor at{b.ops, 0, 0, returnAddress, &b};
    CpuContext* const ctx = f.ctx;
    for (;; ++at.op) {
        if (Verify) {
            // the verifier runs a fused compare as the compare alone; its bc is the next op
            const K plain = Unfused(at.op->kind);
            if (Plain(plain)) {
                Op o = *at.op;
                o.kind = plain;
                VerifyOp(f, o, PcOf(at), b.words[at.op - b.ops]);
                continue;
            }
        }
        switch (RunOp(f, ctx, at)) {
        case Act::Next: continue;
        case Act::Exit: next = at.next; return Step::Continue;
        case Act::Return: return Step::Return;
        case Act::Exec: break;
        }
        const uint32_t pc = PcOf(at);
        f.ctx->pc = pc;
        uint32_t to = pc + 4;
        if (Exec(f, pc, b.words[at.op - b.ops], to, returnAddress) == Step::Return) return Step::Return;
        if (to != pc + 4) { next = to; return Step::Continue; }
    }
}

// The verifier's loop, kept out of Run (Run inlines the plain one).
MKW_PPC_NO_INLINE Step RunBlockVerify(Frame& f, const Block& b, uint32_t& next, uint32_t returnAddress) {
    return RunBlock<true>(f, b, next, returnAddress);
}

}  // namespace

namespace detail {

bool JitGuardedLine(uint32_t segmentStart, uint32_t mark) noexcept {
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
    if (BrawlDirtyMemory::guardedWritten == nullptr) return false;
    // the line's address in the segment's own view (its top bits), as the write guard pages are registered
    const uint32_t address = (segmentStart & ~MemoryInline::kLeanMask) | (mark << 6);
    if (RecompMod::g_executableWriteGuardPages[address >> RecompMod::kExecutableWriteGuardPageShift].load(std::memory_order_relaxed) == 0)
        return false;
    return MemoryInline::LeanStoreNeedsPolicy(address);
#else
    (void)segmentStart;
    (void)mark;
    return false;
#endif
}

void JitSweepLinks() noexcept {
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
    if (__builtin_expect(g_jitEpochFault == 2, 0)) return;
    BlockCache::NoteEpoch(BrawlDirtyMemory::markEpoch.load(std::memory_order_acquire));
#endif
}

uint64_t JitFallbackOp(Frame* f, const Block* b, uint32_t index) noexcept {
    const auto result = [](uint32_t what, uint32_t next) { return (uint64_t{what} << 32) | next; };
    if (diag::ChurnOn()) diag::Fallback(b->pcs[index], static_cast<uint32_t>(b->ops[index].kind), b->words[index]);
    try {
        // RunBlock<false>'s loop body for one op
        Cursor at{b->ops + index, 0, 0, f->returnAddress, b};
        switch (RunOp(*f, f->ctx, at)) {
        case Act::Next: JitNoteEpoch(); return result(0, static_cast<uint32_t>(at.op - b->ops) + 1);
        case Act::Exit: return result(1, at.next);
        case Act::Return: return result(2, 0);
        case Act::Exec: break;
        }
        const uint32_t pc = PcOf(at);
        f->ctx->pc = pc;
        uint32_t to = pc + 4;
        const Step step = Exec(*f, pc, b->words[at.op - b->ops], to, f->returnAddress);
        JitNoteEpoch();  // Exec may have run guest code (a call): the block goes on and may follow its link slots
        if (step == Step::Return) return result(2, 0);
        if (to != pc + 4) return result(1, to);
        return result(0, static_cast<uint32_t>(at.op - b->ops) + 1);
    } catch (...) {
        f->pending = std::current_exception();
        return result(3, 0);
    }
}

uint64_t JitCallOp(Frame* f, const Block* b, uint32_t index) noexcept {
    const auto result = [](uint32_t what, uint32_t value) { return (uint64_t{what} << 32) | value; };
    try {
        const Op& o = b->ops[index];
        const uint32_t pc = b->pcs[index];
        f->ctx->pc = pc;
        // Not made, the block leaves through a link slot or an inline cache with no validity test of its own first, and Call
        // may have run BrawlRollback::LoopCall (the loop frame): JitNoteEpoch. Made: the block's EmitRecheck follows.
        if (o.kind == K::Bl) {
            if (!Call(*f, o.imm, pc + 4, o.b != 0 ? &b->calls[o.b - 1] : nullptr)) {
                JitNoteEpoch();
                return result(1, o.imm);
            }
        } else {
            const uint32_t target = f->ctx->ctr & ~3u;
            if (!Call(*f, target, pc + 4)) {
                JitNoteEpoch();
                return result(1, target);
            }
        }
        return 0;
    } catch (...) {
        f->pending = std::current_exception();
        return uint64_t{3} << 32;
    }
}

uint64_t JitTailOp(Frame* f, const Block* b, uint32_t index) noexcept {
    const auto result = [](uint32_t what, uint32_t value) { return (uint64_t{what} << 32) | value; };
    try {
        const Op& o = b->ops[index];
        CpuContext* const c = f->ctx;
        const uint32_t target = o.kind == K::B ? o.imm : c->ctr & ~3u;
        if (TranslatedFunctionRegistry::FindByAddressPtr(target) == nullptr) {
            return o.kind == K::B ? result(1, target) : result(4, target);
        }
        c->pc = b->pcs[index];
        uint32_t next = 0;
        if (LoopTailCall(*f, target, next)) return result(1, next);
        if (TailCall(*f, target, f->returnAddress)) return uint64_t{2} << 32;
        return result(1, c->lr & ~3u);
    } catch (...) {
        f->pending = std::current_exception();
        return uint64_t{3} << 32;
    }
}

void JitNormalizeF1(Frame* f) noexcept { SetScalar(*f, 1, Scalar(*f, 1)); }

const void* const* JitRollbackLoopFrameCell() noexcept {
    return reinterpret_cast<const void* const*>(&g_rollbackLoopFrame);
}

// The count of code variants this build links, or 0. A build without them (the interpreter tests) does not define the
// symbol, which a weak reference tolerates; Windows' linker has no weak undefined references, so there the tests define
// it themselves (tests/interpreter_test_support.cpp).
#if defined(_WIN32)
extern "C" const uint32_t g_mkwCodeVariantCount;
#else
extern "C" const uint32_t g_mkwCodeVariantCount __attribute__((weak));
#endif

bool JitContinuationsPossible() noexcept {
    static const bool possible = [] {
#if defined(_WIN32)
        if (g_mkwCodeVariantCount == 0) return false;
#else
        if (&g_mkwCodeVariantCount == nullptr || g_mkwCodeVariantCount == 0) return false;
#endif
        return std::getenv("BRAWL_NO_CODE_VARIANTS") == nullptr && std::getenv("BRAWL_INTERPRET") == nullptr;
    }();
    return possible;
}

bool JitContinuationAt(uint32_t pc) noexcept { return GuestCodePatch::ContinuationAt(pc) != nullptr; }

#if MKW_JIT
const JitBackend* Backend() { return JitX64Backend(); }
#else
const JitBackend* Backend() { return nullptr; }
#endif

}  // namespace detail

namespace {

enum class Mode : uint8_t { Cached, Classic, Verify, Jit };

// SetEngineOverride's engine (an Engine), or -1 for the environment's choice; SetJitThreshold's count.
std::atomic<int> g_engineOverride{-1};
std::atomic<uint32_t> g_jitThreshold{2};
std::atomic<bool> g_fastJitReady{false};
EnterDirectFn g_enterDirect = nullptr;

// The JIT's engine when this build has a backend, the cached interpreter otherwise.
Mode JitOrCached() {
    static const bool jit = Backend() != nullptr;
    return jit ? Mode::Jit : Mode::Cached;
}

// The engine the environment chooses (what InterpreterMode answers with no override set).
Mode DefaultMode() {
    static const Mode mode = [] {
        const char* interp = std::getenv("BRAWL_INTERP");
        if (interp != nullptr && std::strcmp(interp, "classic") == 0) return Mode::Classic;
        const char* verify = std::getenv("BRAWL_INTERP_VERIFY");
        if (verify != nullptr && *verify && *verify != '0') return Mode::Verify;
        const char* jit = std::getenv("BRAWL_JIT");
        if (jit != nullptr && std::strcmp(jit, "0") == 0) return Mode::Cached;
        return JitOrCached();
    }();
    return mode;
}

Mode InterpreterMode() {
    switch (g_engineOverride.load(std::memory_order_relaxed)) {
    case static_cast<int>(Engine::Classic): return Mode::Classic;
    case static_cast<int>(Engine::Verify): return Mode::Verify;
    case static_cast<int>(Engine::Cached): return Mode::Cached;
    case static_cast<int>(Engine::Jit): return JitOrCached();
    default: break;
    }
    return DefaultMode();
}

// Compiles a block the dispatcher has entered often enough (once: a block the backend turns down stays interpreted).
MKW_PPC_NO_INLINE void CompileBlock(BlockCache& cache, const JitBackend& backend, Block& b) {
    if (diag::ChurnOn()) {
        diag::Add(diag::kCompileCalls);
        diag::AddKey(diag::kTCompile, b.start);
    }
    if (cache.jit == nullptr) cache.jit = backend.createCache();
    b.jitState = (cache.jit != nullptr && backend.compile(*cache.jit, b) && b.jitEntry != nullptr) ? 1 : 2;
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
    if (b.jitState == 1 && b.jitGuarded) BlockCache::NoteGuardedBlock(b);
#endif
    cache.UpdateOverLimit();
}

// Everything Run keeps per thread, behind one plain thread_local pointer. A thread_local with a constructor or
// destructor is reached through a wrapper call on every access (and emulated TLS makes any access a call), and Run is
// entered millions of times a second from translated code. Allocated on a thread's first Run, kept for its life.
struct ThreadState {
    DirectEnterState directState{};
    std::unordered_set<uint32_t> seen;  // entries logged
    uint32_t recentEntries[1024] = {};
    BlockCache cache;
};
thread_local ThreadState* t_state = nullptr;

MKW_PPC_NO_INLINE ThreadState* NewState() {
    InstructionCounter* c = t_counter;
    if (c == nullptr) c = NewCounter();
    static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t) && alignof(std::atomic<uint64_t>) == alignof(uint64_t));
    ThreadState* const s = t_state = new ThreadState;
    s->directState.insnCount = reinterpret_cast<uint64_t*>(&c->count);
    s->directState.threadState = s;
    s->directState.fastEntries = s->cache.fastEntries;
    return s;
}

MKW_PPC_FORCE_INLINE ThreadState& State() {
    ThreadState* s = t_state;
    if (s == nullptr) s = NewState();
    return *s;
}

}  // namespace

void DumpRecentBranches() {
    const uint32_t count = std::min(t_recentBranchCount, kRecentBranches);
    std::string trail;
    char item[12];
    for (uint32_t i = 0; i < count; ++i) {
        std::snprintf(item, sizeof(item), " %08X", t_recentBranches[(t_recentBranchCount - count + i) % kRecentBranches]);
        trail += item;
    }
    RT_LOGF(RT_TAG_RUNTIME, "interpreter: last %u branch targets on this thread, oldest first:%s\n", count, trail.c_str());
}

bool Enabled() { return g_enabled.load(std::memory_order_relaxed); }
bool JitAvailable() { return Backend() != nullptr; }
void SetEngineOverride(Engine engine) {
    g_engineOverride.store(static_cast<int>(engine), std::memory_order_relaxed);
    g_fastJitReady.store(false, std::memory_order_release);
}
void ClearEngineOverride() {
    g_engineOverride.store(-1, std::memory_order_relaxed);
    g_fastJitReady.store(false, std::memory_order_release);
}
void SetJitThreshold(uint32_t entries) { g_jitThreshold.store(entries, std::memory_order_relaxed); }
void SetEnabled(bool enabled) { g_enabled.store(enabled, std::memory_order_relaxed); }
uint64_t InstructionCount() {
    std::lock_guard<std::mutex> lock(g_countersMutex);
    uint64_t total = 0;
    for (const InstructionCounter* c : g_counters) total += c->count.load(std::memory_order_relaxed);
    return total;
}
// Code runs at virtual addresses; a call to a low one is a bad pointer (an unrelocated offset), reported as missing.
bool CanRun(uint32_t address) {
    return address >= 0x80000000u && (address & 3u) == 0 && Memory::Contains(address, 4);
}

namespace {

// The thread's rollback loop frames running (Run's, and what its RunLoop reads).
thread_local unsigned t_rollbackLoopDepth = 0;

// What Run decides once per process, at its first call (the function-local statics it used to hold).
struct RunFlags {
    bool coreEnabled, rollbackEnabled;
};
const RunFlags& RunFlagsOnce() {
    static const RunFlags flags = [] {
        RunFlags r{};
        r.coreEnabled = BrawlOnlineCore::Enabled();
        r.rollbackEnabled = std::getenv("BRAWL_ROLLBACK_INPUT") != nullptr || std::getenv("BRAWL_ONLINE_IPC") != nullptr || r.coreEnabled;
        return r;
    }();
    return flags;
}

// Each entry is logged once; the per-thread set keeps the lock off the path of every later call, and a small
// direct-mapped cache of recent entries keeps the set's hashing off it (0.66% of P+'s game thread).
MKW_PPC_NO_INLINE void NoteNewEntry(ThreadState& state, CpuContext* ctx, uint32_t entry) {
    if (state.seen.insert(entry).second) {
        static std::mutex logMutex;
        static std::unordered_set<uint32_t> logged;
        std::lock_guard<std::mutex> lock(logMutex);
        if (logged.size() < 4096 && logged.insert(entry).second) {
            RT_LOGF(RT_TAG_RUNTIME, "interpreter: running untranslated guest code at 0x%08X (called from 0x%08X)\n",
                    entry, ctx->lr);
        }
    }
}
MKW_PPC_FORCE_INLINE void NoteEntry(ThreadState& state, CpuContext* ctx, uint32_t entry) {
    uint32_t& recentEntry = state.recentEntries[(entry >> 2) & 1023u];
    if (recentEntry != entry) {
        recentEntry = entry;
        NoteNewEntry(state, ctx, entry);
    }
}

// The frame a Run starts with. Callers hand FPRs over in the scalar CpuContext form (the translator's entry contract), so a float
// argument consumed by a paired op is splatted to both lanes, as lfs/fadds leave it on the hardware.
MKW_PPC_FORCE_INLINE void StartFrame(Frame& f, CpuContext* ctx) {
    f.pairedMask = 0;
    f.returnAddress = ctx->lr & ~3u;  // every user compares it with a word address; compiled code compares it unmasked
}

// The per-instruction bookkeeping touches these thread_locals on every guest instruction. Resolved once per Run: with emulated TLS,
// every thread_local access is otherwise a runtime call.
struct Trail {
    uint32_t* const branches;
    uint32_t& count;
    uint32_t& lastPc;
    Trail() : branches(t_recentBranches), count(t_recentBranchCount), lastPc(t_lastPc) {}
};

// The dispatcher's step for a block with code: compiled code runs this block and every one it is linked to, and counts its own
// instructions. Rethrows what a helper compiled code called stored in the frame.
MKW_PPC_FORCE_INLINE JitExit EnterCompiled(const JitBackend& backend, CpuContext* ctx, Frame& f, Block& b, uint64_t& executed, const Trail& trail) {
    const uint32_t first = b.start;
    ctx->pc = first;
    if (first != trail.lastPc + 4) trail.branches[trail.count++ % kRecentBranches] = first;
    trail.lastPc = b.end;
    f.executed = 0;
    if (diag::ChurnOn()) {
        diag::Add(diag::kEnters);
        diag::AddKey(diag::kTEnter, b.start);
    }
    const JitExit exit = backend.enter(f, b);
    executed += f.executed;
    if (exit.status == JitStatus::Pending) {
        std::exception_ptr pending = std::move(f.pending);
        f.pending = nullptr;
        CountInstructions(executed);
        std::rethrow_exception(pending);
    }
    return exit;
}

// After compiled code left for exit.next: the block to go on with, linked from the slot the exit left through; null when it starts
// at a hook, which the dispatcher's outer loop handles (f.pc is then where to go on).
MKW_PPC_FORCE_INLINE Block* FollowExit(BlockCache& cache, const JitBackend& backend, Frame& f, const JitExit& exit) {
    if (diag::ChurnOn()) diag::AddKey(diag::kTFind, (uint64_t{diag::kFromFollow} << 32) | exit.next);
    Block* const following = cache.Find(exit.next);
    if (following->hook) {
        f.pc = exit.next;
        return nullptr;
    }
    if (exit.link != nullptr && following->jitEntry != nullptr) {
        diag::Add(diag::kFollowLinks);
        backend.link(*exit.link, *following);
    }
    return following;
}

// What Run (and the lean entry, which has run the first block already) hands the dispatcher's loop.
struct LoopEnv {
    CpuContext* ctx;
    Frame& f;
    uint64_t& executed;
    BlockCache* cache;
    bool stats;
    Mode mode;
    const JitBackend* jitBackend;
    uint32_t jitThreshold;
    Block* first;  // a block already found, compiled and linked for the next pc (not a hook), or null: Find(f.pc)
#if MKW_INTERP_PROFILE
    prof::Table* profileTable;
#endif
};

// The loop of Run: hook pcs, then blocks one after another until the frame returns.
MKW_PPC_NO_INLINE void RunLoop(const LoopEnv& env) {
    CpuContext* const ctx = env.ctx;
    Frame& f = env.f;
    uint64_t& executed = env.executed;
    BlockCache* const cache = env.cache;
    const bool stats = env.stats;
    const Mode mode = env.mode;
    const JitBackend* const jitBackend = env.jitBackend;
    const uint32_t jitThreshold = env.jitThreshold;
    Block* first = env.first;
#if MKW_INTERP_PROFILE
    prof::Table* const profileTable = env.profileTable;
#endif
    const auto& tracePcs = TracePcs();
    const RunFlags& flags = RunFlagsOnce();
    const bool coreEnabled = flags.coreEnabled, rollbackEnabled = flags.rollbackEnabled;
    unsigned& loopDepth = t_rollbackLoopDepth;
    const Trail trail;
    uint32_t* const recentBranches = trail.branches;
    uint32_t& recentBranchCount = trail.count;
    uint32_t& lastPc = trail.lastPc;
    for (;;) {
        const uint32_t pc = f.pc;
        if(pc==0x80949BC0u) BrawlOnline::Stage();
        if(pc==0x809435F0u) BrawlRollback::SeedCrossplayStage();
        // Brawl Online's complete main-loop boundary, including pad bookkeeping,
        // render-side game writes and the application frame counter.
        if(rollbackEnabled && pc==0x800171B4u && !loopDepth) {
            MKW_INTERP_PROFILE_PAUSE();  // the rollback session's own time; each frame's Run below is its own entry
            BrawlRollback::RunLoop(ctx,[](CpuContext* c) {
                ++t_rollbackLoopDepth;Run(c,0x800171B4u);--t_rollbackLoopDepth;
            });
            f.pairedMask = 0;
            continue;
        }
        // Brawl Online's core (Dolphin's Brawlback hooks): the frame end inside the pass.
        if(coreEnabled && loopDepth && pc==0x80017504u) {
            for(uint32_t m=f.pairedMask; m!=0; m&=m-1u) { unsigned i=__builtin_ctz(m); SetScalar(f,i,Scalar(f,i)); }
            if(BrawlOnlineCore::FrameEnd(ctx)) {f.pc=0x80017508u;continue;}
        }
        if(loopDepth && pc==0x80017508u) {
            if (diag::ChurnOn() && cache != nullptr) {
                const uint64_t counts[diag::kCacheCounters] = {cache->built, cache->rebuilt, cache->slotMisses, cache->current,
                                                               cache->linked, cache->claims, cache->retired, cache->epochs,
                                                               cache->partial, cache->linkRefreshes, cache->clears};
                diag::Pass(counts);
            }
            for(uint32_t m=f.pairedMask; m!=0; m&=m-1u) { unsigned i=__builtin_ctz(m); SetScalar(f,i,Scalar(f,i)); }
            if(coreEnabled) BrawlOnlineCore::LoopEnd(ctx);
            CountInstructions(executed);return;
        }
        if(loopDepth && BrawlRollback::DrivesLoop()) {
            if(pc==0x800172D0u) ctx->gpr[20]=1;
            if(pc==0x8001730Cu) {
                ctx->gpr[20]=1;
                Memory::Write32(ctx->gpr[23]+0xFC,Memory::Read16(ctx->gpr[23]+0xF8));
            }
        }
        // Brawl Online's sound hooks (Dolphin's HLE_Misc sound alloc, attach and stop hooks, at the same pcs): the
        // game's detail_SetupSound and BasicSound::Stop run here while the core is on (BrawlOnlineCore::Start).
        if(coreEnabled && (pc==0x801C9AACu || pc==0x801C9B54u || pc==0x801C9BFCu || pc==0x801C9C70u || pc==0x801BC684u)) {
            const uint32_t to=BrawlOnlineCore::SoundHook(ctx,pc);
            if(to!=pc) {
                if(pc==f.entry && (to&~3u)==(f.returnAddress&~3u)) {  // BasicSound::Stop declined: back to its caller
                    SetScalar(f,1,Scalar(f,1));
                    CountInstructions(executed);
                    return;
                }
                f.pc=to;
                continue;
            }
        }
        if (cache != nullptr) {
            // Blocks run one after another here, each found through the last one's links, until the frame returns
            // or the next block starts at a hook (which the loop above handles first).
            if (diag::ChurnOn() && first == nullptr) diag::AddKey(diag::kTFind, (uint64_t{diag::kFromLoop} << 32) | pc);
            Block* b = first != nullptr ? first : cache->Find(pc);
            first = nullptr;
            for (;;) {
                if (jitBackend != nullptr) {
                    if (b->jitEntry == nullptr && b->jitState == 0 && ++b->entries > jitThreshold)
                        CompileBlock(*cache, *jitBackend, *b);
                    if (b->jitEntry != nullptr && JitBlocksMayRun(ctx)) {
                        const JitExit exit = EnterCompiled(*jitBackend, ctx, f, *b, executed, trail);
                        if (exit.status == JitStatus::Return) {
                            CountInstructions(executed);
                            return;
                        }
                        Block* const following = FollowExit(*cache, *jitBackend, f, exit);
                        if (following == nullptr) break;
                        b = following;
                        continue;
                    }
                }
                const uint32_t start = b->start;
                if (diag::ChurnOn() && jitBackend != nullptr) {
                    diag::Add(diag::kInterpBlocks);
                    diag::Add(diag::kInterpOps, b->count);
                    diag::AddKey(diag::kTInterpBlock, start | (b->jitState == 2 ? uint64_t{1} << 32 : 0));
                }
                if (stats) {
                    ++cache->blocksRun;
                    cache->opsRun += b->count;
                    for (uint32_t i = 0; i < b->count; ++i) cache->genericRun += b->ops[i].kind == K::Generic;
                }
                ctx->pc = start;
                if (start != lastPc + 4) recentBranches[recentBranchCount++ % kRecentBranches] = start;
                lastPc = b->end;
                executed += b->count;
                uint32_t next = start;
                const Step step = mode == Mode::Verify ? RunBlockVerify(f, *b, next, f.returnAddress)
                                                       : RunBlock<false>(f, *b, next, f.returnAddress);
                if (stats) cache->Count(*b, step, next);
                if (step == Step::Return) {
                    CountInstructions(executed);
                    return;
                }
                Block* const following = cache->Next(*b, next);
                if (following->hook) {
                    f.pc = next;
                    break;
                }
                b = following;
            }
            continue;
        }
#if MKW_INTERP_PROFILE
        if (profileTable != nullptr) prof::CountPc(profileTable, pc);
#endif
        const uint32_t w = MemoryInline::FlatRead32(pc);
        ctx->pc = pc;
        ++executed;
        if (pc != lastPc + 4) recentBranches[recentBranchCount++ % kRecentBranches] = pc;
        lastPc = pc;
        if (!tracePcs.empty() && std::find(tracePcs.begin(), tracePcs.end(), pc) != tracePcs.end()) TracePc(ctx, pc);
        uint32_t next = pc + 4;
        if (Exec(f, pc, w, next, f.returnAddress) == Step::Return) {
            CountInstructions(executed);
            return;
        }
        f.pc = next;
    }
}

}  // namespace

// Every entry first tries what GuestCodePatch::RunPatched does for a patched function (RunCompiledEntry: compiled code entered
// through jit_enter_direct when the entry's first block has code, is current and is no hook pc); the general entry otherwise.
// Translated code's calls of untranslated code (a mod's module, through RunUntranslatedOrFail) come here.
void Run(CpuContext* ctx, uint32_t entry) {
    if (detail::RunCompiledEntry(ctx, entry)) return;
    detail::RunGeneral(ctx, entry);
}

void detail::RunGeneral(CpuContext* ctx, uint32_t entry) {
    HZ_EXCLUSIVE(kHzInterpTicks);
    if (diag::ChurnOn()) {
        diag::Add(diag::kRuns);
        diag::AddKey(diag::kTRun, entry);
    }
    ThreadState& state = State();
    NoteEntry(state, ctx, entry);
    Frame f(ctx, entry);
    StartFrame(f, ctx);
    std::optional<CpuContextScope> cpuContextScope;
    if (__builtin_expect(TryGetCpuContext() != ctx, 0)) cpuContextScope.emplace(ctx);
    uint64_t executed = 0;
    const auto& tracePcs = TracePcs();
    unsigned& loopDepth = t_rollbackLoopDepth;  // only on the game thread
    const bool rollbackLoopFrame = loopDepth && entry == 0x800171B4u;
    if (rollbackLoopFrame) f.flags |= kFrameFlagLoopFrame;
    struct LoopFrameScope {
        bool owner;
        const Frame* saved;
        ~LoopFrameScope() { if (owner) g_rollbackLoopFrame.store(saved, std::memory_order_relaxed); }
    } loopFrameScope{rollbackLoopFrame, g_rollbackLoopFrame.load(std::memory_order_relaxed)};
    if (rollbackLoopFrame) g_rollbackLoopFrame.store(&f, std::memory_order_relaxed);
#if MKW_INTERP_PROFILE
    std::optional<prof::Scope> profile;
    if (prof::Prefix() != nullptr) profile.emplace(entry, f.returnAddress, executed);
    prof::Table* const profileTable = profile ? profile->table : nullptr;
#endif
    // Tracing needs every pc: it decodes as it goes.
    Mode mode = tracePcs.empty() ? InterpreterMode() : Mode::Classic;
#if MKW_INTERP_PROFILE
    // Exact per-PC diagnostics retain the instruction loop; ordinary builds use cached blocks.
    if (profileTable != nullptr) mode = Mode::Classic;
#endif
    BlockCache* cache = nullptr;
    const bool stats = BlockCache::Stats();
    if (stats && mode == Mode::Jit) mode = Mode::Cached;  // the statistics count every op dispatched
    const JitBackend* const jitBackend = mode == Mode::Jit ? Backend() : nullptr;
    const uint32_t jitThreshold = jitBackend != nullptr ? g_jitThreshold.load(std::memory_order_relaxed) : 0;
    if (mode != Mode::Classic) {
        cache = &state.cache;
        if (cache->depth == 0 && cache->overLimit) cache->Clear();
        ++cache->depth;
        if (stats && (++cache->runs & 0xFFFFFu) == 0) cache->Report();
    }
    struct DepthScope {
        BlockCache* cache;
        ~DepthScope() { if (cache) --cache->depth; }
    } depthScope{cache};
    const LoopEnv env{ctx, f, executed, cache, stats, mode, jitBackend, jitThreshold, nullptr
#if MKW_INTERP_PROFILE
                      , profileTable
#endif
    };
    RunLoop(env);
}

#if MKW_JIT && !MKW_INTERP_PROFILE
void detail::JitDirectContinue(DirectEnterState* ds, Frame* f, uint64_t statusAndNext, JitLink* link) {
    // Clear direct/subcall flags before any RunLoop -> jit_enter call reuses this Frame, so its exits return to Enter().
    const bool wasSubCall = (f->flags & kFrameFlagSubCall) != 0;
    f->flags &= ~(kFrameFlagDirectEnter | kFrameFlagSubCall);
    if (wasSubCall) f->entry = f->pc;  // a subcall frame stores pc = entry only (nothing in compiled code writes either)
    // A subcall frame passes no state (its caller may be any frame of this thread): the thread's own.
    ThreadState& state = ds != nullptr ? *static_cast<ThreadState*>(ds->threadState) : State();
    diag::Add(diag::kDirectContinues);
    BlockCache& cache = state.cache;
    const JitBackend& backend = *Backend();
    if (wasSubCall) NoteEntry(state, f->ctx, f->entry);  // what RunCompiledEntry does for an entry it runs (the log of entries)
    uint64_t executed = f->executed;
    const JitExit exit{static_cast<JitStatus>(statusAndNext >> 32), static_cast<uint32_t>(statusAndNext), link};
    if (exit.status == JitStatus::Pending) {
        std::exception_ptr pending = std::move(f->pending);
        f->pending = nullptr;
        CountInstructions(executed);
        std::rethrow_exception(pending);
    }
    if (exit.status == JitStatus::Return) {
        CountInstructions(executed);
    } else {
        Block* const following = FollowExit(cache, backend, *f, exit);
        const LoopEnv env{f->ctx, *f, executed, &cache, false, Mode::Jit, &backend, g_jitThreshold.load(std::memory_order_relaxed), following};
        RunLoop(env);
    }
    if (wasSubCall) {
        const PatchedHooks& h = GetPatchedHooks();
        if (h.sync != nullptr &&
            (h.pendingFast->load(std::memory_order_acquire) != 0 || h.anyDirty->load(std::memory_order_acquire))) {
            h.sync();
        }
    }
}

MKW_PPC_NO_INLINE bool RunCompiledEntrySlow(CpuContext* ctx, uint32_t entry) {
    diag::Add(diag::kSlowEntries);
    struct Once {
        const JitBackend* backend;
        bool defaultJit;
        bool jitStats;
    };
    static const Once once = [] {
        RunFlagsOnce();
        const bool on = Backend() != nullptr && TracePcs().empty() && !BlockCache::Stats() && !BlockCache::CheckMarks();
        const char* s = std::getenv("BRAWL_JIT_STATS");
        const bool jitStats = s != nullptr && *s != '\0' && *s != '0';
        return Once{on ? Backend() : nullptr, on && DefaultMode() == Mode::Jit, jitStats};
    }();
    if (once.backend == nullptr) return false;
    const int engineOverride = g_engineOverride.load(std::memory_order_relaxed);
    if (engineOverride == -1 ? !once.defaultJit : engineOverride != static_cast<int>(Engine::Jit)) return false;
    const JitBackend& backend = *once.backend;
    if (!once.jitStats && backend.enterDirect != nullptr) {
        g_enterDirect = backend.enterDirect;
        g_fastJitReady.store(true, std::memory_order_release);
    }
    ThreadState& state = State();
    BlockCache& cache = state.cache;
    if (cache.depth == 0 && cache.overLimit) return false;
    if (diag::ChurnOn()) diag::AddKey(diag::kTFind, (uint64_t{diag::kFromEntrySlow} << 32) | entry);
    Block* const b = cache.FindCurrent(entry);
    if (b->hook || b->jitEntry == nullptr || !JitBlocksMayRun(ctx)) return false;
    diag::Add(diag::kSlowEntriesRan);
    NoteEntry(state, ctx, entry);
    if (b->jitChecked != nullptr) {
        cache.FastEntryPut(entry, b->jitChecked);
    }
    std::optional<CpuContextScope> cpuContextScope;
    if (__builtin_expect(TryGetCpuContext() != ctx, 0)) cpuContextScope.emplace(ctx);
    ++cache.depth;
    struct DepthScope {
        BlockCache* cache;
        ~DepthScope() { --cache->depth; }
    } depthScope{&cache};
    if (!once.jitStats && backend.enterDirect != nullptr && b->jitChecked != nullptr) {
        backend.enterDirect(ctx, entry, b->jitChecked, &state.directState);
        return true;
    }
    Frame f(ctx, entry);
    StartFrame(f, ctx);
    uint64_t executed = 0;
    const Trail trail;
    const JitExit exit = EnterCompiled(backend, ctx, f, *b, executed, trail);
    if (exit.status == JitStatus::Return) {
        CountInstructions(executed);
        return true;
    }
    Block* const following = FollowExit(cache, backend, f, exit);
    const LoopEnv env{ctx, f, executed, &cache, false, Mode::Jit, &backend, g_jitThreshold.load(std::memory_order_relaxed), following};
    RunLoop(env);
    return true;
}
const FastEntrySlot* detail::JitThreadFastEntries() noexcept { return State().cache.fastEntries; }
#else
void detail::JitDirectContinue(DirectEnterState*, Frame*, uint64_t, JitLink*) {}
const FastEntrySlot* detail::JitThreadFastEntries() noexcept { return nullptr; }
#endif

// The entry for the common call from translated code (GuestCodePatch::RunPatched's last step): Run(ctx, entry) when `entry` is
// the start of a block that has compiled code and is current, which is all the general entry does besides tests that cannot
// apply then. True when it ran (and returned, or threw, as Run does). False when it did nothing, and the caller makes the
// general call.
bool detail::RunCompiledEntry(CpuContext* ctx, uint32_t entry) {
#if MKW_JIT && !MKW_INTERP_PROFILE
    ThreadState* const state = t_state;
    if (__builtin_expect(state != nullptr && g_fastJitReady.load(std::memory_order_acquire) &&
                             TryGetCpuContext() == ctx &&
                             (!state->cache.overLimit || state->cache.depth != 0), 1)) {
        const void* const checked = state->cache.FastEntryFor(entry);
        if (__builtin_expect(checked != nullptr && JitBlocksMayRun(ctx), 1)) {
            diag::Add(diag::kFastEntries);
            ++state->cache.depth;
            struct DepthScope {
                BlockCache* cache;
                ~DepthScope() { --cache->depth; }
            } depthScope{&state->cache};
            g_enterDirect(ctx, entry, checked, &state->directState);
            return true;
        }
    }
    return RunCompiledEntrySlow(ctx, entry);
#else
    (void)ctx;
    (void)entry;
    return false;
#endif
}

void detail::RunPatchedJit(CpuContext* ctx, uint32_t address) {
    const PatchedHooks& h = g_patchedHooks;
#if MKW_JIT && !MKW_INTERP_PROFILE
    // RunCompiledEntry's fast path, behind RunPatched's redirect test (a call at or above the lowest redirect goes the general
    // way, which resolves it)
    ThreadState* const state = t_state;
    if (__builtin_expect(state != nullptr && address < h.lowestRedirect->load(std::memory_order_acquire) &&
                             g_fastJitReady.load(std::memory_order_acquire) && TryGetCpuContext() == ctx &&
                             (!state->cache.overLimit || state->cache.depth != 0), 1)) {
        const void* const checked = state->cache.FastEntryFor(address);
        if (__builtin_expect(checked != nullptr && JitBlocksMayRun(ctx), 1)) {
            diag::Add(diag::kFastEntries);
            {
                ++state->cache.depth;
                struct DepthScope {
                    BlockCache* cache;
                    ~DepthScope() { --cache->depth; }
                } depthScope{&state->cache};
                g_enterDirect(ctx, address, checked, &state->directState);
            }
            // RunPatched's Sync test: what the function called may have patched more code (GuestCodePatch::RunPatched)
            if (__builtin_expect(h.pendingFast->load(std::memory_order_acquire) != 0 ||
                                     h.anyDirty->load(std::memory_order_acquire) ||
                                     h.openEdgeCount->load(std::memory_order_acquire) != 0, 0))
                h.sync();
            return;
        }
    }
#endif
    h.runPatched(ctx, address);
}

#if MKW_INTERP_PROFILE
// The segment since the last dump, to <prefix>.<n>.<label>.txt: "F frames", "T thread instructions ticks" per thread,
// "E entry calls instructions ticks translated firstCaller" per entry (translated: the address has a translation, so
// the interpreter runs it because its code was patched or redirected), "P pc instructions" per guest PC.
void ProfileDump(const char* label) {
    const char* prefix = prof::Prefix();
    if (prefix == nullptr) return;
    char path[512];
    std::snprintf(path, sizeof path, "%s.%03u.%s.txt", prefix, prof::g_dumps++, label && *label ? label : "none");
    FILE* out = std::fopen(path, "w");
    if (out == nullptr) return;
    std::fprintf(out, "F %llu\n", static_cast<unsigned long long>(prof::g_frame - prof::g_segmentStart));
    prof::g_segmentStart = prof::g_frame;
    std::unordered_map<uint32_t, prof::Entry> merged;
    std::vector<uint64_t> pcs(prof::kMem1Words + prof::kMem2Words);
    std::lock_guard<std::mutex> tablesLock(prof::g_tablesMutex);
    for (prof::Table* t : prof::g_tables) {
        std::lock_guard<std::mutex> lock(t->mutex);
        uint64_t instructions = 0, ticks = 0;
        for (auto& [entry, stat] : t->entries) {
            auto& m = merged[entry];
            m.calls += stat.calls;
            m.instructions += stat.instructions;
            m.ticks += stat.ticks;
            if (m.firstCaller == 0) m.firstCaller = stat.firstCaller;
            instructions += stat.instructions;
            ticks += stat.ticks;
            stat.calls = stat.instructions = stat.ticks = 0;
        }
        for (size_t i = 0; i < pcs.size(); ++i) {
            pcs[i] += t->pcs[i];
            t->pcs[i] = 0;
        }
        std::fprintf(out, "T %s %llu %llu other %llu\n", t->thread.c_str(), static_cast<unsigned long long>(instructions),
                     static_cast<unsigned long long>(ticks), static_cast<unsigned long long>(t->otherInstructions));
        t->otherInstructions = 0;
    }
    for (const auto& [entry, stat] : merged) {
        if (stat.calls == 0) continue;
        std::fprintf(out, "E %08X %llu %llu %llu %d %08X\n", entry, static_cast<unsigned long long>(stat.calls),
                     static_cast<unsigned long long>(stat.instructions), static_cast<unsigned long long>(stat.ticks),
                     TranslatedFunctionRegistry::FindByAddressPtr(entry) != nullptr ? 1 : 0, stat.firstCaller);
    }
    for (size_t i = 0; i < pcs.size(); ++i) {
        if (pcs[i] == 0) continue;
        const uint32_t pc = i < prof::kMem1Words ? 0x80000000u + static_cast<uint32_t>(i) * 4
                                                 : 0x90000000u + static_cast<uint32_t>(i - prof::kMem1Words) * 4;
        std::fprintf(out, "P %08X %llu\n", pc, static_cast<unsigned long long>(pcs[i]));
    }
    std::fclose(out);
}

// Once per presented frame (GuestCodePatch::SyncFrame): a segment ends at each scene change, labelled with the scene
// it covered, and every BRAWL_INTERP_PROFILE_FRAMES frames.
void ProfileFrame() {
    if (prof::Prefix() == nullptr) return;
    ++prof::g_frame;
    static const uint64_t every = [] {
        const char* text = std::getenv("BRAWL_INTERP_PROFILE_FRAMES");
        return text ? std::strtoull(text, nullptr, 10) : 0ull;
    }();
    static std::string scene;
    std::string now = SaveStates::SceneName();
    if (now != scene) {
        ProfileDump(scene.c_str());
        scene = std::move(now);
    } else if (every != 0 && prof::g_frame - prof::g_segmentStart >= every) {
        ProfileDump(scene.c_str());
    }
}
#else
void ProfileDump(const char*) {}
void ProfileFrame() {}
#endif

}  // namespace PpcInterpreter
