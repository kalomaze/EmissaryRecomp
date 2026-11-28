#pragma once
// Per-call-site monomorphic inline caches for translated bctrl/blrl and tail bctr. Included at the end of
// abi_bridge.h.
//
// The translator emits MKW_INDIRECT_CALL(target, ctx) for every indirect call (IrIndirectCall) and
// MKW_INDIRECT_JUMP(target, ctx) for every indirect tail jump (IrIndirectJump, a jump table's default arm) where it
// used to emit InvokeIndirectCpu / InvokeIndirectJump. Each expansion owns one static MkwIndirectSite: the
// RawDispatchRecord the site's last target resolved to. A hit skips the thread-local memo probe (and, on a memo miss,
// the generated table's segment/page/binary search), the out-of-line TryDispatchRawCpuTarget and its guard
// constructor/destructor calls.
//
// A cached record never goes stale. TranslatedFunctionRegistry::Finalize publishes the lookup once, before guest code
// runs (main.cpp, ahead of Aurora and the guest boot), and Register aborts after it; the generated table is constant
// data and the dynamic (native) records are never rebuilt after publication. So FindRawByAddressPtr is a pure function
// of the address for the rest of the process (the thread-local memo in front of it already relies on this and is never
// invalidated), and before publication it returns nullptr, which is never cached. What can change at run time is what
// an address does, and none of that goes through the lookup: a Gecko/Syriinge patch of a translated function is
// caught by the MKW_GUEST_CODE_GUARD in the record's entry (the function's wrapper), native hooks are tested in the
// same wrapper, modules are linked at their own pre-linked home addresses (hle/brawl_modules.cpp), and code without a
// translation (a mod's replacement module, Gecko code) has no record, so its calls always take the slow path and run
// in the interpreter as before.
//
// Threads: each of a site's two slots is one atomic pointer. A slot holds nullptr or a published record, and every
// use compares that record's own immutable address with the target, so a stale or racing read can only miss, never
// call the wrong function. Fills store with release; the hit path loads relaxed (its reads of the record depend on
// the loaded pointer, the records are constant data or were written before publication, and every host thread that
// runs guest code is started after Finalize).
//
// The hit path's slot only ever holds a record that needs no register guard (nonvolatile FPR write mask 0, no GPR
// preservation: 98% of the generated table) and has a nonzero address. For such a record, when the call stays in the
// current CpuContext, TryDispatchRawCpuTarget does exactly two things (PpcNonvolatileFprGuard and
// PpcNonvolatileGprGuard do nothing, no CpuContextScope is built): the execution-address scope and
// record->entry(ctx). The hit does the same, except that it sets the execution address without restoring it (see
// the note above the hit functions: diagnostics only), which makes the call a tail jump. A guarded record is only kept in the site's second slot,
// which the out-of-line slow path reads; that path also takes a call that switches CpuContext, a miss, or an
// untranslated target, runs the old dispatch for it, and refills the site.
//
// The hit path is one shared out-of-line function by default: a site is its argument moves and a call, as before.
// -DMKW_INDIRECT_IC_INLINE=1 inlines it at every site instead (each site gets its own indirect branch), which measured
// no faster on Zen 5 (deterministic Training replay: game-thread cycles per frame 10.69 M inlined vs
// 10.57 M out of line, base 11.19 M) and costs ~20 host instructions per site: +2.6M over 125k sites, enough to push a
// build at 3.93:1 (lean memory, outlined guest ops) past the 4:1 expansion gate. The out-of-line default adds ~1
// instruction per site.
//
// Switches: -DMKW_INDIRECT_IC=0 makes both macros the plain InvokeIndirectCpu / InvokeIndirectJump calls (the old
// generated code, token for token). MKW_INDIRECT_IC=0 in the environment keeps the caches compiled in but never fills
// them, so every call runs the old lookup and dispatch from the slow path. -DMKW_INDIRECT_IC_STATS=1 counts hits and
// misses per site and prints a summary every 5 s ([indirect-ic] lines on stderr; with the out-of-line hit path the
// per-site caller shown is the hit function).

#ifndef MKW_INDIRECT_IC
#define MKW_INDIRECT_IC 1
#endif
#ifndef MKW_INDIRECT_IC_INLINE
#define MKW_INDIRECT_IC_INLINE 0
#endif
#ifndef MKW_INDIRECT_IC_STATS
#define MKW_INDIRECT_IC_STATS 0
#endif

#if MKW_INDIRECT_IC

struct MkwIndirectSite {
    // Static storage, so zero-initialized (empty) at load time: no constructor, no guard variable.
    // record: the hit path's, always unguarded. other: read only by the slow path, a guarded record (which the hit
    // path must not call) or the unguarded record a refill displaced, so a site with a guarded target or two
    // alternating targets skips the lookup. Each slot is validated on its own, like record.
    std::atomic<const RawDispatchRecord*> record;
    std::atomic<const RawDispatchRecord*> other;
#if MKW_INDIRECT_IC_STATS
    std::atomic<uint64_t> hits;
    std::atomic<uint64_t> slow;
    std::atomic<uint32_t> fills;
    std::atomic<uint32_t> registered;
#endif
};

void MkwInvokeIndirectCpuSiteSlow(uint32_t target, CpuContext* ctx, MkwIndirectSite& site);
void MkwInvokeIndirectJumpSiteSlow(uint32_t target, CpuContext* ctx, MkwIndirectSite& site);

// The site's record is the target's (so it is unguarded and its address, the target, is nonzero) and the call stays
// in the current CpuContext. ctx is never null here: the macros are emitted only into translated bodies, which read
// and write through ctx throughout (a null one would fault long before an indirect call); the slow path keeps its
// own null-ctx fallback for every other caller. So the old `ctx != nullptr` term (a test and a branch per call) is
// dropped; TryGetCpuContext() == ctx still sends a call from another context to the slow path.
MKW_PPC_FORCE_INLINE bool MkwIndirectSiteFastHit(const RawDispatchRecord* record, uint32_t target,
                                                 CpuContext* ctx) noexcept {
    return record != nullptr && record->address == target && TryGetCpuContext() == ctx;
}

// The hit records its target in RecompMod::g_currentTranslatedExecutionAddress and does not restore the previous value
// on return: every reader of it is crash/access-violation diagnostics ("active function"), which then name the last
// indirect target this thread entered instead of the outer scope's. Without the restore the hit is a tail jump: no
// saved registers, no call/return pair, no landing pad (~15 of its ~25 ops per call IBS profile).
// Direct calls never set the value at all, so it was already "last scope that set it", not the current function.

#if MKW_INDIRECT_IC_STATS
#define MKW_INDIRECT_IC_COUNT_HIT(site) \
    (site).hits.store((site).hits.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed)
#else
#define MKW_INDIRECT_IC_COUNT_HIT(site) ((void)0)
#endif

#if MKW_INDIRECT_IC_INLINE
#define MKW_INDIRECT_IC_HIT_FUNCTION MKW_PPC_FORCE_INLINE
#else
#define MKW_INDIRECT_IC_HIT_FUNCTION MKW_PPC_NO_INLINE inline
#endif

// The site's other slot, when it holds this target's record and that record is unguarded (as the hit path's slot
// always is: no non-volatile FPR write mask, no GPR preservation) and the call stays in the current CpuContext: the
// same two things as the hit. A site alternating between two targets otherwise took the out-of-line slow path, which
// also swapped the two slots (two stores), on every other call. A guarded record in that slot still goes to the slow
// path, which builds its guards.
MKW_PPC_FORCE_INLINE const RawDispatchRecord* MkwIndirectSiteOtherHit(MkwIndirectSite& site, uint32_t target,
                                                                       CpuContext* ctx) noexcept {
    const RawDispatchRecord* other = site.other.load(std::memory_order_relaxed);
    if (other != nullptr && other->address == target && other->nonvolatileFprWriteMask == 0 &&
        !other->preserveNonvolatileGprs && TryGetCpuContext() == ctx)
        return other;
    return nullptr;
}

// bctrl: InvokeIndirectCpu.
MKW_INDIRECT_IC_HIT_FUNCTION void MkwInvokeIndirectCpuSite(uint32_t target, CpuContext* ctx, MkwIndirectSite& site) {
    const RawDispatchRecord* record = site.record.load(std::memory_order_relaxed);
    if (__builtin_expect(MkwIndirectSiteFastHit(record, target, ctx), 1)) {
        MKW_INDIRECT_IC_COUNT_HIT(site);
        HZ_EXCLUSIVE_PAUSE_IF_ACTIVE();
        RecompMod::g_currentTranslatedExecutionAddress = target;
        record->entry(ctx);
        return;
    }
    if (const RawDispatchRecord* other = MkwIndirectSiteOtherHit(site, target, ctx)) {
        MKW_INDIRECT_IC_COUNT_HIT(site);
        HZ_EXCLUSIVE_PAUSE_IF_ACTIVE();
        RecompMod::g_currentTranslatedExecutionAddress = target;
        other->entry(ctx);
        return;
    }
    MkwInvokeIndirectCpuSiteSlow(target, ctx, site);
}

// Tail bctr: InvokeIndirectJump (which has no HZ pause and no target-0 check of its own: a filled site never matches
// target 0, and the slow path hands an unregistered target to InvokeIndirectJump itself).
MKW_INDIRECT_IC_HIT_FUNCTION void MkwInvokeIndirectJumpSite(uint32_t target, CpuContext* ctx, MkwIndirectSite& site) {
    const RawDispatchRecord* record = site.record.load(std::memory_order_relaxed);
    if (__builtin_expect(MkwIndirectSiteFastHit(record, target, ctx), 1)) {
        MKW_INDIRECT_IC_COUNT_HIT(site);
        RecompMod::g_currentTranslatedExecutionAddress = target;
        record->entry(ctx);
        return;
    }
    if (const RawDispatchRecord* other = MkwIndirectSiteOtherHit(site, target, ctx)) {
        MKW_INDIRECT_IC_COUNT_HIT(site);
        RecompMod::g_currentTranslatedExecutionAddress = target;
        other->entry(ctx);
        return;
    }
    MkwInvokeIndirectJumpSiteSlow(target, ctx, site);
}

// One static site per expansion (a block-scope static: each macro use is its own object).
#define MKW_INDIRECT_CALL(target, ctx)                              \
    do {                                                            \
        static MkwIndirectSite mkwIndirectSite;                     \
        MkwInvokeIndirectCpuSite((target), (ctx), mkwIndirectSite); \
    } while (false)
#define MKW_INDIRECT_JUMP(target, ctx)                               \
    do {                                                             \
        static MkwIndirectSite mkwIndirectSite;                      \
        MkwInvokeIndirectJumpSite((target), (ctx), mkwIndirectSite); \
    } while (false)

#else

#define MKW_INDIRECT_CALL(target, ctx) InvokeIndirectCpu(target, ctx)
#define MKW_INDIRECT_JUMP(target, ctx) InvokeIndirectJump(target, ctx)

#endif

// Profiling builds of the translated code (MKW_INDIRECT_PROFILE_SITES=1 in the translator's environment) emit
// MKW_INDIRECT_PROFILE_COUNT(function, site, target) before each indirect call and MKW_INDIRECT_PROFILE_JUMP before
// each indirect tail jump: function is the translated function's entry, site the bctrl/blrl/bctr's guest address. The
// counts go to $BRAWL_INDIRECT_PROFILE (rewritten every 2^24 calls and at exit), one "FUNCTION SITE TARGET COUNT"
// line per pair (jumps as "J FUNCTION SITE TARGET COUNT"): the translator's --indirect-profile input
// (Translator.Core/CodeGen/IndirectCallProfile.cs). Without the variable nothing is counted.
void MkwIndirectProfileCount(uint32_t function, uint32_t site, uint32_t target, bool jump);
#define MKW_INDIRECT_PROFILE_COUNT(function, site, target) MkwIndirectProfileCount((function), (site), (target), false)
#define MKW_INDIRECT_PROFILE_JUMP(function, site, target) MkwIndirectProfileCount((function), (site), (target), true)
