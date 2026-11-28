#pragma once
// Per-call-site inline caches for translated bctrl/blrl and tail bctr. Included at the end of abi_bridge.h.
//
// The translator emits MKW_INDIRECT_CALL(target, ctx) for every indirect call (IrIndirectCall) and
// MKW_INDIRECT_JUMP(target, ctx) for every indirect tail jump (IrIndirectJump, a jump table's default arm) where it
// used to emit InvokeIndirectCpu / InvokeIndirectJump. Each expansion owns one static MkwIndirectSite: the entries of
// the last targets the site resolved to. A hit skips the thread-local memo probe (and, on a memo miss, the generated
// table's segment/page/binary search), the out-of-line TryDispatchRawCpuTarget and its guard constructor/destructor
// calls.
//
// A cached entry never goes stale. TranslatedFunctionRegistry::Finalize publishes the lookup once, before guest code
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
// Layout: one 32-byte, 32-aligned object (one cache line) per site.
//   ways[3]  unguarded records (nonvolatile FPR write mask 0, no GPR preservation: 98% of the generated table), each
//            packed into one 64-bit word: the guest address in the low half, the entry's displacement from an anchor
//            in the executable's text (MkwIndirectEntryAnchor) as a nonzero int32 in the high half. A hit is one load
//            and a compare; it never reads the record, whose line in the 2 MB generated table was the second
//            dependent load (often a cache miss) before the jump. An empty way is 0 and never matches (its high half
//            is zero, so not even target 0). A record whose entry is not within +-2 GiB of the anchor is not packed
//            (it goes to `guarded`); in practice every entry is in the executable.
//   guarded  one record that needs its registers guarded (or could not be packed). Its call is the out-of-line
//            MkwIndirectSiteGuardedCall: the guards and the entry, without the slow path's lookup and refill.
// The slow path fills a way by shifting the ways down (the oldest drops out) and putting the new target first; a hit
// in any way moves nothing (no stores on hits). Three ways cover Brawl's common 2-3-target vtable sites (fighter /
// item / gfTask methods).
//
// Behind the ways, one table shared by every site (g_mkwIndirectSharedWays, indirect_call_site_hit.h: 4096 ways
// keyed by target, direct-mapped, 32 KiB) catches what a site's ways miss: a megamorphic site (one vtable slot over
// many object classes, e.g. a per-object update loop: one such site was 40% of all slow-path calls in a P+ replay)
// otherwise took the slow path's frame, lookup and refill on most calls. Every way the slow path packs is stored
// there too, and a hit in it is shifted into the site's way 0 like a fill (or a site whose targets changed after its
// ways were filled would scan them stale on every call). Measured on P+ ranked replays (d4): calls answered without
// the slow path 89% -> 94% with three ways; with the table the slow paths fell from 0.76% to 0.15% of cycles (what is
// left: untranslated targets, guarded churn), the whole indirect machinery 2.91% -> 2.20%, updates -1.6%.
//
// Threads: each way, shared way and the guarded slot is one atomic word. A way holds 0 or a packed (address, entry) pair written
// in one store, so a racing read sees a whole pair or nothing; the guarded slot holds nullptr or a published record
// whose immutable address is compared with the target. A stale or racing read can only miss, never call the wrong
// function. Fills store with release; the hit path loads relaxed.
//
// The hit does what TryDispatchRawCpuTarget does for an unguarded record when the call stays in the current
// CpuContext (PpcNonvolatileFprGuard and PpcNonvolatileGprGuard do nothing, no CpuContextScope is built): the
// execution-address store and the entry, except that it sets the execution address without restoring it (diagnostics
// only: every reader of RecompMod::g_currentTranslatedExecutionAddress is crash/access-violation reporting, which
// then names the last indirect target this thread entered instead of the outer scope's; direct calls never set it
// at all), which makes the call a tail jump. A call from another CpuContext goes to the out-of-line slow path, which
// runs the old dispatch for it, as does a miss or an untranslated target.
//
// The hit path is one shared out-of-line function per kind (call, jump) by default, defined in abi_bridge.cpp from
// indirect_call_site_hit.h: a site is its argument moves and a call, and changing the hit path rebuilds one file, not
// every translated one. -DMKW_INDIRECT_IC_INLINE=1 inlines it at every site instead (each site gets its own indirect
// branch), which measured no faster (deterministic Training replay: game-thread cycles per frame 10.69 M
// inlined vs 10.57 M out of line, base 11.19 M) and costs ~20 host instructions per site: +2.6M over 125k sites,
// enough to push a build at 3.93:1 (lean memory, outlined guest ops) past the 4:1 expansion gate.
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

inline constexpr unsigned kMkwIndirectSiteWays = 3;

struct alignas(32) MkwIndirectSite {
    // Static storage, so zero-initialized (empty) at load time: no constructor, no guard variable.
    std::atomic<uint64_t> ways[kMkwIndirectSiteWays];
    std::atomic<const RawDispatchRecord*> guarded;
#if MKW_INDIRECT_IC_STATS
    std::atomic<uint64_t> hits;
    std::atomic<uint64_t> slow;
    std::atomic<uint32_t> fills;
    std::atomic<uint32_t> registered;
#endif
};

void MkwInvokeIndirectCpuSiteSlow(uint32_t target, CpuContext* ctx, MkwIndirectSite& site);
void MkwInvokeIndirectJumpSiteSlow(uint32_t target, CpuContext* ctx, MkwIndirectSite& site);
// The guarded slot's call: its record matched the target and the call stays in the current CpuContext.
void MkwIndirectSiteGuardedCall(const RawDispatchRecord* record, CpuContext* ctx);
// The text address the ways' entry displacements are relative to (a function in abi_bridge.cpp; never called).
void MkwIndirectEntryAnchor();

#if MKW_INDIRECT_IC_INLINE
#include "indirect_call_site_hit.h"
#else
void MkwInvokeIndirectCpuSite(uint32_t target, CpuContext* ctx, MkwIndirectSite& site);
void MkwInvokeIndirectJumpSite(uint32_t target, CpuContext* ctx, MkwIndirectSite& site);
#endif

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
