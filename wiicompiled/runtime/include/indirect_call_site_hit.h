#pragma once
// The per-site inline caches' hit path and way encoding (see indirect_call_site.h). Included by abi_bridge.cpp (the
// out-of-line hit functions and the slow paths), hle/game_natives.cpp (gfTask's per-type cache) and, with
// -DMKW_INDIRECT_IC_INLINE=1, by indirect_call_site.h itself.

#include "abi_bridge.h"

#if MKW_INDIRECT_IC

using MkwIndirectEntry = void (*)(CpuContext*);

MKW_PPC_FORCE_INLINE uintptr_t MkwIndirectAnchorAddress() noexcept {
    return reinterpret_cast<uintptr_t>(&MkwIndirectEntryAnchor);
}

// A way for the record: its address low, its entry's displacement from the anchor high. 0 (never a hit) when the
// record has no entry or address 0, or its entry is the anchor itself or not within int32 of it.
inline uint64_t MkwIndirectPackWay(const RawDispatchRecord* record) noexcept {
    if (record == nullptr || record->entry == nullptr || record->address == 0) return 0;
    const intptr_t displacement =
        static_cast<intptr_t>(reinterpret_cast<uintptr_t>(record->entry) - MkwIndirectAnchorAddress());
    if (displacement == 0 || displacement != static_cast<intptr_t>(static_cast<int32_t>(displacement))) return 0;
    return (static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(displacement))) << 32) | record->address;
}

// The way holds the target: same address, nonzero displacement (so an empty way never matches target 0).
MKW_PPC_FORCE_INLINE bool MkwIndirectWayMatches(uint64_t way, uint32_t target) noexcept {
    return static_cast<uint32_t>(way) == target && (way >> 32) != 0;
}

MKW_PPC_FORCE_INLINE MkwIndirectEntry MkwIndirectWayEntry(uint64_t way) noexcept {
    const intptr_t displacement = static_cast<int32_t>(static_cast<uint32_t>(way >> 32));
    return reinterpret_cast<MkwIndirectEntry>(MkwIndirectAnchorAddress() + static_cast<uintptr_t>(displacement));
}

// Second tier, shared by every site: ways keyed by target (direct-mapped), read only after the site's own ways miss.
// A megamorphic site (one vtable slot over many object classes, e.g. a per-object update loop) misses its three ways
// on most calls; the table answers those without the slow path. The slow path stores every way it packs here too.
// Same encoding and race argument as a site's ways; global, not thread-local (the mapping is the same for every thread
// and fills are rare once warm). 4096 entries, 32 KiB.
inline constexpr unsigned kMkwIndirectSharedBits = 12;
extern std::atomic<uint64_t> g_mkwIndirectSharedWays[size_t{1} << kMkwIndirectSharedBits];
MKW_PPC_FORCE_INLINE std::atomic<uint64_t>& MkwIndirectSharedWay(uint32_t target) noexcept {
    return g_mkwIndirectSharedWays[((target >> 2) * 0x9E3779B1u) >> (32 - kMkwIndirectSharedBits)];
}

#if MKW_INDIRECT_IC_STATS
#define MKW_INDIRECT_IC_COUNT_HIT(site) \
    (site).hits.store((site).hits.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed)
#else
#define MKW_INDIRECT_IC_COUNT_HIT(site) ((void)0)
#endif

// kJump false: bctrl (InvokeIndirectCpu, with its HZ pause). kJump true: tail bctr (InvokeIndirectJump, which has
// no HZ pause and no target-0 check of its own: a filled way never matches target 0, and the slow path hands an
// unregistered target to InvokeIndirectJump itself).
//
// The context test comes first and covers every way: ctx is never null here (the macros are emitted only into
// translated bodies, which read and write through ctx throughout); the slow path keeps its own null-ctx fallback for
// every other caller.
template <bool kJump>
MKW_PPC_FORCE_INLINE void MkwIndirectSiteDispatch(uint32_t target, CpuContext* ctx, MkwIndirectSite& site) {
    if (__builtin_expect(TryGetCpuContext() == ctx, 1)) {
        uint64_t way = site.ways[0].load(std::memory_order_relaxed);
        if (__builtin_expect(!MkwIndirectWayMatches(way, target), 0)) {
            way = site.ways[1].load(std::memory_order_relaxed);
            if (!MkwIndirectWayMatches(way, target)) {
                way = site.ways[2].load(std::memory_order_relaxed);
                if (!MkwIndirectWayMatches(way, target)) {
                    way = MkwIndirectSharedWay(target).load(std::memory_order_relaxed);
                    if (MkwIndirectWayMatches(way, target)) {
                        // Into the site first, as a slow-path fill would, or a site whose targets changed since its
                        // ways were filled would scan them stale on every call from now on.
                        site.ways[2].store(site.ways[1].load(std::memory_order_relaxed), std::memory_order_relaxed);
                        site.ways[1].store(site.ways[0].load(std::memory_order_relaxed), std::memory_order_relaxed);
                        site.ways[0].store(way, std::memory_order_relaxed);
                        goto hit;
                    }
                    const RawDispatchRecord* guarded = site.guarded.load(std::memory_order_relaxed);
                    if (guarded != nullptr && guarded->address == target) {
                        MKW_INDIRECT_IC_COUNT_HIT(site);
                        MkwIndirectSiteGuardedCall(guarded, ctx);
                        return;
                    }
                    goto slow;
                }
            }
        }
    hit:
        MKW_INDIRECT_IC_COUNT_HIT(site);
        if constexpr (!kJump) {
            HZ_EXCLUSIVE_PAUSE_IF_ACTIVE();  // a scope: it must cover the call
            RecompMod::g_currentTranslatedExecutionAddress = target;
            MkwIndirectWayEntry(way)(ctx);
        } else {
            RecompMod::g_currentTranslatedExecutionAddress = target;
            MkwIndirectWayEntry(way)(ctx);
        }
        return;
    }
slow:
    if constexpr (kJump) {
        MkwInvokeIndirectJumpSiteSlow(target, ctx, site);
    } else {
        MkwInvokeIndirectCpuSiteSlow(target, ctx, site);
    }
}

#if MKW_INDIRECT_IC_INLINE
MKW_PPC_FORCE_INLINE void MkwInvokeIndirectCpuSite(uint32_t target, CpuContext* ctx, MkwIndirectSite& site) {
    MkwIndirectSiteDispatch<false>(target, ctx, site);
}
MKW_PPC_FORCE_INLINE void MkwInvokeIndirectJumpSite(uint32_t target, CpuContext* ctx, MkwIndirectSite& site) {
    MkwIndirectSiteDispatch<true>(target, ctx, site);
}
#endif

#endif
