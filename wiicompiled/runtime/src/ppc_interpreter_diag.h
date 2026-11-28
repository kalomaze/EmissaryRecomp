// Diagnostics of the block cache and its JIT backend for performance work (docs/jit-design.md). Private to src/. Nothing here
// changes what the engine does.
//
//  * BRAWL_JIT_CHURN=<file>, in a build with MKW_JIT_CHURN set to 1 (below, or -DMKW_JIT_CHURN=1; the hooks sit on the
//    dispatcher's hot paths, so a normal build compiles them out): per pass of the game loop (Run's frame end at 0x80017508), one line of counters with a monotonic
//    timestamp (CLOCK_MONOTONIC ns, as `perf record -k mono` stamps samples), and every kMapEvery passes the per-key tables
//    (blocks run in the interpreter, compiles, retires, fallback ops ...), cumulative. Rates per game update come from two
//    lines' differences over a window's time. `%p` in the name is replaced by the process id.
//  * BRAWL_JIT_PROBE=<pc>,<pc>...: compiled code calls ProbeHit before the op at each pc, which logs the registers and the
//    guest words BRAWL_JIT_PROBE_READ names (items: `A` the word at hex address A, `A:o` the word at [A] + o, `rN:o` the word
//    at rN + o, `rN:o:p` the word at [rN + o] + p; o and p are signed hex): the first 40 hits of each pc, then one in 4096.
//    Any build; unset, it costs a look at an empty list per op compiled.
#pragma once

#include <cstdint>

#ifndef MKW_JIT_CHURN
#define MKW_JIT_CHURN 0
#endif

struct CpuContext;

namespace PpcInterpreter::diag {

extern const bool g_churn;
inline bool ChurnOn() { return MKW_JIT_CHURN && g_churn; }

enum Counter : uint32_t {
    // the dispatcher and block cache
    kFinds,             // BlockCache::Find (any caller)
    kInterpBlocks,      // blocks the dispatcher ran in the interpreter while the JIT was on
    kInterpOps,         // their guest instructions
    kCompileCalls,      // CompileBlock (the dispatcher asking the backend)
    kEnters,            // backend enter from C++ (EnterCompiled)
    kFollowLinks,       // FollowExit asked the backend to link an exit slot
    kFastEntries,       // RunCompiledEntry's fast path (fastEntries hit, jit_enter_direct)
    kSlowEntries,       // RunCompiledEntrySlow
    kSlowEntriesRan,    // ... that entered compiled code
    kDirectContinues,   // JitDirectContinue
    kRuns,              // Run
    kConsumes,          // ConsumingMarks (rollback's WriteWatch consumes the marks)
    kRestores,          // RestoredMarks (a snapshot restore)
    kRestoreRetired,    // blocks RestoredMarks retired
    kConsumeRetired,    // blocks ConsumingMarks retired
    kFallbackOps,       // JitFallbackOp calls (compiled code's interpreter calls; EmitSlowPath's too)
    // the backend
    kJitCompiled,       // blocks compiled
    kJitCompileFailed,  // compile turned down
    kJitLinks,          // link slots (or cache entries) aimed at a block
    kJitRetires,        // backend retire (a block replaced or dropped)
    kJitRetireSlots,    // slots reset by those retires
    kJitStalePaths,     // stale paths through the shared counting code (before a block has a stub)
    kJitStaleStubs,     // stale stubs made
    kJitStubRuns,       // stale stub runs
    kJitStubLeaves,     // ... that left for the dispatcher (a word differed or the epoch is old)
    kJitCacheMisses,    // inline-cache misses (left for the dispatcher with a cache slot)
    kJitProbeHits,      // BRAWL_JIT_PROBE calls
    kGuardDrops,        // guarded blocks dropped by a write under them (BlockCache::GuardedWritten)
    kCounters
};
inline constexpr const char* kCounterNames[kCounters] = {
    "finds", "interp_blocks", "interp_ops", "compile_calls", "enters", "follow_links", "fast_entries", "slow_entries",
    "slow_entries_ran", "direct_continues", "runs", "consumes", "restores", "restore_retired", "consume_retired",
    "fallback_ops", "jit_compiled", "jit_compile_failed", "jit_links", "jit_retires", "jit_retire_slots", "jit_stale_paths",
    "jit_stale_stubs", "jit_stub_runs", "jit_stub_leaves", "jit_cache_misses", "jit_probe_hits", "guard_drops",
};
// The counters as a block of memory compiled code can increment (`inc qword [abs]`): CounterCell(c).
uint64_t* CounterCell(Counter c);
inline void Add(Counter c, uint64_t n = 1) {
    if (ChurnOn()) *CounterCell(c) += n;
}

enum Table : uint32_t {
    kTInterpBlock,    // block start -> times run in the interpreter while the JIT was on (key bit 32: it failed to compile)
    kTCompile,        // block start -> compiles
    kTRetire,         // block start -> backend retires
    kTRebuild,        // block start -> rebuilt (code changed or retired, Find built it again)
    kTFallbackSite,   // op pc << 32 | its kind key (as kTFallbackKind's) -> calls
    kTFallbackKind,   // op kind (Generic: 0x10000 | primary << 10 | xo) -> calls
    kTStale,          // block start -> stale paths through the dispatcher (Find's Refresh)
    kTCacheLink,      // an inline-cache entry aimed at a block: evicted key << 32 | the block's start (key 1: was empty)
    kTFind,           // caller << 32 | pc -> Find calls (caller: FindCaller)
    kTRun,            // Run entry pc -> calls
    kTEnter,          // block start -> backend enters from C++
    kTables
};
inline constexpr const char* kTableNames[kTables] = {"interp_block", "compile", "retire", "rebuild", "fallback_site",
                                                     "fallback_kind", "stale", "cache_link", "find", "run", "enter"};
// Who called Find (kTFind's high word).
enum FindCaller : uint32_t { kFromFollow = 1, kFromLoop = 2, kFromNext = 3, kFromEntrySlow = 4 };
void AddKey(Table t, uint64_t key, uint64_t n = 1);
// A JitFallbackOp call: op of kind `kind` (an index of kKindNames), word w, at pc.
void Fallback(uint32_t pc, uint32_t kind, uint32_t w);

// The block cache's own counters (BlockCache's members), handed to Pass in this order.
enum CacheCounter : uint32_t { kCBuilt, kCRebuilt, kCSlotMisses, kCCurrent, kCLinked, kCClaims, kCRetired, kCEpochs,
                               kCPartial, kCLinkRefreshes, kCClears, kCacheCounters };
inline constexpr const char* kCacheCounterNames[kCacheCounters] = {"built", "rebuilt", "slot_misses", "current", "linked",
                                                                   "claims", "retired", "epochs", "partial",
                                                                   "link_refreshes", "clears"};
// Once per pass of the game loop.
void Pass(const uint64_t (&cache)[kCacheCounters]);

// BRAWL_JIT_PROBE.
bool ProbeAt(uint32_t pc);
void ProbeHit(const CpuContext* ctx, uint32_t pairedMask) noexcept;

}  // namespace PpcInterpreter::diag
