// The JIT backend core's own mechanics (src/jit/jit_x64.cpp), which the differential tests reach only indirectly: a block's
// exit linked to the next block and run through the link, a block replaced while another is linked to it (its incoming
// slots reset, then linked again to the new block), the validity test of a linked block, the cache dropped once its code
// passes the limit with links live (the interpreter's rule: BRAWL_JIT_CODE_LIMIT), and the runtime switches
// (BRAWL_JIT_FALLBACK, BRAWL_JIT_PERF_MAP). Guest programs run through PpcInterpreter::Run with Engine::Jit; the backend's
// counters (JitX64GetStats) show what the core did.
//
// A program is a straight run of `addi r3, r3, 1` (a block ends after 32 ops and falls through to the next one, the only
// exit the core links while every op is a fallback call), then blr, so the result in r3 counts the ops that ran.
#include "jit_test_env.h"
#include "abi_bridge.h"
#include "brawl_dirty_memory.h"
#include "memory.h"
#include "ppc_interpreter.h"
#include "../src/jit/jit_x64.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using PpcInterpreter::detail::JitX64GetStats;
using PpcInterpreter::detail::JitX64Stats;

constexpr uint32_t kReturn = 0x80020000u;  // the lr programs return to: no code there
constexpr uint32_t kCodeBase = 0x80100000u;

int g_failures = 0;
#define CHECK(cond)                                                                                  \
    do {                                                                                             \
        if (!(cond)) {                                                                               \
            std::fprintf(stderr, "jit core: line %d: check failed: %s\n", __LINE__, #cond);          \
            ++g_failures;                                                                            \
        }                                                                                            \
    } while (0)

constexpr uint32_t Addi(uint32_t rt, uint32_t ra, int32_t imm) {
    return (14u << 26) | (rt << 21) | (ra << 16) | (static_cast<uint32_t>(imm) & 0xFFFFu);
}
constexpr uint32_t kBlr = 0x4E800020u;

// `ops` addi instructions of +1, then blr, at base.
void Place(uint32_t base, uint32_t ops) {
    for (uint32_t i = 0; i < ops; ++i) Memory::Write32(base + 4u * i, Addi(3, 3, 1));
    Memory::Write32(base + 4u * ops, kBlr);
}

uint32_t Run(uint32_t entry) {
    CpuContext ctx{};
    ctx.lr = kReturn;
    ctx.gpr[1] = 0x80180000u;
    CpuContextScope scope(&ctx);
    PpcInterpreter::Run(&ctx, entry);
    return ctx.gpr[3];
}

// Blocks hold 32 ops: 100 addi are blocks of 32, 32, 32 and 4 + blr.
void TestLinkRetireRelink() {
    const uint32_t base = kCodeBase;
    Place(base, 100);
    const JitX64Stats s0 = JitX64GetStats();
    for (int i = 0; i < 3; ++i) CHECK(Run(base) == 100);
    const JitX64Stats s1 = JitX64GetStats();
    CHECK(s1.compiled - s0.compiled == 4);   // four blocks, compiled once each
    CHECK(s1.links - s0.links == 3);         // block 1 -> 2 -> 3 -> 4, linked on the second run
    // Replace the second block's code (a guest store marks it): its incoming slot is reset, the run goes the long way
    // round and the new block is linked again.
    Memory::Write32(base + 4u * 40, Addi(3, 3, 2));
    CHECK(Run(base) == 101);
    const JitX64Stats s2 = JitX64GetStats();
    CHECK(s2.retires > s1.retires);
    CHECK(s2.compiled - s1.compiled >= 1);
    for (int i = 0; i < 3; ++i) CHECK(Run(base) == 101);
    const JitX64Stats s3 = JitX64GetStats();
    CHECK(s3.links > s2.links);              // relinked to the replacement
    // Put the first word back: the block compiled before is replaced by an identical one, and results stay right.
    Memory::Write32(base + 4u * 40, Addi(3, 3, 1));
    for (int i = 0; i < 3; ++i) CHECK(Run(base) == 100);
    // The very first block replaced (it has no incoming slot; its own exit slot stays linked to the old second block).
    Memory::Write32(base, Addi(3, 3, 5));
    for (int i = 0; i < 3; ++i) CHECK(Run(base) == 104);
    // A new epoch with nothing written: every block revalidates in place and keeps its links (they are suspended at the new
    // epoch, as a link does not test its target's epoch, and aimed again when the refresh finds the target's words unchanged:
    // no new link is made).
    BrawlDirtyMemory::MarksCleared();
    const JitX64Stats s4 = JitX64GetStats();
    for (int i = 0; i < 3; ++i) CHECK(Run(base) == 104);
    const JitX64Stats s5 = JitX64GetStats();
    CHECK(s5.compiled == s4.compiled);
    CHECK(s5.links == s4.links);
    // The same after a restore notification that covers the code (rollback's: everything built since the snapshot is dropped).
    BrawlDirtyMemory::MarksRestored(0);
    for (int i = 0; i < 3; ++i) CHECK(Run(base) == 104);
    const JitX64Stats s6 = JitX64GetStats();
    CHECK(s6.compiled == s5.compiled && s6.links == s5.links);
}

// A loop back to a block's own start: straight to its entry when nothing in the block before the jump can have changed its code
// (no store, no call: the blr after it does not count), through its validity test otherwise (a store to its own words must be
// seen on the next iteration: jit_stale_tests).
// The loop: addi r3,r3,1; cmpw r3,r4; blt start (a fused pair whose taken exit is the back edge); blr.
void TestSelfLoop() {
    const uint32_t base = kCodeBase + 0x6000;
    constexpr uint32_t kCmpwR3R4 = (31u << 26) | (3u << 16) | (4u << 11);
    const auto bltTo = [](uint32_t from, uint32_t to) { return (16u << 26) | (12u << 21) | (0u << 16) | ((to - from) & 0xFFFCu); };
    Memory::Write32(base, Addi(3, 3, 1));
    Memory::Write32(base + 4, kCmpwR3R4);
    Memory::Write32(base + 8, bltTo(base + 8, base));
    Memory::Write32(base + 12, kBlr);
    const auto run = [&](uint32_t n) {
        CpuContext ctx{};
        ctx.lr = kReturn;
        ctx.gpr[1] = 0x80180000u;
        ctx.gpr[4] = n;
        CpuContextScope scope(&ctx);
        PpcInterpreter::Run(&ctx, base);
        return ctx.gpr[3];
    };
    const JitX64Stats before = JitX64GetStats();
    CHECK(run(10) == 10);
    const JitX64Stats a = JitX64GetStats();
    CHECK(a.selfLoops - before.selfLoops == 1);
    const uint64_t counted = PpcInterpreter::InstructionCount();
    CHECK(run(1000) == 1000);
    const JitX64Stats b = JitX64GetStats();
    CHECK(b.links == a.links);                                        // the back edge is no link
    CHECK(b.enters - a.enters == 1);                                  // nor a dispatcher round
    CHECK(PpcInterpreter::InstructionCount() - counted == 4u * 1000);  // every iteration counts the block's 4 ops
    // the loop's code changed from outside between runs: the new word runs
    Memory::Write32(base, Addi(3, 3, 2));
    CHECK(run(1000) == 1000);
    CHECK(run(7) == 8);
    // with a store in the loop (stw r3,0x40(r5), r5 = base: the word after the loop's line) the back edge goes through the test
    const JitX64Stats c = JitX64GetStats();
    Memory::Write32(base + 12, (36u << 26) | (3u << 21) | (5u << 16) | 0x40u);
    Memory::Write32(base + 16, bltTo(base + 16, base));
    Memory::Write32(base + 8, (31u << 26) | (3u << 16) | (4u << 11));  // cmpw r3,r4 (the old blt's place)
    Memory::Write32(base + 4, Addi(0, 0, 0));                           // li r0,0
    Memory::Write32(base + 20, kBlr);
    CpuContext ctx{};
    ctx.lr = kReturn;
    ctx.gpr[1] = 0x80180000u;
    ctx.gpr[4] = 50;
    ctx.gpr[5] = base;
    {
        CpuContextScope scope(&ctx);
        PpcInterpreter::Run(&ctx, base);
    }
    CHECK(ctx.gpr[3] == 50 && Memory::Read32(base + 0x40) == 50);  // addi r3,r3,2 each time round, until r3 is 50
    CHECK(JitX64GetStats().selfLoops == c.selfLoops);
}

// A linked block whose code changed without anything reaching the dispatcher between: the target's checked entry sees the
// mark and leaves, so the changed word runs.
void TestLinkedBlockSeesChange() {
    const uint32_t base = kCodeBase + 0x2000;
    Place(base, 70);
    for (int i = 0; i < 4; ++i) CHECK(Run(base) == 70);  // linked by now
    for (uint32_t word : {40u, 41u, 64u, 33u, 32u}) {
        Memory::Write32(base + 4u * word, Addi(3, 3, 3));
        CHECK(Run(base) == 72);
        Memory::Write32(base + 4u * word, Addi(3, 3, 1));
        CHECK(Run(base) == 70);
    }
}

// The inline cache of a dynamic exit (BlockEmitter::ExitDynamicCached; here bctr's, a 32-entry table selected by (pc >> 2) & 31):
// a target is filled in by the dispatcher's link on the first miss, hit afterwards without the dispatcher, two targets of
// different entries alternate with no misses, two that select the same entry evict each other, and a target whose code is
// replaced empties its entry (retire resets the key) so the next jump to it goes through the dispatcher and relinks. The
// jump's block is `mtctr r4; bctr`; each target is `addi r3, r3, k; blr`. Needs the hit and miss counters (BRAWL_JIT_STATS).
constexpr uint32_t kMtctrR4 = (31u << 26) | (4u << 21) | (9u << 16) | (467u << 1);
constexpr uint32_t kBctr = 0x4E800420u;

uint32_t RunTo(uint32_t entry, uint32_t target) {
    CpuContext ctx{};
    ctx.lr = kReturn;
    ctx.gpr[1] = 0x80180000u;
    ctx.gpr[4] = target;
    CpuContextScope scope(&ctx);
    PpcInterpreter::Run(&ctx, entry);
    return ctx.gpr[3];
}

void TestInlineCache() {
    const uint32_t base = kCodeBase + 0x40000;
    const uint32_t t1 = base + 0x100, t2 = base + 0x204, t3 = base + 0x308, t4 = base + 0x400;  // entries 0, 1, 2 and 0 again
    Memory::Write32(base, kMtctrR4);
    Memory::Write32(base + 4, kBctr);
    const uint32_t targets[] = {t1, t2, t3, t4};
    for (uint32_t k = 0; k < 4; ++k) {
        Memory::Write32(targets[k], Addi(3, 3, 1 << k));
        Memory::Write32(targets[k] + 4, kBlr);
    }
    const auto counters = [] { return JitX64GetStats(); };
    // The dispatcher links an exit's slot only to a block that is already compiled, and compiles a block when it first
    // enters it: the first jump to a target is a miss that does not link, the second is a miss that does, the third hits.
    JitX64Stats a = counters();
    CHECK(RunTo(base, t1) == 1);
    JitX64Stats b = counters();
    CHECK(b.cacheMisses - a.cacheMisses == 1 && b.cacheHits == a.cacheHits);
    CHECK(RunTo(base, t1) == 1);
    JitX64Stats c = counters();
    CHECK(c.cacheMisses - b.cacheMisses == 1 && c.cacheHits == b.cacheHits);  // links now
    for (int i = 0; i < 5; ++i) CHECK(RunTo(base, t1) == 1);
    JitX64Stats d = counters();
    CHECK(d.cacheMisses == c.cacheMisses && d.cacheHits - c.cacheHits == 5);
    CHECK(d.enters - c.enters == 5);  // one enter per run: the jump to the target never went through the dispatcher
    // two blocks of different entries alternating
    for (int i = 0; i < 2; ++i) {
        CHECK(RunTo(base, t2) == 2);
        CHECK(RunTo(base, t3) == 4);
    }
    const JitX64Stats e = counters();
    CHECK(e.cacheMisses - d.cacheMisses == 4);
    for (int i = 0; i < 10; ++i) {
        CHECK(RunTo(base, t1) == 1);
        CHECK(RunTo(base, t2) == 2);
        CHECK(RunTo(base, t3) == 4);
    }
    const JitX64Stats f = counters();
    CHECK(f.cacheMisses == e.cacheMisses && f.cacheHits - e.cacheHits == 30);
    // two blocks selecting the same entry: each evicts the other, and every jump still lands right
    CHECK(RunTo(base, t4) == 8);
    CHECK(RunTo(base, t4) == 8);
    const JitX64Stats g = counters();
    CHECK(g.cacheMisses - f.cacheMisses == 2);
    for (int i = 0; i < 4; ++i) {
        CHECK(RunTo(base, t1) == 1);
        CHECK(RunTo(base, t4) == 8);
    }
    const JitX64Stats h = counters();
    CHECK(h.cacheMisses - g.cacheMisses == 8 && h.cacheHits == g.cacheHits);
    // the other entries were not disturbed
    CHECK(RunTo(base, t2) == 2);
    CHECK(RunTo(base, t3) == 4);
    const JitX64Stats i0 = counters();
    CHECK(i0.cacheMisses == h.cacheMisses && i0.cacheHits - h.cacheHits == 2);
    // t2's code is replaced: the stale block is left by its own check (a dispatcher round), the rebuild retires it, which
    // empties the cache entry, so the next jump goes through the dispatcher and links the replacement
    Memory::Write32(t2, Addi(3, 3, 20));
    const JitX64Stats j0 = counters();
    CHECK(RunTo(base, t2) == 20);
    const JitX64Stats j1 = counters();
    CHECK(j1.retires > j0.retires);
    CHECK(j1.cacheMisses == j0.cacheMisses && j1.cacheHits - j0.cacheHits == 1);  // the old entry still matched: its block left by its own check
    for (int k = 0; k < 2; ++k) CHECK(RunTo(base, t2) == 20);
    const JitX64Stats j2 = counters();
    // emptied: one miss (the rebuilt block is compiled already, so it links), then a hit
    CHECK(j2.cacheMisses - j1.cacheMisses == 1 && j2.cacheHits - j1.cacheHits == 1);
    for (int k = 0; k < 3; ++k) CHECK(RunTo(base, t2) == 20);
    const JitX64Stats j3 = counters();
    CHECK(j3.cacheMisses == j2.cacheMisses && j3.cacheHits - j2.cacheHits == 3);
}

// The code limit: past it the interpreter drops the whole cache (blocks, links and code) at the next outermost Run.
void TestCodeLimit() {
    const JitX64Stats before = JitX64GetStats();
    for (uint32_t i = 0; i < 400; ++i) {
        const uint32_t base = kCodeBase + 0x10000 + (i % 200) * 0x400;
        Place(base, 70 + i % 5);
        for (int run = 0; run < 3; ++run) CHECK(Run(base) == 70 + i % 5);
    }
    const JitX64Stats after = JitX64GetStats();
    CHECK(after.compiled - before.compiled > 400);
    CHECK(after.cachesDestroyed > before.cachesDestroyed);  // dropped at least once, with links live
    CHECK(after.cachesCreated > before.cachesCreated);      // and made again
    // everything still works after a drop, including code run before it
    Place(kCodeBase, 100);
    for (int i = 0; i < 3; ++i) CHECK(Run(kCodeBase) == 100);
}

}  // namespace

int main() {
    // A small limit so the third test drops the cache a few times (read once, on the interpreter's first outermost Run).
    setenv("BRAWL_JIT_CODE_LIMIT", "100000", 1);
    setenv("BRAWL_JIT_STATS", "1", 1);  // the inline cache's hit and miss counters exist only with it (read once, at the first compile)
    Memory::Init(0x200000);
    TranslatedFunctionRegistry::Finalize();
    if (!PpcInterpreter::JitAvailable()) {
        std::printf("jit core: no JIT backend on this host, nothing to test\n");
        return 0;
    }
    PpcInterpreter::SetEngineOverride(PpcInterpreter::Engine::Jit);
    PpcInterpreter::SetJitThreshold(0);
    TestLinkRetireRelink();
    TestLinkedBlockSeesChange();
    TestSelfLoop();
    TestInlineCache();
    TestCodeLimit();
    const JitX64Stats s = JitX64GetStats();
    std::printf("jit core: %llu blocks compiled, %llu links, %llu retires, %llu caches made and %llu destroyed, %d failures\n",
                static_cast<unsigned long long>(s.compiled), static_cast<unsigned long long>(s.links),
                static_cast<unsigned long long>(s.retires), static_cast<unsigned long long>(s.cachesCreated),
                static_cast<unsigned long long>(s.cachesDestroyed), g_failures);
    return g_failures == 0 ? 0 : 1;
}
