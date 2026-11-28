// The JIT's stale stubs (src/jit/jit_x64.cpp, StaleCell): a block whose dirty marks stay set because its own code stores to a data
// word inside its 64-byte mark range fails its entry check on every linked entry and would go round the dispatcher each time.
// After a threshold of failures its stale path jumps to a stub that compares the block's guest words in place and enters the block
// when all are unchanged (a stale epoch never gets there: the block's full entry sends it to the dispatcher, and link slots are
// aimed only at blocks of the current epoch).
//
// Every program here is run under Engine::Classic and Engine::Jit (threshold 0) and the whole contexts and the words it wrote
// are compared; the backend's counters show that the stub was made and that the dispatcher round trips stopped:
//
//  * a loop with a data word under its marks: the stub is made, the enters stop (`enters` needs BRAWL_JIT_STATS, which main sets),
//    and the stub clears no mark;
//  * a real code change: the loop stores over one of its own instructions every iteration, a different word from iteration 150 on,
//    which must be noticed on the very next entry (the result counts the iterations each version ran);
//  * the same for a block of two segments (a `b` followed), the changed word in the second;
//  * an epoch bump (what rollback's consume does) between two runs of a block that has a stub: the slots aimed in the old epoch
//    are reset, the dispatcher refreshes the block, and the same stub serves the new one;
//  * a retire and rebuild: the code is rewritten from outside between runs, the old block and its stub are dropped, the new block
//    gets its own;
//  * other threads with their own caches and stubs, and the cache dropped past its code limit with stubs live (BRAWL_JIT_CODE_LIMIT,
//    small here: the interpreter never destroys a thread's cache otherwise);
//  * the same programs with the stub made wrong on purpose four ways (BRAWL_JIT_STALE_FAULT 1, 3, 4, 5): each must make a program
//    fail.
//
// Run under valgrind --smc-check=all or with ASan for the destruction cases.
#include "jit_test_env.h"
#include "abi_bridge.h"
#include "brawl_dirty_memory.h"
#include "memory.h"
#include "memory_access.h"
#include "ppc_interpreter.h"
#include "../src/jit/jit_x64.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using PpcInterpreter::detail::JitX64GetStats;
using PpcInterpreter::detail::JitX64Stats;

constexpr uint32_t kReturn = 0x80020000u;
constexpr uint32_t kBase = 0x80400000u;
constexpr uint32_t kBlr = 0x4E800020u;
constexpr uint32_t kSpan = 0x2000u;  // the address space of one program

int g_failures = 0;
uint32_t g_slot = 0;
bool g_quiet = false;
#define CHECK(cond, ...)                                                                              \
    do {                                                                                              \
        if (!(cond)) {                                                                                \
            if (!g_quiet) {                                                                           \
                std::fprintf(stderr, "jit stale: line %d: check failed: %s: ", __LINE__, #cond);      \
                std::fprintf(stderr, __VA_ARGS__);                                                    \
                std::fprintf(stderr, "\n");                                                           \
            }                                                                                         \
            ++g_failures;                                                                             \
        }                                                                                             \
    } while (0)

constexpr uint32_t D(uint32_t op, uint32_t rt, uint32_t ra, int32_t imm) {
    return (op << 26) | (rt << 21) | (ra << 16) | (static_cast<uint32_t>(imm) & 0xFFFFu);
}
constexpr uint32_t Addi(uint32_t rt, uint32_t ra, int32_t i) { return D(14, rt, ra, i); }
constexpr uint32_t Stw(uint32_t rs, uint32_t ra, int32_t d) { return D(36, rs, ra, d); }
constexpr uint32_t Bc(uint32_t bo, uint32_t bi, int32_t byteDisp) {
    return (16u << 26) | (bo << 21) | (bi << 16) | (static_cast<uint32_t>(byteDisp) & 0xFFFCu);
}
constexpr uint32_t Bd(int32_t byteDisp) { return (18u << 26) | (static_cast<uint32_t>(byteDisp) & 0x03FFFFFCu); }

struct Program {
    uint32_t base;
    std::vector<std::pair<uint32_t, uint32_t>> words;  // address, word
    uint32_t entry = 0;                                // where Run starts (0: base)
};

void Place(const Program& p) {
    for (const auto& w : p.words) Memory::Write32(w.first, w.second);
}

CpuContext InitialContext(uint32_t base, uint32_t iterations) {
    CpuContext c{};
    c.lr = kReturn;
    c.gpr[1] = 0x80180000u;
    c.gpr[6] = base;
    c.gpr[9] = base + 0x1000u;
    c.ctr = iterations;
    return c;
}

// The classic interpreter and the JIT from the same context and code; the context after and the words the program may write.
struct Result {
    CpuContext ctx;
    uint32_t words[4];
};
Result RunOnce(PpcInterpreter::Engine engine, const Program& p, const CpuContext& init, bool place = true) {
    if (place) Place(p);
    PpcInterpreter::SetEngineOverride(engine);
    PpcInterpreter::SetJitThreshold(0);
    Result r{};
    r.ctx = init;
    r.ctx.pc = 0;
    {
        CpuContextScope scope(&r.ctx);
        PpcInterpreter::Run(&r.ctx, p.entry != 0 ? p.entry : p.base);
    }
    r.ctx.pc = 0;
    PpcInterpreter::ClearEngineOverride();
    r.words[0] = Memory::Read32(p.base + 0x30);
    r.words[1] = Memory::Read32(p.base + 8);
    r.words[2] = Memory::Read32(p.base + 0x1004);
    r.words[3] = Memory::Read32(p.base + 4);
    return r;
}

bool Same(const Result& a, const Result& b) {
    return std::memcmp(&a.ctx, &b.ctx, sizeof a.ctx) == 0 && std::memcmp(a.words, b.words, sizeof a.words) == 0;
}

void Compare(const char* what, const Program& p, const CpuContext& init, JitX64Stats* delta = nullptr) {
    const Result want = RunOnce(PpcInterpreter::Engine::Classic, p, init);
    const JitX64Stats before = JitX64GetStats();
    const Result got = RunOnce(PpcInterpreter::Engine::Jit, p, init);
    const JitX64Stats after = JitX64GetStats();
    if (delta != nullptr) {
        *delta = after;
        delta->enters -= before.enters;
        delta->staleStubs -= before.staleStubs;
        delta->staleStubFailures -= before.staleStubFailures;
    }
    CHECK(Same(want, got), "%s: the JIT's result differs (r5 %u vs %u, r7 %u vs %u, ctr %u vs %u)", what, want.ctx.gpr[5], got.ctx.gpr[5],
          want.ctx.gpr[7], got.ctx.gpr[7], want.ctx.ctr, got.ctx.ctr);
}

// --- the programs -----------------------------------------------------------------------------------------------------------

uint32_t NextBase() { return kBase + (g_slot++ % 64u) * kSpan; }

// stw r5,0x30(r6); addi r5,r5,1; bdnz; blr: a data word at base + 0x30, in the mark range of the loop's 4 words.
Program Loop() {
    const uint32_t a = NextBase();
    Program p{a, {}};
    p.words = {{a, Stw(5, 6, 0x30)}, {a + 4, Addi(5, 5, 1)}, {a + 8, Bc(16, 0, -8)}, {a + 12, kBlr}};
    return p;
}

// The instruction word a loop stores over its own `addi r7,r7,K` every iteration, built from r5 (the iteration count): K = 1 up to
// iteration `change`, 5 after, with no branch (a taken bc would split the loop into blocks, and the dispatcher's refresh of one
// would re-check the others: Recheck). r8 = ((change - r5) >>arith 31 & 4) + 1 + 0x38E70000, the word of `addi r7,r7,K`.
void BuildWord(std::vector<std::pair<uint32_t, uint32_t>>& w, uint32_t at, uint32_t change) {
    w.push_back({at + 0x00, D(8, 8, 5, static_cast<int32_t>(change))});          // subfic r8,r5,change
    w.push_back({at + 0x04, (31u << 26) | (8u << 21) | (8u << 16) | (31u << 11) | (824u << 1)});  // srawi r8,r8,31
    w.push_back({at + 0x08, D(28, 8, 8, 4)});                                    // andi. r8,r8,4
    w.push_back({at + 0x0C, Addi(8, 8, 1)});
    w.push_back({at + 0x10, D(15, 8, 8, 0x38E7)});                               // addis r8,r8,0x38E7
}

// The loop with a code change: it rewrites its own `addi r7,r7,K` each iteration, K becoming 5 after iteration `change`. The word
// is the first of the block (`first`) or the third.
Program Rewriting(uint32_t change, bool first) {
    const uint32_t a = NextBase();
    Program p{a, {}};
    const uint32_t target = Addi(7, 7, 1), data = Stw(5, 6, 0x30), inc = Addi(5, 5, 1);
    const uint32_t tOff = first ? 0 : 8;
    if (first) p.words = {{a + 0x00, target}, {a + 0x04, data}, {a + 0x08, inc}};
    else p.words = {{a + 0x00, data}, {a + 0x04, inc}, {a + 0x08, target}};
    BuildWord(p.words, a + 0x0C, change);
    p.words.push_back({a + 0x20, Stw(8, 6, static_cast<int32_t>(tOff))});
    p.words.push_back({a + 0x24, Bc(16, 0, -0x24)});
    p.words.push_back({a + 0x28, kBlr});
    return p;
}

// Two segments: the first stores to its own data word and `b`s to the second, which adds, and rewrites its own `addi r7,r7,K`
// (held by r9) like Rewriting, then loops. The marks that stay set are the first segment's and the second's; the changed word is
// in the second, its first word when `first`.
Program TwoSegments(uint32_t change, bool first) {
    const uint32_t a = NextBase(), s2 = a + 0x1000;
    Program p{a, {}};
    p.words = {{a + 0x00, Stw(5, 6, 0x30)}, {a + 0x04, Addi(5, 5, 1)}, {a + 0x08, Bd(static_cast<int32_t>(s2 - (a + 0x08)))}};
    const uint32_t target = Addi(7, 7, 1), other = Addi(10, 10, 1);
    if (first) p.words.insert(p.words.end(), {{s2 + 0x00, target}, {s2 + 0x04, other}});
    else p.words.insert(p.words.end(), {{s2 + 0x00, other}, {s2 + 0x04, target}});
    BuildWord(p.words, s2 + 0x08, change);
    p.words.push_back({s2 + 0x1C, Stw(8, 9, first ? 0 : 4)});
    p.words.push_back({s2 + 0x20, Bc(16, 0, static_cast<int32_t>(a - (s2 + 0x20)))});
    p.words.push_back({s2 + 0x24, kBlr});
    return p;
}

// The loop reached through another block (an always-taken bc, linked to the loop's checked entry), so that the loop's stale path
// is what meets a new epoch.
Program Reached() {
    Program p = Loop();
    const uint32_t x = p.base + 0x100;
    p.words.push_back({x, Bc(20, 0, static_cast<int32_t>(p.base - x))});
    p.words.push_back({x + 4, kBlr});
    p.entry = x;
    return p;
}

constexpr uint32_t kIterations = 600;
constexpr uint32_t kThreshold = 8;  // BRAWL_JIT_STALE_STUB, set in main

// --- the cases ----------------------------------------------------------------------------------------------------------

void CaseLoop() {
    const Program p = Loop();
    JitX64Stats d{};
    const uint64_t counted = PpcInterpreter::InstructionCount();
    Compare("loop", p, InitialContext(p.base, kIterations), &d);
    // the instruction count the dispatcher collects (the blocks add theirs to a register the enter's epilogue hands over): the
    // classic run counts 3 per iteration and 1, the JIT's block of 4 ops counts 4 per entry, through the stub too
    CHECK(PpcInterpreter::InstructionCount() - counted == 3u * kIterations + 1u + 4u * kIterations,
          "instructions counted %llu, expected %u", static_cast<unsigned long long>(PpcInterpreter::InstructionCount() - counted),
          3u * kIterations + 1u + 4u * kIterations);
    CHECK(d.staleStubs >= 1, "no stub was made (%llu)", static_cast<unsigned long long>(d.staleStubs));
    CHECK(d.enters <= kThreshold + 8, "%llu dispatcher entries for %u iterations", static_cast<unsigned long long>(d.enters), kIterations);
    // the stub clears no mark: the last store's mark is still set (the dispatcher's refresh would have cleared it)
    const uint32_t mark = (p.base & MemoryInline::kLeanMask) >> 6;
    CHECK(BrawlDirtyMemory::leanBlocks[mark] != 0, "the mark under the loop was cleared");
}

void CaseRewrite() {
    for (const bool first : {false, true}) {
        const Program p = Rewriting(150, first);
        const CpuContext init = InitialContext(p.base, 400);
        JitX64Stats d{};
        Compare(first ? "rewrite (first word)" : "rewrite", p, init, &d);
        CHECK(d.staleStubs >= 1, "no stub was made before the change (%llu)", static_cast<unsigned long long>(d.staleStubs));
    }
}

void CaseTwoSegments() {
    for (const bool first : {false, true}) {
        const Program p = TwoSegments(150, first);
        const CpuContext init = InitialContext(p.base, 400);
        JitX64Stats d{};
        Compare(first ? "two segments (first word)" : "two segments", p, init, &d);
        CHECK(d.staleStubs >= 1, "no stub was made (%llu)", static_cast<unsigned long long>(d.staleStubs));
    }
}

// Two runs of one loop with an epoch bump between: the same stub serves both, and meeting the new epoch costs a dispatcher
// round trip (the refresh of the entry block resets the slots aimed in the old epoch: the loop is linked again).
void CaseEpoch() {
    const Program p = Reached();
    const CpuContext init = InitialContext(p.base, kIterations);
    const Result want = RunOnce(PpcInterpreter::Engine::Classic, p, init);
    const Result first = RunOnce(PpcInterpreter::Engine::Jit, p, init);
    CHECK(Same(want, first), "epoch: the first run differs");
    // the first run's exit from the entry block was not linked (the loop was compiled after it); a second run links it
    const Result again = RunOnce(PpcInterpreter::Engine::Jit, p, init, false);
    CHECK(Same(want, again), "epoch: the second run differs");
    const JitX64Stats mid = JitX64GetStats();
    BrawlDirtyMemory::MarksCleared();  // a consume: a new epoch, the marks stay as they are
    const Result second = RunOnce(PpcInterpreter::Engine::Jit, p, init, false);
    const JitX64Stats end = JitX64GetStats();
    CHECK(Same(want, second), "epoch: the run after the epoch bump differs");
    CHECK(end.staleStubs == mid.staleStubs, "epoch: a second stub was made (%llu -> %llu)", static_cast<unsigned long long>(mid.staleStubs),
          static_cast<unsigned long long>(end.staleStubs));
    const uint64_t enters = end.enters - mid.enters;
    CHECK(enters >= 2 && enters <= 4, "epoch: %llu dispatcher entries for the run after the bump (the stub declines the old epoch once)",
          static_cast<unsigned long long>(enters));
}

// A loop with a stub, its code rewritten from outside, run again: the old block and its stub are dropped, the new block gets one.
void CaseRetire() {
    Program p = Loop();
    const CpuContext init = InitialContext(p.base, kIterations);
    RunOnce(PpcInterpreter::Engine::Jit, p, init);
    const JitX64Stats mid = JitX64GetStats();
    p.words[1].second = Addi(5, 5, 2);  // addi r5,r5,2
    const Result want = RunOnce(PpcInterpreter::Engine::Classic, p, init);
    const Result got = RunOnce(PpcInterpreter::Engine::Jit, p, init);
    CHECK(Same(want, got), "retire: the rewritten loop differs (r5 %u vs %u)", want.ctx.gpr[5], got.ctx.gpr[5]);
    CHECK(JitX64GetStats().staleStubs > mid.staleStubs, "retire: the rebuilt block got no stub");
    // and the data word rewritten into the loop's own code range (a write to a word the block holds, from outside)
    Memory::Write32(p.base + 8, Bc(16, 0, -8));  // the same word again: a mark, no change
    const Result again = RunOnce(PpcInterpreter::Engine::Jit, p, init, false);
    CHECK(Same(want, again), "retire: a run after a write that changed nothing differs");
}

// Other threads run the loop too: each has its own cache (and its own stubs; the interpreter keeps a thread's cache for good).
void CaseThreads() {
    for (int t = 0; t < 3; ++t) {
        const Program p = Loop();
        const CpuContext init = InitialContext(p.base, kIterations);
        const Result want = RunOnce(PpcInterpreter::Engine::Classic, p, init);
        bool same = false;
        std::thread worker([&] {
            const Result got = RunOnce(PpcInterpreter::Engine::Jit, p, init);
            same = Same(want, got);
        });
        worker.join();
        CHECK(same, "threads: the worker's run differs");
    }
}

// The cache dropped past its code limit (BRAWL_JIT_CODE_LIMIT, set small) while stubs are live, then used again.
void CaseCodeLimit() {
    const Program p = Loop();
    const CpuContext init = InitialContext(p.base, kIterations);
    const Result want = RunOnce(PpcInterpreter::Engine::Classic, p, init);
    const uint64_t destroyedBefore = JitX64GetStats().cachesDestroyed;
    for (int round = 0; round < 6; ++round) {
        const Result got = RunOnce(PpcInterpreter::Engine::Jit, p, init, round == 0);
        CHECK(Same(want, got), "code limit: round %d differs", round);
        // straight code of many blocks, to pass the limit
        const uint32_t base = 0x80800000u + round * 0x20000u;
        for (uint32_t i = 0; i < 2000; ++i) Memory::Write32(base + 4 * i, Addi(3, 3, 1));
        Memory::Write32(base + 8000, kBlr);
        CpuContext c{};
        c.lr = kReturn;
        c.gpr[1] = 0x80180000u;
        PpcInterpreter::SetEngineOverride(PpcInterpreter::Engine::Jit);
        {
            CpuContextScope scope(&c);
            PpcInterpreter::Run(&c, base);
        }
        PpcInterpreter::ClearEngineOverride();
        CHECK(c.gpr[3] == 2000, "code limit: the straight code ran %u ops", c.gpr[3]);
    }
    CHECK(JitX64GetStats().cachesDestroyed > destroyedBefore, "code limit: the cache was never dropped (stubs live in it)");
}

void AllCases(bool withLimit) {
    CaseLoop();
    CaseRewrite();
    CaseTwoSegments();
    CaseEpoch();
    CaseRetire();
    CaseThreads();
    if (withLimit) CaseCodeLimit();
}

}  // namespace

int main() {
    // read once by the backend: set before the first use
    setenv("BRAWL_JIT_STATS", "1", 1);
    setenv("BRAWL_JIT_STALE_STUB", "8", 1);
    setenv("BRAWL_JIT_CODE_LIMIT", "100000", 1);
    Memory::Init(0x1800000);
    TranslatedFunctionRegistry::Finalize();
    if (!PpcInterpreter::JitAvailable()) {
        std::printf("jit stale: no JIT backend in this build; nothing to test\n");
        return 0;
    }
    AllCases(true);
    const int baseline = g_failures;
    const JitX64Stats s = JitX64GetStats();
    std::printf("jit stale: %llu stubs made (%llu failed), %d failures\n", static_cast<unsigned long long>(s.staleStubs),
                static_cast<unsigned long long>(s.staleStubFailures), baseline);

    // the stub made wrong on purpose, four ways: each must show
    int notNoticed = 0;
    g_quiet = true;
    for (int fault : {1, 3, 4, 5}) {
        setenv("BRAWL_JIT_STALE_FAULT", std::to_string(fault).c_str(), 1);
        const int before = g_failures;
        AllCases(false);
        const int noticed = g_failures - before;
        std::printf("jit stale: fault %d: %d checks failed\n", fault, noticed);
        if (noticed == 0) ++notNoticed;
        g_failures = before;  // the faults' failures are the expected outcome
    }
    unsetenv("BRAWL_JIT_STALE_FAULT");
    g_quiet = false;
    CHECK(notNoticed == 0, "%d of the 4 deliberate faults were not noticed", notNoticed);
    std::printf("jit stale: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
