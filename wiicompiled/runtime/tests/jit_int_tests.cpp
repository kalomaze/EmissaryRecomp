// The JIT's integer group (src/jit/jit_x64_int.cpp) against the classic interpreter, one instruction at a time: every
// integer, compare and spr-move kind with its Rc forms, register aliasings (destination equal to a source, a == b, r0 as
// a source) and immediate and rotate edge cases, each run from many register states (every pair of edge values in two
// source registers, XER carry and summary-overflow cycling, a random CR) under Engine::Classic and Engine::Jit
// (threshold 0, so the block is compiled before its first run), and the whole contexts compared bytewise: a CR field must hold
// exactly the 64-bit value the interpreter stores (jit_test_cr.h), whatever kind of value it started as.
//
// An instruction is placed as `insn; mflr r6; mtlr r31; blr` (r31 holds the return address; mtspr 8 may change lr).
// BRAWL_JIT_FALLBACK=int in the environment runs the same test with the group's native code off, which must pass too.
#include "abi_bridge.h"
#include "jit_test_cr.h"
#include "memory.h"
#include "ppc_interpreter.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kReturn = 0x80020000u;
constexpr uint32_t kCodeBase = 0x80100000u;
constexpr uint32_t kBlr = 0x4E800020u;

constexpr uint32_t kEdges[] = {0u, 1u, 2u, 3u, 31u, 32u, 33u, 63u, 64u, 0x7FFFFFFFu, 0x80000000u, 0x80000001u,
                               0xFFFFFFFFu, 0xFFFFFFFEu, 0x12345678u, 0xFFFF0000u, 0x0000FFFFu, 0x00008000u, 0x000000FFu,
                               0xFFFFFF80u, 0x00007FFFu, 0xFFFF8000u};
constexpr size_t kEdgeCount = sizeof(kEdges) / sizeof(kEdges[0]);

struct Insn {
    std::string name;
    uint32_t word;
    bool wide;  // reads two registers whose values matter: all pairs of edges; otherwise a smaller set
};
std::vector<Insn> g_insns;

constexpr uint32_t D(uint32_t op, uint32_t rt, uint32_t ra, uint32_t imm) {
    return (op << 26) | (rt << 21) | (ra << 16) | (imm & 0xFFFFu);
}
constexpr uint32_t X(uint32_t rt, uint32_t ra, uint32_t rb, uint32_t xo, uint32_t rc = 0) {
    return (31u << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (xo << 1) | rc;
}
constexpr uint32_t M(uint32_t op, uint32_t rs, uint32_t ra, uint32_t shOrRb, uint32_t mb, uint32_t me, uint32_t rc) {
    return (op << 26) | (rs << 21) | (ra << 16) | (shOrRb << 11) | (mb << 6) | (me << 1) | rc;
}
constexpr uint32_t Spr(uint32_t xo, uint32_t rt, uint32_t spr) {
    return X(rt, spr & 31u, spr >> 5, xo);
}

char g_buf[128];
template <typename... A>
std::string Fmt(const char* f, A... a) {
    std::snprintf(g_buf, sizeof g_buf, f, a...);
    return g_buf;
}

void Add(std::string name, uint32_t word, bool wide) { g_insns.push_back({std::move(name), word, wide}); }

// --- the instruction lists ----------------------------------------------------------------------------------------------

constexpr uint32_t kRegs[] = {0, 3, 4, 5};

void AddImmediates() {
    const int32_t imms[] = {0, 1, -1, 0x7FFF, -0x8000, 0x1234, -0x1234, 0x5A5A, 0xFF, 0x8000, 0xFFFF, 31, 32, 0x7F, -0x80};
    for (int32_t i : imms) {
        for (uint32_t rd : kRegs)
            for (uint32_t ra : kRegs) {
                Add(Fmt("addi r%u,r%u,%d", rd, ra, i), D(14, rd, ra, i), false);
                Add(Fmt("addis r%u,r%u,%d", rd, ra, i), D(15, rd, ra, i), false);
                Add(Fmt("mulli r%u,r%u,%d", rd, ra, i), D(7, rd, ra, i), false);
                Add(Fmt("subfic r%u,r%u,%d", rd, ra, i), D(8, rd, ra, i), false);
                Add(Fmt("ori r%u,r%u,%d", ra, rd, i), D(24, rd, ra, i), false);
                Add(Fmt("oris r%u,r%u,%d", ra, rd, i), D(25, rd, ra, i), false);
                Add(Fmt("xori r%u,r%u,%d", ra, rd, i), D(26, rd, ra, i), false);
                Add(Fmt("xoris r%u,r%u,%d", ra, rd, i), D(27, rd, ra, i), false);
                Add(Fmt("andi. r%u,r%u,%d", ra, rd, i), D(28, rd, ra, i), false);
                Add(Fmt("andis. r%u,r%u,%d", ra, rd, i), D(29, rd, ra, i), false);
            }
    }
}

void AddCompares() {
    const int32_t imms[] = {0, 1, -1, 0x7FFF, -0x8000, 0x1234, 31, 64, 0xFF, 0x8000, 0xFFFF};
    for (uint32_t crf = 0; crf < 8; ++crf) {
        for (uint32_t ra : kRegs) {
            for (int32_t i : imms) {
                Add(Fmt("cmpwi cr%u,r%u,%d", crf, ra, i), D(11, crf << 2, ra, i), false);
                Add(Fmt("cmplwi cr%u,r%u,%d", crf, ra, i), D(10, crf << 2, ra, i), false);
            }
            for (uint32_t rb : kRegs) {
                Add(Fmt("cmpw cr%u,r%u,r%u", crf, ra, rb), X(crf << 2, ra, rb, 0), true);
                Add(Fmt("cmplw cr%u,r%u,r%u", crf, ra, rb), X(crf << 2, ra, rb, 32), true);
            }
        }
    }
}

void AddThreeRegister() {
    struct Op { const char* name; uint32_t xo; bool rcAllowed; bool dstIsRa; };
    // Arithmetic: rt, ra, rb (destination rt). Logical: rs, ra, rb (destination ra, sources rs and rb).
    const Op arith[] = {{"add", 266, false, false}, {"subf", 40, false, false}, {"mullw", 235, false, false}};
    const Op logic[] = {{"or", 444, true, true}, {"and", 28, true, true}, {"xor", 316, false, true}, {"andc", 60, false, true},
                        {"nor", 124, false, true}, {"slw", 24, false, true}, {"srw", 536, false, true}};
    for (uint32_t x : kRegs)
        for (uint32_t y : kRegs)
            for (uint32_t z : kRegs) {
                for (const Op& op : arith) Add(Fmt("%s r%u,r%u,r%u", op.name, x, y, z), X(x, y, z, op.xo), true);
                for (const Op& op : logic)
                    for (uint32_t rc = 0; rc <= (op.rcAllowed ? 1u : 0u); ++rc)
                        Add(Fmt("%s%s r%u,r%u,r%u", op.name, rc ? "." : "", y, x, z), X(x, y, z, op.xo, rc), true);
            }
    for (uint32_t x : kRegs)
        for (uint32_t y : kRegs) {
            Add(Fmt("neg r%u,r%u", x, y), X(x, y, 0, 104), false);
            Add(Fmt("extsb r%u,r%u", y, x), X(x, y, 0, 954), false);
            Add(Fmt("extsh r%u,r%u", y, x), X(x, y, 0, 922), false);
            Add(Fmt("cntlzw r%u,r%u", y, x), X(x, y, 0, 26), false);
            for (uint32_t sh = 0; sh < 32; ++sh) Add(Fmt("srawi r%u,r%u,%u", y, x, sh), X(x, y, sh, 824), false);
        }
}

void AddRotates() {
    const uint32_t bounds[] = {0, 1, 8, 15, 16, 23, 24, 30, 31};
    const uint32_t pairs[][2] = {{3, 3}, {3, 4}, {4, 3}, {0, 3}, {3, 0}, {0, 0}};
    for (const auto& p : pairs) {
        const uint32_t s = p[0], d = p[1];
        for (uint32_t rc = 0; rc < 2; ++rc) {
            for (uint32_t sh = 0; sh < 32; ++sh) {
                if (sh % 5 == 0 || sh == 31 || sh == 1)
                    for (uint32_t mb : bounds)
                        for (uint32_t me : bounds) {
                            Add(Fmt("rlwinm%s r%u,r%u,%u,%u,%u", rc ? "." : "", d, s, sh, mb, me), M(21, s, d, sh, mb, me, rc), false);
                            if (!rc) Add(Fmt("rlwimi r%u,r%u,%u,%u,%u", d, s, sh, mb, me), M(20, s, d, sh, mb, me, 0), true);
                        }
                // slwi, srwi, clrlwi, extrwi: the shift and mask forms with their own code
                Add(Fmt("slwi%s r%u,r%u,%u", rc ? "." : "", d, s, sh), M(21, s, d, sh, 0, 31 - sh, rc), false);
                if (sh != 0) Add(Fmt("srwi%s r%u,r%u,%u", rc ? "." : "", d, s, 32 - sh), M(21, s, d, 32 - sh, sh, 31, rc), false);
                Add(Fmt("rotlwi%s r%u,r%u,%u", rc ? "." : "", d, s, sh), M(21, s, d, sh, 0, 31, rc), false);
                Add(Fmt("rlwinm%s r%u,r%u,%u,24,31", rc ? "." : "", d, s, sh), M(21, s, d, sh, 24, 31, rc), false);
                Add(Fmt("rlwinm%s r%u,r%u,%u,16,31", rc ? "." : "", d, s, sh), M(21, s, d, sh, 16, 31, rc), false);
                Add(Fmt("rlwinm%s r%u,r%u,%u,31,31", rc ? "." : "", d, s, sh), M(21, s, d, sh, 31, 31, rc), false);
                if (!rc) {
                    Add(Fmt("rlwimi r%u,r%u,%u,mb=sh,me=31-sh", d, s, sh), M(20, s, d, sh, sh, 31 - sh, 0), true);
                    Add(Fmt("rlwimi r%u,r%u,%u,24,31", d, s, sh), M(20, s, d, sh, 24, 31, 0), true);
                    Add(Fmt("rlwimi r%u,r%u,%u,0,31", d, s, sh), M(20, s, d, sh, 0, 31, 0), true);
                }
            }
        }
    }
    for (uint32_t d : kRegs)
        for (uint32_t s : kRegs)
            for (uint32_t b : kRegs)
                for (uint32_t rc = 0; rc < 2; ++rc)
                    for (uint32_t mb : {0u, 5u, 16u, 24u, 31u})
                        for (uint32_t me : {0u, 7u, 16u, 31u})
                            Add(Fmt("rlwnm%s r%u,r%u,r%u,%u,%u", rc ? "." : "", d, s, b, mb, me), M(23, s, d, b, mb, me, rc), true);
    for (uint32_t d : kRegs)
        for (uint32_t s : kRegs)
            for (uint32_t b : kRegs)
                Add(Fmt("rlwnm r%u,r%u,r%u,0,31", d, s, b), M(23, s, d, b, 0, 31, 0), true);
}

void AddSpecials() {
    const uint32_t sprs[] = {1, 8, 9, 26, 27, 912, 913, 914, 915, 916, 917, 918, 919, 920, 1008, 1009, 22, 18, 272, 528};
    for (uint32_t r : {0u, 3u, 4u}) {
        Add(Fmt("mfcr r%u", r), X(r, 0, 0, 19), false);
        for (uint32_t spr : sprs) {
            Add(Fmt("mtspr %u,r%u", spr, r), Spr(467, r, spr), false);
            if (spr != 22 && spr != 18 && spr != 272 && spr != 528) Add(Fmt("mfspr r%u,%u", r, spr), Spr(339, r, spr), false);
        }
    }
}

// --- running -----------------------------------------------------------------------------------------------------------------

CpuContext StateFor(size_t i, bool wide, size_t k) {
    CpuContext c{};
    uint64_t s = 0x9E3779B97F4A7C15ull * (i * 1315423911ull + k + 1);
    auto next = [&] { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return static_cast<uint32_t>((s * 0x2545F4914F6CDD1Dull) >> 32); };
    for (auto& g : c.gpr) g = next();
    const size_t x = wide ? k / kEdgeCount : k % kEdgeCount, y = wide ? k % kEdgeCount : (k / kEdgeCount) % kEdgeCount;
    c.gpr[3] = kEdges[x % kEdgeCount];
    c.gpr[4] = kEdges[y % kEdgeCount];
    c.gpr[5] = (k & 1) ? kEdges[(x + y) % kEdgeCount] : next();
    c.gpr[0] = (k & 2) ? kEdges[(x * 3 + y) % kEdgeCount] : next();
    c.gpr[1] = 0x80180000u;
    JitTestCr::Randomize(c, next);
    const uint32_t xers[] = {0u, 0x20000000u, 0x60000000u, 0xE0000000u};  // SO set: compiled code does not run (JitBlocksMayRun)
    c.xer = xers[k % 4] | (next() & 0x7Fu);
    c.ctr = next();
    c.lr = kReturn;
    c.gpr[31] = kReturn;
    for (auto& q : c.gqr) q = next();
    c.srr0 = next();
    c.srr1 = next();
    c.hid0 = next();
    c.hid1 = next();
    c.hid2 = next();
    return c;
}

CpuContext RunOne(PpcInterpreter::Engine engine, uint32_t entry, const CpuContext& init) {
    PpcInterpreter::SetEngineOverride(engine);
    CpuContext ctx = init;
    ctx.pc = 0;
    {
        CpuContextScope scope(&ctx);
        PpcInterpreter::Run(&ctx, entry);
    }
    ctx.pc = 0;
    return ctx;
}

}  // namespace

int main() {
    Memory::Init(0x200000);
    TranslatedFunctionRegistry::Finalize();
    if (!PpcInterpreter::JitAvailable()) {
        std::printf("jit int: no JIT backend in this build; nothing to test\n");
        return 0;
    }
    AddImmediates();
    AddCompares();
    AddThreeRegister();
    AddRotates();
    AddSpecials();
    PpcInterpreter::SetJitThreshold(0);

    uint64_t runs = 0, failures = 0;
    constexpr uint32_t kSlots = 8192;
    for (size_t i = 0; i < g_insns.size(); ++i) {
        const Insn& insn = g_insns[i];
        const uint32_t base = kCodeBase + (static_cast<uint32_t>(i) % kSlots) * 16u;
        Memory::Write32(base, insn.word);
        Memory::Write32(base + 4, Spr(339, 6, 8));   // mflr r6: what mtspr 8 left
        Memory::Write32(base + 8, Spr(467, 31, 8));  // mtlr r31: the return address, set in every state
        Memory::Write32(base + 12, kBlr);
        const size_t states = insn.wide ? kEdgeCount * kEdgeCount : 64;
        for (size_t k = 0; k < states; ++k) {
            const CpuContext init = StateFor(i, insn.wide, k);
            const CpuContext want = RunOne(PpcInterpreter::Engine::Classic, base, init);
            const CpuContext got = RunOne(PpcInterpreter::Engine::Jit, base, init);
            ++runs;
            if (std::memcmp(&want, &got, sizeof want) != 0) {
                if (++failures <= 12) {
                    std::fprintf(stderr, "jit int: %s (0x%08X) differs for state %zu\n", insn.name.c_str(), insn.word, k);
                    for (int g = 0; g < 32; ++g)
                        if (want.gpr[g] != got.gpr[g])
                            std::fprintf(stderr, "  r%d: want %08X got %08X (init %08X)\n", g, want.gpr[g], got.gpr[g], init.gpr[g]);
                    std::fputs(JitTestCr::Diff(want, got, init).c_str(), stderr);
                    if (want.xer != got.xer) std::fprintf(stderr, "  xer: want %08X got %08X (init %08X)\n", want.xer, got.xer, init.xer);
                }
            }
        }
    }
    PpcInterpreter::ClearEngineOverride();
    std::printf("jit int: %zu instructions, %llu runs, %llu failures\n", g_insns.size(),
                static_cast<unsigned long long>(runs), static_cast<unsigned long long>(failures));
    return failures == 0 ? 0 : 1;
}
