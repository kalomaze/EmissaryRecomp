// The Gecko code handler's native body (src/hle/gecko_handler.cpp, BrawlGecko_CodeHandler) against the handler binary
// itself (projects/brawl/gecko/codehandler.bin, entry 0x800018A8) run by the interpreter.
//
// A scenario is an initial image of a small universe of guest memory (the handler and its Gecko registers, a code list,
// a stack, target zones in MEM1 and MEM2 under the cached and uncached views, a pointer table) and a deterministic
// script of passes and events. Every engine runs the whole script from the same image:
//
//   REF           the interpreter runs the binary
//   NATIVE        BrawlGecko_CodeHandler with the write plan off (when the build has a plan; otherwise the only body)
//   PLAN          the same with the plan on (BrawlGecko_PlanControl(1)), only when the symbol exists
//   PLAN_NOMARKS  the plan without trust in dirty marks (mode 2)
//
// After every pass a digest is recorded: the register file (GPRs, CR, XER, CTR, LR, f2, f3) and the universe's bytes;
// every 16 passes and at the end also all of MEM1 and MEM2. The digest sequences of all engines must be equal. The
// rollback events use the real BrawlRollback::WriteWatch over chosen 64 KiB pages (their marks are consumed by Capture,
// a restore rewrites the pages unmarked), in the order SnapshotSave / SnapshotRestore use; other events are game writes
// through Memory::Write8/16/32, unmarked rewrites followed by BrawlDirtyMemory::MarksCleared (a save-state load), the
// watch dropped and made again over another page set, and no watch at all.
//
// Code lists: the hand-written ones (one per behaviour, each under five tracking configurations) and random ones
// (GECKO_TEST_RANDOM=<count>, default 500). C0 bodies are real PowerPC words (a small assembler below) that dump the
// register file they see into memory, store to targets, Gecko registers and list bytes, and change r6/r7/r8/r16.
//
// Environment: GECKO_TEST_RANDOM=<n>, GECKO_TEST_SEED=<first seed>, GECKO_TEST_ONLY=<seed> (one random scenario),
// GECKO_TEST_NAME=<substring> (hand-written scenarios whose name contains it), GECKO_TEST_INJECT=1 (a result is
// corrupted on purpose: the run must report a mismatch and fail), GECKO_TEST_VERBOSE=1.
#include "abi_bridge.h"
#include "brawl_dirty_memory.h"
#include "brawl_write_watch.h"
#include "memory.h"
#include "memory_access.h"
#include "ppc_interpreter.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifndef GECKO_TEST_HANDLER_BIN
#error "GECKO_TEST_HANDLER_BIN must name the handler binary"
#endif

// Present once the write plan exists (src/hle/gecko_handler.cpp); weak, so that today's handler links too.
extern "C" void BrawlGecko_PlanControl(int mode) __attribute__((weak));  // 0 off, 1 on, 2 on without mark trust
extern "C" void BrawlGecko_PlanCounters(uint64_t* out, size_t count) __attribute__((weak));
extern "C" void BrawlGecko_CodeHandler(CpuContext* ctx);
extern "C" void func_800018A8_body(CpuContext* ctx);
extern "C" void func_80001E88(CpuContext* ctx);

constexpr uint32_t kEntry = 0x800018A8u;

// What gecko_handler.cpp links against here. The translated handler is the binary under the interpreter; the 8A/8C
// subroutine runs in the binary too (its caller has set lr).
extern "C" void func_800018A8_body(CpuContext* ctx) { PpcInterpreter::Run(ctx, kEntry); }
extern "C" void func_80001E88(CpuContext* ctx) { PpcInterpreter::Run(ctx, 0x80001E88u); }
namespace AxDspHle {
void JoinMixWorker() {}
}  // namespace AxDspHle

namespace {

using std::vector;

// ---------------------------------------------------------------------------------------------------------------------
// The universe

constexpr uint32_t kHandlerBase = 0x80001800u;
constexpr uint32_t kListPtr = 0x80001804u;
constexpr uint32_t kGr = 0x80001808u;        // r7 at the handler's start: gr0..gr15, then twelve 8-byte blocks
constexpr uint32_t kListA = 0x80566528u;     // up to 64 KiB
constexpr uint32_t kListB = 0x80578000u;
constexpr uint32_t kStackTop = 0x805B9000u;  // r1 at entry; the frame and the dumps sit below it
constexpr uint32_t kAlt = 0x805D0000u;       // a second Gecko register table
constexpr uint32_t kPtrTable = 0x80700000u;  // safe values for ba / po: loaded by 40/48 codes
constexpr uint32_t kReturn = 0x80030000u;    // lr at entry: no code there, the interpreter stops on reaching it

struct Zone {
    uint32_t off, size;
};
constexpr Zone kZones[3] = {{0x400000u, 0x40000u}, {0x100000u, 0x20000u}, {0x1100000u, 0x20000u}};
constexpr uint32_t kBases[4] = {0x80000000u, 0xC0000000u, 0x90000000u, 0xD0000000u};

vector<uint32_t> g_pages;  // the universe's 64 KiB pages (cached views)

void BuildPages() {
    g_pages = {0x80000000u, 0x80560000u, 0x80570000u, 0x805B0000u, 0x805D0000u, 0x80700000u};
    for (uint32_t bank : {0x80000000u, 0x90000000u})
        for (const Zone& z : kZones)
            for (uint32_t at = 0; at < z.size; at += 0x10000u) g_pages.push_back(bank + z.off + at);
}

uint8_t* Host(uint32_t address, uint32_t size) { return Memory::GetPointer(address, size); }

#if MKW_LEAN_MEMORY
inline uint32_t Canon(uint32_t address) { return address & 0x3FFFFFFFu; }
uint8_t MarkOf(uint32_t address) { return BrawlDirtyMemory::leanBlocks[Canon(address) >> 6]; }
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Hashing

inline uint64_t Mix(uint64_t h, uint64_t v) {
    h ^= v;
    h *= 0x9E3779B97F4A7C15ull;
    return h ^ (h >> 31);
}
uint64_t HashBytes(const uint8_t* p, size_t n, uint64_t h) {  // n a multiple of 8
    uint64_t a = h, b = h ^ 0x1111, c = h ^ 0x2222, d = h ^ 0x3333;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        uint64_t w[4];
        std::memcpy(w, p + i, 32);
        a = Mix(a, w[0]);
        b = Mix(b, w[1]);
        c = Mix(c, w[2]);
        d = Mix(d, w[3]);
    }
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        a = Mix(a, w);
    }
    return Mix(Mix(Mix(Mix(h, a), b), c), d);
}

// ---------------------------------------------------------------------------------------------------------------------
// A tiny PowerPC assembler: enough for the C0 bodies.

namespace Ppc {
constexpr uint32_t D(uint32_t op, uint32_t rt, uint32_t ra, int32_t imm) {
    return (op << 26) | (rt << 21) | (ra << 16) | (static_cast<uint32_t>(imm) & 0xFFFFu);
}
constexpr uint32_t Lis(uint32_t r, uint32_t imm) { return D(15, r, 0, static_cast<int32_t>(imm & 0xFFFFu)); }
constexpr uint32_t Ori(uint32_t ra, uint32_t rs, uint32_t imm) { return D(24, rs, ra, static_cast<int32_t>(imm & 0xFFFFu)); }
constexpr uint32_t Addi(uint32_t rt, uint32_t ra, int32_t imm) { return D(14, rt, ra, imm); }
constexpr uint32_t Li(uint32_t rt, int32_t imm) { return Addi(rt, 0, imm); }
constexpr uint32_t Lwz(uint32_t rt, int32_t d, uint32_t ra) { return D(32, rt, ra, d); }
constexpr uint32_t Stw(uint32_t rs, int32_t d, uint32_t ra) { return D(36, rs, ra, d); }
constexpr uint32_t Stb(uint32_t rs, int32_t d, uint32_t ra) { return D(38, rs, ra, d); }
constexpr uint32_t Sth(uint32_t rs, int32_t d, uint32_t ra) { return D(44, rs, ra, d); }
constexpr uint32_t Stmw(uint32_t rs, int32_t d, uint32_t ra) { return D(47, rs, ra, d); }
constexpr uint32_t Stfd(uint32_t fs, int32_t d, uint32_t ra) { return D(54, fs, ra, d); }
constexpr uint32_t Mfcr(uint32_t r) { return 0x7C000026u | (r << 21); }
constexpr uint32_t Mfxer(uint32_t r) { return 0x7C0102A6u | (r << 21); }
constexpr uint32_t Mflr(uint32_t r) { return 0x7C0802A6u | (r << 21); }
constexpr uint32_t Mfctr(uint32_t r) { return 0x7C0902A6u | (r << 21); }
constexpr uint32_t Mtxer(uint32_t r) { return 0x7C0103A6u | (r << 21); }
constexpr uint32_t Mtctr(uint32_t r) { return 0x7C0903A6u | (r << 21); }
constexpr uint32_t Mtcrf(uint32_t fxm, uint32_t r) { return 0x7C000120u | (r << 21) | (fxm << 12); }
constexpr uint32_t Blr = 0x4E800020u;
constexpr uint32_t Nop = 0x60000000u;
constexpr uint32_t B(int32_t byteDisp) { return 0x48000000u | (static_cast<uint32_t>(byteDisp) & 0x03FFFFFCu); }

void Load32(vector<uint32_t>& w, uint32_t r, uint32_t v) {
    w.push_back(Lis(r, v >> 16));
    w.push_back(Ori(r, r, v & 0xFFFFu));
}
}  // namespace Ppc

// ---------------------------------------------------------------------------------------------------------------------
// Scenarios

struct Entry {
    uint32_t gpr[32];
    uint32_t cr, xer, ctr, lr, msr;
    uint64_t f2, f3;
};

struct Mut {  // words or bytes of the list (or elsewhere) the game or a C0 body may rewrite without breaking the list
    uint32_t addr;
    uint32_t len;
    vector<vector<uint8_t>> variants;  // each of `len` bytes
    bool any = false;                  // any bytes at all are fine
};

struct Step {
    enum Kind { Pass, Write, WriteBytes, Save, Restore, Unmarked, WatchOn, WatchOff } kind = Pass;
    uint32_t addr = 0, value = 0, size = 0;
    vector<uint8_t> bytes;
    int slot = 0;
    vector<uint32_t> pages;
    Entry entry{};
};

struct Span {
    uint32_t addr, len;
};

struct Scenario {
    std::string name;
    uint64_t seed = 0;
    vector<std::pair<uint32_t, vector<uint8_t>>> image;
    vector<Step> steps;
    vector<uint32_t> tracked;  // pages the watch covers from the start (empty: no watch)
    bool expectReplay = false, expectTrust = false;
    bool consumeEachPass = false;  // a rollback save (marks consumed) before every pass: a pass then starts with clean marks
    // what the script generator may touch
    vector<Span> targets, condMem;
    vector<Mut> muts;
    bool gregWrites = true;
    uint32_t listAddr = kListA;
};

void Put(Scenario& s, uint32_t addr, const vector<uint8_t>& bytes) { s.image.push_back({addr, bytes}); }
void Put32(Scenario& s, uint32_t addr, uint32_t v) {
    Put(s, addr, {uint8_t(v >> 24), uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)});
}
vector<uint8_t> Bytes(const vector<uint32_t>& words) {
    vector<uint8_t> b;
    for (uint32_t v : words) {
        b.push_back(uint8_t(v >> 24));
        b.push_back(uint8_t(v >> 16));
        b.push_back(uint8_t(v >> 8));
        b.push_back(uint8_t(v));
    }
    return b;
}

using Rng = std::mt19937_64;
inline uint32_t R(Rng& g, uint32_t n) { return n == 0 ? 0 : static_cast<uint32_t>(g() % n); }
inline bool Chance(Rng& g, uint32_t percent) { return R(g, 100) < percent; }

Entry RandomEntry(Rng& g, uint32_t r1) {
    Entry e{};
    for (auto& r : e.gpr) r = static_cast<uint32_t>(g());
    e.gpr[1] = r1;
    e.cr = static_cast<uint32_t>(g());
    e.xer = static_cast<uint32_t>(g()) & 0xE000007Fu;
    e.ctr = static_cast<uint32_t>(g());
    e.lr = kReturn;
    e.msr = 0x00002032u ^ (R(g, 4) << 14) ^ (R(g, 2) << 4);
    const auto dbl = [&] {
        const uint32_t k = R(g, 10);
        double d = 0.0;
        if (k == 0) d = 0.0;
        else if (k == 1) d = -0.0;
        else if (k == 2) d = 1.0 / 0.0;
        else d = (static_cast<double>(static_cast<int64_t>(g() >> 11) - (int64_t{1} << 52)) / 4503599627370496.0) * (1 << R(g, 40));
        uint64_t bits;
        std::memcpy(&bits, &d, 8);
        return bits;
    };
    e.f2 = dbl();
    e.f3 = dbl();
    return e;
}

// ---------------------------------------------------------------------------------------------------------------------
// The code list builder, shared by the hand-written and the random lists

struct Gen {
    Scenario sc;
    vector<uint32_t> w;
    uint32_t base = kListA;
    uint32_t ba = 0x80000000u, po = 0x80000000u;  // the bases the next code is written for (straight-line execution)
    int slot = 0;

    Gen() { w = {0x00D0C0DEu, 0x00D0C0DEu}; }
    uint32_t Here() const { return base + 4u * static_cast<uint32_t>(w.size()); }
    size_t Lines() const { return w.size() / 2; }
    void C(uint32_t hi, uint32_t lo) {
        w.push_back(hi);
        w.push_back(lo);
    }
    uint32_t Off(bool usePo, uint32_t a) const {
        const uint32_t b = usePo ? po : (ba & 0xFE000000u);
        const uint32_t off = a - b;
        if (off >= 0x02000000u) {
            std::fprintf(stderr, "gecko test: address %08X not reachable from base %08X\n", a, b);
            std::abort();
        }
        return off;
    }
    void Target(uint32_t a, uint32_t len) { sc.targets.push_back({a, len}); }
    static uint32_t T(bool usePo) { return usePo ? 0x10000000u : 0u; }

    // ---- writes
    void W8(uint32_t a, uint8_t v, uint32_t n = 1, bool p = false) {
        C(T(p) | (0x00u << 24) | Off(p, a), ((n - 1) << 16) | v);
        Target(a, n);
    }
    void W16(uint32_t a, uint16_t v, uint32_t n = 1, bool p = false) {
        C(T(p) | (0x02u << 24) | Off(p, a), ((n - 1) << 16) | v);
        Target(a, 2 * n);
    }
    void W32(uint32_t a, uint32_t v, bool p = false) {
        C(T(p) | (0x04u << 24) | Off(p, a), v);
        Target(a & ~3u, 4);
    }
    void Str(uint32_t a, const vector<uint8_t>& s, bool p = false) {
        C(T(p) | (0x06u << 24) | Off(p, a), static_cast<uint32_t>(s.size()));
        vector<uint8_t> padded = s;
        while (padded.size() % 8) padded.push_back(0);
        const size_t first = w.size();
        for (size_t i = 0; i < padded.size(); i += 4)
            w.push_back(uint32_t(padded[i]) << 24 | uint32_t(padded[i + 1]) << 16 | uint32_t(padded[i + 2]) << 8 | padded[i + 3]);
        if (!s.empty()) {
            Target(a, static_cast<uint32_t>(s.size()));
            Mut m;
            m.addr = base + 4u * static_cast<uint32_t>(first);
            m.len = static_cast<uint32_t>(padded.size());
            m.any = true;
            sc.muts.push_back(m);
        }
    }
    void Serial(uint32_t a, uint32_t init, uint32_t size /*0,1,2*/, uint32_t count, uint32_t addrStep, uint32_t valStep, bool p = false) {
        C(T(p) | (0x08u << 24) | Off(p, a), init);
        C((size << 28) | (((count - 1) & 0xFFFu) << 16) | (addrStep & 0xFFFFu), valStep);
        Target(a, count * addrStep + 4);
    }
    // ---- ifs: cmp 0 eq, 1 ne, 2 gt, 3 lt (unsigned)
    void If(uint32_t cmp, uint32_t a, uint32_t value, bool sixteen = false, uint32_t ignore = 0, bool endif = false, bool p = false) {
        const uint32_t type = (sixteen ? 0x28u : 0x20u) + 2 * cmp;
        const uint32_t off = Off(p, sixteen ? (a & ~1u) : (a & ~3u)) | (endif ? 1u : 0u);
        C(T(p) | (type << 24) | off, sixteen ? ((ignore << 16) | (value & 0xFFFFu)) : value);
        sc.condMem.push_back({a & ~3u, 4});
    }
    void E2(uint32_t n, bool elseFlag = false, uint32_t baHi = 0, uint32_t poHi = 0) {
        C(0xE2000000u | (elseFlag ? 0x100000u : 0u) | (n & 0x1F), (baHi << 16) | poHi);
    }
    void E0(uint32_t baHi = 0, uint32_t poHi = 0) { C(0xE0000000u, (baHi << 16) | poHi); }
    void Else() { E2(0, true); }
    void Endif(uint32_t n = 1) { E2(n); }
    // ---- 4x
    void SetBa(uint32_t v) {
        C(0x42000000u, v);
        ba = v;
    }
    void SetPo(uint32_t v) {
        C(0x4A000000u, v);
        po = v;
    }
    // ---- 6x
    void Goto(uint32_t lines, uint32_t cond = 2) { C(0x66000000u | (cond << 20) | (lines & 0xFFFFu), 0); }
    void Gosub(uint32_t lines, uint32_t block, uint32_t cond = 0) { C(0x68000000u | (cond << 20) | (lines & 0xFFFFu), block); }
    void Return(uint32_t block, uint32_t cond = 0) { C(0x64000000u | (cond << 20), block); }
    void SetRepeat(uint32_t block, uint32_t count) { C(0x60000000u | (count & 0xFFFFu), block); }
    void ExecRepeat(uint32_t block) { C(0x62000000u, block); }
    // ---- 8x (register indices 0..15; sizes 0:8 1:16 2:32)
    void GrSet(uint32_t idx, uint32_t imm, bool addR12 = false, bool addGr = false) {
        C(0x80000000u | (addR12 ? 0x10000u : 0u) | (addGr ? 0x100000u : 0u) | idx, imm);
    }
    void GrLoad(uint32_t idx, uint32_t size, uint32_t addr, bool addR12 = false) {
        C(0x82000000u | (size << 20) | (addR12 ? 0x10000u : 0u) | idx, addr);
    }
    void GrStore(uint32_t idx, uint32_t size, uint32_t addr, uint32_t count = 1, bool addR12 = false) {
        C(0x84000000u | (size << 20) | (addR12 ? 0x10000u : 0u) | (((count - 1) & 0xFFFu) << 4) | idx, addr);
        Target(addr + (addR12 ? 0x80000000u : 0u), count * (size == 0 ? 1 : size == 1 ? 2 : 4));
    }
    void GrOpImm(uint32_t op, uint32_t dst, uint32_t imm, uint32_t flags = 0) {
        C(0x86000000u | (op << 20) | (flags << 16) | dst, imm);
    }
    void GrOpReg(uint32_t op, uint32_t dst, uint32_t src, uint32_t flags = 0) {
        C(0x88000000u | (op << 20) | (flags << 16) | dst, src);
    }
    void Copy(bool dstOffset, uint32_t dstIdx, uint32_t srcIdx, uint32_t count, uint32_t off) {
        C((dstOffset ? 0x8A000000u : 0x8C000000u) | (count << 8) | (srcIdx << 4) | dstIdx, off);
    }
    // ---- Cx
    void Body(uint32_t type, uint32_t hiLow, const vector<uint32_t>& words) {
        vector<uint32_t> padded = words;
        while (padded.size() % 2) padded.push_back(Ppc::Nop);
        C(type | hiLow, static_cast<uint32_t>(padded.size() / 2));
        for (uint32_t v : padded) w.push_back(v);
    }
    void C0(const vector<uint32_t>& words) { Body(0xC0000000u, 0, words); }
    void C2(uint32_t a, const vector<uint32_t>& words, bool p = false) {
        Body(T(p) | 0xC2000000u, Off(p, a) & ~3u, words);
        Target(a, 8);
    }
    void C4(uint32_t a, const vector<uint32_t>& words, bool p = false) {
        Body(T(p) | 0xC4000000u, Off(p, a) & ~3u, words);
        Target(a, 8);
    }
    void C6(uint32_t a, uint32_t target, bool link = false, bool p = false) {
        C(T(p) | 0xC6000000u | (Off(p, a) & ~3u) | (link ? 1u : 0u), target);
        Target(a, 4);
    }
    void CC(uint32_t lo) { C(0xCC000000u, lo); }
    void F0() { C(0xF0000000u, 0); }

    // ---- bodies
    uint32_t DumpOffset() { return static_cast<uint32_t>(-0x100 - 0xA0 * (slot++ % 150)); }
    vector<uint32_t> Dump() {
        using namespace Ppc;
        const int32_t o = static_cast<int32_t>(DumpOffset());
        return {Stmw(0, o, 1), Mfcr(0), Stw(0, o + 0x80, 1), Mfxer(0), Stw(0, o + 0x84, 1), Mfctr(0), Stw(0, o + 0x88, 1),
                Mflr(0),       Stw(0, o + 0x8C, 1), Stfd(2, o + 0x90, 1), Stfd(3, o + 0x98, 1), Lwz(0, o, 1), Blr};
    }
    static vector<uint32_t> StoreWord(uint32_t addr, uint32_t value, uint32_t size = 4) {
        using namespace Ppc;
        vector<uint32_t> b;
        Load32(b, 0, addr);
        Load32(b, 2, value);
        b.push_back(size == 4 ? Stw(2, 0, 0) : size == 2 ? Sth(2, 0, 0) : Stb(2, 0, 0));
        b.push_back(Blr);
        return b;
    }
    static vector<uint32_t> StoreGr(uint32_t idx, uint32_t value) {
        using namespace Ppc;
        vector<uint32_t> b;
        Load32(b, 2, value);
        b.push_back(Stw(2, static_cast<int32_t>(4 * idx), 7));
        b.push_back(Blr);
        return b;
    }
    struct RegSet {
        bool r6 = false, r16 = false, r8 = false, r7 = false;
        uint32_t v6 = 0, v16 = 0, v8 = 0, v7 = 0;
    };
    static vector<uint32_t> SetRegs(const RegSet& rs) {
        using namespace Ppc;
        vector<uint32_t> b;
        if (rs.r6) Load32(b, 6, rs.v6);
        if (rs.r16) Load32(b, 16, rs.v16);
        if (rs.r8) Load32(b, 8, rs.v8);
        if (rs.r7) Load32(b, 7, rs.v7);
        b.push_back(Blr);
        return b;
    }
    // Every register but the handler's loop state (r4 is the body's size, used after the call) gets a new value, and CR, XER and CTR too.
    static vector<uint32_t> Scramble(Rng& g) {
        using namespace Ppc;
        vector<uint32_t> b;
        for (uint32_t r = 2; r < 32; ++r) {
            if (r == 4 || r == 6 || r == 7 || r == 8 || r == 15 || r == 16) continue;  // r4 holds the body's size across the call
            Load32(b, r, static_cast<uint32_t>(g()));
        }
        Load32(b, 0, static_cast<uint32_t>(g()));
        b.push_back(Mtcrf(0xFF, 0));
        Load32(b, 0, static_cast<uint32_t>(g()) & 0xE000007Fu);
        b.push_back(Mtxer(0));
        Load32(b, 0, static_cast<uint32_t>(g()));
        b.push_back(Mtctr(0));
        Load32(b, 0, static_cast<uint32_t>(g()));
        b.push_back(Blr);
        return b;
    }
    // Calls the function at `addr` (a hooked target) and returns: lr is saved in r0 around it.
    static vector<uint32_t> CallThrough(uint32_t addr) {
        using namespace Ppc;
        vector<uint32_t> b;
        b.push_back(Mflr(0));
        b.push_back(Stw(0, -8, 1));
        Load32(b, 2, addr);
        b.push_back(Mtctr(2));
        b.push_back(0x4E800421u);  // bctrl
        b.push_back(Lwz(0, -8, 1));
        b.push_back(0x7C0803A6u);  // mtlr r0
        b.push_back(Blr);
        return b;
    }

    // Finish: the list's words and the image.
    Scenario Finish(const std::string& name) {
        vector<uint32_t> all = w;
        all.push_back(0xF0000000u);
        all.push_back(0);
        sc.name = name;
        sc.listAddr = base;
        Put(sc, base, Bytes(all));
        return sc;
    }
};

// Initial memory common to every scenario: the handler, its registers, the pointer table, the zones' contents.
void BaseImage(Scenario& s, Rng& g, uint32_t listAddr) {
    static vector<uint8_t> bin;
    if (bin.empty()) {
        std::FILE* f = std::fopen(GECKO_TEST_HANDLER_BIN, "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "gecko test: cannot open the handler binary\n");
            std::exit(2);
        }
        uint8_t buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) bin.insert(bin.end(), buf, buf + n);
        std::fclose(f);
        if (bin.size() != 2736) {
            std::fprintf(stderr, "gecko test: unexpected handler binary size %zu\n", bin.size());
            std::exit(2);
        }
    }
    Put(s, kHandlerBase, bin);
    Put32(s, kListPtr, listAddr);
    const auto table = [&](uint32_t at, const uint32_t ptrs[4]) {
        for (uint32_t i = 0; i < 16; ++i) Put32(s, at + 4 * i, i < 4 ? ptrs[i] : static_cast<uint32_t>(g()));
    };
    const uint32_t p1[4] = {0x80400000u, 0x90100000u, 0x81100000u, 0xC0420000u};
    const uint32_t p2[4] = {0x90400040u, 0x91100100u, 0xD0100200u, 0x80100080u};
    table(kGr, p1);
    table(kAlt, p2);
    for (uint32_t i = 0; i < 64; ++i) Put32(s, kPtrTable + 4 * i, kBases[i % 4]);
    // contents of the zones: the first 8 KiB of each page, mostly small values so that compares go both ways
    for (uint32_t bank : {0x80000000u, 0x90000000u})
        for (const Zone& z : kZones)
            for (uint32_t at = 0; at < z.size; at += 0x10000u) {
                vector<uint8_t> b(0x2000);
                for (size_t i = 0; i + 4 <= b.size(); i += 4) {
                    uint32_t v = 0;
                    switch (R(g, 6)) {
                    case 0: v = 0; break;
                    case 1: v = 1; break;
                    case 2: v = 0xFFFFFFFFu; break;
                    case 3: v = R(g, 8); break;
                    default: v = static_cast<uint32_t>(g()); break;
                    }
                    for (int k = 0; k < 4; ++k) b[i + k] = uint8_t(v >> (24 - 8 * k));
                }
                Put(s, bank + z.off + at, b);
            }
}

// ---------------------------------------------------------------------------------------------------------------------
// The script: passes and events

vector<uint32_t> PickPages(Rng& g, int config) {
    vector<uint32_t> p;
    switch (config) {
    case 0: break;                       // no watch
    case 1: p = g_pages; break;          // everything
    case 2:                              // the code stream and the handler's page only
        p = {0x80000000u, 0x80560000u, 0x80570000u};
        break;
    case 3:                              // the targets only
        for (uint32_t a : g_pages)
            if (a >= 0x80100000u && a != 0x80560000u && a != 0x80570000u && a != 0x805B0000u && a != 0x805D0000u && a != 0x80700000u)
                p.push_back(a);
        break;
    default:                             // a random half
        for (uint32_t a : g_pages)
            if (Chance(g, 50)) p.push_back(a);
        if (p.empty()) p.push_back(g_pages[0]);
        break;
    }
    return p;
}

// Game writes reach memory through any of its views.
bool g_aliasViews = true;
uint32_t View(uint32_t a, uint64_t roll) { return g_aliasViews && (roll % 100) < 15 ? (a | 0x40000000u) : a; }
uint64_t g_viewRoll = 0;

void AddWrite(Scenario& s, uint32_t addr, uint32_t value, uint32_t size) {
    Step st;
    st.kind = Step::Write;
    st.addr = View(addr, g_viewRoll = g_viewRoll * 6364136223846793005ull + 1442695040888963407ull + addr);
    st.value = value;
    st.size = size;
    s.steps.push_back(st);
}
void AddBytes(Scenario& s, Step::Kind kind, uint32_t addr, const vector<uint8_t>& bytes) {
    Step st;
    st.kind = kind;
    st.addr = addr;
    st.bytes = bytes;
    s.steps.push_back(st);
}

uint32_t RandomValue(Rng& g) {
    switch (R(g, 5)) {
    case 0: return 0;
    case 1: return 1;
    case 2: return 0xFFFFFFFFu;
    case 3: return R(g, 16);
    default: return static_cast<uint32_t>(g());
    }
}

void AddMutation(Scenario& s, Rng& g, bool unmarked) {
    if (s.muts.empty()) return;
    const Mut& m = s.muts[R(g, static_cast<uint32_t>(s.muts.size()))];
    vector<uint8_t> bytes(m.len);
    if (m.any && Chance(g, 60)) {  // a few bytes of it
        const uint32_t at = R(g, m.len), n = 1 + R(g, std::min<uint32_t>(m.len - at, 8));
        vector<uint8_t> part(n);
        for (auto& b : part) b = static_cast<uint8_t>(g());
        if (unmarked) AddBytes(s, Step::Unmarked, m.addr + at, part);
        else AddBytes(s, Step::WriteBytes, m.addr + at, part);
        return;
    }
    if (m.any) {
        for (auto& b : bytes) b = static_cast<uint8_t>(g());
    } else {
        bytes = m.variants[R(g, static_cast<uint32_t>(m.variants.size()))];
    }
    AddBytes(s, unmarked ? Step::Unmarked : Step::WriteBytes, m.addr, bytes);
}

void AddScript(Scenario& s, Rng& g, int passes, uint32_t r1, bool events = true) {
    bool watch = !s.tracked.empty();
    vector<int> saved;
    if (watch) saved.push_back(0);
    int nextSlot = 1;
    for (int p = 0; p < passes; ++p) {
        const uint32_t nev = (events && p > 0) ? R(g, 4) : 0;
        for (uint32_t e = 0; e < nev; ++e) {
            switch (R(g, 12)) {
            case 0: case 1: {  // a game write over a target
                if (s.targets.empty()) break;
                const Span& t = s.targets[R(g, static_cast<uint32_t>(s.targets.size()))];
                const uint32_t size = 1u << R(g, 3);
                const uint32_t at = t.addr + R(g, std::max(1u, t.len));
                AddWrite(s, at & ~(size - 1), RandomValue(g), size);
                break;
            }
            case 2: {  // a write in the same 64-byte line as a target
                if (s.targets.empty()) break;
                const Span& t = s.targets[R(g, static_cast<uint32_t>(s.targets.size()))];
                const uint32_t size = 1u << R(g, 3);
                const uint32_t at = (t.addr & ~63u) + R(g, 64);
                AddWrite(s, at & ~(size - 1), RandomValue(g), size);
                break;
            }
            case 3: {  // an if operand
                if (s.condMem.empty()) break;
                const Span& t = s.condMem[R(g, static_cast<uint32_t>(s.condMem.size()))];
                AddWrite(s, t.addr, RandomValue(g), 4);
                break;
            }
            case 4:  // a Gecko register
                if (s.gregWrites) AddWrite(s, kGr + 4 * (4 + R(g, 12)), RandomValue(g), 4);
                break;
            case 5: case 6: AddMutation(s, g, false); break;
            case 7:
                if (watch && saved.size() < 8) {
                    Step st;
                    st.kind = Step::Save;
                    st.slot = nextSlot++;
                    saved.push_back(st.slot);
                    s.steps.push_back(st);
                }
                break;
            case 8:
                if (watch && !saved.empty()) {
                    Step st;
                    st.kind = Step::Restore;
                    // back one to six saves, or the base image
                    const size_t back = std::min<size_t>(saved.size() - 1, R(g, 6));
                    st.slot = saved[saved.size() - 1 - back];
                    s.steps.push_back(st);
                }
                break;
            case 9: {  // unmarked rewrite of a target or of the list, then the epoch bump (a save-state load)
                if (Chance(g, 50) && !s.targets.empty()) {
                    const Span& t = s.targets[R(g, static_cast<uint32_t>(s.targets.size()))];
                    vector<uint8_t> b(1 + R(g, 16));
                    for (auto& x : b) x = static_cast<uint8_t>(g());
                    AddBytes(s, Step::Unmarked, t.addr, b);
                } else {
                    AddMutation(s, g, true);
                }
                break;
            }
            case 10:
                if (Chance(g, 30)) {  // the watch dropped, or made again over another set
                    Step st;
                    if (watch) {
                        st.kind = Step::WatchOff;
                        watch = false;
                        saved.clear();
                    } else {
                        st.kind = Step::WatchOn;
                        st.pages = PickPages(g, 1 + static_cast<int>(R(g, 4)));
                        watch = true;
                        saved = {0};
                        nextSlot = 1;
                    }
                    s.steps.push_back(st);
                }
                break;
            default: {  // a plain 4-byte rewrite of a target with its likely current value zero
                if (s.targets.empty()) break;
                const Span& t = s.targets[R(g, static_cast<uint32_t>(s.targets.size()))];
                AddWrite(s, t.addr & ~3u, 0, 4);
                break;
            }
            }
        }
        Step pass;
        pass.kind = Step::Pass;
        pass.entry = RandomEntry(g, r1);
        s.steps.push_back(pass);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The engines and the runner

enum class EngineKind { Ref, Native, Plan, PlanNoMarks };
const char* EngineName(EngineKind e) {
    switch (e) {
    case EngineKind::Ref: return "REF";
    case EngineKind::Native: return "NATIVE";
    case EngineKind::Plan: return "PLAN";
    default: return "PLAN_NOMARKS";
    }
}

struct Digest {
    uint64_t regs = 0, mem = 0, full = 0;
    bool operator==(const Digest& o) const { return regs == o.regs && mem == o.mem && full == o.full; }
};

struct Detail {
    Entry regs{};
    vector<uint8_t> universe;
    vector<uint8_t> mem1, mem2;
};

struct RunOut {
    vector<Digest> digests;
    vector<std::string> errors;
};

BrawlRollback::SnapshotJobs& Jobs() {
    static BrawlRollback::SnapshotJobs jobs;
    return jobs;
}

constexpr size_t kCounterCount = 11;
struct Counters {
    uint64_t v[kCounterCount] = {};
};
Counters ReadCounters() {
    Counters c;
    if (BrawlGecko_PlanCounters != nullptr) BrawlGecko_PlanCounters(c.v, kCounterCount);
    return c;
}
enum { kPasses, kSteps, kReplayed, kRecorded, kDropped, kKeyMiss, kCompared, kTrusted, kRewritten, kVerifyChecks, kVerifyMismatch };

struct Snap {
    vector<uint8_t> bytes;
    uint32_t seq = 0;
};

struct Watch {
    std::unique_ptr<BrawlRollback::WriteWatch> ww;
    vector<uint32_t> pages;  // sorted by physical key
    size_t total = 0;
    std::map<int, Snap> snaps;
};

void ReadPages(const vector<uint32_t>& pages, vector<uint8_t>& out) {
    out.resize(pages.size() * 0x10000);
    for (size_t i = 0; i < pages.size(); ++i) std::memcpy(out.data() + i * 0x10000, Host(pages[i], 0x10000), 0x10000);
}

void MakeWatch(Watch& w, const vector<uint32_t>& pages) {
    w.snaps.clear();
    w.pages = pages;
    std::sort(w.pages.begin(), w.pages.end(), [](uint32_t a, uint32_t b) {
        return BrawlRollback::WriteWatch::PhysicalKey(a) < BrawlRollback::WriteWatch::PhysicalKey(b);
    });
    w.ww = std::make_unique<BrawlRollback::WriteWatch>(reinterpret_cast<uintptr_t>(MemoryInline::g_leanBase));
    BrawlDirtyMemory::hosts.push_back({reinterpret_cast<uintptr_t>(Memory::GetPointer(Memory::kMem1CachedBase, Memory::kMem1BankSize)),
                                       Memory::kMem1BankSize, Memory::kMem1CachedBase});
    BrawlDirtyMemory::hosts.push_back({reinterpret_cast<uintptr_t>(Memory::GetPointer(Memory::kMem2CachedBase, Memory::kMem2Size)),
                                       Memory::kMem2Size, Memory::kMem2CachedBase});
    size_t packed = 0;
    for (uint32_t page : w.pages) {
        w.ww->Add(Host(page, 0x10000), 0x10000, packed, page);
        packed += 0x10000;
    }
    w.total = packed;
    // the initial save: the base image, then Enable
    Snap s;
    ReadPages(w.pages, s.bytes);
    s.seq = BrawlDirtyMemory::consumeSeq.load(std::memory_order_acquire);
    w.snaps[0] = std::move(s);
    w.ww->Enable();
}

void WatchSave(Watch& w, int slot) {
    w.ww->Capture(w.total, Jobs());
    Snap s;
    ReadPages(w.pages, s.bytes);
    s.seq = BrawlDirtyMemory::consumeSeq.load(std::memory_order_acquire);
    w.snaps[slot] = std::move(s);
}

void WatchRestore(Watch& w, int slot) {
    const Snap& s = w.snaps.at(slot);
    BrawlDirtyMemory::MarksRestored(s.seq);
    w.ww->Capture(w.total, Jobs());
    for (size_t i = 0; i < w.pages.size(); ++i) std::memcpy(Host(w.pages[i], 0x10000), s.bytes.data() + i * 0x10000, 0x10000);
}

void ResetUniverse() {
    for (uint32_t page : g_pages) {
        std::memset(Host(page, 0x10000), 0, 0x10000);
#if MKW_LEAN_MEMORY
        std::memset(BrawlDirtyMemory::leanBlocks + (Canon(page) >> 6), 0, 1024);
#endif
    }
    BrawlDirtyMemory::MarksCleared();
}

void ZeroEverything() {
    std::memset(Host(0x80000000u, Memory::kMem1Size), 0, Memory::kMem1Size);
    std::memset(Host(0x90000000u, Memory::kMem2Size), 0, Memory::kMem2Size);
    BrawlDirtyMemory::MarksCleared();
}

// The universe's digest: a page's hash is made again only when its bytes differ from what was hashed last.
struct UniverseHash {
    vector<uint8_t> seen;
    vector<uint64_t> hashes;
    bool valid = false;
    uint64_t Digest() {
        if (!valid) {
            seen.assign(g_pages.size() * 0x10000, 0);
            hashes.assign(g_pages.size(), 0);
        }
        uint64_t h = 0x1234567;
        for (size_t i = 0; i < g_pages.size(); ++i) {
            const uint8_t* now = Host(g_pages[i], 0x10000);
            if (!valid || std::memcmp(seen.data() + i * 0x10000, now, 0x10000) != 0) {
                std::memcpy(seen.data() + i * 0x10000, now, 0x10000);
                hashes[i] = HashBytes(now, 0x10000, i);
            }
            h = Mix(h, hashes[i]);
        }
        valid = true;
        return h;
    }
};

// All of MEM1 and MEM2, in 1 MiB chunks over a few threads.
uint64_t HashAll() {
    constexpr size_t kChunk = size_t{1} << 20, kThreads = 8;
    const size_t chunks1 = Memory::kMem1Size / kChunk, chunks = chunks1 + Memory::kMem2Size / kChunk;
    const uint8_t* mem1 = Host(0x80000000u, Memory::kMem1Size);
    const uint8_t* mem2 = Host(0x90000000u, Memory::kMem2Size);
    vector<uint64_t> hashes(chunks);
    vector<std::thread> threads;
    for (size_t t = 0; t < kThreads; ++t)
        threads.emplace_back([&, t] {
            for (size_t i = t; i < chunks; i += kThreads)
                hashes[i] = HashBytes(i < chunks1 ? mem1 + i * kChunk : mem2 + (i - chunks1) * kChunk, kChunk, i);
        });
    for (auto& th : threads) th.join();
    uint64_t h = 0x7654321;
    for (uint64_t v : hashes) h = Mix(h, v);
    return h;
}

uint64_t HashRegs(const CpuContext& c) {
    uint64_t h = 0xABCDEF;
    for (uint32_t r : c.gpr) h = Mix(h, r);
    h = Mix(h, PpcGetCr(&c));
    h = Mix(h, c.xer);
    h = Mix(h, c.ctr);
    h = Mix(h, c.lr);
    h = Mix(h, c.fpr[2].raw);
    return Mix(h, c.fpr[3].raw);
}

CpuContext MakeContext(const Entry& e) {
    CpuContext c{};
    std::memcpy(c.gpr, e.gpr, sizeof c.gpr);
    PpcSetCr(&c, e.cr);
    c.xer = e.xer;
    c.ctr = e.ctr;
    c.lr = e.lr;
    c.msr = e.msr;
    c.fpr[2].raw = e.f2;
    c.fpr[3].raw = e.f3;
    return c;
}
Entry EntryOf(const CpuContext& c) {
    Entry e{};
    std::memcpy(e.gpr, c.gpr, sizeof e.gpr);
    e.cr = PpcGetCr(&c);
    e.xer = c.xer;
    e.ctr = c.ctr;
    e.lr = c.lr;
    e.msr = c.msr;
    e.f2 = c.fpr[2].raw;
    e.f3 = c.fpr[3].raw;
    return e;
}

bool g_verbose = false;

RunOut RunScenario(const Scenario& sc, EngineKind engine, int stopAtPass = -1, Detail* detail = nullptr) {
    RunOut out;
    ResetUniverse();
    for (const auto& [addr, bytes] : sc.image) std::memcpy(Host(addr, static_cast<uint32_t>(bytes.size())), bytes.data(), bytes.size());
    BrawlDirtyMemory::MarksCleared();
    if (BrawlGecko_PlanControl != nullptr) {
        BrawlGecko_PlanControl(engine == EngineKind::Plan ? 1 : engine == EngineKind::PlanNoMarks ? 2 : 0);
    }
    Watch watch;
    if (!sc.tracked.empty()) MakeWatch(watch, sc.tracked);
    int passNo = 0;
    vector<uint8_t> before;
    UniverseHash universeHash;
    for (const Step& st : sc.steps) {
        switch (st.kind) {
        case Step::Write:
            if (st.size == 1) Memory::Write8(st.addr, static_cast<uint8_t>(st.value));
            else if (st.size == 2) Memory::Write16(st.addr, static_cast<uint16_t>(st.value));
            else Memory::Write32(st.addr, st.value);
            break;
        case Step::WriteBytes:
            for (size_t i = 0; i < st.bytes.size(); ++i) Memory::Write8(st.addr + static_cast<uint32_t>(i), st.bytes[i]);
            break;
        case Step::Save:
            if (watch.ww) WatchSave(watch, st.slot);
            break;
        case Step::Restore:
            if (watch.ww) WatchRestore(watch, st.slot);
            break;
        case Step::Unmarked:
            std::memcpy(Host(st.addr, static_cast<uint32_t>(st.bytes.size())), st.bytes.data(), st.bytes.size());
            BrawlDirtyMemory::MarksCleared();
            break;
        case Step::WatchOff:
            watch.ww.reset();
            watch.snaps.clear();
            break;
        case Step::WatchOn:
            MakeWatch(watch, st.pages);
            break;
        case Step::Pass: {
            ++passNo;
            if (sc.consumeEachPass && watch.ww) WatchSave(watch, 100 + passNo % 4);
#if MKW_LEAN_MEMORY
            const bool check = watch.ww != nullptr;
            if (check) ReadPages(watch.pages, before);
#endif
            CpuContext ctx = MakeContext(st.entry);
            {
                CpuContextScope scope(&ctx);
                if (engine == EngineKind::Ref) {
                    PpcInterpreter::SetEngineOverride(PpcInterpreter::Engine::Classic);
                    PpcInterpreter::Run(&ctx, kEntry);
                    PpcInterpreter::ClearEngineOverride();
                } else {
                    BrawlGecko_CodeHandler(&ctx);
                }
            }
            ctx.pc = 0;
#if MKW_LEAN_MEMORY
            if (check) {
                // every 64-byte line of a tracked page that the pass changed carries a mark
                for (size_t i = 0; i < watch.pages.size(); ++i) {
                    const uint8_t* now = Host(watch.pages[i], 0x10000);
                    const uint8_t* was = before.data() + i * 0x10000;
                    if (std::memcmp(now, was, 0x10000) == 0) continue;
                    for (uint32_t line = 0; line < 0x10000; line += 64)
                        if (std::memcmp(now + line, was + line, 64) != 0 && MarkOf(watch.pages[i] + line) == 0) {
                            char msg[160];
                            std::snprintf(msg, sizeof msg, "pass %d: line %08X of a tracked page changed without a dirty mark", passNo,
                                          watch.pages[i] + line);
                            if (out.errors.size() < 8) out.errors.push_back(msg);
                        }
                }
            }
#endif
            Digest d;
            d.regs = HashRegs(ctx);
            d.mem = universeHash.Digest();
            if (passNo % 16 == 0) d.full = HashAll();
            out.digests.push_back(d);
            if (passNo == stopAtPass && detail != nullptr) {
                detail->regs = EntryOf(ctx);
                detail->universe.clear();
                for (uint32_t page : g_pages) {
                    const uint8_t* p = Host(page, 0x10000);
                    detail->universe.insert(detail->universe.end(), p, p + 0x10000);
                }
                detail->mem1.assign(Host(0x80000000u, Memory::kMem1Size), Host(0x80000000u, Memory::kMem1Size) + Memory::kMem1Size);
                detail->mem2.assign(Host(0x90000000u, Memory::kMem2Size), Host(0x90000000u, Memory::kMem2Size) + Memory::kMem2Size);
                watch.ww.reset();
                return out;
            }
            break;
        }
        }
    }
    if (out.digests.empty() || out.digests.back().full == 0) {
        Digest d = out.digests.empty() ? Digest{} : out.digests.back();
        d.full = HashAll();
        if (out.digests.empty()) out.digests.push_back(d);
        else out.digests.back() = d;
    }
    watch.ww.reset();
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Reporting

const char* RegName(int i) {
    static char names[48][8];
    static bool init = false;
    if (!init) {
        for (int k = 0; k < 32; ++k) std::snprintf(names[k], 8, "r%d", k);
        std::snprintf(names[32], 8, "cr");
        std::snprintf(names[33], 8, "xer");
        std::snprintf(names[34], 8, "ctr");
        std::snprintf(names[35], 8, "lr");
        std::snprintf(names[36], 8, "f2");
        std::snprintf(names[37], 8, "f3");
        init = true;
    }
    return names[i];
}

void Explain(const Scenario& sc, EngineKind a, EngineKind b, int pass) {
    Detail da, db;
    RunScenario(sc, a, pass, &da);
    RunScenario(sc, b, pass, &db);
    uint64_t va[38], vb[38];
    const auto flat = [](const Entry& e, uint64_t* v) {
        for (int i = 0; i < 32; ++i) v[i] = e.gpr[i];
        v[32] = e.cr;
        v[33] = e.xer;
        v[34] = e.ctr;
        v[35] = e.lr;
        v[36] = e.f2;
        v[37] = e.f3;
    };
    flat(da.regs, va);
    flat(db.regs, vb);
    bool found = false;
    for (int i = 0; i < 38 && !found; ++i)
        if (va[i] != vb[i]) {
            std::fprintf(stderr, "    first differing register %s: %s %llX, %s %llX\n", RegName(i), EngineName(a), (unsigned long long)va[i],
                         EngineName(b), (unsigned long long)vb[i]);
            found = true;
        }
    for (size_t i = 0; i < da.universe.size() && !found; ++i)
        if (da.universe[i] != db.universe[i]) {
            const uint32_t addr = g_pages[i / 0x10000] + static_cast<uint32_t>(i % 0x10000);
            std::fprintf(stderr, "    first differing byte at %08X: %s %02X, %s %02X\n", addr, EngineName(a), da.universe[i], EngineName(b),
                         db.universe[i]);
            found = true;
        }
    for (size_t i = 0; i < da.mem1.size() && !found; ++i)
        if (da.mem1[i] != db.mem1[i]) {
            std::fprintf(stderr, "    first differing byte outside the universe at %08X: %s %02X, %s %02X\n", 0x80000000u + (uint32_t)i,
                         EngineName(a), da.mem1[i], EngineName(b), db.mem1[i]);
            found = true;
        }
    for (size_t i = 0; i < da.mem2.size() && !found; ++i)
        if (da.mem2[i] != db.mem2[i]) {
            std::fprintf(stderr, "    first differing byte outside the universe at %08X: %s %02X, %s %02X\n", 0x90000000u + (uint32_t)i,
                         EngineName(a), da.mem2[i], EngineName(b), db.mem2[i]);
            found = true;
        }
    if (!found) std::fprintf(stderr, "    (the pass's end state is the same; only a later state or the full-memory hash differs)\n");
}

// ---------------------------------------------------------------------------------------------------------------------
// The suite

struct Totals {
    Counters c[2];  // plan modes 1 and 2
    uint64_t scenarios = 0, passes = 0, trustedInTracked = 0, replayedInExpecting = 0, trustedInExpecting = 0;
    int failures = 0;
    std::vector<EngineKind> engines;
};
Totals g_totals;
bool g_inject = false, g_injected = false;

void RunSuiteScenario(const Scenario& sc) {
    ++g_totals.scenarios;
    vector<RunOut> runs;
    Counters before = ReadCounters();
    Counters delta[4];
    for (size_t e = 0; e < g_totals.engines.size(); ++e) {
        const Counters b = ReadCounters();
        runs.push_back(RunScenario(sc, g_totals.engines[e]));
        const Counters a = ReadCounters();
        for (size_t k = 0; k < kCounterCount; ++k) delta[e].v[k] = a.v[k] - b.v[k];
    }
    (void)before;
    if (g_inject && !g_injected && runs.size() > 1) {
        runs[1].digests[runs[1].digests.size() / 2].regs ^= 1;
        g_injected = true;
    }
    g_totals.passes += runs[0].digests.size();
    bool bad = false;
    for (size_t e = 0; e < runs.size(); ++e) {
        for (const auto& err : runs[e].errors) {
            std::fprintf(stderr, "FAIL scenario '%s' seed %llu engine %s: %s\n", sc.name.c_str(), (unsigned long long)sc.seed,
                         EngineName(g_totals.engines[e]), err.c_str());
            bad = true;
        }
        if (e == 0) continue;
        if (runs[e].digests.size() != runs[0].digests.size()) {
            std::fprintf(stderr, "FAIL scenario '%s' seed %llu engine %s: %zu digests vs %zu\n", sc.name.c_str(), (unsigned long long)sc.seed,
                         EngineName(g_totals.engines[e]), runs[e].digests.size(), runs[0].digests.size());
            bad = true;
            continue;
        }
        for (size_t p = 0; p < runs[0].digests.size(); ++p) {
            if (runs[e].digests[p] == runs[0].digests[p]) continue;
            std::fprintf(stderr, "FAIL scenario '%s' seed %llu: engine %s differs from REF at pass %zu%s\n", sc.name.c_str(),
                         (unsigned long long)sc.seed, EngineName(g_totals.engines[e]), p + 1,
                         runs[e].digests[p].regs != runs[0].digests[p].regs ? " (registers)" : "");
            if (!g_injected || !g_inject) Explain(sc, EngineKind::Ref, g_totals.engines[e], static_cast<int>(p + 1));
            else std::fprintf(stderr, "    (corrupted on purpose by GECKO_TEST_INJECT)\n");
            bad = true;
            break;
        }
    }
    if (bad) {
        ++g_totals.failures;
        ZeroEverything();
    }
    for (size_t e = 0; e < g_totals.engines.size(); ++e) {
        if (g_totals.engines[e] == EngineKind::Plan || g_totals.engines[e] == EngineKind::PlanNoMarks) {
            const int mode = g_totals.engines[e] == EngineKind::Plan ? 0 : 1;
            for (size_t k = 0; k < kCounterCount; ++k) g_totals.c[mode].v[k] += delta[e].v[k];
            if (g_totals.engines[e] == EngineKind::Plan) {
                if (!sc.tracked.empty()) g_totals.trustedInTracked += delta[e].v[kTrusted];
                if (sc.expectReplay) g_totals.replayedInExpecting += delta[e].v[kReplayed];
                if (sc.expectTrust) {
                    g_totals.trustedInExpecting += delta[e].v[kTrusted];
                }
            }
        }
    }
    if (g_verbose) std::fprintf(stderr, "scenario '%s' seed %llu: %zu passes %s\n", sc.name.c_str(), (unsigned long long)sc.seed,
                                runs[0].digests.size(), bad ? "FAILED" : "ok");
}

// ---------------------------------------------------------------------------------------------------------------------
// Hand-written lists

using Builder = std::function<void(Gen&, Rng&)>;
struct Hand {
    const char* name;
    Builder build;
    int passes = 10;
};

constexpr uint32_t kX = 0x80410000u;  // compare operands
constexpr uint32_t kZ = 0x80411000u;  // targets written under conditions

void HWrites(Gen& g, Rng&) {
    g.W8(0x80400000, 0x11);
    g.W8(0x80400010, 0x22, 3);
    g.W8(0x80400021, 0x33, 40);
    g.W16(0x80400100, 0x1234);
    g.W16(0x80400111, 0xBEEF, 5);
    g.W16(0x80400200, 0xCAFE, 33);
    g.W32(0x80400300, 0xDEADBEEF);
    g.W32(0x80400307, 0x01020304);  // an unaligned address aligns down
    g.W32(0x80400300, 0x0BADF00D);  // the same word again
    // the 0x01000000 address bit under the 0x80 base
    g.W32(0x81100040, 0xA5A5A5A5);
    g.W8(0x81100100, 0x7F, 64);
    g.W16(0x81100200, 0x1111, 20);
    // po-relative codes (0x10 types) into MEM2, then its 0x01000000 bit
    g.SetPo(0x90000000);
    g.W32(0x90100000, 0x12345678, true);
    g.W8(0x90100010, 0x44, 5, true);
    g.W16(0x90100020, 0x5555, 7, true);
    g.W32(0x91100000, 0xCAFEF00D, true);
    // the uncached views
    g.SetPo(0xC0000000);
    g.W32(0xC0410000, 0x0A0B0C0D, true);
    g.SetBa(0x90000000);
    g.W32(0x90400000, 0x01234567);
    g.W8(0x91100010, 0x99, 12);
    g.SetBa(0xC0000000);
    g.W16(0xC0400110, 0x7777, 3);
    g.SetBa(0xD0000000);
    g.W32(0xD0100100, 0x13572468);
    g.SetBa(0x80000000);
    g.W32(0x80400300, 0x77777777);  // over the first one
}

void HStrings(Gen& g, Rng& r) {
    static const uint32_t lens[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 33, 100};
    uint32_t at = 0x80400000;
    for (uint32_t len : lens) {
        vector<uint8_t> b(len);
        for (auto& x : b) x = static_cast<uint8_t>(r());
        g.Str(at + (len % 3), b);
        at += 0x100;
    }
    // a string over a later code's target, and one over an earlier code's target
    g.Str(0x80401000, vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
    g.W32(0x80401008, 0xAABBCCDD);
    g.Str(0x80401004, vector<uint8_t>{0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8});
    g.W32(0x80402000, 0x01010101);
    g.Str(0x80402002, vector<uint8_t>{9, 9, 9, 9, 9, 9});
    // MEM2 and the uncached view
    g.SetPo(0x90000000);
    g.Str(0x90100300, vector<uint8_t>{'a', 'b', 'c', 'd', 'e'}, true);
    g.SetBa(0xC0000000);
    g.Str(0xC0400400, vector<uint8_t>{'x', 'y', 'z'});
    // the same bytes already in memory the second time round
    g.SetBa(0x80000000);
    g.Str(0x80403000, vector<uint8_t>(40, 0x5A));
    g.Str(0x80403000, vector<uint8_t>(40, 0x5A));
}

void HSerial(Gen& g, Rng&) {
    g.Serial(0x80400000, 0x11, 0, 8, 1, 3);
    g.Serial(0x80400100, 0x1234, 1, 6, 2, 0x111);
    g.Serial(0x80400200, 0xA0000000, 2, 5, 4, 0x01000001);
    g.Serial(0x80400300, 0x55, 0, 1, 1, 0);
    g.Serial(0x80400400, 0x77, 0, 4, 3, 0);   // overlapping steps
    g.Serial(0x80400400, 0x88, 1, 4, 1, 1);   // over the previous
    g.Serial(0x80400500, 0xFFFFFFFF, 2, 12, 2, 1);
    g.SetPo(0x91100000);
    g.Serial(0x91100000 + 0x40, 0x31, 1, 9, 6, 7, true);
}

void HIfs(Gen& g, Rng&) {
    Put32(g.sc, kX, 5);
    Put32(g.sc, kX + 4, 0x00120034);
    Put32(g.sc, kX + 8, 0xFFFFFFF0u);
    Put32(g.sc, kX + 12, 0);
    uint32_t k = 0;
    for (uint32_t cmp = 0; cmp < 4; ++cmp)
        for (uint32_t v : {4u, 5u, 6u}) {
            g.If(cmp, kX, v);
            g.W32(kZ + 4 * k, 0x1000 + k);
            g.Endif();
            ++k;
        }
    for (uint32_t cmp = 0; cmp < 4; ++cmp)  // 16-bit, with an ignore mask
        for (uint32_t v : {0x33u, 0x34u, 0x35u}) {
            g.If(cmp, kX + 6, v, true, 0xFF00);
            g.W32(kZ + 4 * k, 0x2000 + k);
            g.Endif();
            ++k;
        }
    g.If(1, kX + 4, 0x0012, true, 0);  // the other halfword
    g.W32(kZ + 4 * k++, 0x3000);
    g.Endif();
    g.If(2, kX + 8, 0x7FFFFFFF);  // unsigned: greater
    g.W32(kZ + 4 * k++, 0x3001);
    g.Endif();
    // nested, else, endif bit, two levels popped
    g.If(0, kX, 5);
    g.If(2, kX + 4, 0x00100000);
    g.W32(kZ + 0x100, 1);
    g.Else();
    g.W32(kZ + 0x104, 2);
    g.Endif(1);
    g.W32(kZ + 0x108, 3);
    g.Else();
    g.W32(kZ + 0x10C, 4);
    g.Endif(1);
    g.W32(kZ + 0x110, 5);
    g.If(0, kX, 5);
    g.W32(kZ + 0x120, 6);
    g.If(1, kX, 5, false, 0, true);  // endif of the first, then a new if
    g.W32(kZ + 0x124, 7);
    g.Endif(1);
    g.If(0, kX, 5);
    g.If(0, kX + 12, 0);
    g.If(0, kX + 12, 0);
    g.W32(kZ + 0x130, 8);
    g.E2(3);
    g.W32(kZ + 0x134, 9);
    // po-based compares
    g.SetPo(0x90000000);
    g.If(0, 0x90100000, 0, false, 0, false, true);
    g.W32(kZ + 0x140, 10);
    g.Endif();
}

void HFourX(Gen& g, Rng&) {
    // set ba / po directly, from a pointer table, from a register, relative to the next code
    g.SetBa(0x90000000);
    g.W32(0x90400000, 0x11111111);
    g.C(0x40000000, kPtrTable + 4);  // ba = [table + 4]: 0xC0000000
    g.ba = 0xC0000000;
    g.W32(0xC0400004, 0x22222222);
    g.C(0x48000000, kPtrTable + 8);  // po = [table + 8]: 0x90000000
    g.po = 0x90000000;
    g.W32(0x90400008, 0x33333333, true);
    g.C(0x4A001000u | 0, 0x100);     // po = 0x100 + gr0 (0x80400000)
    g.po = 0x80400100;
    g.W32(0x80400104, 0x44444444, true);
    g.C(0x42010000u, 0x400000);      // ba = r12 + 0x400000, r12 = ba's own base 0xC0000000
    g.ba = 0xC0400000;
    g.W32(0xC0400010, 0x55555555);
    g.C(0x4A100000u, 0x400000u);     // po = 0x400000 + po (po is 0x80400100 so far)
    g.po = 0x80800100u;
    g.C(0x4A000000u, 0x80400000u);   // back to something sane
    g.po = 0x80400000u;
    g.C(0x44000000u, 0x400200);      // store ba at r12 + 0x400200
    g.Target(0xC0400200, 4);
    g.C(0x4C000000u, 0x400204);      // store po at r12 + 0x400204
    g.Target(0x80400204, 4);
    g.GrSet(4, 0x400300);
    g.C(0x44001000u | 4, 0x40);      // store ba at r12 + 0x40 + gr4
    g.Target(0x80400340, 4);
    g.Target(0xC0400340, 4);
    g.C(0x46000000u | 0x10, 0);      // ba = next code + 16
    g.ba = 0x80000000u;
    g.W32(0x80400020, 0x66666666);
    g.SetBa(0x80000000);
    g.SetPo(0x80000000);
}

void HGoto(Gen& g, Rng&) {
    Put32(g.sc, kX, 5);
    g.W32(kZ + 0, 1);
    g.Goto(2);
    g.W32(kZ + 4, 2);
    g.W32(kZ + 8, 3);
    g.W32(kZ + 12, 4);
    g.Goto(1, 1);  // only while skipped: not taken here
    g.W32(kZ + 16, 5);
    // a gosub: gosub S; A; goto E; S: B; return; E:
    g.Gosub(2, 3);
    g.W32(kZ + 0x20, 6);
    g.Goto(2);
    g.W32(kZ + 0x24, 7);
    g.Return(3);
    g.W32(kZ + 0x28, 8);
    // all of it under a false condition: gosub / return / goto skipped or not by their condition fields
    g.If(0, kX, 6);
    g.Goto(3, 0);
    g.W32(kZ + 0x30, 9);
    g.Goto(1, 2);
    g.W32(kZ + 0x34, 10);
    g.W32(kZ + 0x38, 11);
    g.Gosub(2, 4, 0);
    g.W32(kZ + 0x3C, 12);
    g.Return(4, 0);
    g.Goto(1, 1);
    g.W32(kZ + 0x40, 13);
    g.W32(kZ + 0x44, 14);
    g.Endif();
    g.W32(kZ + 0x48, 15);
    // a gosub inside a true condition with an if in the subroutine
    g.If(0, kX, 5);
    g.Gosub(5, 5);
    g.W32(kZ + 0x50, 16);
    g.Goto(4);
    g.If(1, kX, 5);
    g.W32(kZ + 0x54, 17);
    g.Endif();
    g.Return(5);
    g.Endif();
}

void HRepeat(Gen& g, Rng&) {
    Put32(g.sc, kX, 5);
    g.SetRepeat(0, 3);
    g.W32(kZ + 0, 0x1111);
    g.W8(kZ + 8, 0x22, 3);
    g.ExecRepeat(0);
    g.W32(kZ + 0x10, 0x3333);
    // nested with a second block
    g.SetRepeat(1, 2);
    g.SetRepeat(2, 1);
    g.W16(kZ + 0x20, 0x4444, 2);
    g.ExecRepeat(2);
    g.W16(kZ + 0x28, 0x5555);
    g.ExecRepeat(1);
    // an exec without a set (the count is 0, or what an earlier run left)
    g.ExecRepeat(7);
    // under conditions
    g.If(0, kX, 6);
    g.SetRepeat(3, 5);
    g.W32(kZ + 0x30, 0x6666);
    g.ExecRepeat(3);
    g.Endif();
    g.If(0, kX, 5);
    g.SetRepeat(4, 2);
    g.If(1, kX, 5);
    g.W32(kZ + 0x34, 0x7777);
    g.Endif();
    g.W32(kZ + 0x38, 0x8888);
    g.ExecRepeat(4);
    g.Endif();
}

void HRegs(Gen& g, Rng&) {
    // 80 set / add / with r12; 82 loads; 84 stores
    Put32(g.sc, kX, 0x11223344);
    Put32(g.sc, kX + 4, 0x55667788);
    g.GrSet(4, 0x1234);
    g.GrSet(5, 0x100, true);              // + r12
    g.GrSet(5, 0x10, false, true);        // + itself
    g.GrSet(6, 0xFFFF0000u, true, true);
    g.GrLoad(7, 2, kX);
    g.GrLoad(8, 1, kX + 5);
    g.GrLoad(9, 0, kX + 3);
    g.GrLoad(10, 2, 0x400000 + 0x10000, true);  // + r12
    for (uint32_t i = 4; i < 11; ++i) g.GrStore(i, 2, kZ + 0x100 + 4 * i);
    g.GrStore(4, 2, kZ + 0x200, 4);
    g.GrStore(4, 1, kZ + 0x220, 5);
    g.GrStore(5, 0, kZ + 0x240, 7);
    g.GrStore(5, 2, 0x10000 + 0x400000 + 0x200, 2, true);
    g.Target(0x80000000u + 0x410200, 8);
}

void HOps(Gen& g, Rng&) {
    Put32(g.sc, kX, 0x3FC00000);  // 1.5f
    Put32(g.sc, kX + 4, 0x40000000);
    for (uint32_t op = 0; op <= 11; ++op) {
        g.GrLoad(4, 2, kX);
        g.GrLoad(5, 2, kX + 4);
        g.GrOpImm(op, 4, 0x00000003);
        g.GrStore(4, 2, kZ + 0x10 * op);
        g.GrLoad(4, 2, kX);
        g.GrOpReg(op, 4, 5);
        g.GrStore(4, 2, kZ + 0x10 * op + 4);
        g.GrSet(6, 0xFFFFFF00u);
        g.GrOpImm(op, 6, 0x24);          // shifts and rotates with amounts above and below 32
        g.GrStore(6, 2, kZ + 0x10 * op + 8);
        g.GrSet(6, 0x80000001u);
        g.GrOpImm(op, 6, 0xFFFFFFF9u);   // sraw with a negative value (carry)
        g.GrStore(6, 2, kZ + 0x10 * op + 12);
    }
    // flags: operate on memory a register points to; a source read through a register
    g.GrOpImm(0, 0, 0x10, 1);            // gr0 = pointer 0x80400000: the word there gets + 0x10
    g.GrOpReg(2, 4, 1, 2);               // source is the word at gr1 (0x90100000)
    g.GrOpImm(0, 4, 0x400100, 2);        // source is the word at r12 + imm
    g.GrOpReg(3, 1, 0, 3);
    g.GrStore(4, 2, kZ + 0x300);
    g.Target(0x80400000, 4);
    g.Target(0x90100000, 4);
}

void HCopy(Gen& g, Rng& r) {
    vector<uint8_t> pat(0x200);
    for (auto& b : pat) b = static_cast<uint8_t>(r());
    Put(g.sc, 0x80400000, pat);
    Put(g.sc, 0x90100000, pat);
    g.Copy(true, 0, 1, 16, 0x40);
    g.Copy(false, 1, 0, 33, 0x20);
    g.Copy(true, 15, 0, 64, 0x400200);
    g.Copy(false, 2, 15, 8, 0x400000);
    g.Copy(true, 0, 0, 32, 4);   // overlapping, forward
    g.Copy(false, 0, 0, 32, 4);  // overlapping, the other way
    g.Copy(true, 0, 0, 0, 0);    // nothing
    g.Copy(true, 0, 0, 1, 0);    // onto itself
    g.Copy(true, 1, 0, 0x180, 0);
    g.Copy(true, 1, 0, 0x180, 0);  // already copied
    g.Copy(true, 3, 2, 12, 0);
    g.Target(0x80400000, 0x300);
    g.Target(0x90100000, 0x300);
    g.Target(0x80400200, 0x100);
}

void HC0Dump(Gen& g, Rng&) {
    Put32(g.sc, kX, 5);
    g.W32(kZ, 1);
    g.C0(g.Dump());
    g.GrSet(4, 0x77);
    g.C0(g.Dump());
    g.If(0, kX, 6);   // false: skipped
    g.C0(g.Dump());
    g.W32(kZ + 4, 2);
    g.Endif();
    g.If(0, kX, 5);
    g.C0(g.Dump());
    g.If(1, kX, 5);
    g.C0(g.Dump());
    g.Endif();
    g.Endif();
    g.SetBa(0x90000000);
    g.SetPo(0xC0000000);
    g.C0(g.Dump());
    g.W8(0x90400000, 7, 2);
    g.C0(g.Dump());
    g.SetBa(0x80000000);
}

void HC0Writes(Gen& g, Rng& r) {
    Put32(g.sc, kX, 5);
    const uint32_t firstCode = g.Here();
    g.W32(kZ + 0x80, 1);  // a code whose value a later C0 rewrites
    g.sc.muts.push_back({firstCode + 4, 4, {}, true});
    g.C0(Gen::StoreWord(kZ, 0x11111111));
    g.W32(kZ + 4, 0x22222222);
    g.C0(Gen::StoreWord(kZ + 4, 0x33333333));  // over the code's own target
    g.C0(Gen::StoreWord(kZ + 9, 0x4455, 2));
    g.C0(Gen::StoreWord(kZ + 0x20, 0x66, 1));
    g.C0(Gen::StoreGr(4, 0xABCD1234));
    g.GrStore(4, 2, kZ + 0x30);
    g.C0(Gen::StoreGr(0, 0x80400040));  // a pointer register
    g.Copy(true, 0, 0, 4, 0x100);
    g.Target(0x80400000, 0x200);
    // list words: a later code's value, the first code's value, a string payload before this point
    const uint32_t laterAt = g.Here() + 0x20 + 4;  // the lo word of the code after this body's
    g.sc.muts.push_back({laterAt, 4, {}, true});
    g.C0(Gen::StoreWord(laterAt, 0xDEAD0001));
    g.W32(kZ + 0x40, 1);
    g.W32(kZ + 0x44, 2);
    g.C0(Gen::StoreWord(firstCode + 4, 0x0BAD0BAD));
    g.Str(kZ + 0x60, vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    g.C0(Gen::StoreWord(g.Here() - 12, 0x41424344));  // inside that string's payload
    g.W32(kZ + 0x70, static_cast<uint32_t>(r()));
}

void HC0Regs(Gen& g, Rng&) {
    Put32(g.sc, kX, 5);
    Gen::RegSet rs;
    rs.r6 = true; rs.v6 = 0x90000000;
    g.C0(Gen::SetRegs(rs));
    g.ba = 0x90000000;
    g.W32(0x90400000, 0x12121212);
    g.W32(0x90400004, 0x34343434);
    rs = {};
    rs.r16 = true; rs.v16 = 0xC0000000;
    g.C0(Gen::SetRegs(rs));
    g.po = 0xC0000000;
    g.W32(0xC0400100, 0x56565656, true);
    rs = {};
    rs.r6 = true; rs.v6 = 0x80000000;
    rs.r16 = true; rs.v16 = 0x90000000;
    g.C0(Gen::SetRegs(rs));
    g.ba = 0x80000000;
    g.po = 0x90000000;
    g.W32(0x80400200, 0x78787878);
    g.W32(0x90400200, 0x9A9A9A9A, true);
    // the execution status: skip a stretch, then a terminator
    rs = {};
    rs.r8 = true; rs.v8 = 1;
    g.C0(Gen::SetRegs(rs));
    g.W32(0x80400300, 0x01010101);
    g.C0(g.Dump());
    g.W32(0x80400304, 0x02020202);
    g.E0();
    g.W32(0x80400308, 0x03030303);
    // r7: another table of Gecko registers for a while
    g.GrSet(4, 0xF00D);
    rs = {};
    rs.r7 = true; rs.v7 = kAlt;
    g.C0(Gen::SetRegs(rs));
    g.GrSet(4, 0xBEEF);
    g.GrStore(4, 2, 0x80400400);
    g.C0(g.Dump());
    rs = {};
    rs.r7 = true; rs.v7 = kGr;
    g.C0(Gen::SetRegs(rs));
    g.GrStore(4, 2, 0x80400404);
    g.Target(0x80400400, 8);
}

void HScramble(Gen& g, Rng& r) {
    Put32(g.sc, kX, 5);
    g.W32(kZ, 1);
    g.C0(Gen::Scramble(r));
    g.W32(kZ + 4, 2);
    g.C0(g.Dump());
    g.If(0, kX, 5);
    g.C0(Gen::Scramble(r));
    g.GrSet(5, 3);
    g.C0(g.Dump());
    g.Endif();
    g.C0(Gen::Scramble(r));
    g.Str(kZ + 0x10, vector<uint8_t>{1, 2, 3});
    g.C0(Gen::Scramble(r));
    g.Serial(kZ + 0x30, 1, 0, 3, 1, 1);
    g.C0(g.Dump());
    g.W32(kZ + 8, 3);
    g.C0(Gen::Scramble(r));
    g.W16(kZ + 0x40, 4, 3);
    g.C0(g.Dump());
}

void HHooks(Gen& g, Rng&) {
    using namespace Ppc;
    const uint32_t t1 = 0x80402000, t2 = 0x80402100;
    Put32(g.sc, t1 + 4, Blr);
    Put32(g.sc, t2 + 4, Blr);
    g.C2(t1, {Li(0, 0x55), Stw(0, -0x60, 1), Addi(3, 3, 7), Blr});
    g.sc.targets.pop_back();
    g.C0(Gen::CallThrough(t1));
    g.C0(g.Dump());
    g.C4(t2, {Li(0, 0x66), Stw(0, -0x64, 1), Nop, Nop, Nop, Nop});
    g.sc.targets.pop_back();
    g.C0(Gen::CallThrough(t2));
    g.C0(g.Dump());
    // hooks over data targets, one skipped
    Put32(g.sc, kX, 5);
    g.C2(0x80402200, {1, 2, 3, 4});
    g.If(0, kX, 6);
    g.C2(0x80402300, {5, 6, 7, 8});
    g.C4(0x80402400, {9, 10});
    g.Endif();
    g.C2(0x80402200, {1, 2, 3, 4, 5, 6});  // the same target, a longer body
    g.SetPo(0x90000000);
    g.C2(0x90100400, {11, 12}, true);
    g.SetPo(0x80000000);
}

void HC6(Gen& g, Rng&) {
    Put32(g.sc, kX, 5);
    g.C6(0x80402000, 0x80002000);
    g.C6(0x80402004, 0x80402000, true);
    g.C6(0x80402008, 0x80402008);          // a branch to itself (never run)
    g.If(0, kX, 6);
    g.C6(0x8040200C, 0x80003000);
    g.Endif();
    g.C6(0x80402010, 0x80400000, false, false);
    g.SetPo(0x90000000);
    g.C6(0x90100000, 0x90100010, true, true);
    g.SetPo(0x80000000);
}

void HCc(Gen& g, Rng&) {
    Put32(g.sc, kX, 5);
    for (uint32_t lo : {0u, 1u, 2u, 3u}) {
        g.W32(kZ + 0x10 * lo, 0x100 + lo);
        g.CC(lo);
        g.W32(kZ + 0x10 * lo + 4, 0x200 + lo);
        g.CC(lo ^ 1);
        g.W32(kZ + 0x10 * lo + 8, 0x300 + lo);
        g.E0();
    }
    g.If(0, kX, 6);
    g.CC(2);
    g.W32(kZ + 0x100, 1);
    g.CC(1);
    g.Endif();
    g.W32(kZ + 0x104, 2);
    g.sc.muts.push_back({g.Here() - 8 + 4, 4, {}, true});
}

void HEnd(Gen& g, Rng&) {
    Put32(g.sc, kX, 5);
    g.SetBa(0x90000000);
    g.SetPo(0xC0000000);
    g.E0(0x8000, 0x9000);  // ba and po from the terminator
    g.ba = 0x80000000;
    g.po = 0x90000000;
    g.W32(0x80400000, 1);
    g.W32(0x90400000, 2, true);
    g.If(0, kX, 5);
    g.If(0, kX, 5);
    g.E2(1, false, 0xC000, 0x8000);
    g.ba = 0xC0000000;
    g.po = 0x80000000;
    g.W32(0xC0400004, 3);
    g.E2(1, true);          // else of the outer
    g.W32(0xC0400008, 4);
    g.E2(1);
    g.W32(0xC040000C, 5);
    g.E2(0x1F);             // more than the depth
    g.E0(0x8000, 0x8000);
    g.ba = g.po = 0x80000000;
    g.W32(0x80400010, 6);
    g.F0();
    g.W32(0x80400014, 7);   // after the end: never run
    g.W32(0x80400018, 8);
}

void HSkipped(Gen& g, Rng&) {
    using namespace Ppc;
    Put32(g.sc, kX, 5);
    for (uint32_t cond : {6u, 5u}) {
        g.If(0, kX, cond);
        g.C0(g.Dump());
        g.C0(Gen::StoreWord(kZ + 0x40 * cond, 0xAAAA0000 + cond));
        g.Str(kZ + 0x40 * cond + 8, vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        g.GrSet(4, 0x1000 + cond);
        g.GrStore(4, 2, kZ + 0x40 * cond + 0x20);
        g.GrOpImm(0, 4, 1);
        g.GrStore(4, 2, kZ + 0x40 * cond + 0x24);
        g.Copy(true, 0, 1, 8, 0x40);
        g.C2(kZ + 0x40 * cond + 0x30, {1, 2, 3, 4});
        g.Serial(kZ + 0x40 * cond + 0x38, 1, 0, 3, 1, 1);
        g.C6(kZ + 0x40 * cond + 0x3C, 0x80402000);
        g.SetBa(0x90000000);
        g.SetPo(0x90000000);
        g.W32(0x90400000 + cond, 7);
        g.SetRepeat(1, 2);
        g.ExecRepeat(1);
        g.Gosub(1, 2);
        g.CC(1);
        g.Endif();
        g.SetBa(0x80000000);
        g.SetPo(0x80000000);
        g.W32(kZ + 0x40 * cond + 0x3C + 4, 9);
    }
}

void HHeader(Gen& g, Rng&) {
    Put32(g.sc, kX, 5);
    // a code that rewrites a later and an earlier code's header
    const uint32_t first = g.Here();
    g.W32(kZ, 0x1111);
    g.W32(kZ + 4, 0x2222);
    const uint32_t body = g.Here();
    (void)body;
    const uint32_t later = g.Here() + 8 * 8;
    g.C0(Gen::StoreWord(later, 0x04000000u | 0x411040u));  // the later code's header: a write to another place
    g.C0(Gen::StoreWord(first, 0x04000000u | 0x411080u));  // the first code's header: for the next pass
    g.W32(kZ + 8, 0x3333);
    g.W32(kZ + 0xC, 0x4444);
    g.W32(kZ + 0x10, 0x5555);
    g.W32(kZ + 0x14, 0x6666);
    g.sc.muts.push_back({first, 4, {{0x04, 0x41, 0x10, 0x00}, {0x04, 0x41, 0x10, 0x40}, {0x04, 0x41, 0x10, 0x80}}, false});
    g.sc.muts.push_back({later, 4, {{0x04, 0x41, 0x10, 0x40}, {0x04, 0x41, 0x10, 0x44}, {0x04, 0x41, 0x10, 0x48}}, false});
    g.sc.muts.push_back({later + 4, 4, {}, true});
}

void HTwoStores(Gen& g, Rng&) {
    Put32(g.sc, kX, 5);
    g.W32(kZ, 0x11111111);
    g.W32(kZ, 0x22222222);
    g.W8(kZ + 8, 0x33, 4);
    g.W16(kZ + 9, 0x4444, 2);
    g.If(0, kX, 6);
    g.W32(kZ + 0x10, 0x55555555);
    g.Endif();
    g.W32(kZ + 0x10, 0x66666666);
    g.If(0, kX, 5);
    g.W32(kZ + 0x10, 0x77777777);
    g.Endif();
    g.Str(kZ + 0x20, vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8});
    g.W32(kZ + 0x24, 0xAAAAAAAA);
    g.W32(kZ + 0x20, 0xBBBBBBBB);
    g.C2(kZ + 0x30, {1, 2});
    g.W32(kZ + 0x30, 0xCCCCCCCC);
    g.Serial(kZ + 0x40, 1, 2, 4, 4, 1);
    g.W32(kZ + 0x44, 0xDDDDDDDD);
}

// Types the native body hands to the interpreter, if the binary handles the encodings.
void HBail(Gen& g, Rng&) {
    Put32(g.sc, kX, 5);
    g.W32(kZ, 1);
    g.C(0xA0000000u | 0x410000, 0x00000005);
    g.W32(kZ + 4, 2);
    g.C(0xCE000000u | 0x410004, 0x00000001);
    g.W32(kZ + 8, 3);
}
// F2 and F4: a C2/C4 hook that is only installed when the checksum (xor of halfwords) of the words at the target matches.
void HBailF(Gen& g, Rng&) {
    const uint32_t t = 0x80402000, u = 0x80402100, v = 0x80402200;
    Put32(g.sc, t + 4, 0x12345678);
    Put32(g.sc, u + 4, 0x12345678);
    Put32(g.sc, v + 4, 0xCAFE0001);
    Put32(g.sc, kX, 5);
    g.W32(kZ, 1);
    g.C(0xF2000000u | (t - 0x80000000u), (2u << 24) | (0x444Cu << 8) | 1u);  // matches: the hook is installed
    g.w.push_back(0x11111111u);
    g.w.push_back(0x22222222u);
    g.W32(kZ + 4, 2);
    g.C(0xF2000000u | (u - 0x80000000u), (2u << 24) | (0x1111u << 8) | 1u);  // does not match: skipped
    g.w.push_back(0x33333333u);
    g.w.push_back(0x44444444u);
    g.C(0xF4000000u | (v - 0x80000000u), (2u << 24) | (0xCAFFu << 8) | 2u);  // F4: from po
    for (uint32_t k = 0; k < 4; ++k) g.w.push_back(0x55550000u + k);
    g.If(0, kX, 6);
    g.C(0xF2000000u | (t - 0x80000000u), (2u << 24) | (0x444Cu << 8) | 1u);  // skipped under a false condition
    g.w.push_back(0x66666666u);
    g.w.push_back(0x77777777u);
    g.Endif();
    g.W32(kZ + 8, 3);
}

// F6: search memory for the code's words; the list's own header is rewritten with the answer.
void HBailF6(Gen& g, Rng&) {
    Put32(g.sc, 0x80400104, 0xA1B2C3D4u);
    Put32(g.sc, 0x80400108, 0x55667788u);
    g.W32(kZ, 1);
    g.C(0xF6000001u, 0x80408041u);  // from 0x80400000 (the low halfword is the end: 0x80410000)
    g.w.push_back(0xA1B2C3D4u);
    g.w.push_back(0x55667788u);
    g.W32(kZ + 4, 2);
    g.W32(kZ + 8, 3);
}
void HBailF6Miss(Gen& g, Rng&) {
    g.W32(kZ, 1);
    g.C(0xF6000001u, 0x80408041u);
    g.w.push_back(0x0BADBEEFu);
    g.w.push_back(0x0BADF00Du);
    g.W32(kZ + 4, 2);
    g.W32(kZ + 8, 3);
}

void HBailC8(Gen& g, Rng&) {
    Put32(g.sc, kX, 5);
    g.W32(kZ, 1);
    g.C(0xC8000000u | 0x410000, 0x00000005);
    g.W32(kZ + 4, 2);
    g.C(0xCA000000u | 0x410000, 0x00000005);
    g.W32(kZ + 8, 3);
}

void HListSwitch(Gen& g, Rng& r) {
    (void)r;
    // list B is a different list; the pointer at 0x80001804 is rewritten by the game between passes
    Gen b;
    b.base = kListB;
    b.W32(kZ + 0x200, 0xB0B0B0B0);
    b.W32(0x80400000, 0xB1B1B1B1);
    b.Str(kZ + 0x210, vector<uint8_t>{9, 8, 7, 6, 5, 4, 3, 2, 1});
    b.C2(kZ + 0x230, {1, 2, 3, 4});
    Scenario sb = b.Finish("b");
    g.sc.image.push_back(sb.image.back());
    for (auto& t : sb.targets) g.sc.targets.push_back(t);
    for (auto& m : sb.muts) g.sc.muts.push_back(m);
    g.W32(kZ, 0xA0A0A0A0);
    g.W32(0x80400000, 0xA1A1A1A1);
    g.Str(kZ + 0x10, vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    g.C2(kZ + 0x30, {5, 6, 7, 8});
    const auto be = [](uint32_t v) { return vector<uint8_t>{uint8_t(v >> 24), uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)}; };
    g.sc.muts.push_back({kListPtr, 4, {be(kListA), be(kListB), be(kListA | 0x40000000u), be(kListB | 0x40000000u), be(0x80600000u)}, false});
}

void HPure(Gen& g, Rng& r) {
    // a long run of unconditional writes, strings and hooks, with an impure code every so often
    Put32(g.sc, kX, 5);
    uint32_t at = 0x80400000;
    for (int i = 0; i < 400; ++i) {
        switch (i % 5) {
        case 0: g.W32(at, 0x1000u + i); break;
        case 1: g.W16(at + 2, 0x2000u + i, 3); break;
        case 2: g.Str(at + 5, vector<uint8_t>(1 + (i % 11), static_cast<uint8_t>(i))); break;
        case 3: g.C2(at + 0x20, {1u + i, 2u}); break;
        default: g.W8(at + 0x30, static_cast<uint8_t>(i), 4); break;
        }
        at += 0x40;
        if (at >= 0x80430000) at = 0x80400000;
        if (i % 97 == 96) {
            g.If(0, kX, 5);
            g.W32(kZ + 4 * (i % 50), 7);
            g.Endif();
        }
        if (i % 131 == 130) g.C0(g.Dump());
    }
    (void)r;
}

vector<Hand>& HandList() {
    static vector<Hand> list = {
        {"writes", HWrites},       {"strings", HStrings},    {"serial", HSerial},      {"ifs", HIfs},
        {"four-x", HFourX},        {"goto", HGoto},          {"repeat", HRepeat},      {"gecko-regs", HRegs},
        {"gecko-ops", HOps},       {"copy", HCopy},          {"c0-dump", HC0Dump},     {"c0-writes", HC0Writes},
        {"c0-regs", HC0Regs},      {"c0-scramble", HScramble}, {"hooks", HHooks},      {"c6", HC6},
        {"cc", HCc},               {"end", HEnd},            {"skipped", HSkipped},    {"header", HHeader},
        {"two-stores", HTwoStores}, {"bail-a-ce", HBail},    {"bail-f2-f4", HBailF},   {"bail-f6", HBailF6},
        {"bail-f6-miss", HBailF6Miss}, {"bail-c8-ca", HBailC8},
        {"list-switch", HListSwitch, 16}, {"pure", HPure, 14},
    };
    return list;
}

Scenario MakeHand(const Hand& h, int config, uint64_t seed) {
    Rng g(seed);
    Gen gen;
    BaseImage(gen.sc, g, kListA);
    h.build(gen, g);
    Scenario s = gen.Finish(std::string(h.name) + "/cfg" + std::to_string(config));
    Put32(s, kListPtr, gen.base);
    s.seed = seed;
    s.tracked = PickPages(g, config);
    s.consumeEachPass = config == 1 || config == 3;
    AddScript(s, g, h.passes, kStackTop);
    return s;
}

// ---------------------------------------------------------------------------------------------------------------------
// Random lists

struct RandomList {
    Gen g;
    Rng& r;
    vector<uint32_t> hot;  // zone offsets of the places the codes write and read
    struct Pending {
        size_t at;  // word index of the body's first word
    };
    vector<Pending> pendingWrites;
    int nextBlock = 0;
    uint32_t units = 0, limit = 0;

    explicit RandomList(Rng& rng) : r(rng) {
        const uint32_t spread = R(r, 10);  // few places (codes overwrite each other), some, or many (most stores find their value there)
        const int places = spread == 0 ? 6 : spread < 4 ? 28 : 400;
        for (int i = 0; i < places; ++i) {
            const Zone& z = kZones[R(r, 3)];
            hot.push_back(z.off + R(r, (z.size - 0x400) / 64) * 64 + 4 * R(r, 12));
        }
    }
    uint32_t HotOff(uint32_t extra = 0) { return hot[R(r, static_cast<uint32_t>(hot.size()))] + extra; }
    uint32_t Base() { return kBases[R(r, 4)]; }
    uint32_t WordAt(size_t index) const { return g.base + 4u * static_cast<uint32_t>(index); }
    void TargetBoth(uint32_t off, uint32_t len) {
        g.Target(0x80000000u + off, len);
        g.Target(0x90000000u + off, len);
    }
    void MutWord(size_t index, bool any, vector<vector<uint8_t>> variants = {}) {
        Mut m;
        m.addr = WordAt(index);
        m.len = 4;
        m.any = any;
        m.variants = std::move(variants);
        g.sc.muts.push_back(m);
    }
    static vector<uint8_t> BE(uint32_t v) { return {uint8_t(v >> 24), uint8_t(v >> 16), uint8_t(v >> 8), uint8_t(v)}; }

    void Write() {
        const bool p = Chance(r, 30);
        const uint32_t pbit = p ? 0x10000000u : 0u;
        const uint32_t off = HotOff(4 * R(r, 8));
        const size_t at = g.w.size();
        switch (R(r, 3)) {
        case 0: {
            const uint32_t n = 1 + R(r, 8);
            g.C(pbit | off, ((n - 1) << 16) | R(r, 256));
            TargetBoth(off, n);
            vector<vector<uint8_t>> v;
            for (int i = 0; i < 3; ++i) v.push_back(BE(((n - 1) << 16) | R(r, 256)));
            MutWord(at + 1, false, v);
            break;
        }
        case 1: {
            const uint32_t n = 1 + R(r, 6);
            g.C(pbit | (0x02u << 24) | off, ((n - 1) << 16) | R(r, 65536));
            TargetBoth(off, 2 * n);
            break;
        }
        default: {
            g.C(pbit | (0x04u << 24) | off, static_cast<uint32_t>(r()));
            TargetBoth(off & ~3u, 4);
            MutWord(at + 1, true);
            vector<vector<uint8_t>> v;
            for (int i = 0; i < 4; ++i) v.push_back(BE(pbit | (0x04u << 24) | HotOff(4 * R(r, 8))));
            MutWord(at, false, v);
            break;
        }
        }
    }
    void Str() {
        const bool p = Chance(r, 20);
        const uint32_t len = Chance(r, 8) ? 0 : 1 + R(r, 40);
        vector<uint8_t> b(len);
        for (auto& x : b) x = Chance(r, 20) ? 0 : static_cast<uint8_t>(r());
        const uint32_t off = HotOff(R(r, 64));
        const size_t first = g.w.size();
        g.C((p ? 0x10000000u : 0u) | (0x06u << 24) | off, len);
        vector<uint8_t> padded = b;
        while (padded.size() % 8) padded.push_back(0);
        for (size_t i = 0; i < padded.size(); i += 4)
            g.w.push_back(uint32_t(padded[i]) << 24 | uint32_t(padded[i + 1]) << 16 | uint32_t(padded[i + 2]) << 8 | padded[i + 3]);
        if (len) {
            TargetBoth(off, len);
            Mut m;
            m.addr = WordAt(first + 2);
            m.len = static_cast<uint32_t>(padded.size());
            m.any = true;
            g.sc.muts.push_back(m);
        }
    }
    void Serial() {
        const bool p = Chance(r, 20);
        const uint32_t off = HotOff(R(r, 32));
        const uint32_t size = R(r, 3), count = 1 + R(r, 8), step = 1 + R(r, 12);
        const size_t at = g.w.size();
        g.C((p ? 0x10000000u : 0u) | (0x08u << 24) | off, static_cast<uint32_t>(r()));
        g.C((size << 28) | ((count - 1) << 16) | step, static_cast<uint32_t>(r()));
        TargetBoth(off, count * step + 4);
        MutWord(at + 1, true);
        MutWord(at + 3, true);
    }
    void Hook() {
        const bool p = Chance(r, 20);
        const uint32_t off = HotOff() & ~3u;
        const uint32_t lines = 1 + R(r, 4);
        const uint32_t type = Chance(r, 50) ? 0xC2000000u : 0xC4000000u;
        const size_t at = g.w.size();
        g.C((p ? 0x10000000u : 0u) | type | off, lines);
        for (uint32_t i = 0; i < 2 * lines; ++i) g.w.push_back(static_cast<uint32_t>(r()));
        TargetBoth(off, 8);
        for (uint32_t i = 0; i < 2 * lines - 1; ++i) MutWord(at + 2 + i, true);
    }
    void Branch() {
        const uint32_t off = HotOff() & ~3u;
        const size_t at = g.w.size();
        g.C((Chance(r, 20) ? 0x10000000u : 0u) | 0xC6000000u | off | (Chance(r, 30) ? 1u : 0u), static_cast<uint32_t>(r()) & ~3u);
        TargetBoth(off, 4);
        MutWord(at + 1, true);
    }
    void If(bool endif = false) {
        const bool p = Chance(r, 20);
        const bool sixteen = Chance(r, 40);
        const uint32_t cmp = R(r, 4);
        const uint32_t off = (HotOff() & ~3u) + (sixteen ? 2 * R(r, 2) : 0);
        const uint32_t type = (sixteen ? 0x28u : 0x20u) + 2 * cmp;
        const size_t at = g.w.size();
        uint32_t lo = Chance(r, 40) ? R(r, 4) : static_cast<uint32_t>(r());
        if (sixteen && Chance(r, 50)) lo &= 0x0000FFFFu;
        g.C((p ? 0x10000000u : 0u) | (type << 24) | off | (endif ? 1u : 0u), lo);
        g.sc.condMem.push_back({0x80000000u + (off & ~3u), 4});
        g.sc.condMem.push_back({0x90000000u + (off & ~3u), 4});
        MutWord(at + 1, true);
        vector<vector<uint8_t>> v;
        for (int i = 0; i < 3; ++i) {
            const uint32_t t = (sixteen ? 0x28u : 0x20u) + 2 * R(r, 4);
            v.push_back(BE((p ? 0x10000000u : 0u) | (t << 24) | off | (endif ? 1u : 0u)));
        }
        MutWord(at, false, v);
    }
    void FourXSafe() {
        const uint32_t form = R(r, 4);
        const bool po = Chance(r, 50);
        const uint32_t pb = po ? 0x08000000u : 0u;
        switch (form) {
        case 0: g.C(0x42000000u | pb, Base() | R(r, 0x100)); break;
        case 1: g.C(0x40000000u | pb, kPtrTable + 4 * R(r, 64)); break;
        case 2: {
            const uint32_t off = HotOff() & ~3u;
            g.C(0x44000000u | pb, off);  // Store32(r12 + off, ba or po)
            TargetBoth(off, 4);
            break;
        }
        default:
            if (!po) g.C(0x46000000u | R(r, 64), 0);
            else g.C(0x42000000u | pb, Base());
            break;
        }
    }
    void CopySafe() {
        const uint32_t count = R(r, 0x60);
        if (Chance(r, 50)) {  // pointer registers on both sides, small offsets
            g.Copy(Chance(r, 50), R(r, 4), R(r, 4), count, R(r, 0x80));
        } else if (Chance(r, 50)) {  // destination base + zone offset
            g.Copy(true, 15, R(r, 4), count, HotOff());
        } else {
            g.Copy(false, R(r, 4), 15, count, HotOff());
        }
    }
    void GrSafe() {
        const uint32_t form = R(r, 7);
        const uint32_t d = 4 + R(r, 12), s = R(r, 16);
        const size_t at = g.w.size();
        switch (form) {
        case 0:
            g.GrSet(d, static_cast<uint32_t>(r()), false, false);
            MutWord(at + 1, true);
            break;
        case 1: g.GrSet(R(r, 4), 0x80000000u + HotOff(), false, false); break;
        case 2: g.GrLoad(d, R(r, 3), 0x80000000u + HotOff(R(r, 16))); break;
        case 3: {
            const uint32_t count = 1 + R(r, 6), size = R(r, 3);
            const uint32_t off = HotOff(R(r, 16));
            g.GrStore(Chance(r, 25) ? R(r, 16) : d, size, 0x80000000u + off, count);
            TargetBoth(off, count * 4);
            break;
        }
        case 4: g.GrOpImm(R(r, 13), d, static_cast<uint32_t>(r()), 0); break;
        case 5: g.GrOpReg(R(r, 13), d, s, 0); break;
        default: CopySafe(); break;
        }
    }
    void Body() {
        using namespace Ppc;
        const uint32_t form = R(r, 10);
        switch (form) {
        case 0: case 1: case 2: g.C0(g.Dump()); break;
        case 3: case 4: {
            const uint32_t off = HotOff(R(r, 16));
            const uint32_t size = 1u << R(r, 3);
            g.C0(Gen::StoreWord(kBases[R(r, 4)] + off, static_cast<uint32_t>(r()), size == 8 ? 4 : size));
            TargetBoth(off, 4);
            break;
        }
        case 5: g.C0(Gen::StoreGr(4 + R(r, 12), static_cast<uint32_t>(r()))); break;
        case 6: case 7: {  // a list byte: patched once the whole list is known
            pendingWrites.push_back({g.w.size() + 2});
            g.C0(Gen::StoreWord(0x80000000u, 0));
            break;
        }
        case 8: {
            Gen::RegSet rs;
            if (Chance(r, 60)) { rs.r6 = true; rs.v6 = Base(); }
            if (Chance(r, 60)) { rs.r16 = true; rs.v16 = Base(); }
            if (!rs.r6 && !rs.r16) { rs.r16 = true; rs.v16 = Base(); }
            g.C0(Gen::SetRegs(rs));
            break;
        }
        default: g.C0(Gen::Scramble(r)); break;
        }
    }
    void Simple() {
        const uint32_t k = R(r, 100);
        if (k < 30) Write();
        else if (k < 38) Str();
        else if (k < 42) Serial();
        else if (k < 54) Hook();
        else if (k < 60) Body();
        else if (k < 65) FourXSafe();
        else if (k < 76) GrSafe();
        else if (k < 79) Branch();
        else if (k < 81) g.CC(R(r, 4));
        else Write();
    }
    void Units(uint32_t n, uint32_t depth, bool pure = false) {
        for (uint32_t i = 0; i < n && units < limit; ++i) {
            ++units;
            const uint32_t k = R(r, 100);
            if (pure || depth >= 3) {
                if (pure) {
                    const uint32_t q = R(r, 10);
                    if (q < 5) Write();
                    else if (q < 7) Str();
                    else Hook();
                } else {
                    Simple();
                }
                continue;
            }
            if (k < 12) {
                IfBlock(depth);
            } else if (k < 15) {
                // a forward goto over a few codes
                const size_t at = g.w.size();
                g.Goto(0, Chance(r, 70) ? 2 : (Chance(r, 50) ? 0 : 1));
                Units(1 + R(r, 4), depth + 1);
                const uint32_t lines = static_cast<uint32_t>((g.w.size() - at - 2) / 2);
                g.w[at] |= lines;
            } else if (k < 17 && nextBlock < 4) {
                // gosub S; A; goto E; S: B; return; E:
                const uint32_t block = 6 + (nextBlock++ % 4);
                const size_t gs = g.w.size();
                g.Gosub(0, block, 0);
                Units(1 + R(r, 3), depth + 1);
                const size_t go = g.w.size();
                g.Goto(0, 2);
                g.w[gs] |= static_cast<uint32_t>((g.w.size() - gs - 2) / 2);
                Units(1 + R(r, 3), depth + 1);
                g.Return(block, 0);
                g.w[go] |= static_cast<uint32_t>((g.w.size() - go - 2) / 2);
            } else if (k < 20 && nextBlock < 4) {
                const uint32_t block = nextBlock++ % 4;
                g.SetRepeat(block, R(r, 4));
                Units(1 + R(r, 4), depth + 1);
                g.ExecRepeat(block);
            } else if (k < 22 && depth == 0) {
                // r8 set from a body, then the skipped stretch ends at E0
                Gen::RegSet rs;
                rs.r8 = true;
                rs.v8 = 1;
                g.C0(Gen::SetRegs(rs));
                Units(1 + R(r, 4), depth + 1, true);
                g.E0(Chance(r, 30) ? kBases[R(r, 4)] >> 16 : 0, Chance(r, 30) ? kBases[R(r, 4)] >> 16 : 0);
            } else if (k < 24 && depth == 0) {
                // another Gecko register table for a few simple codes
                Gen::RegSet rs;
                rs.r7 = true;
                rs.v7 = kAlt;
                g.C0(Gen::SetRegs(rs));
                const uint32_t m = 1 + R(r, 3);
                for (uint32_t j = 0; j < m; ++j) {
                    const uint32_t q = R(r, 4);
                    if (q == 0) g.C0(g.Dump());
                    else if (q == 1) g.GrLoad(4 + R(r, 12), 2, 0x80000000u + HotOff());
                    else if (q == 2) g.GrOpImm(R(r, 11), 4 + R(r, 12), static_cast<uint32_t>(r()));
                    else Write();
                }
                rs.v7 = kGr;
                g.C0(Gen::SetRegs(rs));
            } else if (k < 30 && depth == 0) {
                Units(10 + R(r, 120), depth + 1, true);
            } else {
                Simple();
            }
        }
    }
    void IfBlock(uint32_t depth) {
        If();
        Units(1 + R(r, 6), depth + 1);
        if (Chance(r, 30)) {
            g.Else();
            Units(1 + R(r, 4), depth + 1);
        }
        if (Chance(r, 20)) {  // endif and a new if in one code
            If(true);
            Units(1 + R(r, 4), depth + 1);
        }
        g.Endif(1);
    }
    void PatchWrites() {
        for (const Pending& p : pendingWrites) {
            if (g.sc.muts.empty()) break;
            const Mut& m = g.sc.muts[R(r, static_cast<uint32_t>(g.sc.muts.size()))];
            uint32_t addr = m.addr;
            uint32_t value = static_cast<uint32_t>(r());
            if (m.len == 4 && !m.any) {
                const auto& v = m.variants[R(r, static_cast<uint32_t>(m.variants.size()))];
                value = uint32_t(v[0]) << 24 | uint32_t(v[1]) << 16 | uint32_t(v[2]) << 8 | v[3];
            } else {
                addr += 4 * R(r, m.len / 4);
            }
            using namespace Ppc;
            g.w[p.at] = Lis(0, addr >> 16);
            g.w[p.at + 1] = Ori(0, 0, addr & 0xFFFFu);
            g.w[p.at + 2] = Lis(2, value >> 16);
            g.w[p.at + 3] = Ori(2, 2, value & 0xFFFFu);
        }
    }
};

Scenario MakeRandom(uint64_t seed) {
    Rng g(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull);
    RandomList rl(g);
    const uint32_t k = R(g, 100);
    rl.limit = k < 55 ? 15 + R(g, 80) : k < 92 ? 100 + R(g, 400) : 1200 + R(g, 1200);
    const uint32_t b = R(g, 10);
    rl.g.base = b < 7 ? kListA : kListB;
    const uint32_t listPtr = (Chance(g, 12) ? 0x40000000u : 0u) | rl.g.base;
    BaseImage(rl.g.sc, g, listPtr);
    rl.Units(rl.limit, 0);
    rl.PatchWrites();
    Scenario s = rl.g.Finish("random");
    Put32(s, kListPtr, listPtr);
    s.seed = seed;
    const uint32_t c = R(g, 100);
    const int config = c < 14 ? 0 : c < 38 ? 1 : c < 52 ? 2 : c < 66 ? 3 : 4;
    s.tracked = PickPages(g, config);
    s.consumeEachPass = Chance(g, 35);
    const int passes = 6 + static_cast<int>(R(g, 18));
    AddScript(s, g, passes, Chance(g, 80) ? kStackTop : kStackTop + 0x1000);
    return s;
}

// ---------------------------------------------------------------------------------------------------------------------

void PrintCounters(const char* label, const Counters& c) {
    static const char* names[kCounterCount] = {"passes",   "steps interpreted", "runs replayed",    "runs recorded",
                                               "runs dropped", "key misses",    "extents compared", "extents trusted",
                                               "extents rewritten", "verify checks", "verify mismatches"};
    std::printf("%s:", label);
    for (size_t k = 0; k < kCounterCount; ++k) std::printf(" %s %llu;", names[k], (unsigned long long)c.v[k]);
    std::printf("\n");
}

// The harness itself: what one pass of the binary does to a few hand-written lists, so that a list that silently does
// nothing (or a list the binary walks differently than meant) is noticed.
int SelfCheck() {
    int bad = 0;
    const auto onePass = [&](const char* name) {
        for (const Hand& h : HandList())
            if (std::strcmp(h.name, name) == 0) {
                Scenario s = MakeHand(h, 0, 77);
                vector<Step> steps;
                for (const Step& st : s.steps)
                    if (st.kind == Step::Pass) {
                        steps.push_back(st);
                        break;
                    }
                s.steps = steps;
                RunScenario(s, EngineKind::Ref);
                return;
            }
        std::fprintf(stderr, "self check: no list '%s'\n", name);
        ++bad;
    };
    const auto expect = [&](const char* what, uint32_t addr, uint32_t want, uint32_t mask = 0xFFFFFFFFu) {
        const uint32_t got = Memory::Read32(addr);
        if ((got & mask) != (want & mask)) {
            std::fprintf(stderr, "self check: %s: %08X holds %08X, wanted %08X\n", what, addr, got, want);
            ++bad;
        }
    };
    const uint32_t frame = kStackTop - 168;
    onePass("writes");
    expect("writes: last store wins", 0x80400300, 0x77777777);
    expect("writes: MEM2 through po", 0x91100000, 0xCAFEF00D);
    expect("writes: uncached view", 0xC0400110, 0x77777777, 0xFFFF0000);
    expect("writes: ba in MEM2", 0x90400000, 0x01234567);
    expect("writes: 8-bit repeat", 0x80400040, 0x33333333);
    onePass("strings");
    expect("strings: repeated string", 0x80403000, 0x5A5A5A5A);
    expect("strings: a later string over an earlier store", 0x80401008, 0xE5E6E7E8);
    onePass("ifs");
    expect("ifs: eq true", kZ + 4, 0x1001);
    expect("ifs: else branch", kZ + 0x104, 2, 0);  // (either branch is fine: only that the list ran)
    expect("ifs: after the blocks", kZ + 0x110, 5);
    onePass("goto");
    expect("goto: before", kZ, 1);
    expect("goto: after the skipped lines", kZ + 12, 4);
    expect("gosub: the subroutine and the return", kZ + 0x24, 7);
    expect("gosub: after the return", kZ + 0x20, 6);
    onePass("repeat");
    expect("repeat: after the loops", kZ + 0x10, 0x3333);
    onePass("c0-dump");
    expect("c0-dump: r1 in the first dump", frame - 0x100 + 4, frame);
    expect("c0-dump: lr in the first dump", frame - 0x100 + 0x8C, 0x80001F14u);
    onePass("hooks");
    expect("hooks: a branch at the first target", 0x80402000, 0x48000000u, 0xFC000000u);
    expect("hooks: the hook ran (it stored its marker)", frame - 0x60, 0x55);
    expect("hooks: the second hook ran", frame - 0x64, 0x66);
    onePass("bail-f2-f4");
    expect("f2: matching checksum installs the hook", 0x80402000, 0x48000000u, 0xFC000000u);
    expect("f2: wrong checksum installs nothing", 0x80402100, 0);
    expect("f4: matching checksum installs the hook", 0x80402200, 0x48000000u, 0xFC000000u);
    onePass("bail-f6");
    expect("f6: the search found the words", kListA + 16, 0xF6000301u);
    expect("f6: and left their address in the header", kListA + 20, 0x80400104u);
    onePass("bail-f6-miss");
    expect("f6: not found", kListA + 16, 0xF6000101u);
    onePass("cc");
    expect("cc: the list ran to the end", kZ + 0x104, 2);
    return bad;
}

int Main() {
    const auto start = std::chrono::steady_clock::now();
    Memory::Init(Memory::Config::WiiDefaults());
    TranslatedFunctionRegistry::Finalize();
    BuildPages();
    g_verbose = std::getenv("GECKO_TEST_VERBOSE") != nullptr;
    g_inject = std::getenv("GECKO_TEST_INJECT") != nullptr;
    const auto envNum = [](const char* name, uint64_t dflt) {
        const char* v = std::getenv(name);
        return v != nullptr && v[0] != '\0' ? std::strtoull(v, nullptr, 0) : dflt;
    };
    uint64_t randomCount = envNum("GECKO_TEST_RANDOM", 500);
    const uint64_t firstSeed = envNum("GECKO_TEST_SEED", 1);
    const bool only = std::getenv("GECKO_TEST_ONLY") != nullptr;
    const char* nameFilter = std::getenv("GECKO_TEST_NAME");

    if (SelfCheck() != 0) {
        std::fprintf(stderr, "gecko handler: the harness's own checks failed\n");
        return 1;
    }
    g_totals.engines = {EngineKind::Ref, EngineKind::Native};
    const bool plan = BrawlGecko_PlanControl != nullptr && BrawlGecko_PlanCounters != nullptr;
    if (plan) {
        g_totals.engines.push_back(EngineKind::Plan);
        g_totals.engines.push_back(EngineKind::PlanNoMarks);
    }
    std::printf("gecko handler: engines REF, NATIVE%s\n", plan ? ", PLAN, PLAN_NOMARKS (write plan present)" : " (no write plan in this build)");

    if (g_inject) {
        // a result corrupted on purpose must be reported as a mismatch
        const Scenario s = MakeHand(HandList()[0], 1, 4242);
        RunSuiteScenario(s);
        if (g_totals.failures == 0) {
            std::printf("gecko handler: the injected corruption was NOT detected\n");
            return 0;
        }
        std::printf("gecko handler: the injected corruption was detected\n");
        return 1;
    }

    if (only) {
        const uint64_t seed = envNum("GECKO_TEST_ONLY", 1);
        if (std::getenv("GECKO_TEST_DUMP") != nullptr) {  // the list's words
            const Scenario d = MakeRandom(seed);
            for (const auto& [addr, bytes] : d.image)
                if (addr == d.listAddr)
                    for (size_t i = 0; i + 8 <= bytes.size(); i += 8)
                        std::printf("%08X: %02X%02X%02X%02X %02X%02X%02X%02X\n", addr + static_cast<uint32_t>(i), bytes[i], bytes[i + 1],
                                    bytes[i + 2], bytes[i + 3], bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]);
            std::fflush(stdout);
        }
        RunSuiteScenario(MakeRandom(seed));
        std::printf("gecko handler: random scenario %llu: %s\n", (unsigned long long)seed, g_totals.failures ? "FAILED" : "ok");
        return g_totals.failures ? 1 : 0;
    }

    uint64_t handSeed = 1000;
    for (const Hand& h : HandList()) {
        if (nameFilter != nullptr && std::strstr(h.name, nameFilter) == nullptr) continue;
        for (int config = 0; config < 5; ++config) RunSuiteScenario(MakeHand(h, config, handSeed++));
        if (std::strcmp(h.name, "pure") == 0 || std::strcmp(h.name, "writes") == 0) {
            // quiet: no events, everything tracked: the plan replays and trusts the marks
            Rng g(handSeed++);
            Gen gen;
            BaseImage(gen.sc, g, kListA);
            h.build(gen, g);
            Scenario s = gen.Finish(std::string(h.name) + "/quiet");
            Put32(s, kListPtr, gen.base);
            s.seed = handSeed;
            s.tracked = PickPages(g, 1);
            s.expectReplay = s.expectTrust = true;
            AddScript(s, g, 40, kStackTop, false);
            RunSuiteScenario(s);
        }
    }
    const auto handDone = std::chrono::steady_clock::now();
    if (nameFilter == nullptr)
        for (uint64_t i = 0; i < randomCount; ++i) RunSuiteScenario(MakeRandom(firstSeed + i));
    const double hand = std::chrono::duration<double>(handDone - start).count();
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("gecko handler: %llu scenarios, %llu passes per engine, %d failed; hand-written %.1f s, all %.1f s\n",
                (unsigned long long)g_totals.scenarios, (unsigned long long)g_totals.passes, g_totals.failures, hand, total);
    int bad = g_totals.failures;
    if (plan) {
        PrintCounters("gecko plan (mode 1)", g_totals.c[0]);
        PrintCounters("gecko plan (mode 2)", g_totals.c[1]);
        const uint64_t replayed = g_totals.c[0].v[kReplayed] + g_totals.c[1].v[kReplayed];
        if (replayed == 0) {
            std::fprintf(stderr, "FAIL: no run was ever replayed\n");
            ++bad;
        }
        if (g_totals.trustedInTracked == 0) {
            std::fprintf(stderr, "FAIL: no extent was ever trusted in a scenario with tracked pages\n");
            ++bad;
        }
        if (nameFilter == nullptr && (g_totals.replayedInExpecting == 0 || g_totals.trustedInExpecting == 0)) {
            std::fprintf(stderr, "FAIL: the quiet scenarios (tracked, no events) replayed %llu runs and trusted %llu extents\n",
                         (unsigned long long)g_totals.replayedInExpecting, (unsigned long long)g_totals.trustedInExpecting);
            ++bad;
        }
        if (g_totals.c[0].v[kVerifyMismatch] + g_totals.c[1].v[kVerifyMismatch] != 0) {
            std::fprintf(stderr, "FAIL: the plan's verify mode found mismatches\n");
            ++bad;
        }
    }
    return bad == 0 ? 0 : 1;
}

}  // namespace

int main() { return Main(); }
