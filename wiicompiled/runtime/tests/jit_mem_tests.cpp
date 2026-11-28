// The JIT's loads and stores against the classic interpreter, on what the differential test does not compare: rollback's
// dirty marks, what the checked store path observed, the exception of a store to an unmapped address, and the float
// conversions on every input class.
//
// Each case is a few guest instructions run once under Engine::Classic and once under Engine::Jit (threshold 0, so the
// block is compiled before it runs), from the same registers, the same guest bytes and cleared marks. Compared: every
// register, CR field value (bytewise) and FPR bit, the bytes and the dirty marks around each address the program touches, the address of the last
// store the executable-write check saw, and the exception a run ends with (its type, text and, for an access violation,
// address and length). A float load is followed by a store of the register as a scalar and as a pair, which shows the
// register's form.
//
// The memory is the Wii map (MEM1, MEM2, the locked cache, MEM1's mirrors); one 64 KiB chunk of MEM1 is registered as
// executable code, so the lean store policy sends its stores to the checked path.
#include "abi_bridge.h"
#include "brawl_dirty_memory.h"
#include "jit_test_cr.h"
#include "isa/ppc_isa_fpenv.h"
#include "lean_window.h"
#include "memory.h"
#include "ppc_interpreter.h"
#include "../src/jit/jit_x64.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace RecompMod {
extern std::atomic<uint32_t> g_testLastExecutableWrite;  // interpreter_test_support.cpp
}

namespace {

using PpcInterpreter::Engine;

// --- encoders -------------------------------------------------------------------------------------------------------------
constexpr uint32_t D(uint32_t op, uint32_t rt, uint32_t ra, int32_t imm) {
    return (op << 26) | (rt << 21) | (ra << 16) | (static_cast<uint32_t>(imm) & 0xFFFFu);
}
constexpr uint32_t X(uint32_t rt, uint32_t ra, uint32_t rb, uint32_t xo) {
    return (31u << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (xo << 1);
}
constexpr uint32_t Psq(uint32_t op, uint32_t frt, uint32_t ra, uint32_t w, uint32_t i, int32_t disp) {
    return (op << 26) | (frt << 21) | (ra << 16) | (w << 15) | (i << 12) | (static_cast<uint32_t>(disp) & 0xFFFu);
}
constexpr uint32_t kBlr = 0x4E800020u;
constexpr uint32_t Fmr(uint32_t d, uint32_t b) { return (63u << 26) | (d << 21) | (b << 11) | (72u << 1); }
constexpr uint32_t PsMr(uint32_t d, uint32_t b) { return (4u << 26) | (d << 21) | (b << 11) | (72u << 1); }

// --- the machine --------------------------------------------------------------------------------------------------------
constexpr uint32_t kCodeBase = 0x80100000u;
constexpr uint32_t kCodeLimit = 0x80200000u;
constexpr uint32_t kScratch = 0x80340000u;  // where the observers of a float load store
constexpr uint32_t kReturn = 0x80020000u;
constexpr uint32_t kFlaggedChunk = 0x80500000u;  // registered as code: the policy flags this 64 KiB chunk
constexpr int kDisp = 8;

int g_failures = 0;
uint64_t g_cases = 0;
uint64_t g_threw = 0, g_observedCodeWrite = 0, g_marked = 0;  // what the cases reached, so a test that stopped reaching it is noticed
uint32_t g_codeNext = kCodeBase;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint32_t Next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return static_cast<uint32_t>((s * 0x2545F4914F6CDD1Dull) >> 32);
    }
    uint64_t Next64() { return (uint64_t{Next()} << 32) | Next(); }
};

struct Snap {
    CpuContext ctx{};
    bool threw = false;
    std::string what;
    uint32_t violationAddress = 0;
    uint64_t violationLength = 0;
    std::vector<uint8_t> before, bytes, marks;  // before: the bytes the run started from
    uint32_t lastExecutableWrite = 0;
};

// What a case compares around: the guest addresses it touches (a window of 256 bytes below to 256 above each, and the
// marks of the blocks there).
struct Case {
    std::string name;
    std::vector<uint32_t> words;
    CpuContext init{};
    std::vector<uint32_t> windows;
    bool compareBytes = true;       // false where the addresses have no backing (a store that must throw)
    // false for programs with several stores, which the JIT may mark as one group (jit_x64_mem.cpp): its marks must then be
    // a subset of the interpreter's and cover every block whose bytes the run changed, instead of equal to them
    bool exactMarks = true;
    std::function<void()> prepare;  // the guest bytes the run starts from (deterministic)
};

uint32_t Canonical(uint32_t address) { return address & MemoryInline::kLeanMask; }

CpuContext BaseContext(uint64_t seed) {
    Rng r(seed);
    CpuContext c{};
    for (auto& g : c.gpr) g = r.Next();
    for (auto& f : c.fpr) f.raw = r.Next64();
    c.gpr[1] = 0x80031800u;
    c.gpr[4] = kDisp;
    c.gpr[6] = kScratch;
    JitTestCr::Randomize(c, [&] { return r.Next(); });
    c.xer = r.Next() & 0xE000007Fu;
    c.ctr = r.Next();
    c.lr = kReturn;
    c.gqr[0] = 0;
    return c;
}

// The classic interpreter and the JIT, one run each from the same state.
Snap RunOne(Engine engine, const Case& c, uint32_t entry) {
    if (c.prepare) c.prepare();
    for (uint32_t w : c.windows) {
        const uint32_t first = (Canonical(w) >> 6) - 8, last = (Canonical(w) >> 6) + 12;
        std::memset(BrawlDirtyMemory::leanBlocks + first, 0, last - first + 1);
    }
    RecompMod::g_testLastExecutableWrite.store(0);
    PpcInterpreter::SetEngineOverride(engine);
    PpcInterpreter::SetJitThreshold(0);
    Snap s;
    for (uint32_t w : c.windows) {
        if (!c.compareBytes) break;
        const uint8_t* p = MemoryInline::LeanHost((w & ~63u) - 256u);
        s.before.insert(s.before.end(), p, p + 576);
    }
    CpuContext ctx = c.init;
    ctx.pc = 0;
    {
        CpuContextScope scope(&ctx);
        try {
            PpcInterpreter::Run(&ctx, entry);
        } catch (const Memory::AccessViolation& e) {
            s.threw = true;
            s.what = e.what();
            s.violationAddress = e.address();
            s.violationLength = e.length();
        } catch (const std::exception& e) {
            s.threw = true;
            s.what = e.what();
        } catch (...) {
            s.threw = true;
            s.what = "unknown exception";
        }
    }
    ctx.pc = 0;
    s.ctx = ctx;
    for (uint32_t w : c.windows) {
        if (c.compareBytes) {
            const uint8_t* p = MemoryInline::LeanHost((w & ~63u) - 256u);
            s.bytes.insert(s.bytes.end(), p, p + 576);
        }
        const uint8_t* m = BrawlDirtyMemory::leanBlocks + (Canonical(w) >> 6) - 8;
        s.marks.insert(s.marks.end(), m, m + 21);
    }
    s.lastExecutableWrite = RecompMod::g_testLastExecutableWrite.load();
    return s;
}

uint64_t g_groupedMarks = 0;  // cases where the JIT left out a mark the interpreter set (relaxed comparison only)

std::string Diff(const Snap& a, const Snap& b, bool exactMarks) {
    std::string out;
    char line[256];
    const auto word = [&](const char* name, uint64_t x, uint64_t y) {
        if (x == y) return;
        std::snprintf(line, sizeof line, "    %s: classic=%llX jit=%llX\n", name, static_cast<unsigned long long>(x),
                      static_cast<unsigned long long>(y));
        out += line;
    };
    if (a.threw != b.threw) word("threw", a.threw, b.threw);
    if (a.what != b.what) out += "    exception text: '" + a.what + "' vs '" + b.what + "'\n";
    word("violation address", a.violationAddress, b.violationAddress);
    word("violation length", a.violationLength, b.violationLength);
    for (int i = 0; i < 32; ++i) {
        char n[16];
        std::snprintf(n, sizeof n, "r%d", i);
        word(n, a.ctx.gpr[i], b.ctx.gpr[i]);
        std::snprintf(n, sizeof n, "f%d", i);
        word(n, a.ctx.fpr[i].raw, b.ctx.fpr[i].raw);
    }
    for (uint32_t i = 0; i < 8; ++i) {  // bytewise (stwcx. stores CR0's exact value; no other field may change)
        char n[16];
        std::snprintf(n, sizeof n, "crv[%u]", i);
        word(n, static_cast<uint64_t>(a.ctx.crv[i]), static_cast<uint64_t>(b.ctx.crv[i]));
    }
    word("lr", a.ctx.lr, b.ctx.lr);
    word("ctr", a.ctx.ctr, b.ctx.ctr);
    word("last executable write", a.lastExecutableWrite, b.lastExecutableWrite);
    for (size_t i = 0; i < a.bytes.size() && i < b.bytes.size(); ++i)
        if (a.bytes[i] != b.bytes[i]) {
            std::snprintf(line, sizeof line, "    byte %zu (window %zu +%zu): classic=%02X jit=%02X\n", i, i / 576, i % 576,
                          a.bytes[i], b.bytes[i]);
            out += line;
            if (out.size() > 1500) break;
        }
    bool fewer = false;
    for (size_t i = 0; i < a.marks.size() && i < b.marks.size(); ++i)
        if (a.marks[i] != b.marks[i]) {
            if (!exactMarks && a.marks[i] != 0 && b.marks[i] == 0) {
                fewer = true;
                continue;
            }
            std::snprintf(line, sizeof line, "    mark %zu (window %zu block %zu): classic=%u jit=%u\n", i, i / 21, i % 21,
                          a.marks[i], b.marks[i]);
            out += line;
        }
    if (!exactMarks) {  // the JIT's marks cover each block whose bytes it changed (window w's bytes are its blocks 4..12)
        g_groupedMarks += fewer;
        for (size_t i = 0; i < b.bytes.size() && i < b.before.size(); ++i) {
            const size_t mark = (i / 576) * 21 + 4 + (i % 576) / 64;
            if (b.bytes[i] != b.before[i] && mark < b.marks.size() && b.marks[mark] == 0) {
                std::snprintf(line, sizeof line, "    byte %zu (window %zu +%zu) changed, its block not marked by the jit\n", i,
                              i / 576, i % 576);
                out += line;
                break;
            }
        }
    }
    return out;
}

void Run(const Case& c) {
    // addresses recycle: a new program replaces the old one's blocks
    if (g_codeNext + 4u * c.words.size() + 16 >= kCodeLimit) g_codeNext = kCodeBase;
    const uint32_t entry = g_codeNext;
    g_codeNext += 4u * static_cast<uint32_t>(c.words.size()) + 0x20u;
    g_codeNext = (g_codeNext + 31u) & ~31u;
    for (size_t i = 0; i < c.words.size(); ++i) Memory::Write32(entry + 4u * static_cast<uint32_t>(i), c.words[i]);
    const Snap classic = RunOne(Engine::Classic, c, entry);
    const Snap jit = RunOne(Engine::Jit, c, entry);
    ++g_cases;
    g_threw += classic.threw;
    g_observedCodeWrite += classic.lastExecutableWrite != 0;
    for (uint8_t m : classic.marks)
        if (m != 0) {
            ++g_marked;
            break;
        }
    const std::string diff = Diff(classic, jit, c.exactMarks);
    if (!diff.empty()) {
        if (++g_failures <= 12) {
            std::printf("FAIL %s\n", c.name.c_str());
            for (size_t i = 0; i < c.words.size(); ++i) std::printf("    %08X", c.words[i]);
            std::printf("\n%s", diff.c_str());
        }
    }
}

// Random bytes around an address, with `word` (and `second` after it) planted at it.
std::function<void()> Fill(uint32_t address, uint64_t seed, uint32_t word, uint32_t second = 0x5A5AA5A5u) {
    return [=] {
        Rng r(seed);
        uint8_t* p = MemoryInline::LeanHost((address & ~63u) - 256u);
        for (int i = 0; i < 576; ++i) p[i] = static_cast<uint8_t>(r.Next());
        const uint32_t be[2] = {__builtin_bswap32(word), __builtin_bswap32(second)};
        std::memcpy(MemoryInline::LeanHost(address), be, 8);
        uint8_t* s = MemoryInline::LeanHost(kScratch - 64);
        for (int i = 0; i < 192; ++i) s[i] = 0x33;
    };
}

// --- the addresses ------------------------------------------------------------------------------------------------------
struct Place {
    const char* name;
    uint32_t base;
    bool flagged;
};
const Place kPlaces[] = {
    {"mem1", 0x80300000u, false},
    {"mem1-chunk-edge", 0x80310000u, false},          // a store crossing it moves from one unflagged chunk to the next
    {"mem2", 0x90100000u, false},
    {"mem1-uncached", 0xC0320000u, false},
    {"mem1-physical", 0x00330000u, false},
    {"mem1-mirror", 0x86300000u, false},
    {"locked-cache", 0xE0001000u, false},
    {"code-chunk", kFlaggedChunk + 0x1000u, true},    // inside the registered code chunk, its pages without code
    {"code-chunk-start", kFlaggedChunk, true},
    {"code-chunk-end", kFlaggedChunk + 0x10000u, true},  // below it: unflagged, above: flagged
};
// Offsets from a place's base: a block's start, middle and end, the straddles, and a chunk's edge for every width.
const int kDeltas[] = {-132, -129, -64, -9, -8, -4, -3, -2, -1, 0, 1, 2, 3, 4, 7, 8, 28, 29, 31, 32, 33, 56, 57, 58, 59, 60,
                       61, 62, 63, 64, 65, 120, 121, 123, 124, 125, 127, 128};

// --- stores ---------------------------------------------------------------------------------------------------------------
struct StoreKind {
    const char* name;
    uint32_t (*encode)(uint32_t value, uint32_t base);
    bool update;
    bool isFloat, isPsq;
    uint32_t width;  // bytes the program writes at most
};
const StoreKind kStores[] = {
    {"stw", [](uint32_t d, uint32_t a) { return D(36, d, a, kDisp); }, false, false, false, 4},
    {"stwu", [](uint32_t d, uint32_t a) { return D(37, d, a, kDisp); }, true, false, false, 4},
    {"stb", [](uint32_t d, uint32_t a) { return D(38, d, a, kDisp); }, false, false, false, 1},
    {"stbu", [](uint32_t d, uint32_t a) { return D(39, d, a, kDisp); }, true, false, false, 1},
    {"sth", [](uint32_t d, uint32_t a) { return D(44, d, a, kDisp); }, false, false, false, 2},
    {"sthu", [](uint32_t d, uint32_t a) { return D(45, d, a, kDisp); }, true, false, false, 2},
    {"stwx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 151); }, false, false, false, 4},
    {"stbx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 215); }, false, false, false, 1},
    {"sthx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 407); }, false, false, false, 2},
    {"stfs", [](uint32_t d, uint32_t a) { return D(52, d, a, kDisp); }, false, true, false, 4},
    {"stfsu", [](uint32_t d, uint32_t a) { return D(53, d, a, kDisp); }, true, true, false, 4},
    {"stfsx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 663); }, false, true, false, 4},
    {"stfd", [](uint32_t d, uint32_t a) { return D(54, d, a, kDisp); }, false, true, false, 8},
    {"stfdu", [](uint32_t d, uint32_t a) { return D(55, d, a, kDisp); }, true, true, false, 8},
    {"psq_st w0", [](uint32_t d, uint32_t a) { return Psq(60, d, a, 0, 0, kDisp); }, false, true, true, 8},
    {"psq_stu w0", [](uint32_t d, uint32_t a) { return Psq(61, d, a, 0, 0, kDisp); }, true, true, true, 8},
    {"psq_st w1", [](uint32_t d, uint32_t a) { return Psq(60, d, a, 1, 0, kDisp); }, false, true, true, 4},
    {"psq_stu w1", [](uint32_t d, uint32_t a) { return Psq(61, d, a, 1, 0, kDisp); }, true, true, true, 4},
    {"psq_st q1 w0", [](uint32_t d, uint32_t a) { return Psq(60, d, a, 0, 1, kDisp); }, false, true, true, 8},
    // the forms Decode leaves Generic, which jit_x64_generic.cpp and the memory group give native code
    {"stwux", [](uint32_t d, uint32_t a) { return X(d, a, 4, 183); }, true, false, false, 4},
    {"stbux", [](uint32_t d, uint32_t a) { return X(d, a, 4, 247); }, true, false, false, 1},
    {"sthux", [](uint32_t d, uint32_t a) { return X(d, a, 4, 439); }, true, false, false, 2},
    {"stwbrx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 662); }, false, false, false, 4},
    {"sthbrx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 918); }, false, false, false, 2},
    {"stwcx.", [](uint32_t d, uint32_t a) { return X(d, a, 4, 150) | 1u; }, false, false, false, 4},
    {"stwx ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 151); }, false, false, false, 4},
    {"stbx ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 215); }, false, false, false, 1},
    {"sthx ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 407); }, false, false, false, 2},
    {"stwbrx ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 662); }, false, false, false, 4},
    {"stfsux", [](uint32_t d, uint32_t a) { return X(d, a, 4, 695); }, true, true, false, 4},
    {"stfdx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 727); }, false, true, false, 8},
    {"stfdux", [](uint32_t d, uint32_t a) { return X(d, a, 4, 759); }, true, true, false, 8},
    {"stfsx ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 663); }, false, true, false, 4},
    {"stfdx ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 727); }, false, true, false, 8},
    {"stfiwx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 983); }, false, true, false, 4},
};

// A store of every kind to every place and delta, with the register it stores in its forms.
void IntegerStores() {
    for (const StoreKind& k : kStores) {
        if (k.isFloat) continue;
        for (const Place& p : kPlaces) {
            for (int delta : kDeltas) {
                const uint32_t ea = p.base + static_cast<uint32_t>(delta);
                Case c;
                c.name = std::string(k.name) + " " + p.name + " " + std::to_string(delta);
                c.init = BaseContext(1000 + delta);
                c.init.gpr[3] = ea - kDisp;
                c.words = {k.encode(5, 3), kBlr};
                c.windows = {ea};
                c.prepare = Fill(ea, 77, 0xA1B2C3D4u);
                Run(c);
            }
        }
    }
}

void MultipleStores() {
    for (uint32_t first : {31u, 30u, 29u, 28u, 20u, 10u, 5u, 1u, 0u}) {
        for (const Place& p : kPlaces) {
            for (int delta : kDeltas) {
                const uint32_t ea = p.base + static_cast<uint32_t>(delta);
                for (int store = 0; store < 2; ++store) {
                    Case c;
                    c.name = std::string(store ? "stmw r" : "lmw r") + std::to_string(first) + " " + p.name + " " +
                             std::to_string(delta);
                    c.init = BaseContext(2000 + delta);
                    c.init.gpr[3] = ea - kDisp;
                    c.words = {D(store ? 47 : 46, first, 3, kDisp), kBlr};
                    c.windows = {ea};
                    c.prepare = Fill(ea, 78, 0x01020304u);
                    Run(c);
                }
            }
        }
    }
}

// Doubles whose single narrowing takes each arm: the window 874..896 of the exponent, its edges, zeros, NaNs and
// infinities, ordinary values, and random bits.
std::vector<uint64_t> StoreValues() {
    std::vector<uint64_t> v;
    Rng r(4242);
    for (uint64_t sign : {uint64_t{0}, uint64_t{1} << 63})
        for (uint64_t exp : {0ull, 1ull, 872ull, 873ull, 874ull, 875ull, 880ull, 889ull, 895ull, 896ull, 897ull, 898ull, 1023ull,
                             1150ull, 1151ull, 1152ull, 1153ull, 2046ull, 2047ull})
            for (uint64_t frac : {0ull, 1ull, 0x8000000000000ull, 0xFFFFFFFFFFFFFull, 0x1FFFFFull, 0x1FFFFFFull, 0x123456789ABCDull,
                                  0x4000000000000ull})
                v.push_back(sign | (exp << 52) | frac);
    for (int i = 0; i < 40; ++i) v.push_back(r.Next64());
    return v;
}

// The register the store reads, made Unknown (as the context has it), Scalar (fmr) or Paired (ps_mr) in the program.
void FloatStores() {
    const std::vector<uint64_t> values = StoreValues();
    for (const StoreKind& k : kStores) {
        if (!k.isFloat) continue;
        for (int form = 0; form < 3; ++form) {
            size_t n = 0;
            for (uint64_t bits : values) {
                // every value to MEM1; the places and deltas with a few values each
                const Place& p = kPlaces[n % (sizeof kPlaces / sizeof *kPlaces)];
                const int delta = kDeltas[(n * 7) % (sizeof kDeltas / sizeof *kDeltas)];
                ++n;
                const uint32_t ea = p.base + static_cast<uint32_t>(delta);
                for (uint32_t gqr : {0u, 0x00040004u, 0x3D043D04u, 0x00070007u}) {
                    if (!k.isPsq && gqr != 0) continue;
                    for (uint32_t fpscr : {0u, 4u}) {
                        Case c;
                        c.name = std::string(k.name) + " form " + std::to_string(form) + " " + p.name + " " +
                                 std::to_string(delta) + " value " + std::to_string(bits) + " gqr " + std::to_string(gqr);
                        c.init = BaseContext(3000 + n);
                        c.init.gpr[3] = ea - kDisp;
                        c.init.fpr[5].raw = bits;
                        c.init.gqr[0] = gqr;
                        c.init.fpscr = fpscr;
                        if (form == 1) c.words.push_back(Fmr(5, 5));
                        if (form == 2) c.words.push_back(PsMr(5, 5));
                        c.words.push_back(k.encode(5, 3));
                        c.words.push_back(kBlr);
                        c.windows = {ea};
                        c.prepare = Fill(ea, 79, 0x0, 0x0);
                        Run(c);
                    }
                }
            }
        }
    }
}

// The FPR cache (src/jit/jit_x64_fpr.cpp) around the slow paths of loads and stores: FPRs written before the access and read
// after it are held dirty in xmm slots, which a slow path (the flagged chunk's store routine, stmw's per-word loop, lfs's
// conversion of a denormal or NaN word, psq_l's quieting) must store before its call and reload after it, with their forms.
void CacheAcrossSlowPaths() {
    const auto fa = [](uint32_t op, uint32_t d, uint32_t a, uint32_t b, uint32_t c, uint32_t xo) {
        return (op << 26) | (d << 21) | (a << 16) | (b << 11) | (c << 6) | (xo << 1);
    };
    const uint32_t before[] = {
        fa(59, 6, 1, 2, 0, 21),  // fadds f6, f1, f2: dirty, Scalar
        Fmr(7, 6),               // fmr f7, f6
        fa(4, 9, 1, 2, 0, 21),   // ps_add f9, f1, f2: dirty, Paired
    };
    const uint32_t after[] = {
        fa(63, 8, 6, 7, 0, 21),  // fadd f8, f6, f7
        fa(4, 10, 9, 6, 0, 21),  // ps_add f10, f9, f6 (f6 converted to a pair)
    };
    std::vector<std::pair<std::string, uint32_t>> accesses;
    for (const StoreKind& k : kStores) accesses.push_back({k.name, k.encode(5, 3)});
    accesses.push_back({"stmw r28", D(47, 28, 3, kDisp)});
    accesses.push_back({"lfs", D(48, 5, 3, kDisp)});
    accesses.push_back({"psq_l", Psq(56, 5, 3, 0, 0, kDisp)});
    size_t n = 0;
    for (const auto& acc : accesses) {
        for (const Place& p : kPlaces) {
            for (int delta : {-8, -4, 0, 60, 61}) {
                const uint32_t ea = p.base + static_cast<uint32_t>(delta);
                // a denormal and a NaN for the loads' slow paths, an ordinary pair otherwise
                for (uint32_t word : {0x3F800000u, 0x00012345u, 0x7FA00001u}) {
                    Case c;
                    c.name = "cache around " + acc.first + " " + p.name + " " + std::to_string(delta) + " word " + std::to_string(word);
                    c.init = BaseContext(5000 + n++);
                    c.init.gpr[3] = ea - kDisp;
                    c.init.fpr[1].raw = 0x3FF4000000000000ull;
                    c.init.fpr[2].raw = 0x4008000000000000ull;
                    c.init.fpr[5].raw = 0x3FD5555555555555ull;
                    for (uint32_t w : before) c.words.push_back(w);
                    c.words.push_back(acc.second);
                    for (uint32_t w : after) c.words.push_back(w);
                    c.words.push_back(kBlr);
                    c.windows = {ea};
                    c.prepare = Fill(ea, 80, word, word);
                    Run(c);
                }
            }
        }
    }
}

// --- an unmapped store: the exception, and rA not updated ---------------------------------------------------------------
void UnmappedStores() {
    for (const StoreKind& k : kStores) {
        for (uint32_t ea : {0x84000010u, 0xF0000000u, 0x00000008u + 0x30000000u, 0xCD007000u + 0x01000000u}) {
            Case c;
            c.name = std::string(k.name) + " unmapped " + std::to_string(ea);
            c.init = BaseContext(5000 + ea);
            c.init.gpr[3] = ea - kDisp;
            c.init.fpr[5].raw = 0x3FF0000000000000ull;
            c.words = {k.encode(5, 3), kBlr};
            c.windows = {};
            c.compareBytes = false;
            Run(c);
        }
    }
    for (uint32_t ea : {0x84000010u, 0xF0000000u}) {
        Case c;
        c.name = "stmw unmapped " + std::to_string(ea);
        c.init = BaseContext(5100 + ea);
        c.init.gpr[3] = ea - kDisp;
        c.words = {D(47, 29, 3, kDisp), kBlr};
        c.compareBytes = false;
        Run(c);
    }
}

// --- loads ----------------------------------------------------------------------------------------------------------------
struct LoadKind {
    const char* name;
    uint32_t (*encode)(uint32_t dest, uint32_t base);
    bool update, isFloat, isPsq, isDouble;
};
const LoadKind kLoads[] = {
    {"lwz", [](uint32_t d, uint32_t a) { return D(32, d, a, kDisp); }, false, false, false, false},
    {"lwzu", [](uint32_t d, uint32_t a) { return D(33, d, a, kDisp); }, true, false, false, false},
    {"lbz", [](uint32_t d, uint32_t a) { return D(34, d, a, kDisp); }, false, false, false, false},
    {"lbzu", [](uint32_t d, uint32_t a) { return D(35, d, a, kDisp); }, true, false, false, false},
    {"lhz", [](uint32_t d, uint32_t a) { return D(40, d, a, kDisp); }, false, false, false, false},
    {"lhzu", [](uint32_t d, uint32_t a) { return D(41, d, a, kDisp); }, true, false, false, false},
    {"lha", [](uint32_t d, uint32_t a) { return D(42, d, a, kDisp); }, false, false, false, false},
    {"lwzx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 23); }, false, false, false, false},
    {"lbzx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 87); }, false, false, false, false},
    {"lhzx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 279); }, false, false, false, false},
    {"lfs", [](uint32_t d, uint32_t a) { return D(48, d, a, kDisp); }, false, true, false, false},
    {"lfsu", [](uint32_t d, uint32_t a) { return D(49, d, a, kDisp); }, true, true, false, false},
    {"lfsx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 535); }, false, true, false, false},
    {"lfd", [](uint32_t d, uint32_t a) { return D(50, d, a, kDisp); }, false, true, false, true},
    {"lfdu", [](uint32_t d, uint32_t a) { return D(51, d, a, kDisp); }, true, true, false, true},
    {"psq_l w0", [](uint32_t d, uint32_t a) { return Psq(56, d, a, 0, 0, kDisp); }, false, true, true, true},
    {"psq_lu w0", [](uint32_t d, uint32_t a) { return Psq(57, d, a, 0, 0, kDisp); }, true, true, true, true},
    {"psq_l w1", [](uint32_t d, uint32_t a) { return Psq(56, d, a, 1, 0, kDisp); }, false, true, true, false},
    {"psq_lu w1", [](uint32_t d, uint32_t a) { return Psq(57, d, a, 1, 0, kDisp); }, true, true, true, false},
    {"psq_l q1 w0", [](uint32_t d, uint32_t a) { return Psq(56, d, a, 0, 1, kDisp); }, false, true, true, true},
    // the forms Decode leaves Generic
    {"lwzux", [](uint32_t d, uint32_t a) { return X(d, a, 4, 55); }, true, false, false, false},
    {"lbzux", [](uint32_t d, uint32_t a) { return X(d, a, 4, 119); }, true, false, false, false},
    {"lhzux", [](uint32_t d, uint32_t a) { return X(d, a, 4, 311); }, true, false, false, false},
    {"lhax", [](uint32_t d, uint32_t a) { return X(d, a, 4, 343); }, false, false, false, false},
    {"lhaux", [](uint32_t d, uint32_t a) { return X(d, a, 4, 375); }, true, false, false, false},
    {"lhau", [](uint32_t d, uint32_t a) { return D(43, d, a, kDisp); }, true, false, false, false},
    {"lwbrx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 534); }, false, false, false, false},
    {"lhbrx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 790); }, false, false, false, false},
    {"lwarx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 20); }, false, false, false, false},
    {"lwzx ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 23); }, false, false, false, false},
    {"lbzx ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 87); }, false, false, false, false},
    {"lhzx ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 279); }, false, false, false, false},
    {"lhax ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 343); }, false, false, false, false},
    {"lwbrx ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 534); }, false, false, false, false},
    {"lfsux", [](uint32_t d, uint32_t a) { return X(d, a, 4, 567); }, true, true, false, false},
    {"lfdx", [](uint32_t d, uint32_t a) { return X(d, a, 4, 599); }, false, true, false, true},
    {"lfdux", [](uint32_t d, uint32_t a) { return X(d, a, 4, 631); }, true, true, false, true},
    {"lfsx ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 535); }, false, true, false, false},
    {"lfdx ra=0", [](uint32_t d, uint32_t a) { return X(d, 0, a, 599); }, false, true, false, true},
};

void IntegerLoads() {
    for (const LoadKind& k : kLoads) {
        if (k.isFloat) continue;
        for (const Place& p : kPlaces) {
            for (int delta : kDeltas) {
                const uint32_t ea = p.base + static_cast<uint32_t>(delta);
                Case c;
                c.name = std::string(k.name) + " " + p.name + " " + std::to_string(delta);
                c.init = BaseContext(6000 + delta);
                c.init.gpr[3] = ea - kDisp;
                c.words = {k.encode(5, 3), kBlr};
                c.windows = {ea};
                c.prepare = Fill(ea, 80, 0x80FF7F01u + static_cast<uint32_t>(delta));
                Run(c);
            }
        }
    }
}

// Words and doubles of every class, planted at the address; the register is then stored as a scalar and as a pair.
void FloatLoads() {
    std::vector<uint32_t> singles;
    Rng r(777);
    for (uint32_t sign : {0u, 0x80000000u})
        for (uint32_t exp : {0u, 1u, 2u, 126u, 127u, 128u, 253u, 254u, 255u})
            for (uint32_t frac : {0u, 1u, 0x400000u, 0x7FFFFFu, 0x200000u, 0x123456u, 0x400001u})
                singles.push_back(sign | (exp << 23) | frac);
    for (int i = 0; i < 30; ++i) singles.push_back(r.Next());
    const std::vector<uint64_t> doubles = StoreValues();
    for (const LoadKind& k : kLoads) {
        if (!k.isFloat) continue;
        const size_t count = k.isDouble && !k.isPsq ? doubles.size() : singles.size();
        for (size_t i = 0; i < count; ++i) {
            const Place& p = kPlaces[i % (sizeof kPlaces / sizeof *kPlaces)];
            const int delta = kDeltas[(i * 5) % (sizeof kDeltas / sizeof *kDeltas)];
            const uint32_t ea = p.base + static_cast<uint32_t>(delta);
            uint32_t first, second;
            if (k.isDouble && !k.isPsq) {
                first = static_cast<uint32_t>(doubles[i] >> 32);
                second = static_cast<uint32_t>(doubles[i]);
            } else {
                first = singles[i];
                second = singles[(i * 7 + 3) % singles.size()];
            }
            for (uint32_t gqr : {0u, 0x3F000000u, 0x00040004u, 0x00050005u, 0x00060006u, 0x00070007u, 0x3D043D04u}) {
                if (!k.isPsq && gqr != 0) continue;
                for (uint32_t fpscr : {0u, 4u}) {
                    Case c;
                    c.name = std::string(k.name) + " " + p.name + " " + std::to_string(delta) + " words " + std::to_string(first) +
                             "," + std::to_string(second) + " gqr " + std::to_string(gqr);
                    c.init = BaseContext(7000 + i);
                    c.init.gpr[3] = ea - kDisp;
                    c.init.gqr[0] = gqr;
                    c.init.fpscr = fpscr;
                    c.words = {k.encode(5, 3), D(52, 5, 6, 0) /* stfs f5,0(r6) */, D(54, 5, 6, 16) /* stfd f5,16(r6) */,
                               Psq(60, 5, 6, 0, 7, 32) /* psq_st f5,32(r6),0,7: GQR7 = 0, float */, kBlr};
                    c.init.gqr[7] = 0;
                    c.windows = {ea, kScratch};
                    c.prepare = Fill(ea, 81, first, second);
                    Run(c);
                }
            }
        }
    }
}

// A load or store in a block with others: a stack frame saved and restored (stwu, stmw, lmw), as a function does.
void Frames() {
    for (uint32_t first : {31u, 27u, 14u}) {
        for (const Place& p : kPlaces) {
            Case c;
            c.name = std::string("frame ") + p.name;
            const uint32_t sp = p.base + 0x200;
            c.init = BaseContext(8000 + first);
            c.init.gpr[1] = sp;
            const uint32_t n = 32 - first;
            c.words = {D(37, 1, 1, -static_cast<int32_t>(16 + 4 * n)),  // stwu r1,-N(r1)
                       0x7C0802A6u,                                       // mflr r0
                       D(47, first, 1, 8),                                // stmw
                       D(36, 0, 1, static_cast<int32_t>(16 + 4 * n + 4)),  // stw r0, N+4(r1)
                       D(32, 0, 1, static_cast<int32_t>(16 + 4 * n + 4)),  // lwz r0
                       D(46, first, 1, 8),                                // lmw
                       D(32, 1, 1, 0),                                    // lwz r1,0(r1)
                       0x7C0803A6u,                                       // mtlr r0
                       kBlr};
            c.windows = {sp, sp - 16 - 4 * n};
            c.exactMarks = false;
            c.prepare = Fill(sp - 64, 82, 0x12121212u);
            Run(c);
        }
    }
}

// --- store groups: stores off one base value in one block (jit_x64_mem.cpp) -------------------------------------------------
// Each program stores r5..r9 (and r20 for a read back) off r3 = the place's base + delta, so its group's bytes start in
// every position of a block and a chunk, reach the next block or two, the next chunk, the flagged chunk from either side
// and its guarded page (the leader's slow path: on with the members, or leave), with update forms and the base moved by
// addi, addis and mr between the stores.
constexpr uint32_t kAddiR4R3_16 = 0x38830010u;  // addi r4, r3, 16
constexpr uint32_t kMrR6R3 = 0x7C661B78u;        // mr r6, r3
constexpr uint32_t kAddisR3R3_0 = 0x3C630000u;   // addis r3, r3, 0 (a new value as far as the analysis knows)
constexpr uint32_t kMflrR0 = 0x7C0802A6u;
struct GroupProgram {
    const char* name;
    std::vector<uint32_t> words;
};
const std::vector<GroupProgram>& GroupPrograms() {
    static const std::vector<GroupProgram> programs = {
        {"pair", {D(36, 5, 3, 0), D(36, 6, 3, 4), kBlr}},
        {"down", {D(36, 5, 3, 60), D(44, 6, 3, 2), D(38, 7, 3, -1), kBlr}},
        {"two blocks", {D(36, 5, 3, 0), D(36, 6, 3, 61), D(38, 7, 3, -1), kBlr}},
        {"three blocks", {D(36, 5, 3, -4), D(36, 6, 3, 121), D(44, 7, 3, 60), D(38, 8, 3, 30), kBlr}},
        {"too far", {D(36, 5, 3, 0), D(36, 6, 3, 200), D(36, 7, 3, 4), D(36, 8, 3, 196), kBlr}},
        {"update", {D(37, 5, 3, -16), D(36, 6, 3, 20), D(45, 7, 3, 8), D(36, 8, 3, -6), kBlr}},
        {"update byte", {D(39, 5, 3, 1), D(39, 6, 3, 1), D(39, 7, 3, 1), D(38, 8, 3, -3), kBlr}},
        {"addi mr", {kAddiR4R3_16, D(36, 5, 3, 0), D(36, 6, 4, 4), kMrR6R3, D(38, 7, 6, 30), D(44, 8, 4, -2), kBlr}},
        {"new base", {D(36, 5, 3, 0), kAddisR3R3_0, D(36, 6, 3, 4), D(36, 7, 3, 8), kBlr}},
        {"loads between", {D(36, 5, 3, 0), D(32, 20, 3, 4), D(36, 20, 3, 8), D(32, 21, 3, 0), D(36, 9, 3, 12), kBlr}},
        {"same word", {D(36, 5, 3, 8), D(38, 6, 3, 9), D(44, 7, 3, 10), D(32, 20, 3, 8), kBlr}},
        {"frame", {D(37, 3, 3, -32), kMflrR0, D(36, 0, 3, 36), D(36, 31, 3, 28), D(36, 30, 3, 24), D(36, 9, 3, 8), kBlr}},
        {"far apart", {D(36, 5, 3, -32768), D(36, 6, 3, 32764), D(36, 7, 3, -32764), kBlr}},
        {"update alone", {D(37, 5, 3, 8), 0x38E30004u /* addi r7, r3, 4: r3 read after the update */, D(45, 6, 7, 1000), kBlr}},
    };
    return programs;
}

void StoreGroups() {
    for (const GroupProgram& g : GroupPrograms()) {
        for (const Place& p : kPlaces) {
            const bool far = std::string(g.name) == "far apart";
            if (far && p.base >> 28 == 0xEu) continue;  // the locked cache is 16 KiB: nothing backs +-32 KiB around it
            for (int delta : kDeltas) {
                const uint32_t ea = p.base + static_cast<uint32_t>(delta);
                Case c;
                c.name = std::string("group ") + g.name + " " + p.name + " " + std::to_string(delta);
                c.init = BaseContext(9000 + delta);
                c.init.gpr[3] = ea;
                c.words = g.words;
                c.windows = {ea, ea + 128};
                if (far) c.windows = {ea - 32768, ea + 32764, ea - 32764};
                c.exactMarks = false;
                c.prepare = [ea, w = c.windows] {
                    for (uint32_t at : w) Fill(at, 83 + at, 0x5A5A5A5Au)();
                };
                Run(c);
            }
        }
    }
    // Groups whose bytes straddle a view's end (0x7FFFFFFF / 0x80000000, the 32-bit wrap) or MEM2's: the leader stores and
    // the block leaves, the member's own store raises the exception, as in the interpreter.
    struct Edge {
        const char* name;
        uint32_t base;
        std::vector<uint32_t> windows;
    };
    const Edge edges[] = {
        {"view start", 0x80000000u, {0x80000200u}},
        {"wrap", 0x00000000u, {}},
        {"mem2 end", 0x94000000u, {0x93FFFEC0u}},
    };
    for (const Edge& edge : edges) {
        for (int delta : {-16, -12, -8, -4, -1, 0, 4}) {
            for (int order = 0; order < 2; ++order) {
                const uint32_t ea = edge.base + static_cast<uint32_t>(delta);
                Case c;
                c.name = std::string("group edge ") + edge.name + " " + std::to_string(delta) + " order " + std::to_string(order);
                c.init = BaseContext(9500 + delta);
                c.init.gpr[3] = ea;
                if (order == 0) c.words = {D(36, 5, 3, 8), D(36, 6, 3, 12), D(32, 20, 3, 8), D(36, 7, 3, 0), D(36, 8, 3, 4), kBlr};
                else c.words = {D(36, 7, 3, 0), D(36, 8, 3, 4), D(36, 5, 3, 8), D(36, 6, 3, 12), D(32, 20, 3, 8), kBlr};
                c.windows = edge.windows;
                c.exactMarks = false;
                if (!c.windows.empty()) c.prepare = Fill(c.windows[0], 84, 0x5A5A5A5Au);
                Run(c);
            }
        }
    }
}

// Addresses whose sum passes 2^32 (base + displacement, rA + rB): the 32-bit address wraps to MEM1's physical view at
// 0x100. The JIT's unmasked loads reach it through the aliased view's tail or compute the sum in 32 bits
// (lean_window.h, g_aliased); stores off such a base go through a group (the members' unmasked stores).
void WrappingAddresses() {
    const uint32_t target = 0x100u;
    for (const LoadKind& k : kLoads) {
        if (k.isFloat) continue;
        for (uint32_t delta : {0u, 1u, 2u, 4u, 0x3Cu, 0x7F00u}) {
            const uint32_t ea = target + delta;
            for (int big = 0; big < 2; ++big) {
                Case c;
                c.name = std::string(k.name) + " wrapping " + std::to_string(delta) + (big ? " big" : "");
                c.init = BaseContext(6500 + delta);
                const uint32_t w = k.encode(5, 3);
                const bool dform = (w >> 26) != 31;
                if (dform) {  // base + displacement
                    c.words = {big ? (w & 0xFFFF0000u) | 0x7FF0u : w, kBlr};
                    c.init.gpr[3] = ea - (big ? 0x7FF0u : static_cast<uint32_t>(kDisp));
                } else {  // rA + rB with rB = -0x1000 (as unsigned), or rA = 0 (rB alone)
                    c.words = {w, kBlr};
                    c.init.gpr[3] = ea + 0x1000u;
                    c.init.gpr[4] = 0xFFFFF000u;
                    if (((w >> 16) & 31) == 0) c.init.gpr[3] = ea;
                }
                c.windows = {ea + 0x100u};
                c.prepare = Fill(ea, 85, 0x89ABCDEFu + delta);
                Run(c);
            }
        }
    }
    for (const GroupProgram& g : GroupPrograms()) {
        if (std::string(g.name) == "far apart") continue;
        for (uint32_t delta : {0u, 0x40u, 0x80u}) {
            Case c;
            c.name = std::string("group ") + g.name + " wrapping " + std::to_string(delta);
            c.init = BaseContext(9700 + delta);
            c.init.gpr[3] = 0xFFFFFF00u + delta;  // with the displacements moved up by 0x200, r3 + d passes 2^32
            c.words = g.words;
            for (uint32_t& word : c.words) {
                const uint32_t po = word >> 26;
                if ((po == 36 || po == 37 || po == 38 || po == 39 || po == 44 || po == 45 || po == 32) && ((word >> 16) & 31) == 3)
                    word = (word & 0xFFFF0000u) | ((word + 0x200u) & 0xFFFFu);
            }
            c.windows = {target + 0x100u, target + 0x200u};
            c.exactMarks = false;
            c.prepare = [] {
                Fill(0x200u, 86, 0x5A5A5A5Au)();
                Fill(0x300u, 87, 0x5A5A5A5Au)();
            };
            Run(c);
        }
    }
}

void SetGuardPage(uint32_t page, bool on) {
    RecompMod::g_executableWriteGuardCoarsePages[page >> RecompMod::kExecutableWriteGuardCoarsePageShift].store(on);
    RecompMod::g_executableWriteGuardMidPages[page >> RecompMod::kExecutableWriteGuardMidPageShift].store(on);
    RecompMod::g_executableWriteGuardPages[page >> RecompMod::kExecutableWriteGuardPageShift].store(on);
    Memory::RefreshWritableFastPathsForExecutableRanges();
}

void SetCodeChunk(bool on) {
    RecompMod::g_executableWriteGuardEnabled.store(on);
    SetGuardPage(kFlaggedChunk + 0x800u, on);  // a 4 KiB guard page inside the chunk, at its start
}

// --- stores that fault (the store view, lean_window.h; jit_x64_mem.cpp) ----------------------------------------------------
// A store's site faults where the policy flags its chunk, then is a jump to its slow entry. Loops run a block's sites again
// after they were patched, into the flagged chunk from below and out of it from inside; the policy of a chunk changes under a
// compiled block (the store view follows the policy, memory.cpp), which then goes on storing as the interpreter does.
constexpr uint32_t kToggleChunk = 0x80600000u;

// Both engines from the same state, the program at `entry` already written (the JIT's block, compiled by an earlier run, stays).
void RunAt(const Case& c, uint32_t entry) {
    const Snap classic = RunOne(Engine::Classic, c, entry);
    const Snap jit = RunOne(Engine::Jit, c, entry);
    ++g_cases;
    g_threw += classic.threw;
    g_observedCodeWrite += classic.lastExecutableWrite != 0;
    const std::string diff = Diff(classic, jit, c.exactMarks);
    if (!diff.empty() && ++g_failures <= 12) std::printf("FAIL %s\n%s", c.name.c_str(), diff.c_str());
}

void FaultingStores() {
    const auto mtctr = [](uint32_t r) { return 0x7C0903A6u | (r << 21); };
    const auto bdnz = [](int32_t back) { return 0x42000000u | (static_cast<uint32_t>(back) & 0xFFFCu); };
    // stwu / stbu / stfsu / psq_stu in a loop of `n` stores 4 (1) bytes apart, from below the flagged chunk into it, and from
    // inside it out
    struct LoopStore {
        const char* name;
        uint32_t word;  // the store, rS / frS 5, rA 3
        uint32_t step;
    };
    const LoopStore stores[] = {
        {"stwu", D(37, 5, 3, 4), 4}, {"stbu", D(39, 5, 3, 1), 1}, {"sthu", D(45, 5, 3, 2), 2},
        {"stfsu", D(53, 5, 3, 4), 4}, {"stfdu", D(55, 5, 3, 8), 8}, {"psq_stu w0", Psq(61, 5, 3, 0, 0, 8), 8},
    };
    for (const LoopStore& s : stores) {
        for (const uint32_t start : {kFlaggedChunk - 3 * s.step, kFlaggedChunk + 0x10000u - 3 * s.step, kFlaggedChunk + 0x400u}) {
            for (const uint32_t n : {2u, 6u}) {
                Case c;
                c.name = std::string("loop ") + s.name + " from " + std::to_string(start) + " x" + std::to_string(n);
                c.init = BaseContext(12000 + start + n);
                c.init.xer &= ~0x80000000u;  // compiled code runs only while XER[SO] is clear
                c.init.gpr[3] = start - s.step;
                c.init.gpr[7] = n;
                c.init.fpr[5].raw = 0x3FF8000000000000ull;
                // li r7 is in the context; mtctr r7; loop: store; addi r5, r5, 1; bdnz loop; blr
                c.words = {mtctr(7), s.word, 0x38A50001u, bdnz(-8), kBlr};
                c.windows = {start, start + n * s.step};
                c.exactMarks = false;
                c.prepare = [w = c.windows] {
                    for (uint32_t at : w) Fill(at, 88 + at, 0x5A5A5A5Au)();
                };
                Run(c);
            }
        }
    }
    // A compiled block's store into a chunk whose policy changes between its runs: clear (it stores), flagged (it faults and
    // takes the checked path; its site becomes a jump), clear (the slow entry stores through the window), flagged again.
    struct Kind {
        const char* name;
        uint32_t word;
    };
    const Kind kinds[] = {
        {"stw", D(36, 5, 3, kDisp)}, {"sthu", D(45, 5, 3, kDisp)}, {"stbx", X(5, 3, 4, 215)}, {"stfs", D(52, 5, 3, kDisp)},
        {"stfdu", D(55, 5, 3, kDisp)}, {"psq_st w0", Psq(60, 5, 3, 0, 0, kDisp)}, {"psq_st w1", Psq(60, 5, 3, 1, 0, kDisp)},
        {"group", D(36, 5, 3, kDisp)},  // with a member: stw r6, kDisp + 4(r3)
    };
    RecompMod::g_executableWriteGuardEnabled.store(true);
    for (const Kind& k : kinds) {
        for (const int delta : {0, 60, 0x1000}) {
            const bool group = std::string(k.name) == "group";
            const uint32_t ea = kToggleChunk + 0x2000u + static_cast<uint32_t>(delta);
            if (g_codeNext + 32 >= kCodeLimit) g_codeNext = kCodeBase;
            const uint32_t entry = g_codeNext;
            g_codeNext += 64;
            std::vector<uint32_t> words = {k.word};
            if (group) words.push_back(D(36, 6, 3, kDisp + 4));
            words.push_back(kBlr);
            for (size_t i = 0; i < words.size(); ++i) Memory::Write32(entry + 4u * static_cast<uint32_t>(i), words[i]);
            uint64_t compiled = 0, faults = 0;
            for (int phase = 0; phase < 4; ++phase) {
                SetGuardPage(kToggleChunk + 0x8000u, (phase & 1) != 0);
                if (phase == 1) {  // the block of phase 0 runs on, and its site faults now (with the store view)
                    compiled = PpcInterpreter::detail::JitX64GetStats().compiled;
                    faults = PpcInterpreter::detail::JitX64GetFaultStats().faults;
                }
                Case c;
                c.name = std::string("policy change ") + k.name + " " + std::to_string(delta) + " phase " + std::to_string(phase);
                c.init = BaseContext(13000 + static_cast<uint64_t>(delta));
                c.init.xer &= ~0x80000000u;  // compiled code runs only while XER[SO] is clear
                c.init.gpr[3] = ea - kDisp;
                c.init.fpr[5].raw = 0x400C000000000000ull;
                c.words = words;
                c.windows = {ea};
                c.exactMarks = false;
                c.prepare = Fill(ea, 90 + phase, 0x0, 0x0);
                RunAt(c, entry);
            }
            const bool recompiled = PpcInterpreter::detail::JitX64GetStats().compiled != compiled;
            const bool faulted = PpcInterpreter::detail::JitX64GetFaultStats().faults > faults;
            if (std::getenv("BRAWL_JIT_FALLBACK") == nullptr && (recompiled || faulted != LeanWindow::g_storeView) &&
                ++g_failures <= 12)
                std::printf("FAIL policy change %s %d: recompiled %d, faulted %d\n", k.name, delta, recompiled, faulted);
        }
    }
    SetGuardPage(kToggleChunk + 0x8000u, false);
}

}  // namespace

int main() {
    Memory::Init(Memory::Config::WiiDefaults());
    TranslatedFunctionRegistry::Finalize();
    if (!PpcInterpreter::JitAvailable()) {
        std::printf("jit mem tests: no JIT backend in this build\n");
        return 0;
    }
    SetCodeChunk(true);
    if (!MemoryInline::LeanStoreNeedsPolicy(kFlaggedChunk + 0x1000u) || MemoryInline::LeanStoreNeedsPolicy(0x80300000u)) {
        std::printf("jit mem tests: the code chunk is not flagged as expected\n");
        return 2;
    }
    const auto stage = [](const char* name, void (*fn)()) {
        const uint64_t before = g_cases;
        const int failed = g_failures;
        fn();
        std::printf("  %-28s %7llu cases%s\n", name, static_cast<unsigned long long>(g_cases - before),
                    g_failures != failed ? "  FAILED" : "");
    };
    stage("integer stores", IntegerStores);
    stage("stmw and lmw", MultipleStores);
    stage("float stores", FloatStores);
    stage("fpr cache around slow paths", CacheAcrossSlowPaths);
    stage("unmapped stores", UnmappedStores);
    stage("integer loads", IntegerLoads);
    stage("float and quantized loads", FloatLoads);
    stage("frames", Frames);
    stage("store groups", StoreGroups);
    stage("wrapping addresses", WrappingAddresses);
    stage("faulting stores", FaultingStores);
    PpcInterpreter::ClearEngineOverride();
    SetCodeChunk(false);
    std::printf("  reached: %llu cases ended in an exception, %llu with a store into guarded code, %llu with dirty marks set\n",
                static_cast<unsigned long long>(g_threw), static_cast<unsigned long long>(g_observedCodeWrite),
                static_cast<unsigned long long>(g_marked));
    if (g_threw == 0 || g_observedCodeWrite == 0 || g_marked == 0) {
        std::printf("jit mem tests: a path the cases are for was never reached\n");
        ++g_failures;
    }
    const PpcInterpreter::detail::JitX64StoreGroupStats groups = PpcInterpreter::detail::JitX64GetStoreGroupStats();
    std::printf("  store groups: %llu leaders, %llu members; at run time %llu back from the range test, %llu slow paths "
                "going on, %llu leaving; %llu cases with fewer marks than the interpreter\n",
                static_cast<unsigned long long>(groups.leaders), static_cast<unsigned long long>(groups.members),
                static_cast<unsigned long long>(groups.rangeBack), static_cast<unsigned long long>(groups.slowGoOn),
                static_cast<unsigned long long>(groups.slowLeave), static_cast<unsigned long long>(g_groupedMarks));
    // With the store view a leader tests no range (its members fault for themselves): no range test comes back.
    const bool storeView = LeanWindow::g_storeView;
    if (std::getenv("BRAWL_JIT_FALLBACK") == nullptr &&
        (groups.leaders == 0 || groups.members == 0 || (groups.rangeBack == 0) != storeView || groups.slowGoOn == 0 ||
         groups.slowLeave == 0 || g_groupedMarks == 0)) {
        std::printf("jit mem tests: a store group path was never reached\n");
        ++g_failures;
    }
    const PpcInterpreter::detail::JitX64FaultStats faults = PpcInterpreter::detail::JitX64GetFaultStats();
    std::printf("  fault stores %s: %llu sites, %llu faults handled, %llu at a site already patched\n", storeView ? "on" : "off",
                static_cast<unsigned long long>(faults.sites), static_cast<unsigned long long>(faults.faults),
                static_cast<unsigned long long>(faults.repeated));
    if (std::getenv("BRAWL_JIT_FALLBACK") == nullptr &&
        (storeView ? faults.sites == 0 || faults.faults == 0 || faults.repeated != 0 : faults.sites != 0)) {
        std::printf("jit mem tests: the stores' faults are not what the store view says\n");
        ++g_failures;
    }
    // The Generic forms in the tables (indexed update, byte-reversed, algebraic, rA = 0, float indexed, stwcx., stfiwx) must have
    // been compiled natively, none as a call to the interpreter's handler (unless the environment turns the native code off).
    if (std::getenv("BRAWL_JIT_FALLBACK") == nullptr && std::getenv("BRAWL_JIT_GENERIC") == nullptr) {
        const uint64_t native = PpcInterpreter::detail::JitX64KindOps(PpcInterpreter::detail::K::Generic, true);
        const uint64_t fallback = PpcInterpreter::detail::JitX64KindOps(PpcInterpreter::detail::K::Generic, false);
        std::printf("  generic loads and stores: %llu compiled natively, %llu as the interpreter's handler\n",
                    static_cast<unsigned long long>(native), static_cast<unsigned long long>(fallback));
        if (native < 300 || fallback != 0) {
            std::printf("jit mem tests: the Generic loads and stores were not all compiled natively\n");
            ++g_failures;
        }
    }
    std::printf("jit mem tests: %llu cases, %d failures\n", static_cast<unsigned long long>(g_cases), g_failures);
    return g_failures == 0 ? 0 : 1;
}
