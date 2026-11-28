// The run-time compiler's floating point group (src/jit/jit_x64_fp.cpp) against the classic interpreter, for every op kind of
// the group, over the inputs that tell implementations apart.
//
// For each kind a short guest function (forms prepared for the sources, the op, blr) runs under the classic interpreter and
// under the JIT (threshold 0: compiled before its first run) from the same register state, and every GPR, every FPR's raw
// bits, the CR field values (bytewise: fcmpu stores one of four exact constants, and the other fields must be left as they were), lr, ctr, xer and fpscr must come out equal. The state's sources hold edge values: zeros, denormals, the
// smallest and largest normals, values that need single-precision rounding (ties and near ties, both directions), the single
// range's ends, infinities, quiet and signalling NaNs with payloads and both signs; for paired kinds each lane has its own.
// The same is run for
//
//  * FPSCR[NI] clear and set,
//  * each source arriving as Scalar (as every register is when a function is entered) and as Paired (stfd, then psq_l of
//    the same bits; Form::Unknown cannot be produced by Run, which starts every register as Scalar), and every source as a
//    single made in the same block (frsp r, r: the block knows it is a single-precision value, BlockEmitter::KnownSingle, so
//    fmuls's multiplier is not rounded, fadds's and fsubs's sums are not range-checked, frsp narrows without a check),
//  * the destination being each source, and two sources being the same register,
//  * the full cross product of the first two sources' tables with the native forms (Scalar for the scalar kinds, Paired for the
//    paired ones), a reduced one for the other forms and the third source, and seeded random bit patterns biased to special
//    exponents, with random forms.
//
// The code the JIT makes is counted (BRAWL_JIT_FP_COUNTERS: the test sets it): every kind must have run natively, each
// conversion of a source from the other form and each path to the interpreter (a NaN result, a single result in the range the
// flush rule decides, an exact tie, a multiplier that is a NaN or a denormal) must have happened, as the kind's code has them.
// BRAWL_JIT_FP_HELPERS=1 makes every arithmetic kind call its helper instead, and the same test must pass.
//
// FP_TEST_RANDOM=<n>: random states per kind (default 1000000); FP_TEST_SEED; FP_TEST_FILTER=<name>: only that kind.
#include "jit_test_env.h"
#include "abi_bridge.h"
#include "jit_test_cr.h"
#include "isa/ppc_isa_fpenv.h"
#include "memory.h"
#include "ppc_interpreter.h"
#include "../src/ppc_interpreter_internal.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace PpcInterpreter::detail::x64 {
uint64_t* JitFpCounters();  // [256][12], src/jit/jit_x64_fp.cpp
}

namespace {

using PpcInterpreter::Engine;
using PpcInterpreter::detail::K;

constexpr uint32_t kDataBase = 0x80030000u;
constexpr uint32_t kProbe = kDataBase + 0x1000u;
constexpr uint32_t kReturn = 0x80020000u;
constexpr uint32_t kCodeBase = 0x80100000u;
constexpr uint32_t kCodeStep = 0x40u;
constexpr uint32_t kCodeLimit = 0x80200000u - kCodeStep;

std::string Fmt(const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof buffer, format, args);
    va_end(args);
    return buffer;
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint32_t Next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return static_cast<uint32_t>((s * 0x2545F4914F6CDD1Dull) >> 32);
    }
    uint32_t Below(uint32_t n) { return Next() % n; }
    uint64_t Next64() { return (static_cast<uint64_t>(Next()) << 32) | Next(); }
};

// --- encoders ----------------------------------------------------------------------------------------------------------
constexpr uint32_t Fa(uint32_t op, uint32_t d, uint32_t a, uint32_t b, uint32_t c, uint32_t xo5) {
    return (op << 26) | (d << 21) | (a << 16) | (b << 11) | (c << 6) | (xo5 << 1);
}
constexpr uint32_t Fx(uint32_t op, uint32_t d, uint32_t a, uint32_t b, uint32_t xo10) {
    return (op << 26) | (d << 21) | (a << 16) | (b << 11) | (xo10 << 1);
}
constexpr uint32_t Fcmpu(uint32_t crf, uint32_t a, uint32_t b) { return (63u << 26) | (crf << 23) | (a << 16) | (b << 11); }
constexpr uint32_t Stfd(uint32_t fs, int32_t d, uint32_t ra) { return (54u << 26) | (fs << 21) | (ra << 16) | (static_cast<uint32_t>(d) & 0xFFFFu); }
// psq_l frD, 0(rA), W=0, I=0 (GQR0 is 0: two floats)
constexpr uint32_t PsqL(uint32_t fd, int32_t d, uint32_t ra) { return (56u << 26) | (fd << 21) | (ra << 16) | (static_cast<uint32_t>(d) & 0xFFFu); }
constexpr uint32_t Blr() { return (19u << 26) | (20u << 21) | (16u << 1); }

// --- the kinds ---------------------------------------------------------------------------------------------------------
// Shape: which instruction fields the op reads (a, b, c) and how it is encoded. The sources are numbered in the order the
// differential test and the JIT's helpers take them: the registers of a, then b, then c as the shape lists.
enum class Shape { AB, AC, ABC, B10, AB10, Cmp };
struct Kind {
    const char* name;
    K kind;
    bool paired;  // the op takes its sources as pairs
    uint32_t op, xo;
    Shape shape;
    unsigned nsrc;           // sources the op reads
    unsigned convSlots;      // operand conversions its code has (counter slots 2 ...)
    bool inlineArith;        // has inline arithmetic with paths to the interpreter (unless the build lacks FMA or helpers are forced)
    bool needsFma;
    bool reasonRange, reasonTie, reasonMultiplier;
};
// clang-format off
const Kind kKinds[] = {
    {"fadds",   K::Fadds,   false, 59, 21, Shape::AB,  2, 2, true,  false, true,  false, false},
    {"fsubs",   K::Fsubs,   false, 59, 20, Shape::AB,  2, 2, true,  false, true,  false, false},
    {"fmuls",   K::Fmuls,   false, 59, 25, Shape::AC,  2, 2, true,  false, true,  false, true },
    {"fdivs",   K::Fdivs,   false, 59, 18, Shape::AB,  2, 2, true,  false, true,  false, false},
    {"fmadds",  K::Fmadds,  false, 59, 29, Shape::ABC, 3, 3, true,  true,  true,  true,  true },
    {"fmsubs",  K::Fmsubs,  false, 59, 28, Shape::ABC, 3, 3, true,  true,  true,  true,  true },
    {"fnmsubs", K::Fnmsubs, false, 59, 30, Shape::ABC, 3, 3, true,  true,  true,  true,  true },
    {"fnmadds", K::Fnmadds, false, 59, 31, Shape::ABC, 3, 3, true,  true,  true,  true,  true },
    {"fadd",    K::Fadd,    false, 63, 21, Shape::AB,  2, 2, true,  false, false, false, false},
    {"fsub",    K::Fsub,    false, 63, 20, Shape::AB,  2, 2, true,  false, false, false, false},
    {"fmul",    K::Fmul,    false, 63, 25, Shape::AC,  2, 2, true,  false, false, false, false},
    {"fdiv",    K::Fdiv,    false, 63, 18, Shape::AB,  2, 2, true,  false, false, false, false},
    {"fmadd",   K::Fmadd,   false, 63, 29, Shape::ABC, 3, 3, true,  true,  false, false, false},
    {"fmsub",   K::Fmsub,   false, 63, 28, Shape::ABC, 3, 3, true,  true,  false, false, false},
    {"fneg",    K::Fneg,    false, 63, 40, Shape::B10, 1, 1, false, false, false, false, false},
    {"fabs",    K::Fabs,    false, 63, 264, Shape::B10, 1, 1, false, false, false, false, false},
    {"fmr",     K::Fmr,     false, 63, 72, Shape::B10, 1, 0, false, false, false, false, false},
    {"frsp",    K::Frsp,    false, 63, 12, Shape::B10, 1, 1, true,  false, true,  false, false},
    {"fcmpu",   K::Fcmpu,   false, 63, 0,  Shape::Cmp, 2, 2, false, false, false, false, false},
    {"ps_add",  K::PsAdd,   true,  4, 21, Shape::AB,  2, 2, true,  false, false, false, false},
    {"ps_sub",  K::PsSub,   true,  4, 20, Shape::AB,  2, 2, true,  false, false, false, false},
    {"ps_mul",  K::PsMul,   true,  4, 25, Shape::AC,  2, 2, true,  false, false, false, false},
    {"ps_madd", K::PsMadd,  true,  4, 29, Shape::ABC, 3, 3, true,  true,  false, false, false},
    {"ps_msub", K::PsMsub,  true,  4, 28, Shape::ABC, 3, 3, true,  true,  false, false, false},
    {"ps_nmadd", K::PsNmadd, true, 4, 31, Shape::ABC, 3, 3, true,  true,  false, false, false},
    {"ps_nmsub", K::PsNmsub, true, 4, 30, Shape::ABC, 3, 3, true,  true,  false, false, false},
    {"ps_muls0", K::PsMuls0, true, 4, 12, Shape::AC,  2, 2, true,  false, false, false, false},
    {"ps_muls1", K::PsMuls1, true, 4, 13, Shape::AC,  2, 2, true,  false, false, false, false},
    {"ps_madds0", K::PsMadds0, true, 4, 14, Shape::ABC, 3, 3, true, true, false, false, false},
    {"ps_madds1", K::PsMadds1, true, 4, 15, Shape::ABC, 3, 3, true, true, false, false, false},
    {"ps_sum0", K::PsSum0,  true,  4, 10, Shape::ABC, 3, 3, false, false, false, false, false},
    {"ps_sum1", K::PsSum1,  true,  4, 11, Shape::ABC, 3, 3, false, false, false, false, false},
    {"ps_merge00", K::PsMerge00, true, 4, 528, Shape::AB10, 2, 2, false, false, false, false, false},
    {"ps_merge01", K::PsMerge01, true, 4, 560, Shape::AB10, 2, 2, false, false, false, false, false},
    {"ps_merge10", K::PsMerge10, true, 4, 592, Shape::AB10, 2, 2, false, false, false, false, false},
    {"ps_merge11", K::PsMerge11, true, 4, 624, Shape::AB10, 2, 2, false, false, false, false, false},
    {"ps_mr",   K::PsMr,    true,  4, 72, Shape::B10, 1, 0, false, false, false, false, false},
    {"ps_neg",  K::PsNeg,   true,  4, 40, Shape::B10, 1, 1, false, false, false, false, false},
    {"ps_abs",  K::PsAbs,   true,  4, 264, Shape::B10, 1, 1, false, false, false, false, false},
    // Decode leaves ps_sel Generic (its counters are K::Generic's, which only it of the kinds here uses)
    {"ps_sel",  K::Generic, true,  4, 23, Shape::ABC, 3, 3, false, false, false, false, false},
};
// clang-format on

// The instruction fields (0 a, 1 b, 2 c) each shape reads, in order.
unsigned Fields(Shape s, unsigned out[3]) {
    switch (s) {
    case Shape::AB: case Shape::AB10: case Shape::Cmp: out[0] = 0; out[1] = 1; return 2;
    case Shape::AC: out[0] = 0; out[1] = 2; return 2;
    case Shape::ABC: out[0] = 0; out[1] = 1; out[2] = 2; return 3;
    case Shape::B10: out[0] = 1; return 1;
    }
    return 0;
}
uint32_t Encode(const Kind& k, uint32_t d, uint32_t a, uint32_t b, uint32_t c) {
    switch (k.shape) {
    case Shape::AB: return Fa(k.op, d, a, b, 0, k.xo);
    case Shape::AC: return Fa(k.op, d, a, 0, c, k.xo);
    case Shape::ABC: return Fa(k.op, d, a, b, c, k.xo);
    case Shape::B10: return Fx(k.op, d, 0, b, k.xo);
    case Shape::AB10: return Fx(k.op, d, a, b, k.xo);
    case Shape::Cmp: return Fcmpu(3, a, b);
    }
    return 0;
}

// --- operand tables ----------------------------------------------------------------------------------------------------
std::vector<uint64_t> g_doubles, g_pairs;
std::vector<uint32_t> g_singles;

void BuildTables() {
    const uint64_t mags[] = {
        0x0000000000000000ull, 0x0000000000000001ull, 0x000FFFFFFFFFFFFFull, 0x000123456789ABCDull, 0x0010000000000000ull,
        0x0010000000000001ull, 0x3690000000000000ull, 0x369FFFFFFFFFFFFFull, 0x36A0000000000000ull, 0x37E0000000000000ull,
        0x3800000000000000ull, 0x380FFFFFE0000000ull, 0x380FFFFFFFFFFFFFull, 0x3810000000000000ull, 0x3810000000000001ull,
        0x3E70000000000000ull, 0x3FE0000000000000ull, 0x3FF0000000000000ull, 0x3FF0000000000001ull, 0x3FF0000008000000ull,
        0x3FF000000FFFFFFFull, 0x3FF0000010000000ull, 0x3FF0000010000001ull, 0x3FF0000030000000ull, 0x3FF8000000000000ull,
        0x4000000000000000ull, 0x4008000000000000ull, 0x4170000010000000ull, 0x40091EB851EB851Full, 0x3FB999999999999Aull,
        0x41E0000000000000ull, 0x47D0000000000000ull, 0x47EFFFFFE0000000ull, 0x47EFFFFFEFFFFFFFull, 0x47EFFFFFF0000000ull,
        0x47F0000000000000ull, 0x7FDFFFFFFFFFFFFFull, 0x7FEFFFFFFFFFFFFFull, 0x7FF0000000000000ull,
    };
    for (uint64_t m : mags) {
        g_doubles.push_back(m);
        g_doubles.push_back(m | 0x8000000000000000ull);
    }
    for (uint64_t n : {0x7FF8000000000000ull, 0xFFF8000000000000ull, 0x7FF8000000001234ull, 0xFFF8DEAD00000001ull,
                       0x7FF0000000000001ull, 0xFFF4000000000123ull, 0x7FF7FFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull})
        g_doubles.push_back(n);

    const uint32_t smags[] = {0x00000000u, 0x00000001u, 0x007FFFFFu, 0x00800000u, 0x00800001u, 0x0D800000u, 0x33800000u,
                              0x34000000u, 0x3DCCCCCDu, 0x3F800000u, 0x3F800001u, 0x3FC00000u, 0x40400000u, 0x4B800000u,
                              0x71800000u, 0x7F7FFFFEu, 0x7F7FFFFFu, 0x7F800000u};
    for (uint32_t m : smags) {
        g_singles.push_back(m);
        g_singles.push_back(m | 0x80000000u);
    }
    for (uint32_t n : {0x7FC00000u, 0xFFC00000u, 0x7FC01234u, 0xFFC0BEEFu, 0x7F800001u, 0xFF8ABCDEu, 0x7FBFFFFFu, 0xFFFFFFFFu})
        g_singles.push_back(n);
    const size_t n = g_singles.size();
    for (size_t i = 0; i < n; ++i) {
        g_pairs.push_back((static_cast<uint64_t>(g_singles[i]) << 32) | g_singles[i]);                   // equal lanes
        g_pairs.push_back((static_cast<uint64_t>(g_singles[i]) << 32) | g_singles[(i * 5 + 3) % n]);     // ps0, ps1 differ
    }
}

// What a source reads from, by the form it arrives in and whether its op is a paired one.
// (Form::Unknown cannot be produced: PpcInterpreter::Run starts every register as Scalar.)
enum class SrcForm : uint8_t { Scalar, Paired, Single };
const std::vector<uint64_t>& TableFor(SrcForm f, bool pairedOp) {
    (void)pairedOp;
    return f == SrcForm::Paired ? g_pairs : g_doubles;
}

// A random bit pattern biased to special exponents, as a double or as two singles.
uint64_t RandomDouble(Rng& r) {
    const uint64_t sign = static_cast<uint64_t>(r.Below(2)) << 63;
    uint64_t exp;
    switch (r.Below(8)) {
    case 0: exp = 0; break;
    case 1: exp = 0x7FF; break;
    case 2: exp = 896 + r.Below(3); break;               // around 2^-126 as a double
    case 3: exp = 1023 + 127 + r.Below(3); break;        // around the single range's top
    case 4: exp = 1023 - 149 + r.Below(4); break;
    case 5: exp = 1023 + (r.Below(60)) - 30; break;
    case 6: exp = 1; break;
    default: exp = r.Below(0x800); break;
    }
    uint64_t frac;
    switch (r.Below(6)) {
    case 0: frac = 0; break;
    case 1: frac = 1; break;
    case 2: frac = 0xFFFFFFFFFFFFFull; break;
    case 3: frac = r.Next64() & 0xFFFFFFFF0000000ull; break;  // a single's fraction
    case 4: frac = (r.Next64() & 0xFFFFFFFF0000000ull) | (r.Below(2) ? 0x10000000ull : 0x0FFFFFFFull); break;
    default: frac = r.Next64() & 0xFFFFFFFFFFFFFull; break;
    }
    return sign | (exp << 52) | frac;
}
uint32_t RandomSingle(Rng& r) {
    const uint32_t sign = r.Below(2) << 31;
    uint32_t exp;
    switch (r.Below(6)) {
    case 0: exp = 0; break;
    case 1: exp = 0xFF; break;
    case 2: exp = 1; break;
    case 3: exp = 254; break;
    case 4: exp = 127 + r.Below(5) - 2; break;
    default: exp = r.Below(256); break;
    }
    uint32_t frac;
    switch (r.Below(5)) {
    case 0: frac = 0; break;
    case 1: frac = 1; break;
    case 2: frac = 0x7FFFFF; break;
    case 3: frac = 0x400000 | r.Below(0x400000); break;
    default: frac = r.Next() & 0x7FFFFF; break;
    }
    return sign | (exp << 23) | frac;
}
uint64_t RandomValue(Rng& r, SrcForm f, bool pairedOp) {
    const bool pairLike = f == SrcForm::Paired;
    (void)pairedOp;
    if (r.Below(3) == 0) {
        const auto& t = TableFor(f, pairedOp);
        return t[r.Below(static_cast<uint32_t>(t.size()))];
    }
    if (pairLike) return (static_cast<uint64_t>(RandomSingle(r)) << 32) | RandomSingle(r);
    return RandomDouble(r);
}

// --- running -----------------------------------------------------------------------------------------------------------
unsigned g_failures = 0;
uint64_t g_states = 0;
uint32_t g_nextCode = kCodeBase;

CpuContext BaseState(Rng& r, uint32_t fpscr) {
    CpuContext c{};
    for (auto& g : c.gpr) g = r.Next();
    c.gpr[1] = kDataBase + 0x1800u;
    c.gpr[30] = kProbe;
    JitTestCr::Randomize(c, [&] { return r.Next(); });
    c.xer = r.Next();
    c.ctr = r.Next();
    c.lr = kReturn;
    c.fpscr = fpscr;
    for (auto& f : c.fpr) f.raw = r.Next64();
    for (auto& q : c.gqr) q = 0;
    return c;
}

struct Program {
    uint32_t classic, jit;
    uint32_t words[16];
    unsigned count;
};

void Place(uint32_t base, const uint32_t* words, unsigned count) {
    for (unsigned i = 0; i < count; ++i) Memory::Write32(base + 4u * i, words[i]);
}

void RunStates(Engine engine, uint32_t entry, const std::vector<CpuContext>& in, std::vector<CpuContext>& out) {
    PpcInterpreter::SetEngineOverride(engine);
    PpcInterpreter::SetJitThreshold(0);
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        CpuContext ctx = in[i];
        ctx.pc = 0;
        {
            CpuContextScope scope(&ctx);  // NI mode, and the context the float helpers read
            PpcInterpreter::Run(&ctx, entry);
        }
        ctx.pc = 0;
        out[i] = ctx;
    }
    PpcInterpreter::ClearEngineOverride();
}

bool Equal(const CpuContext& a, const CpuContext& b) {
    return std::memcmp(a.gpr, b.gpr, sizeof a.gpr) == 0 && std::memcmp(a.fpr, b.fpr, sizeof a.fpr) == 0 && JitTestCr::Same(a, b) &&
           a.lr == b.lr && a.ctr == b.ctr && a.xer == b.xer && a.fpscr == b.fpscr;
}

const char* FormName(SrcForm f) { return f == SrcForm::Scalar ? "scalar" : f == SrcForm::Paired ? "paired" : "single"; }

struct Setup {
    const Kind* kind;
    SrcForm form[3];
    unsigned reg[3];  // register of source i
    unsigned dest;
    unsigned alias;
    unsigned fieldOf[3];
    unsigned axisOf[3];
};

// Builds the program, runs every state under both engines, compares.
void RunProgram(const Setup& s, const std::vector<CpuContext>& states) {
    if (states.empty()) return;
    const Kind& k = *s.kind;
    uint32_t words[16];
    unsigned n = 0;
    bool prepared[32] = {};
    for (unsigned i = 0; i < k.nsrc; ++i) {
        const unsigned r = s.reg[i];
        if (prepared[r]) continue;
        prepared[r] = true;
        if (s.form[i] == SrcForm::Paired) {  // the same 64 bits, loaded as two singles: held as a pair
            words[n++] = Stfd(r, 0, 30);
            words[n++] = PsqL(r, 0, 30);
        } else if (s.form[i] == SrcForm::Single) {  // rounded to a single in the block: known to be one
            words[n++] = Fx(63, r, 0, r, 12);
        }
    }
    uint32_t field[3] = {0, 0, 0};  // the registers of a, b, c
    for (unsigned i = 0; i < k.nsrc; ++i) field[s.fieldOf[i]] = s.reg[i];
    words[n++] = Encode(k, s.dest, field[0], field[1], field[2]);
    words[n++] = Blr();
    if (g_nextCode + 2 * kCodeStep > kCodeLimit) {
        std::fprintf(stderr, "jit fp: out of code space\n");
        std::exit(2);
    }
    const uint32_t classic = g_nextCode, jit = g_nextCode + kCodeStep;
    g_nextCode += 2 * kCodeStep;
    Place(classic, words, n);
    Place(jit, words, n);
    std::vector<CpuContext> want, got;
    RunStates(Engine::Classic, classic, states, want);
    RunStates(Engine::Jit, jit, states, got);
    g_states += states.size();
    for (size_t i = 0; i < states.size(); ++i) {
        if (Equal(want[i], got[i])) continue;
        ++g_failures;
        if (g_failures <= 12) {
            std::string out = Fmt("FAIL %s: sources", k.name);
            for (unsigned j = 0; j < k.nsrc; ++j)
                out += Fmt(" f%u(%s)=%016llX", s.reg[j], FormName(s.form[j]), static_cast<unsigned long long>(states[i].fpr[s.reg[j]].raw));
            out += Fmt(" dest f%u alias %u NI=%u\n", s.dest, s.alias, (states[i].fpscr >> 2) & 1u);
            for (int r = 0; r < 32; ++r)
                if (want[i].fpr[r].raw != got[i].fpr[r].raw)
                    out += Fmt("    f%d: classic=%016llX jit=%016llX\n", r, static_cast<unsigned long long>(want[i].fpr[r].raw),
                               static_cast<unsigned long long>(got[i].fpr[r].raw));
            out += JitTestCr::Diff(want[i], got[i], states[i]);
            if (want[i].fpscr != got[i].fpscr) out += Fmt("    fpscr: classic=%08X jit=%08X\n", want[i].fpscr, got[i].fpscr);
            std::fputs(out.c_str(), stderr);
        }
        break;  // one report per program
    }
}

// The values the sources take in a program: for each source its table, as the indexes a loop visits.
struct Axis {
    const std::vector<uint64_t>* table;
    std::vector<size_t> picks;
};
std::vector<size_t> Strided(size_t n, size_t stride, size_t offset) {
    std::vector<size_t> v;
    for (size_t i = offset % stride; i < n; i += stride) v.push_back(i);
    return v;
}
std::vector<size_t> All(size_t n) { return Strided(n, 1, 0); }
// A handful of values for a third source: ones, a rounding case, zeros, infinity, NaNs, a denormal.
std::vector<size_t> ThirdPicks(const std::vector<uint64_t>& t, bool pairedForm) {
    const uint64_t want[] = {
        pairedForm ? 0x3F8000003F800000ull : 0x3FF0000000000000ull, pairedForm ? 0x3FC0000000000000ull | 0x33800000ull : 0x3E70000000000000ull,
        0x0000000000000000ull,
        pairedForm ? 0xBF800000BF800000ull : 0xBFF8000000000000ull,
        pairedForm ? 0x7F8000007FC00000ull : 0x7FF0000000000000ull,
        pairedForm ? 0x7FC012347F800001ull : 0xFFF8DEAD00000001ull,
        pairedForm ? 0x007FFFFF00000001ull : 0x000123456789ABCDull,
        pairedForm ? 0x3F8000013F800001ull : 0x3FF0000010000001ull};
    std::vector<size_t> v;
    for (uint64_t w : want)
        for (size_t i = 0; i < t.size(); ++i)
            if (t[i] == w) {
                v.push_back(i);
                break;
            }
    if (v.size() < 4)
        for (size_t i = 0; i < t.size(); i += t.size() / 6 + 1) v.push_back(i);
    return v;
}

void MakeStates(const Setup& s, const Axis axes[3], unsigned nAxes, Rng& rng, std::vector<CpuContext>& out) {
    // The distinct registers' tables crossed (the sources that share a register share an axis).
    size_t idx[3] = {0, 0, 0};
    for (idx[0] = 0; idx[0] < axes[0].picks.size(); ++idx[0])
        for (idx[1] = 0; nAxes < 2 ? idx[1] == 0 : idx[1] < axes[1].picks.size(); ++idx[1])
            for (idx[2] = 0; nAxes < 3 ? idx[2] == 0 : idx[2] < axes[2].picks.size(); ++idx[2])
                for (uint32_t ni : {0u, 4u}) {
                    CpuContext c = BaseState(rng, ni);
                    for (unsigned i = 0; i < s.kind->nsrc; ++i) {
                        const Axis& ax = axes[s.axisOf[i]];
                        c.fpr[s.reg[i]].raw = (*ax.table)[ax.picks[idx[s.axisOf[i]]]];
                    }
                    out.push_back(c);
                }
}

void RunKind(const Kind& k, uint64_t randomStates, uint64_t seed) {
    Rng rng(seed + static_cast<uint64_t>(k.kind) * 7919);
    unsigned fields[3];
    const unsigned n = Fields(k.shape, fields);
    // Aliases: none; the destination being source i; source i being source j.
    struct Alias { int destSource; unsigned same[2]; bool hasSame; };
    std::vector<Alias> aliases;
    aliases.push_back({-1, {0, 0}, false});
    for (unsigned i = 0; i < n; ++i) aliases.push_back({static_cast<int>(i), {0, 0}, false});
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = i + 1; j < n; ++j) aliases.push_back({-1, {i, j}, true});
    const unsigned combos = 1u << n;
    std::vector<Setup> programs;
    for (unsigned ai = 0; ai < aliases.size(); ++ai) {
        for (unsigned fc = 0; fc < combos; ++fc) {
            Setup s{};
            s.kind = &k;
            s.alias = ai;
            unsigned x = fc;
            for (unsigned i = 0; i < n; ++i) {
                s.form[i] = static_cast<SrcForm>(x % 2);
                x /= 2;
                s.fieldOf[i] = fields[i];
                s.reg[i] = 2 + i;  // not f1: a function starts with f1 held as a scalar
            }
            const Alias& al = aliases[ai];
            if (al.hasSame) {
                s.reg[al.same[1]] = s.reg[al.same[0]];
                if (s.form[al.same[1]] != s.form[al.same[0]]) continue;  // one register has one form
            }
            s.dest = al.destSource >= 0 ? s.reg[al.destSource] : 5;
            programs.push_back(s);
        }
        // every source a single made in the block (a scalar kind's knowledge; a paired kind converts it as any scalar)
        Setup s{};
        s.kind = &k;
        s.alias = ai;
        for (unsigned i = 0; i < n; ++i) {
            s.form[i] = SrcForm::Single;
            s.fieldOf[i] = fields[i];
            s.reg[i] = 2 + i;
        }
        const Alias& al = aliases[ai];
        if (al.hasSame) s.reg[al.same[1]] = s.reg[al.same[0]];
        s.dest = al.destSource >= 0 ? s.reg[al.destSource] : 5;
        programs.push_back(s);
    }
    const SrcForm native = k.paired ? SrcForm::Paired : SrcForm::Scalar;
    uint64_t programIndex = 0;
    for (Setup& s : programs) {
        ++programIndex;
        Axis axes[3];
        unsigned nAxes = 0;
        bool allNative = true;
        for (unsigned i = 0; i < n; ++i) {
            if (s.form[i] != native) allNative = false;
            unsigned owner = i;
            for (unsigned j = 0; j < i; ++j)
                if (s.reg[j] == s.reg[i]) {
                    owner = j;
                    break;
                }
            if (owner != i) {
                s.axisOf[i] = s.axisOf[owner];
                continue;
            }
            s.axisOf[i] = nAxes;
            axes[nAxes].table = &TableFor(s.form[i], k.paired);
            ++nAxes;
        }
        for (unsigned a = 0; a < nAxes; ++a) {
            const size_t len = axes[a].table->size();
            const bool pairForm = axes[a].table == &g_pairs;
            if (nAxes == 3 && a == 2) {
                axes[a].picks = ThirdPicks(*axes[a].table, pairForm);
                if (allNative && s.alias == 0)  // the all-native program: every fourth value too
                    for (size_t i : Strided(len, 4, 0))
                        if (std::find(axes[a].picks.begin(), axes[a].picks.end(), i) == axes[a].picks.end()) axes[a].picks.push_back(i);
            }
            else if (nAxes == 1 || (allNative && s.alias == 0)) axes[a].picks = All(len);
            else axes[a].picks = Strided(len, nAxes == 2 ? 3 : 6, programIndex);
        }
        std::vector<CpuContext> states;
        MakeStates(s, axes, nAxes, rng, states);
        RunProgram(s, states);
        // random states for this program: random forms are the program's, random values
        const uint64_t perProgram = randomStates / programs.size();
        std::vector<CpuContext> random;
        random.reserve(perProgram);
        for (uint64_t i = 0; i < perProgram; ++i) {
            CpuContext c = BaseState(rng, rng.Below(2) ? 4u : 0u);
            for (unsigned j = 0; j < k.nsrc; ++j) {
                bool owns = true;
                for (unsigned q = 0; q < j; ++q)
                    if (s.reg[q] == s.reg[j]) owns = false;
                if (owns) c.fpr[s.reg[j]].raw = RandomValue(rng, s.form[j], k.paired);
            }
            random.push_back(c);
        }
        RunProgram(s, random);
    }
}

}  // namespace

int main() {
    // Count what the JIT's float code does (read when the first block is compiled); BRAWL_JIT_FP_HELPERS may be set by the caller.
    setenv("BRAWL_JIT_FP_COUNTERS", "1", 1);
    Memory::Init(0x200000);
    TranslatedFunctionRegistry::Finalize();
    BuildTables();
    const bool jit = PpcInterpreter::JitAvailable();
    const char* helpersText = std::getenv("BRAWL_JIT_FP_HELPERS");
    const bool helpers = helpersText != nullptr && *helpersText != '\0' && *helpersText != '0';
    uint64_t randomStates = 1000000;
    uint64_t seed = 1;
    if (const char* t = std::getenv("FP_TEST_RANDOM")) randomStates = std::strtoull(t, nullptr, 10);
    if (const char* t = std::getenv("FP_TEST_SEED")) seed = std::strtoull(t, nullptr, 10);
    const char* filter = std::getenv("FP_TEST_FILTER");
    std::printf("jit fp: JIT backend %s%s; %zu doubles, %zu pairs; %llu random states per kind\n", jit ? "available" : "not available",
                helpers ? " (helpers forced)" : "", g_doubles.size(), g_pairs.size(), static_cast<unsigned long long>(randomStates));

    const auto start = std::chrono::steady_clock::now();
    const uint32_t savedControl = MkwGetHostFpControl();
    for (const Kind& k : kKinds) {
        if (filter != nullptr && std::strcmp(filter, k.name) != 0) continue;
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t before = g_states;
        const unsigned failedBefore = g_failures;
        RunKind(k, randomStates, seed);
        std::printf("  %-10s %9llu states %6.2f s%s\n", k.name, static_cast<unsigned long long>(g_states - before),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                    g_failures != failedBefore ? "  FAILED" : "");
        std::fflush(stdout);
    }
    MkwRestoreHostMxcsr(savedControl);

    // What the JIT's code did.
    unsigned missing = 0;
    if (jit) {
        const uint64_t* counters = PpcInterpreter::detail::x64::JitFpCounters();
        std::printf("\n  %-10s %9s %8s %8s %8s %8s %8s %8s %8s %8s\n", "kind", "native", "slow", "conv0", "conv1", "conv2", "nan", "range",
                    "tie", "mult");
        for (const Kind& k : kKinds) {
            if (filter != nullptr && std::strcmp(filter, k.name) != 0) continue;
            const uint64_t* c = counters + static_cast<unsigned>(k.kind) * 12;
            std::printf("  %-10s %9llu %8llu %8llu %8llu %8llu %8llu %8llu %8llu %8llu\n", k.name, static_cast<unsigned long long>(c[0]),
                        static_cast<unsigned long long>(c[1]), static_cast<unsigned long long>(c[2]), static_cast<unsigned long long>(c[3]),
                        static_cast<unsigned long long>(c[4]), static_cast<unsigned long long>(c[5]), static_cast<unsigned long long>(c[6]),
                        static_cast<unsigned long long>(c[7]), static_cast<unsigned long long>(c[8]));
            const auto need = [&](const char* what, uint64_t value) {
                if (value == 0) {
                    std::fprintf(stderr, "MISSING %s: %s never happened\n", k.name, what);
                    ++missing;
                }
            };
            need("native completion (the kind fell back to the generic call?)", c[0]);
            for (unsigned i = 0; i < k.convSlots; ++i) need(Fmt("conversion of source %u", i).c_str(), c[2 + i]);
#if defined(__FMA__)
            const bool hasInline = k.inlineArith && !helpers;
#else
            const bool hasInline = k.inlineArith && !k.needsFma && !helpers;
#endif
            if (hasInline) {
                need("the path to the interpreter", c[1]);
                if (k.kind != K::Frsp) need("a NaN result", c[5]);  // frsp narrows a NaN natively
                if (k.reasonRange) need("a single result in the flushed range", c[6]);
                if (k.reasonTie) need("an exact tie", c[7]);
                if (k.reasonMultiplier) need("a NaN or denormal multiplier", c[8]);
            }
        }
    }
    std::printf("\njit fp: %.1f s, %llu states, %u failure%s, %u missing counter%s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                static_cast<unsigned long long>(g_states), g_failures, g_failures == 1 ? "" : "s", missing, missing == 1 ? "" : "s");
    if (g_failures != 0 || missing != 0) {
        std::printf("jit fp: FAILED\n");
        return 1;
    }
    std::printf("jit fp: ok\n");
    return 0;
}
