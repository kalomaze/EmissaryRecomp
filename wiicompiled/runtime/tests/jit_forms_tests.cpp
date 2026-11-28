// The JIT's floating point form knowledge (src/jit/jit_x64_forms.cpp): what a block knows of the form bytes of the FPRs op by op,
// so that a float op skips the test of a form it knows, converts without a test when the form is known to differ, and does not
// write a form byte that already holds the value. A wrong "known" is a wrong result (or a stale form byte that only a later block
// reads), which random programs can miss, so every program here is directed, run under Engine::Classic and Engine::Jit
// (threshold 0) from several states, and the whole contexts (CR fields bytewise: jit_test_cr.h), the memory the program wrote and the form each register is left in
// compared.
//
// How a register's form is seen: every program ends by branching (an always-taken bc: a block boundary, so the next block has
// no knowledge) to one shared probe block that stores f1-f9 with stfs, stfd and psq_st (which read a register as Scalar and as
// Paired: the three differ for a register holding a value that is not the same in both views) and returns. So a form byte the
// compiled code left wrong shows in the probe's stores.
//
//  * `A; B` for every FPR-writing kind A (loads, the arithmetic, the moves, the quantized load, and the Generic fsel, fnabs,
//    fnmadd, fnmsub, fres, frsqrte, fctiw, fctiwz, lfdx, lfsux) and every FPR-reading kind B (the same arithmetic, the stores,
//    fcmpu, the moves), with A's destination being B's source 1, 2 or 3 or none of them;
//  * `A; X; B` for each barrier X: nothing that touches forms (an integer op, a Generic integer op, a conditional branch not
//    taken), and the ones that end the knowledge: a taken branch (a block boundary), a call to untranslated code, a Generic op the
//    analysis does not know (mffs), a thin call to a registered function (which normalises its float arguments), a call of a
//    patched function (a registered function that runs interpreted guest code in a Run of its own);
//  * random chains of 2 to 9 float ops over a few registers (the same register changing form several times, sources and
//    destinations aliasing, slow paths from NaN inputs), with barriers dropped in at random, and chains of 6 to 14 over f1-f12
//    (more registers than the FPR cache's slots: evictions of dirty slots, refills, flushes at the barriers);
//  * each program from register states whose forms are Scalar, Paired and mixed, prepared in an earlier block (so the sequence
//    starts with nothing known) or in the same block (so the sources are known), and values from the edge tables, NI on and off.
// The counters (JitX64GetFormStats) show the knowledge was used where it must be: for `A; B` in one block the form tests emitted
// are exactly those of the sources not known, and none was skipped across a barrier.
//
// The analysis is then made wrong five ways (BRAWL_JIT_FORMS_FAULT, read at every compile): the tests must fail for each.
// BRAWL_JIT_FALLBACK, BRAWL_JIT_FP_HELPERS or BRAWL_JIT_NO_FORMS in the environment turn the counter checks off.
//
// FORMS_RANDOM=<n>: random chains (default 3000); FORMS_SEED.
#include "jit_test_env.h"
#include "jit_test_cr.h"
#include "abi_bridge.h"
#include "brawl_dirty_memory.h"
#include "isa/ppc_isa_fpenv.h"
#include "memory.h"
#include "ppc_interpreter.h"
#include "../src/jit/jit_x64.h"

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace GuestCodePatch {  // the test double (interpreter_test_support.cpp)
void TestSetInterpreted(uint32_t address, uint32_t at);
uint32_t TestInterpretedAt(uint32_t address);
void Sync();
}  // namespace GuestCodePatch

namespace {

using PpcInterpreter::Engine;
using PpcInterpreter::detail::JitX64FormStats;
using PpcInterpreter::detail::JitX64GetFormStats;

constexpr uint32_t kData = 0x80030000u;      // the registers' initial values, by r30
constexpr uint32_t kProbe = kData + 0x400u;  // what the probe block stores
constexpr uint32_t kReturn = 0x80020000u;
constexpr uint32_t kProbeFn = 0x80012000u;
constexpr uint32_t kLeaf = 0x80012100u;      // `blr`
constexpr uint32_t kCodeBase = 0x80100000u;
constexpr uint32_t kCodeStep = 0x200u;
constexpr uint32_t kCodeSlots = 2048;
constexpr uint32_t kThin = 0x80040000u + 0x10u * 1u;     // a registered function whose float ABI has arguments
constexpr uint32_t kPatched = 0x80041000u;               // a patched function (a registered function that runs guest code)
constexpr uint32_t kPatchedBody = 0x80012200u;

uint64_t g_failures = 0, g_programs = 0, g_runs = 0;
bool g_checkCounters = true, g_quiet = false;
std::string g_context;

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
    uint64_t Next64() { return (uint64_t{Next()} << 32) | Next(); }
};

// --- encoders ----------------------------------------------------------------------------------------------------------------
constexpr uint32_t D(uint32_t op, uint32_t rt, uint32_t ra, int32_t imm) {
    return (op << 26) | (rt << 21) | (ra << 16) | (static_cast<uint32_t>(imm) & 0xFFFFu);
}
constexpr uint32_t X(uint32_t rt, uint32_t ra, uint32_t rb, uint32_t xo, uint32_t rc = 0) {
    return (31u << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (xo << 1) | rc;
}
constexpr uint32_t Fa(uint32_t op, uint32_t d, uint32_t a, uint32_t b, uint32_t c, uint32_t xo5) {
    return (op << 26) | (d << 21) | (a << 16) | (b << 11) | (c << 6) | (xo5 << 1);
}
constexpr uint32_t Fx(uint32_t op, uint32_t d, uint32_t a, uint32_t b, uint32_t xo10) {
    return (op << 26) | (d << 21) | (a << 16) | (b << 11) | (xo10 << 1);
}
constexpr uint32_t Psq(uint32_t op, uint32_t f, uint32_t ra, uint32_t w, uint32_t i, int32_t disp) {
    return (op << 26) | (f << 21) | (ra << 16) | (w << 15) | (i << 12) | (static_cast<uint32_t>(disp) & 0xFFFu);
}
constexpr uint32_t Bc(uint32_t bo, uint32_t bi, int32_t byteDisp) {
    return (16u << 26) | (bo << 21) | (bi << 16) | (static_cast<uint32_t>(byteDisp) & 0xFFFCu);
}
constexpr uint32_t Bl(int32_t byteDisp) { return (18u << 26) | (static_cast<uint32_t>(byteDisp) & 0x03FFFFFCu) | 1u; }
constexpr uint32_t kBlr = 0x4E800020u;
constexpr uint32_t Mtlr(uint32_t rs) { return X(rs, 8, 0, 467); }
constexpr uint32_t Cmpwi(uint32_t crf, uint32_t ra, int32_t i) { return D(11, crf << 2, ra, i); }

// --- the kinds -----------------------------------------------------------------------------------------------------------------
// word(d, a, b, c): the instruction with these registers in its fields (c is the frC field of the A-forms). `src`: the fields
// (bit 0 a, 1 b, 2 c, 3 d) whose register the op reads through a form test and conversion; `dform`: the form it leaves its
// destination in (S scalar, P paired, C copies a source's, N writes none); `psqStore`: a quantized store, which needs no test
// for a register whose form is known (Paired: native; Scalar: the shim, unconditionally).
enum class DForm : uint8_t { S, P, C, N };
struct Kind {
    const char* name;
    uint32_t (*word)(uint32_t d, uint32_t a, uint32_t b, uint32_t c);
    uint8_t src;
    DForm dform;
    bool psqStore = false;
    bool store = false;  // reads the register in field d
};
constexpr uint8_t kA = 1, kB = 2, kC = 4, kD = 8;
#define W(expr) [](uint32_t d, uint32_t a, uint32_t b, uint32_t c) -> uint32_t { (void)d; (void)a; (void)b; (void)c; return expr; }
// clang-format off
const Kind kKinds[] = {
    {"lfs", W(D(48, d, 30, 0x40)), 0, DForm::S},
    {"lfd", W(D(50, d, 30, 0x48)), 0, DForm::S},
    {"lfsu", W(D(49, d, 29, 0x40)), 0, DForm::S},
    {"lfdu", W(D(51, d, 29, 0x48)), 0, DForm::S},
    {"lfsx", W(X(d, 30, 24, 535)), 0, DForm::S},
    {"lfdx", W(X(d, 30, 24, 599)), 0, DForm::S},
    {"lfsux", W(X(d, 29, 24, 567)), 0, DForm::S},
    {"psq_l", W(Psq(56, d, 30, 0, 0, 0x50)), 0, DForm::P},
    {"psq_l w1", W(Psq(56, d, 30, 1, 0, 0x50)), 0, DForm::P},
    {"psq_lu", W(Psq(57, d, 29, 0, 0, 0x50)), 0, DForm::P},
    {"fadds", W(Fa(59, d, a, b, 0, 21)), kA | kB, DForm::S},
    {"fsubs", W(Fa(59, d, a, b, 0, 20)), kA | kB, DForm::S},
    {"fmuls", W(Fa(59, d, a, 0, c, 25)), kA | kC, DForm::S},
    {"fdivs", W(Fa(59, d, a, b, 0, 18)), kA | kB, DForm::S},
    {"fmadds", W(Fa(59, d, a, b, c, 29)), kA | kB | kC, DForm::S},
    {"fmsubs", W(Fa(59, d, a, b, c, 28)), kA | kB | kC, DForm::S},
    {"fnmsubs", W(Fa(59, d, a, b, c, 30)), kA | kB | kC, DForm::S},
    {"fnmadds", W(Fa(59, d, a, b, c, 31)), kA | kB | kC, DForm::S},
    {"fadd", W(Fa(63, d, a, b, 0, 21)), kA | kB, DForm::S},
    {"fsub", W(Fa(63, d, a, b, 0, 20)), kA | kB, DForm::S},
    {"fmul", W(Fa(63, d, a, 0, c, 25)), kA | kC, DForm::S},
    {"fdiv", W(Fa(63, d, a, b, 0, 18)), kA | kB, DForm::S},
    {"fmadd", W(Fa(63, d, a, b, c, 29)), kA | kB | kC, DForm::S},
    {"fmsub", W(Fa(63, d, a, b, c, 28)), kA | kB | kC, DForm::S},
    {"frsp", W(Fx(63, d, 0, b, 12)), kB, DForm::S},
    {"fneg", W(Fx(63, d, 0, b, 40)), kB, DForm::S},
    {"fabs", W(Fx(63, d, 0, b, 264)), kB, DForm::S},
    {"fmr", W(Fx(63, d, 0, b, 72)), 0, DForm::C},
    {"fcmpu", W((63u << 26) | (3u << 23) | (a << 16) | (b << 11)), kA | kB, DForm::N},
    {"ps_add", W(Fa(4, d, a, b, 0, 21)), kA | kB, DForm::P},
    {"ps_sub", W(Fa(4, d, a, b, 0, 20)), kA | kB, DForm::P},
    {"ps_mul", W(Fa(4, d, a, 0, c, 25)), kA | kC, DForm::P},
    {"ps_madd", W(Fa(4, d, a, b, c, 29)), kA | kB | kC, DForm::P},
    {"ps_msub", W(Fa(4, d, a, b, c, 28)), kA | kB | kC, DForm::P},
    {"ps_nmadd", W(Fa(4, d, a, b, c, 31)), kA | kB | kC, DForm::P},
    {"ps_nmsub", W(Fa(4, d, a, b, c, 30)), kA | kB | kC, DForm::P},
    {"ps_muls0", W(Fa(4, d, a, 0, c, 12)), kA | kC, DForm::P},
    {"ps_muls1", W(Fa(4, d, a, 0, c, 13)), kA | kC, DForm::P},
    {"ps_madds0", W(Fa(4, d, a, b, c, 14)), kA | kB | kC, DForm::P},
    {"ps_madds1", W(Fa(4, d, a, b, c, 15)), kA | kB | kC, DForm::P},
    {"ps_sum0", W(Fa(4, d, a, b, c, 10)), kA | kB | kC, DForm::P},
    {"ps_sum1", W(Fa(4, d, a, b, c, 11)), kA | kB | kC, DForm::P},
    {"ps_merge00", W(Fx(4, d, a, b, 528)), kA | kB, DForm::P},
    {"ps_merge01", W(Fx(4, d, a, b, 560)), kA | kB, DForm::P},
    {"ps_merge10", W(Fx(4, d, a, b, 592)), kA | kB, DForm::P},
    {"ps_merge11", W(Fx(4, d, a, b, 624)), kA | kB, DForm::P},
    {"ps_mr", W(Fx(4, d, 0, b, 72)), 0, DForm::C},
    {"ps_neg", W(Fx(4, d, 0, b, 40)), kB, DForm::P},
    {"ps_abs", W(Fx(4, d, 0, b, 264)), kB, DForm::P},
    // Generic ops with native code (jit_x64_generic.cpp, jit_x64_fp.cpp)
    {"fsel", W(Fa(63, d, a, b, c, 23)), kA | kB | kC, DForm::S},
    {"fnabs", W(Fx(63, d, 0, b, 136)), kB, DForm::S},
    {"fnmadd", W(Fa(63, d, a, b, c, 31)), kA | kB | kC, DForm::S},
    {"fnmsub", W(Fa(63, d, a, b, c, 30)), kA | kB | kC, DForm::S},
    {"fres", W(Fa(59, d, 0, b, 0, 24)), kB, DForm::S},
    {"frsqrte", W(Fx(63, d, 0, b, 26)), kB, DForm::S},
    {"fctiw", W(Fx(63, d, 0, b, 14)), kB, DForm::S},
    {"fctiwz", W(Fx(63, d, 0, b, 15)), kB, DForm::S},
    // readers only
    {"stfs", W(D(52, d, 30, 0x60)), kD, DForm::N, false, true},
    {"stfd", W(D(54, d, 30, 0x68)), kD, DForm::N, false, true},
    {"stfsx", W(X(d, 30, 27, 663)), kD, DForm::N, false, true},
    {"stfdx", W(X(d, 30, 26, 727)), kD, DForm::N, false, true},
    {"stfiwx", W(X(d, 30, 25, 983)), 0, DForm::N, false, true},
    {"psq_st", W(Psq(60, d, 30, 0, 0, 0x70)), kD, DForm::N, true, true},
    {"psq_st w1", W(Psq(60, d, 30, 1, 0, 0x78)), kD, DForm::N, true, true},
};
// clang-format on
constexpr unsigned kKindCount = sizeof kKinds / sizeof *kKinds;
const Kind& KindNamed(const char* name) {
    for (const Kind& k : kKinds)
        if (std::strcmp(k.name, name) == 0) return k;
    std::abort();
}

unsigned Popcount(unsigned v) { return static_cast<unsigned>(__builtin_popcount(v)); }

// The registers a kind reads through form tests, with its fields set to (a, b, c, d): the set as a bit mask.
uint32_t SourceRegs(const Kind& k, uint32_t d, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t m = 0;
    if (k.src & kA) m |= 1u << a;
    if (k.src & kB) m |= 1u << b;
    if (k.src & kC) m |= 1u << c;
    if (k.src & kD) m |= 1u << d;
    return m;
}

// --- running -------------------------------------------------------------------------------------------------------------------
struct Snap {
    CpuContext ctx{};
    uint8_t probe[0x100] = {};
    uint8_t scratch[0x100] = {};
    bool threw = false;
};

void Place(uint32_t base, const std::vector<uint32_t>& words) {
    for (size_t i = 0; i < words.size(); ++i) Memory::Write32(base + 4u * static_cast<uint32_t>(i), words[i]);
}

struct State {
    CpuContext ctx;
    std::array<uint64_t, 32> mem;  // each register's initial 64 bits, in guest memory at kData + 8 * r
    uint32_t pairedMask;           // the registers prepared Paired (psq_l), the others Scalar (lfd)
};

Snap RunOne(Engine engine, uint32_t entry, const State& st) {
    for (unsigned r = 0; r < 32; ++r) {
        Memory::Write32(kData + 8u * r, static_cast<uint32_t>(st.mem[r] >> 32));
        Memory::Write32(kData + 8u * r + 4, static_cast<uint32_t>(st.mem[r]));
    }
    std::memset(MemoryInline::LeanHost(kProbe), 0xEE, 0x100);
    std::memset(MemoryInline::LeanHost(kData + 0x100), 0xEE, 0x100);
    PpcInterpreter::SetEngineOverride(engine);
    PpcInterpreter::SetJitThreshold(0);
    Snap s;
    s.ctx = st.ctx;
    s.ctx.pc = 0;
    {
        CpuContextScope scope(&s.ctx);  // NI mode and the context the float helpers read
        try {
            PpcInterpreter::Run(&s.ctx, entry);
        } catch (...) {
            s.threw = true;
        }
    }
    s.ctx.pc = 0;
    PpcInterpreter::ClearEngineOverride();
    std::memcpy(s.probe, MemoryInline::LeanHost(kProbe), sizeof s.probe);
    std::memcpy(s.scratch, MemoryInline::LeanHost(kData + 0x60), sizeof s.scratch - 0x20);
    return s;
}

bool Same(const Snap& a, const Snap& b) {
    return std::memcmp(a.probe, b.probe, sizeof a.probe) == 0 && std::memcmp(a.scratch, b.scratch, sizeof a.scratch) == 0 &&
           a.threw == b.threw && std::memcmp(a.ctx.gpr, b.ctx.gpr, sizeof a.ctx.gpr) == 0 &&
           std::memcmp(a.ctx.fpr, b.ctx.fpr, sizeof a.ctx.fpr) == 0 && JitTestCr::Same(a.ctx, b.ctx) && a.ctx.xer == b.ctx.xer &&
           a.ctx.fpscr == b.ctx.fpscr && a.ctx.ctr == b.ctx.ctr;
}

const uint64_t kValues[] = {0x3FF8000040200000ull, 0x4009000000000000ull, 0xBFE0000000000000ull, 0x7FF8000000001234ull,
                            0x0000000000000001ull, 0x3FF0000000000000ull, 0x40490FDB3F800000ull, 0xC0000000C0800000ull,
                            0x7FF0000000000000ull, 0x8000000000000000ull, 0x3F8000003F800000ull, 0x47EFFFFFE0000000ull,
                            0x00000000000000FFull, 0xFFF8000000000000ull, 0x3FD5555555555555ull, 0x4330000000000000ull};

State MakeState(Rng& r, uint32_t formVariant) {
    State st{};
    for (auto& g : st.ctx.gpr) g = r.Next();
    st.ctx.gpr[1] = 0x80180000u;
    st.ctx.gpr[29] = kData;  // bases of the update forms
    st.ctx.gpr[30] = kData;  // the base of everything else; the registers' initial values are at kData
    st.ctx.gpr[31] = 5;
    st.ctx.gpr[28] = kReturn;  // what the barriers' mtlr restore
    st.ctx.gpr[24] = 0x80;     // the index of the indexed loads
    st.ctx.gpr[27] = 0x100;    // the indexes of the indexed stores: stfsx, stfdx, stfiwx
    st.ctx.gpr[26] = 0x108;
    st.ctx.gpr[25] = 0x110;
    st.ctx.lr = kReturn;
    st.ctx.ctr = 0;
    JitTestCr::Randomize(st.ctx, [&] { return r.Next(); });
    st.ctx.xer = 0;
    st.ctx.fpscr = (r.Below(2) != 0) ? 4u : 0u;  // NI
    for (unsigned i = 0; i < 32; ++i) {
        const uint64_t v = r.Below(3) == 0 ? r.Next64() : kValues[r.Below(sizeof kValues / sizeof *kValues)];
        st.mem[i] = v;
        st.ctx.fpr[i].raw = v;
    }
    // data the loads read: more of the same
    for (unsigned i = 0; i < 16; ++i) st.mem[8 + i] = r.Below(2) != 0 ? kValues[r.Below(sizeof kValues / sizeof *kValues)] : r.Next64();
    switch (formVariant % 4) {
    case 0: st.pairedMask = 0; break;
    case 1: st.pairedMask = 0xFFFFFFFEu; break;
    case 2: st.pairedMask = 0xAAAAAAAAu; break;
    default: st.pairedMask = r.Next(); break;
    }
    return st;
}

// The prologue that puts f1-f12 in the forms of `pairedMask` from memory (lfd: Scalar, psq_l with GQR0 = 0: Paired).
void PrepWords(std::vector<uint32_t>& w, uint32_t pairedMask) {
    for (uint32_t r = 1; r <= 12; ++r) {
        if (pairedMask & (1u << r)) w.push_back(Psq(56, r, 30, 0, 0, static_cast<int32_t>(8 * r)));
        else w.push_back(D(50, r, 30, static_cast<int32_t>(8 * r)));
    }
}

void WriteSupport() {
    // the probe block: f1-f9 as Scalar (stfs, stfd) and as Paired (psq_st), then back to the caller
    std::vector<uint32_t> w;
    for (uint32_t r = 1; r <= 9; ++r) {
        w.push_back(D(52, r, 30, static_cast<int32_t>(0x400 + 0x00 + 4 * r)));          // stfs
        w.push_back(D(54, r, 30, static_cast<int32_t>(0x400 + 0x40 + 8 * r)));          // stfd
        w.push_back(Psq(60, r, 30, 0, 7, static_cast<int32_t>(0x400 + 0x90 + 8 * (r - 1) - 0x0)));  // psq_st, GQR7 = 0
    }
    w.push_back(Mtlr(28));
    w.push_back(kBlr);
    Place(kProbeFn, w);
    Memory::Write32(kLeaf, kBlr);
    // the patched function's interpreted body: it rewrites forms and values
    Place(kPatchedBody, {Fx(4, 2, 0, 3, 592) /* ps_merge10 f2,f0,f3 */, Fa(59, 3, 2, 3, 0, 21) /* fadds f3,f2,f3 */, kBlr});
}

// Functions the barriers call.
void FnThin(CpuContext* c) {
    c->fpr[2].raw ^= 0x0000000100000000ull;
    c->gpr[3] += 1;
}
void FnPatched(CpuContext* c) {
    const uint32_t at = GuestCodePatch::TestInterpretedAt(kPatched);
    if (at != 0) {
        PpcInterpreter::Run(c, at);
        GuestCodePatch::Sync();
        return;
    }
    c->gpr[3] += 100;
}
void RegisterFunction(uint32_t address, const char* name, void (*fn)(CpuContext*)) {
    TranslatedFunctionInfo info{};
    info.address = address;
    info.name = name;
    info.rawCpuInvoker = fn;
    info.entryPoint = reinterpret_cast<void*>(fn);
    info.kind = FunctionKind::BaseTranslated;
    info.nonvolatileFprWriteMask = 0;
    TranslatedFunctionRegistry::Register(std::move(info));
}

// The barriers between A and B. keep: the knowledge goes on; boundary: B starts a new block.
enum class Barrier { None, IntOp, GenericInt, NotTakenBc, TakenBc, Call, GenericUnknown, ThinCall, PatchedCall, kCount };
const char* const kBarrierNames[] = {"none", "int op", "generic int op", "bc not taken", "bc taken", "bl leaf", "mffs", "thin call", "patched call"};
bool Keeps(Barrier b) { return b == Barrier::None || b == Barrier::IntOp || b == Barrier::GenericInt || b == Barrier::NotTakenBc; }

void EmitBarrier(std::vector<uint32_t>& w, uint32_t at, Barrier b) {
    const auto here = [&] { return at + 4u * static_cast<uint32_t>(w.size()); };
    switch (b) {
    case Barrier::None: break;
    case Barrier::IntOp: w.push_back(D(14, 20, 20, 1)); break;
    case Barrier::GenericInt: w.push_back(D(12, 20, 20, 1)); break;  // addic r20,r20,1
    case Barrier::NotTakenBc:
        w.push_back(Cmpwi(7, 31, 5));                    // r31 == 5
        w.push_back(Bc(4, 4 * 7 + 2, 8));                // bne cr7: not taken
        break;
    case Barrier::TakenBc: w.push_back(Bc(20, 0, 8)); w.push_back(D(14, 21, 21, 1)); break;  // skips an addi
    case Barrier::Call:
        w.push_back(Bl(static_cast<int32_t>(kLeaf - here())));
        w.push_back(Mtlr(28));
        break;
    case Barrier::GenericUnknown: w.push_back(Fx(63, 10, 0, 0, 583)); break;  // mffs f10
    case Barrier::ThinCall:
        w.push_back(Bl(static_cast<int32_t>(kThin - here())));
        w.push_back(Mtlr(28));
        break;
    case Barrier::PatchedCall:
        w.push_back(Bl(static_cast<int32_t>(kPatched - here())));
        w.push_back(Mtlr(28));
        break;
    case Barrier::kCount: break;
    }
}

uint32_t NextCode() {
    static uint32_t slot = 0;
    return kCodeBase + (slot++ % kCodeSlots) * kCodeStep;
}

void Report(const std::string& what, const Snap& want, const Snap& got, const State& st) {
    ++g_failures;
    if (g_quiet || g_failures > 12) return;
    std::fprintf(stderr, "jit forms: %s: differs (forms %08X)\n", what.c_str(), st.pairedMask);
    for (int i = 0; i < 32; ++i)
        if (want.ctx.fpr[i].raw != got.ctx.fpr[i].raw)
            std::fprintf(stderr, "  f%d: want %016llX got %016llX\n", i, static_cast<unsigned long long>(want.ctx.fpr[i].raw),
                         static_cast<unsigned long long>(got.ctx.fpr[i].raw));
    for (size_t i = 0; i < sizeof want.probe; ++i)
        if (want.probe[i] != got.probe[i]) {
            std::fprintf(stderr, "  probe byte %zu: want %02X got %02X\n", i, want.probe[i], got.probe[i]);
            break;
        }
    if (want.threw != got.threw) std::fprintf(stderr, "  threw: want %d got %d\n", want.threw, got.threw);
    std::fputs(JitTestCr::Diff(want.ctx, got.ctx, st.ctx).c_str(), stderr);
}

// Runs a program (words at its address, the probe appended by the caller's branch) from `states` under both engines.
// Returns the form counters' change across the first compile.
JitX64FormStats RunProgram(const std::string& what, const std::vector<uint32_t>& words, uint32_t entry, const std::vector<State>& states) {
    Place(entry, words);
    // the marks the writes of the words set: a block built now is current, so a validity test after a call passes (as it does
    // in a steady state), and the compile counters below count the blocks the program is made of
    std::memset(BrawlDirtyMemory::leanBlocks + ((entry & MemoryInline::kLeanMask) >> 6), 0, (4u * static_cast<uint32_t>(words.size()) >> 6) + 2u);
    ++g_programs;
    JitX64FormStats delta{};
    for (size_t k = 0; k < states.size(); ++k) {
        const Snap want = RunOne(Engine::Classic, entry, states[k]);
        const JitX64FormStats before = JitX64GetFormStats();
        const uint64_t compiledBefore = PpcInterpreter::detail::JitX64GetStats().compiled;
        const Snap got = RunOne(Engine::Jit, entry, states[k]);
        const JitX64FormStats after = JitX64GetFormStats();
        if (k == 0 && std::getenv("FORMS_DEBUG") != nullptr && what.find(std::getenv("FORMS_DEBUG")) != std::string::npos)
            std::fprintf(stderr, "%s: %llu blocks compiled, %llu analysed\n", what.c_str(),
                         static_cast<unsigned long long>(PpcInterpreter::detail::JitX64GetStats().compiled - compiledBefore),
                         static_cast<unsigned long long>(after.blocksAnalysed - before.blocksAnalysed));
        if (k == 0) {
            delta.blocksAnalysed = after.blocksAnalysed - before.blocksAnalysed;
            delta.testsSkipped = after.testsSkipped - before.testsSkipped;
            delta.conversionsForced = after.conversionsForced - before.conversionsForced;
            delta.testsEmitted = after.testsEmitted - before.testsEmitted;
            delta.writesSkipped = after.writesSkipped - before.writesSkipped;
            delta.writesEmitted = after.writesEmitted - before.writesEmitted;
            delta.sharedLoads = after.sharedLoads - before.sharedLoads;
        }
        ++g_runs;
        if (!Same(want, got)) Report(what + " " + g_context, want, got, states[k]);
    }
    return delta;
}

// `prep; [bc to the next block]; A; barrier; B; bc to the probe`. known: the preparation is in A's block. The preparation depends
// on each state's form mask, so the program is made once per distinct mask (at the same address: the block is rebuilt). The
// counters returned are those of the first mask's compile.
void Sequence(const std::string& what, const std::vector<std::vector<uint32_t>>& body, const std::vector<Barrier>& barriers, bool known,
              const std::vector<State>& states, JitX64FormStats* delta = nullptr) {
    const uint32_t at = NextCode();
    std::vector<uint32_t> masks;
    for (const State& s : states)
        if (std::find(masks.begin(), masks.end(), s.pairedMask) == masks.end()) masks.push_back(s.pairedMask);
    bool first = true;
    for (uint32_t mask : masks) {
        std::vector<uint32_t> prog;
        PrepWords(prog, mask);
        if (!known) {  // a jump through ctr to the next op: a block boundary, nothing of the first block after it
            const uint32_t next = at + 4u * (static_cast<uint32_t>(prog.size()) + 4u);
            prog.push_back(D(15, 12, 0, static_cast<int32_t>(next >> 16)));
            prog.push_back(D(24, 12, 12, static_cast<int32_t>(next & 0xFFFFu)));
            prog.push_back(X(12, 9, 0, 467));  // mtctr r12
            prog.push_back(0x4E800420u);       // bctr
        }
        for (size_t part = 0; part < body.size(); ++part) {
            for (uint32_t x : body[part]) prog.push_back(x);
            if (part + 1 < body.size()) EmitBarrier(prog, at, barriers[part]);
        }
        // to the probe block through ctr (a bctr is a block boundary; a b to untranslated code would be followed, and a bc reaches
        // 32 KiB only)
        prog.push_back(D(15, 12, 0, static_cast<int32_t>(kProbeFn >> 16)));
        prog.push_back(D(24, 12, 12, static_cast<int32_t>(kProbeFn & 0xFFFFu)));
        prog.push_back(X(12, 9, 0, 467));  // mtctr r12
        prog.push_back(0x4E800420u);       // bctr
        std::vector<State> some;
        for (const State& s : states)
            if (s.pairedMask == mask) some.push_back(s);
        const JitX64FormStats d = RunProgram(what, prog, at, some);
        if (first && delta != nullptr) *delta = d;
        first = false;
    }
}

std::vector<State> StatesFor(Rng& r, unsigned count) {
    std::vector<State> v;
    for (unsigned i = 0; i < count; ++i) v.push_back(MakeState(r, i));
    return v;
}

uint32_t g_expectFailures = 0;
uint64_t g_probeTests = 0, g_patchedTests = 0, g_patchedForced = 0;
#define CHECK(cond, ...)                                                                              \
    do {                                                                                              \
        if (!(cond)) {                                                                                \
            ++g_failures;                                                                             \
            if (!g_quiet && ++g_expectFailures <= 20) {                                               \
                std::fprintf(stderr, "jit forms: check failed: %s: ", #cond);                         \
                std::fprintf(stderr, __VA_ARGS__);                                                    \
                std::fprintf(stderr, "\n");                                                           \
            }                                                                                         \
        }                                                                                             \
    } while (0)

void ClearCode() {
    for (uint32_t a = kCodeBase; a < kCodeBase + kCodeSlots * kCodeStep; a += 4) Memory::Write32(a, 0);
}

// The matrix: every A (writer) x every B x the aliasing of A's destination with B's sources x the barriers.
void Matrix(bool full) {
    Rng rng(11);
    const uint32_t aliases[4] = {9, 1, 2, 3};
    unsigned pair = 0;
    for (const Kind& a : kKinds) {
        if (a.dform == DForm::N) continue;
        for (const Kind& b : kKinds) {
            ++pair;
            for (uint32_t alias = 0; alias < 4; ++alias) {
                const uint32_t ad = aliases[alias];
                const uint32_t bd = b.store ? 1u : 8u;
                const uint32_t aSrc = SourceRegs(a, ad, 4, 5, 6), bSrc = SourceRegs(b, bd, 1, 2, 3);
                for (int barrier = 0; barrier < static_cast<int>(Barrier::kCount); ++barrier) {
                    const Barrier x = static_cast<Barrier>(barrier);
                    // the barriers that keep the knowledge are cheap: the matrix of them is sampled except for none; the rest by 1/4
                    if (!full && (pair + alias + barrier) % 6 != 0) continue;
                    if (x != Barrier::None && (pair + alias * 3 + barrier) % 4 != 0) continue;
                    const bool known = (pair + barrier) % 7 == 3;
                    g_context = Fmt("[%s ; %s ; %s, d=f%u%s]", a.name, kBarrierNames[barrier], b.name, ad, known ? ", forms known" : "");
                    JitX64FormStats d{};
                    Sequence(g_context, {{a.word(ad, 4, 5, 6)}, {b.word(bd, 1, 2, 3)}}, {x}, known, StatesFor(rng, 4), &d);
                    if (!g_checkCounters || known) continue;
                    // forms not known on entry to A's block: A's sources are tests; B's too, except a source A just wrote when the
                    // block goes on (a barrier that keeps the knowledge) and A's destination form is known
                    const bool destKnown = a.dform == DForm::S || a.dform == DForm::P;
                    const bool alias_ = (Keeps(x) || x == Barrier::TakenBc) && destKnown && ((bSrc >> ad) & 1u) != 0 && b.src != 0;
                    // a quantized store of a known form tests nothing either (a known Scalar goes to the shim without a test: the
                    // form bit in memory may still be pending)
                    const bool skips = alias_;
                    // B is compiled in A's block, and also in a block of its own where the barrier ends the block at run time: after
                    // a taken bc (the compiled block goes on past it; the knowledge with it) and after a call to untranslated code
                    // (which leaves the block: B follows the return, in a new block, and also the call's resume path in the old one)
                    unsigned tests = Popcount(aSrc) + Popcount(bSrc) - (skips ? 1u : 0u);
                    if (x == Barrier::TakenBc || x == Barrier::Call) tests += Popcount(bSrc);
                    tests += static_cast<unsigned>(g_probeTests);
                    unsigned used = skips ? 1u : 0u;
                    if (x == Barrier::PatchedCall) {
                        tests += static_cast<unsigned>(g_patchedTests);
                        used += static_cast<unsigned>(g_patchedForced);
                    }
                    CHECK(d.testsEmitted == tests, "%s: %llu form tests emitted, expected %u", g_context.c_str(),
                          static_cast<unsigned long long>(d.testsEmitted), tests);
                    CHECK(d.testsSkipped + d.conversionsForced == used, "%s: %llu tests skipped or conversions forced, expected %u",
                          g_context.c_str(), static_cast<unsigned long long>(d.testsSkipped + d.conversionsForced), used);
                }
            }
        }
    }
}

// Random chains of float ops over a few registers, with random barriers.
void Chains(unsigned count, uint64_t seed) {
    Rng rng(seed);
    for (unsigned n = 0; n < count; ++n) {
        const unsigned len = 2 + rng.Below(8);
        std::vector<std::vector<uint32_t>> body;
        std::vector<Barrier> barriers;
        std::string what = "chain";
        for (unsigned i = 0; i < len; ++i) {
            const Kind& k = kKinds[rng.Below(kKindCount)];
            const uint32_t d = 1 + rng.Below(6), a = 1 + rng.Below(6), b = 1 + rng.Below(6), c = 1 + rng.Below(6);
            body.push_back({k.word(d, a, b, c)});
            what += Fmt(" %s(f%u,f%u,f%u,f%u)", k.name, d, a, b, c);
            const uint32_t pick = rng.Below(100);
            barriers.push_back(pick < 70 ? Barrier::None : static_cast<Barrier>(1 + rng.Below(static_cast<uint32_t>(Barrier::kCount) - 1)));
        }
        g_context = "";
        Sequence(what, body, barriers, rng.Below(2) != 0, StatesFor(rng, 4));
    }
}

// Longer chains over f1-f12, more registers than the FPR cache has slots (jit_x64_fpr.cpp): slots are evicted and refilled
// while dirty, around slow paths (NaN inputs), barriers that flush the cache (calls, Generic ops, taken branches) and ones that
// keep it (a bc not taken stores the dirty slots and keeps the clean ones).
void WideChains(unsigned count, uint64_t seed) {
    Rng rng(seed);
    for (unsigned n = 0; n < count; ++n) {
        const unsigned len = 6 + rng.Below(9);
        std::vector<std::vector<uint32_t>> body;
        std::vector<Barrier> barriers;
        std::string what = "wide chain";
        for (unsigned i = 0; i < len; ++i) {
            const Kind& k = kKinds[rng.Below(kKindCount)];
            const uint32_t d = 1 + rng.Below(12), a = 1 + rng.Below(12), b = 1 + rng.Below(12), c = 1 + rng.Below(12);
            body.push_back({k.word(d, a, b, c)});
            what += Fmt(" %s(f%u,f%u,f%u,f%u)", k.name, d, a, b, c);
            const uint32_t pick = rng.Below(100);
            barriers.push_back(pick < 85 ? Barrier::None : static_cast<Barrier>(1 + rng.Below(static_cast<uint32_t>(Barrier::kCount) - 1)));
        }
        g_context = "";
        Sequence(what, body, barriers, rng.Below(2) != 0, StatesFor(rng, 4));
    }
}

// Counters that must not fire: a barrier that ends the knowledge, then a source the block does not know.
void Directed() {
    if (!g_checkCounters) return;
    Rng rng(5);
    const Kind& writer = KindNamed("fmuls");
    const Kind& reader = KindNamed("fadds");
    for (int barrier = 0; barrier < static_cast<int>(Barrier::kCount); ++barrier) {
        const Barrier x = static_cast<Barrier>(barrier);
        JitX64FormStats d{};
        g_context = Fmt("[directed %s]", kBarrierNames[barrier]);
        // fmuls f1,f4,f6 ; barrier ; fadds f8,f1,f2: the read of f1 is known only when the barrier keeps the knowledge
        Sequence(g_context, {{writer.word(1, 4, 5, 6)}, {reader.word(8, 1, 2, 3)}}, {x}, false, StatesFor(rng, 4), &d);
        const uint64_t extra = x == Barrier::PatchedCall ? g_patchedForced : 0;
        // a taken bc ends the block at run time, but the compiled block goes on past it with the knowledge
        CHECK(d.testsSkipped + d.conversionsForced == ((Keeps(x) || x == Barrier::TakenBc) ? 1u : 0u) + extra,
              "%s: %llu tests skipped and conversions forced", g_context.c_str(),
              static_cast<unsigned long long>(d.testsSkipped + d.conversionsForced));
    }
    // a conversion forced and a form write skipped: ps_add f1 (Paired) ; fmuls f8,f1,f2 reads f1 as Scalar (a conversion, no test);
    // ps_add f1 twice: the second write of f1's byte is skipped
    JitX64FormStats d{};
    g_context = "[directed conversion]";
    Sequence(g_context, {{KindNamed("ps_add").word(1, 4, 5, 6)}, {KindNamed("fmuls").word(8, 1, 0, 3)}}, {Barrier::None}, false,
             StatesFor(rng, 4), &d);
    CHECK(d.conversionsForced == 1, "%s: %llu conversions forced", g_context.c_str(), static_cast<unsigned long long>(d.conversionsForced));
    g_context = "[directed repeated write]";
    Sequence(g_context, {{KindNamed("ps_add").word(1, 4, 5, 6)}, {KindNamed("ps_mul").word(1, 2, 0, 3)}}, {Barrier::None}, false,
             StatesFor(rng, 4), &d);
    CHECK(d.writesSkipped >= 1, "%s: no form write was skipped", g_context.c_str());
}

void AllPhases(bool full, unsigned chains) {
    ClearCode();
    Matrix(full);
    Chains(chains, 77);
    WideChains(chains / 2, 78);
    Directed();
}

}  // namespace

int main() {
    // The JIT's cache is dropped at every outermost Run (the code limit is one byte): every program is compiled from nothing, so
    // no block of an earlier program at the same address is found, and the counters of a program's first run are its own.
    setenv("BRAWL_JIT_CODE_LIMIT", "1", 1);
    Memory::Init(0x1800000);
    RegisterFunction(kThin, "FormsThin", FnThin);
    RegisterFunction(kPatched, "FormsPatched", FnPatched);
    TranslatedFunctionRegistry::Finalize();
    if (!PpcInterpreter::JitAvailable()) {
        std::printf("jit forms: no JIT backend in this build; nothing to test\n");
        return 0;
    }
    const auto env = [](const char* n) { const char* v = std::getenv(n); return v != nullptr && *v != '\0' && *v != '0'; };
    g_checkCounters = !env("BRAWL_JIT_FALLBACK") && !env("BRAWL_JIT_FP_HELPERS") && !env("BRAWL_JIT_NO_FORMS") &&
                      std::getenv("BRAWL_JIT_FORMS_FAULT") == nullptr;
    WriteSupport();
    GuestCodePatch::TestSetInterpreted(kPatched, kPatchedBody);
    const unsigned chains = std::getenv("FORMS_RANDOM") != nullptr ? static_cast<unsigned>(std::atoi(std::getenv("FORMS_RANDOM"))) : 3000u;
    // The programs end in the shared probe block, whose form tests (and, for a patched call, those of the patched function's
    // interpreted body) every program's counters include: measured with programs of no float tests.
    {
        Rng r(1);
        JitX64FormStats d{};
        Sequence("[base]", {{KindNamed("lfs").word(8, 1, 2, 3)}}, {}, false, StatesFor(r, 1), &d);
        g_probeTests = d.testsEmitted;
        JitX64FormStats v{};
        Sequence("[patched base]", {{KindNamed("lfs").word(8, 1, 2, 3)}, {KindNamed("lfs").word(8, 1, 2, 3)}}, {Barrier::PatchedCall},
                 false, StatesFor(r, 1), &v);
        g_patchedTests = v.testsEmitted - d.testsEmitted;
        g_patchedForced = v.testsSkipped + v.conversionsForced;
        g_failures = 0;
    }
    AllPhases(true, chains);
    const JitX64FormStats stats = JitX64GetFormStats();
    std::printf("jit forms: %llu programs, %llu runs, %llu failures; blocks analysed %llu, tests skipped %llu, conversions forced %llu, "
                "tests emitted %llu, "
                "form writes skipped %llu / emitted %llu, shared loads %llu\n",
                static_cast<unsigned long long>(g_programs), static_cast<unsigned long long>(g_runs), static_cast<unsigned long long>(g_failures),
                static_cast<unsigned long long>(stats.blocksAnalysed), static_cast<unsigned long long>(stats.testsSkipped),
                static_cast<unsigned long long>(stats.conversionsForced), static_cast<unsigned long long>(stats.testsEmitted),
                static_cast<unsigned long long>(stats.writesSkipped), static_cast<unsigned long long>(stats.writesEmitted),
                static_cast<unsigned long long>(stats.sharedLoads));
    uint64_t result = g_failures;
    if (env("BRAWL_JIT_NO_FORMS") || std::getenv("BRAWL_JIT_FORMS_FAULT") != nullptr || env("BRAWL_JIT_FALLBACK")) return result == 0 ? 0 : 1;

    // the analysis made wrong five ways: each must make a program fail
    int missed = 0;
    g_quiet = true;
    g_checkCounters = false;
    // fault 5 (a double result taken for a single) matters only to the inline arithmetic, which forced helpers do not use
    const int faults = env("BRAWL_JIT_FP_HELPERS") ? 4 : 5;
    for (int fault = 1; fault <= faults; ++fault) {
        setenv("BRAWL_JIT_FORMS_FAULT", std::to_string(fault).c_str(), 1);
        const uint64_t before = g_failures;
        AllPhases(false, chains / 3);
        const uint64_t noticed = g_failures - before;
        std::printf("jit forms: fault %d: %llu programs failed\n", fault, static_cast<unsigned long long>(noticed));
        if (noticed == 0) ++missed;
        g_failures = before;
    }
    unsetenv("BRAWL_JIT_FORMS_FAULT");
    g_quiet = false;
    if (missed != 0) {
        std::printf("jit forms: %d of the %d deliberate faults were not noticed\n", missed, faults);
        ++result;
    }
    return result == 0 ? 0 : 1;
}
