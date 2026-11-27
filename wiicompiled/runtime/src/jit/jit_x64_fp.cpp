// Native code for the floating point (Fadds ... PsAbs) of the x86-64 backend, written against the emitter API in jit_x64.h.
//
// The results are the interpreter's bit for bit (docs/jit-design.md, rule 1). Three kinds of code get there:
//
//  * Pure bit movement (fmr, fneg, fabs, ps_mr, ps_neg, ps_abs, the merges, and the form conversions every op does on a
//    source held in the other form) is written out: the same bits RunOp's functions produce, shown by the tests.
//  * Arithmetic whose result is the host instruction's result whenever that result is not a NaN is emitted as the
//    instruction, then a check that sends a NaN result (which operand's NaN the host picks depends on the operand order the
//    compiler gave each copy of an expression, so the interpreter's own copy decides) to the interpreter's own handler, inline at
//    the end of the op. The single-precision ops add the checks that make PpcForceSingleValueBody's flush a no-op (the result is
//    zero or not below the smallest single normal) and send the rest there too; the fused single-precision ones also send the
//    exact-tie case of PpcAccurateSingleMaddIntermediateInline.
//    All of them read both operands in the form the op wants, converted as Scalar() and Paired() convert.
//  * The rest (and every op when BRAWL_JIT_FP_HELPERS is set, or an instruction the build lacks) calls a function below
//    whose body is the expression RunOp evaluates, compiled from the same inline functions with the same options; a NaN result
//    of one with two or more sources goes to the interpreter's handler as well.
//
// The conversions use the instructions the interpreter's conversions compile to (cvtss2sd of ps0, cvtsd2ss twice), under
// the same MXCSR, so a denormal or a NaN converts the same way here.
#include "jit_x64.h"
#include "jit_x64_generic.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "isa/ppc_isa_float.h"
#include "isa/ppc_isa_fpenv.h"

// The same guard as the interpreter's: a guest comparison or operation must keep its exact result.
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "The PowerPC interpreter requires strict floating-point semantics"
#endif

namespace PpcInterpreter::detail::x64 {

// Counts of what each kind's code did, kept only when BRAWL_JIT_FP_COUNTERS is set: [kind][slot], slot 0 completed
// natively, 1 took the interpreter at the end of the op, 2 + i converted operand i, 5 + r the reason r of a slow path
// (see SlowPaths). The unit test reads them.
static uint64_t g_fpCounters[256][12];
uint64_t* JitFpCounters() { return &g_fpCounters[0][0]; }

namespace {

using namespace Xbyak::util;
using Xbyak::Label;

bool EnvSet(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && *v != '0';
}
bool CountersOn() {
    static const bool on = EnvSet("BRAWL_JIT_FP_COUNTERS");
    return on;
}
// Every arithmetic kind calls its helper (the test runs this way too: the helpers are code of their own).
bool HelpersOnly() {
    static const bool on = EnvSet("BRAWL_JIT_FP_HELPERS");
    return on;
}

// --- the helpers: RunOp's expressions on operand values -------------------------------------------------------------------
// A Paired value travels as its 64 raw bits in a double, as RunOp passes it.

double HFadds(double a, double b) noexcept { return static_cast<double>(PpcForceSingleValueInline(PpcFaddInline(a, b))); }
double HFsubs(double a, double b) noexcept { return static_cast<double>(PpcForceSingleValueInline(PpcFsubInline(a, b))); }
double HFmuls(double a, double c) noexcept { return PpcFmulsInline(a, c); }
double HFdivs(double a, double b) noexcept { return static_cast<double>(PpcForceSingleValueInline(PpcFdivInline(a, b))); }
double HFmadds(double a, double c, double b) noexcept { return PpcFmaddsInline(a, c, b); }
double HFmsubs(double a, double c, double b) noexcept { return PpcFmsubsInline(a, c, b); }
double HFnmsubs(double a, double c, double b) noexcept { return PpcFnmsubsInline(a, c, b); }
double HFnmadds(double a, double c, double b) noexcept { return PpcFnmaddsInline(a, c, b); }
double HFadd(double a, double b) noexcept { return PpcFaddInline(a, b); }
double HFsub(double a, double b) noexcept { return PpcFsubInline(a, b); }
double HFmul(double a, double c) noexcept { return PpcFmulInline(a, c); }
double HFdiv(double a, double b) noexcept { return PpcFdivInline(a, b); }
double HFmadd(double a, double c, double b) noexcept { return PpcFmaddInline(a, c, b); }
double HFmsub(double a, double c, double b) noexcept { return PpcFmsubInline(a, c, b); }
double HFnmadd(double a, double c, double b) noexcept { return PpcFnmaddInline(a, c, b); }
double HFnmsub(double a, double c, double b) noexcept { return PpcFnmsubInline(a, c, b); }
double HFres(double b) noexcept { return static_cast<double>(PpcForceSingleValueInline(PpcApproximateReciprocalInline(b))); }
double HFrsqrte(double b) noexcept { return PpcApproximateReciprocalSquareRootInline(b); }
double HFctiw(double b) noexcept { return PPC_Fctiw(b); }
double HFctiwz(double b) noexcept { return PPC_Fctiwz(b); }
double HFrsp(double b) noexcept { return static_cast<double>(PpcForceSingleValueInline(b)); }
double HPsAdd(double a, double b) noexcept { return PPC_PsAddInline(a, b); }
double HPsSub(double a, double b) noexcept { return PPC_PsSubInline(a, b); }
double HPsMul(double a, double c) noexcept { return PPC_PsMulInline(a, c); }
double HPsMadd(double a, double c, double b) noexcept { return PPC_PsMaddInline(a, c, b); }
double HPsMsub(double a, double c, double b) noexcept { return PPC_PsMsubInline(a, c, b); }
double HPsNmadd(double a, double c, double b) noexcept { return PPC_PsNmaddInline(a, c, b); }
double HPsNmsub(double a, double c, double b) noexcept { return PPC_PsNmsubInline(a, c, b); }
double HPsMuls0(double a, double c) noexcept { return PPC_PsMuls0Inline(a, c); }
double HPsMuls1(double a, double c) noexcept { return PPC_PsMuls1Inline(a, c); }
double HPsMadds0(double a, double c, double b) noexcept { return PPC_PsMadds0Inline(a, c, b); }
double HPsMadds1(double a, double c, double b) noexcept { return PPC_PsMadds1Inline(a, c, b); }
double HPsSum0(double a, double b, double c) noexcept { return PPC_PsSum0Inline(a, b, c); }
double HPsSum1(double a, double b, double c) noexcept { return PPC_PsSum1Inline(a, b, c); }

// An arithmetic kind: its operand form, its sources in the order its helper takes them, and the helper.
struct Info {
    bool paired = false;
    unsigned n = 0;
    uint32_t src[3] = {};
    const void* fn = nullptr;
};

bool GetInfo(const Op& o, Info& i) {
    const auto set = [&](bool paired, unsigned n, uint32_t s0, uint32_t s1, uint32_t s2, const void* fn) {
        i.paired = paired;
        i.n = n;
        i.src[0] = s0;
        i.src[1] = s1;
        i.src[2] = s2;
        i.fn = fn;
        return true;
    };
    const uint32_t a = o.a, b = o.b, c = o.imm;  // imm holds frC
    switch (o.kind) {
    case K::Fadds: return set(false, 2, a, b, 0, reinterpret_cast<const void*>(&HFadds));
    case K::Fsubs: return set(false, 2, a, b, 0, reinterpret_cast<const void*>(&HFsubs));
    case K::Fmuls: return set(false, 2, a, c, 0, reinterpret_cast<const void*>(&HFmuls));
    case K::Fdivs: return set(false, 2, a, b, 0, reinterpret_cast<const void*>(&HFdivs));
    case K::Fmadds: return set(false, 3, a, c, b, reinterpret_cast<const void*>(&HFmadds));
    case K::Fmsubs: return set(false, 3, a, c, b, reinterpret_cast<const void*>(&HFmsubs));
    case K::Fnmsubs: return set(false, 3, a, c, b, reinterpret_cast<const void*>(&HFnmsubs));
    case K::Fnmadds: return set(false, 3, a, c, b, reinterpret_cast<const void*>(&HFnmadds));
    case K::Fadd: return set(false, 2, a, b, 0, reinterpret_cast<const void*>(&HFadd));
    case K::Fsub: return set(false, 2, a, b, 0, reinterpret_cast<const void*>(&HFsub));
    case K::Fmul: return set(false, 2, a, c, 0, reinterpret_cast<const void*>(&HFmul));
    case K::Fdiv: return set(false, 2, a, b, 0, reinterpret_cast<const void*>(&HFdiv));
    case K::Fmadd: return set(false, 3, a, c, b, reinterpret_cast<const void*>(&HFmadd));
    case K::Fmsub: return set(false, 3, a, c, b, reinterpret_cast<const void*>(&HFmsub));
    case K::Frsp: return set(false, 1, b, 0, 0, reinterpret_cast<const void*>(&HFrsp));
    case K::PsAdd: return set(true, 2, a, b, 0, reinterpret_cast<const void*>(&HPsAdd));
    case K::PsSub: return set(true, 2, a, b, 0, reinterpret_cast<const void*>(&HPsSub));
    case K::PsMul: return set(true, 2, a, c, 0, reinterpret_cast<const void*>(&HPsMul));
    case K::PsMadd: return set(true, 3, a, c, b, reinterpret_cast<const void*>(&HPsMadd));
    case K::PsMsub: return set(true, 3, a, c, b, reinterpret_cast<const void*>(&HPsMsub));
    case K::PsNmadd: return set(true, 3, a, c, b, reinterpret_cast<const void*>(&HPsNmadd));
    case K::PsNmsub: return set(true, 3, a, c, b, reinterpret_cast<const void*>(&HPsNmsub));
    case K::PsMuls0: return set(true, 2, a, c, 0, reinterpret_cast<const void*>(&HPsMuls0));
    case K::PsMuls1: return set(true, 2, a, c, 0, reinterpret_cast<const void*>(&HPsMuls1));
    case K::PsMadds0: return set(true, 3, a, c, b, reinterpret_cast<const void*>(&HPsMadds0));
    case K::PsMadds1: return set(true, 3, a, c, b, reinterpret_cast<const void*>(&HPsMadds1));
    case K::PsSum0: return set(true, 3, a, b, c, reinterpret_cast<const void*>(&HPsSum0));
    case K::PsSum1: return set(true, 3, a, b, c, reinterpret_cast<const void*>(&HPsSum1));
    default: return false;
    }
}

// --- pieces of native code --------------------------------------------------------------------------------------------------

constexpr unsigned kSlotDone = 0, kSlotSlow = 1, kSlotConv = 2, kSlotReason = 5;
// Why an op goes to the interpreter: a NaN result, a single-precision result in the flushed range, the exact tie of a
// fused single-precision op, a multiplier that is a NaN or a nonzero denormal.
constexpr unsigned kReasonNan = 0, kReasonRange = 1, kReasonTie = 2, kReasonMultiplier = 3;

// The labels an op's checks jump to. With counters on each reason has a stub of its own that counts and joins the common
// path; otherwise every check goes to the common path.
struct SlowPaths {
    Label common;
    Label reason[4];
    bool used[4] = {};
    Label& At(unsigned r) {
        used[r] = true;
        return CountersOn() ? reason[r] : common;
    }
    bool Any() const { return used[0] || used[1] || used[2] || used[3]; }
};
constexpr uint8_t kFormScalar = static_cast<uint8_t>(Form::Scalar);
constexpr uint8_t kFormPaired = static_cast<uint8_t>(Form::Paired);

// One count (r11 is free between ops).
void Count(BlockEmitter& e, K kind, unsigned slot) {
    if (!CountersOn()) return;
    e.a.mov(r11, reinterpret_cast<uint64_t>(&g_fpCounters[static_cast<unsigned>(kind)][slot]));
    e.a.inc(e.a.qword[r11]);
}

// What the block knows of a register's form byte on entry to the op being emitted (jit_x64_forms.cpp). A source whose form is
// known needs no test: Scalar(f, r) of a Scalar register is the raw double, of a Paired one the conversion, unconditionally.
Form Known(const BlockEmitter& e, uint32_t r) { return e.KnownForm(e.index, r); }
using FormEvent = BlockEmitter::FormEvent;

// Scalar(f, r) into the low half of x: the raw double, or ps0 widened when the register is held as a pair
// (PPC_PsToScalarInline: ps0 is the high word, cvtss2sd).
void LoadScalar(BlockEmitter& e, K kind, const Xbyak::Xmm& x, uint32_t r, unsigned slot) {
    auto& a = e.a;
    Label skip;
    a.movsd(x, e.Fpr(r));
    const Form known = Known(e, r);
    if (known == Form::Scalar) {
        BlockEmitter::NoteForm(FormEvent::TestSkipped);
        return;
    }
    if (known == Form::Paired) {
        BlockEmitter::NoteForm(FormEvent::ConversionForced);
        Count(e, kind, slot);
        a.psrlq(x, 32);
        a.cvtss2sd(x, x);
        return;
    }
    BlockEmitter::NoteForm(FormEvent::TestEmitted);
    a.test(e.PairedMask(), 1u << (r & 31u));
    a.jz(skip, Xbyak::CodeGenerator::T_SHORT);
    Count(e, kind, slot);
    a.psrlq(x, 32);
    a.cvtss2sd(x, x);
    a.L(skip);
}

// Paired(f, r): the raw bits, or a register held as a scalar narrowed to both halves (PPC_PsFromScalarInline).
void LoadPaired(BlockEmitter& e, K kind, const Xbyak::Xmm& x, uint32_t r, unsigned slot) {
    auto& a = e.a;
    Label skip;
    a.movsd(x, e.Fpr(r));
    const Form known = Known(e, r);
    if (known == Form::Paired) {
        BlockEmitter::NoteForm(FormEvent::TestSkipped);
        return;
    }
    if (known == Form::Scalar) {
        BlockEmitter::NoteForm(FormEvent::ConversionForced);
        Count(e, kind, slot);
        a.cvtsd2ss(x, x);
        a.shufps(x, x, 0);
        return;
    }
    BlockEmitter::NoteForm(FormEvent::TestEmitted);
    a.test(e.PairedMask(), 1u << (r & 31u));
    a.jnz(skip, Xbyak::CodeGenerator::T_SHORT);
    Count(e, kind, slot);
    a.cvtsd2ss(x, x);
    a.shufps(x, x, 0);
    a.L(skip);
}

// Form bit of d := form, unless the bit is known to hold it already.
void WriteForm(BlockEmitter& e, uint32_t d, uint8_t form) {
    if (static_cast<uint8_t>(Known(e, d)) == form) {
        BlockEmitter::NoteForm(FormEvent::WriteSkipped);
        return;
    }
    BlockEmitter::NoteForm(FormEvent::WriteEmitted);
    const uint32_t bit = 1u << (d & 31u);
    if (form == kFormPaired) e.a.or_(e.PairedMask(), bit);
    else e.a.and_(e.PairedMask(), ~bit);
}

void StoreScalar(BlockEmitter& e, uint32_t d, const Xbyak::Xmm& x) {
    e.a.movsd(e.Fpr(d), x);
    WriteForm(e, d, kFormScalar);
}
void StorePaired(BlockEmitter& e, uint32_t d, const Xbyak::Xmm& x) {
    e.a.movsd(e.Fpr(d), x);
    WriteForm(e, d, kFormPaired);
}

// The operands of an arithmetic op, converted in place into xmm0 ...: a register that two operands name is loaded once.
void LoadOperands(BlockEmitter& e, K kind, const Info& info) {
    for (unsigned i = 0; i < info.n; ++i) {
        const Xbyak::Xmm x(i);
        unsigned j = 0;
        while (j < i && info.src[j] != info.src[i]) ++j;
        if (j < i) {
            e.a.movaps(x, Xbyak::Xmm(j));
            BlockEmitter::NoteForm(FormEvent::SharedLoad);
        } else if (info.paired) {
            LoadPaired(e, kind, x, info.src[i], kSlotConv + i);
        } else {
            LoadScalar(e, kind, x, info.src[i], kSlotConv + i);
        }
    }
}

// The double in xmm0 is a NaN: to `slow`.
void ScalarNanCheck(BlockEmitter& e, SlowPaths& slow) {
    e.a.ucomisd(xmm0, xmm0);
    e.a.jp(slow.At(kReasonNan));
}

// The pair in xmm0 has a NaN lane (the low two lanes are the guest's: PpcPairNanLaneBitsInline): to `slow`.
void PairNanCheck(BlockEmitter& e, SlowPaths& slow) {
    e.a.movaps(xmm3, xmm0);
    e.a.cmpunordps(xmm3, xmm3);
    e.a.movmskps(eax, xmm3);
    e.a.test(al, 3);
    e.a.jnz(slow.At(kReasonNan));
}

// PpcForceSingleValueBody on xmm0, then widened: its flush of a value below the smallest single normal (to the sign alone,
// when NI is set) is the identity for zero and for anything at or above 2^-126, so a magnitude strictly between is sent to
// `slow`; everything else is cvtsd2ss. A NaN (not sent to `slow` here: the callers that can produce one check it first, and
// frsp wants it narrowed) is above the threshold. m - 1 >= 2^-126 - 1 as unsigned is m == 0 or m >= 2^-126. `negate`:
// fnmadds and fnmsubs negate the single result (not a NaN here).
void SingleRound(BlockEmitter& e, SlowPaths& slow, bool negate) {
    auto& a = e.a;
    a.movq(rax, xmm0);
    a.btr(rax, 63);
    a.sub(rax, 1);
    a.mov(rcx, 0x380FFFFFFFFFFFFFull);
    a.cmp(rax, rcx);
    a.jb(slow.At(kReasonRange));
    a.cvtsd2ss(xmm0, xmm0);
    a.cvtss2sd(xmm0, xmm0);
    if (negate) {
        a.movq(rax, xmm0);
        a.btc(rax, 63);
        a.movq(xmm0, rax);
    }
}

// The multiplier c of fmuls and the fused single ops, in xmm1, rounded to a 25-bit significand as PpcForce25BitInline does
// for a value that is neither a NaN (kept as is, and the result is a NaN: `slow`) nor a nonzero denormal (its own arm: `slow`):
// (bits & ~0x7FFFFFF) + (bits & 0x8000000).
void Force25C(BlockEmitter& e, SlowPaths& slow) {
    auto& a = e.a;
    a.movq(rdx, xmm1);
    a.mov(rax, rdx);
    a.btr(rax, 63);
    a.mov(rcx, 0x7FF0000000000000ull);
    a.cmp(rax, rcx);
    a.ja(slow.At(kReasonMultiplier));
    a.sub(rax, 1);
    a.mov(rcx, 0x000FFFFFFFFFFFFFull);
    a.cmp(rax, rcx);
    a.jb(slow.At(kReasonMultiplier));
    a.mov(rax, rdx);
    a.mov(rcx, 0xFFFFFFFFF8000000ull);
    a.and_(rdx, rcx);
    a.and_(rax, 0x8000000);
    a.add(rdx, rax);
    a.movq(xmm1, rdx);
}

// The exact-tie case of PpcAccurateSingleMaddIntermediateInline (the fma result's discarded 29 bits are exactly half): the
// interpreter corrects the direction of the discarded error, so the op goes there.
void TieCheck(BlockEmitter& e, SlowPaths& slow) {
    auto& a = e.a;
    a.movq(rax, xmm0);
    a.mov(ecx, eax);
    a.and_(ecx, 0x1FFFFFFF);
    a.cmp(ecx, 0x10000000);
    a.je(slow.At(kReasonTie));
}

// The sign mask of both words of a pair (or the magnitude mask) in x.
void PairMask(BlockEmitter& e, const Xbyak::Xmm& x, uint64_t value) {
    e.a.mov(rax, value);
    e.a.movq(x, rax);
}

// An arithmetic op as a call of its helper: operands in xmm0, xmm1, xmm2 in the helper's order (converted in place, no call
// in between), the result stored with the destination's form. A NaN result of an op with two or more sources goes to the
// interpreter's handler instead (nothing has been written yet): which operand's NaN the host picks depends on the operand
// order the compiler gave that copy of the expression (the helper's, or the interpreter's own), and only the interpreter's
// copy is the reference.
void EmitHelperCall(BlockEmitter& e, const Op& o, const Info& info) {
    auto& a = e.a;
    LoadOperands(e, o.kind, info);
    e.CallRaw(info.fn);
    if (info.n < 2) {
        if (info.paired) StorePaired(e, o.d, xmm0);
        else StoreScalar(e, o.d, xmm0);
        Count(e, o.kind, kSlotDone);
        return;
    }
    Label slow, done;
    if (info.paired) {
        a.movaps(xmm3, xmm0);
        a.cmpunordps(xmm3, xmm3);
        a.movmskps(eax, xmm3);
        a.test(al, 3);
        a.jnz(slow);
        StorePaired(e, o.d, xmm0);
    } else {
        a.ucomisd(xmm0, xmm0);
        a.jp(slow);
        StoreScalar(e, o.d, xmm0);
    }
    Count(e, o.kind, kSlotDone);
    a.jmp(done);
    a.L(slow);
    Count(e, o.kind, kSlotSlow);
    e.EmitSlowPath(e.index);
    a.L(done);
}

// Inline arithmetic. Emits nothing and returns false when the kind has no inline form in this build.
bool EmitInline(BlockEmitter& e, const Op& o, const Info& info, SlowPaths& slow) {
    auto& a = e.a;
    const K k = o.kind;
    enum class Shape { None, Scalar2, Single2, Fmuls, Frsp, Fma, FmaSingle, Pair2, PairMuls, PairFma } shape = Shape::None;
    enum class Op2 { Add, Sub, Mul, Div } op2 = Op2::Add;
    bool subtract = false, negate = false, broadcast1 = false;
    switch (k) {
    case K::Fadd: shape = Shape::Scalar2; op2 = Op2::Add; break;
    case K::Fsub: shape = Shape::Scalar2; op2 = Op2::Sub; break;
    case K::Fmul: shape = Shape::Scalar2; op2 = Op2::Mul; break;
    case K::Fdiv: shape = Shape::Scalar2; op2 = Op2::Div; break;
    case K::Fadds: shape = Shape::Single2; op2 = Op2::Add; break;
    case K::Fsubs: shape = Shape::Single2; op2 = Op2::Sub; break;
    case K::Fdivs: shape = Shape::Single2; op2 = Op2::Div; break;
    case K::Fmuls: shape = Shape::Fmuls; break;
    case K::Frsp: shape = Shape::Frsp; break;
    case K::PsAdd: shape = Shape::Pair2; op2 = Op2::Add; break;
    case K::PsSub: shape = Shape::Pair2; op2 = Op2::Sub; break;
    case K::PsMul: shape = Shape::Pair2; op2 = Op2::Mul; break;
    case K::PsMuls0: shape = Shape::PairMuls; broadcast1 = false; break;
    case K::PsMuls1: shape = Shape::PairMuls; broadcast1 = true; break;
#if defined(__FMA__)
    case K::Fmadd: shape = Shape::Fma; subtract = false; break;
    case K::Fmsub: shape = Shape::Fma; subtract = true; break;
    case K::Fmadds: shape = Shape::FmaSingle; subtract = false; negate = false; break;
    case K::Fmsubs: shape = Shape::FmaSingle; subtract = true; negate = false; break;
    case K::Fnmadds: shape = Shape::FmaSingle; subtract = false; negate = true; break;
    case K::Fnmsubs: shape = Shape::FmaSingle; subtract = true; negate = true; break;
    case K::PsMadd: shape = Shape::PairFma; subtract = false; negate = false; break;
    case K::PsMsub: shape = Shape::PairFma; subtract = true; negate = false; break;
    case K::PsNmadd: shape = Shape::PairFma; subtract = false; negate = true; break;
    case K::PsNmsub: shape = Shape::PairFma; subtract = true; negate = true; break;
    case K::PsMadds0: shape = Shape::PairFma; subtract = false; negate = false; broadcast1 = false; break;
    case K::PsMadds1: shape = Shape::PairFma; subtract = false; negate = false; broadcast1 = true; break;
#endif
    default: break;
    }
    if (shape == Shape::None) return false;
    const bool broadcast = k == K::PsMuls0 || k == K::PsMuls1 || k == K::PsMadds0 || k == K::PsMadds1;

    LoadOperands(e, k, info);
    if (broadcast) a.shufps(xmm1, xmm1, broadcast1 ? 0x00 : 0x55);  // PpcBroadcastPs1Inline / PpcBroadcastPs0Inline

    const auto arith = [&]() {
        switch (op2) {
        case Op2::Add: if (info.paired) a.addps(xmm0, xmm1); else a.addsd(xmm0, xmm1); break;
        case Op2::Sub: if (info.paired) a.subps(xmm0, xmm1); else a.subsd(xmm0, xmm1); break;
        case Op2::Mul: if (info.paired) a.mulps(xmm0, xmm1); else a.mulsd(xmm0, xmm1); break;
        case Op2::Div: a.divsd(xmm0, xmm1); break;
        }
    };

    switch (shape) {
    case Shape::Scalar2:
        arith();
        ScalarNanCheck(e, slow);
        break;
    case Shape::Single2:
        arith();
        ScalarNanCheck(e, slow);
        SingleRound(e, slow, false);
        break;
    case Shape::Fmuls:
        Force25C(e, slow);
        a.mulsd(xmm0, xmm1);
        ScalarNanCheck(e, slow);
        SingleRound(e, slow, false);
        break;
    case Shape::Frsp:
        SingleRound(e, slow, false);
        break;
    case Shape::Fma:
        if (subtract) a.vfmsub213sd(xmm0, xmm1, xmm2);
        else a.vfmadd213sd(xmm0, xmm1, xmm2);
        ScalarNanCheck(e, slow);
        break;
    case Shape::FmaSingle:
        Force25C(e, slow);
        if (subtract) a.vfmsub213sd(xmm0, xmm1, xmm2);
        else a.vfmadd213sd(xmm0, xmm1, xmm2);
        ScalarNanCheck(e, slow);
        TieCheck(e, slow);
        SingleRound(e, slow, negate);
        break;
    case Shape::Pair2:
        arith();
        PairNanCheck(e, slow);
        break;
    case Shape::PairMuls:
        a.mulps(xmm0, xmm1);
        PairNanCheck(e, slow);
        break;
    case Shape::PairFma:
        if (subtract) a.vfmsub213ps(xmm0, xmm1, xmm2);
        else a.vfmadd213ps(xmm0, xmm1, xmm2);
        PairNanCheck(e, slow);
        if (negate) {  // no lane is a NaN here, so every lane is negated, as PpcNegateNonNanLanesInline does
            PairMask(e, xmm3, 0x8000000080000000ull);
            a.xorps(xmm0, xmm3);
        }
        break;
    case Shape::None: break;
    }
    if (info.paired) StorePaired(e, o.d, xmm0);
    else StoreScalar(e, o.d, xmm0);
    Count(e, k, kSlotDone);
    return true;
}

}  // namespace

uint32_t EmitFp(BlockEmitter& e, uint32_t index) {
    const Op& o = e.block.ops[index];
    auto& a = e.a;
    const K k = o.kind;
    switch (k) {
    case K::Fmr:
    case K::PsMr: {  // copies the raw bits and the source's form (every register is Scalar or Paired at run time)
        const Form known = Known(e, o.b);
        if (o.d == o.b) {
            Count(e, k, kSlotDone);
            return 1;
        }
        a.mov(rax, e.Fpr(o.b));
        a.mov(e.Fpr(o.d), rax);
        if (known != Form::Unknown) {
            WriteForm(e, o.d, static_cast<uint8_t>(known));
        } else {
            a.mov(ecx, e.PairedMask());
            a.bt(ecx, static_cast<uint8_t>(o.b & 31u));
            a.sbb(edx, edx);
            a.xor_(edx, ecx);
            a.and_(edx, 1u << (o.d & 31u));
            a.xor_(ecx, edx);
            a.mov(e.PairedMask(), ecx);
        }
        Count(e, k, kSlotDone);
        return 1;
    }
    case K::Fneg:
    case K::Fabs: {  // -Scalar(b), fabs(Scalar(b)): the sign bit of the double, set or cleared, a NaN's too
        Label skip;
        const Form known = Known(e, o.b);
        a.mov(rax, e.Fpr(o.b));
        if (known != Form::Scalar) {
            if (known == Form::Unknown) {
                BlockEmitter::NoteForm(FormEvent::TestEmitted);
                a.test(e.PairedMask(), 1u << (o.b & 31u));
                a.jz(skip, Xbyak::CodeGenerator::T_SHORT);
            } else {
                BlockEmitter::NoteForm(FormEvent::ConversionForced);
            }
            Count(e, k, kSlotConv);
            a.movq(xmm0, rax);
            a.psrlq(xmm0, 32);
            a.cvtss2sd(xmm0, xmm0);
            a.movq(rax, xmm0);
            a.L(skip);
        } else {
            BlockEmitter::NoteForm(FormEvent::TestSkipped);
        }
        if (k == K::Fneg) a.btc(rax, 63);
        else a.btr(rax, 63);
        a.mov(e.Fpr(o.d), rax);
        WriteForm(e, o.d, kFormScalar);
        Count(e, k, kSlotDone);
        return 1;
    }
    case K::Fcmpu: {  // PpcCrvCompareFloat: the value of nibble 8 (less), 4 (greater), 2 (equal) or 1 (unordered)
        if (e.CrDead(index, o.d >> 2)) {  // nothing reads the field before a later op overwrites it (BlockEmitter::CrDead)
            BlockEmitter::NoteCr(BlockEmitter::CrEvent::WriteSkipped);
            return 1;
        }
        LoadScalar(e, k, xmm0, o.a, kSlotConv);
        if (o.b == o.a) {
            a.movaps(xmm1, xmm0);
            BlockEmitter::NoteForm(FormEvent::SharedLoad);
        } else {
            LoadScalar(e, k, xmm1, o.b, kSlotConv + 1);
        }
        a.ucomisd(xmm0, xmm1);  // above: CF = ZF = 0; below: CF; unordered: ZF = PF = CF = 1
        a.mov(rax, static_cast<uint64_t>(kPpcCrvEq));
        static_assert(kPpcCrvGt >= 0 && kPpcCrvGt <= 0xFFFFFFFF);
        a.mov(ecx, static_cast<uint32_t>(kPpcCrvGt));
        a.mov(rdx, static_cast<uint64_t>(kPpcCrvLt));
        a.mov(rsi, static_cast<uint64_t>(kPpcCrvSo));
        a.cmova(rax, rcx);
        a.cmovb(rax, rdx);
        a.cmovp(rax, rsi);
        a.mov(e.Crv(o.d >> 2), rax);
        Count(e, k, kSlotDone);
        return 1;
    }
    case K::PsNeg:
    case K::PsAbs:  // the sign bit of each single (std::abs and unary minus on floats: no arithmetic)
        LoadPaired(e, k, xmm0, o.b, kSlotConv);
        if (k == K::PsNeg) {
            PairMask(e, xmm1, 0x8000000080000000ull);
            a.xorps(xmm0, xmm1);
        } else {
            PairMask(e, xmm1, 0x7FFFFFFF7FFFFFFFull);
            a.andps(xmm0, xmm1);
        }
        StorePaired(e, o.d, xmm0);
        Count(e, k, kSlotDone);
        return 1;
    case K::PsMerge00:
    case K::PsMerge01:
    case K::PsMerge10:
    case K::PsMerge11: {
        // result.ps0 (lane 1) from a's ps0 (lane 1) or ps1 (lane 0), result.ps1 (lane 0) from b's
        const unsigned aLane = (k == K::PsMerge00 || k == K::PsMerge01) ? 1u : 0u;
        const unsigned bLane = (k == K::PsMerge00 || k == K::PsMerge10) ? 1u : 0u;
        LoadPaired(e, k, xmm0, o.a, kSlotConv);
        LoadPaired(e, k, xmm1, o.b, kSlotConv + 1);
        a.shufps(xmm1, xmm0, static_cast<uint8_t>(bLane | bLane << 2 | aLane << 4 | aLane << 6));  // { b[bLane] x2, a[aLane] x2 }
        a.shufps(xmm1, xmm1, 0x08);                                                                 // lanes 0 and 2
        StorePaired(e, o.d, xmm1);
        Count(e, k, kSlotDone);
        return 1;
    }
    default: break;
    }

    Info info;
    if (!GetInfo(o, info)) return 0;
    SlowPaths slow;
    Label done;
    if (HelpersOnly() || !EmitInline(e, o, info, slow)) {
        EmitHelperCall(e, o, info);
        return 1;
    }
    if (slow.Any()) {
        // A NaN result, a single-precision result in the flushed range, a tie: the interpreter's own handler, which reads
        // the operands again (nothing has been written).
        a.jmp(done);
        if (CountersOn()) {
            for (unsigned r = 0; r < 4; ++r) {
                if (!slow.used[r]) continue;
                a.L(slow.reason[r]);
                Count(e, k, kSlotReason + r);
                a.jmp(slow.common);
            }
        }
        a.L(slow.common);
        Count(e, k, kSlotSlow);
        e.EmitSlowPath(index);
        a.L(done);
    }
    return 1;
}

// The float ops Decode leaves Generic: fsel, fnabs, fnmadd, fnmsub, fres, frsqrte, fctiw, fctiwz (Rc clear: the forms with Rc
// set are the interpreter's). Each reads its sources as Scalar(f, r) and sets the destination Scalar, as Exec's cases do.
uint32_t EmitFpGeneric(BlockEmitter& e, uint32_t w) {
    const uint32_t po = w >> 26, d = (w >> 21) & 31, a = (w >> 16) & 31, b = (w >> 11) & 31, c = (w >> 6) & 31;
    const uint32_t xo5 = (w >> 1) & 31, xo10 = (w >> 1) & 1023;
    if ((w & 1u) != 0) return 0;
    const Op o{K::Generic, static_cast<uint8_t>(d), static_cast<uint8_t>(a), static_cast<uint8_t>(b), c};
    auto& as = e.a;
    Info info;
    if (po == 59) {
        if (xo5 != 24) return 0;
        info.paired = false;
        info.n = 1;
        info.src[0] = b;
        info.fn = reinterpret_cast<const void*>(&HFres);
        EmitHelperCall(e, o, info);
        return 1;
    }
    if (po != 63) return 0;
    if (xo5 == 23) {  // fsel: a >= 0.0 ? c : b (a NaN picks b)
        LoadScalar(e, K::Generic, xmm0, a, kSlotConv);
        LoadScalar(e, K::Generic, xmm1, c, kSlotConv + 1);
        LoadScalar(e, K::Generic, xmm2, b, kSlotConv + 2);
        as.xorpd(xmm3, xmm3);
        as.ucomisd(xmm0, xmm3);  // CF is set when a < 0 and when unordered
        as.movq(rax, xmm1);
        as.movq(rcx, xmm2);
        as.cmovb(rax, rcx);
        as.mov(e.Fpr(d), rax);
        WriteForm(e, d, kFormScalar);
        return 1;
    }
    if (xo5 == 30 || xo5 == 31) {
        info.paired = false;
        info.n = 3;
        info.src[0] = a;
        info.src[1] = c;
        info.src[2] = b;
        info.fn = reinterpret_cast<const void*>(xo5 == 30 ? &HFnmsub : &HFnmadd);
        EmitHelperCall(e, o, info);
        return 1;
    }
    if (xo10 == 136) {  // fnabs: -fabs(Scalar(b)): the sign bit set, a NaN's too
        LoadScalar(e, K::Generic, xmm0, b, kSlotConv);
        as.movq(rax, xmm0);
        as.bts(rax, 63);
        as.mov(e.Fpr(d), rax);
        WriteForm(e, d, kFormScalar);
        return 1;
    }
    const void* fn = nullptr;
    switch (xo10) {
    case 14: fn = reinterpret_cast<const void*>(&HFctiw); break;
    case 15: fn = reinterpret_cast<const void*>(&HFctiwz); break;
    case 26: fn = reinterpret_cast<const void*>(&HFrsqrte); break;
    default: return 0;
    }
    info.paired = false;
    info.n = 1;
    info.src[0] = b;
    info.fn = fn;
    EmitHelperCall(e, o, info);
    return 1;
}

}  // namespace PpcInterpreter::detail::x64
