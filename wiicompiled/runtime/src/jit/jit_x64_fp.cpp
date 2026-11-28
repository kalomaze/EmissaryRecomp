// Native code for the floating point (Fadds ... PsAbs) of the x86-64 backend, written against the emitter API in jit_x64.h.
//
// The results are the interpreter's bit for bit (docs/jit-design.md, rule 1). Three kinds of code get there:
//
//  * Pure bit movement (fmr, fneg, fabs, ps_mr, ps_neg, ps_abs, the merges, and the form conversions every op does on a
//    source held in the other form) is written out: the same bits RunOp's functions produce, shown by the tests.
//  * Arithmetic whose result is the host instruction's result whenever that result is not a NaN is emitted as the
//    instruction, then a check that sends a NaN result (which operand's NaN the host picks depends on the operand order the
//    compiler gave each copy of an expression, so the interpreter's own copy decides) to the interpreter's own handler, out of
//    line. The single-precision ops add the checks that make PpcForceSingleValueBody's flush a no-op (the result is zero or not
//    below the smallest single normal) and send the rest there too; the fused single-precision ones also send the exact-tie case
//    of PpcAccurateSingleMaddIntermediateInline, and fmuls and the fused single ones a multiplier that PpcForce25BitInline would
//    change. All of them read their operands in the form the op wants, converted as Scalar() and Paired() convert.
//  * The rest (and every op when BRAWL_JIT_FP_HELPERS is set, or an instruction the build lacks) calls a function below
//    whose body is the expression RunOp evaluates, compiled from the same inline functions with the same options; a NaN result
//    of one with two or more sources goes to the interpreter's handler as well.
//
// What the block knows lets checks go (jit_x64_forms.cpp): a source whose form is known needs no test, and one known to be a
// single-precision value (BlockEmitter::KnownSingle) needs no 25-bit rounding as a multiplier (PpcForce25BitInline leaves it as
// it is), and a sum or difference of two of them needs no flush check: such a sum below 2^-126 is a float denormal exactly
// (both are multiples of 2^-149), which cvtsd2ss returns as it is with NI clear, and flushes to the signed zero with NI set
// (FTZ), as PpcForceSingleValueBody's mask does.
//
// Operands come from the FPR cache (jit_x64_fpr.cpp) and results go back to it: a register the block reads again stays in its
// xmm slot. An op's slow path stores the cache's snapshot (dirty slots, pending forms, dirty GPRs), runs the interpreter's
// handler, reloads the snapshot and the destination's slot, and rejoins.
//
// The conversions use the instructions the interpreter's conversions compile to (cvtss2sd of ps0, cvtsd2ss twice), under
// the same MXCSR, so a denormal or a NaN converts the same way here.
#include "jit_x64.h"
#include "jit_x64_generic.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "isa/ppc_isa_float.h"
#include "isa/ppc_isa_fpenv.h"

// The same guard as the interpreter's: a guest comparison or operation must keep its exact result.
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "The PowerPC interpreter requires strict floating-point semantics"
#endif

namespace PpcInterpreter::detail::x64 {

// Counts of what each kind's code did, kept only when BRAWL_JIT_FP_COUNTERS is set: [kind][slot], slot 0 completed
// natively, 1 took the interpreter, 2 + i converted operand i, 5 + r the reason r of a slow path (see the reasons below).
// The unit test reads them.
static uint64_t g_fpCounters[256][12];
uint64_t* JitFpCounters() { return &g_fpCounters[0][0]; }

namespace {

using namespace Xbyak::util;
using Xbyak::Label;
using Xmm = Xbyak::Xmm;
using CG = Xbyak::CodeGenerator;
using FormEvent = BlockEmitter::FormEvent;

#if defined(__AVX__)
constexpr bool kAvx = true;
#else
constexpr bool kAvx = false;
#endif

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
// BRAWL_JIT_NO_FUSED_FCMP=1 (a test switch): an fcmpu always writes its field and the bc after it reads the field in memory.
bool FusedFcmpEnabled() {
    static const bool on = !EnvSet("BRAWL_JIT_NO_FUSED_FCMP");
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

// --- the SSE / AVX instructions used, in one place ---------------------------------------------------------------------------
// With AVX the three-operand forms; without, the two-operand forms after a copy (d must then differ from the second source
// unless it is the first).

void Mov(CG& a, const Xmm& d, const Xmm& s) {
    if (d.getIdx() == s.getIdx()) return;
    if (kAvx) a.vmovaps(d, s);
    else a.movaps(d, s);
}
void LoadQ(CG& a, const Xmm& d, const Xbyak::Address& m) {  // clears the upper quadword
    if (kAvx) a.vmovsd(d, m);
    else a.movsd(d, m);
}
void StoreQ(CG& a, const Xbyak::Address& m, const Xmm& s) {
    if (kAvx) a.vmovsd(m, s);
    else a.movsd(m, s);
}
void LoadS(CG& a, const Xmm& d, const Xbyak::Address& m) {
    if (kAvx) a.vmovss(d, m);
    else a.movss(d, m);
}
void MovqToGpr(CG& a, const Xbyak::Reg64& r, const Xmm& s) {
    if (kAvx) a.vmovq(r, s);
    else a.movq(r, s);
}
void MovqFromGpr(CG& a, const Xmm& d, const Xbyak::Reg64& r) {
    if (kAvx) a.vmovq(d, r);
    else a.movq(d, r);
}
void CvtSs2Sd(CG& a, const Xmm& d, const Xmm& s) {
    if (kAvx) a.vcvtss2sd(d, s, s);
    else a.cvtss2sd(d, s);
}
void CvtSd2Ss(CG& a, const Xmm& d, const Xmm& s) {
    if (kAvx) a.vcvtsd2ss(d, s, s);
    else a.cvtsd2ss(d, s);
}
void Psrlq32(CG& a, const Xmm& d, const Xmm& s) {  // ps0 (the high word) to the low word
    if (kAvx) {
        a.vpsrlq(d, s, 32);
        return;
    }
    Mov(a, d, s);
    a.psrlq(d, 32);
}
void Shufps(CG& a, const Xmm& d, const Xmm& s1, const Xmm& s2, uint8_t imm) {
    if (kAvx) {
        a.vshufps(d, s1, s2, imm);
        return;
    }
    Mov(a, d, s1);
    a.shufps(d, s2, imm);
}
void Ucomisd(CG& a, const Xmm& x, const Xmm& y) {
    if (kAvx) a.vucomisd(x, y);
    else a.ucomisd(x, y);
}
enum class Bin { AddSd, SubSd, MulSd, DivSd, AddPs, SubPs, MulPs, XorPd, AndPd, OrPd, XorPs, AndPs, Unpcklps };
void Binary(CG& a, Bin op, const Xmm& d, const Xmm& x, const Xbyak::Operand& y) {
    if (kAvx) {
        switch (op) {
        case Bin::AddSd: a.vaddsd(d, x, y); break;
        case Bin::SubSd: a.vsubsd(d, x, y); break;
        case Bin::MulSd: a.vmulsd(d, x, y); break;
        case Bin::DivSd: a.vdivsd(d, x, y); break;
        case Bin::AddPs: a.vaddps(d, x, y); break;
        case Bin::SubPs: a.vsubps(d, x, y); break;
        case Bin::MulPs: a.vmulps(d, x, y); break;
        case Bin::XorPd: a.vxorpd(d, x, y); break;
        case Bin::AndPd: a.vandpd(d, x, y); break;
        case Bin::OrPd: a.vorpd(d, x, y); break;
        case Bin::XorPs: a.vxorps(d, x, y); break;
        case Bin::AndPs: a.vandps(d, x, y); break;
        case Bin::Unpcklps: a.vunpcklps(d, x, y); break;
        }
        return;
    }
    Mov(a, d, x);
    switch (op) {
    case Bin::AddSd: a.addsd(d, y); break;
    case Bin::SubSd: a.subsd(d, y); break;
    case Bin::MulSd: a.mulsd(d, y); break;
    case Bin::DivSd: a.divsd(d, y); break;
    case Bin::AddPs: a.addps(d, y); break;
    case Bin::SubPs: a.subps(d, y); break;
    case Bin::MulPs: a.mulps(d, y); break;
    case Bin::XorPd: a.xorpd(d, y); break;
    case Bin::AndPd: a.andpd(d, y); break;
    case Bin::OrPd: a.orpd(d, y); break;
    case Bin::XorPs: a.xorps(d, y); break;
    case Bin::AndPs: a.andps(d, y); break;
    case Bin::Unpcklps: a.unpcklps(d, y); break;
    }
}

// --- forward jumps into the op's cold code, without labels (the block's label pool is small; jit_x64_mem.cpp does the same) ---

uint32_t Here(CG& a) { return static_cast<uint32_t>(a.getSize()); }
// jcc rel32 with a zero displacement; the cold code resolves it.
uint32_t JccPatch(CG& a, uint8_t cc) {
    a.db(0x0F);
    a.db(0x80 | cc);
    const uint32_t at = Here(a);
    a.dd(0);
    return at;
}
void ResolveAt(CG& a, uint32_t at) {
    uint8_t* const base = const_cast<uint8_t*>(a.getCode());
    const int32_t rel = static_cast<int32_t>(a.getSize() - (at + 4));
    std::memcpy(base + at, &rel, sizeof(rel));
}
void JmpTo(CG& a, uint32_t offset) { a.jmp(static_cast<const void*>(a.getCode() + offset), CG::T_NEAR); }
enum Cc : uint8_t { kCcB = 0x2, kCcE = 0x4, kCcNe = 0x5, kCcA = 0x7, kCcP = 0xA };

// --- counters ----------------------------------------------------------------------------------------------------------------

constexpr unsigned kSlotDone = 0, kSlotSlow = 1, kSlotConv = 2, kSlotReason = 5;
// Why an op goes to the interpreter: a NaN result, a single-precision result in the flushed range, the exact tie of a
// fused single-precision op, a multiplier that PpcForce25BitInline would change (or a NaN one).
constexpr unsigned kReasonNan = 0, kReasonRange = 1, kReasonTie = 2, kReasonMultiplier = 3;

// One count (rcx: the counts are placed where no scratch value is live and the flags are free).
void Count(BlockEmitter& e, K kind, unsigned slot) {
    if (!CountersOn()) return;
    e.a.mov(rcx, reinterpret_cast<uint64_t>(&g_fpCounters[static_cast<unsigned>(kind)][slot]));
    e.a.inc(e.a.qword[rcx]);
}

// What the block knows of a register's form on entry to the op being emitted (jit_x64_forms.cpp). A source whose form is
// known needs no test: Scalar(f, r) of a Scalar register is the raw double, of a Paired one the conversion, unconditionally.
Form Known(const BlockEmitter& e, uint32_t r) { return e.KnownForm(e.index, r); }

// --- operands and results ----------------------------------------------------------------------------------------------------

// Where FPR r's raw bits are for the op: its slot when cached, or loaded into one when the block reads r again (`cache`, and a
// slot can be had without taking one of `pin`'s), else `scratch`, loaded from memory.
Xmm RawSrc(BlockEmitter& e, uint32_t r, const Xmm& scratch, uint32_t pin, bool cache = true) {
    if (e.FindCachedFpr(r) >= 0 || (cache && e.FprReadLater(r) && e.FprSlotFree(pin))) return e.ReadFpr(r, pin);
    LoadQ(e.a, scratch, e.Fpr(r));
    return scratch;
}

// Scalar(f, r) in a register: r's slot when r holds a known Scalar, else `scratch` (a known Paired register converted; an
// unknown one tested and converted when it holds a pair: PPC_PsToScalarInline, ps0 is the high word, cvtss2sd).
Xmm ScalarSrc(BlockEmitter& e, K kind, uint32_t r, const Xmm& scratch, uint32_t pin, unsigned slot, bool cache = true) {
    CG& a = e.a;
    const Form known = Known(e, r);
    if (known == Form::Scalar) {
        BlockEmitter::NoteForm(FormEvent::TestSkipped);
        return RawSrc(e, r, scratch, pin, cache);
    }
    if (known == Form::Paired) {
        BlockEmitter::NoteForm(FormEvent::ConversionForced);
        Count(e, kind, slot);
        if (e.FindCachedFpr(r) >= 0 || (cache && e.FprReadLater(r) && e.FprSlotFree(pin))) Psrlq32(a, scratch, e.ReadFpr(r, pin));
        else LoadS(a, scratch, e.FprWord(r, 1));  // ps0 alone
        CvtSs2Sd(a, scratch, scratch);
        return scratch;
    }
    BlockEmitter::NoteForm(FormEvent::TestEmitted);
    Mov(a, scratch, RawSrc(e, r, scratch, pin, cache));
    Label skip;
    a.test(e.PairedMask(), 1u << (r & 31u));
    a.jz(skip, CG::T_SHORT);
    Count(e, kind, slot);
    Psrlq32(a, scratch, scratch);
    CvtSs2Sd(a, scratch, scratch);
    a.L(skip);
    return scratch;
}

// Paired(f, r) in a register: r's slot when r holds a known pair, else `scratch` (a register held as a scalar narrowed to both
// halves: PPC_PsFromScalarInline).
Xmm PairedSrc(BlockEmitter& e, K kind, uint32_t r, const Xmm& scratch, uint32_t pin, unsigned slot, bool cache = true) {
    CG& a = e.a;
    const Form known = Known(e, r);
    if (known == Form::Paired) {
        BlockEmitter::NoteForm(FormEvent::TestSkipped);
        return RawSrc(e, r, scratch, pin, cache);
    }
    if (known == Form::Scalar) {
        BlockEmitter::NoteForm(FormEvent::ConversionForced);
        Count(e, kind, slot);
        CvtSd2Ss(a, scratch, RawSrc(e, r, scratch, pin, cache));
        Shufps(a, scratch, scratch, scratch, 0);
        return scratch;
    }
    BlockEmitter::NoteForm(FormEvent::TestEmitted);
    Mov(a, scratch, RawSrc(e, r, scratch, pin, cache));
    Label skip;
    a.test(e.PairedMask(), 1u << (r & 31u));
    a.jnz(skip, CG::T_SHORT);
    Count(e, kind, slot);
    CvtSd2Ss(a, scratch, scratch);
    Shufps(a, scratch, scratch, scratch, 0);
    a.L(skip);
    return scratch;
}

// The result becomes FPR d, with form `form`: `emit(x)` puts it in xmm x, which is d's slot when the block reads d again (and a
// slot can be had without taking one of `pin`'s), else `scratch`, stored to memory (d's slot, if any, dropped). Answers d's
// slot afterwards, or -1.
template <typename EmitTo>
int WriteResult(BlockEmitter& e, uint32_t d, Form form, const Xmm& scratch, uint32_t pin, const EmitTo& emit) {
    return e.WriteFpr(d, form, scratch, pin, emit);
}
int WriteResult(BlockEmitter& e, uint32_t d, Form form, const Xmm& x, uint32_t pin = 0) {
    return WriteResult(e, d, form, x, pin, [&](const Xmm& to) { Mov(e.a, to, x); });
}

// --- the slow path -----------------------------------------------------------------------------------------------------------

// The jumps of an op's checks to its slow path, and why each goes there.
struct SlowJumps {
    static constexpr unsigned kMax = 6;
    uint32_t at[kMax];
    uint8_t reason[kMax];
    uint8_t n = 0;
    void Jcc(CG& a, uint8_t cc, unsigned r) {
        if (n == kMax) throw Xbyak::Error(Xbyak::ERR_BAD_PARAMETER);
        at[n] = JccPatch(a, cc);
        reason[n++] = static_cast<uint8_t>(r);
    }
    bool Any() const { return n != 0; }
};

// The interpreter's handler for the op (it reads the operands again: nothing of the op has been written to memory), with the
// cache stored around it: the snapshot taken where the checks branch off, reloaded after, and the destination's slot (the one
// the main path gives it at the join) loaded from what the handler wrote.
struct SlowCold {
    SlowJumps jumps;
    BlockEmitter::GprSnapshot snap;
    uint32_t join, index;
    int8_t dSlot;
    uint8_t d;
    K kind;
    void operator()(BlockEmitter& e) const {
        CG& a = e.a;
        Label common;
        if (CountersOn()) {
            for (unsigned r = 0; r < 4; ++r) {
                bool any = false;
                for (unsigned i = 0; i < jumps.n; ++i)
                    if (jumps.reason[i] == r) {
                        ResolveAt(a, jumps.at[i]);
                        any = true;
                    }
                if (!any) continue;
                Count(e, kind, kSlotReason + r);
                a.jmp(common, CG::T_NEAR);
            }
        } else {
            for (unsigned i = 0; i < jumps.n; ++i) ResolveAt(a, jumps.at[i]);
        }
        a.L(common);
        Count(e, kind, kSlotSlow);
        e.EmitFlushSnapshot(snap);
        e.EmitSlowPath(index);
        e.EmitReloadSnapshotVolatile(snap);
        if (dSlot >= 0 && snap.fpr[dSlot] != static_cast<int8_t>(d)) LoadQ(a, BlockEmitter::FprCacheReg(static_cast<uint32_t>(dSlot)), e.Fpr(d));
        JmpTo(a, join);
    }
};

void EmitSlowCold(BlockEmitter& e, const SlowJumps& jumps, const BlockEmitter::GprSnapshot& snap, int dSlot, uint32_t d, K kind) {
    if (!jumps.Any()) return;
    SlowCold c{};
    c.jumps = jumps;
    c.snap = snap;
    c.join = Here(e.a);
    c.index = e.index;
    c.dSlot = static_cast<int8_t>(dSlot);
    c.d = static_cast<uint8_t>(d);
    c.kind = kind;
    e.Cold(c);
}

// --- checks ------------------------------------------------------------------------------------------------------------------

// The double in x is a NaN: to the slow path.
void ScalarNanCheck(BlockEmitter& e, const Xmm& x, SlowJumps& slow) {
    Ucomisd(e.a, x, x);
    slow.Jcc(e.a, kCcP, kReasonNan);
}

// The pair in x has a NaN lane (the low two lanes are the guest's: PpcPairNanLaneBitsInline): to the slow path. tmp: a scratch.
void PairNanCheck(BlockEmitter& e, const Xmm& x, const Xmm& tmp, SlowJumps& slow) {
    CG& a = e.a;
    if (kAvx) {
        a.vcmpunordps(tmp, x, x);
        a.vmovmskps(eax, tmp);
    } else {
        a.movaps(tmp, x);
        a.cmpunordps(tmp, tmp);
        a.movmskps(eax, tmp);
    }
    a.test(al, 3);
    slow.Jcc(a, kCcNe, kReasonNan);
}

// PpcForceSingleValueBody's flush (to the sign alone, when NI is set) is the identity for zero and for anything at or above
// 2^-126, so a magnitude strictly between goes to the slow path; the rest is cvtsd2ss. A NaN (sent to the slow path before this
// by the callers that can produce one; frsp wants it narrowed) is above the threshold. With BMI2: ecx = bits 62..31 of the double
// (the magnitude's top 32 bits), and ecx - 1 < 0x701FFFFF exactly when the magnitude is in [2^31, 2^-126 as bits); a magnitude
// below 2^31 as bits (a double subnormal) is let through: cvtsd2ss gives its signed zero both with NI set (DAZ) and as the mask
// does, and with NI clear the mask keeps the value and the interpreter's conversion is this same instruction. `bitsInRax`: the
// caller has the double's bits in rax already.
void RangeCheck(BlockEmitter& e, const Xmm& x, SlowJumps& slow, bool bitsInRax = false) {
    CG& a = e.a;
    if (!bitsInRax) MovqToGpr(a, rax, x);
#if defined(__BMI2__)
    a.rorx(rcx, rax, 31);
    a.sub(ecx, 1);
    a.cmp(ecx, 0x701FFFFFu);
    slow.Jcc(a, kCcB, kReasonRange);
#else
    a.mov(rcx, rax);
    a.btr(rcx, 63);
    a.sub(rcx, 1);
    a.mov(rdx, 0x380FFFFFFFFFFFFFull);
    a.cmp(rcx, rdx);
    slow.Jcc(a, kCcB, kReasonRange);
#endif
}

// A multiplier c that PpcForce25BitInline would change: bits 0..27 not all clear (a value with them clear is left as it is by
// both its arms, and a NaN is not rounded at all), to the slow path. Free when the block knows c is a single.
void MultiplierCheck(BlockEmitter& e, uint32_t c, const Xmm& xc, SlowJumps& slow) {
    if (e.KnownSingle(e.index, c)) return;
    MovqToGpr(e.a, rax, xc);
    e.a.test(eax, 0x0FFFFFFFu);
    slow.Jcc(e.a, kCcNe, kReasonMultiplier);
}

// The exact-tie case of PpcAccurateSingleMaddIntermediateInline (the fma result's discarded 29 bits are exactly half): the
// interpreter corrects the direction of the discarded error, so the op goes there. Leaves the bits in rax.
void TieCheck(BlockEmitter& e, const Xmm& x, SlowJumps& slow) {
    CG& a = e.a;
    MovqToGpr(a, rax, x);
    a.lea(ecx, a.ptr[rax - 0x10000000]);
    a.test(ecx, 0x1FFFFFFFu);
    slow.Jcc(a, kCcE, kReasonTie);
}

// --- the arithmetic ----------------------------------------------------------------------------------------------------------

// An arithmetic op as a call of its helper: operands in xmm0, xmm1, xmm2 in the helper's order (converted in place), the caches'
// volatile slots flushed, the call, the result stored with the destination's form. A NaN result of an op with two or more sources
// goes to the interpreter's handler instead (nothing has been written yet): which operand's NaN the host picks depends on the
// operand order the compiler gave that copy of the expression (the helper's, or the interpreter's own), and only the
// interpreter's copy is the reference.
void EmitHelperCall(BlockEmitter& e, const Op& o, const Info& info) {
    CG& a = e.a;
    for (unsigned i = 0; i < info.n; ++i) {
        const Xmm x(i);
        unsigned j = 0;
        while (j < i && info.src[j] != info.src[i]) ++j;
        if (j < i) {
            Mov(a, x, Xmm(j));
            BlockEmitter::NoteForm(FormEvent::SharedLoad);
        } else if (info.paired) {
            Mov(a, x, PairedSrc(e, o.kind, info.src[i], x, 0, kSlotConv + i, false));
        } else {
            Mov(a, x, ScalarSrc(e, o.kind, info.src[i], x, 0, kSlotConv + i, false));
        }
    }
    e.FlushVolatileGprs();  // the call takes every volatile register, the FPR cache's included (stores leave xmm0-xmm2 alone)
    e.CallRaw(info.fn);
    SlowJumps slow;
    const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
    if (info.n >= 2) {
        if (info.paired) PairNanCheck(e, xmm0, xmm3, slow);
        else ScalarNanCheck(e, xmm0, slow);
    }
    const int dSlot = WriteResult(e, o.d, info.paired ? Form::Paired : Form::Scalar, xmm0);
    Count(e, o.kind, kSlotDone);
    EmitSlowCold(e, slow, snap, dSlot, o.d, o.kind);
}

// Inline arithmetic. Emits nothing and returns false when the kind has no inline form in this build.
bool EmitInline(BlockEmitter& e, const Op& o, const Info& info) {
    CG& a = e.a;
    const K k = o.kind;
    enum class Shape { None, Scalar2, Single2, Fmuls, Frsp, Fma, FmaSingle, Pair2, PairMuls, PairFma } shape = Shape::None;
    Bin op2 = Bin::AddSd;
    bool subtract = false, negate = false, broadcast1 = false;
    switch (k) {
    case K::Fadd: shape = Shape::Scalar2; op2 = Bin::AddSd; break;
    case K::Fsub: shape = Shape::Scalar2; op2 = Bin::SubSd; break;
    case K::Fmul: shape = Shape::Scalar2; op2 = Bin::MulSd; break;
    case K::Fdiv: shape = Shape::Scalar2; op2 = Bin::DivSd; break;
    case K::Fadds: shape = Shape::Single2; op2 = Bin::AddSd; break;
    case K::Fsubs: shape = Shape::Single2; op2 = Bin::SubSd; break;
    case K::Fdivs: shape = Shape::Single2; op2 = Bin::DivSd; break;
    case K::Fmuls: shape = Shape::Fmuls; break;
    case K::Frsp: shape = Shape::Frsp; break;
    case K::PsAdd: shape = Shape::Pair2; op2 = Bin::AddPs; break;
    case K::PsSub: shape = Shape::Pair2; op2 = Bin::SubPs; break;
    case K::PsMul: shape = Shape::Pair2; op2 = Bin::MulPs; break;
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
    const bool fused = shape == Shape::Fma || shape == Shape::FmaSingle || shape == Shape::PairFma;

    // The sources, in the helper's order (a, then c or b, then b), each in its slot or in scratch xmm1 + i; a register named
    // twice is read once. A fused op's accumulator is xmm0: its first source (a) is loaded there when it is not cached.
    Xmm src[3] = {xmm1, xmm2, xmm3};
    uint32_t pin = 0;
    for (unsigned i = 0; i < info.n; ++i) pin |= 1u << (info.src[i] & 31u);
    for (unsigned i = 0; i < info.n; ++i) {
        unsigned j = 0;
        while (j < i && info.src[j] != info.src[i]) ++j;
        if (j < i) {
            src[i] = src[j];
            BlockEmitter::NoteForm(FormEvent::SharedLoad);
            continue;
        }
        const Xmm scratch = (fused && i == 0) ? xmm0 : Xmm(1 + i);
        const bool cache = info.src[i] != o.d;  // a source the op overwrites is not worth a slot: later reads read the result
        src[i] = info.paired ? PairedSrc(e, k, info.src[i], scratch, pin, kSlotConv + i, cache)
                             : ScalarSrc(e, k, info.src[i], scratch, pin, kSlotConv + i, cache);
    }
    // From here to the result nothing changes the caches: this is what the slow path stores and reloads.
    const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
    SlowJumps slow;
    const uint32_t c = info.n >= 2 ? info.src[1] : 0;
    if (broadcast) {  // PpcBroadcastPs1Inline / PpcBroadcastPs0Inline of c, into xmm3
        // a third source loaded into scratch xmm3 moves to xmm1 (free: a fused op's first source is in xmm0 or a slot)
        if (info.n == 3 && src[2].getIdx() == 3) {
            Mov(a, xmm1, xmm3);
            src[2] = xmm1;
        }
        Shufps(a, xmm3, src[1], src[1], broadcast1 ? 0x00 : 0x55);
        src[1] = xmm3;
    }
    // The destination may be a source: compute into xmm0 and write the result after the checks (the slow path reads the sources).
    switch (shape) {
    case Shape::Scalar2:
        Binary(a, op2, xmm0, src[0], src[1]);
        ScalarNanCheck(e, xmm0, slow);
        break;
    case Shape::Single2:
        Binary(a, op2, xmm0, src[0], src[1]);
        ScalarNanCheck(e, xmm0, slow);
        // the sum or difference of two singles below 2^-126 is exact (see the top of the file): no flush check
        if (op2 == Bin::DivSd || !e.KnownSingle(e.index, info.src[0]) || !e.KnownSingle(e.index, info.src[1])) RangeCheck(e, xmm0, slow);
        break;
    case Shape::Fmuls:
        MultiplierCheck(e, c, src[1], slow);
        Binary(a, Bin::MulSd, xmm0, src[0], src[1]);
        ScalarNanCheck(e, xmm0, slow);
        RangeCheck(e, xmm0, slow);
        break;
    case Shape::Frsp:
        // frsp of a single: cvtsd2ss returns it, and flushes it with NI set as the mask does (a float denormal exactly)
        if (!e.KnownSingle(e.index, info.src[0])) RangeCheck(e, src[0], slow);
        break;
#if defined(__FMA__)
    case Shape::Fma:
    case Shape::FmaSingle:
        if (shape == Shape::FmaSingle) MultiplierCheck(e, c, src[1], slow);
        Mov(a, xmm0, src[0]);
        if (subtract) a.vfmsub213sd(xmm0, src[1], src[2]);
        else a.vfmadd213sd(xmm0, src[1], src[2]);
        ScalarNanCheck(e, xmm0, slow);
        if (shape == Shape::FmaSingle) {
            TieCheck(e, xmm0, slow);
            RangeCheck(e, xmm0, slow, true);
        }
        break;
    case Shape::PairFma:
        Mov(a, xmm0, src[0]);
        if (subtract) a.vfmsub213ps(xmm0, src[1], src[2]);
        else a.vfmadd213ps(xmm0, src[1], src[2]);
        PairNanCheck(e, xmm0, xmm1, slow);
        if (negate) Binary(a, Bin::XorPs, xmm0, xmm0, e.Const128(0x8000000080000000ull));  // no NaN lane: every lane negated
        break;
#else
    case Shape::Fma:
    case Shape::FmaSingle:
    case Shape::PairFma:
        break;
#endif
    case Shape::Pair2:
        Binary(a, op2, xmm0, src[0], src[1]);
        PairNanCheck(e, xmm0, xmm1, slow);
        break;
    case Shape::PairMuls:
        Binary(a, Bin::MulPs, xmm0, src[0], src[1]);
        PairNanCheck(e, xmm0, xmm1, slow);
        break;
    case Shape::None: break;
    }
    int dSlot;
    if (shape == Shape::Single2 || shape == Shape::Fmuls || shape == Shape::Frsp || shape == Shape::FmaSingle) {
        // the single rounding, its widening into the destination
        CvtSd2Ss(a, xmm0, shape == Shape::Frsp ? src[0] : xmm0);
        dSlot = WriteResult(e, o.d, Form::Scalar, xmm0, 0, [&](const Xmm& to) {
            CvtSs2Sd(a, to, xmm0);
            if (negate) Binary(a, Bin::XorPd, to, to, e.Const128(0x8000000000000000ull));  // fnmadds, fnmsubs: not a NaN here
        });
    } else {
        dSlot = WriteResult(e, o.d, info.paired ? Form::Paired : Form::Scalar, xmm0);
    }
    Count(e, k, kSlotDone);
    EmitSlowCold(e, slow, snap, dSlot, o.d, k);
    return true;
}

// fmr, ps_mr: the raw bits and the source's form (every register is Scalar or Paired at run time).
void EmitMove(BlockEmitter& e, const Op& o) {
    CG& a = e.a;
    const Form known = Known(e, o.b);
    if (o.d == o.b) return;
    const uint32_t pin = 1u << (o.b & 31u);
    const Xmm src = RawSrc(e, o.b, xmm0, pin);
    if (known != Form::Unknown) {
        WriteResult(e, o.d, known, src, pin);
        return;
    }
    // the source's form is not known: its bit in memory is right; copy it to the destination's bit now (no pending write for d)
    WriteResult(e, o.d, Form::Unknown, src, pin);
    Label paired, done;
    a.bt(e.PairedMask(), static_cast<uint8_t>(o.b & 31u));
    a.jc(paired, CG::T_SHORT);
    a.btr(e.PairedMask(), static_cast<uint8_t>(o.d & 31u));
    a.jmp(done, CG::T_SHORT);
    a.L(paired);
    a.bts(e.PairedMask(), static_cast<uint8_t>(o.d & 31u));
    a.L(done);
}

// PpcCrvCompareFloat's value in rax from the flags of `ucomisd x, y` (swapped: of `ucomisd y, x`): the field value of nibble 8
// (less), 4 (greater), 2 (equal) or 1 (unordered). Keeps the flags; clobbers rcx, rdx, rsi.
void CrFromUcomisd(CG& a, bool swapped) {
    a.mov(rax, static_cast<uint64_t>(kPpcCrvEq));
    static_assert(kPpcCrvGt >= 0 && kPpcCrvGt <= 0xFFFFFFFF);
    a.mov(ecx, static_cast<uint32_t>(kPpcCrvGt));
    a.mov(rdx, static_cast<uint64_t>(kPpcCrvLt));
    a.mov(rsi, static_cast<uint64_t>(kPpcCrvSo));
    if (!swapped) {  // above: CF = ZF = 0; below: CF; unordered: ZF = PF = CF = 1
        a.cmova(rax, rcx);
        a.cmovb(rax, rdx);
    } else {
        a.cmova(rax, rdx);
        a.cmovb(rax, rcx);
    }
    a.cmovp(rax, rsi);
}

// The taken path of an fcmpu fused with its bc: the field written when the main path did not (it is dead in the block, but the
// block is left with the condition register visible), the dirty GPRs stored (the FPR cache has no dirty slot here), the exit.
struct FcmpTakenCold {
    uint32_t at[2];
    uint8_t n;
    bool writeField, swapped;
    uint8_t field;
    uint32_t target;
    BlockEmitter::GprSnapshot snap;
    void operator()(BlockEmitter& e) const {
        for (unsigned i = 0; i < n; ++i) ResolveAt(e.a, at[i]);
        if (writeField) {  // the compare's flags are still those of the ucomisd
            CrFromUcomisd(e.a, swapped);
            e.a.mov(e.Crv(field), rax);
        }
        e.EmitFlushSnapshot(snap);
        e.ExitLinked(target);
    }
};

// fcmpu followed by a bc that tests a bit of its field, with no ctr decrement: the bc branches on the flags of the compare. The
// field is written before the branch when the block reads it later (CrDead after the bc), else only on the taken path. The bc's
// FPR flush (the dirty slots and pending forms before any branch, jit_x64.cpp's EmitOps) is made here. Answers 2 (the pair), or 0
// when the next op is no such bc. With the field written before the branch, the dirty GPR slots are stored before it too, so that
// the taken path is nothing but the jump: a direct conditional link (ExitLinkedIf), as the integer compares' (EmitCondBranch).
uint32_t EmitFcmpuBc(BlockEmitter& e, uint32_t index) {
    const Op& o = e.block.ops[index];
    if (index + 1 >= e.block.count) return 0;
    const Op& bc = e.block.ops[index + 1];
    const uint32_t field = static_cast<uint32_t>(o.d) >> 2;
    if (bc.kind != K::Bc || (bc.d & 0x04u) == 0 || (bc.d & 0x10u) != 0 || (bc.a >> 2) != field) return 0;
    CG& a = e.a;
    const uint32_t bit = bc.a & 3u;
    const bool want = (bc.d & 0x08u) != 0;
    const bool swapped = bit == 0;  // less: `ucomisd b, a` and above, which an unordered pair does not satisfy
    const bool dead = e.CrDead(index + 1, field);
    if (dead) e.FlushDirtyFprs();
    else e.FlushDirtyGprs();  // the FPR cache's dirty slots and pending forms too (stores and the form mask's `or` / `and`)
    const uint32_t pin = (1u << (o.a & 31u)) | (1u << (o.b & 31u));
    const Xmm x = ScalarSrc(e, K::Fcmpu, o.a, xmm0, pin, kSlotConv);
    Xmm y = x;
    if (o.b == o.a) BlockEmitter::NoteForm(FormEvent::SharedLoad);
    else y = ScalarSrc(e, K::Fcmpu, o.b, xmm1, pin, kSlotConv + 1);
    const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
    if (swapped) Ucomisd(a, y, x);
    else Ucomisd(a, x, y);
    if (!dead) {
        CrFromUcomisd(a, swapped);  // movs and cmovs: the flags stay those of the ucomisd
        a.mov(e.Crv(field), rax);
        BlockEmitter::NoteCr(BlockEmitter::CrEvent::FusedLive);
        switch (bit) {
        case 0:
        case 1: e.ExitLinkedIf(want ? x64::Cc::A : x64::Cc::BE, bc.imm); break;
        case 2:
            if (want) {  // equal: ZF and not PF
                Label skip;
                a.jp(skip, CG::T_NEAR);
                e.ExitLinkedIf(x64::Cc::E, bc.imm);
                a.L(skip);
            } else {
                e.ExitLinkedIf(x64::Cc::P, bc.imm);
                e.ExitLinkedIf(x64::Cc::NE, bc.imm);
            }
            break;
        default: e.ExitLinkedIf(want ? x64::Cc::P : x64::Cc::NP, bc.imm); break;
        }
        Count(e, K::Fcmpu, kSlotDone);
        return 2;
    }
    BlockEmitter::NoteCr(BlockEmitter::CrEvent::FusedFlagsOnly);
    FcmpTakenCold cold{};
    switch (bit) {
    case 0:
    case 1: cold.at[cold.n++] = JccPatch(a, want ? kCcA : 0x6 /* be */); break;
    case 2:
        if (want) {  // equal: ZF and not PF
            Label skip;
            a.jp(skip, CG::T_SHORT);
            cold.at[cold.n++] = JccPatch(a, kCcE);
            a.L(skip);
        } else {
            cold.at[cold.n++] = JccPatch(a, kCcP);
            cold.at[cold.n++] = JccPatch(a, kCcNe);
        }
        break;
    default: cold.at[cold.n++] = JccPatch(a, want ? kCcP : 0xB /* np */); break;
    }
    cold.writeField = dead;
    cold.swapped = swapped;
    cold.field = static_cast<uint8_t>(field);
    cold.target = bc.imm;
    cold.snap = snap;
    e.Cold(cold);
    Count(e, K::Fcmpu, kSlotDone);
    return 2;
}

}  // namespace

uint32_t EmitFp(BlockEmitter& e, uint32_t index) {
    const Op& o = e.block.ops[index];
    CG& a = e.a;
    const K k = o.kind;
    switch (k) {
    case K::Fmr:
    case K::PsMr:
        EmitMove(e, o);
        Count(e, k, kSlotDone);
        return 1;
    case K::Fneg:
    case K::Fabs: {  // -Scalar(b), fabs(Scalar(b)): the sign bit of the double, set or cleared, a NaN's too
        const Xmm b = ScalarSrc(e, k, o.b, xmm0, 1u << (o.b & 31u), kSlotConv, o.b != o.d);
        WriteResult(e, o.d, Form::Scalar, xmm0, 1u << (o.b & 31u), [&](const Xmm& to) {
            if (k == K::Fneg) Binary(a, Bin::XorPd, to, b, e.Const128(0x8000000000000000ull));
            else Binary(a, Bin::AndPd, to, b, e.Const128(0x7FFFFFFFFFFFFFFFull));
        });
        Count(e, k, kSlotDone);
        return 1;
    }
    case K::Fcmpu: {  // PpcCrvCompareFloat: the value of nibble 8 (less), 4 (greater), 2 (equal) or 1 (unordered)
        if (e.CrDead(index, o.d >> 2)) {  // nothing reads the field before a later op overwrites it (BlockEmitter::CrDead)
            BlockEmitter::NoteCr(BlockEmitter::CrEvent::WriteSkipped);
            return 1;
        }
        if (FusedFcmpEnabled()) {
            if (const uint32_t used = EmitFcmpuBc(e, index)) return used;
        }
        const uint32_t pin = (1u << (o.a & 31u)) | (1u << (o.b & 31u));
        const Xmm x = ScalarSrc(e, k, o.a, xmm0, pin, kSlotConv);
        Xmm y = x;
        if (o.b == o.a) BlockEmitter::NoteForm(FormEvent::SharedLoad);
        else y = ScalarSrc(e, k, o.b, xmm1, pin, kSlotConv + 1);
        Ucomisd(a, x, y);
        CrFromUcomisd(a, false);
        a.mov(e.Crv(o.d >> 2), rax);
        Count(e, k, kSlotDone);
        return 1;
    }
    case K::PsNeg:
    case K::PsAbs: {  // the sign bit of each single (std::abs and unary minus on floats: no arithmetic)
        const Xmm b = PairedSrc(e, k, o.b, xmm0, 1u << (o.b & 31u), kSlotConv, o.b != o.d);
        WriteResult(e, o.d, Form::Paired, xmm0, 1u << (o.b & 31u), [&](const Xmm& to) {
            if (k == K::PsNeg) Binary(a, Bin::XorPs, to, b, e.Const128(0x8000000080000000ull));
            else Binary(a, Bin::AndPs, to, b, e.Const128(0x7FFFFFFF7FFFFFFFull));
        });
        Count(e, k, kSlotDone);
        return 1;
    }
    case K::PsMerge00:
    case K::PsMerge01:
    case K::PsMerge10:
    case K::PsMerge11: {
        // result.ps0 (lane 1) from a's ps0 (lane 1) or ps1 (lane 0), result.ps1 (lane 0) from b's
        const unsigned aLane = (k == K::PsMerge00 || k == K::PsMerge01) ? 1u : 0u;
        const unsigned bLane = (k == K::PsMerge00 || k == K::PsMerge10) ? 1u : 0u;
        const uint32_t pin = (1u << (o.a & 31u)) | (1u << (o.b & 31u));
        const Xmm xa = PairedSrc(e, k, o.a, xmm0, pin, kSlotConv, o.a != o.d);
        const Xmm xb = o.b == o.a ? xa : PairedSrc(e, k, o.b, xmm1, pin, kSlotConv + 1, o.b != o.d);
        if (o.b == o.a) BlockEmitter::NoteForm(FormEvent::SharedLoad);
        WriteResult(e, o.d, Form::Paired, xmm2, pin, [&](const Xmm& to) {
            if (kAvx && aLane == 0 && bLane == 0) {
                Binary(a, Bin::Unpcklps, to, xb, xa);  // { b.ps1, a.ps1 }
            } else if (kAvx && aLane == 1 && bLane == 0) {
                a.vblendps(to, xb, xa, 0x2);  // lane 1 from a, lane 0 from b
            } else {
                Shufps(a, xmm3, xb, xa, static_cast<uint8_t>(bLane | bLane << 2 | aLane << 4 | aLane << 6));  // { b[bLane] x2, a[aLane] x2 }
                Shufps(a, to, xmm3, xmm3, 0x08);                                                              // lanes 0 and 2
            }
        });
        Count(e, k, kSlotDone);
        return 1;
    }
    default: break;
    }

    Info info;
    if (!GetInfo(o, info)) return 0;
    if (HelpersOnly() || !EmitInline(e, o, info)) EmitHelperCall(e, o, info);
    return 1;
}

// The float ops Decode leaves Generic: fsel, fnabs, fnmadd, fnmsub, fres, frsqrte, fctiw, fctiwz (Rc clear: the forms with Rc
// set are the interpreter's). Each reads its sources as Scalar(f, r) and sets the destination Scalar, as Exec's cases do.
// And ps_sel, which reads them as Paired(f, r) and sets the destination Paired. The caches are empty on entry (a Generic op
// flushes them) and flushed after.
uint32_t EmitFpGeneric(BlockEmitter& e, uint32_t w) {
    const uint32_t po = w >> 26, d = (w >> 21) & 31, a = (w >> 16) & 31, b = (w >> 11) & 31, c = (w >> 6) & 31;
    const uint32_t xo5 = (w >> 1) & 31, xo10 = (w >> 1) & 1023;
    if ((w & 1u) != 0) return 0;
    const Op o{K::Generic, static_cast<uint8_t>(d), static_cast<uint8_t>(a), static_cast<uint8_t>(b), c};
    CG& as = e.a;
    if (po == 4) {
        if (xo5 != 23) return 0;
        // ps_sel (PPC_PsSelInline(Paired c, Paired a, Paired b)): each lane is c's when a's is >= -0.0, else b's; a NaN lane of
        // a picks b. Only bits are moved, so the result is the sources' lanes exactly as Paired(f, r) makes them.
        const Xmm xa = PairedSrc(e, K::Generic, a, xmm0, 0, kSlotConv, false);
        Mov(as, xmm0, xa);
        const Xmm xc = PairedSrc(e, K::Generic, c, xmm1, 0, kSlotConv + 1, false);
        Mov(as, xmm1, xc);
        const Xmm xb = PairedSrc(e, K::Generic, b, xmm2, 0, kSlotConv + 2, false);
        Mov(as, xmm2, xb);
        as.xorps(xmm3, xmm3);
        as.cmpleps(xmm3, xmm0);  // 0.0 <= a, per lane: ordered, and -0.0 compares equal to 0.0
        as.andps(xmm1, xmm3);
        as.andnps(xmm3, xmm2);
        as.orps(xmm1, xmm3);
        WriteResult(e, d, Form::Paired, xmm1);
        Count(e, K::Generic, kSlotDone);
        return 1;
    }
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
        const Xmm xa = ScalarSrc(e, K::Generic, a, xmm0, 0, kSlotConv, false);
        Mov(as, xmm0, xa);
        const Xmm xc = ScalarSrc(e, K::Generic, c, xmm1, 0, kSlotConv + 1, false);
        Mov(as, xmm1, xc);
        const Xmm xb = ScalarSrc(e, K::Generic, b, xmm2, 0, kSlotConv + 2, false);
        Mov(as, xmm2, xb);
        as.xorpd(xmm3, xmm3);
        as.ucomisd(xmm0, xmm3);  // CF is set when a < 0 and when unordered
        as.movq(rax, xmm1);
        as.movq(rcx, xmm2);
        as.cmovb(rax, rcx);
        e.DropFpr(d);
        as.mov(e.Fpr(d), rax);
        e.SetFprForm(d, Form::Scalar);
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
        const Xmm xb = ScalarSrc(e, K::Generic, b, xmm0, 0, kSlotConv, false);
        WriteResult(e, d, Form::Scalar, xmm0, 0, [&](const Xmm& to) { Binary(as, Bin::OrPd, to, xb, e.Const128(0x8000000000000000ull)); });
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
