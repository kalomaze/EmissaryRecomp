// The floating point register forms in compiled blocks (docs/jit-design.md, "FPR forms"): what a block knows, op by op, about
// the form bytes (Frame::form) the float code tests and writes.
//
// Every float op reads its sources "as Scalar" or "as Paired": it looks at the register's form byte and converts the value when
// the register holds the other form, and it writes its destination's byte. Within a block the byte of a register is known
// after the op that wrote it (every kind that writes a register sets a definite form: RunOp's SetScalar / SetPaired), so the
// test, and the write of a byte that already holds the value, can go. AnalyseForms walks the block once, when it is compiled,
// and keeps for every op the registers whose form is known on entry to the op's code.
//
// Knowledge is about the form the register holds. The byte in memory follows it lazily (the writes are pending in the emitter until
// the next flush before a call, an exit or a cold path's call: BlockEmitter::SetFprForm); a register the block does not know has no
// pending write, so its byte in memory is right whenever code tests it. Knowledge starts empty at the block's entry (the
// dispatcher and the links enter with whatever forms the frame holds) and an op changes it only as listed here:
//
//  * a kind that writes an FPR sets its destination: Scalar (loads lfs, lfd and their forms, the double and single arithmetic,
//    fneg, fabs, frsp, and the Generic float ops: DescribeGeneric), Paired (psq_l, the paired arithmetic and moves);
//  * fmr copies the source's form: known if the source's is. ps_mr gives Scalar for a Scalar source and Paired for a Paired
//    one; for a source that is not known the destination is not known;
//  * everything else that is positively listed leaves every register as it was: the integer ops, the stores, the compares, the
//    branches that fall through;
//  * a call (bl, bctrl: the callee may leave the forms as it likes), a Generic op that
//    DescribeGeneric does not list, and anything not listed here empties the knowledge.
//
// A float op's slow path (the interpreter's handler for the same op, run for a NaN or a rounding case) writes the same
// destination byte and no other, so it does not change what is known after the op. An op's own sources are read before its
// destination is written, so aliasing is no problem: the knowledge on entry describes the sources.
//
// Alongside the forms: the registers known Scalar whose double is a single-precision value widened (BlockEmitter::KnownSingle;
// its fraction's low 29 bits are clear and it is no double subnormal, so PpcForce25BitInline leaves it as it is, and a sum or
// difference of two of them below 2^-126 is exact). A Paired register's scalar view (ps0 widened by cvtss2sd) is always one. The
// writers that make one: lfs and its forms (the manual's conversion of any word, or cvtss2sd), the single-precision arithmetic
// and frsp (a float widened, whatever the operands), and fmr, fneg, fabs, ps_mr of a register whose scalar view is one (they
// move bits or change the sign only). Every other writer makes the register not known single.
//
// The forms are kept in memory lazily (BlockEmitter::SetFprForm, jit_x64_fpr.cpp): what this analysis knows is the form the
// register holds, and the emitter writes the byte before anything reads it in memory.
//
// BRAWL_JIT_NO_FORMS=1 turns the analysis off (every form unknown, every byte written).
// BRAWL_JIT_FORMS_FAULT=<n> (a test aid: it makes the analysis wrong on purpose so that the tests can show they notice) is read
// at every compile: 1 keeps the knowledge across calls and unlisted ops, 2 gives fmr and ps_mr a Scalar result whatever their
// source, 3 makes the loads set Paired, 4 treats a paired op's destination as already Paired, 5 takes every Scalar result for a
// single-precision value (lfd's and the double arithmetic's too).
#include "jit_x64.h"
#include "jit_x64_generic.h"

#include <atomic>
#include <cstdlib>

namespace PpcInterpreter::detail::x64 {

namespace {

struct FormCounters {
    std::atomic<uint64_t> analysed{0}, testsSkipped{0}, conversionsForced{0}, testsEmitted{0}, writesSkipped{0}, writesEmitted{0},
        sharedLoads{0};
};
FormCounters g_forms;

bool EnvOn(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && *v != '0';
}

int Fault() {
    const char* v = std::getenv("BRAWL_JIT_FORMS_FAULT");
    return v != nullptr ? std::atoi(v) : 0;
}

// What an op does to the knowledge.
struct Effect {
    enum Kind : uint8_t { Keep, SetScalar, SetPaired, Copy, Mr, Clear } kind = Keep;
    uint8_t dst = 0, src = 0;
};

Effect EffectOf(const Op& o) {
    using E = Effect;
    switch (o.kind) {
    // integer, compares, spr moves, memory that moves no float
    case K::Jmp:
    case K::Li: case K::Addi: case K::Lis: case K::Addis: case K::Mulli: case K::Ori: case K::Oris: case K::Xori: case K::Xoris:
    case K::AndiRc: case K::AndisRc: case K::Rlwinm: case K::RlwinmRc: case K::Rlwimi: case K::Cmpwi: case K::Cmplwi: case K::Cmpw:
    case K::Cmplw: case K::Add: case K::Subf: case K::Neg: case K::Mullw: case K::Or: case K::OrRc: case K::And: case K::Xor:
    case K::Andc: case K::Nor: case K::Slw: case K::Srw: case K::Srawi: case K::Extsb: case K::Extsh: case K::Cntlzw: case K::Mflr:
    case K::Mtlr: case K::Mfctr: case K::Mtctr: case K::Mfcr: case K::AndRc: case K::Subfic: case K::Rlwnm: case K::RlwnmRc:
    case K::Mfspr: case K::Mtspr:
    case K::Lwz: case K::Lwzu: case K::Lbz: case K::Lbzu: case K::Lhz: case K::Lhzu: case K::Lha: case K::Stw: case K::Stwu:
    case K::Stb: case K::Stbu: case K::Sth: case K::Sthu: case K::Lmw: case K::Stmw: case K::Lwzx: case K::Lbzx: case K::Lhzx:
    case K::Stwx: case K::Stbx: case K::Sthx:
    case K::Stfs: case K::Stfsu: case K::Stfd: case K::Stfdu: case K::Stfsx: case K::PsqSt: case K::PsqStu:
    case K::Fcmpu:
    // branches that go on when not taken
    case K::Bc: case K::CmpwiBc: case K::CmplwiBc: case K::CmpwBc: case K::CmplwBc: case K::RlwinmRcBc: case K::Bclr:
        return {E::Keep, 0, 0};
    case K::Lfs: case K::Lfsu: case K::Lfd: case K::Lfdu: case K::Lfsx:
        return {Fault() == 3 ? E::SetPaired : E::SetScalar, o.d, 0};
    case K::PsqL: case K::PsqLu: return {E::SetPaired, o.d, 0};
    case K::Fadds: case K::Fsubs: case K::Fmuls: case K::Fdivs: case K::Fmadds: case K::Fmsubs: case K::Fnmsubs: case K::Fnmadds:
    case K::Fadd: case K::Fsub: case K::Fmul: case K::Fdiv: case K::Fmadd: case K::Fmsub: case K::Fneg: case K::Fabs: case K::Frsp:
        return {E::SetScalar, o.d, 0};
    case K::Fmr: return {Fault() == 2 ? E::SetScalar : E::Copy, o.d, o.b};
    case K::PsMr: return {Fault() == 2 ? E::SetScalar : E::Mr, o.d, o.b};
    case K::PsAdd: case K::PsSub: case K::PsMul: case K::PsMadd: case K::PsMsub: case K::PsNmadd: case K::PsNmsub: case K::PsMuls0:
    case K::PsMuls1: case K::PsMadds0: case K::PsMadds1: case K::PsSum0: case K::PsSum1: case K::PsMerge00: case K::PsMerge01:
    case K::PsMerge10: case K::PsMerge11: case K::PsNeg: case K::PsAbs:
        return {E::SetPaired, o.d, 0};
    case K::Generic: {
        const GenericFacts g = DescribeGeneric(o.imm);
        switch (g.form) {
        case GenericFacts::FormKeep: return {E::Keep, 0, 0};
        case GenericFacts::FormScalar: return {E::SetScalar, g.formReg, 0};
        case GenericFacts::FormBarrier: break;
        }
        return {E::Clear, 0, 0};
    }
    default: return {E::Clear, 0, 0};  // calls, jumps, anything not listed
    }
}

}  // namespace

void BlockEmitter::AnalyseForms() {
    const uint32_t n = block.count;
    g_forms.analysed.fetch_add(1, std::memory_order_relaxed);
    if (EnvOn("BRAWL_JIT_NO_FORMS")) {
        for (uint32_t i = 0; i <= n; ++i) formScalar_[i] = formPaired_[i] = formSingle_[i] = 0;
        return;
    }
    const int fault = Fault();
    uint32_t scalar = 0, paired = 0;  // the registers known Scalar / Paired before the op looked at
    uint32_t single = 0;              // of `scalar`, those known to hold a single-precision value
    const auto set = [&](uint32_t r, uint32_t form) {  // form: 0 not known, 1 scalar, 2 paired
        const uint32_t bit = 1u << (r & 31u);
        scalar &= ~bit;
        paired &= ~bit;
        single &= ~bit;
        if (form == 1) scalar |= bit;
        else if (form == 2) paired |= bit;
    };
    const auto scalarViewSingle = [&](uint32_t r) { return (((paired | single) >> (r & 31u)) & 1u) != 0; };
    for (uint32_t i = 0; i < n; ++i) {
        formScalar_[i] = scalar;
        formPaired_[i] = paired;
        formSingle_[i] = single;
        const Op& o = block.ops[i];
        const Effect fx = EffectOf(o);
        // a Scalar result that is a single (before `set` forgets the sources)
        bool makesSingle = false;
        switch (o.kind) {
        case K::Lfs: case K::Lfsu: case K::Lfsx:
        case K::Fadds: case K::Fsubs: case K::Fmuls: case K::Fdivs: case K::Fmadds: case K::Fmsubs: case K::Fnmsubs: case K::Fnmadds:
        case K::Frsp:
            makesSingle = true;
            break;
        case K::Fneg: case K::Fabs: case K::Fmr: case K::PsMr:
            makesSingle = scalarViewSingle(o.b);
            break;
        default: break;
        }
        switch (fx.kind) {
        case Effect::Keep: break;
        case Effect::SetScalar: set(fx.dst, 1); break;
        case Effect::SetPaired: set(fx.dst, 2); break;
        case Effect::Copy: {
            const uint32_t bit = 1u << (fx.src & 31u);
            set(fx.dst, (scalar & bit) != 0 ? 1u : (paired & bit) != 0 ? 2u : 0u);
            break;
        }
        case Effect::Mr: {
            const uint32_t bit = 1u << (fx.src & 31u);
            set(fx.dst, (scalar & bit) != 0 ? 1u : (paired & bit) != 0 ? 2u : 0u);  // an unknown source: Scalar or Paired
            break;
        }
        case Effect::Clear:
            if (fault != 1) scalar = paired = single = 0;
            break;
        }
        if (fx.kind != Effect::Clear && fx.kind != Effect::Keep) {
            const uint32_t bit = 1u << (fx.dst & 31u);
            // fault 5 (wrong on purpose): every Scalar result "is" a single, lfd's and the double arithmetic's too
            if ((scalar & bit) != 0 && (makesSingle || fault == 5)) single |= bit;
        }
        if (fault == 4 && fx.kind == Effect::SetPaired) {  // wrong on purpose: the destination "was" Paired already
            formPaired_[i] |= 1u << (fx.dst & 31u);
            formScalar_[i] &= ~(1u << (fx.dst & 31u));
        }
        // a fused compare and bc: the bc half is looked at with the same knowledge
        if ((o.kind == K::CmpwiBc || o.kind == K::CmplwiBc || o.kind == K::CmpwBc || o.kind == K::CmplwBc || o.kind == K::RlwinmRcBc) &&
            i + 1 < n) {
            formScalar_[i + 1] = scalar;
            formPaired_[i + 1] = paired;
            formSingle_[i + 1] = single;
            ++i;
        }
    }
    formScalar_[n] = formPaired_[n] = formSingle_[n] = 0;
}

void BlockEmitter::NoteForm(FormEvent e) {
    switch (e) {
    case FormEvent::TestSkipped: g_forms.testsSkipped.fetch_add(1, std::memory_order_relaxed); break;
    case FormEvent::ConversionForced: g_forms.conversionsForced.fetch_add(1, std::memory_order_relaxed); break;
    case FormEvent::TestEmitted: g_forms.testsEmitted.fetch_add(1, std::memory_order_relaxed); break;
    case FormEvent::WriteSkipped: g_forms.writesSkipped.fetch_add(1, std::memory_order_relaxed); break;
    case FormEvent::WriteEmitted: g_forms.writesEmitted.fetch_add(1, std::memory_order_relaxed); break;
    case FormEvent::SharedLoad: g_forms.sharedLoads.fetch_add(1, std::memory_order_relaxed); break;
    }
}

}  // namespace PpcInterpreter::detail::x64

namespace PpcInterpreter::detail {

JitX64FormStats JitX64GetFormStats() {
    using x64::g_forms;
    const auto get = [](const std::atomic<uint64_t>& c) { return c.load(std::memory_order_relaxed); };
    return JitX64FormStats{get(g_forms.analysed), get(g_forms.testsSkipped), get(g_forms.conversionsForced),
                           get(g_forms.testsEmitted), get(g_forms.writesSkipped), get(g_forms.writesEmitted), get(g_forms.sharedLoads)};
}

}  // namespace PpcInterpreter::detail
