// Native code for the instructions Decode leaves as K::Generic (the interpreter runs them through Exec), written against the
// emitter API in jit_x64.h, and what the analyses may assume of each (DescribeGeneric).
//
// Every function reproduces the matching case of Exec (ppc_interpreter.cpp) bit for bit; the case is named in a comment where
// it is not obvious. The families:
//
//  * XO-form arithmetic with carry (addic, addic., subfc, addc, adde, subfe, addze, subfze, addme, subfme): the carry in is
//    XER[CA] through `bt`, the carry out is the host carry after the add, written back to XER[CA];
//  * mulhw, mulhwu, divw, divwu (the divide-by-zero and INT_MIN / -1 results of PPC_Divw and PPC_Divwu in cold paths);
//  * the Rc forms of the ops that have a kind without Rc (add., subf., neg., mullw., xor., andc., nor., slw., srw., srawi.,
//    sraw., extsb., extsh., cntlzw., rlwimi.): the plain form (EmitIntOp), then CR0 from the register it wrote;
//  * orc, nand, eqv, sraw;
//  * the condition register ops: crand ... cror, mcrf, mtcrf;
//  * the loads and stores of the memory group's Generic forms (EmitMemGeneric), the float ops (EmitFpGeneric), and the ops that
//    do nothing (sync, eieio, cache hints, isync).
//
// An OE form (overflow enable) is left to the interpreter's handler: nothing in the hot code uses one. DescribeGeneric holds
// for those as for the native forms.
#include "jit_x64_generic.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace PpcInterpreter::detail::x64 {

namespace {

using namespace Xbyak::util;
using CG = Xbyak::CodeGenerator;

constexpr uint32_t kXerCa = 0x20000000u;

bool EnvOn(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && *v != '0';
}

struct Fields {
    uint32_t po, d, a, b, xo, rc, oe;
    explicit Fields(uint32_t w)
        : po(w >> 26), d((w >> 21) & 31), a((w >> 16) & 31), b((w >> 11) & 31), xo((w >> 1) & 0x3FF), rc(w & 1u), oe((w >> 10) & 1u) {}
};

// The XO-form arithmetic cases of Exec's first switch (xo with the OE bit masked off).
bool IsXoArithmetic(uint32_t xo9) {
    switch (xo9) {
    case 8: case 10: case 11: case 40: case 75: case 104: case 136: case 138: case 200: case 202: case 232: case 234:
    case 235: case 266: case 459: case 491:
        return true;
    default: return false;
    }
}

// The X-form logical and shift cases that write rA and, with Rc, CR0 (Exec's `writeA`).
bool IsWriteA(uint32_t xo) {
    switch (xo) {
    case 24: case 26: case 28: case 60: case 124: case 284: case 316: case 412: case 444: case 476: case 536: case 792:
    case 824: case 922: case 954:
        return true;
    default: return false;
    }
}

// Cache and ordering instructions that Exec ignores.
bool IsNop31(uint32_t xo) {
    switch (xo) {
    case 4: case 54: case 86: case 210: case 242: case 246: case 278: case 470: case 598: case 854: case 982:
        return true;
    default: return false;
    }
}

uint32_t Mask(uint32_t mb, uint32_t me) {
    const uint32_t a = 0xFFFFFFFFu >> mb;
    const uint32_t b = me == 31 ? 0xFFFFFFFFu : ~(0xFFFFFFFFu >> (me + 1));
    return mb <= me ? (a & b) : (a | b);
}

// --- pieces -------------------------------------------------------------------------------------------------------------------

// XER[CA] = the host carry flag.
void SetCaFromCf(BlockEmitter& e) {
    e.a.sbb(ecx, ecx);
    e.a.and_(ecx, kXerCa);
    e.a.and_(e.Xer(), ~kXerCa);
    e.a.or_(e.Xer(), ecx);
}

// CR0 from a value (SetCr0: a signed compare against zero) held in `reg` (32 bit), unless a later op overwrites CR0 first.
void Cr0From(BlockEmitter& e, const Xbyak::Reg32& reg) {
    if (e.CrDead(e.index, 0)) {
        BlockEmitter::NoteCr(BlockEmitter::CrEvent::WriteSkipped);
        return;
    }
    e.a.movsxd(rax, reg);
    e.CrStore(0);
}

// CR0 from guest register r, as the op just wrote it.
void Cr0FromGpr(BlockEmitter& e, uint32_t r) {
    if (e.CrDead(e.index, 0)) {
        BlockEmitter::NoteCr(BlockEmitter::CrEvent::WriteSkipped);
        return;
    }
    const int hit = e.FindCachedGpr(r);
    if (hit >= 0) e.a.movsxd(rax, BlockEmitter::GprCacheReg(static_cast<uint32_t>(hit)));
    else e.a.movsxd(rax, e.Gpr(r));
    e.CrStore(0);
}

// The plain form of an Int-group kind, then CR0 when rc: dst is the register the plain op wrote.
uint32_t PlainThenCr0(BlockEmitter& e, K kind, const Fields& f, uint32_t dst, uint32_t imm = 0) {
    const Op o{kind, static_cast<uint8_t>(f.d), static_cast<uint8_t>(f.a), static_cast<uint8_t>(f.b), imm};
    if (EmitIntOp(e, o) == 0) return 0;
    if (f.rc != 0) Cr0FromGpr(e, dst);
    return 1;
}

// sraw: r[a] = r[d] >> (r[b] & 63) arithmetic as 64 bits (counts 32..63 give the sign), XER[CA] = r[d] negative and a 1 bit was
// shifted out (for a count over 31 every bit counts, and a negative value has its sign bit).
void EmitSraw(BlockEmitter& e, const Fields& f) {
    auto& c = e.a;
    c.movsxd(rax, e.Gpr(f.d));
    c.mov(ecx, e.Gpr(f.b));
    c.and_(ecx, 63);
    c.mov(rdx, rax);
#if defined(__BMI2__)
    c.bzhi(rdx, rdx, rcx);  // the bits that are shifted out
#else
    c.mov(esi, 1);
    c.shl(rsi, cl);
    c.dec(rsi);
    c.and_(rdx, rsi);
#endif
    c.mov(r8, rax);
    c.sar(r8, 63);    // all ones when the value is negative (before rA is written: it may be rS)
    c.sar(rax, cl);
    c.mov(e.Gpr(f.a), eax);
    c.neg(rdx);       // CF: any lost bit
    c.sbb(edx, edx);  // all ones when so
    c.and_(edx, r8d);
    c.and_(edx, kXerCa);
    c.and_(e.Xer(), ~kXerCa);
    c.or_(e.Xer(), edx);
    if (f.rc != 0) Cr0FromGpr(e, f.a);
}

// The XO-form arithmetic of Exec's first switch, OE clear.
uint32_t EmitXo(BlockEmitter& e, const Fields& f, uint32_t xo9) {
    auto& c = e.a;
    switch (xo9) {
    case 266: return PlainThenCr0(e, K::Add, f, f.d);
    case 40: return PlainThenCr0(e, K::Subf, f, f.d);
    case 104: return PlainThenCr0(e, K::Neg, f, f.d);
    case 235: return PlainThenCr0(e, K::Mullw, f, f.d);
    case 10:  // addc
        c.mov(eax, e.Gpr(f.a));
        c.add(eax, e.Gpr(f.b));
        c.mov(e.Gpr(f.d), eax);
        SetCaFromCf(e);
        break;
    case 8:  // subfc: ~a + b + 1, the carry is "no borrow"
        c.mov(eax, e.Gpr(f.b));
        c.sub(eax, e.Gpr(f.a));
        c.mov(e.Gpr(f.d), eax);
        c.cmc();
        SetCaFromCf(e);
        break;
    case 138:  // adde
    case 234:  // addme: a + 0xFFFFFFFF + CA
    case 202:  // addze: a + 0 + CA
        c.mov(eax, e.Gpr(f.a));
        c.bt(e.Xer(), 29);
        if (xo9 == 138) c.adc(eax, e.Gpr(f.b));
        else if (xo9 == 234) c.adc(eax, -1);
        else c.adc(eax, 0);
        c.mov(e.Gpr(f.d), eax);
        SetCaFromCf(e);
        break;
    case 136:  // subfe: ~a + b + CA
    case 232:  // subfme: ~a + 0xFFFFFFFF + CA
    case 200:  // subfze: ~a + 0 + CA
        c.mov(eax, e.Gpr(f.a));
        c.not_(eax);
        c.bt(e.Xer(), 29);
        if (xo9 == 136) c.adc(eax, e.Gpr(f.b));
        else if (xo9 == 232) c.adc(eax, -1);
        else c.adc(eax, 0);
        c.mov(e.Gpr(f.d), eax);
        SetCaFromCf(e);
        break;
    case 75:  // mulhw
    case 11:  // mulhwu
        c.mov(eax, e.Gpr(f.a));
        if (xo9 == 75) c.imul(e.Gpr(f.b));
        else c.mul(e.Gpr(f.b));
        c.mov(e.Gpr(f.d), edx);
        if (f.rc != 0) Cr0From(e, edx);
        return 1;
    case 491: {  // divw
        Xbyak::Label& special = e.NewLabel();
        Xbyak::Label& zero = e.NewLabel();
        Xbyak::Label& back = e.NewLabel();
        c.mov(eax, e.Gpr(f.a));
        c.mov(ecx, e.Gpr(f.b));
        c.lea(edx, c.ptr[rcx + 1]);
        c.cmp(edx, 1);
        c.jbe(special, CG::T_NEAR);  // a divisor of 0 or -1
        c.cdq();
        c.idiv(ecx);
        c.L(back);
        c.mov(e.Gpr(f.d), eax);
        e.Cold([&special, &zero, &back](BlockEmitter& k) {
            k.a.L(special);
            k.a.test(ecx, ecx);
            k.a.jz(zero, CG::T_SHORT);
            k.a.neg(eax);  // x / -1 = -x; OF only for INT_MIN, which gives -1 as PPC_Divw does
            k.a.mov(edx, -1);
            k.a.cmovo(eax, edx);
            k.a.jmp(back, CG::T_NEAR);
            k.a.L(zero);
            k.a.cdq();  // a divisor of 0: -1 for a negative dividend, else 0
            k.a.mov(eax, edx);
            k.a.jmp(back, CG::T_NEAR);
        });
        if (f.rc != 0) Cr0From(e, eax);
        return 1;
    }
    case 459: {  // divwu
        Xbyak::Label& zero = e.NewLabel();
        Xbyak::Label& back = e.NewLabel();
        c.mov(eax, e.Gpr(f.a));
        c.mov(ecx, e.Gpr(f.b));
        c.test(ecx, ecx);
        c.jz(zero, CG::T_NEAR);
        c.xor_(edx, edx);
        c.div(ecx);
        c.L(back);
        c.mov(e.Gpr(f.d), eax);
        e.Cold([&zero, &back](BlockEmitter& k) {
            k.a.L(zero);
            k.a.xor_(eax, eax);
            k.a.jmp(back, CG::T_NEAR);
        });
        if (f.rc != 0) Cr0From(e, eax);
        return 1;
    }
    default: return 0;
    }
    // the carry forms: the result is in eax still
    if (f.rc != 0) Cr0From(e, eax);
    return 1;
}

// The X-form logical, shift and extend ops that write rA (Exec's writeA).
uint32_t EmitWriteA(BlockEmitter& e, const Fields& f) {
    auto& c = e.a;
    switch (f.xo) {
    case 316: return PlainThenCr0(e, K::Xor, f, f.a);
    case 60: return PlainThenCr0(e, K::Andc, f, f.a);
    case 124: return PlainThenCr0(e, K::Nor, f, f.a);
    case 24: return PlainThenCr0(e, K::Slw, f, f.a);
    case 536: return PlainThenCr0(e, K::Srw, f, f.a);
    case 824: return PlainThenCr0(e, K::Srawi, f, f.a);
    case 954: return PlainThenCr0(e, K::Extsb, f, f.a);
    case 922: return PlainThenCr0(e, K::Extsh, f, f.a);
    case 26: return PlainThenCr0(e, K::Cntlzw, f, f.a);
    case 444: return PlainThenCr0(e, K::Or, f, f.a);   // reached for or with Rc only when no kind took it
    case 28: return PlainThenCr0(e, K::And, f, f.a);
    case 792: EmitSraw(e, f); return 1;
    case 412:  // orc: r[d] | ~r[b]
        c.mov(eax, e.Gpr(f.b));
        c.not_(eax);
        c.or_(eax, e.Gpr(f.d));
        break;
    case 476:  // nand
        c.mov(eax, e.Gpr(f.d));
        c.and_(eax, e.Gpr(f.b));
        c.not_(eax);
        break;
    case 284:  // eqv
        c.mov(eax, e.Gpr(f.d));
        c.xor_(eax, e.Gpr(f.b));
        c.not_(eax);
        break;
    default: return 0;
    }
    c.mov(e.Gpr(f.a), eax);
    if (f.rc != 0) Cr0From(e, eax);
    return 1;
}

// crnor ... cror (opcode 19, xo 33 ... 449): bt, ba and bb are bit numbers (field * 4 + 0 lt, 1 gt, 2 eq, 3 so). The interpreter's
// code is called: it turns the target field's value into a nibble, sets the bit and takes the nibble's table value.
void EmitCrLogical(BlockEmitter& e, uint32_t xo, uint32_t bt, uint32_t ba, uint32_t bb) {
    uint32_t op;
    switch (xo) {
    case 33: op = 0; break;   // crnor
    case 129: op = 1; break;  // crandc
    case 193: op = 2; break;  // crxor
    case 225: op = 3; break;  // crnand
    case 257: op = 4; break;  // crand
    case 289: op = 5; break;  // creqv
    case 417: op = 6; break;  // crorc
    default: op = 7; break;   // cror
    }
    e.CrLogic(op, bt, ba, bb);
}

// mcrf: CR field bf = CR field bfa.
void EmitMcrf(BlockEmitter& e, uint32_t bf, uint32_t bfa) {
    if (bf == bfa) return;
    e.a.mov(rax, e.Crv(bfa));
    e.a.mov(e.Crv(bf), rax);
}

// --- diagnostics ------------------------------------------------------------------------------------------------------------

constexpr uint32_t kKeys = 64u * 1024u;  // primary opcode (6 bits) and the 10 bits after bit 0
struct Counts {
    std::atomic<uint64_t> compiledNative[kKeys], compiledFallback[kKeys];
    uint64_t run[kKeys];
    Counts() {
        for (uint32_t i = 0; i < kKeys; ++i) {
            compiledNative[i].store(0, std::memory_order_relaxed);
            compiledFallback[i].store(0, std::memory_order_relaxed);
            run[i] = 0;
        }
    }
    ~Counts() {
        if (!GenericStatsOn()) return;
        uint64_t total = 0;
        for (uint32_t i = 0; i < kKeys; ++i) total += run[i];
        std::fprintf(stderr, "jit generic ops executed: %llu\n", static_cast<unsigned long long>(total));
        for (int rank = 0; rank < 40; ++rank) {
            uint32_t best = kKeys;
            for (uint32_t i = 0; i < kKeys; ++i)
                if (run[i] != 0 && (best == kKeys || run[i] > run[best])) best = i;
            if (best == kKeys) break;
            std::fprintf(stderr, "  op %2u xo %4u: %12llu run, compiled %llu native %llu fallback\n", best >> 10, best & 1023u,
                         static_cast<unsigned long long>(run[best]),
                         static_cast<unsigned long long>(compiledNative[best].load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(compiledFallback[best].load(std::memory_order_relaxed)));
            run[best] = 0;
        }
    }
};
Counts& Stats() {
    static Counts counts;
    return counts;
}
uint32_t KeyOf(uint32_t w) { return ((w >> 26) << 10) | ((w >> 1) & 0x3FFu); }

}  // namespace

bool GenericNativeEnabled() {
    static const bool on = [] { const char* v = std::getenv("BRAWL_JIT_GENERIC"); return v == nullptr || *v != '0'; }();
    return on;
}

bool GenericStatsOn() {
    static const bool on = EnvOn("BRAWL_JIT_GENERIC_STATS");
    return on;
}

uint64_t* GenericRunCounter(uint32_t word) { return &Stats().run[KeyOf(word)]; }

// --- facts -----------------------------------------------------------------------------------------------------------------------

GenericFacts DescribeGeneric(uint32_t w) {
    using F = GenericFacts;
    const Fields f(w);
    const F barrier{};
    const F pass{F::CrPass, F::FormKeep, 0, 0};
    const F store{F::CrBarrier, F::FormKeep, 0, 0};  // a store's slow path can throw: the condition register stays visible
    const auto writer0 = [&] { return F{F::CrWriter, F::FormKeep, 0, 0}; };
    const auto floatLoad = [&] { return F{F::CrPass, F::FormScalar, 0, static_cast<uint8_t>(f.d)}; };
    switch (f.po) {
    case 3: return pass;  // twi
    case 12: return pass;
    case 13: return writer0();
    case 19:
        switch (f.xo) {
        case 0: case 33: case 129: case 193: case 225: case 257: case 289: case 417: case 449:
            return F{F::CrBarrier, F::FormKeep, 0, 0};  // reads (and writes) the condition register
        case 150: return pass;  // isync
        default: return barrier;
        }
    case 20: return f.rc != 0 ? writer0() : pass;  // rlwimi.
    case 31:
        if (IsXoArithmetic(f.xo & 0x1FFu)) return f.rc != 0 ? writer0() : pass;  // OE only touches XER
        if (IsWriteA(f.xo)) return f.rc != 0 ? writer0() : pass;
        if (IsNop31(f.xo)) return pass;
        switch (f.xo) {
        case 144: return F{F::CrBarrier, F::FormKeep, 0, 0};  // mtcrf
        case 83: return pass;                                    // mfmsr
        case 20: case 23: case 55: case 87: case 119: case 279: case 311: case 343: case 375: case 534: case 790: return pass;
        case 535: case 567: case 599: case 631: return floatLoad();
        case 150: case 151: case 183: case 215: case 247: case 407: case 439: case 662: case 918: case 663: case 695: case 727:
        case 759: case 983: case 1014:
            return store;
        default: return barrier;
        }
    case 32: case 33: case 34: case 35: case 40: case 41: case 42: case 43: return pass;
    case 36: case 37: case 38: case 39: case 44: case 45: case 46: case 47: return store;
    case 48: case 49: case 50: case 51: return floatLoad();
    case 52: case 53: case 54: case 55: return store;
    case 59:
        if (f.rc == 0 && ((w >> 1) & 31u) == 24) return F{F::CrPass, F::FormScalar, 0, static_cast<uint8_t>(f.d)};  // fres
        return barrier;
    case 63: {
        if (f.rc != 0) return barrier;
        const uint32_t xo5 = (w >> 1) & 31u, xo10 = (w >> 1) & 1023u;
        if (xo5 == 23 || xo5 == 30 || xo5 == 31) return F{F::CrPass, F::FormScalar, 0, static_cast<uint8_t>(f.d)};  // fsel, fnmsub, fnmadd
        if (xo10 == 14 || xo10 == 15 || xo10 == 26 || xo10 == 136)  // fctiw, fctiwz, frsqrte, fnabs
            return F{F::CrPass, F::FormScalar, 0, static_cast<uint8_t>(f.d)};
        return barrier;
    }
    default: return barrier;
    }
}

// --- the entry point -------------------------------------------------------------------------------------------------------------

namespace {

uint32_t EmitGenericWord(BlockEmitter& e, uint32_t w) {
    const Fields f(w);
    switch (f.po) {
    case 3: return 1;  // twi: a debug assert, nothing here
    case 12:    // addic
    case 13: {  // addic.
        auto& c = e.a;
        c.mov(eax, e.Gpr(f.a));
        c.add(eax, static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(w & 0xFFFFu))));
        c.mov(e.Gpr(f.d), eax);
        SetCaFromCf(e);
        if (f.po == 13) Cr0From(e, eax);
        return 1;
    }
    case 19:
        switch (f.xo) {
        case 0: EmitMcrf(e, f.d >> 2, f.a >> 2); return 1;
        case 33: case 129: case 193: case 225: case 257: case 289: case 417: case 449:
            EmitCrLogical(e, f.xo, f.d, f.a, f.b);
            return 1;
        case 150: return 1;  // isync
        default: return 0;
        }
    case 20:  // rlwimi. (the form without Rc is a kind)
        if (f.rc == 0) return 0;
        return PlainThenCr0(e, K::Rlwimi, f, f.a, Mask((w >> 6) & 31u, (w >> 1) & 31u));
    case 31:
        if (IsXoArithmetic(f.xo & 0x1FFu)) {
            if (f.oe != 0) return 0;  // overflow forms: the interpreter's
            return EmitXo(e, f, f.xo & 0x1FFu);
        }
        if (IsWriteA(f.xo)) return EmitWriteA(e, f);
        if (IsNop31(f.xo)) return 1;
        if (f.xo == 144) {
            e.CrFromGpr(f.d, (w >> 12) & 0xFFu);
            return 1;
        }
        return EmitMemGeneric(e, w);
    case 59:
    case 63: return EmitFpGeneric(e, w);
    default: return EmitMemGeneric(e, w);
    }
}

}  // namespace

uint32_t EmitGeneric(BlockEmitter& e, uint32_t index) {
    const uint32_t w = e.block.ops[index].imm;
    uint32_t used = 0;
    if (GenericNativeEnabled()) used = EmitGenericWord(e, w);
    if (GenericStatsOn()) (used != 0 ? Stats().compiledNative : Stats().compiledFallback)[KeyOf(w)].fetch_add(1, std::memory_order_relaxed);
    return used;
}

}  // namespace PpcInterpreter::detail::x64
