// Native code for the integer arithmetic, logic, rotates, compares and spr moves (Li ... Mtspr) of the x86-64 backend,
// written against the emitter API in jit_x64.h. Every function here reproduces the matching case of RunOp
// (ppc_interpreter.cpp) bit for bit; the Decode comments there say what d, a, b and imm hold per kind.
//
// Conventions: eax is the result register, ecx and edx are scratch, and no
// value lives in a host register across ops. Memory operands are used directly where that is shorter. A compare makes its CR
// field's value as the interpreter does (PpcCrvCompare: the 64-bit difference of the extended operands, BlockEmitter::CrStore
// adds XER[SO]) and stores it with one move; XER[CA] is updated without branches.
#include "jit_x64.h"
#include "jit_x64_generic.h"

#include <cstddef>



namespace PpcInterpreter::detail::x64 {

namespace {

using namespace Xbyak::util;

constexpr uint32_t kXerCa = 0x20000000u;

// Context dword accessor for special registers.
Xbyak::Address CtxDword(BlockEmitter& e, size_t offset) { return e.a.dword[kCtxBase + CtxDisp(offset)]; }

// --- small emit helpers ----------------------------------------------------------------------------------------------

// dst = src, through cached GPRs when live later (nothing when they are the same register).
void EmitMove(BlockEmitter& e, uint32_t dst, uint32_t src) {
    if (dst == src) return;
    if (e.GprReadLater(dst)) {
        if (e.WantCacheGpr(src)) {
            const Xbyak::Reg32& s = e.ReadGpr(src);
            const Xbyak::Reg32& d = e.WriteGpr(dst, 1u << src);
            e.a.mov(d, s);
        } else {
            const Xbyak::Reg32& d = e.WriteGpr(dst);
            e.a.mov(d, e.Gpr(src));
        }
        return;
    }
    const int hit = e.FindCachedGpr(src);
    if (hit >= 0) {
        const Xbyak::Reg32& s = BlockEmitter::GprCacheReg(static_cast<uint32_t>(hit));
        e.DropGpr(dst);
        e.a.mov(e.Gpr(dst), s);
        return;
    }
    if (e.GprReadLater(src)) {
        const Xbyak::Reg32& s = e.ReadGpr(src);
        e.DropGpr(dst);
        e.a.mov(e.Gpr(dst), s);
        return;
    }
    e.DropGpr(dst);
    e.a.mov(eax, e.Gpr(src));
    e.a.mov(e.Gpr(dst), eax);
}

// The words of the guest special registers the sprs mfspr and mtspr reach (the table of ReadSpr and WriteSpr), or
// false for a spr without one.
bool SprWord(BlockEmitter& e, uint32_t spr, Xbyak::Address& out) {
    switch (spr) {
    case 1: out = e.Xer(); return true;
    case 8: out = e.Lr(); return true;
    case 9: out = e.Ctr(); return true;
    case 26: out = CtxDword(e, offsetof(CpuContext, srr0)); return true;
    case 27: out = CtxDword(e, offsetof(CpuContext, srr1)); return true;
    case 920: out = CtxDword(e, offsetof(CpuContext, hid2)); return true;
    case 1008: out = CtxDword(e, offsetof(CpuContext, hid0)); return true;
    case 1009: out = CtxDword(e, offsetof(CpuContext, hid1)); return true;
    default: break;
    }
    if (spr >= 912 && spr <= 919) { out = e.Gqr(spr - 912); return true; }
    return false;
}

// --- CR fields -------------------------------------------------------------------------------------------------------

// CR0 from the value in reg (SetCr0: a signed compare against zero, whose difference is the value itself).
void EmitSetCr0From(BlockEmitter& e, const Xbyak::Reg32& reg) {
    e.a.movsxd(rax, reg);
    e.CrStore(0);
}

// A compare or record form whose CR field no later op of the block reads (BlockEmitter::CrDead) writes nothing: a compare
// is then no code at all, and a record form is its plain form.
bool CrDeadNote(BlockEmitter& e, uint32_t field) {
    if (!e.CrDead(e.index, field)) return false;
    BlockEmitter::NoteCr(BlockEmitter::CrEvent::WriteSkipped);
    return true;
}

// --- logical operations ------------------------------------------------------------------------------------------------

enum class Logic { And, Or, Xor };

void LogicRegReg(BlockEmitter& e, Logic op, const Xbyak::Reg32& d, const Xbyak::Reg32& s) {
    switch (op) {
    case Logic::And: e.a.and_(d, s); break;
    case Logic::Or: e.a.or_(d, s); break;
    case Logic::Xor: e.a.xor_(d, s); break;
    }
}
void LogicRegImm(BlockEmitter& e, Logic op, const Xbyak::Reg32& r, uint32_t imm) {
    switch (op) {
    case Logic::And: e.a.and_(r, imm); break;
    case Logic::Or: e.a.or_(r, imm); break;
    case Logic::Xor: e.a.xor_(r, imm); break;
    }
}

// r[dst] = r[src] op imm (ori, andi., xori and their shifted forms). The Rc forms also set CR0.
void EmitLogicImm(BlockEmitter& e, Logic op, uint32_t dst, uint32_t src, uint32_t imm, bool rc) {
    if (!rc && op != Logic::And && imm == 0) {
        EmitMove(e, dst, src);
        return;
    }
    if (dst == src) {
        const Xbyak::Reg32& d = e.ModifyGpr(dst);
        if (op == Logic::And && imm == 0) e.a.xor_(d, d);
        else if (imm != 0 || op == Logic::And) LogicRegImm(e, op, d, imm);
        if (rc) EmitSetCr0From(e, d);
        return;
    }
    const Xbyak::Reg32& s = e.ReadGpr(src);
    const Xbyak::Reg32& d = e.WriteGpr(dst, 1u << src);
    if (op == Logic::And && imm == 0) {
        e.a.xor_(d, d);
    } else {
        e.a.mov(d, s);
        if (imm != 0 || op == Logic::And) LogicRegImm(e, op, d, imm);
    }
    if (rc) EmitSetCr0From(e, d);
}

// r[dst] = r[s1] op r[s2], with every aliasing of the three registers; Rc: also CR0.
void EmitLogicReg(BlockEmitter& e, Logic op, uint32_t dst, uint32_t s1, uint32_t s2, bool rc) {
    if (s1 == s2) {
        if (op == Logic::Xor) {
            const Xbyak::Reg32& d = e.WriteGpr(dst);
            e.a.xor_(d, d);
            if (rc) EmitSetCr0From(e, d);
            return;
        }
        EmitMove(e, dst, s1);
        if (rc) EmitSetCr0From(e, e.ReadGpr(dst));
        return;
    }
    if (dst == s1 || dst == s2) {
        const uint32_t other = dst == s1 ? s2 : s1;
        const Xbyak::Reg32& o = e.ReadGpr(other);
        const Xbyak::Reg32& d = e.ModifyGpr(dst, 1u << other);
        LogicRegReg(e, op, d, o);
        if (rc) EmitSetCr0From(e, d);
        return;
    }
    const Xbyak::Reg32& r1 = e.ReadGpr(s1);
    const Xbyak::Reg32& r2 = e.ReadGpr(s2, 1u << s1);
    const Xbyak::Reg32& d = e.WriteGpr(dst, (1u << s1) | (1u << s2));
    e.a.mov(d, r1);
    LogicRegReg(e, op, d, r2);
    if (rc) EmitSetCr0From(e, d);
}

// --- rotates -------------------------------------------------------------------------------------------------------------

void EmitRlwinm(BlockEmitter& e, const Op& o, bool rc) {
    auto& c = e.a;
    const uint32_t src = o.d, dst = o.a, sh = o.b, mask = o.imm;
    if (mask == 0) {
        const Xbyak::Reg32& d = e.WriteGpr(dst);
        c.xor_(d, d);
        if (rc) EmitSetCr0From(e, d);
        return;
    }
    if (sh == 0 && mask == 0xFFFFFFFFu) {
        EmitMove(e, dst, src);
        if (rc) EmitSetCr0From(e, e.ReadGpr(dst));
        return;
    }
    const Xbyak::Reg32& s = e.ReadGpr(src);
    const Xbyak::Reg32& d = dst == src ? e.ModifyGpr(dst) : e.WriteGpr(dst, 1u << src);
    if (sh == 0 && mask == 0xFFu) {
        c.movzx(d, s.cvt8());
    } else if (sh == 0 && mask == 0xFFFFu) {
        c.movzx(d, s.cvt16());
    } else {
        if (dst != src) c.mov(d, s);
        if (sh != 0 && mask == (0xFFFFFFFFu << sh)) {
            c.shl(d, static_cast<uint8_t>(sh));
        } else if (sh != 0 && mask == (0xFFFFFFFFu >> (32u - sh))) {
            c.shr(d, static_cast<uint8_t>(32u - sh));
        } else {
            if (sh != 0) c.rol(d, static_cast<uint8_t>(sh));
            if (mask != 0xFFFFFFFFu) c.and_(d, mask);
        }
    }
    if (rc) EmitSetCr0From(e, d);
}

// r[a] = (rotl(r[d], sh) & mask) | (r[a] & ~mask)
void EmitRlwimi(BlockEmitter& e, const Op& o) {
    const uint32_t src = o.d, dst = o.a, sh = o.b, mask = o.imm;
    if (mask == 0) return;
    if (mask == 0xFFFFFFFFu) {
        if (sh == 0) { EmitMove(e, dst, src); return; }
        const Xbyak::Reg32& s = e.ReadGpr(src);
        const Xbyak::Reg32& d = dst == src ? e.ModifyGpr(dst) : e.WriteGpr(dst, 1u << src);
        if (dst != src) e.a.mov(d, s);
        e.a.rol(d, static_cast<uint8_t>(sh));
        return;
    }
    const Xbyak::Reg32& s = e.ReadGpr(src);
    if (sh == 0 && mask == 0xFFu) {
        e.a.movzx(eax, s.cvt8());
    } else {
        e.a.mov(eax, s);
        if (sh != 0) e.a.rol(eax, static_cast<uint8_t>(sh));
        e.a.and_(eax, mask);
    }
    const Xbyak::Reg32& d = e.ModifyGpr(dst);
    e.a.and_(d, ~mask);
    e.a.or_(d, eax);
}

// r[a] = rotl(r[d], r[b] & 31) & mask
void EmitRlwnm(BlockEmitter& e, const Op& o, bool rc) {
    auto& c = e.a;
    c.mov(ecx, e.ReadGpr(o.b));
    const Xbyak::Reg32& s = e.ReadGpr(o.d, 1u << o.b);
    const Xbyak::Reg32& d = o.a == o.d ? e.ModifyGpr(o.a) : e.WriteGpr(o.a, 1u << o.d);
    if (o.a != o.d) c.mov(d, s);
    c.rol(d, cl);  // the count is taken mod 32 by the hardware, as & 31u
    if (o.imm != 0xFFFFFFFFu) c.and_(d, o.imm);
    if (rc) EmitSetCr0From(e, d);
}

// --- arithmetic ------------------------------------------------------------------------------------------------------------

// r[d] = r[a] + imm
void EmitAddImm(BlockEmitter& e, uint32_t d, uint32_t a, uint32_t imm) {
    if (imm == 0) { EmitMove(e, d, a); return; }
    if (d == a) {
        if (e.WantCacheGpr(d)) e.a.add(e.ModifyGpr(d), imm);
        else e.a.add(e.Gpr(d), imm);
        return;
    }
    if (e.GprReadLater(d)) {
        if (e.WantCacheGpr(a)) {
            const Xbyak::Reg32& s = e.ReadGpr(a);
            const Xbyak::Reg32& dst = e.WriteGpr(d, 1u << a);
            e.a.lea(dst, e.a.ptr[s.cvt64() + static_cast<int32_t>(imm)]);
        } else {
            const Xbyak::Reg32& dst = e.WriteGpr(d);
            e.a.mov(dst, e.Gpr(a));
            e.a.add(dst, imm);
        }
        return;
    }
    if (e.WantCacheGpr(a)) {
        const Xbyak::Reg32& s = e.ReadGpr(a);
        e.DropGpr(d);
        e.a.lea(eax, e.a.ptr[s.cvt64() + static_cast<int32_t>(imm)]);
        e.a.mov(e.Gpr(d), eax);
        return;
    }
    e.DropGpr(d);
    e.a.mov(eax, e.Gpr(a));
    e.a.add(eax, imm);
    e.a.mov(e.Gpr(d), eax);
}

void EmitAdd(BlockEmitter& e, const Op& o) {
    if (o.d == o.a || o.d == o.b) {
        const uint32_t other = o.d == o.a ? o.b : o.a;
        const Xbyak::Reg32& rOther = e.ReadGpr(other);
        const Xbyak::Reg32& dst = e.ModifyGpr(o.d, 1u << other);
        e.a.add(dst, rOther);
        return;
    }
    const Xbyak::Reg32& ra = e.ReadGpr(o.a);
    const Xbyak::Reg32& rb = e.ReadGpr(o.b, 1u << o.a);
    const Xbyak::Reg32& dst = e.WriteGpr(o.d, (1u << o.a) | (1u << o.b));
    e.a.lea(dst, e.a.ptr[ra.cvt64() + rb.cvt64()]);
}

// r[d] = r[b] - r[a]
void EmitSubf(BlockEmitter& e, const Op& o) {
    if (o.a == o.b) {
        const Xbyak::Reg32& dst = e.WriteGpr(o.d);
        e.a.xor_(dst, dst);
        return;
    }
    if (o.d == o.b) {
        const Xbyak::Reg32& ra = e.ReadGpr(o.a);
        const Xbyak::Reg32& dst = e.ModifyGpr(o.d, 1u << o.a);
        e.a.sub(dst, ra);
        return;
    }
    if (o.d == o.a) {
        const Xbyak::Reg32& rb = e.ReadGpr(o.b);
        const Xbyak::Reg32& dst = e.ModifyGpr(o.d, 1u << o.b);
        e.a.neg(dst);
        e.a.add(dst, rb);
        return;
    }
    const Xbyak::Reg32& ra = e.ReadGpr(o.a);
    const Xbyak::Reg32& rb = e.ReadGpr(o.b, 1u << o.a);
    const Xbyak::Reg32& dst = e.WriteGpr(o.d, (1u << o.a) | (1u << o.b));
    e.a.mov(dst, rb);
    e.a.sub(dst, ra);
}

// Sets XER[CA] when ecx is all ones and clears it when ecx is 0 (nothing else in ecx is allowed).
void EmitSetCaFromMask(BlockEmitter& e) {
    e.a.and_(ecx, kXerCa);
    e.a.and_(e.Xer(), ~kXerCa);
    e.a.or_(e.Xer(), ecx);
}

// r[a] = (int32)r[d] >> sh; XER[CA] = r[d] is negative and a 1 bit was shifted out. CA is "sign and any of the low sh
// bits": with x = r & (0x80000000 | (1 << sh) - 1), that is x > 0x80000000.
void EmitSrawi(BlockEmitter& e, const Op& o) {
    auto& c = e.a;
    const uint32_t sh = o.b;
    if (sh == 0) {
        EmitMove(e, o.a, o.d);
        c.and_(e.Xer(), ~kXerCa);
        return;
    }
    const uint32_t keep = 0x80000000u | ((1u << sh) - 1u);
    c.mov(eax, e.ReadGpr(o.d));
    c.mov(ecx, eax);
    if (keep != 0xFFFFFFFFu) c.and_(ecx, keep);
    c.mov(edx, 0x80000000u);
    c.cmp(edx, ecx);  // CF: 0x80000000 < x
    c.sbb(ecx, ecx);
    c.sar(eax, static_cast<uint8_t>(sh));
    c.mov(e.WriteGpr(o.a), eax);
    EmitSetCaFromMask(e);
}

// r[d] = imm - r[a]; XER[CA] = no borrow (imm >= r[a] unsigned), as AddCarry(~a, imm, 1).
void EmitSubfic(BlockEmitter& e, const Op& o) {
    auto& c = e.a;
    const Xbyak::Reg32& ra = e.ReadGpr(o.a);
    c.mov(edx, o.imm);
    c.sub(edx, ra);  // CF: borrow
    c.mov(e.WriteGpr(o.d), edx);
    c.cmc();
    c.sbb(ecx, ecx);
    EmitSetCaFromMask(e);
}

// r[a] = r[d] << (r[b] & 63) as 64 bits, low half: counts 32..63 give 0, as PPC_Slw and PPC_Srw.
void EmitShiftReg(BlockEmitter& e, const Op& o, bool left) {
    auto& c = e.a;
    c.mov(ecx, e.ReadGpr(o.b));
    c.mov(eax, e.ReadGpr(o.d, 1u << o.b));  // zero-extends into rax
#if defined(__BMI2__)
    if (left) c.shlx(rax, rax, rcx);
    else c.shrx(rax, rax, rcx);
#else
    if (left) c.shl(rax, cl);
    else c.shr(rax, cl);
#endif
    c.mov(e.WriteGpr(o.a), eax);
}

void EmitCntlzw(BlockEmitter& e, const Op& o) {
    auto& c = e.a;
    const Xbyak::Reg32& src = e.ReadGpr(o.d);
    const Xbyak::Reg32& dst = o.a == o.d ? e.ModifyGpr(o.a) : e.WriteGpr(o.a, 1u << o.d);
#if defined(__LZCNT__)
    c.lzcnt(dst, src);
#else
    // bsr leaves its destination undefined for 0: pick 63 then, and 63 ^ 31 = 32 while 31 - bsr = bsr ^ 31 otherwise.
    c.mov(edx, 63u);
    c.bsr(eax, src);
    c.cmovz(eax, edx);
    c.xor_(eax, 31u);
    c.mov(dst, eax);
#endif
}

}  // namespace

uint32_t EmitIntOp(BlockEmitter& e, const Op& o) {
    auto& c = e.a;
    switch (o.kind) {
    case K::Li:
    case K::Lis: {
        if (e.GprReadLater(o.d)) {
            const Xbyak::Reg32& dst = e.WriteGpr(o.d);
            if (o.imm == 0) c.xor_(dst, dst);
            else c.mov(dst, o.imm);
        } else {
            e.DropGpr(o.d);
            c.mov(e.Gpr(o.d), o.imm);
        }
        return 1;
    }
    case K::Addi:
    case K::Addis: EmitAddImm(e, o.d, o.a, o.imm); return 1;
    case K::Mulli: {
        const Xbyak::Reg32& src = e.ReadGpr(o.a);
        const Xbyak::Reg32& dst = o.d == o.a ? e.ModifyGpr(o.d) : e.WriteGpr(o.d, 1u << o.a);
        c.imul(dst, src, static_cast<int>(o.imm));
        return 1;
    }
    case K::Ori:
    case K::Oris: EmitLogicImm(e, Logic::Or, o.a, o.d, o.imm, false); return 1;
    case K::Xori:
    case K::Xoris: EmitLogicImm(e, Logic::Xor, o.a, o.d, o.imm, false); return 1;
    case K::AndiRc:
    case K::AndisRc: EmitLogicImm(e, Logic::And, o.a, o.d, o.imm, !CrDeadNote(e, 0)); return 1;
    case K::Rlwinm: EmitRlwinm(e, o, false); return 1;
    case K::RlwinmRc: EmitRlwinm(e, o, !CrDeadNote(e, 0)); return 1;
    case K::Rlwimi: EmitRlwimi(e, o); return 1;
    case K::Cmpwi:
        if (CrDeadNote(e, o.d >> 2)) return 1;
        c.movsxd(rax, e.ReadGpr(o.a));
        e.CrDiffImm(true, o.imm, false);
        e.CrStore(o.d >> 2);
        return 1;
    case K::Cmplwi:
        if (CrDeadNote(e, o.d >> 2)) return 1;
        c.mov(eax, e.ReadGpr(o.a));
        e.CrDiffImm(false, o.imm, false);
        e.CrStore(o.d >> 2);
        return 1;
    case K::Cmpw:
    case K::Cmplw: {
        if (CrDeadNote(e, o.d >> 2)) return 1;
        const bool isSigned = o.kind == K::Cmpw;
        const Xbyak::Reg32& ra = e.ReadGpr(o.a);
        if (isSigned) c.movsxd(rax, ra);
        else c.mov(eax, ra);
        e.CrDiffGpr(isSigned, o.b);
        e.CrStore(o.d >> 2);
        return 1;
    }
    case K::Add: EmitAdd(e, o); return 1;
    case K::Subf: EmitSubf(e, o); return 1;
    case K::Neg: {
        const Xbyak::Reg32& src = e.ReadGpr(o.a);
        const Xbyak::Reg32& dst = o.d == o.a ? e.ModifyGpr(o.d) : e.WriteGpr(o.d, 1u << o.a);
        if (o.d != o.a) c.mov(dst, src);
        c.neg(dst);
        return 1;
    }
    case K::Mullw: {
        const Xbyak::Reg32& ra = e.ReadGpr(o.a);
        const Xbyak::Reg32& rb = e.ReadGpr(o.b, 1u << o.a);
        const Xbyak::Reg32& dst = e.WriteGpr(o.d, (1u << o.a) | (1u << o.b));
        if (o.d == o.a) {
            c.imul(dst, rb);
        } else if (o.d == o.b) {
            c.imul(dst, ra);
        } else {
            c.mov(dst, ra);
            c.imul(dst, rb);
        }
        return 1;
    }
    case K::Or: EmitLogicReg(e, Logic::Or, o.a, o.d, o.b, false); return 1;
    case K::OrRc: EmitLogicReg(e, Logic::Or, o.a, o.d, o.b, !CrDeadNote(e, 0)); return 1;
    case K::And: EmitLogicReg(e, Logic::And, o.a, o.d, o.b, false); return 1;
    case K::AndRc: EmitLogicReg(e, Logic::And, o.a, o.d, o.b, !CrDeadNote(e, 0)); return 1;
    case K::Xor: EmitLogicReg(e, Logic::Xor, o.a, o.d, o.b, false); return 1;
    case K::Andc: {  // r[a] = r[d] & ~r[b]
        if (o.d == o.b) {
            const Xbyak::Reg32& dst = e.WriteGpr(o.a);
            c.xor_(dst, dst);
            return 1;
        }
        const Xbyak::Reg32& rb = e.ReadGpr(o.b);
        const Xbyak::Reg32& rd = e.ReadGpr(o.d, 1u << o.b);
        c.mov(eax, rb);
        c.not_(eax);
        c.and_(eax, rd);
        c.mov(e.WriteGpr(o.a), eax);
        return 1;
    }
    case K::Nor: {  // r[a] = ~(r[d] | r[b])
        if (o.d == o.b) {
            const Xbyak::Reg32& s = e.ReadGpr(o.d);
            const Xbyak::Reg32& dst = o.a == o.d ? e.ModifyGpr(o.a) : e.WriteGpr(o.a, 1u << o.d);
            if (o.a != o.d) c.mov(dst, s);
            c.not_(dst);
            return 1;
        }
        const Xbyak::Reg32& rd = e.ReadGpr(o.d);
        const Xbyak::Reg32& rb = e.ReadGpr(o.b, 1u << o.d);
        const Xbyak::Reg32& dst = e.WriteGpr(o.a, (1u << o.d) | (1u << o.b));
        if (o.a == o.d) c.or_(dst, rb);
        else if (o.a == o.b) c.or_(dst, rd);
        else { c.mov(dst, rd); c.or_(dst, rb); }
        c.not_(dst);
        return 1;
    }
    case K::Slw: EmitShiftReg(e, o, true); return 1;
    case K::Srw: EmitShiftReg(e, o, false); return 1;
    case K::Srawi: EmitSrawi(e, o); return 1;
    case K::Extsb: {
        const Xbyak::Reg32& s = e.ReadGpr(o.d);
        const Xbyak::Reg32& dst = o.a == o.d ? e.ModifyGpr(o.a) : e.WriteGpr(o.a, 1u << o.d);
        c.movsx(dst, s.cvt8());
        return 1;
    }
    case K::Extsh: {
        const Xbyak::Reg32& s = e.ReadGpr(o.d);
        const Xbyak::Reg32& dst = o.a == o.d ? e.ModifyGpr(o.a) : e.WriteGpr(o.a, 1u << o.d);
        c.movsx(dst, s.cvt16());
        return 1;
    }
    case K::Cntlzw: EmitCntlzw(e, o); return 1;
    case K::Mflr:
        if (e.GprReadLater(o.d)) c.mov(e.WriteGpr(o.d), e.Lr());
        else { e.DropGpr(o.d); c.mov(eax, e.Lr()); c.mov(e.Gpr(o.d), eax); }
        return 1;
    case K::Mtlr:
        if (e.WantCacheGpr(o.d)) c.mov(e.Lr(), e.ReadGpr(o.d));
        else { c.mov(eax, e.Gpr(o.d)); c.mov(e.Lr(), eax); }
        return 1;
    case K::Mfctr:
        if (e.GprReadLater(o.d)) c.mov(e.WriteGpr(o.d), e.Ctr());
        else { e.DropGpr(o.d); c.mov(eax, e.Ctr()); c.mov(e.Gpr(o.d), eax); }
        return 1;
    case K::Mtctr:
        if (e.WantCacheGpr(o.d)) c.mov(e.Ctr(), e.ReadGpr(o.d));
        else { c.mov(eax, e.Gpr(o.d)); c.mov(e.Ctr(), eax); }
        return 1;
    case K::Mfcr:
        e.FlushVolatileGprs();
        e.CrPack();
        c.mov(e.WriteGpr(o.d), eax);
        return 1;
    case K::Subfic: EmitSubfic(e, o); return 1;
    case K::Rlwnm: EmitRlwnm(e, o, false); return 1;
    case K::RlwnmRc: EmitRlwnm(e, o, !CrDeadNote(e, 0)); return 1;
    case K::Mfspr: {
        Xbyak::Address word = e.Lr();
        if (!SprWord(e, o.imm, word)) return 0;  // an unknown spr logs: the handler's
        c.mov(e.WriteGpr(o.d), word);
        return 1;
    }
    case K::Mtspr: {
        Xbyak::Address word = e.Lr();
        if (!SprWord(e, o.imm, word)) return 1;  // WriteSpr ignores the rest (decrementer, performance monitors, BATs)
        c.mov(word, e.ReadGpr(o.d));
        return 1;
    }
    default: break;
    }
    return 0;
}

uint32_t EmitInt(BlockEmitter& e, uint32_t index) {
    (void)index;
    return EmitIntOp(e, e.CurrentOp());
}

}  // namespace PpcInterpreter::detail::x64
