// The condition register in compiled blocks (docs/jit-design.md): which CR fields a block's ops need to write at all, and the
// code that makes a field's value.
//
// A compare or record form writes its CR field through memory (ctx->crv[field]). Most of those writes are never read: the next
// compare of the same field overwrites them before a branch, mfcr or the end of the block. AnalyseCr finds them once, when
// the block is compiled; the integer and branch groups skip the write (CrDead) or, for a compare fused with its bc, write the
// field only on the taken path.
//
// A field written by op i is dead after i when, walking the ops after i on the fall-through path, a later op overwrites the
// field in full before one that can read the condition register or leave the block is reached. Only op kinds that are
// positively classified as neither are skipped over (Pass); everything not listed is a Barrier and keeps every field live:
//
//  * leaves the block or can: every branch and call (a taken bc leaves with the whole condition register visible), the
//    block's end, a store (its slow path can throw an access violation, which unwinds with ctx->crv as it is), the
//    quantized loads and stores (the same), lmw (it faults at the window's unbacked end);
//  * reads it: bc and bclr (also the bc half of a fused pair), mfcr, and the Generic ops DescribeGeneric does not list as
//    passing or writing (mtcrf, mcrf, crand ..., and every word it does not know);
//  * runs code that may do either: mfspr and mtspr (an unknown spr takes the interpreter's handler).
//
// What may be passed over: the integer ops that only move GPRs, lr, ctr and XER[CA] (the Rc forms are Writers), the Generic
// words DescribeGeneric lists (addic, the carry and multiply and divide forms, orc/nand/eqv, indexed loads, ...; the Rc forms are
// Writers of cr0), integer and float loads other than lmw, and the arithmetic and move ops of the float group (their slow path is the interpreter's
// handler for the same op, which touches registers and forms only). A Writer overwrites one field in full: the compares
// (Cmpwi ... Cmplw, Fcmpu), and the record forms, which set cr0. A fused compare and bc kills its compare's field (the compare
// comes first) and then, being a bc, reads everything: the fields before the pair are all live except that one.
#include "jit_x64.h"
#include "jit_x64_generic.h"

#include <atomic>
#include <cstdlib>

#include "isa/ppc_isa_crv.h"  // MKW_LEAN_FLAGS, the field encoding

namespace PpcInterpreter::detail {
namespace x64 {

namespace {

using namespace Xbyak::util;

enum class CrUse : uint8_t { Pass, Writer, Barrier };

struct CrClass {
    CrUse use;
    uint32_t field;  // Writer
};

bool IsFusedCompare(K k) {
    return k == K::CmpwiBc || k == K::CmplwiBc || k == K::CmpwBc || k == K::CmplwBc || k == K::RlwinmRcBc;
}

CrClass Classify(const Op& o) {
    switch (o.kind) {
    // overwrite one field in full
    case K::Cmpwi: case K::Cmplwi: case K::Cmpw: case K::Cmplw: case K::Fcmpu:
        return {CrUse::Writer, static_cast<uint32_t>(o.d) >> 2};
    case K::AndiRc: case K::AndisRc: case K::RlwinmRc: case K::OrRc: case K::AndRc: case K::RlwnmRc:
        return {CrUse::Writer, 0};
    // neither read nor leave
    case K::Jmp:
    case K::Li: case K::Addi: case K::Lis: case K::Addis: case K::Mulli: case K::Ori: case K::Oris: case K::Xori:
    case K::Xoris: case K::Rlwinm: case K::Rlwimi: case K::Add: case K::Subf: case K::Neg: case K::Mullw: case K::Or:
    case K::And: case K::Xor: case K::Andc: case K::Nor: case K::Slw: case K::Srw: case K::Srawi: case K::Extsb:
    case K::Extsh: case K::Cntlzw: case K::Mflr: case K::Mtlr: case K::Mfctr: case K::Mtctr: case K::Subfic:
    case K::Rlwnm:
    case K::Lwz: case K::Lwzu: case K::Lbz: case K::Lbzu: case K::Lhz: case K::Lhzu: case K::Lha: case K::Lwzx:
    case K::Lbzx: case K::Lhzx: case K::Lfs: case K::Lfsu: case K::Lfd: case K::Lfdu: case K::Lfsx:
    case K::Fadds: case K::Fsubs: case K::Fmuls: case K::Fdivs: case K::Fmadds: case K::Fmsubs: case K::Fnmsubs:
    case K::Fnmadds: case K::Fadd: case K::Fsub: case K::Fmul: case K::Fdiv: case K::Fmadd: case K::Fmsub: case K::Fneg:
    case K::Fabs: case K::Fmr: case K::Frsp: case K::PsAdd: case K::PsSub: case K::PsMul: case K::PsMadd: case K::PsMsub:
    case K::PsNmadd: case K::PsNmsub: case K::PsMuls0: case K::PsMuls1: case K::PsMadds0: case K::PsMadds1:
    case K::PsSum0: case K::PsSum1: case K::PsMerge00: case K::PsMerge01: case K::PsMerge10: case K::PsMerge11:
    case K::PsMr: case K::PsNeg: case K::PsAbs:
        return {CrUse::Pass, 0};
    case K::Generic: {  // by the word (jit_x64_generic.cpp): the same for the native code and the interpreter's handler
        const GenericFacts g = DescribeGeneric(o.imm);
        switch (g.cr) {
        case GenericFacts::CrPass: return {CrUse::Pass, 0};
        case GenericFacts::CrWriter: return {CrUse::Writer, g.crField};
        case GenericFacts::CrBarrier: break;
        }
        return {CrUse::Barrier, 0};
    }
    default:
        return {CrUse::Barrier, 0};
    }
}

struct CrCounters {
    std::atomic<uint64_t> writesSkipped{0}, fusedFlagsOnly{0}, compactWrites{0}, analysed{0}, fusedLive{0};
};
CrCounters g_cr;

}  // namespace

void BlockEmitter::AnalyseCr() {
    const uint32_t n = block.count;
    // BRAWL_JIT_NO_CR_LIVENESS=1 keeps every field live (a diagnostic: the code of the unoptimised compare and bc forms).
    static const bool off = [] { const char* v = std::getenv("BRAWL_JIT_NO_CR_LIVENESS"); return v != nullptr && *v != '\0' && *v != '0'; }();
    if (off) {
        for (uint8_t& d : crDead_) d = 0;
        g_cr.analysed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    uint32_t start[kMaxBlock + 1];
    uint8_t span[kMaxBlock + 1];
    uint32_t items = 0;
    for (uint32_t i = 0; i < n;) {
        const uint8_t s = IsFusedCompare(block.ops[i].kind) && i + 1 < n ? 2 : 1;
        start[items] = i;
        span[items++] = s;
        i += s;
    }
    uint8_t dead = 0;  // the fields dead before the item being looked at (after the one before it in the walk)
    for (uint32_t k = items; k-- > 0;) {
        const uint32_t i = start[k];
        crDead_[i] = dead;
        if (span[k] == 2) crDead_[i + 1] = dead;
        if (span[k] == 2) {
            // the compare writes its field, then the bc leaves with everything visible
            const Op& o = block.ops[i];
            const uint32_t field = o.kind == K::RlwinmRcBc ? 0u : static_cast<uint32_t>(o.d) >> 2;
            dead = static_cast<uint8_t>(1u << field);
            continue;
        }
        const CrClass c = Classify(block.ops[i]);
        switch (c.use) {
        case CrUse::Pass: break;
        case CrUse::Writer: dead = static_cast<uint8_t>(dead | (1u << c.field)); break;
        case CrUse::Barrier: dead = 0; break;
        }
    }
    crDead_[n] = 0;  // End: leaves
    g_cr.analysed.fetch_add(1, std::memory_order_relaxed);
}

// --- field values -------------------------------------------------------------------------------------------------------------
//
// What the interpreter stores for a compare is PpcCrvCompare's: the 64-bit difference of the operands, sign- or zero-extended,
// with XER[SO] copied in through PpcCrvWithSo. The code here makes that value in rax and stores it: three or four instructions
// for a compare, and the SO path out of line.

void BlockEmitter::CrDiffImm(bool isSigned, uint32_t imm, bool flags) {
    if (imm == 0) {
        if (flags) a.test(rax, rax);
    } else if (isSigned || imm < 0x80000000u) {  // a sign-extended 32-bit immediate: the signed operand, or an unsigned one below 2^31
        a.sub(rax, imm);
    } else {
        a.mov(ecx, imm);
        a.sub(rax, rcx);
    }
}

void BlockEmitter::CrDiffGpr(bool isSigned, uint32_t reg) {
    const int hit = FindCachedGpr(reg);
    if (hit >= 0) {
        const Xbyak::Reg32& src = GprCacheReg(static_cast<uint32_t>(hit));
        if (isSigned) a.movsxd(rcx, src);
        else a.mov(ecx, src);
    } else {
        if (isSigned) a.movsxd(rcx, Gpr(reg));
        else a.mov(ecx, Gpr(reg));
    }
    a.sub(rax, rcx);
}

void BlockEmitter::CrStore(uint32_t field) {
    if (!MKW_LEAN_FLAGS) {  // XER[SO] is copied into the field; translated code never sets it, so the call is out of line
        Xbyak::Label& so = NewLabel();
        Xbyak::Label& store = NewLabel();
        const GprSnapshot snap = SnapshotGprs();
        a.cmp(Xer(), 0);
        a.js(so);
        a.L(store);
        Cold([&so, &store, snap](BlockEmitter& cold) {
            cold.a.L(so);
            cold.EmitFlushSnapshot(snap);
            cold.CallHelper(&PpcCrvWithSo, rax);
            cold.EmitReloadSnapshotVolatile(snap);
            cold.a.jmp(store);
        });
    }
    a.mov(Crv(field), rax);
    NoteCr(CrEvent::CompactWrite);
}

// mfcr: the interpreter's code (PpcGetCr) behind a noexcept call.
namespace {

uint32_t CrPackCall(Frame* f) noexcept { return PpcGetCr(f->ctx); }

// Extracts CR bit `bit` (0 LT, 1 GT, 2 EQ, 3 SO) of field `field` into `dst` (0 or 1). Clobbers `tmp` and flags.
void EmitExtractCrBit(BlockEmitter& e, uint32_t field, uint32_t bit, const Xbyak::Reg64& dst, const Xbyak::Reg64& tmp) {
    auto& a = e.a;
    const Xbyak::Address cr = e.Crv(field);
    switch (bit & 3u) {
    case 0:  // LT: bit 62
        a.mov(dst, cr);
        a.shr(dst, 62);
        a.and_(dst.cvt32(), 1);
        break;
    case 1:  // GT: signed > 0
        a.cmp(cr, 0);
        a.setg(dst.cvt8());
        a.movzx(dst.cvt32(), dst.cvt8());
        break;
    case 2:  // EQ: low 32 bits == 0
        a.cmp(a.dword[cr.getRegExp()], 0);
        a.sete(dst.cvt8());
        a.movzx(dst.cvt32(), dst.cvt8());
        break;
    default:  // SO: bit 59 ^ bit 62
        a.mov(dst, cr);
        a.mov(tmp, dst);
        a.shr(tmp, 3);
        a.xor_(dst, tmp);
        a.shr(dst, 59);
        a.and_(dst.cvt32(), 1);
        break;
    }
}

}  // namespace

void BlockEmitter::CrPack() { CallHelper(&CrPackCall, kFrame); }

// Native crnor..cror (op 0..7 as PpcCrvLogic's) on CR bits bt, ba, bb: computes the result bit in eax (0 or 1),
// extracts the other 3 bits of field bt>>2 into a nibble, and writes kPpcCrvTable[nibble] into Crv(bt>>2).
void BlockEmitter::CrLogic(uint32_t op, uint32_t bt, uint32_t ba, uint32_t bb) {
    const uint32_t btField = (bt >> 2) & 7u, btBit = bt & 3u;
    const uint32_t baField = (ba >> 2) & 7u, baBit = ba & 3u;
    const uint32_t bbField = (bb >> 2) & 7u, bbBit = bb & 3u;
    // Compute result bit in eax (0 or 1).
    if (ba == bb) {
        if (op == 2) {  // crxor a, a -> 0 (crclr)
            a.xor_(eax, eax);
        } else if (op == 5) {  // creqv a, a -> 1 (crset)
            a.mov(eax, 1);
        } else if (op == 1) {  // crandc a, a -> 0
            a.xor_(eax, eax);
        } else if (op == 6) {  // crorc a, a -> 1
            a.mov(eax, 1);
        } else {
            EmitExtractCrBit(*this, baField, baBit, rax, rcx);
            if (op == 0 || op == 3) a.xor_(eax, 1);  // crnor/crnand -> !a (crnot); crand/cror -> a (crmove)
        }
    } else {
        EmitExtractCrBit(*this, baField, baBit, rax, rdx);
        EmitExtractCrBit(*this, bbField, bbBit, rcx, rdx);
        switch (op) {
        case 0: a.or_(eax, ecx); a.xor_(eax, 1); break;   // crnor
        case 1: a.xor_(ecx, 1); a.and_(eax, ecx); break;  // crandc
        case 2: a.xor_(eax, ecx); break;                  // crxor
        case 3: a.and_(eax, ecx); a.xor_(eax, 1); break;  // crnand
        case 4: a.and_(eax, ecx); break;                  // crand
        case 5: a.xor_(eax, ecx); a.xor_(eax, 1); break;  // creqv
        case 6: a.xor_(ecx, 1); a.or_(eax, ecx); break;   // crorc
        default: a.or_(eax, ecx); break;                  // cror
        }
    }
    // Extract the 4-bit nibble of target field btField into edx (skipping the overwritten bit btBit), then OR in eax.
    a.mov(rsi, Crv(btField));
    const uint32_t shift = 3u - btBit;
    if (shift != 0) a.shl(eax, static_cast<uint8_t>(shift));
    if (btBit != 0) {  // LT -> bit 3
        a.mov(rcx, rsi);
        a.shr(rcx, 59);
        a.and_(ecx, 8);
        a.or_(eax, ecx);
    }
    if (btBit != 1) {  // GT -> bit 2
        a.test(rsi, rsi);
        a.setg(cl);
        a.movzx(ecx, cl);
        a.shl(ecx, 2);
        a.or_(eax, ecx);
    }
    if (btBit != 2) {  // EQ -> bit 1
        a.test(esi, esi);
        a.sete(cl);
        a.movzx(ecx, cl);
        a.add(ecx, ecx);
        a.or_(eax, ecx);
    }
    if (btBit != 3) {  // SO -> bit 0
        a.mov(rcx, rsi);
        a.mov(rdx, rsi);
        a.shr(rdx, 3);
        a.xor_(rcx, rdx);
        a.shr(rcx, 59);
        a.and_(ecx, 1);
        a.or_(eax, ecx);
    }
    a.mov(rsi, reinterpret_cast<uintptr_t>(&kPpcCrvTable[0]));
    a.mov(rax, a.qword[rsi + rax * 8]);
    a.mov(Crv(btField), rax);
}

void BlockEmitter::CrFromGpr(uint32_t rs, uint32_t crm) {
    if (crm == 0) return;
    a.mov(edx, Gpr(rs));
    a.mov(rsi, reinterpret_cast<uintptr_t>(&kPpcCrvTable[0]));
    for (uint32_t k = 0; k < 8; ++k) {
        if ((crm & (0x80u >> k)) == 0) continue;
        const uint32_t shift = 28u - 4u * k;  // field k's nibble of the word
        a.mov(ecx, edx);
        if (shift != 0) a.shr(ecx, static_cast<uint8_t>(shift));
        if (shift != 28) a.and_(ecx, 0xF);
        a.mov(rax, a.qword[rsi + rcx * 8]);
        a.mov(Crv(k), rax);
    }
}

void BlockEmitter::NoteCr(CrEvent e) {
    switch (e) {
    case CrEvent::WriteSkipped: g_cr.writesSkipped.fetch_add(1, std::memory_order_relaxed); break;
    case CrEvent::FusedFlagsOnly: g_cr.fusedFlagsOnly.fetch_add(1, std::memory_order_relaxed); break;
    case CrEvent::CompactWrite: g_cr.compactWrites.fetch_add(1, std::memory_order_relaxed); break;
    case CrEvent::FusedLive: g_cr.fusedLive.fetch_add(1, std::memory_order_relaxed); break;
    }
}

}  // namespace x64

JitX64CrStats JitX64GetCrStats() {
    return JitX64CrStats{x64::g_cr.analysed.load(std::memory_order_relaxed), x64::g_cr.writesSkipped.load(std::memory_order_relaxed),
                         x64::g_cr.fusedFlagsOnly.load(std::memory_order_relaxed), x64::g_cr.compactWrites.load(std::memory_order_relaxed),
                         x64::g_cr.fusedLive.load(std::memory_order_relaxed)};
}

}  // namespace PpcInterpreter::detail
