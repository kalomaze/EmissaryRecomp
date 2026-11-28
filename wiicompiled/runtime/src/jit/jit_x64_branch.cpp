// Native code for the branches and calls (Bc, fused compare+bc, B ... Bctrl) of the x86-64 backend, written against the emitter
// API in jit_x64.h. Each kind does what RunOp does for it (and Exec where RunOp defers to it), with the same side effects in
// the same order; what the decode or the compile can settle is settled here instead of at every execution.
//
//  Bc and the fused compare+bc kinds: native. BO is decoded here: only the ctr decrement and the tests it needs are emitted;
//      taken goes to ExitLinked. A bc tests one CR field's value in memory as isa/ppc_isa_crv.h defines each bit (EmitJumpIfCrBit).
//      A fused pair writes the compare's CR field, then branches on the host flags of the compare when the bc tests lt, gt or eq
//      of the field just written (else it tests the CR bit in memory).
//  B:  native when its target is no translated function and no code variant can continue there (ExitLinked); a tail call
//      to a translated function is JitThinTail (JitTailOp in the rollback loop frame); with code variants possible, or the
//      registry not yet published, the fallback (RunOp asks ContinuationAt on every execution, which a block cannot settle).
//  Bl: native when the site was resolved at decode, the target is untranslated and runnable: lr and the loop-frame test
//      inline, then ExitLinked. A translated target with a raw dispatch record: lr, pc and the loop-frame test inline (the
//      loop frame goes the general way), the callee's float arguments tested in line, then JitThinCall with the target's
//      JitCallTarget (everything Call and TryDispatchRawCpuTarget look up, settled at compile time), and a validity test.
//      Otherwise JitCallOp (Call) and a validity test when the call was made.
//  Blr, Bclr: native; the frame's return normalises f1, anything else leaves through an inline cache.
//  Bctr: an inline cache probe (untranslated targets only ever get in it); then a cell holding the last translated targets and
//      their JitCallTargets: a hit is JitThinTail, a miss JitTailOpCached, which fills the cell; the fallback until the
//      registry is published.
//  Bctrl: a cell of its own: a hit is JitThinCall, a miss JitCallOpCached, which fills the cell; the interpret-here answer leaves
//      through an inline cache.
#include "jit_x64.h"

#include <cstddef>
#include <cstdlib>

namespace PpcInterpreter::detail::x64 {

namespace {

using namespace Xbyak::util;
using Xbyak::CodeGenerator;

constexpr uint32_t kReturnPending = 3;  // JitCallOp / JitTailOp: what values (ppc_interpreter_internal.h)
constexpr uint32_t kReturnFrame = 2;
constexpr uint32_t kJumpHere = 4;

// What a fused compare left in the flags, to branch on them instead of reading the CR field back.
struct Compare {
    enum Source : uint8_t { Imm, Reg, Test } source;  // eax against an immediate | against a guest register | against 0 (a record form's result)
    bool isSigned;
    uint32_t field;
    int32_t imm;
    uint32_t reg;
};

void EmitCompareFlags(BlockEmitter& e, const Compare& c) {
    switch (c.source) {
    case Compare::Imm:
        if (c.imm == 0) e.a.test(eax, eax);  // the same flags as cmp eax, 0
        else e.a.cmp(eax, static_cast<uint32_t>(c.imm));
        break;
    case Compare::Reg:
        if (e.WantCacheGpr(c.reg)) e.a.cmp(eax, e.ReadGpr(c.reg));
        else e.a.cmp(eax, e.Gpr(c.reg));
        break;
    case Compare::Test: e.a.test(eax, eax); break;
    }
}

// The compare's CR field value in rax, as PpcCrvCompare makes it (before XER[SO]: BlockEmitter::CrStore), from the first operand
// in eax. With `flags` the host flags are those of the 64-bit difference, whose signed conditions (l, g, e) are the compare's for
// both signednesses: the extended operands differ by less than 2^33 in magnitude.
void EmitCompareValue(BlockEmitter& e, const Compare& c, bool flags) {
    if (c.isSigned) e.a.movsxd(rax, eax);
    switch (c.source) {
    case Compare::Imm: e.CrDiffImm(c.isSigned, static_cast<uint32_t>(c.imm), flags); break;
    case Compare::Reg: e.CrDiffGpr(c.isSigned, c.reg); break;
    case Compare::Test:
        if (flags) e.a.test(rax, rax);
        break;
    }
}

// Jumps to `target` when CR bit `bi` (as BI numbers it: field bi >> 2, then 0 lt, 1 gt, 2 eq, 3 so) of the field in memory is
// `want`, by the field value's own tests (isa/ppc_isa_crv.h): lt is bit 62, gt is positive, eq is a zero low word, so is bit 59
// differing from bit 62. Clobbers the flags, and rcx and rdx for so; eax is kept.
void EmitJumpIfCrBit(BlockEmitter& e, uint32_t bi, bool want, Xbyak::Label& target, CodeGenerator::LabelType type) {
    CodeGenerator& a = e.a;
    const Xbyak::Address field = e.Crv(bi >> 2);
    switch (bi & 3u) {
    case 0:
        a.test(a.byte[field.getRegExp() + 7], 0x40);  // bit 62: bit 6 of the top byte
        if (want) a.jnz(target, type);
        else a.jz(target, type);
        break;
    case 1:
        a.cmp(field, 0);
        if (want) a.jg(target, type);
        else a.jle(target, type);
        break;
    case 2:
        a.cmp(a.dword[field.getRegExp()], 0);
        if (want) a.je(target, type);
        else a.jne(target, type);
        break;
    default:
        a.mov(rcx, field);
        a.mov(rdx, rcx);
        a.shr(rdx, 3);  // bit 59 of rdx is bit 62 of the field
        a.xor_(rcx, rdx);
        a.bt(rcx, 59);
        if (want) a.jc(target, type);
        else a.jnc(target, type);
        break;
    }
}

// Jumps to `taken` when CR bit `bit` (0 lt, 1 gt, 2 eq) of the compare in the flags is `want`.
void EmitJumpIfBit(BlockEmitter& e, const Compare& c, uint32_t bit, bool want, Xbyak::Label& taken) {
    CodeGenerator& a = e.a;
    const auto T = CodeGenerator::T_NEAR;
    if (want) {
        switch (bit) {
        case 0: c.isSigned ? a.jl(taken, T) : a.jb(taken, T); break;
        case 1: c.isSigned ? a.jg(taken, T) : a.ja(taken, T); break;
        default: a.je(taken, T); break;
        }
    } else {
        switch (bit) {
        case 0: c.isSigned ? a.jge(taken, T) : a.jae(taken, T); break;
        case 1: c.isSigned ? a.jle(taken, T) : a.jbe(taken, T); break;
        default: a.jne(taken, T); break;
        }
    }
}

// The taken path of a bc: leave for the target. When the bc's compare did not write its CR field (`cmp` and `dead`: nothing
// in the block reads the field after it, BlockEmitter::CrDead), the field is written here, from the compare made again (eax still
// holds its first operand), because the block is left with the condition register visible.
void EmitTaken(BlockEmitter& e, uint32_t target, const Compare* cmp, bool dead, const BlockEmitter::GprSnapshot& snap) {
    e.EmitFlushSnapshot(snap);
    if (dead) {
        EmitCompareValue(e, *cmp, false);
        e.CrStore(cmp->field);
    }
    e.ExitLinked(target);
}

// A bc (op `bc`, already past the compare of a fused pair): BranchTaken's ctr decrement and tests, then the exit to the
// target when taken; falls through when not. `dead`: the compare did not write its field (see EmitTaken; `cmp` is then the
// fused compare, whose flags a test of its own field uses, and eax still holds its first operand); the taken path lies out of
// line, where the field is written, and the tests branch to it. Without `dead` the tests read the field in memory.
void EmitBc(BlockEmitter& e, const Op& bc, const Compare* cmp, bool dead) {
    CodeGenerator& a = e.a;
    const uint32_t bo = bc.d, bi = bc.a;
    const auto T = CodeGenerator::T_SHORT;
    const bool useCtr = (bo & 0x04u) == 0, useCond = (bo & 0x10u) == 0;
    const bool onFlags = dead && useCond && (bi >> 2) == cmp->field && (bi & 3u) != 3u;
    const bool want = (bo & 0x08u) != 0;
    if (!useCtr && !useCond) {
        const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
        EmitTaken(e, bc.imm, cmp, dead, snap);
        return;
    }
    if (dead) {
        // Pre-warm the second compare operand before any conditional branch (useCtr) so the GPR cache state
        // is identical on both the ctr-fail skip path and the compare fall-through path.
        if (onFlags && cmp->source == Compare::Reg && e.WantCacheGpr(cmp->reg)) (void)e.ReadGpr(cmp->reg);
        Xbyak::Label& taken = e.NewLabel();
        Xbyak::Label skip;
        if (useCtr) {  // --ctr, always, even when the condition then fails; ZF is the new ctr == 0
            a.sub(e.Ctr(), 1);
            if (useCond) {
                if ((bo & 0x02u) != 0) a.jnz(skip, T);
                else a.jz(skip, T);
            } else if ((bo & 0x02u) != 0) {
                a.jz(taken, CodeGenerator::T_NEAR);
            } else {
                a.jnz(taken, CodeGenerator::T_NEAR);
            }
        }
        if (useCond) {
            if (onFlags) {
                EmitCompareFlags(e, *cmp);
                EmitJumpIfBit(e, *cmp, bi & 3u, want, taken);
            } else {
                EmitJumpIfCrBit(e, bi, want, taken, CodeGenerator::T_NEAR);
            }
        }
        a.L(skip);
        const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
        const Compare c = *cmp;
        const uint32_t target = bc.imm;
        e.Cold([&taken, c, target, snap](BlockEmitter& cold) {
            cold.a.L(taken);
            EmitTaken(cold, target, &c, true, snap);
        });
        return;
    }
    const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
    Xbyak::Label skip;
    if (useCtr) {  // --ctr, always, even when the condition then fails; ZF is the new ctr == 0
        a.sub(e.Ctr(), 1);
        if ((bo & 0x02u) != 0) a.jnz(skip, T);
        else a.jz(skip, T);
    }
    if (useCond) EmitJumpIfCrBit(e, bi, !want, skip, T);
    e.EmitFlushSnapshot(snap);
    e.ExitLinked(bc.imm);
    a.L(skip);
}

// BRAWL_JIT_NO_FUSED_FLAGS=1 (a test switch): a fused pair whose field is live writes the field first and then branches on the
// field in memory, instead of branching on the host flags.
bool FusedOnFlagsEnabled() {
    static const bool on = [] { const char* v = std::getenv("BRAWL_JIT_NO_FUSED_FLAGS"); return v == nullptr || *v == '\0' || *v == '0'; }();
    return on;
}

// A fused compare and bc whose CR field is live and which the bc tests by a bit of the compare's result (not SO): the compare
// is made once (as the field's 64-bit value, whose flags are the compare's), the branch is on those flags, and the value is stored
// on both paths. eax holds the compare's first operand. The store (CrStore) comes after the branch, so the jump comes straight
// after the compare and neither path has to make it again. A bc that also decrements ctr does that first; when the ctr test alone
// decides "not taken" the compare is made on that path too (cold).
void EmitBcLive(BlockEmitter& e, const Op& bc, const Compare& c) {
    CodeGenerator& a = e.a;
    const uint32_t bo = bc.d, bi = bc.a;
    const bool useCtr = (bo & 0x04u) == 0;
    const bool want = (bo & 0x08u) != 0;
    if (useCtr && c.source == Compare::Reg) e.FlushGpr(c.reg);  // ctrFail cold path reads Gpr(c.reg) from memory
    const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
    Xbyak::Label& taken = e.NewLabel();
    Xbyak::Label* ctrFail = nullptr;
    Xbyak::Label& join = e.NewLabel();
    if (useCtr) {  // --ctr, always; ZF is the new ctr == 0
        ctrFail = &e.NewLabel();
        a.sub(e.Ctr(), 1);
        if ((bo & 0x02u) != 0) a.jnz(*ctrFail, CodeGenerator::T_NEAR);
        else a.jz(*ctrFail, CodeGenerator::T_NEAR);
    }
    EmitCompareValue(e, c, true);
    Compare onFlags = c;
    onFlags.isSigned = true;  // the flags are those of the 64-bit difference
    EmitJumpIfBit(e, onFlags, bi & 3u, want, taken);
    e.CrStore(c.field);  // not taken
    a.L(join);
    const Compare cmp = c;
    const uint32_t target = bc.imm;
    e.Cold([&taken, cmp, target, snap](BlockEmitter& cold) {
        cold.a.L(taken);
        cold.EmitFlushSnapshot(snap);
        cold.CrStore(cmp.field);
        cold.ExitLinked(target);
    });
    if (ctrFail != nullptr) {
        e.Cold([ctrFail, &join, cmp](BlockEmitter& cold) {
            cold.a.L(*ctrFail);
            EmitCompareValue(cold, cmp, false);
            cold.CrStore(cmp.field);
            cold.a.jmp(join, CodeGenerator::T_NEAR);
        });
    }
    BlockEmitter::NoteCr(BlockEmitter::CrEvent::FusedLive);
}

uint32_t EmitFusedCompare(BlockEmitter& e, uint32_t index) {
    const Op& o = e.block.ops[index];
    CodeGenerator& a = e.a;
    const auto loadToEax = [&](uint32_t r) {
        if (e.WantCacheGpr(r)) a.mov(eax, e.ReadGpr(r));
        else a.mov(eax, e.Gpr(r));
    };
    Compare c{};
    switch (o.kind) {
    case K::CmpwiBc:
        c = {Compare::Imm, true, static_cast<uint32_t>(o.d) >> 2u, static_cast<int32_t>(o.imm), 0};
        loadToEax(o.a);
        break;
    case K::CmplwiBc:
        c = {Compare::Imm, false, static_cast<uint32_t>(o.d) >> 2u, static_cast<int32_t>(o.imm), 0};
        loadToEax(o.a);
        break;
    case K::CmpwBc:
        c = {Compare::Reg, true, static_cast<uint32_t>(o.d) >> 2u, 0, o.b};
        loadToEax(o.a);
        break;
    case K::CmplwBc:
        c = {Compare::Reg, false, static_cast<uint32_t>(o.d) >> 2u, 0, o.b};
        loadToEax(o.a);
        break;
    default:  // RlwinmRcBc: r[a] = rotl(r[d], b) & mask; CR0 from it
        c = {Compare::Test, true, 0, 0, 0};
        loadToEax(o.d);
        if (o.b != 0) a.rol(eax, static_cast<uint8_t>(o.b & 31u));
        a.and_(eax, o.imm);
        if (e.GprReadLater(o.a)) {
            a.mov(e.WriteGpr(o.a), eax);
        } else {
            e.DropGpr(o.a);
            a.mov(e.Gpr(o.a), eax);
        }
        break;
    }
    const Op& bc = e.block.ops[index + 1];
    // No later op reads the field before it is overwritten: the compare's flags decide the branch and the field is written on
    // the taken path only. A bc that tests the field's SO bit reads it from the field in memory (the flags have no SO), so that case writes it first.
    const bool readsOwnSo = (bc.d & 0x10u) == 0 && (bc.a >> 2) == c.field && (bc.a & 3u) == 3u;
    const bool dead = e.CrDead(index, c.field) && !readsOwnSo;
    if (dead) {
        BlockEmitter::NoteCr(BlockEmitter::CrEvent::FusedFlagsOnly);
    } else if (FusedOnFlagsEnabled() && (bc.d & 0x10u) == 0 && (bc.a >> 2) == c.field && (bc.a & 3u) != 3u) {
        // The field is live and the bc tests one of its lt, gt, eq bits: branch on the flags of the compare, and store the field
        // on each path from that same difference (the cold taken path and the code after the branch).
        EmitBcLive(e, bc, c);
        return 2;
    } else {
        EmitCompareValue(e, c, false);
        e.CrStore(c.field);
    }
    EmitBc(e, bc, dead ? &c : nullptr, dead);
    return 2;
}

// The frame returned (lr is the address the frame was entered with): f1 becomes the scalar the caller takes (inline when
// it is not in paired form), then the exit. Else a jump through lr, through the inline cache.
void EmitReturnOrJump(BlockEmitter& e) {
    CodeGenerator& a = e.a;
    Xbyak::Label notReturn;
    Xbyak::Label& paired = e.NewLabel();
    a.mov(eax, e.Lr());
    a.and_(eax, ~3u);
    a.mov(ecx, a.dword[kFrame + static_cast<int>(offsetof(Frame, returnAddress))]);
    a.and_(ecx, ~3u);
    a.cmp(eax, ecx);
    a.jne(notReturn, CodeGenerator::T_SHORT);
    a.test(e.PairedMask(), 2u);  // bit 1: f1 is Paired
    a.jnz(paired);
    e.ExitReturn();
    e.Cold([&paired](BlockEmitter& c) {
        c.a.L(paired);
        c.CallHelper(&JitNormalizeF1, kFrame);
        c.ExitReturn();
    });
    a.L(notReturn);
    e.ExitDynamicCached(0);
}

uint32_t EmitBclr(BlockEmitter& e, uint32_t index) {
    CodeGenerator& a = e.a;
    const Op& o = e.block.ops[index];
    const uint32_t bo = o.d, bi = o.a;
    const auto T = CodeGenerator::T_NEAR;  // the return and the cache probe lie between
    Xbyak::Label skip;
    if ((bo & 0x04u) == 0) {
        a.sub(e.Ctr(), 1);
        if ((bo & 0x02u) != 0) a.jnz(skip, T);
        else a.jz(skip, T);
    }
    if ((bo & 0x10u) == 0) EmitJumpIfCrBit(e, bi, (bo & 0x08u) == 0, skip, T);
    EmitReturnOrJump(e);
    a.L(skip);
    return 1;
}

// After JitCallOp / JitTailOp: rax = what << 32 | value. Sets rcx = what and the flags for "what != 0".
void EmitShimResult(BlockEmitter& e) {
    e.a.mov(rcx, rax);
    e.a.shr(rcx, 32);
}

// The answer of JitCallOp other than "the call was made" (rcx = what, rax = what << 32 | the target): pending, or a call
// to interpret in this frame: leave for the target (Bl: linked; Bctrl: through the inline cache, eax is the target).
void EmitNotMade(BlockEmitter& e, bool isBl, uint32_t target) {
    Xbyak::Label pending;
    e.a.cmp(ecx, kReturnPending);
    e.a.je(pending);
    if (isBl) e.ExitLinked(target);
    else e.ExitDynamicCached(0);
    e.a.L(pending);
    e.ExitPending();
}

// The call through JitCallOp. Made: the block goes on after its validity test. Not made: `notMade` (cold).
void EmitCallThroughShim(BlockEmitter& e, uint32_t index, bool isBl, uint32_t target) {
    Xbyak::Label& notMade = e.NewLabel();
    const uint32_t pc = e.PcOf(index);
    e.CallHelper(&JitCallOp, kFrame, &e.block, index);
    e.a.mov(rcx, rax);
    e.a.shr(rcx, 32);
    e.a.jnz(notMade, CodeGenerator::T_NEAR);
    e.EmitRecheck(pc + 4);
    e.Cold([&notMade, isBl, target](BlockEmitter& c) {
        c.a.L(notMade);
        EmitNotMade(c, isBl, target);
    });
}

// Jumps to `loop` when the frame is the rollback loop frame (BrawlRollback::LoopCall decides its calls).
void EmitLoopFrameTest(BlockEmitter& e, Xbyak::Label& loop) {
    e.a.test(e.FrameFlags(), kFrameFlagLoopFrame);
    e.a.jnz(loop, CodeGenerator::T_NEAR);
}

// Tests whether any float argument register named in `t.argMask` is currently held in paired form; if so, calls
// JitNormalizeArgs out of line and rejoins. Always leaves all of t's float arguments in scalar form.
void EmitStaticArgTest(BlockEmitter& e, const JitCallTarget& t) {
    if (t.argMask == 0) return;
    Xbyak::Label& convert = e.NewLabel();
    Xbyak::Label& done = e.NewLabel();
    e.a.test(e.PairedMask(), t.argMask);
    e.a.jnz(convert, CodeGenerator::T_NEAR);
    e.a.L(done);
    const JitCallTarget* const tp = &t;
    e.Cold([&convert, &done, tp](BlockEmitter& c) {
        c.a.L(convert);
        c.CallHelper(&JitNormalizeArgs, kFrame, tp);
        c.a.jmp(done, CodeGenerator::T_NEAR);
    });
}

// Finishes a direct tail call whose saved return address (lr & ~3) sits in ebp: clears f1's paired bit, restores lr,
// and either returns from the frame or continues at ebp through the inline cache.
void EmitDirectTailFinish(BlockEmitter& e) {
    CodeGenerator& a = e.a;
    Xbyak::Label returned;
    a.and_(e.PairedMask(), ~2u);
    a.mov(e.Lr(), ebp);
    a.mov(ecx, a.dword[kFrame + static_cast<int>(offsetof(Frame, returnAddress))]);
    a.and_(ecx, ~3u);
    a.cmp(ebp, ecx);
    a.je(returned, CodeGenerator::T_SHORT);
    a.mov(eax, ebp);
    e.ExitDynamicCached(0);
    a.L(returned);
    e.ExitReturn();
}

// A call to a translated function with a raw dispatch record (`t`), as thin as Call makes it. The caller has stored nothing yet.
// In the rollback loop frame the call goes through JitCallOp.
void EmitThinBl(BlockEmitter& e, uint32_t index, const JitCallTarget& t) {
    CodeGenerator& a = e.a;
    const uint32_t pc = e.PcOf(index);
    const uint32_t target = t.target;
    Xbyak::Label& loopFrame = e.NewLabel();
    Xbyak::Label& resume = e.NewLabel();
    a.mov(e.Pc(), pc);
    a.mov(e.Lr(), pc + 4);
    EmitLoopFrameTest(e, loopFrame);
    EmitStaticArgTest(e, t);
    if (kDirectCalls && (t.fastDirect || (t.fprsOnly && !t.returnsPaired && !t.special))) {
        if (t.fastDirect) {
            if (t.patchedByte != nullptr && *t.patchedByte != 0) {
                a.mov(eax, target);
                a.mov(rdx, reinterpret_cast<uint64_t>(&t));
                e.CallSubCall();
            } else {
                a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
                e.CallRaw(reinterpret_cast<const void*>(t.entry));
            }
        } else {
            a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
            a.mov(ArgReg(1), reinterpret_cast<uint64_t>(&t));
            e.CallRaw(reinterpret_cast<const void*>(&JitDirectFprsCall));
        }
        a.and_(e.PairedMask(), ~2u);
        a.L(resume);
        e.EmitRecheck(pc + 4);
        e.Cold([&loopFrame, &resume, index, target](BlockEmitter& c) {
            c.a.L(loopFrame);
            c.CallHelper(&JitCallOp, kFrame, &c.block, index);
            c.a.mov(rcx, rax);
            c.a.shr(rcx, 32);
            c.a.jz(resume, CodeGenerator::T_NEAR);
            EmitNotMade(c, true, target);
        });
        return;
    }
    Xbyak::Label& pending = e.NewLabel();
    e.CallHelper(&JitThinCallPrepared, kFrame, &t);
    a.test(rax, rax);
    a.jnz(pending, CodeGenerator::T_NEAR);
    a.L(resume);
    e.EmitRecheck(pc + 4);
    e.Cold([&pending, &loopFrame, &resume, index, target](BlockEmitter& c) {
        c.a.L(pending);
        c.ExitPending();
        c.a.L(loopFrame);
        c.CallHelper(&JitCallOp, kFrame, &c.block, index);
        c.a.mov(rcx, rax);
        c.a.shr(rcx, 32);
        c.a.jz(resume, CodeGenerator::T_NEAR);
        EmitNotMade(c, true, target);
    });
}

// The call op's target, when it is a translated function the thin shims can call: a Bl resolved at decode, translated, with a
// raw dispatch record and nothing Call treats specially.
const JitCallTarget* ThinCallTarget(const BlockEmitter& e, const Op& o) {
    if (o.b == 0 || e.block.calls[o.b - 1].info == nullptr) return nullptr;
    const JitCallTarget* const t = JitCallTargetFor(o.imm);
    return t != nullptr && !t->special ? t : nullptr;
}

uint32_t EmitBl(BlockEmitter& e, uint32_t index) {
    CodeGenerator& a = e.a;
    const Op& o = e.block.ops[index];
    const uint32_t pc = e.PcOf(index);
    const uint32_t target = o.imm;
    if (const JitCallTarget* const t = ThinCallTarget(e, o)) {
        EmitThinBl(e, index, *t);
        return 1;
    }
    // A site resolved at decode with an untranslated, runnable target: Call sets lr, asks whether this is the rollback loop
    // frame, and answers "interpret it here". Everything else (translated, native, patched targets; unknown sites) is Call.
    const bool leaf = o.b != 0 && e.block.calls[o.b - 1].info == nullptr && PpcInterpreter::CanRun(target);
    if (!leaf) {
        EmitCallThroughShim(e, index, true, target);
        return 1;
    }
    Xbyak::Label& loopFrame = e.NewLabel();
    Xbyak::Label& cont = e.NewLabel();
    a.mov(e.Pc(), pc);
    a.mov(e.Lr(), pc + 4);
    EmitLoopFrameTest(e, loopFrame);
    e.ExitLinked(target);
    // Reached only from the loop-frame path below, when BrawlRollback::LoopCall skipped the call
    a.L(cont);
    e.EmitRecheck(pc + 4);
    e.Cold([&loopFrame, &cont, index, target](BlockEmitter& c) {
        c.a.L(loopFrame);
        c.CallHelper(&JitCallOp, kFrame, &c.block, index);
        c.a.mov(rcx, rax);
        c.a.shr(rcx, 32);
        c.a.jz(cont, CodeGenerator::T_NEAR);
        EmitNotMade(c, true, target);
    });
    return 1;
}

uint32_t EmitBctrl(BlockEmitter& e, uint32_t index) {
    // The registry is fixed once published: before that, nothing may be remembered.
    if (!TranslatedFunctionRegistry::IsLookupPublished()) {
        EmitCallThroughShim(e, index, false, 0);
        return 1;
    }
    CodeGenerator& a = e.a;
    const uint32_t pc = e.PcOf(index);
    Xbyak::Label& cell = e.NewCell();
    Xbyak::Label& generic = e.NewLabel();
    Xbyak::Label& resume = e.NewLabel();
    a.mov(e.Pc(), pc);
    a.mov(e.Lr(), pc + 4);
    EmitLoopFrameTest(e, generic);
    // The cell holds the last translated targets this site called and their records; the key is the ctr it was found for.
    a.mov(eax, e.Ctr());
    a.and_(eax, ~3u);
    e.CellProbe(cell, generic);
    if (kDirectCalls) {
        Xbyak::Label& guarded = e.NewLabel();
        Xbyak::Label& norm = e.NewLabel();
        Xbyak::Label& afterNorm = e.NewLabel();
        Xbyak::Label& subCallPath = e.NewLabel();
        Xbyak::Label& afterCall = e.NewLabel();
        Xbyak::Label aotCall;
        a.cmp(a.byte[rdx + static_cast<int>(offsetof(JitCallTarget, fastDirect))], 0);
        a.je(guarded, CodeGenerator::T_NEAR);
        a.mov(ecx, a.dword[rdx + static_cast<int>(offsetof(JitCallTarget, argMask))]);
        a.test(e.PairedMask(), ecx);
        a.jnz(norm, CodeGenerator::T_NEAR);
        a.L(afterNorm);
        a.mov(rcx, a.qword[rdx + static_cast<int>(offsetof(JitCallTarget, patchedByte))]);
        a.test(rcx, rcx);
        a.jz(aotCall, CodeGenerator::T_SHORT);
        a.cmp(a.byte[rcx], 0);
        a.jne(subCallPath, CodeGenerator::T_NEAR);
        a.L(aotCall);
        a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
        a.call(a.qword[rdx + static_cast<int>(offsetof(JitCallTarget, entry))]);
        a.L(afterCall);
        a.and_(e.PairedMask(), ~2u);
        a.L(resume);
        e.EmitRecheck(pc + 4);
        e.Cold([&subCallPath, &afterCall](BlockEmitter& c) {
            c.a.L(subCallPath);
            c.a.mov(eax, c.a.dword[rdx + static_cast<int>(offsetof(JitCallTarget, target))]);
            c.CallSubCall();
            c.a.jmp(afterCall, CodeGenerator::T_NEAR);
        });
        e.Cold([&norm, &afterNorm, &guarded, &generic, &resume, &cell, index](BlockEmitter& c) {
            c.a.L(norm);
            c.a.mov(rbp, rdx);
            c.CallHelper(&JitNormalizeArgs, kFrame, rdx);
            c.a.mov(rdx, rbp);
            c.a.jmp(afterNorm, CodeGenerator::T_NEAR);
            c.a.L(guarded);
            c.CallHelper(&JitThinCall, kFrame, rdx);
            c.a.test(rax, rax);
            c.a.jz(resume, CodeGenerator::T_NEAR);
            c.ExitPending();
            c.a.L(generic);
            c.a.lea(r10, c.a.ptr[c.a.rip + cell]);
            c.CallHelper(&JitCallOpCached, kFrame, &c.block, index, r10);
            c.a.mov(rcx, rax);
            c.a.shr(rcx, 32);
            c.a.jz(resume, CodeGenerator::T_NEAR);
            EmitNotMade(c, false, 0);
        });
        return 1;
    }
    Xbyak::Label& pending = e.NewLabel();
    e.CallHelper(&JitThinCall, kFrame, rdx);
    a.test(rax, rax);
    a.jnz(pending, CodeGenerator::T_NEAR);
    a.L(resume);
    e.EmitRecheck(pc + 4);
    e.Cold([&pending, &generic, &resume, &cell, index](BlockEmitter& c) {
        c.a.L(pending);
        c.ExitPending();
        c.a.L(generic);
        c.a.lea(r10, c.a.ptr[c.a.rip + cell]);
        c.CallHelper(&JitCallOpCached, kFrame, &c.block, index, r10);
        c.a.mov(rcx, rax);
        c.a.shr(rcx, 32);
        c.a.jz(resume, CodeGenerator::T_NEAR);
        EmitNotMade(c, false, 0);
    });
    return 1;
}

// After a tail shim (rax = what << 32 | value): 2 the frame returned, 3 pending, else go on at the value (an inline cache, then
// the dispatcher).
void EmitTailResult(BlockEmitter& e) {
    Xbyak::Label pending, returned;
    EmitShimResult(e);
    e.a.cmp(ecx, kReturnFrame);
    e.a.je(returned);
    e.a.cmp(ecx, kReturnPending);
    e.a.je(pending);
    e.ExitDynamicCached(0);  // goes on at lr (the tail caller was inlined code of this frame): a block of its own, probed first
    e.a.L(returned);
    e.ExitReturn();
    e.a.L(pending);
    e.ExitPending();
}

uint32_t EmitB(BlockEmitter& e, uint32_t index) {
    CodeGenerator& a = e.a;
    const Op& o = e.block.ops[index];
    // RunOp asks GuestCodePatch::ContinuationAt on every execution. Never non-null in a process without code variants
    // (or with them switched off); an answer that is non-null now is for the fallback (a stub answers by address alone).
    if (JitContinuationsPossible() || JitContinuationAt(o.imm)) return e.EmitFallback(index);
    bool translated;
    if (o.b != 0) translated = o.b == 1;
    else if (TranslatedFunctionRegistry::IsLookupPublished()) translated = TranslatedFunctionRegistry::FindByAddressPtr(o.imm) != nullptr;
    else return e.EmitFallback(index);
    if (!translated) {
        e.ExitLinked(o.imm);
        return 1;
    }
    // A tail call: the callee runs, then the frame returns, or goes on at lr (JitTailOp). When the callee has a raw dispatch
    // record, call directly when fastDirect, else JitThinTailPrepared; the loop frame goes the general way.
    const JitCallTarget* const t = JitCallTargetFor(o.imm);
    if (t != nullptr) {
        Xbyak::Label& loopFrame = e.NewLabel();
        a.mov(e.Pc(), e.PcOf(index));
        EmitLoopFrameTest(e, loopFrame);
        EmitStaticArgTest(e, *t);
        if (kDirectCalls && t->fastDirect) {
            a.mov(ebp, e.Lr());
            a.and_(ebp, ~3u);
            a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
            e.CallRaw(reinterpret_cast<const void*>(t->entry));
            EmitDirectTailFinish(e);
        } else {
            e.CallHelper(&JitThinTailPrepared, kFrame, t);
            EmitTailResult(e);
        }
        e.Cold([&loopFrame, index](BlockEmitter& c) {
            c.a.L(loopFrame);
            c.CallHelper(&JitTailOp, kFrame, &c.block, index);
            EmitTailResult(c);
        });
        return 1;
    }
    e.CallHelper(&JitTailOp, kFrame, &e.block, index);
    EmitTailResult(e);
    return 1;
}

uint32_t EmitBctr(BlockEmitter& e, uint32_t index) {
    CodeGenerator& a = e.a;
    // A cached target is one the registry did not know when the dispatcher linked it, and the registry is fixed once
    // published: before that, nothing may be cached.
    if (!TranslatedFunctionRegistry::IsLookupPublished()) return e.EmitFallback(index);
    Xbyak::Label pending, returned, plain, generic;
    const uint32_t pc = e.PcOf(index);
    Xbyak::Label& cell = e.NewCell();
    a.mov(eax, e.Ctr());
    a.and_(eax, ~3u);
    e.CacheProbe(1);
    // The cell holds the last translated targets this site jumped to and their records; eax is the target.
    e.CellProbe(cell, generic);
    a.mov(e.Pc(), pc);
    EmitLoopFrameTest(e, generic);
    if (kDirectCalls) {
        Xbyak::Label& guarded = e.NewLabel();
        Xbyak::Label& norm = e.NewLabel();
        Xbyak::Label& afterNorm = e.NewLabel();
        a.cmp(a.byte[rdx + static_cast<int>(offsetof(JitCallTarget, fastDirect))], 0);
        a.je(guarded, CodeGenerator::T_NEAR);
        a.mov(ecx, a.dword[rdx + static_cast<int>(offsetof(JitCallTarget, argMask))]);
        a.test(e.PairedMask(), ecx);
        a.jnz(norm, CodeGenerator::T_NEAR);
        a.L(afterNorm);
        a.mov(ebp, e.Lr());
        a.and_(ebp, ~3u);
        a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
        a.call(a.qword[rdx + static_cast<int>(offsetof(JitCallTarget, entry))]);
        EmitDirectTailFinish(e);
        e.Cold([&norm, &afterNorm, &guarded](BlockEmitter& c) {
            c.a.L(norm);
            c.a.mov(rbp, rdx);
            c.CallHelper(&JitNormalizeArgs, kFrame, rdx);
            c.a.mov(rdx, rbp);
            c.a.jmp(afterNorm, CodeGenerator::T_NEAR);
            c.a.L(guarded);
            c.CallHelper(&JitThinTail, kFrame, rdx);
            EmitTailResult(c);
        });
    } else {
        e.CallHelper(&JitThinTail, kFrame, rdx);
        EmitTailResult(e);
    }
    a.L(generic);
    a.lea(r10, a.ptr[a.rip + cell]);
    e.CallHelper(&JitTailOpCached, kFrame, &e.block, index, r10);
    EmitShimResult(e);
    a.cmp(ecx, kJumpHere);
    a.je(plain);
    a.cmp(ecx, kReturnFrame);
    a.je(returned);
    a.cmp(ecx, kReturnPending);
    a.je(pending);
    e.ExitDynamic();
    a.L(returned);
    e.ExitReturn();
    a.L(pending);
    e.ExitPending();
    a.L(plain);
    e.ExitDynamicMiss(1);
    return 1;
}

}  // namespace

uint32_t EmitBranch(BlockEmitter& e, uint32_t index) {
    const Op& o = e.block.ops[index];
    switch (o.kind) {
    case K::Bc:
        EmitBc(e, o, nullptr, false);
        return 1;
    case K::CmpwiBc: case K::CmplwiBc: case K::CmpwBc: case K::CmplwBc: case K::RlwinmRcBc:
        return EmitFusedCompare(e, index);
    case K::B: return EmitB(e, index);
    case K::Bl: return EmitBl(e, index);
    case K::Blr:
        EmitReturnOrJump(e);
        return 1;
    case K::Bclr: return EmitBclr(e, index);
    case K::Bctr: return EmitBctr(e, index);
    case K::Bctrl: return EmitBctrl(e, index);
    default: return 0;
    }
}

}  // namespace PpcInterpreter::detail::x64
