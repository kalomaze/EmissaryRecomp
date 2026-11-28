// Native code for the branches and calls (Bc, fused compare+bc, B ... Bctrl) of the x86-64 backend, written against the emitter
// API in jit_x64.h. Each kind does what RunOp does for it (and Exec where RunOp defers to it), with the same side effects in
// the same order; what the decode or the compile can settle is settled here instead of at every execution.
//
//  Bc and the fused compare+bc kinds (and a compare or record form whose bc follows past followed b's, EmitCompareBranch):
//      native. BO is decoded here: only the ctr decrement and the tests it needs are emitted; taken is a direct conditional link
//      (ExitLinkedIf), every dirty GPR slot and the compare's field stored before it (EmitCondBranch). A bc tests one CR field's
//      value in memory as isa/ppc_isa_crv.h defines each bit (CrBitTest). A pair branches on the host flags of the compare's
//      field value when the bc tests lt, gt or eq of the field just written (else it tests the CR bit in memory).
//  B:  native when its target is no translated function and no code variant can continue there (ExitLinked); a tail call
//      to a translated function is JitThinTail (JitTailOp in the rollback loop frame); with code variants possible, or the
//      registry not yet published, the fallback (RunOp asks ContinuationAt on every execution, which a block cannot settle).
//  Bl: native when the site was resolved at decode, the target is untranslated and runnable: lr and the loop-frame test
//      inline, then ExitLinked. A translated target with a raw dispatch record: lr, pc and the loop-frame test inline (the
//      loop frame goes the general way), the callee's float arguments tested in line, then a direct call of the entry when it
//      needs no guard, else JitThinCall with the target's JitCallTarget (everything Call and TryDispatchRawCpuTarget look up,
//      settled at compile time), and a validity test. A patched target (guard byte set at compile time) is called in a subcall
//      frame linked to its block while it stays patched (EmitSubFrameBl). Otherwise JitCallOp (Call) and a validity test when the
//      call was made.
//  Blr, Bclr: native; the frame's return normalises f1, anything else leaves through an inline cache.
//  Bctr: the newest target of its cell, an inline cache probe (untranslated targets only ever get in it), the cell's other targets
//      (translated targets: each with what JitFastCallFor says to call for it, as a tail call), a miss JitTailOpCached, which fills
//      the cell; the fallback until the registry is published.
//  Bctrl: a cell of its own, the same way: a hit calls what the entry says, a miss is JitCallOpCached, which fills the cell; the
//      interpret-here answer leaves through an inline cache.
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

// What a fused compare compares: r[a] against an immediate, against r[b], or the result a record form wrote to r[a] against 0.
struct Compare {
    enum Source : uint8_t { Imm, Reg, Test } source;
    bool isSigned;
    uint32_t field;
    int32_t imm;
    uint32_t a, b;
    bool nonNegative = false;  // Test: the result is known not negative (its field value is the zero-extended result)
};

// Calls f with guest register r as an operand: its cache register (loaded into the cache first when a later op reads it), else its
// word in memory. In cold code (no cache) always the word, which the code there has flushed.
template <typename F>
void WithGpr(BlockEmitter& e, uint32_t r, F&& f) {
    if (e.WantCacheGpr(r)) f(e.ReadGpr(r));
    else f(e.Gpr(r));
}

// rax = the compare's field value as PpcCrvCompare makes it (no SO: BlockEmitter::JumpIfSo). With `flags` the host flags are those of
// the value against 0, whose signed conditions (l, g, e) are the compare's for both signednesses: the extended operands differ by
// less than 2^33 in magnitude. `result` (Test): the host register holding the record form's result, or null to read r[a].
void EmitCompareValue(BlockEmitter& e, const Compare& c, bool flags, const Xbyak::Reg32* result) {
    CodeGenerator& a = e.a;
    const auto load = [&](const Xbyak::Operand& op) {
        if (c.isSigned) a.movsxd(rax, op);
        else a.mov(eax, op);
    };
    switch (c.source) {
    case Compare::Imm:
        WithGpr(e, c.a, load);
        e.CrDiffImm(c.isSigned, static_cast<uint32_t>(c.imm), flags);
        break;
    case Compare::Reg:
        WithGpr(e, c.a, load);
        e.CrDiffGpr(c.isSigned, c.b);
        break;
    case Compare::Test:
        if (result != nullptr) a.movsxd(rax, *result);
        else WithGpr(e, c.a, [&](const Xbyak::Operand& op) { a.movsxd(rax, op); });
        if (flags) a.test(rax, rax);
        break;
    }
}

// BRAWL_JIT_NO_FUSED_FLAGS=1 (a test switch): a fused pair whose field is live writes the field first and then branches on the
// field in memory, instead of branching on the host flags.
bool FusedOnFlagsEnabled() {
    static const bool on = [] { const char* v = std::getenv("BRAWL_JIT_NO_FUSED_FLAGS"); return v == nullptr || *v == '\0' || *v == '0'; }();
    return on;
}

// The condition "CR bit `bit` (0 lt, 1 gt, 2 eq) of the compare is `want`" on the host flags of a compare of that signedness.
Cc CompareCc(bool isSigned, uint32_t bit, bool want) {
    const Cc c = bit == 0 ? (isSigned ? Cc::L : Cc::B) : bit == 1 ? (isSigned ? Cc::G : Cc::A) : Cc::E;
    return want ? c : Invert(c);
}

// Sets the host flags for CR bit `bi` (as BI numbers it: field bi >> 2, then 0 lt, 1 gt, 2 eq, 3 so) of the field in memory and
// answers the condition for "the bit is `want`", by the field value's own tests (isa/ppc_isa_crv.h): lt is bit 62, gt is positive,
// eq is a zero low word, so is bit 59 differing from bit 62. Clobbers rcx and rdx for so; eax is kept.
Cc CrBitTest(BlockEmitter& e, uint32_t bi, bool want) {
    CodeGenerator& a = e.a;
    const Xbyak::Address field = e.Crv(bi >> 2);
    switch (bi & 3u) {
    case 0:
        a.test(a.byte[field.getRegExp() + 7], 0x40);  // bit 62: bit 6 of the top byte
        return want ? Cc::NE : Cc::E;
    case 1:
        a.cmp(field, 0);
        return want ? Cc::G : Cc::LE;
    case 2:
        a.cmp(a.dword[field.getRegExp()], 0);
        return want ? Cc::E : Cc::NE;
    default:
        a.mov(rcx, field);
        a.mov(rdx, rcx);
        a.shr(rdx, 3);  // bit 59 of rdx is bit 62 of the field
        a.xor_(rcx, rdx);
        a.bt(rcx, 59);
        return want ? Cc::B : Cc::AE;
    }
}

// Jumps to `target` when CR bit `bi` of the field in memory is `want` (CrBitTest).
void EmitJumpIfCrBit(BlockEmitter& e, uint32_t bi, bool want, Xbyak::Label& target, CodeGenerator::LabelType type) {
    Jcc(e.a, CrBitTest(e, bi, want), target, type);
}

// A bc, op `bci` (for a fused pair `cmp` is its compare, an op before it, whose code comes first): BranchTaken's ctr decrement and tests,
// the exit to the target when taken; falls through when not. `result`, `resultFlags`: a record form's result register (Test), and
// whether the host flags are already those of the result against 0.
//
// The taken path leaves the block, so on it every dirty GPR slot is stored and the compare's field is written (the block leaves with
// the condition register visible). Both happen before the branch, on both paths, so that the taken path is nothing but the jump: a
// direct conditional link to the target block (ExitLinkedIf). A dirty slot would be stored once anyway, at the block's next flush,
// unless a later op of the block rewrites it first; and a dead field (CrDead) costs its value and one store on the fall-through
// path. Measured against keeping that work out of line on the taken path (a jump to cold code, then the jump to the target), this
// is fewer instructions and taken branches for the game's branches, which are taken about half the time.
void EmitCondBranch(BlockEmitter& e, uint32_t bci, const Compare* cmp, const Xbyak::Reg32* result, bool resultFlags) {
    CodeGenerator& a = e.a;
    const Op& bc = e.block.ops[bci];
    const uint32_t bo = bc.d, bi = bc.a, target = bc.imm;
    const bool useCtr = (bo & 0x04u) == 0, useCond = (bo & 0x10u) == 0;
    const bool want = (bo & 0x08u) != 0;
    bool onFlags = false;
    // Only moves when a record form's flags are live here: the FPR cache's pending forms (an `or` / `and`) were stored before
    // the record form (EmitOps for the fused kinds, EmitCompareBranch for the pairs).
    e.FlushDirtyGprs();
    Xbyak::Reg64 value = rax;  // what holds the field value
    if (cmp != nullptr) {
        const bool readsOwn = useCond && (bi >> 2) == cmp->field;
        onFlags = readsOwn && (bi & 3u) != 3u;  // a test of its SO bit reads the field in memory
        const bool dead = e.CrDead(bci, cmp->field) && onFlags == readsOwn;
        if (!dead && onFlags && !FusedOnFlagsEnabled()) onFlags = false;
        if (dead) BlockEmitter::NoteCr(BlockEmitter::CrEvent::FusedFlagsOnly);
        else if (onFlags) BlockEmitter::NoteCr(BlockEmitter::CrEvent::FusedLive);
        const bool flags = onFlags && !useCtr && !(cmp->source == Compare::Test && resultFlags);
        if (cmp->source == Compare::Test && cmp->nonNegative && result != nullptr) {
            value = result->cvt64();  // zero-extended: the field value itself
            if (flags) a.test(*result, *result);
        } else {
            EmitCompareValue(e, *cmp, flags, result);
        }
        e.CrStore(cmp->field, value);
    }
    if (!useCtr && !useCond) {  // always taken
        e.ExitLinked(target);
        return;
    }
    Xbyak::Label skip;
    Cc taken = Cc::NE;
    if (useCtr) {  // --ctr, always, even when the condition then fails; ZF is the new ctr == 0
        a.sub(e.Ctr(), 1);
        const Cc ctrOk = (bo & 0x02u) != 0 ? Cc::E : Cc::NE;
        if (useCond) Jcc(a, Invert(ctrOk), skip, CodeGenerator::T_SHORT);
        else taken = ctrOk;
    }
    if (useCond) {
        if (onFlags) {  // the flags of the field value (made again after the ctr test)
            if (useCtr) a.test(value, value);
            taken = CompareCc(true, bi & 3u, want);
        } else {
            taken = CrBitTest(e, bi, want);
        }
    }
    e.ExitLinkedIf(taken, target);
    a.L(skip);
}

// A record form's result (r[a] of RlwinmRc(Bc), AndiRc, AndisRc, AndRc, OrRc): written to r[a] and left in the answered host register
// (r[a]'s cache register when a later op reads it, else eax); `flags`: the host flags are those of the result against 0. Fills c
// (Test, cr0). Null for another kind (nothing emitted).
const Xbyak::Reg32* EmitRecordResult(BlockEmitter& e, const Op& o, Compare& c, bool& flags) {
    CodeGenerator& a = e.a;
    uint32_t dst = 0;
    flags = false;
    bool nonNegative = false;
    const Xbyak::Reg32* r = &eax;
    switch (o.kind) {
    case K::RlwinmRc:
    case K::RlwinmRcBc: {  // r[a] = rotl(r[d], b) & mask
        const uint32_t sh = o.b & 31u, mask = o.imm;
        dst = o.a;
        if (e.GprReadLater(o.a) && o.a == o.d) {
            r = &e.ModifyGpr(o.a);
            if (sh != 0) a.rol(*r, static_cast<uint8_t>(sh));
        } else {
            const auto rotate = [&](const Xbyak::Reg32& d) {
                WithGpr(e, o.d, [&](const Xbyak::Operand& op) {
                    if (sh != 0) a.rorx(d, op, static_cast<uint8_t>((32u - sh) & 31u));
                    else a.mov(d, op);
                });
            };
            if (e.GprReadLater(o.a)) {
                if (e.WantCacheGpr(o.d)) (void)e.ReadGpr(o.d);
                r = &e.WriteGpr(o.a, 1u << o.d);
            }
            rotate(*r);
        }
        if (mask != 0xFFFFFFFFu) {
            a.and_(*r, mask);
            flags = true;
        }
        nonNegative = (mask & 0x80000000u) == 0;
        break;
    }
    case K::AndiRc:
    case K::AndisRc:  // r[a] = r[d] & imm
        dst = o.a;
        if (e.GprReadLater(o.a)) {
            if (o.a == o.d) {
                r = &e.ModifyGpr(o.a);
            } else {
                if (e.WantCacheGpr(o.d)) (void)e.ReadGpr(o.d);
                r = &e.WriteGpr(o.a, 1u << o.d);
                WithGpr(e, o.d, [&](const Xbyak::Operand& op) { a.mov(*r, op); });
            }
        } else {
            WithGpr(e, o.d, [&](const Xbyak::Operand& op) { a.mov(eax, op); });
        }
        a.and_(*r, o.imm);
        flags = true;
        nonNegative = (o.imm & 0x80000000u) == 0;
        break;
    case K::AndRc:
    case K::OrRc: {  // r[a] = r[d] op r[b] (or. with d == b: mr.)
        dst = o.a;
        const bool isAnd = o.kind == K::AndRc;
        if (e.GprReadLater(o.a)) {
            if (e.WantCacheGpr(o.d)) (void)e.ReadGpr(o.d);
            if (e.WantCacheGpr(o.b)) (void)e.ReadGpr(o.b, 1u << o.d);
            r = (o.a == o.d || o.a == o.b) ? &e.ModifyGpr(o.a, (1u << o.d) | (1u << o.b)) : &e.WriteGpr(o.a, (1u << o.d) | (1u << o.b));
        }
        const uint32_t first = o.a == o.b ? o.b : o.d, second = o.a == o.b ? o.d : o.b;
        if (!(e.GprReadLater(o.a) && o.a == first)) WithGpr(e, first, [&](const Xbyak::Operand& op) { a.mov(*r, op); });
        if (second != first) {
            WithGpr(e, second, [&](const Xbyak::Operand& op) {
                if (isAnd) a.and_(*r, op);
                else a.or_(*r, op);
            });
            flags = true;
        }
        break;
    }
    default:
        return nullptr;
    }
    if (r == &eax) {
        e.DropGpr(dst);
        a.mov(e.Gpr(dst), eax);
    }
    c = Compare{Compare::Test, true, 0, 0, dst, 0, nonNegative};
    return r;
}

uint32_t EmitFusedCompare(BlockEmitter& e, uint32_t index) {
    const Op& o = e.block.ops[index];
    Compare c{};
    const Xbyak::Reg32* result = nullptr;
    bool resultFlags = false;
    switch (o.kind) {
    case K::CmpwiBc: c = {Compare::Imm, true, static_cast<uint32_t>(o.d) >> 2u, static_cast<int32_t>(o.imm), o.a, 0}; break;
    case K::CmplwiBc: c = {Compare::Imm, false, static_cast<uint32_t>(o.d) >> 2u, static_cast<int32_t>(o.imm), o.a, 0}; break;
    case K::CmpwBc: c = {Compare::Reg, true, static_cast<uint32_t>(o.d) >> 2u, 0, o.a, o.b}; break;
    case K::CmplwBc: c = {Compare::Reg, false, static_cast<uint32_t>(o.d) >> 2u, 0, o.a, o.b}; break;
    default: result = EmitRecordResult(e, o, c, resultFlags); break;  // RlwinmRcBc
    }
    EmitCondBranch(e, index + 1, &c, result, resultFlags);
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
    a.cmp(eax, a.dword[kFrame + static_cast<int>(offsetof(Frame, returnAddress))]);  // kept & ~3 (Frame::returnAddress)
    a.jne(notReturn, CodeGenerator::T_SHORT);
    a.test(a.byte[kFrame + static_cast<int>(offsetof(Frame, pairedMask))], 2u);  // bit 1: f1 is Paired
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
// Emitted in cold code only. Call ran BrawlRollback::LoopCall (the loop frame), which may have run C++ that set XER[SO]: then
// through the dispatcher, not a link (BlockEmitter::JumpIfSo; the link epoch is JitCallOp's JitNoteEpoch).
void EmitNotMade(BlockEmitter& e, bool isBl, uint32_t target) {
    Xbyak::Label pending, so;
    e.a.cmp(ecx, kReturnPending);
    e.a.je(pending);
    e.JumpIfSo(so);
    if (isBl) e.ExitLinked(target);
    else e.ExitDynamicCached(0);
    e.a.L(pending);
    e.ExitPending();
    e.a.L(so);
    if (isBl) e.ExitTo(target);
    else e.ExitDynamic();  // eax: the target
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
// The registers the form analysis knows to be scalar here need no test (JitNormalizeArgs leaves a scalar one as it is).
void EmitStaticArgTest(BlockEmitter& e, const JitCallTarget& t) {
    uint32_t mask = t.argMask;
    for (uint32_t m = mask; m != 0; m &= m - 1u) {
        const uint32_t r = static_cast<uint32_t>(__builtin_ctz(m));
        if (e.KnownForm(e.index, r) == Form::Scalar) mask &= ~(1u << r);
    }
    if (mask == 0) return;
    Xbyak::Label& convert = e.NewLabel();
    Xbyak::Label& done = e.NewLabel();
    e.a.test(e.PairedMask(), mask);
    e.a.jnz(convert, CodeGenerator::T_NEAR);
    e.a.L(done);
    const JitCallTarget* const tp = &t;
    e.Cold([&convert, &done, tp](BlockEmitter& c) {
        c.a.L(convert);
        c.CallHelper(&JitNormalizeArgs, kFrame, tp);
        c.a.jmp(done, CodeGenerator::T_NEAR);
    });
}

// NormalizeCallFloatReturn after the call op e.index made: f1's paired bit cleared, unless the form analysis knows f1 is scalar
// before the call (a callee leaves f1 scalar, every way a call op calls one: the bit is clear already).
void EmitF1Scalar(BlockEmitter& e) {
    if (e.KnownForm(e.index, 1) == Form::Scalar) return;
    e.a.and_(e.PairedMask(), ~2u);
}

// Finishes a direct tail call whose saved return address (lr & ~3) sits in ebp: clears f1's paired bit, restores lr,
// and either returns from the frame or continues at ebp through the inline cache.
void EmitDirectTailFinish(BlockEmitter& e) {
    CodeGenerator& a = e.a;
    Xbyak::Label returned;
    EmitF1Scalar(e);
    a.mov(e.Lr(), ebp);
    a.cmp(ebp, a.dword[kFrame + static_cast<int>(offsetof(Frame, returnAddress))]);  // kept & ~3 (Frame::returnAddress)
    a.je(returned, CodeGenerator::T_SHORT);
    a.mov(eax, ebp);
    e.ExitDynamicAfterCall(0);  // through the dispatcher when the callee set XER[SO] or moved the epoch
    a.L(returned);
    e.ExitReturn();
}

// A bl to a translated function `t` that is patched (its guard byte set when the block is compiled), the common JIT->JIT call:
// while the guard byte stays set and no redirect applies, the callee's block is called in a subcall frame
// (BlockEmitter::CallSubFrame), which is what calling the wrapper does then (RunPatched, RunCompiledEntry, jit_enter_direct)
// without the C++ in between. Otherwise the wrapper, or JitCallOp in the rollback loop frame (Call decides its calls, before the
// float arguments are touched).
void EmitSubFrameBl(BlockEmitter& e, uint32_t index, const JitCallTarget& t) {
    CodeGenerator& a = e.a;
    const uint32_t pc = e.PcOf(index);
    const uint32_t target = t.target;
    const auto T = CodeGenerator::T_NEAR;
    Xbyak::Label& loopFrame = e.NewLabel();
    Xbyak::Label& wrapper = e.NewLabel();  // the wrapper, float arguments done
    Xbyak::Label& resume = e.NewLabel();
    Xbyak::Label& made = e.NewLabel();
    a.mov(e.Lr(), pc + 4);
    a.test(a.byte[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagLoopFrame);
    a.jnz(loopFrame, T);
    EmitStaticArgTest(e, t);
    a.cmp(AbsMem(a, t.patchedByte, rcx, 8), 0);
    a.je(wrapper, T);
    a.cmp(AbsMem(a, GetPatchedHooks().lowestRedirect, rcx, 32), target);
    a.jbe(wrapper, T);  // lowestRedirect <= target: RunPatched resolves it
    e.CallSubFrame(target, pc + 4);
    a.L(resume);
    EmitF1Scalar(e);  // NormalizeCallFloatReturn (the callee's frame return made f1 scalar in its own frame)
    a.L(made);
    e.EmitRecheck(pc + 4);
    const JitCallTarget* const tp = &t;
    e.Cold([&loopFrame, &wrapper, &resume, &made, index, target, pc, tp](BlockEmitter& c) {
        CodeGenerator& g = c.a;
        g.L(wrapper);
        g.mov(c.Pc(), pc);
        g.lea(ArgReg(0), g.ptr[kCtxBase - kCtxBias]);
        c.CallRaw(reinterpret_cast<const void*>(tp->entry));
        g.jmp(resume, CodeGenerator::T_NEAR);
        g.L(loopFrame);
        c.CallHelper(&JitCallOp, kFrame, &c.block, index);
        g.mov(rcx, rax);
        g.shr(rcx, 32);
        g.jz(made, CodeGenerator::T_NEAR);
        EmitNotMade(c, true, target);
    });
}

// A call to a translated function with a raw dispatch record (`t`), as thin as Call makes it. The caller has stored nothing yet.
// In the rollback loop frame the call goes through JitCallOp.
void EmitThinBl(BlockEmitter& e, uint32_t index, const JitCallTarget& t) {
    CodeGenerator& a = e.a;
    const uint32_t pc = e.PcOf(index);
    const uint32_t target = t.target;
    if (kDirectCalls && t.fastDirect && t.patchedByte != nullptr && *t.patchedByte != 0 && SubFramesPossible()) {
        EmitSubFrameBl(e, index, t);
        return;
    }
    Xbyak::Label& loopFrame = e.NewLabel();
    Xbyak::Label& resume = e.NewLabel();
    a.mov(e.Pc(), pc);
    a.mov(e.Lr(), pc + 4);
    EmitLoopFrameTest(e, loopFrame);
    EmitStaticArgTest(e, t);
    if (kDirectCalls && (t.fastDirect || (t.fprsOnly && !t.returnsPaired && !t.special))) {
        if (t.fastDirect) {
            a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
            e.CallRaw(reinterpret_cast<const void*>(t.entry));
        } else {
            a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
            a.mov(ArgReg(1), reinterpret_cast<uint64_t>(&t));
            e.CallRaw(reinterpret_cast<const void*>(&JitDirectFprsCall));
        }
        EmitF1Scalar(e);
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

// The ways of a cell after the first (the site tests the first itself): jumps to `hit` with rdx = the matching entry, else to `miss`.
// eax is the target. An entry hit 8 times while not the first becomes the first (CellPromoteRoutine): the cell keeps the targets in
// the order the misses put them in (the newest first), and a site's common target moves up to the one test the site makes first.
// Clobbers rcx, xmm0-xmm3.
void CellWays(BlockEmitter& e, Xbyak::Label& cell, Xbyak::Label& hit, Xbyak::Label& miss) {
    CodeGenerator& a = e.a;
    Xbyak::Label wayHit;
    a.lea(rcx, a.ptr[a.rip + cell]);
    for (uint32_t i = 1; i < kCellWays; ++i) {
        const int at = static_cast<int>(i * sizeof(JitTargetCell));
        a.cmp(eax, a.dword[rcx + at + static_cast<int>(offsetof(JitTargetCell, target))]);
        a.lea(rdx, a.ptr[rcx + at]);  // lea leaves the flags
        a.je(wayHit, CodeGenerator::T_NEAR);
    }
    a.jmp(miss, CodeGenerator::T_NEAR);
    a.L(wayHit);
    a.add(a.byte[rdx + static_cast<int>(offsetof(JitTargetCell, hits))], 32);  // carries on the 8th
    a.jnc(hit, CodeGenerator::T_NEAR);
    // Not CallRaw: its far form loads the address into rax, and eax (the target) is read after the hit (jit_subcall). r10 is
    // free here: the GPR cache is flushed before a bctr or bctrl.
    static const bool alwaysLong = [] { const char* v = std::getenv("BRAWL_JIT_FAR_JUMPS"); return v != nullptr && *v != '\0' && *v != '0'; }();
    const void* const promote = CellPromoteRoutine();
    const int64_t distance = reinterpret_cast<const uint8_t*>(promote) - (a.getCurr() + 5);
    if (!alwaysLong && distance >= INT32_MIN && distance <= INT32_MAX) {
        a.call(promote);
    } else {
        a.mov(r10, reinterpret_cast<uint64_t>(promote));
        a.call(r10);
    }
    a.jmp(hit, CodeGenerator::T_NEAR);
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
    // The cell holds the last translated targets this site called, their records and what to call for each (JitFastCallFor: the
    // entry, or a routine that does what JitThinCall does); the key is the ctr it was found for.
    a.mov(eax, e.Ctr());
    a.and_(eax, ~3u);
    if (kDirectCalls) {
        // The newest target of the cell first, then the inline cache of untranslated targets (1, shared with the block's bctr:
        // both put only pcs that are no translated function in it, and a call to one in this frame is interpreted here: lr is set,
        // it is a jump), then the cell's other targets. A miss of all of them is JitCallOpCached; its interpret-here answer
        // (here, not in the loop frame, where Call decides otherwise: an untranslated target) links cache 1.
        Xbyak::Label& ways = e.NewLabel();
        Xbyak::Label& hit = e.NewLabel();
        Xbyak::Label& miss = e.NewLabel();
        a.cmp(eax, a.dword[a.rip + cell]);
        a.jne(ways, CodeGenerator::T_NEAR);
        a.lea(rdx, a.ptr[a.rip + cell]);
        a.L(hit);
        a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
        a.call(a.qword[rdx + static_cast<int>(offsetof(JitTargetCell, fast))]);
        EmitF1Scalar(e);  // NormalizeCallFloatReturn
        a.L(resume);
        e.EmitRecheck(pc + 4);
        e.Cold([&ways, &hit, &miss, &resume, &cell, index](BlockEmitter& c) {
            c.a.L(ways);
            c.CacheProbe(1);
            CellWays(c, cell, hit, miss);
            c.a.L(miss);
            c.a.lea(r10, c.a.ptr[c.a.rip + cell]);
            c.CallHelper(&JitCallOpCached, kFrame, &c.block, index, r10);
            c.a.mov(rcx, rax);
            c.a.shr(rcx, 32);
            c.a.jz(resume, CodeGenerator::T_NEAR);
            Xbyak::Label pending;
            c.a.cmp(ecx, kReturnPending);
            c.a.je(pending);
            c.ExitDynamicMiss(1);  // eax: the target
            c.a.L(pending);
            c.ExitPending();
        });
        e.Cold([&generic, &resume, &cell, index](BlockEmitter& c) {  // the rollback loop frame: Call decides
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
    e.CellProbe(cell, generic);
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
    // goes on at lr (the tail caller was inlined code of this frame): a block of its own, probed first; through the dispatcher
    // when the callee set XER[SO] or moved the epoch
    e.ExitDynamicAfterCall(0);
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
    if (kDirectCalls) {
        // The cell holds the last translated targets this site jumped to (a tail call), their records and what to call for each
        // (JitFastCallFor); the inline cache (1) the untranslated ones, plain jumps. The two hold different pcs, so their order is
        // free: the cell's newest target first, the inline cache, the cell's other targets.
        Xbyak::Label& hit = e.NewLabel();
        Xbyak::Label notFirst;
        a.cmp(eax, a.dword[a.rip + cell]);
        a.jne(notFirst, CodeGenerator::T_NEAR);
        a.lea(rdx, a.ptr[a.rip + cell]);
        a.L(hit);
        a.mov(e.Pc(), pc);
        EmitLoopFrameTest(e, generic);
        a.mov(ebp, e.Lr());
        a.and_(ebp, ~3u);
        a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
        a.call(a.qword[rdx + static_cast<int>(offsetof(JitTargetCell, fast))]);  // keeps rbp: no subcall frame (JitFastCallFor)
        EmitDirectTailFinish(e);
        a.L(notFirst);
        e.CacheProbe(1);
        CellWays(e, cell, hit, generic);
    } else {
        e.CacheProbe(1);
        e.CellProbe(cell, generic);
        a.mov(e.Pc(), pc);
        EmitLoopFrameTest(e, generic);
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

uint32_t EmitCompareBranch(BlockEmitter& e, uint32_t index) {
    const Block& b = e.block;
    uint32_t j = index + 1;
    while (j < b.count && b.ops[j].kind == K::Jmp) ++j;
    if (j >= b.count || b.ops[j].kind != K::Bc || KindForced(K::Bc)) return 0;
    const Op& o = b.ops[index];
    const Op& bc = b.ops[j];
    Compare c{};
    uint32_t field = 0;
    switch (o.kind) {
    case K::Cmpwi: case K::Cmplwi: case K::Cmpw: case K::Cmplw: field = static_cast<uint32_t>(o.d) >> 2u; break;
    case K::RlwinmRc: case K::AndiRc: case K::AndisRc: case K::AndRc: case K::OrRc: break;
    default: return 0;
    }
    if ((bc.d & 0x10u) != 0 || (bc.a >> 2) != field) return 0;  // the pair gains only when the bc tests this op's field
    // What the core does before a Bc of its own (EmitOps): the FPR cache's dirty slots and pending forms stored first. Here it
    // must come before the record form, whose flags the branch uses: a pending form's store is an `or` / `and` that would
    // clobber them in EmitCondBranch's flush.
    e.FlushDirtyFprs();
    const Xbyak::Reg32* result = nullptr;
    bool resultFlags = false;
    switch (o.kind) {
    case K::Cmpwi: c = {Compare::Imm, true, field, static_cast<int32_t>(o.imm), o.a, 0}; break;
    case K::Cmplwi: c = {Compare::Imm, false, field, static_cast<int32_t>(o.imm), o.a, 0}; break;
    case K::Cmpw: c = {Compare::Reg, true, field, 0, o.a, o.b}; break;
    case K::Cmplw: c = {Compare::Reg, false, field, 0, o.a, o.b}; break;
    default: result = EmitRecordResult(e, o, c, resultFlags); break;
    }
    EmitCondBranch(e, j, &c, result, resultFlags);
    return j - index + 1;
}

uint32_t EmitBranch(BlockEmitter& e, uint32_t index) {
    const Op& o = e.block.ops[index];
    switch (o.kind) {
    case K::Bc:
        EmitCondBranch(e, index, nullptr, nullptr, false);
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
