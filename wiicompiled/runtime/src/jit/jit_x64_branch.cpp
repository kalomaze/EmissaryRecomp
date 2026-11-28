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
//      to a translated function is JitThinTail (JitTailOp in the rollback loop frame); to a target a code variant is keyed at,
//      or before the registry is published, the fallback (RunOp asks ContinuationAt on every execution, and its answer there
//      follows guest memory).
//  Bl: native when the site was resolved at decode, the target is untranslated and runnable: lr and the loop-frame test
//      inline, then ExitLinked. A translated target with a raw dispatch record: lr and the loop-frame test inline (the loop
//      frame goes the general way; ctx->pc, a diagnostic, is stored by the shims that run Call, not before direct calls), the
//      callee's float arguments tested in line, then a direct call of the entry when it needs no guard, else JitThinCall with the
//      target's JitCallTarget (everything Call and TryDispatchRawCpuTarget look up, settled at compile time), and a validity test.
//      A patched target (guard byte set at compile time) has its block called with a host call while it stays patched
//      (EmitSubFrameBl). Otherwise JitCallOp (Call) and a validity test when the call was made.
//  Blr, Bclr: native; the return of the frame's own function normalises f1 and leaves, that of a function called with a host call
//      normalises f1 and returns with `ret` (the caller drops the return slot), anything else leaves through an inline cache. A compare and the bclr after it
//      branch on the compare's flags (EmitCompareReturn).
//  Bctr: its two inline ways (direct calls of its common translated targets, JitCallWays; not at a switch's table), the newest target
//      of its cell, an inline cache probe (untranslated targets only ever get in it), the cell's other targets (translated targets:
//      each with what JitFastCallFor says to call for it, as a tail call), a miss JitTailOpCached, which fills the cell; the fallback
//      until the registry is published.
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
        } else if (cmp->source == Compare::Imm && !cmp->isSigned && cmp->imm == 0 && e.WantCacheGpr(cmp->a)) {
            value = e.ReadGpr(cmp->a).cvt64();  // an unsigned compare with 0: the zero-extended register is the difference
            if (flags) a.test(value, value);
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

// The function returned (lr is the address it was entered with, Frame::returnAddress): f1 becomes the scalar the caller takes
// (inline when it is not in paired form), then the return. Else a jump through lr, through the inline cache.
//
// Linux x86-64: (lr & ~3) ^ Frame::returnAddress is 0 for the frame's own function's return, 1 (the tag) for the return of a
// function a block called with a host call (`ret` to the caller's code after the caller's word is back: ExitSharedReturn, which
// the return stack predicts), anything else a jump.
void EmitReturnOrJump(BlockEmitter& e) {
    CodeGenerator& a = e.a;
    const auto T = CodeGenerator::T_NEAR;
    const bool f1Scalar = e.KnownForm(e.index, 1) == Form::Scalar;
    if (!kDirectCalls) {
        Xbyak::Label notReturn;
        Xbyak::Label& paired = e.NewLabel();
        a.mov(eax, e.Lr());
        a.and_(eax, ~3u);
        a.cmp(eax, e.FrameReturn());  // kept & ~3 (Frame::returnAddress)
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
        return;
    }
    Xbyak::Label& frame = e.NewLabel();
    Xbyak::Label& other = e.NewLabel();
    Xbyak::Label& sharedTail = e.NewLabel();
    Xbyak::Label& frameTail = e.NewLabel();
    Xbyak::Label& sharedF1 = e.NewLabel();
    Xbyak::Label& frameF1 = e.NewLabel();
    a.mov(eax, e.Lr());
    a.and_(eax, ~3u);
    a.xor_(eax, e.FrameReturn());
    a.jz(frame, T);
    a.cmp(eax, 1);
    a.jne(other, T);
    if (!f1Scalar) {
        a.test(a.byte[kFrame + static_cast<int>(offsetof(Frame, pairedMask))], 2u);  // bit 1: f1 is Paired
        a.jnz(sharedF1, T);
    }
    a.L(sharedTail);
    e.ExitSharedReturn();
    a.L(frame);
    if (!f1Scalar) {
        a.test(a.byte[kFrame + static_cast<int>(offsetof(Frame, pairedMask))], 2u);
        a.jnz(frameF1, T);
    }
    a.L(frameTail);
    e.ExitFrameReturn();
    a.L(other);
    a.xor_(eax, e.FrameReturn());  // lr & ~3 again
    e.ExitDynamicCached(0);
    e.Cold([&sharedF1, &frameF1, &sharedTail, &frameTail](BlockEmitter& c) {
        c.a.L(sharedF1);
        c.CallHelper(&JitNormalizeF1, kFrame);
        c.a.jmp(sharedTail, CodeGenerator::T_NEAR);
        c.a.L(frameF1);
        c.CallHelper(&JitNormalizeF1, kFrame);
        c.a.jmp(frameTail, CodeGenerator::T_NEAR);
    });
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

// Finishes a direct tail call whose saved return address (lr & ~3) sits in ebp: restores lr, and either returns (the function
// returned: f1 is the callee's scalar; the frame or the caller's word it goes back to has no use for f1's paired bit) or clears f1's
// paired bit and continues at ebp through the inline cache.
void EmitDirectTailFinish(BlockEmitter& e) {
    CodeGenerator& a = e.a;
    a.mov(e.Lr(), ebp);
    if (kDirectCalls) {
        // ebp + 1 is the tagged field of a function called with a host call that returns (the common case: a tail call is how
        // most functions a switch dispatches to end), ebp the frame's own function's return address
        Xbyak::Label& notShared = e.NewLabel();
        Xbyak::Label& jump = e.NewLabel();
        a.lea(eax, a.ptr[rbp + 1]);
        a.cmp(eax, e.FrameReturn());
        a.jne(notShared, CodeGenerator::T_NEAR);
        e.ExitSharedReturn();
        a.L(notShared);
        a.cmp(ebp, e.FrameReturn());
        a.jne(jump, CodeGenerator::T_NEAR);
        e.ExitFrameReturn();
        a.L(jump);
        EmitF1Scalar(e);
        a.mov(eax, ebp);
        e.ExitDynamicAfterCall(0);  // through the dispatcher when the callee set XER[SO] or moved the epoch
        return;
    }
    Xbyak::Label returned;
    EmitF1Scalar(e);
    a.cmp(ebp, e.FrameReturn());  // kept & ~3 (Frame::returnAddress)
    a.je(returned, CodeGenerator::T_SHORT);
    a.mov(eax, ebp);
    e.ExitDynamicAfterCall(0);  // through the dispatcher when the callee set XER[SO] or moved the epoch
    a.L(returned);
    e.ExitReturn();
}

// A bl to a translated function `t` that is patched (its guard byte set when the block is compiled), the common JIT->JIT call:
// while the guard byte stays set and no redirect applies, the callee's block is called with a host call in this Frame, which is
// what calling the wrapper does then (RunPatched, RunCompiledEntry, jit_enter_direct: a Run of the function in a frame of its own)
// without the C++ in between. The only per-call state of that frame is pairedMask (0: the callee starts with every FPR scalar) and
// returnAddress (pc + 4): the caller's pair goes on the stack (the return slot, `push`), the callee's is stored tagged (bit 0:
// returned to with `ret`, ExitSharedReturn, after which the caller drops the slot), and the callee's return puts the caller's back. Then RunPatched's Sync test,
// NormalizeCallFloatReturn and the validity test. Otherwise the wrapper, or JitCallOp in the rollback loop frame (Call decides its
// calls, before the float arguments are touched).
void EmitSubFrameBl(BlockEmitter& e, uint32_t index, const JitCallTarget& t) {
    CodeGenerator& a = e.a;
    const uint32_t pc = e.PcOf(index);
    const uint32_t target = t.target;
    const auto T = CodeGenerator::T_NEAR;
    Xbyak::Label& loopFrame = e.NewLabel();
    Xbyak::Label& wrapper = e.NewLabel();  // the wrapper, float arguments done
    Xbyak::Label& resume = e.NewLabel();
    Xbyak::Label& made = e.NewLabel();
    Xbyak::Label& sync = e.NewLabel();
    a.mov(e.Lr(), pc + 4);
    a.test(a.byte[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagLoopFrame);
    a.jnz(loopFrame, T);
    EmitStaticArgTest(e, t);
    a.push(e.FrameWord());
    a.mov(rax, static_cast<uint64_t>((pc + 4) | 1u) << 32);  // pairedMask 0, returnAddress pc + 4 tagged
    a.mov(e.FrameWord(), rax);
    e.CallLinked(target, t.patchedByte, wrapper);  // the guard byte and the redirect bound: tested by the link's stub
    a.lea(rsp, a.ptr[rsp + 8]);  // the return slot: the callee's `ret` leaves it (ExitSharedReturn)
    const PatchedHooks& hooks = GetPatchedHooks();
    a.cmp(AbsMem(a, hooks.pendingFast, rcx, 64), 0);
    a.jne(sync, T);
    a.cmp(AbsMem(a, hooks.anyDirty, rcx, 8), 0);
    a.jne(sync, T);
    a.L(resume);
    EmitF1Scalar(e);  // NormalizeCallFloatReturn (the callee's return made f1 scalar in its own state)
    a.L(made);
    e.EmitRecheck(pc + 4);
    const JitCallTarget* const tp = &t;
    e.Cold([&loopFrame, &wrapper, &resume, &made, &sync, index, target, pc, tp](BlockEmitter& c) {
        CodeGenerator& g = c.a;
        g.L(sync);
        c.CallRaw(reinterpret_cast<const void*>(GetPatchedHooks().sync));
        g.jmp(resume, CodeGenerator::T_NEAR);
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

// The FPRs an fprsOnly record's call keeps in host registers (kThinSaveMagic), in register order; 0 when its mask has none or more
// than kThinSaveMax (JitDirectFprsCall then).
unsigned ThinSaveFprs(const JitCallTarget& t, uint8_t* fprs) {
    const uint32_t mask = t.fprMask & kPpcAllNonvolatileFprMask;
    if (mask == 0 || static_cast<unsigned>(__builtin_popcount(mask)) > kThinSaveMax) return 0;
    unsigned n = 0;
    for (uint32_t m = mask; m != 0; m &= m - 1u) fprs[n++] = static_cast<uint8_t>(__builtin_ctz(m));
    return n;
}

// JitDirectFprsCall's work for a record of one to three FPRs, in line (kThinSaveMagic): their raw bits in rbp, r14, r12 across the
// direct call, stored back after it, then the pinned r14 and r12 set again. As on the fastDirect path, the execution address (a
// fault report's diagnostic) is not stored and the ambient context is the caller's (compiled code runs under its Run's).
void EmitThinSaveCall(BlockEmitter& e, const JitCallTarget& t, const uint8_t* fprs, unsigned n) {
    CodeGenerator& a = e.a;
    static const Xbyak::Reg64* const regs[kThinSaveMax] = {&rbp, &kEpochCell, &kLean};
    for (unsigned k = 0; k < n; ++k) a.mov(*regs[k], e.Fpr(fprs[k]));
    a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
    e.CallRaw(reinterpret_cast<const void*>(t.entry));
    const uint32_t d = ThinSaveDescriptor(fprs, n);  // nop dword [rax + rax + d]: what the personality routine reads
    a.db(0x0F), a.db(0x1F), a.db(0x84), a.db(0x00), a.dd(d);
    for (unsigned k = 0; k < n; ++k) a.mov(e.Fpr(fprs[k]), *regs[k]);
    if (n > 1) a.mov(kEpochCell, reinterpret_cast<uint64_t>(&BrawlDirtyMemory::markEpoch));
    if (n > 2) a.mov(kLean, AbsMem(a, &MemoryInline::g_leanBase, kLean, 64));
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
    a.mov(e.Lr(), pc + 4);
    EmitLoopFrameTest(e, loopFrame);
    EmitStaticArgTest(e, t);
    const bool guardedDirect = t.preserveGprs && !t.returnsPaired && !t.special && GuardedDirectOn();
    if (kDirectCalls && (t.fastDirect || (t.fprsOnly && !t.returnsPaired && !t.special) || guardedDirect)) {
        uint8_t fprs[kThinSaveMax];
        const unsigned saves = t.fprsOnly && ThinSavesOn() ? ThinSaveFprs(t, fprs) : 0;
        if (t.fastDirect) {
            a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
            e.CallRaw(reinterpret_cast<const void*>(t.entry));
        } else if (saves != 0) {
            EmitThinSaveCall(e, t, fprs, saves);
        } else if (guardedDirect) {
            a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
            a.mov(ArgReg(1), reinterpret_cast<uint64_t>(&t));
            e.CallRaw(reinterpret_cast<const void*>(&JitDirectGuardedCall));
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

// After a call site's shim (JitTailOpCached / JitCallOpCached, which put a translated target at the front of the site's cell): the
// cell's first entry into the code cache's table (BlockEmitter::TargetTable). Keeps rax (the shim's answer); clobbers rcx, rdx,
// rsi, xmm0, xmm1.
void TableFill(BlockEmitter& e, bool tail, Xbyak::Label& cell) {
    uint8_t* const table = e.TargetTable(tail);
    if (table == nullptr) return;
    CodeGenerator& a = e.a;
    Xbyak::Label skip;
    a.lea(rcx, a.ptr[a.rip + cell]);
    a.mov(edx, a.dword[rcx]);
    a.cmp(edx, JitTargetCell::kNoKey);
    a.je(skip, CodeGenerator::T_SHORT);
    a.and_(edx, (BlockEmitter::kTargetTableEntries - 1) << 2);
    a.mov(rsi, reinterpret_cast<uint64_t>(table));
    a.lea(rdx, a.ptr[rsi + rdx * 8]);
    a.movups(xmm0, a.ptr[rcx]);
    a.movups(xmm1, a.ptr[rcx + 16]);
    a.movups(a.ptr[rdx], xmm0);
    a.movups(a.ptr[rdx + 16], xmm1);
    a.L(skip);
}

// The code cache's table of translated targets (BlockEmitter::TargetTable) for eax: jumps to `hit` with rdx = the entry, else falls
// through. A site whose targets are more than its cell's ways (a virtual call over many classes) finds them there instead of going to
// its shim. Every 8th hit of a table entry (its `hits` byte, counted over all sites) also copies it into this site's first way, so
// that a site's common target comes back to the one test the site makes first (as the cell's own ways move up). Clobbers rcx, rdx,
// xmm0, xmm1.
void TableProbe(BlockEmitter& e, bool tail, Xbyak::Label& cell, Xbyak::Label& hit) {
    uint8_t* const table = e.TargetTable(tail);
    if (table == nullptr) return;
    CodeGenerator& a = e.a;
    static_assert(sizeof(JitTargetCell) == 32 && offsetof(JitTargetCell, target) == 0);
    Xbyak::Label miss;
    a.mov(ecx, eax);
    a.and_(ecx, (BlockEmitter::kTargetTableEntries - 1) << 2);
    a.mov(rdx, reinterpret_cast<uint64_t>(table));
    a.lea(rdx, a.ptr[rdx + rcx * 8]);
    a.cmp(eax, a.dword[rdx]);
    a.jne(miss, CodeGenerator::T_SHORT);
    a.add(a.byte[rdx + static_cast<int>(offsetof(JitTargetCell, hits))], 32);  // carries on the 8th
    a.jnc(hit, CodeGenerator::T_NEAR);
    a.movups(xmm0, a.ptr[rdx]);
    a.movups(xmm1, a.ptr[rdx + 16]);
    a.lea(rdx, a.ptr[a.rip + cell]);
    a.movups(a.ptr[rdx], xmm0);
    a.movups(a.ptr[rdx + 16], xmm1);
    a.jmp(hit, CodeGenerator::T_NEAR);
    a.L(miss);
}

// The ways of a cell after the first (the site tests the first itself): jumps to `hit` with rdx = the matching entry, else to `miss`.
// eax is the target. An entry hit 8 times while not the first becomes the first (CellPromoteRoutine): the cell keeps the targets in
// the order the misses put them in (the newest first), and a site's common target moves up to the one test the site makes first.
// Clobbers rcx, rdx, xmm0-xmm3.
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

// The site's inline ways get the cell entry `entry` (JitPatchCallWays; `fill`: only into an empty way 0); nothing once the site's
// budget is spent. Clobbers every scratch register.
void PatchWays(BlockEmitter& e, uint32_t site, const Xbyak::Reg64& entry, bool fill) {
    CodeGenerator& a = e.a;
    Xbyak::Label spent;
    a.cmp(a.dword[a.rip + e.WaysRecord(site) + static_cast<int>(offsetof(JitCallWays, budget))], 0);
    a.je(spent, CodeGenerator::T_NEAR);
    if (entry.getIdx() != ArgReg(1).getIdx()) a.mov(ArgReg(1), entry);
    a.lea(ArgReg(0), a.ptr[a.rip + e.WaysRecord(site)]);
    a.mov(ArgReg(2).cvt32(), fill ? 1 : 0);
    e.CallRaw(reinterpret_cast<const void*>(&JitPatchCallWays));
    a.L(spent);
}

// At a call site's slow hit (rdx: the cell or table entry): every 32nd hit of the entry out of line writes it into the site's inline
// ways (JitTargetCell::wayHits), then goes on at `call` with rdx and eax as they were. rbp is free at a bctrl or bctr (the GPR cache
// is flushed before them, and the call sets ebp after this). Clobbers the flags.
void CountWayHit(BlockEmitter& e, uint32_t site, Xbyak::Label& call) {
    CodeGenerator& a = e.a;
    Xbyak::Label& hot = e.NewLabel();
    a.add(a.byte[rdx + static_cast<int>(offsetof(JitTargetCell, wayHits))], 8);  // carries on the 32nd
    a.jc(hot, CodeGenerator::T_NEAR);
    a.L(call);
    e.Cold([&hot, &call, site](BlockEmitter& c) {
        c.a.L(hot);
        c.a.mov(rbp, rdx);
        PatchWays(c, site, rdx, false);
        c.a.mov(rdx, rbp);
        c.a.mov(eax, c.Ctr());
        c.a.and_(eax, ~3u);
        c.a.jmp(call, CodeGenerator::T_NEAR);
    });
}

// A bctrl's or bctr's probe-order switch (BlockEmitter::EmitOrderSwitch), eax the masked target: once switched, cache 1 is probed
// out of line first, a hit leaves the block through it, a miss goes on here. Clobbers ecx, rdx and the flags only when switched (as
// the probes after it do).
void EmitCacheFirstSwitch(BlockEmitter& e) {
    if (!e.OrderSwitchRoom()) return;
    Xbyak::Label& cacheFirst = e.NewLabel();
    Xbyak::Label& back = e.NewLabel();
    if (!e.EmitOrderSwitch(cacheFirst)) return;
    e.a.L(back);
    e.Cold([&cacheFirst, &back](BlockEmitter& c) {
        c.a.L(cacheFirst);
        c.CacheProbe(1);
        c.a.jmp(back, CodeGenerator::T_NEAR);
    });
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
    a.mov(e.Lr(), pc + 4);
    EmitLoopFrameTest(e, generic);
    // The cell holds the last translated targets this site called, their records and what to call for each (JitFastCallFor: the
    // entry, or a routine that does what JitThinCall does); the key is the ctr it was found for.
    a.mov(eax, e.Ctr());
    a.and_(eax, ~3u);  // the empty entries' keys are 1 (JitTargetCell::kNoKey, JitLink::kNoKey): only a masked ctr never matches them
    if (kDirectCalls) {
        // The site's inline ways first (JitCallWays: direct calls of its common targets), then out of line the cell's newest target,
        // the code cache's table of translated targets, the inline cache of untranslated targets (1, shared with the block's bctr:
        // both put only pcs that are no translated function in it, and a call to one in this frame is interpreted here: lr is set,
        // it is a jump), the cell's other targets. A miss of all of them is JitCallOpCached; its interpret-here answer (here, not
        // in the loop frame, where Call decides otherwise: an untranslated target) links cache 1. The site's first target (a miss)
        // and a target hit out of line 32 times (CountWayHit) are written into the ways.
        Xbyak::Label& ways = e.NewLabel();
        Xbyak::Label& hit = e.NewLabel();
        Xbyak::Label& miss = e.NewLabel();
        Xbyak::Label& way1 = e.NewLabel();
        Xbyak::Label& slow = e.NewLabel();
        Xbyak::Label& after = e.NewLabel();
        Xbyak::Label& never = e.NewLabel();
        Xbyak::Label& call = e.NewLabel();
        const uint32_t site = e.NewWays();
        EmitCacheFirstSwitch(e);
        a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
        e.EmitWayCompare(site, 0);
        a.jne(way1, CodeGenerator::T_NEAR);
        e.EmitWayCall(site, 0, never);
        a.L(after);
        EmitF1Scalar(e);  // NormalizeCallFloatReturn
        a.L(resume);
        e.EmitRecheck(pc + 4);
        e.Cold([&way1, &slow, &after, &never, &hit, &ways, &cell, &call, site](BlockEmitter& c) {
            CodeGenerator& g = c.a;
            g.L(way1);
            c.EmitWayCompare(site, 1);
            g.jne(slow, CodeGenerator::T_NEAR);
            c.EmitWayCall(site, 1, never);
            g.jmp(after, CodeGenerator::T_NEAR);
            g.L(never);
            g.ud2();  // a way's call before its key is written: no masked ctr equals kNoKey
            g.L(slow);
            g.cmp(eax, g.dword[g.rip + cell]);
            g.jne(ways, CodeGenerator::T_NEAR);
            g.lea(rdx, g.ptr[g.rip + cell]);
            g.L(hit);
            CountWayHit(c, site, call);
            g.lea(ArgReg(0), g.ptr[kCtxBase - kCtxBias]);
            g.call(g.qword[rdx + static_cast<int>(offsetof(JitTargetCell, fast))]);
            g.jmp(after, CodeGenerator::T_NEAR);
        });
        e.Cold([&ways, &hit, &miss, &resume, &cell, index, site](BlockEmitter& c) {
            CodeGenerator& g = c.a;
            g.L(ways);
            TableProbe(c, false, cell, hit);
            c.CacheProbe(1);
            CellWays(c, cell, hit, miss);
            g.L(miss);
            g.lea(r10, g.ptr[g.rip + cell]);
            c.CallHelper(&JitCallOpCached, kFrame, &c.block, index, r10);
            TableFill(c, false, cell);
            g.mov(rcx, rax);
            g.shr(rcx, 32);
            Xbyak::Label pending, notMade;
            g.jnz(notMade, CodeGenerator::T_NEAR);
            g.lea(rdx, g.ptr[g.rip + cell]);  // the call was made: the target it filled the cell with (if any) to an empty way
            PatchWays(c, site, rdx, true);
            g.jmp(resume, CodeGenerator::T_NEAR);
            g.L(notMade);
            g.cmp(ecx, kReturnPending);
            g.je(pending);
            c.ExitDynamicMiss(1);  // eax: the target
            g.L(pending);
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
    // RunOp asks GuestCodePatch::ContinuationAt on every execution, and its answer follows guest memory: a target some code
    // variant is keyed at goes to the fallback; any other target can never be a continuation.
    if (JitContinuationPossibleAt(o.imm)) return e.EmitFallback(index);
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

// Whether the bctr at `index` jumps through a switch's table: ctr set by the op before from a register the op before that loaded with
// lwzx (`lwzx rT, rB, rI; mtctr rT; bctr`). Its targets are code of the function (cases), never translated functions: the inline
// cache comes first. Only an order (a heuristic): every target is found either way.
bool JumpTableSite(const Block& b, uint32_t index) {
    uint32_t i = index;
    const auto previous = [&b](uint32_t& at) {
        while (at > 0 && b.ops[at - 1].kind == K::Jmp) --at;
        return at > 0 ? &b.ops[--at] : nullptr;
    };
    const Op* const mtctr = previous(i);
    if (mtctr == nullptr || mtctr->kind != K::Mtctr) return false;
    const Op* const load = previous(i);
    return load != nullptr && load->kind == K::Lwzx && load->d == mtctr->d;
}

uint32_t EmitBctr(BlockEmitter& e, uint32_t index) {
    CodeGenerator& a = e.a;
    // A cached target is one the registry did not know when the dispatcher linked it, and the registry is fixed once
    // published: before that, nothing may be cached.
    if (!TranslatedFunctionRegistry::IsLookupPublished()) return e.EmitFallback(index);
    const uint32_t pc = e.PcOf(index);
    Xbyak::Label& cell = e.NewCell();
    Xbyak::Label& generic = e.NewLabel();
    a.mov(eax, e.Ctr());
    a.and_(eax, ~3u);  // the empty entries' keys are 1 (JitTargetCell::kNoKey, JitLink::kNoKey): only a masked ctr never matches them
    uint32_t site = ~0u;  // the site's inline ways (JitCallWays), if it has them
    if (kDirectCalls) {
        // The cell holds the last translated targets this site jumped to (a tail call), their records and what to call for each
        // (JitFastCallFor); the inline cache (1) the untranslated ones, plain jumps. The two hold different pcs, so their order is
        // free: the site's inline ways (JitCallWays: direct calls of its common translated targets), out of line the cell's newest
        // target, the code cache's table, the inline cache, the cell's other targets; at a switch's table (its targets are code of
        // the function) the inline cache first and no ways. The first target and one hit out of line 32 times go in the ways.
        Xbyak::Label& hit = e.NewLabel();
        Xbyak::Label& ways = e.NewLabel();
        const bool table = JumpTableSite(e.block, index);
        // the call (rdx: the cell entry, or `way` >= 0: that inline way's direct call); keeps rbp (JitFastCallFor: never jit_subcall)
        const auto call = [&e, &generic](int way, uint32_t at, Xbyak::Label* never) {
            CodeGenerator& g = e.a;
            EmitLoopFrameTest(e, generic);
            g.mov(ebp, e.Lr());
            g.and_(ebp, ~3u);
            g.lea(ArgReg(0), g.ptr[kCtxBase - kCtxBias]);
            if (way < 0) g.call(g.qword[rdx + static_cast<int>(offsetof(JitTargetCell, fast))]);
            else e.EmitWayCall(at, static_cast<uint32_t>(way), *never);
        };
        if (table) {
            e.CacheProbe(1);
            a.cmp(eax, a.dword[a.rip + cell]);
            a.jne(ways, CodeGenerator::T_NEAR);
            a.lea(rdx, a.ptr[a.rip + cell]);
            a.L(hit);
            call(-1, 0, nullptr);
            EmitDirectTailFinish(e);
        } else {
            Xbyak::Label& way1 = e.NewLabel();
            Xbyak::Label& slow = e.NewLabel();
            Xbyak::Label& after = e.NewLabel();
            Xbyak::Label& never = e.NewLabel();
            site = e.NewWays();
            EmitCacheFirstSwitch(e);
            e.EmitWayCompare(site, 0);
            a.jne(way1, CodeGenerator::T_NEAR);
            call(0, site, &never);
            a.L(after);
            EmitDirectTailFinish(e);
            // after the exits (as the probes below): way 1, the cell's first entry
            a.L(way1);
            e.EmitWayCompare(site, 1);
            a.jne(slow, CodeGenerator::T_NEAR);
            call(1, site, &never);
            a.jmp(after, CodeGenerator::T_NEAR);
            a.L(never);
            a.ud2();  // a way's call before its key is written: no masked ctr equals kNoKey
            a.L(slow);
            a.cmp(eax, a.dword[a.rip + cell]);
            a.jne(ways, CodeGenerator::T_NEAR);
            a.lea(rdx, a.ptr[a.rip + cell]);
            a.L(hit);
            CountWayHit(e, site, e.NewLabel());
            call(-1, 0, nullptr);
            a.jmp(after, CodeGenerator::T_NEAR);
        }
        a.L(ways);
        TableProbe(e, true, cell, hit);
        if (!table) e.CacheProbe(1);
        CellWays(e, cell, hit, generic);
    } else {
        e.CacheProbe(1);
        e.CellProbe(cell, generic);
        a.mov(e.Pc(), pc);
        EmitLoopFrameTest(e, generic);
        e.CallHelper(&JitThinTail, kFrame, rdx);
        EmitTailResult(e);
    }
    e.Cold([&generic, &cell, index, site](BlockEmitter& c) {
        CodeGenerator& g = c.a;
        Xbyak::Label pending, returned, plain;
        g.L(generic);
        g.lea(r10, g.ptr[g.rip + cell]);
        c.CallHelper(&JitTailOpCached, kFrame, &c.block, index, r10);
        if (kDirectCalls) TableFill(c, true, cell);
        if (site != ~0u) {  // the target the shim filled the cell with (if any) to an empty way; rbp (free here) keeps the answer
            g.mov(rbp, rax);
            g.lea(rdx, g.ptr[g.rip + cell]);
            PatchWays(c, site, rdx, true);
            g.mov(rax, rbp);
        }
        EmitShimResult(c);
        g.cmp(ecx, kJumpHere);
        g.je(plain);
        g.cmp(ecx, kReturnFrame);
        g.je(returned);
        g.cmp(ecx, kReturnPending);
        g.je(pending);
        c.ExitDynamic();
        g.L(returned);
        c.ExitReturn();
        g.L(pending);
        c.ExitPending();
        g.L(plain);
        c.ExitDynamicMiss(1);  // eax: the target
    });
    return 1;
}

}  // namespace

// A compare or record form and the conditional return (bclr without a ctr decrement) that tests lt, gt or eq of its field: the field
// is written (the return leaves the block) and the return is taken on the host flags of the field value, not on the field read
// back from memory. The GPR cache is flushed first (EmitOps does it before a Bclr of its own); the FPR cache's pending forms were
// stored before the compare.
void EmitCompareReturn(BlockEmitter& e, uint32_t bci, const Compare& c, const Xbyak::Reg32* result, bool resultFlags) {
    CodeGenerator& a = e.a;
    const Op& bclr = e.block.ops[bci];
    const bool flags = !(c.source == Compare::Test && resultFlags);
    Xbyak::Reg64 value = rax;
    if (c.source == Compare::Test && c.nonNegative && result != nullptr) {
        value = result->cvt64();  // zero-extended: the field value itself
        if (flags) a.test(*result, *result);
    } else {
        EmitCompareValue(e, c, flags, result);
    }
    e.CrStore(c.field, value);
    e.FlushAllGprs();  // stores only: the flags stay
    Xbyak::Label skip;
    Jcc(a, Invert(CompareCc(true, bclr.a & 3u, (bclr.d & 0x08u) != 0)), skip, CodeGenerator::T_NEAR);
    EmitReturnOrJump(e);
    a.L(skip);
}

uint32_t EmitCompareBranch(BlockEmitter& e, uint32_t index) {
    const Block& b = e.block;
    uint32_t j = index + 1;
    while (j < b.count && b.ops[j].kind == K::Jmp) ++j;
    if (j >= b.count) return 0;
    const Op& next = b.ops[j];
    // bclr: no ctr decrement, a condition that is not SO (whose bit the host flags do not have)
    const bool isReturn = next.kind == K::Bclr && !KindForced(K::Bclr) && (next.d & 0x04u) != 0 && (next.d & 0x10u) == 0 &&
                          (next.a & 3u) != 3u;
    if (!isReturn && (next.kind != K::Bc || KindForced(K::Bc))) return 0;
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
    if (isReturn) EmitCompareReturn(e, j, c, result, resultFlags);
    else EmitCondBranch(e, j, &c, result, resultFlags);
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
