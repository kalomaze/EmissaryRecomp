// The intra-block FPR cache of the x86-64 backend (jit_x64.h, "intra-block FPR cache"): guest FPRs in xmm registers within a
// block, their raw 64 bits in a slot's low quadword exactly as CpuContext::fpr holds them, and the Frame::pairedMask writes the
// float ops make kept pending until the next flush.
//
// What may hold a value between ops, and when it must be in memory instead:
//  * A slot is clean (memory holds its value) or dirty. Anything that calls out or leaves the block needs memory up to date:
//    FlushAllGprs / FlushVolatileGprs (every call and exit path of the core and the branch group goes through one of them)
//    store the dirty slots and the pending forms and empty the cache; a Bc stores them before it (FlushDirtyFprs, the clean
//    slots stay for the fall-through path); a cold path stores the snapshot taken where it branched off and reloads it after
//    its call (GprSnapshot carries the FPR slots and the pending forms).
//  * A pending form bit is kept only for a register whose form the block knows (it was written in the block): every op that
//    tests a form at run time tests one the block does not know, and that register's bit in memory is right, since nothing in
//    the block wrote it. "Known" changes only at the ops jit_x64_forms.cpp lists, and the ones that end the knowledge (calls,
//    unlisted ops) are behind a FlushAllGprs, which writes the pending bits first.
//  * A slot's register is a guest FPR, never two slots for one register.
#include "jit_x64.h"

#include <atomic>
#include <cstdlib>

namespace PpcInterpreter::detail::x64 {

namespace {

using namespace Xbyak::util;

struct FprCounters {
    std::atomic<uint64_t> hits{0}, loads{0}, stores{0}, formFlushes{0};
};
FprCounters g_fpr;

bool EnvOn(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && *v != '0';
}

}  // namespace

bool BlockEmitter::FprCacheOn() {
    static const bool on = !EnvOn("BRAWL_JIT_NO_FPR_CACHE");
    return on;
}

const Xbyak::Xmm& BlockEmitter::FprCacheReg(uint32_t slot) {
#if defined(_WIN32)
    static const Xbyak::Xmm* const regs[kFprCacheSlots] = {&xmm4, &xmm5};
#else
    static const Xbyak::Xmm* const regs[kFprCacheSlots] = {&xmm6, &xmm7, &xmm8, &xmm9, &xmm10, &xmm11, &xmm12, &xmm13, &xmm14, &xmm15};
#endif
    return *regs[slot];
}

void BlockEmitter::NoteFpr(FprEvent e) {
    switch (e) {
    case FprEvent::Hit: g_fpr.hits.fetch_add(1, std::memory_order_relaxed); break;
    case FprEvent::Load: g_fpr.loads.fetch_add(1, std::memory_order_relaxed); break;
    case FprEvent::Store: g_fpr.stores.fetch_add(1, std::memory_order_relaxed); break;
    case FprEvent::FormFlush: g_fpr.formFlushes.fetch_add(1, std::memory_order_relaxed); break;
    }
}

void BlockEmitter::StoreFpr(uint32_t d, const Xbyak::Xmm& x) {
#if defined(__AVX__)
    a.vmovsd(Fpr(d), x);
#else
    a.movsd(Fpr(d), x);
#endif
}

void BlockEmitter::LoadFpr(const Xbyak::Xmm& x, uint32_t r) {
#if defined(__AVX__)
    a.vmovsd(x, Fpr(r));
#else
    a.movsd(x, Fpr(r));
#endif
}

bool BlockEmitter::FprSlotFree(uint32_t pinMask) const {
    if (inCold_ || !FprCacheOn()) return false;
    for (uint32_t s = 0; s < kFprCacheSlots; ++s)
        if (fprSlots_[s].guest < 0 || ((pinMask >> fprSlots_[s].guest) & 1u) == 0) return true;
    return false;
}

int BlockEmitter::FindCachedFpr(uint32_t r) const {
    if (inCold_) return -1;
    const int8_t g = static_cast<int8_t>(r & 31u);
    for (uint32_t s = 0; s < kFprCacheSlots; ++s)
        if (fprSlots_[s].guest == g) return static_cast<int>(s);
    return -1;
}

// An empty slot, else the least recently used clean one, else the least recently used dirty one (stored first). pinMask: guest
// FPRs whose slots the op still needs.
uint32_t BlockEmitter::AllocFprSlot(uint32_t pinMask) {
    int bestClean = -1, bestDirty = -1;
    for (uint32_t s = 0; s < kFprCacheSlots; ++s) {
        const FprSlot& sl = fprSlots_[s];
        if (sl.guest < 0) return s;
        if (((pinMask >> sl.guest) & 1u) != 0) continue;
        int& best = sl.dirty ? bestDirty : bestClean;
        if (best < 0 || sl.lastUse < fprSlots_[best].lastUse) best = static_cast<int>(s);
    }
    if (bestClean < 0 && bestDirty < 0) throw Xbyak::Error(Xbyak::ERR_BAD_PARAMETER);  // every slot pinned: never with <= 4 operands
    const uint32_t chosen = static_cast<uint32_t>(bestClean >= 0 ? bestClean : bestDirty);
    if (fprSlots_[chosen].dirty) {
        StoreFpr(static_cast<uint32_t>(fprSlots_[chosen].guest), FprCacheReg(chosen));
        NoteFpr(FprEvent::Store);
    }
    fprSlots_[chosen] = FprSlot{};
    return chosen;
}

const Xbyak::Xmm& BlockEmitter::ReadFpr(uint32_t r, uint32_t pinMask) {
    const int hit = FindCachedFpr(r);
    if (hit >= 0) {
        fprSlots_[hit].lastUse = ++fprTick_;
        NoteFpr(FprEvent::Hit);
        return FprCacheReg(static_cast<uint32_t>(hit));
    }
    const uint32_t s = AllocFprSlot(pinMask | (1u << (r & 31u)));
    LoadFpr(FprCacheReg(s), r);  // the load clears the slot's upper quadword
    fprSlots_[s] = FprSlot{static_cast<int8_t>(r & 31u), false, ++fprTick_};
    NoteFpr(FprEvent::Load);
    return FprCacheReg(s);
}

const Xbyak::Xmm& BlockEmitter::ReserveFpr(uint32_t r, uint32_t pinMask) {
    const int hit = FindCachedFpr(r);
    if (hit >= 0) {
        fprSlots_[hit].lastUse = ++fprTick_;
        return FprCacheReg(static_cast<uint32_t>(hit));
    }
    const uint32_t s = AllocFprSlot(pinMask | (1u << (r & 31u)));
    fprSlots_[s] = FprSlot{static_cast<int8_t>(r & 31u), false, ++fprTick_};
    return FprCacheReg(s);
}

void BlockEmitter::MarkFprDirty(uint32_t r) {
    const int hit = FindCachedFpr(r);
    if (hit >= 0) fprSlots_[hit].dirty = true;
}

void BlockEmitter::DropFpr(uint32_t r) {
    const int hit = FindCachedFpr(r);
    if (hit >= 0) fprSlots_[hit] = FprSlot{};
}

void BlockEmitter::SetFprForm(uint32_t r, Form form) {
    const uint32_t bit = 1u << (r & 31u);
    if (form == Form::Unknown) {  // the op writes r's bit in memory itself (fmr of a register whose form is not known)
        formPending_ &= ~bit;
        formPendingPaired_ &= ~bit;
        return;
    }
    // the form r had is this one: memory holds it already, or the pending write does (its value is always r's known form)
    if (KnownForm(index, r) == form) {
        NoteForm(FormEvent::WriteSkipped);
        return;
    }
    NoteForm(FormEvent::WriteEmitted);
    formPending_ |= bit;
    if (form == Form::Paired) formPendingPaired_ |= bit;
    else formPendingPaired_ &= ~bit;
}

void BlockEmitter::FlushPendingForms() {
    if (inCold_ || formPending_ == 0) return;
    const uint32_t set = formPending_ & formPendingPaired_, clear = formPending_ & ~formPendingPaired_;
    if (set != 0) a.or_(PairedMask(), set);
    if (clear != 0) a.and_(PairedMask(), ~clear);
    formPending_ = formPendingPaired_ = 0;
    NoteFpr(FprEvent::FormFlush);
}

void BlockEmitter::FlushDirtyFprs() {
    if (inCold_) return;
    for (uint32_t s = 0; s < kFprCacheSlots; ++s) {
        FprSlot& sl = fprSlots_[s];
        if (sl.guest >= 0 && sl.dirty) {
            StoreFpr(static_cast<uint32_t>(sl.guest), FprCacheReg(s));
            sl.dirty = false;
            NoteFpr(FprEvent::Store);
        }
    }
    FlushPendingForms();
}

void BlockEmitter::FlushAllFprs() {
    if (inCold_) return;
    FlushDirtyFprs();
    for (FprSlot& sl : fprSlots_) sl = FprSlot{};
}

void BlockEmitter::SnapshotFprs(GprSnapshot& snap) const {
    snap.fprDirty = 0;
    for (uint32_t s = 0; s < kFprCacheSlots; ++s) {
        snap.fpr[s] = inCold_ ? int8_t{-1} : fprSlots_[s].guest;
        if (snap.fpr[s] >= 0 && fprSlots_[s].dirty) snap.fprDirty |= static_cast<uint16_t>(1u << s);
    }
    snap.formPending = inCold_ ? 0u : formPending_;
    snap.formPaired = inCold_ ? 0u : formPendingPaired_;
}

void BlockEmitter::EmitFlushFprSnapshot(const GprSnapshot& snap) {
    for (uint32_t s = 0; s < kFprCacheSlots; ++s)
        if ((snap.fprDirty & (1u << s)) != 0) StoreFpr(static_cast<uint32_t>(snap.fpr[s]), FprCacheReg(s));
    const uint32_t set = snap.formPending & snap.formPaired, clear = snap.formPending & ~snap.formPaired;
    if (set != 0) a.or_(PairedMask(), set);
    if (clear != 0) a.and_(PairedMask(), ~clear);
}

void BlockEmitter::EmitReloadFprSnapshot(const GprSnapshot& snap) {
    for (uint32_t s = 0; s < kFprCacheSlots; ++s)
        if (snap.fpr[s] >= 0) LoadFpr(FprCacheReg(s), static_cast<uint32_t>(snap.fpr[s]));
}

Xbyak::Address BlockEmitter::Const128(uint64_t lo, uint64_t hi) {
    uint32_t i = 0;
    while (i < constCount_ && (constLo_[i] != lo || constHi_[i] != hi)) ++i;
    if (i == constCount_) {
        if (constCount_ == kMaxConsts) throw Xbyak::Error(Xbyak::ERR_BAD_PARAMETER);  // the block is not compiled
        constLo_[i] = lo;
        constHi_[i] = hi;
        ++constCount_;
    }
    return a.ptr[a.rip + consts_[i]];
}

void BlockEmitter::EmitConsts() {
    if (constCount_ == 0) return;
    a.align(16);
    for (uint32_t i = 0; i < constCount_; ++i) {
        a.L(consts_[i]);
        a.dq(constLo_[i]);
        a.dq(constHi_[i]);
    }
}

// Backward: the FPRs a later op of the segment reads before writing them (segments end where AnalyseGprUses' do). A hint for
// what to keep in the cache; the generated code is correct whatever it says.
void BlockEmitter::AnalyseFprUses() {
    uint32_t live = 0;
    fprReadLater_[block.count] = 0;
    for (uint32_t i = block.count; i-- > 0;) {
        const Op& o = block.ops[i];
        fprReadLater_[i] = live;
        const uint32_t d = 1u << (o.d & 31u), a = 1u << (o.a & 31u), b = 1u << (o.b & 31u), c = 1u << (o.imm & 31u);
        switch (o.kind) {
        case K::B: case K::Bl: case K::Blr: case K::Bclr: case K::Bctr: case K::Bctrl: case K::Generic: case K::End:
            live = 0;
            fprReadLater_[i] = 0;
            break;
        case K::Lfs: case K::Lfsu: case K::Lfd: case K::Lfdu: case K::Lfsx: case K::PsqL: case K::PsqLu:
            live &= ~d;
            break;
        case K::Stfs: case K::Stfsu: case K::Stfd: case K::Stfdu: case K::Stfsx: case K::PsqSt: case K::PsqStu:
            live |= d;
            break;
        case K::Fadds: case K::Fsubs: case K::Fdivs: case K::Fadd: case K::Fsub: case K::Fdiv: case K::PsAdd: case K::PsSub:
        case K::PsMerge00: case K::PsMerge01: case K::PsMerge10: case K::PsMerge11:
            live = (live & ~d) | a | b;
            break;
        case K::Fmuls: case K::Fmul: case K::PsMul: case K::PsMuls0: case K::PsMuls1:
            live = (live & ~d) | a | c;
            break;
        case K::Fmadds: case K::Fmsubs: case K::Fnmsubs: case K::Fnmadds: case K::Fmadd: case K::Fmsub: case K::PsMadd:
        case K::PsMsub: case K::PsNmadd: case K::PsNmsub: case K::PsMadds0: case K::PsMadds1: case K::PsSum0: case K::PsSum1:
            live = (live & ~d) | a | b | c;
            break;
        case K::Fneg: case K::Fabs: case K::Fmr: case K::Frsp: case K::PsMr: case K::PsNeg: case K::PsAbs:
            live = (live & ~d) | b;
            break;
        case K::Fcmpu:
            live |= a | b;
            break;
        default:
            break;
        }
    }
}

}  // namespace PpcInterpreter::detail::x64

namespace PpcInterpreter::detail {

JitX64FprStats JitX64GetFprStats() {
    using x64::g_fpr;
    const auto get = [](const std::atomic<uint64_t>& c) { return c.load(std::memory_order_relaxed); };
    return JitX64FprStats{get(g_fpr.hits), get(g_fpr.loads), get(g_fpr.stores), get(g_fpr.formFlushes)};
}

}  // namespace PpcInterpreter::detail
