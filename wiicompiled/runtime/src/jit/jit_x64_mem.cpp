// Native code for the loads and stores (Lwz ... PsqStu) of the x86-64 backend, written against the emitter API in
// jit_x64.h (docs/jit-design.md).
//
// Every op does what its handler in ppc_interpreter.cpp (RunOp) does, in the same order of effects: a load reads, then
// writes its register, then an update form writes rA; a store that leaves the lean window's fast path (the policy byte of
// its 64 KiB chunk is set: MMIO, executable pages, anything outside the guest regions) calls the same routine the
// interpreter's FlatWriteN ends in for such a chunk, before rA is written, so an access violation it throws leaves rA as
// it was. A guest access is [r12 + (ea & kLeanMask)] (memory_access.h): loads are and, load, byte swap; stores are the
// policy test, the dirty marks (LeanTryStore's 16-bit pair), then the swapped bytes.
//
// Registers inside an op: eax holds the effective address, ecx its canonical form (ea & kLeanMask), edx the value (host
// order); esi and edi are temporaries. Slow paths are Cold code that calls a noexcept shim of this file (an exception
// is stored in frame.pending and the code exits through ExitPending) and rejoins the main path.
#include "jit_x64.h"
#include "jit_x64_generic.h"

#include <cstring>
#include <exception>

#include "brawl_dirty_memory.h"
#include "isa/ppc_isa_float.h"
#include "isa/ppc_isa_quantized.h"
#include "memory_access.h"

namespace PpcInterpreter::detail::x64 {
namespace {

using namespace Xbyak::util;
using CG = Xbyak::CodeGenerator;

#if !MKW_LEAN_MEMORY
#error "the x86-64 JIT is built for lean memory only"
#endif
static_assert(MemoryInline::kLeanPolicyShift == 16, "the policy byte index below shifts by 16");

constexpr int kPolicyDisp = static_cast<int>(MemoryInline::kLeanPolicyOffset);
constexpr int kDirtyDisp = static_cast<int>(MemoryInline::kLeanDirtyOffset);
constexpr uint32_t kMask = MemoryInline::kLeanMask;
static_assert(kMask == 0x3FFFFFFFu, "the code below assumes the 1 GiB lean window");

// --- the shims the slow paths call -----------------------------------------------------------------------------------------
// None throws: an exception is kept in the frame and the shim answers 1, so the code calls ExitPending.

uint32_t Keep(Frame* f) noexcept {
    f->pending = std::current_exception();
    return 1;
}

// What FlatWrite8/16/32 and the 64-bit float store end in when the chunk's policy byte is set (LeanStoreBody).
template <typename T>
uint32_t StoreFlagged(Frame* f, uint32_t ea, uint64_t value) noexcept {
    try {
        MemoryInline::WriteResolvedFallback<T>(ea, static_cast<T>(value));
        return 0;
    } catch (...) {
        return Keep(f);
    }
}

// Stmw: one FlatWrite32 per register, as the interpreter does (a range of words that touches a flagged chunk).
uint32_t StmwFlagged(Frame* f, uint32_t ea, uint32_t first) noexcept {
    try {
        for (uint32_t k = first; k < 32; ++k, ea += 4) MemoryInline::FlatWrite32(ea, f->ctx->gpr[k]);
        return 0;
    } catch (...) {
        return Keep(f);
    }
}

// lfs of a word that is not a normal single or a zero: the manual's conversion (PpcConvertToDoubleBitsInline).
void LfsOther(Frame* f, uint32_t raw, uint32_t d) noexcept {
    SetScalar(*f, d, PpcBitCastToDoubleInline(PpcConvertToDoubleBitsInline(raw)));
}

// Scalar(f, r) for a register in Paired form: its raw bits (the register and its form are not changed).
uint64_t ScalarBits(Frame* f, uint32_t r) noexcept { return PpcBitCastToU64Inline(Scalar(*f, r)); }

// ConvertPpcDoubleToSingleBits for the exponents 874..896.
uint32_t SubnormalSingle(uint64_t bits) noexcept {
    return MemoryInline::ConvertPpcDoubleToSubnormalSingleBits(bits, static_cast<uint32_t>((bits >> 52) & 0x7FFu));
}

// psq_l / psq_st with the quantization the GQR selects at run time. packed = d | (W << 3 | I) << 8.
uint32_t PsqLoad(Frame* f, uint32_t ea, uint32_t packed) noexcept {
    try {
        const uint32_t wi = packed >> 8;
        SetPaired(*f, packed & 0xFFu, PPC_PsqL(ea, wi >> 3, wi & 7u));
        return 0;
    } catch (...) {
        return Keep(f);
    }
}
uint32_t PsqStore(Frame* f, uint32_t ea, uint32_t packed) noexcept {
    try {
        const uint32_t wi = packed >> 8;
        PPC_PsqSt(ea, Paired(*f, packed & 0xFFu), wi >> 3, wi & 7u);
        return 0;
    } catch (...) {
        return Keep(f);
    }
}
// psq_l (W = 0, float GQR) of a pair with a signalling NaN lane: the lanes quieted as the loader does; bits as the
// swapped load leaves them (ps0 in the high word).
void PsqPairQuieted(Frame* f, uint64_t bits, uint32_t d) noexcept {
    SetPaired(*f, d, PpcBitCastToDoubleInline(PpcLoadPairPsqFloatBitsPackedInline(bits)));
}

// --- decoding the kinds ---------------------------------------------------------------------------------------------------

struct Shape {
    uint8_t width;  // bytes
    bool update, indexed, sign;
    bool zeroBase = false;  // rA = 0 in the word: the address is rB, or the displacement, alone (Exec's EaX and Ea)
    bool reverse = false;   // lwbrx, lhbrx, stwbrx, sthbrx: the bytes as they lie in memory, not swapped
};

bool IntLoadShape(K k, Shape& s) {
    switch (k) {
    case K::Lwz: s = {4, false, false, false}; return true;
    case K::Lwzu: s = {4, true, false, false}; return true;
    case K::Lbz: s = {1, false, false, false}; return true;
    case K::Lbzu: s = {1, true, false, false}; return true;
    case K::Lhz: s = {2, false, false, false}; return true;
    case K::Lhzu: s = {2, true, false, false}; return true;
    case K::Lha: s = {2, false, false, true}; return true;
    case K::Lwzx: s = {4, false, true, false}; return true;
    case K::Lbzx: s = {1, false, true, false}; return true;
    case K::Lhzx: s = {2, false, true, false}; return true;
    default: return false;
    }
}

bool IntStoreShape(K k, Shape& s) {
    switch (k) {
    case K::Stw: s = {4, false, false, false}; return true;
    case K::Stwu: s = {4, true, false, false}; return true;
    case K::Stb: s = {1, false, false, false}; return true;
    case K::Stbu: s = {1, true, false, false}; return true;
    case K::Sth: s = {2, false, false, false}; return true;
    case K::Sthu: s = {2, true, false, false}; return true;
    case K::Stwx: s = {4, false, true, false}; return true;
    case K::Stbx: s = {1, false, true, false}; return true;
    case K::Sthx: s = {2, false, true, false}; return true;
    default: return false;
    }
}

// --- emission helpers -----------------------------------------------------------------------------------------------------

// eax = the effective address: r[a] + imm, or r[a] + r[b] (32-bit wrap).
void EmitEa(BlockEmitter& e, const Op& o, bool indexed, bool zeroBase = false) {
    Xbyak::CodeGenerator& a = e.a;
    if (zeroBase) {
        if (indexed) {
            if (e.WantCacheGpr(o.b)) a.mov(eax, e.ReadGpr(o.b));
            else a.mov(eax, e.Gpr(o.b));
        } else {
            a.mov(eax, o.imm);
        }
        return;
    }
    if (e.WantCacheGpr(o.a)) {
        const Xbyak::Reg32& ra = e.ReadGpr(o.a);
        if (indexed) {
            if (e.WantCacheGpr(o.b)) {
                const Xbyak::Reg32& rb = e.ReadGpr(o.b, 1u << o.a);
                a.lea(eax, a.ptr[ra.cvt64() + rb.cvt64()]);
            } else {
                a.mov(eax, ra);
                a.add(eax, e.Gpr(o.b));
            }
        } else if (o.imm != 0) {
            a.lea(eax, a.ptr[ra.cvt64() + static_cast<int32_t>(o.imm)]);
        } else {
            a.mov(eax, ra);
        }
    } else {
        a.mov(eax, e.Gpr(o.a));
        if (indexed) {
            if (e.WantCacheGpr(o.b)) a.add(eax, e.ReadGpr(o.b));
            else a.add(eax, e.Gpr(o.b));
        } else if (o.imm != 0) {
            a.add(eax, o.imm);
        }
    }
}

// ecx = the canonical offset of the address in eax.
void EmitCanonical(BlockEmitter& e) {
    e.a.mov(ecx, eax);
    e.a.and_(ecx, kMask);
}

// ecx = the canonical offset of the effective address, for an op that does not need the address itself afterwards (not an
// update form, and no slow path that passes it on): one register less to copy.
void EmitCanonicalEa(BlockEmitter& e, const Op& o, bool indexed, bool zeroBase = false) {
    Xbyak::CodeGenerator& a = e.a;
    if (zeroBase) {
        if (indexed) {
            if (e.WantCacheGpr(o.b)) a.mov(ecx, e.ReadGpr(o.b));
            else a.mov(ecx, e.Gpr(o.b));
        } else {
            a.mov(ecx, o.imm);
        }
        a.and_(ecx, kMask);
        return;
    }
    if (e.WantCacheGpr(o.a)) {
        const Xbyak::Reg32& ra = e.ReadGpr(o.a);
        if (indexed) {
            if (e.WantCacheGpr(o.b)) {
                const Xbyak::Reg32& rb = e.ReadGpr(o.b, 1u << o.a);
                a.lea(ecx, a.ptr[ra.cvt64() + rb.cvt64()]);
            } else {
                a.mov(ecx, ra);
                a.add(ecx, e.Gpr(o.b));
            }
        } else if (o.imm != 0) {
            a.lea(ecx, a.ptr[ra.cvt64() + static_cast<int32_t>(o.imm)]);
        } else {
            a.mov(ecx, ra);
        }
    } else {
        a.mov(ecx, e.Gpr(o.a));
        if (indexed) {
            if (e.WantCacheGpr(o.b)) a.add(ecx, e.ReadGpr(o.b));
            else a.add(ecx, e.Gpr(o.b));
        } else if (o.imm != 0) {
            a.add(ecx, o.imm);
        }
    }
    a.and_(ecx, kMask);
}

// An update form's write of rA (the address is in eax; a slow path that clobbered it rebuilds it first from memory).
void EmitRebuildEa(Xbyak::CodeGenerator& a, BlockEmitter& e, uint32_t ra, uint32_t imm, bool indexed = false, uint32_t rb = 0) {
    a.mov(eax, e.Gpr(ra));
    if (indexed) a.add(eax, e.Gpr(rb));
    else if (imm != 0) a.add(eax, imm);
}

// The byte-swapped load of `width` bytes at [r12 + rcx] into `dst` (rdx for 8), zero-extended.
void EmitLoadSwappedTo(Xbyak::CodeGenerator& a, unsigned width, const Xbyak::Reg32& dst, bool reverse = false) {
    const Xbyak::RegExp at = kLean + rcx;
    switch (width) {
    case 1: a.movzx(dst, a.byte[at]); break;
    case 2:
        a.movzx(dst, a.word[at]);
        if (!reverse) a.rol(dst.cvt16(), 8);
        break;
    case 4:
        if (reverse) {  // lwbrx: the bytes as memory holds them
            a.mov(dst, a.dword[at]);
            break;
        }
#if defined(__MOVBE__)
        a.movbe(dst, a.dword[at]);
#else
        a.mov(dst, a.dword[at]);
        a.bswap(dst);
#endif
        break;
    default:
#if defined(__MOVBE__)
        a.movbe(dst.cvt64(), a.qword[at]);
#else
        a.mov(dst.cvt64(), a.qword[at]);
        a.bswap(dst.cvt64());
#endif
        break;
    }
}

void EmitLoadSwapped(Xbyak::CodeGenerator& a, unsigned width, bool reverse = false) {
    EmitLoadSwappedTo(a, width, edx, reverse);
}

// --- forward jumps to cold code ----------------------------------------------------------------------------------------
// The emitter's label pool (BlockEmitter::kMaxLabels, shared by the groups) is small next to the number of slow paths a
// block of loads and stores has, so this file does not spend labels: a forward jump to cold code is written with a
// zero displacement and remembers where, the cold code patches it when it is emitted (it is emitted into the same,
// writable buffer), and a jump back from cold code goes to an offset noted in the main path. Offsets are from the
// start of the block's code.

struct Patch {
    uint32_t at;  // offset of the 32-bit displacement
};

enum Cc : uint8_t { kB = 0x2, kE = 0x4, kNe = 0x5, kBe = 0x6, kA = 0x7 };

Patch EmitJcc(Xbyak::CodeGenerator& a, Cc cc) {
    a.db(0x0F);
    a.db(0x80 | cc);
    const Patch p{static_cast<uint32_t>(a.getSize())};
    a.dd(0);
    return p;
}

// The jump written at p goes to here.
void Resolve(Xbyak::CodeGenerator& a, Patch p) {
    uint8_t* const base = const_cast<uint8_t*>(a.getCode());
    const int32_t rel = static_cast<int32_t>(a.getSize() - (p.at + 4));
    std::memcpy(base + p.at, &rel, sizeof(rel));
}

uint32_t Here(Xbyak::CodeGenerator& a) { return static_cast<uint32_t>(a.getSize()); }
void JmpTo(Xbyak::CodeGenerator& a, uint32_t offset) { a.jmp(static_cast<const void*>(a.getCode() + offset), CG::T_NEAR); }

// Up to a few jumps to one cold entry.
struct Patches {
    static constexpr unsigned kMax = 8;
    uint32_t at[kMax];
    uint8_t n;
    void Add(Patch p) { at[n++] = p.at; }
    void Resolve(Xbyak::CodeGenerator& a) const {
        for (unsigned i = 0; i < n; ++i) x64::Resolve(a, Patch{at[i]});
    }
};

// After a shim that answers 1 on an exception: leave the block with it pending.
void EmitPendingCheck(BlockEmitter& e) {
    Xbyak::Label ok;
    e.a.test(eax, eax);
    e.a.jz(ok);
    e.ExitPending();
    e.a.L(ok);
}

using StoreShim = uint32_t (*)(Frame*, uint32_t, uint64_t);
StoreShim ShimFor(unsigned width) {
    switch (width) {
    case 1: return &StoreFlagged<uint8_t>;
    case 2: return &StoreFlagged<uint16_t>;
    case 4: return &StoreFlagged<uint32_t>;
    default: return &StoreFlagged<uint64_t>;
    }
}

// The slow path of a store whose address is in eax and value in rdx: the flagged chunk's routine, then back to the
// join point (an update form rebuilds the address first: the call clobbered it).
struct StoreCold {
    Patch slow;
    uint32_t join, pcIndex, imm;
    uint8_t ra, rb, width;
    bool update, indexed, reverse;
    BlockEmitter::GprSnapshot snap;
    void operator()(BlockEmitter& e) const {
        Xbyak::CodeGenerator& a = e.a;
        Resolve(a, slow);
        e.EmitFlushSnapshot(snap);
        e.StorePc(pcIndex);
        if (reverse) {  // the shim writes big-endian: hand it the value swapped, so memory gets the register's own byte order
            if (width == 4) a.bswap(edx);
            else if (width == 2) a.rol(dx, 8);
        }
        e.CallHelper(ShimFor(width), kFrame, eax, rdx);
        EmitPendingCheck(e);
        e.EmitReloadSnapshotVolatile(snap);
        if (update) EmitRebuildEa(a, e, ra, imm, indexed, rb);
        JmpTo(a, join);
    }
};

// The policy test, the dirty marks and the swapped store of the value in edx / rdx, for the address in eax, ecx = its
// canonical form. A set policy byte jumps to the returned patch (the cold path) before anything is written.
Patch EmitStoreChecked(BlockEmitter& e, unsigned width, bool reverse = false) {
    Xbyak::CodeGenerator& a = e.a;
    a.mov(esi, ecx);
    a.shr(esi, MemoryInline::kLeanPolicyShift);
    a.cmp(a.byte[kLean + rsi + kPolicyDisp], 0);
    const Patch slow = EmitJcc(a, kNe);
#if MKW_LEAN_DIRTY_MARKS
    a.mov(esi, ecx);
    a.shr(esi, 6);
    a.mov(a.word[kLean + rsi + kDirtyDisp], 0x0101);
#endif
    const Xbyak::RegExp at = kLean + rcx;
    switch (width) {
    case 1: a.mov(a.byte[at], dl); break;
    case 2:
        if (!reverse) a.rol(dx, 8);
        a.mov(a.word[at], dx);
        break;
    case 4:
        if (reverse) {  // stwbrx: the register's bytes as they are
            a.mov(a.dword[at], edx);
            break;
        }
#if defined(__MOVBE__)
        a.movbe(a.dword[at], edx);
#else
        a.bswap(edx);
        a.mov(a.dword[at], edx);
#endif
        break;
    default:
#if defined(__MOVBE__)
        a.movbe(a.qword[at], rdx);
#else
        a.bswap(rdx);
        a.mov(a.qword[at], rdx);
#endif
        break;
    }
    return slow;
}

// --- the integer ops ------------------------------------------------------------------------------------------------------

uint32_t EmitIntLoad(BlockEmitter& e, const Op& o, const Shape& s) {
    Xbyak::CodeGenerator& a = e.a;
    if (s.update) {
        EmitEa(e, o, s.indexed);
        EmitCanonical(e);
    } else {
        EmitCanonicalEa(e, o, s.indexed, s.zeroBase);
    }
    if (e.GprReadLater(o.d)) {
        const Xbyak::Reg32& dst = e.WriteGpr(o.d);
        EmitLoadSwappedTo(a, s.width, dst, s.reverse);
        if (s.sign) a.movsx(dst, dst.cvt16());
    } else {
        e.DropGpr(o.d);
        EmitLoadSwappedTo(a, s.width, edx, s.reverse);
        if (s.sign) a.movsx(edx, dx);
        a.mov(e.Gpr(o.d), edx);
    }
    if (s.update) {
        if (e.GprReadLater(o.a)) a.mov(e.WriteGpr(o.a, 1u << o.d), eax);
        else { e.DropGpr(o.a); a.mov(e.Gpr(o.a), eax); }
    }
    return 1;
}

uint32_t EmitIntStore(BlockEmitter& e, const Op& o, const Shape& s, bool fromFpr = false) {
    Xbyak::CodeGenerator& a = e.a;
    EmitEa(e, o, s.indexed, s.zeroBase);
    // after the address: the address does not read rS, an update form's rA is written last. stfiwx stores the low word of the
    // register's bits whatever its form
    if (fromFpr) a.mov(edx, e.FprWord(o.d, 0));
    else if (e.WantCacheGpr(o.d)) a.mov(edx, e.ReadGpr(o.d));
    else a.mov(edx, e.Gpr(o.d));
    EmitCanonical(e);
    const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
    const Patch slow = EmitStoreChecked(e, s.width, s.reverse);
    const uint32_t join = Here(a);
    if (s.update) {
        if (e.GprReadLater(o.a)) a.mov(e.WriteGpr(o.a), eax);
        else { e.DropGpr(o.a); a.mov(e.Gpr(o.a), eax); }
    }
    e.Cold(StoreCold{slow, join, e.index, o.imm, o.a, o.b, s.width, s.update, s.indexed, s.reverse, snap});
    return 1;
}

// lmw: one load per register (unrolled), each as FlatRead32 at ea + 4 * k. The words lie in the lean window one after
// the other (the last 128 bytes below the window's top are unbacked, the first word faults there as in the interpreter).
uint32_t EmitLmw(BlockEmitter& e, const Op& o) {
    Xbyak::CodeGenerator& a = e.a;
    EmitEa(e, o, false);
    EmitCanonical(e);
    e.FlushAllGprs();
    for (uint32_t k = o.d; k < 32; ++k) {
        const Xbyak::RegExp at = kLean + rcx + static_cast<int>(4 * (k - o.d));
#if defined(__MOVBE__)
        a.movbe(edx, a.dword[at]);
#else
        a.mov(edx, a.dword[at]);
        a.bswap(edx);
#endif
        a.mov(e.Gpr(k), edx);
    }
    return 1;
}

// stmw: the interpreter does one FlatWrite32 per register. The words are 4 bytes apart and cover at most 128 bytes, so
// they start in the chunk of the first word or the one of the last (a policy byte per 64 KiB: test those two) and
// the pairs of dirty marks they set are the union of the pair at the first word's block and the pair at the last word's
// (at most 2 blocks apart). If either chunk is flagged, a shim runs the whole per-word loop as the interpreter does.
struct StmwCold {
    Patches slow;
    uint32_t join, pcIndex, first;
    BlockEmitter::GprSnapshot snap;
    void operator()(BlockEmitter& e) const {
        slow.Resolve(e.a);
        e.StorePc(pcIndex);
        e.CallHelper(&StmwFlagged, kFrame, eax, first);
        EmitPendingCheck(e);
        e.EmitReloadSnapshotVolatile(snap);
        JmpTo(e.a, join);
    }
};

uint32_t EmitStmw(BlockEmitter& e, const Op& o) {
    Xbyak::CodeGenerator& a = e.a;
    const uint32_t n = 32 - o.d;
    StmwCold cold{};
    EmitEa(e, o, false);
    e.FlushDirtyGprs();
    cold.snap = e.SnapshotGprs();
    EmitCanonical(e);
    a.mov(esi, ecx);
    a.shr(esi, MemoryInline::kLeanPolicyShift);
    a.cmp(a.byte[kLean + rsi + kPolicyDisp], 0);
    cold.slow.Add(EmitJcc(a, kNe));
    if (n > 1) {
        a.lea(edx, a.ptr[rax + static_cast<int>(4 * (n - 1))]);
        a.and_(edx, kMask);
        a.mov(esi, edx);
        a.shr(esi, MemoryInline::kLeanPolicyShift);
        a.cmp(a.byte[kLean + rsi + kPolicyDisp], 0);
        cold.slow.Add(EmitJcc(a, kNe));
    }
#if MKW_LEAN_DIRTY_MARKS
    a.mov(esi, ecx);
    a.shr(esi, 6);
    a.mov(a.word[kLean + rsi + kDirtyDisp], 0x0101);
    if (n > 1) {
        a.shr(edx, 6);
        a.mov(a.word[kLean + rdx + kDirtyDisp], 0x0101);
    }
#endif
    for (uint32_t k = o.d; k < 32; ++k) {
        const Xbyak::RegExp at = kLean + rcx + static_cast<int>(4 * (k - o.d));
        a.mov(edx, e.Gpr(k));
#if defined(__MOVBE__)
        a.movbe(a.dword[at], edx);
#else
        a.bswap(edx);
        a.mov(a.dword[at], edx);
#endif
    }
    cold.join = Here(a);
    cold.pcIndex = e.index;
    cold.first = o.d;
    e.Cold(cold);
    return 1;
}

// --- floating point loads and stores --------------------------------------------------------------------------------------

// Form bit of r := form, unless the block knows it holds it already (jit_x64_forms.cpp).
void EmitSetForm(BlockEmitter& e, uint32_t r, Form form) {
    if (e.KnownForm(e.index, r) == form) {
        BlockEmitter::NoteForm(BlockEmitter::FormEvent::WriteSkipped);
        return;
    }
    BlockEmitter::NoteForm(BlockEmitter::FormEvent::WriteEmitted);
    const uint32_t bit = 1u << (r & 31u);
    if (form == Form::Paired) e.a.or_(e.PairedMask(), bit);
    else e.a.and_(e.PairedMask(), ~bit);
}
void EmitSetScalarForm(BlockEmitter& e, uint32_t r) { EmitSetForm(e, r, Form::Scalar); }

// lfs: a normal single or a zero widens exactly with cvtss2sd (PpcConvertToDoubleBitsInline); denormals, infinities and
// NaNs (the word is in edx) go to the manual's algorithm in a shim and rejoin.
struct LfsCold {
    Patch other;
    uint32_t join, imm;
    uint8_t d, ra, rb;
    bool update, indexed;
    BlockEmitter::GprSnapshot snap;
    void operator()(BlockEmitter& e) const {
        Resolve(e.a, other);
        e.EmitFlushSnapshot(snap);
        e.CallHelper(&LfsOther, kFrame, edx, static_cast<uint32_t>(d));
        e.EmitReloadSnapshotVolatile(snap);
        if (update) EmitRebuildEa(e.a, e, ra, imm, indexed, rb);
        JmpTo(e.a, join);
    }
};

uint32_t EmitLfs(BlockEmitter& e, const Op& o, bool update, bool indexed, bool zeroBase = false) {
    Xbyak::CodeGenerator& a = e.a;
    EmitEa(e, o, indexed, zeroBase);
    EmitCanonical(e);
    EmitLoadSwapped(a, 4);
    const BlockEmitter::GprSnapshot snap = e.SnapshotGprs();
    Patch other{};
#if !MKW_FAST_FLOAT
    a.mov(esi, edx);
    a.and_(esi, 0x7FFFFFFF);
    a.sub(esi, 0x00800000);
    a.cmp(esi, 0x7F000000);
    Xbyak::Label normal;
    a.jb(normal);
    a.test(edx, 0x7FFFFFFF);
    other = EmitJcc(a, kNe);  // neither a normal single nor a zero
    a.L(normal);
#endif
    a.movd(xmm0, edx);
    a.cvtss2sd(xmm0, xmm0);
    a.movq(e.Fpr(o.d), xmm0);
    EmitSetScalarForm(e, o.d);
    const uint32_t join = Here(a);
    if (update) a.mov(e.WriteGpr(o.a), eax);
#if !MKW_FAST_FLOAT
    e.Cold(LfsCold{other, join, o.imm, o.d, o.a, o.b, update, indexed, snap});
#else
    (void)snap;
#endif
    return 1;
}

uint32_t EmitLfd(BlockEmitter& e, const Op& o, bool update, bool indexed = false, bool zeroBase = false) {
    Xbyak::CodeGenerator& a = e.a;
    EmitEa(e, o, indexed, zeroBase);
    EmitCanonical(e);
    EmitLoadSwapped(a, 8);  // FlatReadFloat64 under lean memory is FlatLoad<uint64_t>
    a.mov(e.Fpr(o.d), rdx);
    EmitSetScalarForm(e, o.d);
    if (update) a.mov(e.WriteGpr(o.a), eax);
    return 1;
}

// stfs / stfd: the register read as a scalar (a Paired register through ScalarBits in a shim), stfs's narrowing in
// integer ops (the subnormal-single window in a shim), then the checked store.
struct StfCold {
    Patch paired, subnormal, slow;
    uint32_t haveValue, haveBits, join, pcIndex, imm;
    uint8_t d, ra, rb, width;
    bool update, hasSubnormal, indexed, havePaired;
    BlockEmitter::GprSnapshot snap;
    void operator()(BlockEmitter& e) const {
        Xbyak::CodeGenerator& a = e.a;
        if (havePaired) {
            Resolve(a, paired);
            e.EmitFlushSnapshot(snap);
            e.CallHelper(&ScalarBits, kFrame, static_cast<uint32_t>(d));
            e.EmitReloadSnapshotVolatile(snap);
            a.mov(rdx, rax);
            JmpTo(a, haveValue);
        }
        if (hasSubnormal) {
            Resolve(a, subnormal);
            e.EmitFlushSnapshot(snap);
            e.CallHelper(&SubnormalSingle, rdx);
            e.EmitReloadSnapshotVolatile(snap);
            a.mov(edx, eax);
            JmpTo(a, haveBits);
        }
        Resolve(a, slow);
        e.EmitFlushSnapshot(snap);
        e.StorePc(pcIndex);
        e.CallHelper(ShimFor(width), kFrame, eax, rdx);
        EmitPendingCheck(e);
        e.EmitReloadSnapshotVolatile(snap);
        if (update) EmitRebuildEa(a, e, ra, imm, indexed, rb);
        JmpTo(a, join);
    }
};

uint32_t EmitStf(BlockEmitter& e, const Op& o, bool single, bool update, bool indexed, bool zeroBase = false) {
    Xbyak::CodeGenerator& a = e.a;
    StfCold cold{};
    const auto prewarmEa = [&]() {
        if (!zeroBase) {
            if (e.WantCacheGpr(o.a)) (void)e.ReadGpr(o.a);
            if (indexed && e.WantCacheGpr(o.b)) (void)e.ReadGpr(o.b, 1u << o.a);
        } else if (indexed && e.WantCacheGpr(o.b)) {
            (void)e.ReadGpr(o.b);
        }
    };
    const Form known = e.KnownForm(e.index, o.d);
    if (known == Form::Paired) {  // no test: ps0 widened, in line
        BlockEmitter::NoteForm(BlockEmitter::FormEvent::ConversionForced);
        e.FlushVolatileGprs();
        e.CallHelper(&ScalarBits, kFrame, static_cast<uint32_t>(o.d));
        a.mov(rdx, rax);
        prewarmEa();
        cold.snap = e.SnapshotGprs();
    } else {
        prewarmEa();
        cold.snap = e.SnapshotGprs();
        a.mov(rdx, e.Fpr(o.d));
        if (known == Form::Scalar) {  // no test: the register holds a double
            BlockEmitter::NoteForm(BlockEmitter::FormEvent::TestSkipped);
        } else {
            BlockEmitter::NoteForm(BlockEmitter::FormEvent::TestEmitted);
            a.test(e.PairedMask(), 1u << (o.d & 31u));
            cold.paired = EmitJcc(a, kNe);
            cold.havePaired = true;
        }
    }
    cold.haveValue = Here(a);
    if (single) {
#if MKW_FAST_FLOAT
        a.movq(xmm0, rdx);
        a.cvtsd2ss(xmm0, xmm0);
        a.movd(edx, xmm0);
#else
        cold.hasSubnormal = true;
        a.mov(rcx, rdx);
        a.shr(rcx, 52);
        a.and_(ecx, 0x7FF);
        a.sub(ecx, 874);
        a.cmp(ecx, 22);
        cold.subnormal = EmitJcc(a, kBe);
        a.mov(rsi, rdx);
        a.shr(rsi, 32);
        a.and_(esi, 0xC0000000u);
        a.shr(rdx, 29);
        a.and_(edx, 0x3FFFFFFF);
        a.or_(edx, esi);
        cold.haveBits = Here(a);
#endif
    }
    EmitEa(e, o, indexed, zeroBase);
    EmitCanonical(e);
    cold.width = single ? 4 : 8;
    cold.slow = EmitStoreChecked(e, cold.width);
    cold.join = Here(a);
    if (update) a.mov(e.WriteGpr(o.a), eax);
    cold.pcIndex = e.index;
    cold.imm = o.imm;
    cold.d = o.d;
    cold.ra = o.a;
    cold.rb = o.b;
    cold.update = update;
    cold.indexed = indexed;
    e.Cold(cold);
    return 1;
}

// --- quantized loads and stores -------------------------------------------------------------------------------------------

// psq_l: the float GQR (the test PPC_PsqLInline makes) is native; any other state calls PPC_PsqL in a shim.
struct PsqLCold {
    Patch generic, quiet0, quiet1;
    uint32_t join, pcIndex, imm;
    uint8_t d, b, ra;
    bool update, hasQuiet;
    BlockEmitter::GprSnapshot snap;
    void operator()(BlockEmitter& e) const {
        Xbyak::CodeGenerator& a = e.a;
        if (hasQuiet) {
            Resolve(a, quiet0);
            Resolve(a, quiet1);
            e.EmitFlushSnapshot(snap);
            e.CallHelper(&PsqPairQuieted, kFrame, rdx, static_cast<uint32_t>(d));
            e.EmitReloadSnapshotVolatile(snap);
            if (update) EmitRebuildEa(a, e, ra, imm);
            JmpTo(a, join);
        }
        Resolve(a, generic);
        e.EmitFlushSnapshot(snap);
        e.StorePc(pcIndex);
        e.CallHelper(&PsqLoad, kFrame, eax, static_cast<uint32_t>(d) | (static_cast<uint32_t>(b) << 8));
        EmitPendingCheck(e);
        e.EmitReloadSnapshotVolatile(snap);
        if (update) EmitRebuildEa(a, e, ra, imm);
        JmpTo(a, join);
    }
};

uint32_t EmitPsqL(BlockEmitter& e, const Op& o, bool update) {
    Xbyak::CodeGenerator& a = e.a;
    const uint32_t w = o.b >> 3, i = o.b & 7u;
    PsqLCold cold{};
    EmitEa(e, o, false);
    cold.snap = e.SnapshotGprs();
    if (w == 0) a.test(e.Gqr(i), 0xFFFF0000u);  // type 0 and load scale 0
    else a.test(e.Gqr(i), 0x00070000u);         // type 0
    cold.generic = EmitJcc(a, kNe);
    EmitCanonical(e);
    if (w == 0) {
        EmitLoadSwapped(a, 8);  // ps0 in the high word, ps1 in the low
#if !MKW_FAST_FLOAT
        // a signalling NaN lane is quieted by the loader: that is the shim's
        cold.hasQuiet = true;
        a.mov(esi, edx);
        a.and_(esi, 0x7FFFFFFF);
        a.cmp(esi, 0x7F800000);
        cold.quiet0 = EmitJcc(a, kA);
        a.mov(rsi, rdx);
        a.shr(rsi, 32);
        a.and_(esi, 0x7FFFFFFF);
        a.cmp(esi, 0x7F800000);
        cold.quiet1 = EmitJcc(a, kA);
#endif
    } else {
        // PpcLoadSinglePsqFloatFastInline: ps0 = the word as it is, ps1 = 1.0f
        EmitLoadSwapped(a, 4);
        a.shl(rdx, 32);
        a.mov(esi, 0x3F800000);
        a.or_(rdx, rsi);
    }
    a.mov(e.Fpr(o.d), rdx);
    EmitSetForm(e, o.d, Form::Paired);
    cold.join = Here(a);
    if (update) a.mov(e.WriteGpr(o.a), eax);
    cold.pcIndex = e.index;
    cold.imm = o.imm;
    cold.d = o.d;
    cold.b = o.b;
    cold.ra = o.a;
    cold.update = update;
    e.Cold(cold);
    return 1;
}

// psq_st: native for a paired float store (W = 0, the GQR's store halfword 0) of a register that is not in Scalar form
// and whose lanes need no flushing or quieting, and for no other case: everything else calls PPC_PsqSt in a shim, which
// does what the interpreter does (policy, marks, MMIO, the quantizers).
struct PsqStCold {
    Patches generic;
    uint32_t join, pcIndex, imm;
    uint8_t d, b, ra;
    bool update;
    BlockEmitter::GprSnapshot snap;
    void operator()(BlockEmitter& e) const {
        Xbyak::CodeGenerator& a = e.a;
        generic.Resolve(a);
        e.EmitFlushSnapshot(snap);
        e.StorePc(pcIndex);
        e.CallHelper(&PsqStore, kFrame, eax, static_cast<uint32_t>(d) | (static_cast<uint32_t>(b) << 8));
        EmitPendingCheck(e);
        e.EmitReloadSnapshotVolatile(snap);
        if (update) EmitRebuildEa(a, e, ra, imm);
        JmpTo(a, join);
    }
};

uint32_t EmitPsqSt(BlockEmitter& e, const Op& o, bool update) {
    Xbyak::CodeGenerator& a = e.a;
    const uint32_t w = o.b >> 3, i = o.b & 7u;
    EmitEa(e, o, false);
    PsqStCold cold{};
    cold.snap = e.SnapshotGprs();
    if (w == 0) a.test(e.Gqr(i), 0x0000FFFFu);  // paired float: store type 0 and store scale 0
    else a.test(e.Gqr(i), 0x00000007u);         // single float: store type 0 (ignores store scale)
    cold.generic.Add(EmitJcc(a, kNe));
    if (e.KnownForm(e.index, o.d) == Form::Paired) {
        BlockEmitter::NoteForm(BlockEmitter::FormEvent::TestSkipped);
    } else {
        BlockEmitter::NoteForm(BlockEmitter::FormEvent::TestEmitted);
        a.test(e.PairedMask(), 1u << (o.d & 31u));
        cold.generic.Add(EmitJcc(a, kE));
    }
    if (w == 0) a.mov(rdx, e.Fpr(o.d));
    else a.mov(edx, e.FprWord(o.d, 1));  // ps0 sits in the high word of the paired FPR
#if !MKW_FAST_FLOAT
    // The store keeps a lane unless it is a subnormal (flushed to a signed zero) or a NaN (quieted): those, and only
    // those, are the shim's. Per lane, m = bits & 0x7FFFFFFF: a normal, an infinity or a zero is kept as it is.
    const int lanes = w == 0 ? 2 : 1;
    for (int lane = 0; lane < lanes; ++lane) {
        Xbyak::Label keep;
        if (lane == 0) {
            a.mov(esi, edx);
        } else {
            a.mov(rsi, rdx);
            a.shr(rsi, 32);
        }
        a.and_(esi, 0x7FFFFFFF);
        a.lea(edi, a.ptr[rsi - 0x00800000]);
        a.cmp(edi, 0x7F000000);
        a.jbe(keep);
        a.test(esi, esi);
        cold.generic.Add(EmitJcc(a, kNe));
        a.L(keep);
    }
#endif
    EmitCanonical(e);
    if (w == 0) {
        // both ends of the 8 bytes: a pair can straddle two chunks (LeanStoreNeedsPolicy(addr) and (addr + 7))
        a.mov(esi, ecx);
        a.shr(esi, MemoryInline::kLeanPolicyShift);
        a.cmp(a.byte[kLean + rsi + kPolicyDisp], 0);
        cold.generic.Add(EmitJcc(a, kNe));
        a.lea(edi, a.ptr[rax + 7]);
        a.and_(edi, kMask);
        a.shr(edi, MemoryInline::kLeanPolicyShift);
        a.cmp(a.byte[kLean + rdi + kPolicyDisp], 0);
        cold.generic.Add(EmitJcc(a, kNe));
#if MKW_LEAN_DIRTY_MARKS
        a.mov(esi, ecx);
        a.shr(esi, 6);
        a.mov(a.word[kLean + rsi + kDirtyDisp], 0x0101);
#endif
        // the packed register byte-swapped: ps0 (the high word) lands at the lower address, big-endian
#if defined(__MOVBE__)
        a.movbe(a.qword[kLean + rcx], rdx);
#else
        a.bswap(rdx);
        a.mov(a.qword[kLean + rcx], rdx);
#endif
    } else {
        cold.generic.Add(EmitStoreChecked(e, 4));
    }
    cold.join = Here(a);
    if (update) a.mov(e.WriteGpr(o.a), eax);
    cold.pcIndex = e.index;
    cold.imm = o.imm;
    cold.d = o.d;
    cold.b = o.b;
    cold.ra = o.a;
    cold.update = update;
    e.Cold(cold);
    return 1;
}

uint32_t EmitMemOp(BlockEmitter& e, uint32_t index) {
    const Op& o = e.block.ops[index];
    Shape s;
    if (IntLoadShape(o.kind, s)) return EmitIntLoad(e, o, s);
    if (IntStoreShape(o.kind, s)) return EmitIntStore(e, o, s);
    switch (o.kind) {
    case K::Lmw: return EmitLmw(e, o);
    case K::Stmw: return EmitStmw(e, o);
    case K::Lfs: return EmitLfs(e, o, false, false);
    case K::Lfsu: return EmitLfs(e, o, true, false);
    case K::Lfsx: return EmitLfs(e, o, false, true);
    case K::Lfd: return EmitLfd(e, o, false);
    case K::Lfdu: return EmitLfd(e, o, true);
    case K::Stfs: return EmitStf(e, o, true, false, false);
    case K::Stfsu: return EmitStf(e, o, true, true, false);
    case K::Stfsx: return EmitStf(e, o, true, false, true);
    case K::Stfd: return EmitStf(e, o, false, false, false);
    case K::Stfdu: return EmitStf(e, o, false, true, false);
    case K::PsqL: return EmitPsqL(e, o, false);
    case K::PsqLu: return EmitPsqL(e, o, true);
    case K::PsqSt: return EmitPsqSt(e, o, false);
    case K::PsqStu: return EmitPsqSt(e, o, true);
    default: return 0;
    }
}

}  // namespace

uint32_t EmitMem(BlockEmitter& e, uint32_t index) { return EmitMemOp(e, index); }

// The loads and stores Decode leaves Generic: rA = 0 (an absolute address), the update, byte-reversed, algebraic and float forms
// without a kind of their own, lwarx, stwcx. and stfiwx. Each does what Exec does for the word; 0 for any other word and for a
// form Exec handles in a way that is not worth native code (an update form with rA = 0).
uint32_t EmitMemGeneric(BlockEmitter& e, uint32_t w) {
    const uint32_t po = w >> 26, d = (w >> 21) & 31, a = (w >> 16) & 31, b = (w >> 11) & 31, xo = (w >> 1) & 0x3FF;
    Op o{K::Generic, static_cast<uint8_t>(d), static_cast<uint8_t>(a), static_cast<uint8_t>(b),
         static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(w & 0xFFFFu)))};
    Shape s{};
    enum class Kind { None, IntLoad, IntStore, Lfs, Lfd, Stfs, Stfd } kind = Kind::None;
    bool stfiwx = false, stwcx = false, update = false, indexed = false;
    if (po == 31) {
        indexed = true;
        switch (xo) {
        case 20: case 23: s = {4, false, true, false}; kind = Kind::IntLoad; break;       // lwarx, lwzx
        case 55: s = {4, true, true, false}; kind = Kind::IntLoad; break;                 // lwzux
        case 87: s = {1, false, true, false}; kind = Kind::IntLoad; break;                // lbzx
        case 119: s = {1, true, true, false}; kind = Kind::IntLoad; break;                // lbzux
        case 279: s = {2, false, true, false}; kind = Kind::IntLoad; break;               // lhzx
        case 311: s = {2, true, true, false}; kind = Kind::IntLoad; break;                // lhzux
        case 343: s = {2, false, true, true}; kind = Kind::IntLoad; break;                // lhax
        case 375: s = {2, true, true, true}; kind = Kind::IntLoad; break;                 // lhaux
        case 534: s = {4, false, true, false, false, true}; kind = Kind::IntLoad; break;  // lwbrx
        case 790: s = {2, false, true, false, false, true}; kind = Kind::IntLoad; break;  // lhbrx
        case 151: s = {4, false, true, false}; kind = Kind::IntStore; break;              // stwx
        case 183: s = {4, true, true, false}; kind = Kind::IntStore; break;               // stwux
        case 215: s = {1, false, true, false}; kind = Kind::IntStore; break;              // stbx
        case 247: s = {1, true, true, false}; kind = Kind::IntStore; break;               // stbux
        case 407: s = {2, false, true, false}; kind = Kind::IntStore; break;              // sthx
        case 439: s = {2, true, true, false}; kind = Kind::IntStore; break;               // sthux
        case 662: s = {4, false, true, false, false, true}; kind = Kind::IntStore; break; // stwbrx
        case 918: s = {2, false, true, false, false, true}; kind = Kind::IntStore; break; // sthbrx
        case 150: s = {4, false, true, false}; kind = Kind::IntStore; stwcx = true; break;  // stwcx.
        case 983: s = {4, false, true, false}; kind = Kind::IntStore; stfiwx = true; break;
        case 535: kind = Kind::Lfs; break;
        case 567: kind = Kind::Lfs; update = true; break;
        case 599: kind = Kind::Lfd; break;
        case 631: kind = Kind::Lfd; update = true; break;
        case 663: kind = Kind::Stfs; break;
        case 695: kind = Kind::Stfs; update = true; break;
        case 727: kind = Kind::Stfd; break;
        case 759: kind = Kind::Stfd; update = true; break;
        default: return 0;
        }
    } else {
        switch (po) {
        case 32: s = {4, false, false, false}; kind = Kind::IntLoad; break;
        case 33: s = {4, true, false, false}; kind = Kind::IntLoad; break;
        case 34: s = {1, false, false, false}; kind = Kind::IntLoad; break;
        case 35: s = {1, true, false, false}; kind = Kind::IntLoad; break;
        case 40: s = {2, false, false, false}; kind = Kind::IntLoad; break;
        case 41: s = {2, true, false, false}; kind = Kind::IntLoad; break;
        case 42: s = {2, false, false, true}; kind = Kind::IntLoad; break;
        case 43: s = {2, true, false, true}; kind = Kind::IntLoad; break;  // lhau
        case 36: s = {4, false, false, false}; kind = Kind::IntStore; break;
        case 37: s = {4, true, false, false}; kind = Kind::IntStore; break;
        case 38: s = {1, false, false, false}; kind = Kind::IntStore; break;
        case 39: s = {1, true, false, false}; kind = Kind::IntStore; break;
        case 44: s = {2, false, false, false}; kind = Kind::IntStore; break;
        case 45: s = {2, true, false, false}; kind = Kind::IntStore; break;
        case 48: kind = Kind::Lfs; break;
        case 49: kind = Kind::Lfs; update = true; break;
        case 50: kind = Kind::Lfd; break;
        case 51: kind = Kind::Lfd; update = true; break;
        case 52: kind = Kind::Stfs; break;
        case 53: kind = Kind::Stfs; update = true; break;
        case 54: kind = Kind::Stfd; break;
        case 55: kind = Kind::Stfd; update = true; break;
        default: return 0;
        }
    }
    if (kind == Kind::IntLoad || kind == Kind::IntStore) update = s.update;
    if (a == 0) {
        if (update) return 0;  // Exec writes r0 with the address: not worth a native form
        s.zeroBase = true;
    }
    switch (kind) {
    case Kind::IntLoad: return EmitIntLoad(e, o, s);
    case Kind::IntStore: {
        const uint32_t used = EmitIntStore(e, o, s, stfiwx);
        if (stwcx) {  // the store always succeeds: CR0 = EQ, and XER[SO] copied (the values of nibble 2 and 3)
            Xbyak::CodeGenerator& c = e.a;
            c.mov(rax, static_cast<uint64_t>(PpcCrvEncode(0x2u)));
            c.mov(rcx, static_cast<uint64_t>(PpcCrvEncode(0x3u)));
            c.cmp(e.Xer(), 0);
            c.cmovs(rax, rcx);
            c.mov(e.Crv(0), rax);
        }
        return used;
    }
    case Kind::Lfs: return EmitLfs(e, o, update, indexed, a == 0);
    case Kind::Lfd: return EmitLfd(e, o, update, indexed, a == 0);
    case Kind::Stfs: return EmitStf(e, o, true, update, indexed, a == 0);
    case Kind::Stfd: return EmitStf(e, o, false, update, indexed, a == 0);
    case Kind::None: break;
    }
    return 0;
}

}  // namespace PpcInterpreter::detail::x64
