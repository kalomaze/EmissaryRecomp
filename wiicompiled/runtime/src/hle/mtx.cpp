#include "guest_memory_alias.h"
// Native bodies for the hot Revolution SDK matrix routines (revolution/MTX/mtx.c): PSMTXCopy, PSMTXConcat,
// PSMTXConcatArray and PSMTXInverse, the paired-single assembly under nw4r g3d's and the game's bone and hitbox
// math, and C_MTXLightPerspective.
//
// Each native is bit-exact with its translated original, func_X_body, which is the spec: it writes the same bytes to
// the same guest memory (the stwu back chain included; the translation keeps no other stack stores), leaves the same
// dirty marks, and leaves the same register file, volatile registers included. MKW_NATIVE_BODY (abi_bridge.h) makes
// the translator's wrapper func_X tail-call the native after its guest code guard, so every caller reaches it at the
// translated call's cost. Inputs a native does not handle tail-call func_X_body before any guest-visible effect: a
// GQR0 that is not the float encoding, a range that wraps the lean window or needs the checked store path (MMIO,
// executable pages), or an overlap whose access order matters.
//
// Paired-single arithmetic: the translated helpers (isa/ppc_isa_float.h) run each ps_* op as one host SIMD op on two
// lanes followed by a NaN fixup, so a 4-wide op across a matrix row computes the same lane values. Every operand and
// intermediate of a row reaches that row's results through a product or an addend, so a row without a NaN result
// needed no fixup; a row with one is recomputed with the fixup after every op, from psq_l-quieted inputs.
//
// BRAWL_HLE_VERIFY=1: each call runs the translated original and the native from the same state, compares register
// file, written memory and dirty marks byte for byte, logs mismatches, and keeps the original's results.
// BRAWL_HLE_FUZZ=N (with BRAWL_HLE_VERIFY): a native's first call also runs N randomized trials through the same
// comparison (special values, aliasing, NI on and off, quantized GQRs) in an unused part of the lean window.

#include "abi_bridge.h"
#include "brawl_dirty_memory.h"
#include "insn_count.h"
#include "memory.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

MKW_NATIVE_BODY(0x801EC184u, BrawlMtx_PSMTXCopy);
MKW_NATIVE_BODY(0x801EC1B8u, BrawlMtx_PSMTXConcat);
MKW_NATIVE_BODY(0x801EC284u, BrawlMtx_PSMTXConcatArray);
MKW_NATIVE_BODY(0x801EC41Cu, BrawlMtx_PSMTXInverse);
// Not registered, C_MTXLightPerspective (0x801ECBF0, BrawlMtx_C_MTXLightPerspective, tan inlined):
// NAT_INSN's counts (NOTE3) gave 788 instructions a call against 810 for its translation with the translated tan
// (281 of them), so the native body saved nothing past tan, and tan saved 19 a call. game_natives.cpp now has
// __kernel_tan (0x803FFED4) as a native, which the translated tan calls. The code stays for a P+ measurement;
// registering it again is a line here. sqrt (0x80400B48): game_natives.cpp's GameNat_IeeeSqrt (the same native as
// one here once, now removed: two native bodies for one address stop the translator).

extern "C" void func_801EC184_body(CpuContext* ctx);
extern "C" void func_801EC1B8_body(CpuContext* ctx);
extern "C" void func_801EC284_body(CpuContext* ctx);
extern "C" void func_801EC41C_body(CpuContext* ctx);
extern "C" void func_801ECBF0_body(CpuContext* ctx);
extern "C" void func_80400AB8(CpuContext* ctx);  // tanf, C_MTXLightPerspective's callee
// guest_code_patch.cpp's table: every translated function's start, ascending; index = its g_mkwGuestCodePatched byte.
extern "C" const uint32_t g_mkwGuardedFunctionStarts[];
extern "C" const uint32_t g_mkwGuardedFunctionCount;
// Present only in a translation with MKW_EABI_REGS entries (vanilla); g3d_calc.cpp detects it the same way.
extern "C" __attribute__((weak)) uint64_t func_801D7AE4_eabi(CpuContext* ctx, uint32_t r3);

namespace {

#if MKW_LEAN_MEMORY  // the natives are written against the lean window (memory_access.h); elsewhere they fall back

using F4 = float __attribute__((ext_vector_type(4)));
using I4 = int32_t __attribute__((ext_vector_type(4)));
using B16 = uint8_t __attribute__((ext_vector_type(16)));

// ---------------------------------------------------------------------------------------------------------------
// Lanes. A matrix row is four floats in memory order; guest memory is big-endian.
// ---------------------------------------------------------------------------------------------------------------

inline B16 SwapWords(B16 b)
{
    return __builtin_shufflevector(b, b, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
}

inline F4 LoadRow(const uint8_t* host)
{
    B16 bytes;
    std::memcpy(&bytes, host, sizeof(bytes));
    return __builtin_bit_cast(F4, SwapWords(bytes));
}

// psq_l's float rule per lane (PpcLoadPairPsqFloatBitsLanesInline): a signalling NaN is quieted.
inline F4 Quiet(F4 v)
{
    const I4 bits = __builtin_bit_cast(I4, v);
    const I4 nan = (bits & 0x7FFFFFFF) > 0x7F800000;
    return __builtin_bit_cast(F4, bits | (nan & 0x00400000));
}

// psq_st's float rule per lane (PpcStorePairPsqFloatBitsLanesInline): a subnormal becomes its signed zero, a NaN is
// quieted. Then the bytes, big-endian.
inline void StoreRow(uint8_t* host, F4 v)
{
    const I4 bits = __builtin_bit_cast(I4, v);
    const I4 magnitude = bits & 0x7FFFFFFF;
    const I4 subnormal = magnitude < 0x00800000;
    const I4 quieted = bits | ((magnitude > 0x7F800000) & 0x00400000);
    const I4 stored = (subnormal & (bits & INT32_MIN)) | (~subnormal & quieted);
    const B16 bytes = SwapWords(__builtin_bit_cast(B16, stored));
    std::memcpy(host, &bytes, sizeof(bytes));
}

// The packed FPR form (ppc_isa_context.h): ps0 in the high word, ps1 in the low word.
inline void SetPairs(PPC_FPR* fpr, F4 v)  // fpr[0] = (v0, v1), fpr[1] = (v2, v3)
{
    const F4 packed = __builtin_shufflevector(v, v, 1, 0, 3, 2);
    std::memcpy(fpr, &packed, sizeof(packed));
}
inline void SetPairLow(PPC_FPR& fpr, F4 v)  // (v0, v1)
{
    const F4 packed = __builtin_shufflevector(v, v, 1, 0, 3, 2);
    std::memcpy(&fpr, &packed, sizeof(PPC_FPR));
}
inline void SetPairHigh(PPC_FPR& fpr, F4 v)  // (v2, v3)
{
    const F4 packed = __builtin_shufflevector(v, v, 3, 2, 1, 0);
    std::memcpy(&fpr, &packed, sizeof(PPC_FPR));
}

template <int Lane>
inline F4 Splat(F4 v)
{
    return __builtin_shufflevector(v, v, Lane, Lane, Lane, Lane);
}

inline bool AnyLane(I4 mask)
{
    uint64_t words[2];
    std::memcpy(words, &mask, sizeof(words));
    return (words[0] | words[1]) != 0;
}

inline bool AnyNan(F4 v)
{
    return AnyLane(v != v);
}

// a * b + c per lane with one rounding: the instruction PpcFmaddPairInline uses on each host.
inline F4 FusedMulAdd(F4 a, F4 b, F4 c)
{
#if defined(__x86_64__) && defined(__FMA__)
    return __builtin_bit_cast(F4, _mm_fmadd_ps(__builtin_bit_cast(__m128, a), __builtin_bit_cast(__m128, b),
                                               __builtin_bit_cast(__m128, c)));
#elif defined(__x86_64__)
    return __builtin_bit_cast(F4, PpcFmaLanesNoFmaInline(__builtin_bit_cast(__m128, a), __builtin_bit_cast(__m128, b),
                                                         __builtin_bit_cast(__m128, c), 1.0f));
#elif defined(__aarch64__)
    return __builtin_bit_cast(F4, vfmaq_f32(__builtin_bit_cast(float32x4_t, c), __builtin_bit_cast(float32x4_t, a),
                                            __builtin_bit_cast(float32x4_t, b)));
#endif
}

// The NaN a ps_* lane produces (PpcResolveNanLanesInline): the first NaN operand in guest order, quieted, or the
// default NaN for an invalid operation on non-NaN operands.
[[gnu::noinline, gnu::cold]] F4 ResolveNanLanes(F4 result, F4 a, F4 b, F4 c)
{
    uint32_t r[4], x[4], y[4], z[4];
    std::memcpy(r, &result, sizeof(r));
    std::memcpy(x, &a, sizeof(x));
    std::memcpy(y, &b, sizeof(y));
    std::memcpy(z, &c, sizeof(z));
    const auto isNan = [](uint32_t bits) { return (bits & 0x7FFFFFFFu) > 0x7F800000u; };
    for (int i = 0; i < 4; ++i)
        if (isNan(r[i]))
            r[i] = isNan(x[i]) ? x[i] | 0x00400000u
                 : isNan(y[i]) ? y[i] | 0x00400000u
                 : isNan(z[i]) ? z[i] | 0x00400000u : 0x7FC00000u;
    std::memcpy(&result, r, sizeof(r));
    return result;
}

// ps_mul (PpcMulPairInline): Exact applies the NaN fixup.
template <bool Exact>
inline F4 Mul(F4 a, F4 b)
{
    const F4 r = a * b;
    if (Exact && AnyNan(r)) return ResolveNanLanes(r, a, b, F4{0.0f, 0.0f, 0.0f, 0.0f});
    return r;
}

// ps_madd (PpcFmaddPairInline): multiplicand * multiplier + addend; the fixup takes them in guest order a, b, c.
template <bool Exact>
inline F4 Madd(F4 multiplicand, F4 multiplier, F4 addend)
{
    const F4 r = FusedMulAdd(multiplicand, multiplier, addend);
    if (Exact && AnyNan(r)) return ResolveNanLanes(r, multiplicand, addend, multiplier);
    return r;
}

// Row r of PSMTXConcat's a·b: lanes 0-1 are ps_muls0, ps_madds1, ps_madds0 of a[r][0..2] down b's columns 0-1, lanes
// 2-3 the same for columns 2-3 plus ps_madds1 of the unit pair {0, 1} (f31) by a[r][3]. `unit` is {-, -, u0, u1}.
template <bool Exact>
inline F4 ConcatRow(F4 a, F4 b0, F4 b1, F4 b2, F4 unit)
{
    F4 acc = Mul<Exact>(b0, Splat<0>(a));
    acc = Madd<Exact>(b1, Splat<1>(a), acc);
    acc = Madd<Exact>(b2, Splat<2>(a), acc);
    return __builtin_shufflevector(acc, Madd<Exact>(unit, Splat<3>(a), acc), 0, 1, 6, 7);
}

// PSMTXConcatArray's row: the same, except its unit term is a full ps_madd (f27 by a[r][2..3]), so column 2 adds
// u0 * a[r][2] and column 3 u1 * a[r][3].
template <bool Exact>
inline F4 ConcatArrayRow(F4 a, F4 b0, F4 b1, F4 b2, F4 unit)
{
    F4 acc = Mul<Exact>(b0, Splat<0>(a));
    acc = Madd<Exact>(b1, Splat<1>(a), acc);
    acc = Madd<Exact>(b2, Splat<2>(a), acc);
    return __builtin_shufflevector(acc, Madd<Exact>(unit, a, acc), 0, 1, 6, 7);
}

// ---------------------------------------------------------------------------------------------------------------
// Lean memory (memory_access.h): GuestMemoryAlias::Offset(address) is the offset into one host window.
// ---------------------------------------------------------------------------------------------------------------

inline uint8_t* Window() { return MemoryInline::g_leanBase; }
inline uint32_t Canonical(uint32_t address) { return GuestMemoryAlias::Offset(address); }

// psq stores mark through MarkHost, relative to flatBase; Memory::Init makes that the window.
inline bool LeanReady()
{
    return BrawlDirtyMemory::flatBase == reinterpret_cast<uintptr_t>(MemoryInline::g_leanBase);
}

// Require a mapped, contiguous host range before bypassing scalar accessors.
inline bool Fits(uint32_t address, uint64_t size)
{
    return MemoryInline::LeanRangeMapped(address, size);
}

// No 64 KiB chunk of the range sends a store to the checked path; the original's psq stores test the policy byte
// of each pair's first and last byte.
inline bool PlainStores(uint32_t canonical, uint64_t size)
{
    const uint8_t* const policy = Window() + MemoryInline::kLeanPolicyOffset;
    const uint64_t last = (canonical + size - 1) >> MemoryInline::kLeanPolicyShift;
    for (uint64_t chunk = canonical >> MemoryInline::kLeanPolicyShift; chunk <= last; ++chunk)
        if (policy[chunk] != 0) return false;
    return true;
}

// The same for one 48-byte matrix, which spans at most two chunks.
inline bool PlainStores48(const uint8_t* window, uint32_t canonical)
{
    const uint8_t* const policy = window + MemoryInline::kLeanPolicyOffset;
    return (policy[canonical >> MemoryInline::kLeanPolicyShift] |
            policy[(canonical + 47u) >> MemoryInline::kLeanPolicyShift]) == 0;
}

// The dirty marks of a matrix's six 8-byte psq stores: each store's MarkHost marks its 64-byte block and the next,
// so together the blocks from the first store's to the last one's, plus one. Two pair marks cover exactly that.
// None in a build without dirty marks (MKW_LEAN_DIRTY_MARKS=0: no translated store marks either).
inline void MarkMatrixStores(uint8_t* dirty, uint32_t canonical)
{
#if MKW_LEAN_DIRTY_MARKS
    const uint16_t both = 0x0101;
    std::memcpy(dirty + (canonical >> 6), &both, sizeof(both));
    std::memcpy(dirty + ((canonical + 40u) >> 6), &both, sizeof(both));
#else
    (void)dirty;
    (void)canonical;
#endif
}

// The {-, -, ps0, ps1} unit pair a psq_l at `host` loads.
inline F4 LoadUnit(const uint8_t* host)
{
    uint32_t words[2];
    std::memcpy(words, host, sizeof(words));
    return F4{0.0f, 0.0f, __builtin_bit_cast(float, __builtin_bswap32(words[0])),
              __builtin_bit_cast(float, __builtin_bswap32(words[1]))};
}

constexpr uint32_t kConcatUnit = 0x8059EA60u;  // f31 = {0.0f, 1.0f}: lis r6, 0x805A; subi r6, r6, 5536

// PSMTXConcat's loads, product, stores and register file. The fast pass (Exact = false) returns false before any
// store when a result lane is NaN; the exact pass then redoes it from the unchanged inputs.
template <bool Exact>
[[gnu::always_inline]] inline bool ConcatRun(CpuContext* ctx, uint8_t* m, uint8_t* dirty, uint32_t a, uint32_t b,
                                             uint32_t ab)
{
    F4 a0 = LoadRow(m + a), a1 = LoadRow(m + a + 16), a2 = LoadRow(m + a + 32);
    F4 b0 = LoadRow(m + b), b1 = LoadRow(m + b + 16), b2 = LoadRow(m + b + 32);
    F4 unit = LoadUnit(m + Canonical(kConcatUnit));
    if constexpr (Exact) {
        a0 = Quiet(a0); a1 = Quiet(a1); a2 = Quiet(a2);
        b0 = Quiet(b0); b1 = Quiet(b1); b2 = Quiet(b2);
        unit = Quiet(unit);
    }
    // Every load precedes every store in the original, so ab may alias a or b.
    const F4 r0 = ConcatRow<Exact>(a0, b0, b1, b2, unit);
    const F4 r1 = ConcatRow<Exact>(a1, b0, b1, b2, unit);
    const F4 r2 = ConcatRow<Exact>(a2, b0, b1, b2, unit);
    if (!Exact && AnyLane((r0 != r0) | (r1 != r1) | (r2 != r2))) [[unlikely]] return false;
    StoreRow(m + ab, r0);
    StoreRow(m + ab + 16, r1);
    StoreRow(m + ab + 32, r2);
    MarkMatrixStores(dirty, ab);

    // The original's register file: f0-f13 as left by its schedule, f14, f15, f31 restored, r6 the unit's address.
    ctx->gpr[6] = kConcatUnit;
    SetPairHigh(ctx->fpr[0], r2);
    SetPairHigh(ctx->fpr[1], a0);
    SetPairLow(ctx->fpr[2], r2);
    SetPairHigh(ctx->fpr[3], a1);
    SetPairs(&ctx->fpr[4], a2);
    SetPairs(&ctx->fpr[6], b0);
    SetPairs(&ctx->fpr[8], b1);
    SetPairs(&ctx->fpr[10], b2);
    SetPairs(&ctx->fpr[12], r0);
    return true;
}

[[gnu::noinline, gnu::cold]] void ConcatExact(CpuContext* ctx, uint8_t* m, uint8_t* dirty, uint32_t a, uint32_t b,
                                              uint32_t ab)
{
    ConcatRun<true>(ctx, m, dirty, a, b, ab);
}

// PSMTXConcat(a = r3, b = r4, ab = r5).
[[gnu::always_inline]] inline bool ConcatNative(CpuContext* ctx)
{
    uint8_t* const m = Window();
    uint8_t* const dirty = BrawlDirtyMemory::leanBlocks;
    if (ctx->gqr[0] != 0 || BrawlDirtyMemory::flatBase != reinterpret_cast<uintptr_t>(m)) return false;
    const uint32_t a = Canonical(ctx->gpr[3]);
    const uint32_t b = Canonical(ctx->gpr[4]);
    const uint32_t ab = Canonical(ctx->gpr[5]);
    if (!Fits(ctx->gpr[3], 48) || !Fits(ctx->gpr[4], 48) || !Fits(ctx->gpr[5], 48) || !PlainStores48(m, ab)) return false;

    // stwu r1, -64(r1) first, as in the original (its f14, f15, f31 saves are host locals in the translation).
    const uint32_t r1 = ctx->gpr[1];
    MemoryInline::FlatWriteRam32(r1 - 64u, r1);
    if (!ConcatRun<false>(ctx, m, dirty, a, b, ab)) [[unlikely]] ConcatExact(ctx, m, dirty, a, b, ab);
    return true;
}

// One matrix of PSMTXConcatArray: dst = a·src. The fast pass returns false before storing when a result lane is
// NaN. Leaves the source rows (psq_l-quieted in the exact pass) and row 0 of the product in s and r0.
template <bool Exact>
[[gnu::always_inline]] inline bool ConcatArrayMatrix(uint8_t* dirty, const F4 (&a)[3], F4 unit, const uint8_t* src,
                                                     uint8_t* dst, uint32_t dstCanonical, F4 (&s)[3], F4& r0)
{
    s[0] = LoadRow(src);
    s[1] = LoadRow(src + 16);
    s[2] = LoadRow(src + 32);
    if constexpr (Exact) {
        s[0] = Quiet(s[0]); s[1] = Quiet(s[1]); s[2] = Quiet(s[2]);
    }
    r0 = ConcatArrayRow<Exact>(a[0], s[0], s[1], s[2], unit);
    const F4 r1 = ConcatArrayRow<Exact>(a[1], s[0], s[1], s[2], unit);
    const F4 r2 = ConcatArrayRow<Exact>(a[2], s[0], s[1], s[2], unit);
    if (!Exact && AnyLane((r0 != r0) | (r1 != r1) | (r2 != r2))) [[unlikely]] return false;
    StoreRow(dst, r0);
    StoreRow(dst + 16, r1);
    StoreRow(dst + 32, r2);
    MarkMatrixStores(dirty, dstCanonical);
    return true;
}

[[gnu::noinline, gnu::cold]] void ConcatArrayMatrixExact(uint8_t* dirty, const F4 (&a)[3], F4 unit, const uint8_t* src,
                                                         uint8_t* dst, uint32_t dstCanonical, F4 (&s)[3], F4& r0)
{
    ConcatArrayMatrix<true>(dirty, a, unit, src, dst, dstCanonical, s, r0);
}

// PSMTXConcatArray(a = r3, srcBase = r4, dstBase = r5, count = r6): dst[i] = a·src[i].
[[gnu::always_inline]] inline bool ConcatArrayNative(CpuContext* ctx)
{
    uint8_t* const m = Window();
    uint8_t* const dirty = BrawlDirtyMemory::leanBlocks;
    const uint32_t count = ctx->gpr[6];
    // count < 2 leaves ctr at 0 for the loop's bdnz, which then runs 2^32 more times: the original's business.
    if (ctx->gqr[0] != 0 || BrawlDirtyMemory::flatBase != reinterpret_cast<uintptr_t>(m) || count < 2) return false;
    const uint32_t a = Canonical(ctx->gpr[3]);
    const uint32_t src = Canonical(ctx->gpr[4]);
    const uint32_t dst = Canonical(ctx->gpr[5]);
    const uint32_t unitAddress = ctx->gpr[13] - 22976u;  // f27 = {0, 1}: addi r6, r13, -22976
    const uint32_t unitCanonical = Canonical(unitAddress);
    const uint64_t size = 48ull * count;
    if (!Fits(ctx->gpr[3], 48) || !Fits(ctx->gpr[4], size) || !Fits(ctx->gpr[5], size) || !Fits(unitAddress, 8) || !PlainStores(dst, size))
        return false;
    // The original loads src[i + 1] while it still stores dst[i]; this loop finishes dst[i] first. The two agree
    // unless dst[i] overlaps src[i + 1].
    const int64_t shift = static_cast<int64_t>(dst) - static_cast<int64_t>(src);
    if (shift > 0 && shift < 96) return false;

    const uint32_t r1 = ctx->gpr[1];
    MemoryInline::FlatWriteRam32(r1 - 96u, r1);

    // a and the unit are loaded once, before any store; quieting them up front is what psq_l did.
    const F4 rows[3] = {Quiet(LoadRow(m + a)), Quiet(LoadRow(m + a + 16)), Quiet(LoadRow(m + a + 32))};
    const F4 unit = Quiet(LoadUnit(m + unitCanonical));
    F4 s[3], r0;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t d = dst + 48u * i;
        if (!ConcatArrayMatrix<false>(dirty, rows, unit, m + src + 48u * i, m + d, d, s, r0)) [[unlikely]]
            ConcatArrayMatrixExact(dirty, rows, unit, m + src + 48u * i, m + d, d, s, r0);
    }

    const uint32_t advance = 48u * (count - 1u);
    ctx->gpr[0] = count - 1u;
    ctx->gpr[4] += advance;
    ctx->gpr[5] += advance;
    ctx->gpr[6] = unitAddress;
    ctx->ctr = 0;
    SetPairs(&ctx->fpr[0], rows[0]);
    SetPairs(&ctx->fpr[2], rows[1]);
    SetPairs(&ctx->fpr[4], rows[2]);
    SetPairs(&ctx->fpr[6], s[0]);  // the last source matrix
    SetPairs(&ctx->fpr[8], s[1]);
    SetPairs(&ctx->fpr[10], s[2]);
    SetPairs(&ctx->fpr[12], r0);  // the last product's row 0; f27-f31 restored
    return true;
}

// PSMTXCopy(src = r3, dst = r4): six psq_l/psq_st pairs, so the stored floats get psq_st's rule, the registers
// psq_l's.
[[gnu::always_inline]] inline bool CopyNative(CpuContext* ctx)
{
    uint8_t* const m = Window();
    uint8_t* const dirty = BrawlDirtyMemory::leanBlocks;
    if (ctx->gqr[0] != 0 || BrawlDirtyMemory::flatBase != reinterpret_cast<uintptr_t>(m)) return false;
    const uint32_t src = Canonical(ctx->gpr[3]);
    const uint32_t dst = Canonical(ctx->gpr[4]);
    if (!Fits(ctx->gpr[3], 48) || !Fits(ctx->gpr[4], 48) || !PlainStores48(m, dst)) return false;
    // The original interleaves loads and stores; loading all first agrees unless a store lands on a later load.
    const int64_t shift = static_cast<int64_t>(dst) - static_cast<int64_t>(src);
    if (shift > 0 && shift < 48) return false;

    const F4 v0 = Quiet(LoadRow(m + src)), v1 = Quiet(LoadRow(m + src + 16)), v2 = Quiet(LoadRow(m + src + 32));
    StoreRow(m + dst, v0);
    StoreRow(m + dst + 16, v1);
    StoreRow(m + dst + 32, v2);
    MarkMatrixStores(dirty, dst);
    SetPairs(&ctx->fpr[0], v0);
    SetPairs(&ctx->fpr[2], v1);
    SetPairs(&ctx->fpr[4], v2);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Single FPRs as the translated code holds them: a double, or a pair packed as ps0 in the high word and ps1 in the
// low word, seen here as lanes {ps1, ps0, 0, 0}. The Pair* ops are the ps_* helpers' host instructions without
// their NaN fixup: a native using them checks every value it leaves in a register or in memory for NaN and hands
// such calls to the original before any guest-visible effect. A NaN-free result means no op produced a NaN (each
// op's NaN reaches a result through its product or addend; compares do not depend on a NaN's payload).
// ---------------------------------------------------------------------------------------------------------------

inline F4 Lanes(double packed)
{
    const double pair[2] = {packed, 0.0};
    return __builtin_bit_cast(F4, pair);
}
inline double Packed(F4 v)
{
    double pair[2];
    std::memcpy(pair, &v, sizeof(pair));
    return pair[0];
}
inline double PairMul(double a, double b) { return Packed(Lanes(a) * Lanes(b)); }
inline double PairMuls0(double a, double c) { return Packed(Lanes(a) * Splat<1>(Lanes(c))); }  // ps0 is lane 1
inline double PairAdd(double a, double b) { return Packed(Lanes(a) + Lanes(b)); }
inline double PairSub(double a, double b) { return Packed(Lanes(a) - Lanes(b)); }
inline double PairMadd(double a, double c, double b) { return Packed(FusedMulAdd(Lanes(a), Lanes(c), Lanes(b))); }
inline double PairMsub(double a, double c, double b) { return Packed(FusedMulAdd(Lanes(a), Lanes(c), -Lanes(b))); }
inline double PairNmadd(double a, double c, double b) { return Packed(-FusedMulAdd(Lanes(a), Lanes(c), Lanes(b))); }
inline double PairNmsub(double a, double c, double b) { return Packed(-FusedMulAdd(Lanes(a), Lanes(c), -Lanes(b))); }

// Any NaN among the float views of two FPRs (a double NaN is one too, through its high word).
inline I4 NanIn(double x, double y)
{
    const double pair[2] = {x, y};
    const F4 v = __builtin_bit_cast(F4, pair);
    return v != v;
}

// psq_l w=0, float GQR, without psq_l's sNaN quieting (only NaN inputs differ, and those reach a result).
inline double LoadPair(const uint8_t* host)
{
    uint64_t raw;
    std::memcpy(&raw, host, sizeof(raw));
    return PpcBitCastToDoubleInline(__builtin_bswap64(raw));
}

// psq_l w=1, float GQR (PpcLoadSinglePsqFloatFastInline): {value, 1.0f}.
inline double LoadSingleOne(const uint8_t* host)
{
    uint32_t raw;
    std::memcpy(&raw, host, sizeof(raw));
    return PpcBitCastToDoubleInline((static_cast<uint64_t>(__builtin_bswap32(raw)) << 32) | 0x3F800000u);
}

// lfs (MemoryInline::FlatReadFloat32): the exact widening.
inline double LoadFloat(const uint8_t* host)
{
    uint32_t raw;
    std::memcpy(&raw, host, sizeof(raw));
    return PpcBitCastToDoubleInline(PpcConvertToDoubleBitsInline(__builtin_bswap32(raw)));
}

// psq_st w=0, float GQR: psq_st's per-lane rule, ps0 first.
inline void StorePair(uint8_t* host, double packed)
{
    const F4 v = Lanes(packed);
    const I4 bits = __builtin_bit_cast(I4, __builtin_shufflevector(v, v, 1, 0, 2, 3));
    const I4 magnitude = bits & 0x7FFFFFFF;
    const I4 subnormal = magnitude < 0x00800000;
    const I4 quieted = bits | ((magnitude > 0x7F800000) & 0x00400000);
    const B16 bytes = SwapWords(__builtin_bit_cast(B16, (subnormal & (bits & INT32_MIN)) | (~subnormal & quieted)));
    std::memcpy(host, &bytes, 8);
}

// psq_st w=1, float GQR (PpcStoreSinglePsqFloatFastInline): ps0 widened, then narrowed with flush to zero.
inline void StoreSingle(uint8_t* host, double packed)
{
    const uint32_t bits =
        PpcConvertToSingleFTZInline(PpcBitCastToU64Inline(static_cast<double>(PpcGetPs0Inline(packed))));
    const uint32_t be = __builtin_bswap32(bits);
    std::memcpy(host, &be, sizeof(be));
}

inline void Store32(uint8_t* host, uint32_t value)
{
    const uint32_t be = __builtin_bswap32(value);
    std::memcpy(host, &be, sizeof(be));
}

inline void Store64(uint8_t* host, uint64_t value)
{
    const uint64_t be = __builtin_bswap64(value);
    std::memcpy(host, &be, sizeof(be));
}

// stfs (MemoryInline::FlatWriteFloat32's conversion).
inline void StoreFloat(uint8_t* host, double value)
{
    Store32(host, MemoryInline::ConvertPpcDoubleToSingleBits(value));
}

// The dirty marks of stores starting at offsets 0 through `last` < 128 of a range, none more than 64 bytes apart:
// each marks its block and the next, so together the blocks from the first's to the last's plus one, which the
// first's and the last's pairs cover (they are at most two blocks apart).
inline void MarkStores(uint8_t* dirty, uint32_t canonical, uint32_t last)
{
#if MKW_LEAN_DIRTY_MARKS
    const uint16_t both = 0x0101;
    std::memcpy(dirty + (canonical >> 6), &both, sizeof(both));
    std::memcpy(dirty + ((canonical + last) >> 6), &both, sizeof(both));
#else
    (void)dirty;
    (void)canonical;
    (void)last;
#endif
}

// PSMTXInverse(src = r3, inv = r4): r3 = 0 for a singular matrix (nothing stored), else 1. The original's schedule
// with the Pair* ops; a NaN anywhere in its results goes to the original.
[[gnu::always_inline]] inline bool InverseNative(CpuContext* ctx)
{
    uint8_t* const m = Window();
    uint8_t* const dirty = BrawlDirtyMemory::leanBlocks;
    if (ctx->gqr[0] != 0 || BrawlDirtyMemory::flatBase != reinterpret_cast<uintptr_t>(m)) return false;
    const uint32_t src = Canonical(ctx->gpr[3]);
    const uint32_t dst = Canonical(ctx->gpr[4]);
    if (!Fits(ctx->gpr[3], 48) || !Fits(ctx->gpr[4], 48) || !PlainStores48(m, dst)) return false;
    // Every load precedes every store, so inv may alias src.
    const uint8_t* const s = m + src;
    double f0 = LoadSingleOne(s), f1 = LoadPair(s + 4), f2 = LoadSingleOne(s + 16);
    double f6 = PPC_PsMerge10Inline(f1, f0);
    double f3 = LoadPair(s + 20), f4 = LoadSingleOne(s + 32);
    double f7 = PPC_PsMerge10Inline(f3, f2);
    double f5 = LoadPair(s + 36);
    double f11 = PairMul(f3, f6);
    double f13 = PairMul(f5, f7);
    double f8 = PPC_PsMerge10Inline(f5, f4);
    f11 = PairMsub(f1, f7, f11);
    double f12 = PairMul(f1, f8);
    f13 = PairMsub(f3, f8, f13);
    double f10 = PairMul(f3, f4);
    f12 = PairMsub(f5, f6, f12);
    double f9 = PairMul(f0, f5);
    f8 = PairMul(f1, f2);
    f6 = PairSub(f6, f6);
    f10 = PairMsub(f2, f5, f10);
    f7 = PairMul(f0, f13);
    f9 = PairMsub(f1, f4, f9);
    f7 = PairMadd(f2, f12, f7);
    f8 = PairMsub(f0, f3, f8);
    f7 = PairMadd(f4, f11, f7);
    uint32_t cr = ctx->cr;
    SetCRFloatResident(cr, 0, PpcGetPs0Inline(f7), PpcGetPs0Inline(f6));  // fcmpu cr0, f7, f6
    const bool singular = (cr & 0x20000000u) != 0;
    double stored0 = 0.0, stored1 = 0.0;
    if (!singular) {
        f0 = PPC_Fres(f7);
        f6 = PairAdd(f0, f0);
        f5 = PairMul(f0, f0);
        f0 = PairNmsub(f7, f5, f6);
        f1 = LoadFloat(s + 12);
        f13 = PairMuls0(f13, f0);
        f2 = LoadFloat(s + 28);
        f12 = PairMuls0(f12, f0);
        f3 = LoadFloat(s + 44);
        f11 = PairMuls0(f11, f0);
        stored0 = PPC_PsMerge00Inline(f13, f12);
        f10 = PairMuls0(f10, f0);
        stored1 = PPC_PsMerge11Inline(f13, f12);
        f9 = PairMuls0(f9, f0);
        f6 = PairMul(f13, PPC_PsFromScalarInline(f1));
        f8 = PairMuls0(f8, f0);
        f6 = PairMadd(f12, PPC_PsFromScalarInline(f2), f6);
        f6 = PairNmadd(f11, PPC_PsFromScalarInline(f3), f6);
        f7 = PairMul(f10, PPC_PsFromScalarInline(f1));
        f5 = PPC_PsMerge00Inline(f11, f6);
        f4 = PPC_PsMerge11Inline(f11, f6);
        f7 = PairMadd(f9, PPC_PsFromScalarInline(f2), f7);
        f7 = PairNmadd(f8, PPC_PsFromScalarInline(f3), f7);
    }
    if (AnyLane(NanIn(f0, f1) | NanIn(f2, f3) | NanIn(f4, f5) | NanIn(f6, f7) | NanIn(f8, f9) | NanIn(f10, f11) |
                NanIn(f12, f13) | NanIn(stored0, stored1))) [[unlikely]]
        return false;
    if (!singular) {
        uint8_t* const d = m + dst;
        StorePair(d, stored0);
        StorePair(d + 16, stored1);
        StoreSingle(d + 32, f10);
        StoreSingle(d + 36, f9);
        StoreSingle(d + 40, f8);
        StorePair(d + 8, f5);
        StorePair(d + 24, f4);
        StoreSingle(d + 44, f7);
        MarkStores(dirty, dst, 44);
    }
    ctx->gpr[3] = singular ? 0u : 1u;
    ctx->cr = cr;
    const double out[14] = {f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13};
    for (int i = 0; i < 14; ++i) ctx->fpr[i].d = out[i];
    return true;
}

int32_t GuardIndex(uint32_t address)
{
    const uint32_t* const begin = g_mkwGuardedFunctionStarts;
    const uint32_t* const end = begin + g_mkwGuardedFunctionCount;
    const uint32_t* const it = std::lower_bound(begin, end, address);
    return it != end && *it == address ? static_cast<int32_t>(it - begin) : -1;
}
// A callee the native evaluates itself: translated, and its code not patched.
inline bool CalleeUnpatched(int32_t guard) { return guard >= 0 && g_mkwGuestCodePatched[guard] == 0; }

inline double LoadDouble(uint32_t address)
{
    return MemoryInline::FlatReadFloat64(address);
}

// BRAWL_HLE_TAN=0: C_MTXLightPerspective calls the translated tan, for A/B measurements on one binary.
const bool g_tanNative = [] {
    const char* value = std::getenv("BRAWL_HLE_TAN");
    return value == nullptr || value[0] != '0';
}();

// tan (0x80400AB8) as C_MTXLightPerspective calls it, for |x| in [2^-28, pi/4]: its argument test, then fdlibm's
// __kernel_tan (0x803FFED4) with iy = 1, both as translated:
// - the same frame stores (tan: back chain, f1 at +8, LR at +36; __kernel_tan: back chain, f1 at +8 (again after the
//   |x| >= 0.6744 reduction), z at +24, and that path's int-to-double words at +32..+47; its f30/f31 saves are kept
//   in registers by the translation, not stored);
// - the same double ops in the same order, through the translation's helpers (PpcF*Inline), so every result is the
//   original's, NaN and non-IEEE mode included;
// - the registers the translated bodies write back: r3 = 1, r4, f1, f2 (r1 and LR as on entry, which tan restores).
// False, with no effect, for other arguments, with tan or __kernel_tan patched, or a frame not plain lean memory; and
// in P+, whose translated bodies also write back r0, r5-r8, f0, f3-f13, CR and XER.
bool TanNative(CpuContext* ctx)
{
    static const int32_t kGuardTan = GuardIndex(0x80400AB8u);
    static const int32_t kGuardKernelTan = GuardIndex(0x803FFED4u);
    if (!g_tanNative || func_801D7AE4_eabi == nullptr || !CalleeUnpatched(kGuardTan) ||
        !CalleeUnpatched(kGuardKernelTan))
        return false;
    uint8_t* const m = Window();
    const uint32_t r1 = ctx->gpr[1];
    const uint32_t r1Tan = r1 - 32u, r1Kernel = r1Tan - 80u;
    const uint32_t k = Canonical(r1Kernel);
    // __kernel_tan's stwu r1, -80(r1) through tan's stw r0, 36(r1): [r1Kernel, r1Tan + 40).
    if (!Fits(r1Kernel, 120) || !PlainStores(k, 120)) return false;
    const uint64_t xBits = ctx->fpr[1].raw;
    const uint32_t hx = static_cast<uint32_t>(xBits >> 32);
    const int32_t ix = static_cast<int32_t>(hx & 0x7FFFFFFFu);
    if (ix > 0x3FE921FB || ix < 0x3E300000) return false;  // tan's reduction path; __kernel_tan's tiny-|x| path
    const uint32_t r2 = ctx->gpr[2];

    double f1 = ctx->fpr[1].d;
    double f2 = LoadDouble(r2 - 16928u);  // tan: lfd f2 (y)
    const bool big = ix >= 0x3FE59428;
    if (big) {
        if (static_cast<int32_t>(hx) < 0) {
            f1 = -f1;
            f2 = -f2;
        }
        const double pio4 = LoadDouble(r2 - 17104u), pio4lo = LoadDouble(r2 - 17096u);
        f1 = PpcFsubInline(pio4, f1);
        const double w = PpcFsubInline(pio4lo, f2);
        f2 = LoadDouble(r2 - 17088u);
        f1 = PpcFaddInline(f1, w);
    }
    const uint32_t t = 0x80420368u;  // T[0..12] (lis r6, 0x8042; addi r5, r6, 872)
    const double z = PpcFmulInline(f1, f1);
    const double w = PpcFmulInline(z, z);
    double f5 = LoadDouble(t + 96), f8 = LoadDouble(t + 88), f6 = LoadDouble(t + 80);
    double f10 = LoadDouble(t + 72);
    double f4 = PpcFmulInline(z, f1);
    double f7 = PpcFmulInline(w, f5);
    f5 = LoadDouble(t);
    double f9 = LoadDouble(t + 64);
    double f11 = PpcFmulInline(w, f8);
    const double f31 = LoadDouble(t + 56);
    f8 = LoadDouble(t + 48);
    f6 = PpcFaddInline(f6, f7);
    const double f13 = LoadDouble(t + 40);
    double f30 = PpcFaddInline(f10, f11);
    f7 = LoadDouble(t + 32);
    f5 = PpcFmulInline(f5, f4);
    const double f12 = LoadDouble(t + 24);
    f10 = PpcFmulInline(w, f6);
    f6 = LoadDouble(t + 16);
    f11 = LoadDouble(t + 8);
    f30 = PpcFmulInline(w, f30);
    f9 = PpcFaddInline(f9, f10);
    f10 = PpcFaddInline(f31, f30);
    f9 = PpcFmulInline(w, f9);
    f10 = PpcFmulInline(w, f10);
    f8 = PpcFaddInline(f8, f9);
    f9 = PpcFaddInline(f13, f10);
    f8 = PpcFmulInline(w, f8);
    f9 = PpcFmulInline(w, f9);
    f7 = PpcFaddInline(f7, f8);
    f8 = PpcFaddInline(f12, f9);
    f7 = PpcFmulInline(w, f7);
    f8 = PpcFmulInline(w, f8);
    double f3 = PpcFaddInline(f6, f7);
    f6 = PpcFaddInline(f11, f8);
    f3 = PpcFmulInline(z, f3);
    f3 = PpcFaddInline(f6, f3);
    f3 = PpcFmulInline(f4, f3);
    f3 = PpcFaddInline(f2, f3);
    double f0 = PpcFmulInline(z, f3);
    f6 = PpcFaddInline(f2, f0);
    f6 = PpcFaddInline(f6, f5);
    f4 = PpcFaddInline(f1, f6);

    uint8_t* const kf = m + k;
    Store32(kf, r1Tan);
    Store64(kf + 8, __builtin_bit_cast(uint64_t, f1));
    Store64(kf + 24, __builtin_bit_cast(uint64_t, z));
    uint32_t r4 = 0x3FE60000u;
    double result = f4;
    if (big) {
        // (double)iy and the sign's (double)(1 - ((hx >> 30) & 2)) by the 0x43300000 word pairs, then
        // v - 2 * (x - (w * w / (w + v) - r)) times that sign.
        r4 = 0x43300000u;
        const uint32_t iyWord = 1u ^ 0x80000000u;
        const uint32_t signWord = (1u - ((hx >> 30) & 2u)) ^ 0x80000000u;
        Store32(kf + 36, iyWord);
        Store32(kf + 32, r4);
        Store32(kf + 44, signWord);
        Store32(kf + 40, r4);
        f5 = LoadDouble(r2 - 17072u);
        f3 = PpcFmulInline(f4, f4);
        f0 = LoadDouble(r2 - 17080u);
        f7 = PpcFsubInline(__builtin_bit_cast(double, (uint64_t{0x43300000u} << 32) | iyWord), f5);
        f2 = PpcFaddInline(f4, f7);
        double sign = PpcFsubInline(__builtin_bit_cast(double, (uint64_t{0x43300000u} << 32) | signWord), f5);
        f2 = PpcFdivInline(f3, f2);
        f2 = PpcFsubInline(f2, f6);
        f1 = PpcFsubInline(f1, f2);
        f0 = PpcFmulInline(f0, f1);
        f0 = PpcFsubInline(f7, f0);
        result = PpcFmulInline(sign, f0);
    }
    uint8_t* const tf = kf + 80;
    Store32(tf, r1);
    Store64(tf + 8, xBits);
    Store32(tf + 36, ctx->lr);
    uint8_t* const dirty = BrawlDirtyMemory::leanBlocks;
    MarkStores(dirty, k, big ? 44u : 24u);
    MarkStores(dirty, k + 80u, 36u);
    ctx->gpr[3] = 1u;
    ctx->gpr[4] = r4;
    ctx->fpr[1].d = result;
    ctx->fpr[2].d = f2;
    return true;
}

// C_MTXLightPerspective(m = r3, fovY = f1, aspect = f2, scaleS = f3, scaleT = f4, transS = f5, transT = f6): the
// original's frame stores, its tanf call (the translated callee, handed the register file the original hands it)
// and its matrix stores, with the frame's and the matrix's store checks made once. The scalar float ops are the
// translated helpers themselves.
bool LightPerspectiveNative(CpuContext* ctx)
{
    uint8_t* const m = Window();
    uint8_t* const dirty = BrawlDirtyMemory::leanBlocks;
    // The frame's psq_st take GQR0's store half.
    if ((ctx->gqr[0] & 0xFFFFu) != 0 || BrawlDirtyMemory::flatBase != reinterpret_cast<uintptr_t>(m)) return false;
    const uint32_t r1 = ctx->gpr[1];
    const uint32_t r2 = ctx->gpr[2];
    const uint32_t out = ctx->gpr[3];
    const uint32_t frame = Canonical(r1 - 96u);
    const uint32_t matrix = Canonical(out);
    // stwu r1, -96(r1) through stw r0, 100(r1): [r1 - 96, r1 + 8), the frame and the caller's LR slot.
    if (!Fits(r1 - 96u, 104) || !PlainStores(frame, 104) || !Fits(out, 48) || !PlainStores48(m, matrix)) return false;

    uint8_t* const f = m + frame;
    const uint32_t lr = ctx->lr;
    Store32(f, r1);
    Store32(f + 100, lr);
    Store64(f + 80, ctx->fpr[31].raw);
    StorePair(f + 88, PPC_PsFromScalarInline(ctx->fpr[31].d));
    Store64(f + 64, ctx->fpr[30].raw);
    StorePair(f + 72, PPC_PsFromScalarInline(ctx->fpr[30].d));
    Store64(f + 48, ctx->fpr[29].raw);
    StorePair(f + 56, PPC_PsFromScalarInline(ctx->fpr[29].d));
    Store64(f + 32, ctx->fpr[28].raw);
    StorePair(f + 40, PPC_PsFromScalarInline(ctx->fpr[28].d));
    Store64(f + 16, ctx->fpr[27].raw);
    StorePair(f + 24, PPC_PsFromScalarInline(ctx->fpr[27].d));
    Store32(f + 12, ctx->gpr[31]);
    MarkStores(dirty, frame, 100);

    const double f7 = MemoryInline::FlatReadFloat32(r2 - 23488u);
    double f1 = PpcFmulsInline(f7, ctx->fpr[1].d);
    const double f0 = MemoryInline::FlatReadFloat32(r2 - 23472u);
    f1 = PpcFmulsInline(f0, f1);
    // The register file the original hands tanf. P+'s translation writes every register back before the call: f27-f31
    // take aspect..transT, r31 the matrix, r0 the LR, f0 and f7 the constants. The vanilla one (MKW_EABI_REGS) keeps
    // those in host registers and writes only LR, r1 and f1, so a callee's frame saves (__rem_pio2's, for an angle
    // past pi/4) store the context's entry values: the native leaves them there too, and keeps the arguments itself.
    const bool eabi = func_801D7AE4_eabi != nullptr;
    const PPC_FPR args[5] = {ctx->fpr[2], ctx->fpr[3], ctx->fpr[4], ctx->fpr[5], ctx->fpr[6]};
    ctx->lr = 0x801ECC54u;
    ctx->gpr[1] = r1 - 96u;
    ctx->fpr[1].d = f1;
    if (!eabi) {
        ctx->gpr[0] = lr;
        ctx->gpr[31] = out;
        ctx->fpr[0].d = f0;
        ctx->fpr[7].d = f7;
        for (int i = 0; i < 5; ++i) ctx->fpr[27 + i] = args[i];
    }
    if (!TanNative(ctx)) func_80400AB8(ctx);

    // As the original: r1, r2 and the matrix pointer from before the call, f1 from the callee, f27-f31 from the callee
    // (P+: non-volatile, through the context) or the original's host registers (vanilla).
    const PPC_FPR* const a = eabi ? args : &ctx->fpr[27];
    const double f5 = static_cast<double>(PpcForceSingleValueInline(ctx->fpr[1].d));
    const double f3 = MemoryInline::FlatReadFloat32(r2 - 23492u);
    double f4 = MemoryInline::FlatReadFloat32(r2 - 23496u);
    double g1 = -(a[4].d);
    double g0 = MemoryInline::FlatReadFloat32(r2 - 23476u);
    const double f2 = -(a[3].d);
    f4 = static_cast<double>(PpcForceSingleValueInline(PpcFdivInline(f4, f5)));
    // The callee could have changed the matrix's store policy (a mod patching it, in principle): checked again.
    const bool plain = PlainStores48(m, matrix);
    uint8_t* const o = m + matrix;
    const auto store = [&](uint32_t offset, double value) {
        if (plain) StoreFloat(o + offset, value);
        else MemoryInline::FlatWriteFloat32(out + offset, value);
    };
    store(24, g1);
    store(40, g0);
    store(4, f3);
    store(8, f2);
    store(12, f3);
    g1 = static_cast<double>(PpcForceSingleValueInline(PpcFdivInline(f4, a[0].d)));
    store(16, f3);
    store(28, f3);
    store(32, f3);
    store(36, f3);
    store(44, f3);
    g1 = PpcFmulsInline(a[1].d, g1);
    g0 = PpcFmulsInline(f4, a[2].d);
    store(0, g1);
    store(20, g0);
    if (plain) MarkStores(dirty, matrix, 44);

    // The epilogue reloads the saved registers from the frame (the callee could have changed them there).
    uint64_t saved64[5];
    uint32_t saved32[2];
    for (int i = 0; i < 5; ++i) {
        std::memcpy(&saved64[i], f + 16 + 16 * i, 8);
        saved64[i] = __builtin_bswap64(saved64[i]);
    }
    std::memcpy(&saved32[0], f + 12, 4);
    std::memcpy(&saved32[1], f + 100, 4);
    const uint32_t r0 = __builtin_bswap32(saved32[1]);
    ctx->lr = r0;
    ctx->gpr[0] = r0;
    ctx->gpr[1] = r1;
    ctx->gpr[31] = __builtin_bswap32(saved32[0]);
    ctx->fpr[0].d = g0;
    ctx->fpr[1].d = g1;
    ctx->fpr[2].d = f2;
    ctx->fpr[3].d = f3;
    ctx->fpr[4].d = f4;
    ctx->fpr[5].d = f5;
    for (int i = 0; i < 5; ++i) ctx->fpr[27 + i].raw = saved64[i];  // f27 at +16 ... f31 at +80
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// BRAWL_HLE_VERIFY / BRAWL_HLE_FUZZ
// ---------------------------------------------------------------------------------------------------------------

struct GuestRange {
    uint32_t address;
    uint32_t size;
};

struct GuestRanges {
    GuestRange items[4];
    int count = 0;
    void Add(uint32_t address, uint32_t size) { items[count++] = {address, size}; }
};

struct Fuzzer;

struct NativeSpec {
    const char* name;
    void (*body)(CpuContext*);
    bool (*native)(CpuContext*);
    // Every guest range the original may write for these inputs (outputs, its stack frame); false when it cannot be
    // verified (a range not plain RAM, or inputs the original itself mishandles).
    bool (*writes)(const CpuContext&, GuestRanges&);
    // Sets up one random trial in the fuzzer's scratch window.
    void (*setup)(CpuContext&, Fuzzer&);
    std::atomic<uint64_t> calls{0}, compared{0}, fallbacks{0}, skipped{0}, mismatches{0};
    std::atomic<bool> fuzzed{false};
    int site = -2;  // NAT_INSN's site, registered at the first counted call (-2: not yet)
};

// 0: natives. kVerify (BRAWL_HLE_VERIFY=1): differential checks. kOriginals (BRAWL_HLE_MTX=0): the translated
// originals behind the same wrappers, for A/B measurements on one binary.
constexpr uint8_t kVerify = 1;
constexpr uint8_t kOriginals = 2;
const uint8_t g_mode = [] {
    const auto enabled = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    };
    const char* natives = std::getenv("BRAWL_HLE_MTX");
    if (natives != nullptr && natives[0] == '0') return kOriginals;
    return enabled("BRAWL_HLE_VERIFY") ? kVerify : uint8_t{0};
}();
std::mutex g_logMutex;
std::atomic<int> g_mismatchLogs{0};
constexpr int kMaxMismatchLogs = 40;

// Range bytes, then each range's dirty bytes (its blocks and the one a pair mark spills into).
void Capture(const GuestRanges& ranges, std::vector<uint8_t>& out)
{
    out.clear();
    for (int i = 0; i < ranges.count; ++i) {
        const uint32_t c = Canonical(ranges.items[i].address);
        const uint32_t size = ranges.items[i].size;
        out.insert(out.end(), Window() + c, Window() + c + size);
        const uint8_t* const dirty = BrawlDirtyMemory::leanBlocks;
        out.insert(out.end(), dirty + (c >> 6), dirty + ((c + size - 1) >> 6) + 2);
    }
}

void Restore(const GuestRanges& ranges, const std::vector<uint8_t>& in)
{
    BrawlDirtyMemory::MarksCleared();
    size_t at = 0;
    for (int i = 0; i < ranges.count; ++i) {
        const uint32_t c = Canonical(ranges.items[i].address);
        const uint32_t size = ranges.items[i].size;
        std::memcpy(Window() + c, in.data() + at, size);
        at += size;
        const size_t blocks = ((c + size - 1) >> 6) + 2 - (c >> 6);
        std::memcpy(BrawlDirtyMemory::leanBlocks + (c >> 6), in.data() + at, blocks);
        at += blocks;
    }
}

bool Verifiable(const GuestRanges& ranges)
{
    for (int i = 0; i < ranges.count; ++i) {
        const uint32_t c = Canonical(ranges.items[i].address);
        if (ranges.items[i].size == 0 || !Fits(ranges.items[i].address, ranges.items[i].size) || !PlainStores(c, ranges.items[i].size))
            return false;
    }
    return LeanReady();
}

void DescribeContextWord(size_t offset, char* out, size_t outSize)
{
    if (offset < offsetof(CpuContext, cr)) { std::snprintf(out, outSize, "r%zu", offset / 4); return; }
    if (offset < offsetof(CpuContext, fpr)) {
        static const char* const names[] = {"cr", "lr", "ctr", "xer", "fpscr", "pc"};
        std::snprintf(out, outSize, "%s", names[(offset - offsetof(CpuContext, cr)) / 4]);
        return;
    }
    if (offset < offsetof(CpuContext, gqr)) {
        const size_t f = offset - offsetof(CpuContext, fpr);
        std::snprintf(out, outSize, "f%zu.%s", f / 8, (f % 8) < 4 ? "ps1" : "ps0");
        return;
    }
    if (offset < offsetof(CpuContext, hid0)) { std::snprintf(out, outSize, "gqr%zu", (offset - offsetof(CpuContext, gqr)) / 4); return; }
    std::snprintf(out, outSize, "spr+%zu", offset - offsetof(CpuContext, hid0));
}

void LogMismatch(const NativeSpec& spec, const char* origin, const CpuContext& in, const CpuContext& original,
                 const CpuContext& native, const GuestRanges& ranges, const std::vector<uint8_t>& memOriginal,
                 const std::vector<uint8_t>& memNative)
{
    if (g_mismatchLogs.fetch_add(1, std::memory_order_relaxed) >= kMaxMismatchLogs) return;
    std::lock_guard<std::mutex> lock(g_logMutex);
    std::fprintf(stderr,
                 "[hle-verify] MISMATCH %s (%s): r1=%08X r3=%08X r4=%08X r5=%08X r6=%08X r13=%08X gqr0=%08X fpscr=%08X ni=%d\n",
                 spec.name, origin, in.gpr[1], in.gpr[3], in.gpr[4], in.gpr[5], in.gpr[6], in.gpr[13], in.gqr[0],
                 in.fpscr, MkwHostNiActiveInline() ? 1 : 0);
    int lines = 0;
    for (size_t off = 0; off + 4 <= sizeof(CpuContext) && lines < 12; off += 4) {
        uint32_t o = 0, n = 0;
        std::memcpy(&o, reinterpret_cast<const uint8_t*>(&original) + off, 4);
        std::memcpy(&n, reinterpret_cast<const uint8_t*>(&native) + off, 4);
        if (o == n) continue;
        char name[32];
        DescribeContextWord(off, name, sizeof(name));
        std::fprintf(stderr, "[hle-verify]   %-8s original %08X native %08X\n", name, o, n);
        ++lines;
    }
    size_t at = 0;
    for (int i = 0; i < ranges.count; ++i) {
        const uint32_t size = ranges.items[i].size;
        const uint32_t c = Canonical(ranges.items[i].address);
        const size_t blocks = ((c + size - 1) >> 6) + 2 - (c >> 6);
        for (size_t k = 0; k < size + blocks && lines < 24; ++k) {
            if (memOriginal[at + k] == memNative[at + k]) continue;
            if (k < size) {
                const size_t word = k & ~size_t{3};
                std::fprintf(stderr, "[hle-verify]   mem %08X: original %02X%02X%02X%02X native %02X%02X%02X%02X\n",
                             ranges.items[i].address + static_cast<uint32_t>(word),
                             memOriginal[at + word], memOriginal[at + word + 1], memOriginal[at + word + 2], memOriginal[at + word + 3],
                             memNative[at + word], memNative[at + word + 1], memNative[at + word + 2], memNative[at + word + 3]);
                k = word + 3;
            } else {
                std::fprintf(stderr, "[hle-verify]   dirty block %zu of range %08X: original %u native %u\n",
                             k - size, ranges.items[i].address, memOriginal[at + k], memNative[at + k]);
            }
            ++lines;
        }
        at += size + blocks;
    }
}

enum class VerifyResult { Compared, Mismatch, Fallback, Skipped };

// The registers a caller can read after the call. In P+ the whole context. In the vanilla translation
// (MKW_EABI_REGS) a body writes back only what its callers read, where the natives write P+'s register file: there,
// r1, r2, r13-r31, f14-f31, LR, CR fields 2-4, the GQRs and the special registers past them always, and every other
// register where the original changed it (one it wrote back with its entry value goes unchecked).
bool SameRegisters(const CpuContext& in, const CpuContext& original, const CpuContext& native)
{
    if (func_801D7AE4_eabi == nullptr) return std::memcmp(&native, &original, sizeof(CpuContext)) == 0;
    for (uint32_t r = 0; r < 32; ++r) {
        const bool kept = r == 1 || r == 2 || r >= 13;
        if (native.gpr[r] != original.gpr[r] && (kept || original.gpr[r] != in.gpr[r])) return false;
    }
    for (uint32_t f = 0; f < 32; ++f) {
        if (native.fpr[f].raw != original.fpr[f].raw && (f >= 14 || original.fpr[f].raw != in.fpr[f].raw))
            return false;
    }
    if (native.lr != original.lr) return false;
    if (((native.cr ^ original.cr) & (0x00FFF000u | (original.cr ^ in.cr))) != 0) return false;
    if (native.ctr != original.ctr && original.ctr != in.ctr) return false;
    if (native.xer != original.xer && original.xer != in.xer) return false;
    if (native.fpscr != original.fpscr && original.fpscr != in.fpscr) return false;
    const size_t from = offsetof(CpuContext, gqr);
    return std::memcmp(reinterpret_cast<const uint8_t*>(&native) + from,
                       reinterpret_cast<const uint8_t*>(&original) + from, sizeof(CpuContext) - from) == 0;
}

// The same bytes, and dirty marks that cover every block whose range bytes changed. A translation with
// MKW_LEAN_RANGES marks a lowered segment range's span at once (memory_access.h LeanMarkSpan), so the original can
// mark blocks it did not write or leave the second block of a store's pair unmarked; rollback needs the written
// blocks marked, not the same extra marks.
bool SameMemory(const GuestRanges& ranges, const std::vector<uint8_t>& before, const std::vector<uint8_t>& original,
                const std::vector<uint8_t>& native)
{
    if (native == original) return true;
    size_t at = 0;
    for (int i = 0; i < ranges.count; ++i) {
        const uint32_t c = Canonical(ranges.items[i].address);
        const uint32_t size = ranges.items[i].size;
        if (std::memcmp(native.data() + at, original.data() + at, size) != 0) return false;
        const uint32_t first = c >> 6;
        const uint32_t blocks = ((c + size - 1) >> 6) + 2 - first;
        for (uint32_t b = 0; b < blocks; ++b) {
            if (native[at + size + b] != 0) continue;
            const uint32_t lo = std::max((first + b) << 6, c) - c;
            const uint32_t hi = std::min((first + b + 1) << 6, c + size) - c;
            for (uint32_t k = lo; k < hi; ++k)
                if (native[at + k] != before[at + k]) return false;
        }
        at += size + blocks;
    }
    return true;
}

// Original, then native from the same state; the original's results stay.
VerifyResult VerifyOne(NativeSpec& spec, CpuContext* ctx, const char* origin)
{
    thread_local std::vector<uint8_t> before, afterOriginal, afterNative;
    GuestRanges ranges;
    if (!spec.writes(*ctx, ranges) || !Verifiable(ranges)) {
        spec.body(ctx);
        return VerifyResult::Skipped;
    }
    const CpuContext in = *ctx;
    Capture(ranges, before);
    // NAT_INSN=1 (insn_count.h): game calls' original and native instructions, site "mtx.<name>"; a declined native
    // costs its attempt plus the original it falls back to.
    const bool counting = InsnCount::On() && std::strcmp(origin, "game") == 0;
    const uint64_t originalStart = counting ? InsnCount::Read() : 0;
    spec.body(ctx);
    const uint64_t originalInsns = counting ? InsnCount::Since(originalStart) : 0;
    const CpuContext original = *ctx;
    Capture(ranges, afterOriginal);
    *ctx = in;
    Restore(ranges, before);
    VerifyResult result = VerifyResult::Fallback;
    const uint64_t nativeStart = counting ? InsnCount::Read() : 0;
    const bool ranNatively = spec.native(ctx);
    if (counting) {
        const uint64_t nativeInsns = InsnCount::Since(nativeStart);
        if (spec.site == -2) spec.site = InsnCount::Site((std::string("mtx.") + spec.name).c_str());
        InsnCount::Add(spec.site, InsnCount::kOriginal, originalInsns);
        InsnCount::Add(spec.site, InsnCount::kNative, ranNatively ? nativeInsns : nativeInsns + originalInsns);
    }
    if (ranNatively) {
        Capture(ranges, afterNative);
        const bool same = SameRegisters(in, original, *ctx) && SameMemory(ranges, before, afterOriginal, afterNative);
        if (!same) LogMismatch(spec, origin, in, original, *ctx, ranges, afterOriginal, afterNative);
        result = same ? VerifyResult::Compared : VerifyResult::Mismatch;
    }
    *ctx = original;
    Restore(ranges, afterOriginal);
    return result;
}

void Count(NativeSpec& spec, VerifyResult result)
{
    switch (result) {
    case VerifyResult::Compared: spec.compared.fetch_add(1, std::memory_order_relaxed); break;
    case VerifyResult::Mismatch: spec.compared.fetch_add(1, std::memory_order_relaxed);
                                 spec.mismatches.fetch_add(1, std::memory_order_relaxed); break;
    case VerifyResult::Fallback: spec.fallbacks.fetch_add(1, std::memory_order_relaxed); break;
    case VerifyResult::Skipped: spec.skipped.fetch_add(1, std::memory_order_relaxed); break;
    }
}

// --- fuzzing -----------------------------------------------------------------------------------------------------

// The MEM2 mapping's top 8 KiB, past the game's 64 MiB, the module homes and the FST (in the IOS/IPC reserve,
// system_bridge.cpp): mapped, so even the original's checked paths (quantized GQRs) accept it, and unused.
constexpr uint32_t kFuzzScratchSize = 0x2000u;
constexpr uint32_t kFuzzScratch = Memory::kMem2CachedEnd - kFuzzScratchSize;

struct Fuzzer {
    uint64_t state;
    // Per trial: 0 finite values only (the fast passes' edges: subnormals, signed zeros, overflow), 1 also
    // infinities, 2 also NaNs.
    uint32_t classes = 2;
    uint32_t Next()
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return static_cast<uint32_t>(state >> 16);
    }
    uint32_t Below(uint32_t n) { return n ? Next() % n : 0; }
    // A float's bits, weighted towards the values that break naive implementations.
    uint32_t FloatBits()
    {
        const uint32_t sign = (Next() & 1u) << 31;
        uint32_t kind = Below(16);
        if ((kind == 2 && classes < 1) || ((kind == 3 || kind == 4) && classes < 2)) kind = 10;
        switch (kind) {
        case 0: return sign;                                              // ±0
        case 1: return sign | (1u + Below(0x007FFFFFu));                  // subnormal
        case 2: return sign | 0x7F800000u;                                // ±inf
        case 3: return sign | 0x7FC00000u | Below(0x00400000u);           // quiet NaN
        case 4: return sign | 0x7F800000u | (1u + Below(0x003FFFFFu));    // signalling NaN
        case 5: return sign | 0x7F7FFFFFu;                                // ±FLT_MAX
        case 6: return sign | (0x00800000u + Below(0x01000000u));         // just above FLT_MIN
        case 7: return sign | ((0xE0u + Below(0x1Fu)) << 23) | Below(0x00800000u);  // products overflow
        case 8: return sign | ((0x01u + Below(0x20u)) << 23) | Below(0x00800000u);  // products underflow
        case 9: return 0x3F800000u;                                       // 1.0
        default: return sign | ((0x70u + Below(0x20u)) << 23) | Below(0x00800000u); // ordinary magnitudes
        }
    }
    void Fill(uint32_t address, uint32_t bytes)
    {
        for (uint32_t i = 0; i < bytes; i += 4) {
            const uint32_t be = __builtin_bswap32(FloatBits());
            std::memcpy(Window() + Canonical(address + i), &be, 4);
        }
    }
    // A 4-byte aligned address in the scratch window with room for `span` bytes.
    uint32_t Address(uint32_t span) { return kFuzzScratch + 0x200u + 4u * Below((kFuzzScratchSize - 0x400u - span) / 4u); }
    // An address near `base`: often exactly on it, else a few words or matrices away (overlaps both ways).
    uint32_t Near(uint32_t base, uint32_t span)
    {
        static const int32_t kShifts[] = {0, 0, 4, -4, 8, -8, 16, -16, 40, -40, 44, 48, -48, 52, 92, 96, -96, 100, 144};
        const int32_t shift = kShifts[Below(sizeof(kShifts) / sizeof(kShifts[0]))];
        const uint32_t candidate = base + static_cast<uint32_t>(shift);
        if (candidate < kFuzzScratch + 0x200u || candidate + span > kFuzzScratch + kFuzzScratchSize - 0x200u)
            return Address(span);
        return candidate;
    }
};

void RandomizeRegisters(CpuContext& ctx, Fuzzer& f)
{
    for (uint32_t i = 0; i < 32; ++i) {
        const uint64_t hi = f.Next(), lo = f.Next();
        ctx.fpr[i].raw = (hi << 32) | lo;
    }
    for (uint32_t i : {0u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u}) ctx.gpr[i] = f.Next();
    ctx.ctr = f.Next();
    ctx.gpr[1] = kFuzzScratch + kFuzzScratchSize - 0x100u;  // frames below this land in the scratch window
    // Mostly the float encoding; sometimes a quantized one (types 4-7, any scale), which the natives hand back to
    // the original.
    ctx.gqr[0] = 0u;
    if (f.Below(8) == 0) {
        static const uint32_t kTypes[] = {0u, 4u, 5u, 6u, 7u};
        ctx.gqr[0] = ((f.Next() & 0x3Fu) << 24) | (kTypes[f.Below(5)] << 16) | ((f.Next() & 0x3Fu) << 8) | kTypes[f.Below(5)];
    }
}

void RunFuzz(NativeSpec& spec, const CpuContext& live, uint32_t trials)
{
    const uint32_t scratch = Canonical(kFuzzScratch);
    if (!Fits(kFuzzScratch, kFuzzScratchSize) || !PlainStores(scratch, kFuzzScratchSize) || !LeanReady()) {
        std::fprintf(stderr, "[hle-fuzz] %s: scratch window unavailable, skipped\n", spec.name);
        return;
    }
    GuestRanges whole;
    whole.Add(kFuzzScratch, kFuzzScratchSize);
    std::vector<uint8_t> saved;
    Capture(whole, saved);
    const uint32_t savedFpControl = MkwGetHostFpControl();
    Fuzzer f{0x9E3779B97F4A7C15ull ^ reinterpret_cast<uintptr_t>(spec.name)};
    uint64_t compared = 0, fallbacks = 0, skipped = 0, mismatches = 0, niTrials = 0;
    for (uint32_t t = 0; t < trials; ++t) {
        CpuContext ctx = live;
        f.classes = f.Below(3);
        RandomizeRegisters(ctx, f);
        f.Fill(kFuzzScratch, kFuzzScratchSize);
        spec.setup(ctx, f);
        const bool ni = (f.Next() & 1u) != 0;
        ctx.fpscr = (ctx.fpscr & ~4u) | (ni ? 4u : 0u);
        MkwApplyHostNiMode(ctx.fpscr);
        niTrials += ni;
        switch (VerifyOne(spec, &ctx, "fuzz")) {
        case VerifyResult::Compared: ++compared; break;
        case VerifyResult::Mismatch: ++compared; ++mismatches; break;
        case VerifyResult::Fallback: ++fallbacks; break;
        case VerifyResult::Skipped: ++skipped; break;
        }
        MkwRestoreHostMxcsr(savedFpControl);
    }
    Restore(whole, saved);
    std::lock_guard<std::mutex> lock(g_logMutex);
    std::fprintf(stderr, "[hle-fuzz] %s: trials %u compared %llu (NI on in %llu) fallbacks %llu skipped %llu mismatches %llu\n",
                 spec.name, trials, static_cast<unsigned long long>(compared), static_cast<unsigned long long>(niTrials),
                 static_cast<unsigned long long>(fallbacks), static_cast<unsigned long long>(skipped),
                 static_cast<unsigned long long>(mismatches));
}

uint32_t FuzzTrials()
{
    static const uint32_t trials = [] {
        const char* value = std::getenv("BRAWL_HLE_FUZZ");
        return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
    }();
    return trials;
}

void PrintSummary(const NativeSpec& spec)
{
    std::lock_guard<std::mutex> lock(g_logMutex);
    std::fprintf(stderr, "[hle-verify] %s: calls %llu compared %llu fallbacks %llu skipped %llu mismatches %llu\n",
                 spec.name, static_cast<unsigned long long>(spec.calls.load()),
                 static_cast<unsigned long long>(spec.compared.load()),
                 static_cast<unsigned long long>(spec.fallbacks.load()),
                 static_cast<unsigned long long>(spec.skipped.load()),
                 static_cast<unsigned long long>(spec.mismatches.load()));
}

// The non-native modes (g_mode != 0).
[[gnu::noinline, gnu::cold]] void SpecialCall(NativeSpec& spec, CpuContext* ctx)
{
    if (g_mode == kOriginals) return spec.body(ctx);
    if (FuzzTrials() != 0 && !spec.fuzzed.exchange(true)) RunFuzz(spec, *ctx, FuzzTrials());
    const uint64_t calls = spec.calls.fetch_add(1, std::memory_order_relaxed) + 1;
    Count(spec, VerifyOne(spec, ctx, "game"));
    if ((calls & (calls - 1)) == 0 ? calls <= 16384 : (calls & 0x3FFF) == 0) PrintSummary(spec);
}

// --- per-native write ranges and trial setups ----------------------------------------------------------------------

bool ConcatWrites(const CpuContext& ctx, GuestRanges& ranges)
{
    ranges.Add(ctx.gpr[5], 48);
    ranges.Add(ctx.gpr[1] - 128u, 128);  // its frame and below: only the back chain at r1 - 64 may change
    return true;
}

void ConcatSetup(CpuContext& ctx, Fuzzer& f)
{
    ctx.gpr[3] = f.Address(48);
    ctx.gpr[4] = f.Below(8) == 0 ? ctx.gpr[3] : f.Address(48);
    switch (f.Below(4)) {
    case 0: ctx.gpr[5] = ctx.gpr[3]; break;
    case 1: ctx.gpr[5] = ctx.gpr[4]; break;
    case 2: ctx.gpr[5] = f.Near(ctx.gpr[f.Below(2) ? 3 : 4], 48); break;
    default: ctx.gpr[5] = f.Address(48); break;
    }
}

bool ConcatArrayWrites(const CpuContext& ctx, GuestRanges& ranges)
{
    const uint32_t count = ctx.gpr[6];
    if (count < 2 || count > 0x10000u) return false;  // the original's bdnz never ends for count < 2
    ranges.Add(ctx.gpr[5], 48u * count);
    ranges.Add(ctx.gpr[1] - 160u, 160);
    return true;
}

void ConcatArraySetup(CpuContext& ctx, Fuzzer& f)
{
    const uint32_t count = 2u + f.Below(5);
    ctx.gpr[6] = count;
    ctx.gpr[3] = f.Address(48);
    ctx.gpr[4] = f.Address(48u * count);
    ctx.gpr[5] = f.Below(3) == 0 ? ctx.gpr[4] : f.Near(f.Below(2) ? ctx.gpr[4] : ctx.gpr[3], 48u * count);
    // The unit pair (r13 - 22976) in the scratch window: usually {0, 1}, else whatever Fill left there.
    const uint32_t unit = f.Address(8);
    ctx.gpr[13] = unit + 22976u;
    if (f.Below(2) == 0) {
        const uint32_t pair[2] = {0u, __builtin_bswap32(0x3F800000u)};
        std::memcpy(Window() + Canonical(unit), pair, sizeof(pair));
    }
}

bool CopyWrites(const CpuContext& ctx, GuestRanges& ranges)
{
    ranges.Add(ctx.gpr[4], 48);
    ranges.Add(ctx.gpr[1] - 64u, 64);  // no frame: nothing below r1 may change
    return true;
}

void CopySetup(CpuContext& ctx, Fuzzer& f)
{
    ctx.gpr[3] = f.Address(48);
    ctx.gpr[4] = f.Below(4) == 0 ? f.Address(48) : f.Near(ctx.gpr[3], 48);
}

bool InverseWrites(const CpuContext& ctx, GuestRanges& ranges)
{
    ranges.Add(ctx.gpr[4], 48);
    ranges.Add(ctx.gpr[1] - 64u, 64);  // no frame: nothing below r1 may change
    return true;
}

void InverseSetup(CpuContext& ctx, Fuzzer& f)
{
    const uint32_t src = f.Address(48);
    ctx.gpr[3] = src;
    ctx.gpr[4] = f.Below(4) == 0 ? src : f.Address(48);
    // Often a matrix of small integers (exact products), sometimes made singular: a zero row or column, or two
    // equal rows.
    if (f.Below(2) == 0) {
        float values[12];
        for (float& v : values) v = static_cast<float>(static_cast<int>(f.Below(9)) - 4);
        switch (f.Below(4)) {
        case 0: for (int c = 0; c < 4; ++c) values[4 * f.Below(3) + c] = 0.0f; break;
        case 1: { const uint32_t c = f.Below(3); for (int r = 0; r < 3; ++r) values[4 * r + c] = 0.0f; } break;
        case 2: for (int c = 0; c < 4; ++c) values[8 + c] = values[c]; break;
        default: break;
        }
        for (int i = 0; i < 12; ++i) {
            const uint32_t be = __builtin_bswap32(__builtin_bit_cast(uint32_t, values[i]));
            std::memcpy(Window() + Canonical(src + 4u * i), &be, 4);
        }
    }
}

bool LightPerspectiveWrites(const CpuContext& ctx, GuestRanges& ranges)
{
    ranges.Add(ctx.gpr[3], 48);
    ranges.Add(ctx.gpr[1] - 1024u, 1024 + 8);  // its frame, the caller's LR slot, and tanf's frames below
    return true;
}

void LightPerspectiveSetup(CpuContext& ctx, Fuzzer& f)
{
    ctx.gpr[3] = f.Address(48);
    // The arguments as doubles (single values, as the callers pass), occasionally arbitrary double bits.
    for (int i = 1; i <= 6; ++i) {
        if (f.Below(8) == 0) continue;  // keep RandomizeRegisters' raw bits
        ctx.fpr[i].d = static_cast<double>(__builtin_bit_cast(float, f.FloatBits()));
    }
}

#else  // !MKW_LEAN_MEMORY: no fast path.

bool ConcatNative(CpuContext*) { return false; }
bool ConcatArrayNative(CpuContext*) { return false; }
bool CopyNative(CpuContext*) { return false; }
bool InverseNative(CpuContext*) { return false; }
bool LightPerspectiveNative(CpuContext*) { return false; }
constexpr uint8_t g_mode = 0;
struct NativeSpec {};
void SpecialCall(NativeSpec&, CpuContext*) {}

#endif  // MKW_LEAN_MEMORY

#if MKW_LEAN_MEMORY
NativeSpec g_copySpec{"PSMTXCopy", func_801EC184_body, CopyNative, CopyWrites, CopySetup};
NativeSpec g_concatSpec{"PSMTXConcat", func_801EC1B8_body, ConcatNative, ConcatWrites, ConcatSetup};
NativeSpec g_concatArraySpec{"PSMTXConcatArray", func_801EC284_body, ConcatArrayNative, ConcatArrayWrites,
                             ConcatArraySetup};
NativeSpec g_inverseSpec{"PSMTXInverse", func_801EC41C_body, InverseNative, InverseWrites, InverseSetup};
NativeSpec g_lightPerspectiveSpec{"C_MTXLightPerspective", func_801ECBF0_body, LightPerspectiveNative,
                                  LightPerspectiveWrites, LightPerspectiveSetup};
#else
NativeSpec g_copySpec, g_concatSpec, g_concatArraySpec, g_inverseSpec, g_lightPerspectiveSpec;
#endif

}  // namespace

extern "C" void BrawlMtx_PSMTXCopy(CpuContext* ctx)
{
    if (g_mode != 0) [[unlikely]] return SpecialCall(g_copySpec, ctx);
    if (CopyNative(ctx)) [[likely]] return;
    [[clang::musttail]] return func_801EC184_body(ctx);
}

extern "C" void BrawlMtx_PSMTXConcat(CpuContext* ctx)
{
    if (g_mode != 0) [[unlikely]] return SpecialCall(g_concatSpec, ctx);
    if (ConcatNative(ctx)) [[likely]] return;
    [[clang::musttail]] return func_801EC1B8_body(ctx);
}

extern "C" void BrawlMtx_PSMTXConcatArray(CpuContext* ctx)
{
    if (g_mode != 0) [[unlikely]] return SpecialCall(g_concatArraySpec, ctx);
    if (ConcatArrayNative(ctx)) [[likely]] return;
    [[clang::musttail]] return func_801EC284_body(ctx);
}

extern "C" void BrawlMtx_PSMTXInverse(CpuContext* ctx)
{
    if (g_mode != 0) [[unlikely]] return SpecialCall(g_inverseSpec, ctx);
    if (InverseNative(ctx)) [[likely]] return;
    [[clang::musttail]] return func_801EC41C_body(ctx);
}

extern "C" void BrawlMtx_C_MTXLightPerspective(CpuContext* ctx)
{
    if (g_mode != 0) [[unlikely]] return SpecialCall(g_lightPerspectiveSpec, ctx);
    if (LightPerspectiveNative(ctx)) [[likely]] return;
    [[clang::musttail]] return func_801ECBF0_body(ctx);
}
