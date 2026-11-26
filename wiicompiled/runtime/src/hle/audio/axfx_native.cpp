#include "guest_memory_alias.h"
// axfx_native.cpp - Brawl's AXFX reverb and delay callbacks, natively (vanilla rev 1 main.dol):
//   AXFXReverbHiExpCallback 0x80202A58 (the hottest guest function in a match: ~7% of the game thread)
//   AXFXDelayCallback       0x80203A10
//
// Bit-exact with the translated originals (func_80202A58_body / func_80203A10_body are the spec; no C source of these
// SDK functions is public). Each native performs, per channel, the original's Broadway operations in the original's
// order with the same operands (isa/ppc_isa_float.h's fmuls/fadds/fsubs semantics, PPC_Fctiwz; memory_access.h's lfs
// widening and stfs narrowing - the shortcuts taken below are proven identities, see "Broadway single-precision
// arithmetic"), writes the same bytes to every guest location the original writes (delay lines, sample buffers, the
// effect struct, the whole stack frame including its dead slots), leaves the same dirty marks and returns the same
// register file. The translated code reloads every struct field and position per sample and channel, and resolves,
// policy-tests and dirty-marks each guest store; here each buffer is resolved, tested and marked once per call, the
// struct lives in locals, and the reverb runs its three channels' dependency chains interleaved.
//
// Each call first proves what that relies on: every position inside its line, every buffer resolvable as one host
// range (writable where written), and no overlap between the effect struct, the stack frame, the .sdata2 constants
// and the delay lines / sample buffers. The reverb also requires every written line or buffer to overlap nothing else
// (its interleaving reorders accesses across channels); the delay keeps the original's access order, so its lines
// and buffers may alias one another. Otherwise the call runs the translated original, so the result is exact either
// way. The game's effects meet these conditions on every call (BRAWL_HLE_VERIFY reports the fallbacks).
//
// Registration: REGISTER_NATIVE_FUNCTION_AS keeps the base translation (the fallback and the verifier call it). The
// translator's emit-build-shards (--native-source-dir runtime/src) then drops both addresses from the generated
// indirect-dispatch table and leaves their direct calls (AXFXReverbHiCallback 0x8020270C -> 0x80202A58, nw4r's FxDelay
// 0x801C0398 -> 0x80203A10) as InvokeDirectCpu, which reaches the registry's native winner. Re-emit the build shards
// whenever these registrations change: the runtime rejects a generated table entry whose winner is native. A function
// whose guest code a mod patched still runs in the interpreter (its translated guard).
//
// BRAWL_HLE_VERIFY=1: each call runs the translated original and the native from the same state and compares the
// register file, every byte the function can write and (lean memory) the dirty marks; mismatches are logged
// ("[audio] axfx ..."), and the original's results are kept. BRAWL_AXFX_NATIVE=0: run the translated originals.

#include "abi_bridge.h"
#include "axfx_async.h"
#include "brawl_dirty_memory.h"
#include "insn_count.h"
#include "memory.h"
#include "ppc_runtime.h"
#include "runtime_log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>  // the vector loops' masked loads and stores
#endif

#if defined(__clang__)
#pragma clang fp contract(off)  // discrete fmuls/fadds, as translated (CMake adds -ffp-contract=off for this file too)
#endif

extern "C" void func_80202A58(CpuContext* ctx);
extern "C" void func_80202A58_body(CpuContext* ctx);
extern "C" void func_80203A10(CpuContext* ctx);
extern "C" void func_80203A10_body(CpuContext* ctx);
// The generated guest_code_guard table (guest_code_patch.cpp): every translated function's start, ascending.
extern "C" const uint32_t g_mkwGuardedFunctionStarts[];
extern "C" const uint32_t g_mkwGuardedFunctionCount;
// Present only in a translation with MKW_EABI_REGS entries (vanilla); g3d_calc.cpp and mtx.cpp detect it the same way.
extern "C" __attribute__((weak)) uint64_t func_801D7AE4_eabi(CpuContext* ctx, uint32_t r3);

namespace {

constexpr uint32_t kSamples = 96;           // per AX frame (both callbacks loop 96 times)

// Audio buffers and delay lines are guest memory: arbitrary mods can read them.
// Use Broadway-exact processing by default. Approximate host-float processing
// requires AXFX_FAST=1; AXFX_EXACT=1 and verification always take precedence.
bool ExactAudio() {
    static const bool exact = [] {
        const char* exactEnv = std::getenv("AXFX_EXACT");
        const char* fastEnv = std::getenv("AXFX_FAST");
        const char* verifyEnv = std::getenv("BRAWL_HLE_VERIFY");
        const bool asked = exactEnv && std::strcmp(exactEnv, "1") == 0;
        const bool fast = fastEnv && std::strcmp(fastEnv, "1") == 0;
        const bool verify = verifyEnv && *verifyEnv && std::strcmp(verifyEnv, "0") != 0;
        RT_LOGF(RT_TAG_AUDIO, "axfx: %s reverb%s\n", asked || verify || !fast ? "exact (Broadway float)" : "fast (host float)",
                verify && !asked ? " for BRAWL_HLE_VERIFY" : "");
        return asked || verify || !fast;
    }();
    return exact;
}
constexpr uint32_t kBufferBytes = kSamples * 4;

// ---- guest memory through a proven host range ----

inline uint32_t Load32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return __builtin_bswap32(v);
}
inline uint64_t Load64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return __builtin_bswap64(v);
}
inline void Store32(uint8_t* p, uint32_t v) {
    v = __builtin_bswap32(v);
    std::memcpy(p, &v, 4);
}
// lfs / stfs (memory_access.h FlatReadFloat32Body / FlatWriteFloat32): exact widening and Broadway's narrowing.
// PpcConvertToDoubleBitsInline and ConvertPpcDoubleToSingleBits each call an out-of-line arm for the rare inputs
// (subnormals, NaN/inf; tiny values); with the default calling convention (MKW_OUTLINE_GUEST_OPS=0) even an untaken
// call makes the compiler keep every live double of the samples loop in memory, since all XMM registers are
// caller-saved. These are the same functions with those arms (PpcConvertToDoubleBitsSlow,
// ConvertPpcDoubleToSubnormalSingleBits) copied verbatim inline.
inline uint64_t ConvertToDoubleBits(uint32_t value) {
    float single;
    std::memcpy(&single, &value, sizeof(single));
#if MKW_FAST_FLOAT
    return PpcBitCastToU64Inline(static_cast<double>(single));
#endif
    const uint32_t magnitude = value & 0x7FFFFFFFu;
    if (magnitude - 0x00800000u < 0x7F000000u || magnitude == 0) [[likely]]
        return PpcBitCastToU64Inline(static_cast<double>(single));
    // PpcConvertToDoubleBitsSlow:
    uint64_t x = value;
    uint64_t exp = (x >> 23) & 0xFFu;
    uint64_t frac = x & 0x007FFFFFu;
    if (exp > 0 && exp < 255) {
        const uint64_t y = !(exp >> 7);
        const uint64_t z = (y << 61) | (y << 60) | (y << 59);
        return ((x & 0xC0000000ULL) << 32) | z | ((x & 0x3FFFFFFFULL) << 29);
    }
    if (exp == 0 && frac != 0) {
        exp = 1023 - 126;
        do {
            frac <<= 1;
            --exp;
        } while ((frac & 0x00800000u) == 0);
        return ((x & 0x80000000ULL) << 32) | (exp << 52) | ((frac & 0x007FFFFFULL) << 29);
    }
    const uint64_t y = exp >> 7;
    const uint64_t z = (y << 61) | (y << 60) | (y << 59);
    return ((x & 0xC0000000ULL) << 32) | z | ((x & 0x3FFFFFFFULL) << 29);
}
inline uint32_t ConvertToSingleBits(double value) {
#if MKW_FAST_FLOAT
    const float single = static_cast<float>(value);
    uint32_t singleBits = 0;
    std::memcpy(&singleBits, &single, sizeof(singleBits));
    return singleBits;
#endif
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = static_cast<uint32_t>((bits >> 52) & 0x7FFu);
    if (exponent - 874u <= 22u) [[unlikely]] {
        // ConvertPpcDoubleToSubnormalSingleBits:
        uint32_t narrowed = static_cast<uint32_t>(0x80000000ULL | ((bits & 0x000FFFFFFFFFFFFFULL) >> 21));
        narrowed >>= (905u - exponent);
        narrowed |= static_cast<uint32_t>((bits >> 32) & 0x80000000ULL);
        return narrowed;
    }
    return static_cast<uint32_t>(((bits >> 32) & 0xC0000000ULL) | ((bits >> 29) & 0x3FFFFFFFULL));
}
inline double Widen(uint32_t bits) { return PpcBitCastToDoubleInline(ConvertToDoubleBits(bits)); }
inline uint32_t Narrow(double value) { return ConvertToSingleBits(value); }
inline double LoadF(const uint8_t* p) { return Widen(Load32(p)); }

// ---- Broadway single-precision arithmetic ----
// The translated code spells fmuls as PpcFmulsInline(a, c) and fadds/fsubs as
// static_cast<double>(PpcForceSingleValueInline(PpcFadd/FsubInline(a, b))). Two facts give identical, shorter forms:
//  - Every operand the reverb feeds to arithmetic is single-exact: an lfs widening, or a previous single result
//    (static_cast<double> of a float). PpcFmulsBody first rounds C to 25 bits (PpcForce25BitInline, skipped for NaN),
//    which changes only the low 29 fraction bits of a normal double and otherwise acts only on double subnormals; a
//    single-exact C has those bits zero and is never a double subnormal, so the rounding is the identity and fmuls is
//    single(PpcFmulInline(a, c)). (Under MKW_FAST_FLOAT both sides are double(float(a * c)).)
//  - PpcForceSingleValueBody masks a value to its sign when |v| < g_mkwNiFlushThreshold, which is 0.0 unless FPSCR[NI]
//    is set: then the mask is never set and the body is static_cast<float>(v). Ni = false is used only for a call that
//    starts with the threshold at 0.0 (nothing in the call changes it); Ni = true keeps the body.
// Single-exactness also makes an stfs/lfs round trip of a chain value (Narrow, then Widen) the identity - including
// NaNs and subnormal singles - so the reverb keeps its stack temporaries and low-pass state as doubles between samples
// and narrows them once, for the stores that leave them in guest memory.
template <bool Ni>
inline double Single(double v) {
    if constexpr (Ni) return static_cast<double>(PpcForceSingleValueInline(v));
    else return static_cast<double>(static_cast<float>(v));
}
// PpcFmul/Fadd/FsubBody: the host operation, then PpcResolveNan2Inline's guest-order NaN. Its out-of-line arm
// (PpcResolveNanSlow: the first NaN operand, quieted, else the default NaN; the third operand is 0.0 here) is copied
// inline for the reason given at ConvertToDoubleBits.
inline double ResolveNan2(double result, double a, double b) {
#if MKW_FAST_FLOAT
    (void)a;
    (void)b;
    return result;
#else
    if (!std::isnan(result)) [[likely]] return result;
    constexpr uint64_t quiet = 0x0008000000000000ull;
    if (std::isnan(a)) return PpcBitCastToDoubleInline(PpcBitCastToU64Inline(a) | quiet);
    if (std::isnan(b)) return PpcBitCastToDoubleInline(PpcBitCastToU64Inline(b) | quiet);
    return PpcBitCastToDoubleInline(0x7FF8000000000000ull);
#endif
}
template <bool Ni> inline double Fmuls(double a, double c) { return Single<Ni>(ResolveNan2(a * c, a, c)); }
template <bool Ni> inline double Fadds(double a, double b) { return Single<Ni>(ResolveNan2(a + b, a, b)); }
template <bool Ni> inline double Fsubs(double a, double b) { return Single<Ni>(ResolveNan2(a - b, a, b)); }

// A single result kept as the float it was rounded to: its stfs bits are the float's own. (Narrow of
// static_cast<double>(f) gives back f's bits for every float, NaNs included; with NI the flush leaves no subnormal f,
// so the widening is exact under DAZ too.) This skips Broadway's integer narrowing for the stores to the lines.
struct SingleResult {
    float f;
    double d() const { return static_cast<double>(f); }
};
template <bool Ni>
inline SingleResult RoundF(double v) {
    if constexpr (Ni) return {PpcForceSingleValueInline(v)};
    else return {static_cast<float>(v)};
}
template <bool Ni> inline SingleResult FaddsF(double a, double b) { return RoundF<Ni>(ResolveNan2(a + b, a, b)); }
template <bool Ni> inline SingleResult FsubsF(double a, double b) { return RoundF<Ni>(ResolveNan2(a - b, a, b)); }
inline void StoreSingle(uint8_t* p, SingleResult value) {
    uint32_t bits;
    std::memcpy(&bits, &value.f, 4);
    Store32(p, bits);
}
inline uint32_t Fctiwz(double value, double& fpr) {
    fpr = PPC_Fctiwz(value);
    return PPC_FprLowWordInline(fpr);
}

// A position's step: the guest stores pos + 1, then 0 unless pos + 1 < length (unsigned).
inline uint32_t Next(uint32_t pos, uint32_t length) { return pos + 1 < length ? pos + 1 : 0; }

// The dirty marks the translated 4-byte stores to [start, start + bytes) leave. Lean memory: each store marks its
// 64-byte block and the next (LeanTryStore / LeanStoreRamBody); the other models mark the blocks a store touches.
// The range lies in one view (Resolve), so its canonical offsets are contiguous.
inline void MarkStores4(uint32_t start, uint64_t bytes) {
    if (bytes == 0) return;
#if MKW_LEAN_MEMORY && !MKW_LEAN_DIRTY_MARKS
    (void)start;  // no dirty marks in this build (no translated store marks either; rollback refuses to run)
#elif MKW_LEAN_MEMORY
    const uint32_t canonical = GuestMemoryAlias::Offset(start);
    const uint32_t first = canonical >> 6;
    const uint32_t last = ((canonical + static_cast<uint32_t>(bytes) - 4) >> 6) + 1;
    std::memset(MemoryInline::g_leanBase + MemoryInline::kLeanDirtyOffset + first, 1, last - first + 1);
#else
    BrawlDirtyMemory::Mark(start, static_cast<size_t>(bytes));
#endif
}

// A guest range resolved to one host range (memory_access.h ResolveRangeHost: mapped, within one view, and - when
// written - clear of the store policy, i.e. not MMIO or executable).
struct Span {
    uint32_t start = 0;
    uint32_t bytes = 0;
    uint8_t* host = nullptr;
};

inline bool Resolve(Span& span, uint32_t start, uint64_t bytes, bool write) {
    if (bytes == 0 || bytes > MemoryInline::kPageSize) return false;
    const uint32_t last = start + static_cast<uint32_t>(bytes - 1);
    if (last < start || ((start ^ last) & ~0x3FFFFFFFu) != 0) return false;  // wraps, or straddles two views
    span.start = start;
    span.bytes = static_cast<uint32_t>(bytes);
    span.host = MemoryInline::ResolveRangeHost(start, 0, span.bytes, write);
    return span.host != nullptr;
}

// Alias test on the lean window offset, which folds the physical, cached and uncached views together (memory_access.h).
// For the other memory models it is conservative: a false overlap only sends the call to the translated original.
inline bool Overlap(const Span& a, const Span& b) {
    const uint32_t ca = GuestMemoryAlias::Offset(a.start), cb = GuestMemoryAlias::Offset(b.start);
    return ca < cb + b.bytes && cb < ca + a.bytes;
}

// A delay line walked one position per sample from `pos`. Proven pos < length, so every visited position is in
// [0, length) and lies in `span`, the bounding range of the visited positions (one lap of 96 or the whole line).
struct Ring {
    uint32_t line = 0;
    uint32_t pos = 0;
    uint32_t length = 0;
    uintptr_t host0 = 0;  // host address of position 0 (span.host - 4 * the span's first position)
    Span span;

    uint8_t* At(uint32_t position) const { return reinterpret_cast<uint8_t*>(host0 + uintptr_t{4} * position); }
};

bool SetupRing(Ring& ring, uint32_t line, uint32_t pos, uint32_t length, bool write) {
    if (pos >= length) return false;  // the guest's first access would land past the line: run the original
    ring.line = line;
    ring.pos = pos;
    ring.length = length;
    const uint64_t end = uint64_t{pos} + kSamples;
    const uint32_t lo = end <= length ? pos : 0;
    const uint64_t hi = end <= length ? end : length;
    // The guest address is line + (position << 2) mod 2^32; Resolve proves [line + 4 lo, + 4 (hi - lo)) is one range.
    if (!Resolve(ring.span, line + (lo << 2), (hi - lo) * 4, write)) return false;
    ring.host0 = reinterpret_cast<uintptr_t>(ring.span.host) - uintptr_t{4} * lo;
    return true;
}

// The marks of the ring's stores: one per visited position, [pos, min(pos + 96, length)) then laps from 0.
void MarkRing(const Ring& ring) {
    const uint64_t end = uint64_t{ring.pos} + kSamples;
    const uint64_t firstEnd = std::min<uint64_t>(end, ring.length);
    MarkStores4(ring.line + (ring.pos << 2), (firstEnd - ring.pos) * 4);
    if (end > ring.length) MarkStores4(ring.line, std::min<uint64_t>(ring.length, end - ring.length) * 4);
}

// Sample buffers and delay lines may not overlap the ranges the natives keep in locals or write only at the end.
struct Pinned {
    const Span* frame;
    const Span* effect;
    const Span* constants;  // null for the delay (it reads no constants)
};

inline bool Clear(const Span& span, bool write, const Pinned& pinned) {
    if (Overlap(span, *pinned.frame) || Overlap(span, *pinned.effect)) return false;
    return !(write && pinned.constants && Overlap(span, *pinned.constants));
}

int64_t GuardIndexOf(uint32_t address) {
    const uint32_t* begin = g_mkwGuardedFunctionStarts;
    const uint32_t* end = begin + g_mkwGuardedFunctionCount;
    const uint32_t* it = std::lower_bound(begin, end, address);
    return it != end && *it == address ? it - begin : -1;
}

// A guest range a call touches, in canonical lean offsets [start, end) (Overlap's folding of the views).
struct Segment {
    uint32_t start, end;
    bool write;
};

inline void AddSegment(Segment* segments, uint32_t& count, const Span& span, bool write) {
    const uint32_t start = GuestMemoryAlias::Offset(span.start);
    segments[count++] = {start, start + span.bytes, write};
}

// The delay's samples loop state (Delay::Run below), here for the worker's jobs.
namespace Delay {
// The 96 samples: the buffers and lines, and what Run's final stores and registers take from the last sample.
struct Loop {
    uint8_t* in[3];
    Ring line[3];
    uint32_t pos[3], length[3], feedback[3], output[3];
    uint32_t b = 0, d = 0, written2 = 0, inc2 = 0, product2 = 0;
};
}  // namespace Delay

// ======================================================================================================================
// AXFXReverbHiExpCallback(AXFX_BUFFERUPDATE* r3, AXFX_REVERBHI_EXP* r4)
// ======================================================================================================================
namespace Reverb {

constexpr uint32_t kAddress = 0x80202A58u;
constexpr uint32_t kFrameBytes = 112;                // stwu r1, -112(r1)
constexpr uint32_t kFrameSpan = kFrameBytes + 8;     // the frame, the caller's back chain and its LR save word

// AXFX_REVERBHI_EXP
constexpr uint32_t kEarlyLine = 0x00;          // f32* [3 channels]
constexpr uint32_t kEarlyPos = 0x0C;           // u32 [3 taps]
constexpr uint32_t kEarlyLength = 0x18;
constexpr uint32_t kEarlyCoef = 0x20;          // f32 [3 taps]
constexpr uint32_t kPreDelayLine = 0x2C;       // f32* [3 channels]
constexpr uint32_t kPreDelayPos = 0x38;
constexpr uint32_t kPreDelayLength = 0x3C;
constexpr uint32_t kCombLine = 0x44;           // f32* [3 channels][3 combs]
constexpr uint32_t kCombPos = 0x68;            // u32 [3 combs]
constexpr uint32_t kCombLength = 0x74;         // u32 [3 combs]
constexpr uint32_t kCombCoef = 0x8C;           // f32 [3 combs]
constexpr uint32_t kAllpassLine = 0x98;        // f32* [3 channels][2 allpasses]
constexpr uint32_t kAllpassPos = 0xB0;         // u32 [2]
constexpr uint32_t kAllpassLength = 0xB8;      // u32 [2]
constexpr uint32_t kLastAllpassLine = 0xC8;    // f32* [3 channels]
constexpr uint32_t kLastAllpassPos = 0xD4;     // u32 [3 channels]
constexpr uint32_t kLastAllpassLength = 0xE0;  // u32 [3 channels]
constexpr uint32_t kAllpassCoef = 0xF8;
constexpr uint32_t kLastLpfOut = 0xFC;         // f32 [3 channels]
constexpr uint32_t kLpfCoef = 0x108;
constexpr uint32_t kActive = 0x10C;            // nonzero: skip this frame (the callback clears bit 1 and returns)
constexpr uint32_t kCrosstalk = 0x12C;
constexpr uint32_t kFusedGain = 0x134;
constexpr uint32_t kBusIn = 0x138;             // AXFX_BUS* (3 sample buffers)
constexpr uint32_t kBusOut = 0x13C;
constexpr uint32_t kOutGain = 0x140;
constexpr uint32_t kSendGain = 0x144;
constexpr uint32_t kEffectBytes = 0x148;

// .sdata2 (r2-relative): 0.0f, 1.0f, two gain scales, then the int->double bias 0x4330000080000000.
constexpr int32_t kConstantsOffset = -23044;
constexpr uint32_t kConstantsBytes = 28;
constexpr uint32_t kZero = 0, kOne = 4, kFusedScale = 8, kCrosstalkScale = 12, kBias = 20;

std::atomic<uint64_t> g_bypassed{0};  // native calls that took the skip path (statistics)

// The guest ranges the samples loop touches, for the disjointness sweep.
constexpr uint32_t kMaxSegments = 3 * (3 + 3 + 1 + 3 + 2 + 1);  // per channel: 3 buffers, 3 taps, 7 lines

// True when no written range overlaps any other range. The samples loop interleaves the three channels stage by stage,
// which reorders accesses across channels; with every write disjoint from everything else the order is unobservable.
bool Disjoint(Segment* segments, uint32_t count) {
    std::sort(segments, segments + count, [](const Segment& a, const Segment& b) { return a.start < b.start; });
    uint32_t endAll = 0, endWritten = 0;  // the furthest end of the ranges so far, and of the written ones
    for (uint32_t k = 0; k < count; ++k) {
        const Segment& s = segments[k];
        if (s.start < (s.write ? endAll : endWritten)) return false;
        endAll = std::max(endAll, s.end);
        if (s.write) endWritten = std::max(endWritten, s.end);
    }
    return true;
}

// One call's proven state: resolved buffers and lines, positions, and the effect's coefficients; Process runs the 96
// samples on it and leaves what the final stores and the register file need.
struct Plan {
    const uint8_t* s;          // the effect struct (host)
    const uint8_t* constants;  // the .sdata2 constants (host)
    bool busIn, busOut, preDelay;
    Span in[3], bi[3], bo[3];
    Ring early[3][3], pre[3], comb[3][3], ap[3][2], last[3];
    uint32_t earlyPos[3], earlyLength, preDelayPos, preDelayLength;
    uint32_t combPos[3], combLength[3], apPos[2], apLength[2], lastPos[3], lastLength[3];
    // Results.
    double lpf[3];  // the low-pass state (single-exact; its struct word is Narrow of it)
    double mix[3];  // each channel's output sample, the single the original keeps at r1+20 (new frame)
    double f[14];   // f0-f13 at the original's return
    uint32_t lastConvert;  // the low word of the last int->double conversion (r1+60 or r1+68)
    uint32_t lastApInc;    // r8 at the original's return: the second allpass position + 1 before its wrap
};

}  // namespace Reverb

// ======================================================================================================================
// AXFX_ASYNC: the samples loops on a worker thread
// ======================================================================================================================
// On the Wii these callbacks run inside the AI interrupt (__AXOutNewFrame -> __AXProcessAux -> the bus's callback), so
// the game never sees an effect half processed. Here a call does on the game thread everything that does not depend
// on the samples: Run's proofs, the effect struct's new positions, the stack frame, the register file. It hands the
// 96 samples to a worker as a job. The job runs the same samples loop (ProcessFastV, ProcessFast, Process, LoopVector,
// LoopScalar) on the same inputs: the lines, the bus buffers, a copy of the struct's parameters taken at the call. A
// reverb job also writes the struct's low-pass words. Join() waits until every job handed over is done and applies
// what the call could not leave itself (below).
//
// Who else reads or writes what a job reads or writes, and where they join (vanilla rev 1 main.dol):
// - The DSP. The next AX frame's command list reads the processed buffer (DispatchCommandList), and the mixer publishes
//   its aux uploads into the bus buffers (JoinMix). AxDspHle's JoinMix joins first. It runs before every AI block (both
//   AI tick paths), and on DSP init, stop and reconfiguration and on ARAM remaps. DispatchCommandList joins as well.
// - The next callback of the same effect: in the next AI block, after that join.
// - Another callback in this AX frame whose ranges overlap a pending job's (a second effect on the same bus, or an
//   effect reading another's AXFX_BUS): Run joins first (Async::Overlaps / Conflicts).
// - The effects' owners. These functions join first (MKW_NATIVE_BODY, at the end of this file):
//   - AXFXReverbHi Init, Shutdown and Settings, and the ReverbHiExp functions they call;
//   - AXFXDelay Init, Settings and Shutdown;
//   - AXRegisterAux{A,B,C}Callback;
//   - __AXAuxInit.
//   No other code reaches an effect's struct, its lines or the aux buffers outside the callbacks:
//   - nw4r's FxReverbHi and FxDelay, and Brawl's own reverb (0x80270ED8, 0x80272658), call those functions;
//   - the lines are allocated and freed only inside them;
//   - the aux buffers (0x804E4960, three buses by three frames) are touched only by __AXProcessAux (its own bus's
//     callback; memset only for a bus with no callback), __AXAuxInit and the DSP. The DC range operations around each
//     callback notify GX only and read nothing.
// - Whole-memory readers and writers join first: det-hash images (det_hash.cpp), savestates (save_states.cpp), and
//   rollback's Save, Load and hashes (brawl_rollback.cpp).
// - A translated original (fallback, verifier, a mod's patched guard): Dispatch joins first.
// What the original leaves that depends on the samples, and a call cannot leave in time:
// - The reverb's f2 and f7-f13. They are volatile, and its callers never read them. An asynchronous call leaves
//   them as they were.
// - Six dead words of the reverb's stack frame: r1-92 .. r1-72, the last sample's mix and f9, f8, f7. The call
//   writes a sentinel there. Join writes the job's values over any sentinel still in place; a guest store since
//   then wins, as it would have over the original's values. This is exact unless guest code reads those dead words in
//   between.
// The delay leaves everything exactly. Its returned values come from the last sample, which reads only memory no
// earlier sample wrote, provided its lines 1 and 2 are at least 96 long (required for an asynchronous call).
//
// AXFX_ASYNC selects the mode:
// - 0: off (default). Mods may access effect buffers outside the known join sites.
// - 1: on (explicit opt-in for a workload whose access paths have been audited).
// - check: no worker. Each job runs at its join. From its hand-over until then its written ranges hold a poison
//   pattern, so any early reader gets garbage, and every range it touches is checked unchanged at the join, so any
//   writer is counted ("[audio] axfx async check").
// AXFX_ASYNC_SPIN_US (30) sets how long the worker spins for the next job before it parks.
namespace Async {

enum class Mode : uint8_t { kOff, kOn, kCheck };
Mode GetMode();  // defined with the configuration below
inline bool On() { return GetMode() != Mode::kOff; }

enum class Kind : uint8_t { kReverb, kDelay };
enum class Samples : uint8_t { kVector, kFast, kExact, kExactNi, kScalar };  // the samples loop the call chose

constexpr uint32_t kSlots = 16;  // jobs handed over and not yet joined (an AX frame hands over one per effect)
// The reverb frame's dead words that depend on the samples (offsets from the new r1): the last sample's mix (three),
// then f9, f8, f7. A call leaves kSentinel + k in word k until Join.
constexpr uint32_t kFillWords = 6;
constexpr uint32_t kFillOffsets[kFillWords] = {20, 24, 28, 32, 36, 40};
constexpr uint32_t kSentinel = 0x7FA5F100u;
constexpr uint32_t kMaxRanges = 48;  // a job's guest ranges, for AXFX_ASYNC=check (a reverb has 40 at most)
constexpr uint32_t kMaxExpect = 32;  // a call's own results, for AXFX_ASYNC=check (a reverb has 26)

struct Range {
    uint32_t guest;
    uint8_t* host;
    uint32_t bytes;
    bool write;
};

struct Job {
    Kind kind = Kind::kReverb;
    Samples samples = Samples::kVector;
    uint32_t fpControl = 0;  // the game thread's MXCSR or FPCR at the call (the worker applies it)
    // The reverb. The plan's s and constants point at the copies here, so the job reads no guest struct.
    Reverb::Plan plan{};
    uint8_t effect[Reverb::kEffectBytes] = {};
    uint8_t constants[Reverb::kConstantsBytes] = {};
    uint8_t* lpf = nullptr;    // host: the struct's kLastLpfOut words (the job stores them)
    uint8_t* frame = nullptr;  // host: the frame's first word (new r1), for the dead words
    uint32_t fill[kFillWords] = {};
    // The delay.
    Delay::Loop loop{};
    // AXFX_ASYNC=check: every range the job touches, and their bytes at the hand-over; and what the call left from
    // its own computation of the loop's results (positions, returned values), compared with the loop's at the join.
    Range ranges[kMaxRanges];
    uint32_t rangeCount = 0;
    std::vector<uint8_t> saved;
    uint32_t expect[kMaxExpect] = {};
    uint32_t expectCount = 0;

    void AddRange(const Span& span, bool write) {
        if (rangeCount < kMaxRanges) ranges[rangeCount++] = {span.start, span.host, span.bytes, write};
    }
    void SetExpect(const uint32_t* values, uint32_t count) {
        expectCount = std::min(count, kMaxExpect);
        std::memcpy(expect, values, expectCount * sizeof(uint32_t));
    }
};

// Game thread only (the guest's fibers), like the callbacks.
// A slot for the next job; joins first when every slot is pending.
Job& Acquire();
// Records the next job's footprint (Reverb::Footprint: its buffers, a range covering its lines, the low-pass words) as
// pending, so that later calls' Overlaps and Conflicts see it. Coarse on purpose: a false overlap only joins early.
void Hold(const Segment* segments, uint32_t count);
// Hands the job over.
void Push(Job& job);
// Counts a call that ran its samples loop in place although asynchronous calls are on (statistics).
void InPlace(Kind kind);
// Whether jobs are handed over and not yet joined.
bool Pending();
// Whether a range, or any of a call's ranges, overlaps a pending job's.
bool Overlaps(uint32_t start, uint32_t bytes);
bool Conflicts(const Segment* segments, uint32_t count);
void Join();

}  // namespace Async

namespace Reverb {

// The 96 samples, channels interleaved stage by stage (three independent dependency chains instead of one long one).
// Within each channel every operation, operand order and memory access is the original's (loc_80202B14..80202F54).
template <bool Ni>
void Process(Plan& p) {
    const uint8_t* s = p.s;
    const uint8_t* k2 = p.constants;
    // loc_80202AD8's constants (f-register names are the original's).
    const double f1 = LoadF(s + kLpfCoef);
    const double f4 = Fsubs<Ni>(LoadF(k2 + kOne), f1);
    const double f5 = Fmuls<Ni>(LoadF(k2 + kFusedScale), LoadF(s + kFusedGain));
    const double f3 = LoadF(s + kAllpassCoef);
    const double f6 = Fmuls<Ni>(LoadF(k2 + kCrosstalkScale), LoadF(s + kCrosstalk));
    const double f0 = PpcBitCastToDoubleInline(Load64(k2 + kBias));
    const double zero = LoadF(k2 + kZero);
    const double outGain = LoadF(s + kOutGain), sendGain = LoadF(s + kSendGain);
    double earlyCoef[3], combCoef[3], lpf[3];
    for (uint32_t k = 0; k < 3; ++k) {
        earlyCoef[k] = LoadF(s + kEarlyCoef + 4 * k);
        combCoef[k] = LoadF(s + kCombCoef + 4 * k);
        lpf[k] = LoadF(s + kLastLpfOut + 4 * k);
    }
    uint32_t earlyPos[3], combPos[3], apPos[2], lastPos[3], preDelayPos = p.preDelayPos;
    for (uint32_t k = 0; k < 3; ++k) {
        earlyPos[k] = p.earlyPos[k];
        combPos[k] = p.combPos[k];
        lastPos[k] = p.lastPos[k];
    }
    apPos[0] = p.apPos[0];
    apPos[1] = p.apPos[1];

    double f2 = 0, f7 = 0, f8 = 0, f9 = 0, f10 = 0, f11 = 0, f12 = 0, f13 = 0, mix[3] = {};
    uint32_t convert = 0, lastApInc = 0;
    for (uint32_t i = 0; i < kSamples; ++i) {
        SingleResult input[3];
        double early[3], dry[3], combSum[3], a7[3], a8[3], a9[3];
        for (uint32_t c = 0; c < 3; ++c) {  // the input as a float: (in [+ bus in]) through the 2^52 + 2^31 bias
            const uint32_t sample = Load32(p.in[c].host + 4 * i);
            convert = (p.busIn ? sample + Load32(p.bi[c].host + 4 * i) : sample) ^ 0x80000000u;
            input[c] = FsubsF<Ni>(PpcBitCastToDoubleInline((uint64_t{0x43300000u} << 32) | convert), f0);
        }
        for (uint32_t c = 0; c < 3; ++c) {  // early reflections: three taps of the channel's line
            double a = LoadF(p.early[c][0].At(earlyPos[0]));
            double b = LoadF(p.early[c][1].At(earlyPos[1]));
            a = Fmuls<Ni>(earlyCoef[0], a);
            b = Fmuls<Ni>(earlyCoef[1], b);
            const double d = Fmuls<Ni>(earlyCoef[2], LoadF(p.early[c][2].At(earlyPos[2])));
            early[c] = Fadds<Ni>(d, Fadds<Ni>(a, b));
        }
        for (uint32_t c = 0; c < 3; ++c) {  // pre-delay
            if (p.preDelay) {
                uint8_t* q = p.pre[c].At(preDelayPos);
                dry[c] = LoadF(q);
                StoreSingle(q, input[c]);
            } else {
                dry[c] = input[c].d();
            }
            combSum[c] = zero;
        }
        for (uint32_t k = 0; k < 3; ++k) {  // three parallel combs
            for (uint32_t c = 0; c < 3; ++c) {
                uint8_t* q = p.comb[c][k].At(combPos[k]);
                const double old = LoadF(q);
                const double fed = Fmuls<Ni>(old, combCoef[k]);
                combSum[c] = Fadds<Ni>(combSum[c], old);
                StoreSingle(q, FaddsF<Ni>(dry[c], fed));
            }
        }
        for (uint32_t c = 0; c < 3; ++c) {  // first allpass
            uint8_t* q = p.ap[c][0].At(apPos[0]);
            const double old = LoadF(q);
            const SingleResult w = FaddsF<Ni>(combSum[c], Fmuls<Ni>(old, f3));
            StoreSingle(q, w);
            a7[c] = Fmuls<Ni>(w.d(), f3);
            a8[c] = old;
        }
        for (uint32_t c = 0; c < 3; ++c) {  // second allpass
            const double x = Fsubs<Ni>(a8[c], a7[c]);
            uint8_t* q = p.ap[c][1].At(apPos[1]);
            const double old = LoadF(q);
            const SingleResult w = FaddsF<Ni>(x, Fmuls<Ni>(old, f3));
            a8[c] = Fmuls<Ni>(w.d(), f3);
            StoreSingle(q, w);
            a9[c] = old;
        }
        for (uint32_t c = 0; c < 3; ++c) {  // low-pass
            const double x = Fsubs<Ni>(a9[c], a8[c]);
            const double held = Fmuls<Ni>(f1, lpf[c]);
            lpf[c] = Fadds<Ni>(Fmuls<Ni>(f4, x), held);
        }
        for (uint32_t c = 0; c < 3; ++c) {  // the channel's last allpass, then its output sample
            uint8_t* q = p.last[c].At(lastPos[c]);
            const double old = LoadF(q);
            const SingleResult w = FaddsF<Ni>(lpf[c], Fmuls<Ni>(old, f3));
            StoreSingle(q, w);
            const double out = Fsubs<Ni>(old, Fmuls<Ni>(w.d(), f3));
            lastPos[c] = Next(lastPos[c], p.lastLength[c]);
            mix[c] = Fadds<Ni>(Fmuls<Ni>(out, f5), early[c]);
        }
        // Crosstalk mix and output (loc_80202D38).
        f12 = mix[1];
        f11 = mix[2];
        f10 = mix[0];
        f2 = Fadds<Ni>(f12, f11);
        f7 = Fadds<Ni>(f10, f11);
        f9 = Fmuls<Ni>(f2, f6);
        f8 = Fmuls<Ni>(f7, f6);
        f13 = Fadds<Ni>(f10, f12);
        f9 = Fadds<Ni>(f10, f9);
        f8 = Fadds<Ni>(f12, f8);
        f7 = Fmuls<Ni>(f13, f6);
        f2 = Fmuls<Ni>(f9, outGain);
        f7 = Fadds<Ni>(f11, f7);
        Store32(p.in[0].host + 4 * i, Fctiwz(f2, f2));
        Store32(p.in[1].host + 4 * i, Fctiwz(Fmuls<Ni>(f8, outGain), f2));
        Store32(p.in[2].host + 4 * i, Fctiwz(Fmuls<Ni>(f7, outGain), f2));
        if (p.busOut) {
            Store32(p.bo[0].host + 4 * i, Fctiwz(Fmuls<Ni>(f9, sendGain), f2));
            Store32(p.bo[1].host + 4 * i, Fctiwz(Fmuls<Ni>(f8, sendGain), f2));
            Store32(p.bo[2].host + 4 * i, Fctiwz(Fmuls<Ni>(f7, sendGain), f2));
        }
        for (uint32_t k = 0; k < 3; ++k) earlyPos[k] = Next(earlyPos[k], p.earlyLength);
        if (p.preDelay) preDelayPos = Next(preDelayPos, p.preDelayLength);
        for (uint32_t k = 0; k < 3; ++k) combPos[k] = Next(combPos[k], p.combLength[k]);
        apPos[0] = Next(apPos[0], p.apLength[0]);
        lastApInc = apPos[1] + 1;
        apPos[1] = Next(apPos[1], p.apLength[1]);
    }

    for (uint32_t k = 0; k < 3; ++k) {
        p.earlyPos[k] = earlyPos[k];
        p.combPos[k] = combPos[k];
        p.lastPos[k] = lastPos[k];
        p.lpf[k] = lpf[k];
        p.mix[k] = mix[k];
    }
    p.apPos[0] = apPos[0];
    p.apPos[1] = apPos[1];
    p.preDelayPos = preDelayPos;
    p.lastConvert = convert;
    p.lastApInc = lastApInc;
    const double f[14] = {f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13};
    std::memcpy(p.f, f, sizeof(f));
}

// ---- Approximate samples loop (explicit AXFX_FAST=1 only) ----
// The same algorithm, operations and order in plain host single precision, with subnormals flushed to zero for the
// call (no denormal stalls while a tail decays to silence) and a saturating float-to-int output. For single-precision
// operands Broadway's fmuls/fadds/fsubs (an exact double product or sum, then one rounding to single) equal IEEE single
// operations, so this computes Process's values except at subnormals, NaN payloads and FPSCR[NI]'s flush threshold.
inline float LoadFloat(const uint8_t* p) {
    const uint32_t bits = Load32(p);
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
}
inline void StoreFloat(uint8_t* p, float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    Store32(p, bits);
}
// fctiwz's saturation: below the range or NaN -> 0x80000000, from 2^31 up -> 0x7FFFFFFF, else truncation. Branch-free
// (fmax drops a NaN; 2147483520 is the largest float below 2^31).
inline uint32_t ToIntSat(float value) {
    const int32_t clamped = static_cast<int32_t>(std::fmin(std::fmax(value, -2147483648.0f), 2147483520.0f));
    return value >= 2147483648.0f ? 0x7FFFFFFFu : static_cast<uint32_t>(clamped);
}

// The loop's state, stepped N samples at a time: N = 8 while no line position wraps inside the step, else 1. A step
// runs each stage as a loop over its samples (vectorized for N = 8), every sample performing the per-sample
// algorithm's operations in its order, so steps of 8 and of 1 compute identical values. In a step of 8 every line read
// still precedes the step's writes in per-sample order: a slot written now is read again only `length` > 8 samples on.
// The low-pass is a recursion and stays per sample.
struct Fast {
    float f1, f3, f4, f5, f6, zero, outGain, sendGain, earlyCoef[3], combCoef[3], lpf[3];
    uint32_t earlyPos[3], combPos[3], apPos[2], lastPos[3], preDelayPos;
    // The last sample's values, which Finish leaves in the stack frame and the registers.
    float mix[3] = {}, f7 = 0, f8 = 0, f9 = 0, f10 = 0, f11 = 0, f12 = 0, f13 = 0;
    uint32_t convert = 0, lastApInc = 0, lastOut = 0;

    bool Fits(const Plan& p, uint32_t n) const {
        bool ok = apPos[0] + n <= p.apLength[0] && apPos[1] + n <= p.apLength[1];
        for (uint32_t k = 0; k < 3; ++k)
            ok &= earlyPos[k] + n <= p.earlyLength && combPos[k] + n <= p.combLength[k] &&
                  lastPos[k] + n <= p.lastLength[k];
        return ok && (!p.preDelay || preDelayPos + n <= p.preDelayLength);
    }

    // The positions after n samples, wrapping as Next does (a step ends at most on a line's last slot).
    void Advance(const Plan& p, uint32_t n) {
        const auto step = [n](uint32_t& pos, uint32_t length) { pos = pos + n < length ? pos + n : 0; };
        for (uint32_t k = 0; k < 3; ++k) {
            step(earlyPos[k], p.earlyLength);
            step(combPos[k], p.combLength[k]);
            step(lastPos[k], p.lastLength[k]);
        }
        if (p.preDelay) step(preDelayPos, p.preDelayLength);
        step(apPos[0], p.apLength[0]);
        lastApInc = apPos[1] + n;
        step(apPos[1], p.apLength[1]);
    }

    template <uint32_t N>
    void Step(const Plan& p, uint32_t i) {
        // Locals, not members: a guest store (through uint8_t*) could alias *this, which would reload every member
        // after every store and keep the compiler from vectorizing.
        const float f1 = this->f1, f3 = this->f3, f4 = this->f4, f5 = this->f5, f6 = this->f6, zero = this->zero;
        const float outGain = this->outGain, sendGain = this->sendGain;
        const float earlyCoef[3] = {this->earlyCoef[0], this->earlyCoef[1], this->earlyCoef[2]};
        const float combCoef[3] = {this->combCoef[0], this->combCoef[1], this->combCoef[2]};
        const uint32_t earlyPos[3] = {this->earlyPos[0], this->earlyPos[1], this->earlyPos[2]};
        const uint32_t combPos[3] = {this->combPos[0], this->combPos[1], this->combPos[2]};
        const uint32_t apPos[2] = {this->apPos[0], this->apPos[1]};
        const uint32_t lastPos[3] = {this->lastPos[0], this->lastPos[1], this->lastPos[2]};
        const uint32_t preDelayPos = this->preDelayPos;
        float lpf[3] = {this->lpf[0], this->lpf[1], this->lpf[2]};
        float input[3][N], early[3][N], dry[3][N], combSum[3][N], a7[3][N], a8[3][N], a9[3][N], lp[3][N], out[3][N];
        uint32_t convert = 0;
        for (uint32_t c = 0; c < 3; ++c) {
            const uint8_t* src = p.in[c].host + 4 * i;
            const uint8_t* bus = p.busIn ? p.bi[c].host + 4 * i : nullptr;
            uint32_t sum = 0;
            for (uint32_t j = 0; j < N; ++j) {
                sum = bus ? Load32(src + 4 * j) + Load32(bus + 4 * j) : Load32(src + 4 * j);
                input[c][j] = static_cast<float>(static_cast<int32_t>(sum));
            }
            convert = sum ^ 0x80000000u;  // the original's int->double conversion word (left in its frame)
        }
        for (uint32_t c = 0; c < 3; ++c) {
            const uint8_t* t0 = p.early[c][0].At(earlyPos[0]);
            const uint8_t* t1 = p.early[c][1].At(earlyPos[1]);
            const uint8_t* t2 = p.early[c][2].At(earlyPos[2]);
            for (uint32_t j = 0; j < N; ++j) {
                const float a = earlyCoef[0] * LoadFloat(t0 + 4 * j);
                const float b = earlyCoef[1] * LoadFloat(t1 + 4 * j);
                const float d = earlyCoef[2] * LoadFloat(t2 + 4 * j);
                early[c][j] = d + (a + b);
            }
        }
        for (uint32_t c = 0; c < 3; ++c) {
            if (p.preDelay) {
                uint8_t* q = p.pre[c].At(preDelayPos);
                for (uint32_t j = 0; j < N; ++j) {
                    dry[c][j] = LoadFloat(q + 4 * j);
                    StoreFloat(q + 4 * j, input[c][j]);
                }
            } else {
                for (uint32_t j = 0; j < N; ++j) dry[c][j] = input[c][j];
            }
            for (uint32_t j = 0; j < N; ++j) combSum[c][j] = zero;
        }
        for (uint32_t k = 0; k < 3; ++k)
            for (uint32_t c = 0; c < 3; ++c) {
                uint8_t* q = p.comb[c][k].At(combPos[k]);
                for (uint32_t j = 0; j < N; ++j) {
                    const float old = LoadFloat(q + 4 * j);
                    const float fed = old * combCoef[k];
                    combSum[c][j] = combSum[c][j] + old;
                    StoreFloat(q + 4 * j, dry[c][j] + fed);
                }
            }
        for (uint32_t c = 0; c < 3; ++c) {
            uint8_t* q = p.ap[c][0].At(apPos[0]);
            for (uint32_t j = 0; j < N; ++j) {
                const float old = LoadFloat(q + 4 * j);
                const float w = combSum[c][j] + old * f3;
                StoreFloat(q + 4 * j, w);
                a7[c][j] = w * f3;
                a8[c][j] = old;
            }
        }
        for (uint32_t c = 0; c < 3; ++c) {
            uint8_t* q = p.ap[c][1].At(apPos[1]);
            for (uint32_t j = 0; j < N; ++j) {
                const float x = a8[c][j] - a7[c][j];
                const float old = LoadFloat(q + 4 * j);
                const float w = x + old * f3;
                a8[c][j] = w * f3;
                StoreFloat(q + 4 * j, w);
                a9[c][j] = old;
            }
        }
        for (uint32_t c = 0; c < 3; ++c) {
            float x[N];
            for (uint32_t j = 0; j < N; ++j) x[j] = a9[c][j] - a8[c][j];
            float state = lpf[c];
            for (uint32_t j = 0; j < N; ++j) {
                const float held = f1 * state;
                state = f4 * x[j] + held;
                lp[c][j] = state;
            }
            lpf[c] = state;
        }
        for (uint32_t c = 0; c < 3; ++c) {
            uint8_t* q = p.last[c].At(lastPos[c]);
            for (uint32_t j = 0; j < N; ++j) {
                const float old = LoadFloat(q + 4 * j);
                const float w = lp[c][j] + old * f3;
                StoreFloat(q + 4 * j, w);
                out[c][j] = (old - w * f3) * f5 + early[c][j];
            }
            mix[c] = out[c][N - 1];
        }
        // Crosstalk and output (loc_80202D38): each output channel plus the other two scaled.
        float o0[N], o1[N], o2[N];
        for (uint32_t j = 0; j < N; ++j) {
            const float m0 = out[0][j], m1 = out[1][j], m2 = out[2][j];
            o0[j] = m0 + (m1 + m2) * f6;
            o1[j] = m1 + (m0 + m2) * f6;
            o2[j] = m2 + (m0 + m1) * f6;
        }
        for (uint32_t j = 0; j < N; ++j) {
            Store32(p.in[0].host + 4 * (i + j), ToIntSat(o0[j] * outGain));
            Store32(p.in[1].host + 4 * (i + j), ToIntSat(o1[j] * outGain));
            Store32(p.in[2].host + 4 * (i + j), ToIntSat(o2[j] * outGain));
        }
        if (p.busOut)
            for (uint32_t j = 0; j < N; ++j) {
                Store32(p.bo[0].host + 4 * (i + j), ToIntSat(o0[j] * sendGain));
                Store32(p.bo[1].host + 4 * (i + j), ToIntSat(o1[j] * sendGain));
                Store32(p.bo[2].host + 4 * (i + j), ToIntSat(o2[j] * sendGain));
            }
        for (uint32_t c = 0; c < 3; ++c) this->lpf[c] = lpf[c];
        this->convert = convert;
        f10 = mix[0];
        f12 = mix[1];
        f11 = mix[2];
        f13 = f10 + f12;
        f9 = o0[N - 1];
        f8 = o1[N - 1];
        f7 = o2[N - 1];
        lastOut = ToIntSat(f7 * (p.busOut ? sendGain : outGain));
    }
};

void ProcessFast(Plan& p) {
    const uint32_t savedFp = MkwGetHostFpControl();
    MkwSetHostFpControl(savedFp | kMkwFpControlFlushToZeroBits);
    const uint8_t* s = p.s;
    const uint8_t* k2 = p.constants;
    Fast f;
    f.f1 = LoadFloat(s + kLpfCoef);
    f.f4 = LoadFloat(k2 + kOne) - f.f1;
    f.f5 = LoadFloat(k2 + kFusedScale) * LoadFloat(s + kFusedGain);
    f.f3 = LoadFloat(s + kAllpassCoef);
    f.f6 = LoadFloat(k2 + kCrosstalkScale) * LoadFloat(s + kCrosstalk);
    f.zero = LoadFloat(k2 + kZero);
    f.outGain = LoadFloat(s + kOutGain);
    f.sendGain = LoadFloat(s + kSendGain);
    for (uint32_t k = 0; k < 3; ++k) {
        f.earlyCoef[k] = LoadFloat(s + kEarlyCoef + 4 * k);
        f.combCoef[k] = LoadFloat(s + kCombCoef + 4 * k);
        f.lpf[k] = LoadFloat(s + kLastLpfOut + 4 * k);
        f.earlyPos[k] = p.earlyPos[k];
        f.combPos[k] = p.combPos[k];
        f.lastPos[k] = p.lastPos[k];
    }
    f.apPos[0] = p.apPos[0];
    f.apPos[1] = p.apPos[1];
    f.preDelayPos = p.preDelayPos;
    for (uint32_t i = 0; i < kSamples;) {
        if (i + 8 <= kSamples && f.Fits(p, 8)) {
            f.Step<8>(p, i);
            f.Advance(p, 8);
            i += 8;
        } else {
            f.Step<1>(p, i);
            f.Advance(p, 1);
            ++i;
        }
    }
    MkwSetHostFpControl(savedFp);

    for (uint32_t k = 0; k < 3; ++k) {
        p.earlyPos[k] = f.earlyPos[k];
        p.combPos[k] = f.combPos[k];
        p.lastPos[k] = f.lastPos[k];
        p.lpf[k] = f.lpf[k];
        p.mix[k] = f.mix[k];
    }
    p.apPos[0] = f.apPos[0];
    p.apPos[1] = f.apPos[1];
    p.preDelayPos = f.preDelayPos;
    p.lastConvert = f.convert;
    p.lastApInc = f.lastApInc;
    // The volatile registers Finish leaves (the guest's callers never read them), from this loop's last sample.
    const double regs[14] = {PpcBitCastToDoubleInline(Load64(k2 + kBias)), f.f1,
                             PpcPackIntegerWordInline(static_cast<int32_t>(f.lastOut)), f.f3, f.f4, f.f5, f.f6, f.f7,
                             f.f8, f.f9, f.f10, f.f11, f.f12, f.f13};
    std::memcpy(p.f, regs, sizeof(regs));
}

// ---- Approximate samples loop in explicit vectors ----
// ProcessFast's Step<8> is fully unrolled by the compiler and stays scalar (IBS: 1% of its ops 8-wide), and any line
// position wrapping within the next 8 samples sends it to Step<1>, sample by sample. ProcessFastV runs the same
// per-sample operations, operand for operand (so the same IEEE single results, under the same flush-to-zero setting:
// a vector lane rounds as the scalar operation does), on 8-lane vectors, always 8 samples a step: a line access that
// wraps within the step reads or writes its slots up to the line's end and from its start, with masked loads and
// stores that touch only those slots. With every line at least 8 slots long a step's slots are distinct, each read
// before it is written as in per-sample order (a shorter line runs ProcessFast). Its output is ProcessFast's, byte for
// byte (AXFX_CHECK=1 compares them).
using F8 = float __attribute__((ext_vector_type(8)));
using I8 = int32_t __attribute__((ext_vector_type(8)));
using U8 = uint32_t __attribute__((ext_vector_type(8)));
using B32 = uint8_t __attribute__((ext_vector_type(32)));

inline U8 SwapWords(U8 v) {
    const B32 b = __builtin_bit_cast(B32, v);
    return __builtin_bit_cast(U8, __builtin_shufflevector(b, b, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19,
                                                          18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28));
}
inline U8 RawLoad8(const uint8_t* p) {
    U8 v;
    std::memcpy(&v, p, 32);
    return v;
}
inline void RawStore8(uint8_t* p, U8 v) { std::memcpy(p, &v, 32); }
// The lanes where `mask` is all ones, from p + 4 * lane; the others are 0 and their memory is not accessed.
inline U8 RawLoadMasked(const uint8_t* p, I8 mask) {
#if defined(__AVX2__)
    return __builtin_bit_cast(U8, _mm256_maskload_epi32(reinterpret_cast<const int*>(p), __builtin_bit_cast(__m256i, mask)));
#else
    U8 v = 0;
    for (int k = 0; k < 8; ++k)
        if (mask[k] != 0) {
            uint32_t w;
            std::memcpy(&w, p + 4 * k, 4);
            v[k] = w;
        }
    return v;
#endif
}
inline void RawStoreMasked(uint8_t* p, I8 mask, U8 v) {
#if defined(__AVX2__)
    _mm256_maskstore_epi32(reinterpret_cast<int*>(p), __builtin_bit_cast(__m256i, mask), __builtin_bit_cast(__m256i, v));
#else
    for (int k = 0; k < 8; ++k)
        if (mask[k] != 0) {
            const uint32_t w = v[k];
            std::memcpy(p + 4 * k, &w, 4);
        }
#endif
}
inline U8 LoadWords(const uint8_t* p) { return SwapWords(RawLoad8(p)); }
inline void StoreWords(uint8_t* p, U8 v) { RawStore8(p, SwapWords(v)); }
inline F8 LoadF8(const uint8_t* p) { return __builtin_bit_cast(F8, LoadWords(p)); }
inline void StoreF8(uint8_t* p, F8 v) { StoreWords(p, __builtin_bit_cast(U8, v)); }

// A line's 8 slots from position `pos` (< length, length >= 8): up to the line's end, then from its start.
inline I8 HeadLanes(uint32_t head) {
    const I8 lane = {0, 1, 2, 3, 4, 5, 6, 7};
    return lane < static_cast<int32_t>(head);
}
inline uint8_t* WrapBase(const Ring& r, uint32_t head) {  // lane k >= head lands on position k - head
    return reinterpret_cast<uint8_t*>(r.host0 - uintptr_t{4} * head);
}
inline U8 LoadRing(const Ring& r, uint32_t pos, uint32_t length) {
    const uint32_t head = length - pos;
    if (head >= 8) [[likely]] return LoadWords(r.At(pos));
    const I8 inHead = HeadLanes(head);
    return SwapWords(RawLoadMasked(r.At(pos), inHead) | RawLoadMasked(WrapBase(r, head), ~inHead));
}
inline void StoreRing(const Ring& r, uint32_t pos, uint32_t length, U8 v) {
    const uint32_t head = length - pos;
    if (head >= 8) [[likely]] {
        StoreWords(r.At(pos), v);
        return;
    }
    const I8 inHead = HeadLanes(head);
    const U8 raw = SwapWords(v);
    RawStoreMasked(r.At(pos), inHead, raw);
    RawStoreMasked(WrapBase(r, head), ~inHead, raw);
}
inline F8 LoadRingF(const Ring& r, uint32_t pos, uint32_t length) {
    return __builtin_bit_cast(F8, LoadRing(r, pos, length));
}
inline void StoreRingF(const Ring& r, uint32_t pos, uint32_t length, F8 v) {
    StoreRing(r, pos, length, __builtin_bit_cast(U8, v));
}
// A position 8 samples on (length >= 8), and the position of the step's last sample.
inline uint32_t Plus8(uint32_t pos, uint32_t length) { return pos + 8 >= length ? pos + 8 - length : pos + 8; }
inline uint32_t Plus7(uint32_t pos, uint32_t length) { return pos + 7 >= length ? pos + 7 - length : pos + 7; }
// ToIntSat per lane: fmax/fmin are maxnum/minnum (a NaN gives the other operand), the clamped value converts exactly,
// and from 2^31 up the result is 0x7FFFFFFF.
inline U8 ToIntSat8(F8 v) {
    const F8 clamped = __builtin_elementwise_min(__builtin_elementwise_max(v, F8(-2147483648.0f)), F8(2147483520.0f));
    const U8 t = __builtin_bit_cast(U8, __builtin_convertvector(clamped, I8));
    const U8 high = __builtin_bit_cast(U8, v >= 2147483648.0f);  // all ones where v >= 2^31
    return (t & ~high) | (U8(0x7FFFFFFFu) & high);
}

struct FastV {
    float f1, f3, f4, f5, f6, zero, outGain, sendGain, earlyCoef[3], combCoef[3], lpf[3];
    uint32_t earlyPos[3], combPos[3], apPos[2], lastPos[3], preDelayPos;
    float mix[3] = {}, f7 = 0, f8 = 0, f9 = 0, f10 = 0, f11 = 0, f12 = 0, f13 = 0;
    uint32_t convert = 0, lastApInc = 0, lastOut = 0;

    // Every line at least 8 slots long: a step's 8 slots of a line are distinct.
    static bool LongLines(const Plan& p) {
        bool ok = p.earlyLength >= 8 && p.apLength[0] >= 8 && p.apLength[1] >= 8 && (!p.preDelay || p.preDelayLength >= 8);
        for (uint32_t k = 0; k < 3; ++k) ok = ok && p.combLength[k] >= 8 && p.lastLength[k] >= 8;
        return ok;
    }

    // The positions 8 samples on: eight of Fast::Advance's steps of one (each wraps to 0 at its line's end).
    void Advance8(const Plan& p) {
        for (uint32_t k = 0; k < 3; ++k) {
            earlyPos[k] = Plus8(earlyPos[k], p.earlyLength);
            combPos[k] = Plus8(combPos[k], p.combLength[k]);
            lastPos[k] = Plus8(lastPos[k], p.lastLength[k]);
        }
        if (p.preDelay) preDelayPos = Plus8(preDelayPos, p.preDelayLength);
        apPos[0] = Plus8(apPos[0], p.apLength[0]);
        lastApInc = Plus7(apPos[1], p.apLength[1]) + 1;  // the last sample's position + 1, before its wrap
        apPos[1] = Plus8(apPos[1], p.apLength[1]);
    }

    // Samples i .. i + 7, as Fast::Step<8> (whose lines could not wrap inside the step).
    void Step(const Plan& p, uint32_t i) {
        constexpr uint32_t n = 8;
        const F8 f1 = this->f1, f3 = this->f3, f4 = this->f4, f5 = this->f5, f6 = this->f6, zero = this->zero;
        const F8 outGain = this->outGain, sendGain = this->sendGain;
        const uint32_t earlyPos[3] = {this->earlyPos[0], this->earlyPos[1], this->earlyPos[2]};
        const uint32_t combPos[3] = {this->combPos[0], this->combPos[1], this->combPos[2]};
        const uint32_t apPos[2] = {this->apPos[0], this->apPos[1]};
        const uint32_t lastPos[3] = {this->lastPos[0], this->lastPos[1], this->lastPos[2]};
        const uint32_t preDelayPos = this->preDelayPos;
        F8 input[3], early[3], dry[3], combSum[3], a7[3], a8[3], a9[3], out[3];
        uint32_t convert = 0;
        for (uint32_t c = 0; c < 3; ++c) {
            U8 sum = LoadWords(p.in[c].host + 4 * i);
            if (p.busIn) sum += LoadWords(p.bi[c].host + 4 * i);
            input[c] = __builtin_convertvector(__builtin_bit_cast(I8, sum), F8);
            convert = sum[n - 1] ^ 0x80000000u;
        }
        for (uint32_t c = 0; c < 3; ++c) {
            const F8 a = this->earlyCoef[0] * LoadRingF(p.early[c][0], earlyPos[0], p.earlyLength);
            const F8 b = this->earlyCoef[1] * LoadRingF(p.early[c][1], earlyPos[1], p.earlyLength);
            const F8 d = this->earlyCoef[2] * LoadRingF(p.early[c][2], earlyPos[2], p.earlyLength);
            early[c] = d + (a + b);
        }
        for (uint32_t c = 0; c < 3; ++c) {
            if (p.preDelay) {
                dry[c] = LoadRingF(p.pre[c], preDelayPos, p.preDelayLength);
                StoreRingF(p.pre[c], preDelayPos, p.preDelayLength, input[c]);
            } else {
                dry[c] = input[c];
            }
            combSum[c] = zero;
        }
        for (uint32_t k = 0; k < 3; ++k)
            for (uint32_t c = 0; c < 3; ++c) {
                const F8 old = LoadRingF(p.comb[c][k], combPos[k], p.combLength[k]);
                const F8 fed = old * this->combCoef[k];
                combSum[c] = combSum[c] + old;
                StoreRingF(p.comb[c][k], combPos[k], p.combLength[k], dry[c] + fed);
            }
        for (uint32_t c = 0; c < 3; ++c) {
            const F8 old = LoadRingF(p.ap[c][0], apPos[0], p.apLength[0]);
            const F8 w = combSum[c] + old * f3;
            StoreRingF(p.ap[c][0], apPos[0], p.apLength[0], w);
            a7[c] = w * f3;
            a8[c] = old;
        }
        for (uint32_t c = 0; c < 3; ++c) {
            const F8 x = a8[c] - a7[c];
            const F8 old = LoadRingF(p.ap[c][1], apPos[1], p.apLength[1]);
            const F8 w = x + old * f3;
            a8[c] = w * f3;
            StoreRingF(p.ap[c][1], apPos[1], p.apLength[1], w);
            a9[c] = old;
        }
        float lpf[3] = {this->lpf[0], this->lpf[1], this->lpf[2]};
        F8 lp[3] = {};
        for (uint32_t c = 0; c < 3; ++c) {  // the low-pass recursion, per sample
            const F8 x = a9[c] - a8[c];
            float state = lpf[c];
            for (uint32_t j = 0; j < n; ++j) {
                const float held = this->f1 * state;
                state = this->f4 * x[j] + held;
                lp[c][j] = state;
            }
            lpf[c] = state;
        }
        for (uint32_t c = 0; c < 3; ++c) {
            const F8 old = LoadRingF(p.last[c], lastPos[c], p.lastLength[c]);
            const F8 w = lp[c] + old * f3;
            StoreRingF(p.last[c], lastPos[c], p.lastLength[c], w);
            out[c] = (old - w * f3) * f5 + early[c];
            mix[c] = out[c][n - 1];
        }
        const F8 o0 = out[0] + (out[1] + out[2]) * f6;
        const F8 o1 = out[1] + (out[0] + out[2]) * f6;
        const F8 o2 = out[2] + (out[0] + out[1]) * f6;
        StoreWords(p.in[0].host + 4 * i, ToIntSat8(o0 * outGain));
        StoreWords(p.in[1].host + 4 * i, ToIntSat8(o1 * outGain));
        StoreWords(p.in[2].host + 4 * i, ToIntSat8(o2 * outGain));
        if (p.busOut) {
            StoreWords(p.bo[0].host + 4 * i, ToIntSat8(o0 * sendGain));
            StoreWords(p.bo[1].host + 4 * i, ToIntSat8(o1 * sendGain));
            StoreWords(p.bo[2].host + 4 * i, ToIntSat8(o2 * sendGain));
        }
        for (uint32_t c = 0; c < 3; ++c) this->lpf[c] = lpf[c];
        this->convert = convert;
        f10 = mix[0];
        f12 = mix[1];
        f11 = mix[2];
        f13 = f10 + f12;
        f9 = o0[n - 1];
        f8 = o1[n - 1];
        f7 = o2[n - 1];
        lastOut = ToIntSat(f7 * (p.busOut ? this->sendGain : this->outGain));
    }
};

// ProcessFast with FastV's steps; the same results. A line shorter than 8 slots runs ProcessFast itself.
void ProcessFastV(Plan& p) {
    if (!FastV::LongLines(p)) return ProcessFast(p);
    const uint32_t savedFp = MkwGetHostFpControl();
    MkwSetHostFpControl(savedFp | kMkwFpControlFlushToZeroBits);
    const uint8_t* s = p.s;
    const uint8_t* k2 = p.constants;
    FastV f;
    f.f1 = LoadFloat(s + kLpfCoef);
    f.f4 = LoadFloat(k2 + kOne) - f.f1;
    f.f5 = LoadFloat(k2 + kFusedScale) * LoadFloat(s + kFusedGain);
    f.f3 = LoadFloat(s + kAllpassCoef);
    f.f6 = LoadFloat(k2 + kCrosstalkScale) * LoadFloat(s + kCrosstalk);
    f.zero = LoadFloat(k2 + kZero);
    f.outGain = LoadFloat(s + kOutGain);
    f.sendGain = LoadFloat(s + kSendGain);
    for (uint32_t k = 0; k < 3; ++k) {
        f.earlyCoef[k] = LoadFloat(s + kEarlyCoef + 4 * k);
        f.combCoef[k] = LoadFloat(s + kCombCoef + 4 * k);
        f.lpf[k] = LoadFloat(s + kLastLpfOut + 4 * k);
        f.earlyPos[k] = p.earlyPos[k];
        f.combPos[k] = p.combPos[k];
        f.lastPos[k] = p.lastPos[k];
    }
    f.apPos[0] = p.apPos[0];
    f.apPos[1] = p.apPos[1];
    f.preDelayPos = p.preDelayPos;
    static_assert(kSamples % 8 == 0);
    for (uint32_t i = 0; i < kSamples; i += 8) {
        f.Step(p, i);
        f.Advance8(p);
    }
    MkwSetHostFpControl(savedFp);

    for (uint32_t k = 0; k < 3; ++k) {
        p.earlyPos[k] = f.earlyPos[k];
        p.combPos[k] = f.combPos[k];
        p.lastPos[k] = f.lastPos[k];
        p.lpf[k] = f.lpf[k];
        p.mix[k] = f.mix[k];
    }
    p.apPos[0] = f.apPos[0];
    p.apPos[1] = f.apPos[1];
    p.preDelayPos = f.preDelayPos;
    p.lastConvert = f.convert;
    p.lastApInc = f.lastApInc;
    const double regs[14] = {PpcBitCastToDoubleInline(Load64(k2 + kBias)), f.f1,
                             PpcPackIntegerWordInline(static_cast<int32_t>(f.lastOut)), f.f3, f.f4, f.f5, f.f6, f.f7,
                             f.f8, f.f9, f.f10, f.f11, f.f12, f.f13};
    std::memcpy(p.f, regs, sizeof(regs));
}

// AXFX_VECTOR=0: the scalar loops (ProcessFast, the delay's), for on/off counts on one binary.
bool VectorAudio() {
    static const bool on = [] {
        const char* v = std::getenv("AXFX_VECTOR");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    return on;
}

// AXFX_CHECK=1 (fast reverb, delay): run today's loop and the vector one from the same state, compare everything they
// leave, keep today's; NAT_INSN=1 counts them as the original and native columns (sites axfx.reverb.fast,
// axfx.delay.loop).
bool CheckMode() {
    static const bool on = [] {
        const char* v = std::getenv("AXFX_CHECK");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return on;
}
std::atomic<uint64_t> g_checkCalls{0}, g_checkMismatches{0};
void LogCheck(const char* what, uint64_t calls, uint64_t mismatches) {
    if (calls == 1 || calls % 2000 == 0 || mismatches != 0)
        RT_LOGF(RT_TAG_AUDIO, "axfx check %s: calls=%" PRIu64 " mismatches=%" PRIu64 "\n", what, calls, mismatches);
}
// Every byte range the samples loop writes, saved and compared as a unit.
struct Snapshot {
    std::vector<std::pair<uint8_t*, uint32_t>> spans;
    std::vector<uint8_t> bytes;
    void Add(const Span& s) { spans.emplace_back(s.host, s.bytes); }
    void Save() {
        bytes.clear();
        for (auto& [host, n] : spans) bytes.insert(bytes.end(), host, host + n);
    }
    void Restore() const {
        size_t at = 0;
        for (auto& [host, n] : spans) {
            std::memcpy(host, bytes.data() + at, n);
            at += n;
        }
    }
};

void CheckFast(Plan& p) {
    static const int site = InsnCount::On() ? InsnCount::Site("axfx.reverb.fast") : -1;
    thread_local Snapshot snap, ref;
    snap.spans.clear();
    for (uint32_t c = 0; c < 3; ++c) {
        snap.Add(p.in[c]);
        if (p.busOut) snap.Add(p.bo[c]);
        if (p.preDelay) snap.Add(p.pre[c].span);
        for (uint32_t k = 0; k < 3; ++k) snap.Add(p.comb[c][k].span);
        for (uint32_t k = 0; k < 2; ++k) snap.Add(p.ap[c][k].span);
        snap.Add(p.last[c].span);
    }
    ref.spans = snap.spans;
    snap.Save();
    Plan old = p;
    uint64_t t = InsnCount::On() ? InsnCount::Read() : 0;
    ProcessFast(old);
    if (InsnCount::On()) InsnCount::Add(site, InsnCount::kOriginal, InsnCount::Since(t));
    ref.Save();
    snap.Restore();
    t = InsnCount::On() ? InsnCount::Read() : 0;
    ProcessFastV(p);
    if (InsnCount::On()) InsnCount::Add(site, InsnCount::kNative, InsnCount::Since(t));
    snap.Save();  // the vector loop's bytes
    bool same = snap.bytes == ref.bytes;
    const auto eq = [](const auto& a, const auto& b) { return std::memcmp(&a, &b, sizeof(a)) == 0; };
    same = same && eq(old.earlyPos, p.earlyPos) && eq(old.combPos, p.combPos) && eq(old.lastPos, p.lastPos) &&
           eq(old.apPos, p.apPos) && old.preDelayPos == p.preDelayPos && eq(old.lpf, p.lpf) && eq(old.mix, p.mix) &&
           eq(old.f, p.f) && old.lastConvert == p.lastConvert && old.lastApInc == p.lastApInc;
    const uint64_t calls = g_checkCalls.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t mismatches = g_checkMismatches.load(std::memory_order_relaxed) + (same ? 0 : 1);
    if (!same) {
        g_checkMismatches.fetch_add(1, std::memory_order_relaxed);
        if (mismatches <= 8) {
            size_t at = 0;
            for (auto& [host, n] : snap.spans) {
                (void)host;
                if (std::memcmp(snap.bytes.data() + at, ref.bytes.data() + at, n) != 0) {
                    uint32_t k = 0;
                    while (snap.bytes[at + k] == ref.bytes[at + k]) ++k;
                    RT_LOGF(RT_TAG_AUDIO, "axfx check MISMATCH reverb: span of %u bytes, byte %u: today %02X vector "
                            "%02X\n", n, k, ref.bytes[at + k], snap.bytes[at + k]);
                    break;
                }
                at += n;
            }
        }
    }
    LogCheck("reverb.fast", calls, mismatches);
    ref.Restore();  // today's bytes and results stay
    p = old;
}

// The prologue's stores: stwu's back chain, the LR save word in the caller's frame, stmw r25-r31 (_savegpr_25).
void StorePrologue(CpuContext* ctx, uint32_t r1, const Span& frame) {
    const uint32_t sp = r1 - kFrameBytes;
    MemoryInline::WriteResolved32(frame.host, 0u, sp, r1);
    MemoryInline::WriteResolved32(frame.host, 116u, sp + 116, ctx->lr);
    for (uint32_t r = 25; r <= 31; ++r)
        MemoryInline::WriteResolved32(frame.host, 84u + (r - 25) * 4, sp + 84 + (r - 25) * 4, ctx->gpr[r]);
}

// f0, f1 and f3-f6 at the original's return (f[2] unused): the bias and the parameters' products, as the samples loop
// the call chose computes them (Process's Broadway operations, or the fast loops' IEEE singles under flush-to-zero).
template <bool Ni>
void ExactParamRegisters(const Plan& p, double f[7]) {
    const uint8_t* s = p.s;
    const uint8_t* k2 = p.constants;
    f[1] = LoadF(s + kLpfCoef);
    f[4] = Fsubs<Ni>(LoadF(k2 + kOne), f[1]);
    f[5] = Fmuls<Ni>(LoadF(k2 + kFusedScale), LoadF(s + kFusedGain));
    f[3] = LoadF(s + kAllpassCoef);
    f[6] = Fmuls<Ni>(LoadF(k2 + kCrosstalkScale), LoadF(s + kCrosstalk));
}

void ParamRegisters(const Plan& p, Async::Samples samples, double f[7]) {
    const uint8_t* s = p.s;
    const uint8_t* k2 = p.constants;
    f[0] = PpcBitCastToDoubleInline(Load64(k2 + kBias));
    f[2] = 0;
    if (samples == Async::Samples::kExact) {
        ExactParamRegisters<false>(p, f);
    } else if (samples == Async::Samples::kExactNi) {
        ExactParamRegisters<true>(p, f);
    } else {
        const uint32_t savedFp = MkwGetHostFpControl();
        MkwSetHostFpControl(savedFp | kMkwFpControlFlushToZeroBits);
        const float f1 = LoadFloat(s + kLpfCoef);
        const float f4 = LoadFloat(k2 + kOne) - f1;
        const float f5 = LoadFloat(k2 + kFusedScale) * LoadFloat(s + kFusedGain);
        const float f6 = LoadFloat(k2 + kCrosstalkScale) * LoadFloat(s + kCrosstalk);
        MkwSetHostFpControl(savedFp);
        f[1] = f1;
        f[3] = LoadFloat(s + kAllpassCoef);
        f[4] = f4;
        f[5] = f5;
        f[6] = f6;
    }
}

// ---- Run's proofs: per call, or per effect configuration (AXFX_PROOF) ----
// Run's preconditions (each must hold for the native to run):
// - every buffer and line range resolves to one host range, writable where written;
// - every position lies inside its line;
// - no buffer or line overlaps the frame or the effect struct, and no written one the .sdata2 constants;
// - no written range overlaps any other range (Disjoint).
// ProveFresh proves them on this call's ranges (a ring's range is what its 96 samples visit). AXFX_PROOF (default on)
// proves them on whole lines, which contain every position's range, once per effect configuration (Proven), and per
// address triple for the bus buffers (a bus hands its callback one of three). A call then checks only what can change
// within one configuration:
// - the frame (its stack moves);
// - the positions;
// - the store policy under every written line and buffer (memory.cpp's executable and MMIO chunks change at runtime);
// - the configuration itself: the struct's line pointers, lengths and bus pointers, compared word for word, and a
//   generation that the effect functions' hooks bump (EffectsChanged).
// A configuration whose whole lines overlap (where its calls' ranges might not) keeps the per-call proof, as do calls
// with bus in or bus out. AXFX_PROOF=0: per call only. AXFX_PROOF=check: both, compared field for field (the plan each
// produces, and that a cached yes is a per-call yes); the per-call result is used. Lean memory only (there,
// ResolveRangeHost is address arithmetic plus the policy bytes PolicyClear reads).
enum class ProofMode : uint8_t { kFresh, kCached, kCheck };

ProofMode GetProofMode() {
    static const ProofMode mode = [] {
#if MKW_LEAN_MEMORY && !MKW_FLAT_MEMORY
        const char* v = std::getenv("AXFX_PROOF");
        const ProofMode m = v == nullptr || *v == '\0' || std::strcmp(v, "1") == 0 ? ProofMode::kCached
                            : std::strcmp(v, "check") == 0                          ? ProofMode::kCheck
                                                                                    : ProofMode::kFresh;
#else
        const ProofMode m = ProofMode::kFresh;
#endif
        RT_LOGF(RT_TAG_AUDIO, "axfx: reverb proofs %s\n",
                m == ProofMode::kCached  ? "per effect configuration (AXFX_PROOF)"
                : m == ProofMode::kCheck ? "per configuration and per call, compared (AXFX_PROOF=check)"
                                         : "per call");
        return m;
    }();
    return mode;
}

uint64_t g_configGeneration = 0;  // game thread: bumped by the effect functions' hooks (EffectsChanged)

struct ProofStats {
    uint64_t calls = 0;        // calls the cache could serve (no bus in or out)
    uint64_t cached = 0;       // served by a cached proof
    uint64_t built = 0;        // configurations proven
    uint64_t uncacheable = 0;  // configurations whose whole lines do not pass (their calls prove each call)
    uint64_t triples = 0;      // buffer triples proven
    uint64_t declined = 0;     // a cached configuration's call that the per-call checks sent to ProveFresh
    uint64_t checked = 0, mismatches = 0;  // AXFX_PROOF=check
};
ProofStats g_proofStats;

constexpr uint32_t kRings = 30;     // per channel: 3 early taps, the pre-delay, 3 combs, 2 allpasses, the last allpass
constexpr uint32_t kMaxChunks = 8;  // store-policy chunks (64 KiB) under a configuration's written lines

// A bus's buffer triple, proven: resolvable and writable as one range each, clear of the struct, the constants, every
// line and each other.
struct Triple {
    uint32_t in[3];
    Span span[3];
    uint32_t chunks[6];
    uint32_t chunkCount;
};

struct Proven {
    bool used = false;
    bool cacheable = false;
    uint64_t generation = 0;
    uint32_t fx = 0, r2 = 0;
    uint8_t config[kEffectBytes] = {};  // the struct when proven (SameConfig compares its configuration words)
    Span effect, constants;
    Plan plan{};  // s, constants, flags, lengths, and each ring's line, length and host0 (its span: the whole line)
    Segment lines[kRings];  // whole lines, merged: sorted and disjoint (the frame and the buffers must avoid them)
    uint32_t lineCount = 0;
    Segment cover{};  // every line (Footprint's cover)
    uint32_t chunks[kMaxChunks];
    uint32_t chunkCount = 0;
    Triple triples[4];
    uint32_t tripleCount = 0, nextTriple = 0;
};
Proven g_proven[2];  // the game's effects: one reverb (two entries, so a second one does not evict it)
uint32_t g_provenNext = 0;

// The struct's configuration words (positions excluded): line pointers, lengths, bus pointers.
template <uint32_t Begin, uint32_t End>
inline bool SameBytes(const uint8_t* a, const uint8_t* b) {
    return std::memcmp(a + Begin, b + Begin, End - Begin) == 0;
}
inline bool SameConfig(const uint8_t* a, const uint8_t* b) {
    return SameBytes<kEarlyLine, kEarlyPos>(a, b) && SameBytes<kEarlyLength, kEarlyLength + 4>(a, b) &&
           SameBytes<kPreDelayLine, kPreDelayPos>(a, b) && SameBytes<kPreDelayLength, kPreDelayLength + 4>(a, b) &&
           SameBytes<kCombLine, kCombPos>(a, b) && SameBytes<kCombLength, kCombLength + 12>(a, b) &&
           SameBytes<kAllpassLine, kAllpassPos>(a, b) && SameBytes<kAllpassLength, kAllpassLength + 8>(a, b) &&
           SameBytes<kLastAllpassLine, kLastAllpassPos>(a, b) &&
           SameBytes<kLastAllpassLength, kLastAllpassLength + 12>(a, b) && SameBytes<kBusIn, kBusOut + 4>(a, b);
}

// The store-policy chunks a written range lies in (false: too many).
bool AddChunks(uint32_t* chunks, uint32_t& count, uint32_t max, const Span& span) {
#if MKW_LEAN_MEMORY
    const uint32_t start = GuestMemoryAlias::Offset(span.start);
    const uint32_t first = start >> MemoryInline::kLeanPolicyShift;
    const uint32_t last = (start + span.bytes - 1) >> MemoryInline::kLeanPolicyShift;
    for (uint32_t chunk = first; chunk <= last; ++chunk) {
        bool known = false;
        for (uint32_t k = 0; k < count && !known; ++k) known = chunks[k] == chunk;
        if (known) continue;
        if (count == max) return false;
        chunks[count++] = chunk;
    }
    return true;
#else
    (void)chunks, (void)count, (void)max, (void)span;
    return false;
#endif
}

// What ResolveRangeHost reads for a written range besides its address: the chunks' store policy, now.
inline bool PolicyClear(const uint32_t* chunks, uint32_t count) {
#if MKW_LEAN_MEMORY
    const uint8_t* const policy = MemoryInline::g_leanBase + MemoryInline::kLeanPolicyOffset;
    for (uint32_t k = 0; k < count; ++k)
        if (policy[chunks[k]] != 0) return false;
    return true;
#else
    (void)chunks, (void)count;
    return false;
#endif
}

// Whether [start, end) overlaps a range of a sorted, disjoint list.
bool OverlapsSorted(const Segment* list, uint32_t count, uint32_t start, uint32_t end) {
    uint32_t lo = 0, hi = count;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (list[mid].end <= start)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < count && list[lo].start < end;
}
inline bool OverlapsSorted(const Segment* list, uint32_t count, const Span& span) {
    const uint32_t start = GuestMemoryAlias::Offset(span.start);
    return OverlapsSorted(list, count, start, start + span.bytes);
}

// Every ring of a plan, with whether its walk writes (pre-delay rings only with a pre-delay).
template <typename F>
void ForEachRing(Plan& p, F&& f) {
    for (uint32_t c = 0; c < 3; ++c) {
        for (uint32_t t = 0; t < 3; ++t) f(p.early[c][t], false);
        if (p.preDelay) f(p.pre[c], true);
        for (uint32_t k = 0; k < 3; ++k) f(p.comb[c][k], true);
        for (uint32_t k = 0; k < 2; ++k) f(p.ap[c][k], true);
        f(p.last[c], true);
    }
}

// The call's proof on its own ranges (Run's preconditions above); fills the plan.
bool ProveFresh(Plan& p, const uint8_t* s, const Span& constants, const Pinned& pinned, const uint32_t in[3],
                const uint32_t bi[3], const uint32_t bo[3], uint32_t busIn, uint32_t busOut) {
    p.s = s;
    p.constants = constants.host;
    p.busIn = busIn != 0;
    p.busOut = busOut != 0;
    Segment segments[kMaxSegments];
    uint32_t count = 0;
    for (uint32_t c = 0; c < 3; ++c) {
        if (!Resolve(p.in[c], in[c], kBufferBytes, true) || !Clear(p.in[c], true, pinned)) return false;
        AddSegment(segments, count, p.in[c], true);
        if (p.busIn) {
            if (!Resolve(p.bi[c], bi[c], kBufferBytes, false) || !Clear(p.bi[c], false, pinned)) return false;
            AddSegment(segments, count, p.bi[c], false);
        }
        if (p.busOut) {
            if (!Resolve(p.bo[c], bo[c], kBufferBytes, true) || !Clear(p.bo[c], true, pinned)) return false;
            AddSegment(segments, count, p.bo[c], true);
        }
    }

    for (uint32_t k = 0; k < 3; ++k) {
        p.earlyPos[k] = Load32(s + kEarlyPos + 4 * k);
        p.combPos[k] = Load32(s + kCombPos + 4 * k);
        p.combLength[k] = Load32(s + kCombLength + 4 * k);
        p.lastPos[k] = Load32(s + kLastAllpassPos + 4 * k);
        p.lastLength[k] = Load32(s + kLastAllpassLength + 4 * k);
    }
    for (uint32_t k = 0; k < 2; ++k) {
        p.apPos[k] = Load32(s + kAllpassPos + 4 * k);
        p.apLength[k] = Load32(s + kAllpassLength + 4 * k);
    }
    p.earlyLength = Load32(s + kEarlyLength);
    p.preDelayPos = Load32(s + kPreDelayPos);
    p.preDelayLength = Load32(s + kPreDelayLength);
    p.preDelay = p.preDelayLength != 0;

    // Each ring proven in range and clear of the pinned ranges; written rings recorded for the sweep.
    const auto ring = [&](Ring& r, uint32_t line, uint32_t pos, uint32_t length, bool write) {
        if (!SetupRing(r, line, pos, length, write) || !Clear(r.span, write, pinned)) return false;
        AddSegment(segments, count, r.span, write);
        return true;
    };
    for (uint32_t c = 0; c < 3; ++c) {
        const uint32_t earlyLine = Load32(s + kEarlyLine + 4 * c);
        for (uint32_t t = 0; t < 3; ++t)
            if (!ring(p.early[c][t], earlyLine, p.earlyPos[t], p.earlyLength, false)) return false;
        if (p.preDelay &&
            !ring(p.pre[c], Load32(s + kPreDelayLine + 4 * c), p.preDelayPos, p.preDelayLength, true))
            return false;
        for (uint32_t k = 0; k < 3; ++k)
            if (!ring(p.comb[c][k], Load32(s + kCombLine + 12 * c + 4 * k), p.combPos[k], p.combLength[k], true))
                return false;
        for (uint32_t k = 0; k < 2; ++k)
            if (!ring(p.ap[c][k], Load32(s + kAllpassLine + 8 * c + 4 * k), p.apPos[k], p.apLength[k], true))
                return false;
        if (!ring(p.last[c], Load32(s + kLastAllpassLine + 4 * c), p.lastPos[c], p.lastLength[c], true)) return false;
    }
    return Disjoint(segments, count);
}

Proven* FindProven(uint32_t fx, uint32_t r2, const uint8_t* s) {
    for (Proven& e : g_proven)
        if (e.used && e.fx == fx && e.r2 == r2 && e.generation == g_configGeneration && SameConfig(s, e.config))
            return &e;
    return nullptr;
}

// After a fresh proof (no bus in or out): the configuration's proof on whole lines, kept whether it passes or not.
void BuildProven(uint32_t fx, uint32_t r2, const uint8_t* s, const Plan& fresh, const Span& effect,
                 const Span& constants) {
    Proven* slot = nullptr;
    for (Proven& e : g_proven)
        if (e.used && e.fx == fx) slot = &e;
    if (slot == nullptr) {
        slot = &g_proven[g_provenNext];
        g_provenNext = (g_provenNext + 1) % 2;
    }
    Proven& e = *slot;
    ++g_proofStats.built;
    e.used = true;
    e.cacheable = false;
    e.generation = g_configGeneration;
    e.fx = fx;
    e.r2 = r2;
    std::memcpy(e.config, s, kEffectBytes);
    e.effect = effect;
    e.constants = constants;
    e.plan = fresh;
    e.lineCount = e.chunkCount = e.tripleCount = e.nextTriple = 0;
    Segment whole[kRings];
    uint32_t n = 0;
    bool ok = true;
    ForEachRing(e.plan, [&](Ring& r, bool write) {
        Span w;
        if (!ok || !Resolve(w, r.line, uint64_t{r.length} * 4, write) || Overlap(w, effect) ||
            (write && Overlap(w, constants)) || reinterpret_cast<uintptr_t>(w.host) != r.host0 ||
            (write && !AddChunks(e.chunks, e.chunkCount, kMaxChunks, w))) {
            ok = false;
            return;
        }
        r.span = w;
        AddSegment(whole, n, w, write);
    });
    if (!ok || !Disjoint(whole, n)) {  // (Disjoint sorts `whole`)
        ++g_proofStats.uncacheable;
        return;
    }
    for (uint32_t k = 0; k < n; ++k) {  // merged: overlapping read lines (the early taps) become one range
        if (e.lineCount != 0 && whole[k].start < e.lines[e.lineCount - 1].end)
            e.lines[e.lineCount - 1].end = std::max(e.lines[e.lineCount - 1].end, whole[k].end);
        else
            e.lines[e.lineCount++] = {whole[k].start, whole[k].end, true};
    }
    e.cover = {e.lines[0].start, e.lines[e.lineCount - 1].end, true};
    e.cacheable = true;
}

// The proven triple for these buffers, proving it first if new (null: they do not pass).
const Triple* FindTriple(Proven& e, const uint32_t in[3]) {
    for (uint32_t k = 0; k < e.tripleCount; ++k)
        if (e.triples[k].in[0] == in[0] && e.triples[k].in[1] == in[1] && e.triples[k].in[2] == in[2])
            return &e.triples[k];
    Triple t{};
    for (uint32_t c = 0; c < 3; ++c) {
        t.in[c] = in[c];
        if (!Resolve(t.span[c], in[c], kBufferBytes, true) || Overlap(t.span[c], e.effect) ||
            Overlap(t.span[c], e.constants) || OverlapsSorted(e.lines, e.lineCount, t.span[c]) ||
            !AddChunks(t.chunks, t.chunkCount, 6, t.span[c]))
            return nullptr;
        for (uint32_t d = 0; d < c; ++d)
            if (Overlap(t.span[c], t.span[d])) return nullptr;
    }
    ++g_proofStats.triples;
    const uint32_t slot = e.tripleCount < 4 ? e.tripleCount++ : (e.nextTriple++ % 4);
    e.triples[slot] = t;
    return &e.triples[slot];
}

// A call of a proven configuration: its per-call checks, and its plan from the configuration's.
bool ProveCached(const Proven& e, const Triple& t, Plan& p, const uint8_t* s, const Span& frame) {
    if (OverlapsSorted(e.lines, e.lineCount, frame) || Overlap(t.span[0], frame) || Overlap(t.span[1], frame) ||
        Overlap(t.span[2], frame) || !PolicyClear(e.chunks, e.chunkCount) || !PolicyClear(t.chunks, t.chunkCount))
        return false;
    p = e.plan;
    bool inside = true;
    for (uint32_t k = 0; k < 3; ++k) {
        p.earlyPos[k] = Load32(s + kEarlyPos + 4 * k);
        p.combPos[k] = Load32(s + kCombPos + 4 * k);
        p.lastPos[k] = Load32(s + kLastAllpassPos + 4 * k);
        inside &= p.earlyPos[k] < p.earlyLength && p.combPos[k] < p.combLength[k] && p.lastPos[k] < p.lastLength[k];
    }
    for (uint32_t k = 0; k < 2; ++k) {
        p.apPos[k] = Load32(s + kAllpassPos + 4 * k);
        inside &= p.apPos[k] < p.apLength[k];
    }
    p.preDelayPos = Load32(s + kPreDelayPos);
    if (p.preDelay) inside &= p.preDelayPos < p.preDelayLength;
    if (!inside) return false;
    for (uint32_t c = 0; c < 3; ++c) {
        for (uint32_t k = 0; k < 3; ++k) p.early[c][k].pos = p.earlyPos[k];
        if (p.preDelay) p.pre[c].pos = p.preDelayPos;
        for (uint32_t k = 0; k < 3; ++k) p.comb[c][k].pos = p.combPos[k];
        for (uint32_t k = 0; k < 2; ++k) p.ap[c][k].pos = p.apPos[k];
        p.last[c].pos = p.lastPos[c];
        p.in[c] = t.span[c];
    }
    return true;
}

// The rings' per-call ranges, as SetupRing computes them (a cached plan carries whole lines; Footprint, the marks and
// the samples loops do not read these, the check modes' snapshots do).
void FillSpans(Plan& p) {
    ForEachRing(p, [](Ring& r, bool) {
        const uint64_t end = uint64_t{r.pos} + kSamples;
        const uint32_t lo = end <= r.length ? r.pos : 0;
        const uint64_t hi = end <= r.length ? end : r.length;
        r.span = {r.line + (lo << 2), static_cast<uint32_t>((hi - lo) * 4),
                  reinterpret_cast<uint8_t*>(r.host0 + uintptr_t{4} * lo)};
    });
}

// AXFX_PROOF=check: the field where two plans differ (null: none).
const char* PlanDifference(Plan& a, Plan& b) {
    const auto span = [](const Span& x, const Span& y) { return x.start == y.start && x.bytes == y.bytes && x.host == y.host; };
    if (a.s != b.s || a.constants != b.constants) return "struct or constants";
    if (a.busIn != b.busIn || a.busOut != b.busOut || a.preDelay != b.preDelay) return "flags";
    for (uint32_t c = 0; c < 3; ++c)
        if (!span(a.in[c], b.in[c])) return "buffer";
    if (std::memcmp(a.earlyPos, b.earlyPos, sizeof(a.earlyPos)) != 0 || a.earlyLength != b.earlyLength ||
        a.preDelayPos != b.preDelayPos || a.preDelayLength != b.preDelayLength ||
        std::memcmp(a.combPos, b.combPos, sizeof(a.combPos)) != 0 ||
        std::memcmp(a.combLength, b.combLength, sizeof(a.combLength)) != 0 ||
        std::memcmp(a.apPos, b.apPos, sizeof(a.apPos)) != 0 ||
        std::memcmp(a.apLength, b.apLength, sizeof(a.apLength)) != 0 ||
        std::memcmp(a.lastPos, b.lastPos, sizeof(a.lastPos)) != 0 ||
        std::memcmp(a.lastLength, b.lastLength, sizeof(a.lastLength)) != 0)
        return "positions or lengths";
    const Ring* ra[kRings];
    const Ring* rb[kRings];
    uint32_t na = 0, nb = 0;
    ForEachRing(a, [&](Ring& r, bool) { ra[na++] = &r; });
    ForEachRing(b, [&](Ring& r, bool) { rb[nb++] = &r; });
    if (na != nb) return "ring count";
    for (uint32_t k = 0; k < na; ++k)
        if (ra[k]->line != rb[k]->line || ra[k]->pos != rb[k]->pos || ra[k]->length != rb[k]->length ||
            ra[k]->host0 != rb[k]->host0 || !span(ra[k]->span, rb[k]->span))
            return "ring";
    return nullptr;
}

void LogProofs() {
    const ProofStats& g = g_proofStats;
    RT_LOGF(RT_TAG_AUDIO,
            "axfx reverb proofs: calls=%" PRIu64 " cached=%" PRIu64 " (declined %" PRIu64 "), configurations proven=%" PRIu64
            " (uncacheable %" PRIu64 "), buffer triples=%" PRIu64 "\n",
            g.calls, g.cached, g.declined, g.built, g.uncacheable, g.triples);
    if (GetProofMode() == ProofMode::kCheck)
        RT_LOGF(RT_TAG_AUDIO, "axfx reverb proofs check: compared=%" PRIu64 " mismatches=%" PRIu64 "\n", g.checked,
                g.mismatches);
}

// AXFX_ASYNC: what a job reads or writes, coarsely, for the overlap tests of calls made while it is pending (each bus
// buffer, one range covering every line, the low-pass words). An effect's lines lie in its own work buffer, so the
// cover adds only false overlaps, which join early.
constexpr uint32_t kMaxFootprint = 11;
uint32_t Footprint(const Plan& p, uint32_t fx, Segment out[kMaxFootprint]) {
    uint32_t n = 0;
    for (uint32_t c = 0; c < 3; ++c) {
        AddSegment(out, n, p.in[c], true);
        if (p.busIn) AddSegment(out, n, p.bi[c], false);
        if (p.busOut) AddSegment(out, n, p.bo[c], true);
    }
    uint32_t lo = UINT32_MAX, hi = 0;
    const auto cover = [&](const Ring& r) {
        const uint32_t start = GuestMemoryAlias::Offset(r.span.start);
        lo = std::min(lo, start);
        hi = std::max(hi, start + r.span.bytes);
    };
    for (uint32_t c = 0; c < 3; ++c) {
        for (uint32_t t = 0; t < 3; ++t) cover(p.early[c][t]);
        if (p.preDelay) cover(p.pre[c]);
        for (uint32_t k = 0; k < 3; ++k) cover(p.comb[c][k]);
        for (uint32_t k = 0; k < 2; ++k) cover(p.ap[c][k]);
        cover(p.last[c]);
    }
    out[n++] = {lo, hi, true};
    const uint32_t lpf = GuestMemoryAlias::Offset((fx + kLastLpfOut));
    out[n++] = {lpf, lpf + 12, true};
    return n;
}

// The same for a call of a proven configuration (no bus in or out): the buffers, the whole lines' cover, the low-pass
// words.
uint32_t CachedFootprint(const Proven& e, const Plan& p, uint32_t fx, Segment out[kMaxFootprint]) {
    uint32_t n = 0;
    for (uint32_t c = 0; c < 3; ++c) AddSegment(out, n, p.in[c], true);
    out[n++] = e.cover;
    const uint32_t lpf = GuestMemoryAlias::Offset((fx + kLastLpfOut));
    out[n++] = {lpf, lpf + 12, true};
    return n;
}

// AXFX_ASYNC (above): the call's proven plan, as a job; here everything the original leaves that does not depend on the
// samples, as Run's tail leaves it.
void RunAsync(CpuContext* ctx, const Plan& p, const Span& frame, const Span& effect, const Span& constants,
              const uint32_t in[3], const uint32_t bi[3], const uint32_t bo[3], const Segment* footprint,
              uint32_t footprintCount) {
    const uint32_t r1 = ctx->gpr[1];
    const uint32_t r3 = ctx->gpr[3];
    const uint32_t fx = ctx->gpr[4];
    const uint32_t sp = r1 - kFrameBytes;
    Async::Job& job = Async::Acquire();
    job.kind = Async::Kind::kReverb;
    job.samples = !ExactAudio() ? (VectorAudio() ? Async::Samples::kVector : Async::Samples::kFast)
                                : (g_mkwNiFlushThreshold == 0.0 ? Async::Samples::kExact : Async::Samples::kExactNi);
    job.fpControl = MkwGetHostFpControl();
    std::memcpy(job.effect, effect.host, kEffectBytes);
    std::memcpy(job.constants, constants.host, kConstantsBytes);
    job.plan = p;
    job.plan.s = job.effect;
    job.plan.constants = job.constants;
    job.lpf = effect.host + kLastLpfOut;
    job.frame = frame.host;
    MarkStores4(fx + kLastLpfOut, 12);  // the job's struct stores, marked first as the translated stores are

    // The struct's positions after 96 samples: each proven inside its line, so 96 steps of Next are one modulo.
    const auto after = [](uint32_t pos, uint32_t length, uint32_t n) {
        return static_cast<uint32_t>((uint64_t{pos} + n) % length);
    };
    for (uint32_t k = 0; k < 3; ++k)
        MemoryInline::FlatWrite32(fx + kEarlyPos + 4 * k, after(p.earlyPos[k], p.earlyLength, kSamples));
    if (p.preDelay) MemoryInline::FlatWrite32(fx + kPreDelayPos, after(p.preDelayPos, p.preDelayLength, kSamples));
    for (uint32_t k = 0; k < 3; ++k)
        MemoryInline::FlatWrite32(fx + kCombPos + 4 * k, after(p.combPos[k], p.combLength[k], kSamples));
    for (uint32_t k = 0; k < 2; ++k)
        MemoryInline::FlatWrite32(fx + kAllpassPos + 4 * k, after(p.apPos[k], p.apLength[k], kSamples));
    for (uint32_t c = 0; c < 3; ++c)
        MemoryInline::FlatWrite32(fx + kLastAllpassPos + 4 * c, after(p.lastPos[c], p.lastLength[c], kSamples));

    // The frame, but for the last sample's mix and f9, f8, f7: sentinels until Join writes the job's values.
    StorePrologue(ctx, r1, frame);
    for (uint32_t c = 0; c < 3; ++c) MemoryInline::FlatWriteRam32(sp + 44 + 4 * c, in[c] + kBufferBytes);
    // The last int->double conversion: the third channel's last input word (+ bus in). No earlier sample writes it,
    // and no line or other buffer overlaps it (Disjoint).
    const uint32_t sample = Load32(p.in[2].host + 4 * (kSamples - 1));
    const uint32_t lastConvert =
        (p.busIn ? sample + Load32(p.bi[2].host + 4 * (kSamples - 1)) : sample) ^ 0x80000000u;
    if (p.busIn) {
        for (uint32_t c = 0; c < 3; ++c) MemoryInline::FlatWrite32(sp + 8 + 4 * c, bi[c] + kBufferBytes);
        MemoryInline::FlatWriteRam32(sp + 56, 0x43300000u);
        MemoryInline::FlatWriteRam32(sp + 60, lastConvert);
    } else {
        MemoryInline::FlatWriteRam32(sp + 64, 0x43300000u);
        MemoryInline::FlatWriteRam32(sp + 68, lastConvert);
    }
    for (uint32_t k = 0; k < 3; ++k) MemoryInline::FlatWrite32(sp + Async::kFillOffsets[k], Async::kSentinel + k);
    for (uint32_t k = 3; k < Async::kFillWords; ++k)
        MemoryInline::FlatWriteRam32(sp + Async::kFillOffsets[k], Async::kSentinel + k);

    // The register file, but for f2 and f7-f13 (the last sample's values; volatile, and the callers never read them).
    ctx->gpr[0] = ctx->lr;
    ctx->gpr[3] = p.busOut ? bo[0] + kBufferBytes : (p.busIn ? bi[1] : r3);
    ctx->gpr[5] = p.busOut ? bo[1] + kBufferBytes : (p.busIn ? bi[0] : in[1]);
    ctx->gpr[6] = p.busOut ? bo[2] + kBufferBytes : in[0];
    ctx->gpr[7] = p.apLength[1];
    ctx->gpr[8] = after(p.apPos[1], p.apLength[1], kSamples - 1) + 1;  // the last sample's position + 1
    ctx->gpr[9] = fx + 8;
    ctx->gpr[10] = in[1] + kBufferBytes - 4;
    ctx->gpr[11] = r1;
    ctx->gpr[12] = in[0] + kBufferBytes - 4;
    double f[7];
    ParamRegisters(p, job.samples, f);
    for (const uint32_t r : {0u, 1u, 3u, 4u, 5u, 6u}) ctx->fpr[r].d = f[r];
    uint32_t cr = ctx->cr;
    SetCRResident(cr, ctx->xer, 0, kSamples, kSamples);  // the loop's final cmplwi r31, 96
    ctx->cr = cr;
    ctx->ctr = 0;

    Async::Hold(footprint, footprintCount);  // pending from here
    if (Async::GetMode() == Async::Mode::kCheck) {
        const Span lpfSpan{fx + kLastLpfOut, 12, effect.host + kLastLpfOut};
        job.rangeCount = 0;
        for (uint32_t c = 0; c < 3; ++c) {
            job.AddRange(p.in[c], true);
            if (p.busIn) job.AddRange(p.bi[c], false);
            if (p.busOut) job.AddRange(p.bo[c], true);
            for (uint32_t t = 0; t < 3; ++t) job.AddRange(p.early[c][t].span, false);
            if (p.preDelay) job.AddRange(p.pre[c].span, true);
            for (uint32_t k = 0; k < 3; ++k) job.AddRange(p.comb[c][k].span, true);
            for (uint32_t k = 0; k < 2; ++k) job.AddRange(p.ap[c][k].span, true);
            job.AddRange(p.last[c].span, true);
        }
        job.AddRange(lpfSpan, true);
        uint32_t expect[Async::kMaxExpect];
        uint32_t m = 0;
        for (uint32_t k = 0; k < 3; ++k) expect[m++] = after(p.earlyPos[k], p.earlyLength, kSamples);
        expect[m++] = p.preDelay ? after(p.preDelayPos, p.preDelayLength, kSamples) : p.preDelayPos;
        for (uint32_t k = 0; k < 3; ++k) expect[m++] = after(p.combPos[k], p.combLength[k], kSamples);
        for (uint32_t k = 0; k < 2; ++k) expect[m++] = after(p.apPos[k], p.apLength[k], kSamples);
        for (uint32_t c = 0; c < 3; ++c) expect[m++] = after(p.lastPos[c], p.lastLength[c], kSamples);
        expect[m++] = ctx->gpr[8];
        expect[m++] = lastConvert;
        for (const uint32_t r : {0u, 1u, 3u, 4u, 5u, 6u}) {
            uint64_t u;
            std::memcpy(&u, &f[r], sizeof(u));
            expect[m++] = static_cast<uint32_t>(u >> 32);
            expect[m++] = static_cast<uint32_t>(u);
        }
        job.SetExpect(expect, m);
    }
    Async::Push(job);
}

bool Run(CpuContext* ctx) {
    const uint32_t r1 = ctx->gpr[1];
    const uint32_t r2 = ctx->gpr[2];
    const uint32_t r3 = ctx->gpr[3];
    const uint32_t fx = ctx->gpr[4];
    const uint32_t sp = r1 - kFrameBytes;

    Span frame, effect;
    if (!Resolve(frame, sp, kFrameSpan, true) || !Resolve(effect, fx, kEffectBytes, true) || Overlap(frame, effect))
        return false;
    const uint8_t* s = effect.host;
    // A job handed over earlier in this AX frame must be done with what this call reads first or writes in place.
    if (Async::Pending() && (Async::Overlaps(sp, kFrameSpan) || Async::Overlaps(fx, kEffectBytes) ||
                             Async::Overlaps(r3, 12)))
        Async::Join();

    const uint32_t active = Load32(s + kActive);
    if (active != 0) {
        // Bypassed this frame: clear the flag's bit 1 and return.
        g_bypassed.fetch_add(1, std::memory_order_relaxed);
        StorePrologue(ctx, r1, frame);
        MemoryInline::FlatWrite32(fx + kActive, active & ~2u);
        ctx->gpr[0] = ctx->lr;
        ctx->gpr[11] = r1;
        uint32_t cr = ctx->cr;
        SetCRResident(cr, ctx->xer, 0, static_cast<int32_t>(active), static_cast<int32_t>(0));
        ctx->cr = cr;
        return true;
    }

    Span constants, update;
    if (!Resolve(constants, r2 + static_cast<uint32_t>(kConstantsOffset), kConstantsBytes, false) ||
        Overlap(constants, frame) || Overlap(constants, effect))
        return false;
    if (!Resolve(update, r3, 12, false) || Overlap(update, frame)) return false;
    uint32_t in[3], bi[3] = {}, bo[3] = {};
    for (uint32_t c = 0; c < 3; ++c) in[c] = Load32(update.host + 4 * c);
    const uint32_t busIn = Load32(s + kBusIn);
    const uint32_t busOut = Load32(s + kBusOut);
    if (busIn != 0) {
        Span bus;
        if (!Resolve(bus, busIn, 12, false) || Overlap(bus, frame)) return false;
        for (uint32_t c = 0; c < 3; ++c) bi[c] = Load32(bus.host + 4 * c);
    }
    if (busOut != 0) {
        Span bus;
        if (!Resolve(bus, busOut, 12, false) || Overlap(bus, frame)) return false;
        for (uint32_t c = 0; c < 3; ++c) bo[c] = Load32(bus.host + 4 * c);
    }

    // The proof (AXFX_PROOF above): the configuration's cached one with this call's checks, or this call's own.
    const Pinned pinned{&frame, &effect, &constants};
    Plan p;
    const ProofMode proofMode = busIn == 0 && busOut == 0 ? GetProofMode() : ProofMode::kFresh;
    Proven* proven = nullptr;
    bool cached = false;  // p is ProveCached's plan
    bool fresh = false;   // p is ProveFresh's plan
    if (proofMode != ProofMode::kFresh) {
        const uint64_t calls = ++g_proofStats.calls;
        if (calls == 1 || calls % 20000 == 0) LogProofs();
        proven = FindProven(fx, r2, s);
        const Triple* triple = proven != nullptr && proven->cacheable ? FindTriple(*proven, in) : nullptr;
        if (proofMode == ProofMode::kCached) {
            cached = triple != nullptr && ProveCached(*proven, *triple, p, s, frame);
            if (triple != nullptr && !cached) ++g_proofStats.declined;
        } else if (triple != nullptr) {
            // Both, compared: a cached yes must be a per-call yes with the same plan. The per-call result stays.
            Plan pc;
            const bool cachedYes = ProveCached(*proven, *triple, pc, s, frame);
            const bool freshYes = ProveFresh(p, s, constants, pinned, in, bi, bo, busIn, busOut);
            fresh = true;
            if (cachedYes) FillSpans(pc);
            const char* difference = cachedYes ? (freshYes ? PlanDifference(pc, p) : "cached yes, per-call no") : nullptr;
            ++g_proofStats.checked;
            if (difference != nullptr && ++g_proofStats.mismatches <= 8)
                RT_LOGF(RT_TAG_AUDIO, "axfx reverb proofs check MISMATCH: %s\n", difference);
            if (!freshYes) return false;
        }
    }
    if (!cached && !fresh) {
        if (!ProveFresh(p, s, constants, pinned, in, bi, bo, busIn, busOut)) return false;
        if (proofMode != ProofMode::kFresh && proven == nullptr) BuildProven(fx, r2, s, p, effect, constants);
    }
    if (cached) {
        ++g_proofStats.cached;
        if (CheckMode() || Async::GetMode() == Async::Mode::kCheck) FillSpans(p);  // their snapshots read the spans
    }
    // The buffers, lines and bus blocks too. The values read from them above may have been mid-write: read them again.
    Segment footprint[kMaxFootprint];
    const uint32_t footprintCount =
        !Async::On() ? 0 : (cached ? CachedFootprint(*proven, p, fx, footprint) : Footprint(p, fx, footprint));
    if (Async::Pending() && (Async::Conflicts(footprint, footprintCount) || (busIn != 0 && Async::Overlaps(busIn, 12)) ||
                             (busOut != 0 && Async::Overlaps(busOut, 12)))) {
        Async::Join();
        return Run(ctx);
    }

    // Proven: from here on the call runs natively. Marks first, as each translated store marks before it stores.
    for (uint32_t c = 0; c < 3; ++c) {
        MarkStores4(in[c], kBufferBytes);
        if (p.busOut) MarkStores4(bo[c], kBufferBytes);
        if (p.preDelay) MarkRing(p.pre[c]);
        for (uint32_t k = 0; k < 3; ++k) MarkRing(p.comb[c][k]);
        for (uint32_t k = 0; k < 2; ++k) MarkRing(p.ap[c][k]);
        MarkRing(p.last[c]);
    }

    if (Async::On()) {
        RunAsync(ctx, p, frame, effect, constants, in, bi, bo, footprint, footprintCount);
        return true;
    }

    if (!ExactAudio()) {
        if (CheckMode())
            CheckFast(p);
        else if (VectorAudio())
            ProcessFastV(p);
        else
            ProcessFast(p);
    } else if (g_mkwNiFlushThreshold == 0.0)
        Process<false>(p);
    else
        Process<true>(p);

    // The effect struct's final state: every field the original stores, with the value it stored last.
    for (uint32_t k = 0; k < 3; ++k) MemoryInline::FlatWrite32(fx + kEarlyPos + 4 * k, p.earlyPos[k]);
    if (p.preDelay) MemoryInline::FlatWrite32(fx + kPreDelayPos, p.preDelayPos);
    for (uint32_t k = 0; k < 3; ++k) MemoryInline::FlatWrite32(fx + kCombPos + 4 * k, p.combPos[k]);
    for (uint32_t k = 0; k < 2; ++k) MemoryInline::FlatWrite32(fx + kAllpassPos + 4 * k, p.apPos[k]);
    for (uint32_t c = 0; c < 3; ++c) {
        MemoryInline::FlatWrite32(fx + kLastAllpassPos + 4 * c, p.lastPos[c]);
        MemoryInline::FlatWrite32(fx + kLastLpfOut + 4 * c, Narrow(p.lpf[c]));
    }

    // The stack frame's final state: the prologue, the walked buffer pointers, the conversion double and the temps.
    StorePrologue(ctx, r1, frame);
    for (uint32_t c = 0; c < 3; ++c) MemoryInline::FlatWriteRam32(sp + 44 + 4 * c, in[c] + kBufferBytes);
    if (busIn != 0) {
        for (uint32_t c = 0; c < 3; ++c) MemoryInline::FlatWrite32(sp + 8 + 4 * c, bi[c] + kBufferBytes);
        MemoryInline::FlatWriteRam32(sp + 56, 0x43300000u);
        MemoryInline::FlatWriteRam32(sp + 60, p.lastConvert);
    } else {
        MemoryInline::FlatWriteRam32(sp + 64, 0x43300000u);
        MemoryInline::FlatWriteRam32(sp + 68, p.lastConvert);
    }
    for (uint32_t c = 0; c < 3; ++c) MemoryInline::FlatWrite32(sp + 20 + 4 * c, Narrow(p.mix[c]));
    MemoryInline::FlatWriteRam32(sp + 32, Narrow(p.f[9]));
    MemoryInline::FlatWriteRam32(sp + 36, Narrow(p.f[8]));
    MemoryInline::FlatWriteRam32(sp + 40, Narrow(p.f[7]));

    // The register file at the original's return.
    ctx->gpr[0] = ctx->lr;
    ctx->gpr[3] = busOut != 0 ? bo[0] + kBufferBytes : (busIn != 0 ? bi[1] : r3);
    ctx->gpr[5] = busOut != 0 ? bo[1] + kBufferBytes : (busIn != 0 ? bi[0] : in[1]);
    ctx->gpr[6] = busOut != 0 ? bo[2] + kBufferBytes : in[0];
    ctx->gpr[7] = p.apLength[1];
    ctx->gpr[8] = p.lastApInc;
    ctx->gpr[9] = fx + 8;
    ctx->gpr[10] = in[1] + kBufferBytes - 4;
    ctx->gpr[11] = r1;
    ctx->gpr[12] = in[0] + kBufferBytes - 4;
    for (uint32_t r = 0; r < 14; ++r) ctx->fpr[r].d = p.f[r];
    uint32_t cr = ctx->cr;
    SetCRResident(cr, ctx->xer, 0, kSamples, kSamples);  // the loop's final cmplwi r31, 96
    ctx->cr = cr;
    ctx->ctr = 0;
    return true;
}

}  // namespace Reverb

// ======================================================================================================================
// AXFXDelayCallback(AXFX_BUFFERUPDATE* r3, AXFX_DELAY* r4)
// ======================================================================================================================
namespace Delay {

using Reverb::CheckMode;
using Reverb::I8;
using Reverb::LoadRing;
using Reverb::LoadWords;
using Reverb::LogCheck;
using Reverb::Plus7;
using Reverb::Plus8;
using Reverb::Snapshot;
using Reverb::StoreRing;
using Reverb::StoreWords;
using Reverb::U8;
using Reverb::VectorAudio;

constexpr uint32_t kAddress = 0x80203A10u;
constexpr uint32_t kFrameBytes = 32;  // stwu r1, -32(r1); r28-r31 at 16..28

// AXFX_DELAY
constexpr uint32_t kLine = 0x00;      // s32* [3]
constexpr uint32_t kPos = 0x0C;       // u32 [3]
constexpr uint32_t kLength = 0x18;    // u32 [3]
constexpr uint32_t kFeedback = 0x24;  // s32 [3]
constexpr uint32_t kOutput = 0x30;    // s32 [3]
constexpr uint32_t kActive = 0x3C;
constexpr uint32_t kEffectBytes = 0x40;

std::atomic<uint64_t> g_bypassed{0};  // native calls that took the skip path (statistics)

void StorePrologue(CpuContext* ctx, uint32_t r1) {
    const uint32_t sp = r1 - kFrameBytes;
    MemoryInline::FlatWriteRam32(sp, r1);
    for (uint32_t r = 31; r >= 28; --r) MemoryInline::FlatWriteRam32(sp + 16 + (r - 28) * 4, ctx->gpr[r]);
}

inline uint32_t Scale(uint32_t sample, uint32_t gain) { return static_cast<uint32_t>(static_cast<int32_t>(sample * gain) >> 7); }
inline U8 Scale8(U8 sample, uint32_t gain) { return __builtin_bit_cast(U8, __builtin_bit_cast(I8, sample * gain) >> 7); }

// AXFX_ASYNC: what the last sample (95) reads from a line, without the samples loop. The slot it reads holds its value
// from before the call, rewritten by each earlier visit of this call (a line shorter than 96 is visited more than once):
// visit i stores input word i + Scale(what it read). Valid when no line or buffer overlaps another (each input word is
// then its value from before the call when its sample reads it).
uint32_t LastOld(const Ring& line, const uint8_t* in, uint32_t pos, uint32_t length, uint32_t feedback) {
    const uint32_t last = static_cast<uint32_t>((uint64_t{pos} + kSamples - 1) % length);
    uint32_t old = Load32(line.At(last));
    for (uint32_t i = (kSamples - 1) % length; i < kSamples - 1; i += length)  // the slot's earlier visits
        old = Load32(in + 4 * i) + Scale(old, feedback);
    return old;
}

// The original's order: per sample, the three line reads, the three line writes, then the three sample writes.
void LoopScalar(Loop& L) {
    uint32_t a = 0, b = 0, d = 0, written2 = 0, inc2 = 0, product2 = 0;
    uint32_t* const pos = L.pos;
    for (uint32_t i = 0; i < kSamples; ++i) {
        uint8_t* q0 = L.line[0].At(pos[0]);
        uint8_t* q1 = L.line[1].At(pos[1]);
        uint8_t* q2 = L.line[2].At(pos[2]);
        a = Load32(q0);
        const uint32_t in0 = Load32(L.in[0] + 4 * i);
        b = Load32(q1);
        d = Load32(q2);
        Store32(q0, in0 + Scale(a, L.feedback[0]));
        Store32(q1, Load32(L.in[1] + 4 * i) + Scale(b, L.feedback[1]));
        written2 = Load32(L.in[2] + 4 * i) + Scale(d, L.feedback[2]);
        Store32(q2, written2);
        pos[0] = Next(pos[0], L.length[0]);
        pos[1] = Next(pos[1], L.length[1]);
        inc2 = pos[2] + 1;
        pos[2] = Next(pos[2], L.length[2]);
        Store32(L.in[0] + 4 * i, Scale(a, L.output[0]));
        Store32(L.in[1] + 4 * i, Scale(b, L.output[1]));
        product2 = d * L.output[2];
        Store32(L.in[2] + 4 * i, static_cast<uint32_t>(static_cast<int32_t>(product2) >> 7));
    }
    L.b = b;
    L.d = d;
    L.written2 = written2;
    L.inc2 = inc2;
    L.product2 = product2;
}

// One channel at a time (lines and buffers proven disjoint, lines at least 8 slots), 8 samples a step; a line access
// that wraps inside the step reads and writes the slots up to its end and from its start, as the reverb's do.
void LoopVector(Loop& L) {
    for (uint32_t c = 0; c < 3; ++c) {
        const uint32_t length = L.length[c];
        uint32_t pos = L.pos[c];
        for (uint32_t i = 0; i < kSamples; i += 8) {
            const U8 old = LoadRing(L.line[c], pos, length);
            const U8 written = LoadWords(L.in[c] + 4 * i) + Scale8(old, L.feedback[c]);
            StoreRing(L.line[c], pos, length, written);
            StoreWords(L.in[c] + 4 * i, Scale8(old, L.output[c]));
            if (i + 8 == kSamples) {  // the last sample's values
                if (c == 1) L.b = old[7];
                if (c == 2) {
                    L.d = old[7];
                    L.written2 = written[7];
                    L.inc2 = Plus7(pos, length) + 1;  // its position + 1, before the wrap
                    L.product2 = L.d * L.output[2];
                }
            }
            pos = Plus8(pos, length);
        }
        L.pos[c] = pos;
    }
}

// AXFX_CHECK=1: the scalar and the vector loop from the same state; the scalar loop's bytes and results stay.
void CheckLoop(Loop& L) {
    static const int site = InsnCount::On() ? InsnCount::Site("axfx.delay.loop") : -1;
    thread_local Snapshot snap, ref;
    snap.spans.clear();
    for (uint32_t c = 0; c < 3; ++c) {
        snap.Add(Span{0, kBufferBytes, L.in[c]});
        snap.Add(L.line[c].span);
    }
    ref.spans = snap.spans;
    snap.Save();
    Loop old = L;
    uint64_t t = InsnCount::On() ? InsnCount::Read() : 0;
    LoopScalar(old);
    if (InsnCount::On()) InsnCount::Add(site, InsnCount::kOriginal, InsnCount::Since(t));
    ref.Save();
    snap.Restore();
    t = InsnCount::On() ? InsnCount::Read() : 0;
    LoopVector(L);
    if (InsnCount::On()) InsnCount::Add(site, InsnCount::kNative, InsnCount::Since(t));
    snap.Save();
    const bool same = snap.bytes == ref.bytes && std::memcmp(old.pos, L.pos, sizeof(L.pos)) == 0 && old.b == L.b &&
                      old.d == L.d && old.written2 == L.written2 && old.inc2 == L.inc2 && old.product2 == L.product2;
    static std::atomic<uint64_t> calls{0}, mismatches{0};
    const uint64_t n = calls.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t bad = mismatches.fetch_add(same ? 0 : 1, std::memory_order_relaxed) + (same ? 0 : 1);
    if (!same && bad <= 8)
        RT_LOGF(RT_TAG_AUDIO, "axfx check MISMATCH delay: pos %u/%u/%u vs %u/%u/%u, bytes %s\n", old.pos[0], old.pos[1],
                old.pos[2], L.pos[0], L.pos[1], L.pos[2], snap.bytes == ref.bytes ? "same" : "differ");
    LogCheck("delay.loop", n, bad);
    ref.Restore();
    L = old;
}

bool Run(CpuContext* ctx) {
    const uint32_t r1 = ctx->gpr[1];
    const uint32_t r3 = ctx->gpr[3];
    const uint32_t fx = ctx->gpr[4];

    Span frame, effect;
    if (!Resolve(frame, r1 - kFrameBytes, kFrameBytes, true) || !Resolve(effect, fx, kEffectBytes, true) ||
        Overlap(frame, effect))
        return false;
    const uint8_t* s = effect.host;
    // A job handed over earlier in this AX frame must be done with what this call reads first or writes in place.
    if (Async::Pending() && (Async::Overlaps(r1 - kFrameBytes, kFrameBytes) || Async::Overlaps(fx, kEffectBytes) ||
                             Async::Overlaps(r3, 12)))
        Async::Join();

    const uint32_t active = Load32(s + kActive);
    if (active != 0) {
        g_bypassed.fetch_add(1, std::memory_order_relaxed);
        StorePrologue(ctx, r1);
        MemoryInline::FlatWrite32(fx + kActive, active & ~2u);
        ctx->gpr[0] = active & ~2u;
        uint32_t cr = ctx->cr;
        SetCRResident(cr, ctx->xer, 0, static_cast<int32_t>(active), static_cast<int32_t>(0));
        ctx->cr = cr;
        return true;
    }

    Span update;
    if (!Resolve(update, r3, 12, false) || Overlap(update, frame)) return false;
    const Pinned pinned{&frame, &effect, nullptr};
    uint32_t in[3], feedback[3], output[3], pos[3], length[3];
    Span inSpan[3];
    Ring line[3];
    for (uint32_t c = 0; c < 3; ++c) {
        in[c] = Load32(update.host + 4 * c);
        if (!Resolve(inSpan[c], in[c], kBufferBytes, true) || !Clear(inSpan[c], true, pinned)) return false;
        pos[c] = Load32(s + kPos + 4 * c);
        length[c] = Load32(s + kLength + 4 * c);
        feedback[c] = Load32(s + kFeedback + 4 * c);
        output[c] = Load32(s + kOutput + 4 * c);
        if (!SetupRing(line[c], Load32(s + kLine + 4 * c), pos[c], length[c], true) ||
            !Clear(line[c].span, true, pinned))
            return false;
    }
    // The buffers and lines too. The values read from them above may have been mid-write: read them again.
    // Footprint: each buffer, and one range covering the lines (Reverb::Footprint).
    Segment footprint[4];
    uint32_t count = 0;
    uint32_t lo = UINT32_MAX, hi = 0;
    for (uint32_t c = 0; c < 3; ++c) {
        AddSegment(footprint, count, inSpan[c], true);
        const uint32_t start = GuestMemoryAlias::Offset(line[c].span.start);
        lo = std::min(lo, start);
        hi = std::max(hi, start + line[c].span.bytes);
    }
    footprint[count++] = {lo, hi, true};
    if (Async::Pending() && Async::Conflicts(footprint, count)) {
        Async::Join();
        return Run(ctx);
    }

    for (uint32_t c = 0; c < 3; ++c) {
        MarkStores4(in[c], kBufferBytes);
        MarkRing(line[c]);
    }

    Loop loop;
    for (uint32_t c = 0; c < 3; ++c) {
        loop.in[c] = inSpan[c].host;
        loop.line[c] = line[c];
        loop.pos[c] = pos[c];
        loop.length[c] = length[c];
        loop.feedback[c] = feedback[c];
        loop.output[c] = output[c];
    }
    // The vector loop runs the channels one after another: with every line and buffer disjoint from the others, the
    // order of their accesses is unobservable (the scalar loop keeps the original's, for lines and buffers that alias).
    bool separate = true;  // no line or buffer overlaps another
    for (uint32_t c = 0; c < 3 && separate; ++c)
        for (uint32_t e = 0; e < 3 && separate; ++e)
            separate = !Overlap(line[c].span, inSpan[e]) && (e <= c || (!Overlap(line[c].span, line[e].span) &&
                                                                       !Overlap(inSpan[c], inSpan[e])));
    const bool disjoint = separate && length[0] >= 8 && length[1] >= 8 && length[2] >= 8;  // (the vector loop's)
    uint32_t b, d, written2, inc2, product2;
    if (Async::On() && separate) {
        // AXFX_ASYNC (above): the samples loop as a job. Its returned values come from the last sample: what it reads
        // from lines 1 and 2 (LastOld) and its third input word (written by that sample only, after the read).
        Async::Job& job = Async::Acquire();
        job.kind = Async::Kind::kDelay;
        job.samples = disjoint && VectorAudio() ? Async::Samples::kVector : Async::Samples::kScalar;
        job.fpControl = MkwGetHostFpControl();
        job.loop = loop;
        b = LastOld(line[1], inSpan[1].host, pos[1], length[1], feedback[1]);
        d = LastOld(line[2], inSpan[2].host, pos[2], length[2], feedback[2]);
        written2 = Load32(inSpan[2].host + 4 * (kSamples - 1)) + Scale(d, feedback[2]);
        inc2 = static_cast<uint32_t>((uint64_t{pos[2]} + kSamples - 1) % length[2]) + 1;
        product2 = d * output[2];
        for (uint32_t c = 0; c < 3; ++c) pos[c] = static_cast<uint32_t>((uint64_t{pos[c]} + kSamples) % length[c]);
        Async::Hold(footprint, count);
        if (Async::GetMode() == Async::Mode::kCheck) {
            job.rangeCount = 0;
            for (uint32_t c = 0; c < 3; ++c) {
                job.AddRange(inSpan[c], true);
                job.AddRange(line[c].span, true);
            }
            const uint32_t expect[] = {pos[0], pos[1], pos[2], b, d, written2, inc2, product2};
            job.SetExpect(expect, sizeof(expect) / sizeof(expect[0]));
        }
        Async::Push(job);
    } else {
        if (Async::On()) Async::InPlace(Async::Kind::kDelay);
        if (disjoint && CheckMode())
            CheckLoop(loop);
        else if (disjoint && VectorAudio())
            LoopVector(loop);
        else
            LoopScalar(loop);
        b = loop.b;
        d = loop.d;
        written2 = loop.written2;
        inc2 = loop.inc2;
        product2 = loop.product2;
        for (uint32_t c = 0; c < 3; ++c) pos[c] = loop.pos[c];
    }

    for (uint32_t c = 0; c < 3; ++c) MemoryInline::FlatWrite32(fx + kPos + 4 * c, pos[c]);
    StorePrologue(ctx, r1);

    ctx->gpr[0] = static_cast<uint32_t>(static_cast<int32_t>(product2) >> 7);
    ctx->gpr[3] = in[2] + kBufferBytes;
    ctx->gpr[5] = in[0] + kBufferBytes;
    ctx->gpr[6] = in[1] + kBufferBytes;
    ctx->gpr[7] = b;
    ctx->gpr[8] = d;
    ctx->gpr[9] = output[0];
    ctx->gpr[10] = inc2;
    ctx->gpr[11] = 0;
    ctx->gpr[12] = written2;
    uint32_t cr = ctx->cr;
    SetCRResident(cr, ctx->xer, 0, inc2, length[2]);  // the last cmplw of the third position
    ctx->cr = cr;
    // srawi's carry from the last output product.
    const bool carry = (product2 & 0x80000000u) != 0 && (product2 & 0x7Fu) != 0;
    ctx->xer = (ctx->xer & 0xDFFFFFFFu) | (static_cast<uint32_t>(carry) << 29);
    ctx->ctr = 0;
    return true;
}

}  // namespace Delay

// ======================================================================================================================
// Dispatch, statistics and the in-situ verifier
// ======================================================================================================================

struct Config {
    bool native = true;
    bool verify = false;
};

const Config& GetConfig() {
    static const Config config = [] {
        Config c;
        if (const char* v = std::getenv("BRAWL_AXFX_NATIVE")) c.native = std::strcmp(v, "0") != 0;
        if (const char* v = std::getenv("BRAWL_HLE_VERIFY")) c.verify = v[0] != '\0' && std::strcmp(v, "0") != 0;
        RT_LOGF(RT_TAG_AUDIO, "axfx: native AXFX reverb/delay %s%s\n", c.native ? "on" : "off (BRAWL_AXFX_NATIVE=0)",
                c.native && c.verify ? ", verifying every call against the translation (BRAWL_HLE_VERIFY)" : "");
        return c;
    }();
    return config;
}

// ---- AXFX_ASYNC: the worker, the pending ranges and Join (see the section above Reverb's samples loops) ----
namespace Async {

Mode GetMode() {
    static const Mode mode = [] {
        const Config& config = GetConfig();
        // Mod code may inspect or write effect buffers between the known join sites.
        // An EABI build alone does not establish that no such code can run.
        Mode m = Mode::kOff;
        const char* v = std::getenv("AXFX_ASYNC");
        if (v != nullptr && *v != '\0')
            m = std::strcmp(v, "0") == 0 ? Mode::kOff : (std::strcmp(v, "check") == 0 ? Mode::kCheck : Mode::kOn);
        const char* why = "";
        if (m != Mode::kOff && (!config.native || config.verify)) {
            m = Mode::kOff;
            why = " (the verifier and BRAWL_AXFX_NATIVE=0 run every call in place)";
        } else if (m != Mode::kOff && Reverb::CheckMode()) {
            m = Mode::kOff;
            why = " (AXFX_CHECK compares the loops in place)";
        }
        RT_LOGF(RT_TAG_AUDIO, "axfx: samples loops %s%s\n",
                m == Mode::kOn      ? "on a worker thread (AXFX_ASYNC)"
                : m == Mode::kCheck ? "run at their joins, written ranges poisoned until then (AXFX_ASYNC=check)"
                                    : "in place",
                why);
        return m;
    }();
    return mode;
}

constexpr uint32_t kMaxHeld = 32;  // pending jobs' footprints (a reverb holds 11 ranges at most, a delay 4)
constexpr uint8_t kPoison = 0xA5;   // AXFX_ASYNC=check: what a pending job's written ranges hold

// A spin-wait iteration's hint to the core.
inline void CpuRelax() {
#if defined(__x86_64__)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#endif
}

struct State {
    std::array<Job, kSlots> slots;
    // Game thread.
    uint64_t head = 0;    // jobs handed over
    uint64_t joined = 0;  // jobs joined (their results applied)
    Segment held[kMaxHeld];  // the pending jobs' footprints (Footprint), in hand-over order
    uint32_t heldCount = 0;
    bool started = false;
    bool inlineJobs = false;  // the worker failed to start: each job runs when handed over
    std::chrono::microseconds spin{30};
    uint64_t jobs[2] = {}, joins = 0, waited = 0, waitNs = 0, conflicts = 0, wakes = 0;
    uint64_t filled = 0, overwritten = 0, checkWriters = 0, checkInputs = 0, checkResults = 0, checkResultMismatches = 0;
    uint64_t inPlace[2] = {};  // calls that ran their samples loop in place (not eligible for a job)
    // Shared with the worker.
    alignas(64) std::atomic<uint64_t> published{0};  // written by the game thread
    alignas(64) std::atomic<uint64_t> done{0};       // written by the worker
    std::atomic<bool> parked{false};
    std::atomic<bool> notified{false};  // the game thread has woken the parked worker (it clears this once awake)
    alignas(64) std::mutex mutex;  // parking only
    std::condition_variable cv;
};
// Never destroyed: the detached worker waits on it until the process ends.
State& G() {
    static State* state = new State;
    return *state;
}

void LogAsync(const State& g) {
    RT_LOGF(RT_TAG_AUDIO,
            "axfx async: jobs reverb=%" PRIu64 " delay=%" PRIu64 ", joins=%" PRIu64 " (waited %" PRIu64 ", %.1f us in all)"
            ", conflicts=%" PRIu64 ", worker wakes=%" PRIu64 ", dead words filled=%" PRIu64 " overwritten=%" PRIu64 "\n",
            g.jobs[0], g.jobs[1], g.joins, g.waited, static_cast<double>(g.waitNs) / 1000.0, g.conflicts, g.wakes,
            g.filled, g.overwritten);
    RT_LOGF(RT_TAG_AUDIO, "axfx async: samples loops in place (not eligible): reverb=%" PRIu64 " delay=%" PRIu64 "\n",
            g.inPlace[0], g.inPlace[1]);
    if (GetMode() == Mode::kCheck)
        RT_LOGF(RT_TAG_AUDIO, "axfx async check: written ranges changed while pending=%" PRIu64
                ", read ranges changed while pending=%" PRIu64 ", calls' own results checked=%" PRIu64
                " mismatched=%" PRIu64 "\n", g.checkWriters, g.checkInputs, g.checkResults, g.checkResultMismatches);
}

// A job's samples loop and its stores: on the worker, or (AXFX_ASYNC=check, no worker) at the join.
void RunJob(Job& job) {
    static const int reverbSite = InsnCount::On() ? InsnCount::Site("axfx.reverb.job") : -1;
    static const int delaySite = InsnCount::On() ? InsnCount::Site("axfx.delay.job") : -1;
    MkwSetHostFpControl(job.fpControl);
    const uint64_t t = InsnCount::On() ? InsnCount::Read() : 0;
    if (job.kind == Kind::kReverb) {
        Reverb::Plan& p = job.plan;
        switch (job.samples) {
        case Samples::kVector: Reverb::ProcessFastV(p); break;
        case Samples::kFast: Reverb::ProcessFast(p); break;
        case Samples::kExact: Reverb::Process<false>(p); break;
        case Samples::kExactNi: Reverb::Process<true>(p); break;
        case Samples::kScalar: break;
        }
        for (uint32_t c = 0; c < 3; ++c) Store32(job.lpf + 4 * c, Narrow(p.lpf[c]));
        for (uint32_t c = 0; c < 3; ++c) job.fill[c] = Narrow(p.mix[c]);
        job.fill[3] = Narrow(p.f[9]);
        job.fill[4] = Narrow(p.f[8]);
        job.fill[5] = Narrow(p.f[7]);
    } else if (job.samples == Samples::kVector) {
        Delay::LoopVector(job.loop);
    } else {
        Delay::LoopScalar(job.loop);
    }
    if (InsnCount::On())
        InsnCount::Add(job.kind == Kind::kReverb ? reverbSite : delaySite, InsnCount::kNative, InsnCount::Since(t));
}

// Worker: until a job after `tail` is published. It spins a while first (an AX frame's effects come microseconds
// apart), then parks; the game thread wakes it only when it has parked.
void WaitForJob(State& g, uint64_t tail) {
    const auto until = std::chrono::steady_clock::now() + g.spin;
    for (uint32_t i = 1;; ++i) {
        if (g.published.load(std::memory_order_acquire) != tail) return;
        CpuRelax();
        if (i % 64 == 0 && std::chrono::steady_clock::now() >= until) break;
    }
    std::unique_lock<std::mutex> lock(g.mutex);
    // Cleared before parked is set: a hand-over that sees this park notifies (a stale flag from an earlier wake would
    // make it skip the notify and leave the worker parked with a job published).
    g.notified.store(false, std::memory_order_seq_cst);
    g.parked.store(true, std::memory_order_seq_cst);
    while (g.published.load(std::memory_order_seq_cst) == tail) g.cv.wait(lock);
    g.parked.store(false, std::memory_order_seq_cst);
}

void WorkerMain(State* g) {
    for (uint64_t tail = 0;; ++tail) {
        if (g->published.load(std::memory_order_acquire) == tail) WaitForJob(*g, tail);
        RunJob(g->slots[tail % kSlots]);
        g->done.store(tail + 1, std::memory_order_release);
    }
}

void Start(State& g) {
    g.started = true;
    if (GetMode() != Mode::kOn) return;  // check mode runs its jobs at the joins
    if (const char* v = std::getenv("AXFX_ASYNC_SPIN_US")) g.spin = std::chrono::microseconds(std::atoi(v));
    try {
        std::thread(WorkerMain, &g).detach();
        RT_LOGF(RT_TAG_AUDIO, "axfx: started the AXFX worker thread (spins %lld us before parking)\n",
                static_cast<long long>(g.spin.count()));
    } catch (const std::system_error&) {
        g.inlineJobs = true;
        RT_LOGF(RT_TAG_AUDIO, "axfx: failed to start the AXFX worker; samples loops in place\n");
    }
    // While the process exits, no job may still be writing guest memory. (Only waits: the game-side state stays the
    // game thread's.)
    std::atexit([] {
        State& s = G();
        const uint64_t target = s.published.load(std::memory_order_acquire);
        while (s.done.load(std::memory_order_acquire) < target) std::this_thread::yield();
    });
}

bool OverlapsHeld(const State& g, uint32_t start, uint32_t end) {
    for (uint32_t k = 0; k < g.heldCount; ++k)
        if (g.held[k].start < end && start < g.held[k].end) return true;
    return false;
}

// AXFX_ASYNC=check: at the hand-over, keep the job's ranges and poison the written ones; at the join, count changed
// ranges (a guest writer while pending), put the bytes back and run the job.
void Poison(Job& job) {
    size_t total = 0;
    for (uint32_t k = 0; k < job.rangeCount; ++k) total += job.ranges[k].bytes;
    job.saved.resize(total);
    size_t at = 0;
    for (uint32_t k = 0; k < job.rangeCount; ++k) {
        std::memcpy(job.saved.data() + at, job.ranges[k].host, job.ranges[k].bytes);
        at += job.ranges[k].bytes;
    }
    for (uint32_t k = 0; k < job.rangeCount; ++k)
        if (job.ranges[k].write) std::memset(job.ranges[k].host, kPoison, job.ranges[k].bytes);
}

void CheckRun(State& g, Job& job) {
    size_t at = 0;
    for (uint32_t k = 0; k < job.rangeCount; ++k) {
        const Range& r = job.ranges[k];
        uint32_t changed = r.bytes;
        for (uint32_t b = 0; b < r.bytes && changed == r.bytes; ++b)
            if (r.host[b] != (r.write ? kPoison : job.saved[at + b])) changed = b;
        if (changed != r.bytes) {
            const uint64_t n = r.write ? ++g.checkWriters : ++g.checkInputs;
            if (n <= 8)
                RT_LOGF(RT_TAG_AUDIO, "axfx async check: a pending %s job's %s range 0x%08X (+%u) changed at 0x%08X\n",
                        job.kind == Kind::kReverb ? "reverb" : "delay", r.write ? "written" : "read", r.guest, r.bytes,
                        r.guest + changed);
        }
        at += r.bytes;
    }
    at = 0;
    for (uint32_t k = 0; k < job.rangeCount; ++k) {
        std::memcpy(job.ranges[k].host, job.saved.data() + at, job.ranges[k].bytes);
        at += job.ranges[k].bytes;
    }
    RunJob(job);
    // The call's own positions and returned values against the loop's, in the order the call recorded them.
    uint32_t actual[kMaxExpect];
    uint32_t m = 0;
    if (job.kind == Kind::kReverb) {
        const Reverb::Plan& p = job.plan;
        for (uint32_t k = 0; k < 3; ++k) actual[m++] = p.earlyPos[k];
        actual[m++] = p.preDelayPos;
        for (uint32_t k = 0; k < 3; ++k) actual[m++] = p.combPos[k];
        for (uint32_t k = 0; k < 2; ++k) actual[m++] = p.apPos[k];
        for (uint32_t c = 0; c < 3; ++c) actual[m++] = p.lastPos[c];
        actual[m++] = p.lastApInc;
        actual[m++] = p.lastConvert;
        for (const uint32_t r : {0u, 1u, 3u, 4u, 5u, 6u}) {
            uint64_t u;
            std::memcpy(&u, &p.f[r], sizeof(u));
            actual[m++] = static_cast<uint32_t>(u >> 32);
            actual[m++] = static_cast<uint32_t>(u);
        }
    } else {
        const Delay::Loop& L = job.loop;
        for (uint32_t c = 0; c < 3; ++c) actual[m++] = L.pos[c];
        for (const uint32_t v : {L.b, L.d, L.written2, L.inc2, L.product2}) actual[m++] = v;
    }
    ++g.checkResults;
    for (uint32_t k = 0; k < job.expectCount; ++k) {
        if (k >= m || actual[k] != job.expect[k]) {
            if (++g.checkResultMismatches <= 8)
                RT_LOGF(RT_TAG_AUDIO, "axfx async check: a %s call left value %u = 0x%08X, its loop 0x%08X\n",
                        job.kind == Kind::kReverb ? "reverb" : "delay", k, job.expect[k], k < m ? actual[k] : 0u);
            break;
        }
    }
    if (job.expectCount != m && ++g.checkResultMismatches <= 8)
        RT_LOGF(RT_TAG_AUDIO, "axfx async check: %u values left, %u from the loop\n", job.expectCount, m);
}

// What the call left for Join: the reverb's dead words, over any sentinel still in place.
void Apply(State& g, const Job& job) {
    if (job.kind != Kind::kReverb) return;
    for (uint32_t k = 0; k < kFillWords; ++k) {
        uint8_t* word = job.frame + kFillOffsets[k];
        if (Load32(word) == kSentinel + k) {
            Store32(word, job.fill[k]);
            ++g.filled;
        } else {
            ++g.overwritten;
        }
    }
}

Job& Acquire() {
    State& g = G();
    if (!g.started) Start(g);
    if (g.head - g.joined >= kSlots) Join();
    return g.slots[g.head % kSlots];
}

void Hold(const Segment* segments, uint32_t count) {
    State& g = G();
    if (g.heldCount + count > kMaxHeld) Join();
    for (uint32_t k = 0; k < count; ++k) g.held[g.heldCount++] = segments[k];
}

void Push(Job& job) {
    State& g = G();
    ++g.jobs[job.kind == Kind::kReverb ? 0 : 1];
    const uint64_t n = g.jobs[0] + g.jobs[1];
    if (GetMode() == Mode::kCheck) {
        Poison(job);
        g.published.store(++g.head, std::memory_order_relaxed);  // done advances at the join
    } else if (g.inlineJobs) {
        RunJob(job);
        g.published.store(++g.head, std::memory_order_relaxed);
        g.done.store(g.head, std::memory_order_release);
    } else {
        // Wake a parked worker once: until it is awake, a woken worker still sees every job handed over meanwhile.
        g.published.store(++g.head, std::memory_order_seq_cst);
        if (g.parked.load(std::memory_order_seq_cst) && !g.notified.exchange(true, std::memory_order_seq_cst)) {
            std::lock_guard<std::mutex> lock(g.mutex);
            g.cv.notify_one();
            ++g.wakes;
        }
    }
    if (n == 1 || n % 16384 == 0) LogAsync(g);
}

void InPlace(Kind kind) { ++G().inPlace[kind == Kind::kReverb ? 0 : 1]; }

bool Pending() {
    const State& g = G();
    return g.joined != g.head;
}

bool Overlaps(uint32_t start, uint32_t bytes) {
    State& g = G();
    const uint32_t canonical = GuestMemoryAlias::Offset(start);
    if (!OverlapsHeld(g, canonical, canonical + bytes)) return false;
    ++g.conflicts;
    return true;
}

bool Conflicts(const Segment* segments, uint32_t count) {
    State& g = G();
    for (uint32_t k = 0; k < count; ++k) {
        if (OverlapsHeld(g, segments[k].start, segments[k].end)) {
            ++g.conflicts;
            return true;
        }
    }
    return false;
}

void Join() {
    State& g = G();
    if (g.joined == g.head) return;
    static const int site = InsnCount::On() ? InsnCount::Site("axfx.join") : -1;
    const uint64_t t = InsnCount::On() ? InsnCount::Read() : 0;
    ++g.joins;
    if (GetMode() == Mode::kCheck) {
        for (uint64_t seq = g.joined; seq < g.head; ++seq) CheckRun(g, g.slots[seq % kSlots]);
        g.done.store(g.head, std::memory_order_release);
    } else if (g.done.load(std::memory_order_acquire) < g.head) {
        // Rare: the jobs have the rest of the AX frame (3 ms) before the next AI block's join.
        ++g.waited;
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t i = 1; g.done.load(std::memory_order_acquire) < g.head; ++i) {
            CpuRelax();
            if (i % 4096 == 0) std::this_thread::yield();
        }
        g.waitNs += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    }
    for (uint64_t seq = g.joined; seq < g.head; ++seq) Apply(g, g.slots[seq % kSlots]);
    g.joined = g.head;
    g.heldCount = 0;
    if (InsnCount::On()) InsnCount::Add(site, InsnCount::kNative, InsnCount::Since(t));
}

}  // namespace Async

struct Stats {
    const char* name;
    const std::atomic<uint64_t>* bypassed;  // native runs that took the skip path (the rest processed 96 samples)
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> natives{0};       // ran natively
    std::atomic<uint64_t> fallbacks{0};     // preconditions failed: ran the translated original
    std::atomic<uint64_t> compared{0};      // verifier: calls compared
    std::atomic<uint64_t> mismatches{0};    // verifier: calls whose native result differed
    std::atomic<uint64_t> unverifiable{0};  // verifier: write ranges not resolvable, ran natively unverified
    int site = -1;                          // NAT_INSN's site (insn_count.h)
};

Stats g_reverbStats{"AXFXReverbHiExpCallback", &Reverb::g_bypassed};
Stats g_delayStats{"AXFXDelayCallback", &Delay::g_bypassed};

// NAT_INSN=1 (insn_count.h): sites "axfx.reverb" and "axfx.delay". BRAWL_HLE_VERIFY counts the translated original
// and the native of every compared call (the reverb's exact loop: verify implies AXFX_EXACT); the native mode counts
// the native column only (the default fast reverb's cost per call).
const bool g_sitesRegistered = [] {
    if (!InsnCount::On()) return false;
    g_reverbStats.site = InsnCount::Site("axfx.reverb");
    g_delayStats.site = InsnCount::Site("axfx.delay");
    return true;
}();

void LogStats(const Stats& stats, bool verify) {
    const uint64_t natives = stats.natives.load(), bypassed = stats.bypassed->load();
    if (verify) {
        RT_LOGF(RT_TAG_AUDIO, "axfx verify %s: calls=%" PRIu64 " compared=%" PRIu64 " mismatches=%" PRIu64
                " native=%" PRIu64 " (processed=%" PRIu64 " skipped=%" PRIu64 ") fallbacks=%" PRIu64
                " unverifiable=%" PRIu64 "\n", stats.name, stats.calls.load(), stats.compared.load(),
                stats.mismatches.load(), natives, natives - bypassed, bypassed, stats.fallbacks.load(),
                stats.unverifiable.load());
    } else {
        RT_LOGF(RT_TAG_AUDIO, "axfx %s: calls=%" PRIu64 " native=%" PRIu64 " (processed=%" PRIu64 " skipped=%" PRIu64
                ") fallbacks=%" PRIu64 "\n", stats.name, stats.calls.load(), natives, natives - bypassed, bypassed,
                stats.fallbacks.load());
    }
}

inline void CountCall(Stats& stats, bool verify) {
    const uint64_t n = stats.calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == 1 || n == 100 || n % (verify ? 1000 : 5000) == 0) LogStats(stats, verify);
}

// One guest range the function may write, with its bytes and (lean memory) its dirty marks at each stage.
struct VerifyRange {
    uint32_t start = 0;
    uint32_t bytes = 0;
    uint8_t* host = nullptr;
    uint8_t* marks = nullptr;  // lean dirty bytes of the blocks [first, last + 1] the range's stores can mark
    uint32_t markCount = 0;
    std::vector<uint8_t> before, original, before_marks, original_marks;
};

struct VerifyScratch {
    std::vector<VerifyRange> ranges;
    uint32_t count = 0;
    VerifyRange& Next() {
        if (count == ranges.size()) ranges.emplace_back();
        return ranges[count++];
    }
};

bool AddRange(VerifyScratch& scratch, uint32_t start, uint64_t bytes) {
    if (bytes == 0) return true;
    Span span;
    if (!Resolve(span, start, bytes, false)) return false;
    VerifyRange& r = scratch.Next();
    r.start = start;
    r.bytes = span.bytes;
    r.host = span.host;
#if MKW_LEAN_MEMORY
    const uint32_t canonical = GuestMemoryAlias::Offset(start);
    const uint32_t first = canonical >> 6;
    const uint32_t last = ((canonical + span.bytes - 1) >> 6) + 1;
    r.marks = MemoryInline::g_leanBase + MemoryInline::kLeanDirtyOffset + first;
    r.markCount = last - first + 1;
#else
    r.marks = nullptr;
    r.markCount = 0;
#endif
    return true;
}

// Everything a ring's walk can write: the whole line, and the start position if it lies past the line.
bool AddRingRange(VerifyScratch& scratch, uint32_t line, uint32_t pos, uint32_t length) {
    if (length != 0 && !AddRange(scratch, line, uint64_t{length} * 4)) return false;
    return pos < length || AddRange(scratch, line + (pos << 2), 4);
}

// The ranges each original can write, derived from the guest state independently of the native's own plan.
bool ReverbWriteRanges(const CpuContext* ctx, VerifyScratch& scratch) {
    using namespace Reverb;
    const uint32_t r1 = ctx->gpr[1], fx = ctx->gpr[4];
    if (!AddRange(scratch, r1 - kFrameBytes, kFrameSpan) || !AddRange(scratch, fx, kEffectBytes)) return false;
    const auto read = [](uint32_t address) { return MemoryInline::FlatRead32(address); };
    for (uint32_t c = 0; c < 3; ++c) {
        if (!AddRange(scratch, read(ctx->gpr[3] + 4 * c), kBufferBytes)) return false;
        if (const uint32_t busOut = read(fx + kBusOut); busOut != 0 &&
            !AddRange(scratch, read(busOut + 4 * c), kBufferBytes))
            return false;
        if (const uint32_t length = read(fx + kPreDelayLength); length != 0 &&
            !AddRingRange(scratch, read(fx + kPreDelayLine + 4 * c), read(fx + kPreDelayPos), length))
            return false;
        for (uint32_t k = 0; k < 3; ++k)
            if (!AddRingRange(scratch, read(fx + kCombLine + 12 * c + 4 * k), read(fx + kCombPos + 4 * k),
                              read(fx + kCombLength + 4 * k)))
                return false;
        for (uint32_t k = 0; k < 2; ++k)
            if (!AddRingRange(scratch, read(fx + kAllpassLine + 8 * c + 4 * k), read(fx + kAllpassPos + 4 * k),
                              read(fx + kAllpassLength + 4 * k)))
                return false;
        if (!AddRingRange(scratch, read(fx + kLastAllpassLine + 4 * c), read(fx + kLastAllpassPos + 4 * c),
                          read(fx + kLastAllpassLength + 4 * c)))
            return false;
    }
    return true;
}

bool DelayWriteRanges(const CpuContext* ctx, VerifyScratch& scratch) {
    using namespace Delay;
    const uint32_t r1 = ctx->gpr[1], fx = ctx->gpr[4];
    if (!AddRange(scratch, r1 - kFrameBytes, kFrameBytes) || !AddRange(scratch, fx, kEffectBytes)) return false;
    const auto read = [](uint32_t address) { return MemoryInline::FlatRead32(address); };
    for (uint32_t c = 0; c < 3; ++c) {
        if (!AddRange(scratch, read(ctx->gpr[3] + 4 * c), kBufferBytes)) return false;
        if (!AddRingRange(scratch, read(fx + kLine + 4 * c), read(fx + kPos + 4 * c), read(fx + kLength + 4 * c)))
            return false;
    }
    return true;
}

void LogContextMismatch(const char* name, const CpuContext& want, const CpuContext& got) {
    for (uint32_t r = 0; r < 32; ++r)
        if (want.gpr[r] != got.gpr[r])
            RT_LOGF(RT_TAG_AUDIO, "axfx MISMATCH %s: r%u original %08X native %08X\n", name, r, want.gpr[r], got.gpr[r]);
    for (uint32_t r = 0; r < 32; ++r)
        if (want.fpr[r].raw != got.fpr[r].raw)
            RT_LOGF(RT_TAG_AUDIO, "axfx MISMATCH %s: f%u original %016" PRIX64 " native %016" PRIX64 "\n", name, r,
                    want.fpr[r].raw, got.fpr[r].raw);
    const struct { const char* field; uint32_t a, b; } spr[] = {
        {"cr", want.cr, got.cr}, {"lr", want.lr, got.lr}, {"ctr", want.ctr, got.ctr}, {"xer", want.xer, got.xer},
        {"fpscr", want.fpscr, got.fpscr}, {"pc", want.pc, got.pc}, {"msr", want.msr, got.msr}};
    for (const auto& s : spr)
        if (s.a != s.b) RT_LOGF(RT_TAG_AUDIO, "axfx MISMATCH %s: %s original %08X native %08X\n", name, s.field, s.a, s.b);
}

// The registers a caller can read after the call (mtx.cpp SameRegisters' rule). In P+ the whole context. The vanilla
// translation (MKW_EABI_REGS, detected by an EABI entry's presence) writes back only what the original's callers
// read, where the natives leave P+'s whole register file: there r1, r2, r13-r31, f14-f31, LR, CR fields 2-4, the GQRs
// and the special registers past them always, and any other register where the original changed it.
bool SameRegisters(const CpuContext& in, const CpuContext& original, const CpuContext& native) {
    if (func_801D7AE4_eabi == nullptr) return std::memcmp(&native, &original, sizeof(CpuContext)) == 0;
    for (uint32_t r = 0; r < 32; ++r) {
        const bool kept = r == 1 || r == 2 || r >= 13;
        if (native.gpr[r] != original.gpr[r] && (kept || original.gpr[r] != in.gpr[r])) return false;
    }
    for (uint32_t f = 0; f < 32; ++f)
        if (native.fpr[f].raw != original.fpr[f].raw && (f >= 14 || original.fpr[f].raw != in.fpr[f].raw)) return false;
    if (native.lr != original.lr) return false;
    if (((native.cr ^ original.cr) & (0x00FFF000u | (original.cr ^ in.cr))) != 0) return false;
    if (native.ctr != original.ctr && original.ctr != in.ctr) return false;
    if (native.xer != original.xer && original.xer != in.xer) return false;
    if (native.fpscr != original.fpscr && original.fpscr != in.fpscr) return false;
    const size_t from = offsetof(CpuContext, gqr);
    return std::memcmp(reinterpret_cast<const uint8_t*>(&native) + from,
                       reinterpret_cast<const uint8_t*>(&original) + from, sizeof(CpuContext) - from) == 0;
}

// The first block of a range's marks (index into r.marks) whose range bytes the native changed but did not mark, or
// UINT32_MAX. r.marks holds the native's marks, measured from zero; r.host its bytes; r.before the bytes before.
uint32_t FirstUncoveredBlock(const VerifyRange& r) {
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS  // (without dirty marks neither side marks: nothing to cover)
    const uint32_t c = GuestMemoryAlias::Offset(r.start);
    const uint32_t first = c >> 6;
    for (uint32_t b = 0; b < r.markCount; ++b) {
        if (r.marks[b] != 0) continue;
        const uint32_t lo = std::max((first + b) << 6, c) - c;
        const uint32_t hi = std::min((first + b + 1) << 6, c + r.bytes) - c;
        for (uint32_t k = lo; k < hi; ++k)
            if (r.host[k] != r.before[k]) return b;
    }
#else
    (void)r;
#endif
    return UINT32_MAX;
}

// Run the translated original and the native from the same state, compare, keep the original's results.
void Verify(CpuContext* ctx, Stats& stats, void (*original)(CpuContext*), bool (*native)(CpuContext*),
            bool (*writeRanges)(const CpuContext*, VerifyScratch&)) {
    static thread_local VerifyScratch scratch;
    scratch.count = 0;
    if (!writeRanges(ctx, scratch)) {
        stats.unverifiable.fetch_add(1, std::memory_order_relaxed);
        if (native(ctx)) {
            stats.natives.fetch_add(1, std::memory_order_relaxed);
        } else {
            stats.fallbacks.fetch_add(1, std::memory_order_relaxed);
            original(ctx);
        }
        return;
    }
    const CpuContext before = *ctx;
    // Capture every range before clearing any marks: ranges may share bytes and mark blocks.
    for (uint32_t k = 0; k < scratch.count; ++k) {
        VerifyRange& r = scratch.ranges[k];
        r.before.assign(r.host, r.host + r.bytes);
        r.before_marks.assign(r.marks, r.marks + r.markCount);
    }
    for (uint32_t k = 0; k < scratch.count; ++k) {
        VerifyRange& r = scratch.ranges[k];
        if (r.markCount) std::memset(r.marks, 0, r.markCount);
    }
    // The original. Its marks are measured from zero (and OR-ed back with the earlier ones at the end).
    const uint64_t originalStart = InsnCount::On() ? InsnCount::Read() : 0;
    original(ctx);
    const uint64_t originalInsns = InsnCount::On() ? InsnCount::Since(originalStart) : 0;
    const CpuContext afterOriginal = *ctx;
    for (uint32_t k = 0; k < scratch.count; ++k) {
        VerifyRange& r = scratch.ranges[k];
        r.original.assign(r.host, r.host + r.bytes);
        r.original_marks.assign(r.marks, r.marks + r.markCount);
    }
    // Back to the start; then the native.
    for (uint32_t k = 0; k < scratch.count; ++k) {
        VerifyRange& r = scratch.ranges[k];
        std::memcpy(r.host, r.before.data(), r.bytes);
        if (r.markCount) std::memset(r.marks, 0, r.markCount);
    }
    *ctx = before;
    const uint64_t nativeStart = InsnCount::On() ? InsnCount::Read() : 0;
    const bool ranNatively = native(ctx);
    if (ranNatively) {
        stats.natives.fetch_add(1, std::memory_order_relaxed);
    } else {
        stats.fallbacks.fetch_add(1, std::memory_order_relaxed);
        original(ctx);
    }
    if (InsnCount::On()) {
        const uint64_t nativeInsns = InsnCount::Since(nativeStart);
        InsnCount::Add(stats.site, InsnCount::kOriginal, originalInsns);
        InsnCount::Add(stats.site, InsnCount::kNative, nativeInsns);
    }
    // Compare.
    bool mismatch = !SameRegisters(before, afterOriginal, *ctx);
    const uint64_t priorMismatches = stats.mismatches.load(std::memory_order_relaxed);
    const bool detail = priorMismatches < 8;
    if (mismatch && detail) LogContextMismatch(stats.name, afterOriginal, *ctx);
    for (uint32_t k = 0; k < scratch.count; ++k) {
        const VerifyRange& r = scratch.ranges[k];
        if (std::memcmp(r.host, r.original.data(), r.bytes) != 0) {
            mismatch = true;
            if (detail) {
                uint32_t differing = 0, first = UINT32_MAX;
                for (uint32_t b = 0; b < r.bytes; ++b)
                    if (r.host[b] != r.original[b]) {
                        ++differing;
                        if (first == UINT32_MAX) first = b;
                    }
                const uint32_t word = first & ~3u;
                RT_LOGF(RT_TAG_AUDIO, "axfx MISMATCH %s: range %08X+%u: %u byte(s) differ, first at %08X: original "
                        "%08X native %08X\n", stats.name, r.start, r.bytes, differing, r.start + word,
                        word + 4 <= r.bytes ? Load32(r.original.data() + word) : 0u,
                        word + 4 <= r.bytes ? Load32(r.host + word) : 0u);
            }
        }
        // Marks: the native's must cover every block of the range whose bytes changed. Exact equality is too strict
        // with MKW_LEAN_RANGES: a range-lowered original marks a segment's span at once (memory_access.h
        // LeanMarkSpan), so either side can mark blocks the other does not; rollback needs only the written ones.
        if (const uint32_t block = FirstUncoveredBlock(r); block != UINT32_MAX) {
            mismatch = true;
            if (detail)
                RT_LOGF(RT_TAG_AUDIO, "axfx MISMATCH %s: dirty mark of range %08X+%u, block %u: bytes changed, native "
                        "mark 0 (original %u)\n", stats.name, r.start, r.bytes, block,
                        static_cast<unsigned>(r.original_marks[block]));
        }
    }
    stats.compared.fetch_add(1, std::memory_order_relaxed);
    if (mismatch) {
        const uint64_t n = stats.mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
        if (detail || n % 1000 == 0)
            RT_LOGF(RT_TAG_AUDIO, "axfx MISMATCH %s: call %" PRIu64 " (%s), %" PRIu64 " mismatching call(s) so far\n",
                    stats.name, stats.calls.load(), ranNatively ? "native" : "fallback", n);
    }
    // Keep the original's results: its bytes, its registers, and its marks on top of the earlier ones.
    for (uint32_t k = 0; k < scratch.count; ++k) {
        VerifyRange& r = scratch.ranges[k];
        std::memcpy(r.host, r.original.data(), r.bytes);
        for (uint32_t b = 0; b < r.markCount; ++b) r.marks[b] = r.before_marks[b] | r.original_marks[b];
    }
    *ctx = afterOriginal;
}

template <uint32_t Address>
void Dispatch(CpuContext* ctx, Stats& stats, void (*wrapper)(CpuContext*), void (*original)(CpuContext*),
              bool (*native)(CpuContext*), bool (*writeRanges)(const CpuContext*, VerifyScratch&)) {
    // A mod that patched this function's guest code gets it run in the interpreter, as the translated guard does.
    // Translated or interpreted guest code may touch anything a pending job does: join first (also below).
    static const int64_t guard = GuardIndexOf(Address);
    if (guard < 0) [[unlikely]] {
        Async::Join();
        original(ctx);
        return;
    }
    if (guard >= 0 && g_mkwGuestCodePatched[guard]) [[unlikely]] {
        Async::Join();
        wrapper(ctx);
        return;
    }
    const Config& config = GetConfig();
    if (!config.native) {
        original(ctx);
        return;
    }
    CountCall(stats, config.verify);
    if (config.verify) {
        Verify(ctx, stats, original, native, writeRanges);
        return;
    }
    const uint64_t nativeStart = InsnCount::On() ? InsnCount::Read() : 0;
    const bool ranNatively = native(ctx);
    if (InsnCount::On()) InsnCount::Add(stats.site, InsnCount::kNative, InsnCount::Since(nativeStart));
    if (ranNatively) [[likely]] {
        stats.natives.fetch_add(1, std::memory_order_relaxed);
    } else {
        if (stats.fallbacks.fetch_add(1, std::memory_order_relaxed) == 0)
            RT_LOGF(RT_TAG_AUDIO, "axfx %s: first call outside the native's proven case, ran the translation\n",
                    stats.name);
        Async::Join();
        original(ctx);
    }
}

}  // namespace

void AXFXReverbHiExpCallback_Native(CpuContext* ctx) {
    Dispatch<Reverb::kAddress>(ctx, g_reverbStats, func_80202A58, func_80202A58_body, Reverb::Run, ReverbWriteRanges);
}

void AXFXDelayCallback_Native(CpuContext* ctx) {
    Dispatch<Delay::kAddress>(ctx, g_delayStats, func_80203A10, func_80203A10_body, Delay::Run, DelayWriteRanges);
}

REGISTER_NATIVE_FUNCTION_AS(0x80202A58, AXFXReverbHiExpCallback_Native, "AXFXReverbHiExpCallback_Native");
REGISTER_NATIVE_FUNCTION_AS(0x80203A10, AXFXDelayCallback_Native, "AXFXDelayCallback_Native");

namespace AxfxAsync {
void Join() { Async::Join(); }
}  // namespace AxfxAsync

namespace {
// An effect function is about to run: no job may be pending (AXFX_ASYNC), and every reverb proof cached so far is for
// a configuration it may change (AXFX_PROOF).
void EffectsChanged() {
    Async::Join();
    ++Reverb::g_configGeneration;
}
}  // namespace

// ---- AXFX_ASYNC's joins in the effects' owners ----
// Every function outside the callbacks that reaches an effect's struct, its lines or the aux buffers (the inventory in
// the AXFX_ASYNC section): each joins, then runs its translation. MKW_NATIVE_BODY: the translator's wrapper keeps the
// guest code guard (a mod's patch still runs) and tail-calls these; func_X_body is the translation.
extern "C" void func_80202620_body(CpuContext* ctx);
extern "C" void func_80202684_body(CpuContext* ctx);
extern "C" void func_802026A8_body(CpuContext* ctx);
extern "C" void func_80202798_body(CpuContext* ctx);
extern "C" void func_80202934_body(CpuContext* ctx);
extern "C" void func_80202A04_body(CpuContext* ctx);
extern "C" void func_8020365C_body(CpuContext* ctx);
extern "C" void func_8020385C_body(CpuContext* ctx);
extern "C" void func_80203980_body(CpuContext* ctx);
extern "C" void func_801FFE28_body(CpuContext* ctx);
extern "C" void func_801FFE8C_body(CpuContext* ctx);
extern "C" void func_801FFEF0_body(CpuContext* ctx);
extern "C" void func_801FF764_body(CpuContext* ctx);

MKW_NATIVE_BODY(0x80202620u, AxfxJoin_AXFXReverbHiInit);
MKW_NATIVE_BODY(0x80202684u, AxfxJoin_AXFXReverbHiShutdown);
MKW_NATIVE_BODY(0x802026A8u, AxfxJoin_AXFXReverbHiSettings);
MKW_NATIVE_BODY(0x80202798u, AxfxJoin_AXFXReverbHiExpInit);
MKW_NATIVE_BODY(0x80202934u, AxfxJoin_AXFXReverbHiExpSettings);  // fn_80202934 (called by AXFXReverbHiSettings)
MKW_NATIVE_BODY(0x80202A04u, AxfxJoin_AXFXReverbHiExpShutdown);
MKW_NATIVE_BODY(0x8020365Cu, AxfxJoin_AXFXDelayInit);
MKW_NATIVE_BODY(0x8020385Cu, AxfxJoin_AXFXDelaySettings);
MKW_NATIVE_BODY(0x80203980u, AxfxJoin_AXFXDelayShutdown);
MKW_NATIVE_BODY(0x801FFE28u, AxfxJoin_AXRegisterAuxACallback);
MKW_NATIVE_BODY(0x801FFE8Cu, AxfxJoin_AXRegisterAuxBCallback);
MKW_NATIVE_BODY(0x801FFEF0u, AxfxJoin_AXRegisterAuxCCallback);
MKW_NATIVE_BODY(0x801FF764u, AxfxJoin_AXAuxInit);

void AxfxJoin_AXFXReverbHiInit(CpuContext* ctx) { EffectsChanged(); func_80202620_body(ctx); }
void AxfxJoin_AXFXReverbHiShutdown(CpuContext* ctx) { EffectsChanged(); func_80202684_body(ctx); }
void AxfxJoin_AXFXReverbHiSettings(CpuContext* ctx) { EffectsChanged(); func_802026A8_body(ctx); }
void AxfxJoin_AXFXReverbHiExpInit(CpuContext* ctx) { EffectsChanged(); func_80202798_body(ctx); }
void AxfxJoin_AXFXReverbHiExpSettings(CpuContext* ctx) { EffectsChanged(); func_80202934_body(ctx); }
void AxfxJoin_AXFXReverbHiExpShutdown(CpuContext* ctx) { EffectsChanged(); func_80202A04_body(ctx); }
void AxfxJoin_AXFXDelayInit(CpuContext* ctx) { EffectsChanged(); func_8020365C_body(ctx); }
void AxfxJoin_AXFXDelaySettings(CpuContext* ctx) { EffectsChanged(); func_8020385C_body(ctx); }
void AxfxJoin_AXFXDelayShutdown(CpuContext* ctx) { EffectsChanged(); func_80203980_body(ctx); }
void AxfxJoin_AXRegisterAuxACallback(CpuContext* ctx) { EffectsChanged(); func_801FFE28_body(ctx); }
void AxfxJoin_AXRegisterAuxBCallback(CpuContext* ctx) { EffectsChanged(); func_801FFE8C_body(ctx); }
void AxfxJoin_AXRegisterAuxCCallback(CpuContext* ctx) { EffectsChanged(); func_801FFEF0_body(ctx); }
void AxfxJoin_AXAuxInit(CpuContext* ctx) { EffectsChanged(); func_801FF764_body(ctx); }
