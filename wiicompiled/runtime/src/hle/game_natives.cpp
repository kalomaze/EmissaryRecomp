// Native bodies for unnamed Brawl game code (no decomp names or splits for these; identified from the PPC).
//
// Havok (hk*) animation runs inside Brawl's per-frame fighter update: fn_800969E0 builds an hkaPose from the skeleton
// (hkArray<hkQsTransform> local/model poses with the 0x80000000 DONT_DEALLOCATE flag, hkMemory class 0x15) and
// fn_800977F0 copies each mapped g3d world matrix (MTX34) into the pose's model space, decomposing it into an
// hkQsTransform (translation, rotation quaternion, scale) with Havok's matrix decomposition. That decomposition was the
// hottest unnamed subtree of vanilla Training (IBS ops: fn_802821DC 1.66%, fn_80282CD4 0.77%, plus callees).
//
// QsFromMatrix (0x80284F2C): hkQsTransform from an hkMatrix4 (r3 = out, r4 = matrix): hkMatrixDecomposition::
// decomposeMatrix (0x802821DC) into a 0xD0-byte Decomposition in its own frame, then out = its translation, rotation
// and scale; r3 = (hasSkew ? 0 : 1) << 24 (hkBool success). decomposeMatrix (Havok's names for the callees):
//   0x80287274 hkTransform::set4x4 (basis columns with w = 0, translation with w = 1)
//   polar decomposition, at most 30 times: Q' = 0.5 * (Q + Q^-T): 0x80283524 hkMatrix3::invert(eps 2^-23; a
//     transpose of the adjugate scaled by 1/det, failing when |det| <= eps^3), 0x802829C4 transpose, 0x80283910 add,
//     0x80282F50 scalar multiply; until 0x80283280 hkMatrix3::isApproximatelyEqual(Q', Q, eps) (rows x, y, z)
//   the rotation's columns normalized (frsqrte + one Newton step), negated column 0 when det < 0 (m_flips),
//   0x8028590C/0x802856E4 hkQuaternion::set(hkRotation) (Shepperd), normalized; 0x80285A6C hkRotation::set(the
//   conjugate); 0x80282CD4 hkMatrix3::setMul twice: scaleAndSkew = R^-1 * basis, skew = diag(1/scale) * scaleAndSkew;
//   hasScale (any |scale - 1| > 0.001), hasSkew (!isApproximatelyEqual(skew, identity, 0.001)).
//
// Exactness. The native computes everything first and writes guest memory last, as the final image of the original's
// stores: the Decomposition, the output, every frame's saves and temporaries (dead stack included). The polar loop's
// and the later callees' frames all lie in [s1 - 0x120, s1) (s1 = decomposeMatrix's r1), which the second setMul
// (0xE0 bytes of saved-FPR zeros, its 12 temporaries and back chain) and the final isApproximatelyEqual overwrite: the
// final image does not depend on the iteration count. The vanilla (MKW_EABI_REGS) translation reaches invert,
// setMul and hkRotation::set through their EABI entries, which save the non-volatile FPRs they proved dead as 0; the
// native writes those zeros. QsFromMatrix's own r31 save is 0 too, as its callers' EABI entry wrote it before it
// became a native (the verify reference runs the context body and has that slot zeroed, as g3d_calc.cpp's verify does).
//
// Floating point: host single-precision operations. With every operand a single (all inputs, constants and results
// are), the translated code's helpers (isa/ppc_isa_float.h) compute: for fmuls/fadds/fsubs/fdivs the double result
// rounded once to single (double rounding is innocuous for +, -, *, / at 53 >= 2 * 24 + 2 bits); for fmadds/fmsubs/
// fnmsubs the exactly rounded single of a*c +- b (their double FMA plus the tie correction); then FPSCR[NI]'s flush of
// a pre-round magnitude below 2^-126 and Gekko's NaN rules. The host's FMA and ops give the same bits whenever no NaN,
// infinity or tiny result occurs. The native runs only with FPSCR[NI] on (Brawl's OSInit sets it; the host then
// flushes and flags every tiny result) and round-to-nearest. It clears the host's FP exception flags first and runs
// the original instead when any of invalid, divide-by-zero, overflow, underflow or denormal is set at the end, and
// when an input or constant is not a normal single or zero. One gap remains on x86: it detects tininess after
// rounding, so a product or FMA whose exact value lies just below 2^-126 and rounds up to it is neither flushed nor
// flagged there (NI flushes it); such a result is exactly +-2^-126, which the native tracks on every product and FMA
// (AArch64 flushes on the pre-round value, as NI does, and flags it). Sums and quotients of singles cannot land there.
// frsqrte uses the translated code's estimate helper.
//
// The other natives below (each with its own comment block) follow the same scheme: Havok's hkQsTransform::setMul,
// setInverse, setMulInverseMul and hkQsTransform -> hkMatrix4, hkaPose::syncModelSpace, Brawl mt's MTX34 scale, and
// MSL/fdlibm's __ieee754_sqrt (double precision: host doubles are the helpers' operations exactly). __kernel_tan
// (0x803FFED4) had a native too; NAT_INSN counted it at 262 instructions a call against the translation's 208 (its
// double operations already translate to host operations), so it is gone. A replay native of fn_800977F0 (the glue
// that fills the pose; cheaper bone scans) verified exact but counted 138,783 instructions a call against the
// original's 138,203 with every callee original: not kept.
//
// GAME_NATIVES: 1 (default) natives; 0 the translated originals behind the same wrappers (A/B on one binary); 2 verify:
// every call runs the original and then the native from the same state, compares the written guest bytes (the stack
// window and the output), the rollback dirty marks (native marks must be a subset of the original's and cover every
// changed block: the original's segment ranges over-mark; the frequent natives check their ranges' marks, and all of
// MEM1/MEM2 every 64th call) and the whole register file, logs differences, and keeps the original's results.
// GAME_NATIVES_OFF=name,...: those natives run their originals; GAME_NATIVES_ON=name,...: enables a default-off one
// (kFnDefaultOff). Names: qsfrommatrix qssetmul qstomatrix syncmodel sqrt qsinverse qsmulinvmul mtscale gftask.
// gftask (gfTask::process, a dispatcher) has its own verify and fast native path; see its block.
// NAT_INSN=1 (insn_count.h) counts instructions per call at sites game.<name>: verify, each compared call's original
// and native from the same state; native mode, each call (a declined one with its original).
// The vanilla translation only: a translation without EABI entries (P+) runs the originals.

#include "abi_bridge.h"
#include "indirect_call_site_hit.h"
#include "audio/ax_dsp.h"
#include "brawl_dirty_memory.h"
#include "insn_count.h"
#include "memory.h"
#include "ppc_runtime.h"
#include "runtime_log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#pragma clang fp contract(off)

MKW_NATIVE_BODY(0x80284F2Cu, GameNat_QsFromMatrix);
MKW_NATIVE_BODY(0x8028526Cu, GameNat_QsSetMul);
MKW_NATIVE_BODY(0x80284DB0u, GameNat_QsToMatrix);
MKW_NATIVE_BODY(0x8032EB5Cu, GameNat_SyncModelSpace);
MKW_NATIVE_BODY(0x80400B48u, GameNat_IeeeSqrt);  // the one sqrt native; mtx.cpp no longer registers its copy
MKW_NATIVE_BODY(0x80284FECu, GameNat_QsSetInverse);
MKW_NATIVE_BODY(0x802855F0u, GameNat_QsMulInverseMul);
MKW_NATIVE_BODY(0x8003E6DCu, GameNat_MtScale);
MKW_NATIVE_BODY(0x8002DC74u, GameNat_GfTaskProcess);
MKW_NATIVE_BODY(0x8002E578u, GameNat_GfTaskScheduler);

extern "C" void func_80284F2C_body(CpuContext* ctx);
extern "C" void func_8028526C_body(CpuContext* ctx);
extern "C" void func_80284DB0_body(CpuContext* ctx);
extern "C" void func_8032EB5C_body(CpuContext* ctx);
extern "C" void func_80400B48_body(CpuContext* ctx);
extern "C" void func_80284FEC_body(CpuContext* ctx);
extern "C" void func_802855F0_body(CpuContext* ctx);
extern "C" void func_8003E6DC_body(CpuContext* ctx);
extern "C" void func_8002DC74_body(CpuContext* ctx);
extern "C" void func_8002DC74(CpuContext* ctx);  // its wrapper (guard), as gfTaskScheduler::process calls it
extern "C" void func_8002E578_body(CpuContext* ctx);
// gfTaskScheduler::process's callees, through the EABI entries its translation calls (prototypes as emitted).
extern "C" __attribute__((weak)) MkwEabiPair func_8002E44C_eabi(CpuContext*, uint32_t, uint32_t);
extern "C" __attribute__((weak)) void func_8002EB88_eabi(CpuContext*, uint32_t, uint32_t);
// The translation's guard table (guest_code_patch.cpp): every translated function's start, ascending; a function's
// guard index is its position.
extern "C" const uint32_t g_mkwGuardedFunctionStarts[];
extern "C" const uint32_t g_mkwGuardedFunctionCount;
// The vanilla translation's EABI entries the original's callees are reached through (absent from a translation
// without MKW_EABI_REGS, so weak): their dead non-volatile saves are zeros only there.
extern "C" __attribute__((weak)) void func_80283524_eabi(CpuContext*, uint32_t, uint32_t, PPC_FPR);
extern "C" __attribute__((weak)) MkwEabiPair func_80285A6C_eabi(CpuContext*, uint32_t, uint32_t, uint32_t, PPC_FPR,
                                                                 PPC_FPR);
extern "C" __attribute__((weak)) MkwEabiPair func_80282CD4_eabi(CpuContext*, uint32_t, uint32_t, uint32_t, uint32_t,
                                                                 PPC_FPR, PPC_FPR);

void GameNativesLogStats(double frames);

namespace gamenat {
namespace {

// ---------------------------------------------------------------------------------------------------------------
// Switches and counters (g3d_calc.cpp's)
// ---------------------------------------------------------------------------------------------------------------
enum Mode : uint8_t { kNative = 1, kOriginal = 0, kVerify = 2 };
enum Fn : uint8_t {
    kQsFromMatrix, kQsSetMul, kQsToMatrix, kSyncModelSpace, kIeeeSqrt, kQsSetInverse, kQsMulInverseMul, kMtScale,
    kGfTaskProcess, kGfTaskScheduler, kNumFn
};
const char* const kFnNames[kNumFn] = {"qsfrommatrix", "qssetmul",    "qstomatrix", "syncmodel", "sqrt",
                                      "qsinverse",    "qsmulinvmul", "mtscale",    "gftask",    "gfsched"};

inline bool EabiTranslation() {
    return func_80283524_eabi != nullptr && func_80285A6C_eabi != nullptr && func_80282CD4_eabi != nullptr;
}

// Natives that run their originals unless GAME_NATIVES_ON names them (none now). syncmodel and qsmulinvmul were off
// after a verify run reported mismatches on 13.7 of 21.1 and 20.4 of 20.4 calls/frame: the
// harness's, not theirs (Verify's buffers were per instantiation, so a nested verify, e.g. setMul's inside
// syncModelSpace, overwrote its caller's saved bytes; per native since). Re-verified over a full Training
// match: 0 mismatches for both, and the deterministic hash matches with all of them on.
constexpr bool kFnDefaultOff[kNumFn] = {false, false, false, false, false, false, false, false, false, false};

struct Settings {
    Mode mode = kNative;
    bool off[kNumFn] = {};
    Settings() {
        if (const char* v = std::getenv("GAME_NATIVES"); v != nullptr && v[0] != '\0')
            mode = v[0] == '0' ? kOriginal : v[0] == '2' ? kVerify : kNative;
        const auto listed = [](const char* env, int i) {
            const char* const v = std::getenv(env);
            if (v == nullptr) return false;
            return (std::string(",") + v + ",").find(std::string(",") + kFnNames[i] + ",") != std::string::npos;
        };
        for (int i = 0; i < kNumFn; ++i)
            off[i] = (kFnDefaultOff[i] && !listed("GAME_NATIVES_ON", i)) || listed("GAME_NATIVES_OFF", i);
    }
};
const Settings& Config() {
    static const Settings settings;
    return settings;
}
// gftask's native mode, decided once at startup (a plain load per call; see GfTaskProcess).
const bool g_gfTaskFast = Config().mode == kNative && !Config().off[kGfTaskProcess] && EabiTranslation();
// gftask's per-type target cache (GFTASK_CACHE=0: the inline-cache site alone, as first shipped).
const bool g_gfTaskCache = [] {
    const char* v = std::getenv("GFTASK_CACHE");
    return v == nullptr || v[0] != '0';
}();

struct Counters {
    std::atomic<uint64_t> calls{0}, native{0}, declined{0}, fallback{0}, compared{0}, mismatches{0}, undeclared{0},
        foreign{0}, overmarked{0};
};
Counters g_counters[kNumFn];
// A relaxed load and store, not a locked read-modify-write; these numbers are only logged.
inline void Bump(std::atomic<uint64_t>& counter, uint64_t n = 1) {
    counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

#if MKW_LEAN_MEMORY
// ---------------------------------------------------------------------------------------------------------------
// Lean memory (memory_access.h; g3d_calc.cpp's helpers): guest address & kLeanMask is the offset into one host window;
// rollback's dirty marks are one byte per 64 guest bytes; a 64 KiB chunk with a non-zero policy byte sends stores to
// the checked path.
// ---------------------------------------------------------------------------------------------------------------
inline uint8_t* Window() { return MemoryInline::g_leanBase; }
inline uint32_t Canonical(uint32_t address) { return address & MemoryInline::kLeanMask; }
inline bool LeanReady() {
    return MemoryInline::g_leanBase != nullptr &&
           BrawlDirtyMemory::flatBase == reinterpret_cast<uintptr_t>(MemoryInline::g_leanBase);
}
// The range lies in one view of the window and no store to it takes the checked path.
inline bool PlainRange(uint32_t address, uint64_t size) {
    if (size == 0) return true;
    const uint32_t c = Canonical(address);
    if (c + size > MemoryInline::kLeanWindowSize) return false;
    if (((address ^ (address + static_cast<uint32_t>(size - 1))) & ~MemoryInline::kLeanMask) != 0) return false;
    const uint8_t* const policy = Window() + MemoryInline::kLeanPolicyOffset;
    for (uint64_t chunk = c >> MemoryInline::kLeanPolicyShift; chunk <= (c + size - 1) >> MemoryInline::kLeanPolicyShift;
         ++chunk)
        if (policy[chunk] != 0) return false;
    return true;
}
// The range lies in one view of the window (one host span; guest loads read the window at any address).
inline bool OneView(uint32_t address, uint32_t size) {
    if (size == 0) return true;
    return Canonical(address) + static_cast<uint64_t>(size) <= MemoryInline::kLeanWindowSize &&
           ((address ^ (address + (size - 1))) & ~MemoryInline::kLeanMask) == 0;
}
inline bool Overlap(uint32_t a, uint32_t aSize, uint32_t b, uint32_t bSize) {
    return aSize != 0 && bSize != 0 && Canonical(a) < Canonical(b) + static_cast<uint64_t>(bSize) &&
           Canonical(b) < Canonical(a) + static_cast<uint64_t>(aSize);
}
inline uint32_t Load32(const uint8_t* host) {
    uint32_t v;
    std::memcpy(&v, host, 4);
    return __builtin_bswap32(v);
}
inline void Store32(uint8_t* host, uint32_t v) {
    v = __builtin_bswap32(v);
    std::memcpy(host, &v, 4);
}
inline float AsFloat(uint32_t bits) { return __builtin_bit_cast(float, bits); }
inline uint32_t Bits(float value) { return __builtin_bit_cast(uint32_t, value); }

// Stores of the native's final image: the bytes and the dirty mark of the 64-byte block each lands in (always inside
// the original's marks, which cover at least every block it wrote). A build without dirty marks
// (MKW_LEAN_DIRTY_MARKS=0, brawl_dirty_memory.h: no translated store marks) gets none from the natives either.
#if !defined(MKW_LEAN_DIRTY_MARKS) || MKW_LEAN_DIRTY_MARKS
constexpr bool kDirtyMarks = true;
#else
constexpr bool kDirtyMarks = false;
#endif
struct Image {
    uint8_t* m;
    uint8_t* marks;
    void W32(uint32_t address, uint32_t value) const {
        const uint32_t c = Canonical(address);
        Store32(m + c, value);
        if constexpr (kDirtyMarks) marks[c >> 6] = 1;
    }
    void WF(uint32_t address, float value) const { W32(address, Bits(value)); }
    void Mark(uint32_t address) const {
        if constexpr (kDirtyMarks) marks[Canonical(address) >> 6] = 1;
    }
    void W8(uint32_t address, uint8_t value) const {
        const uint32_t c = Canonical(address);
        m[c] = value;
        if constexpr (kDirtyMarks) marks[c >> 6] = 1;
    }
    // `words` floats at address, from host values.
    void WFs(uint32_t address, const float* values, uint32_t words) const {
        for (uint32_t i = 0; i < words; ++i) WF(address + 4u * i, values[i]);
    }
    // Zero bytes over a span inside one view (the callers' PlainRange), and the marks of every block it touches.
    void ZeroSpan(uint32_t address, uint32_t bytes) const {
        const uint32_t c = Canonical(address);
        std::memset(m + c, 0, bytes);
        if constexpr (kDirtyMarks) std::memset(marks + (c >> 6), 1, ((c + bytes - 1) >> 6) - (c >> 6) + 1);
    }
};

// A normal single or a zero: lfs widens it exactly, stfs narrows it back to the same bits, and the host's single
// arithmetic on it matches the helpers'.
inline bool PlainSingle(uint32_t bits) {
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    return magnitude == 0 || magnitude - 0x00800000u < 0x7F000000u;
}

// ---------------------------------------------------------------------------------------------------------------
// Host FP exception flags: cleared before a native's first floating-point operation, read after all its results are in
// memory (the asm statements take a pointer to them and clobber memory, so no operation moves across either).
// x86 MXCSR: IE 0x01, DE 0x02, ZE 0x04, OE 0x08, UE 0x10 (PE 0x20 is every inexact result). AArch64 FPSR: IOC 0x01,
// DZC 0x02, OFC 0x04, UFC 0x08, IDC 0x80 (IXC 0x10 inexact).
// ---------------------------------------------------------------------------------------------------------------
inline void FpFlagsClear() {
#if defined(__x86_64__)
    uint32_t csr;
    asm volatile("stmxcsr %0" : "=m"(csr) : : "memory");
    csr &= ~0x3Fu;
    asm volatile("ldmxcsr %0" : : "m"(csr) : "memory");
#elif defined(__aarch64__)
    uint64_t fpsr;
    asm volatile("mrs %0, fpsr" : "=r"(fpsr) : : "memory");
    asm volatile("msr fpsr, %0" : : "r"(fpsr & ~uint64_t{0x9F}) : "memory");
#endif
}
inline bool FpFlagsRaised(const void* results) {
#if defined(__x86_64__)
    uint32_t csr;
    asm volatile("stmxcsr %0" : "=m"(csr) : "r"(results) : "memory");
    return (csr & 0x1Fu) != 0;
#elif defined(__aarch64__)
    uint64_t fpsr;
    asm volatile("mrs %0, fpsr" : "=r"(fpsr) : "r"(results) : "memory");
    return (fpsr & 0x8Fu) != 0;
#endif
}

// A value the optimizer cannot see through: keeps it from speculating an operation on a value the original never
// computes with (the host flags would raise for it).
inline float Opaque(float v) {
#if defined(__x86_64__)
    asm("" : "+x"(v));
#elif defined(__aarch64__)
    asm("" : "+w"(v));
#endif
    return v;
}

// The operations: host single precision (see the header). Products and FMAs are tracked for the x86 tininess gap.
struct Ops {
#if defined(__x86_64__)
    // min over tracked results of (|bits| << 1) - 1: 0x00FFFFFF exactly for +-2^-126 (smaller magnitudes are flushed
    // and flagged, a zero wraps to the maximum).
    uint32_t band = UINT32_MAX;
    void Track(float r) { band = std::min(band, (Bits(r) << 1) - 1u); }
    bool Gap() const { return band <= 0x00FFFFFFu; }
#else
    void Track(float) {}
    bool Gap() const { return false; }
#endif
    float Mul(float a, float c) { const float r = a * c; Track(r); return r; }
    float Madd(float a, float c, float b) { const float r = __builtin_fmaf(a, c, b); Track(r); return r; }
    float Msub(float a, float c, float b) { const float r = __builtin_fmaf(a, c, -b); Track(r); return r; }
    float Nmsub(float a, float c, float b) { const float r = __builtin_fmaf(a, c, -b); Track(r); return -r; }
    static float Add(float a, float b) { return a + b; }
    static float Sub(float a, float b) { return a - b; }
    static float Div(float a, float b) { return a / b; }
    // frsp(frsqrte(x)): the translated code's estimate, rounded as its frsp rounds it.
    static float Rsqrte(float x) { return PpcForceSingleValueInline(PPC_Frsqrte(static_cast<double>(x))); }
    // The rsqrt sequence inlined everywhere here (hkMath::sqrtInverse): r = frsp(frsqrte(x)); 0.5r * (3 - r * (x r)),
    // the last factor an fnmsubs. x > 0.
    float SqrtInverse(float x) {
        const float r = Rsqrte(x);
        const float xr = Mul(x, r);
        const float half = Mul(0.5f, r);
        const float t = Nmsub(r, xr, 3.0f);
        return Mul(half, t);
    }
};
#endif  // MKW_LEAN_MEMORY

// ---------------------------------------------------------------------------------------------------------------
// Verify (GAME_NATIVES=2): g3d_calc.cpp's harness, with NativeMarksCover and a whole-register-file compare.
// ---------------------------------------------------------------------------------------------------------------
struct GuestRange {
    uint32_t address, size;
};
struct Ranges {
    static constexpr int kMax = 8;
    GuestRange items[kMax];
    int count = 0;
    bool Add(uint32_t address, uint32_t size) {
        if (count == kMax) return false;
        items[count++] = {address, size};
        return true;
    }
};

#if MKW_LEAN_MEMORY
struct MarkSpan {
    uint32_t firstBlock, blocks;
};
// The AX mix worker's AI output frame (0x804951C0): it stores and marks it from its own thread (see Verify).
constexpr uint32_t kAxOutputPage = 0x00495000u;
constexpr MarkSpan kMarkSpans[] = {
    {0x00000000u >> 6, 0x01800000u >> 6}, {0x10000000u >> 6, 0x04000000u >> 6}, {0x20000000u >> 6, 0x4000u >> 6}};
struct MarkSpans {
    const MarkSpan* items;
    int count;
    MarkSpan local[Ranges::kMax];
};
[[maybe_unused]] void AllSpans(MarkSpans& out) {
    out.items = kMarkSpans;
    out.count = static_cast<int>(sizeof(kMarkSpans) / sizeof(kMarkSpans[0]));
}
// g3d_calc.cpp's local spans: the declared ranges' blocks, each with the eight blocks past its end that a segment
// range's span (or a store's mark pair) may reach.
[[maybe_unused]] void LocalSpans(const Ranges& ranges, MarkSpans& out) {
    for (int i = 0; i < ranges.count; ++i) {
        const uint32_t c = Canonical(ranges.items[i].address);
        out.local[i] = {c >> 6, ((c + ranges.items[i].size - 1) >> 6) - (c >> 6) + 9};
    }
    out.items = out.local;
    out.count = ranges.count;
}
// Verify's mark check over all of MEM1/MEM2/LC costs ~1.4 MB of copies per call: the frequent natives check their
// declared ranges' marks, and all of them on every kFullMarksEvery-th call (g3d_calc.cpp's scheme).
constexpr bool kFnLocalMarks[kNumFn] = {false, true, true, true, true, true, true, true, false, false};
constexpr uint64_t kFullMarksEvery = 64;
void SaveMarks(const MarkSpans& spans, std::vector<uint8_t>& out) {
    size_t bytes = 0;
    for (int i = 0; i < spans.count; ++i) bytes += spans.items[i].blocks;
    out.resize(bytes);
    size_t at = 0;
    for (int i = 0; i < spans.count; ++i) {
        const MarkSpan& s = spans.items[i];
        std::memcpy(out.data() + at, BrawlDirtyMemory::leanBlocks + s.firstBlock, s.blocks);
        at += s.blocks;
    }
}
void ClearMarks(const MarkSpans& spans) {
    for (int i = 0; i < spans.count; ++i)
        std::memset(BrawlDirtyMemory::leanBlocks + spans.items[i].firstBlock, 0, spans.items[i].blocks);
}
// marks = a | b | c
void RestoreMarks(const MarkSpans& spans, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                  const std::vector<uint8_t>& c) {
    size_t at = 0;
    for (int k = 0; k < spans.count; ++k) {
        const MarkSpan& s = spans.items[k];
        uint8_t* const dst = BrawlDirtyMemory::leanBlocks + s.firstBlock;
        for (uint32_t i = 0; i < s.blocks; ++i) dst[i] = a[at + i] | b[at + i] | c[at + i];
        at += s.blocks;
    }
}
// The first marked block (guest canonical address) outside the ranges (a range's marks may spill one block past
// its end, and a segment range's span up to eight blocks from its start), or UINT32_MAX.
uint32_t FirstUndeclared(const MarkSpans& spans, const std::vector<uint8_t>& marks, const Ranges& ranges) {
    size_t at = 0;
    for (int k = 0; k < spans.count; ++k) {
        const MarkSpan& s = spans.items[k];
        for (uint32_t i = 0; i < s.blocks; ++i) {
            if (marks[at + i] == 0) continue;
            const uint32_t block = s.firstBlock + i;
            if ((block << 6) - kAxOutputPage < 0x1000u) continue;  // the AX mix worker's
            bool covered = false;
            for (int r = 0; r < ranges.count && !covered; ++r) {
                const uint32_t c = Canonical(ranges.items[r].address);
                covered = block >= (c >> 6) && block <= ((c + ranges.items[r].size - 1) >> 6) + 8;
            }
            if (!covered) return block << 6;
        }
        at += s.blocks;
    }
    return UINT32_MAX;
}
bool MarkSet(const MarkSpans& spans, const std::vector<uint8_t>& marks, uint32_t block) {
    size_t at = 0;
    for (int s = 0; s < spans.count; ++s) {
        const MarkSpan& span = spans.items[s];
        if (block >= span.firstBlock && block < span.firstBlock + span.blocks) return marks[at + block - span.firstBlock] != 0;
        at += span.blocks;
    }
    return false;
}
// the rule: the native's marks are a subset of the original's and cover every block of the declared
// ranges whose bytes it changed.
bool NativeMarksCover(const MarkSpans& spans, const std::vector<uint8_t>& nativeMarks,
                      const std::vector<uint8_t>& origMarks, const Ranges& ranges, const std::vector<uint8_t>& before,
                      const std::vector<uint8_t>& after) {
    for (size_t i = 0; i < nativeMarks.size(); ++i)
        if (nativeMarks[i] != 0 && origMarks[i] == 0) return false;
    size_t at = 0;
    for (int r = 0; r < ranges.count; ++r) {
        const uint32_t c = Canonical(ranges.items[r].address);
        for (uint32_t k = 0; k < ranges.items[r].size; ++k)
            if (before[at + k] != after[at + k] && !MarkSet(spans, nativeMarks, (c + k) >> 6)) return false;
        at += ranges.items[r].size;
    }
    return true;
}
void Capture(const Ranges& ranges, std::vector<uint8_t>& out) {
    out.clear();
    for (int i = 0; i < ranges.count; ++i) {
        const uint8_t* const host = Window() + Canonical(ranges.items[i].address);
        out.insert(out.end(), host, host + ranges.items[i].size);
    }
}
void Restore(const Ranges& ranges, const std::vector<uint8_t>& in) {
    size_t at = 0;
    for (int i = 0; i < ranges.count; ++i) {
        std::memcpy(Window() + Canonical(ranges.items[i].address), in.data() + at, ranges.items[i].size);
        at += ranges.items[i].size;
    }
}
// The whole register file (the natives reproduce every register the originals leave in the context); every
// difference is listed (original/native).
bool SameContext(const CpuContext& a, const CpuContext& b, char* what, size_t whatSize) {
    size_t at = std::strlen(what);
    const auto add = [&](const char* format, auto... values) {
        if (at < whatSize) at += static_cast<size_t>(std::snprintf(what + at, whatSize - at, format, values...));
    };
    for (uint32_t r = 0; r < 32; ++r)
        if (a.gpr[r] != b.gpr[r]) add("r%u %08X/%08X ", r, a.gpr[r], b.gpr[r]);
    for (uint32_t f = 0; f < 32; ++f)
        if (a.fpr[f].raw != b.fpr[f].raw)
            add("f%u %016llX/%016llX ", f, static_cast<unsigned long long>(a.fpr[f].raw),
                static_cast<unsigned long long>(b.fpr[f].raw));
    if (PpcGetCr(&a) != PpcGetCr(&b)) add("cr %08X/%08X ", PpcGetCr(&a), PpcGetCr(&b));
    if (a.lr != b.lr) add("lr %08X/%08X ", a.lr, b.lr);
    if (a.ctr != b.ctr) add("ctr %08X/%08X ", a.ctr, b.ctr);
    if (a.xer != b.xer) add("xer %08X/%08X ", a.xer, b.xer);
    if (a.fpscr != b.fpscr) add("fpscr %08X/%08X ", a.fpscr, b.fpscr);
    for (uint32_t q = 0; q < 8; ++q)
        if (a.gqr[q] != b.gqr[q]) add("gqr%u %08X/%08X ", q, a.gqr[q], b.gqr[q]);
    return at == 0;
}

std::atomic<int> g_verifyLogs{0};
constexpr int kMaxVerifyLogs = 60;

// NAT_INSN=1 (insn_count.h; g3d_calc.cpp's): a compared call's original and native instructions, sites "game.<name>".
// A declined native costs its attempt plus the original it falls back to, as in the native mode.
int FnSite(Fn fn) {
    static const std::array<int, kNumFn> sites = [] {
        std::array<int, kNumFn> s{};
        for (size_t i = 0; i < s.size(); ++i) s[i] = InsnCount::Site((std::string("game.") + kFnNames[i]).c_str());
        return s;
    }();
    return sites[fn];
}
void CountPair(Fn fn, uint64_t original, uint64_t native) {
    InsnCount::Add(FnSite(fn), InsnCount::kOriginal, original);
    InsnCount::Add(FnSite(fn), InsnCount::kNative, native);
}
// The native mode: a call's instructions (a declined native: its attempt plus the original).
void CountNative(Fn fn, uint64_t instructions) { InsnCount::Add(FnSite(fn), InsnCount::kNative, instructions); }

struct VerifyBuffers {
    std::vector<uint8_t> savedMarks, origMarks, nativeMarks, before, afterOrig, afterNative, rerunMarks;
};
thread_local VerifyBuffers t_verifyBuffers[kNumFn];

// Original, then native, from the same state; the original's results stay. `writes` declares the guest ranges the
// function may write; `fixup` adjusts the original's result as the vanilla callers' EABI entry left it (dead saves);
// `native` returns false where it declines (no effect).
template <typename Writes, typename Fixup, typename Native>
void Verify(Fn fn, CpuContext* ctx, void (*body)(CpuContext*), Writes writes, Fixup fixup, Native native) {
    // Buffers per native, not per instantiation: natives with the same callable types share an instantiation, and an
    // original that calls another native (syncModelSpace's setMul) runs that native's Verify inside this one, which
    // would overwrite `before` before this run restores it (syncmodel and qsmulinvmul mismatched
    // in exactly the bytes they do not write).
    VerifyBuffers& buffers = t_verifyBuffers[fn];
    std::vector<uint8_t>& savedMarks = buffers.savedMarks;
    std::vector<uint8_t>& origMarks = buffers.origMarks;
    std::vector<uint8_t>& nativeMarks = buffers.nativeMarks;
    std::vector<uint8_t>& before = buffers.before;
    std::vector<uint8_t>& afterOrig = buffers.afterOrig;
    std::vector<uint8_t>& afterNative = buffers.afterNative;
    Counters& n = g_counters[fn];
    Ranges ranges;
    if (!LeanReady() || !writes(*ctx, ranges)) {
        n.declined.fetch_add(1, std::memory_order_relaxed);
        const uint32_t r1 = ctx->gpr[1];
        body(ctx);
        fixup(r1);
        return;
    }
    const CpuContext in = *ctx;
    MarkSpans spans;
    if constexpr (!kDirtyMarks) {  // (no store marks on either side: nothing to compare, as axfx_native.cpp's verify)
        spans.items = spans.local;
        spans.count = 0;
    } else if (kFnLocalMarks[fn] && n.calls.load(std::memory_order_relaxed) % kFullMarksEvery != 1) {
        LocalSpans(ranges, spans);
    } else {
        AllSpans(spans);
    }
    // The AX mix worker stores each AI frame into guest memory (with its marks) from its own thread, and only this
    // thread starts a mix: idle, it cannot land inside the windows below.
    AxDspHle::JoinMixWorker();
    SaveMarks(spans, savedMarks);
    ClearMarks(spans);
    Capture(ranges, before);
    const uint64_t originalStart = InsnCount::On() ? InsnCount::Read() : 0;
    body(ctx);
    const uint64_t originalInsns = InsnCount::On() ? InsnCount::Since(originalStart) : 0;
    fixup(in.gpr[1]);
    const CpuContext orig = *ctx;
    SaveMarks(spans, origMarks);
    Capture(ranges, afterOrig);
    const uint32_t undeclared = FirstUndeclared(spans, origMarks, ranges);
    if (undeclared != UINT32_MAX) {
        n.undeclared.fetch_add(1, std::memory_order_relaxed);
        if (g_verifyLogs.fetch_add(1, std::memory_order_relaxed) < kMaxVerifyLogs)
            RT_LOGF(RT_TAG_GX, "[gamenat] verify %s: the original wrote outside the declared ranges (canonical %08X)\n",
                    kFnNames[fn], undeclared);
        nativeMarks.assign(origMarks.size(), 0);
        RestoreMarks(spans, savedMarks, origMarks, nativeMarks);
        return;
    }
    Restore(ranges, before);
    ClearMarks(spans);
    *ctx = in;
    const uint64_t nativeStart = InsnCount::On() ? InsnCount::Read() : 0;
    const bool ranNatively = native(ctx);
    if (InsnCount::On()) {
        const uint64_t nativeInsns = InsnCount::Since(nativeStart);
        CountPair(fn, originalInsns, ranNatively ? nativeInsns : nativeInsns + originalInsns);
    }
    if (!ranNatively) {
        n.declined.fetch_add(1, std::memory_order_relaxed);
        *ctx = orig;
        Restore(ranges, afterOrig);
        nativeMarks.assign(origMarks.size(), 0);
        RestoreMarks(spans, savedMarks, origMarks, nativeMarks);
        return;
    }
    SaveMarks(spans, nativeMarks);
    Capture(ranges, afterNative);
    char what[640] = "";
    bool same = SameContext(orig, *ctx, what, sizeof(what));
    if (afterNative != afterOrig) {  // every differing word counted, the first few listed (original/native)
        same = false;
        size_t at = 0, len = std::strlen(what);
        uint32_t words = 0;
        for (int i = 0; i < ranges.count; ++i) {
            for (uint32_t w = 0; w + 4 <= ranges.items[i].size; w += 4) {
                if (std::memcmp(afterNative.data() + at + w, afterOrig.data() + at + w, 4) == 0) continue;
                if (++words <= 6 && len < sizeof(what))
                    len += static_cast<size_t>(std::snprintf(what + len, sizeof(what) - len, "mem %08X %08X/%08X ",
                                                             ranges.items[i].address + w,
                                                             Load32(afterOrig.data() + at + w),
                                                             Load32(afterNative.data() + at + w)));
            }
            at += ranges.items[i].size;
        }
        if (len < sizeof(what)) std::snprintf(what + len, sizeof(what) - len, "(%u words differ)", words);
    }
    if (same && nativeMarks != origMarks &&
        NativeMarksCover(spans, nativeMarks, origMarks, ranges, before, afterNative)) {
        Bump(n.overmarked);  // the original marked more (segment ranges' spans)
    } else if (same && nativeMarks != origMarks) {
        same = false;
        size_t at = 0;
        bool outsideOnly = true;  // every differing mark outside the declared ranges
        bool axOnly = true;       // and all of them in the AX mixer's output frame page
        for (int s = 0; s < spans.count; ++s) {
            for (uint32_t i = 0; i < spans.items[s].blocks; ++i) {
                if (origMarks[at + i] == nativeMarks[at + i]) continue;
                const uint32_t block = spans.items[s].firstBlock + i;
                axOnly = axOnly && (block << 6) - kAxOutputPage < 0x1000u;
                if (what[0] == '\0')
                    std::snprintf(what, sizeof(what), "dirty mark of canonical %08X: original %u native %u",
                                  block << 6, origMarks[at + i], nativeMarks[at + i]);
                for (int r = 0; r < ranges.count && outsideOnly; ++r) {
                    const uint32_t c = Canonical(ranges.items[r].address);
                    outsideOnly = block < (c >> 6) || block > ((c + ranges.items[r].size - 1) >> 6) + 8;
                }
            }
            at += spans.items[s].blocks;
        }
        // Only there: another host thread's store landing in the window, unless a second native run repeats it. The AX
        // mix worker stores the AI output frame (canonical 0x4951C0's page) and marks it from its own thread; while a
        // full-marks check runs it lands in the native's run and its rerun alike, so marks only there are its.
        if (outsideOnly && axOnly) {
            same = true;
            n.foreign.fetch_add(1, std::memory_order_relaxed);
            what[0] = '\0';
        } else if (outsideOnly) {
            std::vector<uint8_t>& rerunMarks = buffers.rerunMarks;
            Restore(ranges, before);
            ClearMarks(spans);
            CpuContext rerun = in;
            native(&rerun);
            SaveMarks(spans, rerunMarks);
            if (NativeMarksCover(spans, rerunMarks, origMarks, ranges, before, afterNative)) {
                same = true;
                n.foreign.fetch_add(1, std::memory_order_relaxed);
                if (g_verifyLogs.fetch_add(1, std::memory_order_relaxed) < kMaxVerifyLogs)
                    RT_LOGF(RT_TAG_GX, "[gamenat] verify %s: a mark outside the declared ranges, not repeated by a "
                            "second native run (another thread's store): %s\n", kFnNames[fn], what);
                what[0] = '\0';
            }
            for (size_t k = 0; k < nativeMarks.size(); ++k) nativeMarks[k] |= rerunMarks[k];
        }
    }
    n.compared.fetch_add(1, std::memory_order_relaxed);
    if (!same) {
        n.mismatches.fetch_add(1, std::memory_order_relaxed);
        if (g_verifyLogs.fetch_add(1, std::memory_order_relaxed) < kMaxVerifyLogs)
            RT_LOGF(RT_TAG_GX, "[gamenat] verify MISMATCH %s: r1=%08X r3=%08X r4=%08X r5=%08X lr=%08X: %s\n",
                    kFnNames[fn], in.gpr[1], in.gpr[3], in.gpr[4], in.gpr[5], in.lr, what);
    }
    *ctx = orig;
    Restore(ranges, afterOrig);
    RestoreMarks(spans, savedMarks, origMarks, nativeMarks);
}
#endif  // MKW_LEAN_MEMORY

// Dispatch for one native: counts, GAME_NATIVES / GAME_NATIVES_OFF, verify. `fixup` runs after every original run
// the native stands for (the dead saves the vanilla callers' EABI entry left).
template <typename Writes, typename Fixup, typename Native>
[[gnu::always_inline]] inline void Run(Fn fn, CpuContext* ctx, void (*body)(CpuContext*), Writes writes, Fixup fixup,
                                       Native native) {
    const Settings& s = Config();
    Bump(g_counters[fn].calls);
#if MKW_LEAN_MEMORY
    if (s.mode == kNative && !s.off[fn]) {
        if (InsnCount::On()) [[unlikely]] {  // the whole call into game.<name>'s native column
            const uint64_t start = InsnCount::Read();
            const bool ranNatively = native(ctx);
            if (!ranNatively) {
                const uint32_t r1 = ctx->gpr[1];
                body(ctx);
                fixup(r1);
            }
            CountNative(fn, InsnCount::Since(start));
            Bump(ranNatively ? g_counters[fn].native : g_counters[fn].declined);
            return;
        }
        if (native(ctx)) {
            Bump(g_counters[fn].native);
            return;
        }
        Bump(g_counters[fn].declined);
    } else if (s.mode == kVerify && !s.off[fn]) {
        Verify(fn, ctx, body, writes, fixup, native);
        return;
    }
#else
    (void)s;
    (void)writes;
    (void)native;
#endif
    const uint32_t r1 = ctx->gpr[1];  // (not a context copy: this is the GAME_NATIVES=0 arm of every A/B)
    body(ctx);
    fixup(r1);
}

#if MKW_LEAN_MEMORY
// ---------------------------------------------------------------------------------------------------------------
// QsFromMatrix (0x80284F2C) with decomposeMatrix (0x802821DC) and its callees; see the header. Frames (entry r1 = R0):
//   QsFromMatrix  s0 = (R0 & ~15) - 0xF0: back chain R0, LR at R0+4, r31 at R0-4; the Decomposition D at s0+0x10
//   decomposeMatrix s1 = s0 - 0x270: back chain s0, LR 0x80284F58 at s0+4, r27-r31 at s0-20..s0-4 (the context's,
//     as its context body saves them); temporaries at s1+0x20..0x24C; callees' LR 0x80282334 (invert's) at s1+4
//   callees (all frames below s1): final image = setMul's (back chain s1 at s1-0x120, its 12 temporaries at
//     s1-0x110, saved f18-f31 as zeros at [s1-0xE0, s1)) overlaid by the final isApproximatelyEqual's (back chain s1
//     at s1-0x70; rows at s1-0x20, s1-0x40, s1-0x60 as far as it got)
// ---------------------------------------------------------------------------------------------------------------
constexpr uint32_t kQsWindowBelow = 0x480u;  // R0 - 15 - 0xF0 - 0x270 - 0x120 rounded up
constexpr uint32_t kSdata2A = 0x5638u;       // r2 - 0x5638: 10 words (eps 2^-23, 0.5, 0, 3, -1, 1, 0.001, 0, 0, 1)
constexpr uint32_t kSdata2B = 0x55E8u;       // r2 - 0x55E8: 6 words (0.5, 1, 0, 3, 1, 0)
constexpr uint32_t kSdata2C = 0x55A0u;       // r2 - 0x55A0: 2 words (0, 1)
constexpr uint32_t kConstA[10] = {0x34000000u, 0x3F000000u, 0u, 0x40400000u, 0xBF800000u,
                                  0x3F800000u, 0x3A83126Fu, 0u, 0u,          0x3F800000u};
constexpr uint32_t kConstB[6] = {0x3F000000u, 0x3F800000u, 0u, 0x40400000u, 0x3F800000u, 0u};
constexpr uint32_t kConstC[2] = {0u, 0x3F800000u};
constexpr uint32_t kQuatNext = 0x8040F880u;  // {1, 2, 0}
constexpr uint32_t kQuatNextValues[3] = {1u, 2u, 0u};
constexpr uint32_t kIdentity3 = 0x804864F0u;  // hkMatrix3 identity (rows with w = 0)
constexpr uint32_t kIdentity3Values[12] = {0x3F800000u, 0u, 0u, 0u, 0u, 0x3F800000u, 0u, 0u, 0u, 0u, 0x3F800000u, 0u};
constexpr float kEps = 0x1p-23f;     // 0x34000000
constexpr float kEps2 = 0.001f;      // 0x3A83126F

bool SameWords(uint32_t address, const uint32_t* values, uint32_t words) {
    const uint8_t* const p = Window() + Canonical(address);
    for (uint32_t i = 0; i < words; ++i)
        if (Load32(p + 4u * i) != values[i]) return false;
    return true;
}

// hkMatrix3::isApproximatelyEqual(a, b, eps) (0x80283280): rows x, y, z within eps (w compared, ignored). Records
// what its frame and registers end with.
struct ApproxResult {
    bool equal;
    int rows;          // rows whose temporaries it stored (1..3)
    float d[3][4];     // |a - b| per row: x, y, z, w
    uint32_t r4;       // r4 at return: b, or the third row's result (extsb of its all-within flag)
    float f2;          // f2 at return: the last row's |x| (rows 1-2 failing), the third row's |z|
};
ApproxResult ApproxEqual(const float* a, const float* b, uint32_t bAddress, float eps) {
    ApproxResult r{};
    r.r4 = bAddress;
    for (int row = 0; row < 3; ++row) {
        const float* const x = a + 4 * row;
        const float* const y = b + 4 * row;
        r.d[row][3] = std::fabs(Ops::Sub(x[3], y[3]));
        r.d[row][0] = std::fabs(Ops::Sub(x[0], y[0]));
        r.d[row][1] = std::fabs(Ops::Sub(x[1], y[1]));
        r.d[row][2] = std::fabs(Ops::Sub(x[2], y[2]));
        r.rows = row + 1;
        const bool pass = !(eps < r.d[row][0]) && !(eps < r.d[row][1]) && !(eps < r.d[row][2]);
        if (row < 2) {
            r.f2 = r.d[row][0];
            if (!pass) return r;
        } else {
            r.f2 = r.d[row][2];
            r.r4 = pass ? 1u : 0u;
            r.equal = pass;
        }
    }
    return r;
}

// hkMatrix3::setMul (0x80282CD4): out row r = sum_k b[r][k] * a row k, as the original rounds it.
void SetMul(Ops& o, float* out, const float* a, const float* b) {
    for (int r = 0; r < 3; ++r) {
        const float b0 = b[4 * r], b1 = b[4 * r + 1], b2 = b[4 * r + 2];
        float u[3];
        for (int c = 0; c < 3; ++c) u[c] = o.Madd(b0, a[c], o.Mul(b1, a[4 + c]));
        for (int c = 0; c < 3; ++c) out[4 * r + c] = o.Madd(b2, a[8 + c], u[c]);
        out[4 * r + 3] = 0.0f;
    }
}

// hkRotation::set(hkQuaternion) (0x80285A6C): R (rows with w = 0) from q = (x, y, z, w), as the original rounds it.
void QuatToRotation(Ops& o, const float* q, float* R) {
    const float x = q[0], y = q[1], z = q[2], qw = q[3];
    const float y2 = Ops::Add(y, y), z2 = Ops::Add(z, z), x2 = Ops::Add(x, x);
    const float yy = o.Mul(y2, y), zz = o.Mul(z2, z), xx = o.Mul(x2, x), xy = o.Mul(y2, x);
    const float yyzz = Ops::Add(yy, zz), xxzz = Ops::Add(xx, zz), xxyy = Ops::Add(xx, yy);
    const float zw = o.Mul(z2, qw);
    R[0] = Ops::Sub(1.0f, yyzz);
    R[5] = Ops::Sub(1.0f, xxzz);
    R[10] = Ops::Sub(1.0f, xxyy);
    R[1] = Ops::Add(xy, zw);
    R[4] = Ops::Sub(xy, zw);
    const float zx = o.Mul(z2, x), yw = o.Mul(y2, qw), zy = o.Mul(z2, y), xw = o.Mul(x2, qw);
    R[2] = Ops::Sub(zx, yw);
    R[8] = Ops::Add(zx, yw);
    R[6] = Ops::Add(zy, xw);
    R[9] = Ops::Sub(zy, xw);
    R[3] = R[7] = R[11] = 0.0f;
}

struct QsResult {
    float t[16];       // s1+0x210: hkTransform (set4x4)
    float q[12], qPrev[12], w[12], acc[12];  // s1+0x1B0, 0x1E0, 0x180, 0x150 (the polar loop's last iteration)
    float c0[4], c1[4], c2[4];               // s1+0xB0, 0xA0, 0x90: the normalized columns
    float cross[3];                          // s1+0x80 (w = 0)
    float quat[4];                           // s1+0x50: hkQuaternion::set's result
    float qn[4];                             // D+0x10: normalized
    float rinv[12];                          // s1+0xF0: hkRotation of the conjugate
    float s[12];                             // D+0x40: scaleAndSkew
    float absd[3];                           // s1+0x30: |scale - 1|
    float diag[12];                          // s1+0xC0: diag(1/scale)
    float skew[12];                          // D+0x70
    ApproxResult final;                      // isApproximatelyEqual(skew, identity, 0.001)
    uint32_t det0;                           // cr0 of fcmpo det, 0.0 (as the context's cr keeps it)
    uint8_t hasScale, flips;
};

// The decomposition in host registers; false: run the original (a path the native does not take: a non-positive
// length or trace sum, which only NaN-free inputs never reach, or a quaternion with |q|^2 <= 0).
bool Decompose(Ops& o, QsResult& r, const float* m) {
    // set4x4: columns with w = 0, translation with w = 1.
    for (int c = 0; c < 4; ++c) {
        r.t[4 * c] = m[4 * c];
        r.t[4 * c + 1] = m[4 * c + 1];
        r.t[4 * c + 2] = m[4 * c + 2];
        r.t[4 * c + 3] = c == 3 ? 1.0f : 0.0f;
    }
    float q[12];
    std::memcpy(q, r.t, sizeof(q));
    for (int iteration = 1;; ++iteration) {
        float* const qp = r.qPrev;
        float* const w = r.w;
        std::memcpy(qp, q, sizeof(q));
        std::memcpy(w, q, sizeof(q));
        // invert(eps) (0x80283524), m = w.
        const float eps3 = o.Mul(kEps, o.Mul(kEps, kEps));
        const float f2a = o.Mul(w[4], w[10]);
        const float f26 = o.Msub(w[6], w[8], f2a);
        const float f2b = o.Mul(w[6], w[9]);
        const float f9a = o.Mul(w[10], w[1]);
        const float f25 = o.Msub(w[5], w[10], f2b);
        const float f4a = o.Mul(w[5], w[8]);
        const float f2c = o.Mul(w[1], f26);
        const float f24 = o.Msub(w[9], w[2], f9a);
        const float f27 = o.Msub(w[4], w[9], f4a);
        const float f2d = o.Madd(w[0], f25, f2c);
        const float f5a = o.Mul(w[9], w[0]);
        const float f7a = o.Mul(w[8], w[2]);
        const float det = o.Madd(w[2], f27, f2d);
        const float f3a = o.Mul(w[1], w[4]);
        const float f4b = o.Mul(w[2], w[5]);
        const float f9 = o.Msub(w[8], w[1], f5a);
        const float f29 = o.Msub(w[10], w[0], f7a);
        const float f12 = o.Msub(w[1], w[6], f4b);
        const float f2e = o.Mul(w[0], w[6]);
        const float f10 = o.Msub(w[2], w[4], f2e);
        const float f8 = o.Msub(w[0], w[5], f3a);
        if (std::fabs(det) > eps3) {
            const float inv = Ops::Div(1.0f, det);
            const float zero = o.Mul(inv, 0.0f);
            // written, then transposed twice (invert's own transpose and decomposeMatrix's): as written
            w[0] = o.Mul(inv, f25);
            w[1] = o.Mul(inv, f26);
            w[3] = zero;
            w[2] = o.Mul(inv, f27);
            w[4] = o.Mul(inv, f24);
            w[5] = o.Mul(inv, f29);
            w[6] = o.Mul(inv, f9);
            w[7] = zero;
            w[8] = o.Mul(inv, f12);
            w[9] = o.Mul(inv, f10);
            w[10] = o.Mul(inv, f8);
            w[11] = zero;
        } else {  // invert failed: decomposeMatrix's transpose of the copy
            std::swap(w[1], w[4]);
            std::swap(w[2], w[8]);
            std::swap(w[6], w[9]);
        }
        // add (0x80283910) into the copy of q; scale by 0.5 (0x80282F50)
        for (int i = 0; i < 12; ++i) r.acc[i] = Ops::Add(qp[i], w[i]);
        for (int i = 0; i < 12; ++i) q[i] = o.Mul(0.5f, r.acc[i]);
        if (iteration >= 30) break;
        if (ApproxEqual(q, qp, 0, kEps).equal) break;
    }
    std::memcpy(r.q, q, sizeof(q));

    // Normalized columns (stack 0xB0, 0xA0, 0x90); 0.0 for a zero column, the original's +inf for a negative
    // length (never: a sum of squares) is declined.
    float inv[3];
    for (int c = 0; c < 3; ++c) {
        const float* const col = q + 4 * c;
        const float len = o.Madd(col[2], col[2], o.Madd(col[0], col[0], o.Mul(col[1], col[1])));
        if (len == 0.0f) {
            inv[c] = 0.0f;
        } else if (len < 0.0f || !(len > 0.0f)) {
            return false;
        } else {
            inv[c] = o.SqrtInverse(len);
        }
    }
    for (int k = 0; k < 4; ++k) {
        r.c0[k] = o.Mul(q[k], inv[0]);
        r.c1[k] = o.Mul(q[4 + k], inv[1]);
        r.c2[k] = o.Mul(q[8 + k], inv[2]);
    }
    // Determinant c0 . (c1 x c2); a negative one flips column 0.
    {
        const float f1 = o.Mul(r.c1[0], r.c2[2]);
        const float f6 = o.Mul(r.c1[2], r.c2[1]);
        const float f0 = o.Mul(r.c1[1], r.c2[0]);
        r.cross[1] = o.Msub(r.c1[2], r.c2[0], f1);
        r.cross[0] = o.Msub(r.c1[1], r.c2[2], f6);
        r.cross[2] = o.Msub(r.c1[0], r.c2[1], f0);
        const float f2 = o.Mul(r.c0[1], r.cross[1]);
        const float d = o.Madd(r.c0[2], r.cross[2], o.Madd(r.c0[0], r.cross[0], f2));
        if (d < 0.0f)
            for (int k = 0; k < 4; ++k) r.c0[k] = o.Mul(r.c0[k], -1.0f);
        uint32_t cr = 0;
        SetCRFloatResident(cr, 0, static_cast<double>(d), 0.0);
        r.det0 = cr >> 28;
        r.flips = d < 0.0f ? 1 : 0;
    }
    // hkQuaternion::set(hkRotation) (0x802856E4) on the rotation (columns c0, c1, c2).
    float rot[12];
    std::memcpy(rot, r.c0, 16);
    std::memcpy(rot + 4, r.c1, 16);
    std::memcpy(rot + 8, r.c2, 16);
    {
        const float trace = Ops::Add(rot[10], Ops::Add(rot[0], rot[5]));
        if (trace > 0.0f) {
            const float s1 = Ops::Add(1.0f, trace);
            if (!(s1 > 0.0f)) return false;
            const float root = Ops::Div(1.0f, o.SqrtInverse(s1));
            const float half = Ops::Div(0.5f, root);
            r.quat[1] = o.Mul(half, Ops::Sub(rot[8], rot[2]));
            r.quat[0] = o.Mul(half, Ops::Sub(rot[6], rot[9]));
            r.quat[2] = o.Mul(half, Ops::Sub(rot[1], rot[4]));
            r.quat[3] = o.Mul(0.5f, root);
        } else {
            int i = rot[5] > rot[0] ? 1 : 0;
            if (rot[10] > rot[5 * i]) i = 2;
            const int j = static_cast<int>(kQuatNextValues[i]);
            const int k = static_cast<int>(kQuatNextValues[j]);
            const float s1 = Ops::Add(1.0f, Ops::Sub(rot[5 * i], Ops::Add(rot[5 * j], rot[5 * k])));
            if (!(s1 > 0.0f)) return false;
            const float root = Ops::Div(1.0f, o.SqrtInverse(s1));
            const float half = Ops::Div(0.5f, root);
            const float wjk = Ops::Sub(rot[4 * j + k], rot[4 * k + j]);
            const float qi = o.Mul(0.5f, root);
            const float sij = Ops::Add(rot[4 * i + j], rot[4 * j + i]);
            const float sik = Ops::Add(rot[4 * i + k], rot[4 * k + i]);
            r.quat[3] = o.Mul(half, wjk);
            r.quat[i] = qi;
            r.quat[j] = o.Mul(half, sij);
            r.quat[k] = o.Mul(half, sik);
        }
    }
    // Normalized (D+0x10); its conjugate's rotation (0x80285A6C).
    {
        const float* const v = r.quat;
        const float len = o.Madd(v[3], v[3], o.Madd(v[2], v[2], o.Madd(v[0], v[0], o.Mul(v[1], v[1]))));
        float invLen;
        if (len == 0.0f) {
            invLen = 0.0f;
        } else if (!(len > 0.0f)) {
            return false;
        } else {
            invLen = o.SqrtInverse(len);
        }
        for (int k = 0; k < 4; ++k) r.qn[k] = o.Mul(v[k], invLen);
        const float conj[4] = {-r.qn[0], -r.qn[1], -r.qn[2], r.qn[3]};
        QuatToRotation(o, conj, r.rinv);
    }
    // scaleAndSkew = setMul(rinv, basis); scale = its diagonal; hasScale; diag(1/scale); skew.
    SetMul(o, r.s, r.rinv, r.t);
    r.absd[0] = std::fabs(Ops::Sub(r.s[0], 1.0f));
    r.absd[1] = std::fabs(Ops::Sub(r.s[5], 1.0f));
    r.absd[2] = std::fabs(Ops::Sub(r.s[10], 1.0f));
    r.hasScale = (kEps2 < r.absd[0] || kEps2 < r.absd[1] || kEps2 < r.absd[2]) ? 1 : 0;
    std::memset(r.diag, 0, sizeof(r.diag));
    r.diag[0] = Ops::Div(1.0f, r.s[0]);
    r.diag[5] = Ops::Div(1.0f, r.s[5]);
    r.diag[10] = Ops::Div(1.0f, r.s[10]);
    SetMul(o, r.skew, r.diag, r.s);
    static constexpr float kIdentity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    r.final = ApproxEqual(r.skew, kIdentity, kIdentity3, kEps2);
    return true;
}

// The original's stores, final values only (see the frames above), and its registers.
void WriteQs(CpuContext* ctx, const QsResult& r, uint32_t out, uint32_t R0, uint32_t s0, uint32_t s1) {
    const Image im{Window(), BrawlDirtyMemory::leanBlocks};
    const uint32_t D = s0 + 0x10u;
    // QsFromMatrix's frame and the caller's LR slot.
    im.W32(s0, R0);
    im.W32(R0 + 4u, ctx->lr);
    im.W32(R0 - 4u, 0u);
    // decomposeMatrix's prologue.
    im.W32(s1, s0);
    im.W32(s0 + 4u, 0x80284F58u);
    for (uint32_t k = 0; k < 5; ++k) im.W32(s0 - 20u + 4u * k, ctx->gpr[27 + k]);
    // Its temporaries.
    im.W32(s1 + 4u, 0x80282334u);
    static constexpr float kZero4[4] = {0, 0, 0, 0}, kOne4[4] = {1, 1, 1, 1}, kEps24[4] = {kEps2, kEps2, kEps2, kEps2};
    im.WFs(s1 + 0x20u, kZero4, 4);
    im.WFs(s1 + 0x30u, r.absd, 3);
    im.WF(s1 + 0x3Cu, 1.0f);
    im.WFs(s1 + 0x40u, kEps24, 4);
    im.WFs(s1 + 0x50u, r.quat, 4);
    im.WFs(s1 + 0x60u, kOne4, 4);
    const float conj[4] = {-r.qn[0], -r.qn[1], -r.qn[2], r.qn[3]};
    im.WFs(s1 + 0x70u, conj, 4);
    im.WFs(s1 + 0x80u, r.cross, 3);
    im.WF(s1 + 0x8Cu, 0.0f);
    im.WFs(s1 + 0x90u, r.c2, 4);
    im.WFs(s1 + 0xA0u, r.c1, 4);
    im.WFs(s1 + 0xB0u, r.c0, 4);
    im.WFs(s1 + 0xC0u, r.diag, 12);
    im.WFs(s1 + 0xF0u, r.rinv, 12);
    im.WFs(s1 + 0x120u, r.c0, 4);
    im.WFs(s1 + 0x130u, r.c1, 4);
    im.WFs(s1 + 0x140u, r.c2, 4);
    im.WFs(s1 + 0x150u, r.acc, 12);
    im.WFs(s1 + 0x180u, r.w, 12);
    im.WFs(s1 + 0x1B0u, r.q, 12);
    im.WFs(s1 + 0x1E0u, r.qPrev, 12);
    im.WFs(s1 + 0x210u, r.t, 16);
    // The callees' area: the second setMul's frame, then the final isApproximatelyEqual's.
    im.W32(s1 - 0x120u, s1);
    im.WFs(s1 - 0x110u, r.diag, 12);
    im.ZeroSpan(s1 - 0xE0u, 0xE0u);
    im.W32(s1 - 0x70u, s1);
    static constexpr uint32_t kRowAt[3] = {0x50u, 0x30u, 0x10u};
    for (int row = 0; row < r.final.rows; ++row) {
        const uint32_t at = s1 - 0x70u + kRowAt[row];
        im.WFs(at, r.final.d[row], 4);
        im.WFs(at + 0x10u, kEps24, 4);
    }
    // The Decomposition.
    im.WFs(D, r.t + 12, 4);
    im.WFs(D + 0x10u, r.qn, 4);
    const float scale[4] = {r.s[0], r.s[5], r.s[10], 0.0f};
    im.WFs(D + 0x20u, scale, 4);
    im.W8(D + 0x30u, r.hasScale);
    im.W8(D + 0x31u, r.final.equal ? 0 : 1);
    im.W8(D + 0x32u, r.flips);
    im.WFs(D + 0x40u, r.s, 12);
    im.WFs(D + 0x70u, r.skew, 12);
    im.WFs(D + 0xA0u, r.q, 12);
    // The hkQsTransform.
    im.WFs(out, r.t + 12, 4);
    im.WFs(out + 0x10u, r.qn, 4);
    im.WFs(out + 0x20u, scale, 4);
    // Registers: r3 = hkBool(!hasSkew) << 24; r4, f2 as the final isApproximatelyEqual left them; f1 its eps; cr0 of
    // the determinant's compare (decomposeMatrix's context write before hkQuaternion::set); CA of the subfics.
    ctx->gpr[3] = r.final.equal ? 0x01000000u : 0u;
    ctx->gpr[4] = r.final.r4;
    ctx->fpr[1].d = static_cast<double>(kEps2);
    ctx->fpr[2].d = static_cast<double>(r.final.f2);
    ctx->crv[0] = PpcCrvEncode(r.det0 & 0xFu);
    ctx->xer |= 0x20000000u;
}

bool QsFromMatrixNative(CpuContext* ctx) {
    // NI on (Brawl's OSInit sets it): the host flushes, and flags, every tiny result; round to nearest.
    if (!EabiTranslation() || ctx->gqr[0] != 0 || !LeanReady() || !MkwHostNiActiveInline() || (ctx->fpscr & 3u) != 0)
        return false;
    const uint32_t out = ctx->gpr[3], mtx = ctx->gpr[4], R0 = ctx->gpr[1], r2 = ctx->gpr[2];
    const uint32_t s0 = (R0 & ~15u) - 0xF0u, s1 = s0 - 0x270u;
    const uint32_t low = R0 - kQsWindowBelow, windowSize = kQsWindowBelow + 8u;
    if (!PlainRange(low, windowSize) || !PlainRange(out, 0x30u) || !OneView(mtx, 0x3Cu) ||
        Overlap(out, 0x30u, low, windowSize) || Overlap(mtx, 0x3Cu, low, windowSize) ||
        !OneView(r2 - kSdata2A, kSdata2A - kSdata2C + 8u) || !OneView(kQuatNext, 12u) || !OneView(kIdentity3, 48u))
        return false;
    FpFlagsClear();
    // The constants the original loads (patched data runs the original) and the matrix: normal singles or zeros.
    if (!SameWords(r2 - kSdata2A, kConstA, 10) || !SameWords(r2 - kSdata2B, kConstB, 6) ||
        !SameWords(r2 - kSdata2C, kConstC, 2) || !SameWords(kQuatNext, kQuatNextValues, 3) ||
        !SameWords(kIdentity3, kIdentity3Values, 12))
        return false;
    const uint8_t* const pm = Window() + Canonical(mtx);
    float m[16] = {};
    uint32_t plain = 1;
    for (int c = 0; c < 4; ++c)
        for (int k = 0; k < 3; ++k) {
            const uint32_t bits = Load32(pm + 16 * c + 4 * k);
            plain &= PlainSingle(bits) ? 1u : 0u;
            m[4 * c + k] = AsFloat(bits);
        }
    if (!plain) return false;
    Ops o;
    QsResult r;
    if (!Decompose(o, r, m)) return false;
    if (FpFlagsRaised(&r) || o.Gap()) {
        Bump(g_counters[kQsFromMatrix].fallback);
        return false;
    }
    WriteQs(ctx, r, out, R0, s0, s1);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// QsSetMul (0x8028526C): hkQsTransform::setMul(r3 = out, r4 = a, r5 = b): out.t = a.t + a.q.rotate(b.t) (the
// 2 * (dot * q + (w^2 - 1/2) * v + w * (q x v)) form, with the w lane carried), out.q = a.q * b.q, out.s = a.s * b.s.
// hkaPose::syncModelSpace (0x8032EB5C) composes each bone's model transform with it; the IK and look-at solvers call it
// too. A leaf: frame s = (R & ~15) - 0x1B0 (R = entry r1): back chain R; f14-f31 saved at [R - 0x120, R) (zeros: every
// caller reached it through its EABI entry); temporaries at s+0x10..0x7C and the double at s+0x80 (an stfd/lfd of a
// product). Its loads of a.s, b.q.w and b.s follow its stores of out.t, which is harmless when out is a or b exactly
// (those offsets lie past out.t); any other overlap of out with an input, or an input inside the frame, declines.
// Leaves f1 = a.s.w (its last load) and f2 = a.s.z * b.s.z.
// ---------------------------------------------------------------------------------------------------------------
constexpr uint32_t kSetMulHalf = 0x8040F870u;  // 4 x -0.5
constexpr uint32_t kSetMulHalfValues[4] = {0xBF000000u, 0xBF000000u, 0xBF000000u, 0xBF000000u};
constexpr uint32_t kSetMulZero = 0x55F8u;      // r2 - 0x55F8: 0.0
constexpr uint32_t kSetMulSaves = 0x120u;      // f14-f31, two words each
constexpr uint32_t kSetMulWindowBelow = 0x1C0u;  // R - 15 - 0x1B0 rounded up

// setMul's values: the output, its frame's 28 temporaries (s+0x10..0x7C in order) and the product it spills as a
// double at s+0x80 (a.q.x * b.q.z).
struct SetMulValues {
    float out[12], tmp[28], prod;
};
void SetMulCompute(Ops& o, const float* a, const float* b, SetMulValues& v);

// Load 12 words of a QsTransform; false when one is not a normal single or zero.
bool LoadPlain12(uint32_t address, float* out) {
    const uint8_t* const h = Window() + Canonical(address);
    uint32_t plain = 1;
    for (int i = 0; i < 12; ++i) {
        const uint32_t x = Load32(h + 4 * i);
        plain &= PlainSingle(x) ? 1u : 0u;
        out[i] = AsFloat(x);
    }
    return plain != 0;
}

bool SetMulConstantsOk(uint32_t r2) {
    static constexpr uint32_t kZeroWord[1] = {0u};
    return OneView(r2 - kSetMulZero, 4u) && OneView(kSetMulHalf, 16u) && SameWords(kSetMulHalf, kSetMulHalfValues, 4) &&
           SameWords(r2 - kSetMulZero, kZeroWord, 1);
}

// setMul's frame image for entry r1 R: back chain, the EABI-zero FPR saves, its temporaries and spilled double.
void WriteSetMulFrame(const Image& im, uint32_t R, const SetMulValues& v) {
    const uint32_t s = (R & ~15u) - 0x1B0u;
    im.W32(s, R);
    im.ZeroSpan(R - kSetMulSaves, kSetMulSaves);
    im.WFs(s + 0x10u, v.tmp, 28);
    const uint64_t prod = __builtin_bit_cast(uint64_t, static_cast<double>(v.prod));
    im.W32(s + 0x80u, static_cast<uint32_t>(prod >> 32));
    im.W32(s + 0x84u, static_cast<uint32_t>(prod));
}

bool QsSetMulNative(CpuContext* ctx) {
    if (!EabiTranslation() || ctx->gqr[0] != 0 || !LeanReady() || !MkwHostNiActiveInline() || (ctx->fpscr & 3u) != 0)
        return false;
    const uint32_t out = ctx->gpr[3], pa = ctx->gpr[4], pb = ctx->gpr[5], R = ctx->gpr[1];
    const uint32_t low = R - kSetMulWindowBelow;
    if (!PlainRange(low, kSetMulWindowBelow) || !PlainRange(out, 0x30u) || !OneView(pa, 0x30u) || !OneView(pb, 0x30u) ||
        Overlap(out, 0x30u, low, kSetMulWindowBelow) || Overlap(pa, 0x30u, low, kSetMulWindowBelow) ||
        Overlap(pb, 0x30u, low, kSetMulWindowBelow) || (out != pa && Overlap(out, 0x30u, pa, 0x30u)) ||
        (out != pb && Overlap(out, 0x30u, pb, 0x30u)))
        return false;
    FpFlagsClear();
    struct {
        float a[12], b[12];
        SetMulValues v;
    } w;
    if (!SetMulConstantsOk(ctx->gpr[2]) || !LoadPlain12(pa, w.a) || !LoadPlain12(pb, w.b)) return false;
    Ops o;
    SetMulCompute(o, w.a, w.b, w.v);
    if (FpFlagsRaised(&w) || o.Gap()) {
        Bump(g_counters[kQsSetMul].fallback);
        return false;
    }
    const Image im{Window(), BrawlDirtyMemory::leanBlocks};
    WriteSetMulFrame(im, R, w.v);
    im.WFs(out, w.v.out, 12);
    ctx->fpr[1].d = static_cast<double>(w.a[11]);
    ctx->fpr[2].d = static_cast<double>(w.v.out[10]);
    return true;
}

// setMul in the original's order (register names in the comments).
void SetMulCompute(Ops& o, const float* a, const float* b, SetMulValues& v) {
    constexpr float K = -0.5f;
    // In the original's order (register names in the comments).
    float f1 = o.Mul(a[5], b[1]);
    const float f28 = o.Madd(a[7], a[7], K), f29 = o.Madd(a[7], a[7], K), f30 = o.Madd(a[7], a[7], K),
                f31 = o.Madd(a[7], a[7], K);
    f1 = o.Madd(a[4], b[0], f1);
    const float f16a = o.Mul(a[4], b[2]);
    const float f15a = o.Mul(a[6], b[1]);
    const float dot = o.Madd(a[6], b[2], f1);                  // f8
    const float f23 = o.Msub(a[6], b[0], f16a);
    const float f24 = o.Msub(a[5], b[2], f15a);
    float f3 = o.Mul(b[0], f28);
    float f5 = o.Mul(b[1], f29);
    float f4 = o.Mul(b[2], f30);
    float f2 = o.Mul(b[3], f31);
    const float f9a = o.Mul(a[5], b[0]);
    f3 = o.Madd(dot, a[4], f3);
    f2 = o.Madd(dot, a[7], f2);
    const float f22 = o.Msub(a[4], b[1], f9a);
    f4 = o.Madd(dot, a[6], f4);
    const float r3w = o.Madd(0.0f, a[7], f2);                 // f6
    f5 = o.Madd(dot, a[5], f5);
    const float r0x = o.Madd(f24, a[7], f3);                  // f9
    const float r2z = o.Madd(f22, a[7], f4);                  // f7
    const float r1y = o.Madd(f23, a[7], f5);                  // f8
    const float d0 = Ops::Add(r0x, r0x), d2 = Ops::Add(r2z, r2z), d1 = Ops::Add(r1y, r1y);
    v.prod = o.Mul(a[4], b[6]);
    v.out[0] = Ops::Add(a[0], d0);
    v.out[1] = Ops::Add(a[1], d1);
    const float f18a = o.Mul(a[6], b[5]);
    const float d3 = Ops::Add(r3w, r3w);
    v.out[2] = Ops::Add(a[2], d2);
    const float f14 = o.Msub(a[5], b[6], f18a);
    v.out[3] = Ops::Add(a[3], d3);
    const float f16 = o.Msub(a[6], b[4], v.prod);
    // (out.t is stored here in the original; the loads below read offsets past it.)
    float g1 = o.Mul(a[5], b[4]);
    float g0 = o.Mul(a[5], b[5]);
    const float f18 = o.Madd(a[7], b[4], f14);
    g1 = o.Msub(a[4], b[5], g1);
    g0 = o.Madd(a[4], b[4], g0);
    float g9 = o.Madd(a[7], b[5], f16);
    float g8 = o.Madd(a[7], b[6], g1);
    g9 = o.Madd(b[7], a[5], g9);
    const float g16 = o.Madd(a[6], b[6], g0);
    g8 = o.Madd(b[7], a[6], g8);
    const float g14 = o.Madd(a[7], b[7], 0.0f);
    const float g10 = o.Madd(b[7], a[4], f18);
    v.out[10] = o.Mul(a[10], b[10]);
    v.out[7] = o.Msub(a[7], b[7], g16);
    v.out[8] = o.Mul(a[8], b[8]);
    v.out[9] = o.Mul(a[9], b[9]);
    v.out[11] = o.Mul(a[11], b[11]);
    const float g3 = o.Madd(b[7], a[7], g14);
    v.out[4] = g10;
    v.out[5] = g9;
    v.out[6] = g8;
    // Temporaries s+0x10..0x7C.
    float* const t = v.tmp;
    t[0] = g10, t[1] = g9, t[2] = g8, t[3] = g3;                  // 0x10
    t[4] = t[5] = t[6] = t[7] = K;                                // 0x20
    t[8] = t[9] = t[10] = t[11] = a[7];                           // 0x30
    t[12] = f28, t[13] = f29, t[14] = f30, t[15] = f31;           // 0x40
    t[16] = r0x, t[17] = r1y, t[18] = r2z, t[19] = r3w;           // 0x50
    t[20] = f24, t[21] = f23, t[22] = f22, t[23] = 0.0f;          // 0x60
    t[24] = d0, t[25] = d1, t[26] = d2, t[27] = d3;               // 0x70
}

// ---------------------------------------------------------------------------------------------------------------
// QsSetInverse (0x80284FEC): hkQsTransform::setInverse(r3 = out, r4 = in): t' = -2 * rotate(conj q, t) (setMul's form
// with the w lane carried), q' = conj(q), s' = (1/sx, 1/sy, 1/sz, 0) (a zero sw makes its spilled reciprocal 1). Leaf;
// frame s = (R & ~15) - 0xF0: back chain R, f24-f31 at [R - 0x80, R) (zeros: EABI entry), temporaries s+0x10..0x6C.
// It stores out.q.w before it loads in.s, harmless when out is in exactly. Leaves f1 = 1/sz, f2 = 1/sy.
// QsMulInverseMul (0x802855F0): out = setMul(setInverse(a), b) through a QsTransform at s2+0x10 (s2 = (R & ~15) -
// 0x50; back chain R, LR at R+4, r30/r31 at R-8/R-4 from the context: no EABI entry). setMul's frame image, at the same
// r1, covers setInverse's. Leaves r3 = out, r4 = the temporary, f1/f2 setMul's, CA set.
// ---------------------------------------------------------------------------------------------------------------
constexpr uint32_t kInvSaves = 0x80u;
constexpr uint32_t kInvWindowBelow = 0x100u;  // R - 15 - 0xF0 rounded up
constexpr uint32_t kMimWindowBelow = 0x210u;  // R - 15 - 0x50 - 15 - 0x1B0 rounded up

struct SetInverseValues {
    float out[12], tmp[24];  // tmp: s+0x10..0x6C in order
};
bool SetInverseConstantsOk(uint32_t r2) {
    static constexpr uint32_t kRun[3] = {0u, 0xBF800000u, 0x3F800000u};  // r2 - 0x55F8: 0, -1, 1
    return OneView(r2 - kSetMulZero, 12u) && OneView(kSetMulHalf, 16u) &&
           SameWords(kSetMulHalf, kSetMulHalfValues, 4) && SameWords(r2 - kSetMulZero, kRun, 3);
}
void SetInverseCompute(Ops& o, const float* in, SetInverseValues& v) {
    constexpr float K = -0.5f;
    const float tx = in[0], ty = in[1], tz = in[2], tw = in[3], qx = in[4], qy = in[5], qz = in[6], qw = in[7];
    const float f4a = o.Mul(qy, ty);
    const float f26 = o.Madd(qw, qw, K), f29 = o.Madd(qw, qw, K), f12a = o.Madd(qw, qw, K), f10a = o.Madd(qw, qw, K);
    const float f4b = o.Madd(qx, tx, f4a);
    const float f9a = o.Mul(tx, f26);
    const float dot = o.Madd(qz, tz, f4b);
    const float f8a = o.Mul(ty, f29);
    const float f7a = o.Mul(tz, f12a);
    const float f6a = o.Mul(tw, f10a);
    const float f5b = o.Madd(dot, qx, f9a);
    const float f4c = o.Mul(tz, qy);
    const float f30a = o.Madd(dot, qy, f8a);
    const float f28 = o.Madd(dot, qz, f7a);
    const float f13b = o.Madd(dot, qw, f6a);
    const float f24 = o.Msub(ty, qz, f4c);
    const float f12b = o.Mul(tx, qz);
    const float f11b = o.Mul(ty, qx);
    const float f10b = o.Madd(f24, qw, f5b);
    const float f12c = o.Msub(tz, qx, f12b);
    const float f11c = o.Msub(tx, qy, f11b);
    const float f31b = o.Madd(0.0f, qw, f13b);
    const float f30b = o.Madd(f12c, qw, f30a);
    const float isx = Ops::Div(1.0f, in[8]);
    const float f3b = o.Madd(f11c, qw, f28);
    const float d0 = Ops::Add(f10b, f10b);
    const float isy = Ops::Div(1.0f, in[9]);
    const float d2 = Ops::Add(f3b, f3b);
    const float isz = Ops::Div(1.0f, in[10]);
    const float d1 = Ops::Add(f30b, f30b);
    const float d3 = Ops::Add(f31b, f31b);
    v.out[0] = o.Mul(d0, -1.0f);
    v.out[1] = o.Mul(d1, -1.0f);
    v.out[2] = o.Mul(d2, -1.0f);
    v.out[3] = o.Mul(d3, -1.0f);
    v.out[4] = -qx, v.out[5] = -qy, v.out[6] = -qz, v.out[7] = qw;
    v.out[8] = isx, v.out[9] = isy, v.out[10] = isz, v.out[11] = 0.0f;
    // A zero sw spills 1.0 (the original skips the division): divide by 1 instead, through Opaque so the compiler
    // cannot turn 1/(c ? 1 : sw) back into c ? 1 : 1/sw and speculate a 1/0, whose divide-by-zero flag would send every
    // call to the original.
    const float isw = Ops::Div(1.0f, Opaque(in[11] == 0.0f ? 1.0f : in[11]));
    float* const t = v.tmp;
    t[0] = t[1] = t[2] = t[3] = K;                             // 0x10
    t[4] = t[5] = t[6] = t[7] = qw;                            // 0x20
    t[8] = f26, t[9] = f29, t[10] = f12a, t[11] = f10a;        // 0x30
    t[12] = f10b, t[13] = f30b, t[14] = f3b, t[15] = f31b;     // 0x40
    t[16] = f24, t[17] = f12c, t[18] = f11c, t[19] = 0.0f;     // 0x50
    t[20] = isx, t[21] = isy, t[22] = isz, t[23] = isw;        // 0x60
}
void WriteSetInverseFrame(const Image& im, uint32_t R, const SetInverseValues& v) {
    const uint32_t s = (R & ~15u) - 0xF0u;
    im.W32(s, R);
    im.ZeroSpan(R - kInvSaves, kInvSaves);
    im.WFs(s + 0x10u, v.tmp, 24);
}

bool QsSetInverseNative(CpuContext* ctx) {
    if (!EabiTranslation() || ctx->gqr[0] != 0 || !LeanReady() || !MkwHostNiActiveInline() || (ctx->fpscr & 3u) != 0)
        return false;
    const uint32_t out = ctx->gpr[3], pin = ctx->gpr[4], R = ctx->gpr[1];
    const uint32_t low = R - kInvWindowBelow;
    if (!PlainRange(low, kInvWindowBelow) || !PlainRange(out, 0x30u) || !OneView(pin, 0x30u) ||
        Overlap(out, 0x30u, low, kInvWindowBelow) || Overlap(pin, 0x30u, low, kInvWindowBelow) ||
        (out != pin && Overlap(out, 0x30u, pin, 0x30u)))
        return false;
    FpFlagsClear();
    struct {
        float in[12];
        SetInverseValues v;
    } w;
    if (!SetInverseConstantsOk(ctx->gpr[2]) || !LoadPlain12(pin, w.in)) return false;
    Ops o;
    SetInverseCompute(o, w.in, w.v);
    if (FpFlagsRaised(&w) || o.Gap()) {
        Bump(g_counters[kQsSetInverse].fallback);
        return false;
    }
    const Image im{Window(), BrawlDirtyMemory::leanBlocks};
    WriteSetInverseFrame(im, R, w.v);
    im.WFs(out, w.v.out, 12);
    ctx->fpr[1].d = static_cast<double>(w.v.out[10]);
    ctx->fpr[2].d = static_cast<double>(w.v.out[9]);
    return true;
}

bool QsSetInverseWrites(const CpuContext& ctx, Ranges& ranges) {
    return ranges.Add(ctx.gpr[1] - kInvWindowBelow, kInvWindowBelow) && ranges.Add(ctx.gpr[3], 0x30u);
}

bool QsMulInverseMulNative(CpuContext* ctx) {
    if (!EabiTranslation() || ctx->gqr[0] != 0 || !LeanReady() || !MkwHostNiActiveInline() || (ctx->fpscr & 3u) != 0)
        return false;
    const uint32_t out = ctx->gpr[3], pa = ctx->gpr[4], pb = ctx->gpr[5], R = ctx->gpr[1];
    const uint32_t s2 = (R & ~15u) - 0x50u, low = R - kMimWindowBelow, windowSize = kMimWindowBelow + 8u;
    if (!PlainRange(low, windowSize) || !PlainRange(out, 0x30u) || !OneView(pa, 0x30u) || !OneView(pb, 0x30u) ||
        Overlap(out, 0x30u, low, windowSize) || Overlap(pa, 0x30u, low, windowSize) ||
        Overlap(pb, 0x30u, low, windowSize) || (out != pb && Overlap(out, 0x30u, pb, 0x30u)))
        return false;
    FpFlagsClear();
    struct {
        float a[12], b[12];
        SetInverseValues inv;
        SetMulValues mul;
    } w;
    if (!SetInverseConstantsOk(ctx->gpr[2]) || !LoadPlain12(pa, w.a) || !LoadPlain12(pb, w.b)) return false;
    Ops o;
    SetInverseCompute(o, w.a, w.inv);
    SetMulCompute(o, w.inv.out, w.b, w.mul);
    if (FpFlagsRaised(&w) || o.Gap()) {
        Bump(g_counters[kQsMulInverseMul].fallback);
        return false;
    }
    const Image im{Window(), BrawlDirtyMemory::leanBlocks};
    im.W32(s2, R);
    im.W32(R + 4u, ctx->lr);
    im.W32(R - 8u, ctx->gpr[30]);
    im.W32(R - 4u, ctx->gpr[31]);
    im.WFs(s2 + 0x10u, w.inv.out, 12);
    WriteSetMulFrame(im, s2, w.mul);
    im.WFs(out, w.mul.out, 12);
    ctx->gpr[3] = out;
    ctx->gpr[4] = s2 + 0x10u;
    ctx->fpr[1].d = static_cast<double>(w.inv.out[11]);
    ctx->fpr[2].d = static_cast<double>(w.mul.out[10]);
    ctx->xer |= 0x20000000u;
    return true;
}

bool QsMulInverseMulWrites(const CpuContext& ctx, Ranges& ranges) {
    return ranges.Add(ctx.gpr[1] - kMimWindowBelow, kMimWindowBelow + 8u) && ranges.Add(ctx.gpr[3], 0x30u);
}

// ---------------------------------------------------------------------------------------------------------------
// MtScale (0x8003E6DC): Brawl mt library, an MTX34's scale (r3 = matrix, r4 = out Vec3): each column's squared length
// (m0^2 + m4^2 first, then + m8^2; likewise columns 1 and 2), OSReport on a negative one (never: the native declines
// it), and mtSqrtf (0x8003DB08, inlined: 0 for x <= 0, else x * the estimate after two Newton steps of frsqrte) of
// each.
// mtSqrtf's first two products take the raw frsqrte estimate, which the translation rounds with the exact helper
// (PpcFmulsInline: C to 25 bits); the native calls it there. No EABI entry: frame s = R - 0x40 (back chain R, LR at
// R+4, f29-f31 saved from the context with stfd and psq_st at s+0x10..0x3F, r30/r31 at s+8/s+0xC). Leaves f1 = the
// last root, f2 = the last Newton estimate (the previous column's when this one is 0; the entry f2 when all are).
// ---------------------------------------------------------------------------------------------------------------
constexpr uint32_t kMtZero = 0x7BF8u;   // r2 - 0x7BF8: 0.0 (MtScale)
constexpr uint32_t kMtSqrtK = 0x7C28u;  // r2 - 0x7C28: 0.0, 0.5, 3.0 (mtSqrtf)

bool MtScaleNative(CpuContext* ctx) {
    if (!EabiTranslation() || ctx->gqr[0] != 0 || !LeanReady() || !MkwHostNiActiveInline() || (ctx->fpscr & 3u) != 0)
        return false;
    const uint32_t pm = ctx->gpr[3], out = ctx->gpr[4], R = ctx->gpr[1], r2 = ctx->gpr[2], s = R - 0x40u;
    if (!PlainRange(s, 0x48u) || !PlainRange(out, 12u) || !OneView(pm, 0x2Cu) || Overlap(out, 12u, s, 0x48u) ||
        Overlap(pm, 0x2Cu, s, 0x48u) || !OneView(r2 - kMtSqrtK, 12u) || !OneView(r2 - kMtZero, 4u))
        return false;
    FpFlagsClear();
    static constexpr uint32_t kSqrtK[3] = {0u, 0x3F000000u, 0x40400000u};
    static constexpr uint32_t kZeroWord[1] = {0u};
    if (!SameWords(r2 - kMtSqrtK, kSqrtK, 3) || !SameWords(r2 - kMtZero, kZeroWord, 1)) return false;
    struct {
        float m[12], root[3], f2[3];
        bool positive[3];
    } v;
    if (!LoadPlain12(pm, v.m)) return false;
    Ops o;
    const float* const m = v.m;
    const float a0 = o.Mul(m[0], m[0]), a4 = o.Mul(m[4], m[4]);
    const float c0 = Ops::Add(a0, a4);
    const float a8 = o.Mul(m[8], m[8]), a1 = o.Mul(m[1], m[1]), a5 = o.Mul(m[5], m[5]);
    const float a6 = o.Mul(m[6], m[6]), a2 = o.Mul(m[2], m[2]);
    const float x = Ops::Add(a8, c0);
    const float a9 = o.Mul(m[9], m[9]);
    const float c1 = Ops::Add(a1, a5);
    const float a10 = o.Mul(m[10], m[10]);
    const float c2 = Ops::Add(a2, a6);
    const float y = Ops::Add(a9, c1), z = Ops::Add(a10, c2);
    if (x < 0.0f || y < 0.0f || z < 0.0f) return false;  // OSReport
    const float sq[3] = {x, y, z};
    for (int k = 0; k < 3; ++k) {
        const float in = sq[k];
        v.positive[k] = in > 0.0f;
        if (!v.positive[k]) {  // (and NaN: the original's ble)
            v.root[k] = 0.0f;
            continue;
        }
        const double est = PPC_Frsqrte(static_cast<double>(in));
        const float e2 = static_cast<float>(PpcFmulsInline(est, est));
        float h = static_cast<float>(PpcFmulsInline(0.5, est));
        float t = Ops::Sub(3.0f, o.Mul(in, e2));
        h = o.Mul(h, t);
        const float h2 = o.Mul(h, h);
        h = o.Mul(0.5f, h);
        t = Ops::Sub(3.0f, o.Mul(in, h2));
        h = o.Mul(h, t);
        v.f2[k] = h;
        v.root[k] = o.Mul(h, in);
    }
    if (FpFlagsRaised(&v) || o.Gap()) {
        Bump(g_counters[kMtScale].fallback);
        return false;
    }
    uint8_t* const w = Window();
    const Image im{w, BrawlDirtyMemory::leanBlocks};
    im.W32(s, R);
    im.W32(R + 4u, ctx->lr);
    uint8_t* const frame = w + Canonical(s);
    for (uint32_t k = 0; k < 3; ++k) {  // f31 at 0x30/0x38, f30 at 0x20/0x28, f29 at 0x10/0x18
        const PPC_FPR f = ctx->fpr[31 - k];
        const uint32_t at = 0x30u - 0x10u * k;
        im.W32(s + at, static_cast<uint32_t>(f.raw >> 32));
        im.W32(s + at + 4u, static_cast<uint32_t>(f.raw));
        PPC_PsqStSegmentInline<0u, 0u>(ctx, frame, at + 8u, s + at + 8u, PPC_PsFromScalarInline(f.d));
        im.Mark(s + at + 8u);
    }
    im.W32(s + 0xCu, ctx->gpr[31]);
    im.W32(s + 8u, ctx->gpr[30]);
    im.WFs(out, v.root, 3);
    ctx->fpr[1].d = static_cast<double>(v.root[2]);
    for (int k = 2; k >= 0; --k)
        if (v.positive[k]) {
            ctx->fpr[2].d = static_cast<double>(v.f2[k]);
            break;
        }
    return true;
}

bool MtScaleWrites(const CpuContext& ctx, Ranges& ranges) {
    return ranges.Add(ctx.gpr[1] - 0x40u, 0x48u) && ranges.Add(ctx.gpr[4], 12u);
}

// ---------------------------------------------------------------------------------------------------------------
// SyncModelSpace (0x8032EB5C): hkaPose::syncModelSpace(r3 = pose). Unless modelInSync (pose+0x28), for each bone i of
// the skeleton (pose+0: hkaSkeleton, +4 parent indices s16, +0x10 count) whose flag (pose+0x1C bytes) has bit 2:
// model[i] (pose+0x10, 0x30 each) = setMul(model[parent], local[i]) (pose+4), or local[i] for a root (parent -1);
// flag &= ~2; then modelInSync = 1. Frame: stwu -0x20 (back chain R), LR at R+4, r27-r31 at R-20..R-4 (the context's,
// as its context body saves them: no caller has an EABI entry to it). Each setMul call's frame sits at the same r1, so
// the image left is the last call's. The bones run in order on host copies (a parent processed earlier in the call is
// read from them, any other from memory, as the original's reads see it), then everything is written. Leaves r3 = the
// flag array (the skeleton when it has no bones; the pose when in sync), r4 = the last setMul's a (0 without one;
// extsb(modelInSync) when in sync), r5 = the last setMul's b (unchanged without one), f1/f2 the last setMul's
// (unchanged without one).
// ---------------------------------------------------------------------------------------------------------------
constexpr uint32_t kSyncMaxBones = 512;
constexpr uint32_t kSyncWindowBelow = 0x1E0u;  // R - 0x20 - 15 - 0x1B0 rounded up

struct SyncLayout {
    uint32_t pose, skel, count, parents, local, model, flags;
};
// The pose's arrays; false: decline (ranges the native does not handle, or an overlap that orders its stores).
bool SyncReadLayout(uint32_t pose, SyncLayout& l) {
    if (!OneView(pose, 0x2Cu)) return false;
    const uint8_t* const p = Window() + Canonical(pose);
    l.pose = pose;
    l.skel = Load32(p);
    l.local = Load32(p + 4);
    l.model = Load32(p + 0x10);
    l.flags = Load32(p + 0x1C);
    if (!OneView(l.skel, 0x14u)) return false;
    const uint8_t* const sk = Window() + Canonical(l.skel);
    l.parents = Load32(sk + 4);
    const int32_t n = static_cast<int32_t>(Load32(sk + 0x10));
    l.count = n <= 0 ? 0u : static_cast<uint32_t>(n);
    return l.count <= kSyncMaxBones;
}

bool SyncModelSpaceNative(CpuContext* ctx) {
    if (!EabiTranslation() || ctx->gqr[0] != 0 || !LeanReady() || !MkwHostNiActiveInline() || (ctx->fpscr & 3u) != 0)
        return false;
    const uint32_t pose = ctx->gpr[3], R = ctx->gpr[1], s = R - 0x20u;
    const uint32_t low = R - kSyncWindowBelow, windowSize = kSyncWindowBelow + 8u;
    if (!PlainRange(low, windowSize) || !PlainRange(pose + 0x28u, 1u) || Overlap(pose, 0x2Cu, low, windowSize))
        return false;
    SyncLayout l;
    if (!SyncReadLayout(pose, l)) return false;
    uint8_t* const m = Window();
    const Image im{m, BrawlDirtyMemory::leanBlocks};
    const uint8_t inSync = m[Canonical(pose + 0x28u)];
    const auto frame = [&] {
        im.W32(s, R);
        im.W32(R + 4u, ctx->lr);
        for (uint32_t k = 0; k < 5; ++k) im.W32(R - 20u + 4u * k, ctx->gpr[27 + k]);
    };
    if (inSync != 0) {
        frame();
        ctx->gpr[3] = pose;
        ctx->gpr[4] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(inSync)));
        return true;
    }
    const uint32_t n = l.count;
    if (n != 0) {
        const uint32_t mBytes = 0x30u * n;
        if (!PlainRange(l.model, mBytes) || !PlainRange(l.flags, n) || !OneView(l.local, mBytes) ||
            !OneView(l.parents, 2u * n) || Overlap(l.model, mBytes, low, windowSize) ||
            Overlap(l.flags, n, low, windowSize) ||
            Overlap(l.model, mBytes, l.flags, n) || Overlap(l.model, mBytes, l.local, mBytes) ||
            Overlap(l.model, mBytes, l.parents, 2u * n) || Overlap(l.model, mBytes, pose, 0x2Cu) ||
            Overlap(l.model, mBytes, l.skel, 0x14u) || Overlap(l.flags, n, l.local, mBytes) ||
            Overlap(l.flags, n, l.parents, 2u * n) || Overlap(l.flags, n, pose, 0x2Cu) ||
            Overlap(l.flags, n, l.skel, 0x14u))
            return false;
    }
    FpFlagsClear();
    if (!SetMulConstantsOk(ctx->gpr[2])) return false;
    // Host copies: model values computed in this call, which bones were processed, the last setMul's values.
    thread_local float t_model[kSyncMaxBones][12];
    thread_local uint8_t t_done[kSyncMaxBones];
    struct {
        float a[12], b[12];
        SetMulValues v;
        uint32_t aAddress, bAddress;
        bool any;
    } last{};
    std::memset(t_done, 0, n);
    const uint8_t* const flags = m + Canonical(l.flags);
    const uint8_t* const parents = m + Canonical(l.parents);
    Ops o;
    for (uint32_t i = 0; i < n; ++i) {
        if ((flags[i] & 2) == 0) continue;
        const int32_t parent = static_cast<int16_t>((uint16_t(parents[2 * i]) << 8) | parents[2 * i + 1]);
        float b[12];
        if (!LoadPlain12(l.local + 0x30u * i, b)) return false;
        if (parent == -1) {
            std::memcpy(t_model[i], b, sizeof(b));
        } else {
            if (parent < 0 || static_cast<uint32_t>(parent) >= n) return false;  // (not seen: outside the skeleton)
            const uint32_t aAddress = l.model + 0x30u * static_cast<uint32_t>(parent);
            if (t_done[parent])
                std::memcpy(last.a, t_model[parent], sizeof(last.a));
            else if (!LoadPlain12(aAddress, last.a))
                return false;
            std::memcpy(last.b, b, sizeof(b));
            SetMulCompute(o, last.a, last.b, last.v);
            std::memcpy(t_model[i], last.v.out, sizeof(last.v.out));
            last.aAddress = aAddress;
            last.bAddress = l.local + 0x30u * i;
            last.any = true;
        }
        t_done[i] = 1;
    }
    if (FpFlagsRaised(&last) || o.Gap()) {
        Bump(g_counters[kSyncModelSpace].fallback);
        return false;
    }
    frame();
    for (uint32_t i = 0; i < n; ++i) {
        if (!t_done[i]) continue;
        im.WFs(l.model + 0x30u * i, t_model[i], 12);
        im.W8(l.flags + i, static_cast<uint8_t>(flags[i] & ~2u));
    }
    if (last.any) WriteSetMulFrame(im, s, last.v);
    im.W8(pose + 0x28u, 1);
    ctx->gpr[3] = n != 0 ? l.flags : l.skel;
    ctx->gpr[4] = last.any ? last.aAddress : 0u;
    if (last.any) {
        // The original's call to setMul's wrapper (a native, so no EABI entry) stores its argument registers in the
        // context; r5 (b) is the one the original does not write back itself.
        ctx->gpr[5] = last.bAddress;
        ctx->fpr[1].d = static_cast<double>(last.a[11]);
        ctx->fpr[2].d = static_cast<double>(last.v.out[10]);
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// IeeeSqrt (0x80400B48): MSL/fdlibm __ieee754_sqrt (sqrt, 0x80400D94, tail-branches to it): the square root bit by bit
// in integer registers (~1k guest instructions a call), rounded to nearest even with the remainder as sticky bit, so
// for a positive normal double it returns the correctly rounded root, the host's sqrt (the native requires RN, the
// host's mode too). Frame: stwu -0x20 (back chain R), the input stfd at R-0x18, the result's words stored at R-0x10
// and reloaded with lfd (the 1.0 its rounding test spills there first is overwritten). Leaves f1 = the root, r3 = the
// input's biased exponent (ix0 >> 20), r4 = the root's high word. Zero, negative (errno EDOM), subnormal, infinite and
// NaN inputs run the original.
// ---------------------------------------------------------------------------------------------------------------
bool IeeeSqrtNative(CpuContext* ctx) {
    if (!EabiTranslation() || !LeanReady() || (ctx->fpscr & 3u) != 0) return false;
    const uint64_t bits = ctx->fpr[1].raw;
    const uint32_t exponent = static_cast<uint32_t>(bits >> 52);  // sign included: positive only
    if (exponent - 1u >= 0x7FEu) return false;
    const uint32_t R = ctx->gpr[1];
    if (!PlainRange(R - 0x20u, 0x18u)) return false;
    const double root = std::sqrt(ctx->fpr[1].d);
    const uint64_t rootBits = __builtin_bit_cast(uint64_t, root);
    const Image im{Window(), BrawlDirtyMemory::leanBlocks};
    im.W32(R - 0x20u, R);
    im.W32(R - 0x18u, static_cast<uint32_t>(bits >> 32));
    im.W32(R - 0x14u, static_cast<uint32_t>(bits));
    im.W32(R - 0x10u, static_cast<uint32_t>(rootBits >> 32));
    im.W32(R - 0x0Cu, static_cast<uint32_t>(rootBits));
    ctx->fpr[1].d = root;
    ctx->gpr[3] = exponent;
    ctx->gpr[4] = static_cast<uint32_t>(rootBits >> 32);
    return true;
}

bool IeeeSqrtWrites(const CpuContext& ctx, Ranges& ranges) { return ranges.Add(ctx.gpr[1] - 0x20u, 0x20u); }

// The stack window, the model and flag arrays and modelInSync (read from the pose as the original will see them).
bool SyncModelSpaceWrites(const CpuContext& ctx, Ranges& ranges) {
    const uint32_t pose = ctx.gpr[3], R = ctx.gpr[1];
    SyncLayout l;
    if (!LeanReady() || !SyncReadLayout(pose, l) || !ranges.Add(R - kSyncWindowBelow, kSyncWindowBelow + 8u) ||
        !ranges.Add(pose + 0x28u, 4u))
        return false;
    return l.count == 0 || (ranges.Add(l.model, 0x30u * l.count) && ranges.Add(l.flags, l.count));
}

// ---------------------------------------------------------------------------------------------------------------
// QsToMatrix (0x80284DB0): hkQsTransform -> hkMatrix4 (r3 = the QsTransform, r4 = out): R = hkRotation::set(q)
// (0x80285A6C), M = setMul(R, diag(scale)) (0x80282CD4), out = the hkTransform (M, translation) widened by 0x802871F4
// (w lanes 0, translation w 1). The pose's write-back to the g3d matrices (fn_800969E0) and the IK solvers call it.
// Frame s = (R0 & ~15) - 0x130 (R0 = entry r1): back chain R0, LR at R0+4, r30/r31 and f30/f31 saved at [R0 - 0x28, R0)
// (zeros: its callers reached it through its EABI entry); temporaries at s+0x10..0xFC; below s the callees' frames,
// whose final image is setMul's (back chain at s-0x120, R's 12 words at s-0x110, 0xE0 zero bytes). Its loads of the
// QsTransform all precede its stores to out. Leaves r3 = s+0x30, r4 = out, f1/f2 = the translation's z/y.
// ---------------------------------------------------------------------------------------------------------------
constexpr uint32_t kToMatWindowBelow = 0x260u;  // R0 - 15 - 0x130 - 0x120 rounded up
constexpr uint32_t kToMatSaves = 0x28u;

bool HavokConstantsOk(uint32_t r2) {
    static constexpr uint32_t kZeroWord[1] = {0u};
    return OneView(r2 - kSdata2A, kSdata2A - kSdata2C + 8u) && SameWords(r2 - kSdata2A, kConstA, 10) &&
           SameWords(r2 - kSdata2B, kConstB, 6) && SameWords(r2 - kSdata2C, kConstC, 2) &&
           SameWords(r2 - kSetMulZero, kZeroWord, 1);
}

bool QsToMatrixNative(CpuContext* ctx) {
    if (!EabiTranslation() || ctx->gqr[0] != 0 || !LeanReady() || !MkwHostNiActiveInline() || (ctx->fpscr & 3u) != 0)
        return false;
    const uint32_t pq = ctx->gpr[3], out = ctx->gpr[4], R0 = ctx->gpr[1];
    const uint32_t s = (R0 & ~15u) - 0x130u, low = R0 - kToMatWindowBelow, windowSize = kToMatWindowBelow + 8u;
    if (!PlainRange(low, windowSize) || !PlainRange(out, 0x40u) || !OneView(pq, 0x30u) ||
        Overlap(out, 0x40u, low, windowSize) || Overlap(pq, 0x30u, low, windowSize))
        return false;
    FpFlagsClear();
    if (!HavokConstantsOk(ctx->gpr[2])) return false;
    struct {
        float qs[12], rot[12], diag[12], m[12];
    } v;
    const uint8_t* const hq = Window() + Canonical(pq);
    uint32_t plain = 1;
    for (int i = 0; i < 12; ++i) {
        const uint32_t x = Load32(hq + 4 * i);
        plain &= PlainSingle(x) ? 1u : 0u;
        v.qs[i] = AsFloat(x);
    }
    if (!plain) return false;
    Ops o;
    QuatToRotation(o, v.qs + 4, v.rot);
    std::memset(v.diag, 0, sizeof(v.diag));
    v.diag[0] = v.qs[8];
    v.diag[5] = v.qs[9];
    v.diag[10] = v.qs[10];
    SetMul(o, v.m, v.rot, v.diag);
    if (FpFlagsRaised(&v) || o.Gap()) {
        Bump(g_counters[kQsToMatrix].fallback);
        return false;
    }
    const Image im{Window(), BrawlDirtyMemory::leanBlocks};
    im.W32(s, R0);
    im.W32(R0 + 4u, ctx->lr);
    im.ZeroSpan(R0 - kToMatSaves, kToMatSaves);
    static constexpr float kZero8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    im.WFs(s + 0x10u, kZero8, 8);
    im.WFs(s + 0x30u, v.m, 12);
    im.WFs(s + 0x60u, v.qs, 4);
    im.WFs(s + 0x70u, v.m, 12);
    im.WFs(s + 0xA0u, v.diag, 12);
    im.WFs(s + 0xD0u, v.rot, 12);
    im.W32(s - 0x120u, s);
    im.WFs(s - 0x110u, v.rot, 12);
    im.ZeroSpan(s - 0xE0u, 0xE0u);
    const float matrix[16] = {v.m[0], v.m[1], v.m[2], 0.0f, v.m[4], v.m[5],  v.m[6],  0.0f,
                              v.m[8], v.m[9], v.m[10], 0.0f, v.qs[0], v.qs[1], v.qs[2], 1.0f};
    im.WFs(out, matrix, 16);
    ctx->gpr[3] = s + 0x30u;
    ctx->gpr[4] = out;
    ctx->fpr[1].d = static_cast<double>(v.qs[2]);
    ctx->fpr[2].d = static_cast<double>(v.qs[1]);
    ctx->xer |= 0x20000000u;
    return true;
}

bool QsToMatrixWrites(const CpuContext& ctx, Ranges& ranges) {
    return ranges.Add(ctx.gpr[1] - kToMatWindowBelow, kToMatWindowBelow + 8u) && ranges.Add(ctx.gpr[4], 0x40u);
}

bool QsSetMulWrites(const CpuContext& ctx, Ranges& ranges) {
    return ranges.Add(ctx.gpr[1] - kSetMulWindowBelow, kSetMulWindowBelow) && ranges.Add(ctx.gpr[3], 0x30u);
}

bool QsFromMatrixWrites(const CpuContext& ctx, Ranges& ranges) {
    const uint32_t R0 = ctx.gpr[1];
    return ranges.Add(R0 - kQsWindowBelow - 0x40u, kQsWindowBelow + 0x48u) && ranges.Add(ctx.gpr[3], 0x30u);
}

// ---------------------------------------------------------------------------------------------------------------
// GfTaskProcess (0x8002DC74): gfTask::process(ProcessType) (r3 = the task, r4 = the type), ~1,030 calls a frame. A
// frameless dispatcher: types 0-7 through the jump table at 0x80422EE8 (cmplwi r4,7; r0 = r4 << 2; r5 = the entry),
// 8-15 through 0x80422EC8 (r0 = r4 - 8, cmplwi r0,7; r0 <<= 2; r4 = the entry); each case loads the task's vtable
// (+0x3C) and tail-branches (bctr) to its slot 8 + 4 * type (processDefault .. processEnd). Any other type returns.
// The translation lowers each jump table to a switch on the loaded entry and each case to: r0, r5 or r4, r12, cr and
// ctr stored to the context, then MKW_INDIRECT_JUMP (one inline-cache site per case). Its return path stores nothing
// (cr0 and r0 are dead there). The native does the same stores and the same tail jump, through one cache site per type
// (the translation's per-case sites), with a 32-record per-type target cache in front of it (the sites alone missed
// on 47% of calls; see the jump), and skips the switch: an entry other than its case's
// label (0x8002DC9C / 0x8002DD40 + 16 * index) runs the original before any store. cr0 is the unsigned compare of
// the index (0-7) with 7: LT, or EQ for 7, with xer's SO.
// The native mode runs it straight from the wrapper (g_gfTaskFast: no per-call counters, which would cost about as
// much as the switch it saves). GAME_NATIVES=2 and GAME_NATIVES_OFF=gftask go through GfTaskProcessSlow.
// ---------------------------------------------------------------------------------------------------------------
constexpr uint32_t kGfTaskTableLow = 0x80422EE8u, kGfTaskTableHigh = 0x80422EC8u;
constexpr uint32_t kGfTaskCaseLow = 0x8002DC9Cu, kGfTaskCaseHigh = 0x8002DD40u;
constexpr uint32_t kGfTaskProbe = 0x8002DDC4u;  // gfTask::init: a blr (its translation stores nothing)

// Per type: the jump-table entry's address, its expected case label, cr0's LT/EQ bits in place, r0 (index << 2) and
// the register the entry lands in (r5 for 0-7, r4 for 8-15).
struct GfTaskCase {
    uint32_t entryAddress, label, cr0, r0, entryGpr;
};
constexpr GfTaskCase MakeGfTaskCase(uint32_t type) {
    const uint32_t index = type & 7u;
    return {(type >= 8u ? kGfTaskTableHigh : kGfTaskTableLow) + 4u * index,
            (type >= 8u ? kGfTaskCaseHigh : kGfTaskCaseLow) + 16u * index, (index == 7u ? 0x2u : 0x8u) << 28,
            index << 2, type >= 8u ? 4u : 5u};
}
constexpr GfTaskCase kGfTaskCases[16] = {
    MakeGfTaskCase(0),  MakeGfTaskCase(1),  MakeGfTaskCase(2),  MakeGfTaskCase(3),  MakeGfTaskCase(4),  MakeGfTaskCase(5),
    MakeGfTaskCase(6),  MakeGfTaskCase(7),  MakeGfTaskCase(8),  MakeGfTaskCase(9),  MakeGfTaskCase(10), MakeGfTaskCase(11),
    MakeGfTaskCase(12), MakeGfTaskCase(13), MakeGfTaskCase(14), MakeGfTaskCase(15)};

// kProbe (gfsched's verify): every store of the dispatch, then a return in place of the jump, as if the task's
// method were gfTask::init (a blr whose translation stores nothing).
template <bool kProbe = false>
[[gnu::always_inline]] inline bool GfTaskProcessNative(CpuContext* ctx) {
    const uint32_t type = ctx->gpr[4];
    if (type >= 16u) return true;  // bgtlr
    const GfTaskCase& c = kGfTaskCases[type];
    const uint32_t entry = MemoryInline::FlatRead32(c.entryAddress);
    if (entry != c.label) [[unlikely]]
        return false;
    const uint32_t target = MemoryInline::FlatRead32(MemoryInline::FlatRead32(ctx->gpr[3] + 60u) + 8u + 4u * type);
    ctx->gpr[0] = c.r0;
    ctx->gpr[c.entryGpr] = entry;
    ctx->gpr[12] = target;
    ctx->crv[0] = PpcCrvEncode((c.cr0 >> 28) | MKW_CR_SO_BIT(ctx->xer));
    ctx->ctr = target;
    if constexpr (kProbe) return true;
#if MKW_INDIRECT_IC
    // MkwInvokeIndirectJumpSite's hit, inlined (indirect_call_site_hit.h), from a per-type cache of 32 ways keyed by
    // target in front of the type's site. One type dispatches round-robin to every task class's method, so even the
    // site's three ways miss often (two ways missed on 47% of calls: 480 of 1,022 a frame in Training). The cache
    // holds only ways the site's own slow path put in the site for that target (unguarded, packed: a pure function of
    // the address after TranslatedFunctionRegistry::Finalize), and a hit is the site's hit: same test, same
    // execution-address store, same entry. GFTASK_CACHE=0 leaves only the site.
    static MkwIndirectSite sites[16];
    static std::atomic<uint64_t> cache[16][32];
    MkwIndirectSite& site = sites[type];
    std::atomic<uint64_t>& slot = cache[type][(target * 0x9E3779B1u) >> 27];
    const uint64_t way = slot.load(std::memory_order_relaxed);
    if (__builtin_expect(MkwIndirectWayMatches(way, target) && TryGetCpuContext() == ctx, 1)) {
        MKW_INDIRECT_IC_COUNT_HIT(site);
        RecompMod::g_currentTranslatedExecutionAddress = target;
        MkwIndirectWayEntry(way)(ctx);
        return true;
    }
    MkwInvokeIndirectJumpSite(target, ctx, site);
    if (g_gfTaskCache) {
        // The site's ways after the call (the callee may have reused the site since: the test keeps only this target).
        for (const auto& w : site.ways) {
            const uint64_t held = w.load(std::memory_order_relaxed);
            if (MkwIndirectWayMatches(held, target)) {
                slot.store(held, std::memory_order_relaxed);
                break;
            }
        }
    }
#else
    InvokeIndirectJump(target, ctx);
#endif
    return true;
}

// Verify: the callee is whatever the task's class does, so neither side may run it. The vtable slot the call reads
// is pointed at gfTask::init (a blr) while the original and then the native run from the same state, each twice
// (the second run's cache site already holds the probe: a hit on both sides, which the count compares); then the
// slot is restored and the original makes the real call (its results stay).
void GfTaskProcessVerify(CpuContext* ctx) {
    Counters& n = g_counters[kGfTaskProcess];
    const CpuContext in = *ctx;
    const uint32_t type = in.gpr[4];
    uint32_t slot = 0;
    uint8_t saved[4] = {};
    if (type < 16u) {
        slot = MemoryInline::FlatRead32(in.gpr[3] + 60u) + 8u + 4u * type;
        if (!LeanReady() || (slot & 3u) != 0 || !PlainRange(slot, 4u)) {
            Bump(n.declined);
            func_8002DC74_body(ctx);
            return;
        }
        std::memcpy(saved, Window() + Canonical(slot), 4);
        Store32(Window() + Canonical(slot), kGfTaskProbe);
    }
    func_8002DC74_body(ctx);
    *ctx = in;
    const uint64_t originalStart = InsnCount::On() ? InsnCount::Read() : 0;
    func_8002DC74_body(ctx);
    const uint64_t originalInsns = InsnCount::On() ? InsnCount::Since(originalStart) : 0;
    const CpuContext orig = *ctx;
    *ctx = in;
    GfTaskProcessNative(ctx);
    *ctx = in;
    const uint64_t nativeStart = InsnCount::On() ? InsnCount::Read() : 0;
    const bool ranNatively = GfTaskProcessNative(ctx);
    const uint64_t nativeInsns = InsnCount::On() ? InsnCount::Since(nativeStart) : 0;
    if (ranNatively) {
        if (InsnCount::On()) CountPair(kGfTaskProcess, originalInsns, nativeInsns);
        n.compared.fetch_add(1, std::memory_order_relaxed);
        char what[640] = "";
        if (!SameContext(orig, *ctx, what, sizeof(what))) {
            n.mismatches.fetch_add(1, std::memory_order_relaxed);
            if (g_verifyLogs.fetch_add(1, std::memory_order_relaxed) < kMaxVerifyLogs)
                RT_LOGF(RT_TAG_GX, "[gamenat] verify MISMATCH gftask: r3=%08X r4=%08X: %s\n", in.gpr[3], in.gpr[4], what);
        }
    } else {
        Bump(n.declined);
    }
    if (type < 16u) std::memcpy(Window() + Canonical(slot), saved, 4);
    *ctx = in;
    func_8002DC74_body(ctx);
}

// The counted modes: verify, the original (GAME_NATIVES=0 or GAME_NATIVES_OFF=gftask), and the native mode of a
// translation without EABI entries (P+), which runs the original.
thread_local bool t_gfProbe = false;          // gfsched's verify runs both loops with probe dispatches
thread_local bool t_gfProbeDeclined = false;  // a probe dispatch the native declined (the run is not compared)

[[gnu::noinline]] void GfTaskProcessSlow(CpuContext* ctx) {
    if (t_gfProbe) {  // the original loop's dispatch under gfsched's verify
        if (!GfTaskProcessNative<true>(ctx)) t_gfProbeDeclined = true;
        return;
    }
    const Settings& s = Config();
    Bump(g_counters[kGfTaskProcess].calls);
    if (s.mode == kVerify && !s.off[kGfTaskProcess] && EabiTranslation()) {
        GfTaskProcessVerify(ctx);
        return;
    }
    func_8002DC74_body(ctx);
}

// ---------------------------------------------------------------------------------------------------------------
// GfTaskScheduler (0x8002E578): gfTaskScheduler::process(bool) (r3 = the scheduler, r4 = the bool), once a frame,
// around ~1,030 gfTask::process dispatches. Frame: stwu -0x30 (back chain R at s = R - 0x30), LR at R+4, r24-r31
// at R-0x20..R-4 (_savegpr_24: the context's values). updateStatusPre (0x8002E44C, EABI entry); state byte (+0) = 1;
// then for each type 0-15 (all skipped while +8 is nonzero; the type in +1 either way) and each list 0-16 (heads at
// +0x14, list index in the +6 halfword) it walks the tasks: prev (+4) and next (+8) read, (task, next) stored to
// +0xC/+0x10, the flag byte (+0x2C) tested (bool set: 0x40 and not 0x20; clear: 0x40, then a reread, and not 0x02),
// gfTask::process(task, type) called (LR 0x8002E618 / 0x8002E640, r3/r4 stored), and the next task chosen: +8 of the
// task when its flag bit 0 is set, else the list head again when prev was 0, else +0x10 (task code that unlinks the
// next task updates it). Then the pending moves (+0x140 entries of 12 bytes at +0x144: find the task by id in list
// (int8)entry[8]>>... at +0x58, set its +0x30, or unlinkProcessList (0x8002EB88, EABI entry) and append it to its new
// list), +0x140 = 0, state byte = 0, return.
// The native is the translated body step by step: every guest load and store in its order, the same
// call entries with the same context stores (LR, r1 at updateStatusPre; LR, r3, r4 at each dispatch; LR, xer at
// unlinkProcessList), and the same leftovers at return (LR, r1, r3, r4; xer, cr and r24-r31 stay host locals in the
// translation too). Only the dispatch is inlined: the gfTask native (its guard byte clear and its fast mode on), else
// a call of 0x8002DC74's wrapper, as the original makes it.
// ---------------------------------------------------------------------------------------------------------------
int GuardIndexOf(uint32_t address) {
    const uint32_t* const begin = g_mkwGuardedFunctionStarts;
    const uint32_t* const end = begin + g_mkwGuardedFunctionCount;
    const uint32_t* const it = std::lower_bound(begin, end, address);
    return it != end && *it == address ? static_cast<int>(it - begin) : -1;
}
int GfTaskGuard() {
    static const int index = GuardIndexOf(0x8002DC74u);
    return index;
}

// One dispatch from the loop: the original's call site (LR, r3, r4 to the context) and 0x8002DC74.
template <bool kProbe>
[[gnu::always_inline]] inline void SchedDispatch(CpuContext* ctx, uint32_t task, uint32_t type, uint32_t lr, int guard) {
    ctx->lr = lr;
    ctx->gpr[3] = task;
    ctx->gpr[4] = type;
    if constexpr (kProbe) {
        if (g_mkwGuestCodePatched[guard] != 0 || !GfTaskProcessNative<true>(ctx)) t_gfProbeDeclined = true;
        return;
    }
    if (g_gfTaskFast && g_mkwGuestCodePatched[guard] == 0) [[likely]] {
        if (GfTaskProcessNative(ctx)) return;
        Bump(g_counters[kGfTaskProcess].declined);
        func_8002DC74_body(ctx);
        return;
    }
    func_8002DC74(ctx);
}

template <bool kProbe = false>
bool GfTaskSchedulerNative(CpuContext* ctx) {
    const int guard = GfTaskGuard();
    if (!EabiTranslation() || func_8002E44C_eabi == nullptr || func_8002EB88_eabi == nullptr || !LeanReady() ||
        guard < 0)
        return false;
    const uint32_t R = ctx->gpr[1], s = R - 0x30u;
    if (!PlainRange(s, 0x38u)) return false;
    using MemoryInline::FlatRead16;
    using MemoryInline::FlatRead32;
    using MemoryInline::FlatRead8;
    using MemoryInline::FlatWrite16;
    using MemoryInline::FlatWrite32;
    using MemoryInline::FlatWrite8;
    const Image im{Window(), BrawlDirtyMemory::leanBlocks};
    uint32_t r3 = ctx->gpr[3], r4 = ctx->gpr[4], xer = ctx->xer;
    // Prologue: back chain, LR, r24-r31.
    im.W32(s, R);
    im.W32(R + 4u, ctx->lr);
    for (uint32_t k = 0; k < 8; ++k) im.W32(R - 0x20u + 4u * k, ctx->gpr[24 + k]);
    const uint32_t sched = r3, flag = r4;
    ctx->lr = 0x8002E598u;
    ctx->gpr[1] = s;
    const MkwEabiPair status = func_8002E44C_eabi(ctx, r3, r4);
    r3 = static_cast<uint32_t>(status.first);
    r4 = static_cast<uint32_t>(status.second);
    FlatWrite32(sched, (FlatRead32(sched) & 0x00FFFFFFu) | 0x01000000u);
    for (uint32_t type = 0; type < 16; ++type) {
        const uint32_t paused = FlatRead32(sched + 8u);
        r3 = (FlatRead32(sched) & 0xFF00FFFFu) | (type << 16);
        FlatWrite32(sched, r3);
        if (paused != 0) continue;  // (slw. 1 << type: nonzero for every type)
        uint32_t head = sched + 20u;
        for (uint32_t list = 0; list < 17; ++list, head += 4u) {
            uint32_t task = FlatRead32(head);
            FlatWrite16(sched + 6u, static_cast<uint16_t>(list));
            while (task != 0) {
                const uint32_t prev = FlatRead32(task + 4u);
                const uint32_t next = FlatRead32(task + 8u);
                FlatWrite32(sched + 12u, task);
                FlatWrite32(sched + 16u, next);
                if (flag != 0) {
                    r3 = FlatRead8(task + 44u);
                    if ((r3 & 0x40u) != 0 && (r3 & 0x20u) == 0) {
                        SchedDispatch<kProbe>(ctx, task, type, 0x8002E618u, guard);
                        r3 = ctx->gpr[3];
                        r4 = ctx->gpr[4];
                    }
                } else if ((FlatRead8(task + 44u) & 0x40u) != 0 && (FlatRead8(task + 44u) & 0x02u) == 0) {
                    SchedDispatch<kProbe>(ctx, task, type, 0x8002E640u, guard);
                    r3 = ctx->gpr[3];
                    r4 = ctx->gpr[4];
                }
                if ((FlatRead8(task + 44u) & 1u) != 0)
                    task = FlatRead32(task + 8u);
                else if (prev == 0)
                    task = FlatRead32(head);
                else
                    task = FlatRead32(sched + 16u);
            }
        }
    }
    // The pending moves.
    const uint32_t pending = FlatRead16(sched + 320u);
    FlatWrite32(sched + 12u, 0);
    FlatWrite32(sched + 16u, 0);
    if (pending != 0) {
        uint32_t entry = sched + 324u;
        for (uint32_t i = 0; i < pending; ++i, entry += 12u) {
            r4 = FlatRead32(entry + 4u);
            uint32_t r0 = FlatRead32(entry + 8u);
            r0 = PpcRotl32Inline(r0, 8) & 0xFF800000u;  // rlwinm r0,r0,8,0,8
            const bool carry = (r0 & 0x80000000u) != 0 && (r0 & 0x00FFFFFFu) != 0;  // srawi r0,r0,24
            xer = (xer & 0xDFFFFFFFu) | (carry ? 0x20000000u : 0u);
            r0 = static_cast<uint32_t>(static_cast<int32_t>(r0) >> 24) << 2;
            r3 = sched + r0;
            uint32_t task = FlatRead32(r3 + 88u);
            while (task != 0 && r4 != FlatRead32(task + 40u)) task = FlatRead32(task + 16u);
            if (task == 0) continue;
            if ((FlatRead8(task + 44u) & 1u) == 0) {
                const uint32_t list = FlatRead8(entry + 8u);
                r3 = FlatRead32(entry);
                FlatWrite8(r3 + 48u, static_cast<uint8_t>(list));
                continue;
            }
            ctx->lr = 0x8002E720u;
            ctx->xer = xer;
            func_8002EB88_eabi(ctx, sched, task);
            const uint32_t list = FlatRead8(entry + 8u);
            FlatWrite8(task + 48u, static_cast<uint8_t>(list));
            r3 = sched + (list << 2);
            r4 = FlatRead32(r3 + 20u);
            if (r4 == 0) {
                FlatWrite32(r3 + 20u, task);
                continue;
            }
            for (uint32_t link = FlatRead32(r4 + 8u); link != 0; link = FlatRead32(r4 + 8u)) r4 = link;
            FlatWrite32(r4 + 8u, task);
            FlatWrite32(task + 4u, r4);
        }
        FlatWrite16(sched + 320u, 0);
    }
    FlatWrite32(sched, FlatRead32(sched) & 0x00FFFFFFu);
    // Epilogue: r24-r31 come back from the frame into host locals only (the context's are unchanged).
    ctx->lr = FlatRead32(s + 52u);
    ctx->gpr[1] = R;
    ctx->gpr[3] = r3;
    ctx->gpr[4] = r4;
    return true;
}

// The site of the whole call's instructions (NAT_INSN=1, native mode: native or original column).
int SchedSite() {
    static const int site = InsnCount::Site("game.gfsched");
    return site;
}

// Verify (GAME_NATIVES=2): the original loop and the native from the same state, both with probe dispatches (every
// store of gfTask::process, then a return: as if each task's method were gfTask::init; gftask's own verify covers
// the dispatch against its translation). No task code runs, so the lists hold still, and what either side may write
// is: its frame, the scheduler (with its pending entries), and the tasks' first 0x38 bytes (updateStatusPre's flag
// bytes, the pending moves' list links and +0x30) for every task in the 17 process lists, the 40 id lists at +0x58
// (+0x10 links), the list at +0x130 and the pending entries. Those are compared with the whole register file, the
// pair counted (game.gfsched: loop and dispatch, no method), then restored for the original's real call.
struct SchedVerifyBuffers {
    std::vector<GuestRange> ranges;
    std::vector<uint32_t> tasks;
    std::vector<uint8_t> before, afterOrig, afterNative;
};
void CaptureAll(const std::vector<GuestRange>& ranges, std::vector<uint8_t>& out) {
    out.clear();
    for (const GuestRange& r : ranges) {
        const uint8_t* const host = Window() + Canonical(r.address);
        out.insert(out.end(), host, host + r.size);
    }
}
void RestoreAll(const std::vector<GuestRange>& ranges, const std::vector<uint8_t>& in) {
    size_t at = 0;
    for (const GuestRange& r : ranges) {
        std::memcpy(Window() + Canonical(r.address), in.data() + at, r.size);
        at += r.size;
    }
}
// The tasks either side may touch; false when a list does not end within kMaxTasks or a range is not plain.
bool SchedVerifyRanges(uint32_t sched, uint32_t R, SchedVerifyBuffers& b) {
    constexpr size_t kMaxTasks = 4096;
    b.ranges.clear();
    b.tasks.clear();
    const auto walk = [&](uint32_t head, uint32_t link) {
        for (uint32_t t = MemoryInline::FlatRead32(head); t != 0; t = MemoryInline::FlatRead32(t + link)) {
            if (b.tasks.size() >= kMaxTasks) return false;
            b.tasks.push_back(t);
        }
        return true;
    };
    for (uint32_t l = 0; l < 17; ++l)
        if (!walk(sched + 0x14u + 4u * l, 8u)) return false;
    for (uint32_t l = 0; l < 40; ++l)
        if (!walk(sched + 0x58u + 4u * l, 0x10u)) return false;
    if (!walk(sched + 0x130u, 8u)) return false;
    const uint32_t pending = MemoryInline::FlatRead16(sched + 320u);
    for (uint32_t i = 0; i < pending; ++i) b.tasks.push_back(MemoryInline::FlatRead32(sched + 324u + 12u * i));
    std::sort(b.tasks.begin(), b.tasks.end());
    b.tasks.erase(std::unique(b.tasks.begin(), b.tasks.end()), b.tasks.end());
    b.ranges.push_back({R - 0x30u, 0x38u});
    b.ranges.push_back({sched, 0x144u + 12u * pending});
    for (const uint32_t t : b.tasks)
        if (t != 0) b.ranges.push_back({t, 0x38u});
    for (const GuestRange& r : b.ranges)
        if (!PlainRange(r.address, r.size)) return false;
    return true;
}

void GfTaskSchedulerVerify(CpuContext* ctx) {
    Counters& n = g_counters[kGfTaskScheduler];
    thread_local SchedVerifyBuffers b;
    const CpuContext in = *ctx;
    if (!LeanReady() || !SchedVerifyRanges(in.gpr[3], in.gpr[1], b)) {
        Bump(n.declined);
        func_8002E578_body(ctx);
        return;
    }
    AxDspHle::JoinMixWorker();
    CaptureAll(b.ranges, b.before);
    t_gfProbe = true;
    t_gfProbeDeclined = false;
    const uint64_t originalStart = InsnCount::On() ? InsnCount::Read() : 0;
    func_8002E578_body(ctx);
    const uint64_t originalInsns = InsnCount::On() ? InsnCount::Since(originalStart) : 0;
    const CpuContext orig = *ctx;
    CaptureAll(b.ranges, b.afterOrig);
    RestoreAll(b.ranges, b.before);
    *ctx = in;
    const uint64_t nativeStart = InsnCount::On() ? InsnCount::Read() : 0;
    const bool ranNatively = GfTaskSchedulerNative<true>(ctx);
    const uint64_t nativeInsns = InsnCount::On() ? InsnCount::Since(nativeStart) : 0;
    t_gfProbe = false;
    if (!ranNatively || t_gfProbeDeclined) {
        Bump(n.declined);
    } else {
        CaptureAll(b.ranges, b.afterNative);
        if (InsnCount::On()) CountPair(kGfTaskScheduler, originalInsns, nativeInsns);
        n.compared.fetch_add(1, std::memory_order_relaxed);
        char what[640] = "";
        bool same = SameContext(orig, *ctx, what, sizeof(what));
        if (b.afterNative != b.afterOrig) {
            same = false;
            size_t at = 0, len = std::strlen(what);
            uint32_t words = 0;
            for (const GuestRange& r : b.ranges) {
                for (uint32_t w = 0; w + 4 <= r.size; w += 4) {
                    if (std::memcmp(b.afterNative.data() + at + w, b.afterOrig.data() + at + w, 4) == 0) continue;
                    if (++words <= 6 && len < sizeof(what))
                        len += static_cast<size_t>(std::snprintf(what + len, sizeof(what) - len, "mem %08X %08X/%08X ",
                                                                 r.address + w, Load32(b.afterOrig.data() + at + w),
                                                                 Load32(b.afterNative.data() + at + w)));
                }
                at += r.size;
            }
            if (len < sizeof(what)) std::snprintf(what + len, sizeof(what) - len, "(%u words differ)", words);
        }
        if (!same) {
            n.mismatches.fetch_add(1, std::memory_order_relaxed);
            if (g_verifyLogs.fetch_add(1, std::memory_order_relaxed) < kMaxVerifyLogs)
                RT_LOGF(RT_TAG_GX, "[gamenat] verify MISMATCH gfsched: sched=%08X r4=%08X: %s\n", in.gpr[3], in.gpr[4],
                        what);
        }
    }
    RestoreAll(b.ranges, b.before);
    *ctx = in;
    func_8002E578_body(ctx);
}
#endif  // MKW_LEAN_MEMORY

// After an original run a native stands for (vanilla): the non-volatile saves its callers' EABI entry wrote as 0 (the
// context body saves the context's copies there), kBytes below the entry r1. QsFromMatrix: r31 (4 bytes); setMul:
// f14-f31 (0x120); QsToMatrix: r30/r31 and f30/f31 (0x28); setInverse: f24-f31 (0x80).
template <uint32_t kBytes>
void ZeroEabiSaves(uint32_t r1) {
#if MKW_LEAN_MEMORY
    if (EabiTranslation() && LeanReady() && PlainRange(r1 - kBytes, kBytes))
        std::memset(Window() + Canonical(r1 - kBytes), 0, kBytes);
#else
    (void)r1;
#endif
}
constexpr auto QsFromMatrixFixup = &ZeroEabiSaves<4u>;
constexpr auto QsSetMulFixup = &ZeroEabiSaves<0x120u>;
constexpr auto QsToMatrixFixup = &ZeroEabiSaves<0x28u>;
constexpr auto QsSetInverseFixup = &ZeroEabiSaves<0x80u>;
inline void NoFixup(uint32_t) {}

}  // namespace
}  // namespace gamenat

// gfTaskScheduler::process, once a frame: the counted modes cost nothing that matters, so it takes the plain path.
// NAT_INSN=1 in the native mode counts the whole call (task methods included) into game.gfsched: the native column,
// or the original column under GAME_NATIVES_OFF=gfsched, so two runs of one binary give the saving.
extern "C" void GameNat_GfTaskScheduler(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    using namespace gamenat;
    const Settings& s = Config();
    Bump(g_counters[kGfTaskScheduler].calls);
    if (s.mode == kVerify && !s.off[kGfTaskScheduler] && EabiTranslation()) {
        GfTaskSchedulerVerify(ctx);
        return;
    }
    const bool native = s.mode == kNative && !s.off[kGfTaskScheduler];
    const uint64_t start = InsnCount::On() ? InsnCount::Read() : 0;
    if (native && GfTaskSchedulerNative(ctx)) {
        Bump(g_counters[kGfTaskScheduler].native);
    } else {
        if (native) Bump(g_counters[kGfTaskScheduler].declined);
        func_8002E578_body(ctx);
    }
    if (InsnCount::On())
        InsnCount::Add(SchedSite(), native ? InsnCount::kNative : InsnCount::kOriginal, InsnCount::Since(start));
#else
    func_8002E578_body(ctx);
#endif
}

extern "C" void GameNat_GfTaskProcess(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    if (gamenat::g_gfTaskFast) [[likely]] {
        if (gamenat::GfTaskProcessNative(ctx)) return;
        gamenat::Bump(gamenat::g_counters[gamenat::kGfTaskProcess].declined);
        func_8002DC74_body(ctx);
        return;
    }
    gamenat::GfTaskProcessSlow(ctx);
#else
    func_8002DC74_body(ctx);
#endif
}

extern "C" void GameNat_MtScale(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    gamenat::Run(gamenat::kMtScale, ctx, func_8003E6DC_body, gamenat::MtScaleWrites, gamenat::NoFixup,
                 gamenat::MtScaleNative);
#else
    func_8003E6DC_body(ctx);
#endif
}

extern "C" void GameNat_QsSetInverse(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    gamenat::Run(gamenat::kQsSetInverse, ctx, func_80284FEC_body, gamenat::QsSetInverseWrites,
                 gamenat::QsSetInverseFixup, gamenat::QsSetInverseNative);
#else
    func_80284FEC_body(ctx);
    gamenat::QsSetInverseFixup(ctx->gpr[1]);
#endif
}

extern "C" void GameNat_QsMulInverseMul(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    gamenat::Run(gamenat::kQsMulInverseMul, ctx, func_802855F0_body, gamenat::QsMulInverseMulWrites,
                 gamenat::NoFixup, gamenat::QsMulInverseMulNative);
#else
    func_802855F0_body(ctx);
#endif
}

extern "C" void GameNat_IeeeSqrt(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    gamenat::Run(gamenat::kIeeeSqrt, ctx, func_80400B48_body, gamenat::IeeeSqrtWrites, gamenat::NoFixup,
                 gamenat::IeeeSqrtNative);
#else
    func_80400B48_body(ctx);
#endif
}

extern "C" void GameNat_SyncModelSpace(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    gamenat::Run(gamenat::kSyncModelSpace, ctx, func_8032EB5C_body, gamenat::SyncModelSpaceWrites,
                 gamenat::NoFixup, gamenat::SyncModelSpaceNative);
#else
    func_8032EB5C_body(ctx);
#endif
}

extern "C" void GameNat_QsToMatrix(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    gamenat::Run(gamenat::kQsToMatrix, ctx, func_80284DB0_body, gamenat::QsToMatrixWrites, gamenat::QsToMatrixFixup,
                 gamenat::QsToMatrixNative);
#else
    func_80284DB0_body(ctx);
    gamenat::QsToMatrixFixup(ctx->gpr[1]);
#endif
}

extern "C" void GameNat_QsSetMul(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    gamenat::Run(gamenat::kQsSetMul, ctx, func_8028526C_body, gamenat::QsSetMulWrites, gamenat::QsSetMulFixup,
                 gamenat::QsSetMulNative);
#else
    func_8028526C_body(ctx);
    gamenat::QsSetMulFixup(ctx->gpr[1]);
#endif
}

extern "C" void GameNat_QsFromMatrix(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    gamenat::Run(gamenat::kQsFromMatrix, ctx, func_80284F2C_body, gamenat::QsFromMatrixWrites,
                 gamenat::QsFromMatrixFixup, gamenat::QsFromMatrixNative);
#else
    func_80284F2C_body(ctx);
    gamenat::QsFromMatrixFixup(ctx->gpr[1]);
#endif
}

// The fps log's window line (vi.cpp NSMBW_LOG_FPS): per native, calls per frame and how they went.
void GameNativesLogStats(double frames) {
    using namespace gamenat;
    if (frames <= 0) return;
    std::string line;
    for (int i = 0; i < kNumFn; ++i) {
        Counters& c = g_counters[i];
        const uint64_t calls = c.calls.exchange(0, std::memory_order_relaxed);
        const uint64_t native = c.native.exchange(0, std::memory_order_relaxed);
        const uint64_t declined = c.declined.exchange(0, std::memory_order_relaxed);
        const uint64_t fallback = c.fallback.exchange(0, std::memory_order_relaxed);
        const uint64_t compared = c.compared.exchange(0, std::memory_order_relaxed);
        const uint64_t mismatches = c.mismatches.exchange(0, std::memory_order_relaxed);
        const uint64_t undeclared = c.undeclared.exchange(0, std::memory_order_relaxed);
        const uint64_t foreign = c.foreign.exchange(0, std::memory_order_relaxed);
        const uint64_t overmarked = c.overmarked.exchange(0, std::memory_order_relaxed);
        if (calls == 0) continue;
        char buf[280];
        std::snprintf(buf, sizeof(buf), "%s%s %.1f calls (native %.1f, declined %.1f, fp fallback %.1f, verified %.1f, "
                      "mismatch %.1f, undeclared %.1f, foreign %.1f, original over-marked %.1f)",
                      line.empty() ? "" : "; ", kFnNames[i], calls / frames, native / frames, declined / frames,
                      fallback / frames, compared / frames, mismatches / frames, undeclared / frames,
                      foreign / frames, overmarked / frames);
        line += buf;
    }
    if (!line.empty()) RT_LOGF(RT_TAG_VI, "game natives (GAME_NATIVES=%d): %s\n", Config().mode, line.c_str());
}
