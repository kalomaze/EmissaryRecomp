// Native bodies for nw4r g3d's calc side (doldecomp/ogws g3d_calcworld.cpp, g3d_calcview.cpp, g3d_resanmchr.cpp,
// nw4r for MDL0 rev 9, checked against Brawl's translated functions), the per-frame work under ScnMdl's CALC_WORLD
// and CALC_VIEW: CalcSkinning (0x801A5414), the envelope blend that writes skinned matrices into the model's
// world-matrix array; CalcView_LC (0x801A3D68), the view, normal and texture matrices, with ut::LC::StoreBlocks
// (0x8015D8BC); and the CHR keyframe readers (FVS32/48/96: 0x801943DC, 0x8019426C, 0x80194140) that evaluate each
// animated bone's scale, rotation and translation.
//
// Calc output is game state: hitboxes and attachments follow the bones, and rollback snapshots and resimulates the
// world-matrix arrays. So each native writes exactly the guest bytes its translated original writes, the dead stack
// below r1 included (its own frame, saved registers and stack temporaries, and its callees' frames: it reaches every
// callee through the callee's guarded wrapper with the original's r1 and return address), leaves exactly the same
// rollback dirty marks, and leaves r1, r2, r13, the non-volatile registers and LR as the original does. The void
// functions' volatile registers are not reproduced (as for the native draw, g3d_native.cpp): compiled callers never
// read them. The keyframe readers return a float in f1; their vanilla (EABI) translated callers reload r3, r4, f1 and
// f2 from the context, so those natives write back those four with the original's values, and no other volatile
// register (the P+ translation also writes back r0, r5, r6, f0-f11, CR0 and XER, scratch to every caller). Floating point
// uses the translated code's own helpers (isa/ppc_isa_float.h), or one host FMA per lane where those compute the same
// bits (no NaN result), with the helpers' path for anything else.
//
// MKW_NATIVE_BODY (abi_bridge.h) makes the translator's wrapper func_X tail-call the native after its guest code guard
// (a mod's patch of the function wins), and excludes the function from MKW_EABI_REGS register-passing entries, so
// every caller reaches it. Inputs a native does not handle (a GQR0 other than the float encoding, a store the lean
// window would send to the checked path) run the translated body before any guest-visible effect.
//
// G3D_CALC: 1 (default) natives; 0 the translated originals behind the same wrappers (A/B on one binary); 2 verify:
// every call runs the original and then the native from the same state, compares written guest bytes (declared
// ranges: the function's arrays and the stack window), the rollback dirty marks of all of MEM1 and MEM2 (so a write
// outside the declared ranges is caught too; the keyframe readers: their ranges' marks, and all of them every 64th
// call) and the registers above (the readers: also r3, r4, f1, f2), logs differences, and keeps the original's results.
// G3D_CALC_OFF=name,...: those natives run their originals (names: skinning, calcview, fvs32, fvs48, fvs96; cvfull:
// CalcView_LC runs its translated body with only the StoreBlocks notifications deferred, the previous native).
// G3D_CALC_CVGENERIC=1: CalcView_LC's native takes its generic path (each copy leaf through its wrapper) on every
// matrix, the path it otherwise takes only after a billboard callee changed GQR0; for verify runs.

#include "abi_bridge.h"
#include "audio/ax_dsp.h"
#include "brawl_dirty_memory.h"
#include "insn_count.h"
#include "memory.h"
#include "ppc_runtime.h"
#include "runtime_log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

MKW_NATIVE_BODY(0x801A5414u, G3dCalc_CalcSkinning);
MKW_NATIVE_BODY(0x801A3D68u, G3dCalc_CalcViewLC);
MKW_NATIVE_BODY(0x8015D8BCu, G3dCalc_LCStoreBlocks);
MKW_NATIVE_BODY(0x801943DCu, G3dCalc_KeyFVS32);
MKW_NATIVE_BODY(0x8019426Cu, G3dCalc_KeyFVS48);
MKW_NATIVE_BODY(0x80162C04u, G3dCalc_Mtx44Copy);
MKW_NATIVE_BODY(0x801A7DBCu, G3dCalc_CameraMtx);
MKW_NATIVE_BODY(0x801B6708u, G3dCalc_LightSettingCopy);
MKW_NATIVE_BODY(0x80004338u, G3dCalc_Memcpy);
// Not registered: FVS96 (0x80194140, G3dCalc_KeyFVS96) and G3DState's light slot setter (0x801A0DD8,
// G3dCalc_LightSlot). NAT_INSN's per-call counts on vanilla (NOTE3): FVS96 854 instructions a call against its
// translation's 717, the light slot 186 against 134 (the translation got cheaper with MKW_LEAN_RANGES), so both
// natives cost more than they save, and as native bodies they also took the light slot's EABI entry from its
// callers. Their code stays for a P+ measurement (heavier context translation there); registering one again is a
// line here.
MKW_NATIVE_BODY(0x80004388u, G3dCalc_FillMem);
MKW_NATIVE_BODY(0x801A7FC8u, G3dCalc_EnvTexMtx);

extern "C" void func_80162C04_body(CpuContext* ctx);
extern "C" void func_801A7DBC_body(CpuContext* ctx);
extern "C" void func_801B6708_body(CpuContext* ctx);
extern "C" void func_801A0DD8_body(CpuContext* ctx);
extern "C" void func_80004338_body(CpuContext* ctx);
extern "C" void func_80004388_body(CpuContext* ctx);
extern "C" void func_801A7FC8_body(CpuContext* ctx);
extern "C" void func_801A81D4(CpuContext* ctx);  // Camera::UpdateCameraMtx
extern "C" void func_803F6150(CpuContext* ctx);  // memcmp
extern "C" void func_801A5414_body(CpuContext* ctx);
extern "C" void func_801A3D68_body(CpuContext* ctx);
extern "C" void func_8015D8BC_body(CpuContext* ctx);
extern "C" void func_801943DC_body(CpuContext* ctx);
extern "C" void func_8019426C_body(CpuContext* ctx);
extern "C" void func_80194140_body(CpuContext* ctx);
extern "C" void PPC_WriteSpr(uint32_t spr, uint32_t value);             // ppc_helpers.cpp
extern "C" void GxNotifyGuestRamDmaWrite(uint32_t addr, uint32_t size);  // hle/gx/gx_objects.cpp
extern "C" void func_8018ECCC(CpuContext* ctx);  // ResMdl::GetResByteCode(const char*)
extern "C" void func_8018EDF8(CpuContext* ctx);  // ResMdl::GetResNode(u32)
extern "C" void func_801EC1B8(CpuContext* ctx);  // PSMTXConcat (mtx.cpp's native behind its wrapper)
// CalcView_LC's callees, reached through their guarded wrappers as the original reaches them.
extern "C" void func_801EC284(CpuContext* ctx);  // PSMTXConcatArray (mtx.cpp)
extern "C" void func_801EC41C(CpuContext* ctx);  // PSMTXInverse (mtx.cpp)
extern "C" void func_801EC184(CpuContext* ctx);  // PSMTXCopy (mtx.cpp)
extern "C" void func_801EC158(CpuContext* ctx);  // PSMTXIdentity
extern "C" void func_801EC514(CpuContext* ctx);  // PSMTXInvXpose
extern "C" void func_80162798(CpuContext* ctx);  // math::MTX34ToMTX33
extern "C" void func_801627CC(CpuContext* ctx);  // math::MTX34InvTranspose(MTX33*, const MTX34*)
extern "C" void func_801D7AE4(CpuContext* ctx);  // LCQueueWait
extern "C" void func_8015D8BC(CpuContext* ctx);  // ut::LC::StoreBlocks (G3dCalc_LCStoreBlocks below)
// The vanilla (MKW_EABI_REGS) translation's register-passing entries that the originals call; absent from a
// translation without them (P+), whose originals call the wrappers above instead, so weak (null when absent). An
// EABI entry can leave different dead-stack bytes than the context entry: it saves a non-volatile register the
// translation proved dead as 0 (UpdateCameraMtx's r31), where the context entry saves the context's value.
extern "C" __attribute__((weak)) void func_801A81D4_eabi(CpuContext* ctx, uint32_t r1, uint32_t r3);
extern "C" __attribute__((weak)) MkwEabiPair func_803F6150_eabi(CpuContext* ctx, uint32_t r3, uint32_t r4, uint32_t r5);
extern "C" __attribute__((weak)) uint64_t func_801D7AE4_eabi(CpuContext* ctx, uint32_t r3);
// guest_code_patch.cpp's table: every translated function's start, ascending; index = its g_mkwGuestCodePatched byte.
extern "C" const uint32_t g_mkwGuardedFunctionStarts[];
extern "C" const uint32_t g_mkwGuardedFunctionCount;

namespace g3dcalc {
namespace {

// ---------------------------------------------------------------------------------------------------------------
// Switches and counters
// ---------------------------------------------------------------------------------------------------------------
enum Mode : uint8_t { kNative = 1, kOriginal = 0, kVerify = 2 };
enum Fn : uint8_t {
    kSkinning, kCalcViewLC, kKeyFVS32, kKeyFVS48, kKeyFVS96, kMtx44Copy, kCameraMtx, kLightSettingCopy, kLightSlot,
    kMemcpy, kFillMem, kEnvTexMtx, kNumFn
};
const char* const kFnNames[kNumFn] = {"skinning",  "calcview",  "fvs32",     "fvs48",  "fvs96",   "mtx44copy",
                                      "cameramtx", "lightcopy", "lightslot", "memcpy", "fillmem", "envtexmtx"};
// Verify's register check: the ABI state (void functions), or the ABI state plus r3, r4, f1 and f2 (the keyframe
// readers: f1 is their result; the vanilla EABI translation writes back r3, r4 and f2 as well, and its callers reload
// them, so the natives reproduce those too). The other volatile registers are not reproduced: the P+ translation, with
// no EABI contract, writes back r0, r5, r6, f0-f11, CR0 and XER as well, which compiled callers never read.
// kAbiStateR3R4F1: the ABI state plus r3, r4 and f1 (the LightSetting copy: its callee writes back f1, not f2).
enum Check : uint8_t { kAbiState, kAbiStateAndResults, kAbiStateR3R4F1 };
constexpr Check kFnCheck[kNumFn] = {kAbiState,           kAbiState,           kAbiStateAndResults,
                                    kAbiStateAndResults, kAbiStateAndResults, kAbiStateAndResults,
                                    kAbiStateAndResults, kAbiStateR3R4F1, kAbiStateR3R4F1,
                                    kAbiStateAndResults, kAbiStateAndResults, kAbiStateAndResults};
// Verify's dirty-mark check over all of MEM1/MEM2/LC (catching a write outside the declared ranges) costs ~1.4 MB of
// copies per call; the keyframe readers (thousands of calls per frame, writes only to their stack frame) check the
// marks of their declared ranges, and all of them on every kFullMarksEvery-th call; so do the camera natives (a few
// hundred calls per frame).
constexpr bool kFnLocalMarks[kNumFn] = {false, false, true, true, true, true, true, false, true, true, true, true};
// Functions every caller reached through their EABI entry before they became natives (the vanilla translation):
// that entry saves the non-volatile registers it proved dead as 0 (`uint32_t r31 = 0;`) where the context entry
// (func_X_body, verify's reference) saves the context's copies, stale in that translation; callees see the context
// either way. The natives store 0 in those slots as the callers' entry did; verify runs the original as is and then
// zeroes the same slots. Bytes below the entry r1: GetCameraMtx and the light slot setter save r30/r31 at r1-8..r1-1.
constexpr uint32_t kFnEabiDeadSaveBytes[kNumFn] = {0, 0, 0, 0, 0, 0, 8, 0, 8, 0, 0, 0};
// The translation has MKW_EABI_REGS entries (vanilla), detected by one CalcView_LC's original calls.
inline bool EabiTranslation() { return func_801D7AE4_eabi != nullptr; }
constexpr uint64_t kFullMarksEvery = 64;

struct Settings {
    Mode mode = kNative;
    bool off[kNumFn] = {};
    bool calcViewFull = true;      // G3D_CALC_OFF=cvfull clears it
    bool calcViewGeneric = false;  // G3D_CALC_CVGENERIC=1
    uint32_t watch = 0;  // G3D_CALC_WATCH=hex guest address: verify logs that 64-byte block around a mismatch
    Settings() {
        if (const char* v = std::getenv("G3D_CALC"); v != nullptr && v[0] != '\0')
            mode = v[0] == '0' ? kOriginal : v[0] == '2' ? kVerify : kNative;
        if (const char* v = std::getenv("G3D_CALC_WATCH")) watch = static_cast<uint32_t>(std::strtoul(v, nullptr, 16)) & ~63u;
#if MKW_FAST_FLOAT
        // Fast-float builds: with host float operations, these translations cost less per call than the natives
        // (NAT_INSN, NOTE5: GetCameraMtx 202 vs 231, GetEnvironmentTexMtx 154 vs 248, MTX44Copy 104 vs 132), so they
        // run translated by default. G3D_CALC_OFF, when set, replaces this list.
        off[kCameraMtx] = off[kEnvTexMtx] = off[kMtx44Copy] = true;
#endif
        if (const char* v = std::getenv("G3D_CALC_OFF")) {
            const std::string list = std::string(",") + v + ",";
            for (int i = 0; i < kNumFn; ++i)
                off[i] = list.find(std::string(",") + kFnNames[i] + ",") != std::string::npos;
            calcViewFull = list.find(",cvfull,") == std::string::npos;
        }
        if (const char* v = std::getenv("G3D_CALC_CVGENERIC")) calcViewGeneric = v[0] == '1';
    }
};
const Settings& Config() {
    static const Settings settings;
    return settings;
}

struct Counters {
    std::atomic<uint64_t> calls{0}, native{0}, declined{0}, compared{0}, mismatches{0}, undeclared{0}, foreign{0},
        overmarked{0};
};
Counters g_counters[kNumFn];
// A per-call statistics increment: a relaxed load and store, not a locked read-modify-write (two of those per call
// cost ~40 cycles; memcpy alone runs thousands of times a frame). Calls on another thread at the same moment can
// lose a count; these numbers are only logged.
inline void Bump(std::atomic<uint64_t>& counter, uint64_t n = 1) {
    counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------------------------------------------
// Lean memory (memory_access.h): guest address & kLeanMask is the offset into one host window; rollback's dirty marks
// are one byte per 64 guest bytes (BrawlDirtyMemory::leanBlocks); a 64 KiB chunk with a non-zero policy byte sends
// stores to the checked path.
// ---------------------------------------------------------------------------------------------------------------
#if MKW_LEAN_MEMORY
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
inline uint16_t Load16(const uint8_t* host) {
    uint16_t v;
    std::memcpy(&v, host, 2);
    return __builtin_bswap16(v);
}
// Loads read the lean window as FlatLoad does, at any address.
inline uint32_t GuestLoad8(uint32_t address) { return Window()[Canonical(address)]; }
inline uint32_t GuestLoad16(uint32_t address) { return Load16(Window() + Canonical(address)); }
inline uint32_t GuestLoad32(uint32_t address) { return Load32(Window() + Canonical(address)); }
// A plain store's dirty mark: its 64-byte block and the next (BrawlDirtyMemory::LeanMarkBlockPair). None in a build
// without them (MKW_LEAN_DIRTY_MARKS=0: no translated store marks either, and rollback refuses to run).
inline void MarkPair(uint32_t canonical) {
#if MKW_LEAN_DIRTY_MARKS
    const uint16_t both = 0x0101;
    std::memcpy(BrawlDirtyMemory::leanBlocks + (canonical >> 6), &both, sizeof(both));
#else
    (void)canonical;
#endif
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// Verify (G3D_CALC=2)
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
// Spans of dirty-mark blocks. All: MEM1 and MEM2 (canonical 0-0x01800000 and 0x10000000-0x14000000) and the locked
// cache (0x20000000, 16 KiB), where every guest store lands. Local: the declared ranges' blocks, each with the one
// block past its end that a store's mark pair may reach.
struct MarkSpan {
    uint32_t firstBlock, blocks;
};
constexpr MarkSpan kMarkSpans[] = {
    {0x00000000u >> 6, 0x01800000u >> 6}, {0x10000000u >> 6, 0x04000000u >> 6}, {0x20000000u >> 6, 0x4000u >> 6}};
struct MarkSpans {
    const MarkSpan* items;
    int count;
    MarkSpan local[Ranges::kMax];
};
void AllSpans(MarkSpans& out) {
    out.items = kMarkSpans;
    out.count = static_cast<int>(sizeof(kMarkSpans) / sizeof(kMarkSpans[0]));
}
void LocalSpans(const Ranges& ranges, MarkSpans& out) {
    for (int i = 0; i < ranges.count; ++i) {
        const uint32_t c = Canonical(ranges.items[i].address);
        out.local[i] = {c >> 6, ((c + ranges.items[i].size - 1) >> 6) - (c >> 6) + 2};
    }
    out.items = out.local;
    out.count = ranges.count;
}

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
    BrawlDirtyMemory::MarksCleared();
    for (int i = 0; i < spans.count; ++i)
        std::memset(BrawlDirtyMemory::leanBlocks + spans.items[i].firstBlock, 0, spans.items[i].blocks);
}
// marks = a | b | c (overlapping local spans hold the same bytes, so writing them twice is harmless)
void RestoreMarks(const MarkSpans& spans, const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                  const std::vector<uint8_t>& c) {
    BrawlDirtyMemory::MarksCleared();
    size_t at = 0;
    for (int k = 0; k < spans.count; ++k) {
        const MarkSpan& s = spans.items[k];
        uint8_t* const dst = BrawlDirtyMemory::leanBlocks + s.firstBlock;
        for (uint32_t i = 0; i < s.blocks; ++i) dst[i] = a[at + i] | b[at + i] | c[at + i];
        at += s.blocks;
    }
}
// The blocks next to the declared ranges whose bytes Verify captures around the original. A store's mark pair can
// reach the block past a range; with MKW_LEAN_RANGES a lowered segment range marks its span from its lowest access,
// loads included (LeanMarkSpan: two or eight blocks from there, or the exact span), so the original's marks can also
// lie before a declared range or up to seven blocks past it with no write there. Each range gets eight blocks on each
// side, within its MEM1, MEM2 or locked-cache span.
struct Margins {
    MarkSpan items[2 * Ranges::kMax];
    int count = 0;
};
constexpr uint32_t kMarginBlocks = 8;
void RangeMargins(const Ranges& ranges, Margins& out) {
    out.count = 0;
    for (int i = 0; i < ranges.count; ++i) {
        const uint32_t c = Canonical(ranges.items[i].address);
        const uint32_t first = c >> 6, last = (c + ranges.items[i].size - 1) >> 6;
        for (const MarkSpan& s : kMarkSpans) {
            const uint32_t end = s.firstBlock + s.blocks;
            if (first < s.firstBlock || last >= end) continue;
            const uint32_t lo = first >= s.firstBlock + kMarginBlocks ? first - kMarginBlocks : s.firstBlock;
            const uint32_t hi = last + 1 + kMarginBlocks <= end ? last + 1 + kMarginBlocks : end;
            if (lo < first) out.items[out.count++] = {lo, first - lo};
            if (last + 1 < hi) out.items[out.count++] = {last + 1, hi - (last + 1)};
            break;
        }
    }
}
void CaptureMargins(const Margins& margins, std::vector<uint8_t>& out) {
    out.clear();
    for (int i = 0; i < margins.count; ++i) {
        const uint8_t* const host = Window() + (static_cast<size_t>(margins.items[i].firstBlock) << 6);
        out.insert(out.end(), host, host + (static_cast<size_t>(margins.items[i].blocks) << 6));
    }
}
// The first marked block (guest canonical address) that is neither in a declared range nor a margin block whose
// bytes the call left unchanged, or UINT32_MAX.
uint32_t FirstUndeclared(const MarkSpans& spans, const std::vector<uint8_t>& marks, const Ranges& ranges,
                         const Margins& margins, const std::vector<uint8_t>& marginBefore,
                         const std::vector<uint8_t>& marginAfter) {
    size_t at = 0;
    for (int k = 0; k < spans.count; ++k) {
        const MarkSpan& s = spans.items[k];
        for (uint32_t i = 0; i < s.blocks; ++i) {
            if (marks[at + i] == 0) continue;
            const uint32_t block = s.firstBlock + i;
            bool covered = false;
            for (int r = 0; r < ranges.count && !covered; ++r) {
                const uint32_t c = Canonical(ranges.items[r].address);
                covered = block >= (c >> 6) && block <= ((c + ranges.items[r].size - 1) >> 6);
            }
            size_t m = 0;
            for (int j = 0; j < margins.count && !covered; ++j) {
                const MarkSpan& g = margins.items[j];
                if (block >= g.firstBlock && block < g.firstBlock + g.blocks) {
                    const size_t o = m + (static_cast<size_t>(block - g.firstBlock) << 6);
                    covered = std::memcmp(marginBefore.data() + o, marginAfter.data() + o, 64) == 0;
                    if (!covered) break;
                }
                m += static_cast<size_t>(g.blocks) << 6;
            }
            if (!covered) return block << 6;
        }
        at += s.blocks;
    }
    return UINT32_MAX;
}
// The mark of `block` in a span set's saved marks (false outside the spans).
bool MarkSet(const MarkSpans& spans, const std::vector<uint8_t>& marks, uint32_t block) {
    size_t at = 0;
    for (int s = 0; s < spans.count; ++s) {
        const MarkSpan& span = spans.items[s];
        if (block >= span.firstBlock && block < span.firstBlock + span.blocks) return marks[at + block - span.firstBlock] != 0;
        at += span.blocks;
    }
    return false;
}
// A translation with MKW_LEAN_RANGES marks a lowered segment range's span at once (memory_access.h LeanMarkSpan: the
// blocks from its start, up to eight), where the natives mark each store's 64-byte block pair. The two can differ both
// ways: the span can cover blocks the function never wrote, and a store pair's second block can lie past the span's
// last block (a 32-byte fill at 0x...938 marks 0x...980; its span marks 0x...900 and 0x...940). Rollback needs every
// written block marked, not the same extra marks: the native passes when its marks the original lacks lie in the
// declared ranges (plus the one-block spill a pair can add) and its marks cover every block of the declared ranges
// whose bytes it changed (the bytes themselves were compared already).
bool NativeMarksCover(const MarkSpans& spans, const std::vector<uint8_t>& nativeMarks,
                      const std::vector<uint8_t>& origMarks, const Ranges& ranges, const std::vector<uint8_t>& before,
                      const std::vector<uint8_t>& after) {
    size_t at = 0;
    for (int s = 0; s < spans.count; ++s) {
        for (uint32_t i = 0; i < spans.items[s].blocks; ++i) {
            if (nativeMarks[at + i] == 0 || origMarks[at + i] != 0) continue;
            const uint32_t block = spans.items[s].firstBlock + i;
            bool declared = false;
            for (int r = 0; r < ranges.count && !declared; ++r) {
                const uint32_t c = Canonical(ranges.items[r].address);
                declared = block >= (c >> 6) && block <= ((c + ranges.items[r].size - 1) >> 6) + 1;
            }
            if (!declared) return false;
        }
        at += spans.items[s].blocks;
    }
    at = 0;
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
    BrawlDirtyMemory::MarksCleared();
    size_t at = 0;
    for (int i = 0; i < ranges.count; ++i) {
        std::memcpy(Window() + Canonical(ranges.items[i].address), in.data() + at, ranges.items[i].size);
        at += ranges.items[i].size;
    }
}

// What the natives must leave as the original does: r1, r2, r13-r31, f14-f31 (both lanes), LR, CR fields 2-4.
bool SameAbiState(const CpuContext& a, const CpuContext& b, char* what, size_t whatSize) {
    const uint32_t gprs[] = {1, 2};
    for (uint32_t r : gprs)
        if (a.gpr[r] != b.gpr[r]) return std::snprintf(what, whatSize, "r%u %08X/%08X", r, a.gpr[r], b.gpr[r]), false;
    for (uint32_t r = 13; r < 32; ++r)
        if (a.gpr[r] != b.gpr[r]) return std::snprintf(what, whatSize, "r%u %08X/%08X", r, a.gpr[r], b.gpr[r]), false;
    for (uint32_t f = 14; f < 32; ++f)
        if (a.fpr[f].raw != b.fpr[f].raw)
            return std::snprintf(what, whatSize, "f%u %016llX/%016llX", f, static_cast<unsigned long long>(a.fpr[f].raw),
                                 static_cast<unsigned long long>(b.fpr[f].raw)),
                   false;
    if (a.lr != b.lr) return std::snprintf(what, whatSize, "lr %08X/%08X", a.lr, b.lr), false;
    if ((PpcGetCr(&a) & 0x00FFF000u) != (PpcGetCr(&b) & 0x00FFF000u))
        return std::snprintf(what, whatSize, "cr2-4 %08X/%08X", PpcGetCr(&a), PpcGetCr(&b)), false;
    return true;
}

// The ABI state plus the keyframe readers' result registers r3, r4, f1, f2.
bool SameAbiStateAndResults(const CpuContext& a, const CpuContext& b, char* what, size_t whatSize) {
    if (!SameAbiState(a, b, what, whatSize)) return false;
    for (uint32_t r : {3u, 4u})
        if (a.gpr[r] != b.gpr[r]) return std::snprintf(what, whatSize, "r%u %08X/%08X", r, a.gpr[r], b.gpr[r]), false;
    for (uint32_t f : {1u, 2u})
        if (a.fpr[f].raw != b.fpr[f].raw)
            return std::snprintf(what, whatSize, "f%u %016llX/%016llX", f, static_cast<unsigned long long>(a.fpr[f].raw),
                                 static_cast<unsigned long long>(b.fpr[f].raw)),
                   false;
    return true;
}

// The ABI state plus r3, r4 and f1.
bool SameAbiStateR3R4F1(const CpuContext& a, const CpuContext& b, char* what, size_t whatSize) {
    if (!SameAbiState(a, b, what, whatSize)) return false;
    for (uint32_t r : {3u, 4u})
        if (a.gpr[r] != b.gpr[r]) return std::snprintf(what, whatSize, "r%u %08X/%08X", r, a.gpr[r], b.gpr[r]), false;
    if (a.fpr[1].raw != b.fpr[1].raw)
        return std::snprintf(what, whatSize, "f1 %016llX/%016llX", static_cast<unsigned long long>(a.fpr[1].raw),
                             static_cast<unsigned long long>(b.fpr[1].raw)),
               false;
    return true;
}

// NAT_INSN=1 (insn_count.h): a compared call's original and native instructions, sites "g3d.<name>". A declined
// native costs its attempt plus the original it falls back to, as in the native mode.
int FnSite(Fn fn) {
    static const std::array<int, kNumFn> sites = [] {
        std::array<int, kNumFn> s{};
        for (size_t i = 0; i < s.size(); ++i) s[i] = InsnCount::Site((std::string("g3d.") + kFnNames[i]).c_str());
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

// Per native, so the first natives' lines cannot use up the log.
std::atomic<int> g_verifyLogs[kNumFn] = {};
constexpr int kMaxVerifyLogs = 8;

// Original, then native, from the same state; the original's results stay. `writes` declares the guest ranges the
// function may write (the stack window is added here); `native` returns false where it declines (no effect).
template <typename Writes, typename Native>
void Verify(Fn fn, CpuContext* ctx, void (*body)(CpuContext*), Writes writes, Native native) {
    thread_local std::vector<uint8_t> savedMarks, origMarks, nativeMarks, before, afterOrig, afterNative,
        marginBefore, marginAfter;
    Counters& n = g_counters[fn];
    Ranges ranges;
    if (!LeanReady() || !writes(*ctx, ranges) || !ranges.Add(ctx->gpr[1] - 0x800u, 0x808u)) {
        n.declined.fetch_add(1, std::memory_order_relaxed);
        body(ctx);
        return;
    }
    const CpuContext in = *ctx;
    MarkSpans spans;
    if (kFnLocalMarks[fn] && n.calls.load(std::memory_order_relaxed) % kFullMarksEvery != 1)
        LocalSpans(ranges, spans);
    else
        AllSpans(spans);
    // The AX mix worker stores each AI frame into guest memory (with its marks) from its own thread, and only this
    // thread starts a mix: idle, it cannot land inside the windows below.
    AxDspHle::JoinMixWorker();
    SaveMarks(spans, savedMarks);
    ClearMarks(spans);
    Capture(ranges, before);
    Margins margins;
    RangeMargins(ranges, margins);
    CaptureMargins(margins, marginBefore);
    const uint32_t watch = Config().watch;
    uint8_t watched[3][64] = {};
    if (watch != 0) std::memcpy(watched[0], Window() + Canonical(watch), 64);
    const uint64_t originalStart = InsnCount::On() ? InsnCount::Read() : 0;
    body(ctx);
    const uint64_t originalInsns = InsnCount::On() ? InsnCount::Since(originalStart) : 0;
    if (const uint32_t dead = EabiTranslation() ? kFnEabiDeadSaveBytes[fn] : 0u; dead != 0)
        std::memset(Window() + Canonical(in.gpr[1] - dead), 0, dead);  // (inside the stack window; already marked)
    if (watch != 0) std::memcpy(watched[1], Window() + Canonical(watch), 64);
    const CpuContext orig = *ctx;
    SaveMarks(spans, origMarks);
    Capture(ranges, afterOrig);
    CaptureMargins(margins, marginAfter);
    const uint32_t undeclared = FirstUndeclared(spans, origMarks, ranges, margins, marginBefore, marginAfter);
    if (undeclared != UINT32_MAX) {
        n.undeclared.fetch_add(1, std::memory_order_relaxed);
        if (g_verifyLogs[fn].fetch_add(1, std::memory_order_relaxed) < kMaxVerifyLogs)
            RT_LOGF(RT_TAG_GX, "[g3dcalc] verify %s: the original wrote outside the declared ranges (canonical %08X)\n",
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
    if (watch != 0) std::memcpy(watched[2], Window() + Canonical(watch), 64);
    char what[160] = "";
    bool same = kFnCheck[fn] == kAbiStateAndResults ? SameAbiStateAndResults(orig, *ctx, what, sizeof(what))
              : kFnCheck[fn] == kAbiStateR3R4F1     ? SameAbiStateR3R4F1(orig, *ctx, what, sizeof(what))
                                                    : SameAbiState(orig, *ctx, what, sizeof(what));
    if (same && afterNative != afterOrig) {
        same = false;
        size_t at = 0;
        for (int i = 0; i < ranges.count && what[0] == '\0'; ++i) {
            for (uint32_t k = 0; k < ranges.items[i].size; ++k)
                if (afterNative[at + k] != afterOrig[at + k]) {
                    const uint32_t w = k & ~3u;
                    std::snprintf(what, sizeof(what), "mem %08X: original %08X native %08X",
                                  ranges.items[i].address + w, Load32(afterOrig.data() + at + w),
                                  Load32(afterNative.data() + at + w));
                    break;
                }
            at += ranges.items[i].size;
        }
    }
    if (same && nativeMarks != origMarks && NativeMarksCover(spans, nativeMarks, origMarks, ranges, before, afterNative)) {
        Bump(n.overmarked);  // the marks differ (MKW_LEAN_RANGES segment spans); the native's cover its writes
    } else if (same && nativeMarks != origMarks) {
        same = false;
        size_t at = 0;
        // Every mark the native set and the original did not lies outside the declared ranges (the original's own
        // extra marks, MKW_LEAN_RANGES spans, pass NativeMarksCover and do not count here).
        bool outsideOnly = true;
        for (int s = 0; s < spans.count; ++s) {
            for (uint32_t i = 0; i < spans.items[s].blocks; ++i) {
                if (nativeMarks[at + i] == 0 || origMarks[at + i] != 0) continue;
                const uint32_t block = spans.items[s].firstBlock + i;
                if (what[0] == '\0')
                    std::snprintf(what, sizeof(what), "dirty mark of canonical %08X: original 0 native %u", block << 6,
                                  nativeMarks[at + i]);
                for (int r = 0; r < ranges.count && outsideOnly; ++r) {
                    const uint32_t c = Canonical(ranges.items[r].address);
                    outsideOnly = block < (c >> 6) || block > ((c + ranges.items[r].size - 1) >> 6) + 1;
                }
            }
            at += spans.items[s].blocks;
        }
        if (what[0] == '\0') std::snprintf(what, sizeof(what), "a block whose bytes changed left unmarked");
        // Only there: another host thread's store landing in the native's window (a DVD read, say), unless a
        // second native run from the same state repeats it. The marks it left stay (nativeMarks).
        if (outsideOnly) {
            thread_local std::vector<uint8_t> rerunMarks;
            Restore(ranges, before);
            ClearMarks(spans);
            CpuContext rerun = in;
            native(&rerun);
            SaveMarks(spans, rerunMarks);
            if (rerunMarks == origMarks || NativeMarksCover(spans, rerunMarks, origMarks, ranges, before, afterNative)) {
                same = true;
                n.foreign.fetch_add(1, std::memory_order_relaxed);
                if (g_verifyLogs[fn].fetch_add(1, std::memory_order_relaxed) < kMaxVerifyLogs)
                    RT_LOGF(RT_TAG_GX, "[g3dcalc] verify %s: a mark outside the declared ranges, not repeated by a "
                            "second native run (another thread's store): %s\n", kFnNames[fn], what);
                what[0] = '\0';
            }
            for (size_t k = 0; k < nativeMarks.size(); ++k) nativeMarks[k] |= rerunMarks[k];
        }
    }
    n.compared.fetch_add(1, std::memory_order_relaxed);
    if (!same) {
        n.mismatches.fetch_add(1, std::memory_order_relaxed);
        if (g_verifyLogs[fn].fetch_add(1, std::memory_order_relaxed) < kMaxVerifyLogs) {
            RT_LOGF(RT_TAG_GX, "[g3dcalc] verify MISMATCH %s: r1=%08X r3=%08X r4=%08X r5=%08X r6=%08X lr=%08X: %s\n",
                    kFnNames[fn], in.gpr[1], in.gpr[3], in.gpr[4], in.gpr[5], in.gpr[6], in.lr, what);
            for (int k = 0; watch != 0 && k < 3; ++k) {
                char hex[64 * 2 + 16 + 1];
                int at = 0;
                for (int i = 0; i < 64; ++i)
                    at += std::snprintf(hex + at, sizeof(hex) - at, (i & 3) == 3 && i != 63 ? "%02X " : "%02X",
                                        watched[k][i]);
                RT_LOGF(RT_TAG_GX, "[g3dcalc]   watch %08X %s: %s\n", watch,
                        k == 0 ? "before" : k == 1 ? "after original" : "after native", hex);
            }
        }
    }
    *ctx = orig;
    Restore(ranges, afterOrig);
    RestoreMarks(spans, savedMarks, origMarks, nativeMarks);
}
#endif  // MKW_LEAN_MEMORY

// Dispatch for one native: counts, G3D_CALC / G3D_CALC_OFF, verify.
template <typename Writes, typename Native>
[[gnu::always_inline]] inline void Run(Fn fn, CpuContext* ctx, void (*body)(CpuContext*), Writes writes,
                                       Native native) {
    const Settings& s = Config();
    Bump(g_counters[fn].calls);
#if MKW_LEAN_MEMORY
    if (s.mode == kNative && !s.off[fn]) {
        if (InsnCount::On()) [[unlikely]] {  // the whole call into g3d.<name>'s native column
            const uint64_t start = InsnCount::Read();
            const bool ranNatively = native(ctx);
            if (!ranNatively) body(ctx);
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
        Verify(fn, ctx, body, writes, native);
        return;
    }
#else
    (void)s;
    (void)writes;
    (void)native;
#endif
    body(ctx);
}

#if MKW_LEAN_MEMORY
// ---------------------------------------------------------------------------------------------------------------
// Frames: the prologue and epilogue stores of the originals (stwu, mflr/stw, the stmw-style save helpers), through
// the same accessors, so the stack bytes and their dirty marks match.
// ---------------------------------------------------------------------------------------------------------------
// stwu r1,-size(r1); stw r0(lr),size+4(r1); and the saves of r(first)..r31 ending at newR1+size-4, written as the
// _savegpr helpers write them (pairs from the lowest register, a single word last when the count is odd).
inline bool Prologue(CpuContext* ctx, uint32_t size, uint32_t firstSaved, uint32_t& newR1) {
    const uint32_t r1 = ctx->gpr[1];
    newR1 = r1 - size;
    uint8_t* const frame = MemoryInline::ResolveRangeHost(newR1, 0, size + 8u, true);
    if (frame == nullptr) return false;
    MemoryInline::WriteResolved32(frame, 0u, newR1, r1);
    MemoryInline::WriteResolved32(frame, size + 4u, newR1 + size + 4u, ctx->lr);
    uint32_t offset = size - 4u * (32u - firstSaved);
    uint32_t reg = firstSaved;
    for (; reg + 1 < 32; reg += 2, offset += 8)
        if (!MemoryInline::WriteResolvedPair32(
                frame, offset, (static_cast<uint64_t>(ctx->gpr[reg]) << 32) | ctx->gpr[reg + 1])) {
            MemoryInline::WriteResolved32(frame, offset, newR1 + offset, ctx->gpr[reg]);
            MemoryInline::WriteResolved32(frame, offset + 4u, newR1 + offset + 4u, ctx->gpr[reg + 1]);
        }
    if (reg == 31) MemoryInline::WriteResolved32(frame, offset, newR1 + offset, ctx->gpr[31]);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// CalcSkinning (0x801A5414): ogws g3d_calcworld.cpp.
//   r3 = MTX34* pModelMtxArray, r4 = u32* pModelMtxAttribArray, r5 = const ResMdl* (by reference), r6 = NodeMix
//   byte code (0: looked up by name). WEIGHT (3): target u16, count u8, then count x {mtx u16, weight f32}: the
//   target becomes the weighted sum of the skinning temporaries (ps_madds0 per pair, in entry order) and its attribute
//   the root attribute ANDed with each entry's. EVPMTX (anything else but END 1): mtx u16, node u16: skin[mtx] =
//   model[mtx] x node.invModelMtx (PSMTXConcat).
// The original: a 48-byte frame saving r25-r31, each weight stored to 8(r1) and reloaded with lfs, the attribute
// stored after each entry, GetResByteCode (r4 = r13-23336, "NodeMix") and GetResNode / PSMTXConcat through their
// wrappers (return addresses 0x801A544C, 0x801A56B4, 0x801A56C8).
// ---------------------------------------------------------------------------------------------------------------
constexpr uint32_t kSkinTemporary = 0x804A9A80u;  // detail::workmem::GetSkinningMtxTemporary (lis 0x804B, -25984)
constexpr uint32_t kRootAttrib = 0xF0000000u;     // detail::WorldMtxAttr::GetRootMtxAttr

using F4 = float __attribute__((ext_vector_type(4)));
using I4 = int32_t __attribute__((ext_vector_type(4)));

// Four big-endian words, as one byte shuffle (each word's bytes reversed).
using RowBytes = uint8_t __attribute__((ext_vector_type(16)));
inline RowBytes SwapRowWords(RowBytes b) {
    return __builtin_shufflevector(b, b, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
}
inline F4 LoadRow(const uint8_t* host) {
    RowBytes b;
    std::memcpy(&b, host, 16);
    return __builtin_bit_cast(F4, SwapRowWords(b));
}
inline void StoreRow(uint8_t* host, F4 v) {
    const RowBytes b = SwapRowWords(__builtin_bit_cast(RowBytes, v));
    std::memcpy(host, &b, 16);
}

// One WEIGHT command's matrix (entries at `entries`, count n) as the original computes it, from the skinning
// temporaries: 4-wide FMAs (one ps_madds0 pair per half row; identical while no lane is NaN), else the translated
// helpers' sequence. The byte code was validated (ValidSkinningByteCode).
void BlendMatrix(CpuContext* ctx, const uint8_t* entries, uint32_t n, uint32_t zeroBits, F4 out[3]) {
    uint8_t* const m = Window();
    const float zero = __builtin_bit_cast(float, zeroBits);
    F4 a0 = zero, a1 = zero, a2 = zero;
    // Any NaN among the inputs, collected lane-wise and tested once (each entry's horizontal test cost more than its
    // FMAs); the weight's own test as a scalar.
    I4 nanLanes = 0;
    bool nanWeight = false;
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* const e = entries + 6u * i;
        const uint32_t id = (uint32_t(e[0]) << 8) + e[1];
        const float w = __builtin_bit_cast(float, Load32(e + 2));
        const uint8_t* const t = m + Canonical(kSkinTemporary + 48u * id);
        const F4 t0 = LoadRow(t), t1 = LoadRow(t + 16), t2 = LoadRow(t + 32);
        nanLanes |= (t0 != t0) | (t1 != t1) | (t2 != t2);
        nanWeight = nanWeight || w != w;
        const F4 ws = w;
        a0 = __builtin_elementwise_fma(t0, ws, a0);
        a1 = __builtin_elementwise_fma(t1, ws, a1);
        a2 = __builtin_elementwise_fma(t2, ws, a2);
    }
    nanLanes |= (a0 != a0) | (a1 != a1) | (a2 != a2);
    if (!nanWeight && (nanLanes[0] | nanLanes[1] | nanLanes[2] | nanLanes[3]) == 0) {
        out[0] = a0;
        out[1] = a1;
        out[2] = a2;
        return;
    }
    // The translated sequence: psq_l pairs (GQR0 is the float encoding here), lfs of the weight word, ps_madds0.
    const double z = PPC_PsMerge00Inline(PPC_PsFromScalarInline(static_cast<double>(zero)),
                                         PPC_PsFromScalarInline(static_cast<double>(zero)));
    double acc[6] = {z, z, z, z, z, z};
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t* const e = entries + 6u * i;
        const uint32_t id = (uint32_t(e[0]) << 8) + e[1];
        const uint32_t t = kSkinTemporary + 48u * id;
        const double r = PPC_PsFromScalarInline(static_cast<double>(__builtin_bit_cast(float, Load32(e + 2))));
        for (uint32_t k = 0; k < 6; ++k)
            acc[k] = PPC_PsMadds0Inline(PPC_PsqLGqrInline<0u, 0u>(ctx, 0u, t + 8u * k), r, acc[k]);
    }
    for (uint32_t row = 0; row < 3; ++row) {
        PPC_FPR lo, hi;
        lo.d = acc[2 * row];
        hi.d = acc[2 * row + 1];
        out[row] = F4{PpcGetPs0Inline(lo.d), PpcGetPs1Inline(lo.d), PpcGetPs0Inline(hi.d), PpcGetPs1Inline(hi.d)};
    }
}

// The byte code's commands all within the model's matrices, and the whole byte code in one view of the window (the
// native then has no reason to stop half way, and reads it at Canonical(address) + offset). One walk with plain
// window reads (loads need no store policy), the span checked once at the end; the walk stops before it could read
// past the window's end. (Was a PlainRange per command, store-policy loop included: ~0.5% of the game thread.)
bool ValidSkinningByteCode(const uint8_t* m, uint32_t byteCode, uint32_t mtxCount) {
    constexpr uint32_t kLongestCommand = 4u + 6u * 255u;
    const uint32_t start = byteCode;
    for (uint32_t steps = 0; steps < 0x10000u; ++steps) {
        if (Canonical(byteCode) > MemoryInline::kLeanWindowSize - kLongestCommand) return false;
        const uint8_t* const p = m + Canonical(byteCode);
        if (p[0] == 1) return OneView(start, byteCode + 4u - start);
        if (p[0] == 3) {
            const uint32_t n = p[3];
            if ((uint32_t(p[1]) << 8) + p[2] >= mtxCount) return false;
            for (uint32_t i = 0; i < n; ++i)
                if ((uint32_t(p[4 + 6 * i]) << 8) + p[5 + 6 * i] >= mtxCount) return false;
            byteCode += 4u + 6u * n;
        } else {
            if ((uint32_t(p[1]) << 8) + p[2] >= mtxCount) return false;
            byteCode += 5u;
        }
    }
    return false;
}

// Defined with CalcView_LC below (callee guards, GetResNode's lookup); CalcSkinning's EVPMTX path uses them too.
int32_t GuardIndex(uint32_t address);
inline bool CalleeUnpatched(int32_t guard);
inline uint32_t ResNode(uint32_t mdlRef, uint32_t idx);

// NAT_INSN=1 NAT_INSN_PHASES=1: the phases of CalcSkinning and CalcView_LC as sites of their own. Only on request:
// each phase site's reads and add (~150 instructions) count inside the whole call's g3d.<name> count, ~200 a
// skinning call (~30K instructions) and ~9 a CalcView call.
bool PhaseSites() {
    const char* v = std::getenv("NAT_INSN_PHASES");
    return InsnCount::On() && v != nullptr && v[0] == '1';
}

// CalcSkinningNative's phases, native column: its weighted blends, its EVPMTX products (node lookup and PSMTXConcat),
// its byte-code check.
const std::array<int, 3> kSkinSites = [] {
    std::array<int, 3> s{-1, -1, -1};
    if (!PhaseSites()) return s;
    const char* const names[3] = {"g3d.skin.blend", "g3d.skin.evpmtx", "g3d.skin.check"};
    for (size_t i = 0; i < s.size(); ++i) s[i] = InsnCount::Site(names[i]);
    return s;
}();

bool CalcSkinningNative(CpuContext* ctx) {
    if (ctx->gqr[0] != 0 || !LeanReady()) return false;
    const uint32_t modelArray = ctx->gpr[3], attribArray = ctx->gpr[4], mdlRef = ctx->gpr[5];
    uint32_t byteCode = ctx->gpr[6];
    const uint32_t r1 = ctx->gpr[1], lr = ctx->lr;
    uint8_t* const m = Window();
    const uint32_t mdl = MemoryInline::FlatRead32(mdlRef);
    const uint32_t mtxCount = MemoryInline::FlatRead32(mdl + 0x5Cu);  // ResMdlInfo numViewMtx
    if (mtxCount == 0 || mtxCount > 0x2000u) return false;
    // Every store's range: the arrays, the skinning temporaries and the frame (callees write theirs below it).
    if (!PlainRange(modelArray, 48u * mtxCount) || !PlainRange(attribArray, 4u * mtxCount) ||
        !PlainRange(kSkinTemporary, 48u * mtxCount) || !PlainRange(r1 - 0x400u, 0x408u))
        return false;
    const uint32_t zeroBits = MemoryInline::FlatRead32(ctx->gpr[2] - 24532u);  // lfs f0, -24532(r2): 0.0f

    uint32_t frame;
    if (!Prologue(ctx, 48u, 25u, frame)) return false;
    ctx->gpr[1] = frame;
    if (byteCode == 0) {
        ctx->gpr[3] = mdlRef;
        ctx->gpr[4] = ctx->gpr[13] - 23336u;  // "NodeMix"
        ctx->lr = 0x801A544Cu;
        func_8018ECCC(ctx);
        byteCode = ctx->gpr[3];
    }
    // Declining from here repeats only the frame stores and the lookup above, byte for byte.
    std::optional<InsnCount::Scope> checkPhase;
    if (InsnCount::On()) checkPhase.emplace(kSkinSites[2], InsnCount::kNative);
    const bool valid = byteCode == 0 || ValidSkinningByteCode(m, byteCode, mtxCount);
    checkPhase.reset();
    if (!valid) {
        ctx->gpr[1] = r1;
        ctx->lr = lr;
        return false;
    }
    bool weighted = false;
    uint32_t lastWeight = 0;
    while (byteCode != 0) {
        const uint8_t* const p = m + Canonical(byteCode);
        const uint8_t c = p[0];
        if (c == 1) break;  // END
        if (c == 3) {        // WEIGHT
            const uint32_t target = (uint32_t(p[1]) << 8) + p[2];
            const uint32_t n = p[3];
            F4 rows[3];
            {
                InsnCount::Scope phase(kSkinSites[0], InsnCount::kNative);
                BlendMatrix(ctx, p + 4, n, zeroBits, rows);
            }
            uint32_t attrib = kRootAttrib;
            uint8_t* const attribHost = m + Canonical(attribArray);
            uint8_t* const slot = attribHost + 4u * target;
            // The original stores the running value to the target's slot after each entry (an entry naming the target
            // reads it back); only the last store stays. Kept in a register and stored once, unless the slot overlaps
            // this command's entries, which those stores would change before they are read.
            if (slot + 4 <= p + 4 || slot >= p + 4 + 6u * n) {
                for (uint32_t i = 0; i < n; ++i) {
                    const uint8_t* const e = p + 4 + 6u * i;
                    const uint32_t id = (uint32_t(e[0]) << 8) + e[1];
                    attrib &= id == target ? attrib : Load32(attribHost + 4u * id);
                    lastWeight = Load32(e + 2);
                }
                Store32(slot, attrib);
            } else {
                Store32(slot, attrib);
                for (uint32_t i = 0; i < n; ++i) {
                    const uint8_t* const e = p + 4 + 6u * i;
                    const uint32_t id = (uint32_t(e[0]) << 8) + e[1];
                    attrib &= Load32(attribHost + 4u * id);  // the target's own slot reads the running value
                    Store32(slot, attrib);
                    lastWeight = Load32(e + 2);
                }
            }
            MarkPair(Canonical(attribArray + 4u * target));
            weighted = weighted || n != 0;
            const uint32_t dst = Canonical(modelArray + 48u * target);
            StoreRow(m + dst, rows[0]);
            StoreRow(m + dst + 16, rows[1]);
            StoreRow(m + dst + 32, rows[2]);
            MarkPair(dst);
            MarkPair(dst + 40u);
            byteCode += 4u + 6u * n;
        } else {  // EVPMTX
            InsnCount::Scope phase(kSkinSites[1], InsnCount::kNative);
            const uint32_t mtx = (uint32_t(p[1]) << 8) + p[2];
            const uint32_t node = (uint32_t(p[3]) << 8) + p[4];
            // GetResNode (0x8018EDF8): a leaf that stores nothing and writes back only r3 on every path, which ResNode
            // computes (the same loads and tests; CalcView's billboard pass uses it too); r3, r4 and LR are set
            // again for PSMTXConcat below. A patched GetResNode runs through its wrapper.
            static const int32_t kGuardGetResNode = GuardIndex(0x8018EDF8u);
            if (CalleeUnpatched(kGuardGetResNode)) {
                ctx->gpr[3] = ResNode(mdlRef, node);
            } else {
                ctx->gpr[3] = mdlRef;
                ctx->gpr[4] = node;
                ctx->lr = 0x801A56B4u;
                func_8018EDF8(ctx);
            }
            ctx->gpr[4] = ctx->gpr[3] + 160u;  // ResNodeData invModelMtx
            ctx->gpr[3] = modelArray + 48u * mtx;
            ctx->gpr[5] = kSkinTemporary + 48u * mtx;
            ctx->lr = 0x801A56C8u;
            func_801EC1B8(ctx);
            byteCode += 5u;
        }
    }
    // The weight words the original stored at 8(r1) for each lfs: the last one stays.
    if (weighted) {
        Store32(m + Canonical(frame + 8u), lastWeight);
        MarkPair(Canonical(frame + 8u));
    }
    ctx->gpr[1] = r1;
    ctx->lr = lr;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// CalcView_LC (0x801A3D68): ogws g3d_calcview.cpp, the view, normal and texture matrices of a model in batches of 40
// through the locked cache, each batch stored to the arrays with ut::LC::StoreBlocks (0x8015D8BC: mtspr DMA_U,
// DMA_L; the runtime's DMA_L write copies the blocks, marks them dirty and calls GxNotifyGuestRamDmaWrite).
//   r3 = view positions, r4 = view normals (0: none), r5 = model matrices, r6 = their attributes, r7 = matrix count,
//   r8 = view matrix, r9 = ResMdl (by reference), r10 = view texture matrices (0: none, and unused without normals).
//
// StoreBlocks notifications: the function's first calls are DCInvalidateRange over the three arrays (r3 and r10 over
// align32(count * 48), r4 over align32(count * 36)), whose HLE (DcRangeOp) already calls GxNotifyGuestRamDmaWrite
// over them; each later StoreBlocks writes inside those arrays, and nothing in between reads the caches that
// notification resets (texture and TLUT uploads, EFB-copy sources, display-list generations; idempotent, and the
// generations are only compared for equality). So within the call, StoreBlocks does its copy, marks and SPR writes
// as the runtime does, and only queues the notification; at the end, a queued range inside one of the three arrays
// is dropped (already notified), any other is sent. ~0.3k cycles per StoreBlocks call were three mutex round trips.
// ---------------------------------------------------------------------------------------------------------------
struct DmaRange {
    uint32_t start, end;
};
thread_local bool t_deferDmaNotify = false;
thread_local DmaRange t_pendingDma[16];
thread_local int t_pendingDmaCount = 0;

void QueueDmaNotify(uint32_t addr, uint32_t size) {
    for (int i = 0; i < t_pendingDmaCount; ++i) {
        DmaRange& r = t_pendingDma[i];
        if (r.end == addr) { r.end = addr + size; return; }
        if (r.start == addr + size) { r.start = addr; return; }
    }
    if (t_pendingDmaCount == 16) {  // (not seen: 3 arrays) send the oldest
        GxNotifyGuestRamDmaWrite(t_pendingDma[0].start, t_pendingDma[0].end - t_pendingDma[0].start);
        std::memmove(t_pendingDma, t_pendingDma + 1, sizeof(DmaRange) * 15);
        --t_pendingDmaCount;
    }
    t_pendingDma[t_pendingDmaCount++] = {addr, addr + size};
}

// The notifications of one CalcView_LC call (entry r3, r4, r7, r10), deferred from construction to Send().
class DeferredDmaNotify {
public:
    DeferredDmaNotify(uint32_t viewPos, uint32_t viewNrm, uint32_t count, uint32_t viewTex)
        : count_(count),
          arrays_{{viewPos, viewPos + ((count * 48u + 31u) & ~31u)},
                  {viewNrm, viewNrm + ((count * 36u + 31u) & ~31u)},
                  {viewTex, viewTex + ((count * 48u + 31u) & ~31u)}},
          hasArray_{true, viewNrm != 0, viewNrm != 0 && viewTex != 0} {
        t_pendingDmaCount = 0;
        t_deferDmaNotify = true;
    }
    void Send() {
        t_deferDmaNotify = false;
        for (int i = 0; i < t_pendingDmaCount; ++i) {
            const DmaRange& r = t_pendingDma[i];
            bool covered = false;
            // (DcRangeOp notifies only ranges inside guest memory, Memory::Contains)
            for (int a = 0; a < 3 && !covered; ++a)
                covered = hasArray_[a] && count_ != 0 && r.start >= arrays_[a].start && r.end <= arrays_[a].end &&
                          Memory::Contains(arrays_[a].start, arrays_[a].end - arrays_[a].start);
            if (!covered) GxNotifyGuestRamDmaWrite(r.start, r.end - r.start);
        }
        t_pendingDmaCount = 0;
    }

private:
    uint32_t count_;
    DmaRange arrays_[3];
    bool hasArray_[3];
};

// ---------------------------------------------------------------------------------------------------------------
// CalcView_LC's body, native. The translated original spends ~4% of vanilla Training's game-thread instructions in
// its own body: psq_l/psq_st pairs (each a policy test, byte swaps, the NaN/subnormal lane rules and a dirty mark)
// for the normal and texture matrices, and a GetResNode call per matrix for the billboard-parent test.
//
// The native is the original's instruction sequence (0x801A3D68-0x801A4200) with the same calls in the same order,
// through the callees' guarded wrappers with the original's r1 and return addresses: DCInvalidateRange, LCQueueWait,
// PSMTXConcatArray / PSMTXConcat (the view matrices), the billboard functions (bbFunc[], indirect), PSMTXInverse,
// StoreBlocks, and for non-uniformly scaled matrices PSMTXInvXpose and MTX34InvTranspose. So every callee frame below
// r1 holds the same bytes, and the locked cache and the arrays get the same values. It evaluates itself what touches
// no stack:
//   - GetResNode (0x8018EDF8, a frameless leaf: two loads and the dictionary entry), which it declines to replace
//     while that function's code is patched (the original would reach the patched code through its wrapper);
//   - the leaves the original inlines (PSMTXCopy, MTX34ToMTX33, PSMTXIdentity, GetBillboardMtxTemporary), covered by
//     the function's own guard (a patched inlined leaf sends its callers to the interpreter, guest_code_patch.cpp);
//   - the uniformly scaled matrix's texture copy and normal matrix: psq_l then psq_st of each pair with GQR0 the
//     float encoding is, per lane, the psq_st rule (a subnormal becomes its signed zero, a NaN is quieted; psq_l's
//     quieting is the same rule's half), so three 16-byte row loads with that rule and the stores give the same
//     bytes; the texture matrix's translation column is f31 (lfs of r2-24544) stored with stfs's conversion. The
//     stores' dirty marks are one 64-byte block pair per store start: per matrix, the pairs of its first and last
//     store start (under 64 bytes apart) cover exactly that union.
// If a billboard function leaves GQR0 not the float encoding, the rest of the call takes the generic path: those
// leaves through their wrappers (which honour the live GQR0) and stfs through FlatWriteFloat32.
// Its own frame: the prologue's stores as the translation makes them (back chain, LR save, f31 by stfd and psq_st,
// _savegpr_14), and each local slot's final value (one store each: a slot's dirty mark does not depend on how often
// it was written). Registers: r1, LR, the non-volatile registers and CR fields 2-4 as the original leaves them
// (volatile registers are not reproduced, as for the other void natives). CR and CTR take the original's values at
// the calls it hands them to (the billboard functions, MTX34InvTranspose). Declines (before any effect): GQR0 not
// the float encoding, the locked cache not plain lean memory, GetResNode / MTX34ToMTX33 / PSMTXCopy patched.
// ---------------------------------------------------------------------------------------------------------------
constexpr uint32_t kLcBase = 0xE0000000u;          // ut::LC::GetBase (MtxCacheMap: cur view/nrm/tex at +2/4/6 KiB,
constexpr uint32_t kBbTemporary = 0x804A9A80u;     // previous ones at +10/12/14 KiB); GetBillboardMtxTemporary
constexpr uint32_t kBbFuncTable = 0x8040AAB8u;     // bbFunc[] (lis r30, 0x8041; addi r30, r30, -0x5548)

int32_t GuardIndex(uint32_t address) {
    const uint32_t* const begin = g_mkwGuardedFunctionStarts;
    const uint32_t* const end = begin + g_mkwGuardedFunctionCount;
    const uint32_t* const it = std::lower_bound(begin, end, address);
    return it != end && *it == address ? static_cast<int32_t>(it - begin) : -1;
}
// A callee the native evaluates or copies itself: translated, and its code not patched.
inline bool CalleeUnpatched(int32_t guard) { return guard >= 0 && g_mkwGuestCodePatched[guard] == 0; }

struct CalcViewCounters {
    std::atomic<uint64_t> full{0}, fullDeclined{0}, matrices{0}, uniform{0}, invXpose{0}, invTranspose{0},
        billboard{0}, billboardChild{0}, generic{0};
} g_calcView;

using B16 = uint8_t __attribute__((ext_vector_type(16)));
inline I4 LoadWords(const uint8_t* host) {
    B16 bytes;
    std::memcpy(&bytes, host, sizeof(bytes));
    return __builtin_bit_cast(I4, __builtin_shufflevector(bytes, bytes, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14,
                                                          13, 12));
}
inline B16 GuestBytes(I4 words) {
    const B16 bytes = __builtin_bit_cast(B16, words);
    return __builtin_shufflevector(bytes, bytes, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
}
// psq_st's float rule per lane (PpcStorePairPsqFloatBitsLanesInline; PpcStoreSinglePsqFloatFastInline's conversion
// gives the same bits for ps0), which includes psq_l's (PpcLoadPairPsqFloatBitsLanesInline).
inline I4 PsqStoreRule(I4 bits) {
    const I4 magnitude = bits & 0x7FFFFFFF;
    const I4 subnormal = magnitude < 0x00800000;
    const I4 quieted = bits | ((magnitude > 0x7F800000) & 0x00400000);
    return (subnormal & (bits & INT32_MIN)) | (~subnormal & quieted);
}
// MTX34ToMTX33 (0x80162798) from the rows at `src` to `dst`: psq_l of the six pairs; psq_st of pairs 0, 2, 4 (8
// bytes) and of ps0 of pairs 1, 3, 5 (4 bytes) at dst + 0, 8, 12, 20, 24, 32. Returns nothing: no NaN path (bit rules).
inline void ToMtx33(uint32_t dst, uint32_t src) {
    uint8_t* const m = Window();
    const uint8_t* const s = m + Canonical(src);
    const I4 r0 = PsqStoreRule(LoadWords(s)), r1 = PsqStoreRule(LoadWords(s + 16)), r2 = PsqStoreRule(LoadWords(s + 32));
    const uint32_t c = Canonical(dst);
    uint8_t* const d = m + c;
    const B16 b0 = GuestBytes(r0), b1 = GuestBytes(r1), b2 = GuestBytes(r2);
    std::memcpy(d, &b0, 12);
    std::memcpy(d + 12, &b1, 12);
    std::memcpy(d + 24, &b2, 12);
    MarkPair(c);
    MarkPair(c + 32u);
}
// A uniformly scaled matrix's texture matrix (PSMTXCopy 0x801EC184 from the view matrix, then stfs f31 to _23, _13,
// _03: the six pairs, then three words; marks from tex to tex + 44), when `tex`, and its normal matrix (ToMtx33 from
// the view matrix), from one load of the view rows: both read the same view rows (the texture stores go to another
// part of the locked cache), so their psq_l results are one set of rows under the same lane rule.
inline void UniformTexNrm(uint32_t tex, uint32_t nrm, uint32_t view, uint32_t zeroBits) {
    uint8_t* const m = Window();
    const uint8_t* const s = m + Canonical(view);
    const I4 r0 = PsqStoreRule(LoadWords(s)), r1 = PsqStoreRule(LoadWords(s + 16)), r2 = PsqStoreRule(LoadWords(s + 32));
    if (tex != 0) {
        const uint32_t c = Canonical(tex);
        uint8_t* const d = m + c;
        const I4 rows[3] = {r0, r1, r2};
        for (uint32_t row = 0; row < 3; ++row) {
            I4 r = rows[row];
            r[3] = static_cast<int32_t>(zeroBits);
            const B16 b = GuestBytes(r);
            std::memcpy(d + 16u * row, &b, 16);
        }
        MarkPair(c);
        MarkPair(c + 44u);
    }
    const uint32_t c = Canonical(nrm);
    uint8_t* const d = m + c;
    const B16 b0 = GuestBytes(r0), b1 = GuestBytes(r1), b2 = GuestBytes(r2);
    std::memcpy(d, &b0, 12);
    std::memcpy(d + 12, &b1, 12);
    std::memcpy(d + 24, &b2, 12);
    MarkPair(c);
    MarkPair(c + 32u);
}

// GetResNode (0x8018EDF8): ResMdl's node dictionary (offset at +0x14), entry idx + 1's data offset.
inline uint32_t ResNode(uint32_t mdlRef, uint32_t idx) {
    const uint32_t mdl = GuestLoad32(mdlRef);
    const uint32_t dictOffset = GuestLoad32(mdl + 0x14u);
    if (dictOffset == 0) return 0;
    const uint32_t dict = mdl + dictOffset;
    if (dict == 0) return 0;
    const uint32_t data = GuestLoad32(dict + ((idx + 1u) << 4) + 0x14u);
    return data == 0 ? 0u : dict + data;
}
// ResMdlInfo::GetNodeIDFromMtxID(mtx) as the original reads it per matrix (lwz the ResMdl, +0x40 info, +0x24 table
// offset, entry mtx + 1).
inline uint32_t NodeIdOfMtx(uint32_t mdlRef, uint32_t mtx4) {
    const uint32_t info = GuestLoad32(mdlRef) + 0x40u;
    return GuestLoad32(info + GuestLoad32(info + 0x24u) + mtx4 + 4u);
}
// The same two lookups with the ResMdl's pointers read once: the original reads them again for every matrix, but
// with no store or call in between they read the same words, so a lookup filled after the last store or call gives
// the same results (the billboard pass refills it after each callee).
struct NodeLookup {
    uint32_t table = 0, dict = 0;  // the mtx-to-node table (entries at +4), the node dictionary (0: none)
    void Load(uint32_t mdlRef) {
        const uint32_t mdl = GuestLoad32(mdlRef);
        const uint32_t info = mdl + 0x40u;
        table = info + GuestLoad32(info + 0x24u);
        const uint32_t dictOffset = GuestLoad32(mdl + 0x14u);
        dict = dictOffset == 0 ? 0u : mdl + dictOffset;
    }
    uint32_t NodeId(uint32_t mtx4) const { return GuestLoad32(table + mtx4 + 4u); }
    uint32_t Node(uint32_t idx) const {  // ResNode
        if (dict == 0) return 0;
        const uint32_t data = GuestLoad32(dict + ((idx + 1u) << 4) + 0x14u);
        return data == 0 ? 0u : dict + data;
    }
};

inline void Call(CpuContext* ctx, void (*fn)(CpuContext*), uint32_t returnAddress) {
    ctx->lr = returnAddress;
    fn(ctx);
}
// LCQueueWait(len) (r1 = the caller's frame), through the entry the original calls.
inline void QueueWait(CpuContext* ctx, uint32_t len, uint32_t returnAddress) {
    ctx->lr = returnAddress;
    if (func_801D7AE4_eabi != nullptr) {
        func_801D7AE4_eabi(ctx, len);
        return;
    }
    ctx->gpr[3] = len;
    func_801D7AE4(ctx);
}
inline void StoreBlocks(CpuContext* ctx, uint32_t dst, uint32_t src, uint32_t blocks, uint32_t returnAddress) {
    ctx->gpr[3] = dst;
    ctx->gpr[4] = src;
    ctx->gpr[5] = blocks;
    Call(ctx, func_8015D8BC, returnAddress);
}

// CalcViewLCFull's phases (NAT_INSN_PHASES=1, see PhaseSites), native column: the queue wait, the matrix product, the
// billboard and position pass, the LC block stores, the normal and texture pass (a call's prologue and epilogue are
// the rest of its g3d.calcview count).
const std::array<int, 5> kCvSites = [] {
    std::array<int, 5> s{-1, -1, -1, -1, -1};
    if (!PhaseSites()) return s;
    const char* const names[5] = {"g3d.cv.queue", "g3d.cv.concat", "g3d.cv.pos", "g3d.cv.store", "g3d.cv.nrm"};
    for (size_t i = 0; i < s.size(); ++i) s[i] = InsnCount::Site(names[i]);
    return s;
}();

bool CalcViewLCFull(CpuContext* ctx) {
    static const int32_t kGuardGetResNode = GuardIndex(0x8018EDF8u);
    static const int32_t kGuardToMtx33 = GuardIndex(0x80162798u);
    static const int32_t kGuardCopy = GuardIndex(0x801EC184u);
    if (ctx->gqr[0] != 0 || !LeanReady() || !CalleeUnpatched(kGuardGetResNode) || !CalleeUnpatched(kGuardToMtx33) ||
        !CalleeUnpatched(kGuardCopy) || !PlainRange(kLcBase, 0x4000u))
        return false;
    const uint32_t r1 = ctx->gpr[1], lr = ctx->lr, xer = ctx->xer;
    int64_t entryCrv[8];
    std::memcpy(entryCrv, ctx->crv, sizeof(entryCrv));
    const uint32_t viewPos = ctx->gpr[3], viewNrm = ctx->gpr[4], model = ctx->gpr[5], attrib = ctx->gpr[6];
    const uint32_t count = ctx->gpr[7], viewMtx = ctx->gpr[8], mdlRef = ctx->gpr[9], viewTex = ctx->gpr[10];
    const uint32_t sp = r1 - 208u;

    // Prologue: stwu r1,-208(r1); stw r0(lr),212(r1); stfd f31,192(r1); psq_st f31,200(r1),0,0; _savegpr_14 (r11 =
    // r1 + 192), as the translation stores them.
    MemoryInline::FlatWriteRam32(sp, r1);
    MemoryInline::FlatWriteRam32(sp + 212u, lr);
    const double f31 = ctx->fpr[31].d;
    MemoryInline::FlatWriteRamFloat64(sp + 192u, f31);
    PPC_PsqStStackInline<0u, 0u>(ctx, sp + 200u, PPC_PsFromScalarInline(f31));
    {
        const uint32_t saveArea = sp + 192u - 72u;
        uint8_t* const host = MemoryInline::ResolveRangeHost(saveArea, 0, 72u, true);
        for (uint32_t reg = 14, offset = 0; reg < 32; reg += 2, offset += 8)
            if (!MemoryInline::WriteResolvedPair32(
                    host, offset, (static_cast<uint64_t>(ctx->gpr[reg]) << 32) | ctx->gpr[reg + 1])) {
                MemoryInline::WriteResolved32(host, offset, saveArea + offset, ctx->gpr[reg]);
                MemoryInline::WriteResolved32(host, offset + 4u, saveArea + offset + 4u, ctx->gpr[reg + 1]);
            }
    }
    MemoryInline::FlatWriteRam32(sp + 8u, viewPos);
    MemoryInline::FlatWriteRam32(sp + 12u, viewNrm);
    MemoryInline::FlatWriteRam32(sp + 16u, attrib);
    MemoryInline::FlatWriteRam32(sp + 20u, count);
    if (count == 0) return true;

    ctx->gpr[1] = sp;
    const uint32_t posBytes = (count * 48u + 31u) & ~31u;
    ctx->gpr[3] = viewPos;
    ctx->gpr[4] = posBytes;
    ctx->lr = 0x801A3DECu;
    InvokeDirectCpu<0x801D76BCu>(ctx);  // DCInvalidateRange
    uint32_t queueLen = 1;
    if (viewNrm != 0) {
        queueLen = 2;
        ctx->gpr[3] = viewNrm;
        ctx->gpr[4] = (count * 36u + 31u) & ~31u;
        ctx->lr = 0x801A3E20u;
        InvokeDirectCpu<0x801D76BCu>(ctx);
        if (viewTex != 0) {
            queueLen = 3;
            ctx->gpr[3] = viewTex;
            ctx->gpr[4] = posBytes;
            ctx->lr = 0x801A3E3Cu;
            InvokeDirectCpu<0x801D76BCu>(ctx);
        }
    }
    QueueWait(ctx, 0u, 0x801A3E4Cu);
    const double f31Body = MemoryInline::FlatReadFloat32(ctx->gpr[2] - 24544u);  // lfs f31, -0x5FE0(r2)
    const uint32_t zeroBits = MemoryInline::ConvertPpcDoubleToSingleBits(f31Body);
    bool generic = Config().calcViewGeneric;

    DeferredDmaNotify notify(viewPos, viewNrm, count, viewTex);
    uint32_t curV = kLcBase + 0x0800u, curN = kLcBase + 0x1000u, curT = kLcBase + 0x1800u;
    uint32_t prevV = kLcBase + 0x2800u, prevN = kLcBase + 0x3000u, prevT = kLcBase + 0x3800u;
    uint32_t blocks = 0, blocksNrm = 0, base = 0, base4 = 0;
    uint32_t posAt = viewPos, nrmAt = viewNrm, texAt = viewTex, modelAt = model;
    uint64_t uniform = 0, invXpose = 0, invTranspose = 0, billboard = 0, billboardChild = 0;
    while (base < count) {
        uint32_t remain = count - base;
        if (remain > 40u) {
            remain = 40u;
            blocks = 60u;
            blocksNrm = 45u;
        } else {
            blocks = (remain * 48u + 31u) >> 5;
            blocksNrm = (remain * 36u + 31u) >> 5;
        }
        {
            InsnCount::Scope phase(kCvSites[0], InsnCount::kNative);
            QueueWait(ctx, queueLen, 0x801A3EC4u);
        }
        ctx->gpr[3] = viewMtx;
        ctx->gpr[4] = modelAt;
        ctx->gpr[5] = curV;
        if (remain > 1u) {
            ctx->gpr[6] = remain;
            InsnCount::Scope phase(kCvSites[1], InsnCount::kNative);
            Call(ctx, func_801EC284, 0x801A3EE0u);  // PSMTXConcatArray(view, model + base, curV, remain)
        } else {
            InsnCount::Scope phase(kCvSites[1], InsnCount::kNative);
            Call(ctx, func_801EC1B8, 0x801A3EF4u);  // PSMTXConcat
        }
        const uint32_t attrAt = attrib + base4;
        NodeLookup look;
        bool lookFresh = false;  // filled since the last call
        std::optional<InsnCount::Scope> posPhase;
        if (InsnCount::On()) posPhase.emplace(kCvSites[2], InsnCount::kNative);
        for (uint32_t i = 0; i < remain; ++i) {
            const uint32_t attr = GuestLoad32(attrAt + 4u * i);
            const uint32_t view = curV + 48u * i;
            if ((attr & 0xFFu) != 0) {  // billboard: bbFunc[type](&curV[i], model, uniform, view, mdl, base + i)
                ++billboard;
                lookFresh = false;
                const uint32_t mdl = GuestLoad32(mdlRef);
                const uint32_t fn = GuestLoad32(kBbFuncTable + ((attr & 0xFFu) << 2));
                MemoryInline::FlatWriteRam32(sp + 24u, mdl);
                ctx->gpr[3] = view;
                ctx->gpr[4] = model;
                ctx->gpr[5] = (attr >> 29) & 1u;
                ctx->gpr[6] = viewMtx;
                ctx->gpr[7] = sp + 24u;
                ctx->gpr[8] = base + i;
                std::memcpy(ctx->crv, entryCrv, sizeof(entryCrv));
                ctx->crv[0] = PpcCrvCompare(static_cast<int32_t>(attr & 0xFFu), static_cast<int32_t>(0), xer);
                ctx->ctr = fn;
                ctx->lr = 0x801A3F50u;
                InvokeIndirectCpu(fn, ctx);
                generic = generic || ctx->gqr[0] != 0;
                const uint32_t node = ResNode(mdlRef, NodeIdOfMtx(mdlRef, base4 + 4u * i));
                if (node == 0) continue;
                const uint32_t childOffset = GuestLoad32(node + 0x60u);
                if (childOffset == 0 || node + childOffset == 0) continue;
                // Its inverse world matrix times the view matrix, for billboard children (pBbMtxArray).
                ++billboardChild;
                const uint32_t mtxOffset = (base + i) * 48u;
                ctx->gpr[3] = model + mtxOffset;
                ctx->gpr[4] = sp + 32u;
                Call(ctx, func_801EC41C, 0x801A3FC4u);  // PSMTXInverse(model + i, &invWorld)
                if (ctx->gpr[3] == 1u) {
                    ctx->gpr[3] = view;
                    ctx->gpr[4] = sp + 32u;
                    ctx->gpr[5] = kBbTemporary + mtxOffset;
                    Call(ctx, func_801EC1B8, 0x801A3FDCu);
                } else {
                    ctx->gpr[3] = kBbTemporary + mtxOffset;
                    func_801EC158(ctx);  // PSMTXIdentity (inlined leaf: the original's LR stays)
                }
                continue;
            }
            if (!lookFresh) {
                look.Load(mdlRef);
                lookFresh = true;
            }
            const uint32_t nodeId = look.NodeId(base4 + 4u * i);
            if (static_cast<int32_t>(nodeId) < 0) continue;
            const uint32_t node = look.Node(nodeId);
            if (node == 0 || (GuestLoad32(node + 0x14u) & 0x400u) == 0) continue;  // FLAG_BILLBOARD_PARENT
            const uint32_t bbNode = look.Node(GuestLoad32(node + 0x1Cu));   // bbref_nodeid
            const uint32_t bbMtx = bbNode != 0 ? GuestLoad32(bbNode + 0x10u) : 0u;
            ctx->gpr[3] = kBbTemporary + bbMtx * 48u;
            ctx->gpr[4] = model + (base + i) * 48u;
            ctx->gpr[5] = view;
            Call(ctx, func_801EC1B8, 0x801A4068u);  // PSMTXConcat(&pBbMtxArray[bbMtxID], model + i, &curV[i])
            lookFresh = false;
        }
        posPhase.reset();
        {
            InsnCount::Scope phase(kCvSites[3], InsnCount::kNative);
            StoreBlocks(ctx, posAt, curV, blocks, 0x801A4094u);
        }
        if (viewNrm != 0) {
            std::optional<InsnCount::Scope> nrmPhase;
            if (InsnCount::On()) nrmPhase.emplace(kCvSites[4], InsnCount::kNative);
            for (uint32_t i = 0; i < remain; ++i) {
                const uint32_t attr = GuestLoad32(attrAt + 4u * i);
                const uint32_t view = curV + 48u * i, nrm = curN + 36u * i, tex = curT + 48u * i;
                if ((attr & 0x20000000u) != 0) {  // IsAllScaleUniform
                    ++uniform;
                    if (!generic) {
                        UniformTexNrm(viewTex != 0 ? tex : 0u, nrm, view, zeroBits);
                        continue;
                    }
                    if (viewTex != 0) {
                        ctx->gpr[3] = view;
                        ctx->gpr[4] = tex;
                        func_801EC184(ctx);  // PSMTXCopy (inlined leaf)
                        MemoryInline::FlatWriteFloat32(tex + 44u, f31Body);
                        MemoryInline::FlatWriteFloat32(tex + 28u, f31Body);
                        MemoryInline::FlatWriteFloat32(tex + 12u, f31Body);
                    }
                    ctx->gpr[3] = nrm;
                    ctx->gpr[4] = view;
                    func_80162798(ctx);  // MTX34ToMTX33 (inlined leaf)
                } else if (viewTex != 0) {
                    ++invXpose;
                    ctx->gpr[3] = view;
                    ctx->gpr[4] = tex;
                    func_801EC514(ctx);  // PSMTXInvXpose(&curV[i], &curT[i]) (inlined leaf)
                    if (!generic) {
                        ToMtx33(nrm, tex);
                        continue;
                    }
                    ctx->gpr[3] = nrm;
                    ctx->gpr[4] = tex;
                    Call(ctx, func_80162798, 0x801A4114u);
                } else {
                    ++invTranspose;
                    ctx->gpr[3] = nrm;
                    ctx->gpr[4] = view;
                    std::memcpy(ctx->crv, entryCrv, sizeof(entryCrv));
                    ctx->crv[0] = PpcCrvCompare(static_cast<int32_t>(viewTex), static_cast<int32_t>(0), xer);
                    Call(ctx, func_801627CC, 0x801A4124u);  // MTX34InvTranspose(&curN[i], &curV[i])
                }
            }
            nrmPhase.reset();
            InsnCount::Scope phase(kCvSites[3], InsnCount::kNative);
            StoreBlocks(ctx, nrmAt, curN, blocksNrm, 0x801A4160u);
            if (viewTex != 0) StoreBlocks(ctx, texAt, curT, blocks, 0x801A4178u);
        }
        std::swap(curV, prevV);
        std::swap(curN, prevN);
        std::swap(curT, prevT);
        base += 40u;
        base4 += 160u;
        modelAt += 0x780u;
        posAt += 0x780u;
        nrmAt += 0x5A0u;
        texAt += 0x780u;
    }
    notify.Send();

    // The local slots' final values: 8 view positions, 112 view normals, 108 view texture matrices, 116 model
    // matrices (each advanced per batch), 88 queue length, 84/80 the last batch's block counts, and the cache map's
    // swapped pointers (curV and curT stay in r22/r14): 100 previous view, 104 current normal, 96 previous normal,
    // 92 previous texture.
    MemoryInline::FlatWriteRam32(sp + 8u, posAt);
    MemoryInline::FlatWriteRam32(sp + 80u, blocksNrm);
    MemoryInline::FlatWriteRam32(sp + 84u, blocks);
    MemoryInline::FlatWriteRam32(sp + 88u, queueLen);
    MemoryInline::FlatWriteRam32(sp + 92u, prevT);
    MemoryInline::FlatWriteRam32(sp + 96u, prevN);
    MemoryInline::FlatWriteRam32(sp + 100u, prevV);
    MemoryInline::FlatWriteRam32(sp + 104u, curN);
    MemoryInline::FlatWriteRam32(sp + 108u, texAt);
    MemoryInline::FlatWriteRam32(sp + 112u, nrmAt);
    MemoryInline::FlatWriteRam32(sp + 116u, modelAt);
    ctx->gpr[1] = r1;
    ctx->lr = lr;

    Bump(g_calcView.matrices, count);
    Bump(g_calcView.uniform, uniform);
    Bump(g_calcView.invXpose, invXpose);
    Bump(g_calcView.invTranspose, invTranspose);
    Bump(g_calcView.billboard, billboard);
    Bump(g_calcView.billboardChild, billboardChild);
    Bump(g_calcView.generic, generic ? 1u : 0u);
    return true;
}

// CalcView_LC: the native above, else (declined, or G3D_CALC_OFF=cvfull) the translated body with deferred
// StoreBlocks notifications.
bool CalcViewLCNative(CpuContext* ctx) {
    if (Config().calcViewFull) {
        if (CalcViewLCFull(ctx)) {
            Bump(g_calcView.full);
            return true;
        }
        Bump(g_calcView.fullDeclined);
    }
    DeferredDmaNotify notify(ctx->gpr[3], ctx->gpr[4], ctx->gpr[7], ctx->gpr[10]);
    func_801A3D68_body(ctx);
    notify.Send();
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Camera and light state (g3d_camera.cpp, g3d_light.cpp). G3DState::SetCameraProjMtx (0x801A0FD8) fills each of the 32
// camera slots on every ScnRoot pass (0x801AC7C0, ~13 passes a frame): ~400 calls a frame each of GetCameraMtx and
// GetProjectionMtx (whose MTX44Copy is the cost), and LightSetting's copy runs per pass too.
// ---------------------------------------------------------------------------------------------------------------
// `pairs` psq_st of psq_l'd pairs (GQR0 the float encoding), rows of 16 bytes at a time: the psq_st rule per lane.
// The source and destination do not overlap (or are the same range). Marks: the caller's.
inline void CopyPsqRows(uint8_t* d, const uint8_t* s, uint32_t rows) {
    for (uint32_t row = 0; row < rows; ++row) {
        const B16 b = GuestBytes(PsqStoreRule(LoadWords(s + 16u * row)));
        std::memcpy(d + 16u * row, &b, 16);
    }
}

// math::MTX44Copy (0x80162C04, frameless): eight psq_l, then eight psq_st (all loads first, so an overlap reads the
// source as it was). Per lane the psq_st rule; the stores' marks are the pairs from dst to dst + 56. Writes back f1 and
// f2 (psq_l's pairs at src + 8 and src + 16), as the vanilla translation does.
bool Mtx44CopyNative(CpuContext* ctx) {
    const uint32_t dst = ctx->gpr[3], src = ctx->gpr[4];
    if (ctx->gqr[0] != 0 || !LeanReady() || !OneView(src, 64u) || !PlainRange(dst, 64u)) return false;
    uint8_t* const m = Window();
    const uint8_t* const s = m + Canonical(src);
    const I4 rows[4] = {LoadWords(s), LoadWords(s + 16), LoadWords(s + 32), LoadWords(s + 48)};
    const double f1 = PpcLoadPairPsqFloatFromHostInline(s + 8), f2 = PpcLoadPairPsqFloatFromHostInline(s + 16);
    const uint32_t c = Canonical(dst);
    for (uint32_t row = 0; row < 4; ++row) {
        const B16 b = GuestBytes(PsqStoreRule(rows[row]));
        std::memcpy(m + c + 16u * row, &b, 16);
    }
    MarkPair(c);
    MarkPair(c + 56u);
    PpcSetPairedFprInline(ctx->fpr[1], f1);
    PpcSetPairedFprInline(ctx->fpr[2], f2);
    return true;
}

// Camera::GetCameraMtx (0x801A7DBC; r3 = Camera (by reference), r4 = output): a 16-byte frame (back chain, LR, r31,
// r30); with an output and a camera (lwz r3): UpdateCameraMtx (0x801A81D4, through its wrapper with the original's r1,
// LR, r3 and CR) unless the matrix is current (flags @ +0x70 bit 3), then PSMTXCopy (inlined) of the camera matrix (@
// +0) to the output: six psq_l/psq_st pairs, each pair's load before its store, so overlapping ranges (or a callee that
// left the output not plain) take the translated helpers pair by pair. Writes back r3 (the camera data), r4 (the
// output), f1 and f2 (psq_l's pairs at +8 and +16) as the vanilla translation does; nothing without an output/camera.
bool CameraMtxNative(CpuContext* ctx) {
    static const int32_t kGuardCopy = GuardIndex(0x801EC184u);
    const uint32_t r1 = ctx->gpr[1], lr = ctx->lr, handle = ctx->gpr[3], out = ctx->gpr[4];
    if (ctx->gqr[0] != 0 || !LeanReady() || !CalleeUnpatched(kGuardCopy)) return false;
    const uint32_t sp = r1 - 16u;
    const bool eabi = EabiTranslation();  // kFnEabiDeadSaveBytes: the callers' EABI entry saved r31/r30 as 0
    MemoryInline::FlatWriteRam32(sp, r1);
    MemoryInline::FlatWriteRam32(sp + 20u, lr);
    MemoryInline::FlatWriteRam32(sp + 12u, eabi ? 0u : ctx->gpr[31]);
    MemoryInline::FlatWriteRam32(sp + 8u, eabi ? 0u : ctx->gpr[30]);
    if (out == 0) return true;
    const uint32_t cam = GuestLoad32(handle);
    if (cam == 0) return true;
    if ((GuestLoad32(cam + 0x70u) & 8u) == 0) {
        ctx->crv[0] = PpcCrvCompare(static_cast<int32_t>(0), static_cast<int32_t>(0), ctx->xer);  // rlwinm. of flags & 8
        ctx->gpr[1] = sp;
        ctx->lr = 0x801A7DF8u;
        if (func_801A81D4_eabi != nullptr)  // UpdateCameraMtx (r3: the handle), the entry the original calls
            func_801A81D4_eabi(ctx, sp, handle);
        else
            func_801A81D4(ctx);
        ctx->gpr[1] = r1;
        ctx->lr = lr;
    }
    if ((cam == out || !Overlap(cam, 48u, out, 48u)) && OneView(cam, 48u) && PlainRange(out, 48u)) {
        uint8_t* const m = Window();
        const uint8_t* const s = m + Canonical(cam);
        const double f1 = PpcLoadPairPsqFloatFromHostInline(s + 8), f2 = PpcLoadPairPsqFloatFromHostInline(s + 16);
        const uint32_t c = Canonical(out);
        CopyPsqRows(m + c, s, 3);
        MarkPair(c);
        MarkPair(c + 40u);
        PpcSetPairedFprInline(ctx->fpr[1], f1);
        PpcSetPairedFprInline(ctx->fpr[2], f2);
    } else {
        double pair[6];
        for (uint32_t k = 0; k < 6; ++k) {
            pair[k] = PPC_PsqLGqrInline<0u, 0u>(ctx, 0u, cam + 8u * k);
            PPC_PsqStGqrInline<0u, 0u>(ctx, 0u, out + 8u * k, pair[k]);
        }
        PpcSetPairedFprInline(ctx->fpr[1], pair[1]);
        PpcSetPairedFprInline(ctx->fpr[2], pair[2]);
    }
    ctx->gpr[3] = cam;
    ctx->gpr[4] = out;
    return true;
}

// Camera::GetEnvironmentTexMtx (0x801A7FC8; r3 = Camera (by reference), r4 = output), reached through its context
// entry: a 16-byte frame (back chain, LR, r31, r30); with an output and a camera, PSMTXIdentity (inlined: lfs 0.0
// and 1.0 from r2-23492/-23496 and six psq_st of their paired forms, the translation's helpers in its order), then
// six stfs of camera fields (0xCC, 0xD4, -0xD0, 0xD8) and constants (r2-24512, r2-24504), each lfs'd and stored with
// FlatWriteFloat32's conversion. Writes back r3 (the output), f1 and f2 (the last lfs of each) as the translation.
bool EnvTexMtxNative(CpuContext* ctx) {
    static const int32_t kGuardIdentity = GuardIndex(0x801EC158u);
    const uint32_t r1 = ctx->gpr[1], r2 = ctx->gpr[2], out = ctx->gpr[4];
    if (ctx->gqr[0] != 0 || !LeanReady() || !CalleeUnpatched(kGuardIdentity) || (out != 0 && !PlainRange(out, 48u)))
        return false;
    const uint32_t sp = r1 - 16u;
    MemoryInline::FlatWriteRam32(sp, r1);
    MemoryInline::FlatWriteRam32(sp + 20u, ctx->lr);
    MemoryInline::FlatWriteRam32(sp + 12u, ctx->gpr[31]);
    MemoryInline::FlatWriteRam32(sp + 8u, ctx->gpr[30]);
    if (out == 0) return true;
    const uint32_t cam = GuestLoad32(ctx->gpr[3]);
    if (cam == 0) return true;
    uint8_t* const o = Window() + Canonical(out);
    // Fast form. The twelve words the sequence below leaves: the identity's pairs are (1.0, z) at +0 and +40, (z, z)
    // at +8, +24 and +32, (z, 1.0) at +16 (z = the 0.0 constant), then the stfs overwrite words 0, 3, 5, 7, 10 and 11,
    // which covers every 1.0. For a normal or zero single, lfs widens exactly and stfs narrows back to the same bits,
    // fneg flips the sign bit, psq_l/psq_st leave it as is: so the words are those bits (z pairs, camera words 0xCC,
    // 0xD4, 0xD0 negated, 0xD8, constants r2-24512 and r2-24504). Any other value, or an output overlapping a word the
    // sequence reads (the order of its loads and stores would matter), takes the sequence itself.
    {
        const uint32_t z = GuestLoad32(r2 - 23492u), k0 = GuestLoad32(r2 - 24512u), k1 = GuestLoad32(r2 - 24504u);
        const uint32_t c0 = GuestLoad32(cam + 204u), c1 = GuestLoad32(cam + 208u), c2 = GuestLoad32(cam + 212u),
                       c3 = GuestLoad32(cam + 216u);
        const auto plain = [](uint32_t bits) {
            const uint32_t magnitude = bits & 0x7FFFFFFFu;
            return magnitude == 0 || magnitude - 0x00800000u < 0x7F000000u;
        };
        if (plain(z) && plain(k0) && plain(k1) && plain(c0) && plain(c1) && plain(c2) && plain(c3) &&
            !Overlap(out, 48u, cam + 204u, 16u) && !Overlap(out, 48u, r2 - 24512u, 12u) &&
            !Overlap(out, 48u, r2 - 23492u, 4u)) {
            const uint32_t words[12] = {c0, z, z, c2, z, c1 ^ 0x80000000u, z, c3, z, z, k0, k1};
            for (uint32_t i = 0; i < 12; ++i) Store32(o + 4u * i, words[i]);
            MarkPair(Canonical(out));
            MarkPair(Canonical(out + 44u));
            ctx->gpr[3] = out;
            ctx->fpr[1].d = PpcBitCastToDoubleInline(PpcConvertToDoubleBitsInline(k0));
            ctx->fpr[2].d = PpcBitCastToDoubleInline(PpcConvertToDoubleBitsInline(c3));
            return true;
        }
    }
    const double zero = MemoryInline::FlatReadFloat32(r2 - 23492u), one = MemoryInline::FlatReadFloat32(r2 - 23496u);
    const double zeroPair = PPC_PsFromScalarInline(zero), onePair = PPC_PsFromScalarInline(one);
    const double f2Pair = PPC_PsMerge10Inline(onePair, zeroPair), f1Pair = PPC_PsMerge01Inline(zeroPair, onePair);
    PpcStorePairPsqFloatToHostInline(o + 8, zeroPair);
    PpcStorePairPsqFloatToHostInline(o + 24, zeroPair);
    PpcStorePairPsqFloatToHostInline(o + 32, zeroPair);
    PpcStorePairPsqFloatToHostInline(o + 16, f1Pair);
    PpcStorePairPsqFloatToHostInline(o, f2Pair);
    PpcStorePairPsqFloatToHostInline(o + 40, f2Pair);
    const double scaleS = MemoryInline::FlatReadFloat32(cam + 204u), last1 = MemoryInline::FlatReadFloat32(r2 - 24512u);
    MemoryInline::FlatWriteFloat32(out, scaleS);
    const double last0 = MemoryInline::FlatReadFloat32(r2 - 24504u);
    MemoryInline::FlatWriteFloat32(out + 12u, MemoryInline::FlatReadFloat32(cam + 212u));
    MemoryInline::FlatWriteFloat32(out + 20u, -MemoryInline::FlatReadFloat32(cam + 208u));
    const double f2 = MemoryInline::FlatReadFloat32(cam + 216u);
    MemoryInline::FlatWriteFloat32(out + 28u, f2);
    MemoryInline::FlatWriteFloat32(out + 40u, last1);
    MemoryInline::FlatWriteFloat32(out + 44u, last0);
    ctx->gpr[3] = out;
    ctx->fpr[1].d = last1;
    ctx->fpr[2].d = f2;
    return true;
}
bool EnvTexMtxWrites(const CpuContext& ctx, Ranges& ranges) { return ctx.gpr[4] == 0 || ranges.Add(ctx.gpr[4], 48u); }

// LightSetting copy (0x801B6708; r3 = destination, r4 = source; u16 light objects @0, u16 light sets @2, LightObj[]
// @4 (0x44 bytes: flags, then 64 bytes), AmbLightObj[] @8 (4 bytes), LightSet[] @12 (12 bytes)): returns 0 when the
// destination holds fewer of either. Per light object: an enabled one (flags bit 2) is copied (flags by stw, the 64
// bytes by Copy32ByteBlocks 0x8019DFAC: lfd/stfd, raw bits) unless it is the same object, a disabled one has bit 2
// cleared in the destination; its ambient color is copied byte by byte; then every light set, byte by byte. The
// frame: back chain, LR, _savegpr_27. The native copies the same bytes in the same per-element order (stores that
// could see each other's bytes keep their order) when the destination arrays are plain lean memory and overlap no
// source array or either header (an identical array is a self copy: same bytes, same marks), else declines; marks
// per element are its first and last store starts' pairs (dense, under 64 bytes apart). Writes back r3, r4 (as the
// original leaves it: the last light set's destination, else the last ambient source, else r4) and f1 (the callee's
// write-back: the last enabled object's bytes 60-67 as a double).
bool LightSettingCopyNative(CpuContext* ctx) {
    static const int32_t kGuardBlocks = GuardIndex(0x8019DFACu);
    if (!LeanReady() || !CalleeUnpatched(kGuardBlocks)) return false;
    const uint32_t r1 = ctx->gpr[1], dst = ctx->gpr[3], src = ctx->gpr[4];
    if (!OneView(dst, 16u) || !OneView(src, 16u)) return false;
    const uint32_t numObj = GuestLoad16(src), numSet = GuestLoad16(src + 2u);
    const bool fits = !(GuestLoad16(dst) < numObj) && !(GuestLoad16(dst + 2u) < numSet);
    const uint32_t objD = GuestLoad32(dst + 4u), objS = GuestLoad32(src + 4u);
    const uint32_t ambD = GuestLoad32(dst + 8u), ambS = GuestLoad32(src + 8u);
    const uint32_t setD = GuestLoad32(dst + 12u), setS = GuestLoad32(src + 12u);
    const uint32_t objBytes = 68u * numObj, ambBytes = 4u * numObj, setBytes = 12u * numSet;
    if (fits) {
        const uint32_t dAt[3] = {objD, ambD, setD}, sAt[3] = {objS, ambS, setS};
        const uint32_t bytes[3] = {objBytes, ambBytes, setBytes};
        for (int a = 0; a < 3; ++a) {
            if (bytes[a] == 0) continue;
            if (!PlainRange(dAt[a], bytes[a]) || !OneView(sAt[a], bytes[a]) || Overlap(dAt[a], bytes[a], dst, 16u) ||
                Overlap(dAt[a], bytes[a], src, 16u))
                return false;
            for (int b = 0; b < 3; ++b)
                if (!(a == b && dAt[a] == sAt[b]) && Overlap(dAt[a], bytes[a], sAt[b], bytes[b])) return false;
        }
    }

    const uint32_t sp = r1 - 32u;
    MemoryInline::FlatWriteRam32(sp, r1);
    MemoryInline::FlatWriteRam32(sp + 36u, ctx->lr);
    for (uint32_t reg = 27; reg < 32; ++reg) MemoryInline::FlatWriteRam32(sp + 12u + 4u * (reg - 27u), ctx->gpr[reg]);
    if (!fits) {
        ctx->gpr[3] = 0;
        return true;
    }
    uint8_t* const m = Window();
    uint32_t r4 = ctx->gpr[4];
    bool copied = false;
    uint64_t lastF1 = 0;
    for (uint32_t i = 0; i < numObj; ++i) {
        const uint32_t so = objS + 68u * i, dO = objD + 68u * i;
        const uint32_t flags = GuestLoad32(so);
        if ((flags & 4u) != 0) {
            if (dO != so) {
                Store32(m + Canonical(dO), flags);
                std::memcpy(m + Canonical(dO + 4u), m + Canonical(so + 4u), 64);
                MarkPair(Canonical(dO));
                MarkPair(Canonical(dO + 60u));
                std::memcpy(&lastF1, m + Canonical(so + 60u), 8);
                copied = true;
            }
        } else {
            Store32(m + Canonical(dO), GuestLoad32(dO) & ~4u);
            MarkPair(Canonical(dO));
        }
        const uint32_t a = ambD + 4u * i;
        std::memmove(m + Canonical(a), m + Canonical(ambS + 4u * i), 4);
        MarkPair(Canonical(a));
        MarkPair(Canonical(a + 3u));
        r4 = ambS + 4u * i;
    }
    for (uint32_t j = 0; j < numSet; ++j) {
        const uint32_t d = setD + 12u * j;
        std::memmove(m + Canonical(d), m + Canonical(setS + 12u * j), 12);
        MarkPair(Canonical(d));
        MarkPair(Canonical(d + 11u));
        r4 = d;
    }
    ctx->gpr[3] = 1;
    ctx->gpr[4] = r4;
    if (copied) ctx->fpr[1].raw = __builtin_bswap64(lastF1);
    return true;
}

// G3DState's light slot setter (0x801A0DD8; r3 = LightObj (by reference), r4 = slot, 0-31), called per slot on every
// ScnRoot pass: a 32-byte frame (back chain, LR, r31, r30, and the LightObj pointer at 8); for a valid slot and
// object: if the slot is the state's current one (0x804A4F04), memcmp (0x803F6150, through its wrapper with the
// original's r1 and LR) of the object against the slot, and if they differ the current slot is reset (stw 0 to
// 0x804A4F00); then (inlined 0x801B5E38) the object's 48 bytes go to the slot (0x804A4F08 + 48 * slot) by lfd/stfd:
// raw bytes, a load before each store (an overlap takes the same 8-byte order). Writes back r3 and r4 as the
// original leaves them.
constexpr uint32_t kLightState = 0x804A4F00u;
bool LightSlotNative(CpuContext* ctx) {
    if (!LeanReady()) return false;
    const uint32_t r1 = ctx->gpr[1], handle = ctx->gpr[3], slot = ctx->gpr[4];
    const uint32_t sp = r1 - 32u;
    MemoryInline::FlatWriteRam32(sp, r1);  // stwu, then lwz r3, 0(r3)
    const uint32_t obj = GuestLoad32(handle);
    const bool eabi = EabiTranslation();  // kFnEabiDeadSaveBytes: the callers' EABI entry saved r31/r30 as 0
    MemoryInline::FlatWriteRam32(sp + 36u, ctx->lr);
    MemoryInline::FlatWriteRam32(sp + 28u, eabi ? 0u : ctx->gpr[31]);
    MemoryInline::FlatWriteRam32(sp + 24u, eabi ? 0u : ctx->gpr[30]);
    MemoryInline::FlatWriteRam32(sp + 8u, obj);
    uint32_t r3 = obj, r4 = slot;
    if (static_cast<int32_t>(slot) >= 0 && static_cast<int32_t>(slot) < 32 && obj != 0) {
        bool copy = true;
        if (static_cast<int32_t>(GuestLoad32(kLightState + 4u)) == static_cast<int32_t>(slot)) {
            const uint32_t lr = ctx->lr;
            ctx->gpr[1] = sp;
            ctx->lr = 0x801A0E38u;
            if (func_803F6150_eabi != nullptr) {  // memcmp, the entry the original calls
                const MkwEabiPair result = func_803F6150_eabi(ctx, obj, kLightState + 8u + 48u * slot, 48u);
                r3 = static_cast<uint32_t>(result.first);
                r4 = static_cast<uint32_t>(result.second);
            } else {
                ctx->gpr[3] = obj;
                ctx->gpr[4] = kLightState + 8u + 48u * slot;
                ctx->gpr[5] = 48u;
                func_803F6150(ctx);
                r3 = ctx->gpr[3];
                r4 = ctx->gpr[4];
            }
            ctx->gpr[1] = r1;
            ctx->lr = lr;
            copy = r3 != 0;
            if (copy) MemoryInline::FlatWriteRam32(kLightState, 0u);
        }
        if (copy) {
            const uint32_t dst = kLightState + 8u + 48u * slot;
            const uint32_t src = GuestLoad32(sp + 8u);  // (lwz r5, 0(r3) with r3 = r1 + 8)
            r3 = 0;
            r4 = dst;
            if (src != 0) {
                if (!Overlap(src, 48u, dst, 48u) && OneView(src, 48u) && PlainRange(dst, 48u)) {
                    std::memcpy(Window() + Canonical(dst), Window() + Canonical(src), 48);
                    MarkPair(Canonical(dst));
                    MarkPair(Canonical(dst + 40u));
                } else {
                    for (uint32_t k = 0; k < 48u; k += 8u)
                        MemoryInline::FlatWriteFloat64(dst + k, MemoryInline::FlatReadFloat64(src + k));
                }
                r3 = dst;
            }
        }
    }
    ctx->gpr[3] = r3;
    ctx->gpr[4] = r4;
    return true;
}
bool LightSlotWrites(const CpuContext&, Ranges& ranges) { return ranges.Add(kLightState, 8u + 48u * 32u); }

// ---------------------------------------------------------------------------------------------------------------
// MSL memcpy (0x80004338) and __fill_mem (0x80004388, memset's body) in .init: byte loops in the translation (~1.6% of
// vanilla Training's game thread). Both frameless.
// ---------------------------------------------------------------------------------------------------------------
// The marks of a dense run of stores (starts at most 64 bytes apart) from `first` to `lastStart`: each store's block
// pair (BrawlDirtyMemory::LeanMarkBlockPair).
inline void MarkStoreRun(uint32_t first, uint32_t lastStart) {
#if MKW_LEAN_DIRTY_MARKS
    const uint32_t a = Canonical(first) >> 6, b = (Canonical(lastStart) >> 6) + 1u;
    std::memset(BrawlDirtyMemory::leanBlocks + a, 1, b - a + 1u);
#else
    (void)first;
    (void)lastStart;
#endif
}
constexpr uint32_t kMaxMemBytes = 1u << 24;

// memcpy(dst = r3, src = r4, n = r5): byte by byte, backward when src < dst (unsigned), else forward; so the bytes
// are memmove's whenever the guest addresses' order is the host's (it declines a cross-view alias that would reverse
// it). Each byte a plain store: the marks of the run dst..dst + n - 1. Writes back r4 as the original leaves it
// (src + n - 1 forward, src backward); r3 is unchanged.
bool MemcpyNative(CpuContext* ctx) {
    const uint32_t dst = ctx->gpr[3], src = ctx->gpr[4], n = ctx->gpr[5];
    if (n > kMaxMemBytes || !LeanReady() || !OneView(src, n) || !MemoryInline::LeanStoresArePlain(dst, n)) return false;
    const bool backward = src < dst;
    if (n != 0) {
        if (Overlap(src, n, dst, n) && backward != (Canonical(src) < Canonical(dst))) return false;
        std::memmove(Window() + Canonical(dst), Window() + Canonical(src), n);
        MarkStoreRun(dst, dst + n - 1u);
    }
    if (n != 0) ctx->gpr[0] = Window()[Canonical(backward ? dst : dst + n - 1u)];
    ctx->gpr[4] = backward ? src : src + n - 1u;
    ctx->gpr[5] = 0;
    ctx->gpr[6] = backward ? dst : dst + n - 1u;
    ctx->xer |= 0x20000000u;
    ctx->crv[0] = PpcCrvCompare(int32_t{0}, int32_t{0}, ctx->xer);
    return true;
}

// __fill_mem(dst = r3, value = r4, n = r5): n bytes of value & 0xFF. Under 32 bytes, byte stores; else byte stores up
// to a word boundary, words (the byte replicated), then the remaining 0-3 bytes: the last store is a word at
// dst + n - 4 when no byte remains, which decides the run's last mark pair. The destination may lie in a chunk the
// store policy flags only because a 4 KiB page next to it holds guarded code (P+ clears a heap buffer beside its
// Syriinge trampolines every frame): LeanStoresArePlain accepts it when no page of the range holds guarded code.
// Writes back every register the original writes, as it leaves them: r5 = 0, r6 = dst + n - 1, r7 = the byte
// (replicated to a word from 32 bytes on), XER[CA] set by any of its count-down loops (every n > 0), CR0 = r5's
// compare with zero; from 32 bytes on also r0 = 0, r3 = the stwu pointer (the aligned start - 4 + the words' bytes)
// and, when the byte is not zero, r4 = byte | byte << 8 (the first step of its replication).
bool FillMemNative(CpuContext* ctx) {
    const uint32_t dst = ctx->gpr[3], value = ctx->gpr[4], n = ctx->gpr[5];
    if (n > kMaxMemBytes || !LeanReady() || !MemoryInline::LeanStoresArePlain(dst, n)) return false;
    const uint8_t byte = static_cast<uint8_t>(value);
    if (n != 0) {
        std::memset(Window() + Canonical(dst), byte, n);
        ctx->xer |= 0x20000000u;
    }
    if (n < 32u) {
        if (n != 0) MarkStoreRun(dst, dst + n - 1u);
        ctx->gpr[7] = byte;
    } else {
        const uint32_t head = (0u - dst) & 3u, words = (n - head) & ~3u, tail = (n - head) & 3u;
        MarkStoreRun(dst, tail != 0 ? dst + n - 1u : dst + n - 4u);
        ctx->gpr[0] = 0;
        ctx->gpr[3] = dst + head - 4u + words;
        if (byte != 0) ctx->gpr[4] = byte | (static_cast<uint32_t>(byte) << 8);
        ctx->gpr[7] = byte * 0x01010101u;
    }
    ctx->gpr[5] = 0;
    ctx->gpr[6] = dst + n - 1u;
    ctx->crv[0] = PpcCrvCompare(int32_t{0}, int32_t{0}, ctx->xer);
    return true;
}
bool MemcpyWrites(const CpuContext& ctx, Ranges& ranges) {
    return ctx.gpr[5] <= kMaxMemBytes && (ctx.gpr[5] == 0 || ranges.Add(ctx.gpr[3], ctx.gpr[5]));
}

bool Mtx44CopyWrites(const CpuContext& ctx, Ranges& ranges) { return ranges.Add(ctx.gpr[3], 64u); }
bool CameraMtxWrites(const CpuContext& ctx, Ranges& ranges) {
    if (ctx.gpr[4] == 0) return true;
    if (!ranges.Add(ctx.gpr[4], 48u)) return false;
    const uint32_t cam = GuestLoad32(ctx.gpr[3]);
    return cam == 0 || ranges.Add(cam, 0x100u);  // UpdateCameraMtx's writes to the camera
}
bool LightSettingCopyWrites(const CpuContext& ctx, Ranges& ranges) {
    const uint32_t dst = ctx.gpr[3], src = ctx.gpr[4];
    const uint32_t numObj = GuestLoad16(src), numSet = GuestLoad16(src + 2u);
    return (numObj == 0 || (ranges.Add(GuestLoad32(dst + 4u), 68u * numObj) &&
                            ranges.Add(GuestLoad32(dst + 8u), 4u * numObj))) &&
           (numSet == 0 || ranges.Add(GuestLoad32(dst + 12u), 12u * numSet));
}

bool CalcViewLCWrites(const CpuContext& ctx, Ranges& ranges) {
    const uint32_t count = ctx.gpr[7];
    if (count > 0x2000u) return false;
    const uint32_t pos = (count * 48u + 31u) & ~31u, nrm = (count * 36u + 31u) & ~31u;
    if (!ranges.Add(0xE0000000u, 0x4000u) || !ranges.Add(kSkinTemporary, 48u * (count + 1)) ||
        (count != 0 && !ranges.Add(ctx.gpr[3], pos)))
        return false;
    if (count != 0 && ctx.gpr[4] != 0 && !ranges.Add(ctx.gpr[4], nrm)) return false;
    if (count != 0 && ctx.gpr[4] != 0 && ctx.gpr[10] != 0 && !ranges.Add(ctx.gpr[10], pos)) return false;
    return true;
}

bool CalcSkinningWrites(const CpuContext& ctx, Ranges& ranges) {
    const uint32_t mdl = MemoryInline::FlatRead32(ctx.gpr[5]);
    const uint32_t mtxCount = MemoryInline::FlatRead32(mdl + 0x5Cu);
    if (mtxCount == 0 || mtxCount > 0x2000u) return false;
    return ranges.Add(ctx.gpr[3], 48u * mtxCount) && ranges.Add(ctx.gpr[4], 4u * mtxCount) &&
           ranges.Add(kSkinTemporary, 48u * mtxCount);
}

// ---------------------------------------------------------------------------------------------------------------
// CHR keyframe readers: ogws g3d_resanmchr.cpp CalcAnimationFVS<FVS32 / FVS48 / FVS96> (0x801943DC, 0x8019426C,
// 0x80194140), called per animated SRT component from the CHR node evaluator (0x8019459C).
//   r3 = the track: u16 numFrameValues @0, f32 invKeyFrameRange @4; FVS32/48: f32 scale @8, offset @12, keys @16
//   (FVS32: one word per key, frame u8 | value u12 | slope s12; FVS48: s16 frame, u16 value, s16 slope); FVS96: keys
//   @8 (f32 frame, value, slope). f1 = the frame. Returns the value in f1.
// Each native is its original's instruction sequence: the same loads, the same stack frame bytes (stwu's back chain
// and the psq round trips through it: numFrameValues, the estimated index and the quantized frame, FVS32's key
// fields), the same float helpers in the same order, and the registers the vanilla (EABI) original writes back (r1,
// r3/r4 as each does, f1, f2). The quantized loads and stores follow the GQRs' encodings as PPC_PsqL*/PPC_PsqSt*Generic do, inline
// (the originals call the out-of-line generic bodies for every quantized access, and libm for each scale); a reserved
// encoding (1-3) declines.
// ---------------------------------------------------------------------------------------------------------------
// lfs: FlatReadFloat32Body's load and exact widening.
inline double GuestLoadFloat(uint32_t address) {
    return PpcBitCastToDoubleInline(PpcConvertToDoubleBitsInline(GuestLoad32(address)));
}
// Stores go to the reader's own stack frame, proven plain (PlainRange) first: FlatWriteRam* / WriteStack*'s mark
// pair and bytes.
inline void FrameStore8(uint32_t address, uint8_t v) {
    const uint32_t c = Canonical(address);
    MarkPair(c);
    Window()[c] = v;
}
inline void FrameStore16(uint32_t address, uint16_t v) {
    const uint32_t c = Canonical(address);
    MarkPair(c);
    v = __builtin_bswap16(v);
    std::memcpy(Window() + c, &v, 2);
}
inline void FrameStore32(uint32_t address, uint32_t v) {
    const uint32_t c = Canonical(address);
    MarkPair(c);
    Store32(Window() + c, v);
}

// std::ldexp(1.0f, e) for the psq scales' exponents (-32..32, normal floats), without the libm call.
inline float Pow2(int e) { return __builtin_bit_cast(float, static_cast<uint32_t>(127 + e) << 23); }
// PpcDequantizePsqInline.
template <typename T>
inline float Dequantize(T value, uint32_t scale) {
    if (scale == 0u) return static_cast<float>(value);
    return static_cast<float>(value) * (scale < 32u ? Pow2(-static_cast<int>(scale)) : Pow2(static_cast<int>(64u - scale)));
}
// PpcScaleAndClampPsqInline.
template <typename T>
inline T Quantize(float value, uint32_t scale) {
    constexpr float kMin = static_cast<float>(std::numeric_limits<T>::min());
    constexpr float kMax = static_cast<float>(std::numeric_limits<T>::max());
    if (scale == 0u) return static_cast<T>(std::clamp(value, kMin, kMax));
    const float scaled =
        value * (scale < 32u ? Pow2(static_cast<int>(scale)) : Pow2(-(64 - static_cast<int>(scale))));
    return static_cast<T>(std::clamp(scaled, kMin, kMax));
}
inline bool PsqLoadEncoding(uint32_t gqr) {
    const uint32_t type = (gqr >> 16) & 7u;
    return type == 0u || type >= 4u;
}
inline bool PsqStoreEncoding(uint32_t gqr) {
    const uint32_t type = gqr & 7u;
    return type == 0u || type >= 4u;
}
// psq_l fD, 0(address), 1, gqr (W=1): PPC_PsqLGqrInline<1, I> / PPC_PsqLStackInline<1, I>; the paired result.
inline double PsqLoad1(uint32_t gqr, uint32_t address) {
    const uint8_t* const host = Window() + Canonical(address);
    const uint32_t scale = (gqr >> 24) & 0x3Fu;
    float value;
    switch ((gqr >> 16) & 7u) {
    case 0u: value = PpcBitCastToFloatInline(Load32(host)); break;
    case 4u: value = Dequantize<uint8_t>(host[0], scale); break;
    case 5u: value = Dequantize<uint16_t>(Load16(host), scale); break;
    case 6u: value = Dequantize<int8_t>(static_cast<int8_t>(host[0]), scale); break;
    default: value = Dequantize<int16_t>(static_cast<int16_t>(Load16(host)), scale); break;
    }
    return PpcMakePairedResultInline(value, 1.0f).d;
}
// psq_st of PPC_PsFromScalarInline(scalar), W=1, to the frame: PPC_PsqStStackInline<1, I>.
inline void PsqStoreFrame1(uint32_t gqr, uint32_t address, double scalar) {
    const double packed = PPC_PsFromScalarInline(scalar);
    PPC_FPR fpr{};
    fpr.d = packed;
    const uint32_t scale = (gqr >> 8) & 0x3Fu;
    switch (gqr & 7u) {
    case 0u:
        FrameStore32(address, PpcConvertToSingleFTZInline(
                                  PpcBitCastToU64Inline(static_cast<double>(PpcGetPs0Inline(packed)))));
        return;
    case 4u: FrameStore8(address, Quantize<uint8_t>(fpr.paired.ps0, scale)); return;
    case 5u: FrameStore16(address, Quantize<uint16_t>(fpr.paired.ps0, scale)); return;
    case 6u: FrameStore8(address, static_cast<uint8_t>(Quantize<int8_t>(fpr.paired.ps0, scale))); return;
    default: FrameStore16(address, static_cast<uint16_t>(Quantize<int16_t>(fpr.paired.ps0, scale))); return;
    }
}
// psq_l W=1 through one GQR, decoded once per call (PsqLoad1 decodes it at every load): Dequantize's value is
// float(value) * 2^-scale (scale 0: the multiplier 1.0f, which leaves every value as it is).
struct PsqLoader {
    uint32_t type;
    float mul;
    explicit PsqLoader(uint32_t gqr) : type((gqr >> 16) & 7u) {
        const uint32_t scale = (gqr >> 24) & 0x3Fu;
        mul = scale == 0u ? 1.0f : scale < 32u ? Pow2(-static_cast<int>(scale)) : Pow2(static_cast<int>(64u - scale));
    }
    float Value(uint32_t address) const {
        const uint8_t* const host = Window() + Canonical(address);
        switch (type) {
        case 0u: return PpcBitCastToFloatInline(Load32(host));
        case 4u: return static_cast<float>(host[0]) * mul;
        case 5u: return static_cast<float>(Load16(host)) * mul;
        case 6u: return static_cast<float>(static_cast<int8_t>(host[0])) * mul;
        default: return static_cast<float>(static_cast<int16_t>(Load16(host))) * mul;
        }
    }
    // The pair psq_l leaves (ps1 = 1.0), for a value that stays in a register.
    double Pair(uint32_t address) const { return PpcMakePairedResultInline(Value(address), 1.0f).d; }
    // PPC_PsToScalarInline of that pair: static_cast<double>(ps0).
    double Scalar(uint32_t address) const { return static_cast<double>(Value(address)); }
};
// psq_st W=1 of PPC_PsFromScalarInline(scalar) to the frame through one GQR, decoded once per call (PsqStoreFrame1;
// Quantize's scaled value is value * 2^scale, scale 0 the multiplier 1.0f).
struct PsqFrameStorer {
    uint32_t type;
    float mul;
    explicit PsqFrameStorer(uint32_t gqr) : type(gqr & 7u) {
        const uint32_t scale = (gqr >> 8) & 0x3Fu;
        mul = scale == 0u ? 1.0f : scale < 32u ? Pow2(static_cast<int>(scale)) : Pow2(-(64 - static_cast<int>(scale)));
    }
    template <typename T>
    static T Clamp(float scaled) {
        constexpr float kMin = static_cast<float>(std::numeric_limits<T>::min());
        constexpr float kMax = static_cast<float>(std::numeric_limits<T>::max());
        return static_cast<T>(std::clamp(scaled, kMin, kMax));
    }
    void Store(uint32_t address, double scalar) const {
        const double packed = PPC_PsFromScalarInline(scalar);
        if (type == 0u) {
            FrameStore32(address, PpcConvertToSingleFTZInline(
                                      PpcBitCastToU64Inline(static_cast<double>(PpcGetPs0Inline(packed)))));
            return;
        }
        PPC_FPR fpr{};
        fpr.d = packed;
        const float scaled = fpr.paired.ps0 * mul;
        switch (type) {
        case 4u: FrameStore8(address, Clamp<uint8_t>(scaled)); return;
        case 5u: FrameStore16(address, Clamp<uint16_t>(scaled)); return;
        case 6u: FrameStore8(address, static_cast<uint8_t>(Clamp<int8_t>(scaled))); return;
        default: FrameStore16(address, static_cast<uint16_t>(Clamp<int16_t>(scaled))); return;
        }
    }
};
// fsubs
inline double SubSingle(double a, double b) {
    return static_cast<double>(PpcForceSingleValueInline(PpcFsubInline(a, b)));
}

struct KeyValue {
    double f1, f2;
};
// The readers' shared tail between keys 0 and 1 (the same instructions in all three): d = frame - frame0, span =
// frame1 - frame0 (fsubs), values v0, v1 and slopes s0, s1; t = d * recip(span) (fres and one ps Newton step), then
// HermiteInterpolation's fp_contract form. Constants: 1.0 (r2-24876), 2.0 (r2-24872), 3.0 (r2-24868).
inline KeyValue Hermite(uint32_t r2, double d, double span, double v0, double v1, double s0, double s1) {
    const double one = GuestLoadFloat(r2 - 24876u), two = GuestLoadFloat(r2 - 24872u),
                 three = GuestLoadFloat(r2 - 24868u);
    const double dv = SubSingle(v0, v1);
    const double estimate = PPC_Fres(PPC_PsFromScalarInline(span));
    const double twice = PPC_PsAddInline(estimate, estimate);
    const double square = PPC_PsMulInline(estimate, estimate);
    const double recip = PPC_PsNmsubInline(PPC_PsFromScalarInline(span), square, twice);
    // Every C operand below is a single (a paired register's PS0, a single-precision result, a dequantized value), so
    // the runtime's SingleC forms apply: Gekko's 25-bit rounding of C is the identity on a single, NaN included
    // (ppc_isa_float.h, at PpcFmulsSingleCInline), and those forms are the same operations without that step.
    const double t = PpcFmulsSingleCInline(d, PPC_PsToScalarInline(recip));
    const double tm1 = PPC_FmsubsSingleC(d, PPC_PsToScalarInline(recip), one);
    const double a = PPC_FmsubsSingleC(two, t, three);
    const double f2 = PpcFmulsSingleCInline(t, s1);
    const double f4 = PpcFmulsSingleCInline(d, tm1);
    double f0 = PpcFmulsSingleCInline(a, dv);
    const double f1 = PPC_FmaddsSingleC(tm1, s0, f2);
    f0 = PpcFmulsSingleCInline(t, f0);
    f0 = PPC_FmaddsSingleC(t, f0, v0);
    return {PPC_FmaddsSingleC(f4, f1, f0), f2};
}
// A key's value for FVS32/48: scale * dequantized + offset (lfs, lfs, fmadds; C the dequantized single).
inline double ScaledValue(uint32_t track, double dequantized) {
    return PPC_FmaddsSingleC(GuestLoadFloat(track + 8u), PPC_PsToScalarInline(dequantized), GuestLoadFloat(track + 12u));
}
// ScaledValue for a value already taken out of its pair (PPC_PsToScalarInline of the psq_l result).
inline double ScaledScalar(uint32_t track, double value) {
    return PPC_FmaddsSingleC(GuestLoadFloat(track + 8u), value, GuestLoadFloat(track + 12u));
}
inline bool TrackHeaderOk(uint32_t track) { return ((track ^ (track + 15u)) & ~MemoryInline::kLeanMask) == 0; }

// FVS32 (0x801943DC): a 32-byte frame. Writes back r1, r3, r4, f1, f2.
bool KeyFVS32Native(CpuContext* ctx) {
    const uint32_t gqr2 = ctx->gqr[2], gqr3 = ctx->gqr[3], gqr6 = ctx->gqr[6];
    const uint32_t r1 = ctx->gpr[1], r2 = ctx->gpr[2], track = ctx->gpr[3];
    const uint32_t sp = r1 - 32u;
    if (!LeanReady() || !PsqLoadEncoding(gqr2) || !PsqStoreEncoding(gqr2) || !PsqLoadEncoding(gqr3) ||
        !PsqStoreEncoding(gqr3) || !PsqLoadEncoding(gqr6) || !PlainRange(sp, 32u) || !TrackHeaderOk(track))
        return false;
    FrameStore32(sp, r1);
    const double frame = ctx->fpr[1].d;
    const uint32_t numKeys = GuestLoad16(track);
    const uint32_t first = track + 16u;
    const uint32_t last = track + ((numKeys - 1u) << 2) + 16u;
    uint32_t r3 = track, r4 = last;
    double f1, f2 = ctx->fpr[2].d;
    // The three GQRs' psq_l and psq_st, decoded once (PsqLoader, PsqFrameStorer: PsqLoad1's and PsqStoreFrame1's values).
    const PsqLoader keyFrame(gqr2), keyValue(gqr3), keySlope(gqr6);
    const PsqFrameStorer storeIndex(gqr3), storeFrame(gqr2);
    // A key word's value through the frame slot: u12 stored as u16, psq_l GQR3 (its pair stays in f2), scale and offset.
    const auto value = [&](uint32_t key, uint32_t slot) {
        FrameStore16(sp + slot, static_cast<uint16_t>((GuestLoad32(key) >> 12) & 0xFFFu));
        f2 = keyValue.Pair(sp + slot);
        return ScaledValue(track, f2);
    };
    if (frame <= keyFrame.Scalar(first)) {
        f1 = value(first, 22u);
    } else if (keyFrame.Scalar(last) <= frame) {
        f1 = value(last, 20u);
    } else {
        const double first0 = keyFrame.Scalar(first);
        FrameStore16(sp + 18u, static_cast<uint16_t>(numKeys));
        const double f3 = SubSingle(frame, first0);
        f2 = PpcFmulsSingleCInline(f3, keyValue.Scalar(sp + 18u));  // (C singles: see Hermite)
        storeIndex.Store(sp + 28u, PpcFmulsSingleCInline(GuestLoadFloat(track + 4u), f2));
        uint32_t key = track + (GuestLoad16(sp + 28u) << 2) + 16u;
        storeFrame.Store(sp + 24u, frame);
        r4 = GuestLoad8(sp + 24u);
        uint32_t k = GuestLoad8(key);
        if (r4 >= k) {
            do {
                key += 4u;
                k = GuestLoad8(key);
            } while (k <= r4);
            key -= 4u;
        } else {
            do {
                key -= 4u;
                k = GuestLoad8(key);
            } while (r4 < k);
        }
        if (frame == keyFrame.Scalar(key)) {
            f1 = value(key, 16u);
        } else {
            const uint32_t w0 = GuestLoad32(key);
            FrameStore16(sp + 14u, static_cast<uint16_t>((w0 >> 12) & 0xFFFu));
            const double q0 = keyValue.Scalar(sp + 14u);
            FrameStore16(sp + 12u, static_cast<uint16_t>(static_cast<int32_t>(w0 << 20) >> 20));
            const double v0 = ScaledScalar(track, q0);
            const double s0 = keySlope.Scalar(sp + 12u);
            const uint32_t w1 = GuestLoad32(key + 4u);
            FrameStore16(sp + 10u, static_cast<uint16_t>((w1 >> 12) & 0xFFFu));
            const double q1 = keyValue.Scalar(sp + 10u);
            FrameStore16(sp + 8u, static_cast<uint16_t>(static_cast<int32_t>(w1 << 20) >> 20));
            const double v1 = ScaledScalar(track, q1);
            const double s1 = keySlope.Scalar(sp + 8u);
            const double k0 = keyFrame.Scalar(key), k1 = keyFrame.Scalar(key + 4u);
            const KeyValue kv = Hermite(r2, SubSingle(frame, k0), SubSingle(k1, k0), v0, v1, s0, s1);
            f1 = kv.f1;
            f2 = kv.f2;
            r3 = w1;
            r4 = w0;
        }
    }
    ctx->gpr[3] = r3;
    ctx->gpr[4] = r4;
    ctx->fpr[1].d = f1;
    ctx->fpr[2].d = f2;
    return true;
}

// FVS48 (0x8019426C): a 16-byte frame. Writes back r1, r4, f1, f2.
bool KeyFVS48Native(CpuContext* ctx) {
    const uint32_t gqr3 = ctx->gqr[3], gqr6 = ctx->gqr[6], gqr7 = ctx->gqr[7];
    const uint32_t r1 = ctx->gpr[1], r2 = ctx->gpr[2], track = ctx->gpr[3];
    const uint32_t sp = r1 - 16u;
    if (!LeanReady() || !PsqLoadEncoding(gqr3) || !PsqStoreEncoding(gqr3) || !PsqLoadEncoding(gqr6) ||
        !PsqStoreEncoding(gqr6) || !PsqLoadEncoding(gqr7) || !PlainRange(sp, 16u) || !TrackHeaderOk(track))
        return false;
    FrameStore32(sp, r1);
    const double frame = ctx->fpr[1].d;
    const uint32_t numKeys = GuestLoad16(track);
    const uint32_t first = track + 16u;
    const uint32_t last = track + (numKeys - 1u) * 6u + 16u;
    uint32_t r4 = last;
    double f1, f2 = ctx->fpr[2].d;
    // The GQRs' psq_l and psq_st, decoded once (PsqLoad1's and PsqStoreFrame1's values).
    const PsqLoader keyFrame(gqr6), keyValue(gqr3), keySlope(gqr7);
    const PsqFrameStorer storeIndex(gqr3), storeFrame(gqr6);
    if (frame <= keyFrame.Scalar(first)) {
        f2 = keyValue.Pair(first + 2u);
        f1 = ScaledValue(track, f2);
    } else if (keyFrame.Scalar(last) <= frame) {
        f2 = keyValue.Pair(last + 2u);
        f1 = ScaledValue(track, f2);
    } else {
        const double first0 = keyFrame.Scalar(first);
        FrameStore16(sp + 10u, static_cast<uint16_t>(numKeys));
        const double f3 = SubSingle(frame, first0);
        f2 = PpcFmulsSingleCInline(f3, keyValue.Scalar(sp + 10u));  // (C singles: see Hermite)
        storeIndex.Store(sp + 12u, PpcFmulsSingleCInline(GuestLoadFloat(track + 4u), f2));
        uint32_t key = track + GuestLoad16(sp + 12u) * 6u + 16u;
        storeFrame.Store(sp + 8u, frame);
        const int32_t quantized = static_cast<int16_t>(GuestLoad16(sp + 8u));
        r4 = static_cast<uint32_t>(quantized);
        int32_t k = static_cast<int16_t>(GuestLoad16(key));
        if (quantized >= k) {
            do {
                key += 6u;
                k = static_cast<int16_t>(GuestLoad16(key));
            } while (k <= quantized);
            key -= 6u;
        } else {
            do {
                key -= 6u;
                k = static_cast<int16_t>(GuestLoad16(key));
            } while (quantized < k);
        }
        if (frame == keyFrame.Scalar(key)) {
            f2 = keyValue.Pair(key + 2u);
            f1 = ScaledValue(track, f2);
        } else {
            const uint32_t next = key + 6u;
            const double v0 = ScaledScalar(track, keyValue.Scalar(key + 2u));
            const double s0 = keySlope.Scalar(key + 4u);
            const double v1 = ScaledScalar(track, keyValue.Scalar(next + 2u));
            const double s1 = keySlope.Scalar(next + 4u);
            const double k0 = keyFrame.Scalar(key), k1 = keyFrame.Scalar(next);
            const KeyValue kv = Hermite(r2, SubSingle(frame, k0), SubSingle(k1, k0), v0, v1, s0, s1);
            f1 = kv.f1;
            f2 = kv.f2;
            r4 = next;
        }
    }
    ctx->gpr[4] = r4;
    ctx->fpr[1].d = f1;
    ctx->fpr[2].d = f2;
    return true;
}

// FVS96 (0x80194140): a 16-byte frame. Writes back r1, r3, r4, f1, f2.
bool KeyFVS96Native(CpuContext* ctx) {
    const uint32_t gqr3 = ctx->gqr[3];
    const uint32_t r1 = ctx->gpr[1], r2 = ctx->gpr[2], track = ctx->gpr[3];
    const uint32_t sp = r1 - 16u;
    if (!LeanReady() || !PsqLoadEncoding(gqr3) || !PsqStoreEncoding(gqr3) || !PlainRange(sp, 16u) ||
        !TrackHeaderOk(track))
        return false;
    FrameStore32(sp, r1);
    const double frame = ctx->fpr[1].d;
    const uint32_t numKeys = GuestLoad16(track);
    const uint32_t first = track + 8u;
    const uint32_t last = track + (numKeys - 1u) * 12u + 8u;
    uint32_t r3 = track, r4 = last;
    double f1, f2 = ctx->fpr[2].d;
    if (frame <= GuestLoadFloat(first)) {
        f1 = GuestLoadFloat(first + 4u);
    } else if (GuestLoadFloat(last) <= frame) {
        f1 = GuestLoadFloat(last + 4u);
    } else {
        FrameStore16(sp + 8u, static_cast<uint16_t>(numKeys));
        const double f3 = SubSingle(frame, GuestLoadFloat(first));
        f2 = PsqLoad1(gqr3, sp + 8u);
        f2 = PpcFmulsInline(f3, PPC_PsToScalarInline(f2));
        PsqStoreFrame1(gqr3, sp + 12u, PpcFmulsInline(GuestLoadFloat(track + 4u), f2));
        r3 = track + GuestLoad16(sp + 12u) * 12u;
        uint32_t key = r3 + 8u;
        if (frame < GuestLoadFloat(key)) {
            do key -= 12u;
            while (frame < GuestLoadFloat(key));
        } else {
            do key += 12u;
            while (GuestLoadFloat(key) <= frame);
            key -= 12u;
        }
        r4 = key;
        const double k0 = GuestLoadFloat(key);
        if (frame == k0) {
            f1 = GuestLoadFloat(key + 4u);
        } else {
            const KeyValue kv =
                Hermite(r2, SubSingle(frame, k0), SubSingle(GuestLoadFloat(key + 12u), k0), GuestLoadFloat(key + 4u),
                        GuestLoadFloat(key + 16u), GuestLoadFloat(key + 8u), GuestLoadFloat(key + 20u));
            f1 = kv.f1;
            f2 = kv.f2;
        }
    }
    ctx->gpr[3] = r3;
    ctx->gpr[4] = r4;
    ctx->fpr[1].d = f1;
    ctx->fpr[2].d = f2;
    return true;
}

// The readers write only their stack frame (inside verify's stack window).
bool NoArrayWrites(const CpuContext&, Ranges&) { return true; }
#endif  // MKW_LEAN_MEMORY

}  // namespace
}  // namespace g3dcalc

extern "C" void G3dCalc_CalcViewLC(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    g3dcalc::Run(g3dcalc::kCalcViewLC, ctx, func_801A3D68_body, g3dcalc::CalcViewLCWrites,
                 g3dcalc::CalcViewLCNative);
#else
    func_801A3D68_body(ctx);
#endif
}

// ut::LC::StoreBlocks(dst = r3, src = r4, blocks = r5): mtspr DMA_U = dst & 0x1FFFFFFF | (blocks >> 2 & 31), mtspr
// DMA_L = src | (blocks & 3) << 2 | 2 (trigger). Outside CalcView_LC: the translated original. Inside: the same SPR
// writes, copy and marks as the runtime's DMA_L write (ppc_helpers.cpp PPC_WriteSpr 923), its notification queued.
extern "C" void G3dCalc_LCStoreBlocks(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    if (!g3dcalc::t_deferDmaNotify) {
        func_8015D8BC_body(ctx);
        return;
    }
    const uint32_t dst = ctx->gpr[3] & 0x1FFFFFFFu;
    const uint32_t blocksArg = ctx->gpr[5];
    const uint32_t dmaU = ((blocksArg >> 2) & 31u) | dst;
    PPC_WriteSpr(922, dmaU);
    const uint32_t dmaL = ctx->gpr[4] | ((blocksArg << 2) & 12u);
    uint32_t blocks = ((dmaU & 0x1Fu) << 2) | ((dmaL >> 2) & 3u);
    if (blocks == 0) blocks = 128;
    const uint32_t memAddr = dmaU & ~0x1Fu, lcAddr = dmaL & ~0x1Fu, bytes = blocks * 32u;
    uint8_t* const mem = Memory::GetPointer(memAddr, bytes);
    uint8_t* const lc = Memory::GetPointer(lcAddr, bytes);
    if (mem != nullptr && lc != nullptr) {
        if ((dmaL & 0x10u) != 0) {
            BrawlDirtyMemory::Mark(lcAddr, bytes);
            std::memcpy(lc, mem, bytes);
        } else {
            BrawlDirtyMemory::Mark(memAddr, bytes);
            std::memcpy(mem, lc, bytes);
            g3dcalc::QueueDmaNotify(memAddr, bytes);
        }
    }
    PPC_WriteSpr(923, dmaL);  // the shadow the triggered write leaves (T reads back clear)
    ctx->gpr[3] = dst;
    ctx->gpr[6] = dmaL | 2u;
#else
    func_8015D8BC_body(ctx);
#endif
}

extern "C" void G3dCalc_CalcSkinning(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    g3dcalc::Run(g3dcalc::kSkinning, ctx, func_801A5414_body, g3dcalc::CalcSkinningWrites,
                 g3dcalc::CalcSkinningNative);
#else
    func_801A5414_body(ctx);
#endif
}

extern "C" void G3dCalc_KeyFVS32(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    g3dcalc::Run(g3dcalc::kKeyFVS32, ctx, func_801943DC_body, g3dcalc::NoArrayWrites, g3dcalc::KeyFVS32Native);
#else
    func_801943DC_body(ctx);
#endif
}

extern "C" void G3dCalc_KeyFVS48(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    g3dcalc::Run(g3dcalc::kKeyFVS48, ctx, func_8019426C_body, g3dcalc::NoArrayWrites, g3dcalc::KeyFVS48Native);
#else
    func_8019426C_body(ctx);
#endif
}

extern "C" void G3dCalc_KeyFVS96(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    g3dcalc::Run(g3dcalc::kKeyFVS96, ctx, func_80194140_body, g3dcalc::NoArrayWrites, g3dcalc::KeyFVS96Native);
#else
    func_80194140_body(ctx);
#endif
}

extern "C" void G3dCalc_Mtx44Copy(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    g3dcalc::Run(g3dcalc::kMtx44Copy, ctx, func_80162C04_body, g3dcalc::Mtx44CopyWrites, g3dcalc::Mtx44CopyNative);
#else
    func_80162C04_body(ctx);
#endif
}

extern "C" void G3dCalc_CameraMtx(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    g3dcalc::Run(g3dcalc::kCameraMtx, ctx, func_801A7DBC_body, g3dcalc::CameraMtxWrites, g3dcalc::CameraMtxNative);
#else
    func_801A7DBC_body(ctx);
#endif
}

extern "C" void G3dCalc_LightSettingCopy(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    g3dcalc::Run(g3dcalc::kLightSettingCopy, ctx, func_801B6708_body, g3dcalc::LightSettingCopyWrites,
                 g3dcalc::LightSettingCopyNative);
#else
    func_801B6708_body(ctx);
#endif
}

extern "C" void G3dCalc_LightSlot(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    g3dcalc::Run(g3dcalc::kLightSlot, ctx, func_801A0DD8_body, g3dcalc::LightSlotWrites, g3dcalc::LightSlotNative);
#else
    func_801A0DD8_body(ctx);
#endif
}

extern "C" void G3dCalc_Memcpy(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    g3dcalc::Run(g3dcalc::kMemcpy, ctx, func_80004338_body, g3dcalc::MemcpyWrites, g3dcalc::MemcpyNative);
#else
    func_80004338_body(ctx);
#endif
}

extern "C" void G3dCalc_FillMem(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    g3dcalc::Run(g3dcalc::kFillMem, ctx, func_80004388_body, g3dcalc::MemcpyWrites, g3dcalc::FillMemNative);
#else
    func_80004388_body(ctx);
#endif
}

extern "C" void G3dCalc_EnvTexMtx(CpuContext* ctx) {
#if MKW_LEAN_MEMORY
    g3dcalc::Run(g3dcalc::kEnvTexMtx, ctx, func_801A7FC8_body, g3dcalc::EnvTexMtxWrites, g3dcalc::EnvTexMtxNative);
#else
    func_801A7FC8_body(ctx);
#endif
}

// The fps log's window line (vi.cpp NSMBW_LOG_FPS, through G3dNativeLogStats): per native, calls per frame and how
// they went.
void G3dCalcLogStats(double frames) {
    using namespace g3dcalc;
    if (frames <= 0) return;
    InsnCount::Report(frames);  // NAT_INSN=1: every counted site's window (g3d_calc, mtx, axfx and any other user)
    std::string line;
    for (int i = 0; i < kNumFn; ++i) {
        Counters& c = g_counters[i];
        const uint64_t calls = c.calls.exchange(0, std::memory_order_relaxed);
        const uint64_t native = c.native.exchange(0, std::memory_order_relaxed);
        const uint64_t declined = c.declined.exchange(0, std::memory_order_relaxed);
        const uint64_t compared = c.compared.exchange(0, std::memory_order_relaxed);
        const uint64_t mismatches = c.mismatches.exchange(0, std::memory_order_relaxed);
        const uint64_t undeclared = c.undeclared.exchange(0, std::memory_order_relaxed);
        const uint64_t foreign = c.foreign.exchange(0, std::memory_order_relaxed);
        const uint64_t overmarked = c.overmarked.exchange(0, std::memory_order_relaxed);
        if (calls == 0) continue;
        char buf[280];
        std::snprintf(buf, sizeof(buf), "%s%s %.1f calls (native %.1f, declined %.1f, verified %.1f, mismatch %.1f, "
                      "undeclared %.1f, foreign %.1f, span marks differ %.1f)", line.empty() ? "" : "; ",
                      kFnNames[i], calls / frames, native / frames, declined / frames, compared / frames,
                      mismatches / frames, undeclared / frames, foreign / frames, overmarked / frames);
        line += buf;
    }
#if MKW_LEAN_MEMORY
    // CalcView_LC's native (calcview's native/verified calls): its calls, declines and per-matrix paths.
    CalcViewCounters& v = g_calcView;
    if (const uint64_t full = v.full.exchange(0, std::memory_order_relaxed),
        declined = v.fullDeclined.exchange(0, std::memory_order_relaxed);
        full != 0 || declined != 0) {
        char buf[320];
        std::snprintf(buf, sizeof(buf), "; cvfull %.1f calls (declined %.1f): %.1f matrices, uniform %.1f, invxpose "
                      "%.1f, invtranspose %.1f, billboard %.1f (child %.1f), generic calls %.1f",
                      full / frames, declined / frames, v.matrices.exchange(0, std::memory_order_relaxed) / frames,
                      v.uniform.exchange(0, std::memory_order_relaxed) / frames,
                      v.invXpose.exchange(0, std::memory_order_relaxed) / frames,
                      v.invTranspose.exchange(0, std::memory_order_relaxed) / frames,
                      v.billboard.exchange(0, std::memory_order_relaxed) / frames,
                      v.billboardChild.exchange(0, std::memory_order_relaxed) / frames,
                      v.generic.exchange(0, std::memory_order_relaxed) / frames);
        line += buf;
    }
#endif
    if (!line.empty()) RT_LOGF(RT_TAG_VI, "g3d calc (G3D_CALC=%d): %s\n", Config().mode, line.c_str());
}
