// Native nw4r g3d draw path. nw4r::g3d::DrawResMdlDirectly (0x801A64B0) and its
// subtree run as host code for the models it can draw exactly, through the translator's native hook with guest
// fallback (MKW_NATIVE_HOOK in abi_bridge.h): a model or feature it does not handle runs the translated function.
//
// It is a port of the guest code (doldecomp/ogws g3d_draw.cpp, g3d_draw1mat1shp.cpp, g3d_state.cpp: nw4r g3d for
// MDL0 rev 9, checked against Brawl's translated functions). It keeps nw4r's G3DState in guest memory, read and
// written where the guest keeps it, so translated g3d code drawing before or after it sees exactly the state the
// translated DrawResMdlDirectly would have left. It reaches the GX through the same HLE entry points the guest code
// does, with the same arguments in the same order (GX FIFO bytes, GXCallDisplayList, the GX calls nw4r makes); the
// rare, float-heavy loads (a fog change, texture-matrix animation) and the Z sort call the translated nw4r functions
// themselves. Below that boundary it takes four shortcuts with the same effects: its GD bytes applied packet by
// packet without the FIFO parser (GX_HLE_FIFO_WritePackets_G3D), its material lists without the display-list scan
// cache's probe (GX__CallRegisterDisplayList_G3D), its shape lists from per-list records instead of that probe
// (GX__CallShapeDisplayList_G3D), and its shapes' draws submitted to Aurora from a compiled form
// (GXCallDisplayListCompiled: one vertex and one index push per shape, or per matrix group of a skinned shape,
// instead of one per strip).
//
// G3D_NATIVE=1 draws natively; 2 verifies each native call against the translated one in the same frame (see
// g3d_native.h) and draws with the translated one; unset or 0: off. G3D_NATIVE_FAST=0, G3D_NATIVE_SHAPE=0 and
// G3D_NATIVE_COMPILED=0 turn the shortcuts off; G3D_NATIVE_ENVELOPE=0 declines models with skinned shapes.
// G3D_NATIVE_MODELS=a,b limits it to models whose names contain a or b. G3D_NATIVE_LOG=1 logs each model once with
// its verdict. The per-scene caches (compiled lists, shape-list records, model names) go on every scene change.
#include "abi_bridge.h"
#include "g3d_native.h"
#include "guest_native_trampoline.h"
#include "gx_guest_write.h"
#include "memory.h"
#include "runtime_log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#if defined(__x86_64__)
#include <x86intrin.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// The few float operations here must round exactly as the guest's (see LoadResMatChan): no fused multiply-add.
#pragma clang fp contract(off)

extern "C" __attribute__((visibility("hidden"))) MkwNativeHookFn g_mkwNativeHook[4] = {};

// GX HLE entry points (hle/gx): what the translated nw4r code reaches.
extern "C" void GX_HLE_FIFO_WriteBurst(const uint8_t* data, uint32_t sizeBytes);
extern "C" void GX__CallDisplayList_HLE(uint32_t listAddr, uint32_t nbytes);
extern "C" void GX_HLE_SyncGuestProjection();  // gx_transform.cpp
// Their shortcuts for this path's own calls (G3D_NATIVE_FAST, default on): the same effects, less parsing and probing.
extern "C" void GX_HLE_FIFO_WritePackets_G3D(const uint8_t* data, uint32_t sizeBytes);  // gx_fifo.cpp
extern "C" void GX__CallRegisterDisplayList_G3D(uint32_t listAddr, uint32_t nbytes);     // gx_dl.cpp
extern "C" void GX__CallShapeDisplayList_G3D(uint32_t listAddr, uint32_t nbytes);        // gx_dl.cpp
extern "C" void GX__ClearShapeDisplayLists_G3D();                                        // gx_dl.cpp
extern "C" void GX__LoadTexObj_HLE(uint32_t oa, uint32_t tid);
extern "C" void GX__LoadTlut_HLE(uint32_t oa, uint32_t tl);
extern "C" void GX__SetArray_HLE(uint32_t a, uint32_t ba, uint32_t str);
extern "C" void GX__SetZCompLoc_HLE(uint32_t bt);
extern "C" void GX__LoadTexMtxImm_HLE(uint32_t ma, uint32_t id, uint32_t t);
extern "C" void GX__LoadLightObjImm_HLE(uint32_t la, uint32_t lid);
// The replay's by-value forms (gx_transform.cpp, gx_pixel.cpp).
extern "C" void GX_HLE_LoadPosMtxImmHost(const float* m, uint32_t id);
extern "C" void GX_HLE_LoadNrmMtxImmHost(const float* m, uint32_t id);
extern "C" void GX_HLE_LoadTexMtxImmHost(const float* m, uint32_t id, uint32_t t);
extern "C" void GX_HLE_SetFogHost(uint32_t t, float sz, float ez, float nz, float fz, uint32_t color);
// Deferred draws (G3D_NATIVE_ASYNC=2): the HLE's per-thread state and the game thread's half of a deferral
// (gx_dl.cpp, gx_fifo.cpp, gx_frame.cpp, gx_texture.cpp), Aurora's register shadow (GXManage.cpp) and deferred jobs
// (gx_worker.hpp).
extern "C" size_t GX_HLE_VertexStateBytes();
extern "C" void GX_HLE_SaveVertexState(void* out);
extern "C" void GX_HLE_LoadVertexState(const void* in);
extern "C" bool GX_HLE_IdleForDeferral();
extern "C" void GX_HLE_EnsureFrameActive();
extern "C" void GX_HLE_InvalidatePublishedState();
extern "C" void GX_HLE_KeepPublishedStateForNewJob();
extern "C" bool GX_HLE_VertexStatePending();
extern "C" bool GX_HLE_DisplayListRecording();
extern "C" void GX_HLE_SetVertexStatePending(bool pending);
// The carry's rewrite tracking (gx_fifo.cpp; gx_internal.h GxCarryRewriting).
extern "C" bool GX_HLE_VtxRewritesPending();
extern "C" void GX_HLE_VtxRewritesBegin(void* saved);
extern "C" void GX_HLE_VtxRewritesFinish(const void* saved);
extern "C" uint64_t GX_HLE_TakeVtxRewriteDraws();
extern "C" size_t GXShadowVtxBytes();
extern "C" void GXShadowSaveVtx(void* out);
extern "C" void GXShadowLoadVtx(const void* in);
extern "C" void GXShadowAfterDeferredDraw(int zCompLoc);
extern "C" void GXShadowSaveDirty(uint32_t out[2]);
extern "C" void GXShadowLoadDirty(const uint32_t in[2]);
extern "C" bool GX_HLE_ApplyFifoVertexState(const uint8_t* data, uint32_t sizeBytes, uint32_t* drawFmtMask);
extern "C" bool GX_HLE_ApplyListVertexState(uint32_t listAddr, uint32_t nbytes, uint32_t* drawFmtMask);
extern "C" void GX_HLE_AfterDeferredDraw(int zCompLoc, uint32_t drawFmtMask);
extern "C" void GX_HLE_DeferredPublishBegin();  // gx_dl.cpp: GX_HLE_AfterDeferredDraw over a run of mirrored draws
extern "C" void GX_HLE_DeferredPublishNote(uint32_t drawFmtMask);
extern "C" void GX_HLE_DeferredPublishFinish(bool published);
extern "C" void GX_HLE_SetDrawWorker(bool worker);
extern "C" size_t GX_HLE_TexLoadPlanBytes();
extern "C" void GX_HLE_PlanTexLoadForJob(uint32_t oa, uint32_t tid, void* out);
extern "C" void GX_HLE_RunTexLoadPlan(const void* in, uint32_t oa, uint32_t tid);
extern "C" size_t GX_HLE_TlutLoadPlanBytes();
extern "C" void GX_HLE_PlanTlutLoadForJob(uint32_t oa, uint32_t tl, void* out);
extern "C" void GX_HLE_RunTlutLoadPlan(const void* in, uint32_t tl);
extern "C" size_t GXShadowBytes();
extern "C" void GXShadowSave(void* out);
extern "C" void GXShadowUseOwn(void* storage);
extern "C" void GXShadowLoad(const void* in);
extern "C" uint32_t GXShadowDirtyState();
extern "C" void GXShadowFlushDirty();
namespace aurora::gfx::texture_replacement {
bool tlut_mirror_active() noexcept;
}  // namespace aurora::gfx::texture_replacement
namespace aurora::gx::worker {
struct DeferredJob;
DeferredJob* begin_deferred_job(bool* followsJob = nullptr);
void attach_deferred_job(DeferredJob* job) noexcept;
void finish_deferred_job() noexcept;
void flush() noexcept;
}  // namespace aurora::gx::worker
// Translated guest functions this calls.
extern "C" void func_801A64B0_body(CpuContext* ctx);  // nw4r::g3d::DrawResMdlDirectly (the fallback)
extern "C" void func_801A0E7C(CpuContext* ctx);       // G3DState::LoadFog(int)
extern "C" void func_801A04EC(CpuContext* ctx);       // G3DState::LoadResTexSrt(ResTexSrt)
extern "C" void func_801A093C(CpuContext* ctx);       // G3DState::LoadResShpPrimitive(ResShp, MTX34*, MTX34*)
extern "C" void func_801D7AE4(CpuContext* ctx);       // LCQueueWait(u32)
extern "C" void func_801A6378(CpuContext* ctx);       // SetUpMdlZ(u32*, ResMdl, MTX34*, u8* byteCode, replacement)
extern "C" void func_801A6778(CpuContext* ctx);       // std::sort(MdlZ*, MdlZ*, comparator)
// The translator's guest code guard table (generated base_dispatch/guest_code_guard_*.cpp; abi_bridge.h).
extern "C" const uint32_t g_mkwGuardedFunctionStarts[];
extern "C" const uint32_t g_mkwGuardedFunctionCount;

namespace g3dn {

// ---------------------------------------------------------------------------------------------------------------
// Recorder (g3d_native.h)
// ---------------------------------------------------------------------------------------------------------------
// A byte vector whose resize leaves new elements uninitialized (value-initialization zeroed every recorded event's
// bytes just before RecEvent wrote them). Only for buffers that are written before they are read: the recording and
// what holds it (jobs, the verify copy).
template <typename T>
struct DefaultInitAllocator : std::allocator<T> {
    template <typename U>
    struct rebind {
        using other = DefaultInitAllocator<U>;
    };
    DefaultInitAllocator() noexcept = default;
    template <typename U>
    DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept {}
    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
};
// The recording's byte buffer: a vector's interface for what reads it (size, data, [], swap, ==), and an append (Grow)
// whose capacity check is the only work in line - the reallocation is out of line, so the appends that inline it keep
// their registers (RecEvent's std::vector resize kept six callee-saved registers busy on every event).
class RecBuf {
public:
    RecBuf() = default;
    ~RecBuf() { std::free(data_); }
    RecBuf(const RecBuf&) = delete;
    RecBuf& operator=(const RecBuf&) = delete;
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    uint8_t* data() { return data_; }
    const uint8_t* data() const { return data_; }
    uint8_t& operator[](size_t i) { return data_[i]; }
    const uint8_t& operator[](size_t i) const { return data_[i]; }
    void clear() { size_ = 0; }
    void swap(RecBuf& o) noexcept {
        std::swap(data_, o.data_);
        std::swap(size_, o.size_);
        std::swap(cap_, o.cap_);
    }
    // n more bytes, uninitialized (every caller writes them at once); where they start.
    uint8_t* Grow(size_t n) {
        if (cap_ - size_ < n) [[unlikely]] Reserve(size_ + n);
        uint8_t* p = data_ + size_;
        size_ += n;
        return p;
    }
    friend bool operator==(const RecBuf& a, const RecBuf& b) {
        return a.size_ == b.size_ && (a.size_ == 0 || std::memcmp(a.data_, b.data_, a.size_) == 0);
    }

private:
    [[gnu::noinline, gnu::cold]] void Reserve(size_t want) {
        const size_t cap = std::max<size_t>(want, cap_ * 2 + 4096);
        auto* p = static_cast<uint8_t*>(std::realloc(data_, cap));
        if (p == nullptr) std::abort();
        data_ = p;
        cap_ = cap;
    }
    uint8_t* data_ = nullptr;
    size_t size_ = 0, cap_ = 0;
};
using RecBytes = RecBuf;
using PlanBytes = std::vector<uint8_t, DefaultInitAllocator<uint8_t>>;  // a job's load plans (PlanLoads)

thread_local bool g_nativeDraw = false;
thread_local bool g_recOn = false;
thread_local bool g_recSuppress = false;
thread_local bool g_recJob = false;
thread_local int g_recDepth = 0;
struct VerifyGuestByte {
    uint32_t address;
    uint8_t before, native;
};
static std::vector<VerifyGuestByte> s_verifyGuestBytes;
void SaveVerifyGuestByte(uint32_t address) {
    // A texture can be loaded into several maps in one draw. Restore its input
    // once, and compare the final native byte with the translated result.
    for (const auto& byte : s_verifyGuestBytes)
        if (byte.address == address) return;
    s_verifyGuestBytes.push_back({address, Memory::Read8(address), 0});
}
static RecBytes s_rec;
static size_t s_recLastFifo = SIZE_MAX;  // offset of the open FIFO event's header, if the last event is FIFO bytes
// The offsets of the recording's events the game thread acts on once it defers the draw, in order, so it does not walk
// every event: ScanForCarry's (z-compare locations, TLUT loads, list calls: s_recMarks) and PlanLoads' (texture and
// TLUT loads: s_recLoadMarks). Cleared with s_rec at each recording (RunNative); only FIFO bytes are merged into an
// earlier event, and those are never marked, so a marked offset stays put.
// A list call the native draw recorded directly (RecListCall) keeps its range with its mark, so ScanForCarry need not
// decode it from the payload (size 0: decode).
struct RecMark {
    uint32_t at;
    uint32_t listAddr;
    uint32_t listSize;
};
static std::vector<RecMark> s_recMarks;
static std::vector<uint32_t> s_recLoadMarks;
constexpr uint32_t kRecCarryTags = 1u << kRecSetZCompLoc | 1u << kRecLoadTlut | 1u << kRecShapeDL | 1u << kRecRegDL;
constexpr uint32_t kRecLoadTags = 1u << kRecLoadTlut | 1u << kRecLoadTexObj;
static_assert(kRecRegDL < 32, "RecTag fits the mark masks");

namespace {
// A recorded event's payload is a few words of HLE arguments (~16 bytes on average): copied inline, with
// two overlapping moves, instead of a call into the C library's memcpy for each of ~1,700 events a frame.
inline void CopyRecPayload(uint8_t* dst, const void* payload, uint32_t bytes) {
    const auto* src = static_cast<const uint8_t*>(payload);
    if (bytes > 32) {
        std::memcpy(dst, src, bytes);
    } else if (bytes >= 16) {
        uint8_t head[16], tail[16];
        std::memcpy(head, src, 16);
        std::memcpy(tail, src + bytes - 16, 16);
        std::memcpy(dst, head, 16);
        std::memcpy(dst + bytes - 16, tail, 16);
    } else if (bytes >= 8) {
        uint64_t head, tail;
        std::memcpy(&head, src, 8);
        std::memcpy(&tail, src + bytes - 8, 8);
        std::memcpy(dst, &head, 8);
        std::memcpy(dst + bytes - 8, &tail, 8);
    } else if (bytes >= 4) {
        uint32_t head, tail;
        std::memcpy(&head, src, 4);
        std::memcpy(&tail, src + bytes - 4, 4);
        std::memcpy(dst, &head, 4);
        std::memcpy(dst + bytes - 4, &tail, 4);
    } else {
        for (uint32_t i = 0; i < bytes; ++i) dst[i] = src[i];
    }
}
// One event appended: RecEvent, and the native draw's own GX calls (RecWords, RecListCall, Fifo::Flush: in line, with
// the tag and size constants there).
[[gnu::always_inline]] inline void RecEventInline(uint8_t tag, const void* payload, uint32_t bytes) {
    const bool fifo = tag == kRecFifo || tag == kRecFifoG3D;
    // Consecutive FIFO bytes are one event, however written (bytes of the two kinds only with their own kind).
    // Grow leaves the new bytes uninitialized: every one of them is written right after.
    if (fifo && s_recLastFifo != SIZE_MAX && s_rec[s_recLastFifo] == tag) {
        uint32_t len;
        std::memcpy(&len, &s_rec[s_recLastFifo + 1], 4);
        len += bytes;
        std::memcpy(&s_rec[s_recLastFifo + 1], &len, 4);
        CopyRecPayload(s_rec.Grow(bytes), payload, bytes);
        return;
    }
    const size_t at = s_rec.size();
    s_recLastFifo = fifo ? at : SIZE_MAX;
    if ((kRecCarryTags >> (tag & 31)) & 1u) s_recMarks.push_back(RecMark{static_cast<uint32_t>(at), 0, 0});
    if ((kRecLoadTags >> (tag & 31)) & 1u) s_recLoadMarks.push_back(static_cast<uint32_t>(at));
    uint8_t* p = s_rec.Grow(5 + bytes);
    p[0] = tag;
    std::memcpy(p + 1, &bytes, 4);
    CopyRecPayload(p + 5, payload, bytes);
}
}  // namespace

void RecEvent(uint8_t tag, const void* payload, uint32_t bytes) { RecEventInline(tag, payload, bytes); }

void RecGuestWords(uint32_t* out, uint32_t addr, uint32_t words) {
    for (uint32_t i = 0; i < words; ++i) out[i] = Memory::Read32(addr + 4 * i);
}

namespace {

bool s_fast = true;      // G3D_NATIVE_FAST (default on): the HLE shortcuts above
bool s_envelope = true;  // G3D_NATIVE_ENVELOPE (default on): skinned shapes too (0: declined, as in M1)
bool s_shape = true;     // G3D_NATIVE_SHAPE (default on, with G3D_NATIVE_FAST): shape lists from per-list records
uint64_t s_drawRecords = 0;  // DRAW records (Draw1Mat1ShpDirectly calls) drawn natively

// While a job records (RunNative, G3D_NATIVE_ASYNC; game thread): the native draw's own GX calls are recorded here, in
// line, instead of through their HLE entries. An entry reached with the recorder on, suppressing, at depth 0 (as every
// call from the native draw's own code is: the depth is only raised inside an entry) records its event and returns at
// once (g3d_native.h HleRec); with g_recJob its event has the job tags (kRecFifoG3D, kRecShapeDL, kRecRegDL). The
// same event, written here, without the call, the payload built for the entry and its recorder checks. Its argument
// checks before the recorder (a list's zero address or size, empty FIFO bytes) are repeated here.
bool s_recDirect = false;
inline bool RecDirect() { return s_recDirect && g_recDepth == 0; }
// The event of a GX entry recording host-order words (G3D_REC_HLE); false: not recording a job, call the entry.
template <typename... W>
inline bool RecWords(uint8_t tag, W... words) {
    if (!RecDirect()) return false;
    const uint32_t payload[] = {static_cast<uint32_t>(words)...};
    RecEventInline(tag, payload, sizeof payload);
    return true;
}
// A display-list call's event: its FIFO form (0x40, big-endian address and size), as the list entries record it.
inline void RecListCall(uint8_t tag, uint32_t addr, uint32_t size) {
    uint8_t payload[9];
    payload[0] = 0x40;
    const uint32_t a = __builtin_bswap32(addr), s = __builtin_bswap32(size);
    std::memcpy(payload + 1, &a, 4);
    std::memcpy(payload + 5, &s, 4);
    RecEventInline(tag, payload, 9);
    // A shape or register list's event is always a new one, marked last (RecEventInline): it keeps the range.
    if (tag == kRecShapeDL || tag == kRecRegDL) {
        s_recMarks.back().listAddr = addr;
        s_recMarks.back().listSize = size;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Guest memory (big-endian; the same accessors translated code uses, so writes are dirty-tracked)
// ---------------------------------------------------------------------------------------------------------------
inline uint32_t R32(uint32_t a) { return MemoryInline::FlatRead32(a); }
inline int32_t RS32(uint32_t a) { return static_cast<int32_t>(MemoryInline::FlatRead32(a)); }
inline uint16_t R16(uint32_t a) { return MemoryInline::FlatRead16(a); }
inline int16_t RS16(uint32_t a) { return static_cast<int16_t>(MemoryInline::FlatRead16(a)); }
inline uint8_t R8(uint32_t a) { return MemoryInline::FlatRead8(a); }
inline int8_t RS8(uint32_t a) { return static_cast<int8_t>(MemoryInline::FlatRead8(a)); }
inline void W32(uint32_t a, uint32_t v) { MemoryInline::FlatWriteRam32(a, v); }
inline void W16(uint32_t a, uint16_t v) { MemoryInline::FlatWriteRam16(a, v); }
inline void W8(uint32_t a, uint8_t v) { MemoryInline::FlatWriteRam8(a, v); }
// A guest offset field (nw4r ofs_to_ptr): 0 means none.
inline uint32_t Ofs(uint32_t base, uint32_t field) {
    const int32_t o = RS32(base + field);
    return o != 0 ? base + static_cast<uint32_t>(o) : 0;
}

// ---------------------------------------------------------------------------------------------------------------
// nw4r::g3d::G3DState in Brawl's memory (g3d_state.cpp's statics; addresses from the translated loaders)
// ---------------------------------------------------------------------------------------------------------------
constexpr uint32_t kG3D = 0x804A44C0;
constexpr uint32_t kGenMode = kG3D + 0;        // GenMode2: u8 texGens, chans, tevs, inds; u32 cull @4; u32 flag @8
constexpr uint32_t kTcs = kG3D + 12;           // TexCoordScaleState: u32 upToDate; wh[8] {u16 w, h} @4;
                                               //   scale[8] {u16 w, h; u32 opt} @0x24; u8 table[8] @0x64
constexpr uint32_t kTex = kG3D + 128;          // TexState: GXTexObj[8]; u8 validFlag @0x100
constexpr uint32_t kTlut = kG3D + 416;         // TlutState: GXTlutObj[8] (12 bytes); u16 flag @0x60
constexpr uint32_t kShp = kG3D + 516;          // ShpState: ResCacheVtxDescv (12 bytes)
constexpr uint32_t kPreTexMtx = kG3D + 528;    // PreTexMtxState: ScnDependentTexMtxFuncType[8] (u32)
constexpr uint32_t kPosNrm = kG3D + 560;       // PosNrmMtxArrayState: viewPos, viewNrm, viewEnvTex
constexpr uint32_t kFog = kG3D + 2624;         // FogState: u32 flag; s32 currentFogIdx; FogData[]
constexpr uint32_t kLight = kG3D + 4180;       // LightState (offsets in LoadLightSet)
constexpr uint32_t kSda = 0x805A4420;          // r13
constexpr uint32_t kSyncGX = kSda - 16104;     // SyncGX::bNeedSync (u8)
constexpr uint32_t kCurrMtx = kSda - 16100;    // CurrMtxState::mIsUpToDate (u8)
constexpr uint32_t kPostTexMtx = kSda - 16096; // PostTexMtxState::mFlagIdentity (u32)
constexpr uint32_t kTevState = kSda - 16092;   // TevState::mpTevData
constexpr uint32_t kMiscFlag = kSda - 16080;   // Misc::mValidFlag (u32)
constexpr uint32_t kMiscZComp = kSda - 16076;  // Misc::mbZCompLoc (GXBool)
// Verification compares these (G3D_NATIVE=2): g3d_state.cpp's .bss and .sbss.
constexpr uint32_t kSnapBss = kG3D, kSnapBssSize = 0x4000;
constexpr uint32_t kSnapSbss = kSda - 16112, kSnapSbssSize = 48;
// MSL std::sort's pivot generator state (the Z sort advances it): r13 - 23328, r13 - 23324.
constexpr uint32_t kSnapSortSeed = kSda - 23328, kSnapSortSeedSize = 8;

enum : uint32_t { kGenBP = 1, kGenXF = 2, kGenFirst = 4, kGenAll = 3 };
enum : uint32_t { kTcsTexCoord = 1, kTcsTexMap = 2 };

// The FIFO bytes nw4r's GD helpers write, collected and handed to the HLE in one burst before the next other GX
// call (the HLE parses a burst exactly as the same bytes written one store at a time).
// The same bytes as before, in the same order; each call makes room for its whole piece once (a packet is then never
// split across two flushes, which only spared the packet writer its split-packet path) and stores big-endian words
// directly instead of one bounds-checked byte at a time.
class Fifo {
public:
    void U8(uint8_t v) {
        Reserve(1);
        buf_[n_++] = v;
    }
    void U16(uint16_t v) {
        Reserve(2);
        Put16(v);
    }
    void U32(uint32_t v) {
        Reserve(4);
        Put32(v);
    }
    void BP(uint32_t v) {
        Reserve(5);
        buf_[n_++] = 0x61;
        Put32(v);
    }
    void CP(uint8_t reg, uint32_t v) {
        Reserve(6);
        buf_[n_++] = 0x08;
        buf_[n_++] = reg;
        Put32(v);
    }
    void XF(uint16_t addr, uint32_t v) {
        Reserve(9);
        buf_[n_++] = 0x10;
        Put16(0);
        Put16(addr);
        Put32(v);
    }
    void XFHdr(uint16_t addr, uint8_t len) {
        Reserve(5);
        buf_[n_++] = 0x10;
        Put16(uint16_t(len - 1));
        Put16(addr);
    }
    void Flush() {
        if (n_ != 0) {
            if (RecDirect())  // the event GX_HLE_FIFO_WritePackets_G3D (job tag) or GX_HLE_FIFO_WriteBurst records
                RecEventInline(s_fast ? kRecFifoG3D : kRecFifo, buf_, n_);
            else if (s_fast)
                GX_HLE_FIFO_WritePackets_G3D(buf_, n_);
            else
                GX_HLE_FIFO_WriteBurst(buf_, n_);
            n_ = 0;
        }
    }

private:
    void Reserve(uint32_t bytes) {
        if (n_ + bytes > sizeof(buf_)) [[unlikely]] Flush();
    }
    void Put16(uint16_t v) {
        const uint16_t be = __builtin_bswap16(v);
        std::memcpy(buf_ + n_, &be, 2);
        n_ += 2;
    }
    void Put32(uint32_t v) {
        const uint32_t be = __builtin_bswap32(v);
        std::memcpy(buf_ + n_, &be, 4);
        n_ += 4;
    }
    uint8_t buf_[1024];
    uint32_t n_ = 0;
};

// A frame on the guest stack, as a guest caller makes one: scratch memory for arguments passed by reference, and
// the stack the translated functions called from here build their frames under. Restores r1 and LR.
class GuestFrame {
public:
    static constexpr uint32_t kSize = 256;
    explicit GuestFrame(CpuContext* ctx) : ctx_(ctx), savedSp_(ctx->gpr[1]), savedLr_(ctx->lr) {
        sp_ = (savedSp_ - kSize) & ~15u;
        W32(sp_, savedSp_);  // back chain
        ctx->gpr[1] = sp_;
    }
    ~GuestFrame() {
        ctx_->gpr[1] = savedSp_;
        ctx_->lr = savedLr_;
    }
    uint32_t Scratch(uint32_t offset) const { return sp_ + 8 + offset; }
    GuestFrame(const GuestFrame&) = delete;
    GuestFrame& operator=(const GuestFrame&) = delete;

private:
    CpuContext* ctx_;
    uint32_t savedSp_, savedLr_, sp_;
};

// ---------------------------------------------------------------------------------------------------------------
// The draw
// ---------------------------------------------------------------------------------------------------------------
struct Swap {  // Draw1Mat1ShpSwap: replacement data for one material (0: none)
    uint32_t texObj = 0, tlutObj = 0, genMode = 0, tev = 0, pix = 0, tevColor = 0, indMtx = 0, chan = 0,
             texCoordGen = 0, misc = 0, texSrt = 0, vtxPosTable = 0, vtxNrmTable = 0, vtxClrTable = 0;
};

class Drawer {
public:
    Drawer(CpuContext* ctx) : ctx_(ctx), frame_(ctx) {}

    void DrawResMdlDirectly(uint32_t mdl, uint32_t viewPos, uint32_t viewNrm, uint32_t viewEnv, uint32_t opa,
                            uint32_t xlu, uint32_t repl, uint32_t drawMode);
    // DrawResMdlDirectly's guest-visible work without the drawing (g3dn::SetDrawSuppressed).
    void SortOnly(uint32_t mdl, uint32_t viewPos, uint32_t viewNrm, uint32_t viewEnv, uint32_t opa, uint32_t xlu,
                  uint32_t repl, uint32_t drawMode);

private:
    void SortMdlZ(uint32_t mdl, uint32_t bc, uint32_t repl, uint32_t viewPos, uint32_t comparator, uint32_t& mdlZ,
                  uint32_t& numMdlZ);
    // G3DState
    bool SyncGet() {
        const bool need = R8(kSyncGX) != 0;
        W8(kSyncGX, 0);
        return need;
    }
    void SyncSet() { W8(kSyncGX, 1); }
    void GenModeLoadBP();
    void GenModeLoadXF();
    void GDSetGenMode2Ex_BP(uint32_t texGens, uint32_t chans, uint32_t tevs, uint32_t inds, uint32_t cull);
    void TcsLoad(uint32_t texGens);
    // A material's register list (TEV, pixel, TEV colour, indirect matrix, texgen), or a shape's pre-primitive or
    // primitive list.
    enum class List { kMaterial, kShape };
    void CallDL(uint32_t addr, uint32_t size, List kind) {
        fifo_.Flush();
        const bool reg = s_fast && kind == List::kMaterial;
        const bool shape = !reg && s_fast && s_shape;
        if (RecDirect()) {
            // The entry's event (GX__CallDisplayList_HLE records the FIFO tag); like the entries, nothing for a list
            // with no address or size.
            if (addr != 0 && size != 0) RecListCall(reg ? kRecRegDL : shape ? kRecShapeDL : kRecFifo, addr, size);
        } else if (reg) {
            GX__CallRegisterDisplayList_G3D(addr, size);
        } else if (shape) {
            GX__CallShapeDisplayList_G3D(addr, size);
        } else {
            GX__CallDisplayList_HLE(addr, size);
        }
    }

    void SetArray(uint32_t attr, uint32_t base, uint32_t stride) {  // GXSetArray
        if (!RecWords(kRecSetArray, attr, base, stride)) GX__SetArray_HLE(attr, base, stride);
    }
    void LoadFog(int id);
    void LoadResMatMisc(uint32_t misc);
    void LoadResTlutObj(uint32_t tlut);
    void LoadResTexObj(uint32_t tex);
    void LoadResGenMode(uint32_t gm);
    void LoadResTev(uint32_t tev);
    void LoadResMatPix(uint32_t pix);
    void LoadResMatTevColor(uint32_t col);
    void LoadResMatIndMtxAndScale(uint32_t ind);
    void LoadLightSet(int id, uint32_t masks[4], uint32_t& amb);
    void LoadResMatChan(uint32_t chan, const uint32_t masks[4], uint32_t amb, bool lightOff);
    void LoadResMatTexCoordGen(uint32_t gen);
    void LoadResTexSrt(uint32_t srt);
    void LoadResShpPrePrimitive(uint32_t shp);
    void LoadResShpPrimitive(uint32_t shp);
    void SetCurrentMtx(int id);
    void SetCurrMtx();
    void Draw1Mat1Shp(uint32_t mat, uint32_t shp, uint32_t ctrl, const Swap* swap);
    void DrawLoop(uint32_t mdl, uint32_t byteCode, uint32_t repl, uint32_t drawMode, bool zSort, uint32_t viewPos,
                  uint32_t comparator);

    CpuContext* ctx_;
    GuestFrame frame_;
    Fifo fifo_;
};

constexpr uint8_t kCm2Hw[4] = {0, 2, 1, 3};

void Drawer::GDSetGenMode2Ex_BP(uint32_t texGens, uint32_t chans, uint32_t tevs, uint32_t inds, uint32_t cull) {
    // GD SSMASK: numTex, numColors (2 bits), numTevStages, cullMode, numIndStages.
    fifo_.BP(0xFE000000u | 0xFu | (3u << 4) | (0xFu << 10) | (3u << 14) | (7u << 16));
    fifo_.BP(texGens | chans << 4 | (tevs - 1) << 10 | uint32_t(kCm2Hw[cull & 3]) << 14 | inds << 16 | 0x00u << 24);
}

void Drawer::GenModeLoadBP() {
    const uint32_t flag = R32(kGenMode + 8);
    if ((flag & kGenFirst) && (flag & kGenAll) == kGenXF) {
        GDSetGenMode2Ex_BP(R8(kGenMode), R8(kGenMode + 1), R8(kGenMode + 2), R8(kGenMode + 3), R32(kGenMode + 4));
        W32(kGenMode + 8, flag | kGenBP);
    }
}

void Drawer::GenModeLoadXF() {
    const uint32_t flag = R32(kGenMode + 8);
    if ((flag & kGenFirst) && (flag & kGenAll) != kGenAll) {
        const uint32_t texGens = R8(kGenMode), chans = R8(kGenMode + 1);
        GDSetGenMode2Ex_BP(texGens, chans, R8(kGenMode + 2), R8(kGenMode + 3), R32(kGenMode + 4));
        fifo_.XF(0x1009, chans);    // XF NUMCOLORS
        fifo_.XF(0x103F, texGens);  // XF NUMTEX
        W32(kGenMode + 8, flag | kGenAll);
    }
}

void Drawer::TcsLoad(uint32_t texGens) {  // TexCoordScaleState::LoadTexCoordScale
    const uint32_t upToDate = R32(kTcs);
    if (!(upToDate & kTcsTexMap) || (upToDate & kTcsTexCoord) || texGens == 0) return;
    for (uint32_t i = 0; i < texGens; ++i) {
        const uint32_t map = R8(kTcs + 0x64 + i);
        if (map == 0xFF) continue;
        const uint16_t w = R16(kTcs + 4 + map * 4), h = R16(kTcs + 6 + map * 4);
        W16(kTcs + 0x24 + i * 8, w);
        W16(kTcs + 0x26 + i * 8, h);
        // GDSetTexCoordScale2(i, w, 0, 0, h, 0, 0): SSMASK scale, range bias, cylindric wrap; SU_SSIZEi, SU_TSIZEi.
        fifo_.BP(0xFE000000u | 0xFFFFu | 1u << 16 | 1u << 17);
        fifo_.BP(uint32_t(w - 1) | (0x30u + i * 2) << 24);
        fifo_.BP(uint32_t(h - 1) | (0x31u + i * 2) << 24);
    }
    W32(kTcs, R32(kTcs) | kTcsTexCoord);
}

void Drawer::LoadFog(int id) {  // FogState::LoadFog; a load (rare: once per layer) runs the translated one
    const uint32_t flag = R32(kFog);
    if ((flag & 1) && (flag & 2) && id == RS32(kFog + 4)) return;
    fifo_.Flush();
    ctx_->gpr[3] = static_cast<uint32_t>(id);
    ctx_->lr = 0x801A1AD8u;
    func_801A0E7C(ctx_);
}

void Drawer::LoadResMatMisc(uint32_t misc) {  // Misc::GXSetZCompLoc
    if (misc == 0) return;
    const uint8_t beforeTex = R8(misc);
    const uint32_t flag = R32(kMiscFlag);
    if (!(flag & 1) || R8(kMiscZComp) != beforeTex) {
        W32(kMiscFlag, flag | 1);
        W8(kMiscZComp, beforeTex);
        fifo_.Flush();
        if (!RecWords(kRecSetZCompLoc, beforeTex)) GX__SetZCompLoc_HLE(beforeTex);
        SyncSet();
    }
}

void Drawer::LoadResTlutObj(uint32_t tlut) {  // TlutState::LoadResTlutObj
    if (tlut == 0) return;
    const uint32_t used = R32(tlut);
    for (uint32_t i = 0; i < 8; ++i) {
        if (!(used & (1u << i))) continue;
        const uint32_t obj = tlut + 4 + 12 * i, cached = kTlut + 12 * i;
        const uint16_t flag = R16(kTlut + 0x60), mask = uint16_t(1u << i);
        const uint32_t a0 = R32(obj), a1 = R32(obj + 4), a2 = R32(obj + 8);
        if (!(flag & mask) || a0 != R32(cached) || a1 != R32(cached + 4) || a2 != R32(cached + 8)) {
            W16(kTlut + 0x60, flag | mask);
            W32(cached, a0);
            W32(cached + 4, a1);
            W32(cached + 8, a2);
            fifo_.Flush();
            if (!RecWords(kRecLoadTlut, obj, i)) GX__LoadTlut_HLE(obj, i);
            W8(kTex + 0x100, R8(kTex + 0x100) & uint8_t(~(1u << i)));  // TexState::Invalidate(map i)
        }
    }
    SyncSet();
}

void Drawer::LoadResTexObj(uint32_t tex) {  // TexState::LoadResTexObj
    if (tex == 0) return;
    const uint32_t used = R32(tex);
    for (uint32_t i = 0; i < 8; ++i) {
        if (!(used & (1u << i))) continue;
        const uint32_t obj = tex + 4 + 32 * i, cached = kTex + 32 * i;
        const uint8_t valid = R8(kTex + 0x100), mask = uint8_t(1u << i);
        bool equal = (valid & mask) != 0;
        for (uint32_t w = 0; equal && w < 8; ++w) equal = R32(obj + 4 * w) == R32(cached + 4 * w);
        if (!equal) {
            W8(kTex + 0x100, valid | mask);
            for (uint32_t w = 0; w < 8; ++w) W32(cached + 4 * w, R32(obj + 4 * w));
            fifo_.Flush();
            if (!RecWords(kRecLoadTexObj, obj, i)) GX__LoadTexObj_HLE(obj, i);
            // TexCoordScaleState::UpdateTexMapSize: GXGetTexObjWidth/Height from image0.
            const uint32_t image0 = R32(obj + 8);
            W16(kTcs + 4 + i * 4, uint16_t((image0 & 0x3FF) + 1));
            W16(kTcs + 6 + i * 4, uint16_t((image0 >> 10 & 0x3FF) + 1));
            W32(kTcs, (R32(kTcs) | kTcsTexMap) & ~kTcsTexCoord);
        }
    }
    SyncSet();
}

void Drawer::LoadResGenMode(uint32_t gm) {  // GenMode2::SetResGenMode
    if (gm == 0) return;
    uint32_t flag = R32(kGenMode + 8);
    const uint8_t texGens = R8(gm), chans = R8(gm + 1), tevs = R8(gm + 2), inds = R8(gm + 3);
    const uint32_t cull = R32(gm + 4);
    if (R8(kGenMode) != texGens) { W8(kGenMode, texGens); flag &= ~kGenAll; }
    if (R8(kGenMode + 1) != chans) { W8(kGenMode + 1, chans); flag &= ~kGenAll; }
    if (R8(kGenMode + 2) != tevs) { W8(kGenMode + 2, tevs); flag &= ~kGenBP; }
    if (R8(kGenMode + 3) != inds) { W8(kGenMode + 3, inds); flag &= ~kGenBP; }
    if (R32(kGenMode + 4) != cull) { W32(kGenMode + 4, cull); flag &= ~kGenBP; }
    if (!(flag & kGenFirst)) { flag &= ~kGenAll; flag |= kGenFirst; }
    W32(kGenMode + 8, flag);
}

void Drawer::LoadResTev(uint32_t tev) {
    if (tev == 0) return;
    if (R32(kTevState) == tev) return;
    W32(kTevState, tev);
    GenModeLoadBP();
    // ResTev::CallDisplayList: the common part (0x60) and one 0x30 block per two stages, rounded up to 32 bytes.
    const uint32_t stages = R8(tev + 0xC);
    const uint32_t size = (0x60 + (stages + 1) / 2 * 0x30 + 31) & ~31u;
    SyncGet();
    CallDL(tev + 0x20, size, List::kMaterial);
    // TexCoordScaleState::UpdateTable, then LoadTexCoordScale.
    const uint32_t t0 = R32(tev + 0x10), t1 = R32(tev + 0x14);
    uint32_t upToDate = R32(kTcs);
    if (!(upToDate & kTcsTexMap) || t0 != R32(kTcs + 0x64) || t1 != R32(kTcs + 0x68)) {
        W32(kTcs + 0x64, t0);
        W32(kTcs + 0x68, t1);
        W32(kTcs, (upToDate | kTcsTexMap) & ~kTcsTexCoord);
    }
    TcsLoad(R8(kGenMode));
}

void Drawer::LoadResMatPix(uint32_t pix) {
    if (pix == 0) return;
    GenModeLoadBP();
    SyncGet();
    CallDL(pix, 32, List::kMaterial);
}

void Drawer::LoadResMatTevColor(uint32_t col) {
    if (col == 0) return;
    GenModeLoadBP();
    SyncGet();
    CallDL(col, 128, List::kMaterial);
}

void Drawer::LoadResMatIndMtxAndScale(uint32_t ind) {
    if (ind == 0) return;
    GenModeLoadBP();
    const uint32_t inds = R8(kGenMode + 3);
    SyncGet();
    if (inds != 0) CallDL(ind, 64, List::kMaterial);
}

void Drawer::LoadLightSet(int id, uint32_t masks[4], uint32_t& amb) {  // LightState::LoadLightSet
    constexpr uint32_t kNumLight = 128, kNumLightSet = 128;
    constexpr uint32_t kCurSet = kLight + 16, kMasks = kLight + 20, kLoaded = kLight + 36, kObj = kLight + 44,
                       kAmb = kLight + 8748, kSetData = kLight + 9260;
    if (id < 0 || id >= int(kNumLightSet)) {
        masks[0] = masks[1] = masks[2] = masks[3] = 0;
        amb = 0xFFFFFFFFu;
        return;
    }
    const uint32_t set = kSetData + 12 * uint32_t(id);
    const int8_t ambId = RS8(set + 8);
    amb = ambId < 0 ? 0xFFFFFFFFu : R32(kAmb + 4 * uint32_t(ambId));
    if (RS32(kCurSet) == id) {
        for (int i = 0; i < 4; ++i) masks[i] = R32(kMasks + 4 * i);
        return;
    }
    uint32_t diffC = 0, diffA = 0, specC = 0, specA = 0;
    W32(kCurSet, uint32_t(id));
    for (uint32_t i = 0; i < 8; ++i) {
        const int8_t lightId = RS8(set + i);
        if (lightId < 0) continue;
        const uint32_t obj = kObj + 68 * uint32_t(lightId);
        const uint32_t lflag = R32(obj);
        if (!(lflag & 4)) continue;  // FLAG_ENABLE_LIGHT
        int slot = -1;
        for (int j = 0; j < 8; ++j)
            if (RS8(kLoaded + j) == lightId) { slot = j; break; }
        if (slot < 0) {
            for (int j = 0; j < 8; ++j)
                if (RS8(kLoaded + j) == -1) { slot = j; break; }
            if (slot < 0) {
                // Evict the first loaded light this set does not use.
                int j = 0;
                for (; j < 8; ++j) {
                    const int8_t loaded = RS8(kLoaded + j);
                    bool inSet = false;
                    for (uint32_t k = 0; k < 8; ++k) inSet = inSet || RS8(set + k) == loaded;
                    if (!inSet) break;
                }
                slot = j;
            }
            fifo_.Flush();
            if (!RecWords(kRecLoadLightObj, obj + 4, 1u << slot)) GX__LoadLightObjImm_HLE(obj + 4, 1u << slot);
            W8(kLoaded + uint32_t(slot), uint8_t(lightId));
        }
        const uint32_t idFlag = 1u << slot;
        if (lflag & 2) {  // FLAG_SPECULAR
            if (!(lflag & 0x10)) specC |= idFlag;
            if (!(lflag & 0x20)) specA |= idFlag;
        } else {
            if (!(lflag & 0x10)) diffC |= idFlag;
            if (!(lflag & 0x20)) diffA |= idFlag;
        }
    }
    (void)kNumLight;
    W32(kMasks, diffC);
    W32(kMasks + 4, diffA);
    W32(kMasks + 8, specC);
    W32(kMasks + 12, specA);
    masks[0] = diffC;
    masks[1] = diffA;
    masks[2] = specC;
    masks[3] = specA;
}

// One component of LoadResMatChan's ambient colour: u8(0.5f + (amb * mat) * (1.0f / 255.0f)), with the guest's
// rounding (fmuls, then fadds; the constants read from its .sdata2).
inline uint32_t AmbComponentCompute(uint32_t product) {
    static const double kScale = MemoryInline::FlatReadFloat32(0x805A9320u - 24560);  // r2 - 24560: 1/255
    static const double kHalf = MemoryInline::FlatReadFloat32(0x805A9320u - 24564);   // r2 - 24564: 0.5
    const double x = static_cast<double>(PpcForceSingleValueInline(static_cast<double>(product)));
    const double t = static_cast<double>(PpcForceSingleValueInline(PpcFaddInline(kHalf, PpcFmulsInline(kScale, x))));
    uint64_t bits;
    const double c = PPC_Fctiwz(t);
    std::memcpy(&bits, &c, 8);
    return static_cast<uint32_t>(bits) & 0xFF;
}
// The result depends only on amb * mat (0..65025), and the two constants are read once above: each product's value is
// computed the first time it occurs and kept (-1: not yet). Four per material channel load, every draw.
inline uint32_t AmbComponent(uint32_t amb, uint32_t mat) {
    static std::array<int16_t, 255 * 255 + 1> s_byProduct = [] {
        std::array<int16_t, 255 * 255 + 1> table;
        table.fill(-1);
        return table;
    }();
    const uint32_t product = amb * mat;
    int16_t value = s_byProduct[product];
    if (value < 0) [[unlikely]] {
        value = static_cast<int16_t>(AmbComponentCompute(product));
        s_byProduct[product] = value;  // (racing writers store the same value)
    }
    return static_cast<uint32_t>(value);
}

void Drawer::LoadResMatChan(uint32_t chan, const uint32_t masks[4], uint32_t amb, bool lightOff) {
    if (chan == 0) return;
    auto chanCtrl = [&](uint32_t id, uint32_t param, uint32_t lightMask, bool off) {
        param &= ~(0xFu << 2 | 0xFu << 11 | (off ? 2u : 0u));
        param |= (lightMask & 0xF) << 2 | (lightMask >> 4 & 0xF) << 11;
        fifo_.XF(uint16_t(0x100E + (id & 3)), param);
    };
    {
        const uint32_t r = chan;
        const uint32_t flag = R32(r);
        if ((flag & 1) || (flag & 2)) fifo_.XF(0x100C, R32(r + 4));  // MATERIAL0 (COLOR0 for colour and alpha)
        const uint32_t matAmb = R32(r + 8);
        const uint32_t ambColor = AmbComponent(amb >> 24, matAmb >> 24) << 24 |
                                  AmbComponent(amb >> 16 & 0xFF, matAmb >> 16 & 0xFF) << 16 |
                                  AmbComponent(amb >> 8 & 0xFF, matAmb >> 8 & 0xFF) << 8 |
                                  AmbComponent(amb & 0xFF, matAmb & 0xFF);
        if ((flag & 4) || (flag & 8)) fifo_.XF(0x100A, ambColor);  // AMBIENT0
        if (flag & 0x10) chanCtrl(0, R32(r + 0xC), masks[0], lightOff);  // GX_COLOR0
        if (flag & 0x20) chanCtrl(2, R32(r + 0x10), masks[1], lightOff);  // GX_ALPHA0
    }
    if (R8(kGenMode + 1) == 2) {
        const uint32_t r = chan + 20;
        const uint32_t flag = R32(r);
        if ((flag & 1) || (flag & 2)) fifo_.XF(0x100D, R32(r + 4));
        if ((flag & 4) || (flag & 8)) fifo_.XF(0x100B, R32(r + 8));
        if (flag & 0x10) chanCtrl(1, R32(r + 0xC), masks[2], lightOff);  // GX_COLOR1
        if (flag & 0x20) chanCtrl(3, R32(r + 0x10), masks[3], lightOff);  // GX_ALPHA1
    } else {
        chanCtrl(1, 0, 0, false);
        chanCtrl(3, 0, 0, false);
    }
    GenModeLoadXF();
}

void Drawer::LoadResMatTexCoordGen(uint32_t gen) {
    if (gen == 0) return;
    const uint32_t n = R8(kGenMode);
    SyncGet();
    if (n != 0) CallDL(gen, n < 2 ? 32 : n < 4 ? 64 : n < 8 ? 128 : 0xA0, List::kMaterial);
    GenModeLoadXF();
}

void Drawer::LoadResTexSrt(uint32_t srt) {
    // Identity texture matrices (the common case) here; anything computed (an SRT, a scene-dependent mapping)
    // through the translated function.
    const uint32_t flag = R32(srt);
    for (uint32_t i = 0; i < 8; ++i) {
        if (!(flag & (1u << (i * 4)))) continue;
        const uint32_t effect = srt + 0xA8 + 52 * i;
        const bool identity = (flag >> (i * 4) & 0xE) == 0xE && (R8(effect + 3) & 1);
        if (R8(effect + 2) != 0 || !identity) {
            fifo_.Flush();
            W32(frame_.Scratch(0), srt);  // ResTexSrt by value: a reference to a copy
            ctx_->gpr[3] = frame_.Scratch(0);
            ctx_->lr = 0x801A1D88u;
            func_801A04EC(ctx_);
            return;
        }
    }
    uint32_t post = R32(kPostTexMtx);
    for (uint32_t i = 0; i < 8; ++i) {
        W32(kPreTexMtx + 4 * i, 0);  // SCNDEPENDENT_TEXMTX_FUNCTYPE_TEXMTX_NOT_EXIST
        if (!(flag & (1u << (i * 4)))) continue;
        if (!(post & (1u << i))) {
            post |= 1u << i;
            W32(kPostTexMtx, post);
            // GXLoadTexMtxImm(identity, GX_PTTEXMTX0 + 3i, GX_MTX3x4): the matrix in guest memory, as the guest's.
            const uint32_t m = frame_.Scratch(16);
            for (uint32_t w = 0; w < 12; ++w) W32(m + 4 * w, (w == 0 || w == 5 || w == 10) ? 0x3F800000u : 0u);
            fifo_.Flush();
            GX__LoadTexMtxImm_HLE(m, 64 + 3 * i, 0);
        }
    }
}

void Drawer::LoadResShpPrePrimitive(uint32_t shp) {
    if (shp == 0) return;
    TcsLoad(R8(kGenMode));
    GenModeLoadXF();
    const bool sync = SyncGet();
    (void)sync;
    // ShpState::IsCacheEqual
    bool same = true;
    for (uint32_t w = 0; w < 3; ++w) same = same && R32(shp + 0xC + 4 * w) == R32(kShp + 4 * w);
    if (!same)
        for (uint32_t w = 0; w < 3; ++w) W32(kShp + 4 * w, R32(shp + 0xC + 4 * w));
    const uint32_t tag = shp + 0x18;
    const uint32_t dl = tag + static_cast<uint32_t>(RS32(tag + 8)), size = R32(tag + 4);
    if (same)
        CallDL(dl + 32, size - 32, List::kShape);
    else
        CallDL(dl, size, List::kShape);
}

void Drawer::SetCurrentMtx(int id) {  // PreTexMtxState::SetCurrentMtx with every type TEXMTX_NOT_EXIST
    if (id < 0) return;
    SetCurrMtx();
}

void Drawer::SetCurrMtx() {  // CurrMtxState::SetCurrMtx: GDResetCurrentMtx unless up to date
    if (R8(kCurrMtx) != 0) return;
    constexpr uint32_t kIdent = 60;  // GX_IDENTITY
    const uint32_t regA = 0u | kIdent << 6 | kIdent << 12 | kIdent << 18 | kIdent << 24;
    const uint32_t regB = kIdent | kIdent << 6 | kIdent << 12 | kIdent << 18;
    fifo_.CP(0x30, regA);  // MATINDEX_A
    fifo_.CP(0x40, regB);  // MATINDEX_B
    fifo_.XFHdr(0x1018, 2);
    fifo_.U32(regA);
    fifo_.U32(regB);
    W8(kCurrMtx, 1);
}

void Drawer::LoadResShpPrimitive(uint32_t shp) {
    if (shp == 0) return;
    const int32_t id = RS32(shp + 8);
    // An envelope (skinned) shape's matrices are in its primitive list: indexed loads of its palette slots from the
    // view matrix arrays (SetViewPosNrmMtxArray), between its draws.
    const bool envelope = id < 0;  // ResShpData::ID_FLAG_ENVELOPE
    if (!envelope) {
        // A scene-dependent texture matrix (environment or projection mapping, set by a translated LoadResTexSrt):
        // the translated function, which loads it from the view matrices.
        for (uint32_t i = 0; i < 8; ++i) {
            if (R32(kPreTexMtx + 4 * i) != 0) {
                fifo_.Flush();
                W32(frame_.Scratch(0), shp);  // ResShp by value: a reference to a copy
                ctx_->gpr[3] = frame_.Scratch(0);
                ctx_->gpr[4] = 0;  // pViewPos, pViewNrm: none (the indexed matrices)
                ctx_->gpr[5] = 0;
                ctx_->lr = 0x801A1E78u;
                func_801A093C(ctx_);
                return;
            }
        }
    }
    TcsLoad(R8(kGenMode));
    if (!envelope) {
        // GXLoadPosMtxIndx(id, GX_PNMTX0); GXLoadNrmMtxIndx3x3(id, GX_PNMTX0) if the shape has normals.
        fifo_.U8(0x20);
        fifo_.U32(uint32_t(id & 0xFFFF) << 16 | 11u << 12 | 0u);
        if (R32(shp + 0x30) & (1u << 10)) {
            fifo_.U8(0x28);
            fifo_.U32(uint32_t(id & 0xFFFF) << 16 | 8u << 12 | 0x400u);
        }
        SetCurrentMtx(id);
    } else {
        SetCurrMtx();
    }
    GenModeLoadXF();
    SyncGet();
    const uint32_t tag = shp + 0x24;
    CallDL(tag + static_cast<uint32_t>(RS32(tag + 8)), R32(tag + 4), List::kShape);
    if (envelope) W8(kCurrMtx, 0);  // CurrMtxState::Invalidate: the list's loads changed the current matrices
}

void Drawer::Draw1Mat1Shp(uint32_t mat, uint32_t shp, uint32_t ctrl, const Swap* swap) {
    auto pick = [&](uint32_t swapped, uint32_t own) { return swap != nullptr && swapped != 0 ? swapped : own; };
    if (mat != 0) {
        const uint32_t dlData = Ofs(mat, 0x38);
        const uint32_t misc = pick(swap ? swap->misc : 0, mat + 0x1C);
        LoadFog(RS8(misc + 2));
        LoadResMatMisc(misc);
        LoadResTlutObj(pick(swap ? swap->tlutObj : 0, mat + 0x140));
        LoadResTexObj(pick(swap ? swap->texObj : 0, mat + 0x3C));
        LoadResGenMode(pick(swap ? swap->genMode : 0, mat + 0x14));
        LoadResTev(pick(swap ? swap->tev : 0, Ofs(mat, 0x28)));
        LoadResMatPix(pick(swap ? swap->pix : 0, dlData));
        LoadResMatTevColor(pick(swap ? swap->tevColor : 0, dlData ? dlData + 0x20 : 0));
        LoadResMatIndMtxAndScale(pick(swap ? swap->indMtx : 0, dlData ? dlData + 0xA0 : 0));
        uint32_t masks[4], amb;
        LoadLightSet(RS8(misc + 1), masks, amb);
        LoadResMatChan(pick(swap ? swap->chan : 0, mat + 0x3EC), masks, amb, (ctrl & 8) != 0);
        LoadResMatTexCoordGen(pick(swap ? swap->texCoordGen : 0, dlData ? dlData + 0xE0 : 0));
        LoadResTexSrt(pick(swap ? swap->texSrt : 0, mat + 0x1A4));
    }
    if (shp != 0 && !(R32(shp + 0x34) & 2)) {
        LoadResShpPrePrimitive(shp);
        if (swap != nullptr) {
            if (swap->vtxPosTable != 0) {  // ResVtxPos::SetArray
                const uint32_t v = R32(swap->vtxPosTable + 4 * uint32_t(RS16(shp + 0x48)));
                if (v != 0) {
                    fifo_.Flush();
                    SetArray(9, v + static_cast<uint32_t>(RS32(v + 8)), R8(v + 0x1D));
                }
            }
            if (swap->vtxNrmTable != 0) {
                const int nrm = RS16(shp + 0x4A);
                const uint32_t v = nrm >= 0 ? R32(swap->vtxNrmTable + 4 * uint32_t(nrm)) : 0;
                if (v != 0) {
                    fifo_.Flush();
                    SetArray(10, v + static_cast<uint32_t>(RS32(v + 8)), R8(v + 0x1D));
                }
            }
            if (swap->vtxClrTable != 0) {
                for (uint32_t c = 0; c < 2; ++c) {
                    const int clr = RS16(shp + 0x4C + 2 * c);
                    const uint32_t v = clr >= 0 ? R32(swap->vtxClrTable + 4 * uint32_t(clr)) : 0;
                    if (v != 0) {
                        fifo_.Flush();
                        SetArray(11 + c, v + static_cast<uint32_t>(RS32(v + 8)), R8(v + 0x1C));
                    }
                }
            }
        }
        LoadResShpPrimitive(shp);
    }
}

// The dictionary entry idx of a ResMdl resource dictionary (nw4r ResDic::operator[](int)).
inline uint32_t DicEntry(uint32_t mdl, uint32_t dicField, uint32_t idx) {
    const uint32_t dic = Ofs(mdl, dicField);
    if (dic == 0) return 0;
    const int32_t o = RS32(dic + 8 + 16 * (idx + 1) + 12);
    return o != 0 ? dic + static_cast<uint32_t>(o) : 0;
}
constexpr uint32_t kDicNode = 0x14, kDicMat = 0x28, kDicShp = 0x30;

// A Z-sorted draw loop's MdlZ records: the translated SetUpMdlZ and std::sort leave them in nw4r's work memory (their
// order is the guest sort's own, ties included). The sort also advances MSL std::sort's pivot generator, a global every
// later std::sort of the game uses.
void Drawer::SortMdlZ(uint32_t mdl, uint32_t bc, uint32_t repl, uint32_t viewPos, uint32_t comparator, uint32_t& mdlZ,
                      uint32_t& numMdlZ) {
    fifo_.Flush();
    const uint32_t num = frame_.Scratch(192), mdlCopy = frame_.Scratch(196);
    W32(mdlCopy, mdl);
    ctx_->gpr[3] = num;
    ctx_->gpr[4] = mdlCopy;
    ctx_->gpr[5] = viewPos;
    ctx_->gpr[6] = bc;
    ctx_->gpr[7] = repl;
    ctx_->lr = 0x801A6608u;
    func_801A6378(ctx_);  // SetUpMdlZ
    mdlZ = ctx_->gpr[3];
    numMdlZ = R32(num);
    ctx_->gpr[3] = mdlZ;
    ctx_->gpr[4] = mdlZ + 12 * numMdlZ;
    ctx_->gpr[5] = comparator;
    ctx_->lr = 0x801A6624u;
    func_801A6778(ctx_);  // std::sort
}

// The draw loops (DrawResMdlLoop): the visible DRAW records of a byte code in order or, Z-sorted (SORT_OPA_Z /
// SORT_XLU_Z), the MdlZ records SortMdlZ leaves.
void Drawer::DrawLoop(uint32_t mdl, uint32_t bc, uint32_t repl, uint32_t drawMode, bool zSort, uint32_t viewPos,
                      uint32_t comparator) {
    const bool ignoreMat = (drawMode & 4) != 0;
    const uint32_t ctrl = 1u | ((drawMode & 8) ? 8u : 0u);
    uint32_t prevMat = 0;
    Swap swap;
    uint32_t mdlZ = 0, numMdlZ = 0;
    if (zSort) SortMdlZ(mdl, bc, repl, viewPos, comparator, mdlZ, numMdlZ);
    const uint32_t visArray = repl ? R32(repl + 4) : 0;
    for (uint32_t i = 0;; ++i) {
        uint32_t node, mat, shp;
        if (zSort) {
            if (i == numMdlZ) break;
            const uint32_t z = mdlZ + 12 * i;  // MdlZ: f32 Z, u16 priority, nodeID, matID, shpID
            node = DicEntry(mdl, kDicNode, R16(z + 6));
            mat = DicEntry(mdl, kDicMat, R16(z + 8));
            shp = DicEntry(mdl, kDicShp, R16(z + 10));
        } else {
            if (R8(bc) == 1) break;
            const uint32_t rec = bc;
            bc += 8;
            node = DicEntry(mdl, kDicNode, uint32_t(R8(rec + 5)) << 8 | R8(rec + 6));
            const bool visible =
                visArray != 0 ? R8(visArray + R32(node + 0xC)) != 0 : (R32(node + 0x14) & 0x100) != 0;
            if (!visible) continue;
            mat = DicEntry(mdl, kDicMat, uint32_t(R8(rec + 1)) << 8 | R8(rec + 2));
            shp = DicEntry(mdl, kDicShp, uint32_t(R8(rec + 3)) << 8 | R8(rec + 4));
        }
        (void)node;
        if (repl != 0) {  // SetupDraw1Mat1ShpSwap (Brawl's DrawResMdlReplacement: visArray at +4)
            const uint32_t id = R32(mat + 0xC);
            auto entry = [&](uint32_t field, uint32_t stride) {
                const uint32_t a = R32(repl + field);
                return a != 0 ? a + id * stride : 0u;
            };
            swap.texObj = entry(8, 260);
            swap.tlutObj = entry(12, 100);
            swap.texSrt = entry(16, 584);
            swap.chan = entry(20, 40);
            swap.genMode = entry(24, 8);
            swap.misc = entry(28, 12);
            swap.pix = entry(32, 32);
            swap.tevColor = entry(36, 128);
            swap.indMtx = entry(40, 64);
            swap.texCoordGen = entry(44, 160);
            swap.tev = entry(48, 512);
            // The vertex tables only while bit 0 of the replacement's first word (Brawl's flags) is clear.
            const bool vtxTables = (R32(repl) & 1) == 0;
            swap.vtxPosTable = vtxTables ? R32(repl + 52) : 0;
            swap.vtxNrmTable = vtxTables ? R32(repl + 56) : 0;
            swap.vtxClrTable = vtxTables ? R32(repl + 60) : 0;
        }
        Draw1Mat1Shp(ignoreMat || mat == prevMat ? 0 : mat, shp, ctrl, repl ? &swap : nullptr);
        ++s_drawRecords;
        prevMat = mat;
    }
}

void Drawer::DrawResMdlDirectly(uint32_t mdl, uint32_t viewPos, uint32_t viewNrm, uint32_t viewEnv, uint32_t opa,
                                uint32_t xlu, uint32_t repl, uint32_t drawMode) {
    // G3DState::SetViewPosNrmMtxArray
    W32(kPosNrm, viewPos);
    W32(kPosNrm + 4, viewNrm);
    W32(kPosNrm + 8, viewEnv);
    if (viewPos) SetArray(21, viewPos, 48);
    if (viewNrm) SetArray(22, viewNrm, 36);
    if (viewEnv) SetArray(23, viewEnv, 48);
    if (!(drawMode & 16)) {  // LCQueueWait(0) (PPCSync is a no-op here)
        ctx_->gpr[3] = 0;
        ctx_->lr = 0x801A64FCu;
        func_801D7AE4(ctx_);
    }
    // std::sort's comparators: FrontToBack (opaque), BackToFront (translucent).
    if (opa) DrawLoop(mdl, opa, repl, drawMode, (drawMode & 1) != 0, viewPos, 0x801A66D8u);
    if (xlu) DrawLoop(mdl, xlu, repl, drawMode, (drawMode & 2) != 0, viewPos, 0x801A6714u);
    fifo_.Flush();
    W32(kPosNrm, 0);
    W32(kPosNrm + 4, 0);
    W32(kPosNrm + 8, 0);
}

// DrawResMdlDirectly as above without drawing: the guest memory it leaves is the same (the Z sorts' MdlZ records and
// std::sort's pivot generator, G3DState's PosNrm array), except the G3DState caches the shape loads would have set,
// which the next drawn layer's G3DState::Invalidate resets before any draw reads them. No GX state changes.
void Drawer::SortOnly(uint32_t mdl, uint32_t viewPos, uint32_t viewNrm, uint32_t viewEnv, uint32_t opa, uint32_t xlu,
                      uint32_t repl, uint32_t drawMode) {
    W32(kPosNrm, viewPos);
    W32(kPosNrm + 4, viewNrm);
    W32(kPosNrm + 8, viewEnv);
    if (!(drawMode & 16)) {  // LCQueueWait(0)
        ctx_->gpr[3] = 0;
        ctx_->lr = 0x801A64FCu;
        func_801D7AE4(ctx_);
    }
    uint32_t mdlZ = 0, numMdlZ = 0;
    if (opa && (drawMode & 1)) SortMdlZ(mdl, opa, repl, viewPos, 0x801A66D8u, mdlZ, numMdlZ);
    if (xlu && (drawMode & 2)) SortMdlZ(mdl, xlu, repl, viewPos, 0x801A6714u, mdlZ, numMdlZ);
    W32(kPosNrm, 0);
    W32(kPosNrm + 4, 0);
    W32(kPosNrm + 8, 0);
}

// ---------------------------------------------------------------------------------------------------------------
// Eligibility: everything a call will draw must be something the port above draws exactly.
// ---------------------------------------------------------------------------------------------------------------
enum Decline : uint8_t {
    kOk,
    kDrawMode,     // Z sort or an unknown draw mode bit
    kByteCode,     // a byte code record that is not DRAW
    kMissing,      // a record's node, material or shape is missing
    kEnvelope,     // a skinned shape (envelope matrices) with G3D_NATIVE_ENVELOPE=0
    kIndirect,     // a material with a normal-map indirect method (IndMtxOp)
    kModelFilter,  // not in G3D_NATIVE_MODELS
    kPatched,      // a mod patched a function of the subtree
    kNumDecline,
};
const char* const kDeclineNames[kNumDecline] = {"ok",       "draw mode", "byte code",    "missing resource",
                                                "envelope", "indirect",  "model filter", "mod patch"};

// The translated functions this replaces or calls (DrawResMdlDirectly's subtree: nw4r g3d's draw loops, Z sort,
// Draw1Mat1ShpDirectly, the G3DState loaders, the res accessors and GD helpers; MSL's sort; LCQueueWait). A mod's
// patch to any of them (which then runs in the interpreter) makes every call decline, so the patched code runs. A
// patched leaf inlined into one of them marks it patched too (guest_code_patch.cpp).
constexpr uint32_t kSubtree[] = {
    0x801A0C84, 0x801A6378, 0x801A6778, 0x801A6D34, 0x801A72F0, 0x801A66D8, 0x801A6714, 0x801A5DD4, 0x801A5F1C,
    0x801A60E8, 0x801A6208, 0x801A5C0C, 0x801A56F0, 0x801A1A48, 0x801A0E7C, 0x801B5F34, 0x801B5DC0, 0x8019F150,
    0x8019F428, 0x8019F1C0, 0x8019F574, 0x8019F740, 0x8019092C, 0x8019F950, 0x80190AA8, 0x8019F9FC, 0x80190AD8,
    0x8019FB60, 0x8019FAA8, 0x8019163C, 0x80191670, 0x801A0F7C, 0x8019ECA0, 0x8019FD30, 0x801A045C, 0x80191CF0,
    0x8019DA9C, 0x801A04EC, 0x801A4D08, 0x8019EB3C, 0x801A0798, 0x8018FF6C, 0x80192710, 0x801927E0, 0x801928B0,
    0x801A12AC, 0x801A093C, 0x80190044, 0x8019DF2C, 0x8019DEB0, 0x8019DE54, 0x8019DB4C, 0x8019DB1C, 0x8019DBBC,
    0x8018EDA4, 0x8018F2EC, 0x8018F3C8, 0x80191190, 0x801923B0, 0x80191170, 0x801D7AE4, 0x801D50A0, 0x801F45A8,
};
std::vector<uint32_t> s_subtreeGuards;  // their guard indices (g_mkwGuestCodePatched)

void FindSubtreeGuards() {
    const uint32_t* begin = g_mkwGuardedFunctionStarts;
    const uint32_t* end = begin + g_mkwGuardedFunctionCount;
    for (const uint32_t address : kSubtree) {
        const uint32_t* it = std::lower_bound(begin, end, address);
        if (it != end && *it == address) s_subtreeGuards.push_back(static_cast<uint32_t>(it - begin));
    }
}

Decline CheckByteCode(uint32_t mdl, uint32_t bc) {
    for (uint32_t n = 0; n < 4096; ++n, bc += 8) {
        const uint8_t op = R8(bc);
        if (op == 1) return kOk;
        if (op != 4) return kByteCode;
        const uint32_t node = DicEntry(mdl, kDicNode, uint32_t(R8(bc + 5)) << 8 | R8(bc + 6));
        const uint32_t mat = DicEntry(mdl, kDicMat, uint32_t(R8(bc + 1)) << 8 | R8(bc + 2));
        const uint32_t shp = DicEntry(mdl, kDicShp, uint32_t(R8(bc + 3)) << 8 | R8(bc + 4));
        if (node == 0 || mat == 0 || shp == 0) return kMissing;
        if ((R32(shp + 8) & 0x80000000u) && !s_envelope) return kEnvelope;
        // GetIndMtxOp reads the material's own misc: all three indirect matrices must be WARP (none).
        if (R8(mat + 0x1C + 4) != 0 || R8(mat + 0x1C + 5) != 0 || R8(mat + 0x1C + 6) != 0) return kIndirect;
    }
    return kByteCode;
}

struct ModelInfo {
    bool logged = false;
    uint32_t mismatches = 0;  // G3D_NATIVE=2
    std::string name;
};
std::unordered_map<uint32_t, ModelInfo> s_models;

std::string ModelName(uint32_t mdl) {
    const uint32_t name = Ofs(mdl, 0x3C);
    std::string s;
    for (uint32_t i = 0; name != 0 && i < 64; ++i) {
        const char c = static_cast<char>(R8(name + i));
        if (c == 0) break;
        s.push_back(c);
    }
    return s;
}

// G3D_NATIVE_MODELS=a,b,...: only models whose name contains one of these (default: every eligible model).
bool ModelAllowed(const std::string& name) {
    static const std::vector<std::string> filter = [] {
        std::vector<std::string> out;
        if (const char* v = std::getenv("G3D_NATIVE_MODELS")) {
            std::string s(v);
            size_t pos = 0;
            while (pos <= s.size()) {
                const size_t comma = s.find(',', pos);
                const std::string item = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!item.empty()) out.push_back(item);
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        return out;
    }();
    if (filter.empty()) return true;
    for (const auto& f : filter)
        if (name.find(f) != std::string::npos) return true;
    return false;
}

// ---------------------------------------------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------------------------------------------
struct Stats {
    uint64_t calls = 0, native = 0, verifiedOk = 0, verifiedBad = 0;
    uint64_t draws = 0, stageCalls = 0, stageDraws = 0;  // native DRAW records, and those of stage models ("Stg")
    uint64_t declinedDraws = 0;                          // DRAW records in the calls it declined
    uint64_t declined[kNumDecline] = {};
};
Stats s_stats;
int s_mode = 0;  // G3D_NATIVE
bool s_log = false;

Decline Check(const CpuContext* ctx, uint32_t mdl, ModelInfo& info) {
    const uint32_t opa = ctx->gpr[7], xlu = ctx->gpr[8], drawMode = ctx->gpr[10];
    for (const uint32_t guard : s_subtreeGuards)
        if (g_mkwGuestCodePatched[guard]) return kPatched;
    if (drawMode & ~0x1Fu) return kDrawMode;  // Z sorts, IGNORE_MATERIAL, FORCE_LIGHTOFF, NOPPCSYNC
    if (!ModelAllowed(info.name)) return kModelFilter;
    Decline d = kOk;
    if (opa) d = CheckByteCode(mdl, opa);
    if (d == kOk && xlu) d = CheckByteCode(mdl, xlu);
    return d;
}

bool s_compiled = true;  // G3D_NATIVE_COMPILED (default on)

// ---------------------------------------------------------------------------------------------------------------
// Jobs (G3D_NATIVE_ASYNC): a native draw as a recording of its GX HLE calls, replayed later. The draw runs with the
// recorder suppressing every outermost GX HLE call (as verify's native pass does: G3DState, the byte code walk and
// the translated callees run for real), each call recorded with its arguments, matrices and fog by value; the replay
// makes the same calls in the same order. 1: replayed at once on the game thread (checks that the recording is
// complete: frames must be identical to G3D_NATIVE_ASYNC=0).
// ---------------------------------------------------------------------------------------------------------------
int s_async = 2;  // G3D_NATIVE_ASYNC (default 2 = draw worker + vertex-state carry, the verified/timed mode; 0 = inline)
bool s_carry = true;  // G3D_NATIVE_CARRY (async 2): the worker carries the vertex state (0: the game thread mirrors it)
bool s_takeBackMirror = true;  // G3D_NATIVE_TAKEBACK: mirror (default) or wait (MaterializeVertexState)
uint64_t s_pendingFirst = 0;   // game thread: the first deferred job since it last took the vertex state back
struct JobStats {
    // Game thread: jobs recorded, their bytes; deferred to the worker; replayed on the game thread instead (not
    // eligible: __gx dirty state or HLE work in flight; or the vertex-state mirror could not follow the recording).
    uint64_t jobs = 0, bytes = 0, deferred = 0, notEligible = 0, mirrorFallback = 0;
    uint64_t slotWaits = 0;  // game thread: a job slot still held by the worker (all kSlots in flight)
    uint64_t followsJob = 0;  // deferred jobs right after the previous one in the stream (the worker keeps its mirrors)
    // The carry: deferred jobs that started from the worker's own vertex state; times the game thread took the state
    // back (a GX entry needed it), of them the times it waited for the worker, and the wait.
    uint64_t carried = 0, materialized = 0, materializeWaits = 0, materializeWaitNanos = 0;
    uint64_t takeBackMirrored = 0;  // pending jobs the game thread mirrored when it took the state back (not yet done)
    // Take-backs per reason (gx_internal.h GxTakeBackReason, kTakeBackNames), and of them those that put rewrites back.
    uint64_t takeBackBy[8] = {}, takeBackRewrites = 0;
    // Either thread: replayed events, failed replays, worker jobs that ended with dirty shadow state (a cache line of
    // their own: the worker writes them).
    alignas(64) std::atomic<uint64_t> events{0};
    std::atomic<uint64_t> failed{0}, leftoverDirty{0};
};
JobStats s_jobStats;

// A spin-wait iteration's hint to the core.
inline void CpuRelax() {
#if defined(__x86_64__)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield");
#endif
}

inline float BitsF32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// The recorded calls, in order; false at an event it cannot replay (none expected: verify found no kRecOther). With
// plans (a deferred draw's, PlanLoads), its texture and TLUT loads run their plans instead of the HLE entries.
bool Replay(const RecBytes& rec, const uint8_t* plans = nullptr) {
    size_t p = 0;
    uint64_t events = 0;  // added to the shared count once (the game thread writes the stats next to it)
    struct AddEvents {
        uint64_t& n;
        ~AddEvents() { s_jobStats.events.fetch_add(n, std::memory_order_relaxed); }
    } addEvents{events};
    while (p + 5 <= rec.size()) {
        const uint8_t tag = rec[p];
        uint32_t len;
        std::memcpy(&len, &rec[p + 1], 4);
        const uint8_t* d = rec.data() + p + 5;
        const auto word = [d](uint32_t i) {
            uint32_t w;
            std::memcpy(&w, d + 4 * i, 4);
            return w;
        };
        const auto be32 = [d](uint32_t off) {
            return uint32_t(d[off]) << 24 | uint32_t(d[off + 1]) << 16 | uint32_t(d[off + 2]) << 8 | d[off + 3];
        };
        float m[12] = {};
        ++events;
        switch (tag) {
        case kRecFifo: GX_HLE_FIFO_WriteBurst(d, len); break;
        case kRecFifoG3D: GX_HLE_FIFO_WritePackets_G3D(d, len); break;
        case kRecShapeDL: GX__CallShapeDisplayList_G3D(be32(1), be32(5)); break;
        case kRecRegDL: GX__CallRegisterDisplayList_G3D(be32(1), be32(5)); break;
        case kRecLoadTexObj:
            if (plans == nullptr) {
                GX__LoadTexObj_HLE(word(0), word(1));
            } else {
                GX_HLE_RunTexLoadPlan(plans, word(0), word(1));
                plans += GX_HLE_TexLoadPlanBytes();
            }
            break;
        case kRecLoadTlut:
            if (plans == nullptr) {
                GX__LoadTlut_HLE(word(0), word(1));
            } else {
                GX_HLE_RunTlutLoadPlan(plans, word(1));
                plans += GX_HLE_TlutLoadPlanBytes();
            }
            break;
        case kRecSetArray: GX__SetArray_HLE(word(0), word(1), word(2)); break;
        case kRecSetZCompLoc: GX__SetZCompLoc_HLE(word(0)); break;
        case kRecLoadLightObj: GX__LoadLightObjImm_HLE(word(0), word(1)); break;
        case kRecSetFog:
            GX_HLE_SetFogHost(word(0), BitsF32(word(1)), BitsF32(word(2)), BitsF32(word(3)), BitsF32(word(4)), word(5));
            break;
        case kRecLoadPosMtxImm:
            for (uint32_t i = 0; i < 12; ++i) m[i] = BitsF32(word(1 + i));
            GX_HLE_LoadPosMtxImmHost(m, word(0));
            break;
        case kRecLoadNrmMtxImm:
            for (uint32_t i = 0; i < 12; ++i) m[i] = BitsF32(word(1 + i));
            GX_HLE_LoadNrmMtxImmHost(m, word(0));
            break;
        case kRecLoadTexMtxImm:  // u32 id, u32 type, then 12 (3x4) or 8 (2x4) words
            for (uint32_t i = 0; i + 2 < len / 4 && i < 12; ++i) m[i] = BitsF32(word(2 + i));
            GX_HLE_LoadTexMtxImmHost(m, word(0), word(1));
            break;
        default: return false;
        }
        p += 5 + len;
    }
    return true;
}

// G3D_NATIVE_ASYNC=2: the replay on the draw worker, into a deferred job (aurora gx_worker.hpp) whose Job item the
// game thread puts in its command stream where the draw was, so the GX thread decodes it there. The worker replays on
// its own HLE state and register shadow (per thread), loaded with the game thread's at the recording; the game thread
// brings its own state to where the replay leaves it (vertex state from the recording's CP writes and arrays; the
// texture and TLUT loads planned, with their table work and guest writes, and their register shadow; __gx;
// GX_HLE_AfterDeferredDraw) and goes on. The worker touches no guest-thread table (gx_internal.h g_texObjGuard): a
// load's plan carries a copy of its host object. Inputs the replay reads later are the ones the GX thread already reads
// later (resource display lists, vertex and matrix arrays, texture and palette data): a GX sync waits for the GX
// thread, which waits for the job.
struct Job {
    RecBytes rec;                      // the recording
    PlanBytes plans;                   // its texture and TLUT loads' plans, in order (PlanLoads)
    std::vector<uint8_t> vertexState;  // the game thread's HLE vertex state at the recording
    std::vector<uint8_t> shadow;       // its Aurora register shadow
    uint32_t mxcsr = 0;                // its float environment (the HLE and Aurora convert floats)
    bool compiled = false;
    // Nothing entered the command stream since the previous job's item (begin_deferred_job): the GX thread decodes
    // this job from the state the previous one left, which the worker's published-state mirrors describe, so it keeps
    // them (otherwise invalidated: every vertex descriptor, array and texture published again).
    bool followsJob = false;
    // The vertex-state carry (G3D_NATIVE_CARRY, default on): the game thread's vertex state was stale at the recording
    // (an earlier deferred draw's, not taken back yet), so the worker starts from its own, as the previous job left
    // it (vertexState unused), keeping its own vertex fields of the shadow too. At the end of every job the worker
    // publishes both for the game thread to take back (GX_HLE_MaterializeVertexStateSlow).
    bool carry = false;
    std::vector<uint8_t> endVertexState;
    std::vector<uint8_t> endShadowVtx;
    aurora::gx::worker::DeferredJob* aurora = nullptr;
};

// Jobs go to the worker through a ring of slots reused in turn (their buffers keep their capacity): the game thread
// fills the slot after the last one it handed over and publishes it (published_), the worker replays slots in order
// and counts them done (done_). Between jobs the worker spins a while (a frame's draws come a few dozen microseconds
// apart) before it parks; the game thread wakes it only when it has parked.
class DrawWorker {
public:
    static constexpr uint64_t kSlots = 64;
    // Game thread: the slot for its next job (waits while the worker still holds it). Handed over by Push, or left
    // unpublished: the next Acquire returns it again.
    // One slot short of the ring: the last job done keeps its end state (MaterializeVertexState reads it).
    Job& Acquire() {
        Start();
        if (head_ - done_.load(std::memory_order_acquire) >= kSlots - 1) {
            ++s_jobStats.slotWaits;
            aurora::gx::worker::flush();  // as LastJobDone
            WaitDone(head_ - kSlots + 2);
        }
        return slots_[head_ % kSlots];
    }
    // Game thread: jobs handed over, jobs the worker has done (their slots' end states published), a job's slot.
    uint64_t Head() const { return head_; }
    uint64_t Done() const { return done_.load(std::memory_order_acquire); }
    const Job& Slot(uint64_t seq) const { return slots_[seq % kSlots]; }
    void Push() {
        published_.store(++head_, std::memory_order_seq_cst);
        if (parked_.load(std::memory_order_seq_cst)) {
            std::lock_guard<std::mutex> lock(mutex_);
            cv_.notify_one();
        }
    }
    // Game thread: the last job it handed over, once the worker has done it (waits for that); waited: whether it had
    // to. Its end state stays put until the game thread hands over kSlots more jobs.
    const Job& LastJobDone(bool* waited) const {
        *waited = done_.load(std::memory_order_acquire) < head_;
        if (*waited) {
            // The GX thread must be able to reach the job's item (a worker in sync() waits for it): hand it over.
            aurora::gx::worker::flush();
            WaitDone(head_);
        }
        return slots_[(head_ - 1) % kSlots];
    }
private:
    void Start() {
        if (started_) return;
        started_ = true;
        if (const char* v = std::getenv("G3D_NATIVE_SPIN_US")) spin_ = std::chrono::microseconds(std::atoi(v));
        std::thread([this] { Run(); }).detach();  // runs as long as the process (the GX thread waits on its jobs)
    }
    // Game thread: until the worker has done `target` jobs. It does not park with jobs published, so this spins.
    void WaitDone(uint64_t target) const {
        for (uint32_t i = 1; done_.load(std::memory_order_acquire) < target; ++i) {
            CpuRelax();
            if (i % 4096 == 0) std::this_thread::yield();
        }
    }
    // Worker: until a job after `tail` is published.
    void WaitForJob(uint64_t tail) {
        const auto until = std::chrono::steady_clock::now() + spin_;
        for (uint32_t i = 1;; ++i) {
            if (published_.load(std::memory_order_acquire) != tail) return;
            CpuRelax();
            if (i % 64 == 0 && std::chrono::steady_clock::now() >= until) break;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        parked_.store(true, std::memory_order_seq_cst);
        while (published_.load(std::memory_order_seq_cst) == tail) cv_.wait(lock);
        parked_.store(false, std::memory_order_relaxed);
    }
    void Run() {
        GX_HLE_SetDrawWorker(true);
        std::vector<uint8_t> shadow(GXShadowBytes());
        GXShadowUseOwn(shadow.data());
        std::vector<uint8_t> ownVtx(GXShadowVtxBytes());
        for (uint64_t tail = 0;; ++tail) {
            if (published_.load(std::memory_order_acquire) == tail) WaitForJob(tail);
            Job& job = slots_[tail % kSlots];
            // The game thread's float environment (x86 MXCSR, AArch64 FPCR: FZ carries FPSCR[NI]), so replayed
            // conversions treat denormals as the game thread would; on AArch64 the worker used the default FPCR.
            MkwSetHostFpControl(job.mxcsr);
            if (job.carry) {  // its own vertex state, and its own vertex fields over the game thread's shadow
                GXShadowSaveVtx(ownVtx.data());
                GXShadowLoad(job.shadow.data());
                GXShadowLoadVtx(ownVtx.data());
            } else {
                GX_HLE_LoadVertexState(job.vertexState.data());
                GXShadowLoad(job.shadow.data());
            }
            if (job.followsJob) {
                GX_HLE_KeepPublishedStateForNewJob();
            } else {
                GX_HLE_InvalidatePublishedState();
            }
            aurora::gx::worker::attach_deferred_job(job.aurora);
            g_nativeDraw = job.compiled;
            if (!Replay(job.rec, job.plans.data())) s_jobStats.failed.fetch_add(1, std::memory_order_relaxed);
            g_nativeDraw = false;
            if (GXShadowDirtyState() != 0) {
                GXShadowFlushDirty();
                s_jobStats.leftoverDirty.fetch_add(1, std::memory_order_relaxed);
            }
            aurora::gx::worker::finish_deferred_job();
            if (s_carry) {  // published before done_ (release): the game thread reads them after seeing the job done
                job.endVertexState.resize(GX_HLE_VertexStateBytes());
                GX_HLE_SaveVertexState(job.endVertexState.data());
                job.endShadowVtx.resize(GXShadowVtxBytes());
                GXShadowSaveVtx(job.endShadowVtx.data());
            }
            done_.store(tail + 1, std::memory_order_release);
        }
    }

    std::array<Job, kSlots> slots_;
    uint64_t head_ = 0;      // game thread: jobs it handed over
    bool started_ = false;   // game thread
    std::chrono::microseconds spin_{100};  // G3D_NATIVE_SPIN_US
    alignas(64) std::atomic<uint64_t> published_{0};  // written by the game thread
    alignas(64) std::atomic<uint64_t> done_{0};       // written by the worker, as is the next
    std::atomic<bool> parked_{false};
    alignas(64) std::mutex mutex_;  // parking only
    std::condition_variable cv_;
};
// Never destroyed: its detached thread waits on it until the process ends.
DrawWorker& Worker() {
    static DrawWorker* worker = new DrawWorker;
    return *worker;
}

bool MirrorVertexState(int& zCompLoc, uint32_t& drawFmtMask, const RecBytes& rec = s_rec);
// gx_internal.h GxTakeBackReason (not included here), for the carry stats.
constexpr int kTakeBackNativeDraw = 6;
constexpr const char* kTakeBackNames[8] = {"entry", "GXBegin", "FIFO draw", "FIFO CP write", "FIFO indexed XF",
                                           "FIFO list call", "native draw", "log full"};

// The vertex-state carry: the game thread takes the vertex state back when a GX entry point needs it (gx_internal.h
// GX_HLE_MaterializeIfPending) or a draw is replayed here. G3D_NATIVE_TAKEBACK=mirror (default): from the end state
// the worker published for the last pending job it has done - or, if it has done none, the game thread's own stale
// state, which is exactly the state the first pending job started from (MaterializeVertexState loads it back if this
// thread rewrote parts of it) - then the
// pending jobs it has not done yet mirrored here from their recordings, as without the carry (MirrorVertexState and
// the shadow's descriptors and formats per job: GX_HLE_AfterDeferredDraw); no waiting. If a recording cannot be
// mirrored, and with G3D_NATIVE_TAKEBACK=wait: wait for the last job and take the worker's state after it (what
// replaying the draws here would have left). Also while the guest records a display list: mirroring republishes the
// jobs' descriptors and formats to Aurora (GX_HLE_AfterDeferredDraw), which would encode them into the list Aurora
// records (GX__BeginDisplayList_HLE takes the state back first, so this is a guard, not a path taken).
void MaterializeJobsVertexState() {
    GX_HLE_SetVertexStatePending(false);
    ++s_jobStats.materialized;
    DrawWorker& w = Worker();
    const uint64_t head = w.Head();
    if (s_takeBackMirror && !GX_HLE_DisplayListRecording()) {
        uint64_t from = w.Done();
        if (from >= head) {  // all done: the last one's end state
            const Job& last = w.Slot(head - 1);
            GX_HLE_LoadVertexState(last.endVertexState.data());
            GXShadowLoadVtx(last.endShadowVtx.data());
            return;
        }
        if (from > s_pendingFirst) {
            const Job& done = w.Slot(from - 1);
            GX_HLE_LoadVertexState(done.endVertexState.data());
            GXShadowLoadVtx(done.endShadowVtx.data());
        } else {
            from = s_pendingFirst;
        }
        // GX_HLE_AfterDeferredDraw clears the shadow's dirty bits, as the job's replay leaves them right after its
        // draw; the bits this thread's GX calls set since then (a gen mode, SU texture registers...) still have to be
        // sent, so they are put back after the mirror (they include no vertex bits: setting those takes the state back
        // first).
        uint32_t dirty[2];
        GXShadowSaveDirty(dirty);
        bool mirrored = true;
        // Each mirrored draw's GX_HLE_AfterDeferredDraw, published once for the run (gx_dl.cpp
        // GX_HLE_DeferredPublishFinish: only the last publications survive, and the dirty bits are put back below).
        GX_HLE_DeferredPublishBegin();
        for (uint64_t seq = from; seq < head && mirrored; ++seq) {
            int zCompLoc = -1;  // set on this thread when the job was deferred
            uint32_t drawFmtMask = 0;
            mirrored = MirrorVertexState(zCompLoc, drawFmtMask, w.Slot(seq).rec);
            if (mirrored) GX_HLE_DeferredPublishNote(drawFmtMask);
        }
        GX_HLE_DeferredPublishFinish(mirrored);
        GXShadowLoadDirty(dirty);
        s_jobStats.takeBackMirrored += head - from;
        if (mirrored) return;
    }
    const auto start = std::chrono::steady_clock::now();
    bool waited = false;
    const Job& last = w.LastJobDone(&waited);
    if (waited) {
        ++s_jobStats.materializeWaits;
        s_jobStats.materializeWaitNanos += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
    }
    GX_HLE_LoadVertexState(last.endVertexState.data());
    GXShadowLoadVtx(last.endShadowVtx.data());
}

// The take-back. If this thread rewrote parts of the vertex state while it was pending (gx_internal.h
// GxCarryRewriting), the jobs' end state is rebuilt from the stale state they started from, the logged Aurora calls
// replayed on it, and the rewritten parts put back (GX_HLE_VtxRewritesBegin / Finish): sequential execution's state.
std::vector<uint8_t> s_rewriteSaved;  // game thread
void MaterializeVertexState(int reason) {
    s_jobStats.takeBackBy[reason >= 0 && reason < 8 ? reason : 0] += 1;
    const bool rewrites = GX_HLE_VtxRewritesPending();
    if (rewrites) {
        ++s_jobStats.takeBackRewrites;
        s_rewriteSaved.resize(GX_HLE_VertexStateBytes());
        GX_HLE_VtxRewritesBegin(s_rewriteSaved.data());
    }
    MaterializeJobsVertexState();
    if (rewrites) GX_HLE_VtxRewritesFinish(s_rewriteSaved.data());
}

// The list ranges ScanForCarry has marked (below). Marks are sticky (gx_guest_write.h: g_trackedPages bits are only
// ever set), so marking a range again changes nothing: a direct-mapped memo of ranges already marked skips it. The
// same model lists are deferred every frame (~1,200 list calls a frame, each an address canonicalization and a page
// mask test before). Game thread.
inline void NoteListTracked(uint32_t addr, uint32_t size) {
    static std::array<uint64_t, 1024> s_noted{};  // addr << 32 | size, + 1 (0: empty)
    const uint64_t key = (uint64_t{addr} << 32 | size) + 1;
    uint64_t& slot = s_noted[((addr >> 5) * 0x9E3779B1u) >> 22];
    if (slot == key) return;
    GxGuestWrite::NoteTrackedRange(addr, size);
    slot = key;
}

// With the carry, the game thread's whole half of a recorded draw beyond the loads: the last z-compare location it
// set (-1: none), and false when a TLUT load must stay on this thread (Aurora's texture-replacement palette mirror).
// Only the marked events (s_recMarks: every event of these tags, in order).
bool ScanForCarry(int& zCompLoc) {
    for (const RecMark& mark : s_recMarks) {
        if (mark.listSize != 0) {  // a list call recorded directly (RecListCall): its range, as decoded below
            NoteListTracked(mark.listAddr, mark.listSize);
            continue;
        }
        const uint32_t p = mark.at;
        const uint8_t tag = s_rec[p];
        uint32_t len;
        std::memcpy(&len, &s_rec[p + 1], 4);
        if (tag == kRecSetZCompLoc) {
            uint32_t w;
            std::memcpy(&w, &s_rec[p + 5], 4);
            zCompLoc = static_cast<int>(w & 1);
        } else if (tag == kRecLoadTlut && aurora::gfx::texture_replacement::tlut_mirror_active()) {
            return false;
        } else if ((tag == kRecShapeDL || tag == kRecRegDL) && len >= 9) {
            // The worker reads these lists by their write generations (gx_dl.cpp): mark their pages here, before the
            // hand-off, as MirrorVertexState does through GX_HLE_ApplyListVertexState (gx_guest_write.h: a consumer
            // on another thread is exact only if the guest thread marked its range first). Before, only the worker
            // marked them, racing the game thread's AnyTracked check on its next write.
            const uint8_t* d = &s_rec[p + 5];
            const auto be32 = [d](uint32_t off) {
                return uint32_t(d[off]) << 24 | uint32_t(d[off + 1]) << 16 | uint32_t(d[off + 2]) << 8 | d[off + 3];
            };
            NoteListTracked(be32(1), be32(5));
        }
    }
    return true;
}

// The game thread's half of a recorded draw (in s_rec), part one: its HLE vertex state as the replay leaves it. False
// when it could not follow (a list it cannot walk, a draw in FIFO bytes) or a TLUT load must stay on this thread
// (Aurora's texture-replacement palette mirror): the caller restores the vertex state and replays the draw itself.
// Nothing here touches the guest-thread tables or guest memory, so a false costs nothing to undo.
bool MirrorVertexState(int& zCompLoc, uint32_t& drawFmtMask, const RecBytes& rec) {  // rec: s_rec
    size_t p = 0;
    while (p + 5 <= rec.size()) {
        const uint8_t tag = rec[p];
        uint32_t len;
        std::memcpy(&len, &rec[p + 1], 4);
        const uint8_t* d = rec.data() + p + 5;
        const auto word = [d](uint32_t i) {
            uint32_t w;
            std::memcpy(&w, d + 4 * i, 4);
            return w;
        };
        const auto be32 = [d](uint32_t off) {
            return uint32_t(d[off]) << 24 | uint32_t(d[off + 1]) << 16 | uint32_t(d[off + 2]) << 8 | d[off + 3];
        };
        uint32_t mask = 0;
        switch (tag) {
        case kRecFifo:
        case kRecFifoG3D:
            if (!GX_HLE_ApplyFifoVertexState(d, len, &mask)) return false;
            break;
        case kRecShapeDL:
        case kRecRegDL:
            if (!GX_HLE_ApplyListVertexState(be32(1), be32(5), &mask)) return false;
            break;
        case kRecSetArray: GX__SetArray_HLE(word(0), word(1), word(2)); break;  // HLE state only
        case kRecSetZCompLoc: zCompLoc = static_cast<int>(word(0) & 1); break;
        case kRecLoadTlut:
            if (aurora::gfx::texture_replacement::tlut_mirror_active()) return false;
            break;
        default: break;  // texture loads (PlanLoads), XF loads, lights, fog: no HLE vertex state
        }
        drawFmtMask |= mask;
        p += 5 + len;
    }
    return true;
}

// Part two, once the draw is deferred: its texture and TLUT loads planned in order (gx_texture.cpp: the table half,
// the guest writes, this thread's register shadow of the maps), the plans for the worker's replay.
// Only the marked loads (s_recLoadMarks: every texture and TLUT load, in order). The plans' bytes are left
// uninitialized by the resize (PlanBytes): each plan writes all of its own.
void PlanLoads(PlanBytes& plans) {
    plans.clear();
    for (const uint32_t p : s_recLoadMarks) {
        const uint8_t tag = s_rec[p];
        const uint8_t* d = s_rec.data() + p + 5;
        const auto word = [d](uint32_t i) {
            uint32_t w;
            std::memcpy(&w, d + 4 * i, 4);
            return w;
        };
        if (tag == kRecLoadTexObj) {
            const size_t at = plans.size();
            plans.resize(at + GX_HLE_TexLoadPlanBytes());
            GX_HLE_PlanTexLoadForJob(word(0), word(1), plans.data() + at);
        } else if (tag == kRecLoadTlut) {
            const size_t at = plans.size();
            plans.resize(at + GX_HLE_TlutLoadPlanBytes());
            GX_HLE_PlanTlutLoadForJob(word(0), word(1), plans.data() + at);
        }
    }
}

void RunNative(CpuContext* ctx, uint32_t mdl, bool compiled) {
    // Verify already owns the recorder and suppresses every GX effect. Recording
    // an async job here would clear that recording, replace its event tags, and
    // replay/publish real work before the translated comparison. Keep this pass
    // inline; ordinary native draws still use the configured worker path.
    if (s_async != 0 && !g_recOn) {
        const bool defer = s_async == 2 && GXShadowDirtyState() == 0 && GX_HLE_IdleForDeferral();
        if (s_async == 2 && !defer) ++s_jobStats.notEligible;
        // The replay here (below) calls the HLE directly, not through its entry thunks: take the vertex state back. A
        // deferred draw carried on from the worker's state must not follow vertex-state rewrites made here since the
        // last deferral (the worker does not have them): take the state back first, and the job starts from it.
        if (GX_HLE_VertexStatePending() && (!defer || GX_HLE_VtxRewritesPending())) {
            MaterializeVertexState(kTakeBackNativeDraw);
        }
        Job* job = nullptr;
        bool carry = false;
        if (defer) {
            job = &Worker().Acquire();
            carry = s_carry && GX_HLE_VertexStatePending();
            if (!carry) {
                job->vertexState.resize(GX_HLE_VertexStateBytes());
                GX_HLE_SaveVertexState(job->vertexState.data());
            }
            job->shadow.resize(GXShadowBytes());
            GXShadowSave(job->shadow.data());
        }
        s_rec.clear();
        s_recMarks.clear();
        s_recLoadMarks.clear();
        s_recLastFifo = SIZE_MAX;
        g_recOn = g_recSuppress = g_recJob = true;
        s_recDirect = true;
        {
            Drawer drawer(ctx);
            drawer.DrawResMdlDirectly(mdl, ctx->gpr[4], ctx->gpr[5], ctx->gpr[6], ctx->gpr[7], ctx->gpr[8],
                                      ctx->gpr[9], ctx->gpr[10]);
        }
        s_recDirect = false;
        g_recOn = g_recSuppress = g_recJob = false;
        ++s_jobStats.jobs;
        s_jobStats.bytes += s_rec.size();
        if (defer) {
            int zCompLoc = -1;
            uint32_t drawFmtMask = 0;
            if (s_carry ? ScanForCarry(zCompLoc) : MirrorVertexState(zCompLoc, drawFmtMask)) {
                GX_HLE_EnsureFrameActive();
                if (aurora::gx::worker::DeferredJob* aj = aurora::gx::worker::begin_deferred_job(&job->followsJob)) {
                    s_jobStats.followsJob += job->followsJob;
                    if (s_carry) {
                        // The shadow's vertex fields stay stale with the vertex state (taken back with it); the rest
                        // as the replay leaves it.
                        GXShadowAfterDeferredDraw(zCompLoc);
                        GX_HLE_InvalidatePublishedState();
                    } else {
                        GX_HLE_AfterDeferredDraw(zCompLoc, drawFmtMask);
                    }
                    PlanLoads(job->plans);
                    job->rec.swap(s_rec);  // s_rec gets the slot's old buffer, cleared before the next recording
                    job->mxcsr = MkwGetHostFpControl();  // MXCSR or FPCR (the worker applies it)
                    job->compiled = compiled;
                    job->carry = carry;
                    s_jobStats.carried += carry;
                    job->aurora = aj;
                    if (s_carry && !carry) s_pendingFirst = Worker().Head();  // this job's sequence number
                    Worker().Push();
                    if (s_carry) GX_HLE_SetVertexStatePending(true);
                    ++s_jobStats.deferred;
                    return;
                }
            }
            // Replay here, as an undeferred draw (no load planned yet), from the exact vertex state: taken back from
            // the worker if it was stale, else the one the recording started with (undoing a partial mirror). The slot
            // stays unpublished.
            ++s_jobStats.mirrorFallback;
            if (carry) {
                MaterializeVertexState(kTakeBackNativeDraw);
            } else {
                GX_HLE_LoadVertexState(job->vertexState.data());
            }
        }
        g_nativeDraw = compiled;
        if (!Replay(s_rec)) s_jobStats.failed.fetch_add(1, std::memory_order_relaxed);
        g_nativeDraw = false;
        return;
    }
    g_nativeDraw = compiled;
    Drawer drawer(ctx);
    drawer.DrawResMdlDirectly(mdl, ctx->gpr[4], ctx->gpr[5], ctx->gpr[6], ctx->gpr[7], ctx->gpr[8], ctx->gpr[9],
                              ctx->gpr[10]);
    g_nativeDraw = false;
}

// G3D_NATIVE=2: the native call recorded with every GX call suppressed, its G3DState restored, then the translated
// call recorded for real; both must match.
struct Snapshot {
    std::vector<uint8_t> bss, sbss, seed;
    void Take() {
        bss.assign(Memory::GetPointer(kSnapBss, kSnapBssSize), Memory::GetPointer(kSnapBss, kSnapBssSize) + kSnapBssSize);
        sbss.assign(Memory::GetPointer(kSnapSbss, kSnapSbssSize),
                    Memory::GetPointer(kSnapSbss, kSnapSbssSize) + kSnapSbssSize);
        seed.assign(Memory::GetPointer(kSnapSortSeed, kSnapSortSeedSize),
                    Memory::GetPointer(kSnapSortSeed, kSnapSortSeedSize) + kSnapSortSeedSize);
    }
    // Writes back the words that differ (guest memory holds big-endian bytes; Read32 / W32 swap both ways).
    void Restore() const {
        auto put = [](uint32_t base, const std::vector<uint8_t>& bytes) {
            for (uint32_t off = 0; off < bytes.size(); off += 4) {
                const uint32_t want = uint32_t(bytes[off]) << 24 | uint32_t(bytes[off + 1]) << 16 |
                                      uint32_t(bytes[off + 2]) << 8 | bytes[off + 3];
                if (R32(base + off) != want) W32(base + off, want);
            }
        };
        put(kSnapBss, bss);
        put(kSnapSbss, sbss);
        put(kSnapSortSeed, seed);
    }
};

std::string Hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789ABCDEF";
    std::string s;
    for (size_t i = 0; i < n; ++i) {
        s.push_back(d[p[i] >> 4]);
        s.push_back(d[p[i] & 15]);
        if (i % 4 == 3) s.push_back(' ');
    }
    return s;
}

// The first difference between two recordings, described for the log.
std::string FirstDifference(const RecBytes& a, const RecBytes& b) {
    size_t pa = 0, pb = 0;
    for (int ev = 0;; ++ev) {
        const bool ea = pa >= a.size(), eb = pb >= b.size();
        if (ea || eb) {
            if (ea && eb) return "equal";
            return "event " + std::to_string(ev) + ": " + (ea ? "native ended" : "translated ended") + " first";
        }
        uint32_t la, lb;
        std::memcpy(&la, &a[pa + 1], 4);
        std::memcpy(&lb, &b[pb + 1], 4);
        if (a[pa] != b[pb] || la != lb || std::memcmp(&a[pa + 5], &b[pb + 5], la) != 0) {
            size_t at = 0;
            const uint32_t common = std::min(la, lb);
            while (at < common && a[pa + 5 + at] == b[pb + 5 + at]) ++at;
            const size_t from = at > 8 ? at - 8 : 0;
            return "event " + std::to_string(ev) + " tag " + std::to_string(a[pa]) + "/" + std::to_string(b[pb]) +
                   " len " + std::to_string(la) + "/" + std::to_string(lb) + " first diff byte " + std::to_string(at) +
                   "\n    native     @" + std::to_string(from) + ": " +
                   Hex(&a[pa + 5 + from], std::min<size_t>(40, la > from ? la - from : 0)) +
                   "\n    translated @" + std::to_string(from) + ": " +
                   Hex(&b[pb + 5 + from], std::min<size_t>(40, lb > from ? lb - from : 0));
        }
        pa += 5 + la;
        pb += 5 + lb;
    }
}

void Verify(CpuContext* ctx, uint32_t mdl, ModelInfo& info) {
    // The translated call needs its arguments back after the native one's translated callees used the volatile
    // registers.
    uint32_t gpr[13];
    std::memcpy(gpr, ctx->gpr, sizeof(gpr));
    const uint32_t cr = ctx->cr, xer = ctx->xer, ctr = ctx->ctr, lr = ctx->lr;
    Snapshot before, afterNative, afterGuest;
    before.Take();
    s_verifyGuestBytes.clear();
    s_rec.clear();
    s_recMarks.clear();
    s_recLoadMarks.clear();
    s_recLastFifo = SIZE_MAX;
    g_recOn = g_recSuppress = true;
    RunNative(ctx, mdl, false);
    g_recOn = g_recSuppress = false;
    RecBytes nativeRec;
    nativeRec.swap(s_rec);
    afterNative.Take();
    for (auto& byte : s_verifyGuestBytes) {
        byte.native = Memory::Read8(byte.address);
        Memory::Write8(byte.address, byte.before);
    }
    before.Restore();
    std::memcpy(ctx->gpr, gpr, sizeof(gpr));
    ctx->cr = cr;
    ctx->xer = xer;
    ctx->ctr = ctr;
    ctx->lr = lr;
    s_rec.clear();
    s_recMarks.clear();
    s_recLoadMarks.clear();
    s_recLastFifo = SIZE_MAX;
    g_recOn = true;
    func_801A64B0_body(ctx);
    g_recOn = false;
    afterGuest.Take();
    const bool eventsEqual = nativeRec == s_rec;
    const bool guestWritesEqual = std::all_of(s_verifyGuestBytes.begin(), s_verifyGuestBytes.end(),
        [](const VerifyGuestByte& byte) { return Memory::Read8(byte.address) == byte.native; });
    const bool stateEqual = guestWritesEqual &&
        afterNative.bss == afterGuest.bss && afterNative.sbss == afterGuest.sbss && afterNative.seed == afterGuest.seed;
    if (eventsEqual && stateEqual) {
        ++s_stats.verifiedOk;
        return;
    }
    ++s_stats.verifiedBad;
    static int logged = 0;
    if (info.mismatches++ < 3 && logged++ < 120) {
        std::string what = eventsEqual ? std::string("events equal") : FirstDifference(nativeRec, s_rec);
        if (!guestWritesEqual) what += "\n    guest HLE writes differ";
        if (!stateEqual) {
            for (size_t i = 0; i < afterNative.bss.size(); i += 4)
                if (std::memcmp(&afterNative.bss[i], &afterGuest.bss[i], 4) != 0) {
                    what += "\n    G3DState differs at bss+" + std::to_string(i) + ": native " +
                            Hex(&afterNative.bss[i], 4) + " translated " + Hex(&afterGuest.bss[i], 4);
                    break;
                }
            for (size_t i = 0; i < afterNative.sbss.size(); i += 4)
                if (std::memcmp(&afterNative.sbss[i], &afterGuest.sbss[i], 4) != 0) {
                    what += "\n    G3DState differs at sbss+" + std::to_string(i) + ": native " +
                            Hex(&afterNative.sbss[i], 4) + " translated " + Hex(&afterGuest.sbss[i], 4);
                    break;
                }
        }
        RT_LOGF(RT_TAG_GX, "[g3dn] verify MISMATCH %s (%zu/%zu event bytes): %s\n", info.name.c_str(), nativeRec.size(),
                s_rec.size(), what.c_str());
    }
}

// Scene changes (G3dNativeNoteScene, from the game's scene reports on whichever thread makes them): the drawing thread
// drops its per-model entries (their names would go stale when a later scene's model takes the address).
std::atomic<uint32_t> s_sceneChanges{0};
uint32_t s_sceneChangesSeen = 0;

std::atomic<bool> s_drawSuppressed{false};  // SetDrawSuppressed

bool DrawResMdlDirectlyHook(CpuContext* ctx) {
    // The projection the model is drawn with: what code (Brawl Online's plugin, around a HUD model's draw) left in the
    // GX state since GXSetProjection, before the draw is recorded or deferred to the worker (gx_transform.cpp).
    GX_HLE_SyncGuestProjection();
    if (const uint32_t n = s_sceneChanges.load(std::memory_order_relaxed); n != s_sceneChangesSeen) {
        s_sceneChangesSeen = n;
        s_models.clear();
    }
    ++s_stats.calls;
    const uint32_t mdl = R32(ctx->gpr[3]);
    auto [it, inserted] = s_models.try_emplace(mdl);
    ModelInfo& info = it->second;
    if (inserted) info.name = ModelName(mdl);
    const Decline d = Check(ctx, mdl, info);
    if (s_log && !info.logged) {
        info.logged = true;
        RT_LOGF(RT_TAG_GX, "[g3dn] model %s @0x%08X: drawMode 0x%X opa %d xlu %d replacement %d: %s\n",
                info.name.c_str(), mdl, ctx->gpr[10], ctx->gpr[7] != 0, ctx->gpr[8] != 0, ctx->gpr[9] != 0,
                kDeclineNames[d]);
    }
    if (d != kOk) {
        ++s_stats.declined[d];
        for (const uint32_t bc : {ctx->gpr[7], ctx->gpr[8]})
            for (uint32_t p = bc, n = 0; p != 0 && n < 4096 && R8(p) != 1; p += 8, ++n)
                if (R8(p) == 4) ++s_stats.declinedDraws;
        return false;
    }
    if (s_drawSuppressed.load(std::memory_order_relaxed)) {
        Drawer drawer(ctx);
        drawer.SortOnly(mdl, ctx->gpr[4], ctx->gpr[5], ctx->gpr[6], ctx->gpr[7], ctx->gpr[8], ctx->gpr[9], ctx->gpr[10]);
        return true;
    }
    if (s_mode == 2) {
        Verify(ctx, mdl, info);
        return true;  // the translated call drew it
    }
    const uint64_t recordsBefore = s_drawRecords;
    RunNative(ctx, mdl, s_compiled);
    ++s_stats.native;
    s_stats.draws += s_drawRecords - recordsBefore;
    if (info.name.find("Stg") != std::string::npos) {
        ++s_stats.stageCalls;
        s_stats.stageDraws += s_drawRecords - recordsBefore;
    }
    return true;
}

struct Install {
    Install() {
        // On by default (G3D_NATIVE=0: the translated draw path; 2: verify against it). Exact: verify mode found 0
        // mismatches over whole vanilla Training runs, 42.7 calls a frame, and a call it does not handle,
        // or whose subtree a mod patched, runs the translated function.
        const char* v = std::getenv("G3D_NATIVE");
        s_mode = v != nullptr ? std::atoi(v) : 1;
        s_log = std::getenv("G3D_NATIVE_LOG") != nullptr;
        if (const char* async = std::getenv("G3D_NATIVE_ASYNC")) s_async = std::atoi(async);
        if (const char* carry = std::getenv("G3D_NATIVE_CARRY")) s_carry = carry[0] != '0';
        if (const char* takeBack = std::getenv("G3D_NATIVE_TAKEBACK")) s_takeBackMirror = std::string(takeBack) != "wait";
        const char* compiled = std::getenv("G3D_NATIVE_COMPILED");
        s_compiled = compiled == nullptr || compiled[0] != '0';
        const char* fast = std::getenv("G3D_NATIVE_FAST");
        s_fast = fast == nullptr || fast[0] != '0';
        const char* envelope = std::getenv("G3D_NATIVE_ENVELOPE");
        s_envelope = envelope == nullptr || envelope[0] != '0';
        const char* shape = std::getenv("G3D_NATIVE_SHAPE");
        s_shape = shape == nullptr || shape[0] != '0';
        if (s_mode == 1 || s_mode == 2) {
            FindSubtreeGuards();
            g_mkwNativeHook[0] = DrawResMdlDirectlyHook;
            GuestCodePatch::RegisterOriginalNative(0x801A64B0u, [](CpuContext* ctx) {
                if (!DrawResMdlDirectlyHook(ctx)) func_801A64B0_body(ctx);
            });
            RT_LOGF(RT_TAG_GX, "[g3dn] native nw4r g3d draws: G3D_NATIVE=%d, fast %d, compiled %d, envelope %d, shape "
                    "%d, async %d, carry %d, %zu/%zu subtree functions guarded against mod patches\n",
                    s_mode, s_fast ? 1 : 0, s_compiled ? 1 : 0, s_envelope ? 1 : 0, s_shape ? 1 : 0, s_async,
                    s_carry ? 1 : 0, s_subtreeGuards.size(), std::size(kSubtree));
        }
    }
} s_install;

}  // namespace

void SetDrawSuppressed(bool suppressed) { s_drawSuppressed.store(suppressed, std::memory_order_relaxed); }
}  // namespace g3dn

// gx_internal.h GX_HLE_MaterializeIfPending's slow path (the vertex-state carry).
extern "C" void GX_HLE_MaterializeVertexStateSlow(int reason) { g3dn::MaterializeVertexState(reason); }

extern "C" void GXClearCompiledDisplayLists();  // Aurora GXDispList.cpp

// Called on every scene the game reports entering (wii_remote_input.cpp NoteGameReport, from OSReport's "a -> b"
// lines): the previous scene's models and display lists are gone, so their cached forms go too.
void G3dNativeNoteScene() {
    if (g_mkwNativeHook[0] == nullptr) return;
    g3dn::s_sceneChanges.fetch_add(1, std::memory_order_relaxed);
    GXClearCompiledDisplayLists();
    GX__ClearShapeDisplayLists_G3D();
}

// Called by the fps log (vi.cpp, NSMBW_LOG_FPS) every window of `frames` presented frames.
void G3dCalcLogStats(double frames);  // g3d_calc.cpp: the calc natives' counters

extern "C" void GX_HLE_LogRawBeginStats(double frames);  // gx_fifo.cpp: immediate-mode begins on the raw path
extern "C" void GX_HLE_LogTexPlanStats(double frames);   // gx_texture.cpp: deferred draws' texture plans
extern "C" void GX_HLE_LogListImageStats(double frames);  // gx_dl.cpp: the take-back mirror's list writes

void G3dNativeLogStats(double frames) {
    using g3dn::s_stats;
    G3dCalcLogStats(frames);
    GX_HLE_LogRawBeginStats(frames);
    GX_HLE_LogTexPlanStats(frames);
    GX_HLE_LogListImageStats(frames);
    if (g_mkwNativeHook[0] == nullptr || frames <= 0) return;
    std::string declined;
    for (int i = 1; i < g3dn::kNumDecline; ++i)
        if (s_stats.declined[i] != 0)
            declined += std::string(", ") + g3dn::kDeclineNames[i] + " " +
                        std::to_string(s_stats.declined[i] / frames).substr(0, 5);
    RT_LOGF(RT_TAG_VI, "g3d native: %.1f DrawResMdlDirectly calls per frame, %.1f native, %.1f verified ok, %.1f "
            "verify mismatch; declined%s; DRAW records: %.1f native (stage models: %.1f calls, %.1f records), %.1f "
            "in declined calls\n",
            s_stats.calls / frames, s_stats.native / frames, s_stats.verifiedOk / frames,
            s_stats.verifiedBad / frames, declined.empty() ? " none" : declined.c_str(), s_stats.draws / frames,
            s_stats.stageCalls / frames, s_stats.stageDraws / frames, s_stats.declinedDraws / frames);
    s_stats = {};
    if (g3dn::s_async != 0) {
        g3dn::JobStats& j = g3dn::s_jobStats;
        const uint64_t events = j.events.exchange(0), failed = j.failed.exchange(0), dirty = j.leftoverDirty.exchange(0);
        RT_LOGF(RT_TAG_VI, "g3d native jobs (G3D_NATIVE_ASYNC=%d): %.1f/frame (%.1f deferred to the draw worker, %.1f "
                "not eligible, %.1f mirror fallback), %.0f recorded bytes and %.0f events per job, %.1f/frame failed to "
                "replay, %.2f/frame left dirty state, %.2f/frame waits for a free job slot, %.1f/frame following the "
                "previous job\n",
                g3dn::s_async, j.jobs / frames, j.deferred / frames, j.notEligible / frames, j.mirrorFallback / frames,
                j.jobs ? double(j.bytes) / j.jobs : 0.0, j.jobs ? double(events) / j.jobs : 0.0, failed / frames,
                dirty / frames, j.slotWaits / frames, j.followsJob / frames);
        j.jobs = j.bytes = j.deferred = j.notEligible = j.mirrorFallback = j.slotWaits = j.followsJob = 0;
        if (g3dn::s_async == 2 && g3dn::s_carry) {
            RT_LOGF(RT_TAG_VI, "g3d native vertex-state carry (take-back %s): %.1f/frame jobs carried, %.2f/frame taken "
                    "back by the game thread, %.2f/frame jobs mirrored then, %.2f/frame waits for the worker, %.4f "
                    "ms/frame waiting\n",
                    g3dn::s_takeBackMirror ? "mirror" : "wait", j.carried / frames, j.materialized / frames,
                    j.takeBackMirrored / frames, j.materializeWaits / frames, j.materializeWaitNanos / frames / 1e6);
            j.carried = j.materialized = j.materializeWaits = j.materializeWaitNanos = j.takeBackMirrored = 0;
            std::string by;
            for (int i = 0; i < 8; ++i) {
                if (j.takeBackBy[i] == 0) continue;
                char part[64];
                std::snprintf(part, sizeof part, "%s%s %.2f", by.empty() ? "" : ", ", g3dn::kTakeBackNames[i],
                              j.takeBackBy[i] / frames);
                by += part;
                j.takeBackBy[i] = 0;
            }
            RT_LOGF(RT_TAG_VI, "g3d native carry rewrites: %.1f/frame draws with no take-back, %.2f/frame take-backs "
                    "putting rewrites back; taken back at: %s\n",
                    GX_HLE_TakeVtxRewriteDraws() / frames, j.takeBackRewrites / frames, by.empty() ? "none" : by.c_str());
            j.takeBackRewrites = 0;
        }
    }
}
