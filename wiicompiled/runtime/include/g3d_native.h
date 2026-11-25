#pragma once
// Native nw4r g3d draw path (hle/g3d_native.cpp): the HLE-boundary recorder it is verified with.
//
// G3D_NATIVE=2 checks every native DrawResMdlDirectly call against the translated one, in the same frame: the native
// path runs first with the recorder suppressing GPU effects (guest texture-object writes are journaled and restored),
// then the translated function runs for real while recorded. The two event streams, and the
// G3DState bytes each leaves in guest memory, and complete CPU register results must be identical. This does not
// check every guest memory write (notably dead stack bytes), so a pass alone cannot certify arbitrary-mod safety.
// Native drawing is opt-in because it currently fails the register comparison. The recorder sits at the GX HLE entry points both
// paths reach: guest FIFO writes, GXCallDisplayList, and the GX calls nw4r g3d makes.
#include <cstdint>
#include <cstring>

namespace g3dn {

inline uint32_t F32Bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}
// The words of a guest matrix (big-endian in memory, host order here), for matrix-load events.
void RecGuestWords(uint32_t* out, uint32_t addr, uint32_t words);
// Verification must keep guest-visible HLE writes while suppressing GPU effects.
// Save a byte before such a write so the translated pass starts from the same input.
void SaveVerifyGuestByte(uint32_t address);

enum RecTag : uint8_t {
    kRecFifo = 1,        // bytes written to the GX FIFO; a direct GXCallDisplayList records as its FIFO form (0x40)
    kRecLoadTexObj,      // u32 guest GXTexObj, u32 map
    kRecLoadTlut,        // u32 guest GXTlutObj, u32 tlut
    kRecSetArray,        // u32 attr, u32 base, u32 stride
    kRecSetFog,          // u32 type, f32 startz, endz, nearz, farz, u32 color
    kRecSetZCompLoc,     // u32 beforeTex
    kRecLoadPosMtxImm,   // u32 id, 12 words
    kRecLoadNrmMtxImm,   // u32 id, 12 words
    kRecLoadTexMtxImm,   // u32 id, u32 type, 12 or 8 words
    kRecLoadLightObj,    // u32 guest GXLightObj, u32 light
    kRecSetCurrentMtx,   // u32 id
    kRecOther,           // u32 guest address of another GX HLE entry (unexpected inside the subtree)
    // Job recording only (g_recJob: a native draw recorded for later replay, G3D_NATIVE_ASYNC): the shortcuts keep
    // their own tags so the replay takes them again (verify records them as the FIFO bytes the translated path makes).
    kRecFifoG3D,         // bytes for GX_HLE_FIFO_WritePackets_G3D
    kRecShapeDL,         // the FIFO form of a call (0x40, u32 address, u32 size) for GX__CallShapeDisplayList_G3D
    kRecRegDL,           // the same, for GX__CallRegisterDisplayList_G3D
};

// A thread_local with a constant initializer, declared so: other translation units then access it directly instead
// of through a TLS wrapper call that checks for a dynamic initializer (every HLE entry reads the recorder flags).
#ifndef MKW_TLS_CONSTINIT
#if defined(__clang__)
#define MKW_TLS_CONSTINIT [[clang::require_constant_initialization]]
#elif defined(__GNUC__) && __GNUC__ >= 10
#define MKW_TLS_CONSTINIT __constinit
#else
#define MKW_TLS_CONSTINIT
#endif
#endif

// Recording a native draw for replay (G3D_NATIVE_ASYNC) rather than for verification.
MKW_TLS_CONSTINIT extern thread_local bool g_recJob;

// True while the native path draws (G3D_NATIVE=1): a display list it calls with draw commands goes to Aurora through
// GXCallDisplayListCompiled (unless G3D_NATIVE_COMPILED=0). Per thread, as the recorder state below: the game thread
// records a draw while the draw worker replays an earlier one (G3D_NATIVE_ASYNC=2).
MKW_TLS_CONSTINIT extern thread_local bool g_nativeDraw;

// Rollback's light advances (brawl_rollback.cpp): while set, a native DrawResMdlDirectly call leaves the guest memory a
// draw would (its Z sorts, std::sort's pivot generator, G3DState's PosNrm array) but draws nothing; a call the native
// path declines still runs the translated function. Without the native path (G3D_NATIVE=0) models draw in full.
void SetDrawSuppressed(bool suppressed);

MKW_TLS_CONSTINIT extern thread_local bool g_recOn;        // recording
MKW_TLS_CONSTINIT extern thread_local bool g_recSuppress;  // ... and the outermost HLE call returns without effect
MKW_TLS_CONSTINIT extern thread_local int g_recDepth;
void RecEvent(uint8_t tag, const void* payload, uint32_t bytes);

// One per instrumented GX HLE entry point: records the outermost entry only (an HLE calling another HLE is one
// event), and tells it to return at once when the recorder suppresses.
struct HleRec {
    bool counted = false;
    bool skip = false;
    HleRec(uint8_t tag, const void* payload, uint32_t bytes) {
        if (__builtin_expect(g_recOn, 0)) {
            counted = true;
            if (g_recDepth++ == 0) {
                RecEvent(tag, payload, bytes);
                skip = g_recSuppress;
            }
        }
    }
    ~HleRec() {
        if (counted) --g_recDepth;
    }
    HleRec(const HleRec&) = delete;
    HleRec& operator=(const HleRec&) = delete;
};

}  // namespace g3dn

// Records the call (payload words, host order) and returns from the enclosing void HLE function when suppressed.
#define G3D_REC_HLE(tag, ...)                                                   \
    const uint32_t g3dRecPayload[] = {__VA_ARGS__};                             \
    g3dn::HleRec g3dRec(tag, g3dRecPayload, sizeof(g3dRecPayload));             \
    if (__builtin_expect(g3dRec.skip, 0)) return
