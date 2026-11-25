#include <memory>
#include <tuple>
#include <map>
// gx_texture.cpp - Texture Object and TLUT Functions
#include "gx_internal.h"
#include "runtime_log.h"

#include <algorithm>
#include <cstdlib>

// The draw worker (gx_internal.h t_gxDrawWorker) leaves guest memory and the guest-thread tables to the game thread,
// which plans a deferred draw's texture and TLUT loads when it records them (GX_HLE_Plan*ForJob below).
thread_local bool t_gxDrawWorker = false;
extern "C" void GX_HLE_SetDrawWorker(bool worker) { t_gxDrawWorker = worker; }

// Helper to write GXTexObj structure to guest memory in SDK format
// This is needed because other code may read the structure directly
static void WriteGuestTexObj(uint32_t addr, uint32_t dataAddr, uint16_t width, uint16_t height, 
                             uint32_t format, uint32_t wrapS, uint32_t wrapT, bool mipmap,
                             bool writeTlut, uint32_t tlut) {
    dataAddr = CanonicalizeGxMainRamAddress(dataAddr);
    // The struct ends all zero but for the words written below: only the user data word (+0x10) and, without a TLUT,
    // +0x18 need their zero (the rest are rewritten below, so their zero stores changed nothing).
    Memory::Write32(addr + 0x10, 0);
    if (!writeTlut) {
        Memory::Write32(addr + 0x18, 0);
    }

    uint32_t word0 = (wrapS & 0x3u) | ((wrapT & 0x3u) << 2) | 0x10u;
    uint32_t word1 = 0;
    if (!mipmap) {
        word0 = (word0 & 0xFFFFFF10u) | (wrapS & 0x3u) | ((wrapT & 0x3u) << 2) | 0x90u;
    } else {
        word0 = (word0 & 0xFFFFFF10u) | (wrapS & 0x3u) | ((wrapT & 0x3u) << 2) |
                (((format - 8u) < 3u) ? 0xB0u : 0xD0u);
        const uint32_t maxDim = std::max<uint32_t>(width, height);
        const uint32_t maxLod = maxDim > 0 ? (31u - static_cast<uint32_t>(__builtin_clz(maxDim))) : 0u;
        word1 |= (std::min<uint32_t>(maxLod * 16u, 0xFFu) << 8);
    }

    uint32_t blockShiftX = 2;
    uint32_t blockShiftY = 2;
    uint8_t blockType = 2;
    switch (format & 0xFu) {
    case 0:
    case 8:
        blockShiftX = 3;
        blockShiftY = 3;
        blockType = 1;
        break;
    case 1:
    case 2:
    case 9:
        blockShiftX = 3;
        blockShiftY = 2;
        blockType = 2;
        break;
    case 3:
    case 4:
    case 5:
    case 10:
        blockShiftX = 2;
        blockShiftY = 2;
        blockType = 2;
        break;
    case 6:
        blockShiftX = 2;
        blockShiftY = 2;
        blockType = 3;
        break;
    case 14:
        blockShiftX = 3;
        blockShiftY = 3;
        blockType = 0;
        break;
    default:
        blockShiftX = 2;
        blockShiftY = 2;
        blockType = 2;
        break;
    }

    const uint32_t word2 =
        ((width - 1u) & 0x3FFu) |
        (((height - 1u) & 0x3FFu) << 10) |
        ((format & 0xFu) << 20);
    const uint32_t word3 = (dataAddr >> 5) & 0x00FFFFFFu;
    const uint32_t blocksX = (static_cast<uint32_t>(width) + ((1u << blockShiftX) - 1u)) >> blockShiftX;
    const uint32_t blocksY = (static_cast<uint32_t>(height) + ((1u << blockShiftY) - 1u)) >> blockShiftY;
    const uint16_t blockCount = static_cast<uint16_t>((blocksX * blocksY) & 0x7FFFu);
    uint8_t flags = mipmap ? 0x03u : 0x02u;

    Memory::Write32(addr + 0x00, word0);
    Memory::Write32(addr + 0x04, word1);
    Memory::Write32(addr + 0x08, word2);
    Memory::Write32(addr + 0x0C, word3);
    Memory::Write32(addr + 0x14, format);
    if (writeTlut) {
        Memory::Write32(addr + 0x18, tlut);
    }
    Memory::Write16(addr + 0x1C, blockCount);
    Memory::Write8(addr + 0x1E, blockType);
    Memory::Write8(addr + 0x1F, flags);
}

// SDK min-filter enum -> the hardware encoding stored in GXTexObj word0[7:5]
// (GX2HWFiltConv). Entries 6 and 7 have no SDK filter and read back as 0; the
// inverse table (HW -> SDK) lives in gx_objects.cpp.
static const uint8_t kGxToHwMinFilter[8] = {0, 4, 1, 5, 2, 6, 0, 0};

// Helper to update LOD info in guest GXTexObj structure
static void WriteGuestTexObjLOD(uint32_t addr, uint32_t minFilt, uint32_t magFilt,
                                 float minLod, float maxLod, float lodBias,
                                 bool biasClamp, bool edgeLod, uint32_t maxAniso) {
    // Read existing word0 and update relevant bits
    uint32_t word0 = Memory::Read32(addr + 0x00);

    // bits [4] = magFilter, bits [7:5] = hardware minFilter encoding
    uint32_t minFiltBits = (minFilt < 6u) ? kGxToHwMinFilter[minFilt] : 4;
    word0 = (word0 & ~0xF0) | ((magFilt & 1) << 4) | ((minFiltBits & 7) << 5);
    
    // bits [8] = edgeLod (inverted), bits [17:9] = lodBias (signed, scaled by 32)
    int8_t biasScaled = (int8_t)(lodBias * 32.0f);
    word0 = (word0 & ~0x3FF00) | ((edgeLod ? 0 : 1) << 8) | ((uint32_t)(uint8_t)biasScaled << 9);
    
    // bits [20:19] = maxAniso, bit [21] = biasClamp
    word0 = (word0 & ~0x380000) | ((maxAniso & 0x3) << 19) | ((biasClamp ? 1 : 0) << 21);
    
    Memory::Write32(addr + 0x00, word0);
    
    // word1: bits [7:0] = minLod * 16, bits [15:8] = maxLod * 16
    uint8_t minLodScaled = (uint8_t)(minLod * 16.0f);
    uint8_t maxLodScaled = (uint8_t)(maxLod * 16.0f);
    uint32_t word1 = Memory::Read32(addr + 0x04);
    word1 = (word1 & 0xFFFF0000) | minLodScaled | ((uint32_t)maxLodScaled << 8);
    Memory::Write32(addr + 0x04, word1);
}

// ============================================================================
// Texture Object Initialization
// ============================================================================

static uint32_t g_unsupportedFormatLogCount = 0;

// Must cover every format aurora's convert_texture() decodes; anything missing here is refused
// and never reaches aurora (GX_TF_C14X2 was once missing despite aurora decoding it, silently
// dropping C14X2 binds). GX_CTF_YUVA8 stays listed for THP EFB-copy frames, which never go through convert_texture.
static bool IsAuroraLoadableTexFormat(uint32_t fmt) noexcept {
    switch (fmt) {
    case GX_TF_I4:
    case GX_TF_I8:
    case GX_TF_IA4:
    case GX_TF_IA8:
    case GX_TF_C4:
    case GX_TF_C8:
    case GX_TF_C14X2:
    case GX_TF_RGB565:
    case GX_TF_RGB5A3:
    case GX_TF_Z8:
    case GX_TF_Z16:
    case GX_TF_Z24X8:
    case GX_TF_RGBA8:
    case GX_CTF_YUVA8:
    case GX_TF_CMPR:
        return true;
    default:
        return false;
    }
}

static GXTexWrapMode SanitizeWrapMode(uint32_t rawWrap) noexcept {
    switch (rawWrap) {
    case GX_CLAMP:
    case GX_REPEAT:
    case GX_MIRROR:
        return static_cast<GXTexWrapMode>(rawWrap);
    default:
        return GX_CLAMP;
    }
}

static GXTexFilter SanitizeMinFilter(uint32_t rawFilter) noexcept {
    switch (rawFilter) {
    case GX_NEAR:
    case GX_LINEAR:
    case GX_NEAR_MIP_NEAR:
    case GX_LIN_MIP_NEAR:
    case GX_NEAR_MIP_LIN:
    case GX_LIN_MIP_LIN:
        return static_cast<GXTexFilter>(rawFilter);
    // Some decoded hardware encodings surface as 6/7; map to the strongest valid mip filter.
    case 6:
    case 7:
        return GX_LIN_MIP_LIN;
    default:
        return GX_LINEAR;
    }
}

static GXTexFilter SanitizeMagFilter(uint32_t rawFilter) noexcept {
    return (rawFilter == GX_NEAR) ? GX_NEAR : GX_LINEAR;
}

static GXAnisotropy SanitizeAniso(uint32_t rawAniso) noexcept {
    switch (rawAniso) {
    case GX_ANISO_1:
    case GX_ANISO_2:
    case GX_ANISO_4:
        return static_cast<GXAnisotropy>(rawAniso);
    default:
        return GX_ANISO_4;
    }
}

static bool IsReasonableTextureDimensions(uint16_t width, uint16_t height) noexcept {
    // Hardware textures are small; reject absurd dimensions that can destabilize host texture creation.
    constexpr uint16_t kMaxDimension = 4096;
    return width > 0 && height > 0 && width <= kMaxDimension && height <= kMaxDimension;
}

static float SanitizeLodValue(float lod, float fallback) noexcept {
    if (!std::isfinite(lod)) {
        return fallback;
    }
    return std::max(0.0f, lod);
}

static float SanitizeLodBias(float lodBias) noexcept {
    if (!std::isfinite(lodBias)) {
        return 0.0f;
    }
    return std::clamp(lodBias, -4.0f, 3.99f);
}

// Republish the host texobj's LOD/filter block from `meta`, with the three
// fields each caller may be overriding passed in explicitly.
static void ApplyHostTexObjLod(GXTexObj* obj, const TexObjMeta& meta, GXTexFilter minFilter,
                               GXTexFilter magFilter, float lodBias, GXAnisotropy maxAniso) {
    GXInitTexObjLOD(obj, minFilter, magFilter, meta.minLod, meta.maxLod, lodBias,
                    meta.biasClamp ? GX_TRUE : GX_FALSE, meta.edgeLod ? GX_TRUE : GX_FALSE,
                    maxAniso);
}

static uint32_t ComputeMaxMipLevel(uint16_t width, uint16_t height) noexcept {
    const uint32_t w = std::max<uint32_t>(1u, width);
    const uint32_t h = std::max<uint32_t>(1u, height);
    // Halving a side to 1 takes floor(log2) steps, and the levels run until both sides are 1.
    return static_cast<uint32_t>(31 - __builtin_clz(std::max(w, h)));
}

extern "C" void GX__InitTexObj_HLE(uint32_t oa, uint32_t da, uint32_t w, uint32_t h, uint32_t f, uint32_t ws, uint32_t wt, uint32_t m) {
    if (w == 0 || h == 0) {
        auto* cpu = TryGetCpuContext();
        RT_LOGF(RT_TAG_GX,
                "GXInitTexObj zero size oa=0x%08X da=0x%08X w=%u h=%u f=%u ws=%u wt=%u m=%u pc=0x%08X lr=0x%08X\n",
                oa, da, w, h, f, ws, wt, m, cpu ? cpu->pc : 0u, cpu ? cpu->lr : 0u);
    }
    const uint32_t canonicalDataAddr = CanonicalizeGxMainRamAddress(da);
    std::lock_guard<GuestThreadGuard> guard(g_texObjGuard); GXTexObj* obj = CreateHostTexObj(oa);
    TexObjMeta& meta = GetTexObjMetaForImageWrite(oa, canonicalDataAddr, (u16)w, (u16)h, f, m != 0);
    meta.dataAddr=canonicalDataAddr; meta.width=(u16)w; meta.height=(u16)h; meta.format=f; meta.wrapS=ws; meta.wrapT=wt; meta.mipmap=(m!=0); meta.userData=0; meta.needsUpload=true;
    GXInitTexObj(obj, GuestToHostPtr(da), (u16)w, (u16)h, (GXTexFmt)f, (GXTexWrapMode)ws, (GXTexWrapMode)wt, (GXBool)m); MarkHostTexObjConstructed(oa);
    // Also write to guest memory so reads work
    WriteGuestTexObj(oa, canonicalDataAddr, (u16)w, (u16)h, f, ws, wt, m != 0, false, 0);
}
PPC_NATIVE_OVERRIDE_VOID(801f28f0, GX__InitTexObj_HLE, (uint32_t oa, uint32_t da, uint32_t w, uint32_t h, uint32_t f, uint32_t ws, uint32_t wt, uint32_t m), (oa, da, w, h, f, ws, wt, m));

extern "C" void GX__InitTexObj_switch_80170938(uint32_t oa, uint32_t da, uint32_t w, uint32_t h, uint32_t f, uint32_t ws, uint32_t wt, uint32_t m) { GX__InitTexObj_HLE(oa, da, w, h, f, ws, wt, m); }

extern "C" void GX__InitTexObj_caseD_0_8017093c(uint32_t oa, uint32_t da, uint32_t w, uint32_t h, uint32_t f, uint32_t ws, uint32_t wt, uint32_t m) { GX__InitTexObj_HLE(oa, da, w, h, f, ws, wt, m); }

extern "C" void GX__InitTexObj_caseD_1_80170950(uint32_t oa, uint32_t da, uint32_t w, uint32_t h, uint32_t f, uint32_t ws, uint32_t wt, uint32_t m) { GX__InitTexObj_HLE(oa, da, w, h, f, ws, wt, m); }

extern "C" void GX__InitTexObj_caseD_3_80170964(uint32_t oa, uint32_t da, uint32_t w, uint32_t h, uint32_t f, uint32_t ws, uint32_t wt, uint32_t m) { GX__InitTexObj_HLE(oa, da, w, h, f, ws, wt, m); }

extern "C" void GX__InitTexObj_caseD_6_80170978(uint32_t oa, uint32_t da, uint32_t w, uint32_t h, uint32_t f, uint32_t ws, uint32_t wt, uint32_t m) { GX__InitTexObj_HLE(oa, da, w, h, f, ws, wt, m); }

extern "C" void GX__InitTexObj_caseD_e_8017098c(uint32_t oa, uint32_t da, uint32_t w, uint32_t h, uint32_t f, uint32_t ws, uint32_t wt, uint32_t m) { GX__InitTexObj_HLE(oa, da, w, h, f, ws, wt, m); }

extern "C" void GX__InitTexObj_caseD_7_801709a0(uint32_t oa, uint32_t da, uint32_t w, uint32_t h, uint32_t f, uint32_t ws, uint32_t wt, uint32_t m) { GX__InitTexObj_HLE(oa, da, w, h, f, ws, wt, m); }

// ============================================================================
// Texture Object Configuration
// ============================================================================

extern "C" void GX__InitTexObjCI_HLE(uint32_t oa, uint32_t da, uint32_t w, uint32_t h, uint32_t f, uint32_t ws, uint32_t wt, uint32_t m, uint32_t tl) {
    if (w == 0 || h == 0) {
        auto* cpu = TryGetCpuContext();
        RT_LOGF(RT_TAG_GX,
                "GXInitTexObjCI zero size oa=0x%08X da=0x%08X w=%u h=%u f=%u ws=%u wt=%u m=%u tl=%u pc=0x%08X lr=0x%08X\n",
                oa, da, w, h, f, ws, wt, m, tl, cpu ? cpu->pc : 0u, cpu ? cpu->lr : 0u);
    }
    const uint32_t canonicalDataAddr = CanonicalizeGxMainRamAddress(da);
    std::lock_guard<GuestThreadGuard> guard(g_texObjGuard); GXTexObj* obj = CreateHostTexObj(oa);
    TexObjMeta& meta = GetTexObjMetaForImageWrite(oa, canonicalDataAddr, (u16)w, (u16)h, f, m != 0);
    meta.dataAddr=canonicalDataAddr; meta.width=(u16)w; meta.height=(u16)h; meta.format=f; meta.wrapS=ws; meta.wrapT=wt; meta.mipmap=(m!=0); meta.tlut=tl; meta.userData=0; meta.needsUpload=true;
    GXInitTexObjCI(obj, GuestToHostPtr(da), (u16)w, (u16)h, (GXCITexFmt)f, (GXTexWrapMode)ws, (GXTexWrapMode)wt, (GXBool)m, tl); MarkHostTexObjConstructed(oa);
    // Also write to guest memory so reads work
    WriteGuestTexObj(oa, canonicalDataAddr, (u16)w, (u16)h, f, ws, wt, m != 0, true, tl);
    // GXInitTexObjCI clears bit1 in the flags byte; keep guest memory consistent.
    try {
        uint8_t flags = Memory::Read8(oa + 0x1F);
        Memory::Write8(oa + 0x1F, static_cast<uint8_t>(flags & 0xFD));
    } catch (...) {
    }
}
PPC_NATIVE_OVERRIDE_VOID(801f2afc, GX__InitTexObjCI_HLE, (uint32_t oa, uint32_t da, uint32_t w, uint32_t h, uint32_t f, uint32_t ws, uint32_t wt, uint32_t m, uint32_t tl), (oa, da, w, h, f, ws, wt, m, tl));

extern "C" void GX__InitTexObjLOD_HLE(uint32_t oa, uint32_t mif, uint32_t maf, float mil, float mal, float lb, uint32_t bc, uint32_t el, uint32_t ma) {
    std::lock_guard<GuestThreadGuard> guard(g_texObjGuard); GXTexObj* obj = GetHostTexObj(oa);
    float fmal = std::isfinite(mal) && mal >= 0.f ? mal : 0.f, fmil = std::isfinite(mil) && mil >= 0.f ? std::min(mil, fmal) : 0.f;
    TexObjMeta& meta = GetTexObjMetaForMaxLodWrite(oa, fmal);
    meta.minFilter=mif; meta.magFilter=maf; meta.minLod=fmil; meta.maxLod=fmal; meta.lodBias=lb; meta.biasClamp=(bc!=0); meta.edgeLod=(el!=0); meta.maxAniso=ma;
    GXInitTexObjLOD(obj, (GXTexFilter)mif, (GXTexFilter)maf, fmil, fmal, lb, (GXBool)bc, (GXBool)el, (GXAnisotropy)ma);
    // Also write LOD info to guest memory
    WriteGuestTexObjLOD(oa, mif, maf, fmil, fmal, lb, bc != 0, el != 0, ma);
}
PPC_NATIVE_OVERRIDE_VOID(801f2b44, GX__InitTexObjLOD_HLE, (uint32_t oa, uint32_t mif, uint32_t maf, float mil, float mal, float lb, uint32_t bc, uint32_t el, uint32_t ma), (oa, mif, maf, mil, mal, lb, bc, el, ma));

extern "C" void GX__InitTexObjWrapMode_HLE(uint32_t oa, uint32_t ws, uint32_t wt) {
    std::lock_guard<GuestThreadGuard> guard(g_texObjGuard);
    GXTexObj* obj = GetHostTexObj(oa);
    TexObjMeta& meta = GetTexObjMetaKeepingInterval(oa);
    meta.wrapS = ws;
    meta.wrapT = wt;
    try {
        uint32_t word0 = Memory::Read32(oa + 0x00);
        word0 = (word0 & ~0xFu) | (ws & 0x3u) | ((wt & 0x3u) << 2);
        Memory::Write32(oa + 0x00, word0);
    } catch (...) {
    }
    GXInitTexObjWrapMode(obj, (GXTexWrapMode)ws, (GXTexWrapMode)wt);
}
PPC_NATIVE_OVERRIDE_VOID(801f2c58, GX__InitTexObjWrapMode_HLE, (uint32_t oa, uint32_t ws, uint32_t wt), (oa, ws, wt));

extern "C" void GX__InitTexObjTlut_HLE(uint32_t oa, uint32_t tl) {
    std::lock_guard<GuestThreadGuard> guard(g_texObjGuard);
    GXTexObj* obj = GetHostTexObj(oa);
    TexObjMeta& meta = GetTexObjMetaKeepingInterval(oa);
    meta.tlut = tl;
    GXInitTexObjTlut(obj, tl);
    // Keep guest GXTexObj coherent for later GXLoadTexObj from memory.
    try {
        Memory::Write32(oa + 0x18, tl);
    } catch (...) {
    }
}
PPC_NATIVE_OVERRIDE_VOID(801f2c6c, GX__InitTexObjTlut_HLE, (uint32_t oa, uint32_t tl), (oa, tl));

// GXInitTexObjFilter - sets mag/min filter modes
extern "C" void GX__InitTexObjFilter_HLE(uint32_t oa, uint32_t minFilter, uint32_t magFilter) {
    std::lock_guard<GuestThreadGuard> guard(g_texObjGuard);
    TexObjMeta& meta = GetTexObjMetaKeepingInterval(oa);
    meta.minFilter = minFilter;
    meta.magFilter = magFilter;
    
    try {
        uint32_t word0 = Memory::Read32(oa + 0x00);
        // bit 4: magFilter - set if magFilter == GX_LINEAR (1), clear otherwise
        // Original code used cntlzw to check: cntlzw(magFilter-1) >> 1 & 0x10
        // This sets bit 4 when magFilter-1 is 0 (i.e., magFilter == 1)
        uint32_t magBit = (magFilter == 1) ? 0x10 : 0;
        word0 = (word0 & 0xFFFFFFEF) | magBit;
        // bits [7:5]: minFilter from LUT
        uint32_t minBits = (kGxToHwMinFilter[minFilter & 7] & 7) << 5;
        word0 = (word0 & 0xFFFFFF0F) | magBit | minBits;
        Memory::Write32(oa + 0x00, word0);
    } catch (...) {}

    // Update host texture object if it exists
    if (GXTexObj* obj = TryGetHostTexObj(oa)) {
        ApplyHostTexObjLod(obj, meta, (GXTexFilter)minFilter, (GXTexFilter)magFilter, meta.lodBias,
                           (GXAnisotropy)meta.maxAniso);
    }
}
PPC_NATIVE_OVERRIDE_VOID(801f2c74, GX__InitTexObjFilter_HLE, (uint32_t oa, uint32_t minFilter, uint32_t magFilter), (oa, minFilter, magFilter));

// GXInitTexObjLODBias - sets LOD bias value
extern "C" void GX__InitTexObjLODBias_80170b94(uint32_t oa, float bias) {
    std::lock_guard<GuestThreadGuard> guard(g_texObjGuard);
    TexObjMeta& meta = GetTexObjMetaKeepingInterval(oa);

    // Clamp bias to [-4.0, 3.99...] range as per SDK
    constexpr float kMinBias = -4.0f;
    constexpr float kMaxBias = 3.99f; // Actually ~3.990000009536743
    float clampedBias = bias;
    if (clampedBias < kMinBias) clampedBias = kMinBias;
    if (clampedBias >= 4.0f) clampedBias = kMaxBias;
    
    meta.lodBias = clampedBias;
    
    // Update guest GXTexObj word0 with bias (scale by 32, store in bits [16:9])
    try {
        uint32_t word0 = Memory::Read32(oa + 0x00);
        int8_t biasScaled = static_cast<int8_t>(clampedBias * 32.0f);
        word0 = (word0 & 0xFFFE01FF) | ((static_cast<uint32_t>(static_cast<uint8_t>(biasScaled)) & 0xFF) << 9);
        Memory::Write32(oa + 0x00, word0);
    } catch (...) {}
    
    // Update host texture object if it exists
    if (GXTexObj* obj = TryGetHostTexObj(oa)) {
        ApplyHostTexObjLod(obj, meta, (GXTexFilter)meta.minFilter, (GXTexFilter)meta.magFilter,
                           clampedBias, (GXAnisotropy)meta.maxAniso);
    }
}

extern "C" void GX__InitTexObjUserData_80170be8(uint32_t oa, uint32_t userData) {
    std::lock_guard<GuestThreadGuard> guard(g_texObjGuard);
    TexObjMeta& meta = GetTexObjMetaKeepingInterval(oa);
    meta.userData = userData;

    // The RVL SDK stores this opaque guest pointer verbatim at GXTexObj + 0x10.
    WriteGuest32(oa + 0x10, userData, "GXInitTexObjUserData");

    // Aurora keeps an expanded host-side object, so mirror the pointer when
    // that representation has already been constructed.
    if (GXTexObj* obj = TryGetHostTexObj(oa)) {
        GXInitTexObjUserData(obj, GuestToHostPtr(userData));
    }
}

// ============================================================================
// Texture Loading
// ============================================================================

// A refused GXLoadTexObj must not leave the previous material's texture bound in that TEV
// slot (rejection paths used to just return, so draws sampled stale, unrelated art). Bind
// aurora's empty texture instead: null image data resolves to a 1x1 transparent placeholder,
// which is also the correct result for the common case of an intentionally-transparent hide texture.
// One per producer thread (the game thread's; a draw worker's for a deferred draw's rejected load, RunTexLoadPlan):
// Aurora's GXLoadTexObj writes the map's register ids into the object it loads.
static GXTexObj* UnloadableTexturePlaceholder() {
    static thread_local GXTexObj s_placeholder{};
    static thread_local bool s_placeholderReady = false;
    if (!s_placeholderReady) {
        GXInitTexObj(&s_placeholder, nullptr, 1, 1, GX_TF_RGBA8, GX_CLAMP, GX_CLAMP, GX_FALSE);
        s_placeholderReady = true;
    }
    return &s_placeholder;
}

static void BindUnloadableTexturePlaceholder(uint32_t tid) {
    if (tid >= g_boundTexMaps.size()) {
        return;
    }
    // A zeroed binding record can never compare equal to a real load (those
    // always carry a non-zero objAddr), so the CanSkipHostLoad fast path cannot
    // mistake a later genuine bind for a no-op.
    g_boundTexMaps[tid] = BoundTexInfo{};
    GXLoadTexObj(UnloadableTexturePlaceholder(), (GXTexMapID)tid);
}

// One host texture object per (image, sampler) for loads recorded into display lists; never freed
// (lists are replayed for as long as the map is loaded, and the set is small).
static GXTexObj* PersistentDlTexObj(const TexObjMeta& meta, GXTexWrapMode wrapS, GXTexWrapMode wrapT,
                                    GXTexFilter minFilter, GXTexFilter magFilter, GXAnisotropy aniso, uint32_t size) {
    static std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>, std::unique_ptr<GXTexObj>> objs;
    const auto key = std::make_tuple(CanonicalizeGxMainRamAddress(meta.dataAddr),
                                     (static_cast<uint32_t>(meta.width) << 16) | meta.height, meta.format,
                                     (static_cast<uint32_t>(wrapS) << 8) | wrapT,
                                     (static_cast<uint32_t>(minFilter) << 8) | magFilter, meta.mipmap ? 1u : 0u);
    auto& slot = objs[key];
    if (!slot) {
        slot = std::make_unique<GXTexObj>();
        GXInitTexObj(slot.get(), GuestToHostPtr(meta.dataAddr, size), meta.width, meta.height,
                     (GXTexFmt)meta.format, wrapS, wrapT, meta.mipmap ? GX_TRUE : GX_FALSE);
        ApplyHostTexObjLod(slot.get(), meta, minFilter, magFilter, meta.lodBias, aniso);
    }
    return slot.get();
}

// GXLoadTexObj in two halves. The table half (PlanTexLoad: the guest GXTexObj, the texture tables and host objects,
// under g_texObjGuard, so on the guest thread) works out what the load does; the per-thread half (RunTexLoadPlan: this
// thread's map bindings and Aurora register shadow) does it. Inline, both run here. A deferred native draw
// (g3d_native.cpp) plans on the game thread when it records the load and runs the plan on the draw worker, with a
// copy of the host object (GX_HLE_PlanTexLoadForJob / GX_HLE_RunTexLoadPlan below).
struct TexLoadPlan {
    enum Kind : uint8_t { kNothing, kPlaceholder, kLoad };
    Kind kind = kNothing;  // kNothing: an invalid map id; kPlaceholder: a rejected load
    bool needsInit = false;
    bool tlutChanged = false;
    bool textureDataUploaded = false;
    bool isPalette = false;
    bool spmPersistent = false;  // SPM display-list recording: loads a PersistentDlTexObj, always
    TexObjMeta meta{};           // as loaded (sanitized)
    BoundTexInfo bound{};        // the map's binding after the load
    GXTexObj* obj = nullptr;     // the host object to load
};
static uint32_t s_texLoadExceptionLogCount = 0;

// Aurora digests a loaded object's bytes under a write generation it takes on the GX thread (or a draw worker's job):
// their pages are marked on this (the guest) thread, before the hand-off (gx_guest_write.h g_trackedPages). The whole
// mip chain bounds what any LOD setting reads. Marks are sticky, so an image (data, size, format) marked before is
// skipped: a direct-mapped memo of marked images, filled only where the mark happened. Guest thread (PlanTexLoad).
static void NoteTexObjTracked(GXTexObj* obj) {
    const void* data = GXGetTexObjData(obj);
    if (data == nullptr) return;
    const uint16_t w = GXGetTexObjWidth(obj);
    const uint16_t h = GXGetTexObjHeight(obj);
    const uint32_t fmt = static_cast<uint32_t>(GXGetTexObjFmt(obj));
    struct Marked {
        const void* data;
        uint64_t image;
    };
    static std::array<Marked, 512> s_marked{};
    const uint64_t image = uint64_t{w} | uint64_t{h} << 16 | uint64_t{fmt} << 32;
    Marked& m = s_marked[((reinterpret_cast<uintptr_t>(data) >> 5) * 0x9E3779B97F4A7C15ull) >> 55];
    if (m.data == data && m.image == image) return;
    if (GxGuestWrite::NoteTrackedHostRange(data, GXGetTexBufferSize(w, h, fmt, GX_TRUE, 255))) m = {data, image};
}

static TexLoadPlan PlanTexLoadFull(uint32_t oa, uint32_t tid) {
    static uint32_t s_invalidMetaLogCount = 0;
    static uint32_t s_invalidTidLogCount = 0;
    static uint32_t s_invalidDimLogCount = 0;
    uint32_t& s_hostExceptionLogCount = s_texLoadExceptionLogCount;
    static uint32_t s_metaLookupLogCount = 0;
    static uint32_t s_unknownFormatLogCount = 0;
    static uint32_t s_invalidDataLogCount = 0;
    TexLoadPlan plan;
    std::lock_guard<GuestThreadGuard> guard(g_texObjGuard);

    if (tid >= g_boundTexMaps.size()) {
        if (s_invalidTidLogCount < 128) {
            RT_LOGF(RT_TAG_GX, "GXLoadTexObj: invalid tex map id %u (oa=0x%08X)\n", tid, oa);
            ++s_invalidTidLogCount;
        }
        return plan;
    }
    
    // Check if we have metadata for this texture, or try to extract it from guest memory
    TexObjMeta meta;
    if (!TryGetOrExtractTexObjMeta(oa, meta)) {
        if (s_metaLookupLogCount++ < 64) {
            RT_LOGF(RT_TAG_GX,
                    "GXLoadTexObj rejected reason=no-metadata oa=0x%08X tid=%u\n", oa, tid);
        }
        plan.kind = TexLoadPlan::kPlaceholder;
        return plan;
    }

    if (meta.dataAddr == 0 || meta.width == 0 || meta.height == 0) {
        if (s_invalidMetaLogCount++ < 64) {
            RT_LOGF(RT_TAG_GX,
                    "GXLoadTexObj rejected reason=invalid-meta oa=0x%08X tid=%u data=0x%08X %ux%u\n",
                    oa, tid, meta.dataAddr, meta.width, meta.height);
        }
        plan.kind = TexLoadPlan::kPlaceholder;
        return plan;
    }
    if (!IsReasonableTextureDimensions(meta.width, meta.height)) {
        if (s_invalidDimLogCount++ < 128) {
            RT_LOGF(RT_TAG_GX,
                    "GXLoadTexObj rejected reason=bad-dimensions oa=0x%08X tid=%u %ux%u fmt=0x%X\n",
                    oa, tid, meta.width, meta.height, meta.format);
        }
        plan.kind = TexLoadPlan::kPlaceholder;
        return plan;
    }
    // One gate, two reasons: every aurora-loadable format is also a known one,
    // so a separate IsKnownTexFormat rejection could never fire on anything this
    // check would let through. The reason string still distinguishes "GX has no
    // such format" from "aurora cannot decode it", which is what triage needs.
    if (!IsAuroraLoadableTexFormat(meta.format)) {
        const bool known = IsKnownTexFormat(meta.format);
        uint32_t& logCount = known ? g_unsupportedFormatLogCount : s_unknownFormatLogCount;
        if (logCount++ < 128) {
            RT_LOGF(RT_TAG_GX,
                    "GXLoadTexObj rejected reason=%s oa=0x%08X tid=%u fmt=0x%X %ux%u\n",
                    known ? "unsupported-format" : "unknown-format", oa, tid, meta.format,
                    meta.width, meta.height);
        }
        plan.kind = TexLoadPlan::kPlaceholder;
        return plan;
    }
    const float maxMipLevel = static_cast<float>(ComputeMaxMipLevel(meta.width, meta.height));
    const float minLodSafe = SanitizeLodValue(meta.minLod, 0.0f);
    float maxLodSafe = SanitizeLodValue(meta.maxLod, minLodSafe);
    maxLodSafe = std::min(maxLodSafe, maxMipLevel);
    const float clampedMinLodSafe = std::min(minLodSafe, maxLodSafe);
    const float lodBiasSafe = SanitizeLodBias(meta.lodBias);
    const uint8_t maxLod = (maxLodSafe > 0.0f) ? ((maxLodSafe > 255.0f) ? 255u : static_cast<uint8_t>(maxLodSafe)) : 0u;
    const uint32_t size = CachedTexBufferSize(meta.width, meta.height, meta.format, (GXBool)meta.mipmap, maxLod);
    if (size == 0 || !Memory::Contains(meta.dataAddr, size)) {
        if (s_invalidDataLogCount++ < 64) {
            auto* cpu = TryGetCpuContext();
            RT_LOGF(RT_TAG_GX,
                    "GXLoadTexObj rejected reason=data-out-of-range "
                    "oa=0x%08X tid=%u data=0x%08X size=0x%X %ux%u fmt=0x%X mip=%u maxLod=%u pc=0x%08X lr=0x%08X\n",
                    oa, tid, meta.dataAddr, size, meta.width, meta.height, meta.format,
                    meta.mipmap ? 1u : 0u, static_cast<uint32_t>(maxLod),
                    cpu ? cpu->pc : 0u, cpu ? cpu->lr : 0u);
        }
        plan.kind = TexLoadPlan::kPlaceholder;
        return plan;
    }
    const GXTexWrapMode wrapS = SanitizeWrapMode(meta.wrapS);
    const GXTexWrapMode wrapT = SanitizeWrapMode(meta.wrapT);
    const GXTexFilter minFilter = SanitizeMinFilter(meta.minFilter);
    const GXTexFilter magFilter = SanitizeMagFilter(meta.magFilter);
    const GXAnisotropy maxAnisoSafe = SanitizeAniso(meta.maxAniso);
    meta.wrapS = static_cast<uint32_t>(wrapS);
    meta.wrapT = static_cast<uint32_t>(wrapT);
    meta.minFilter = static_cast<uint32_t>(minFilter);
    meta.magFilter = static_cast<uint32_t>(magFilter);
    meta.maxAniso = static_cast<uint32_t>(maxAnisoSafe);
    meta.minLod = clampedMinLodSafe;
    meta.maxLod = maxLodSafe;
    meta.lodBias = lodBiasSafe;
    
    // Check if host texture object exists, create if needed
    GXTexObj* obj = TryGetHostTexObj(oa);
    const bool isPalette = IsPaletteTexFormat(meta.format);
    bool needsInit = (obj == nullptr);
    // The object's slot, as TryGetHostTexObj just found it (FindTexObjSlot's one-entry memo, not another hash lookup;
    // slots are never erased).
    const TexObjMeta* cachedMeta = FindTexObjSlotForPlan(oa);
    if (!needsInit && cachedMeta != nullptr) {
        const TexObjMeta& cached = *cachedMeta;
        if (cached.width != meta.width || cached.height != meta.height || cached.format != meta.format ||
            cached.wrapS != meta.wrapS || cached.wrapT != meta.wrapT || cached.mipmap != meta.mipmap ||
            cached.minFilter != meta.minFilter || cached.magFilter != meta.magFilter ||
            cached.minLod != meta.minLod || cached.maxLod != meta.maxLod ||
            cached.lodBias != meta.lodBias || cached.biasClamp != meta.biasClamp ||
            cached.edgeLod != meta.edgeLod || cached.maxAniso != meta.maxAniso ||
            cached.tlut != meta.tlut || cached.userData != meta.userData) {
            needsInit = true;
        }
    }
    // The host twin must describe the same image as the guest struct; one that drifted (built blank
    // by a wrap/LOD call before any GXInitTexObj) is rebuilt rather than patched with a data pointer.
    if (!needsInit && obj != nullptr &&
        (GXGetTexObjWidth(obj) != meta.width || GXGetTexObjHeight(obj) != meta.height ||
         static_cast<uint32_t>(GXGetTexObjFmt(obj)) != meta.format)) {
        needsInit = true;
    }
    bool tlutChanged = false;
    bool textureDataUploaded = false;
    try {
        if (needsInit) {
            if (!obj) {
                obj = CreateHostTexObj(oa);
            }
            void* dp = GuestToHostPtr(meta.dataAddr, size);
            if (dp) {
                if (isPalette) {
                    GXInitTexObjCI(obj, dp, meta.width, meta.height, (GXCITexFmt)meta.format,
                                   wrapS, wrapT,
                                   meta.mipmap ? GX_TRUE : GX_FALSE, meta.tlut);
                } else {
                    GXInitTexObj(obj, dp, meta.width, meta.height, (GXTexFmt)meta.format,
                                 wrapS, wrapT,
                                 meta.mipmap ? GX_TRUE : GX_FALSE);
                }
                ApplyHostTexObjLod(obj, meta, minFilter, magFilter, meta.lodBias, maxAnisoSafe);
                GXInitTexObjUserData(obj, GuestToHostPtr(meta.userData));
            }
            MarkHostTexObjConstructed(oa);
            // Write through GetTexObjMetaForWrite so the DCStoreRange interval index is
            // told this entry's backing may have moved.
            GetTexObjMetaForWrite(oa, meta) = meta;
        } else if (isPalette && cachedMeta != nullptr && cachedMeta->tlut != meta.tlut) {
            GXInitTexObjTlut(obj, meta.tlut);
            GetTexObjMetaKeepingInterval(oa).tlut = meta.tlut;
            tlutChanged = true;
        }

        const void* dp = GuestToHostPtr(meta.dataAddr, size);
        if (dp && meta.needsUpload) {
            GXInitTexObjData(obj, dp);
            meta.needsUpload = false;
            textureDataUploaded = true;
        }
        // Everything the binding contract compares apart from objAddr is the
        // shared sampler state, so copy that base wholesale; only dataAddr needs
        // to be canonicalized on the way in.
        BoundTexInfo newBound{};
        static_cast<GxTextureBindingContract::SamplerState&>(newBound) = meta;
        newBound.objAddr = oa;
        newBound.dataAddr = CanonicalizeGxMainRamAddress(meta.dataAddr);
        GetTexObjMetaForWrite(oa, meta) = meta;
        plan.needsInit = needsInit;
        plan.tlutChanged = tlutChanged;
        plan.textureDataUploaded = textureDataUploaded;
        plan.isPalette = isPalette;
        plan.meta = meta;
        plan.bound = newBound;
        // SPM builds map materials into display lists from a stack GXTexObj reused for every
        // material, so the shared host twin (keyed by that stack address) would give every recorded
        // load the same texture identity. Recorded loads get a permanent host object per image.
        plan.spmPersistent = SpmAuroraDlRecording() && !isPalette;
        plan.obj = plan.spmPersistent
                       ? PersistentDlTexObj(meta, wrapS, wrapT, minFilter, magFilter, maxAnisoSafe, size)
                       : obj;
        plan.kind = TexLoadPlan::kLoad;
        NoteTexObjTracked(plan.obj);
    } catch (const std::exception& ex) {
        if (s_hostExceptionLogCount++ < 64) {
            RT_LOGF(RT_TAG_GX,
                    "GXLoadTexObj rejected reason=host-exception (%s) oa=0x%08X tid=%u fmt=0x%X %ux%u data=0x%08X\n",
                    ex.what(), oa, tid, meta.format, meta.width, meta.height, meta.dataAddr);
        }
        plan.kind = TexLoadPlan::kPlaceholder;
        return plan;
    } catch (...) {
        if (s_hostExceptionLogCount++ < 64) {
            RT_LOGF(RT_TAG_GX,
                    "GXLoadTexObj rejected reason=host-exception-unknown oa=0x%08X tid=%u fmt=0x%X %ux%u data=0x%08X\n",
                    oa, tid, meta.format, meta.width, meta.height, meta.dataAddr);
        }
        plan.kind = TexLoadPlan::kPlaceholder;
        return plan;
    }
    return plan;
}

// Repeated loads. A plan that changed nothing it read - the object's slot metadata, the guest struct's bytes (equal to
// the slot's shadow, so the metadata lookup took its fast path) and the host object, each the same before and after -
// is a function of those and of what PlanTexLoad checks below (the map id's range, SPM's list recording; the format,
// size and address checks and the host pointer follow from the metadata). A later load of the same object finding the
// same three gets the same plan again, and the writes the full path would make are of values already there: the
// slot's metadata and guest shadow, an interval patch with nothing to move, tracked pages already marked (sticky).
// So it returns the remembered plan. Vanilla Training: 125 of 134 deferred texture loads a frame are such repeats
// (the same materials every frame; the other 9, objects re-initialized that frame). Guest thread, under
// g_texObjGuard, like the table it reads.
//
// Each object's plan is remembered on its own slot (TexObjSlot::planMemo), so no object evicts another's: a
// direct-mapped table of 256 entries remembered only 72 of the 134 (the rest lost to collisions at half load).
struct GxTexPlanMemo {
    bool valid = false;
    uint64_t shadow[4] = {};  // the guest struct's bytes (and the slot's shadow)
    TexObjMeta meta{};        // the slot's metadata
    GXTexObj host{};          // the host object's bytes
    TexLoadPlan plan{};
};
namespace {
// GX_TEXPLAN_CHECK=1: a remembered plan is also made the full way and compared (the plan, and the inputs it leaves).
const bool s_texPlanCheck = [] {
    const char* v = std::getenv("GX_TEXPLAN_CHECK");
    return v != nullptr && v[0] == '1';
}();
uint64_t s_texPlanJobs = 0, s_texPlanRemembered = 0, s_texPlanChecked = 0, s_texPlanCheckBad = 0;  // guest thread

// The guest struct's bytes as gx_objects.cpp's ReadTexObjShadow reads them (the slot's guestShadow form).
bool ReadGuestTexObjBytes(uint32_t addr, uint64_t (&out)[4]) noexcept {
    try {
        out[0] = Memory::Read64(addr + 0x00);
        out[1] = Memory::Read64(addr + 0x08);
        out[2] = Memory::Read64(addr + 0x10);
        out[3] = Memory::Read64(addr + 0x18);
    } catch (...) {
        return false;
    }
    return true;
}
bool SameWords(const uint64_t (&a)[4], const uint64_t (&b)[4]) {
    return ((a[0] ^ b[0]) | (a[1] ^ b[1]) | (a[2] ^ b[2]) | (a[3] ^ b[3])) == 0;
}
// Every field, floats by their bits (a NaN or a signed zero must not compare as another value).
bool SameSamplerBits(const GxTextureBindingContract::SamplerState& a, const GxTextureBindingContract::SamplerState& b) {
    const auto bits = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        return u;
    };
    return a.dataAddr == b.dataAddr && a.width == b.width && a.height == b.height && a.format == b.format &&
           a.wrapS == b.wrapS && a.wrapT == b.wrapT && a.tlut == b.tlut && a.minFilter == b.minFilter &&
           a.magFilter == b.magFilter && bits(a.minLod) == bits(b.minLod) && bits(a.maxLod) == bits(b.maxLod) &&
           bits(a.lodBias) == bits(b.lodBias) && a.maxAniso == b.maxAniso && a.biasClamp == b.biasClamp &&
           a.edgeLod == b.edgeLod && a.mipmap == b.mipmap;
}
bool SameMetaBits(const TexObjMeta& a, const TexObjMeta& b) {
    return SameSamplerBits(a, b) && a.userData == b.userData && a.needsUpload == b.needsUpload;
}
bool SamePlanBits(const TexLoadPlan& a, const TexLoadPlan& b) {  // check mode
    return a.kind == b.kind && a.needsInit == b.needsInit && a.tlutChanged == b.tlutChanged &&
           a.textureDataUploaded == b.textureDataUploaded && a.isPalette == b.isPalette &&
           a.spmPersistent == b.spmPersistent && SameMetaBits(a.meta, b.meta) && SameSamplerBits(a.bound, b.bound) &&
           a.bound.objAddr == b.bound.objAddr && a.obj == b.obj;
}
// What a plan reads of the object, taken before and after it.
struct TexPlanInputs {
    bool valid = false;  // a slot with a constructed host object, the guest bytes equal to its shadow
    uint64_t shadow[4] = {};
    TexObjMeta meta{};
    GXTexObj host{};
    GXTexObj* hostPtr = nullptr;
};
TexPlanInputs TakeTexPlanInputs(const TexObjSlot* slot, uint32_t oa) {
    TexPlanInputs in;
    uint64_t guest[4];
    if (slot == nullptr || !slot->hasGuestShadow || !slot->host || !slot->host->constructed ||
        !ReadGuestTexObjBytes(oa, guest) || !SameWords(guest, slot->guestShadow)) {
        return in;
    }
    in.valid = true;
    std::memcpy(in.shadow, guest, sizeof guest);
    in.meta = *slot;
    in.hostPtr = slot->host->PublicPtr();
    std::memcpy(&in.host, in.hostPtr, sizeof(GXTexObj));
    return in;
}
}  // namespace

// The SDK's GXLoadTexObj(PreLoaded) puts the map's BP register ids into the top bytes of the guest object's mode0,
// mode1, image0 and image3 words (Brawl 0x801F2E4C: rlwimi ..., 24, 0, 7, then stored back), as the words it sends.
// Code that finds a loaded object by those words sees them: Brawl Online's plugin points the room CSS state models'
// texture objects (ScnMdl material buffers) at its relabelled copies by matching image3 = 0x94-0x97/0xB4-0xB7 | image
// (without them the recomp kept Brawl's own "Selecting Character..." and "Selecting Stage..." panels). Done before the
// plan, so the slot's guest shadow and the plan memo hold the stamped bytes and repeated loads stay on the fast paths;
// written only where a byte changes. Every other writer of these words keeps their top bytes.
static void StampTexObjRegisterIds(uint32_t oa, uint32_t tid) {
    static constexpr uint8_t kMode0[8] = {0x80, 0x81, 0x82, 0x83, 0xA0, 0xA1, 0xA2, 0xA3};
    static constexpr uint8_t kMode1[8] = {0x84, 0x85, 0x86, 0x87, 0xA4, 0xA5, 0xA6, 0xA7};
    static constexpr uint8_t kImage0[8] = {0x88, 0x89, 0x8A, 0x8B, 0xA8, 0xA9, 0xAA, 0xAB};
    static constexpr uint8_t kImage3[8] = {0x94, 0x95, 0x96, 0x97, 0xB4, 0xB5, 0xB6, 0xB7};
    if (oa == 0 || tid >= 8) return;
    try {
        const struct { uint32_t off; uint8_t id; } words[4] = {
            {0x00, kMode0[tid]}, {0x04, kMode1[tid]}, {0x08, kImage0[tid]}, {0x0C, kImage3[tid]}};
        for (const auto& w : words) {
            if (Memory::Read8(oa + w.off) != w.id) {
                if (g3dn::g_recOn && g3dn::g_recSuppress && !g3dn::g_recJob)
                    g3dn::SaveVerifyGuestByte(oa + w.off);
                Memory::Write8(oa + w.off, w.id);
            }
        }
    } catch (...) {
    }
}

// remember: a deferred draw's load (GX_HLE_PlanTexLoadForJob). An inline load (GX__LoadTexObj_HLE) never repeats so:
// its GXLoadTexObj writes the map's register ids into the host object, so its next plan finds other host bytes.
static TexLoadPlan PlanTexLoad(uint32_t oa, uint32_t tid, bool remember = false) {
    if (!remember) return PlanTexLoadFull(oa, tid);
    std::lock_guard<GuestThreadGuard> guard(g_texObjGuard);
    ++s_texPlanJobs;
    if (tid >= g_boundTexMaps.size() || SpmAuroraDlRecording()) return PlanTexLoadFull(oa, tid);
    const TexObjSlot* slot = FindTexObjSlotForPlan(oa);
    GxTexPlanMemo* memo = slot != nullptr ? slot->planMemo : nullptr;
    const TexPlanInputs before = TakeTexPlanInputs(slot, oa);
    if (before.valid && memo != nullptr && memo->valid && before.hostPtr == memo->plan.obj &&
        SameWords(before.shadow, memo->shadow) && SameMetaBits(before.meta, memo->meta) &&
        std::memcmp(&before.host, &memo->host, sizeof(GXTexObj)) == 0) {
        ++s_texPlanRemembered;
        if (s_texPlanCheck) {
            const TexLoadPlan full = PlanTexLoadFull(oa, tid);
            const TexObjSlot* after = FindTexObjSlotForPlan(oa);
            const TexPlanInputs left = TakeTexPlanInputs(after, oa);
            const bool same = SamePlanBits(full, memo->plan) && after == slot && left.valid &&
                              left.hostPtr == before.hostPtr && SameWords(left.shadow, before.shadow) &&
                              SameMetaBits(left.meta, before.meta) &&
                              std::memcmp(&left.host, &before.host, sizeof(GXTexObj)) == 0;
            ++s_texPlanChecked;
            if (!same && s_texPlanCheckBad++ < 20) {
                RT_LOGF(RT_TAG_GX, "[texplan] remembered plan MISMATCH oa=0x%08X tid=%u\n", oa, tid);
            }
        }
        return memo->plan;
    }
    TexLoadPlan plan = PlanTexLoadFull(oa, tid);
    if (memo != nullptr) memo->valid = false;
    if (before.valid && plan.kind == TexLoadPlan::kLoad && !plan.needsInit && !plan.tlutChanged &&
        !plan.textureDataUploaded && !plan.spmPersistent && plan.obj == before.hostPtr) {
        const TexObjSlot* after = FindTexObjSlotForPlan(oa);
        const TexPlanInputs left = TakeTexPlanInputs(after, oa);
        if (after == slot && left.valid && left.hostPtr == before.hostPtr && SameWords(left.shadow, before.shadow) &&
            SameMetaBits(left.meta, before.meta) && std::memcmp(&left.host, &before.host, sizeof(GXTexObj)) == 0) {
            if (slot->planMemo == nullptr) slot->planMemo = new GxTexPlanMemo;
            memo = slot->planMemo;
            memo->valid = true;
            std::memcpy(memo->shadow, before.shadow, sizeof memo->shadow);
            memo->meta = before.meta;
            memo->host = before.host;
            memo->plan = plan;
        }
    }
    return plan;
}

// The per-thread half: loads obj (the plan's host object, or a copy of it) into map tid unless this thread's binding of
// the map makes it redundant. False where the load ends without its guest writes: nothing to do, or the placeholder
// bound (a rejected load, or a host exception here).
static bool RunTexLoadPlan(const TexLoadPlan& plan, uint32_t oa, uint32_t tid, GXTexObj* obj) {
    if (plan.kind == TexLoadPlan::kNothing) return false;
    if (plan.kind == TexLoadPlan::kPlaceholder) {
        BindUnloadableTexturePlaceholder(tid);
        return false;
    }
    const TexObjMeta& meta = plan.meta;
    const bool isPalette = plan.isPalette;
    try {
        const BoundTexInfo& oldBound = g_boundTexMaps[tid];
        const bool canSkipHostLoad = GxTextureBindingContract::CanSkipHostLoad(
            oldBound, plan.bound, plan.needsInit, plan.tlutChanged, plan.textureDataUploaded);

        g_boundTexMaps[tid] = plan.bound;
        if (SpmDumpFrame()) {
            RT_LOGF(RT_TAG_GX, "[drawdump] loadtex map %u oa 0x%08X data 0x%08X %ux%u fmt 0x%X pal %d tlut %u skip %d init %d up %d\n", tid, oa,
                    meta.dataAddr, meta.width, meta.height, meta.format, isPalette ? 1 : 0, meta.tlut, canSkipHostLoad ? 1 : 0,
                    plan.needsInit ? 1 : 0, plan.textureDataUploaded ? 1 : 0);
        }
        if (SpmAuroraDlRecording()) {
            static int s_texLog = 0;
            if (s_texLog++ < 60) RT_LOGF(RT_TAG_GX, "[spmdl] recorded tex load map %u data 0x%08X %ux%u fmt 0x%X pal %d\n", tid, meta.dataAddr, meta.width, meta.height, meta.format, isPalette ? 1 : 0);
        }
        if (plan.spmPersistent || !canSkipHostLoad) {
            GXLoadTexObj(obj, (GXTexMapID)tid);
        }
    } catch (const std::exception& ex) {
        if (s_texLoadExceptionLogCount++ < 64) {
            RT_LOGF(RT_TAG_GX,
                    "GXLoadTexObj rejected reason=host-exception (%s) oa=0x%08X tid=%u fmt=0x%X %ux%u data=0x%08X\n",
                    ex.what(), oa, tid, meta.format, meta.width, meta.height, meta.dataAddr);
        }
        BindUnloadableTexturePlaceholder(tid);
        return false;
    } catch (...) {
        if (s_texLoadExceptionLogCount++ < 64) {
            RT_LOGF(RT_TAG_GX,
                    "GXLoadTexObj rejected reason=host-exception-unknown oa=0x%08X tid=%u fmt=0x%X %ux%u data=0x%08X\n",
                    oa, tid, meta.format, meta.width, meta.height, meta.dataAddr);
        }
        BindUnloadableTexturePlaceholder(tid);
        return false;
    }
    return true;
}

static void WriteTexLoadGuestEffects() {
    try { uint32_t gd = Memory::Read32(kGXDataPtrAddr); if (gd) { Memory::Write32(gd + 0x5FCu, Memory::Read32(gd + 0x5FCu) | 1u); Memory::Write16(gd + 2, 0); } } catch (...) {}
}

extern "C" void GX__LoadTexObj_HLE(uint32_t oa, uint32_t tid) {
    // These are guest-visible SDK effects, including during differential recording.
    StampTexObjRegisterIds(oa, tid);
    G3D_REC_HLE(g3dn::kRecLoadTexObj, oa, tid);
    const TexLoadPlan plan = PlanTexLoad(oa, tid);
    if (!RunTexLoadPlan(plan, oa, tid, plan.obj)) return;
    WriteTexLoadGuestEffects();
}
PPC_NATIVE_OVERRIDE_VOID(801f2fb0, GX__LoadTexObj_HLE, (uint32_t oa, uint32_t tid), (oa, tid));

// A deferred native draw's texture load (g3d_native.cpp), as GX__LoadTexObj_HLE in two places. The game thread, when it
// records the load: the table half, the guest writes (where the inline load makes them: a load that reaches Aurora; a
// host exception in the worker's GXLoadTexObj, which skips them inline, is not foreseen here, none expected), and
// Aurora's register shadow of the map as that GXLoadTexObj leaves it (this thread's SU flush reads it); the plan
// carries a copy of the host object (the game thread may rebuild the object before the worker loads it). The draw
// worker: the per-thread half with that copy.
extern "C" void GXShadowSetTexMap(const GXTexObj* obj, uint32_t map);  // aurora GXTexture.cpp
struct TexLoadJobPlan {
    TexLoadPlan plan;
    GXTexObj obj;
};
extern "C" size_t GX_HLE_TexLoadPlanBytes() { return sizeof(TexLoadJobPlan); }
extern "C" void GX_HLE_PlanTexLoadForJob(uint32_t oa, uint32_t tid, void* out) {
    StampTexObjRegisterIds(oa, tid);
    TexLoadJobPlan job{};
    job.plan = PlanTexLoad(oa, tid, /*remember=*/true);
    if (job.plan.kind == TexLoadPlan::kLoad) {
        job.obj = *job.plan.obj;
        GXShadowSetTexMap(&job.obj, tid);
        WriteTexLoadGuestEffects();
    } else if (job.plan.kind == TexLoadPlan::kPlaceholder) {
        GXShadowSetTexMap(UnloadableTexturePlaceholder(), tid);
    }
    job.plan.obj = nullptr;
    std::memcpy(out, &job, sizeof job);
}
extern "C" void GX_HLE_RunTexLoadPlan(const void* in, uint32_t oa, uint32_t tid) {
    TexLoadJobPlan job;
    std::memcpy(&job, in, sizeof job);
    RunTexLoadPlan(job.plan, oa, tid, &job.obj);
}

// The stats line (g3d_native.cpp G3dNativeLogStats): per frame. Guest thread.
extern "C" void GX_HLE_LogTexPlanStats(double frames) {
    if (frames <= 0 || s_texPlanJobs == 0) return;
    RT_LOGF(RT_TAG_VI, "gx texture plans (deferred draws): %.1f/frame, remembered %.1f/frame, full %.1f/frame; "
            "checked %.1f/frame (GX_TEXPLAN_CHECK=1), %llu mismatches\n",
            s_texPlanJobs / frames, s_texPlanRemembered / frames, (s_texPlanJobs - s_texPlanRemembered) / frames,
            s_texPlanChecked / frames, static_cast<unsigned long long>(s_texPlanCheckBad));
    s_texPlanJobs = s_texPlanRemembered = s_texPlanChecked = 0;
}

// GXLoadTexObjPreLoaded(obj, region, id): the map id is the THIRD argument; the region (r4) only
// describes TMEM placement, which Aurora does not model.
extern "C" void GX__LoadTexObjPreLoaded_HLE(uint32_t oa, uint32_t /*region*/, uint32_t tid) { GX__LoadTexObj_HLE(oa, tid); }
PPC_NATIVE_OVERRIDE_VOID(801f2e4c, GX__LoadTexObjPreLoaded_HLE, (uint32_t oa, uint32_t region, uint32_t tid), (oa, region, tid));

// ============================================================================
// Texture Object Getters
// ============================================================================

extern "C" void GX__GetTexObjAll_80170bf8(uint32_t oa, uint32_t dp, uint32_t wp, uint32_t hp, uint32_t fp, uint32_t wsp, uint32_t wtp, uint32_t mp) {
    std::lock_guard<GuestThreadGuard> guard(g_texObjGuard);
    TexObjMeta meta;
    if (!TryGetOrExtractTexObjMeta(oa, meta)) {
        RT_LOGF(RT_TAG_GX, "GXGetTexObjAll: failed to get metadata for GXTexObj @0x%08X\n", oa);
        // Return zeros on failure
        if (dp) Memory::Write32(dp, 0); if (wp) Memory::Write16(wp, 0); if (hp) Memory::Write16(hp, 0); 
        if (fp) Memory::Write32(fp, 0); if (wsp) Memory::Write32(wsp, 0); if (wtp) Memory::Write32(wtp, 0); if (mp) Memory::Write8(mp, 0);
        return;
    }
    if (dp) Memory::Write32(dp, meta.dataAddr); if (wp) Memory::Write16(wp, meta.width); if (hp) Memory::Write16(hp, meta.height); if (fp) Memory::Write32(fp, meta.format); if (wsp) Memory::Write32(wsp, meta.wrapS); if (wtp) Memory::Write32(wtp, meta.wrapT); if (mp) Memory::Write8(mp, meta.mipmap ? 1 : 0);
}

extern "C" void GX__GetTexObjLODAll_80170cbc(uint32_t oa, uint32_t mifp, uint32_t mafp, uint32_t milp, uint32_t malp, uint32_t lbp, uint32_t bcp, uint32_t elp, uint32_t map) {
    std::lock_guard<GuestThreadGuard> guard(g_texObjGuard);
    TexObjMeta meta;
    if (!TryGetOrExtractTexObjMeta(oa, meta)) {
        RT_LOGF(RT_TAG_GX, "GXGetTexObjLODAll: failed to get metadata for GXTexObj @0x%08X\n", oa);
        // Return zeros on failure
        if (mifp) Memory::Write32(mifp, 0); if (mafp) Memory::Write32(mafp, 0); if (milp) Memory::WriteFloat32(milp, 0.0f);
        if (malp) Memory::WriteFloat32(malp, 0.0f); if (lbp) Memory::WriteFloat32(lbp, 0.0f); if (bcp) Memory::Write8(bcp, 0);
        if (elp) Memory::Write8(elp, 0); if (map) Memory::Write32(map, 0);
        return;
    }
    if (mifp) Memory::Write32(mifp, meta.minFilter); if (mafp) Memory::Write32(mafp, meta.magFilter); if (milp) Memory::WriteFloat32(milp, meta.minLod); if (malp) Memory::WriteFloat32(malp, meta.maxLod); if (lbp) Memory::WriteFloat32(lbp, meta.lodBias); if (bcp) Memory::Write8(bcp, meta.biasClamp ? 1 : 0); if (elp) Memory::Write8(elp, meta.edgeLod ? 1 : 0); if (map) Memory::Write32(map, meta.maxAniso);
}

// ============================================================================
// TLUT (Texture Lookup Table)
// ============================================================================

extern "C" void GX__InitTlutObj_HLE(uint32_t oa, uint32_t da, uint32_t f, uint32_t e) {
    std::lock_guard<GuestThreadGuard> guard(g_tlutObjGuard); GXTlutObj* obj = CreateHostTlutObj(oa); TlutObjMeta& meta = GetTlutObjMeta(oa);
    if (meta.dataAddr != CanonicalizeGxMainRamAddress(da) || meta.format != f || meta.entries != (u16)e) NoteTlutObjRangeChangedLocked();
    meta.dataAddr=CanonicalizeGxMainRamAddress(da); meta.format=f; meta.entries=(u16)e; meta.dirty=false;
    // Fill the guest struct's hardware layout like the SDK does, so GXLoadTlut's inline-object path below reads this
    // object as current instead of stale stack bytes (Brawl's HUD reuses one stack object for every portrait).
    WriteGuest32(oa, (f & 3u) << 10, "GXInitTlutObj"); WriteGuest32(oa + 4, 0x64000000u | ((CanonicalizeGxMainRamAddress(da) >> 5) & 0xFFFFFFu), "GXInitTlutObj");
    try { Memory::Write16(oa + 8, static_cast<uint16_t>(e)); } catch (const Memory::AccessViolation&) {}
    // Leave the object unconstructed on a bad descriptor. A later GXLoadTlut
    // then takes GetHostTlutObj's soft-fail path and gets a fresh empty object
    // instead of aurora reading entries*2 bytes off an unvalidated pointer.
    if (!ValidateTlutData(oa, meta)) return;
    // Aurora digests the palette under a write generation it takes on another thread: marked here, where the host
    // object gets its data (gx_guest_write.h g_trackedPages; also PlanTlutLoad's rebuild).
    GxGuestWrite::NoteTrackedRange(da, static_cast<uint32_t>((u16)e) * 2u);
    GXInitTlutObj(obj, GuestToHostPtr(da), (GXTlutFmt)f, (u16)e); MarkHostTlutObjConstructed(oa);
}
PPC_NATIVE_OVERRIDE_VOID(801f3004, GX__InitTlutObj_HLE, (uint32_t oa, uint32_t da, uint32_t f, uint32_t e), (oa, da, f, e));

// A TLUT object the game built without GXInitTlutObj (Brawl's NW4R G3D fills it inline): read the hardware layout
// {u32 tlut (format in bits 10-11), u32 loadTlut0 (address >> 5), u16 numEntries}. The hardware layout carries the
// LOADTLUT0 register id (0x64) in loadTlut0's top byte (GXInitTlutObj above writes the same layout). A reused object
// is rebuilt when its contents change: GX__InitTlutObj_HLE, which writes guest memory, so a deferred native draw
// (g3d_native.cpp) runs this on the game thread when it records the load, and its replay finds the object current.
extern "C" void GX_HLE_PrepareTlut(uint32_t oa) {
    if (oa != 0 && (Memory::Read32(oa + 4) >> 24) == 0x64u) {
        const uint32_t format = (Memory::Read32(oa) >> 10) & 3u;
        const uint32_t data = (Memory::Read32(oa + 4) & 0xFFFFFFu) << 5;
        const uint32_t entries = Memory::Read16(oa + 8);
        bool current = false;
        {
            std::lock_guard<GuestThreadGuard> guard(g_tlutObjGuard);
            const auto it = g_TlutObjMeta.find(oa);
            current = it != g_TlutObjMeta.end() && it->second.dataAddr == CanonicalizeGxMainRamAddress(data) &&
                      it->second.format == format && it->second.entries == entries;
        }
        if (!current && data != 0 && entries != 0) {
            GX__InitTlutObj_HLE(oa, data, format, entries);
        }
    }
}

// GXLoadTlut's table half (guest thread: g_tlutObjGuard): the object rebuilt where the guest changed it, then the host
// object to load; null for an invalid TLUT index (logged; no load and no guest write).
static GXTlutObj* PlanTlutLoad(uint32_t oa, uint32_t tl) {
    GX_HLE_PrepareTlut(oa);
    if (SpmDumpFrame()) {
        auto it = g_TlutObjMeta.find(oa);
        RT_LOGF(RT_TAG_GX, "[drawdump] loadtlut oa 0x%08X slot %u known %d data 0x%08X n %u fmt %u\n", oa, tl, it != g_TlutObjMeta.end() ? 1 : 0,
                it != g_TlutObjMeta.end() ? it->second.dataAddr : 0u, it != g_TlutObjMeta.end() ? it->second.entries : 0u,
                it != g_TlutObjMeta.end() ? it->second.format : 0u);
    } std::lock_guard<GuestThreadGuard> guard(g_tlutObjGuard); if (tl >= kMaxTluts) { RT_LOGF(RT_TAG_GX, "GXLoadTlut: invalid TLUT index %u (oa=0x%08X)\n", tl, oa); return nullptr; } auto metaIt = g_TlutObjMeta.find(oa); if (metaIt != g_TlutObjMeta.end() && metaIt->second.dirty) { if (ValidateTlutData(oa, metaIt->second)) { GXTlutObj* rebuild = CreateHostTlutObj(oa); GxGuestWrite::NoteTrackedRange(metaIt->second.dataAddr, static_cast<uint32_t>(metaIt->second.entries) * 2u); GXInitTlutObj(rebuild, GuestToHostPtr(metaIt->second.dataAddr), (GXTlutFmt)metaIt->second.format, metaIt->second.entries); MarkHostTlutObjConstructed(oa); } metaIt->second.dirty = false; }
    return GetHostTlutObj(oa);
}

static void WriteTlutLoadGuestEffects() {
    try { uint32_t gd = Memory::Read32(kGXDataPtrAddr); if (gd) Memory::Write16(gd + 2, 0); } catch (...) {}
}

extern "C" void GX__LoadTlut_HLE(uint32_t oa, uint32_t tl) {
    G3D_REC_HLE(g3dn::kRecLoadTlut, oa, tl);
    GXTlutObj* obj = PlanTlutLoad(oa, tl);
    if (obj == nullptr) return;
    GXLoadTlut(obj, (GXTlut)tl);
    WriteTlutLoadGuestEffects();
}
PPC_NATIVE_OVERRIDE_VOID(801f302c, GX__LoadTlut_HLE, (uint32_t oa, uint32_t tl), (oa, tl));

// A deferred native draw's TLUT load (g3d_native.cpp), as GX__LoadTexObj's job pair above: the game thread plans it and
// makes its guest write when it records it, the plan carrying a copy of the host object; the draw worker loads the copy.
// Aurora's texture-replacement palette mirror, when on, is producer state of its own (g3d_native.cpp keeps such loads
// on the game thread).
struct TlutLoadJobPlan {
    bool load;
    GXTlutObj obj;
};
extern "C" size_t GX_HLE_TlutLoadPlanBytes() { return sizeof(TlutLoadJobPlan); }
extern "C" void GX_HLE_PlanTlutLoadForJob(uint32_t oa, uint32_t tl, void* out) {
    TlutLoadJobPlan job{};
    if (GXTlutObj* obj = PlanTlutLoad(oa, tl)) {
        job.load = true;
        job.obj = *obj;
        WriteTlutLoadGuestEffects();
    }
    std::memcpy(out, &job, sizeof job);
}
extern "C" void GX_HLE_RunTlutLoadPlan(const void* in, uint32_t tl) {
    TlutLoadJobPlan job;
    std::memcpy(&job, in, sizeof job);
    if (job.load) GXLoadTlut(&job.obj, (GXTlut)tl);
}

// ============================================================================
// Texture Invalidation and Coordinate Control
// ============================================================================

extern "C" void GX__InvalidateTexAll_HLE() {
    // Real GX invalidates its internal texture cache here. Aurora forwards the
    // invalidate to the renderer, which keeps unchanged source uploads hot but
    // revalidates reused guest buffers before serving cached texture handles.
    GXInvalidateTexAll();
}
PPC_NATIVE_OVERRIDE_VOID(801F3194, GX__InvalidateTexAll_HLE, (), ());

extern "C" void GX__SetTexCoordScaleManually_80171180(uint32_t c, uint32_t en, uint32_t ss, uint32_t ts) { GXSetTexCoordScaleManually((GXTexCoordID)c, (GXBool)en, (u16)ss, (u16)ts); try{ uint32_t gd=Memory::Read32(kGXDataPtrAddr); if(gd){ Memory::Write32(gd+0x5E4u, (Memory::Read32(gd+0x5E4u)&~(1u<<c))|((en&1u)<<c)); if(en){ uint32_t sa=gd+0x108u+c*4u, ta=gd+0x128u+c*4u; Memory::Write32(sa, (Memory::Read32(sa)&0xFFFF0000u)|((ss-1)&0xFFFFu)); Memory::Write32(ta, (Memory::Read32(ta)&0xFFFF0000u)|((ts-1)&0xFFFFu)); Memory::Write16(gd+2, 0); } } }catch(...){} }

extern "C" void GX__SetTexCoordBias_801711fc(uint32_t c, uint32_t se, uint32_t te) { GXSetTexCoordBias((GXTexCoordID)c, (GXBool)se, (GXBool)te); try{ uint32_t gd=Memory::Read32(kGXDataPtrAddr); if(gd){ uint32_t sa=gd+0x108u+c*4u, ta=gd+0x128u+c*4u; Memory::Write32(sa, (Memory::Read32(sa)&0xFFFEFFFFu)|((se&1u)<<16)); Memory::Write32(ta, (Memory::Read32(ta)&0xFFFEFFFFu)|((te&1u)<<16)); if(Memory::Read32(gd+0x5E4u)&(1u<<c)) Memory::Write16(gd+2, 0); } }catch(...){} }

// GXInitTexObjData (SPM 0x8028ED7C): repoints a texture object at new image data. SPM's smartTexObj
// uses it for every animated paper-character frame; NSMBW never called it, so it ran translated and
// changed only the guest struct, leaving the native twin on the first image (characters drew nothing).
extern "C" void GX__InitTexObjData_SPM(uint32_t oa, uint32_t da) {
    const uint32_t canonical = CanonicalizeGxMainRamAddress(da);
    {
        std::lock_guard<GuestThreadGuard> guard(g_texObjGuard);
        if (g_TexObjMeta.find(oa) != g_TexObjMeta.end()) {  // else extracted from the guest struct on load
            TexObjMeta& meta = GetTexObjMeta(oa);
            meta.dataAddr = canonical;
            meta.needsUpload = true;
        }
    }
    try {  // the SDK's image3 word: physical address >> 5
        Memory::Write32(oa + 0x0C, (canonical >> 5) & 0x00FFFFFFu);
    } catch (const Memory::AccessViolation&) {
    }
}
PPC_NATIVE_OVERRIDE_VOID(801F2C48, GX__InitTexObjData_SPM, (uint32_t oa, uint32_t da), (oa, da));
