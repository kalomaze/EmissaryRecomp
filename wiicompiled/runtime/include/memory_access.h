#pragma once

#include "guest_flat_memory.h"
#include "memory.h"
#include "isa/ppc_isa_float.h"
#include "brawl_dirty_memory.h"
#include "recomp_mod_loader.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

// Hooks for GX HLE FIFO handling
extern "C" {
    void GX_HLE_FIFO_WriteFloat(float val);
    void GX_HLE_FIFO_Write32(uint32_t val);
    void GX_HLE_FIFO_Write16(uint16_t val);
    void GX_HLE_FIFO_Write8(uint8_t val);
    void GX_HLE_FIFO_WriteBurst(const uint8_t* data, uint32_t sizeBytes);
}

namespace MemoryInline {
#if defined(_WIN32)
#define MKW_MEMORY_FORCE_INLINE __forceinline
#define MKW_MEMORY_NO_INLINE __declspec(noinline)
#else
// See runtime/include/isa/ppc_isa_config.h for why non-Windows Clang needs the GNU-attribute
// spellings instead of the MS-extension keywords.
#define MKW_MEMORY_FORCE_INLINE __attribute__((always_inline)) inline
#define MKW_MEMORY_NO_INLINE __attribute__((noinline))
#endif
#define MKW_MEMORY_COLD __attribute__((cold))
#ifndef MKW_NONFLAT_INLINE
#define MKW_NONFLAT_INLINE 1  // without the flat view: try the page table inline before calling out (0 = always call)
#endif
// MKW_LEAN_MEMORY=1: every guest access is `base + (addr & 0x3FFFFFFF)` into one 1 GiB host window, with no page
// table and no fault handler (so it works on Horizon too). The mask folds the physical, cached (0x8/0x9) and
// uncached (0xC/0xD) views onto one copy: MEM1 at +0, the MMIO window at +0x0C000000, MEM2 at +0x10000000, the
// locked cache at +0x20000000. Loads are and+load+swap. Stores test one byte per 64 KiB (the store policy at
// kLeanPolicyOffset: MMIO, executable pages, everything outside the guest regions), take the checked Memory::Write*
// path only where it is set, and mark rollback's dirty map (kLeanDirtyOffset, brawl_dirty_memory.h). On every OS
// only the regions, those two tables and the MMIO window (reading zero) are backed (memory.cpp, BackLeanWindow).
#ifndef MKW_LEAN_MEMORY
#define MKW_LEAN_MEMORY 0
#endif
#if MKW_LEAN_MEMORY && MKW_FLAT_MEMORY
#error "MKW_LEAN_MEMORY replaces MKW_FLAT_MEMORY: build with MKW_FLAT_MEMORY=OFF"
#endif
#if MKW_LEAN_MEMORY
inline constexpr uint32_t kLeanMask = 0x3FFFFFFFu;
inline constexpr uint64_t kLeanWindowSize = 1ull << 30;
inline constexpr uint32_t kLeanPolicyShift = 16;
// The store policy (one byte per 64 KiB, rebuilt by RefreshFastPathTables) and rollback's dirty marks (one byte per
// 64 bytes, brawl_dirty_memory.h) live inside the window, past every guest region (MEM2 ends at 0x15000000, the
// locked cache is 16 KiB at 0x20000000). A store then needs one base register for its policy test, its mark and
// itself; as separate globals each cost a reload per access (a guest store may alias any global). Their own
// 64 KiB chunks are policy-flagged, so a stray guest store there takes the checked path instead.
inline constexpr uint32_t kLeanDirtyOffset = 0x3E000000u;   // kLeanWindowSize / 64 + 1 bytes
inline constexpr uint32_t kLeanPolicyOffset = 0x3F800000u;  // kLeanWindowSize >> kLeanPolicyShift bytes
// The window's host base, set once by Memory::Init (SetLeanBase) before any guest code runs and never changed after.
// Every reader sees the object through a const declaration, so the compiler keeps it in a register across guest
// stores (a guest store is a byte copy, which may alias any non-const global: as a plain global the base was loaded
// again after every store). The one writable definition is src/lean_base.cpp, built outside the unity batches.
#if defined(MKW_LEAN_BASE_DEFINITION)
extern uint8_t* g_leanBase;
#else
extern uint8_t* const g_leanBase;
#endif
void SetLeanBase(uint8_t* base);
// The base read through the writable definition, out of line: for code that runs while Memory::Init sets it (a load
// of the const view may be scheduled before the SetLeanBase call that precedes it).
uint8_t* LeanBaseAtInit();
MKW_MEMORY_FORCE_INLINE uint8_t* LeanHost(uint32_t address) { return g_leanBase + (address & kLeanMask); }
MKW_MEMORY_FORCE_INLINE bool LeanStoreNeedsPolicy(uint32_t address) {
    return g_leanBase[kLeanPolicyOffset + ((address & kLeanMask) >> kLeanPolicyShift)] != 0;
}
// A plain lean store: policy test, dirty mark (BrawlDirtyMemory::Mark's 16-bit pair), then the bytes, all off one
// load of the base. False when the policy byte sends the store to the checked path.
template <typename T>
MKW_MEMORY_FORCE_INLINE bool LeanTryStore(uint32_t address, T swapped) {
    uint8_t* const base = g_leanBase;
    const uint32_t canonical = address & kLeanMask;
    if (base[kLeanPolicyOffset + (canonical >> kLeanPolicyShift)] != 0) [[unlikely]] return false;
#if MKW_LEAN_DIRTY_MARKS
    const uint16_t both = 0x0101;
    std::memcpy(base + kLeanDirtyOffset + (canonical >> 6), &both, sizeof(both));
#endif
    std::memcpy(base + canonical, &swapped, sizeof(T));
    return true;
}
// Segment ranges (ResolveSegmentRange below): the range policy, one byte per 64 KiB chunk, set where the chunk or the
// one after it needs the checked path (RefreshLeanStorePolicy), and the reserved, never backed guard after the window.
inline constexpr uint32_t kLeanRangePolicyOffset = kLeanPolicyOffset + 0x8000u;  // same size as the store policy
inline constexpr uint64_t kLeanGuardSize = 1ull << 16;
#endif
inline constexpr uint32_t kPageShift = 20;
inline constexpr uint32_t kPageSize = 1u << kPageShift;
inline constexpr uint32_t kPageMask = kPageSize - 1u;
inline constexpr uint32_t kPageCount = 1u << (32 - kPageShift);
inline constexpr uint32_t kMaxFastScalarSize = 8;
inline constexpr uint32_t kWritableSubPageShift = RecompMod::kExecutableWriteGuardPageShift;
inline constexpr uint32_t kWritableSubPageSize = 1u << kWritableSubPageShift;
inline constexpr uint32_t kWritableSubPagesPerPage = kPageSize / kWritableSubPageSize;

struct PageEntry {
    uint8_t* base = nullptr;
    uint32_t limit = 0;
};

extern PageEntry g_pageTable[kPageCount];
// Encoded (host page base - guest page base) + 1 for full pages whose next
// page is contiguous. This permits any native access up to 8 bytes without a
// per-access mask/limit check. Zero retains the general PageEntry fallback.
extern uintptr_t g_fullPageBias[kPageCount];
// Same encoding, but only for pages proven not to contain executable bytes.
// Executable-range registration invalidates entries before guest execution.
extern uintptr_t g_fullWritablePageBias[kPageCount];

// Allocated only for mapped 1 MiB pages that contain both executable and data
// 4 KiB pages. Entries use the same encoded host bias as the coarse table.
// Exact executable bits remain authoritative and are checked at lookup time,
// including both sides of a cross-4-KiB access.
struct SparseWritablePageTable {
    uintptr_t encodedBias[kWritableSubPagesPerPage]{};
};
extern const SparseWritablePageTable* g_sparseWritablePageTables[kPageCount];

// The cold paths translated code calls take MKW_COLD_OP_CC (isa/ppc_isa_config.h, preserve_all): a call site then
// spills nothing, so a function whose only calls are these needs no callee-saved registers.
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC uint8_t Read8Slow(uint32_t addr);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC uint16_t Read16Slow(uint32_t addr);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC uint32_t Read32Slow(uint32_t addr);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC uint64_t Read64Slow(uint32_t addr);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC float ReadFloat32Slow(uint32_t addr);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC double ReadFloat64Slow(uint32_t addr);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC void Write8Slow(uint32_t addr, uint8_t val);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC void Write16Slow(uint32_t addr, uint16_t val);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC void Write32Slow(uint32_t addr, uint32_t val);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC void Write64Slow(uint32_t addr, uint64_t val);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC void WriteFloat32Slow(uint32_t addr, double val);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC void WriteFloat64Slow(uint32_t addr, double val);
template <typename T>
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC T ReadResolvedFallback(uint32_t addr);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC float ReadResolvedFallbackFloat32(uint32_t addr);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC double ReadResolvedFallbackFloat64(uint32_t addr);
template <typename T>
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC void WriteResolvedFallback(uint32_t addr, T value);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC void WriteResolvedFallbackFloat32(uint32_t addr, double val);
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC void WriteResolvedFallbackFloat64(uint32_t addr, double val);
#if MKW_LEAN_MEMORY
// Every guest store into [addr, addr + size) ends as the dirty marks and the bytes in the window, on whichever path the
// policy sends it: the chunks are unflagged, or the range lies in MEM1's or MEM2's cached view where
// WriteResolvedFallback's fast exit takes it (no guarded code in any of its 4 KiB pages). For natives that write a
// whole range at once (memcpy, memset's body): plain host stores plus the marks of the run are then exact.
bool LeanStoresArePlain(uint32_t addr, uint32_t size);
#endif
#if MKW_OUTLINE_GUEST_OPS && MKW_LEAN_MEMORY
// Lean stores and exact single loads/stores, out of line (src/guest_ops_outlined.cpp): FlatStore/FlatStoreRam's lean
// bodies and lfs/stfs's conversions, each the inline code below compiled once.
MKW_MEMORY_NO_INLINE MKW_GUEST_OP_CC void LeanStore8(uint32_t address, uint8_t value);
MKW_MEMORY_NO_INLINE MKW_GUEST_OP_CC void LeanStore16(uint32_t address, uint16_t value);
MKW_MEMORY_NO_INLINE MKW_GUEST_OP_CC void LeanStore32(uint32_t address, uint32_t value);
MKW_MEMORY_NO_INLINE MKW_GUEST_OP_CC void LeanStore64(uint32_t address, uint64_t value);
MKW_MEMORY_NO_INLINE MKW_GUEST_OP_CC void LeanStoreRam8(uint32_t address, uint8_t value);
MKW_MEMORY_NO_INLINE MKW_GUEST_OP_CC void LeanStoreRam16(uint32_t address, uint16_t value);
MKW_MEMORY_NO_INLINE MKW_GUEST_OP_CC void LeanStoreRam32(uint32_t address, uint32_t value);
MKW_MEMORY_NO_INLINE MKW_GUEST_OP_CC void LeanStoreRam64(uint32_t address, uint64_t value);
MKW_MEMORY_NO_INLINE MKW_GUEST_OP_CC void LeanStoreFloat32(uint32_t address, double value);
MKW_MEMORY_NO_INLINE MKW_GUEST_OP_CC void LeanStoreRamFloat32(uint32_t address, double value);
MKW_MEMORY_NO_INLINE MKW_GUEST_OP_CC double LeanLoadFloat32(uint32_t address);
#endif

constexpr bool IsMmioAddress(uint32_t addr) {
    return addr >= 0xCC000000u && addr < 0xCE000000u;
}

// Modeled device blocks. Other MMIO addresses currently read zero through the
// compatibility path and discard helper-based writes. This is not a complete
// hardware model: Dolphin also has MMIO aliases that this policy does not model.
constexpr bool IsDeviceMmioAddress(uint32_t addr) {
    const uint32_t block = addr & 0xFFFF0000u;
    return IsMmioAddress(addr) && (block == 0xCC000000u || block == 0xCD000000u || block == 0xCD800000u);
}

constexpr bool IsGpuFifoAddress(uint32_t addr) {
    return addr >= 0xCC008000u && addr < 0xCC009000u;
}

// Page protections can't cover this: an MMIO write must reach GX HLE with its value or be
// reported, and a fault record can't carry the value, so this mask/compare sits in front of
// every flat store instead.
MKW_MEMORY_FORCE_INLINE bool FlatWriteNeedsPolicy(uint32_t address) {
#if MKW_LEAN_MEMORY
    (void)address;
    return false;  // FlatStore's policy byte covers the MMIO window (and executable pages)
#else
    return (address & 0xFE000000u) == 0xCC000000u;  // 0xCC000000..0xCDFFFFFF
#endif
}

// Gekko stfs conversion is a bit-level narrowing operation.  In particular it
// does not behave like a host double-to-float cast for values which were left
// in double precision, and it has hardware-tested handling for tiny values.
// ConvertPpcDoubleToSingleBits' subnormal-single arm (exponents 874..896), out of line: rare, and inlined at every
// stfs it was most of the conversion's code.
MKW_MEMORY_NO_INLINE MKW_MEMORY_COLD MKW_COLD_OP_CC inline uint32_t ConvertPpcDoubleToSubnormalSingleBits(uint64_t bits, uint32_t exponent) {
    uint32_t narrowed = static_cast<uint32_t>(
        0x80000000ULL | ((bits & 0x000FFFFFFFFFFFFFULL) >> 21));
    narrowed >>= (905u - exponent);
    narrowed |= static_cast<uint32_t>((bits >> 32) & 0x80000000ULL);
    return narrowed;
}

MKW_MEMORY_FORCE_INLINE uint32_t ConvertPpcDoubleToSingleBits(double value) {
#if MKW_FAST_FLOAT  // isa/ppc_isa_float.h: the host's rounding narrowing instead of Gekko's truncating one
    const float single = static_cast<float>(value);
    uint32_t singleBits = 0;
    std::memcpy(&singleBits, &single, sizeof(singleBits));
    return singleBits;
#endif
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = static_cast<uint32_t>((bits >> 52) & 0x7FFu);
    // The subnormal-single arm applies to exactly the exponents 874..896; every
    // other exponent takes the plain sign/exponent/fraction narrowing below.
    // Inside that window the exponent field is nonzero, so the magnitude cannot
    // be zero and needs no separate test - the zero case (exponent 0) reaches
    // the narrowing exactly as it did when the two arms shared that test.
    if (exponent - 874u <= 22u) [[unlikely]]
        return ConvertPpcDoubleToSubnormalSingleBits(bits, exponent);

    // Results below the documented conversion range are architecturally
    // undefined; this is the behavior measured on Gekko/Broadway hardware.
    return static_cast<uint32_t>(
        ((bits >> 32) & 0xC0000000ULL) | ((bits >> 29) & 0x3FFFFFFFULL));
}

MKW_MEMORY_FORCE_INLINE float PpcSingleBitsToFloat(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

MKW_MEMORY_FORCE_INLINE bool TryGetPointerFast(uint32_t address, size_t length, uint8_t*& pointer) {
    const uint32_t page = address >> kPageShift;
    if (length <= 8) {
        const uintptr_t encodedBias = g_fullPageBias[page];
        if (encodedBias != 0) {
            pointer = reinterpret_cast<uint8_t*>((encodedBias - 1u) + address);
            return true;
        }
    }
    const uint32_t offset = address & kPageMask;
    const auto& entry = g_pageTable[page];
    if (!entry.base || offset > entry.limit || length > entry.limit - offset) {
        pointer = nullptr;
        return false;
    }
    pointer = entry.base + offset;
    return true;
}

inline uint8_t* GetPointerFast(uint32_t address, size_t length) {
    uint8_t* pointer = nullptr;
    return TryGetPointerFast(address, length, pointer) ? pointer : nullptr;
}

MKW_MEMORY_FORCE_INLINE bool TryGetWritablePointerFast(
    uint32_t address, size_t length, uint8_t*& pointer) {
    // The sparse-table hit below distinguishes itself from the small-mapping
    // hit through `pointer`, so the out-parameter must start null regardless
    // of what the caller passed in. (A caller handing in an uninitialized
    // pointer used to turn every sparse hit into a write through stack
    // garbage - random host memory corruption.)
    pointer = nullptr;
    if (length == 0 || length > 8 || address > UINT32_MAX - (length - 1))
        return false;
    const uint32_t coarsePage = address >> kPageShift;
    uintptr_t encodedBias = g_fullWritablePageBias[coarsePage];
    const uint32_t endAddress = address + static_cast<uint32_t>(length - 1);
    const uint32_t firstExactPage = address >> kWritableSubPageShift;
    const uint32_t lastExactPage = endAddress >> kWritableSubPageShift;
    if (encodedBias != 0 && (endAddress >> kPageShift) != coarsePage &&
        RecompMod::g_executableWriteGuardPages[lastExactPage].load(
            std::memory_order_relaxed) != 0) {
        return false;
    }
    if (encodedBias == 0) {
        const auto* subTable = g_sparseWritablePageTables[coarsePage];
        if (subTable != nullptr) {
            encodedBias = subTable->encodedBias[
                (address & kPageMask) >> kWritableSubPageShift];
            if (encodedBias == 0)
                return false;
        } else {
            // Small mappings such as Broadway's 16 KiB locked cache cannot
            // populate the full-1-MiB bias table. Keep them native by proving
            // the exact access against the ordinary page entry, then applying
            // the same executable-write policy as a sparse-table hit.
            const uint32_t offset = address & kPageMask;
            const auto& entry = g_pageTable[coarsePage];
            if (!entry.base || offset + length > entry.limit)
                return false;
            pointer = entry.base + offset;
        }

        // The exact 4 KiB guard is the final authority. Checking it on every
        // checked/sparse hit also makes later executable-range registration
        // safe when a prebuilt table still contains the old mapped bias.
        if (RecompMod::g_executableWriteGuardPages[firstExactPage].load(
                std::memory_order_relaxed) != 0 ||
            (lastExactPage != firstExactPage &&
             RecompMod::g_executableWriteGuardPages[lastExactPage].load(
                 std::memory_order_relaxed) != 0)) {
            return false;
        }

        if (pointer != nullptr)
            return true;
    }
    if (encodedBias == 0)
        return false;
    pointer = reinterpret_cast<uint8_t*>((encodedBias - 1u) + address);
    return true;
}

// Flat form: guest_flat_memory.h's page protections already answer mapped/non-executable, so resolving is pure address arithmetic.
// Two checks stay inline: a wrapped guest address can't survive 64-bit `host + rangeOffset`, and an MMIO write's value isn't recoverable from a
// fault record, so a write touching that window must resolve null and fall back to Memory::Write*. Checking both range endpoints is a complete
// proof since length <= kPageSize (1 MiB) can't straddle the 32 MiB MMIO window.
MKW_MEMORY_FORCE_INLINE uint8_t* ResolveRangeHost(uint32_t base, int32_t minOffset, uint32_t length,
                                                 bool needsWrite) {
    const uint32_t guestStart = base + static_cast<uint32_t>(minOffset);
    if (length == 0 || length > kPageSize || guestStart > UINT32_MAX - (length - 1)) return nullptr;
#if MKW_FLAT_MEMORY
    if (GuestFlat::RequiresCheckedAccess()) return nullptr;
    if (needsWrite &&
        (FlatWriteNeedsPolicy(guestStart) || FlatWriteNeedsPolicy(guestStart + (length - 1))))
        [[unlikely]] return nullptr;
    return MKW_FLAT_GUEST_BASE + guestStart;
#elif MKW_LEAN_MEMORY
    const uint32_t last = guestStart + length - 1;
    if (((guestStart ^ last) & ~kLeanMask) != 0) return nullptr;  // straddles two views: no single host range
    if (needsWrite) {
        for (uint32_t chunk = (guestStart & kLeanMask) >> kLeanPolicyShift;
             chunk <= ((last & kLeanMask) >> kLeanPolicyShift); ++chunk)
            if (g_leanBase[kLeanPolicyOffset + chunk] != 0) return nullptr;
    }
    return LeanHost(guestStart);
#else
    const uint32_t last = guestStart + length - 1;
    const uint32_t page = guestStart >> kPageShift;
    if ((last >> kPageShift) != page || IsMmioAddress(guestStart) || IsMmioAddress(last))
        return nullptr;
    if (needsWrite) {
        for (uint32_t p = guestStart >> kWritableSubPageShift; p <= (last >> kWritableSubPageShift); ++p)
            if (RecompMod::g_executableWriteGuardPages[p].load(std::memory_order_relaxed))
                return nullptr;
    }
    return GetPointerFast(guestStart, length);
#endif
}

// Guest-address byte order. Distinct from isa/big_endian.h, which is the
// host-pointer codec; do not "unify" them.
inline uint16_t ByteSwap16(uint16_t value) {
    return __builtin_bswap16(value);
}

inline uint32_t ByteSwap32(uint32_t value) {
    return __builtin_bswap32(value);
}

inline uint64_t ByteSwap64(uint64_t value) {
    return __builtin_bswap64(value);
}

template <typename T>
inline T MaybeByteSwap(T value) {
    if constexpr (sizeof(T) == 1) {
        return value;
    } else if constexpr (sizeof(T) == 2) {
        return static_cast<T>(ByteSwap16(static_cast<uint16_t>(value)));
    } else if constexpr (sizeof(T) == 4) {
        return static_cast<T>(ByteSwap32(static_cast<uint32_t>(value)));
    } else if constexpr (sizeof(T) == 8) {
        return static_cast<T>(ByteSwap64(static_cast<uint64_t>(value)));
    } else {
        return value;
    }
}

template <typename T>
MKW_MEMORY_FORCE_INLINE bool ReadResolvedScalar(uint8_t* host, uint32_t rangeOffset, T& outValue) {
    if (!host) return false;
    if constexpr (sizeof(T) == 1) {
        outValue = host[rangeOffset];
    } else {
        T value = 0;
        std::memcpy(&value, host + rangeOffset, sizeof(T));
        outValue = MaybeByteSwap(value);
    }
    return true;
}

struct ResolvedLoadPair {
    uint32_t first = 0;
    uint32_t second = 0;
    bool valid = false;
};

// Lean memory: a resolved access is just a lean access at its own address (the third argument every scalar form
// carries). The block's range proof buys nothing when a load is and+load and a store tests its own policy byte, and
// the host-pointer forms cost more: a range test with a policy loop per block, a range-checked dirty mark per store
// and a fallback per access. The pair forms carry no address, so they report "not taken" and the translator's
// per-word fallback (two scalar resolved accesses, with addresses) does the work. With the host unused, Clang drops
// ResolveRangeHost wherever the psq resolved forms (which still read through the host) don't use it.
MKW_MEMORY_FORCE_INLINE ResolvedLoadPair ReadResolvedPair16(
    uint8_t* host, uint32_t rangeOffset) {
#if MKW_LEAN_MEMORY
    (void)host; (void)rangeOffset;
    return {};
#endif
    uint32_t packed = 0;
    if (!ReadResolvedScalar(host, rangeOffset, packed)) return {};
    return {packed >> 16, packed & 0xFFFFu, true};
}

MKW_MEMORY_FORCE_INLINE ResolvedLoadPair ReadResolvedPair32(uint8_t* host, uint32_t rangeOffset) {
#if MKW_LEAN_MEMORY
    (void)host; (void)rangeOffset;
    return {};
#endif
    uint64_t packed = 0;
    if (!ReadResolvedScalar(host, rangeOffset, packed)) return {};
    return {static_cast<uint32_t>(packed >> 32), static_cast<uint32_t>(packed), true};
}

template <typename Packed>
MKW_MEMORY_FORCE_INLINE bool WriteResolvedPairFast(
    uint8_t* host, uint32_t rangeOffset, Packed packed) {
    if (!host) return false;
    BrawlDirtyMemory::MarkHost(host + rangeOffset,sizeof(Packed));
    const Packed swapped = MaybeByteSwap(packed);
    std::memcpy(host + rangeOffset, &swapped, sizeof(swapped));
    return true;
}

MKW_MEMORY_FORCE_INLINE bool WriteResolvedPair16(
    uint8_t* host, uint32_t rangeOffset, uint32_t packed) {
#if MKW_LEAN_MEMORY
    (void)host; (void)rangeOffset; (void)packed;
    return false;  // see ReadResolvedPair16
#endif
    return WriteResolvedPairFast(host, rangeOffset, packed);
}

MKW_MEMORY_FORCE_INLINE bool WriteResolvedPair32(
    uint8_t* host, uint32_t rangeOffset, uint64_t packed) {
#if MKW_LEAN_MEMORY
    (void)host; (void)rangeOffset; (void)packed;
    return false;  // see ReadResolvedPair16
#endif
    return WriteResolvedPairFast(host, rangeOffset, packed);
}

template <typename T>
MKW_MEMORY_FORCE_INLINE bool WriteResolvedScalar(uint8_t* host, uint32_t rangeOffset, T value) {
    if (!host) return false;
    BrawlDirtyMemory::MarkHost(host + rangeOffset,sizeof(T));
    if constexpr (sizeof(T) == 1) {
        host[rangeOffset] = static_cast<uint8_t>(value);
    } else {
        const T swapped = MaybeByteSwap(value);
        std::memcpy(host + rangeOffset, &swapped, sizeof(T));
    }
    return true;
}

template <typename T>
inline bool TryReadMappedScalar(uint32_t address, T& outValue) {
#if MKW_LEAN_MEMORY
    T value{};
    std::memcpy(&value, LeanHost(address), sizeof(T));
    outValue = MaybeByteSwap(value);
    return true;
#endif
    uint8_t* ptr = nullptr;
    if (TryGetPointerFast(address, sizeof(T), ptr)) {
        if constexpr (sizeof(T) == 1) {
            outValue = *ptr;
        } else {
            T value = 0;
            std::memcpy(&value, ptr, sizeof(T));
            outValue = MaybeByteSwap(value);
        }
        return true;
    }
    return false;
}

template <typename T>
MKW_MEMORY_FORCE_INLINE bool TryReadGuestScalar(uint32_t address, T& outValue) {
    const uintptr_t encodedBias = g_fullPageBias[address >> kPageShift];
    if (encodedBias == 0) [[unlikely]]
        return false;
    auto* ptr = reinterpret_cast<uint8_t*>((encodedBias - 1u) + address);
    if constexpr (sizeof(T) == 1) {
        outValue = *ptr;
    } else {
        T value = 0;
        std::memcpy(&value, ptr, sizeof(T));
        outValue = MaybeByteSwap(value);
    }
    return true;
}

template <typename T>
MKW_MEMORY_FORCE_INLINE bool TryWriteGuestScalar(uint32_t address, T value) {
    static_assert(sizeof(T) >= 1 && sizeof(T) <= kMaxFastScalarSize);
    // A mixed executable/data 1 MiB page zeroes the coarse writable bias even though most of
    // its 4 KiB sub-pages are plain data; MKW's THP buffers share such a page with .text, which
    // used to force ~15% of total CPU through the cold path. The sparse sub-page tier below
    // keeps those stores native while the exact 4 KiB executable guards stay authoritative.
    if (address > UINT32_MAX - static_cast<uint32_t>(sizeof(T) - 1u)) [[unlikely]]
        return false;
    const uint32_t coarsePage = address >> kPageShift;
    const uint32_t endAddress = address + static_cast<uint32_t>(sizeof(T) - 1u);
    if ((endAddress >> kPageShift) != coarsePage) [[unlikely]]
        return false;
    const uintptr_t encodedBias = g_fullWritablePageBias[coarsePage];
    uint8_t* ptr = nullptr;
    if (encodedBias != 0) {
        ptr = reinterpret_cast<uint8_t*>((encodedBias - 1u) + address);
    } else if (!TryGetWritablePointerFast(address, sizeof(T), ptr)) [[unlikely]] {
        return false;
    }
    BrawlDirtyMemory::Mark(address,sizeof(T));
        if constexpr (sizeof(T) == 1) {
        *ptr = static_cast<uint8_t>(value);
    } else {
        const T swapped = MaybeByteSwap(value);
        std::memcpy(ptr, &swapped, sizeof(T));
    }
    return true;
}

template <typename T>
inline bool WriteStackScalarFast(uint32_t address, T value) {
#if MKW_LEAN_MEMORY
    return LeanTryStore<T>(address, MaybeByteSwap(value));
#endif
    uint8_t* ptr = nullptr;
    if (TryGetPointerFast(address, sizeof(T), ptr)) {
        BrawlDirtyMemory::Mark(address,sizeof(T));
        if constexpr (sizeof(T) == 1) {
            *ptr = static_cast<uint8_t>(value);
        } else {
            const T swapped = MaybeByteSwap(value);
            std::memcpy(ptr, &swapped, sizeof(T));
        }
        return true;
    }
    return false;
}

inline uint8_t ReadStack8(uint32_t address) {
    uint8_t value = 0;
    return TryReadMappedScalar(address, value) ? value : Memory::Read8(address);
}

inline uint16_t ReadStack16(uint32_t address) {
    uint16_t value = 0;
    return TryReadMappedScalar(address, value) ? value : Memory::Read16(address);
}

inline uint32_t ReadStack32(uint32_t address) {
    uint32_t value = 0;
    return TryReadMappedScalar(address, value) ? value : Memory::Read32(address);
}

inline uint64_t ReadStack64(uint32_t address) {
    uint64_t value = 0;
    return TryReadMappedScalar(address, value) ? value : Memory::Read64(address);
}



MKW_MEMORY_FORCE_INLINE void WriteStack8(uint32_t address, uint8_t value) {
    if (!WriteStackScalarFast(address, value)) {
        Memory::Write8(address, value);
    }
}

MKW_MEMORY_FORCE_INLINE void WriteStack16(uint32_t address, uint16_t value) {
    if (!WriteStackScalarFast(address, value)) {
        Memory::Write16(address, value);
    }
}

MKW_MEMORY_FORCE_INLINE void WriteStack32(uint32_t address, uint32_t value) {
    if (!WriteStackScalarFast(address, value)) {
        Memory::Write32(address, value);
    }
}

MKW_MEMORY_FORCE_INLINE void WriteStack64(uint32_t address, uint64_t value) {
    if (!WriteStackScalarFast(address, value)) {
        Memory::Write64(address, value);
    }
}

inline void WriteStackFloat32(uint32_t address, double value) {
    const uint32_t bits = ConvertPpcDoubleToSingleBits(value);
    if (!WriteStackScalarFast(address, bits)) {
        Memory::WriteFloat32(address, value);
    }
}

inline void WriteStackFloat64(uint32_t address, double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if (!WriteStackScalarFast(address, bits)) {
        Memory::WriteFloat64(address, value);
    }
}

template <typename T>
MKW_MEMORY_FORCE_INLINE T FlatLoad(uint32_t address);
template <typename T>
MKW_MEMORY_FORCE_INLINE void FlatStore(uint32_t address, T value);

template <typename T, typename SlowRead>
MKW_MEMORY_FORCE_INLINE T ReadResolved(uint8_t* host, uint32_t rangeOffset, uint32_t address,
                      SlowRead slow) {
#if MKW_LEAN_MEMORY
    (void)host; (void)rangeOffset; (void)slow;
    return FlatLoad<T>(address);  // see ReadResolvedPair16
#endif
    T value = 0;
    if (ReadResolvedScalar(host, rangeOffset, value)) {
        return value;
    }
#if !MKW_FLAT_MEMORY && !MKW_LEAN_MEMORY && MKW_NONFLAT_INLINE
    // The block's whole range failed to resolve (it touches a guarded, MMIO or unmapped page somewhere): this one
    // access is usually still plain RAM, so try it on its own before calling out.
    if (TryReadGuestScalar<T>(address, value)) return value;
#endif
    [[unlikely]] return slow(address);
}

MKW_MEMORY_FORCE_INLINE uint8_t ReadResolved8(uint8_t* r, uint32_t o, uint32_t a) { return ReadResolved<uint8_t>(r, o, a, ReadResolvedFallback<uint8_t>); }
MKW_MEMORY_FORCE_INLINE uint16_t ReadResolved16(uint8_t* r, uint32_t o, uint32_t a) { return ReadResolved<uint16_t>(r, o, a, ReadResolvedFallback<uint16_t>); }
MKW_MEMORY_FORCE_INLINE uint32_t ReadResolved32(uint8_t* r, uint32_t o, uint32_t a) { return ReadResolved<uint32_t>(r, o, a, ReadResolvedFallback<uint32_t>); }
// Live via isa/ppc_isa_quantized.h (the psq resolved tier packs two lanes into
// one 64-bit access); generated code never names it directly.
MKW_MEMORY_FORCE_INLINE uint64_t ReadResolved64(uint8_t* r, uint32_t o, uint32_t a) { return ReadResolved<uint64_t>(r, o, a, ReadResolvedFallback<uint64_t>); }
// lfs widens the stored single to an FPR double without flushing subnormals,
// even when FPSCR[NI] enables host DAZ/FZ for arithmetic. A C++ float->double
// conversion would silently turn those loads into zero on such hosts.
MKW_MEMORY_FORCE_INLINE double FlatReadFloat32(uint32_t address);
MKW_MEMORY_FORCE_INLINE void FlatWriteFloat32(uint32_t address, double value);
MKW_MEMORY_FORCE_INLINE double ReadResolvedFloat32(uint8_t* r, uint32_t o, uint32_t a) {
#if MKW_LEAN_MEMORY
    (void)r; (void)o;
    return FlatReadFloat32(a);  // the same and+load and widening (see ReadResolvedPair16)
#endif
    return PpcBitCastToDoubleInline(PpcConvertToDoubleBitsInline(ReadResolved32(r, o, a)));
}
MKW_MEMORY_FORCE_INLINE double ReadResolvedFloat64(uint8_t* r, uint32_t o, uint32_t a) {
#if MKW_LEAN_MEMORY
    (void)r; (void)o;
    const uint64_t leanBits = FlatLoad<uint64_t>(a);
    double leanValue; std::memcpy(&leanValue, &leanBits, sizeof(leanValue)); return leanValue;
#endif
    uint64_t bits = 0;
    if (!ReadResolvedScalar(r, o, bits)) [[unlikely]] return ReadResolvedFallbackFloat64(a);
    double value; std::memcpy(&value, &bits, sizeof(value)); return value;
}

template <typename T, typename SlowWrite>
MKW_MEMORY_FORCE_INLINE void WriteResolved(uint8_t* host, uint32_t rangeOffset, uint32_t address, T value,
                          SlowWrite slow) {
#if MKW_LEAN_MEMORY
    (void)host; (void)rangeOffset; (void)slow;
    FlatStore<T>(address, value);  // its own policy test and dirty mark (see ReadResolvedPair16)
    return;
#endif
    if (WriteResolvedScalar(host, rangeOffset, value)) {
        return;
    }
#if !MKW_FLAT_MEMORY && !MKW_LEAN_MEMORY && MKW_NONFLAT_INLINE
    if (TryWriteGuestScalar<T>(address, value)) return;  // as in ReadResolved: this access alone is usually RAM
#endif
    [[unlikely]] slow(address, value);
}

MKW_MEMORY_FORCE_INLINE void WriteResolved8(uint8_t* r, uint32_t o, uint32_t a, uint8_t v) { WriteResolved(r, o, a, v, WriteResolvedFallback<uint8_t>); }
MKW_MEMORY_FORCE_INLINE void WriteResolved16(uint8_t* r, uint32_t o, uint32_t a, uint16_t v) { WriteResolved(r, o, a, v, WriteResolvedFallback<uint16_t>); }
MKW_MEMORY_FORCE_INLINE void WriteResolved32(uint8_t* r, uint32_t o, uint32_t a, uint32_t v) { WriteResolved(r, o, a, v, WriteResolvedFallback<uint32_t>); }
// Live via isa/ppc_isa_quantized.h, as ReadResolved64 above.
MKW_MEMORY_FORCE_INLINE void WriteResolved64(uint8_t* r, uint32_t o, uint32_t a, uint64_t v) { WriteResolved(r, o, a, v, WriteResolvedFallback<uint64_t>); }
MKW_MEMORY_FORCE_INLINE void WriteResolvedFloat32(uint8_t* r, uint32_t o, uint32_t a, double v) {
#if MKW_LEAN_MEMORY
    (void)r; (void)o;
    FlatWriteFloat32(a, v);  // see ReadResolvedPair16
    return;
#endif
    const uint32_t bits = ConvertPpcDoubleToSingleBits(v);
    if (WriteResolvedScalar(r, o, bits)) return;
    [[unlikely]] WriteResolvedFallbackFloat32(a, v);
}
MKW_MEMORY_FORCE_INLINE void WriteResolvedFloat64(uint8_t* r, uint32_t o, uint32_t a, double v) {
#if MKW_LEAN_MEMORY
    (void)r; (void)o;
    uint64_t leanBits; std::memcpy(&leanBits, &v, sizeof(leanBits));
    FlatStore<uint64_t>(a, leanBits);  // see ReadResolvedPair16
    return;
#endif
    uint64_t bits; std::memcpy(&bits, &v, sizeof(bits));
    if (WriteResolvedScalar(r, o, bits)) return;
    [[unlikely]] WriteResolvedFallbackFloat64(a, v);
}

// ---------------------------------------------------------------------------------------------------------------------
// Segment ranges: the translator's resolved accesses with MKW_LEAN_RANGES (GuestMemoryRangeLowering: two or more
// accesses off one base inside one straight-line segment, no call or branch between them, at most 4 KiB; runtime code
// keeps ResolveRangeHost and the accessors above). Lean memory resolves a segment range once and each access is then
// one byte-swapped load or store off it:
// - a read range is `base + (start & mask)`, with no test. Every access of the range runs whenever its first does (one
//   segment). An access computed off the range differs from its own masked address only when the range straddles a
//   1 GiB view boundary, and then another access of the same range reads canonical [0x3FFFF000, 0x40000000), which is
//   unbacked on every OS: the per-access code faults in that same segment. The 64 KiB past the window are reserved and
//   never backed (kLeanGuardSize, memory.cpp), so the range's access faults too instead of reaching other host memory.
//   A run that does not fault reads the same bytes.
// - a write range tests one byte of the range policy (set where the 64 KiB chunk or the next one needs the checked
//   path, so it covers any range of at most 64 KiB) and marks the range's dirty blocks once; its stores are then plain
//   byte-swapped stores. A null host sends every store of the range to WriteResolvedFallback (cold), which is what a
//   lean store does in a flagged chunk and, for plain RAM, ends in the same dirty mark and bytes. The marks precede the
//   stores with no snapshot point between (a call ends a segment); a block marked for a store that then lands in a
//   checked chunk is only copied by the next snapshot, unchanged. A straddling write range touches chunk 0x3FFF, which
//   is flagged: it never resolves.
// Other memory models: ResolveRangeHost and the resolved accessors above.
// ---------------------------------------------------------------------------------------------------------------------
#if MKW_LEAN_MEMORY
MKW_MEMORY_FORCE_INLINE void LeanMarkSpan(uint32_t canonical, uint32_t length) {
#if !MKW_LEAN_DIRTY_MARKS
    (void)canonical; (void)length;
    return;
#endif
    uint8_t* const marks = g_leanBase + kLeanDirtyOffset + (canonical >> 6);
    if (length <= 65) {  // at most two blocks: the per-store mark's 16-bit pair
        const uint16_t two = 0x0101;
        std::memcpy(marks, &two, sizeof(two));
    } else if (length <= 449) {  // at most eight blocks
        const uint64_t eight = 0x0101010101010101ull;
        std::memcpy(marks, &eight, sizeof(eight));
    } else {
        std::memset(marks, 1, ((canonical + length - 1) >> 6) - (canonical >> 6) + 1);
    }
}
template <typename T>
MKW_MEMORY_FORCE_INLINE T LeanSegmentLoad(uint8_t* host, uint32_t rangeOffset) {
    T value{};
    std::memcpy(&value, host + rangeOffset, sizeof(T));  // a read range is never null
    return MaybeByteSwap(value);
}
template <typename T>
MKW_MEMORY_FORCE_INLINE bool LeanSegmentStore(uint8_t* host, uint32_t rangeOffset, T value) {
    if (host == nullptr) [[unlikely]] return false;
    const T swapped = MaybeByteSwap(value);
    std::memcpy(host + rangeOffset, &swapped, sizeof(T));
    return true;
}
#endif
MKW_MEMORY_FORCE_INLINE uint8_t* ResolveSegmentRange(uint32_t base, int32_t minOffset, uint32_t length,
                                                    bool needsWrite) {
#if MKW_LEAN_MEMORY
    const uint32_t canonical = (base + static_cast<uint32_t>(minOffset)) & kLeanMask;
    if (needsWrite) {
        if (g_leanBase[kLeanRangePolicyOffset + (canonical >> kLeanPolicyShift)] != 0) [[unlikely]] return nullptr;
        LeanMarkSpan(canonical, length);
    }
    return g_leanBase + canonical;
#else
    return ResolveRangeHost(base, minOffset, length, needsWrite);
#endif
}
// A write range the translator proved lies in the stack (StackAddressFacts: r1 plus a constant, the proof that selects
// FlatStoreRam): a stack slot is never MMIO or an executable page and the stacks lie in MEM1/MEM2, so, like
// FlatStoreRam, it skips the policy byte and still marks. With no null arm the range is never null, and saying so lets
// the compiler drop each segment store's null test after it.
MKW_MEMORY_FORCE_INLINE uint8_t* ResolveSegmentStackRange(uint32_t base, int32_t minOffset, uint32_t length) {
#if MKW_LEAN_MEMORY
    const uint32_t canonical = (base + static_cast<uint32_t>(minOffset)) & kLeanMask;
    LeanMarkSpan(canonical, length);
    uint8_t* const host = g_leanBase + canonical;
#if defined(__clang__)
    __builtin_assume(host != nullptr);
#else
    if (host == nullptr) __builtin_unreachable();
#endif
    return host;
#else
    return ResolveRangeHost(base, minOffset, length, true);
#endif
}
#if MKW_LEAN_MEMORY
#define MKW_SEGMENT_READ(T, fallback) return LeanSegmentLoad<T>(r, o)
#define MKW_SEGMENT_WRITE(T, fallback)                          \
    if (LeanSegmentStore<T>(r, o, v)) return;                   \
    [[unlikely]] WriteResolvedFallback<T>(a, v)
#else
#define MKW_SEGMENT_READ(T, fallback) return fallback(r, o, a)
#define MKW_SEGMENT_WRITE(T, fallback) fallback(r, o, a, v)
#endif
MKW_MEMORY_FORCE_INLINE uint8_t SegRead8(uint8_t* r, uint32_t o, uint32_t a) { MKW_SEGMENT_READ(uint8_t, ReadResolved8); }
MKW_MEMORY_FORCE_INLINE uint16_t SegRead16(uint8_t* r, uint32_t o, uint32_t a) { MKW_SEGMENT_READ(uint16_t, ReadResolved16); }
MKW_MEMORY_FORCE_INLINE uint32_t SegRead32(uint8_t* r, uint32_t o, uint32_t a) { MKW_SEGMENT_READ(uint32_t, ReadResolved32); }
MKW_MEMORY_FORCE_INLINE uint64_t SegRead64(uint8_t* r, uint32_t o, uint32_t a) { MKW_SEGMENT_READ(uint64_t, ReadResolved64); }
MKW_MEMORY_FORCE_INLINE void SegWrite8(uint8_t* r, uint32_t o, uint32_t a, uint8_t v) { MKW_SEGMENT_WRITE(uint8_t, WriteResolved8); }
MKW_MEMORY_FORCE_INLINE void SegWrite16(uint8_t* r, uint32_t o, uint32_t a, uint16_t v) { MKW_SEGMENT_WRITE(uint16_t, WriteResolved16); }
MKW_MEMORY_FORCE_INLINE void SegWrite32(uint8_t* r, uint32_t o, uint32_t a, uint32_t v) { MKW_SEGMENT_WRITE(uint32_t, WriteResolved32); }
MKW_MEMORY_FORCE_INLINE void SegWrite64(uint8_t* r, uint32_t o, uint32_t a, uint64_t v) { MKW_SEGMENT_WRITE(uint64_t, WriteResolved64); }
#undef MKW_SEGMENT_READ
#undef MKW_SEGMENT_WRITE
MKW_MEMORY_FORCE_INLINE double SegReadFloat32(uint8_t* r, uint32_t o, uint32_t a) {
#if MKW_LEAN_MEMORY
    return PpcBitCastToDoubleInline(PpcConvertToDoubleBitsInline(SegRead32(r, o, a)));  // lfs's exact widening
#else
    return ReadResolvedFloat32(r, o, a);
#endif
}
MKW_MEMORY_FORCE_INLINE double SegReadFloat64(uint8_t* r, uint32_t o, uint32_t a) {
#if MKW_LEAN_MEMORY
    const uint64_t bits = SegRead64(r, o, a);
    double value; std::memcpy(&value, &bits, sizeof(value)); return value;
#else
    return ReadResolvedFloat64(r, o, a);
#endif
}
MKW_MEMORY_FORCE_INLINE void SegWriteFloat32(uint8_t* r, uint32_t o, uint32_t a, double v) {
#if MKW_LEAN_MEMORY
    if (LeanSegmentStore<uint32_t>(r, o, ConvertPpcDoubleToSingleBits(v))) return;
    [[unlikely]] WriteResolvedFallbackFloat32(a, v);  // the checked stfs (Memory::WriteFloat32)
#else
    WriteResolvedFloat32(r, o, a, v);
#endif
}
MKW_MEMORY_FORCE_INLINE void SegWriteFloat64(uint8_t* r, uint32_t o, uint32_t a, double v) {
#if MKW_LEAN_MEMORY
    uint64_t bits; std::memcpy(&bits, &v, sizeof(bits));
    if (LeanSegmentStore<uint64_t>(r, o, bits)) return;
    [[unlikely]] WriteResolvedFallback<uint64_t>(a, bits);
#else
    WriteResolvedFloat64(r, o, a, v);
#endif
}
// Pair forms: one double-width access off the range (two adjacent words, the first at the lower address).
MKW_MEMORY_FORCE_INLINE ResolvedLoadPair SegReadPair16(uint8_t* r, uint32_t o) {
#if MKW_LEAN_MEMORY
    const uint32_t packed = LeanSegmentLoad<uint32_t>(r, o);
    return {packed >> 16, packed & 0xFFFFu, true};
#else
    return ReadResolvedPair16(r, o);
#endif
}
MKW_MEMORY_FORCE_INLINE ResolvedLoadPair SegReadPair32(uint8_t* r, uint32_t o) {
#if MKW_LEAN_MEMORY
    const uint64_t packed = LeanSegmentLoad<uint64_t>(r, o);
    return {static_cast<uint32_t>(packed >> 32), static_cast<uint32_t>(packed), true};
#else
    return ReadResolvedPair32(r, o);
#endif
}
MKW_MEMORY_FORCE_INLINE bool SegWritePair16(uint8_t* r, uint32_t o, uint32_t packed) {
#if MKW_LEAN_MEMORY
    return LeanSegmentStore<uint32_t>(r, o, packed);
#else
    return WriteResolvedPair16(r, o, packed);
#endif
}
MKW_MEMORY_FORCE_INLINE bool SegWritePair32(uint8_t* r, uint32_t o, uint64_t packed) {
#if MKW_LEAN_MEMORY
    return LeanSegmentStore<uint64_t>(r, o, packed);
#else
    return WriteResolvedPair32(r, o, packed);
#endif
}

// Flat guest memory (audit item T-MEM): the 4 GiB reservation makes a guest access a byte swap
// around `*(T*)(base + addr)`, no page-table load or limit check (interception model documented
// in guest_flat_memory.h). The one exception kept inline is the MMIO write policy, since the
// written value can't be recovered from a fault record.
//
// When a host VM page is larger than a 4 KiB Wii page, guest-view protections
// cannot distinguish adjacent special Wii pages. The general FlatRead*/
// FlatWrite* helpers then use the checked page-table path, which applies executable-write/MMIO policy before touching RAM.
// FlatWriteRam* remains direct because the translator emits it only for
// addresses it has proven are ordinary RAM.

// Without the flat view (Horizon), the Flat* entry points below take the page table inline: a fully mapped
// 1 MiB page reads directly, a writable one stores directly (TryWriteGuestScalar keeps the 4 KiB executable-write
// guard and dirty marking), and only a miss (MMIO, unmapped, guarded) leaves for the checked Memory::* path.
// Calling out on every access made each guest load/store an out-of-line call plus spills around it.
#ifndef MKW_NONFLAT_INLINE
#define MKW_NONFLAT_INLINE 1  // 0: the old out-of-line behaviour, for A/B builds
#endif
#if MKW_LEAN_MEMORY
#define MKW_FLAT_CHECKED() false
#elif MKW_FLAT_MEMORY || !MKW_NONFLAT_INLINE
#define MKW_FLAT_CHECKED() GuestFlat::RequiresCheckedAccess()
#else
#define MKW_FLAT_CHECKED() false
#endif

template <typename T>
MKW_MEMORY_FORCE_INLINE T FlatLoad(uint32_t address) {
#if MKW_FLAT_MEMORY
    T value{};
    std::memcpy(&value, MKW_FLAT_GUEST_BASE + address, sizeof(T));
    return MaybeByteSwap(value);
#elif MKW_LEAN_MEMORY
    T value{};
    std::memcpy(&value, LeanHost(address), sizeof(T));
    return MaybeByteSwap(value);
#else
    T value{};
    if (MKW_NONFLAT_INLINE && TryReadGuestScalar<T>(address, value)) [[likely]] return value;
    return ReadResolvedFallback<T>(address);
#endif
}

#if MKW_LEAN_MEMORY
// The lean store, FlatStore's body: LeanStore8..64 (MKW_OUTLINE_GUEST_OPS) are this, compiled once.
template <typename T>
MKW_MEMORY_FORCE_INLINE void LeanStoreBody(uint32_t address, T value) {
    if (!LeanTryStore<T>(address, MaybeByteSwap(value))) [[unlikely]] WriteResolvedFallback<T>(address, value);
}
#endif

template <typename T>
MKW_MEMORY_FORCE_INLINE void FlatStore(uint32_t address, T value) {
#if MKW_FLAT_MEMORY
    BrawlDirtyMemory::Mark(address,sizeof(T));
    const T swapped = MaybeByteSwap(value);
    std::memcpy(MKW_FLAT_GUEST_BASE + address, &swapped, sizeof(T));
#elif MKW_LEAN_MEMORY && MKW_OUTLINE_GUEST_OPS
    if constexpr (sizeof(T) == 1) LeanStore8(address, static_cast<uint8_t>(value));
    else if constexpr (sizeof(T) == 2) LeanStore16(address, static_cast<uint16_t>(value));
    else if constexpr (sizeof(T) == 4) LeanStore32(address, static_cast<uint32_t>(value));
    else LeanStore64(address, static_cast<uint64_t>(value));
#elif MKW_LEAN_MEMORY
    LeanStoreBody<T>(address, value);
#else
    if (MKW_NONFLAT_INLINE && TryWriteGuestScalar<T>(address, value)) [[likely]] return;
    WriteResolvedFallback<T>(address, value);
#endif
}

MKW_MEMORY_FORCE_INLINE uint8_t FlatRead8(uint32_t address) {
    if (MKW_FLAT_CHECKED()) return Memory::Read8(address);
    return FlatLoad<uint8_t>(address);
}
MKW_MEMORY_FORCE_INLINE uint16_t FlatRead16(uint32_t address) {
    if (MKW_FLAT_CHECKED()) return Memory::Read16(address);
    return FlatLoad<uint16_t>(address);
}
MKW_MEMORY_FORCE_INLINE uint32_t FlatRead32(uint32_t address) {
    if (MKW_FLAT_CHECKED()) return Memory::Read32(address);
    return FlatLoad<uint32_t>(address);
}

// lfs: the load and the exact widening (LeanLoadFloat32 is this, compiled once).
MKW_MEMORY_FORCE_INLINE double FlatReadFloat32Body(uint32_t address) {
    return PpcBitCastToDoubleInline(PpcConvertToDoubleBitsInline(FlatRead32(address)));
}
MKW_MEMORY_FORCE_INLINE double FlatReadFloat32(uint32_t address) {
#if MKW_LEAN_MEMORY && MKW_OUTLINE_GUEST_OPS
    return LeanLoadFloat32(address);
#endif
    return FlatReadFloat32Body(address);
}

MKW_MEMORY_FORCE_INLINE double FlatReadFloat64(uint32_t address) {
    const uint64_t bits = MKW_FLAT_CHECKED()
        ? Memory::Read64(address) : FlatLoad<uint64_t>(address);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

MKW_MEMORY_FORCE_INLINE void FlatWrite8(uint32_t address, uint8_t value) {
    if (MKW_FLAT_CHECKED()) { Memory::Write8(address, value); return; }
    if (FlatWriteNeedsPolicy(address)) [[unlikely]] { Write8Slow(address, value); return; }
    FlatStore<uint8_t>(address, value);
}

MKW_MEMORY_FORCE_INLINE void FlatWrite16(uint32_t address, uint16_t value) {
    if (MKW_FLAT_CHECKED()) { Memory::Write16(address, value); return; }
    if (FlatWriteNeedsPolicy(address)) [[unlikely]] { Write16Slow(address, value); return; }
    FlatStore<uint16_t>(address, value);
}

MKW_MEMORY_FORCE_INLINE void FlatWrite32(uint32_t address, uint32_t value) {
    if (MKW_FLAT_CHECKED()) { Memory::Write32(address, value); return; }
    if (FlatWriteNeedsPolicy(address)) [[unlikely]] { Write32Slow(address, value); return; }
    FlatStore<uint32_t>(address, value);
}


MKW_MEMORY_FORCE_INLINE void FlatWriteFloat32(uint32_t address, double value) {
#if MKW_LEAN_MEMORY && MKW_OUTLINE_GUEST_OPS
    LeanStoreFloat32(address, value);  // the conversion and FlatStore's lean body below, compiled once
    return;
#endif
    if (MKW_FLAT_CHECKED()) { Memory::WriteFloat32(address, value); return; }
    const uint32_t bits = ConvertPpcDoubleToSingleBits(value);
    if (FlatWriteNeedsPolicy(address)) [[unlikely]] { WriteFloat32Slow(address, value); return; }
    FlatStore<uint32_t>(address, bits);
}

MKW_MEMORY_FORCE_INLINE void FlatWriteFloat64(uint32_t address, double value) {
    if (MKW_FLAT_CHECKED()) { Memory::WriteFloat64(address, value); return; }
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if (FlatWriteNeedsPolicy(address)) [[unlikely]] { WriteFloat64Slow(address, value); return; }
    FlatStore<uint64_t>(address, bits);
}

// Check-free stores: emitted ONLY for addresses the translator proved at translate time are ordinary guest RAM (r1-relative stack slots, ~45%
// of flat stores), skipping the MMIO mask/compare that's pure overhead there. Still safe if that proof were ever wrong: the flat view maps
// 0xCC000000..0xCDFFFFFF PAGE_NOACCESS, so a stray MMIO store faults into the same handler and diagnostic as the checked path, just reported
// instead of dispatched inline. Never use these for an address the translator hasn't proven.

// Lean memory has no fault to fall back on, so it relies on the proof directly: a stack slot is never MMIO or an
// executable page, so the store skips the policy byte. It still marks: a stack may lie inside rollback's ranges.
#if MKW_LEAN_MEMORY
// FlatStoreRam's lean body (LeanStoreRam8..64 are this, compiled once).
template <typename T>
MKW_MEMORY_FORCE_INLINE void LeanStoreRamBody(uint32_t address, T value) {
    uint8_t* const base = g_leanBase;
    const uint32_t canonical = address & kLeanMask;
#if MKW_LEAN_DIRTY_MARKS
    const uint16_t both = 0x0101;
    std::memcpy(base + kLeanDirtyOffset + (canonical >> 6), &both, sizeof(both));
#endif
    const T swapped = MaybeByteSwap(value);
    std::memcpy(base + canonical, &swapped, sizeof(T));
}
#endif

template <typename T>
MKW_MEMORY_FORCE_INLINE void FlatStoreRam(uint32_t address, T value) {
#if MKW_LEAN_MEMORY && MKW_OUTLINE_GUEST_OPS
    if constexpr (sizeof(T) == 1) LeanStoreRam8(address, static_cast<uint8_t>(value));
    else if constexpr (sizeof(T) == 2) LeanStoreRam16(address, static_cast<uint16_t>(value));
    else if constexpr (sizeof(T) == 4) LeanStoreRam32(address, static_cast<uint32_t>(value));
    else LeanStoreRam64(address, static_cast<uint64_t>(value));
#elif MKW_LEAN_MEMORY
    LeanStoreRamBody<T>(address, value);
#else
    FlatStore<T>(address, value);
#endif
}

MKW_MEMORY_FORCE_INLINE void FlatWriteRam8(uint32_t address, uint8_t value) {
    FlatStoreRam<uint8_t>(address, value);
}

MKW_MEMORY_FORCE_INLINE void FlatWriteRam16(uint32_t address, uint16_t value) {
    FlatStoreRam<uint16_t>(address, value);
}

MKW_MEMORY_FORCE_INLINE void FlatWriteRam32(uint32_t address, uint32_t value) {
    FlatStoreRam<uint32_t>(address, value);
}


MKW_MEMORY_FORCE_INLINE void FlatWriteRamFloat32(uint32_t address, double value) {
#if MKW_LEAN_MEMORY && MKW_OUTLINE_GUEST_OPS
    LeanStoreRamFloat32(address, value);  // this conversion and store, compiled once
    return;
#endif
    FlatStoreRam<uint32_t>(address, ConvertPpcDoubleToSingleBits(value));
}

MKW_MEMORY_FORCE_INLINE void FlatWriteRamFloat64(uint32_t address, double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    FlatStoreRam<uint64_t>(address, bits);
}

} // namespace MemoryInline

#undef MKW_MEMORY_FORCE_INLINE
#undef MKW_MEMORY_NO_INLINE
#undef MKW_MEMORY_COLD

inline uint8_t Memory::Read8(uint32_t addr) {
    uint8_t value = 0;
    if (!MemoryInline::TryReadGuestScalar(addr, value)) [[unlikely]]
        return MemoryInline::Read8Slow(addr);
    return value;
}

inline uint16_t Memory::Read16(uint32_t addr) {
    uint16_t value = 0;
    if (!MemoryInline::TryReadGuestScalar(addr, value)) [[unlikely]]
        return MemoryInline::Read16Slow(addr);
    return value;
}

inline uint32_t Memory::Read32(uint32_t addr) {
    uint32_t value = 0;
    if (!MemoryInline::TryReadGuestScalar(addr, value)) [[unlikely]]
        return MemoryInline::Read32Slow(addr);
    return value;
}

inline uint64_t Memory::Read64(uint32_t addr) {
    uint64_t value = 0;
    if (!MemoryInline::TryReadGuestScalar(addr, value)) [[unlikely]]
        return MemoryInline::Read64Slow(addr);
    return value;
}

inline float Memory::ReadFloat32(uint32_t addr) {
    uint32_t bits = 0;
    if (!MemoryInline::TryReadGuestScalar(addr, bits)) [[unlikely]]
        return MemoryInline::ReadFloat32Slow(addr);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline double Memory::ReadFloat64(uint32_t addr) {
    uint64_t bits = 0;
    if (!MemoryInline::TryReadGuestScalar(addr, bits)) [[unlikely]]
        return MemoryInline::ReadFloat64Slow(addr);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline void Memory::Write8(uint32_t addr, uint8_t val) {
    if (!MemoryInline::TryWriteGuestScalar(addr, val)) [[unlikely]]
        MemoryInline::Write8Slow(addr, val);
}

inline void Memory::Write16(uint32_t addr, uint16_t val) {
    if (!MemoryInline::TryWriteGuestScalar(addr, val)) [[unlikely]]
        MemoryInline::Write16Slow(addr, val);
}

inline void Memory::Write32(uint32_t addr, uint32_t val) {
    if (!MemoryInline::TryWriteGuestScalar(addr, val)) [[unlikely]]
        MemoryInline::Write32Slow(addr, val);
}

inline void Memory::Write64(uint32_t addr, uint64_t val) {
    if (!MemoryInline::TryWriteGuestScalar(addr, val)) [[unlikely]]
        MemoryInline::Write64Slow(addr, val);
}

inline void Memory::WriteFloat32(uint32_t addr, double val) {
    {
        const uint32_t bits = MemoryInline::ConvertPpcDoubleToSingleBits(val);
        if (MemoryInline::TryWriteGuestScalar(addr, bits))
            return;
    }
    [[unlikely]] MemoryInline::WriteFloat32Slow(addr, val);
}

inline void Memory::WriteFloat64(uint32_t addr, double val) {
    {
        uint64_t bits = 0;
        std::memcpy(&bits, &val, sizeof(bits));
        if (MemoryInline::TryWriteGuestScalar(addr, bits))
            return;
    }
    [[unlikely]] MemoryInline::WriteFloat64Slow(addr, val);
}
