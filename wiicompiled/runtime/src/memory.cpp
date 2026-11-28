#include "memory.h"
#include "hz_perf.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <optional>
#include "ppc_runtime.h"
#include "runtime_log.h"
#include "system_bridge.h"
#include <mutex>
#include <sstream>
#include <unordered_map>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#endif
#if MKW_LEAN_MEMORY
#include "lean_window.h"  // the window's address space and its backing, per OS
#endif

MemoryInline::PageEntry MemoryInline::g_pageTable[MemoryInline::kPageCount]{};
uintptr_t MemoryInline::g_fullPageBias[MemoryInline::kPageCount]{};
uintptr_t MemoryInline::g_fullWritablePageBias[MemoryInline::kPageCount]{};
const MemoryInline::SparseWritablePageTable*
    MemoryInline::g_sparseWritablePageTables[MemoryInline::kPageCount]{};
template <typename T>
T MemoryInline::ReadResolvedFallback(uint32_t addr) {
    HZ_PERF_ADD(kHzFallbackReads, 1);
    if constexpr (sizeof(T) == 1) return Memory::Read8(addr);
    if constexpr (sizeof(T) == 2) return Memory::Read16(addr);
    if constexpr (sizeof(T) == 4) return Memory::Read32(addr);
    return Memory::Read64(addr);
}
template uint8_t MemoryInline::ReadResolvedFallback<uint8_t>(uint32_t);
template uint16_t MemoryInline::ReadResolvedFallback<uint16_t>(uint32_t);
template uint32_t MemoryInline::ReadResolvedFallback<uint32_t>(uint32_t);
template uint64_t MemoryInline::ReadResolvedFallback<uint64_t>(uint32_t);
float MemoryInline::ReadResolvedFallbackFloat32(uint32_t addr) { return Memory::ReadFloat32(addr); }
double MemoryInline::ReadResolvedFallbackFloat64(uint32_t addr) { return Memory::ReadFloat64(addr); }
#if defined(__SWITCH__)
// Fallback writes per guest 4 KiB page, dumped (top pages) and cleared by the Horizon sampler: which pages keep
// stores off the resolved fast path.
static uint32_t g_hzFallbackPageWrites[1u << 20];
extern "C" void HzDumpFallbackPages(void) {
    uint32_t topPage[12] = {}, topCount[12] = {};
    for (uint32_t page = 0; page < (1u << 20); ++page) {
        const uint32_t count = g_hzFallbackPageWrites[page];
        if (count == 0) continue;
        g_hzFallbackPageWrites[page] = 0;
        if (count <= topCount[11]) continue;
        int slot = 11;
        while (slot > 0 && topCount[slot - 1] < count) {
            topCount[slot] = topCount[slot - 1];
            topPage[slot] = topPage[slot - 1];
            --slot;
        }
        topCount[slot] = count;
        topPage[slot] = page;
    }
    char line[512];
    int used = std::snprintf(line, sizeof line, "[horizon] fallback-write pages:");
    for (int i = 0; i < 12 && topCount[i] != 0 && used < 480; ++i)
        used += std::snprintf(line + used, sizeof line - used, " %08X=%u%s", topPage[i] << 12, topCount[i],
                              RecompMod::g_executableWriteGuardPages[topPage[i]].load(std::memory_order_relaxed)
                                  ? "x" : "");
    std::fprintf(stderr, "%s\n", line);
    if (topCount[0] != 0) {
        // Why the inline store path misses the hottest page.
        const uint32_t address = topPage[0] << 12;
        const uint32_t coarse = address >> MemoryInline::kPageShift;
        const auto& entry = MemoryInline::g_pageTable[coarse];
        const auto* sub = MemoryInline::g_sparseWritablePageTables[coarse];
        std::fprintf(stderr,
                     "[horizon] fallback top page %08X: fullBias=%llx writableBias=%llx sparse=%p subBias=%llx "
                     "entryBase=%p limit=%x guard=%d\n",
                     address, (unsigned long long)MemoryInline::g_fullPageBias[coarse],
                     (unsigned long long)MemoryInline::g_fullWritablePageBias[coarse], (const void*)sub,
                     sub ? (unsigned long long)sub->encodedBias[(address & MemoryInline::kPageMask) >>
                                                              MemoryInline::kWritableSubPageShift]
                         : 0ull,
                     (const void*)entry.base, (unsigned)entry.limit,
                     RecompMod::g_executableWriteGuardPages[topPage[0]].load() ? 1 : 0);
    }
    std::fflush(stderr);
}
#endif
#if MKW_FALLBACK_STATS
#include <execinfo.h>
// Debug counters (-DMKW_FALLBACK_STATS=1): translated stores the lean policy sends to the checked path, by 4 KiB page;
// every 2^22 of them, the 16 pages with the most ([fallback] lines on stderr, with whether the page holds guarded
// code). Plain increments: a lost count under a race is fine here.
namespace {
uint32_t g_fallbackPageStores[1u << 20];
uint64_t g_fallbackStores = 0;
void CountFallbackStore(uint32_t addr) {
    ++g_fallbackPageStores[addr >> 12];
    if ((++g_fallbackStores & ((1u << 22) - 1)) != 0) return;
    std::vector<std::pair<uint32_t, uint32_t>> top;
    for (uint32_t page = 0; page < (1u << 20); ++page)
        if (g_fallbackPageStores[page] != 0) top.push_back({g_fallbackPageStores[page], page});
    std::sort(top.begin(), top.end(), std::greater<>());
    std::fprintf(stderr, "[fallback] %llu checked-path stores so far; top pages of the last %u:",
                 (unsigned long long)g_fallbackStores, 1u << 22);
    for (size_t k = 0; k < top.size() && k < 16; ++k) {
        const uint32_t page = top[k].second;  // a 4 KiB page, as the guard table's index
        std::fprintf(stderr, " %08X:%u%s", page << 12, top[k].first,
                     RecompMod::g_executableWriteGuardPages[page].load(std::memory_order_relaxed) ? "(code)" : "");
    }
    std::fprintf(stderr, "\n");
    std::memset(g_fallbackPageStores, 0, sizeof(g_fallbackPageStores));
}
} // namespace
#endif
#if MKW_LEAN_MEMORY
namespace {
// BRAWL_LEAN_FALLBACK_FAST=0: every store into a flagged chunk takes the checked path (the fast path's A/B switch).
const bool g_leanFallbackFast = [] {
    const char* value = std::getenv("BRAWL_LEAN_FALLBACK_FAST");
    return value == nullptr || value[0] != '0';
}();
} // namespace
#endif
template <typename T>
void MemoryInline::WriteResolvedFallback(uint32_t addr, T value) {
    HZ_PERF_ADD(kHzFallbackWrites, 1);
#if MKW_LEAN_MEMORY
    // The lean policy flags a whole 64 KiB chunk when any 4 KiB page in it holds guarded code (RefreshLeanStorePolicy),
    // so a chunk shared by code and data sends its data stores here too: Project+'s Syriinge trampolines at 0x8105E000
    // flag the game heap buffer at 0x8105A000-0x8105D000 that memset clears every frame (~0.6% of the game thread).
    // For a cached-view MEM1/MEM2 store whose pages hold no guarded code, the checked path below ends in exactly the
    // lean store (WriteScalar: no-op test, dirty mark, HandleExecutableWrite finds no guard, the bytes), so do that
    // here: dirty marks and the bytes, as LeanTryStore does for an unflagged chunk.
    {
        const uint32_t offset = addr & 0x0FFFFFFFu;
        const uint32_t top = addr >> 28;
        const uint32_t last = addr + static_cast<uint32_t>(sizeof(T)) - 1u;
        if (g_leanFallbackFast &&
            ((top == 0x8u && offset + sizeof(T) <= Memory::kMem1BankSize) ||
             (top == 0x9u && offset + sizeof(T) <= Memory::kMem2Size)) &&
            RecompMod::g_executableWriteGuardPages[addr >> RecompMod::kExecutableWriteGuardPageShift].load(
                std::memory_order_relaxed) == 0 &&
            RecompMod::g_executableWriteGuardPages[last >> RecompMod::kExecutableWriteGuardPageShift].load(
                std::memory_order_relaxed) == 0) {
            BrawlDirtyMemory::Mark(addr, sizeof(T));
            const T swapped = MaybeByteSwap(value);
            std::memcpy(LeanHost(addr), &swapped, sizeof(T));
            return;
        }
    }
#endif
#if MKW_FALLBACK_STATS
    CountFallbackStore(addr);  // the stores that still take the checked path
#endif
#if defined(__SWITCH__)
    ++g_hzFallbackPageWrites[addr >> 12];
#endif
    if constexpr (sizeof(T) == 1) Memory::Write8(addr, value);
    else if constexpr (sizeof(T) == 2) Memory::Write16(addr, value);
    else if constexpr (sizeof(T) == 4) Memory::Write32(addr, value);
    else Memory::Write64(addr, value);
}
template void MemoryInline::WriteResolvedFallback<uint8_t>(uint32_t, uint8_t);
template void MemoryInline::WriteResolvedFallback<uint16_t>(uint32_t, uint16_t);
template void MemoryInline::WriteResolvedFallback<uint32_t>(uint32_t, uint32_t);
template void MemoryInline::WriteResolvedFallback<uint64_t>(uint32_t, uint64_t);
#if MKW_LEAN_MEMORY
bool MemoryInline::LeanStoresArePlain(uint32_t addr, uint32_t size) {
    if (size == 0) return true;
    const uint32_t last = addr + (size - 1u);
    if (last < addr || ((addr ^ last) & ~kLeanMask) != 0) return false;  // one view of the window
    const uint32_t c = addr & kLeanMask;
    if (static_cast<uint64_t>(c) + size > kLeanWindowSize) return false;
    const uint8_t* const policy = g_leanBase + kLeanPolicyOffset;
    bool flagged = false;
    for (uint32_t chunk = c >> kLeanPolicyShift; chunk <= (c + (size - 1u)) >> kLeanPolicyShift; ++chunk)
        flagged |= policy[chunk] != 0;
    if (!flagged) return true;
    // WriteResolvedFallback's fast exit, for every store of the range at once.
    const uint32_t top = addr >> 28, offset = addr & 0x0FFFFFFFu;
    if (!g_leanFallbackFast ||
        !((top == 0x8u && offset + static_cast<uint64_t>(size) <= Memory::kMem1BankSize) ||
          (top == 0x9u && offset + static_cast<uint64_t>(size) <= Memory::kMem2Size)))
        return false;
    for (uint32_t page = addr >> RecompMod::kExecutableWriteGuardPageShift;
         page <= last >> RecompMod::kExecutableWriteGuardPageShift; ++page)
        if (RecompMod::g_executableWriteGuardPages[page].load(std::memory_order_relaxed) != 0) return false;
    return true;
}
#endif
void MemoryInline::WriteResolvedFallbackFloat32(uint32_t addr, double val) { Memory::WriteFloat32(addr, val); }
void MemoryInline::WriteResolvedFallbackFloat64(uint32_t addr, double val) { Memory::WriteFloat64(addr, val); }
namespace {
struct Region {
    Memory::RegionConfig config;
    std::shared_ptr<std::vector<uint8_t>> storage;
    uint8_t* storagePtr = nullptr;
    size_t storageSize = 0;
};

std::vector<Region>& Regions() {
    static std::vector<Region> regions;
    return regions;
}

std::mutex& RegionMutex() {
    static std::mutex mutex;
    return mutex;
}

std::array<std::unique_ptr<MemoryInline::SparseWritablePageTable>,
           MemoryInline::kPageCount>& SparseWritablePageTableStorage() {
    static std::array<std::unique_ptr<MemoryInline::SparseWritablePageTable>,
                      MemoryInline::kPageCount> storage;
    return storage;
}

void ClearPageTable() {
    auto* table = MemoryInline::g_pageTable;
    for (uint32_t i = 0; i < MemoryInline::kPageCount; ++i) {
        auto& entry = table[i];
        entry.base = nullptr;
        entry.limit = 0;
        MemoryInline::g_fullPageBias[i] = 0;
        MemoryInline::g_fullWritablePageBias[i] = 0;
        MemoryInline::g_sparseWritablePageTables[i] = nullptr;
        SparseWritablePageTableStorage()[i].reset();
    }
}

} // namespace

namespace {

void BuildPageTable() {
    auto* table = MemoryInline::g_pageTable;
    for (const auto& region : Regions()) {
        const uint64_t base = region.config.baseAddress;
        const uint64_t size = region.storageSize;
        const uint64_t end = base + size;
        if (size == 0) {
            continue;
        }

        const uint32_t startPage = static_cast<uint32_t>(base >> MemoryInline::kPageShift);
        const uint32_t endPage = static_cast<uint32_t>((end - 1) >> MemoryInline::kPageShift);
        for (uint32_t page = startPage; page <= endPage; ++page) {
            const uint64_t pageBase = static_cast<uint64_t>(page) << MemoryInline::kPageShift;
            const uint64_t offset = pageBase - base;
            if (offset >= size) {
                continue;
            }
            const uint32_t limit = static_cast<uint32_t>(std::min<uint64_t>(MemoryInline::kPageSize, size - offset));
            table[page].base = region.storagePtr + static_cast<size_t>(offset);
            table[page].limit = limit;
        }
    }

    // The last page of each contiguous mapping remains on the checked path so
    // an access straddling its end cannot escape the mapped region. Every
    // preceding full page can safely service native accesses up to 8 bytes,
    // including a cross-page access into its contiguous successor.
    for (uint32_t page = 0; page + 1 < MemoryInline::kPageCount; ++page) {
        const auto& current = table[page];
        const auto& next = table[page + 1];
        if (!current.base || current.limit != MemoryInline::kPageSize ||
            !next.base || next.limit < MemoryInline::kMaxFastScalarSize - 1u ||
            next.base != current.base + MemoryInline::kPageSize)
            continue;
        const uintptr_t guestPageBase = static_cast<uintptr_t>(page) << MemoryInline::kPageShift;
        const uintptr_t bias = reinterpret_cast<uintptr_t>(current.base) - guestPageBase;
        MemoryInline::g_fullPageBias[page] = bias + 1u;
    }
}

#if MKW_LEAN_MEMORY
// Where the regions put each bank in the lean window: the window offsets of its copies (one per region, guest address &
// kLeanMask, duplicates folded) and a copy's size. Config::WiiDefaults gives one region per bank copy of every view of
// guest_memory_map.h's table, so MEM1's views fold onto 0x00000000, 0x02000000, 0x04000000 and 0x06000000, MEM2's
// onto 0x10000000 and the locked cache onto 0x20000000. A region outside the table is a bank of its own.
struct LeanBankCopies {
    std::string name;
    uint64_t size = 0;
    std::vector<uint64_t> offsets;
};
std::vector<LeanBankCopies> g_leanBankCopies;  // Memory::Init's; RefreshLeanStorePolicy reads it

// The lean store policy (memory_access.h): one byte per 64 KiB of the window, set where a guest store must take
// the checked Memory::Write* path. Every chunk not wholly inside a guest region, so a store no region maps raises
// the checked path's AccessViolation, as without the window (most of those chunks have no backing at all:
// BackLeanWindow); the MMIO window, so GX FIFO and register writes reach their handlers; every chunk holding an
// executable 4 KiB page, so writes into code still reach GuestCodePatch. Indexing by the masked address makes every
// view (physical, cached, uncached) of a page share its flag.
void RefreshLeanStorePolicy() {
    constexpr uint64_t kChunks = MemoryInline::kLeanWindowSize >> MemoryInline::kLeanPolicyShift;
    uint8_t* const policy = MemoryInline::LeanBaseAtInit() + MemoryInline::kLeanPolicyOffset;
    // The range policy (memory_access.h, kLeanRangePolicyOffset) reads "checked" while the store policy changes.
    uint8_t* const rangePolicy = MemoryInline::LeanBaseAtInit() + MemoryInline::kLeanRangePolicyOffset;
    std::memset(rangePolicy, 1, kChunks);
    std::memset(policy, 1, kChunks);
    for (const auto& region : Regions()) {
        const uint64_t start = region.config.baseAddress & MemoryInline::kLeanMask;
        const uint64_t end = start + region.storageSize;
        constexpr uint64_t kChunk = uint64_t{1} << MemoryInline::kLeanPolicyShift;
        for (uint64_t chunk = (start + kChunk - 1) >> MemoryInline::kLeanPolicyShift;
             chunk < (end >> MemoryInline::kLeanPolicyShift); ++chunk)
            policy[chunk] = 0;
    }
    const auto mark = [&](uint32_t address) {
        policy[(address & MemoryInline::kLeanMask) >> MemoryInline::kLeanPolicyShift] = 1;
    };
    for (uint64_t address = 0xCC000000u; address < 0xCE000000u; address += 1u << MemoryInline::kLeanPolicyShift)
        mark(static_cast<uint32_t>(address));
    // The runtime's own tables in the window's top 32 MiB (dirty marks, this policy): no guest store lands there.
    for (uint64_t address = MemoryInline::kLeanDirtyOffset; address < MemoryInline::kLeanWindowSize;
         address += 1u << MemoryInline::kLeanPolicyShift)
        mark(static_cast<uint32_t>(address));
    for (uint32_t coarse = 0; coarse < MemoryInline::kPageCount; ++coarse) {
        if (RecompMod::g_executableWriteGuardCoarsePages[coarse].load(std::memory_order_relaxed) == 0)
            continue;
        const uint32_t first = coarse * MemoryInline::kWritableSubPagesPerPage;
        for (uint32_t sub = 0; sub < MemoryInline::kWritableSubPagesPerPage; ++sub)
            if (RecompMod::g_executableWriteGuardPages[first + sub].load(std::memory_order_relaxed) != 0)
                mark((first + sub) << MemoryInline::kWritableSubPageShift);
    }
    // A bank's copies (MEM1's mirrors) share their flags: a store to code through any copy takes the checked path,
    // where WriteScalar hands GuestCodePatch the code's own address (GuestMemoryMap::Canonical).
    for (const LeanBankCopies& bank : g_leanBankCopies) {
        if (bank.offsets.size() < 2) continue;
        constexpr uint32_t kShift = MemoryInline::kLeanPolicyShift;
        for (uint64_t chunk = 0; chunk < (bank.size >> kShift); ++chunk) {
            uint8_t flags = 0;
            for (const uint64_t offset : bank.offsets) flags |= policy[(offset >> kShift) + chunk];
            for (const uint64_t offset : bank.offsets) policy[(offset >> kShift) + chunk] = flags;
        }
    }
    // A resolved write range of at most 64 KiB starting in chunk c lies in c and c + 1 (the last chunk's next one is
    // past the window), so its one test reads both.
    for (uint64_t chunk = 0; chunk + 1 < kChunks; ++chunk)
        rangePolicy[chunk] = static_cast<uint8_t>(policy[chunk] | policy[chunk + 1]);
}

// The window is reserved address space on every OS (Memory::Init), which backs only what is in use, the same ranges
// everywhere: each guest region, its slice of the dirty map, the store policy, and the whole MMIO window. Horizon has no lazily backed anonymous memory; backing the same ranges on Linux and Windows makes
// them fault wherever the Switch does. The MMIO backing stays zero (every store there takes the checked path), so a
// guest load there reads zero, as without the window (Read*Slow: nothing on the bus outside the modeled devices) and
// on hardware.
// Correct guest code does load there: P+'s pad hook (0x800489F8) dereferences a pointer slot in sora_melee's bss
// (0x80B84EB0) whenever it is >= 0x80000000, and while that module loads the slot holds the game's 0xCC heap fill, so
// the hook reads 0xCCCCCD14 every frame until the module links (a few frames on Horizon, many on a Wii's disc).
// Nothing else is mapped. A guest store there takes the checked path (RefreshLeanStorePolicy) and raises its
// AccessViolation, as without the window. A guest load there faults and ends the process (so does a store the
// translator proved to be a stack slot, which skips the policy byte); without the window the same access (no region
// maps it) raised an AccessViolation, which is fatal from translated code as well. So a run that hits no fatal error
// without the window touches only backed bytes with it.
uint64_t g_leanBackedBytes = 0;
// The window's 64 KiB chunks already backed (aliased views share their backing): BackLeanWindow's, and MEM1's and its
// mirrors' shared views (MapLeanBanks).
uint8_t g_leanChunkBacked[MemoryInline::kLeanWindowSize >> MemoryInline::kLeanPolicyShift];
void BackLeanWindow(uint64_t offset, uint64_t size) {
    constexpr uint32_t kShift = MemoryInline::kLeanPolicyShift;  // backing granularity: one policy chunk, 64 KiB
    uint8_t* const backed = g_leanChunkBacked;
    const uint64_t end = (offset + size + (uint64_t{1} << kShift) - 1) >> kShift;
    for (uint64_t first = offset >> kShift; first < end;) {
        uint64_t last = first;
        while (last < end && !backed[last]) ++last;
        if (last == first) {
            ++first;
            continue;
        }
        uint8_t* const at = MemoryInline::LeanBaseAtInit() + (first << kShift);
        const uint64_t bytes = (last - first) << kShift;
        const uint32_t rc = LeanWindow::Commit(at, bytes);
        if (rc != 0) {
            std::ostringstream message;
            message << "lean memory: could not back window bytes 0x" << std::hex << (first << kShift) << "+0x" << bytes
                    << " (error 0x" << rc << ")";
            throw std::runtime_error(message.str());
        }
        std::memset(backed + first, 1, last - first);
        g_leanBackedBytes += bytes;
        first = last;
    }
}
#endif

void RefreshFastPathTables() {
#if MKW_LEAN_MEMORY
    RefreshLeanStorePolicy();
#endif
    auto& sparseStorage = SparseWritablePageTableStorage();
    for (uint32_t page = 0; page < MemoryInline::kPageCount; ++page) {
        const uintptr_t bias = MemoryInline::g_fullPageBias[page];
        MemoryInline::g_sparseWritablePageTables[page] = nullptr;
        sparseStorage[page].reset();

        // Executable ranges are registered at their bank's primary view (guest_memory_map.h): every other view of
        // the same bytes (MEM1's mirrors, the uncached and physical views) takes the primary page's classification.
        const uint32_t guardPage =
            GuestMemoryMap::Canonical(page << MemoryInline::kPageShift) >> MemoryInline::kPageShift;
        const bool coarseExecutable =
            RecompMod::g_executableWriteGuardCoarsePages[guardPage].load(
                std::memory_order_relaxed) != 0;
        MemoryInline::g_fullWritablePageBias[page] = coarseExecutable ? 0 : bias;
        if (!coarseExecutable || bias == 0)
            continue;

        auto sparse = std::make_unique<MemoryInline::SparseWritablePageTable>();
        bool hasExecutableSubPage = false;
        const uint32_t firstGuardPage =
            guardPage * MemoryInline::kWritableSubPagesPerPage;
        for (uint32_t subPage = 0;
             subPage < MemoryInline::kWritableSubPagesPerPage; ++subPage) {
            const bool executable =
                RecompMod::g_executableWriteGuardPages[firstGuardPage + subPage].load(
                    std::memory_order_relaxed) != 0;
            hasExecutableSubPage |= executable;
            sparse->encodedBias[subPage] = executable ? 0 : bias;
        }

        // A page holding code keeps its table even when every sub-page is code: the fast path's last check reads
        // the exact 4 KiB guard at the store's own address, which for another view of the code (a MEM1 mirror) is
        // not set, so only the table's zero biases keep such a store off the native path. A homogeneous data page
        // already uses g_fullWritablePageBias.
        if (hasExecutableSubPage) {
            MemoryInline::g_sparseWritablePageTables[page] = sparse.get();
            sparseStorage[page] = std::move(sparse);
        }
    }
}

Region* TryResolveRegion(uint32_t address, size_t length) {
    auto& regions = Regions();
    for (auto& region : regions) {
        const uint64_t base = region.config.baseAddress;
        const uint64_t limit = base + region.storageSize;
        const uint64_t addr = address;
        if (addr >= base && addr <= limit && length <= limit - addr) {
            return &region;
        }
    }
    return nullptr;
}

Region& ResolveRegion(uint32_t address, size_t length) {
    if (Region* region = TryResolveRegion(address, length)) {
        return *region;
    }
    // Emit extra context to help diagnose early-boot accesses that miss the map.
    if (auto* cpu = TryGetCpuContext()) {
        const uint32_t active = RecompMod::CurrentTranslatedExecutionAddress();
        RT_LOG(RT_TAG_MEMORY) << "AccessViolation ctx=" << cpu
                  << " pc=0x" << std::hex << cpu->pc
                  << " active=0x" << active
                  << " lr=0x" << cpu->lr
                  << " r1=0x" << cpu->gpr[1]
                  << " r8=0x" << cpu->gpr[8]
                  << " r9=0x" << cpu->gpr[9]
                  << " r12=0x" << cpu->gpr[12]
                  << " r30=0x" << cpu->gpr[30]
                  << " r31=0x" << cpu->gpr[31]
                  << std::dec
                  << " addr=0x" << std::hex << address
                  << " len=" << length << std::dec
                  << " reason=no mapped region" << std::endl;
    } else {
        RT_LOG(RT_TAG_MEMORY) << "AccessViolation ctx=(null) addr=0x" << std::hex << address
                  << " len=" << length << std::dec << " reason=no mapped region" << std::endl;
    }
    RT_LOG(RT_TAG_MEMORY) << "===== DUMPING CPU STATE =====" << std::endl;
    SystemBridge::DumpCpuState(TryGetCpuContext());
    // Hexdump guest memory around pointer-carrying registers so a corrupted
    // structure's surroundings (e.g. ASCII sprayed over a link pointer) are
    // visible in the report without a debugger attached.
    if (auto* cpu = TryGetCpuContext()) {
        for (const int reg : {4, 5, 6, 7, 8, 26, 27, 28, 29, 30, 31}) {
            const uint32_t base = cpu->gpr[reg];
            if (base < 0x80000000u || base >= 0x94000000u)
                continue;
            const uint32_t start = (base - 0x40u) & ~0xFu;
            RT_LOG(RT_TAG_MEMORY) << "hexdump around r" << reg << "=0x" << std::hex << base << ":" << std::endl;
            for (uint32_t row = 0; row < 16; ++row) {
                const uint32_t rowAddr = start + row * 16u;
                RT_LOG(RT_TAG_MEMORY) << "  0x" << std::hex << rowAddr << ":";
                char ascii[17] = {};
                for (uint32_t i = 0; i < 16; ++i) {
                    uint8_t byte = 0;
                    if (!MemoryInline::TryReadGuestScalar(rowAddr + i, byte)) {
                        std::cerr << " ??";
                        ascii[i] = '?';
                        continue;
                    }
                    std::cerr << " " << std::setw(2) << std::setfill('0') << static_cast<uint32_t>(byte);
                    ascii[i] = (byte >= 0x20 && byte < 0x7F) ? static_cast<char>(byte) : '.';
                }
                std::cerr << "  |" << ascii << "|" << std::dec << std::setfill(' ') << std::endl;
            }
        }
    }
#if defined(_WIN32)
    void* frames[32]{};
    const USHORT captured = CaptureStackBackTrace(0, static_cast<DWORD>(std::size(frames)), frames, nullptr);
    const auto imageBase = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
    HANDLE process = GetCurrentProcess();
    static const bool symbolsReady = [] {
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        if (SymInitialize(GetCurrentProcess(), nullptr, TRUE) != FALSE)
            return true;
        // The runtime crash reporter may already own the process-wide DbgHelp
        // session. Reuse it rather than treating ERROR_INVALID_PARAMETER as
        // symbol unavailability.
        return GetLastError() == ERROR_INVALID_PARAMETER;
    }();
    RT_LOG(RT_TAG_MEMORY) << "host stack at first invalid guest access:" << std::endl;
    for (USHORT index = 0; index < captured; ++index) {
        const auto addressValue = reinterpret_cast<uintptr_t>(frames[index]);
        RT_LOG(RT_TAG_MEMORY) << "  #" << index << " absolute=0x" << std::hex << addressValue;
        if (imageBase != 0 && addressValue >= imageBase)
            std::cerr << " image+0x" << (addressValue - imageBase);
        std::array<char, sizeof(SYMBOL_INFO) + MAX_SYM_NAME> symbolBuffer{};
        auto* symbol = reinterpret_cast<SYMBOL_INFO*>(symbolBuffer.data());
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = MAX_SYM_NAME;
        DWORD64 displacement = 0;
        if (symbolsReady && SymFromAddr(process, addressValue, &displacement, symbol))
            std::cerr << " " << symbol->Name << "+0x" << displacement;
        std::cerr << std::dec << std::endl;
    }
#endif
    std::cerr.flush();
    throw Memory::AccessViolation(address, length, "no mapped region");
}

// A region's backing (guest_memory_map.h): its bank's, which every view of the bank shares; anything else owns its own.
GuestFlat::Backing ClassifyBacking(uint32_t baseAddress) {
    const GuestMemoryMap::Location at = GuestMemoryMap::Decode(baseAddress);
    if (at.bank == nullptr) return GuestFlat::Backing::Owned;
    switch (at.bank->bank) {
    case GuestMemoryMap::Bank::Mem1: return GuestFlat::Backing::Mem1;
    case GuestMemoryMap::Bank::Mem2: return GuestFlat::Backing::Mem2;
    case GuestMemoryMap::Bank::LockedCache: break;
    }
    return GuestFlat::Backing::Owned;
}

#if MKW_LEAN_MEMORY

std::vector<LeanBankCopies> LeanBankLayout(const std::vector<Memory::RegionConfig>& regions) {
    std::vector<LeanBankCopies> banks;
    std::vector<std::pair<const GuestMemoryMap::BankInfo*, size_t>> index;
    for (const auto& region : regions) {
        if (region.sizeBytes == 0) continue;
        const uint64_t offset = region.baseAddress & MemoryInline::kLeanMask;
        if (offset + region.sizeBytes > MemoryInline::kLeanWindowSize)
            throw std::runtime_error("lean memory: region '" + region.name + "' does not fit the window");
        const GuestMemoryMap::BankInfo* const bank = GuestMemoryMap::Decode(region.baseAddress).bank;
        auto it = std::find_if(index.begin(), index.end(),
                               [&](const auto& entry) { return bank != nullptr && entry.first == bank; });
        if (it == index.end()) {
            index.emplace_back(bank, banks.size());
            banks.push_back({bank ? bank->name : region.name, region.sizeBytes, {}});
            it = index.end() - 1;
        }
        LeanBankCopies& copies = banks[it->second];
        copies.size = std::max<uint64_t>(copies.size, region.sizeBytes);
        if (std::find(copies.offsets.begin(), copies.offsets.end(), offset) == copies.offsets.end())
            copies.offsets.push_back(offset);
    }
    return banks;
}

// Backs the regions' banks in the window. A bank at one window offset is private memory there (BackLeanWindow). A bank
// at several (MEM1) is one shared object mapped at each, as Dolphin's MEM1 masks every address below 0x08000000 and as
// the flat build's views share one section: a store through any copy is the same store. Each copy's slice of
// rollback's dirty map (kLeanDirtyOffset + offset / 64) is likewise one object mapped at every copy's slice, so that
// store sets the bank's mark, and the byte a pair mark spills past each slice's end
// (BrawlDirtyMemory::LeanMarkBlockPair) is backed: where copies are adjacent it is the next copy's first mark (shared:
// a harmless extra mark of the bank's first block), and after the last copy private memory.
// Every bank reads zero after. Horizon cannot map one object twice (svcMapMemory moves the pages): there the copies are
// separate memory, as before this table, and say so in the log; elsewhere a failed mapping ends the start-up.
void MapLeanBanks(const std::vector<LeanBankCopies>& banks, uint8_t* window) {
    constexpr uint32_t kShift = MemoryInline::kLeanPolicyShift;
    constexpr uint64_t kChunk = uint64_t{1} << kShift;
    const auto backPrivately = [&](const LeanBankCopies& bank) {
        for (const uint64_t offset : bank.offsets) {
            BackLeanWindow(offset, bank.size);
            BackLeanWindow(MemoryInline::kLeanDirtyOffset + (offset >> 6), ((bank.size - 1) >> 6) + 2);
            std::memset(window + offset, 0, bank.size);
        }
    };
    const auto unbacked = [&](uint64_t offset, uint64_t size) {
        for (uint64_t chunk = offset >> kShift; chunk < (offset + size + kChunk - 1) >> kShift; ++chunk)
            if (g_leanChunkBacked[chunk] != 0) return false;
        return true;
    };
    for (const LeanBankCopies& bank : banks) {
        if (bank.offsets.size() == 1) {
            backPrivately(bank);
            continue;
        }
        const uint64_t marksSize = bank.size >> 6;
        std::vector<uint8_t*> data, marks;
        bool shareable = bank.size % kChunk == 0 && marksSize % kChunk == 0;
        for (const uint64_t offset : bank.offsets) {
            const uint64_t marksOffset = MemoryInline::kLeanDirtyOffset + (offset >> 6);
            shareable = shareable && offset % kChunk == 0 && marksOffset % kChunk == 0 && unbacked(offset, bank.size) &&
                        unbacked(marksOffset, marksSize);
            data.push_back(window + offset);
            marks.push_back(window + marksOffset);
        }
        std::string error = "a copy not 64 KiB aligned, or already backed";
        if (shareable && LeanWindow::MapShared(bank.size, data.data(), static_cast<int>(data.size()), &error) &&
            LeanWindow::MapShared(marksSize, marks.data(), static_cast<int>(marks.size()), &error)) {
            for (size_t i = 0; i < data.size(); ++i) {
                const uint64_t dataChunk = static_cast<uint64_t>(data[i] - window) >> kShift;
                const uint64_t marksChunk = static_cast<uint64_t>(marks[i] - window) >> kShift;
                std::memset(g_leanChunkBacked + dataChunk, 1, bank.size >> kShift);
                std::memset(g_leanChunkBacked + marksChunk, 1, marksSize >> kShift);
            }
            g_leanBackedBytes += bank.size + marksSize;
            for (size_t i = 0; i < marks.size(); ++i)  // each slice's spill byte (often the next copy's first, shared)
                BackLeanWindow(static_cast<uint64_t>(marks[i] - window) + marksSize, 1);
            std::memset(data[0], 0, bank.size);
            std::memset(marks[0], 0, marksSize);
            // The copies are one memory: a byte stored through the first reads back through every other.
            for (const auto& views : {std::make_pair(&data, bank.size), std::make_pair(&marks, marksSize)}) {
                uint8_t* const probe = (*views.first)[0] + views.second - 1;
                *probe = 0x5A;
                for (const uint8_t* view : *views.first)
                    if (view[views.second - 1] != 0x5A)
                        throw std::runtime_error("lean memory: " + bank.name + "'s copies do not share their memory");
                *probe = 0;
            }
            RT_LOG(RT_TAG_MEMORY) << "lean window: " << bank.name << "'s " << data.size()
                                  << " copies share one memory and one set of dirty marks" << std::endl;
            continue;
        }
#if defined(__SWITCH__)
        RT_LOG(RT_TAG_MEMORY) << "lean window: " << bank.name << "'s " << data.size()
                              << " copies are separate memory here (" << error
                              << "): its mirrors read zero where Dolphin reads " << bank.name << std::endl;
        backPrivately(bank);
#else
        RT_LOG(RT_TAG_MEMORY) << "lean window: could not share " << bank.name << "'s copies: " << error << std::endl;
        throw std::runtime_error("lean memory: could not share " + bank.name + "'s copies: " + error);
#endif
    }
}
#endif

template <typename T>
T ReadScalar(uint32_t address) {
    // This is the cold *Slow path; Memory::GetPointer already tries the fast
    // probe first, so repeating it here only guaranteed a second miss.
    auto* ptr = Memory::GetPointer(address, sizeof(T));
    if constexpr (sizeof(T) == 1) {
        return *ptr;
    } else {
        T value = 0;
        std::memcpy(&value, ptr, sizeof(T));
        return MemoryInline::MaybeByteSwap(value);
    }
}

template <typename T>
void WriteScalar(uint32_t address, T value) {
    auto* ptr = Memory::GetPointer(address, sizeof(T));
    const T stored = sizeof(T) == 1 ? value : MemoryInline::MaybeByteSwap(value);
    // A store of the value already there changes nothing, so it needs no dirty mark and no code-patch notice. The
    // Gecko code handler rewrites every one of its patches into guarded code at each VBI; each of those used to
    // queue its page for a compare that then found nothing new.
    if (std::memcmp(ptr, &stored, sizeof(T)) == 0) {
        return;
    }
#if MKW_FALLBACK_STATS
    // BRAWL_WATCH_WRITE=<hex address>: the first 8 checked-path stores that change the word there, with host stacks.
    static const uint32_t watch = [] {
        const char* text = std::getenv("BRAWL_WATCH_WRITE");
        return text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 16)) : 0u;
    }();
    static int watchLogs = 0;
    if (watch != 0 && address <= watch && watch < address + sizeof(T) && watchLogs < 8) {
        ++watchLogs;
        std::fprintf(stderr, "[watch] store of %zu byte(s) %llx at %08X (word was %08X)\n", sizeof(T),
                     (unsigned long long)value, address, Memory::Read32(watch & ~3u));
        void* frames[24];
        const int count = backtrace(frames, 24);
        backtrace_symbols_fd(frames, count, 2);
    }
#endif
    BrawlDirtyMemory::Mark(address,sizeof(T));
    // Executable ranges are registered at their bank's primary view: a store through any other (a MEM1 mirror, an
    // uncached or physical view) is checked at the address of the same bytes there.
    if (RecompMod::HandleExecutableWrite(GuestMemoryMap::Canonical(address), sizeof(T),
                                         static_cast<uint64_t>(value))) {
        return;
    }
    std::memcpy(ptr, &stored, sizeof(T));
}

} // namespace

Memory::AccessViolation::AccessViolation(uint32_t address, size_t length, std::string_view reason)
    : std::runtime_error([&]() {
          std::ostringstream oss;
          oss << "Memory access violation at 0x" << std::hex << std::uppercase << address
              << " (+0x" << length << ") :: " << reason;
          return oss.str();
      }()),
      address_(address),
      length_(length),
      reason_(reason) {
    // The handler that reports it runs after unwinding, without the guest context: record that now.
    static std::atomic<int> reported{0};
    if (const CpuContext* cpu = TryGetCpuContext(); cpu && reported.fetch_add(1, std::memory_order_relaxed) < 3) {
        RT_LOG(RT_TAG_MEMORY) << what() << ", guest state where it happened:" << std::endl;
        SystemBridge::DumpCpuState(cpu);
    }
}

void Memory::RefreshWritableFastPathsForExecutableRanges() {
    std::lock_guard<std::mutex> lock(RegionMutex());
    RefreshFastPathTables();
}

// Every view of guest_memory_map.h's table, one region per bank copy: MEM1's four copies in each of its cached,
// uncached and physical views (its mirrors, as Dolphin decodes them), MEM2's three views, the locked cache.
Memory::Config Memory::Config::WiiDefaults() {
    Config config;
    GuestMemoryMap::ForEachCopy([&](const GuestMemoryMap::View& view, uint32_t index, uint32_t start, uint32_t size) {
        config.regions.push_back(RegionConfig{
            .name = index == 0 ? std::string(view.name) : std::string(view.name) + "_MIRROR" + std::to_string(index),
            .baseAddress = start,
            .sizeBytes = size,
        });
    });
    return config;
}

void Memory::Init(size_t mem1Size) {
    Config config;
    config.regions.push_back(RegionConfig{.name = "MEM1", .baseAddress = 0x80000000, .sizeBytes = mem1Size});
    Init(config);
}

void Memory::Init(const Config& config) {
    std::lock_guard<std::mutex> lock(RegionMutex());
    auto& regions = Regions();
    regions.clear();
    ClearPageTable();
    regions.reserve(config.regions.size());

#if MKW_FLAT_MEMORY
    // The flat reservation backs every guest view from one shared section object, so cached/uncached/
    // physical mirrors alias as before. GetPointer still hands out the unprotected host alias, so
    // native code (image loading, DVD reads, HLE) is unaffected by the guest view's protections.
    {
        std::vector<GuestFlat::RegionRequest> flatRegions;
        flatRegions.reserve(config.regions.size());
        for (const auto& regionConfig : config.regions) {
            flatRegions.push_back(GuestFlat::RegionRequest{
                regionConfig.baseAddress, regionConfig.sizeBytes,
                ClassifyBacking(regionConfig.baseAddress)});
        }
        GuestFlat::Initialize(flatRegions);
    }

#elif MKW_LEAN_MEMORY
    // One 1 GiB window for every guest view (memory_access.h, MKW_LEAN_MEMORY): a region's host bytes sit at its
    // masked address, so the physical, cached and uncached views of MEM1/MEM2 alias by construction.
    // Address space only: BackLeanWindow backs the parts in use. The 64 KiB after it (kLeanGuardSize) stay reserved and
    // unbacked, so a resolved read range's access past the window's end faults (memory_access.h, kLeanRangePolicyOffset).
    static uint8_t* const window = [] {
        constexpr uint64_t kReserved = MemoryInline::kLeanWindowSize + MemoryInline::kLeanGuardSize;
        uint8_t* const reserved = LeanWindow::Reserve(kReserved);
#if defined(_WIN32)
        if (reserved == nullptr)
            throw std::runtime_error("lean memory: could not reserve the 1 GiB guest window (it needs Windows 10 1803 "
                                     "or newer: VirtualAlloc2, MapViewOfFile3)");
#endif
        if (reserved == nullptr) throw std::runtime_error("lean memory: could not reserve the 1 GiB guest window");
        return reserved;
    }();
    MemoryInline::SetLeanBase(window);
    BackLeanWindow(MemoryInline::kLeanPolicyOffset, MemoryInline::kLeanWindowSize >> MemoryInline::kLeanPolicyShift);
    // The whole MMIO window 0xCC000000-0xCE000000 reads zero (BackLeanWindow). A Wii answers loads anywhere in it, and
    // guest code loads through garbage pointers into it (the game's 0xCC heap fill: P+'s pad hook reads 0xCCCCCD14
    // while sora_melee loads), so a list of the chunks some check runs happened to read is not safe (an earlier version had one:
    // 5 chunks from Mario/Marth runs). Costs 32 MiB on Horizon (backing is committed); untouched pages cost nothing
    // elsewhere.
    BackLeanWindow(0xCC000000u & MemoryInline::kLeanMask, 0xCE000000u - 0xCC000000u);
    // Rollback's dirty marks (brawl_dirty_memory.h), one byte per 64 window bytes written by every guest store, sit in
    // the window itself (memory_access.h, kLeanDirtyOffset); only the slices under MEM1/MEM2 are ever touched.
    static_assert(MemoryInline::kLeanDirtyOffset + BrawlDirtyMemory::kLeanDirtyBlocks + 1 <= MemoryInline::kLeanPolicyOffset);
    BrawlDirtyMemory::leanBlocks = window + MemoryInline::kLeanDirtyOffset;
    BrawlDirtyMemory::flatBase = reinterpret_cast<uintptr_t>(window);
    g_leanBankCopies = LeanBankLayout(config.regions);
    MapLeanBanks(g_leanBankCopies, window);
#else
    // Cached, uncached and physical guest aliases share one allocation per RAM
    // bank. Other configured regions own their storage independently.
    struct Store { size_t size = 0; std::shared_ptr<std::vector<uint8_t>> bytes; };
    std::unordered_map<uint64_t, Store> stores;
    const auto key = [](uint32_t base) {
        const auto backing = ClassifyBacking(base);
        return (static_cast<uint64_t>(backing) << 32) |
            (backing == GuestFlat::Backing::Owned ? base : 0u);
    };
    const auto offset = [](uint32_t base) -> size_t {
        switch (ClassifyBacking(base)) {
        case GuestFlat::Backing::Mem1: return base & 0x01FFFFFFu;
        case GuestFlat::Backing::Mem2: return base & 0x0FFFFFFFu;
        default: return 0;
        }
    };
    for (const auto& region : config.regions) {
        if (region.sizeBytes > GuestFlat::kGuestSpaceSize - region.baseAddress)
            throw std::invalid_argument("Guest memory region exceeds the 32-bit address space");
        auto& store = stores[key(region.baseAddress)];
        store.size = std::max(store.size, offset(region.baseAddress) + region.sizeBytes);
    }
    for (auto& [_, store] : stores)
        store.bytes = std::make_shared<std::vector<uint8_t>>(store.size);
#endif

    // Map virtual regions onto that backing store, with mirroring.
    for (const auto& regionConfig : config.regions) {
        Region instance;
        instance.config = regionConfig;

#if MKW_FLAT_MEMORY
        instance.storagePtr = GuestFlat::HostPointer(regionConfig.baseAddress);
        if (instance.storagePtr == nullptr && regionConfig.sizeBytes != 0) {
            throw std::runtime_error("Flat guest mapping is missing region '" + regionConfig.name + "'");
        }
#elif MKW_LEAN_MEMORY
        // Backed by MapLeanBanks above.
        instance.storagePtr = MemoryInline::LeanBaseAtInit() + (regionConfig.baseAddress & MemoryInline::kLeanMask);
#else
        instance.storage = stores.at(key(regionConfig.baseAddress)).bytes;
        instance.storagePtr = instance.storage->empty() ? nullptr :
            instance.storage->data() + offset(regionConfig.baseAddress);
#endif
        instance.storageSize = regionConfig.sizeBytes;

        regions.emplace_back(std::move(instance));
    }
#if MKW_LEAN_MEMORY
    RT_LOG(RT_TAG_MEMORY) << "lean window: " << (g_leanBackedBytes >> 10) << " KiB of 1 GiB backed" << std::endl;
#endif

    BuildPageTable();
    RefreshFastPathTables();
}

void Memory::Reset() {
    std::lock_guard<std::mutex> lock(RegionMutex());
    Regions().clear();
    ClearPageTable();
}

namespace {
// MMIO reads have no backing store or generic HLE. Answering zero (the old sparse fallback) turned
// unimplemented devices into silent hangs, so unmapped reads throw like MMIO writes; every device
// we do implement is hooked at the function level, so a hit here means a missing HLE hook.
[[noreturn]] void ThrowMmioReadBlocked(uint32_t addr, size_t length) {
    std::ostringstream reason;
    reason << (MemoryInline::IsGpuFifoAddress(addr)
                   ? "GPU FIFO read blocked (the gather pipe is write-only)"
                   : "MMIO read blocked (non-GPU)")
           << "; add HLE for this device instead of answering zero"
           << " (active=0x" << std::hex << std::uppercase
           << RecompMod::CurrentTranslatedExecutionAddress() << std::dec << std::nouppercase << ")";
    throw Memory::AccessViolation(addr, length, reason.str());
}
} // namespace

uint8_t MemoryInline::Read8Slow(uint32_t addr) {
    if (IsMmioAddress(addr)) {
        if (!IsDeviceMmioAddress(addr)) return 0;  // nothing on the bus there: reads 0, writes dropped
        ThrowMmioReadBlocked(addr, sizeof(uint8_t));
    }
    return ReadScalar<uint8_t>(addr);
}

uint16_t MemoryInline::Read16Slow(uint32_t addr) {
    if (IsMmioAddress(addr)) {
        if (!IsDeviceMmioAddress(addr)) return 0;  // nothing on the bus there: reads 0, writes dropped
        ThrowMmioReadBlocked(addr, sizeof(uint16_t));
    }
    return ReadScalar<uint16_t>(addr);
}

uint32_t MemoryInline::Read32Slow(uint32_t addr) {
    if (IsMmioAddress(addr)) {
        if (!IsDeviceMmioAddress(addr)) return 0;  // nothing on the bus there: reads 0, writes dropped
        ThrowMmioReadBlocked(addr, sizeof(uint32_t));
    }
    const uint32_t value = ReadScalar<uint32_t>(addr);
    return value;
}

uint64_t MemoryInline::Read64Slow(uint32_t addr) {
    if (IsMmioAddress(addr)) {
        if (!IsDeviceMmioAddress(addr)) return 0;  // nothing on the bus there: reads 0, writes dropped
        ThrowMmioReadBlocked(addr, sizeof(uint64_t));
    }
    return ReadScalar<uint64_t>(addr);
}

float MemoryInline::ReadFloat32Slow(uint32_t addr) {
    if (IsMmioAddress(addr)) {
        if (!IsDeviceMmioAddress(addr)) return 0;  // nothing on the bus there: reads 0, writes dropped
        ThrowMmioReadBlocked(addr, sizeof(float));
    }
    const auto bits = ReadScalar<uint32_t>(addr);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

double MemoryInline::ReadFloat64Slow(uint32_t addr) {
    if (IsMmioAddress(addr)) {
        if (!IsDeviceMmioAddress(addr)) return 0;  // nothing on the bus there: reads 0, writes dropped
        ThrowMmioReadBlocked(addr, sizeof(double));
    }
    const auto bits = ReadScalar<uint64_t>(addr);
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void MemoryInline::Write8Slow(uint32_t addr, uint8_t val) {
    if (IsGpuFifoAddress(addr)) {
        GX_HLE_FIFO_Write8(val);
        return;
    }
    if (IsMmioAddress(addr)) {
        if (!IsDeviceMmioAddress(addr)) return;  // nothing on the bus there: reads 0, writes dropped
        throw Memory::AccessViolation(addr, sizeof(val), "MMIO write blocked (non-GPU)");
    }
    WriteScalar(addr, val);
}

void MemoryInline::Write16Slow(uint32_t addr, uint16_t val) {
    if (IsGpuFifoAddress(addr)) {
        GX_HLE_FIFO_Write16(val);
        return;
    }
    if (IsMmioAddress(addr)) {
        if (!IsDeviceMmioAddress(addr)) return;  // nothing on the bus there: reads 0, writes dropped
        throw Memory::AccessViolation(addr, sizeof(val), "MMIO write blocked (non-GPU)");
    }
    WriteScalar(addr, val);
}

void MemoryInline::Write32Slow(uint32_t addr, uint32_t val) {
    if (IsGpuFifoAddress(addr)) {
        GX_HLE_FIFO_Write32(val);
        return;
    }
    if (IsMmioAddress(addr)) {
        if (!IsDeviceMmioAddress(addr)) return;  // nothing on the bus there: reads 0, writes dropped
        throw Memory::AccessViolation(addr, sizeof(val), "MMIO write blocked (non-GPU)");
    }
    WriteScalar(addr, val);
}

void MemoryInline::Write64Slow(uint32_t addr, uint64_t val) {
    if (IsGpuFifoAddress(addr)) {
        // A 64-bit FIFO store emits the high word first. Dispatch the low word
        // separately: it may cross out of the FIFO's address window.
        GX_HLE_FIFO_Write32(static_cast<uint32_t>(val >> 32));
        Write32Slow(addr + 4, static_cast<uint32_t>(val));
        return;
    }
    if (IsMmioAddress(addr)) {
        if (!IsDeviceMmioAddress(addr)) return;  // nothing on the bus there: reads 0, writes dropped
        throw Memory::AccessViolation(addr, sizeof(val), "MMIO write blocked (non-GPU)");
    }
    WriteScalar(addr, val);
}

void MemoryInline::WriteFloat32Slow(uint32_t addr, double val) {
    // stfs stores the IEEE-754 bit pattern of the single-precision value, never a
    // truncated integer.
    const uint32_t bits = ConvertPpcDoubleToSingleBits(val);
    if (IsGpuFifoAddress(addr)) {
        GX_HLE_FIFO_WriteFloat(PpcSingleBitsToFloat(bits));
        return;
    }
    if (IsMmioAddress(addr)) {
        if (!IsDeviceMmioAddress(addr)) return;  // nothing on the bus there: reads 0, writes dropped
        throw Memory::AccessViolation(addr, sizeof(float), "MMIO write blocked (non-GPU)");
    }

    WriteScalar(addr, bits);
}

void MemoryInline::WriteFloat64Slow(uint32_t addr, double val) {
    // stfd stores the full FPR bit pattern, including packed integer/float data.
    uint64_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    Write64Slow(addr, bits);
}

uint8_t* Memory::GetPointer(uint32_t addr) {
    return GetPointer(addr, 1);
}

uint8_t* Memory::GetPointer(uint32_t addr, size_t length) {
    if (auto* ptr = MemoryInline::GetPointerFast(addr, length)) {
        return ptr;
    }
    auto& region = ResolveRegion(addr, length);
    auto offset = static_cast<size_t>(addr - region.config.baseAddress);
    return region.storagePtr + offset;
}

bool Memory::Contains(uint32_t addr, size_t length) {
    if (MemoryInline::GetPointerFast(addr, length)) {
        return true;
    }
    // Contains is a membership predicate: callers probe arbitrary guest values
    // (e.g. gfx-node slots that may hold floats), so a miss must stay silent
    // rather than take ResolveRegion's crash-dump path.
    return TryResolveRegion(addr, length) != nullptr;
}

std::vector<Memory::RegionConfig> Memory::DescribeRegions() {
    std::lock_guard<std::mutex> lock(RegionMutex());
    std::vector<RegionConfig> info;
    info.reserve(Regions().size());
    for (const auto& region : Regions()) {
        info.push_back(RegionConfig{region.config.name, region.config.baseAddress, region.storageSize});
    }
    return info;
}
