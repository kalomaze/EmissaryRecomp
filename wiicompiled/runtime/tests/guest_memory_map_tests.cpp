// The recomp's memory map (guest_memory_map.h, Memory::Init) against Dolphin's address decode, and MEM1's mirrors in
// use: the same bytes and the same rollback marks through every view, the 24-32 MiB tail included. Built in the lean
// and the flat configuration (MKW_BUILD_INTERPRETER_TESTS), with interpreter_test_support.cpp's stubs.
#include "brawl_dirty_memory.h"
#include "brawl_snapshot_store.h"
#include "brawl_write_watch.h"
#include "lean_window.h"
#include "memory.h"
#include "recomp_mod_loader.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace RecompMod {
extern std::atomic<uint32_t> g_testLastExecutableWrite;  // interpreter_test_support.cpp
}

namespace {

int g_failures = 0;
void Check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// Dolphin's decode of an effective address with translation on: the BATs Boot_BS2Emu.cpp's SetupBAT gives a Wii title
// (0x80000000 and 0xC0000000 onto physical 0, 0x90000000 and 0xD0000000 onto physical 0x10000000, 256 MiB each) and the
// game's own locked-cache BAT at 0xE0000000, then PowerPC/MMU.cpp's physical decode: EFB at 0x08000000, MMIO at
// 0x0C000000, MEM1 below 0x08000000 masked with NextPowerOf2(24 MiB) - 1, MEM2 at 0x10000000 below its 64 MiB, the
// locked cache's 256 KiB. Anything else has no BAT (a DSI) or no device.
enum class DolphinBank { None, Mem1, Mem2, LockedCache, Efb, Mmio };
struct DolphinLocation {
    DolphinBank bank = DolphinBank::None;
    uint32_t offset = 0;
};
DolphinLocation DolphinDecode(uint32_t ea) {
    const uint32_t segment = ea >> 28;
    if (segment == 0xE) {
        if (ea >= 0xE0000000u + 0x40000u) return {};
        return {DolphinBank::LockedCache, ea & 0x0FFFFFFFu};
    }
    uint32_t physical = 0;
    if (segment == 0x8 || segment == 0xC) physical = ea & 0x0FFFFFFFu;
    else if (segment == 0x9 || segment == 0xD) physical = 0x10000000u | (ea & 0x0FFFFFFFu);
    else return {};
    if ((physical & 0xF8000000u) == 0x08000000u)
        return {physical < 0x0C000000u ? DolphinBank::Efb : DolphinBank::Mmio, 0};
    if ((physical & 0xF8000000u) == 0) return {DolphinBank::Mem1, physical & 0x01FFFFFFu};
    if ((physical >> 28) == 1 && (physical & 0x0FFFFFFFu) < 0x04000000u)
        return {DolphinBank::Mem2, physical & 0x0FFFFFFFu};
    return {};
}

DolphinBank AsDolphin(const GuestMemoryMap::BankInfo* bank) {
    if (bank == nullptr) return DolphinBank::None;
    switch (bank->bank) {
    case GuestMemoryMap::Bank::Mem1: return DolphinBank::Mem1;
    case GuestMemoryMap::Bank::Mem2: return DolphinBank::Mem2;
    case GuestMemoryMap::Bank::LockedCache: return DolphinBank::LockedCache;
    }
    return DolphinBank::None;
}
bool IsRam(DolphinBank bank) {
    return bank == DolphinBank::Mem1 || bank == DolphinBank::Mem2 || bank == DolphinBank::LockedCache;
}

// The known, accepted differences where the recomp maps RAM and Dolphin does not: physical addresses (below 0x80000000,
// which the lean window folds onto RAM anyway) and the recomp's own area past MEM2's 64 MiB (the modules' homes).
bool AcceptedExtra(uint32_t ea, const GuestMemoryMap::Location& at) {
    return ea < 0x80000000u || (at.bank != nullptr && at.offset >= at.bank->privateFrom);
}

uint32_t HostRead32(const uint8_t* at) {
    uint32_t v;
    std::memcpy(&v, at, 4);
    return v;
}
void HostWrite32(uint8_t* at, uint32_t v) { std::memcpy(at, &v, 4); }

// (a) Every 4 KiB page of the 32-bit space: the bank and offset Dolphin's decode gives against the table's, Memory's
// regions (Memory::Contains, and the same bytes through Memory::GetPointer as through the bank's primary view), and
// under lean memory the window the translated code reads (address & kLeanMask) too.
void TestEveryPageAgainstDolphin() {
    const auto regions = Memory::DescribeRegions();
    uint64_t ram = 0, extras = 0, mismatches = 0;
#if MKW_LEAN_MEMORY
    uint64_t leanFolds = 0, aliasedPages = 0;
#endif
    for (uint64_t page = 0; page < (uint64_t{1} << 32); page += 0x1000) {
        const uint32_t ea = static_cast<uint32_t>(page) + 0x7FC;  // inside the page, word aligned
        const DolphinLocation dolphin = DolphinDecode(ea);
        const GuestMemoryMap::Location at = GuestMemoryMap::Decode(ea);
        const DolphinBank recomp = AsDolphin(at.bank);
        bool ok;
        if (IsRam(dolphin.bank)) {
            ok = recomp == dolphin.bank && at.offset == dolphin.offset;
            ++ram;
        } else {
            ok = recomp == DolphinBank::None || AcceptedExtra(ea, at);
            if (recomp != DolphinBank::None) ++extras;
        }
        // Memory's regions follow the table, and every alias is the bank's own bytes.
        const bool contained = Memory::Contains(ea, 4);
        ok = ok && contained == (at.bank != nullptr);
        if (contained && at.bank != nullptr) {
            uint8_t* const alias = Memory::GetPointer(ea, 4);
            uint8_t* const primary = Memory::GetPointer(at.bank->primary + at.offset, 4);
            const uint32_t before = HostRead32(primary), tag = 0xA5000000u | static_cast<uint32_t>(page >> 12);
            HostWrite32(primary, tag);
            ok = ok && HostRead32(alias) == tag;
            HostWrite32(primary, before);
        }
#if MKW_LEAN_MEMORY
        // What a translated access reads: the window at the masked address, inside some region's copy or no RAM.
        const uint64_t window = ea & MemoryInline::kLeanMask;
        const Memory::RegionConfig* copy = nullptr;
        for (const auto& region : regions)
            if (window >= (region.baseAddress & MemoryInline::kLeanMask) &&
                window < (region.baseAddress & MemoryInline::kLeanMask) + uint64_t{region.sizeBytes})
                copy = &region;
        GuestMemoryMap::Location folded;
        if (copy != nullptr) {
            const uint64_t into = window - (copy->baseAddress & MemoryInline::kLeanMask);
            folded = GuestMemoryMap::Decode(copy->baseAddress + static_cast<uint32_t>(into));
        }
        if (IsRam(dolphin.bank)) {
            ok = ok && AsDolphin(folded.bank) == dolphin.bank && folded.offset == dolphin.offset;
        } else if (folded.bank != nullptr && !AcceptedExtra(ea, folded)) {
            // The lean mask's own folds above 0x80000000: 0xA0000000.. lands on the locked cache (0xE0000000's offset).
            const bool lockedCacheFold = (ea >> 28) == 0xA && folded.bank->bank == GuestMemoryMap::Bank::LockedCache;
            ok = ok && lockedCacheFold;
            leanFolds += lockedCacheFold;
        }
        if (copy != nullptr && folded.bank != nullptr) {
            uint8_t* const primary = Memory::GetPointer(folded.bank->primary + folded.offset, 4);
            const uint32_t before = HostRead32(primary), tag = 0x5A000000u | static_cast<uint32_t>(page >> 12);
            HostWrite32(primary, tag);
            ok = ok && HostRead32(MemoryInline::LeanHost(ea)) == tag;
            // the aliased view (lean_window.h): the same bytes at base + ea, and at base + 4 GiB + ea for the first 64 KiB
            if (LeanWindow::g_aliased) {
                ok = ok && HostRead32(MemoryInline::g_leanBase + ea) == tag;
                if (ea < LeanWindow::kAliasTail)
                    ok = ok && HostRead32(MemoryInline::g_leanBase + (uint64_t{1} << 32) + ea) == tag;
                ++aliasedPages;
            }
            HostWrite32(primary, before);
        }
#endif
        if (!ok && mismatches++ < 8)
            std::fprintf(stderr, "FAIL: page %08X: Dolphin bank %d offset %08X, recomp bank %d offset %08X\n",
                         static_cast<uint32_t>(page), static_cast<int>(dolphin.bank), dolphin.offset,
                         static_cast<int>(recomp), at.offset);
    }
    Check(mismatches == 0, "every page decodes as on Dolphin");
    std::printf("every page: %llu RAM pages as on Dolphin, %llu accepted extra pages (physical views, module homes)",
                static_cast<unsigned long long>(ram), static_cast<unsigned long long>(extras));
#if MKW_LEAN_MEMORY
    std::printf(", %llu lean-window pages at 0xA0000000 folded onto the locked cache, %llu read through the aliased view",
                static_cast<unsigned long long>(leanFolds), static_cast<unsigned long long>(aliasedPages));
#if defined(__linux__)
    Check(aliasedPages != 0 || std::getenv("BRAWL_LEAN_ALIAS") != nullptr, "the aliased view is there (Linux)");
#endif
#endif
    std::printf("\n");
}

// A write-tracked MEM1, as rollback tracks a region set (brawl_online_core.cpp, brawl_rollback.cpp): the bank's
// primary view, whole.
struct TrackedMem1 {
    BrawlRollback::SnapshotStore store;
#if MKW_FLAT_MEMORY
    BrawlRollback::WriteWatch watch{reinterpret_cast<uintptr_t>(MKW_FLAT_GUEST_BASE)};
#elif MKW_LEAN_MEMORY
    BrawlRollback::WriteWatch watch{reinterpret_cast<uintptr_t>(MemoryInline::g_leanBase)};
#else
    BrawlRollback::WriteWatch watch;
#endif
    uint8_t* host = Memory::GetPointer(Memory::kMem1CachedBase, Memory::kMem1BankSize);
    auto Reader() {
        return [h = host](size_t at, uint8_t* out, size_t n) { std::memcpy(out, h + at, n); };
    }
    auto Writer() {
        return [h = host](size_t at, const uint8_t* in, size_t n) { std::memcpy(h + at, in, n); };
    }
    TrackedMem1() {
        watch.Add(host, Memory::kMem1BankSize, 0, Memory::kMem1CachedBase);
        store.Save(0, Memory::kMem1BankSize, Reader());
        watch.Enable();
    }
    bool Marked(uint32_t offset) {
        const auto& dirty = watch.Capture(Memory::kMem1BankSize, store.Jobs());
        for (const uint32_t block : dirty)
            if (block == offset / BrawlRollback::SnapshotStore::BlockSize) return true;
        return false;
    }
};

// (b) Upstream's failing case: a store at 0x8148B5C8 reads back at its mirror 0x8348B5C8 (Brawl's read leaving P+
// online's stage select), and a store through the mirror sets MEM1's mark. Through the translated code's access path.
void TestUpstreamMirrorCase() {
    TrackedMem1 tracked;
    MemoryInline::FlatWrite32(0x8148B5C8u, 0x12345678u);
    Check(MemoryInline::FlatRead32(0x8348B5C8u) == 0x12345678u, "0x8348B5C8 reads what 0x8148B5C8 stored");
    Check(Memory::Read32(0x8348B5C8u) == 0x12345678u, "the checked path reads it too");
    Check(tracked.Marked(0x0148B5C8u), "the primary store is marked");
    MemoryInline::FlatWrite32(0x8348B5CCu, 0xCAFEBABEu);
    Check(MemoryInline::FlatRead32(0x8148B5CCu) == 0xCAFEBABEu, "a store through the mirror reaches MEM1");
    Check(tracked.Marked(0x0148B5CCu), "a store through the mirror sets MEM1's mark");
    std::printf("upstream mirror case: ok\n");
}

// (c) MEM1's 24-32 MiB tail, which Dolphin backs and the game is not told about: a store there reads back through
// every view of MEM1, is marked, and a snapshot restores it.
void TestTail() {
    TrackedMem1 tracked;
    constexpr uint32_t kOffset = 0x01C00010u;
    MemoryInline::FlatWrite32(0x87C00010u, 0x0BADF00Du);  // the last cached mirror
    int views = 0;
    bool visible = true;
    GuestMemoryMap::ForEachAlias(GuestMemoryMap::Bank::Mem1, kOffset, [&](uint32_t alias) {
        ++views;
        visible = visible && MemoryInline::FlatRead32(alias) == 0x0BADF00Du && Memory::Read32(alias) == 0x0BADF00Du;
    });
    Check(views == 12 && visible, "the tail reads back through all twelve views of MEM1");
    const auto& dirty = tracked.watch.Capture(Memory::kMem1BankSize, tracked.store.Jobs());
    bool marked = false;
    for (const uint32_t block : dirty) marked = marked || block == kOffset / BrawlRollback::SnapshotStore::BlockSize;
    Check(marked, "the tail store is marked");
    tracked.store.Save(1, Memory::kMem1BankSize, tracked.Reader(), &dirty, 0);
    MemoryInline::FlatWrite32(0xC1C00010u, 0xDEADBEEFu);  // the uncached view, MEM1 itself
    const auto& later = tracked.watch.Capture(Memory::kMem1BankSize, tracked.store.Jobs());
    tracked.store.Restore(1, 1, later, tracked.Reader(), tracked.Writer());
    Check(MemoryInline::FlatRead32(0x81C00010u) == 0x0BADF00Du, "the snapshot restores the tail");
    std::printf("MEM1 tail: visible through %d views, marked, restored\n", views);
}

#if MKW_LEAN_MEMORY
// A store to code through a MEM1 mirror takes the checked path (the mirror's chunks carry MEM1's store policy) and is
// reported at the code's own address.
void TestCodeWriteThroughMirror() {
    constexpr uint32_t kCode = 0x80004000u;
    RecompMod::g_executableWriteGuardEnabled.store(true);
    RecompMod::g_executableWriteGuardCoarsePages[kCode >> RecompMod::kExecutableWriteGuardCoarsePageShift].store(1);
    RecompMod::g_executableWriteGuardMidPages[kCode >> RecompMod::kExecutableWriteGuardMidPageShift].store(1);
    RecompMod::g_executableWriteGuardPages[kCode >> RecompMod::kExecutableWriteGuardPageShift].store(1);
    Memory::RefreshWritableFastPathsForExecutableRanges();
    bool flagged = true;
    GuestMemoryMap::ForEachAlias(GuestMemoryMap::Bank::Mem1, kCode - Memory::kMem1CachedBase, [&](uint32_t alias) {
        flagged = flagged && MemoryInline::LeanStoreNeedsPolicy(alias);
    });
    Check(flagged, "every view of a code page takes the checked store path");
    RecompMod::g_testLastExecutableWrite.store(0);
    MemoryInline::FlatWrite32(0x86004000u, 0x60000000u);
    Check(RecompMod::g_testLastExecutableWrite.load() == kCode, "a code store through a mirror reaches the code check");
    Check(MemoryInline::FlatRead32(kCode) == 0x60000000u, "and lands in MEM1");
    RecompMod::g_executableWriteGuardPages[kCode >> RecompMod::kExecutableWriteGuardPageShift].store(0);
    RecompMod::g_executableWriteGuardMidPages[kCode >> RecompMod::kExecutableWriteGuardMidPageShift].store(0);
    RecompMod::g_executableWriteGuardCoarsePages[kCode >> RecompMod::kExecutableWriteGuardCoarsePageShift].store(0);
    RecompMod::g_executableWriteGuardEnabled.store(false);
    Memory::RefreshWritableFastPathsForExecutableRanges();
    std::printf("code store through a mirror: ok\n");
}
#endif

}  // namespace

int main() {
    Memory::Init(Memory::Config::WiiDefaults());
    TestEveryPageAgainstDolphin();
    TestUpstreamMirrorCase();
    TestTail();
#if MKW_LEAN_MEMORY
    TestCodeWriteThroughMirror();
#endif
    if (g_failures == 0) std::printf("guest memory map tests passed\n");
    return g_failures == 0 ? 0 : 1;
}
