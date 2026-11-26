#include "memory_access.h"
#include "brawl_write_watch.h"
#include "brawl_snapshot_store.h"
#include "isa/ppc_isa_quantized.h"
#include <array>
#include <cassert>
#include <cstdio>

int main() {
    Memory::Init(Memory::Config::WiiDefaults());
    constexpr std::array<uint32_t, 9> views = {
        0, 0x80000000u, 0x82000000u, 0x84000000u, 0x86000000u,
        0xc0000000u, 0xc2000000u, 0xc4000000u, 0xc6000000u};
    constexpr uint32_t offset = 0x0148b5c8u; // Actual stage-select mirror read.
    for (auto writer : views) {
        const auto value = writer ^ 0x12345678u;
        std::memset(BrawlDirtyMemory::leanBlocks + (offset >> 6), 0, 2);
        MemoryInline::FlatWrite32(writer + offset, value);
        assert(BrawlDirtyMemory::leanBlocks[offset >> 6] != 0);
        for (auto reader : views) {
            assert(Memory::Read32(reader + offset) == value);
            assert(MemoryInline::FlatRead32(reader + offset) == value);
            assert(Memory::GetPointer(reader + offset) == Memory::GetPointer(offset));
            auto* range = MemoryInline::ResolveSegmentRange(reader + offset, 0, 4, false);
            assert(MemoryInline::SegRead32(range, 0, reader + offset) == value);
        }
    }
    MemoryInline::FlatStore<uint64_t>(0xd0123400u, 0x123456789abcdef0ull);
    for (uint32_t address : {0x10123400u, 0x90123400u, 0xd0123400u})
        assert(MemoryInline::FlatLoad<uint64_t>(address) == 0x123456789abcdef0ull);
    MemoryInline::FlatWrite32(0xe0003ffcu, 0xabcdef01u);
    assert(MemoryInline::FlatRead32(0xe0003ffcu) == 0xabcdef01u);
    // Last-page scalar accesses legitimately take the checked fallback.
    MemoryInline::FlatWrite32(0x837ffffcu, 0xdeadbeefu);
    assert(MemoryInline::FlatRead32(0xc17ffffcu) == 0xdeadbeefu);
    // A scalar crossing a policy boundary must honor the second chunk's guard,
    // even when the write arrives through a mirror whose own guard bit is clear.
    constexpr uint32_t guarded = 0x80010000u;
    RecompMod::g_executableWriteGuardPages[guarded >> RecompMod::kExecutableWriteGuardPageShift].store(1);
    RecompMod::g_executableWriteGuardCoarsePages[guarded >> MemoryInline::kPageShift].store(1);
    Memory::RefreshWritableFastPathsForExecutableRanges();
    assert(!MemoryInline::LeanTryStore<uint64_t>(0x8200fffcu, 0));
    assert(!PpcTryGetPsqWritableHostInline(0xc200fffcu));
    RecompMod::g_executableWriteGuardPages[guarded >> RecompMod::kExecutableWriteGuardPageShift].store(0);
    RecompMod::g_executableWriteGuardCoarsePages[guarded >> MemoryInline::kPageShift].store(0);
    Memory::RefreshWritableFastPathsForExecutableRanges();
    // Track a mirrored region, mutate it through every other view, and verify
    // the actual rollback scanner reports the same packed snapshot block.
    {
        BrawlRollback::SnapshotJobs jobs;
        BrawlRollback::WriteWatch watch(reinterpret_cast<uintptr_t>(MemoryInline::g_leanBase));
        watch.Add(Memory::GetPointer(0x83480000u), 65536, 0, 0x83480000u);
        for (auto writer : views) {
            BrawlRollback::SnapshotStore snapshots;
            auto* backing = Memory::GetPointer(0x83480000u);
            const auto read = [&](size_t at, uint8_t* out, size_t size) { std::memcpy(out, backing + at, size); };
            const auto write = [&](size_t at, const uint8_t* in, size_t size) { std::memcpy(backing + at, in, size); };
            Memory::Write32(0x8148b5c8u, 0x24681357u);
            snapshots.Save(0, 65536, read);
            watch.Enable();
            MemoryInline::FlatWrite32(writer + offset, writer);
            const auto& dirty = watch.Capture(65536, jobs);
            assert(std::find(dirty.begin(), dirty.end(), (offset & 65535u) / 64) != dirty.end());
            snapshots.Save(1, 65536, read, &dirty, 0);
            snapshots.Restore(0, 1, {}, read, write);
            for (auto reader : views) assert(MemoryInline::FlatRead32(reader + offset) == 0x24681357u);
        }
    }
    // MEM1 holes, invalid views, metadata, wraparound and end-of-region overruns
    // must never resolve into backing or runtime tables.
    for (auto address : {0x81800000u, 0x83800000u, 0x40000000u,
                         0xbe000000u, 0xbf800000u, 0xffffffffu, 0x817fffffu}) {
        assert(!MemoryInline::LeanRangeMapped(address, 4));
        assert(!MemoryInline::ResolveSegmentRange(address, 0, 4, false));
        assert(!MemoryInline::ResolveSegmentStackRange(address, 0, 4));
        assert(!MemoryInline::LeanTryStore<uint32_t>(address, 0));
        assert(!PpcTryGetPsqWritableHostInline(address));
        assert(!PpcTryGetPsqReadableHostInline(address));
        assert(!MemoryInline::SegReadPair32(nullptr, 0).valid);
    }
    assert(!MemoryInline::ResolveSegmentRange(0x817ffffcu, 0, 8, false));
    assert(!MemoryInline::ResolveSegmentRange(0x80000000u, 0, 0, false));
    // A claimed stack address is not a permission to access runtime metadata.
    const auto savedPolicy = MemoryInline::g_leanBase[MemoryInline::kLeanPolicyOffset];
    bool rejected = false;
    try { MemoryInline::FlatWriteRam32(0xbf800000u, 0); }
    catch (const Memory::AccessViolation&) { rejected = true; }
    assert(rejected && MemoryInline::g_leanBase[MemoryInline::kLeanPolicyOffset] == savedPolicy);
    rejected = false;
    try { (void)MemoryInline::FlatRead32(0xbe000000u); }
    catch (const Memory::AccessViolation&) { rejected = true; }
    assert(rejected);
    // Compare fast-path admission against the configured guest ranges, including
    // every region edge. This oracle does not use the lean offset/page tables.
    const auto config = Memory::Config::WiiDefaults();
    const auto expected = [&](uint32_t address, size_t length) {
        for (const auto& region : config.regions)
            if (address >= region.baseAddress && uint64_t(address) + length <= uint64_t(region.baseAddress) + region.sizeBytes)
                return true;
        return false;
    };
    uint32_t random = 0x42f001u;
    for (unsigned i = 0; i < 100000; ++i) {
        random = random * 1664525u + 1013904223u;
        const size_t length = size_t{1} << (i % 4);
        assert(MemoryInline::LeanRangeMapped(random, length) == expected(random, length));
    }
    for (const auto& region : config.regions)
        for (uint32_t edge : {region.baseAddress, region.baseAddress + uint32_t(region.sizeBytes)})
            for (int delta = -8; delta <= 8; ++delta)
                for (size_t length : {1u, 2u, 4u, 8u}) {
                    const uint32_t address = edge + uint32_t(delta);
                    assert(MemoryInline::LeanRangeMapped(address, length) == expected(address, length));
                }
    // A smaller configured RAM bank must not inherit the full-Wii fallback's
    // assumption that every address below 24 MiB is backed.
    Memory::Init(2u * 1024u * 1024u);
    rejected = false;
    try { MemoryInline::FlatWrite32(0x81000000u, 0x12345678u); }
    catch (const Memory::AccessViolation&) { rejected = true; }
    assert(rejected);
    std::puts("lean alias and range tests passed");
}
