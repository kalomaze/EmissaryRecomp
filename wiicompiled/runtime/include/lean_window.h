// The lean guest-memory window's address space (memory_access.h, MKW_LEAN_MEMORY; Memory::Init in memory.cpp): one
// reservation, backed in parts, where some parts are views of one shared object so that several guest addresses reach
// the same bytes (MEM1's mirrors). The OS calls only: memory.cpp decides what goes where.
#pragma once

#include <cstdint>
#include <string>

namespace LeanWindow {

// `size` bytes of address space, nothing backed; null on failure. Windows: one placeholder, which later calls split
// and replace (Commit, MapShared). Horizon: the window platform/horizon.c set aside at start-up.
uint8_t* Reserve(uint64_t size);

// Private memory reading zero, read-write, over [at, at + size) of the reservation (64 KiB aligned, not backed yet):
// 0, or the OS's error code.
uint32_t Commit(uint8_t* at, uint64_t size);

// One new shared object of `size` bytes (64 KiB aligned), mapped read-write at each of `targets[0..count)` (64 KiB
// aligned, inside the reservation, not backed yet): every target shows the same bytes, which read zero. False, with
// `error` set, on failure or where the OS gives no way to map one object twice (Horizon: svcMapMemory takes its source
// pages away).
bool MapShared(uint64_t size, uint8_t* const* targets, int count, std::string* error);

// The general form: one new shared object of `size` bytes (64 KiB aligned), and views[i] shows its bytes [offset,
// offset + length) at `at` (all 64 KiB aligned, inside the reservation, not backed yet). Views of the same offsets show
// the same bytes, which read zero. False, with `error` set, as MapShared.
struct View {
    uint8_t* at;
    uint64_t offset, length;
};
bool MapSharedViews(uint64_t size, const View* views, int count, std::string* error);

// The aliased view (Linux; Memory::Init sets g_aliased when it made it): everything the window backs is backed again at
// +1, +2, +3 and +4 GiB, the window's first 64 KiB at +5 GiB (the tail), and the 64 KiB below the window and after the
// tail are reserved and never backed. Then `base + ea` reaches the byte at `base + (ea & kLeanMask)` for every 32-bit ea,
// and `base + ea + d` for a displacement |d| < 32 KiB reaches the byte at ((ea + d) mod 2^32) & kLeanMask or faults where
// that byte is unbacked as well (below the window: the window's top 64 KiB, never backed). The JIT's loads use it (no
// mask); everything else keeps masking, which reads the same bytes.
inline constexpr uint64_t kAliasViews = 5;
inline constexpr uint64_t kAliasTail = uint64_t{1} << 16;
extern bool g_aliased;

// The store view (Linux, with the aliased view; BRAWL_LEAN_STORE_VIEW=0 turns it off; Memory::Init sets g_storeView):
// `base + kStoreViewOffset + ea` (+ d, |d| < 32 KiB) reaches the same bytes as the aliased view one GiB up, and is
// writable exactly where a guest store at ea takes the lean fast path: the copy at +3 GiB (ea 0x80000000-0xBFFFFFFF,
// the cached view) is read-write in every chunk whose store policy byte is clear and read-only elsewhere, and every
// other copy (+1, +2, +4 GiB and the tail: the physical and uncached views, and what wraps past 2^32) is read-only. A
// store there that the store policy would send to the checked path faults (memory.cpp keeps the protection in step with
// the policy, RefreshLeanStorePolicy); the JIT stores through it and handles the fault (jit_x64_mem.cpp). Loads through
// the aliased view read every copy as before. The window itself (+0) stays read-write: every other writer masks.
inline constexpr uint64_t kStoreViewOffset = uint64_t{1} << 30;
extern bool g_storeView;

// Read-write or read-only for [at, at + size) of the reservation (backed, 4 KiB aligned): 0, or the OS's error code.
// POSIX only (elsewhere: a non-zero answer, nothing changed).
uint32_t Protect(uint8_t* at, uint64_t size, bool writable);

}  // namespace LeanWindow
