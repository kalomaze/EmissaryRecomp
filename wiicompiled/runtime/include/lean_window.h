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
// +1, +2 and +3 GiB, the window's first 64 KiB at +4 GiB (the tail), and the 64 KiB below the window and after the tail
// are reserved and never backed. Then `base + ea` reaches the byte at `base + (ea & kLeanMask)` for every 32-bit ea, and
// `base + ea + d` for a displacement |d| < 32 KiB reaches the byte at ((ea + d) mod 2^32) & kLeanMask or faults where that
// byte is unbacked as well (below the window: the window's top 64 KiB, never backed). The JIT's loads use it (no mask);
// everything else keeps masking, which reads the same bytes.
inline constexpr uint64_t kAliasViews = 4;
inline constexpr uint64_t kAliasTail = uint64_t{1} << 16;
extern bool g_aliased;

}  // namespace LeanWindow
