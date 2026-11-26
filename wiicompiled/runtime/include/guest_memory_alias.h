#pragma once
#include <cstdint>

namespace GuestMemoryAlias {
// Offset in the lean backing window. MEM1's cached and uncached views repeat
// every 32 MiB within their 128 MiB apertures. This computes an offset only:
// callers must still validate the guest range (including the 24..32 MiB holes).
constexpr uint32_t Offset(uint32_t address) {
    if ((address & 0xB8000000u) == 0x80000000u)
        return address & 0x01FFFFFFu;
    return address & 0x3FFFFFFFu;
}
constexpr uint32_t Cached(uint32_t address) {
    const uint32_t offset = Offset(address);
    if ((address < 0x01800000u || (address & 0xB8000000u) == 0x80000000u) && offset < 0x01800000u)
        return 0x80000000u + offset;
    if ((address & 0xE0000000u) == 0xC0000000u) return address - 0x40000000u;
    if (address >= 0x10000000u && address < 0x15000000u) return address + 0x80000000u;
    return address;
}
}
