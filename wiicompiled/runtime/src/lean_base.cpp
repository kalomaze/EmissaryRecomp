#include "memory_access.h"

#if MKW_LEAN_MEMORY
#if defined(_WIN32)
#include <windows.h>
#elif defined(__SWITCH__)
extern "C" void* HorizonLeanWindowReserve(uint64_t size);
#else
#include <sys/mman.h>
#endif

namespace {
uint8_t* ReserveLeanWindow() {
    constexpr uint64_t size = MemoryInline::kLeanWindowSize + MemoryInline::kLeanGuardSize;
#if defined(_WIN32)
    return static_cast<uint8_t*>(VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS));
#elif defined(__SWITCH__)
    // Horizon reserves this window in constructor priority 103, before ordinary
    // C++ initialization, to avoid fragmentation by thread stacks.
    return static_cast<uint8_t*>(HorizonLeanWindowReserve(size));
#else
    void* result = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return result == MAP_FAILED ? nullptr : static_cast<uint8_t*>(result);
#endif
}
}

namespace MemoryInline {
// Reserve address space once, before guest execution; Memory::Init backs it.
// This is genuinely const in every translation unit, not a const declaration
// hiding a mutable definition. Initialization reports reservation failure.
uint8_t* const g_leanBase = ReserveLeanWindow();
uint8_t* LeanBaseAtInit() { return g_leanBase; }
}
#endif
