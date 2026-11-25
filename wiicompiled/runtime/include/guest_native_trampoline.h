#pragma once

#include <cstdint>

struct CpuContext;

namespace GuestCodePatch {

struct OriginalNativeTarget {
    uint32_t address = 0;
    void (*entry)(CpuContext*) = nullptr;
};

// Register a native hook's unguarded entry, including its translated fallback. Calls through an original-function
// trampoline can use it when the copied prologue and the rest of the original body still match pristine code.
void RegisterOriginalNative(uint32_t address, void (*entry)(CpuContext*));
OriginalNativeTarget ResolveOriginalNativeTrampoline(uint32_t trampoline);

} // namespace GuestCodePatch
