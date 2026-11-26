// The lean window's host base (memory_access.h, MKW_LEAN_MEMORY): the one writable definition. Every other translation
// unit declares it const, so they may keep it in a register across guest stores; this file is built outside the unity
// batches (CMakeLists.txt) so no other source sees this declaration.
#define MKW_LEAN_BASE_DEFINITION 1
#include "memory_access.h"

#if MKW_LEAN_MEMORY
namespace MemoryInline {
uint8_t* g_leanBase = nullptr;
void SetLeanBase(uint8_t* base) { g_leanBase = base; }
uint8_t* LeanBaseAtInit() { return g_leanBase; }
} // namespace MemoryInline
#endif
