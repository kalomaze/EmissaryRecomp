// Run-time patches to the game's translated code (Gecko codes, Syriinge plugin hooks).
//
// Once a mod that patches code is active, translated code ranges are write-trapped (RecompMod's executable guard).
// A written page is compared with its pristine bytes at the next Sync, and every translated function whose bytes
// changed runs in the interpreter from then on, through the MKW_GUEST_CODE_GUARD at its entry (abi_bridge.h).
#pragma once

#include <cstdint>

struct CpuContext;

// Code variants (generated: build_shards/base_dispatch/code_variants_*.cpp, from translate-recursive --code-variants):
// translations of functions as a mod's fixed code patches leave them, hook bodies included. A patched function runs
// its variant instead of the interpreter while guest memory holds exactly the words the variant was translated from:
// guards [firstGuard, firstGuard + guardCount) of g_mkwCodeVariantGuards, each (start address, word count, index of
// its first word in g_mkwCodeVariantWords). Empty in builds without variants.
struct MkwCodeVariant {
    uint32_t address;
    void (*entry)(CpuContext*);
    uint32_t firstGuard;
    uint32_t guardCount;
};
extern "C" const uint32_t g_mkwCodeVariantCount;
extern "C" const MkwCodeVariant g_mkwCodeVariants[];
extern "C" const uint32_t g_mkwCodeVariantGuards[];
extern "C" const uint32_t g_mkwCodeVariantWords[];
// [start, end) pairs: memory outside translated code that variants were translated from (a code list's hook bodies);
// write-trapped like guarded code once a mod patches code, so a change there re-checks the variants.
extern "C" const uint32_t g_mkwCodeVariantRegionCount;
extern "C" const uint32_t g_mkwCodeVariantRegions[];

namespace GuestCodePatch {

// Write-traps the translated code in [start, end); its current bytes are the pristine state.
void Protect(uint32_t start, uint32_t end, const char* name);
// The trap's report of a write to a page holding guarded code. False when the page holds none.
bool NoteWrite(uint32_t address);
// Compares written pages with their pristine bytes; functions whose code changed go to the interpreter.
void Sync();
// Sync, once a frame: also guards again the pages that hold both code and data.
void SyncFrame();
// [start, end) holds its pristine code again (a module was linked again at its home).
void ResetPristine(uint32_t start, uint32_t end);
// A patch a mod made to code not guarded yet (the code handler's boot pass over a resident module): `value` is the
// word as the guest stores it.
void RecordPatch(uint32_t address, uint32_t value);
// Writes every patch known in [start, end) back over freshly linked code and syncs. On a Wii the code handler runs
// many times while a module loads, so its hooks are in place again before the module's prolog runs.
void Reapply(uint32_t start, uint32_t end);
bool Active();
// After the code handler's first pass: functions holding baked words (BRAWL_BAKED_PATCHES) whose code now matches the
// bake run translated from here on.
void ReleaseBaked();
void RunPatched(CpuContext* ctx, uint32_t address);
// A code variant entered at `address` inside a function whose memory holds the words it was translated from (a plugin
// loader's "call the original" stub branches to a replaced function's second instruction), else null. For the
// interpreter's branches to addresses no translated function starts at.
void (*ContinuationAt(uint32_t address))(CpuContext*);
// Whether ContinuationAt(address) can ever answer non-null, whatever guest memory holds: a code variant is keyed at the
// address (true before the variants are set up). For a JIT block to settle a branch to address at compile time.
bool ContinuationPossibleAt(uint32_t address);
// The game linked a module in place over [start, end) (its bytes came from a file read, past the write trap): the code
// variants with words there are checked against guest memory again before they next run.
void ModuleLinkedInPlace(uint32_t start, uint32_t end);
// BRAWL_INTERPRET=<hex start>-<hex end>: every translated function starting in that range runs in the interpreter, a
// check of the interpreter against the translation on the unmodded game (same behaviour, only slower). Once the
// function registry is final.
void InterpretFromEnvironment();
// A mod's replacement of a resident module links where the disc's translated functions are registered: every
// translated function in [start, end) runs the bytes there in the interpreter from now on.
void InterpretReplacedModule(uint32_t start, uint32_t end, const char* name);
// The translated function at `address` runs in the interpreter from now on, for good: a hook of the runtime's at a pc
// inside it (ppc_interpreter.cpp; Brawl Online's sound hooks, which act in the middle of the game's functions as
// Dolphin's HLE hooks do). `why` is logged.
void InterpretForHooks(uint32_t address, const char* why);
// A module the game linked in place instead of at its translated home (a mod's, or any loaded on demand while codes
// are active) is still reached at the home by translated callers (scene tables in sora_scene): every translated
// function in [homeStart, homeEnd) runs the code at the same offset from `target` in the interpreter, until
// ClearRedirect(homeStart) (the module linked at its home again).
void RedirectModule(uint32_t homeStart, uint32_t homeEnd, uint32_t target, const char* name);
void ClearRedirect(uint32_t homeStart);

} // namespace GuestCodePatch
