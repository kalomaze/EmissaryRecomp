#include "abi_bridge.h"
#include "ppc_interpreter.h"
#include "memory.h"
#include "guest_code_patch.h"
#include "guest_native_trampoline.h"
#include <cassert>
#include <iterator>
#include <utility>

extern "C" const uint32_t g_mkwGuardedFunctionCount = 4;
extern "C" const uint32_t g_mkwGuardedFunctionStarts[] = {0x80007000, 0x80007020, 0x80007040, 0x80007060};
extern "C" { uint8_t g_mkwGuestCodePatched[4] = {}; }
extern "C" const uint32_t g_mkwInlinedLeafCount = 1;
extern "C" const uint32_t g_mkwInlinedLeaves[] = {1, 0};

extern "C" const uint32_t g_mkwCodeVariantCount = 0;
extern "C" const MkwCodeVariant g_mkwCodeVariants[1] = {};
extern "C" const uint32_t g_mkwCodeVariantGuards[1] = {};
extern "C" const uint32_t g_mkwCodeVariantWords[1] = {};
extern "C" const uint32_t g_mkwCodeVariantRegionCount = 0;
extern "C" const uint32_t g_mkwCodeVariantRegions[1] = {};

static unsigned originalCalls = 0;
static void OriginalNative(CpuContext* ctx) {
    ++originalCalls;
    Memory::Write32(ctx->gpr[1] - 16, ctx->gpr[1]);
    ++ctx->gpr[3];
}

static void GuardedCallee(CpuContext* ctx) {
    MKW_GUEST_CODE_GUARD(2u, 0x80007040u);
    ++ctx->gpr[3];
}

static void TestOriginalTrampoline() {
    constexpr uint32_t original = 0x80007000, trampoline = 0x80007400, wrapper = 0x80007500;
    const auto branch = [](uint32_t from, uint32_t to) { return 0x48000000u | ((to - from) & 0x03FFFFFCu); };
    const uint32_t body[] = {0x9421fff0, 0x38630001, 0x38210010, 0x4e800020,
                             0x60000000, 0x60000000, 0x60000000, 0x60000000};
    for (unsigned i = 0; i < std::size(body); ++i) Memory::Write32(original + 4*i, body[i]);
    TranslatedFunctionInfo info{};
    info.address = original;
    info.name = "OriginalNative";
    info.rawCpuInvoker = OriginalNative;
    info.entryPoint = reinterpret_cast<void*>(OriginalNative);
    TranslatedFunctionRegistry::Register(std::move(info));
    TranslatedFunctionInfo callee{};
    callee.address = 0x80007040u;
    callee.name = "GuardedCallee";
    callee.rawCpuInvoker = GuardedCallee;
    callee.entryPoint = reinterpret_cast<void*>(GuardedCallee);
    TranslatedFunctionRegistry::Register(std::move(callee));
    Memory::Write32(0x80007040u, 0x38630001u); // addi r3, r3, 1
    Memory::Write32(0x80007044u, 0x4e800020u); // blr
    TranslatedFunctionRegistry::Finalize();
    GuestCodePatch::RegisterOriginalNative(original, OriginalNative);
    GuestCodePatch::Protect(original, 0x80007060, "original trampoline test");
    // The mod writes a callee's code and calls its registered entry immediately,
    // after the guest cache-synchronization sequence, without returning through
    // RunPatched or invoking a trampoline resolver.
    constexpr uint32_t patchCaller = 0x80007700u;
    const uint32_t patchAndCall[] = {0x90850000u, 0x7c0028acu, 0x7c0004acu, 0x7c002facu,
                                     0x4c00012cu, 0x4e800421u, 0x7fc803a6u, 0x4e800020u};
    for (unsigned i = 0; i < std::size(patchAndCall); ++i)
        Memory::Write32(patchCaller + 4u*i, patchAndCall[i]);
    CpuContext patchCtx{};
    patchCtx.gpr[1] = 0x8000f000u;
    patchCtx.gpr[3] = 10;
    patchCtx.gpr[4] = 0x38630002u; // replacement: addi r3, r3, 2
    patchCtx.gpr[5] = patchCtx.ctr = 0x80007040u;
    patchCtx.lr = patchCtx.gpr[30] = 0x80007800u;
    // This CPU-only harness stubs the platform executable-write hook. Supply
    // its pending notification; the interpreted stw still performs the edit.
    assert(GuestCodePatch::NoteWrite(0x80007040u));
    PpcInterpreter::Run(&patchCtx, patchCaller);
    assert(patchCtx.gpr[3] == 12);
    Memory::Write32(0x80007040u, 0x38630001u);
    GuestCodePatch::ResetPristine(0x80007040u, 0x80007060u);

    // isync also separates cached decoding: the store replaces an instruction
    // later in this same straight-line sequence, rather than a registered callee.
    constexpr uint32_t selfMod = 0x80007900u;
    const uint32_t selfModCode[] = {0x90850000u, 0x7c0028acu, 0x7c0004acu, 0x7c002facu,
                                   0x4c00012cu, 0x38630001u, 0x4e800020u};
    for (unsigned i = 0; i < std::size(selfModCode); ++i)
        Memory::Write32(selfMod + 4u*i, selfModCode[i]);
    patchCtx.gpr[3] = 10;
    patchCtx.gpr[5] = selfMod + 20u;
    patchCtx.lr = patchCtx.gpr[30];
    PpcInterpreter::Run(&patchCtx, selfMod);
    assert(patchCtx.gpr[3] == 12);

    Memory::Write32(original, branch(original, wrapper));
    Memory::Write32(trampoline, body[0]);
    Memory::Write32(trampoline + 4, branch(trampoline + 4, original + 4));
    assert(GuestCodePatch::ResolveOriginalNativeTrampoline(trampoline).entry == OriginalNative);

    // The mod wrapper must execute both before and after its call to the original.
    const uint32_t hook[] = {0x3863000a, 0x4e800421, 0x38630064, 0x7fc803a6, 0x4e800020};
    for (unsigned i = 0; i < std::size(hook); ++i) Memory::Write32(wrapper + 4*i, hook[i]);
    CpuContext ctx{};
    ctx.gpr[1] = 0x8000f000;
    ctx.gpr[3] = 1;
    ctx.lr = ctx.gpr[30] = 0x80007600;
    ctx.ctr = trampoline;
    PpcInterpreter::Run(&ctx, wrapper);
    assert(originalCalls == 1 && ctx.gpr[3] == 112 && ctx.gpr[1] == 0x8000f000);

    // Live edits must invalidate a previously accepted trampoline or body, without relying on a Sync/cache epoch.
    Memory::Write32(original + 4, 0x38630002);
    assert(GuestCodePatch::ResolveOriginalNativeTrampoline(trampoline).entry == nullptr);
    ctx.gpr[3] = 1;
    ctx.lr = ctx.gpr[30];
    PpcInterpreter::Run(&ctx, wrapper);
    assert(originalCalls == 1 && ctx.gpr[3] == 113 && ctx.gpr[1] == 0x8000f000);
    Memory::Write32(original + 4, body[1]);
    Memory::Write32(trampoline, 0x9421ffe0);
    assert(GuestCodePatch::ResolveOriginalNativeTrampoline(trampoline).entry == nullptr);
    Memory::Write32(trampoline, body[0]);
    Memory::Write32(trampoline + 4, branch(trampoline + 4, original + 4) | 1u);
    assert(GuestCodePatch::ResolveOriginalNativeTrampoline(trampoline).entry == nullptr);
    Memory::Write32(trampoline + 4, branch(trampoline + 4, original + 4));
    assert(GuestCodePatch::ResolveOriginalNativeTrampoline(trampoline).entry == OriginalNative);
    // A mod may patch an inlined leaf and call its original trampoline before
    // returning through RunPatched. Reject that shortcut before the outer Sync.
    const uint32_t leafWord = Memory::Read32(0x80007020);
    Memory::Write32(0x80007020, 0x60000000);
    assert(GuestCodePatch::NoteWrite(0x80007020));
    assert(GuestCodePatch::ResolveOriginalNativeTrampoline(trampoline).entry == nullptr);
    GuestCodePatch::Sync();
    assert(g_mkwGuestCodePatched[0] && g_mkwGuestCodePatched[1]);
    assert(GuestCodePatch::ResolveOriginalNativeTrampoline(trampoline).entry == nullptr);
    Memory::Write32(0x80007020, leafWord);
    GuestCodePatch::ResetPristine(0x80007020, 0x80007040);
    assert(!g_mkwGuestCodePatched[1]);
    assert(GuestCodePatch::ResolveOriginalNativeTrampoline(trampoline).entry == OriginalNative);

    // Relinking the leaf must not erase a pending write to another function on
    // that page, or replace the neighbor's pristine bytes with its patched bytes.
    Memory::Write32(0x80007040, 0x60000000);
    assert(GuestCodePatch::NoteWrite(0x80007040));
    assert(!g_mkwGuestCodePatched[2]);
    GuestCodePatch::ResetPristine(0x80007020, 0x80007040);
    GuestCodePatch::Sync();
    assert(g_mkwGuestCodePatched[2]);

    // An unchanged native body may embed a leaf which a mod replaces separately.
    // The entry detour is allowed, but must not hide this dependency invalidation.
    GuestCodePatch::InterpretReplacedModule(0x80007020, 0x80007040, "inlined leaf replacement test");
    assert(g_mkwGuestCodePatched[0] && g_mkwGuestCodePatched[1]);
    assert(GuestCodePatch::ResolveOriginalNativeTrampoline(trampoline).entry == nullptr);
    ctx.gpr[1] = 0x8000f000;
    ctx.gpr[3] = 1;
    ctx.lr = ctx.gpr[30];
    PpcInterpreter::Run(&ctx, wrapper);
    assert(originalCalls == 1 && ctx.gpr[3] == 112 && ctx.gpr[1] == 0x8000f000);
    GuestCodePatch::InterpretReplacedModule(original, original + sizeof(body), "replacement test");
    assert(GuestCodePatch::ResolveOriginalNativeTrampoline(trampoline).entry == nullptr);
}

int main() { Memory::Init(0x200000); TestOriginalTrampoline(); }
