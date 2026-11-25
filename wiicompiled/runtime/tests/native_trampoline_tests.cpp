#include "abi_bridge.h"
#include "ppc_interpreter.h"
#include "memory.h"
#include "guest_code_patch.h"
#include "guest_native_trampoline.h"
#include <cassert>
#include <iterator>
#include <utility>

extern "C" const uint32_t g_mkwGuardedFunctionCount = 2;
extern "C" const uint32_t g_mkwGuardedFunctionStarts[] = {0x80007000, 0x80007020};
extern "C" { uint8_t g_mkwGuestCodePatched[2] = {}; }
extern "C" const uint32_t g_mkwInlinedLeafCount = 0;
extern "C" const uint32_t g_mkwInlinedLeaves[] = {0, 0};

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
    TranslatedFunctionRegistry::Finalize();
    GuestCodePatch::RegisterOriginalNative(original, OriginalNative);
    GuestCodePatch::Protect(original, original + sizeof(body), "original trampoline test");
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
    GuestCodePatch::InterpretReplacedModule(original, original + sizeof(body), "replacement test");
    assert(GuestCodePatch::ResolveOriginalNativeTrampoline(trampoline).entry == nullptr);
}

int main() { Memory::Init(0x200000); TestOriginalTrampoline(); }
