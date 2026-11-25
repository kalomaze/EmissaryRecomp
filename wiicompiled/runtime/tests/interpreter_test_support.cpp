// CPU-only interpreter test support. Unexpected graphics or online calls fail.
#include "brawl_rollback.h"
#include "guest_float_abi.h"
#include "system_bridge.h"
#include "ppc_runtime.h"
#include "recomp_mod_loader.h"
#include "guest_native_trampoline.h"

#include <csetjmp>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <string_view>
#include <stdexcept>

// No translated functions here, so no float ABI facts (the generated table, guest_float_abi.h).
extern "C" const uint32_t g_mkwGuestFloatAbiCount = 0u;
extern "C" const MkwGuestFloatAbi g_mkwGuestFloatAbi[1] = {};

bool g_suppressSehReporting = false;
thread_local jmp_buf* g_sehJumpTarget = nullptr;
thread_local uint32_t g_sehLastExceptionCode = 0;
thread_local uintptr_t g_sehLastExceptionAddress = 0;
thread_local uintptr_t g_sehLastAccessedAddress = 0;
thread_local uint32_t g_sehLastAccessType = 0;

void DebugTrackPcImpl(uint32_t) {}
void DebugPumpAurora() {}
#if defined(__GNUC__) || defined(__clang__)
extern "C" void DumpRecentPcTrace(size_t) __attribute__((weak));
void MarkFatalErrorReported() __attribute__((weak));
#endif
extern "C" void DumpRecentPcTrace(size_t) {}
void MarkFatalErrorReported() {}
extern "C" void DumpHostStackTraceForRuntimeHelper() {}
extern "C" void OS_HLE_ProcessAlarms(int) {}
void ShowRuntimeFatalPopup(std::string_view, std::string_view) noexcept
{
    std::abort();
}

namespace RecompMod {
std::atomic<bool> g_executableWriteGuardEnabled{false};
std::atomic<uint8_t> g_executableWriteGuardPages[kExecutableWriteGuardPageCount]{};
std::atomic<uint8_t> g_executableWriteGuardCoarsePages[kExecutableWriteGuardCoarsePageCount]{};
std::atomic<uint8_t> g_executableWriteGuardMidPages[kExecutableWriteGuardMidPageCount]{};
uint32_t CurrentTranslatedExecutionAddress() noexcept { return 0; }
bool HandleExecutableWrite(uint32_t, size_t, uint64_t) { return false; }
void CheckExecutableWrite(uint32_t, size_t, uint64_t) {}
void RegisterExecutableRange(uint32_t, uint32_t, std::string) {}
}

void SystemBridge::DumpCpuState(const CpuContext*) {}
void SystemBridge::DumpCpuState(std::ostream&, const CpuContext*) {}

namespace BrawlOnline { void Stage() { std::abort(); } }
namespace BrawlRollback {
void Run(CpuContext*, GuestFunction) { std::abort(); }
void RunLoop(CpuContext* ctx, GuestFunction pass) { pass(ctx); throw std::runtime_error("test loop completed"); }
void UpdateSystem(CpuContext*, GuestFunction) { std::abort(); }
void SeedCrossplayStage() { std::abort(); }
bool DrivesLoop() { return false; }
void AnchorPads() { std::abort(); }
LoopCallAction LoopCall(CpuContext*, uint32_t) { std::abort(); }
void LoopCallDone(uint32_t) { std::abort(); }
}
extern "C" void GX_HLE_FIFO_Write8(uint8_t) { std::abort(); }
extern "C" void GX_HLE_FIFO_Write16(uint16_t) { std::abort(); }
extern "C" void GX_HLE_FIFO_Write32(uint32_t) { std::abort(); }
extern "C" void GX_HLE_FIFO_WriteFloat(float) { std::abort(); }

namespace RuntimeCrash {
[[noreturn]] void FatalMissingGuestTarget(uint32_t, CpuContext*) noexcept { std::abort(); }
}

namespace InterpreterTest {
unsigned frameEnds = 0, loopEnds = 0, soundHooks = 0;
}
namespace BrawlOnlineCore {
bool Enabled() { return std::getenv("BRAWL_TEST_ONLINE_CORE") != nullptr; }
bool FrameEnd(CpuContext*) { ++InterpreterTest::frameEnds; return true; }
void LoopEnd(CpuContext*) { ++InterpreterTest::loopEnds; }
uint32_t SoundHook(CpuContext*, uint32_t pc) { ++InterpreterTest::soundHooks; return pc + 4; }
}
#if !defined(MKW_TEST_REAL_CODE_PATCH)
namespace GuestCodePatch {
void Sync() {}
OriginalNativeTarget ResolveOriginalNativeTrampoline(uint32_t) { return {}; }
static void TestContinuation(CpuContext* ctx) { ctx->gpr[3] = 0x55; }
void (*ContinuationAt(uint32_t address))(CpuContext*) {
    return address == 0x8000d004u ? TestContinuation : nullptr;
}
}

#endif
