// CPU-only interpreter test support. Unexpected graphics or online calls fail.
#include "brawl_rollback.h"
#include "guest_code_patch.h"
#include "guest_float_abi.h"
#include "system_bridge.h"
#include "ppc_runtime.h"
#include "recomp_mod_loader.h"

#include <csetjmp>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <string_view>
#include <stdexcept>

// No translated functions here, so no float ABI facts (the generated table, guest_float_abi.h), except for the functions the
// call tests register: 64 at 0x80040000 + 0x10 * i ("thin calls": every guard configuration with every float ABI) and 24 at
// 0x80041000 + 0x10 * i (the patched functions of the call tests). Sorted by address, as the generated table is.
namespace {
constexpr uint16_t kTestArgMasks[6] = {0x0000, 0x0002, 0x0006, 0x2000, 0x3FFE, 0x0A0A};
constexpr uint32_t kTestAbiCount = 64 + 24;
struct TestAbiTable {
    MkwGuestFloatAbi entries[kTestAbiCount];
    constexpr TestAbiTable() : entries() {
        for (uint32_t i = 0; i < 64; ++i)
            entries[i] = {0x80040000u + 0x10u * i, kTestArgMasks[i % 6], static_cast<uint16_t>((i / 6) % 2)};
        for (uint32_t i = 0; i < 24; ++i)
            entries[64 + i] = {0x80041000u + 0x10u * i, kTestArgMasks[(i + 1) % 6], static_cast<uint16_t>((i / 3) % 2)};
    }
};
constexpr TestAbiTable kTestAbi;
}  // namespace
extern "C" const uint32_t g_mkwGuestFloatAbiCount = kTestAbiCount;
extern "C" const MkwGuestFloatAbi g_mkwGuestFloatAbi[kTestAbiCount] = {
#define A(i) kTestAbi.entries[i]
#define A8(i) A(i), A(i + 1), A(i + 2), A(i + 3), A(i + 4), A(i + 5), A(i + 6), A(i + 7)
    A8(0), A8(8), A8(16), A8(24), A8(32), A8(40), A8(48), A8(56), A8(64), A8(72), A8(80)
#undef A8
#undef A
};

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
// Called on fatal paths (abi_bridge.h's missing jump target, nand_path.h) and by the locked-cache DMA (ppc_helpers.cpp).
// On Linux --gc-sections drops the code that calls them; Windows' linker still wants every one defined.
void SetRuntimeExitCode(int) {}
namespace RuntimeCrash {
void WriteCrashArtifacts(std::string_view, std::string_view, const uint32_t*) noexcept {}
}
extern "C" void GxNotifyGuestRamDmaWrite(uint32_t, uint32_t) {}
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
// The address of the last store the checked path reported as one into code (guest_memory_map_tests reads it).
std::atomic<uint32_t> g_testLastExecutableWrite{0};
bool HandleExecutableWrite(uint32_t address, size_t, uint64_t) {
    g_testLastExecutableWrite.store(address, std::memory_order_relaxed);
    return false;
}
void CheckExecutableWrite(uint32_t, size_t, uint64_t) {}
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
namespace GuestCodePatch {
static void TestContinuation(CpuContext* ctx) { ctx->gpr[3] = 0x55; }
void (*ContinuationAt(uint32_t address))(CpuContext*) {
    return address == 0x8000d004u ? TestContinuation : nullptr;
}
bool ContinuationPossibleAt(uint32_t address) { return address == 0x8000d004u; }

// A double of the patch machinery for the call tests: a small table saying where a call of a "patched" registered function X runs
// guest code (TestSetInterpreted(X, X'); X' 0: the function's own body runs). The test's function for X does what RunPatched does
// with the answer: PpcInterpreter::Run(ctx, TestInterpretedAt(X)) and Sync() when it is not 0.
namespace {
struct TestPatched {
    uint32_t address = 0, at = 0;
};
TestPatched g_testPatched[32];
unsigned g_testPatchedCount = 0;
std::atomic<unsigned> g_testSyncs{0};
}  // namespace
void TestSetInterpreted(uint32_t address, uint32_t at) {
    for (unsigned i = 0; i < g_testPatchedCount; ++i)
        if (g_testPatched[i].address == address) {
            g_testPatched[i].at = at;
            return;
        }
    if (g_testPatchedCount == 32) std::abort();
    g_testPatched[g_testPatchedCount++] = {address, at};
}
uint32_t TestInterpretedAt(uint32_t address) {
    for (unsigned i = 0; i < g_testPatchedCount; ++i)
        if (g_testPatched[i].address == address) return g_testPatched[i].at;
    return 0;
}
unsigned TestSyncCount() { return g_testSyncs.load(std::memory_order_relaxed); }
void Sync() { g_testSyncs.store(g_testSyncs.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed); }
}
