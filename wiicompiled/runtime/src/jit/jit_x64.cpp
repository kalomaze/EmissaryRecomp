// The x86-64 backend's core (docs/jit-design.md): the enter trampoline and exit stubs, a block's prologue and validity
// test, the op loop that routes each op to its group or to the interpreter's handler, exits and link slots, and the
// code cache. The native code of each op group is in jit_x64_{int,mem,branch,fp}.cpp, written against jit_x64.h.
#include "jit_x64.h"
#include "jit_x64_generic.h"

#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

#include "brawl_dirty_memory.h"
#include "jit/jit_exec_memory.h"
#include "memory_access.h"

#if defined(__linux__)
#include <unistd.h>
#endif

#if defined(__linux__) && defined(__x86_64__)
extern "C" {
void __register_frame(const void* fde);
void __deregister_frame(const void* fde);
}
#endif

namespace PpcInterpreter::detail {
namespace x64 {
namespace {

using namespace Xbyak::util;

// Thrown by the emitter when one of its fixed pools is full: the block is not compiled.
struct EmitError {
    const char* what;
};

// --- options and diagnostics (all read once from the environment) ------------------------------------------------------

bool EnvOn(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && *v != '0';
}

struct Options {
    bool perfMap = false;
    bool stats = false;
    bool execInMemory = false;  // BRAWL_JIT_EXEC_MEM=1: every block adds its count to frame.executed in memory
    bool blockPc = false;       // BRAWL_JIT_BLOCK_PC=1: every block entry stores ctx->pc
    uint32_t staleThreshold = 32;  // BRAWL_JIT_STALE_STUB=<n>: stale paths of a block before it gets a stub; 0 never
    uint32_t dumpPcs[16] = {};
    unsigned dumpCount = 0;
    bool forced[kKinds] = {};  // kinds that always take the fallback call
    Group group[kKinds] = {};

    Options() {
        for (uint32_t k = 0; k < kKinds; ++k) group[k] = GroupOf(static_cast<K>(k));
        perfMap = EnvOn("BRAWL_JIT_PERF_MAP");
        stats = EnvOn("BRAWL_JIT_STATS");
        execInMemory = EnvOn("BRAWL_JIT_EXEC_MEM");
        blockPc = EnvOn("BRAWL_JIT_BLOCK_PC");
        if (const char* v = std::getenv("BRAWL_JIT_STALE_STUB")) staleThreshold = static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
        if (const char* dump = std::getenv("BRAWL_JIT_DUMP")) {
            for (const char* p = dump; *p != '\0' && dumpCount < 16;) {
                char* end = nullptr;
                const unsigned long pc = std::strtoul(p, &end, 16);
                if (end == p) break;
                dumpPcs[dumpCount++] = static_cast<uint32_t>(pc);
                p = end;
                while (*p == ',' || *p == ' ') ++p;
            }
        }
        if (const char* list = std::getenv("BRAWL_JIT_FALLBACK")) ParseFallback(list);
    }

    static bool EqualsNoCase(const std::string& a, const char* b) {
        if (a.size() != std::strlen(b)) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) return false;
        return true;
    }

    // BRAWL_JIT_FALLBACK: comma-separated group names (int, mem, branch, fp), op kind names (kKindNames), or all.
    void ParseFallback(const char* list) {
        static const struct { const char* name; Group group; } kGroups[] = {
            {"int", Group::Int}, {"mem", Group::Mem}, {"branch", Group::Branch}, {"fp", Group::Fp}};
        std::string token;
        for (const char* p = list;; ++p) {
            if (*p != ',' && *p != '\0') {
                if (*p != ' ') token += *p;
                continue;
            }
            if (!token.empty()) {
                bool known = false;
                if (EqualsNoCase(token, "all")) {
                    for (bool& f : forced) f = true;
                    known = true;
                }
                for (const auto& g : kGroups)
                    if (EqualsNoCase(token, g.name)) {
                        for (uint32_t k = 0; k < kKinds; ++k)
                            if (group[k] == g.group) forced[k] = true;
                        known = true;
                    }
                for (uint32_t k = 0; k < kKinds; ++k)
                    if (EqualsNoCase(token, kKindNames[k])) {
                        forced[k] = true;
                        known = true;
                    }
                if (!known) std::fprintf(stderr, "jit: BRAWL_JIT_FALLBACK: unknown group or op kind '%s'\n", token.c_str());
                token.clear();
            }
            if (*p == '\0') break;
        }
    }
};

const Options& Opts() {
    static const Options options;
    return options;
}

struct Counters {
    std::atomic<uint64_t> compiled{0}, compileFailed{0}, codeBytes{0}, links{0}, retires{0}, enters{0}, cachesCreated{0},
        cachesDestroyed{0}, compiledOps{0}, compileNanos{0}, cacheHits{0}, cacheMisses{0}, staleStubs{0}, staleStubFailures{0};
};
Counters g_counters;
// Ops compiled, per kind: [0] with native code, [1] as a call to the interpreter's handler (a fused pair counts once, at
// its first op's kind).
std::atomic<uint64_t> g_kindOps[2][kKinds];

void Bump(std::atomic<uint64_t>& c, uint64_t n = 1) { c.fetch_add(n, std::memory_order_relaxed); }

void LogStats(const char* when) {
    const JitX64Stats s = JitX64GetStats();
    std::fprintf(stderr,
                 "jit (%s): %llu blocks compiled (%llu failed), %llu code bytes (%.1f per guest instruction), %.2f us per "
                 "block, %llu links, %llu retires, %llu enters, %llu caches made, %llu destroyed, inline cache %llu hits "
                 "%llu misses\n",
                 when, static_cast<unsigned long long>(s.compiled), static_cast<unsigned long long>(s.compileFailed),
                 static_cast<unsigned long long>(s.codeBytes),
                 s.compiledOps ? static_cast<double>(s.codeBytes) / s.compiledOps : 0.0,
                 s.compiled ? s.compileNanos / 1000.0 / s.compiled : 0.0, static_cast<unsigned long long>(s.links),
                 static_cast<unsigned long long>(s.retires), static_cast<unsigned long long>(s.enters),
                 static_cast<unsigned long long>(s.cachesCreated), static_cast<unsigned long long>(s.cachesDestroyed),
                 static_cast<unsigned long long>(s.cacheHits), static_cast<unsigned long long>(s.cacheMisses));
}

#if defined(__linux__)
std::mutex g_perfMutex;
FILE* PerfMap() {
    static FILE* const file = [] {
        char path[64];
        std::snprintf(path, sizeof(path), "/tmp/perf-%d.map", static_cast<int>(getpid()));
        FILE* f = std::fopen(path, "a");
        if (f != nullptr) setvbuf(f, nullptr, _IOLBF, 0);
        return f;
    }();
    return file;
}
#endif

void PerfMapAdd(const void* at, size_t size, const char* name, uint32_t pc) {
#if defined(__linux__)
    if (!Opts().perfMap) return;
    std::lock_guard<std::mutex> lock(g_perfMutex);
    if (FILE* f = PerfMap()) {
        if (pc != 0) std::fprintf(f, "%llx %zx %s_%08X\n", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(at)), size, name, pc);
        else std::fprintf(f, "%llx %zx %s\n", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(at)), size, name);
    }
#else
    (void)at; (void)size; (void)name; (void)pc;
#endif
}

void DumpBlock(const Block& b, const void* at, size_t size) {
    static std::atomic<uint32_t> serial{0};
    const Options& o = Opts();
    for (unsigned i = 0; i < o.dumpCount; ++i) {
        if (o.dumpPcs[i] != b.start) continue;
        const uint32_t n = serial.fetch_add(1, std::memory_order_relaxed);
        char path[64];
        if (n == 0) std::snprintf(path, sizeof(path), "jit_%08X.bin", b.start);
        else std::snprintf(path, sizeof(path), "jit_%08X_%u.bin", b.start, n);
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(at, 1, size, f);
            std::fclose(f);
        }
        std::fprintf(stderr, "jit: block %08X (%u ops) at host %p, %zu bytes, dumped to %s\n", b.start, b.count, at, size, path);
    }
}

// --- the host ABI ------------------------------------------------------------------------------------------------------

// Frame layout shared identically by jit_enter and jit_enter_direct so one constant DWARF FDE describes every JIT block
// and direct-enter continuation:
//   [rsp +  0..31]: Win64 shadow space
//   [rsp + 32..39]: out-parameter slot (jit_enter: JitLink**) or DirectEnterState* (jit_enter_direct)
//   [rsp + 40..87]: in-place Frame (48 bytes, used by jit_enter_direct)
// 88 == 8 mod 16, so rsp is 16-byte aligned throughout every block.
constexpr int kTrampolineFrame = 88;
constexpr int kOutLinkSlot = 32;
constexpr int kDirectFrameOffset = 40;
static_assert(sizeof(Frame) == 48, "kTrampolineFrame assumes 48-byte Frame");
static_assert(kDirectFrameOffset + static_cast<int>(sizeof(Frame)) == kTrampolineFrame);
static_assert(offsetof(Frame, flags) == offsetof(Frame, pairedMask) + 4, "pairedMask and flags packed for one 8-byte store");
static_assert(offsetof(DirectEnterState, insnCount) == 0, "exitReturn loads insnCount from [state]");
static_assert(sizeof(FastEntrySlot) == 16 && offsetof(FastEntrySlot, pc) == 0 && offsetof(FastEntrySlot, jitChecked) == 8);
static_assert(kFastEntrySlots == 2048, "subCall hash shift assumes 2048 slots");

#if defined(__linux__) && defined(__x86_64__)
// Compact 64-byte DWARF .eh_frame (CIE + FDE + 0 terminator) describing the constant frame of jit_enter / jit_enter_direct
// on Linux x86-64: CFA = rsp + 144 (8 ret + 48 six callee-saved pushes + 88 kTrampolineFrame), rip at CFA-8, rbp..r15 at
// CFA-16..CFA-56. Registered once per 2 MiB executable chunk (and once for the shared trampoline region).
struct EhFrameRecord {
    alignas(8) uint8_t bytes[64];
    bool registered = false;

    void Register(const void* base, size_t size) {
        if (registered || base == nullptr || size == 0) return;
        std::memset(bytes, 0, sizeof(bytes));
        // CIE (bytes 0..31): length 28, id 0, version 1, aug "", caf 1, daf -8, ra 16
        const uint32_t cieLen = 28;
        std::memcpy(bytes + 0, &cieLen, 4);
        // bytes 4..7 = 0 (CIE_id)
        bytes[8] = 1;     // version
        bytes[9] = 0;     // augmentation "" (DW_EH_PE_absptr)
        bytes[10] = 1;    // code_alignment_factor = 1
        bytes[11] = 0x78; // data_alignment_factor = -8 (SLEB128)
        bytes[12] = 16;   // return_address_register = rip (16)
        // DW_CFA_def_cfa r7 (rsp), 144 (ULEB128: 0x90, 0x01)
        static_assert(8 + 6 * 8 + kTrampolineFrame == 144, "CFA offset must match ULEB128 144");
        bytes[13] = 0x0c; bytes[14] = 7; bytes[15] = 0x90; bytes[16] = 0x01;
        // DW_CFA_offset reg, factored_offset (CFA - 8 * factored_offset)
        bytes[17] = 0x80 | 16; bytes[18] = 1;  // rip  @ CFA - 8
        bytes[19] = 0x80 | 6;  bytes[20] = 2;  // rbp  @ CFA - 16
        bytes[21] = 0x80 | 3;  bytes[22] = 3;  // rbx  @ CFA - 24
        bytes[23] = 0x80 | 12; bytes[24] = 4;  // r12  @ CFA - 32
        bytes[25] = 0x80 | 13; bytes[26] = 5;  // r13  @ CFA - 40
        bytes[27] = 0x80 | 14; bytes[28] = 6;  // r14  @ CFA - 48
        bytes[29] = 0x80 | 15; bytes[30] = 7;  // r15  @ CFA - 56
        bytes[31] = 0x00;                      // DW_CFA_nop
        // FDE (bytes 32..59): length 24, CIE_pointer 36, pc_begin (8), pc_range (8), 4 nops
        const uint32_t fdeLen = 24;
        const uint32_t ciePtr = 36;
        const uint64_t pcBegin = reinterpret_cast<uintptr_t>(base);
        const uint64_t pcRange = static_cast<uint64_t>(size);
        std::memcpy(bytes + 32, &fdeLen, 4);
        std::memcpy(bytes + 36, &ciePtr, 4);
        std::memcpy(bytes + 40, &pcBegin, 8);
        std::memcpy(bytes + 48, &pcRange, 8);
        // bytes 56..59 = 0 (nops), bytes 60..63 = 0 (terminator)
        __register_frame(bytes);
        registered = true;
    }

    ~EhFrameRecord() {
        if (registered) __deregister_frame(bytes);
    }
};
#endif

// The trampoline's raw signature, the same on both ABIs: a packed result (status << 32 | next pc), the link slot through
// an out-parameter.
using EnterFn = uint64_t (*)(Frame*, const void* entry, JitLink** link);

// --- code shared by every block: the enter trampoline and the exits ----------------------------------------------------------

struct Shared {
    JitExecMemory::Region region;
    EnterFn enter = nullptr;
    EnterDirectFn enterDirect = nullptr;
    const void* subCall = nullptr;      // Linux x86-64: eax = target, rdx = const JitCallTarget*
    const void* exitDynamic = nullptr;  // eax = the next pc
    const void* exitSlot = nullptr;     // eax = the next pc, rdx = the link slot
    const void* exitReturn = nullptr;
    const void* exitPending = nullptr;
    const void* staleCount = nullptr;   // rcx = a block's StaleCell: counts the failure, at the threshold asks for the block's stub
#if defined(__linux__) && defined(__x86_64__)
    std::unique_ptr<EhFrameRecord> ehFrame;
#endif
    bool ok = false;
};
Shared g_shared;
EnterFn g_enter = nullptr;  // g_shared.enter, as a plain global: read on every enter

bool BuildShared() {
    Shared& s = g_shared;
    s.region = JitExecMemory::Allocate(4096);
    if (!s.region) return false;
    try {
        Xbyak::CodeGenerator a(s.region.size, s.region.base);
        Xbyak::Label epilogue, directEpilogue, directContinue, exitReturnSlow, exitSlotSlow;
        Xbyak::Label subCallReturn, subCallDone, subCallSync;
        const uint8_t* const enterAt = a.getCurr();
        // One frame for every block: the callee-saved registers, then the shadow space, out slot, and direct Frame area.
        a.push(rbp);
        a.push(rbx);
        a.push(r12);
        a.push(r13);
        a.push(r14);
        a.push(r15);
        if (kWin64) {
            a.push(rsi);
            a.push(rdi);
        }
        a.sub(rsp, kTrampolineFrame);
        a.mov(a.qword[rsp + kOutLinkSlot], ArgReg(2));
        a.mov(kFrame, ArgReg(0));
        a.mov(kCtxBase, a.qword[kFrame + static_cast<int>(offsetof(Frame, ctx))]);
        a.add(kCtxBase, kCtxBias);
        a.mov(kEpochCell, reinterpret_cast<uint64_t>(&BrawlDirtyMemory::markEpoch));
        // The window base is read at every enter: tests re-initialise memory, so it is no constant of the code.
        a.mov(kLean, reinterpret_cast<uint64_t>(&MemoryInline::g_leanBase));
        a.mov(kLean, a.qword[kLean]);
        a.xor_(kExecCount.cvt32(), kExecCount.cvt32());  // the guest instructions counted so far this enter (see kExecCount)
        a.jmp(ArgReg(1));

        // Fast AOT->JIT entry: void enterDirect(CpuContext* ctx, uint32_t entry, const void* jitChecked, DirectEnterState* state).
        // Uses the exact same stack frame layout as jit_enter so chunk FDEs unwind both identically.
        const uint8_t* const enterDirectAt = a.getCurr();
        a.push(rbp);
        a.push(rbx);
        a.push(r12);
        a.push(r13);
        a.push(r14);
        a.push(r15);
        if (kWin64) {
            a.push(rsi);
            a.push(rdi);
        }
        a.sub(rsp, kTrampolineFrame);
        a.mov(a.qword[rsp + kOutLinkSlot], ArgReg(3));
        a.lea(kFrame, a.ptr[rsp + kDirectFrameOffset]);
        a.mov(a.qword[kFrame + static_cast<int>(offsetof(Frame, ctx))], ArgReg(0));
        // pairedMask = 0 (low 32), flags = kFrameFlagDirectEnter (high 32)
        a.mov(rax, static_cast<uint64_t>(kFrameFlagDirectEnter) << 32);
        a.mov(a.qword[kFrame + static_cast<int>(offsetof(Frame, pairedMask))], rax);
        a.mov(a.dword[kFrame + static_cast<int>(offsetof(Frame, pc))], ArgReg(1).cvt32());
        a.mov(eax, a.dword[ArgReg(0) + static_cast<int>(offsetof(CpuContext, lr))]);
        a.mov(a.dword[kFrame + static_cast<int>(offsetof(Frame, returnAddress))], eax);
        a.mov(a.dword[kFrame + static_cast<int>(offsetof(Frame, entry))], ArgReg(1).cvt32());
        a.mov(a.qword[kFrame + static_cast<int>(offsetof(Frame, executed))], 0);
        a.mov(a.qword[kFrame + static_cast<int>(offsetof(Frame, pending))], 0);
        a.lea(kCtxBase, a.ptr[ArgReg(0) + kCtxBias]);
        a.mov(a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, pc))], ArgReg(1).cvt32());
        a.mov(kEpochCell, reinterpret_cast<uint64_t>(&BrawlDirtyMemory::markEpoch));
        a.mov(kLean, reinterpret_cast<uint64_t>(&MemoryInline::g_leanBase));
        a.mov(kLean, a.qword[kLean]);
        a.xor_(kExecCount.cvt32(), kExecCount.cvt32());
        a.jmp(ArgReg(2));

        // Fast JIT->patched-JIT direct subcall (Linux x86-64 only): bypasses AOT wrapper, RunPatched, RunCompiledEntry,
        // and reloading kCtxBase/kLean/kEpochCell, while keeping the exact 144-byte trampoline frame for chunk FDE unwind
        // and preserving hardware RSB via host call/ret. Inputs: eax = target, rdx = const JitCallTarget*.
        if (kDirectCalls) {
            Xbyak::Label fallbackEntry;
            s.subCall = a.getCurr();
            a.test(a.dword[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagDirectEnter | kFrameFlagSubCall);
            a.jz(fallbackEntry, Xbyak::CodeGenerator::T_NEAR);
            a.mov(rcx, a.qword[rdx + static_cast<int>(offsetof(JitCallTarget, patchedByte))]);
            a.test(rcx, rcx);
            a.jz(fallbackEntry, Xbyak::CodeGenerator::T_NEAR);
            a.cmp(a.byte[rcx], 0);
            a.je(fallbackEntry, Xbyak::CodeGenerator::T_NEAR);
            a.mov(r9, reinterpret_cast<uint64_t>(&GetPatchedHooks()));
            a.mov(rcx, a.qword[r9 + static_cast<int>(offsetof(PatchedHooks, lowestRedirect))]);
            a.cmp(eax, a.dword[rcx]);
            a.jae(fallbackEntry, Xbyak::CodeGenerator::T_NEAR);
            a.mov(rsi, a.qword[rsp + 8 + kOutLinkSlot]);  // caller's DirectEnterState*
            a.mov(rdi, a.qword[rsi + static_cast<int>(offsetof(DirectEnterState, fastEntries))]);
            a.mov(ecx, eax);
            a.shr(ecx, 2);
            a.imul(ecx, ecx, 0x9E3779B1u);
            a.shr(ecx, 21);
            a.shl(rcx, 4);
            a.cmp(a.dword[rdi + rcx], eax);
            a.jne(fallbackEntry, Xbyak::CodeGenerator::T_NEAR);
            a.mov(r8, a.qword[rdi + rcx + 8]);
            a.test(r8, r8);
            a.jz(fallbackEntry, Xbyak::CodeGenerator::T_NEAR);
            // Hit: build identical 144-byte trampoline frame so chunk FDEs unwind arbitrarily nested subcalls.
            a.push(rbp);
            a.push(rbx);
            a.push(r12);
            a.push(r13);
            a.push(r14);
            a.push(r15);
            a.sub(rsp, kTrampolineFrame);
            a.mov(a.qword[rsp + kOutLinkSlot], rsi);
            a.lea(kFrame, a.ptr[rsp + kDirectFrameOffset]);
            a.lea(rcx, a.ptr[kCtxBase - kCtxBias]);
            a.mov(a.qword[kFrame + static_cast<int>(offsetof(Frame, ctx))], rcx);
            a.mov(r9, static_cast<uint64_t>(kFrameFlagSubCall) << 32);
            a.mov(a.qword[kFrame + static_cast<int>(offsetof(Frame, pairedMask))], r9);
            a.mov(a.dword[kFrame + static_cast<int>(offsetof(Frame, pc))], eax);
            a.mov(ecx, a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, lr))]);
            a.mov(a.dword[kFrame + static_cast<int>(offsetof(Frame, returnAddress))], ecx);
            a.mov(a.dword[kFrame + static_cast<int>(offsetof(Frame, entry))], eax);
            a.mov(a.qword[kFrame + static_cast<int>(offsetof(Frame, executed))], 0);
            a.mov(a.qword[kFrame + static_cast<int>(offsetof(Frame, pending))], 0);
            a.mov(a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, pc))], eax);
            a.xor_(kExecCount.cvt32(), kExecCount.cvt32());
            a.jmp(r8);
            a.L(fallbackEntry);
            a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
            a.jmp(a.qword[rdx + static_cast<int>(offsetof(JitCallTarget, entry))]);
        }

        // Everything from here to fdeEnd runs with the trampoline frame active (covered by shared FDE).
        const uint8_t* const fdeStart = a.getCurr();

        s.exitReturn = a.getCurr();
        if (kDirectCalls) {
            a.test(a.dword[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagSubCall);
            a.jnz(subCallReturn, Xbyak::CodeGenerator::T_SHORT);
        }
        a.test(a.dword[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagDirectEnter);
        a.jz(exitReturnSlow, Xbyak::CodeGenerator::T_SHORT);
        // Hot direct-enter return: count instructions directly into ThreadState's counter, tear down frame, return.
        a.mov(rcx, a.qword[rsp + kOutLinkSlot]);
        a.mov(rcx, a.qword[rcx]);
        a.add(a.qword[rcx], kExecCount);
        a.L(directEpilogue);
        a.add(rsp, kTrampolineFrame);
        if (kWin64) {
            a.pop(rdi);
            a.pop(rsi);
        }
        a.pop(r15);
        a.pop(r14);
        a.pop(r13);
        a.pop(r12);
        a.pop(rbx);
        a.pop(rbp);
        a.ret();

        if (kDirectCalls) {
            a.L(subCallReturn);
            a.mov(r9, reinterpret_cast<uint64_t>(&GetPatchedHooks()));
            a.mov(rcx, a.qword[r9 + static_cast<int>(offsetof(PatchedHooks, pendingFast))]);
            a.cmp(a.qword[rcx], 0);
            a.jne(subCallSync, Xbyak::CodeGenerator::T_SHORT);
            a.mov(rcx, a.qword[r9 + static_cast<int>(offsetof(PatchedHooks, anyDirty))]);
            a.cmp(a.byte[rcx], 0);
            a.jne(subCallSync, Xbyak::CodeGenerator::T_SHORT);
            a.L(subCallDone);
            // Merge callee's instruction count (kExecCount) with caller's saved count at [rsp + kTrampolineFrame].
            a.add(kExecCount, a.qword[rsp + kTrampolineFrame]);
            a.add(rsp, kTrampolineFrame + 8);
            a.pop(r14);
            a.pop(r13);
            a.pop(r12);
            a.pop(rbx);
            a.pop(rbp);
            a.ret();
            a.L(subCallSync);
            a.call(a.qword[r9 + static_cast<int>(offsetof(PatchedHooks, sync))]);
            a.jmp(subCallDone, Xbyak::CodeGenerator::T_SHORT);
        }

        a.L(exitReturnSlow);
        a.mov(rax, static_cast<uint64_t>(JitStatus::Return) << 32);
        a.jmp(epilogue);

        s.exitDynamic = a.getCurr();
        a.mov(eax, eax);
        a.xor_(edx, edx);
        a.test(a.dword[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagDirectEnter | kFrameFlagSubCall);
        a.jnz(directContinue);
        a.jmp(epilogue);

        s.exitSlot = a.getCurr();
        a.mov(eax, eax);
        a.test(a.dword[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagDirectEnter | kFrameFlagSubCall);
        a.jnz(directContinue);
        a.L(exitSlotSlow);
        a.mov(rcx, a.qword[rsp + kOutLinkSlot]);
        a.mov(a.qword[rcx], rdx);
        a.jmp(epilogue);

        s.exitPending = a.getCurr();
        a.mov(rax, static_cast<uint64_t>(JitStatus::Pending) << 32);
        a.xor_(edx, edx);
        a.test(a.dword[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagDirectEnter | kFrameFlagSubCall);
        a.jnz(directContinue);

        a.L(epilogue);
        a.add(a.qword[kFrame + static_cast<int>(offsetof(Frame, executed))], kExecCount);  // hand the count to the dispatcher
        a.add(rsp, kTrampolineFrame);
        if (kWin64) {
            a.pop(rdi);
            a.pop(rsi);
        }
        a.pop(r15);
        a.pop(r14);
        a.pop(r13);
        a.pop(r12);
        a.pop(rbx);
        a.pop(rbp);
        a.ret();

        // Direct-enter slow continuation: rax = status << 32 | next, rdx = link (or null).
        // Calls JitDirectContinue(DirectEnterState*, Frame*, uint64_t statusAndNext, JitLink* link).
        a.L(directContinue);
        a.add(a.qword[kFrame + static_cast<int>(offsetof(Frame, executed))], kExecCount);
        if (kWin64) {
            a.mov(r9, rdx);
            a.mov(r8, rax);
            a.mov(rdx, kFrame);
            a.mov(rcx, a.qword[rsp + kOutLinkSlot]);
        } else {
            a.mov(rcx, rdx);
            a.mov(rdx, rax);
            a.mov(rsi, kFrame);
            a.mov(rdi, a.qword[rsp + kOutLinkSlot]);
        }
        a.mov(rax, reinterpret_cast<uint64_t>(&JitDirectContinue));
        a.call(rax);
        a.jmp(directEpilogue);

        const uint8_t* const fdeEnd = a.getCurr();

        // The stale paths' counting code (StaleCell): shared, so a block holds only its cell and a jump through it.
        s.staleCount = a.getCurr();
        if (Opts().staleThreshold != 0) {
            Xbyak::Label toPlain;
            a.add(a.dword[rcx + static_cast<int>(offsetof(StaleCell, failures))], 1);
            a.mov(eax, a.dword[rcx + static_cast<int>(offsetof(StaleCell, failures))]);
            a.cmp(eax, Opts().staleThreshold);
            a.jb(toPlain);
            a.mov(ArgReg(0), rcx);
            a.mov(rax, reinterpret_cast<uint64_t>(&InstallStaleStub));
            a.call(rax);  // rsp is as aligned as in a block: this is reached by a jump from one
            a.jmp(rax);   // the stub, or the plain exit when none could be made
            a.L(toPlain);
            a.jmp(a.qword[rcx + static_cast<int>(offsetof(StaleCell, plain))]);
        }

        a.ready();
        s.enter = reinterpret_cast<EnterFn>(const_cast<uint8_t*>(enterAt));
        s.enterDirect = reinterpret_cast<EnterDirectFn>(const_cast<uint8_t*>(enterDirectAt));
#if defined(__linux__) && defined(__x86_64__)
        s.ehFrame = std::make_unique<EhFrameRecord>();
        s.ehFrame->Register(fdeStart, static_cast<size_t>(fdeEnd - fdeStart));
#else
        (void)fdeStart;
        (void)fdeEnd;
#endif
        PerfMapAdd(enterAt, static_cast<size_t>(static_cast<const uint8_t*>(s.exitReturn) - enterAt), "jit_enter", 0);
        PerfMapAdd(s.exitReturn, static_cast<size_t>(a.getCurr() - static_cast<const uint8_t*>(s.exitReturn)), "jit_exits", 0);
    } catch (...) {
        JitExecMemory::Free(s.region);
        s.region = {};
        return false;
    }
    g_enter = s.enter;
    s.ok = true;
    return true;
}

}  // namespace

const Xbyak::Reg64& ArgReg(unsigned i) {
    static const Xbyak::Reg64* const sysv[] = {&rdi, &rsi, &rdx, &rcx, &r8, &r9};
    static const Xbyak::Reg64* const win[] = {&rcx, &rdx, &r8, &r9};
    if (kWin64) return *win[i < 4 ? i : 3];
    return *sysv[i < 6 ? i : 5];
}

Group GroupOf(K k) {
    if (k >= K::Li && k <= K::Mtspr) return Group::Int;
    if (k >= K::Lwz && k <= K::PsqStu) return Group::Mem;
    if (k >= K::Fadds && k <= K::PsAbs) return Group::Fp;
    if (k >= K::Bc && k <= K::Bctrl && k != K::Jmp) return Group::Branch;
    return Group::None;
}

namespace {
// How many ops a fallback call for kind k runs: a fused compare runs its bc too.
uint32_t FallbackSpan(K k) {
    switch (k) {
    case K::CmpwiBc: case K::CmplwiBc: case K::CmpwBc: case K::CmplwBc: case K::RlwinmRcBc: return 2;
    default: return 1;
    }
}
}  // namespace

// --- BlockEmitter --------------------------------------------------------------------------------------------------------

BlockEmitter::BlockEmitter(const Block& b, void* buffer, size_t size) : a(size, buffer), block(b) {
    a.setDefaultJmpNEAR(true);  // a forward jump to a label not yet defined is a 32-bit displacement unless T_SHORT is asked
}
BlockEmitter::~BlockEmitter() = default;

// The context fields kCtxBias puts within a one-byte displacement (jit_x64.h): the GPRs, the CR field values, lr, ctr, xer, fpscr, pc
// and the first four FPRs (and their words).
constexpr bool OneByteDisp(size_t offsetInContext, size_t size) {
    return offsetInContext >= static_cast<size_t>(kCtxBias) - 128u && offsetInContext + size - 1 <= static_cast<size_t>(kCtxBias) + 127u;
}
static_assert(OneByteDisp(offsetof(CpuContext, gpr), sizeof(CpuContext::gpr)));
static_assert(OneByteDisp(offsetof(CpuContext, crv), sizeof(CpuContext::crv)));
static_assert(OneByteDisp(offsetof(CpuContext, lr), 4) && OneByteDisp(offsetof(CpuContext, ctr), 4) &&
              OneByteDisp(offsetof(CpuContext, xer), 4) && OneByteDisp(offsetof(CpuContext, fpscr), 4) &&
              OneByteDisp(offsetof(CpuContext, pc), 4));
static_assert(OneByteDisp(offsetof(CpuContext, fpr), 4 * sizeof(PPC_FPR)));

Xbyak::Address BlockEmitter::Gpr(uint32_t r) const { return a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, gpr) + 4u * r)]; }
Xbyak::Address BlockEmitter::Crv(uint32_t field) const { return a.qword[kCtxBase + CtxDisp(offsetof(CpuContext, crv) + 8u * field)]; }
Xbyak::Address BlockEmitter::Lr() const { return a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, lr))]; }
Xbyak::Address BlockEmitter::Ctr() const { return a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, ctr))]; }
Xbyak::Address BlockEmitter::Xer() const { return a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, xer))]; }
Xbyak::Address BlockEmitter::Pc() const { return a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, pc))]; }
Xbyak::Address BlockEmitter::Gqr(uint32_t i) const { return a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, gqr) + 4u * i)]; }
Xbyak::Address BlockEmitter::Fpr(uint32_t r) const { return a.qword[kCtxBase + CtxDisp(offsetof(CpuContext, fpr) + 8u * r)]; }
Xbyak::Address BlockEmitter::FprWord(uint32_t r, uint32_t half) const {
    return a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, fpr) + 8u * r + 4u * half)];
}
Xbyak::Address BlockEmitter::PairedMask() const { return a.dword[kFrame + static_cast<int>(offsetof(Frame, pairedMask))]; }
Xbyak::Address BlockEmitter::FrameFlags() const { return a.dword[kFrame + static_cast<int>(offsetof(Frame, flags))]; }
Xbyak::Address BlockEmitter::Executed() const { return a.qword[kFrame + static_cast<int>(offsetof(Frame, executed))]; }
Xbyak::Address BlockEmitter::CtxField(size_t offset) const { return a.ptr[kCtxBase + CtxDisp(offset)]; }

// --- intra-block GPR cache ----------------------------------------------------------------------------------------------

const Xbyak::Reg32& BlockEmitter::GprCacheReg(uint32_t slot) {
    static const Xbyak::Reg32* const regs[kGprCacheSlots] = {&ebp, &r8d, &r9d, &r10d, &r11d};
    return *regs[slot];
}

const Xbyak::Reg64& BlockEmitter::GprCacheReg64(uint32_t slot) {
    static const Xbyak::Reg64* const regs[kGprCacheSlots] = {&rbp, &r8, &r9, &r10, &r11};
    return *regs[slot];
}

int BlockEmitter::FindCachedGpr(uint32_t r) const {
    if (inCold_) return -1;
    const int8_t g = static_cast<int8_t>(r);
    for (uint32_t s = 0; s < kGprCacheSlots; ++s) {
        if (gprSlots_[s].guest == g) return static_cast<int>(s);
    }
    return -1;
}

uint32_t BlockEmitter::AllocGprSlot(uint32_t pinMask) {
    // Prefer an empty slot, then the least-recently-used clean slot, then the least-recently-used dirty slot.
    int bestClean = -1;
    int bestDirty = -1;
    for (uint32_t s = 0; s < kGprCacheSlots; ++s) {
        const GprSlot& sl = gprSlots_[s];
        if (sl.guest < 0) return s;
        if ((pinMask & (1u << sl.guest)) != 0) continue;
        if (!sl.dirty) {
            if (bestClean < 0 || sl.lastUse < gprSlots_[bestClean].lastUse) bestClean = static_cast<int>(s);
        } else {
            if (bestDirty < 0 || sl.lastUse < gprSlots_[bestDirty].lastUse) bestDirty = static_cast<int>(s);
        }
    }
    const uint32_t chosen = static_cast<uint32_t>(bestClean >= 0 ? bestClean : bestDirty);
    if (gprSlots_[chosen].dirty) {
        a.mov(Gpr(static_cast<uint32_t>(gprSlots_[chosen].guest)), GprCacheReg(chosen));
    }
    gprSlots_[chosen] = GprSlot{};
    return chosen;
}

const Xbyak::Reg32& BlockEmitter::ReadGpr(uint32_t r, uint32_t pinMask) {
    const int hit = FindCachedGpr(r);
    if (hit >= 0) {
        gprSlots_[hit].lastUse = ++gprTick_;
        return GprCacheReg(static_cast<uint32_t>(hit));
    }
    const uint32_t s = AllocGprSlot(pinMask);
    a.mov(GprCacheReg(s), Gpr(r));
    gprSlots_[s] = GprSlot{static_cast<int8_t>(r), false, ++gprTick_};
    return GprCacheReg(s);
}

const Xbyak::Reg32& BlockEmitter::WriteGpr(uint32_t r, uint32_t pinMask) {
    const int hit = FindCachedGpr(r);
    if (hit >= 0) {
        gprSlots_[hit].dirty = true;
        gprSlots_[hit].lastUse = ++gprTick_;
        return GprCacheReg(static_cast<uint32_t>(hit));
    }
    const uint32_t s = AllocGprSlot(pinMask);
    gprSlots_[s] = GprSlot{static_cast<int8_t>(r), true, ++gprTick_};
    return GprCacheReg(s);
}

const Xbyak::Reg32& BlockEmitter::ModifyGpr(uint32_t r, uint32_t pinMask) {
    const Xbyak::Reg32& reg = ReadGpr(r, pinMask);
    const int hit = FindCachedGpr(r);
    gprSlots_[hit].dirty = true;
    return reg;
}

void BlockEmitter::DropGpr(uint32_t r) {
    const int hit = FindCachedGpr(r);
    if (hit >= 0) gprSlots_[hit] = GprSlot{};
}

void BlockEmitter::FlushGpr(uint32_t r) {
    const int hit = FindCachedGpr(r);
    if (hit >= 0 && gprSlots_[hit].dirty) {
        a.mov(Gpr(r), GprCacheReg(static_cast<uint32_t>(hit)));
        gprSlots_[hit].dirty = false;
    }
}

void BlockEmitter::FlushDirtyGprs() {
    for (uint32_t s = 0; s < kGprCacheSlots; ++s) {
        if (gprSlots_[s].guest >= 0 && gprSlots_[s].dirty) {
            a.mov(Gpr(static_cast<uint32_t>(gprSlots_[s].guest)), GprCacheReg(s));
            gprSlots_[s].dirty = false;
        }
    }
}

void BlockEmitter::FlushVolatileGprs() {
    for (uint32_t s = 1; s < kGprCacheSlots; ++s) {
        if (gprSlots_[s].guest >= 0) {
            if (gprSlots_[s].dirty) a.mov(Gpr(static_cast<uint32_t>(gprSlots_[s].guest)), GprCacheReg(s));
            gprSlots_[s] = GprSlot{};
        }
    }
}

void BlockEmitter::FlushAllGprs() {
    for (uint32_t s = 0; s < kGprCacheSlots; ++s) {
        if (gprSlots_[s].guest >= 0) {
            if (gprSlots_[s].dirty) a.mov(Gpr(static_cast<uint32_t>(gprSlots_[s].guest)), GprCacheReg(s));
            gprSlots_[s] = GprSlot{};
        }
    }
}

BlockEmitter::GprSnapshot BlockEmitter::SnapshotGprs() const {
    GprSnapshot snap{};
    for (uint32_t s = 0; s < kGprCacheSlots; ++s) {
        snap.guest[s] = gprSlots_[s].guest;
        if (gprSlots_[s].guest >= 0 && gprSlots_[s].dirty) snap.dirtyMask |= static_cast<uint8_t>(1u << s);
    }
    return snap;
}

void BlockEmitter::EmitFlushSnapshot(const GprSnapshot& snap) {
    for (uint32_t s = 0; s < kGprCacheSlots; ++s) {
        if ((snap.dirtyMask & (1u << s)) != 0) {
            a.mov(Gpr(static_cast<uint32_t>(snap.guest[s])), GprCacheReg(s));
        }
    }
}

void BlockEmitter::EmitReloadSnapshotVolatile(const GprSnapshot& snap) {
    for (uint32_t s = 1; s < kGprCacheSlots; ++s) {
        if (snap.guest[s] >= 0) {
            a.mov(GprCacheReg(s), Gpr(static_cast<uint32_t>(snap.guest[s])));
        }
    }
}

Xbyak::Label& BlockEmitter::NewLabel() {
    if (labelCount_ == kMaxLabels) throw EmitError{"labels"};
    return labels_[labelCount_++];
}

BlockEmitter::ColdItem& BlockEmitter::NewCold() {
    if (coldCount_ == kMaxCold) throw EmitError{"cold paths"};
    return cold_[coldCount_++];
}

void BlockEmitter::StorePc(uint32_t i) { a.mov(Pc(), PcOf(i)); }

Xbyak::Label& BlockEmitter::NewCell() {
    if (cellCount_ == kMaxCells) throw EmitError{"cells"};
    return cells_[cellCount_++];
}

static_assert(sizeof(JitTargetCell) == 16);

void BlockEmitter::CellProbe(Xbyak::Label& cell, const Xbyak::Label& miss) {
    const int key = static_cast<int>(offsetof(JitTargetCell, target)), record = static_cast<int>(offsetof(JitTargetCell, record));
    Xbyak::Label hit;
    a.lea(rcx, a.ptr[a.rip + cell]);
    for (uint32_t i = 0; i < kCellWays; ++i) {
        const int at = static_cast<int>(i * sizeof(JitTargetCell));
        a.cmp(eax, a.dword[rcx + at + key]);
        a.mov(rdx, a.qword[rcx + at + record]);  // mov leaves the flags
        if (i + 1 < kCellWays) a.je(hit, Xbyak::CodeGenerator::T_SHORT);
        else a.jne(miss, Xbyak::CodeGenerator::T_NEAR);
    }
    a.L(hit);
}

// A jump to a routine of the shared code. The arena is usually mapped next to it (a rel32 jump, 5 bytes); where it is not,
// the target goes through rcx (12 bytes).
void BlockEmitter::JmpAbs(const void* target) {
    // BRAWL_JIT_FAR_JUMPS=1 (a diagnostic for the tests): always the long form, as for an arena that is far from the code
    static const bool alwaysLong = EnvOn("BRAWL_JIT_FAR_JUMPS");  // not `far`: a macro of the Windows headers
    const int64_t distance = reinterpret_cast<const uint8_t*>(target) - (a.getCurr() + 5);
    if (!alwaysLong && distance >= INT32_MIN && distance <= INT32_MAX) {
        a.jmp(target, Xbyak::CodeGenerator::T_NEAR);
        return;
    }
    a.mov(rcx, reinterpret_cast<uint64_t>(target));
    a.jmp(rcx);
}

void BlockEmitter::ExitDynamic() { JmpAbs(g_shared.exitDynamic); }
void BlockEmitter::ExitReturn() { JmpAbs(g_shared.exitReturn); }
void BlockEmitter::ExitPending() { JmpAbs(g_shared.exitPending); }
void BlockEmitter::ExitTo(uint32_t targetPc) {
    a.mov(eax, targetPc);
    ExitDynamic();
}

void BlockEmitter::ExitLinked(uint32_t targetPc) {
    if (IsHookPc(targetPc)) {  // a hook pc is handled in the dispatcher's loop: never linked
        ExitTo(targetPc);
        return;
    }
    static const bool alwaysLong = EnvOn("BRAWL_JIT_FAR_JUMPS");
    if (alwaysLong) {
        uint32_t i = 0;
        while (i < exitCount_ && exits_[i].pc != targetPc) ++i;
        if (i == exitCount_) {
            if (exitCount_ == kMaxExits) throw EmitError{"link slots"};
            exits_[exitCount_].pc = targetPc;
            exits_[exitCount_++].patchOffset = Exit::kNoPatch;
        }
        a.jmp(a.ptr[a.rip + exits_[i].slot]);
        return;
    }
    if (exitCount_ == kMaxExits) throw EmitError{"link slots"};
    const uint32_t i = exitCount_++;
    exits_[i].pc = targetPc;
    exits_[i].patchOffset = static_cast<uint32_t>(a.getSize());
    a.jmp(exits_[i].stub, Xbyak::CodeGenerator::T_NEAR);
}

// A stats-only increment of one of the process counters (BRAWL_JIT_STATS): clobbers r8 and the flags.
static void EmitCount(Xbyak::CodeGenerator& a, std::atomic<uint64_t>& counter) {
    a.mov(r8, reinterpret_cast<uint64_t>(&counter));
    a.inc(a.qword[r8]);
}

void BlockEmitter::CacheProbe(uint32_t which) {
    DynCache& c = caches_[which];
    c.probed = true;
    const uint32_t entries = kCacheEntries[which];
    static_assert((kCacheEntries[0] & (kCacheEntries[0] - 1)) == 0 && (kCacheEntries[1] & (kCacheEntries[1] - 1)) == 0);
    // rcx = byte offset of the entry (pc >> 2) & (entries - 1) selects: sizeof(JitLink) is 64 = 16 * 4
    static_assert(sizeof(JitLink) == 64);
    Xbyak::Label next;
    a.mov(ecx, eax);
    a.and_(ecx, (entries - 1) << 2);
    a.shl(ecx, 4);
    a.lea(rdx, a.ptr[a.rip + c.slots]);
    a.cmp(eax, a.dword[rdx + rcx + static_cast<int>(offsetof(JitLink, key))]);
    a.jne(next, Xbyak::CodeGenerator::T_SHORT);
    if (Opts().stats) EmitCount(a, g_counters.cacheHits);
    a.jmp(a.qword[rdx + rcx]);
    a.L(next);
}

void BlockEmitter::ExitDynamicMiss(uint32_t which) {
    DynCache& c = caches_[which];
    c.probed = true;
    if (c.missEmitted) {
        a.jmp(c.miss);
        return;
    }
    c.missEmitted = true;
    a.L(c.miss);
    if (Opts().stats) EmitCount(a, g_counters.cacheMisses);
    a.lea(rdx, a.ptr[a.rip + c.slots]);
    JmpAbs(g_shared.exitSlot);
}

void BlockEmitter::ExitDynamicCached(uint32_t which) {
    CacheProbe(which);
    ExitDynamicMiss(which);
}

void BlockEmitter::CallRaw(const void* fn) {
    static const bool alwaysLong = EnvOn("BRAWL_JIT_FAR_JUMPS");
    const int64_t distance = reinterpret_cast<const uint8_t*>(fn) - (a.getCurr() + 5);
    if (!alwaysLong && distance >= INT32_MIN && distance <= INT32_MAX) {
        a.call(fn);
        return;
    }
    a.mov(rax, reinterpret_cast<uint64_t>(fn));
    a.call(rax);
}

void BlockEmitter::CallSubCall() {
    static const bool alwaysLong = EnvOn("BRAWL_JIT_FAR_JUMPS");
    const void* const fn = g_shared.subCall;
    const int64_t distance = reinterpret_cast<const uint8_t*>(fn) - (a.getCurr() + 5);
    if (!alwaysLong && distance >= INT32_MIN && distance <= INT32_MAX) {
        a.call(fn);
        return;
    }
    // eax holds target and rdx holds JitCallTarget*: use r10 for far call.
    a.mov(r10, reinterpret_cast<uint64_t>(fn));
    a.call(r10);
}

void BlockEmitter::CallHelperImpl(const void* fn, std::initializer_list<CallArg> list) {
    constexpr unsigned kMax = kMaxCallArgs;
    CallArg arg[kMax];
    unsigned n = 0;
    for (const CallArg& c : list) arg[n++] = c;
    // the host registers each source reads, as a bit mask over register numbers
    uint32_t reads[kMax] = {};
    bool done[kMax] = {};
    for (unsigned i = 0; i < n; ++i) {
        if (arg[i].kind == CallArg::Kind::Reg) {
            if (!arg[i].reg.isREG()) throw EmitError{"call argument register"};
            reads[i] = 1u << arg[i].reg.getIdx();
        } else if (arg[i].kind == CallArg::Kind::Mem) {
            const Xbyak::RegExp e = arg[i].mem->getRegExp();
            if (e.getBase().getBit() != 0) reads[i] |= 1u << e.getBase().getIdx();
            if (e.getIndex().getBit() != 0) reads[i] |= 1u << e.getIndex().getIdx();
        }
    }
    unsigned remaining = n;
    while (remaining != 0) {
        bool progress = false;
        for (unsigned i = 0; i < n; ++i) {
            if (done[i]) continue;
            const Xbyak::Reg64& dest = ArgReg(i);
            const uint32_t destBit = 1u << dest.getIdx();
            bool blocked = false;
            for (unsigned j = 0; j < n; ++j)
                if (j != i && !done[j] && (reads[j] & destBit) != 0) blocked = true;
            if (blocked) continue;
            const CallArg& c = arg[i];
            switch (c.kind) {
            case CallArg::Kind::Imm:
                a.mov(dest, c.imm);
                break;
            case CallArg::Kind::Reg: {
                const int bits = c.reg.getBit();
                if (bits == 64) a.mov(dest, c.reg.cvt64());
                else if (bits == 32) a.mov(dest.cvt32(), c.reg.cvt32());
                else a.movzx(dest.cvt32(), c.reg);
                break;
            }
            case CallArg::Kind::Mem: {
                const int bits = c.mem->getBit();
                if (bits == 64) a.mov(dest, *c.mem);
                else if (bits == 32) a.mov(dest.cvt32(), *c.mem);
                else a.movzx(dest.cvt32(), *c.mem);
                break;
            }
            }
            done[i] = true;
            --remaining;
            progress = true;
        }
        if (progress) continue;
        // Every pending argument's register is read by another: a cycle of register moves. Park one source in r11.
        unsigned i = 0;
        while (i < n && (done[i] || arg[i].kind != CallArg::Kind::Reg)) ++i;
        if (i == n) throw EmitError{"call argument memory operand uses an argument register"};
        a.mov(r11, arg[i].reg.cvt64());
        arg[i].reg = arg[i].reg.getBit() == 64 ? Xbyak::Reg(r11) : Xbyak::Reg(r11.cvt32());
        reads[i] = 1u << r11.getIdx();
    }
    CallRaw(fn);
}

uint32_t BlockEmitter::EmitFallback(uint32_t i) {
    ++fallbackCalls_;
    return EmitSlowPath(i);
}

uint32_t BlockEmitter::EmitSlowPath(uint32_t i) {
    CallHelper(&JitFallbackOp, kFrame, &block, i);
    // rax = what << 32 | next: go on (what 0), or leave the block with what in ecx and next in eax
    a.mov(rcx, rax);
    a.shr(rcx, 32);
    a.jnz(fallbackExit_);
    fallbackUsed_ = true;
    return FallbackSpan(block.ops[i].kind);
}

// One of the block's mark tests: jumps to `fail` when a mark BlockCache::Current would test is set. The marks are bytes and
// the mask keeps whole bytes of the 8-byte window at m.at, so the test reads only the bytes it needs: a byte, a word, a
// dword (compared with zero, or tested under a mask when it has bytes the block does not use), else the whole window.
void BlockEmitter::EmitMarkTest(const MarkCheck& m, const Xbyak::Label& fail, bool shortJump) {
    const auto T = shortJump ? Xbyak::CodeGenerator::T_SHORT : Xbyak::CodeGenerator::T_NEAR;
    uint8_t bytes[8];
    std::memcpy(bytes, &m.mask, 8);
    int lo = -1, hi = -1;
    bool whole = true;  // every byte of the mask is 0 or 0xFF
    for (int i = 0; i < 8; ++i) {
        if (bytes[i] != 0 && bytes[i] != 0xFF) whole = false;
        if (bytes[i] == 0) continue;
        if (lo < 0) lo = i;
        hi = i;
    }
    const int disp = static_cast<int>(MemoryInline::kLeanDirtyOffset + m.at);
    if (whole && hi - lo < 4) {
        const int start = lo < 4 ? lo : 4;  // a dword read from here ends inside the 8-byte window
        uint32_t keep = 0;
        for (int i = 0; i < 4; ++i)
            if (bytes[start + i] != 0) keep |= 0xFFu << (8 * i);
        if (keep == 0xFFu) a.cmp(a.byte[kLean + disp + start], 0);
        else if (keep == 0xFFFFu) a.cmp(a.word[kLean + disp + start], 0);
        else if (keep == 0xFFFFFFFFu) a.cmp(a.dword[kLean + disp + start], 0);
        else a.test(a.dword[kLean + disp + start], keep);
        a.jnz(fail, T);
        return;
    }
    a.mov(rax, m.mask);
    a.test(a.qword[kLean + disp], rax);
    a.jnz(fail, T);
}

void BlockEmitter::EmitValidityTest(const Xbyak::Label& fail, bool shortJump) {
    const auto T = shortJump ? Xbyak::CodeGenerator::T_SHORT : Xbyak::CodeGenerator::T_NEAR;
    // epoch == markEpoch (b.epoch is read from memory: Refresh rewrites it) ...
    a.mov(rcx, reinterpret_cast<uint64_t>(&block.epoch));
    a.mov(eax, a.dword[rcx]);
    a.cmp(eax, a.dword[kEpochCell]);
    a.jne(fail, T);
    // ... and every mark the block's check[] names is clear (BlockCache::Current always tests the first two; an unused one
    // has no mask)
    const uint32_t checks = block.checks > 2 ? block.checks : 2;
    for (uint32_t k = 0; k < checks; ++k)
        if (block.check[k].mask != 0) EmitMarkTest(block.check[k], fail, shortJump);
}

void BlockEmitter::EmitRecheck(uint32_t nextPc) {
    Xbyak::Label& fail = NewLabel();
    EmitValidityTest(fail, false);
    Cold([&fail, nextPc](BlockEmitter& e) {
        e.a.L(fail);
        e.ExitTo(nextPc);
    });
}

// Backward liveness of GPR reads within each straight-line / Bc-only segment of the block (resets at calls,
// unconditional exits, Generic ops, and Lmw). Purely an allocation hint: only GPRs read again later in the segment are
// placed into the 5-slot cache, avoiding evictions and flushes for single-use registers in short blocks.
void BlockEmitter::AnalyseGprUses() {
    uint32_t live = 0;
    gprReadLater_[block.count] = 0;
    for (uint32_t i = block.count; i-- > 0;) {
        const Op& o = block.ops[i];
        if (o.kind == K::B || o.kind == K::Bl || o.kind == K::Blr || o.kind == K::Bclr || o.kind == K::Bctr ||
            o.kind == K::Bctrl || o.kind == K::Generic || o.kind == K::End) {
            live = 0;
            gprReadLater_[i] = 0;
            continue;
        }
        gprReadLater_[i] = live;
        switch (o.kind) {
        case K::Li: case K::Lis: case K::Mflr: case K::Mfctr: case K::Mfcr: case K::Mfspr:
            live &= ~(1u << o.d);
            break;
        case K::Addi: case K::Addis: case K::Mulli: case K::Subfic: case K::Neg:
        case K::Lwz: case K::Lbz: case K::Lhz: case K::Lha:
            live = (live & ~(1u << o.d)) | (1u << o.a);
            break;
        case K::Lwzu: case K::Lbzu: case K::Lhzu:
            live = (live & ~(1u << o.d)) | (1u << o.a);
            break;
        case K::Add: case K::Subf: case K::Mullw:
        case K::Lwzx: case K::Lbzx: case K::Lhzx:
            live = (live & ~(1u << o.d)) | (1u << o.a) | (1u << o.b);
            break;
        case K::Ori: case K::Oris: case K::Xori: case K::Xoris: case K::AndiRc: case K::AndisRc:
        case K::Rlwinm: case K::RlwinmRc: case K::RlwinmRcBc: case K::Srawi: case K::Extsb: case K::Extsh:
        case K::Cntlzw:
            live = (live & ~(1u << o.a)) | (1u << o.d);
            break;
        case K::Rlwimi:
            live |= (1u << o.d) | (1u << o.a);
            break;
        case K::Or: case K::OrRc: case K::And: case K::AndRc: case K::Xor: case K::Andc: case K::Nor:
        case K::Slw: case K::Srw: case K::Rlwnm: case K::RlwnmRc:
            live = (live & ~(1u << o.a)) | (1u << o.d) | (1u << o.b);
            break;
        case K::Cmpwi: case K::Cmplwi: case K::CmpwiBc: case K::CmplwiBc:
            live |= (1u << o.a);
            break;
        case K::Cmpw: case K::Cmplw: case K::CmpwBc: case K::CmplwBc:
            live |= (1u << o.a) | (1u << o.b);
            break;
        case K::Mtlr: case K::Mtctr: case K::Mtspr:
            live |= (1u << o.d);
            break;
        case K::Stw: case K::Stwu: case K::Stb: case K::Stbu: case K::Sth: case K::Sthu:
            live |= (1u << o.a) | (1u << o.d);
            break;
        case K::Stwx: case K::Stbx: case K::Sthx:
            live |= (1u << o.a) | (1u << o.b) | (1u << o.d);
            break;
        case K::Lfs: case K::Lfsu: case K::Lfd: case K::Lfdu:
        case K::Stfs: case K::Stfsu: case K::Stfd: case K::Stfdu:
        case K::PsqL: case K::PsqLu: case K::PsqSt: case K::PsqStu: case K::Stmw:
            live |= (1u << o.a);
            break;
        case K::Lfsx: case K::Stfsx:
            live |= (1u << o.a) | (1u << o.b);
            break;
        case K::Lmw:
            live = (1u << o.a);  // Lmw flushes all GPRs after computing EA
            break;
        default:
            break;
        }
    }
}

void BlockEmitter::EmitPrologue() {
    // The stale exit comes first, so the check's jumps to it are short; nothing falls into it (blocks are entered at their
    // `checked` or `entry` address).
    AnalyseCr();
    AnalyseForms();
    AnalyseGprUses();
    staleStub_ = Opts().staleThreshold != 0 && owner_ != nullptr;
    a.L(stale_);
    if (staleStub_) {  // through the block's StaleCell (rcx its address): the shared counting code, later a stub
        a.lea(rcx, a.ptr[a.rip + staleCell_]);
        a.jmp(a.ptr[rcx]);
    }
    a.L(plainStale_);
    ExitTo(block.start);
    checkedAt_ = a.getCurr();
    EmitValidityTest(stale_, true);
    entryAt_ = a.getCurr();
    // The block's instruction count goes to a register the trampoline's epilogue adds to frame.executed (one add, not a
    // read-modify-write of memory per block); BRAWL_JIT_EXEC_MEM=1 (a test switch) adds it to frame.executed in memory per block.
    if (Opts().execInMemory) a.add(Executed(), block.count);
    else a.add(kExecCount, block.count);
    // ctx->pc is only a diagnostic (fault and trace reports) between the dispatcher's store and the next store of a call or a
    // slow path (StorePc); BRAWL_JIT_BLOCK_PC=1 (a test switch) stores the block's start here.
    if (Opts().blockPc) a.mov(Pc(), block.start);
}

void BlockEmitter::Finish() {
    if (fallbackUsed_) {  // ecx: what (1 leave for eax, 2 return, 3 pending)
        Xbyak::Label returned, pending;
        a.L(fallbackExit_);
        a.cmp(ecx, static_cast<uint32_t>(JitStatus::Pending) + 1);
        a.je(pending);
        a.cmp(ecx, static_cast<uint32_t>(JitStatus::Return) + 1);
        a.je(returned);
        ExitDynamic();
        a.L(returned);
        ExitReturn();
        a.L(pending);
        ExitPending();
    }
    inCold_ = true;
    for (uint32_t i = 0; i < coldCount_; ++i) cold_[i].run(cold_[i].storage, *this);
    inCold_ = false;
    for (DynCache& c : caches_) {  // a probe whose miss path was never emitted (its caller decided against the dispatcher)
        if (!c.probed || c.missEmitted) continue;
        c.missEmitted = true;
        a.L(c.miss);
        a.lea(rdx, a.ptr[a.rip + c.slots]);
        JmpAbs(g_shared.exitSlot);
    }
    for (uint32_t i = 0; i < exitCount_; ++i) {  // each slot's unlinked stub (and far fallback if >2 GiB)
        a.L(exits_[i].stub);
        a.mov(eax, exits_[i].pc);
        a.lea(rdx, a.ptr[a.rip + exits_[i].slot]);
        JmpAbs(g_shared.exitSlot);
        if (exits_[i].patchOffset != Exit::kNoPatch) {
            a.L(exits_[i].far);
            a.jmp(a.ptr[a.rip + exits_[i].slot]);
        }
    }
    codeBytes_ = a.getSize();
    a.align(8);
    for (uint32_t i = 0; i < exitCount_; ++i) {  // the slots (data), filled in below
        a.L(exits_[i].slot);
        for (unsigned w = 0; w < sizeof(JitLink) / 8; ++w) a.dq(0);
    }
    if (staleStub_) {  // the stale path's cell (data): filled in below
        a.L(staleCell_);
        for (unsigned w = 0; w < sizeof(StaleCell) / 8; ++w) a.dq(0);
    }
    for (DynCache& c : caches_) {  // the inline caches' entries
        if (!c.probed) continue;
        a.L(c.slots);
        for (unsigned w = 0; w < kCacheEntries[&c - caches_] * sizeof(JitLink) / 8; ++w) a.dq(0);
    }
    for (uint32_t i = 0; i < cellCount_; ++i) {  // the call sites' cells: no key, no record, kCellWays entries each
        a.L(cells_[i]);
        for (uint32_t w = 0; w < kCellWays; ++w) {
            a.dq(JitTargetCell::kNoKey);
            a.dq(0);
        }
    }
    a.ready();
    if (staleStub_) {
        StaleCell* const cell = reinterpret_cast<StaleCell*>(const_cast<uint8_t*>(staleCell_.getAddress()));
        cell->target = g_shared.staleCount;
        cell->failures = 0;
        cell->block = &block;
        cell->cache = owner_;
        cell->plain = plainStale_.getAddress();
        cell->stubbed = 0;
    }
    uint8_t* const codeBase = const_cast<uint8_t*>(a.getCode());
    for (uint32_t i = 0; i < exitCount_; ++i) {
        JitLink* const slot = reinterpret_cast<JitLink*>(const_cast<uint8_t*>(exits_[i].slot.getAddress()));
        slot->target = slot->unlinked = exits_[i].stub.getAddress();
        slot->next = nullptr;
        slot->prev = nullptr;
        slot->to = nullptr;
        slot->key = JitLink::kNoKey;
        slot->group = 0;
        if (exits_[i].patchOffset != Exit::kNoPatch && codeBase[exits_[i].patchOffset] == 0xE9u) {
            slot->patchSite = codeBase + exits_[i].patchOffset;
            slot->farTramp = exits_[i].far.getAddress();
        } else {
            slot->patchSite = nullptr;
            slot->farTramp = nullptr;
        }
    }
    for (DynCache& c : caches_) {
        if (!c.probed) continue;
        JitLink* const entries = reinterpret_cast<JitLink*>(const_cast<uint8_t*>(c.slots.getAddress()));
        for (uint32_t i = 0; i < kCacheEntries[&c - caches_]; ++i) {
            entries[i].target = entries[i].unlinked = c.miss.getAddress();
            entries[i].next = nullptr;
            entries[i].prev = nullptr;
            entries[i].to = nullptr;
            entries[i].key = JitLink::kNoKey;
            entries[i].group = i == 0 ? kCacheEntries[&c - caches_] : 0;
            entries[i].patchSite = nullptr;
            entries[i].farTramp = nullptr;
        }
    }
}

}  // namespace x64

// --- the backend --------------------------------------------------------------------------------------------------------

struct JitCodeCache {
    JitExecMemory::Arena arena;
    uint64_t compiled = 0;
#if defined(__linux__) && defined(__x86_64__)
    std::vector<std::unique_ptr<x64::EhFrameRecord>> ehFrames;
    void SyncUnwind() {
        const auto& chunks = arena.Chunks();
        while (ehFrames.size() < chunks.size()) {
            const JitExecMemory::Region& c = chunks[ehFrames.size()];
            auto rec = std::make_unique<x64::EhFrameRecord>();
            rec->Register(c.base, c.size);
            ehFrames.push_back(std::move(rec));
        }
    }
    ~JitCodeCache() { ehFrames.clear(); }  // deregister before arena unmaps chunks
#else
    void SyncUnwind() {}
#endif
};

namespace {

using namespace x64;

// Native code of the groups, per kind, or the fallback call.
void EmitOps(BlockEmitter& e) {
    const Options& o = Opts();
    const Block& b = e.block;
    uint32_t i = 0;
    while (i < b.count) {
        const K kind = b.ops[i].kind;
        if (kind == K::Jmp) {  // a followed b: nothing to run, the next op is at the target
            ++i;
            continue;
        }
        e.index = i;
        const uint32_t fallbacksBefore = e.FallbackCount();
        uint32_t used = 0;
        if (kind == K::Generic && GenericStatsOn()) {  // the execution count of this word's (primary opcode, xo)
            e.FlushAllGprs();
            e.a.mov(r8, reinterpret_cast<uint64_t>(GenericRunCounter(b.ops[i].imm)));
            e.a.inc(e.a.qword[r8]);
        }
        if (o.forced[static_cast<unsigned>(kind)] || kind == K::Generic) {
            e.FlushAllGprs();
        } else if (o.group[static_cast<unsigned>(kind)] == Group::Fp) {
            // FP ops never read/write GPRs, so ebp stays live across them; volatile r8-r11 are flushed in case of slow paths.
            e.FlushVolatileGprs();
        } else if (o.group[static_cast<unsigned>(kind)] == Group::Branch) {
            // Bc and fused compare+bc preserve the GPR cache on fall-through and flush dirty regs on their taken paths;
            // all other branches/calls flush before running.
            const bool keepGprs = kind == K::Bc || kind == K::CmpwiBc || kind == K::CmplwiBc || kind == K::CmpwBc ||
                                  kind == K::CmplwBc || kind == K::RlwinmRcBc;
            if (!keepGprs) e.FlushAllGprs();
        }
        if (!o.forced[static_cast<unsigned>(kind)]) {
            if (kind == K::Generic) used = EmitGeneric(e, i);  // the instructions with no kind of their own
            switch (o.group[static_cast<unsigned>(kind)]) {
            case Group::Int: used = EmitInt(e, i); break;
            case Group::Mem: used = EmitMem(e, i); break;
            case Group::Branch: used = EmitBranch(e, i); break;
            case Group::Fp: used = EmitFp(e, i); break;
            case Group::None: break;
            }
        }
        const bool native = used != 0 && e.FallbackCount() == fallbacksBefore;
        if (used == 0) {
            e.FlushAllGprs();
            used = e.EmitFallback(i);
        }
        if (kind == K::Generic) e.FlushAllGprs();  // EmitMemGeneric may populate cache slots during Generic
        if (used == 0 || i + used > b.count) throw EmitError{"op span"};
        Bump(g_kindOps[native ? 0 : 1][static_cast<unsigned>(kind)]);
        i += used;
    }
    // the code after the last op goes on at End's target
    e.FlushAllGprs();
    e.index = b.count;
    e.ExitLinked(b.ops[b.count].imm);
}

JitCodeCache* CreateCache() {
    JitCodeCache* const cache = new (std::nothrow) JitCodeCache;
    if (cache != nullptr) Bump(g_counters.cachesCreated);
    return cache;
}

void DestroyCache(JitCodeCache* cache) {
    if (cache == nullptr) return;
    Bump(g_counters.cachesDestroyed);
    if (Opts().stats) LogStats("cache destroyed");
    delete cache;
}

size_t CodeBytes(const JitCodeCache& cache) { return cache.arena.BytesUsed(); }

bool CompileBlock(JitCodeCache& cache, Block& b);

bool Compile(JitCodeCache& cache, Block& b) {
    if (!Opts().stats) return CompileBlock(cache, b);
    const auto start = std::chrono::steady_clock::now();
    const bool ok = CompileBlock(cache, b);
    Bump(g_counters.compileNanos,
         static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()));
    return ok;
}

bool CompileBlock(JitCodeCache& cache, Block& b) {
    if (b.count == 0 || b.count > kMaxBlock) return false;
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
    // The block's validity test reads the marks through the window base: they must be where the interpreter reads them.
    if (BrawlDirtyMemory::leanBlocks != MemoryInline::g_leanBase + MemoryInline::kLeanDirtyOffset) return false;
#endif
    size_t reserve = 2048 + static_cast<size_t>(b.count) * kMaxBytesPerOp;
    for (int attempt = 0; attempt < 2; ++attempt, reserve *= 8) {
        // A block starts on a cache line of its own: the link slots and cache entries at the end of the block before it are
        // written at run time (Link), and a store into a line that is executing makes the core flush its pipeline (~50 ns).
        void* const memory = cache.arena.Allocate(reserve, 64);
        if (memory == nullptr) break;
        bool tooBig = false;
        try {
            BlockEmitter e(b, memory, reserve);
            e.SetOwner(&cache);
            e.EmitPrologue();
            EmitOps(e);
            e.Finish();
            const size_t total = e.a.getSize();
            cache.arena.Shrink(memory, reserve, total);
            cache.SyncUnwind();
            b.jitChecked = e.Checked();
            b.jitEntry = e.Entry();
            ++cache.compiled;
            Bump(g_counters.compiled);
            Bump(g_counters.codeBytes, total);
            Bump(g_counters.compiledOps, b.count);
            PerfMapAdd(memory, e.CodeBytes(), "jit", b.start);
            DumpBlock(b, memory, e.CodeBytes());
            if (Opts().stats && (g_counters.compiled.load(std::memory_order_relaxed) & 0x3FFF) == 0) LogStats("running");
            return true;
        } catch (const Xbyak::Error& error) {
            tooBig = static_cast<int>(error) == Xbyak::ERR_CODE_IS_TOO_BIG;
            if (!tooBig) std::fprintf(stderr, "jit: block %08X not compiled: %s\n", b.start, error.what());
        } catch (const EmitError& error) {
            std::fprintf(stderr, "jit: block %08X not compiled: out of %s\n", b.start, error.what);
        } catch (...) {
        }
        cache.arena.Shrink(memory, reserve, 0);
        if (!tooBig) break;
    }
    Bump(g_counters.compileFailed);
    return false;
}

// A jump to an absolute address from code made at run time: rel32 when the address is in reach, else through rax.
void JumpTo(Xbyak::CodeGenerator& g, const void* target) {
    const int64_t distance = reinterpret_cast<const uint8_t*>(target) - (g.getCurr() + 5);
    if (distance >= INT32_MIN && distance <= INT32_MAX) {
        g.jmp(target, Xbyak::CodeGenerator::T_NEAR);
        return;
    }
    g.mov(rax, reinterpret_cast<uint64_t>(target));
    g.jmp(rax);
}

// The stale stub of a block (see StaleCell): epoch == markEpoch, then each guest word of the block compared in place with
// the word the block was decoded from (all its segments; the words as guest memory holds them), then the block's entry; any
// difference leaves through the plain stale exit. Null when it cannot be made.
const void* MakeStaleStub(JitCodeCache& cache, const StaleCell& cell) {
    const Block& b = *cell.block;
    uint32_t words = 0;
    for (uint32_t s = 0; s < b.segments; ++s) words += b.segment[s].count;
    const size_t reserve = 256 + 24u * static_cast<size_t>(words);
    void* const memory = cache.arena.Allocate(reserve, 64);
    if (memory == nullptr) return nullptr;
    // BRAWL_JIT_STALE_FAULT=<n> (a test aid: the stub is wrong on purpose so that the tests can show they notice): 1 compares no
    // words, 2 skips the epoch test, 3 compares the first segment only, 4 skips the first word of each segment, 5 never enters.
    const char* const faultText = std::getenv("BRAWL_JIT_STALE_FAULT");
    const int fault = faultText != nullptr ? std::atoi(faultText) : 0;
    try {
        Xbyak::CodeGenerator g(reserve, memory);
        Xbyak::Label leave;
        if (fault != 2) {
            g.mov(rcx, reinterpret_cast<uint64_t>(&b.epoch));
            g.mov(eax, g.dword[rcx]);
            g.cmp(eax, g.dword[kEpochCell]);
            g.jne(leave, Xbyak::CodeGenerator::T_NEAR);
        }
        const uint32_t* raw = b.words + b.count;  // BlockCache::Holds compares the memory against these
        for (uint32_t s = 0; s < b.segments; ++s)
            for (uint32_t k = 0; k < b.segment[s].count; ++k, ++raw) {
                if (fault == 1 || (fault == 3 && s != 0) || (fault == 4 && k == 0)) continue;
                const int disp = static_cast<int>((b.segment[s].start & MemoryInline::kLeanMask) + 4u * k);
                g.cmp(g.dword[kLean + disp], *raw);
                g.jne(leave, Xbyak::CodeGenerator::T_NEAR);
            }
        if (fault != 5) JumpTo(g, b.jitEntry);
        g.L(leave);
        JumpTo(g, cell.plain);
        g.ready();
        cache.arena.Shrink(memory, reserve, g.getSize());
        cache.SyncUnwind();
        return memory;
    } catch (...) {
        cache.arena.Shrink(memory, reserve, 0);
        return nullptr;
    }
}

JitExit Enter(Frame& f, Block& b) {
    if (Opts().stats) Bump(g_counters.enters);
    JitLink* link = nullptr;
    const uint64_t r = g_enter(&f, b.jitEntry, &link);
    return JitExit{static_cast<JitStatus>(r >> 32), static_cast<uint32_t>(r), link};
}

// Takes a slot off the list of the block it is aimed at.
void Unlink(JitLink& slot) {
    if (slot.prev == nullptr) return;
    *slot.prev = slot.next;
    if (slot.next != nullptr) slot.next->prev = slot.prev;
    slot.next = nullptr;
    slot.prev = nullptr;
    slot.to = nullptr;
}

// Patches a direct static exit's 5-byte `jmp rel32` in place, falling back to its indirect far trampoline if out of ±2 GiB.
void PatchDirectJump(JitLink& slot, const void* dest) {
    if (slot.patchSite == nullptr) return;
    int64_t dist = reinterpret_cast<const uint8_t*>(dest) - (slot.patchSite + 5);
    if (dist < INT32_MIN || dist > INT32_MAX) {
        dist = reinterpret_cast<const uint8_t*>(slot.farTramp) - (slot.patchSite + 5);
    }
    const int32_t rel = static_cast<int32_t>(dist);
    std::memcpy(slot.patchSite + 1, &rel, sizeof(rel));
}

// Aims one slot (or cache entry) at `to`'s checked entry and puts it on to's incoming list. A cache entry's key is the
// block's start: the dispatcher got here with `to` found for exactly that pc.
void Aim(JitLink& slot, Block& to) {
    Unlink(slot);
    slot.target = to.jitChecked;
    PatchDirectJump(slot, to.jitChecked);
    slot.to = &to;
    slot.key = to.start;
    slot.next = to.jitIncoming;
    slot.prev = &to.jitIncoming;
    if (slot.next != nullptr) slot.next->prev = &slot.next;
    to.jitIncoming = &slot;
}

// The inline cache whose first entry is `head`: `to` goes in the entry its start selects (the emitted probe computes the same
// index from the pc), replacing what was there.
void LinkCached(JitLink& head, Block& to) {
    JitLink& e = (&head)[(to.start >> 2) & (head.group - 1)];
    if (e.to == &to && e.target == to.jitChecked && e.key == to.start) return;
    Aim(e, to);
    Bump(g_counters.links);
}

void Link(JitLink& slot, Block& to) {
    if (to.jitChecked == nullptr) return;
    if (slot.group != 0) {
        LinkCached(slot, to);
        return;
    }
    if (slot.to == &to && slot.target == to.jitChecked) return;
    Aim(slot, to);
    Bump(g_counters.links);
}

void Retire(Block& b) {
    for (JitLink* slot = b.jitIncoming; slot != nullptr;) {
        JitLink* const next = slot->next;
        slot->target = slot->unlinked;
        PatchDirectJump(*slot, slot->unlinked);
        slot->key = JitLink::kNoKey;
        slot->next = nullptr;
        slot->prev = nullptr;
        slot->to = nullptr;
        slot = next;
    }
    b.jitIncoming = nullptr;
    Bump(g_counters.retires);
}

JitBackend g_backend{CreateCache, DestroyCache, Compile, Enter, Link, Retire, CodeBytes, nullptr};

}  // namespace

const void* x64::InstallStaleStub(StaleCell* cell) noexcept {
    try {
        if (cell->stubbed == 0) {
            cell->stubbed = 1;
            const void* const stub = cell->block->jitEntry != nullptr ? MakeStaleStub(*static_cast<JitCodeCache*>(cell->cache), *cell) : nullptr;
            if (stub != nullptr) {
                cell->target = stub;
                Bump(x64::g_counters.staleStubs);
            } else {
                cell->target = cell->plain;  // no stub: the stale path goes straight to the dispatcher from now on
                Bump(x64::g_counters.staleStubFailures);
            }
        }
    } catch (...) {
        cell->target = cell->plain;
    }
    return cell->target;
}

uint64_t JitX64KindOps(K kind, bool native) {
    return x64::g_kindOps[native ? 0 : 1][static_cast<unsigned>(kind)].load(std::memory_order_relaxed);
}

JitX64Stats JitX64GetStats() {
    const auto get = [](const std::atomic<uint64_t>& c) { return c.load(std::memory_order_relaxed); };
    return JitX64Stats{get(g_counters.compiled), get(g_counters.compileFailed), get(g_counters.codeBytes), get(g_counters.links),
                       get(g_counters.retires), get(g_counters.enters), get(g_counters.cachesCreated),
                       get(g_counters.cachesDestroyed), get(g_counters.compiledOps), get(g_counters.compileNanos),
                       get(g_counters.cacheHits), get(g_counters.cacheMisses), get(g_counters.staleStubs),
                       get(g_counters.staleStubFailures)};
}

// The x86-64 backend, or null when the host does not run generated code, or guest memory is not the lean window with
// dirty marks (the block validity test reads them).
const JitBackend* JitX64Backend() {
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
    static const JitBackend* const backend = []() -> const JitBackend* {
        if (!JitExecMemory::Available()) return nullptr;
        if (!BuildShared()) return nullptr;
        g_backend.enterDirect = x64::g_shared.enterDirect;
        return &g_backend;
    }();
    return backend;
#else
    return nullptr;
#endif
}

}  // namespace PpcInterpreter::detail
