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
#include <unordered_map>

#include "brawl_dirty_memory.h"
#include "isa/ppc_isa_crv.h"  // MKW_LEAN_FLAGS
#include "jit/jit_exec_memory.h"
#include "memory_access.h"
#include "recomp_mod_loader.h"
#include "../ppc_interpreter_diag.h"

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
    std::string dumpDir;  // BRAWL_JIT_DUMP_DIR=<dir>: every block's code and its op/region side table (jit-<pid>.code/.idx)
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
        if (const char* dir = std::getenv("BRAWL_JIT_DUMP_DIR")) dumpDir = dir;
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

}  // namespace

bool KindForced(K kind) { return Opts().forced[static_cast<unsigned>(kind)]; }

namespace {

struct Counters {
    std::atomic<uint64_t> compiled{0}, compileFailed{0}, codeBytes{0}, links{0}, retires{0}, enters{0}, cachesCreated{0},
        cachesDestroyed{0}, compiledOps{0}, compileNanos{0}, cacheHits{0}, cacheMisses{0}, staleStubs{0}, staleStubFailures{0},
        selfLoops{0}, guarded{0};
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
    const JitX64FaultStats f = JitX64GetFaultStats();
    std::fprintf(stderr, "jit (%s): fault stores: %llu sites, %llu faults handled, %llu chunks\n", when,
                 static_cast<unsigned long long>(f.sites), static_cast<unsigned long long>(f.faults),
                 static_cast<unsigned long long>(f.chunks));
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

// BRAWL_JIT_DUMP_DIR: the code of everything the backend makes, appended to <dir>/jit-<pid>.code, and a text index of it in
// <dir>/jit-<pid>.idx: one record per piece of code, its host address, size and offset in the .code file, and for a block its
// ops (kind, pc, word) and code regions (BlockEmitter::Region; begin offset, op). A diagnostic for code-quality work: it
// changes no generated code.
struct CodeDump {
    std::mutex mutex;
    FILE* code = nullptr;
    FILE* index = nullptr;
    uint64_t offset = 0, serial = 0;
    bool tried = false;
    bool Open() {
        if (tried) return code != nullptr && index != nullptr;
        tried = true;
#if defined(__linux__)
        const int pid = static_cast<int>(getpid());
#else
        const int pid = 0;
#endif
        const std::string base = Opts().dumpDir + "/jit-" + std::to_string(pid);
        code = std::fopen((base + ".code").c_str(), "wb");
        index = std::fopen((base + ".idx").c_str(), "w");
        return code != nullptr && index != nullptr;
    }
    // Writes the bytes and the record's header line; the caller adds its own lines under the same lock.
    bool Begin(char tag, const char* name, const void* at, size_t size, uint32_t pc) {
        if (!Open()) return false;
        std::fwrite(at, 1, size, code);
        std::fprintf(index, "%c %llu %s %llx %zx %08X %llu\n", tag, static_cast<unsigned long long>(serial++), name,
                     static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(at)), size, pc, static_cast<unsigned long long>(offset));
        offset += size;
        return true;
    }
};
CodeDump& Dump() {
    static CodeDump dump;
    return dump;
}

void DumpCode(const char* name, const void* at, size_t size, uint32_t pc) {
    if (Opts().dumpDir.empty() || size == 0) return;
    CodeDump& d = Dump();
    std::lock_guard<std::mutex> lock(d.mutex);
    if (d.Begin('S', name, at, size, pc)) std::fflush(d.index);
}

// A block is two records with the same ops: its hot code (at `hot`) and its far code (at `far`), each with the regions that lie in
// it (offsets from the record's own address).
void DumpBlockTable(const BlockEmitter& e, const uint8_t* hot, const uint8_t* far) {
    if (Opts().dumpDir.empty()) return;
    static const char* const kRegionNames[] = {"stale", "check", "entry", "op", "cold", "end", "fallback_exit", "miss", "stub", "epoch"};
    const Block& b = e.block;
    CodeDump& d = Dump();
    std::lock_guard<std::mutex> lock(d.mutex);
    const size_t farStart = e.Layout().farStart;
    for (int part = 0; part < 2; ++part) {
        const uint8_t* const at = part == 0 ? hot : far;
        const size_t size = part == 0 ? e.HotBytes() : e.FarBytes();
        if (size == 0 || !d.Begin('B', "jit", at, size, b.start)) continue;
        for (uint32_t i = 0; i < b.count; ++i)
            std::fprintf(d.index, "O %u %s %08X %08X\n", i, kKindNames[static_cast<unsigned>(b.ops[i].kind)], b.pcs[i], b.words[i]);
        for (uint32_t i = 0; i < e.RegionCount(); ++i) {
            const BlockEmitter::RegionMark& r = e.RegionAt(i);
            const bool inFar = r.begin >= farStart;
            if (inFar != (part == 1)) continue;
            std::fprintf(d.index, "R %u %s %u\n", inFar ? static_cast<unsigned>(r.begin - farStart) : r.begin,
                         kRegionNames[static_cast<unsigned>(r.kind)], r.op);
        }
        std::fprintf(d.index, "E\n");
    }
    std::fflush(d.index);
}

// --- the host ABI ------------------------------------------------------------------------------------------------------

// The enter trampolines' frame (jit_enter, jit_enter_direct):
//   [rsp +  0..31]: Win64 shadow space
//   [rsp + 32..39]: out-parameter slot (jit_enter: JitLink**) or DirectEnterState* (jit_enter_direct)
//   [rsp + 40..87]: in-place Frame (48 bytes, used by jit_enter_direct)
//   [rsp + 88..135]: the callee-saved registers (r15 ... rbp)
// 88 == 8 mod 16, so rsp is 16-byte aligned there.
//
// Blocks run at "block level", kBlockLevel bytes below it (Linux x86-64, where blocks call each other with host calls,
// jit_x64_branch.cpp): [rsp] holds a return address into the code that entered this level (the trampoline, which enters the
// first block with `push; call`, or a calling block), [rsp + 8] the return slot (a calling block's saved Frame word, unused under
// a trampoline). One constant DWARF FDE describes every block: CFA = rsp + 16, the return address at CFA - 16 (EhFrameRecord).
// On Win64 (no FDEs, no host calls between blocks) blocks run at the trampoline's level.
constexpr int kTrampolineFrame = 88;
constexpr int kOutLinkSlot = 32;
constexpr int kDirectFrameOffset = 40;
constexpr int kBlockLevel = kDirectCalls ? 16 : 0;
constexpr int kOutAtBlock = kBlockLevel + kOutLinkSlot;  // the out slot seen from block level
static_assert(sizeof(Frame) == 48, "kTrampolineFrame assumes 48-byte Frame");
static_assert(kDirectFrameOffset + static_cast<int>(sizeof(Frame)) == kTrampolineFrame);
static_assert(offsetof(Frame, returnAddress) == offsetof(Frame, pairedMask) + 4,
              "pairedMask and returnAddress: the word a host call between blocks keeps on the stack");
static_assert(offsetof(DirectEnterState, insnCount) == 0, "exitReturn loads insnCount from [state]");
static_assert(sizeof(FastEntrySlot) == 16 && offsetof(FastEntrySlot, pc) == 0 && offsetof(FastEntrySlot, jitChecked) == 8);
static_assert(kFastEntrySlots == 2048, "subCall hash shift assumes 2048 slots");

#if defined(__linux__) && defined(__x86_64__)
// A DWARF .eh_frame record (CIE + FDE + 0 terminator) describing one constant frame over a range of code, on Linux x86-64:
// CFA = rsp + cfa, the return address at CFA - ra, and with `saved` rbp, rbx, r12 ... r15 at CFA - 16 ... CFA - 56 (the enter
// trampolines' pushes). Registered once per 2 MiB executable chunk (blocks: kBlock) and per range of the shared code.
struct EhFrameRecord {
    struct Shape {
        uint8_t cfa, ra;
        bool saved;
    };
    static constexpr Shape kBlock{16, 16, false};       // block level (see kBlockLevel)
    static constexpr Shape kTrampoline{8 + 6 * 8 + kTrampolineFrame, 8, true};
    static constexpr Shape kRoutine{16, 8, false};     // a routine called from block level that has pushed 8 bytes
    alignas(8) uint8_t bytes[96];
    bool registered = false;

    void Register(const void* base, size_t size, Shape shape = kBlock) {
        if (registered || base == nullptr || size == 0) return;
        std::memset(bytes, 0, sizeof(bytes));
        // CIE: length, id 0, version 1, augmentation "", code alignment 1, data alignment -8, return address register 16 (rip)
        uint8_t* p = bytes + 4;
        *p++ = 0; *p++ = 0; *p++ = 0; *p++ = 0;
        *p++ = 1;
        *p++ = 0;
        *p++ = 1;
        *p++ = 0x78;
        *p++ = 16;
        *p++ = 0x0c; *p++ = 7;                            // DW_CFA_def_cfa rsp, cfa (ULEB128)
        if (shape.cfa < 128) {
            *p++ = shape.cfa;
        } else {
            *p++ = static_cast<uint8_t>(0x80 | (shape.cfa & 0x7F));
            *p++ = static_cast<uint8_t>(shape.cfa >> 7);
        }
        *p++ = 0x80 | 16; *p++ = shape.ra / 8;            // DW_CFA_offset rip, CFA - ra
        if (shape.saved) {
            static const uint8_t kRegs[] = {6, 3, 12, 13, 14, 15};  // rbp, rbx, r12 ... r15 at CFA - 16 ... CFA - 56
            for (unsigned i = 0; i < 6; ++i) { *p++ = static_cast<uint8_t>(0x80 | kRegs[i]); *p++ = static_cast<uint8_t>(2 + i); }
        }
        while ((p - bytes) % 8 != 0) *p++ = 0;            // DW_CFA_nop
        const uint32_t cieLen = static_cast<uint32_t>(p - bytes) - 4;
        std::memcpy(bytes, &cieLen, 4);
        // FDE: length, CIE pointer (back to the CIE), pc_begin, pc_range (absolute 8-byte pointers), padding
        uint8_t* const fde = p;
        const uint32_t fdeLen = 4 + 8 + 8 + 4;
        std::memcpy(fde, &fdeLen, 4);
        const uint32_t ciePtr = static_cast<uint32_t>(fde + 4 - bytes);
        std::memcpy(fde + 4, &ciePtr, 4);
        const uint64_t pcBegin = reinterpret_cast<uintptr_t>(base);
        const uint64_t pcRange = static_cast<uint64_t>(size);
        std::memcpy(fde + 8, &pcBegin, 8);
        std::memcpy(fde + 16, &pcRange, 8);
        // fde + 24 .. 27: padding (nops), fde + 28 .. 31: the terminator
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
    const void* subCall = nullptr;      // Linux x86-64: eax = target, rdx = the cell entry (a bctrl to a patched function)
    const void* subCallReturn = nullptr;  // where the callee of subCall returns (JitLiftContinue finds the callee's entry there)
    const void* exitDynamic = nullptr;  // eax = the next pc
    const void* exitSlot = nullptr;     // eax = the next pc, rdx = the link slot
    const void* exitReturn = nullptr;   // the frame's function returned (the frame's own: Frame::returnAddress untagged)
    const void* sharedReturn = nullptr; // Linux x86-64: a function a block called with a host call returned (tagged)
    const void* lift = nullptr;         // Linux x86-64: an exit from such a function (rax = status << 32 | next, rdx = slot)
    const void* callArgs = nullptr;         // JitTargetCell::fast routines (Linux x86-64): float arguments, then the entry
    const void* callGuarded = nullptr;      //   JitThinCall
    const void* cellPromote = nullptr;      // a call cell's entry to the front (Linux x86-64)
    const void* exitPending = nullptr;
    const void* staleCount = nullptr;   // rcx = a block's StaleCell: counts the failure, at the threshold asks for the block's stub
#if defined(__linux__) && defined(__x86_64__)
    std::unique_ptr<EhFrameRecord> ehFrame[3];
#endif
    bool ok = false;
};
Shared g_shared;
EnterFn g_enter = nullptr;  // g_shared.enter, as a plain global: read on every enter

// A call or jump from the shared code to a fixed address: rel32 when in reach, else through rax (which the caller gives up).
void CallFar(Xbyak::CodeGenerator& a, const void* fn) {
    const int64_t distance = reinterpret_cast<const uint8_t*>(fn) - (a.getCurr() + 5);
    if (distance >= INT32_MIN && distance <= INT32_MAX) {
        a.call(fn);
        return;
    }
    a.mov(rax, reinterpret_cast<uint64_t>(fn));
    a.call(rax);
}

}  // namespace

Xbyak::Address AbsMem(Xbyak::CodeGenerator& a, const void* p, const Xbyak::Reg64& scratch, int bits) {
    const uint64_t v = reinterpret_cast<uintptr_t>(p);
    Xbyak::RegExp e;
    if (v <= 0x7FFFFFFFu) {  // statics of a non-PIE executable: [disp32]
        e = Xbyak::RegExp(static_cast<size_t>(v));
    } else {
        a.mov(scratch, v);
        e = Xbyak::RegExp(scratch);
    }
    switch (bits) {
    case 8: return a.byte[e];
    case 16: return a.word[e];
    case 32: return a.dword[e];
    default: return a.qword[e];
    }
}

namespace {

// What an exit from a function that a block called with a host call does (BuildShared's lift, jit_x64_branch.cpp): the frame
// that function would have had (a Run of its own, as RunPatched gives it) is made from the shared frame's state and continued in
// the dispatcher; when it returns, the caller's code goes on after its call. `level` is the callee's block level ([0] its return
// address into the calling code); `executed` the instructions the blocks counted since the enter trampoline began (r15). May
// throw (what the callee threw): unwinds through the lift code and the calling block to the trampoline.
void JitLiftContinue(Frame* f, uint64_t statusAndNext, JitLink* link, const void* const* level, uint64_t executed);
// The guest entry of the function a call link calls, by the address its call returns to (BlockEmitter::CallLinked): the entry of
// the frame JitLiftContinue makes for it (the log of entries, Frame::entry).
void NoteLiftEntry(const void* returnTo, uint32_t entry);

bool BuildShared() {
    Shared& s = g_shared;
    s.region = JitExecMemory::Allocate(8192);
    if (!s.region) return false;
    try {
        Xbyak::CodeGenerator a(s.region.size, s.region.base);
        Xbyak::Label epilogue, directEpilogue, directContinue, exitReturnSlow, exitSlotSlow, liftAt, sharedReturnAt;
        // the perf map's names of the pieces of this code (BRAWL_JIT_PERF_MAP), each from its mark to the next
        const uint8_t* markAt[20];
        const char* markName[20];
        unsigned marks = 0;
        const auto mark = [&](const char* name) { markAt[marks] = a.getCurr(); markName[marks++] = name; };
        const int frameRet = static_cast<int>(offsetof(Frame, returnAddress));
        const int frameWord = static_cast<int>(offsetof(Frame, pairedMask));  // pairedMask | returnAddress << 32
        const auto pops = [&]() {
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
        };
        // The trampolines enter the first block at block level (kBlockLevel): an unused return slot, then as the return address an
        // address in the trampoline (where an unwinder finds the trampoline's frame: EhFrameRecord::kTrampoline), pushed rather than
        // called so that the return stack keeps only the trampoline's own call from C++, which the exits return to.
        const auto enterBlock = [&](const Xbyak::Reg64& entry) {
            if (kDirectCalls) {
                Xbyak::Label unwindTo;
                a.push(rax);
                a.lea(rax, a.ptr[a.rip + unwindTo]);
                a.push(rax);
                a.jmp(entry);
                a.L(unwindTo);
                a.ud2();  // never reached: nothing returns to it
            } else {
                a.jmp(entry);
            }
        };
        mark("jit_enter");
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
        a.mov(kLean, AbsMem(a, &MemoryInline::g_leanBase, kLean, 64));
        a.xor_(kExecCount.cvt32(), kExecCount.cvt32());  // the guest instructions counted so far this enter (see kExecCount)
        enterBlock(ArgReg(1));

        // Fast AOT->JIT entry: void enterDirect(CpuContext* ctx, uint32_t entry, const void* jitChecked, DirectEnterState* state).
        // Uses the exact same stack frame layout as jit_enter so chunk FDEs unwind both identically.
        mark("jit_enter_direct");
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
        // pairedMask = 0 (low 32), returnAddress = lr & ~3 (high 32): every user compares it with a word address
        a.mov(eax, a.dword[ArgReg(0) + static_cast<int>(offsetof(CpuContext, lr))]);
        a.and_(eax, ~3u);
        a.shl(rax, 32);
        a.mov(a.qword[kFrame + frameWord], rax);
        a.mov(a.dword[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagDirectEnter);
        a.mov(a.dword[kFrame + static_cast<int>(offsetof(Frame, pc))], ArgReg(1).cvt32());
        a.mov(a.dword[kFrame + static_cast<int>(offsetof(Frame, entry))], ArgReg(1).cvt32());
        if (!kDirectCalls || Opts().execInMemory) a.mov(a.qword[kFrame + static_cast<int>(offsetof(Frame, executed))], 0);  // else jit_direct_continue stores it
        a.mov(a.qword[kFrame + static_cast<int>(offsetof(Frame, pending))], 0);
        a.lea(kCtxBase, a.ptr[ArgReg(0) + kCtxBias]);
        a.mov(a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, pc))], ArgReg(1).cvt32());
        a.mov(kEpochCell, reinterpret_cast<uint64_t>(&BrawlDirtyMemory::markEpoch));
        a.mov(kLean, AbsMem(a, &MemoryInline::g_leanBase, kLean, 64));
        a.xor_(kExecCount.cvt32(), kExecCount.cvt32());
        enterBlock(ArgReg(2));
        const uint8_t* const trampolinesEnd = a.getCurr();

        // A bctrl's call of a patched function (JitFastCallFor; a bl's is inline, jit_x64_branch.cpp): what GuestCodePatch::RunPatched
        // and RunCompiledEntry would do for the target, without them. Inputs: eax = target, rdx = the cell entry (JitTargetCell), lr
        // set, the rollback loop frame ruled out. Called from a block. The callee's block is found through the thread's fast-entry
        // table and called with a host call as a bl's callee is: the shared frame's pairedMask | returnAddress on the stack, the
        // callee's (0, lr tagged) in the frame. A target the table does not hold is a call of the target's wrapper.
        Xbyak::Label exitPendingAt;
        const uint8_t* subCallFdeAt = nullptr;
        const uint8_t* subCallFdeEnd = nullptr;
        if (kDirectCalls) {
            Xbyak::Label fallbackEntry, sync, synced;
            mark("jit_subcall");
            s.subCall = a.getCurr();
            a.mov(rdx, a.qword[rdx + static_cast<int>(offsetof(JitTargetCell, record))]);
            a.mov(rcx, a.qword[rdx + static_cast<int>(offsetof(JitCallTarget, patchedByte))]);
            a.test(rcx, rcx);
            a.jz(fallbackEntry, Xbyak::CodeGenerator::T_NEAR);
            a.cmp(a.byte[rcx], 0);
            a.je(fallbackEntry, Xbyak::CodeGenerator::T_NEAR);
            a.mov(rcx, AbsMem(a, &GetPatchedHooks().lowestRedirect, rcx, 64));
            a.cmp(eax, a.dword[rcx]);
            a.jae(fallbackEntry, Xbyak::CodeGenerator::T_NEAR);
            a.push(rax);  // the target and the record across the call; with the return address, the stack aligned for it
            a.push(rdx);
            a.sub(rsp, 8);
            CallFar(a, reinterpret_cast<const void*>(&JitThreadFastEntries));
            a.add(rsp, 8);
            a.mov(rdi, rax);
            a.pop(rdx);
            a.pop(rax);
            a.test(rdi, rdi);  // none (a build whose dispatcher keeps no table)
            a.jz(fallbackEntry, Xbyak::CodeGenerator::T_NEAR);
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
            // The call: rsp = L - 8 (L the site's block level). A slot with the target (JitLiftContinue reads the callee's entry
            // there), the return slot, the call: the callee's block level is L - 32.
            a.sub(rsp, 8);
            a.mov(a.qword[rsp], rax);
            a.push(a.qword[kFrame + frameWord]);
            a.mov(ecx, a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, lr))]);
            a.and_(ecx, ~3u);
            a.or_(ecx, 1u);
            a.shl(rcx, 32);
            a.mov(a.qword[kFrame + frameWord], rcx);
            a.mov(a.dword[kCtxBase + CtxDisp(offsetof(CpuContext, pc))], eax);
            a.call(r8);
            s.subCallReturn = a.getCurr();
            subCallFdeAt = a.getCurr();
            a.lea(rsp, a.ptr[rsp + 8]);  // the return slot (the callee's return leaves it)
            // RunPatched's Sync test (the callee's return has put the caller's word back; rsp = L - 16)
            const PatchedHooks& hooks = GetPatchedHooks();
            a.mov(rcx, AbsMem(a, &hooks.pendingFast, rcx, 64));
            a.cmp(a.qword[rcx], 0);
            a.jne(sync, Xbyak::CodeGenerator::T_SHORT);
            a.mov(rcx, AbsMem(a, &hooks.anyDirty, rcx, 64));
            a.cmp(a.byte[rcx], 0);
            a.jne(sync, Xbyak::CodeGenerator::T_SHORT);
            a.L(synced);
            a.add(rsp, 8);
            a.ret();
            a.L(sync);
            a.call(AbsMem(a, &hooks.sync, rcx, 64));
            a.jmp(synced, Xbyak::CodeGenerator::T_SHORT);
            a.L(fallbackEntry);
            a.lea(ArgReg(0), a.ptr[kCtxBase - kCtxBias]);
            a.jmp(a.qword[rdx + static_cast<int>(offsetof(JitCallTarget, entry))]);
            subCallFdeEnd = a.getCurr();

            // The other routines a call site calls through its cell (JitFastCallFor), from a bctrl or bctr site: rdi = the
            // context, rdx = the cell entry, lr and ctx->pc set, the rollback loop frame ruled out; [rsp] returns to the block.
            const int record = static_cast<int>(offsetof(JitTargetCell, record));
            mark("jit_exits");
            // A callee without guards whose float arguments may be held paired: NormalizeArgs, then the entry.
            // (The cell entry holds the record's argMask and entry: the record itself, a line of its own, stays out of the way.)
            const int cellEntry = static_cast<int>(offsetof(JitTargetCell, entry));
            s.callArgs = a.getCurr();
            {
                Xbyak::Label normalize;
                a.movzx(eax, a.word[rdx + static_cast<int>(offsetof(JitTargetCell, argMask))]);
                a.test(a.dword[kFrame + static_cast<int>(offsetof(Frame, pairedMask))], eax);
                a.jnz(normalize, Xbyak::CodeGenerator::T_SHORT);
                a.jmp(a.qword[rdx + cellEntry]);
                a.L(normalize);
                a.push(rdx);  // the cell entry, and the stack aligned for the call
                a.mov(rdi, kFrame);
                a.mov(rsi, a.qword[rdx + record]);
                CallFar(a, reinterpret_cast<const void*>(&JitNormalizeArgs));
                a.pop(rdx);
                a.lea(rdi, a.ptr[kCtxBase - kCtxBias]);
                a.jmp(a.qword[rdx + cellEntry]);
            }
            // A cell entry that is not the first was hit 8 times: it becomes the first (rcx = the cell, rdx = the entry; the two
            // entries swap). Answers rdx = the cell's first entry. Clobbers xmm0-xmm3.
            s.cellPromote = a.getCurr();
            static_assert(sizeof(JitTargetCell) == 32);
            a.movups(xmm0, a.ptr[rcx]);
            a.movups(xmm1, a.ptr[rcx + 16]);
            a.movups(xmm2, a.ptr[rdx]);
            a.movups(xmm3, a.ptr[rdx + 16]);
            a.movups(a.ptr[rcx], xmm2);
            a.movups(a.ptr[rcx + 16], xmm3);
            a.movups(a.ptr[rdx], xmm0);
            a.movups(a.ptr[rdx + 16], xmm1);
            a.mov(rdx, rcx);
            a.ret();
            // Any other callee: JitThinCall; an exception it stored leaves the block (the return address is dropped: rsp is the
            // block's again).
            s.callGuarded = a.getCurr();
            {
                Xbyak::Label pending;
                a.sub(rsp, 8);
                a.mov(rdi, kFrame);
                a.mov(rsi, a.qword[rdx + record]);
                CallFar(a, reinterpret_cast<const void*>(&JitThinCall));
                a.add(rsp, 8);
                a.test(rax, rax);
                a.jnz(pending, Xbyak::CodeGenerator::T_SHORT);
                a.ret();
                a.L(pending);
                a.add(rsp, 8);
                a.jmp(exitPendingAt, Xbyak::CodeGenerator::T_NEAR);
            }
        }

        // Everything from here to fdeEnd runs at block level (the blocks' FDE).
        const uint8_t* const fdeStart = a.getCurr();

        // The frame's own function returned (blocks test Frame::returnAddress's tag first: BlockEmitter::ExitReturn).
        mark("jit_exit_return");
        s.exitReturn = a.getCurr();
        a.test(a.byte[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagDirectEnter);
        a.jz(exitReturnSlow, Xbyak::CodeGenerator::T_SHORT);
        // Hot direct-enter return: count instructions directly into ThreadState's counter, tear down frame, return.
        a.mov(rcx, a.qword[rsp + kOutAtBlock]);
        a.mov(rcx, a.qword[rcx]);
        a.add(a.qword[rcx], kExecCount);
        a.L(directEpilogue);
        a.add(rsp, kBlockLevel + kTrampolineFrame);
        pops();
        a.ret();

        if (kDirectCalls) {
            // A function a block called with a host call returned (Frame::returnAddress tagged): the caller's pairedMask and
            // returnAddress back from the return slot, then to the caller's code, which drops the slot (a plain `ret`: `ret 8`
            // measured 4.5% more block cycles, see ExitSharedReturn).
            mark("jit_subcall_return");  // the perf map names the profile tools know (cqlib.TRAMPOLINES)
            a.L(sharedReturnAt);
            s.sharedReturn = a.getCurr();
            a.mov(rax, a.qword[rsp + 8]);
            a.mov(a.qword[kFrame + frameWord], rax);
            a.ret();
        }

        mark("jit_exit_return_slow");
        a.L(exitReturnSlow);
        a.mov(rax, static_cast<uint64_t>(JitStatus::Return) << 32);
        a.jmp(epilogue);

        // The exits to the dispatcher: from a function a block called with a host call (tagged) through lift, from a frame the
        // lean entry made through its continuation, else the trampoline's epilogue.
        const auto toLift = [&]() {
            if (!kDirectCalls) return;
            a.test(a.byte[kFrame + frameRet], 1);
            a.jnz(liftAt, Xbyak::CodeGenerator::T_NEAR);
        };
        mark("jit_exit_dynamic");
        s.exitDynamic = a.getCurr();
        a.mov(eax, eax);
        a.xor_(edx, edx);
        toLift();
        a.test(a.byte[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagDirectEnter);
        a.jnz(directContinue);
        a.jmp(epilogue);

        mark("jit_exit_slot");
        s.exitSlot = a.getCurr();
        a.mov(eax, eax);
        toLift();
        a.test(a.byte[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagDirectEnter);
        a.jnz(directContinue);
        a.L(exitSlotSlow);
        a.mov(rcx, a.qword[rsp + kOutAtBlock]);
        a.mov(a.qword[rcx], rdx);
        a.jmp(epilogue);

        mark("jit_exit_pending");
        a.L(exitPendingAt);
        s.exitPending = a.getCurr();
        a.mov(rax, static_cast<uint64_t>(JitStatus::Pending) << 32);
        a.xor_(edx, edx);
        toLift();
        a.test(a.byte[kFrame + static_cast<int>(offsetof(Frame, flags))], kFrameFlagDirectEnter);
        a.jnz(directContinue);

        mark("jit_epilogue");
        a.L(epilogue);
        a.add(a.qword[kFrame + static_cast<int>(offsetof(Frame, executed))], kExecCount);  // hand the count to the dispatcher
        a.add(rsp, kBlockLevel + kTrampolineFrame);
        pops();
        a.ret();

        // Direct-enter slow continuation: rax = status << 32 | next, rdx = link (or null).
        // Calls JitDirectContinue(DirectEnterState*, Frame*, uint64_t statusAndNext, JitLink* link).
        mark("jit_direct_continue");
        a.L(directContinue);
        // The frame's first exit (JitDirectContinue clears the flag that leads here): executed is still the 0 the trampoline
        // stored, or not stored at all (except BRAWL_JIT_EXEC_MEM=1, where blocks add to it).
        if (kDirectCalls && !Opts().execInMemory) a.mov(a.qword[kFrame + static_cast<int>(offsetof(Frame, executed))], kExecCount);
        else a.add(a.qword[kFrame + static_cast<int>(offsetof(Frame, executed))], kExecCount);
        if (kWin64) {
            a.mov(r9, rdx);
            a.mov(r8, rax);
            a.mov(rdx, kFrame);
            a.mov(rcx, a.qword[rsp + kOutAtBlock]);
        } else {
            a.mov(rcx, rdx);
            a.mov(rdx, rax);
            a.mov(rsi, kFrame);
            a.mov(rdi, a.qword[rsp + kOutAtBlock]);
        }
        CallFar(a, reinterpret_cast<const void*>(&JitDirectContinue));
        a.jmp(directEpilogue);

        if (kDirectCalls) {
            // An exit to the dispatcher from a function a block called with a host call (rax = status << 32 | next, rdx = link,
            // at the callee's block level): JitLiftContinue runs it in a frame of its own until it returns, then the shared return.
            mark("jit_direct_continue");  // the lift (named as the continuation it is, for the profile tools)
            a.L(liftAt);
            s.lift = a.getCurr();
            a.mov(rdi, kFrame);
            a.mov(rsi, rax);
            a.mov(rcx, rsp);
            a.mov(r8, kExecCount);
            CallFar(a, reinterpret_cast<const void*>(&JitLiftContinue));
            a.xor_(kExecCount.cvt32(), kExecCount.cvt32());  // JitLiftContinue counted every instruction r15 held
            a.jmp(sharedReturnAt);
        }

        // The stale paths' counting code (StaleCell): shared, so a block holds only its cell and a jump through it.
        mark("jit_stale_count");
        s.staleCount = a.getCurr();
        if (Opts().staleThreshold != 0) {
            Xbyak::Label toPlain;
            if (diag::ChurnOn()) {  // BRAWL_JIT_CHURN: count it
                a.mov(rax, reinterpret_cast<uint64_t>(diag::CounterCell(diag::kJitStalePaths)));
                a.inc(a.qword[rax]);
            }
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
        const uint8_t* const fdeEnd = a.getCurr();

        a.ready();
        s.enter = reinterpret_cast<EnterFn>(const_cast<uint8_t*>(enterAt));
        s.enterDirect = reinterpret_cast<EnterDirectFn>(const_cast<uint8_t*>(enterDirectAt));
#if defined(__linux__) && defined(__x86_64__)
        // the trampolines (what matters is where their call of the first block returns), the bctrl routine (its call of the
        // callee), and the code that runs at block level
        for (auto& r : s.ehFrame) r = std::make_unique<EhFrameRecord>();
        s.ehFrame[0]->Register(enterAt, static_cast<size_t>(trampolinesEnd - enterAt), EhFrameRecord::kTrampoline);
        if (subCallFdeAt != nullptr)
            s.ehFrame[1]->Register(subCallFdeAt, static_cast<size_t>(subCallFdeEnd - subCallFdeAt), EhFrameRecord::kRoutine);
        s.ehFrame[2]->Register(fdeStart, static_cast<size_t>(fdeEnd - fdeStart), EhFrameRecord::kBlock);
#else
        (void)fdeStart;
        (void)fdeEnd;
        (void)trampolinesEnd;
        (void)subCallFdeAt;
        (void)subCallFdeEnd;
#endif
        markAt[marks] = a.getCurr();
        for (unsigned i = 0; i < marks; ++i) {
            const size_t size = static_cast<size_t>(markAt[i + 1] - markAt[i]);
            if (size == 0) continue;
            PerfMapAdd(markAt[i], size, markName[i], 0);
            DumpCode(markName[i], markAt[i], size, 0);
        }
    } catch (...) {
        JitExecMemory::Free(s.region);
        s.region = {};
        return false;
    }
    g_enter = s.enter;
    s.ok = true;
    return true;
}

std::mutex g_liftMutex;
std::unordered_map<const void*, uint32_t>* g_liftEntries = nullptr;  // never freed: code may be on a stack at exit

void NoteLiftEntry(const void* returnTo, uint32_t entry) {
    std::lock_guard<std::mutex> lock(g_liftMutex);
    if (g_liftEntries == nullptr) g_liftEntries = new std::unordered_map<const void*, uint32_t>();
    (*g_liftEntries)[returnTo] = entry;
}

uint32_t LiftEntry(const void* const* level) {
    // jit_subcall keeps the target in the slot above the return slot
    if (level[0] == g_shared.subCallReturn) return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(level[2]));
    std::lock_guard<std::mutex> lock(g_liftMutex);
    if (g_liftEntries == nullptr) return 0;
    const auto found = g_liftEntries->find(level[0]);
    return found != g_liftEntries->end() ? found->second : 0;
}

void JitLiftContinue(Frame* f, uint64_t statusAndNext, JitLink* link, const void* const* level, uint64_t executed) {
    // What RunPatched's Run would have had: the function's own pairedMask and return address (untagged: a frame of its own),
    // the exception its code stored; JitDirectContinue then does what it does for a frame the lean entry made, and the Sync test.
    Frame s(f->ctx, LiftEntry(level));
    s.pairedMask = f->pairedMask;
    s.returnAddress = f->returnAddress & ~3u;
    s.flags = kFrameFlagSubCall;
    s.executed = Opts().execInMemory ? 0 : executed;
    s.pending = std::move(f->pending);
    f->pending = nullptr;
    JitDirectContinue(nullptr, &s, statusAndNext, link);
}

}  // namespace

const Xbyak::Reg64& ArgReg(unsigned i) {
    static const Xbyak::Reg64* const sysv[] = {&rdi, &rsi, &rdx, &rcx, &r8, &r9};
    static const Xbyak::Reg64* const win[] = {&rcx, &rdx, &r8, &r9};
    if (kWin64) return *win[i < 4 ? i : 3];
    return *sysv[i < 6 ? i : 5];
}

void Jcc(Xbyak::CodeGenerator& a, Cc cc, const Xbyak::Label& target, Xbyak::CodeGenerator::LabelType type) {
    switch (cc) {
    case Cc::O: a.jo(target, type); break;
    case Cc::NO: a.jno(target, type); break;
    case Cc::B: a.jb(target, type); break;
    case Cc::AE: a.jae(target, type); break;
    case Cc::E: a.je(target, type); break;
    case Cc::NE: a.jne(target, type); break;
    case Cc::BE: a.jbe(target, type); break;
    case Cc::A: a.ja(target, type); break;
    case Cc::S: a.js(target, type); break;
    case Cc::NS: a.jns(target, type); break;
    case Cc::P: a.jp(target, type); break;
    case Cc::NP: a.jnp(target, type); break;
    case Cc::L: a.jl(target, type); break;
    case Cc::GE: a.jge(target, type); break;
    case Cc::LE: a.jle(target, type); break;
    case Cc::G: a.jg(target, type); break;
    }
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

BlockEmitter::BlockEmitter(const Block& b, void* buffer, size_t size, const BlockLayout& layout)
    : a(size, buffer), block(b), layout_(layout) {
    a.setDefaultJmpNEAR(true);  // a forward jump to a label not yet defined is a 32-bit displacement unless T_SHORT is asked
    a.SetLimit(layout.hotLimit);  // the main path first: it may not run into the far code
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
Xbyak::Address BlockEmitter::FrameReturn() const { return a.dword[kFrame + static_cast<int>(offsetof(Frame, returnAddress))]; }
Xbyak::Address BlockEmitter::FrameWord() const { return a.qword[kFrame + static_cast<int>(offsetof(Frame, pairedMask))]; }
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
    FlushDirtyFprs();  // memory current for the FPR cache too (a cold path may reload a snapshot it does not store first)
}

void BlockEmitter::FlushVolatileGprs() {
    for (uint32_t s = 1; s < kGprCacheSlots; ++s) {
        if (gprSlots_[s].guest >= 0) {
            if (gprSlots_[s].dirty) a.mov(Gpr(static_cast<uint32_t>(gprSlots_[s].guest)), GprCacheReg(s));
            gprSlots_[s] = GprSlot{};
        }
    }
    FlushAllFprs();  // every FPR cache slot is volatile too
}

void BlockEmitter::FlushAllGprs() {
    for (uint32_t s = 0; s < kGprCacheSlots; ++s) {
        if (gprSlots_[s].guest >= 0) {
            if (gprSlots_[s].dirty) a.mov(Gpr(static_cast<uint32_t>(gprSlots_[s].guest)), GprCacheReg(s));
            gprSlots_[s] = GprSlot{};
        }
    }
    FlushAllFprs();
}

BlockEmitter::GprSnapshot BlockEmitter::SnapshotGprs() const {
    GprSnapshot snap{};
    for (uint32_t s = 0; s < kGprCacheSlots; ++s) {
        snap.guest[s] = gprSlots_[s].guest;
        if (gprSlots_[s].guest >= 0 && gprSlots_[s].dirty) snap.dirtyMask |= static_cast<uint8_t>(1u << s);
    }
    SnapshotFprs(snap);
    return snap;
}

void BlockEmitter::EmitFlushSnapshot(const GprSnapshot& snap) {
    for (uint32_t s = 0; s < kGprCacheSlots; ++s) {
        if ((snap.dirtyMask & (1u << s)) != 0) {
            a.mov(Gpr(static_cast<uint32_t>(snap.guest[s])), GprCacheReg(s));
        }
    }
    EmitFlushFprSnapshot(snap);
}

void BlockEmitter::EmitReloadSnapshotVolatile(const GprSnapshot& snap) {
    for (uint32_t s = 1; s < kGprCacheSlots; ++s) {
        if (snap.guest[s] >= 0) {
            a.mov(GprCacheReg(s), Gpr(static_cast<uint32_t>(snap.guest[s])));
        }
    }
    EmitReloadFprSnapshot(snap);
}

Xbyak::Label& BlockEmitter::NewLabel() {
    if (labelCount_ == kMaxLabels) throw EmitError{"labels"};
    return labels_[labelCount_++];
}

BlockEmitter::ColdItem& BlockEmitter::NewCold() {
    if (coldCount_ == kMaxCold) throw EmitError{"cold paths"};
    cold_[coldCount_].op = index;
    return cold_[coldCount_++];
}

void BlockEmitter::StorePc(uint32_t i) { a.mov(Pc(), PcOf(i)); }

uint32_t BlockEmitter::AddFaultSite() {
    if (faultSiteCount_ == kMaxFaultSites) throw EmitError{"fault sites"};
    faultStore_[faultSiteCount_] = static_cast<uint32_t>(a.getSize());
    faultSlow_[faultSiteCount_] = 0;
    return faultSiteCount_++;
}

void BlockEmitter::FaultSiteSlowHere(uint32_t site) { faultSlow_[site] = static_cast<uint32_t>(a.getSize()); }

Xbyak::Label& BlockEmitter::NewCell() {
    if (cellCount_ == kMaxCells) throw EmitError{"cells"};
    return cells_[cellCount_++];
}

static_assert(sizeof(JitTargetCell) == 32 && offsetof(JitTargetCell, target) == 0);

void BlockEmitter::CellProbe(Xbyak::Label& cell, const Xbyak::Label& miss) {
    const int key = static_cast<int>(offsetof(JitTargetCell, target)), record = static_cast<int>(offsetof(JitTargetCell, record));
    Xbyak::Label hit;
    a.lea(rcx, a.ptr[a.rip + cell]);
    for (uint32_t i = 0; i < kCellWays; ++i) {
        const int at = static_cast<int>(i * sizeof(JitTargetCell));
        a.cmp(eax, a.dword[rcx + at + key]);
        a.mov(rdx, a.qword[rcx + at + record]);  // mov leaves the flags
        if (i + 1 < kCellWays) a.je(hit, Xbyak::CodeGenerator::T_NEAR);
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
void BlockEmitter::JumpIfAbs(Cc cc, const void* target) {
    static const bool alwaysLong = EnvOn("BRAWL_JIT_FAR_JUMPS");
    const int64_t distance = reinterpret_cast<const uint8_t*>(target) - (a.getCurr() + 6);
    if (!alwaysLong && distance >= INT32_MIN && distance <= INT32_MAX) {
        switch (cc) {
        case Cc::E: a.je(target); return;
        case Cc::NE: a.jne(target); return;
        case Cc::B: a.jb(target); return;
        case Cc::AE: a.jae(target); return;
        default: break;
        }
    }
    Xbyak::Label skip;
    Jcc(a, Invert(cc), skip, Xbyak::CodeGenerator::T_SHORT);
    JmpAbs(target);
    a.L(skip);
}

void BlockEmitter::ExitReturn() {
    // A function a block called with a host call (Frame::returnAddress tagged) returns to its caller's code; the frame's own
    // function leaves the frame.
    if (kDirectCalls && g_shared.sharedReturn != nullptr) {
        a.test(a.byte[kFrame + static_cast<int>(offsetof(Frame, returnAddress))], 1);
        JumpIfAbs(Cc::NE, g_shared.sharedReturn);
    }
    ExitFrameReturn();
}
void BlockEmitter::ExitFrameReturn() { JmpAbs(g_shared.exitReturn); }
void BlockEmitter::ExitSharedReturn() {
    a.mov(rax, a.qword[rsp + 8]);  // the caller's pairedMask | returnAddress, from the return slot
    a.mov(FrameWord(), rax);
    // A plain `ret`; the caller drops the slot after its call (`lea rsp, [rsp + 8]`). `ret 8` here cost 4.5% of the blocks' cycles
    // in interleaved runs. An unwinder never stops between the two: the blocks' FDE gives the caller's rsp as the callee's block
    // level + 16 either way.
    a.ret();
}
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
    // A loop back to the block's own start, in a block whose code cannot have changed since its entry passed a validity test
    // (every way in compares its words or tests its marks): straight to the entry (the count), no slot, no test.
    if (targetPc == block.start && (inCold_ ? coldOp_ : index) < selfLoopPrefix_) {
        a.jmp(entry_, Xbyak::CodeGenerator::T_NEAR);
        Bump(g_counters.selfLoops);
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

void BlockEmitter::ExitLinkedIf(Cc cc, uint32_t targetPc) {
    static const bool alwaysLong = EnvOn("BRAWL_JIT_FAR_JUMPS");
    if (IsHookPc(targetPc) || alwaysLong) {  // nothing to patch: over the unconditional exit
        Xbyak::Label skip;
        Jcc(a, Invert(cc), skip, Xbyak::CodeGenerator::T_NEAR);
        ExitLinked(targetPc);
        a.L(skip);
        return;
    }
    // a conditional loop back to the block's own start (the usual loop form), as ExitLinked's: straight to the entry
    if (targetPc == block.start && (inCold_ ? coldOp_ : index) < selfLoopPrefix_) {
        Jcc(a, cc, entry_, Xbyak::CodeGenerator::T_NEAR);
        Bump(g_counters.selfLoops);
        return;
    }
    if (exitCount_ == kMaxExits) throw EmitError{"link slots"};
    const uint32_t i = exitCount_++;
    exits_[i].pc = targetPc;
    exits_[i].patchOffset = static_cast<uint32_t>(a.getSize()) + 1;  // 0F 8x rel32: patchSite + 1 is the rel32 (see ExitLinkedIf)
    exits_[i].conditional = true;
    Jcc(a, cc, exits_[i].stub, Xbyak::CodeGenerator::T_NEAR);
}

void BlockEmitter::JumpIfSo(const Xbyak::Label& target) {
    if (MKW_LEAN_FLAGS) return;  // compares do not copy SO: nothing depends on it
    a.test(a.byte[kCtxBase + CtxDisp(offsetof(CpuContext, xer) + 3)], 0x80);
    a.jnz(target, Xbyak::CodeGenerator::T_NEAR);
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
    if (diag::ChurnOn()) {  // BRAWL_JIT_CHURN
        a.mov(r8, reinterpret_cast<uint64_t>(diag::CounterCell(diag::kJitCacheMisses)));
        a.inc(a.qword[r8]);
    }
    a.lea(rdx, a.ptr[a.rip + c.slots]);
    JmpAbs(g_shared.exitSlot);
}

void BlockEmitter::ExitDynamicCached(uint32_t which) {
    CacheProbe(which);
    ExitDynamicMiss(which);
}

void BlockEmitter::ExitDynamicAfterCall(uint32_t which) {
    Xbyak::Label stale;
    JumpIfSo(stale);  // the callee set XER[SO]: through the dispatcher, which does not enter compiled code while it is set
    if (g_jitEpochFault != 3) {  // 3: a test aid, the epoch guard left out on purpose
        a.mov(rcx, reinterpret_cast<uint64_t>(JitLinkEpochCell()));
        a.mov(ecx, a.dword[rcx]);
        a.cmp(ecx, a.dword[kEpochCell]);
        a.jne(stale, Xbyak::CodeGenerator::T_NEAR);
    }
    ExitDynamicCached(which);
    a.L(stale);
    ExitDynamic();
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

void BlockEmitter::CallLinked(uint32_t targetPc, const uint8_t* guard, Xbyak::Label& wrapper) {
    if (exitCount_ == kMaxExits) throw EmitError{"link slots"};
    const uint32_t i = exitCount_++;
    exits_[i].pc = targetPc;
    exits_[i].call = true;
    exits_[i].guard = guard;
    exits_[i].wrapper = &wrapper;
    // E8 rel32 to the stub (in this chunk), patched like ExitLinked's jmp, through the far trampoline when the target is out of
    // reach; always the patchable form (BRAWL_JIT_FAR_JUMPS too): Resume tells call links by it
    exits_[i].patchOffset = static_cast<uint32_t>(a.getSize());
    a.call(exits_[i].stub);
    exits_[i].retOffset = static_cast<uint32_t>(a.getSize());
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
    const uint32_t span = EmitSlowPath(i);
    // The interpreter's handler may have set XER[SO] (an OE form, mtxer, a call): then the dispatcher goes on (JumpIfSo). Every
    // caller has flushed the GPR cache.
    if (!MKW_LEAN_FLAGS) {
        Xbyak::Label& so = NewLabel();
        JumpIfSo(so);
        const uint32_t next = PcAfter(i, span);
        Cold([&so, next](BlockEmitter& e) {
            e.a.L(so);
            e.ExitTo(next);
        });
    }
    return span;
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

// epoch == markEpoch, the block's epoch an immediate the core rewrites whenever the block cache changes it (Block::jitEpoch,
// JitBackend::setEpoch): `cmp dword [r14], imm32`, the immediate 4-byte aligned (one store rewrites it). Clobbers nothing.
void BlockEmitter::EmitEpochTest(const Xbyak::Label& fail, bool shortJump) {
    const auto T = shortJump ? Xbyak::CodeGenerator::T_SHORT : Xbyak::CodeGenerator::T_NEAR;
    const size_t misalign = (reinterpret_cast<uintptr_t>(a.getCurr()) + 3) & 3;
    if (misalign != 0) a.nop(4 - misalign);
    if (epochSiteCount_ == kMaxEpochSites) throw EmitError{"epoch tests"};
    a.db(0x41);  // REX.B (kEpochCell is r14)
    a.db(0x81);  // cmp r/m32, imm32 (never the imm8 form: the value is rewritten)
    a.db(0x3E);  // ModRM: /7, [r14]
    epochSites_[epochSiteCount_++] = static_cast<uint32_t>(a.getSize());
    a.dd(block.epoch);
    a.jne(fail, T);
}

// Every mark the block's check[] names is clear (BlockCache::Current always tests the first two; an unused one has no mask).
void BlockEmitter::EmitMarksTest(const Xbyak::Label& fail, bool shortJump) {
    const uint32_t checks = block.checks > 2 ? block.checks : 2;
    for (uint32_t k = 0; k < checks; ++k)
        if (testMask_[k] != 0) EmitMarkTest(MarkCheck{testMask_[k], block.check[k].at}, fail, shortJump);  // guarded lines left out
}

void BlockEmitter::EmitRecheck(uint32_t nextPc) {
    Xbyak::Label& so = NewLabel();
    Xbyak::Label& fail = NewLabel();
    JumpIfSo(so);  // the callee set XER[SO]: the dispatcher goes on (see JumpIfSo)
    EmitEpochTest(fail, false);
    EmitMarksTest(fail, false);  // the guarded lines' marks are not tested: a write there dropped the block (epoch 0)
    // The block's code may have changed (a callee wrote it, or a restore dropped it): the rest of it is the block at nextPc's,
    // reached through a link slot (whose target tests its own validity) while the slots' epoch is the current one
    // (JitLinkEpochCell, as ExitDynamicAfterCall); else through the dispatcher. A block with many exits keeps the plain exit.
    const bool slot = exitCount_ + 8 < kMaxExits && !IsHookPc(nextPc);
    Cold([&so, &fail, nextPc, slot](BlockEmitter& e) {
        e.a.L(fail);
        if (slot) {
            e.a.mov(rcx, reinterpret_cast<uint64_t>(JitLinkEpochCell()));
            e.a.mov(ecx, e.a.dword[rcx]);
            e.a.cmp(ecx, e.a.dword[kEpochCell]);
            e.a.jne(so, Xbyak::CodeGenerator::T_NEAR);
            e.ExitLinked(nextPc);
        }
        e.a.L(so);
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

// How many of block b's first ops cannot write guest memory, move the epoch or retire a block: none of them stores, calls, jumps
// through a register or runs C++ that could (Generic ops, special registers, the quantized loads' helper, ops forced to the fallback
// call). A jump back to b's own start from op k < that number may skip the validity test: b's words matched guest memory when it
// was entered, and nothing since can have changed them.
uint32_t BlockEmitter::SelfLoopPrefix(const Block& b) {
    for (uint32_t i = 0; i < b.count; ++i) {
        const Op& o = b.ops[i];
        if (Opts().forced[static_cast<unsigned>(o.kind)]) return i;
        switch (o.kind) {
        case K::Mfspr: case K::Mtspr: case K::PsqL: case K::PsqLu: return i;
        case K::B:
            // a tail call, not decided at decode, or a jump a code variant may continue (EmitB's fallback call)
            if (o.b != 2 || JitContinuationsPossible() || JitContinuationAt(o.imm)) return i;
            continue;
        case K::Bc: case K::CmpwiBc: case K::CmplwiBc: case K::CmpwBc: case K::CmplwBc: case K::RlwinmRcBc: case K::Jmp:
            continue;
        default:
            break;
        }
        const Group g = GroupOf(o.kind);
        const bool load = (o.kind >= K::Lwz && o.kind <= K::Lha) || o.kind == K::Lmw || (o.kind >= K::Lwzx && o.kind <= K::Lhzx) ||
                          (o.kind >= K::Lfs && o.kind <= K::Lfdu) || o.kind == K::Lfsx;
        if (g == Group::Int || g == Group::Fp || load) continue;
        return i;
    }
    return b.count;
}

// Guarded code (jit_x64.h, Guarded; JitGuardedLine says which lines may go untested and why that is sound). The block's own mark
// windows (block.check, which the interpreter's Current tests whole) with the guarded lines' bytes cleared. BRAWL_JIT_GUARD=0:
// every line tested.
void BlockEmitter::ComputeGuard() {
    const uint32_t checks = block.checks > 2 ? block.checks : 2;
    for (uint32_t k = 0; k < checks; ++k) testMask_[k] = block.check[k].mask;
    guarded_ = false;
    untested_ = false;
    static const bool off = [] { const char* v = std::getenv("BRAWL_JIT_GUARD"); return v != nullptr && *v == '0'; }();
    if (off || block.jitNoGuard) return;
    for (uint32_t s = 0; s < block.segments; ++s) {
        const Segment& g = block.segment[s];
        for (uint32_t m = g.firstMark; m <= g.lastMark && m - g.firstMark < 3u; ++m) {
            if (!JitGuardedLine(g.start, m)) continue;
            for (uint32_t k = 0; k < checks; ++k) {
                const uint32_t byte = m - block.check[k].at;
                if (byte >= 8) continue;
                uint8_t bytes[8];
                std::memcpy(bytes, &testMask_[k], 8);
                if (bytes[byte] != 0) guarded_ = true;
                bytes[byte] = 0;
                std::memcpy(&testMask_[k], bytes, 8);
            }
        }
    }
    untested_ = true;
    for (uint32_t k = 0; k < checks; ++k)
        if (testMask_[k] != 0) untested_ = false;
}

void BlockEmitter::EmitPrologue() {
    AnalyseCr();
    AnalyseForms();
    AnalyseGprUses();
    AnalyseFprUses();
    ComputeGuard();
    staleStub_ = Opts().staleThreshold != 0 && owner_ != nullptr && !untested_;
    selfLoopPrefix_ = SelfLoopPrefix(block);
    // The hot code starts with the entry link slots and inline caches aim at (Block::jitLinked): the marks only, as a slot is
    // aimed at a block only while its epoch is the current one. Set marks go to the stale exit (far code, Finish). Guarded code
    // tests nothing: a write under it suspended its slots.
    linkedAt_ = a.getCurr();
    a.L(linked_);
    MarkRegion(Region::Check, 0);
    EmitMarksTest(stale_, false);
    entryAt_ = a.getCurr();
    a.L(entry_);
    MarkRegion(Region::Entry, 0);
    // The block's instruction count goes to a register the trampoline's epilogue adds to frame.executed (one add, not a
    // read-modify-write of memory per block); BRAWL_JIT_EXEC_MEM=1 (a test switch) adds it to frame.executed in memory per block.
    if (Opts().execInMemory) a.add(Executed(), block.count);
    else a.add(kExecCount, block.count);
    // ctx->pc is only a diagnostic (fault and trace reports) between the dispatcher's store and the next store of a call or a
    // slow path (StorePc); BRAWL_JIT_BLOCK_PC=1 (a test switch) stores the block's start here.
    if (Opts().blockPc) a.mov(Pc(), block.start);
}

void BlockEmitter::Finish() {
    hotBytes_ = a.getSize();
    // --- the far code ---
    a.SetLimit(layout_.farLimit);
    a.setSize(layout_.farStart);
    // The entry from outside compiled code (the dispatcher's fast entries, jit_subcall): the epoch, then the linked entry's
    // marks. A stale epoch goes to the dispatcher (its Refresh compares the words and starts the slots' new epoch); set marks go
    // through the stale cell, whose stub, once made, compares the words in place: the epoch is current there.
    checkedAt_ = a.getCurr();
    MarkRegion(Region::Epoch, 0);
    EmitEpochTest(plainStale_, true);
    a.jmp(linked_, Xbyak::CodeGenerator::T_NEAR);
    MarkRegion(Region::Stale, 0);
    a.L(stale_);
    if (staleStub_) {  // through the block's StaleCell (rcx its address): the shared counting code, later a stub
        a.lea(rcx, a.ptr[a.rip + staleCell_]);
        a.jmp(a.ptr[rcx]);
    }
    a.L(plainStale_);
    ExitTo(block.start);
    inCold_ = true;
    for (uint32_t i = 0; i < coldCount_; ++i) {
        MarkRegion(Region::Cold, cold_[i].op);
        coldOp_ = cold_[i].op;
        cold_[i].run(cold_[i].storage, *this);
    }
    inCold_ = false;
    for (uint32_t i = 0; i < faultSiteCount_; ++i)  // every store into the store view has its slow entry (the handler goes there)
        if (faultSlow_[i] == 0) throw EmitError{"a fault site's slow entry"};
    if (fallbackUsed_) {  // ecx: what (1 leave for eax, 2 return, 3 pending); after the cold code, whose slow paths may use it
        Xbyak::Label returned, pending;
        MarkRegion(Region::FallbackExit, 0);
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
    for (DynCache& c : caches_) {  // a probe whose miss path was never emitted (its caller decided against the dispatcher)
        if (!c.probed || c.missEmitted) continue;
        c.missEmitted = true;
        MarkRegion(Region::Miss, 0);
        a.L(c.miss);
        a.lea(rdx, a.ptr[a.rip + c.slots]);
        JmpAbs(g_shared.exitSlot);
    }
    MarkRegion(Region::Stub, 0);
    for (uint32_t i = 0; i < exitCount_; ++i) {  // each slot's unlinked stub (and far fallback if >2 GiB)
        a.L(exits_[i].stub);
        Xbyak::Label wrapper;
        if (exits_[i].call) {  // a call link's (CallLinked): what the wrapper would decide, first
            a.cmp(AbsMem(a, exits_[i].guard, rcx, 8), 0);
            a.je(wrapper, Xbyak::CodeGenerator::T_SHORT);
            a.cmp(AbsMem(a, GetPatchedHooks().lowestRedirect, rcx, 32), exits_[i].pc);
            a.jbe(wrapper, Xbyak::CodeGenerator::T_SHORT);
        }
        a.mov(eax, exits_[i].pc);
        a.lea(rdx, a.ptr[a.rip + exits_[i].slot]);
        // a call link's: the callee's frame state is set and the call made, so the dispatcher runs it in a frame of its own and
        // links the slot (lift); any other exits as the slot's stub
        JmpAbs(exits_[i].call ? g_shared.lift : g_shared.exitSlot);
        if (exits_[i].call) {  // the wrapper after all: the caller's word back, the call undone (its return stack entry unpaired)
            a.L(wrapper);
            a.mov(rax, a.qword[rsp + 8]);
            a.mov(FrameWord(), rax);
            a.add(rsp, 16);
            a.jmp(*exits_[i].wrapper, Xbyak::CodeGenerator::T_NEAR);
        }
        if (exits_[i].patchOffset != Exit::kNoPatch) {
            a.L(exits_[i].far);
            a.jmp(a.ptr[a.rip + exits_[i].slot]);
        }
    }
    farBytes_ = a.getSize() - layout_.farStart;
    // --- the data, written at run time (the dispatcher's links, the call sites' cells, the stale cell's count) ---
    a.SetLimit(layout_.dataLimit);
    a.setSize(layout_.dataStart);
    for (uint32_t i = 0; i < exitCount_; ++i) {  // the slots, filled in below
        a.L(exits_[i].slot);
        for (unsigned w = 0; w < sizeof(JitLink) / 8; ++w) a.dq(0);
    }
    if (staleStub_) {  // the stale path's cell: filled in below
        a.L(staleCell_);
        for (unsigned w = 0; w < sizeof(StaleCell) / 8; ++w) a.dq(0);
    }
    for (uint32_t i = 0; i < cellCount_; ++i) {  // the call sites' cells: no key, no record, kCellWays entries each
        a.L(cells_[i]);
        for (uint32_t w = 0; w < kCellWays; ++w) {
            a.dq(JitTargetCell::kNoKey);
            a.dq(0);
            a.dq(0);
            a.dq(0);
        }
    }
    a.L(epochRecord_);  // the epoch tests' immediates (Block::jitEpoch): their count, then their addresses, filled in below
    a.dq(epochSiteCount_);
    for (uint32_t i = 0; i < epochSiteCount_; ++i) a.dq(0);
    EmitConsts();  // the float code's constants (BlockEmitter::Const128): read only, 16-byte aligned
    for (DynCache& c : caches_) {  // the inline caches' entries, one line each (a probe reads an entry's key and target)
        if (!c.probed) continue;
        a.align(64);
        a.L(c.slots);
        for (unsigned w = 0; w < kCacheEntries[&c - caches_] * sizeof(JitLink) / 8; ++w) a.dq(0);
    }
    dataBytes_ = a.getSize() - layout_.dataStart;
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
    uint64_t* const record = reinterpret_cast<uint64_t*>(const_cast<uint8_t*>(epochRecord_.getAddress()));
    for (uint32_t i = 0; i < epochSiteCount_; ++i) record[1 + i] = reinterpret_cast<uintptr_t>(codeBase + epochSites_[i]);
    for (uint32_t i = 0; i < exitCount_; ++i) {
        JitLink* const slot = reinterpret_cast<JitLink*>(const_cast<uint8_t*>(exits_[i].slot.getAddress()));
        slot->target = slot->unlinked = exits_[i].stub.getAddress();
        slot->next = nullptr;
        slot->prev = nullptr;
        slot->to = nullptr;
        slot->key = JitLink::kNoKey;
        slot->group = 0;
        const uint32_t at = exits_[i].patchOffset;
        const uint8_t opcode = exits_[i].call ? 0xE8u : 0xE9u;  // call rel32 (CallLinked) or jmp rel32
        const bool patchable = at != Exit::kNoPatch &&
                               (exits_[i].conditional ? codeBase[at - 1] == 0x0Fu && (codeBase[at] & 0xF0u) == 0x80u : codeBase[at] == opcode);
        if (exits_[i].call) NoteLiftEntry(codeBase + exits_[i].retOffset, exits_[i].pc);
        if (patchable) {
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
    // The chunk being filled (null: none yet), in three parts each filled upwards (BlockLayout): hot code [0, hotEnd), far
    // code [hotEnd, farEnd) (stale stubs too), data [farEnd, size). A block whose parts do not all fit starts a new chunk; the
    // room left in the old one is not used again.
    static constexpr size_t kChunk = JitExecMemory::Arena::kDefaultChunkBytes;
    static constexpr size_t kHotEnd = kChunk / 32 * 13, kFarEnd = kChunk / 32 * 23;
    uint8_t* chunk = nullptr;
    size_t hot = 0, far = 0, data = 0;
    // The call sites' tables of translated targets (BlockEmitter::TargetTable): at the start of the first chunk's data.
    static constexpr size_t kTargetTableBytes = 2 * x64::BlockEmitter::kTargetTableEntries * sizeof(JitTargetCell);
    uint8_t* targetTables = nullptr;
    size_t used = 0;  // bytes of all parts of all blocks and stubs (the code limit's measure)
    // Room for hotBytes, farBytes and dataBytes in the parts of the chunk being filled, or a new chunk; false when no memory.
    bool Room(size_t hotBytes, size_t farBytes, size_t dataBytes) {
        if (chunk != nullptr && kHotEnd - hot >= hotBytes && kFarEnd - far >= farBytes && kChunk - data >= dataBytes) return true;
        if (hotBytes > kHotEnd || farBytes > kFarEnd - kHotEnd || dataBytes > kChunk - kFarEnd) return false;
        void* const fresh = arena.Allocate(kChunk, 4096);
        if (fresh == nullptr) return false;
        chunk = static_cast<uint8_t*>(fresh);
        hot = 0;
        far = kHotEnd;
        data = kFarEnd;
        if (targetTables == nullptr) {
            targetTables = chunk + data;
            JitTargetCell* const cells = reinterpret_cast<JitTargetCell*>(targetTables);
            for (uint32_t i = 0; i < 2 * x64::BlockEmitter::kTargetTableEntries; ++i)
                cells[i] = JitTargetCell{JitTargetCell::kNoKey, 0, 0, 0, nullptr, nullptr, nullptr};
            data += kTargetTableBytes;
        }
        SyncUnwind();
        faultChunk = x64::FaultChunkOpen(chunk, kChunk);  // the fault handler's table of the chunk's stores (FaultStores)
        if (faultChunk != nullptr) faultChunks.push_back(faultChunk);
        return true;
    }
    x64::FaultChunk* faultChunk = nullptr;  // the chunk being filled's
    std::vector<x64::FaultChunk*> faultChunks;
    void CloseFaultChunks() {
        for (x64::FaultChunk* c : faultChunks) x64::FaultChunkClose(c);
        faultChunks.clear();
        faultChunk = nullptr;
    }
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
    ~JitCodeCache() {  // deregister before arena unmaps chunks
        CloseFaultChunks();
        ehFrames.clear();
    }
#else
    void SyncUnwind() {}
    ~JitCodeCache() { CloseFaultChunks(); }
#endif
};

uint8_t* x64::BlockEmitter::TargetTable(bool tail) const {
    if (owner_ == nullptr) return nullptr;
    uint8_t* const tables = static_cast<JitCodeCache*>(owner_)->targetTables;
    if (tables == nullptr) return nullptr;
    return tables + (tail ? 0 : kTargetTableEntries * sizeof(JitTargetCell));
}

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
        e.MarkRegion(BlockEmitter::Region::Op, i);
        if (diag::ProbeAt(b.pcs[i])) {  // BRAWL_JIT_PROBE: log the state before this op
            e.FlushAllGprs();
            e.StorePc(i);
            e.a.lea(rax, e.a.ptr[kCtxBase - kCtxBias]);
            e.CallHelper(&diag::ProbeHit, rax, e.PairedMask());
        }
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
            // FP ops never read/write GPRs and keep both caches: their slow paths store and reload snapshots, and a helper
            // call in the main path flushes the volatile slots itself (jit_x64_fp.cpp).
        } else if (o.group[static_cast<unsigned>(kind)] == Group::Branch) {
            // Bc and fused compare+bc preserve the GPR cache on fall-through and flush dirty regs on their taken paths;
            // all other branches/calls flush before running. The FPR cache's dirty slots are stored before any branch
            // (the clean ones stay for the fall-through path).
            const bool keepGprs = kind == K::Bc || kind == K::CmpwiBc || kind == K::CmplwiBc || kind == K::CmpwBc ||
                                  kind == K::CmplwBc || kind == K::RlwinmRcBc;
            if (!keepGprs) e.FlushAllGprs();
            else e.FlushDirtyFprs();
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
    e.MarkRegion(BlockEmitter::Region::End, b.count);
    e.FlushAllGprs();
    e.index = b.count;
    e.ExitLinked(b.ops[b.count].imm);
}

JitCodeCache* CreateCache() {
    if (!FaultStoresInit()) return nullptr;  // the store view without its fault handler: no compiled code (jit_x64_mem.cpp)
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

size_t CodeBytes(const JitCodeCache& cache) { return cache.used; }

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
    // Room asked of each part: a worst case from the op count (kMaxBytesPerOp covers an op's main path and its cold code), and
    // for the data every slot, cell and cache a block can have. A part that outgrows its room throws (JitAsm's limit); the block
    // is then emitted again in a new chunk, where every part has all its room.
    const size_t code = 256 + static_cast<size_t>(b.count) * kMaxBytesPerOp;
    const size_t data = 64 + 8 * (1 + BlockEmitter::kMaxEpochSites) + BlockEmitter::kMaxExits * sizeof(JitLink) + sizeof(StaleCell) +
                        BlockEmitter::kMaxCells * kCellWays * sizeof(JitTargetCell) +
                        (BlockEmitter::kCacheEntries[0] + BlockEmitter::kCacheEntries[1]) * sizeof(JitLink) + 64 * BlockEmitter::kCaches +
                        16 * (BlockEmitter::kMaxConsts + 1);  // the float constants (Const128), 16-byte aligned
    for (int attempt = 0; attempt < 2; ++attempt) {
        const bool fresh = attempt != 0;
        if (fresh) cache.chunk = nullptr;  // a new chunk: the parts of the old one were too small for this block
        if (!cache.Room(code, code, data)) break;
        uint8_t* const memory = cache.chunk + cache.hot;
        const BlockLayout layout{JitCodeCache::kHotEnd - cache.hot, cache.far - cache.hot, JitCodeCache::kFarEnd - cache.hot,
                                 cache.data - cache.hot, JitCodeCache::kChunk - cache.hot};
        bool tooBig = false;
        try {
            BlockEmitter e(b, memory, layout.dataLimit, layout);
            e.SetOwner(&cache);
            if (!Opts().dumpDir.empty()) e.EnableRegions();
            e.EmitPrologue();
            EmitOps(e);
            e.Finish();
            // Its stores' faults must find their slow entries (jit_x64_mem.cpp) before the block can run.
            if (e.FaultSiteCount() != 0) {
                if (cache.faultChunk == nullptr) throw EmitError{"a fault site table"};
                FaultChunkAdd(cache.faultChunk, memory, e);
            }
            // The hot code of the next block starts on a 16-byte boundary (the fetch window of a jump to it).
            cache.hot += (e.HotBytes() + 15) & ~size_t{15};
            cache.far += (e.FarBytes() + 15) & ~size_t{15};
            cache.data += (e.DataBytes() + 7) & ~size_t{7};
            const size_t total = e.HotBytes() + e.FarBytes() + e.DataBytes();
            cache.used += total;
            b.jitChecked = e.Checked();
            b.jitLinked = e.Linked();
            b.jitEntry = e.Entry();
            b.jitEpoch = e.EpochRecord();
            b.jitGuarded = e.Guarded();
            if (b.jitGuarded) Bump(g_counters.guarded);
            ++cache.compiled;
            Bump(g_counters.compiled);
            diag::Add(diag::kJitCompiled);
            Bump(g_counters.codeBytes, total);
            Bump(g_counters.compiledOps, b.count);
            uint8_t* const farAt = memory + layout.farStart;
            PerfMapAdd(memory, e.HotBytes(), "jit", b.start);
            PerfMapAdd(farAt, e.FarBytes(), "jit", b.start);
            DumpBlock(b, memory, e.HotBytes());
            DumpBlockTable(e, memory, farAt);
            if (Opts().stats && (g_counters.compiled.load(std::memory_order_relaxed) & 0x3FFF) == 0) LogStats("running");
            return true;
        } catch (const Xbyak::Error& error) {
            tooBig = static_cast<int>(error) == Xbyak::ERR_CODE_IS_TOO_BIG;
            if (!tooBig) std::fprintf(stderr, "jit: block %08X not compiled: %s\n", b.start, error.what());
        } catch (const EmitError& error) {
            std::fprintf(stderr, "jit: block %08X not compiled: out of %s\n", b.start, error.what);
        } catch (...) {
        }
        if (!tooBig || fresh) break;
    }
    Bump(g_counters.compileFailed);
    diag::Add(diag::kJitCompileFailed);
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

// The stale stub of a block (see StaleCell): each guest word of the block compared in place with the word the block was decoded
// from (all its segments; the words as guest memory holds them), then the block's entry; any difference leaves through the plain
// stale exit. Null when it cannot be made. Only a failed mark test leads here, and only with the block's epoch current: a linked
// entry is reached only then (Block::jitLinked), and the full entry sends a stale epoch to the plain exit.
const void* MakeStaleStub(JitCodeCache& cache, const StaleCell& cell) {
    const Block& b = *cell.block;
    uint32_t words = 0;
    for (uint32_t s = 0; s < b.segments; ++s) words += b.segment[s].count;
    const size_t reserve = 256 + 24u * static_cast<size_t>(words);
    if (!cache.Room(0, reserve, 0)) return nullptr;
    uint8_t* const memory = cache.chunk + cache.far;  // far code: it is entered only after a failed mark test
    // BRAWL_JIT_STALE_FAULT=<n> (a test aid: the stub is wrong on purpose so that the tests can show they notice): 1 compares no
    // words, 3 compares the first segment only, 4 skips the first word of each segment, 5 never enters (2 was an epoch test the
    // stub no longer has).
    const char* const faultText = std::getenv("BRAWL_JIT_STALE_FAULT");
    const int fault = faultText != nullptr ? std::atoi(faultText) : 0;
    try {
        Xbyak::CodeGenerator g(reserve, memory);
        Xbyak::Label leave;
        if (diag::ChurnOn()) {  // BRAWL_JIT_CHURN
            g.mov(rax, reinterpret_cast<uint64_t>(diag::CounterCell(diag::kJitStubRuns)));
            g.inc(g.qword[rax]);
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
        if (diag::ChurnOn()) {
            g.mov(rax, reinterpret_cast<uint64_t>(diag::CounterCell(diag::kJitStubLeaves)));
            g.inc(g.qword[rax]);
        }
        JumpTo(g, cell.plain);
        g.ready();
        PerfMapAdd(memory, g.getSize(), "jit_stale", b.start);
        DumpCode("jit_stale", memory, g.getSize(), b.start);
        cache.far += (g.getSize() + 15) & ~size_t{15};
        cache.used += g.getSize();
        return memory;
    } catch (...) {
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
    slot.target = to.jitLinked;
    PatchDirectJump(slot, to.jitLinked);
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
    if (e.to == &to && e.target == to.jitLinked && e.key == to.start) return;
    if (diag::ChurnOn()) diag::AddKey(diag::kTCacheLink, (uint64_t{e.key} << 32) | to.start);
    Aim(e, to);
    Bump(g_counters.links);
    diag::Add(diag::kJitLinks);
}

void Link(JitLink& slot, Block& to) {
    if (to.jitLinked == nullptr) return;
    if (slot.group != 0) {
        LinkCached(slot, to);
        return;
    }
    if (slot.to == &to && slot.target == to.jitLinked) return;
    Aim(slot, to);
    Bump(g_counters.links);
    diag::Add(diag::kJitLinks);
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
        diag::Add(diag::kJitRetireSlots);
    }
    b.jitIncoming = nullptr;
    Bump(g_counters.retires);
    if (diag::ChurnOn()) {
        diag::Add(diag::kJitRetires);
        diag::AddKey(diag::kTRetire, b.start);
    }
}

// The slots aimed at b leave through their unlinked stubs (the dispatcher then finds b and refreshes it), staying on its list.
void Suspend(Block& b) {
    for (JitLink* slot = b.jitIncoming; slot != nullptr; slot = slot->next) {
        slot->target = slot->unlinked;
        PatchDirectJump(*slot, slot->unlinked);
    }
}

// b's epoch into every epoch test of its code (an aligned 4-byte store each: the code may be on the stack, never running on
// another thread).
void SetEpoch(Block& b) {
    const uint64_t* const record = static_cast<const uint64_t*>(b.jitEpoch);
    if (record == nullptr || g_jitEpochFault == 4) return;  // 4: a test aid, the copies left as they were
    for (uint64_t i = 0; i < record[0]; ++i)
        __atomic_store_n(reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(record[1 + i])), b.epoch, __ATOMIC_RELAXED);
}

// The slots on b's list aimed at its linked entry again, but call links (BlockEmitter::CallLinked: a `call rel32` patch site), which
// only their stubs aim (they test the callee's guard byte and the redirect bound, which no suspension knows about).
void Resume(Block& b) {
    if (b.jitLinked == nullptr) return;
    for (JitLink* slot = b.jitIncoming; slot != nullptr; slot = slot->next) {
        if (slot->patchSite != nullptr && *slot->patchSite == 0xE8u) continue;
        slot->target = b.jitLinked;
        PatchDirectJump(*slot, b.jitLinked);
    }
}

JitBackend g_backend{CreateCache, DestroyCache, Compile, Enter, Link, Retire, CodeBytes, nullptr, Suspend, Resume, SetEpoch};

}  // namespace

const void* x64::InstallStaleStub(StaleCell* cell) noexcept {
    try {
        if (cell->stubbed == 0) {
            cell->stubbed = 1;
            const void* const stub = cell->block->jitEntry != nullptr ? MakeStaleStub(*static_cast<JitCodeCache*>(cell->cache), *cell) : nullptr;
            if (stub != nullptr) {
                cell->target = stub;
                Bump(x64::g_counters.staleStubs);
                diag::Add(diag::kJitStaleStubs);
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

const void* x64::CellPromoteRoutine() { return g_shared.cellPromote; }

bool x64::SubFramesPossible() {
    const PatchedHooks& h = GetPatchedHooks();
    return h.lowestRedirect != nullptr && h.pendingFast != nullptr && h.anyDirty != nullptr && h.sync != nullptr;
}

const void* JitFastCallFor(const JitCallTarget* t, bool tail) noexcept {
    const x64::Shared& s = x64::g_shared;
    if (!x64::kDirectCalls || t == nullptr || !s.ok) return nullptr;
    if (!t->fastDirect) return s.callGuarded;
    if (t->argMask != 0) return s.callArgs;
    // a patched callee of a bctrl with a host call (jit_subcall tests the guard byte again at every call); a bctr's tail call keeps
    // lr in rbp across the call, which the callee's blocks do not keep: the wrapper
    if (!tail && t->patchedByte != nullptr && *t->patchedByte != 0 && x64::SubFramesPossible()) return s.subCall;
    return reinterpret_cast<const void*>(t->entry);
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
                       get(g_counters.staleStubFailures), get(g_counters.selfLoops), get(g_counters.guarded)};
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
