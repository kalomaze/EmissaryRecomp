// gecko_handler.cpp - the Gecko code handler (projects/brawl/gecko/codehandler.bin, entry 0x800018A8) as a native
// body. A mod's code list is walked in full at every hooked frame (Project+: ~2,900 codes, 1,226 of them C2 hooks and
// 260 string writes of ~12 KB, at each VBI in a match and at each AX frame in the menus), and nearly every write it
// makes stores the value already there. The translated handler sent each of those stores to a code page through the
// checked store path (Write32Slow, WriteScalar's compare) and copied each string a byte per iteration: 4.4% of P+'s
// game thread. This port runs the same instructions on the same registers, with two changes in how
// memory is touched: a store to RAM first compares, and skips the store when the bytes are already there (no dirty
// mark and no code-patch notice, as WriteScalar does for the checked path); a string write (06) or a memory copy
// (8A/8C) whose destination already holds its source compares once instead of looping. Every other store goes
// through the translated code's own store path.
//
// Exactness: the guest sees the same machine. Each block below is the handler's block at that address, instruction
// for instruction: the same GPRs, CR fields (SO copies included), XER[CA], f2/f3 and stack frame bytes, so a C0 code
// (which runs as guest code with the handler's whole register file, and may store it) and the caller see what the
// translated handler gives them. Code types Project+ does not use (A0-AE, CE and the other if-like C types, F2/F4,
// F6) hand the rest of the pass, with that exact state, to the interpreter at the handler's own instruction; the
// subroutine at 0x80001E88 (8A/8C) is called as the guest function it is. The wrapper func_800018A8 keeps its guest
// code guard, so a changed handler runs as guest code instead.
//
// The write plan: nearly all of a pass is runs of pure codes, whose effects (register writes and stores to plain RAM)
// are a function of the code stream bytes they read and of the key registers at the run's start (r15, ba, po, r7, the
// execution status and XER[SO]). The walk records such a run when it interprets it (every register assignment, CR field
// and CA write, every store and every code-stream byte read are noted), ending it at the first impure step (a load from
// memory outside the code stream, guest code, a store outside RAM, a float operation, the end of the list). At a later
// pass, a run whose start, key and read bytes match is replayed: the extents its stores leave are compared with memory
// and only the stores of those that differ are redone, then the registers it wrote get their exit values, and the walk
// goes on at the run's end with the registers, CR, XER and memory the step-by-step walk would have left, so guest code
// run by a later C0 code or hand-off sees the same machine. A run whose reads differ is dropped and recorded again
// (with a back-off per start). On rollback-tracked pages (lean memory) an extent is compared only after one of its
// 64-byte dirty marks was seen set since it last compared equal; the marks are noted before a consume clears them, and
// a restore or any other rewrite that does not mark makes the next replay compare everything.
// BRAWL_GECKO_PLAN=0: the plain walk (RunHandler<false>). BRAWL_GECKO_PLAN_MARKS=0: compare every extent at every
// replay. BRAWL_GECKO_PLAN_VERIFY=1: after each replay, interpret the run's steps from the pre-replay state and log
// any difference in registers, CR, XER or memory ([gecko-plan-verify]).
// BRAWL_HLE_VERIFY=1 (as hle/mtx.cpp): every pass runs the translated handler and this port from the same state and
// compares the register file and all of MEM1 and MEM2, and checks that every 64-byte block the port changed carries
// a dirty mark; the translated handler's results stay. BRAWL_GECKO_VERIFY_EVERY=N verifies every Nth pass only.
// BRAWL_GECKO_STATS=1: every 600 passes, what a pass does (codes, stores, stores that changed memory, string bytes,
// and the plan's steps interpreted, runs replayed, recorded and dropped, key misses and extents compared, trusted and
// rewritten).
// BRAWL_HLE_GECKO=0: the translated handler, for A/B measurements on one binary.

#include "abi_bridge.h"
#include "audio/ax_dsp.h"
#include "brawl_dirty_memory.h"
#include "memory.h"
#include "ppc_interpreter.h"
#include "runtime_log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

MKW_NATIVE_BODY(0x800018A8u, BrawlGecko_CodeHandler);

extern "C" void func_800018A8_body(CpuContext* ctx);
extern "C" void func_80001E88(CpuContext* ctx);

namespace {

using MemoryInline::FlatRead16;
using MemoryInline::FlatRead32;
using MemoryInline::FlatRead8;

constexpr uint32_t kMem1Size = 0x01800000u;
constexpr uint32_t kMem2Size = 0x04000000u;

// CR bit numbers (bit 0 is the most significant), as the handler's bt/bf operands name them.
constexpr int kLt0 = 0, kGt0 = 1, kEq0 = 2;
constexpr int kLt3 = 12, kGt3 = 13, kEq3 = 14;
constexpr int kLt4 = 16, kGt4 = 17, kEq4 = 18;
constexpr int kLt5 = 20, kGt5 = 21, kEq5 = 22;
constexpr int kLt6 = 24, kGt6 = 25, kEq6 = 26;
constexpr int kEq7 = 30;
constexpr uint32_t kCa = 0x20000000u;

const auto Enabled = [](const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
};
const bool g_originals = [] {
    const char* value = std::getenv("BRAWL_HLE_GECKO");
    return value != nullptr && value[0] == '0';
}();
const bool g_verify = Enabled("BRAWL_HLE_VERIFY");
const bool g_statsOn = Enabled("BRAWL_GECKO_STATS");

struct Stats {
    std::atomic<uint64_t> passes{0}, codes{0}, stores{0}, changed{0}, stringBytes{0}, stringsSkipped{0}, c0{0}, bails{0};
};
Stats g_stats;
inline void Count(std::atomic<uint64_t>& counter, uint64_t n = 1) {
    if (g_statsOn) counter.fetch_add(n, std::memory_order_relaxed);
}

// BRAWL_GECKO_LOG_CHANGES=N (with BRAWL_GECKO_STATS): pass N logs each store that changes memory, with its code.
const uint64_t g_logChangesPass = [] {
    const char* value = std::getenv("BRAWL_GECKO_LOG_CHANGES");
    return value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
}();
bool g_logChanges = false;
uint32_t g_logCode = 0;
int g_logLines = 0;
template <typename T>
[[gnu::noinline, gnu::cold]] void LogChange(uint32_t address, uint32_t value) {
    if (!g_logChanges || g_logLines++ >= 400) return;
    uint32_t old = 0;
    if constexpr (sizeof(T) == 1) old = FlatRead8(address);
    else if constexpr (sizeof(T) == 2) old = FlatRead16(address);
    else old = FlatRead32(address);
    RT_LOGF(RT_TAG_MOD, "[gecko] change: code %08X (%08X %08X) stores %zu byte(s) at %08X: %08X -> %08X\n", g_logCode,
            FlatRead32(g_logCode), FlatRead32(g_logCode + 4), sizeof(T), address, old,
            value & static_cast<uint32_t>((uint64_t{1} << (8 * sizeof(T))) - 1));
}

inline uint32_t Rotl(uint32_t value, uint32_t amount) {
    amount &= 31u;
    return amount == 0 ? value : (value << amount) | (value >> (32u - amount));
}
constexpr uint32_t Mask(int mb, int me) {
    return mb <= me ? (0xFFFFFFFFu >> mb) & (0xFFFFFFFFu << (31 - me))
                    : ~((0xFFFFFFFFu >> (me + 1)) & (0xFFFFFFFFu << (31 - (mb - 1))));
}
inline uint32_t Rlwinm(uint32_t value, int sh, int mb, int me) { return Rotl(value, static_cast<uint32_t>(sh)) & Mask(mb, me); }
inline uint32_t Rlwimi(uint32_t dest, uint32_t value, int sh, int mb, int me) {
    const uint32_t mask = Mask(mb, me);
    return (Rotl(value, static_cast<uint32_t>(sh)) & mask) | (dest & ~mask);
}
inline uint32_t Extsh(uint32_t value) { return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(value))); }

// Plain guest RAM: MEM1 or MEM2 through its cached or uncached view, the whole access inside it. A no-op compare is
// only made there; anything else (MMIO above all) stores through the checked path as the translation does.
inline bool InRam(uint32_t address, uint32_t size) {
    const uint32_t offset = address & 0x0FFFFFFFu;
    switch (address >> 28) {
    case 0x8: case 0xC: return offset + size <= kMem1Size && offset + size >= offset;
    case 0x9: case 0xD: return offset + size <= kMem2Size && offset + size >= offset;
    default: return false;
    }
}

// Inlined at each of the handler's stores (~3,900 a pass); each returns whether memory changed.
[[gnu::always_inline]] inline bool Store8(uint32_t address, uint32_t value) {
    Count(g_stats.stores);
    if (InRam(address, 1) && FlatRead8(address) == static_cast<uint8_t>(value)) return false;
    Count(g_stats.changed);
    if (g_logChanges) LogChange<uint8_t>(address, value);
    MemoryInline::FlatWrite8(address, static_cast<uint8_t>(value));
    return true;
}
[[gnu::always_inline]] inline bool Store16(uint32_t address, uint32_t value) {
    Count(g_stats.stores);
    if (InRam(address, 2) && FlatRead16(address) == static_cast<uint16_t>(value)) return false;
    Count(g_stats.changed);
    if (g_logChanges) LogChange<uint16_t>(address, value);
    MemoryInline::FlatWrite16(address, static_cast<uint16_t>(value));
    return true;
}
[[gnu::always_inline]] inline bool Store32(uint32_t address, uint32_t value) {
    Count(g_stats.stores);
    if (InRam(address, 4) && FlatRead32(address) == value) return false;
    Count(g_stats.changed);
    if (g_logChanges) LogChange<uint32_t>(address, value);
    MemoryInline::FlatWrite32(address, value);
    return true;
}

// The stores' targets are spread over the game's code and data (Project+: ~3,900 words over ~10 MB), mostly out of
// cache by the next pass, and each store's compare waits for its line. A cursor kPrefetchAhead codes ahead of r15
// (walked by line counts alone, ignoring conditions; reset whenever r15 jumps away from it) prefetches the targets of
// the write and C2 codes it passes with the current ba/po, so those misses overlap. No guest-visible effect.
constexpr int kPrefetchAhead = 24;
struct Prefetcher {
    uint32_t cursor = 0;
    int ahead = 0;
    void Run(uint32_t r15, uint32_t ba, uint32_t po) {
#if MKW_LEAN_MEMORY
        if (cursor < r15 || cursor - r15 > 0x10000u) {
            cursor = r15;
            ahead = 0;
        }
        for (int step = 0; step < 2 && ahead < kPrefetchAhead; ++step, ++ahead) {
            if (!InRam(cursor, 8)) return;
            const uint32_t hi = FlatRead32(cursor), lo = FlatRead32(cursor + 4);
            const uint32_t type = hi >> 24;
            uint32_t lines = 1;
            if (type < 0x20 || (type & 0xFE) == 0xC2) {  // writes (00-1E) and C2 hooks
                const uint32_t base = (type & 0x10) != 0 ? po : (ba & 0xFE000000u);
                __builtin_prefetch(MemoryInline::LeanHost(base + (hi & 0x01FFFFFFu)), 0);  // read: most stores only compare
            }
            switch (type & 0xEE) {
            case 0x06: lines = 1 + (lo + 7) / 8; break;  // string
            case 0x08: lines = 2; break;                  // serial
            default:
                if ((type & 0xFC) == 0xC0) lines = 1 + lo;  // C0/C2 bodies
            }
            if (lines > 0x2000u) return;
            cursor += 8 * lines;
        }
#else
        (void)r15; (void)ba; (void)po;
#endif
    }
    void Advanced() { ahead = ahead > 0 ? ahead - 1 : 0; }
};

// [dest, dest + size) already holds [source, source + size): a byte loop from one to the other changes nothing,
// whatever its direction and however the two overlap.
inline bool AlreadyCopied(uint32_t dest, uint32_t source, uint32_t size) {
    if (!InRam(dest, size) || !InRam(source, size)) return false;
    return std::memcmp(Memory::GetPointer(dest, size), Memory::GetPointer(source, size), size) == 0;
}

// The write plan (see the header comment): runs of pure codes recorded by the walk and replayed from memory
// comparisons. Everything here is in GeckoPlan so that the unity build cannot see a clash.
namespace GeckoPlan {

constexpr uint32_t kMaxSteps = 65536u;
constexpr uint32_t kMaxRunBytes = 4u << 20;
constexpr uint64_t kMaxTotalBytes = 64ull << 20;
constexpr size_t kMaxSlots = size_t{1} << 20;
constexpr uint32_t kMaxVariants = 8;
constexpr uint32_t kMaxBackoffExponent = 10;
constexpr uint32_t kMaxChain = 4096;  // consecutive replays at one boundary before a step is interpreted
constexpr uint32_t kSo = 0x80000000u;

// The plan is on (1), on without mark trust (2) or off (0). BRAWL_GECKO_PLAN=0 selects the plain walk.
std::atomic<int> g_mode{[] {
    const char* value = std::getenv("BRAWL_GECKO_PLAN");
    if (value != nullptr && value[0] == '0') return 0;
    const char* marks = std::getenv("BRAWL_GECKO_PLAN_MARKS");
    return marks != nullptr && marks[0] == '0' ? 2 : 1;
}()};
const bool g_verifyOn = Enabled("BRAWL_GECKO_PLAN_VERIFY");
constexpr size_t kCompareAhead = 16;  // extents: the compares' targets are spread over the game's code and data
bool g_marksOn = false;       // mark trust: needs the hooks and a mode of 1
bool g_hooksInstalled = false;
std::atomic<bool> g_inPass{false};
[[maybe_unused]] std::atomic<uint32_t> g_generation{1};  // bumped when a restore rewrites tracked memory without marking it

struct Key {
    uint32_t r6, r7, r8, r16, so;
};
inline bool SameKey(const Key& a, const Key& b) {
    return a.r6 == b.r6 && a.r7 == b.r7 && a.r8 == b.r8 && a.r16 == b.r16 && a.so == b.so;
}

struct Extent {
    uint32_t address, size, at;  // at: where the extent's bytes are in Run::bytes
};
struct StoreRec {
    uint32_t address, value, size;  // size 1, 2 or 4; value in the low bytes
};
struct ReadRec {
    uint32_t address, size, at;  // at: offset into the recorder's byte buffer
};
struct Owner {
    uint32_t mark, extent;  // a watched mark and the extent (reads, then writes) that covers it
};
struct Window {
    uint64_t mask;
    uint32_t at, first, end;  // 8 marks from `at`, the watched ones; owners [first, end) of those marks
};

struct Run {
    uint32_t start = 0, end = 0, steps = 0;
    Key key{};
    uint32_t gprWritten = 0, gpr[32] = {}, crWritten = 0, cr = 0;
    bool caWritten = false;
    uint32_t ca = 0;
    std::vector<Extent> reads, writes;
    std::vector<uint32_t> groupEnd;  // writes[j]'s stores are stores[groupEnd[j - 1], groupEnd[j])
    std::vector<StoreRec> stores;
    std::vector<uint8_t> bytes;  // the read extents' bytes, then the write extents' images
    uint64_t cost = 0, lastUse = 0;
    bool survived = false;  // replayed at least once
    // marks: the epoch and generation the classification and the trust belong to, a flag byte per extent (reads, then
    // writes; bit 0: always compared, bit 1: a watched mark was seen set since the last equal compare)
    uint32_t epoch = 0, generation = 0, nAlways = 0;
    bool allPending = true, anyPending = false;
    std::vector<uint8_t> flags;
    std::vector<Owner> owners;
    std::vector<Window> windows;
};

struct Slot {
    uint32_t address = 0;
    bool used = false;
    Run* variants[kMaxVariants] = {};
    uint8_t n = 0, exponent = 0;
    uint64_t retryAt = 0;
};

struct Counters {
    uint64_t passes = 0, steps = 0, replayed = 0, recorded = 0, dropped = 0, keyMisses = 0, compared = 0, trusted = 0,
             rewritten = 0, verifyChecks = 0, verifyMismatches = 0;
};

// Extents live in canonical space: every InRam address of one byte (its 0x8/0xC or 0x9/0xD view) maps to one value.
inline uint32_t Canon(uint32_t address) { return address & 0x3FFFFFFFu; }
inline const uint8_t* HostPtr(uint32_t canonical, [[maybe_unused]] uint32_t size) {
#if MKW_LEAN_MEMORY
    return MemoryInline::LeanHost(canonical);
#else
    return Memory::GetPointer(0x80000000u | canonical, size);
#endif
}
inline bool HostEqual(uint32_t address, const uint8_t* image, uint32_t size) {
    const uint8_t* host = HostPtr(address, size);
    if (size == 4) {
        uint32_t a, b;
        std::memcpy(&a, host, 4);
        std::memcpy(&b, image, 4);
        return a == b;
    }
    if (size == 8) {
        uint64_t a, b;
        std::memcpy(&a, host, 8);
        std::memcpy(&b, image, 8);
        return a == b;
    }
    return std::memcmp(host, image, size) == 0;
}

#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
// The mark windows of a run, as the interpreter's code cache scans them: sorted mark indices grouped into 8-byte loads
// with a byte mask. A hit flags the extents that watch a set mark.
inline void ScanWindows(Run& run) {
    const uint8_t* marks = BrawlDirtyMemory::leanBlocks;
    for (const Window& w : run.windows) {
        uint64_t v;
        std::memcpy(&v, marks + w.at, 8);
        if (__builtin_expect((v & w.mask) == 0, 1)) continue;
        for (uint32_t k = w.first; k < w.end; ++k) {
            const Owner& o = run.owners[k];
            if (marks[o.mark] == 0) continue;
            run.flags[o.extent] |= 2;
            run.anyPending = true;
        }
    }
}

// An extent whose every 64-byte line is on a rollback-tracked page is compared only after one of its marks was seen
// set; every other extent is compared at every replay.
inline void Classify(Run& run) {
    const size_t count = run.reads.size() + run.writes.size();
    run.flags.assign(count, 1);
    run.owners.clear();
    run.windows.clear();
    run.nAlways = static_cast<uint32_t>(count);
    if (!g_marksOn || BrawlDirtyMemory::leanBlocks == nullptr) return;
    const auto classify = [&](const Extent& e, size_t index) {
        const uint32_t first = (e.address & BrawlDirtyMemory::kLeanDirtyMask) >> 6;
        const uint32_t last = ((e.address + e.size - 1) & BrawlDirtyMemory::kLeanDirtyMask) >> 6;
        for (uint32_t page = first >> 10; page <= (last >> 10); ++page)
            if (!BrawlDirtyMemory::trackedPages[page & 0x3FFFu].load(std::memory_order_relaxed)) return;
        run.flags[index] = 0;
        --run.nAlways;
        for (uint32_t line = first; line <= last; ++line) run.owners.push_back(Owner{line, static_cast<uint32_t>(index)});
    };
    for (size_t i = 0; i < run.reads.size(); ++i) classify(run.reads[i], i);
    for (size_t j = 0; j < run.writes.size(); ++j) classify(run.writes[j], run.reads.size() + j);
    std::sort(run.owners.begin(), run.owners.end(), [](const Owner& a, const Owner& b) {
        return a.mark != b.mark ? a.mark < b.mark : a.extent < b.extent;
    });
    constexpr uint32_t kLast = static_cast<uint32_t>(BrawlDirtyMemory::kLeanDirtyBlocks) - 7u;
    for (uint32_t k = 0; k < run.owners.size(); ++k) {
        const uint32_t m = run.owners[k].mark;
        if (run.windows.empty() || m - run.windows.back().at > 7u)
            run.windows.push_back(Window{0, m < kLast ? m : kLast, k, k});
        Window& w = run.windows.back();
        uint8_t bytes[8];
        std::memcpy(bytes, &w.mask, 8);
        bytes[m - w.at] = 0xFF;
        std::memcpy(&w.mask, bytes, 8);
        w.end = k + 1;
    }
}
#endif

struct Saved {
    uint32_t r[32] = {};
    uint32_t gprWritten = 0, crWritten = 0, cr = 0, xer = 0, steps = 0;
    bool caWritten = false;
    size_t reads = 0, stores = 0;
};

enum class Mode : uint8_t { None, Record, Verify };

struct Planner {
    Mode mode = Mode::None;
    bool impure = false, capHit = false;
    uint32_t crWritten = 0;  // CR bits (whole fields) the walk wrote since the last start of recording
    bool caWritten = false;
    uint32_t listBase = 0;
    bool haveBase = false;
    uint64_t passNo = 0, useClock = 0, total = 0;
    std::vector<Slot> slots;  // open-addressing table keyed by a boundary's r15, a power of two long (or empty)
    size_t slotCount = 0;
    std::vector<std::unique_ptr<Run>> runs;
    Counters c;
    // recording
    uint32_t recIndex = 0, recStart = 0, recUsed = 0, recSteps = 0;
    Key recKey{};
    std::vector<ReadRec> recReads;
    std::vector<uint8_t> recBytes;
    std::vector<StoreRec> recStores;
    Saved saved;
    // verification
    Run* vRun = nullptr;
    uint32_t vSteps = 0, vExpected[32] = {}, vExpectedCr = 0, vExpectedXer = 0;
    int vLogged = 0;

    bool Recording() const { return mode == Mode::Record; }

    // What the steps report.
    void Impure() {
        if (mode != Mode::None) impure = true;
    }
    void ReadCode(uint32_t address, uint32_t length) {
        if (mode != Mode::Record) return;
        if (!InRam(address, length)) {
            impure = true;
            return;
        }
        if (length > kMaxRunBytes - recUsed) {
            capHit = true;
            return;
        }
        recReads.push_back(ReadRec{Canon(address), length, static_cast<uint32_t>(recBytes.size())});
        const uint8_t* host = HostPtr(Canon(address), length);
        recBytes.insert(recBytes.end(), host, host + length);
        recUsed += length;
    }
    void Stored(uint32_t address, uint32_t size, uint32_t value) {
        if (mode == Mode::None) return;
        if (mode == Mode::Verify) {
            if (!InRam(address, size)) impure = true;
            return;
        }
        if (!InRam(address, size)) {
            impure = true;
            return;
        }
        if (size > kMaxRunBytes - recUsed) {
            capHit = true;
            return;
        }
        recStores.push_back(StoreRec{address, size == 4 ? value : value & ((1u << (8 * size)) - 1u), size});
        recUsed += size;
    }

    // Pass bookkeeping.
    void ResetAll() {
        mode = Mode::None;
        slots.clear();
        slotCount = 0;
        runs.clear();
        total = 0;
    }
    void BeginPass(uint32_t base) {
        ++c.passes;
        ++passNo;
        mode = Mode::None;
        impure = capHit = false;
        if (!haveBase || base != listBase) {
            ResetAll();
            listBase = base;
            haveBase = true;
        }
    }
    static constexpr size_t kNone = ~size_t{0};
    static size_t HashOf(uint32_t address, size_t capacity) {
        return static_cast<size_t>((uint64_t{address >> 3} * 0x9E3779B97F4A7C15ull) >> 40) & (capacity - 1);
    }
    // The table index of the boundary at `address`, or kNone.
    size_t Find(uint32_t address) const {
        if (slots.empty()) return kNone;
        const size_t mask = slots.size() - 1;
        for (size_t i = HashOf(address, slots.size());; i = (i + 1) & mask) {
            if (!slots[i].used) return kNone;
            if (slots[i].address == address) return i;
        }
    }
    // Find, adding the slot (growing the table at half load) when there is none; kNone when the table is full.
    size_t FindOrAdd(uint32_t address) {
        const size_t found = Find(address);
        if (found != kNone) return found;
        if ((slotCount + 1) * 2 > slots.size()) {
            const size_t grown = slots.empty() ? 1024 : slots.size() * 2;
            if (grown > kMaxSlots) {
                if (slotCount + 1 > kMaxSlots / 4 * 3) return kNone;
            } else {
                std::vector<Slot> old(grown);
                old.swap(slots);
                for (const Slot& s : old) {
                    if (!s.used) continue;
                    size_t i = HashOf(s.address, slots.size());
                    while (slots[i].used) i = (i + 1) & (slots.size() - 1);
                    slots[i] = s;
                }
            }
        }
        size_t i = HashOf(address, slots.size());
        while (slots[i].used) i = (i + 1) & (slots.size() - 1);
        slots[i].used = true;
        slots[i].address = address;
        ++slotCount;
        return i;
    }
    void Backoff(size_t index) {
        Slot& s = slots[index];
        if (s.exponent < kMaxBackoffExponent) ++s.exponent;
        s.retryAt = passNo + (uint64_t{1} << s.exponent);
    }
    void RemoveRun(size_t index, Run* run) {
        Slot& s = slots[index];
        uint8_t kept = 0;
        for (uint8_t i = 0; i < s.n; ++i)
            if (s.variants[i] != run) s.variants[kept++] = s.variants[i];
        for (uint8_t i = kept; i < s.n; ++i) s.variants[i] = nullptr;
        s.n = kept;
        total -= run->cost;
        for (size_t i = 0; i < runs.size(); ++i) {
            if (runs[i].get() != run) continue;
            runs[i] = std::move(runs.back());
            runs.pop_back();
            return;
        }
    }

    // Verification of a replay (BRAWL_GECKO_PLAN_VERIFY).
    void VerifyLog(const char* line) {
        if (++vLogged <= 40) RT_LOGF(RT_TAG_MOD, "[gecko-plan-verify] %s\n", line);
    }
    void VerifyEnd(const uint32_t* r, uint32_t cr, uint32_t xer, bool early) {
        bool bad = false;
        char line[160];
        const unsigned start = vRun->start;
        if (early) {
            bad = true;
            std::snprintf(line, sizeof(line), "run %08X: the interpretation ended early or met an impure step", start);
            VerifyLog(line);
        } else {
            if (r[15] != vRun->end) {
                bad = true;
                std::snprintf(line, sizeof(line), "run %08X: end r15 %08X, expected %08X", start, r[15], vRun->end);
                VerifyLog(line);
            }
            for (int i = 0; i < 32; ++i) {
                if (r[i] == vExpected[i]) continue;
                bad = true;
                std::snprintf(line, sizeof(line), "run %08X: r%d replay %08X interpreted %08X", start, i, vExpected[i], r[i]);
                VerifyLog(line);
            }
            if (cr != vExpectedCr) {
                bad = true;
                std::snprintf(line, sizeof(line), "run %08X: cr replay %08X interpreted %08X", start, vExpectedCr, cr);
                VerifyLog(line);
            }
            if (xer != vExpectedXer) {
                bad = true;
                std::snprintf(line, sizeof(line), "run %08X: xer replay %08X interpreted %08X", start, vExpectedXer, xer);
                VerifyLog(line);
            }
            const auto holds = [&](const std::vector<Extent>& list, const char* what) {
                for (const Extent& e : list) {
                    const uint8_t* host = HostPtr(e.address, e.size);
                    const uint8_t* image = vRun->bytes.data() + e.at;
                    if (std::memcmp(host, image, e.size) == 0) continue;
                    uint32_t k = 0;
                    while (host[k] == image[k]) ++k;
                    bad = true;
                    std::snprintf(line, sizeof(line), "run %08X: %s extent differs at canonical %08X", start, what,
                                  e.address + k);
                    VerifyLog(line);
                }
            };
            holds(vRun->reads, "read");
            holds(vRun->writes, "write");
        }
        if (bad) ++c.verifyMismatches;
        mode = Mode::None;
    }

    // Recording.
    void StartRecording(uint32_t address, const uint32_t* r, uint32_t& written, uint32_t xer) {
        const size_t index = FindOrAdd(address);
        if (index == kNone) return;
        recReads.clear();
        recBytes.clear();
        recStores.clear();
        recUsed = recSteps = 0;
        impure = capHit = false;
        crWritten = 0;
        caWritten = false;
        written = 0;
        recIndex = static_cast<uint32_t>(index);
        recStart = r[15];
        recKey = Key{r[6], r[7], r[8], r[16], xer & kSo};
        saved.steps = 0;
        mode = Mode::Record;
    }
    void SaveCandidate(const uint32_t* r, uint32_t written, uint32_t cr, uint32_t xer) {
        std::memcpy(saved.r, r, sizeof(saved.r));
        saved.gprWritten = written;
        saved.crWritten = crWritten;
        saved.caWritten = caWritten;
        saved.cr = cr;
        saved.xer = xer;
        saved.steps = ++recSteps;
        saved.reads = recReads.size();
        saved.stores = recStores.size();
    }
    void FinishRecording() {
        mode = Mode::None;
        if (!Finalize()) Backoff(recIndex);
    }

    // Builds the run for the saved boundary from the recorded data alone. False when there is none to keep.
    bool Finalize() {
        const Saved& s = saved;
        if (s.steps < 2) return false;
        auto run = std::make_unique<Run>();
        run->start = recStart;
        run->end = s.r[15];
        run->steps = s.steps;
        run->key = recKey;
        run->gprWritten = s.gprWritten;
        std::memcpy(run->gpr, s.r, sizeof(run->gpr));
        run->crWritten = s.crWritten;
        run->cr = s.cr;
        run->caWritten = s.caWritten;
        run->ca = s.xer & kCa;

        // Reads: sorted, overlapping and contiguous ones merged; two reads that disagree on a byte discard the run.
        std::vector<uint32_t> order(s.reads);
        for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return recReads[a].address != recReads[b].address ? recReads[a].address < recReads[b].address : a < b;
        });
        for (const uint32_t index : order) {
            const ReadRec& rd = recReads[index];
            const uint8_t* src = recBytes.data() + rd.at;
            if (!run->reads.empty()) {
                Extent& cur = run->reads.back();
                const uint32_t curEnd = cur.address + cur.size;
                if (rd.address <= curEnd) {
                    const uint32_t readEnd = rd.address + rd.size;
                    const uint32_t overlap = rd.address < curEnd ? std::min(curEnd, readEnd) - rd.address : 0;
                    if (overlap != 0 &&
                        std::memcmp(run->bytes.data() + cur.at + (rd.address - cur.address), src, overlap) != 0)
                        return false;
                    if (readEnd > curEnd) {
                        run->bytes.insert(run->bytes.end(), src + overlap, src + rd.size);
                        cur.size = readEnd - cur.address;
                    }
                    continue;
                }
            }
            run->reads.push_back(Extent{rd.address, rd.size, static_cast<uint32_t>(run->bytes.size())});
            run->bytes.insert(run->bytes.end(), src, src + rd.size);
        }

        // Writes: the stores' byte ranges merged into extents, each with the image its stores leave.
        const size_t nStores = s.stores;
        order.assign(nStores, 0);
        for (uint32_t i = 0; i < nStores; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            const uint32_t x = Canon(recStores[a].address), y = Canon(recStores[b].address);
            return x != y ? x < y : a < b;
        });
        for (const uint32_t index : order) {
            const StoreRec& st = recStores[index];
            const uint32_t at = Canon(st.address);
            if (!run->writes.empty() && at <= run->writes.back().address + run->writes.back().size) {
                Extent& cur = run->writes.back();
                cur.size = std::max(cur.address + cur.size, at + st.size) - cur.address;
            } else {
                run->writes.push_back(Extent{at, st.size, 0});
            }
        }
        for (Extent& e : run->writes) {
            e.at = static_cast<uint32_t>(run->bytes.size());
            run->bytes.resize(run->bytes.size() + e.size, 0);
        }
        std::vector<uint32_t> extentOf(nStores);
        std::vector<uint32_t> counts(run->writes.size() + 1, 0);
        for (size_t i = 0; i < nStores; ++i) {
            const StoreRec& st = recStores[i];
            const uint32_t at = Canon(st.address);
            size_t lo = 0, hi = run->writes.size();  // the last extent starting at or below the store
            while (hi - lo > 1) {
                const size_t mid = (lo + hi) / 2;
                if (run->writes[mid].address <= at) lo = mid;
                else hi = mid;
            }
            extentOf[i] = static_cast<uint32_t>(lo);
            ++counts[lo + 1];
            uint8_t* image = run->bytes.data() + run->writes[lo].at + (at - run->writes[lo].address);
            for (uint32_t k = 0; k < st.size; ++k) image[k] = static_cast<uint8_t>(st.value >> (8 * (st.size - 1 - k)));
        }
        for (size_t j = 1; j < counts.size(); ++j) counts[j] += counts[j - 1];
        run->groupEnd.assign(counts.begin() + 1, counts.end());
        run->stores.resize(nStores);
        for (size_t i = 0; i < nStores; ++i) run->stores[counts[extentOf[i]]++] = recStores[i];

        // A write that overlaps a read must leave the bytes the read saw.
        for (size_t a = 0, b = 0; a < run->reads.size() && b < run->writes.size();) {
            const Extent& rd = run->reads[a];
            const Extent& wr = run->writes[b];
            const uint32_t lo = std::max(rd.address, wr.address);
            const uint32_t rdEnd = rd.address + rd.size, wrEnd = wr.address + wr.size;
            const uint32_t hi = std::min(rdEnd, wrEnd);
            if (lo < hi && std::memcmp(run->bytes.data() + rd.at + (lo - rd.address),
                                       run->bytes.data() + wr.at + (lo - wr.address), hi - lo) != 0)
                return false;
            if (rdEnd < wrEnd) ++a;
            else ++b;
        }

        run->cost = sizeof(Run) + run->bytes.size() + run->stores.size() * sizeof(StoreRec) +
                    (run->reads.size() + run->writes.size()) * (sizeof(Extent) + 8);
        run->lastUse = ++useClock;
        Slot& slot = slots[recIndex];
        for (uint8_t i = 0; i < slot.n; ++i)  // the same key again: the new run replaces it
            if (SameKey(slot.variants[i]->key, run->key)) {
                RemoveRun(recIndex, slot.variants[i]);
                break;
            }
        if (slot.n == kMaxVariants) {
            Run* oldest = slot.variants[0];
            for (uint8_t i = 1; i < slot.n; ++i)
                if (slot.variants[i]->lastUse < oldest->lastUse) oldest = slot.variants[i];
            RemoveRun(recIndex, oldest);
        }
        slot.variants[slot.n++] = run.get();
        total += run->cost;
        runs.push_back(std::move(run));
        ++c.recorded;
        if (total > kMaxTotalBytes) ResetAll();
        return true;
    }

    // Replays the run if every read extent still holds its bytes: the write extents are put right (their stores redone
    // where they differ) and the exit registers applied. False, with nothing changed, when a read differs.
    bool Replay(Run& run, uint32_t* r, uint32_t& cr, uint32_t& xer) {
        bool all = true;
        [[maybe_unused]] const uint8_t* flags = nullptr;
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
        const uint32_t epoch = BrawlDirtyMemory::markEpoch.load(std::memory_order_acquire);
        const uint32_t generation = g_generation.load(std::memory_order_relaxed);
        if (run.epoch != epoch) {
            Classify(run);
            run.epoch = epoch;
            run.generation = generation;
            run.allPending = true;
        } else if (run.generation != generation) {
            run.generation = generation;
            run.allPending = true;
        }
        ScanWindows(run);  // here, not at the pass start: an earlier step of this pass may have written a target
        all = run.allPending;
        flags = run.flags.data();
        const bool nothing = !all && !run.anyPending && run.nAlways == 0;
#else
        const bool nothing = false;
#endif
        const size_t nr = run.reads.size(), nw = run.writes.size();
        if (nothing) {
            c.trusted += nr + nw;
        } else {
            for (size_t i = 0; i < nr; ++i) {
                const Extent& e = run.reads[i];
                if (i + kCompareAhead < nr) __builtin_prefetch(HostPtr(run.reads[i + kCompareAhead].address, 1), 0);
                if (!all && flags[i] == 0) {
                    ++c.trusted;
                    continue;
                }
                ++c.compared;
                if (!HostEqual(e.address, run.bytes.data() + e.at, e.size)) return false;
            }
            for (size_t j = 0; j < nw; ++j) {
                const Extent& e = run.writes[j];
                if (j + kCompareAhead < nw) __builtin_prefetch(HostPtr(run.writes[j + kCompareAhead].address, 1), 0);
                if (!all && flags[nr + j] == 0) {
                    ++c.trusted;
                    continue;
                }
                ++c.compared;
                if (HostEqual(e.address, run.bytes.data() + e.at, e.size)) continue;
                ++c.rewritten;
                for (uint32_t k = j == 0 ? 0 : run.groupEnd[j - 1]; k < run.groupEnd[j]; ++k) {
                    const StoreRec& st = run.stores[k];
                    if (st.size == 4) Store32(st.address, st.value);
                    else if (st.size == 2) Store16(st.address, st.value);
                    else Store8(st.address, st.value);
                }
            }
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
            if (run.anyPending || run.allPending) {
                for (uint8_t& f : run.flags) f &= 1;
                run.anyPending = run.allPending = false;
            }
#endif
        }
        for (uint32_t m = run.gprWritten; m != 0; m &= m - 1) {
            const int i = __builtin_ctz(m);
            r[i] = run.gpr[i];
        }
        cr = (cr & ~run.crWritten) | (run.cr & run.crWritten);
        if (run.caWritten) xer = (xer & ~kCa) | run.ca;
        return true;
    }

    // The top of the per-code loop: close the step just finished, then replay runs from here while there are any.
    void Boundary(uint32_t* r, uint32_t& written, uint32_t& cr, uint32_t& xer) {
        if (mode == Mode::Record) {
            if (impure || capHit || recSteps + 1 > kMaxSteps) FinishRecording();
            else SaveCandidate(r, written, cr, xer);
        }
        if (mode == Mode::Verify) {
            ++vSteps;
            if (impure) VerifyEnd(r, cr, xer, true);
            else if (vSteps >= vRun->steps) VerifyEnd(r, cr, xer, false);
        }
        if (mode == Mode::None) {
            for (uint32_t chain = 0; chain < kMaxChain; ++chain) {
                const uint32_t address = r[15];
                if ((address & 7u) != 0 || !InRam(address, 8)) break;
                const size_t index = Find(address);
                if (index != kNone && slots[index].n != 0) {
                    Slot& slot = slots[index];
                    const Key key{r[6], r[7], r[8], r[16], xer & kSo};
                    Run* run = nullptr;
                    for (uint8_t i = 0; i < slot.n; ++i)
                        if (SameKey(slot.variants[i]->key, key)) {
                            run = slot.variants[i];
                            break;
                        }
                    if (run == nullptr) {
                        ++c.keyMisses;
                    } else {
                        uint32_t before[32], beforeCr = cr, beforeXer = xer;
                        if (g_verifyOn) std::memcpy(before, r, sizeof(before));
                        if (Replay(*run, r, cr, xer)) {
                            ++c.replayed;
                            run->lastUse = ++useClock;
                            if (!run->survived) {
                                run->survived = true;
                                slot.exponent = 0;
                                slot.retryAt = 0;
                            }
                            if (g_verifyOn) {  // interpret the same steps from the same state, then compare
                                ++c.verifyChecks;
                                std::memcpy(vExpected, r, sizeof(vExpected));
                                vExpectedCr = cr;
                                vExpectedXer = xer;
                                std::memcpy(r, before, sizeof(before));
                                cr = beforeCr;
                                xer = beforeXer;
                                vRun = run;
                                vSteps = 0;
                                impure = false;
                                mode = Mode::Verify;
                                break;
                            }
                            continue;
                        }
                        ++c.dropped;
                        const bool survived = run->survived;
                        RemoveRun(index, run);
                        if (survived) {
                            slot.exponent = 0;
                            slot.retryAt = 0;
                        } else {
                            Backoff(index);
                        }
                    }
                }
                if (index == kNone || passNo >= slots[index].retryAt) StartRecording(address, r, written, xer);
                break;
            }
        }
        ++c.steps;
    }

    // The pass ends here (F0, the interpreter hand-off): a recording closes with its last boundary.
    void EndPass() {
        if (mode == Mode::Record) FinishRecording();
        else if (mode == Mode::Verify) VerifyEnd(nullptr, 0, 0, true);
    }
};

Planner g_plan;

#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
// BrawlDirtyMemory::planConsumingMarks: note the set marks of every run before a consume clears them.
void OnConsumingMarks() {
    if (g_inPass.load(std::memory_order_acquire)) {
        g_generation.fetch_add(1, std::memory_order_acq_rel);
        return;
    }
    const uint32_t epoch = BrawlDirtyMemory::markEpoch.load(std::memory_order_acquire);
    for (const auto& run : g_plan.runs)
        if (run->epoch == epoch) ScanWindows(*run);
}
void OnRestoredMarks() { g_generation.fetch_add(1, std::memory_order_acq_rel); }
#endif
void InstallHooks() {
#if MKW_LEAN_MEMORY && MKW_LEAN_DIRTY_MARKS
    BrawlDirtyMemory::planConsumingMarks = &OnConsumingMarks;
    BrawlDirtyMemory::planRestoredMarks = &OnRestoredMarks;
    g_hooksInstalled = true;
#endif
    g_marksOn = g_hooksInstalled && g_mode.load() == 1;
}
[[maybe_unused]] const bool g_installed = [] {
    if (g_mode.load() != 0) InstallHooks();
    return true;
}();

// The register file of the plan-mode walk: assigning through a register records that it was assigned.
struct TrackedRegs {
    uint32_t v[32];
    uint32_t written = 0;
    struct Ref {
        TrackedRegs* regs;
        int index;
        Ref(TrackedRegs* r, int i) : regs(r), index(i) {}
        Ref(const Ref&) = default;
        operator uint32_t() const { return regs->v[index]; }
        Ref& operator=(uint32_t value) {
            regs->v[index] = value;
            regs->written |= 1u << index;
            return *this;
        }
        Ref& operator=(const Ref& other) { return *this = static_cast<uint32_t>(other); }
        Ref& operator+=(uint32_t x) { return *this = regs->v[index] + x; }
        Ref& operator-=(uint32_t x) { return *this = regs->v[index] - x; }
        Ref& operator&=(uint32_t x) { return *this = regs->v[index] & x; }
        Ref& operator|=(uint32_t x) { return *this = regs->v[index] | x; }
        Ref& operator^=(uint32_t x) { return *this = regs->v[index] ^ x; }
        Ref& operator>>=(uint32_t x) { return *this = regs->v[index] >> x; }
        Ref& operator<<=(uint32_t x) { return *this = regs->v[index] << x; }
    };
    Ref operator[](int index) { return Ref(this, index); }
};
struct PlainRegs {
    uint32_t v[32];
    uint32_t& operator[](int index) { return v[index]; }
};

}  // namespace GeckoPlan

// Plan = true is the walk with the write plan (it records and replays runs); Plan = false is the plain walk.
template <bool Plan>
void RunHandler(CpuContext* ctx) {
    using Regs = std::conditional_t<Plan, GeckoPlan::TrackedRegs, GeckoPlan::PlainRegs>;
    Regs r;
    [[maybe_unused]] GeckoPlan::Planner& plan = GeckoPlan::g_plan;
    std::memcpy(r.v, ctx->gpr, sizeof(r.v));
    uint32_t cr = PpcGetCr(ctx), xer = ctx->xer, ctr = ctx->ctr;
    PPC_FPR f2 = ctx->fpr[2], f3 = ctx->fpr[3];
    Prefetcher prefetch;

    const auto bt = [&](int bit) { return (cr & (0x80000000u >> bit)) != 0; };
    const auto cmpw = [&](int field, uint32_t a, uint32_t b) {
        if constexpr (Plan) plan.crWritten |= 0xFu << (28 - 4 * field);
        SetCRResident(cr, xer, field, static_cast<int32_t>(a), static_cast<int32_t>(b));
    };
    const auto cmplw = [&](int field, uint32_t a, uint32_t b) {
        if constexpr (Plan) plan.crWritten |= 0xFu << (28 - 4 * field);
        SetCRResident(cr, xer, field, a, b);
    };
    const auto rec = [&](uint32_t value) {  // a record form's CR0
        if constexpr (Plan) plan.crWritten |= 0xF0000000u;
        SetCRResident(cr, xer, 0, static_cast<int32_t>(value), 0);
        return value;
    };
    const auto addic = [&](uint32_t value, int32_t imm) {
        if constexpr (Plan) plan.caWritten = true;
        const uint64_t sum = uint64_t{value} + static_cast<uint32_t>(imm);
        xer = (xer & ~kCa) | ((sum >> 32) != 0 ? kCa : 0u);
        return static_cast<uint32_t>(sum);
    };
    const auto writeBack = [&] {
        std::memcpy(ctx->gpr, r.v, sizeof(r.v));
        PpcSetCr(ctx, cr);
        ctx->xer = xer;
        ctx->ctr = ctr;
        ctx->fpr[2] = f2;
        ctx->fpr[3] = f3;
    };
    const auto readBack = [&] {
        std::memcpy(r.v, ctx->gpr, sizeof(r.v));
        cr = PpcGetCr(ctx);
        xer = ctx->xer;
        ctr = ctx->ctr;
        f2 = ctx->fpr[2];
        f3 = ctx->fpr[3];
    };
    // What the walk tells the plan: a step that is not a function of the code stream and the key registers, the
    // code-stream bytes a step depends on, and its stores (which also report whether memory changed).
    const auto impure = [&] {
        if constexpr (Plan) plan.Impure();
    };
    const auto readCode = [&]([[maybe_unused]] uint32_t address, [[maybe_unused]] uint32_t length) {
        if constexpr (Plan) plan.ReadCode(address, length);
    };
    const auto recordingNow = [&]() -> bool {
        if constexpr (Plan) return plan.Recording();
        else return false;
    };
    const auto st8 = [&](uint32_t address, uint32_t value) {
        Store8(address, value);
        if constexpr (Plan) plan.Stored(address, 1, value);
    };
    const auto st16 = [&](uint32_t address, uint32_t value) {
        Store16(address, value);
        if constexpr (Plan) plan.Stored(address, 2, value);
    };
    const auto st32 = [&](uint32_t address, uint32_t value) {
        Store32(address, value);
        if constexpr (Plan) plan.Stored(address, 4, value);
    };
    // The rest of the pass in the interpreter, from the handler's instruction at pc with this state; it returns at the
    // handler's own blr, to the return address the frame holds.
    const auto bail = [&](uint32_t pc) {
        Count(g_stats.bails);
        if constexpr (Plan) {
            plan.Impure();
            plan.EndPass();
        }
        writeBack();
        ctx->lr = FlatRead32(r[1] + 172);
        PpcInterpreter::Run(ctx, pc);
    };

    // 0x800018A8: the frame, the caller's registers into it.
    {
        const uint32_t frame = r[1] - 168;
        MemoryInline::FlatWrite32(frame, r[1]);
        r[1] = frame;
    }
    MemoryInline::FlatWrite32(r[1] + 8, r[0]);
    r[0] = ctx->lr;
    MemoryInline::FlatWrite32(r[1] + 172, r[0]);
    r[0] = cr;
    MemoryInline::FlatWrite32(r[1] + 12, r[0]);
    r[0] = ctr;
    MemoryInline::FlatWrite32(r[1] + 16, r[0]);
    r[0] = xer;
    MemoryInline::FlatWrite32(r[1] + 20, r[0]);
    for (int i = 3; i < 32; ++i) MemoryInline::FlatWrite32(r[1] + 24 + 4 * static_cast<uint32_t>(i - 3), r[i]);
    r[25] = ctx->msr;
    r[26] = r[25] | 0x2000u;
    r[26] = rec(r[26] & 0xF9FFu);
    // 0x800018E0: mtmsr r26 is a nop (README)
    MemoryInline::FlatWriteFloat64(r[1] + 152, f2.d);
    MemoryInline::FlatWriteFloat64(r[1] + 160, f3.d);
    r[31] = 0x80000000u;
    r[20] = 0xCC000000u;
    r[28] = 0;
    r[21] = r[28] | 0xFFu;
    r[29] = ctx->lr;
    r[15] = FlatRead32(r[31] + 0x1804u);  // the code list (README: from 0x80001804)
    if constexpr (Plan) plan.BeginPass(r[15]);
    r[7] = r[31] | 0x1808u;               // Gecko registers gr0-gr15, then the blocks
    r[6] = 0x80000000u;                   // ba
    r[16] = r[6];                         // po
    r[8] = 0;                             // the execution status stack; bit 0 set: codes are skipped
    r[3] = 0x00D0C0DEu;
    r[4] = FlatRead32(r[15]);
    cmpw(0, r[3], r[4]);
    if (!bt(kEq0)) goto L_1944;
    r[4] = FlatRead32(r[15] + 4);
    cmpw(0, r[3], r[4]);
    if (!bt(kEq0)) goto L_1944;
    r[15] += 8;

L_198C:  // the next code
    if constexpr (Plan) plan.Boundary(r.v, r.written, cr, xer);
    Count(g_stats.codes);
    if (g_logChanges) g_logCode = r[15];
    prefetch.Advanced();
    if constexpr (Plan) {
        if (plan.Recording()) prefetch.Run(r[15], r[6], r[16]);  // otherwise the step is an isolated dynamic code
    } else {
        prefetch.Run(r[15], r[6], r[16]);
    }
    readCode(r[15], 8);
    r[3] = FlatRead32(r[15]);
    r[4] = FlatRead32(r[15] + 4);
    r[15] += 8;
    r[9] = rec(r[8] & 1u);
    cmpw(7, r[9], 0);  // cr7.eq: codes execute
    r[9] = 0;
    r[10] = r[3] >> 29;
    r[5] = Rlwinm(r[3], 7, 29, 31);
    r[11] = rec(r[3] & 0x10000000u);
    r[3] &= 0x01FFFFFFu;
    r[12] = bt(kEq0) ? (r[6] & 0xFE000000u) : r[16];
    cmpw(4, r[5], 0);
    cmpw(0, r[10], 1);
    if (bt(kLt0)) goto L_19F8;
    if (bt(kEq0)) goto L_1AB4;
    cmpw(0, r[10], 3);
    if (bt(kLt0)) goto L_1B84;
    if (bt(kEq0)) goto L_1C2C;
    cmpw(0, r[10], 5);
    if (bt(kLt0)) goto L_1CB8;
    if (bt(kEq0)) return bail(0x80001EC8u);  // A0-AE
    cmpw(0, r[10], 7);
    if (bt(kLt0)) goto L_1EFC;
    goto L_1FE4;

L_19F8:  // 00-0E: writes
    r[12] += r[3];
    cmpw(0, r[5], 3);
    if (bt(kEq0)) goto L_1A48;
    if (bt(kGt0)) goto L_1A64;
    if (!bt(kEq7)) goto L_198C;
    cmpw(4, r[5], 1);
    if (bt(kGt4)) goto L_1A3C;
    r[10] = r[4] >> 16;
L_1A18:
    if (bt(kEq4)) {
        st16(r[9] + r[12], r[4]);
        r[9] += 2;
    } else {
        st8(r[9] + r[12], r[4]);
        r[9] += 1;
    }
    r[10] = rec(addic(r[10], -1));
    if (!bt(kLt0)) goto L_1A18;
    goto L_198C;
L_1A3C:
    r[12] &= ~3u;
    st32(r[12], r[4]);
    goto L_198C;

L_1A48:  // 06: string write, copied from its last byte down
    r[9] = r[4];
    if (!bt(kEq7)) goto L_1F14;
    if (static_cast<int32_t>(r[9] - 1) >= 0) readCode(r[15], r[9]);
    if (static_cast<int32_t>(r[9] - 1) >= 0 && !recordingNow() && AlreadyCopied(r[12], r[15], r[9])) {
        // As the loop leaves it after r9 stores of the bytes already there: r5 the first byte, r9 = -1, CA clear.
        Count(g_stats.stringsSkipped);
        Count(g_stats.stringBytes, r[9]);
        r[5] = FlatRead8(r[15]);
        r[9] = rec(addic(0, -1));
        goto L_1F14;
    }
L_1A50:
    r[9] = rec(addic(r[9], -1));
    if (bt(kLt0)) goto L_1F14;
    Count(g_stats.stringBytes);
    r[5] = FlatRead8(r[9] + r[15]);
    st8(r[9] + r[12], r[5]);
    goto L_1A50;

L_1A64:  // 08: serial write
    r[15] += 8;
    if (!bt(kEq7)) goto L_198C;
    readCode(r[15] - 8, 8);
    r[5] = FlatRead32(r[15] - 8);
    r[11] = FlatRead32(r[15] - 4);
    r[17] = r[5] & 0xFFFFu;
    r[10] = Rlwinm(r[5], 16, 20, 31);
    r[5] >>= 28;
L_1A80:
    cmpw(5, r[5], 1);
    if (bt(kEq5)) st16(r[9] + r[12], r[4]);
    else if (bt(kGt5)) st32(r[9] + r[12], r[4]);
    else st8(r[9] + r[12], r[4]);
    r[4] += r[11];
    r[9] += r[17];
    r[10] = rec(addic(r[10], -1));
    if (!bt(kLt0)) goto L_1A80;
    goto L_198C;

L_1AB4:  // 20-2E: if codes (an endif first when the address ends in 1)
    r[9] = rec(r[3] & 1u);
    if (!bt(kEq0)) {
        r[8] >>= 1;
        r[9] = rec(r[8] & 1u);
        cmpw(7, r[9], 0);
    }
    cmpw(5, r[5], 4);
    cmpw(3, r[10], 5);
    r[8] = Rlwimi(r[8], r[8], 1, 0, 30);
    if (!bt(kEq7)) goto L_1B4C;
    impure();  // an executed if code reads its operand from memory
    if (bt(kGt3)) return bail(0x80001F90u);  // CE's range check (reached from the C types, which bail first)
    r[12] += r[3];
    if (bt(kLt3)) {
        if (bt(kLt5)) {  // 32 bits
            r[12] &= ~3u;
            r[11] = FlatRead32(r[12]);
            goto L_1B14;
        }
        r[12] &= ~1u;  // 16 bits under a mask
        r[11] = FlatRead16(r[12]);
    } else if (bt(kLt5)) {
        goto L_1B14;
    }
    r[9] = ~r[4];
    r[9] >>= 16;
    r[11] &= r[9];
    r[4] &= 0xFFFFu;
L_1B14:
    cmplw(6, r[11], r[4]);
    r[9] = rec(r[5] & 3u);
    if (bt(kEq0)) {  // equal
        if (bt(kEq6)) goto L_1B4C;
        goto L_1B48;
    }
    cmpw(0, r[9], 2);
    if (bt(kEq0)) {  // greater
        if (bt(kGt6)) goto L_1B4C;
        goto L_1B48;
    }
    if (bt(kGt0)) {  // lower
        if (bt(kLt6)) goto L_1B4C;
        goto L_1B48;
    }
    if (!bt(kEq6)) goto L_1B4C;  // not equal
L_1B48:
    r[8] |= 1u;
L_1B4C:
    if (!bt(kEq3)) goto L_198C;
    return bail(0x80001B50u);  // the A codes' counter (reached from A0-AE only, which bail first)

L_1B84:  // 40-4E: ba / po
    if (!bt(kEq7)) goto L_198C;
    r[9] = Rlwinm(r[3], 2, 26, 29);
    r[14] = Rlwinm(r[3], 16, 31, 31);
    cmpw(3, r[14], 0);
    cmpw(4, r[5], 4);
    r[14] = rec(r[5] & 3u);
    cmpw(5, r[14], 2);
    if (bt(kLt5)) goto L_1BB4;
    if (bt(kEq5)) goto L_1BF4;
    r[4] = Extsh(r[3]);
    r[4] += r[15];
    goto L_1C18;
L_1BB4:
    r[5] = rec(Rlwinm(r[3], 20, 31, 31));
    if (!bt(kEq0)) {
        impure();
        r[9] = FlatRead32(r[7] + r[9]);
        r[4] += r[9];
    }
    if (!bt(kEq3)) r[4] = r[12] + r[4];
    cmpw(5, r[14], 1);
    if (!bt(kEq5)) {
        impure();
        r[4] = FlatRead32(r[4]);
    }
    r[3] = rec(Rlwinm(r[3], 12, 31, 31));
    if (bt(kEq0)) goto L_1C18;
    r[4] += static_cast<uint32_t>(bt(kLt4) ? r[6] : r[16]);
    goto L_1C18;
L_1BF4:
    r[5] = rec(Rlwinm(r[3], 20, 31, 31));
    if (!bt(kEq0)) {
        impure();
        r[9] = FlatRead32(r[7] + r[9]);
        r[4] += r[9];
    }
    st32(r[12] + r[4], bt(kLt4) ? r[6] : r[16]);
    goto L_198C;
L_1C18:
    if (bt(kLt4)) r[6] = r[4];
    else r[16] = r[4];
    goto L_198C;

L_1C2C:  // 60-6E: repeat, return, goto, gosub
    r[9] = Rlwinm(r[4], 3, 25, 28);
    r[9] += 64;
    cmpw(0, r[5], 2);
    if (bt(kLt0)) goto L_1C80;
    r[11] = rec(Rlwinm(r[3], 10, 0, 1));
    if (bt(kEq0)) {
        if (!bt(kEq7)) goto L_198C;  // only while codes execute
    } else if (bt(kGt0)) {
        if (bt(kEq7)) goto L_198C;  // only while they are skipped
    }
    cmpw(0, r[5], 3);
    if (bt(kGt0)) {
        st32(r[7] + r[9], r[15]);  // gosub
    } else if (!bt(kEq0)) {
        impure();
        r[15] = FlatRead32(r[7] + r[9]);  // return
        goto L_198C;
    }
    r[4] = Extsh(r[3]);
    r[4] = Rlwinm(r[4], 3, 9, 28);
    r[15] += r[4];
    goto L_198C;
L_1C80:
    if (!bt(kEq7)) goto L_198C;
    r[5] = r[7] + r[9];
    if (bt(kEq4)) {  // set repeat
        r[4] = r[3] & 0xFFFFu;
        st32(r[5], r[15]);
        st32(r[5] + 4, r[4]);
        goto L_198C;
    }
    impure();
    r[9] = FlatRead32(r[5] + 4);  // execute repeat
    cmpw(0, r[9], 0);
    if (bt(kEq0)) goto L_198C;
    r[9] -= 1;
    st32(r[5] + 4, r[9]);
    r[15] = FlatRead32(r[5]);
    goto L_198C;

L_1CB8:  // 80-8E: Gecko registers
    if (!bt(kEq7)) goto L_198C;
    impure();  // register loads, float operations and the copy's guest call
    r[11] = Rlwinm(r[3], 2, 26, 29);
    r[26] = r[7] + r[11];
    r[9] = FlatRead32(r[26]);
    r[14] = Rlwinm(r[3], 12, 30, 31);
    if (bt(kEq4)) goto L_1D50;
    cmpw(4, r[5], 5);
    if (!bt(kLt4)) goto L_1E48;
    cmpw(4, r[5], 3);
    if (!bt(kLt4)) goto L_1D6C;
    cmpw(4, r[5], 1);
    r[5] = rec(Rlwinm(r[3], 16, 31, 31));
    if (!bt(kEq0)) r[4] = r[12] + r[4];
    cmpw(6, r[14], 1);
    if (bt(kEq4)) {  // 82: load
        if (bt(kGt6)) r[4] = FlatRead32(r[4]);
        else if (bt(kEq6)) r[4] = FlatRead16(r[4]);
        else r[4] = FlatRead8(r[4]);
        goto L_1DFC;
    }
    r[19] = Rlwinm(r[3], 28, 20, 31);  // 84: store, repeated
L_1D1C:
    if (bt(kGt6)) {
        st32(r[4], r[9]);
        r[4] += 4;
    } else if (bt(kEq6)) {
        st16(r[4], r[9]);
        r[4] += 2;
    } else {
        st8(r[4], r[9]);
        r[4] += 1;
    }
    r[19] = rec(addic(r[19], -1));
    if (!bt(kLt0)) goto L_1D1C;
    goto L_198C;
L_1D50:  // 80: set
    r[5] = rec(Rlwinm(r[3], 16, 31, 31));
    if (!bt(kEq0)) r[4] += r[12];
    r[5] = rec(r[14] & 1u);
    if (bt(kEq0)) goto L_1DFC;
    r[4] += r[9];
    goto L_1DFC;
L_1D6C:  // 86/88: operation with an immediate / a register
    r[10] = Rlwinm(r[3], 16, 30, 31);
    r[14] = Rlwinm(r[4], 2, 26, 29);
    r[19] = r[7] + r[14];
    if (bt(kEq4)) r[19] = r[15] - 4;
    r[4] = FlatRead32(r[26]);
    r[9] = FlatRead32(r[19]);
    r[11] = rec(r[10] & 1u);
    if (!bt(kEq0)) r[26] = r[4];
    r[11] = rec(r[10] & 2u);
    if (!bt(kEq0)) {
        r[19] = r[9];
        if (bt(kEq4)) r[19] = r[12] + r[19];
    }
    r[5] = rec(Rlwinm(r[3], 12, 28, 31));
    cmpw(0, r[5], 9);
    if (!bt(kLt0)) goto L_1E04;
    // 0x80001DB4: b 0x80002160, the computed jump's explicit form (README)
    r[4] = FlatRead32(r[26]);
    r[9] = FlatRead32(r[19]);
    cmplw(0, r[5], 1);
    if (bt(kEq0)) { r[4] = r[9] * r[4]; goto L_1DFC; }
    cmplw(0, r[5], 2);
    if (bt(kEq0)) { r[4] = r[9] | r[4]; goto L_1DFC; }
    cmplw(0, r[5], 3);
    if (bt(kEq0)) { r[4] = r[9] & r[4]; goto L_1DFC; }
    cmplw(0, r[5], 4);
    if (bt(kEq0)) { r[4] = r[9] ^ r[4]; goto L_1DFC; }
    cmplw(0, r[5], 5);
    if (bt(kEq0)) { r[4] = PPC_Slw(r[9], r[4]); goto L_1DFC; }
    cmplw(0, r[5], 6);
    if (bt(kEq0)) { r[4] = PPC_Srw(r[9], r[4]); goto L_1DFC; }
    cmplw(0, r[5], 7);
    if (bt(kEq0)) { r[4] = Rotl(r[9], r[4]); goto L_1DFC; }
    cmplw(0, r[5], 8);
    if (bt(kEq0)) {  // sraw, CA as the translation computes it
        const uint32_t value = r[9], shift = r[4] & 0x3Fu;
        const bool negative = (value & 0x80000000u) != 0;
        const bool carry = shift != 0 && negative &&
                           (shift >= 32 ? value != 0 : (value & ((1u << shift) - 1u)) != 0);
        if constexpr (Plan) plan.caWritten = true;
        xer = (xer & ~kCa) | (carry ? kCa : 0u);
        r[4] = PPC_Sraw(r[9], r[4]);
        goto L_1DFC;
    }
    r[4] = r[9] + r[4];
L_1DFC:
    st32(r[26], r[4]);
    goto L_198C;
L_1E04:  // 86/88 operations 9 and 10: single-precision add and multiply
    cmpw(0, r[5], 10);
    if (bt(kGt0)) goto L_198C;
    f2.d = MemoryInline::FlatReadFloat32(r[26]);
    f3.d = MemoryInline::FlatReadFloat32(r[19]);
    if (bt(kEq0)) f2.d = PpcFmulsInline(f3.d, f2.d);
    else f2.d = static_cast<double>(PpcForceSingleValueInline(PpcFaddInline(f3.d, f2.d)));
    st32(r[26], MemoryInline::ConvertPpcDoubleToSingleBits(f2.d));
    goto L_198C;
L_1E48:  // 8A/8C: memory copy
    if (!bt(kEq7)) goto L_198C;
    r[9] = Rotl(r[3], 24);
    r[14] = r[12];
    // bl 0x80001E88: the two operands' addresses into r17 and r9 (a guest function with its own guard)
    writeBack();
    ctx->lr = 0x80001E58u;
    func_80001E88(ctx);
    readBack();
    if (bt(kEq4)) r[9] += r[4];
    else r[17] += r[4];
    r[4] = rec(Rlwinm(r[3], 24, 16, 31));
    r[5] = 0;
    if (r[4] != 0 && AlreadyCopied(r[9], r[17], r[4])) {
        // As the loop leaves it after r4 stores of the bytes already there: r10 the last byte, r5 = r4, CR0 eq.
        r[10] = FlatRead8(r[17] + r[4] - 1);
        r[5] = r[4];
        cmpw(0, r[5], r[4]);
    }
    while (!bt(kEq0)) {
        r[10] = FlatRead8(r[5] + r[17]);
        st8(r[5] + r[9], r[10]);
        r[5] += 1;
        cmpw(0, r[5], r[4]);
    }
    goto L_198C;

L_1EFC:  // C0-CE
    r[26] = r[4];
    r[4] <<= 3;
    if (!bt(kEq4)) goto L_1F24;
    if (bt(kEq7)) {  // C0: run the code's instructions as a function, with the handler's registers
        Count(g_stats.c0);
        impure();
        writeBack();
        ctx->lr = 0x80001F14u;
        InvokeIndirectCpu(r[15], ctx);
        readBack();
    }
L_1F14:  // past the code's lines (r4 bytes, rounded up to a line)
    r[15] = r[4] + r[15];
    r[15] += 7;
    r[15] &= ~7u;
    goto L_198C;
L_1F24:
    cmpw(4, r[5], 3);
    if (bt(kGt4)) {
        cmpw(4, r[5], 6);
        if (bt(kEq4)) goto L_1FB0;
        return bail(0x80001AB4u);  // CE and the other if-like C types
    }
    r[5] = 0x48000000u;
    r[12] = r[3] + r[12];
    r[12] &= ~3u;
    if (bt(kEq4)) {  // C6: branch
        if (!bt(kEq7)) goto L_198C;
        r[4] = r[26] & ~3u;
        r[4] = r[4] - r[12];
        r[5] = Rlwimi(r[5], r[4], 0, 6, 29);
        r[5] = Rlwimi(r[5], r[3], 0, 31, 31);
        st32(r[12], r[5]);
        goto L_198C;
    }
    if (!bt(kEq7)) goto L_1F14;  // C2/C4: a branch to the code's instructions, and one back after them
    r[9] = r[15] - r[12];
    r[5] = Rlwimi(r[5], r[9], 0, 6, 29);
    st32(r[12], r[5]);
    r[12] += 4;
    r[11] = r[15] + r[4];
    r[11] -= 4;
    r[9] = r[12] - r[11];
    r[5] = Rlwimi(r[5], r[9], 0, 6, 29);
    st32(r[11], r[5]);
    goto L_1F14;
L_1FB0:  // CC: on/off switch
    r[5] = Rlwinm(r[26], 31, 31, 31);
    r[5] ^= 1u;
    r[3] = rec(r[8] & 1u);
    cmpw(0, r[5], r[3]);
    if (!bt(kEq0)) {
        r[26] = Rlwimi(r[26], r[8], 1, 30, 30);
        r[26] ^= 2u;
        r[5] = rec(Rlwinm(r[26], 31, 31, 31));
        if (!bt(kEq0)) r[26] ^= 1u;
        st32(r[15] - 4, r[26]);
    }
    r[8] = Rlwimi(r[8], r[26], 0, 31, 31);
    goto L_198C;

L_1FE4:  // E0-FE
    cmpw(0, r[11], 0);
    if (bt(kEq0)) goto L_2120;
    cmpw(0, r[5], 1);
    if (bt(kEq0)) return bail(0x80002008u);  // F2
    cmpw(0, r[5], 2);
    if (bt(kEq0)) return bail(0x8000200Cu);  // F4
    cmpw(0, r[5], 3);
    if (bt(kEq0)) return bail(0x80002070u);  // F6
    impure();
    goto L_1944;                             // F0: the end
L_2120:
    if (bt(kEq4)) {  // E0: full terminator
        r[8] = 0;
    } else {  // E2: endifs
        r[9] = rec(r[3] & 0x1Fu);
        r[5] = Rlwinm(r[3], 12, 31, 31);
        r[8] = PPC_Srw(r[8], r[9]);
        r[23] = rec(Rlwinm(r[8], 31, 31, 31));
    }
    if (bt(kEq0)) r[8] ^= r[5];
    r[5] = rec(r[4] & 0xFFFF0000u);
    if (!bt(kEq0)) r[6] = r[5];
    r[5] = rec(Rlwinm(r[4], 16, 0, 15));
    if (!bt(kEq0)) r[16] = r[5];
    goto L_198C;

L_1944:  // the caller's registers back from the frame
    if constexpr (Plan) plan.EndPass();
    f2.d = MemoryInline::FlatReadFloat64(r[1] + 152);
    f3.d = MemoryInline::FlatReadFloat64(r[1] + 160);
    r[25] = ctx->msr;
    {
        const uint32_t lr = FlatRead32(r[1] + 172);
        r[0] = FlatRead32(r[1] + 12);
        cr = r[0];
        r[0] = FlatRead32(r[1] + 16);
        ctr = r[0];
        r[0] = FlatRead32(r[1] + 20);
        xer = r[0];
        for (int i = 3; i < 32; ++i) r[i] = FlatRead32(r[1] + 24 + 4 * static_cast<uint32_t>(i - 3));
        r[0] = FlatRead32(r[1] + 8);
        r[1] += 168;
        writeBack();
        ctx->lr = lr;
    }
}

// One pass: with the plan when it is on and no other pass is running (a nested or concurrent entry walks plainly).
void RunPass(CpuContext* ctx) {
    if (GeckoPlan::g_mode.load(std::memory_order_relaxed) != 0 &&
        !GeckoPlan::g_inPass.exchange(true, std::memory_order_acq_rel)) {
        RunHandler<true>(ctx);
        GeckoPlan::g_inPass.store(false, std::memory_order_release);
        return;
    }
    RunHandler<false>(ctx);
}

// BRAWL_GECKO_STATS: one line per 600 passes.
void ReportStats() {
    static std::atomic<uint64_t> lastReported{0};
    const uint64_t passes = g_stats.passes.load(std::memory_order_relaxed);
    if (passes - lastReported.load(std::memory_order_relaxed) < 600) return;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static uint64_t last[8] = {}, lastPlan[9] = {};
    static const auto start = std::chrono::steady_clock::now();
    const uint64_t now[8] = {passes, g_stats.codes.load(), g_stats.stores.load(), g_stats.changed.load(),
                             g_stats.stringBytes.load(), g_stats.stringsSkipped.load(), g_stats.c0.load(),
                             g_stats.bails.load()};
    const GeckoPlan::Counters& pc = GeckoPlan::g_plan.c;
    const uint64_t plan[9] = {pc.steps, pc.replayed, pc.recorded, pc.dropped, pc.keyMisses, pc.compared, pc.trusted,
                              pc.rewritten, pc.passes};
    const double n = static_cast<double>(now[0] - last[0]);
    if (n <= 0) return;
    RT_LOGF(RT_TAG_MOD,
            "[gecko] %.1f s: %llu passes; per pass %.0f codes, %.0f stores (%.1f changed memory), %.0f string bytes "
            "(%.0f strings already there), %.1f C0 calls, %.2f interpreter hand-offs; plan (%.0f%% of passes): %.0f steps "
            "interpreted, %.1f runs replayed, %.2f recorded, %.2f dropped, %.2f key misses, extents %.0f compared, %.0f "
            "trusted, %.1f rewritten\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
            static_cast<unsigned long long>(now[0]), (now[1] - last[1]) / n, (now[2] - last[2]) / n,
            (now[3] - last[3]) / n, (now[4] - last[4]) / n, (now[5] - last[5]) / n, (now[6] - last[6]) / n,
            (now[7] - last[7]) / n, 100.0 * static_cast<double>(plan[8] - lastPlan[8]) / n,
            static_cast<double>(plan[0] - lastPlan[0]) / n, static_cast<double>(plan[1] - lastPlan[1]) / n,
            static_cast<double>(plan[2] - lastPlan[2]) / n, static_cast<double>(plan[3] - lastPlan[3]) / n,
            static_cast<double>(plan[4] - lastPlan[4]) / n, static_cast<double>(plan[5] - lastPlan[5]) / n,
            static_cast<double>(plan[6] - lastPlan[6]) / n, static_cast<double>(plan[7] - lastPlan[7]) / n);
    std::memcpy(last, now, sizeof(last));
    std::memcpy(lastPlan, plan, sizeof(lastPlan));
    lastReported.store(passes, std::memory_order_relaxed);
}

// BRAWL_HLE_VERIFY: the translated handler, then the port from the same state; the translated results stay.
void VerifyPass(CpuContext* ctx) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static const uint32_t every = [] {
        const char* value = std::getenv("BRAWL_GECKO_VERIFY_EVERY");
        const unsigned long n = value != nullptr ? std::strtoul(value, nullptr, 10) : 1;
        return n == 0 ? 1u : static_cast<uint32_t>(n);
    }();
    static uint64_t pass = 0, compared = 0, mismatches = 0;
    if (pass++ % every != 0) {
        func_800018A8_body(ctx);
        return;
    }
    struct Region { uint32_t base, size; };
    static constexpr Region kRegions[2] = {{0x80000000u, kMem1Size}, {0x90000000u, kMem2Size}};
    static std::vector<uint8_t> before[2], original[2];
#if MKW_LEAN_MEMORY
    static std::vector<uint8_t> marksBefore[2], marksOriginal[2];
    const auto marks = [](const Region& region) { return BrawlDirtyMemory::leanBlocks + ((region.base & 0x3FFFFFFFu) >> 6); };
#endif
    const auto capture = [&](std::vector<uint8_t>* mem, [[maybe_unused]] std::vector<uint8_t>* dirty) {
        for (int i = 0; i < 2; ++i) {
            const uint8_t* host = Memory::GetPointer(kRegions[i].base, kRegions[i].size);
            mem[i].assign(host, host + kRegions[i].size);
#if MKW_LEAN_MEMORY
            dirty[i].assign(marks(kRegions[i]), marks(kRegions[i]) + kRegions[i].size / 64);
#endif
        }
    };
    const auto restore = [&](const std::vector<uint8_t>* mem, [[maybe_unused]] const std::vector<uint8_t>* dirty) {
        BrawlDirtyMemory::MarksCleared();
        for (int i = 0; i < 2; ++i) {
            std::memcpy(Memory::GetPointer(kRegions[i].base, kRegions[i].size), mem[i].data(), kRegions[i].size);
#if MKW_LEAN_MEMORY
            std::memcpy(marks(kRegions[i]), dirty[i].data(), kRegions[i].size / 64);
#endif
        }
    };
    // The AX mix worker stores each AI frame into guest memory from its own thread, and only this thread starts a
    // mix (as hle/g3d_calc.cpp's verification): idle, it cannot land between the captures below.
    AxDspHle::JoinMixWorker();
#if MKW_LEAN_MEMORY
    capture(before, marksBefore);
#else
    capture(before, nullptr);
#endif
    const CpuContext in = *ctx;
    func_800018A8_body(ctx);
    const CpuContext afterOriginal = *ctx;
#if MKW_LEAN_MEMORY
    capture(original, marksOriginal);
    restore(before, marksBefore);
#else
    capture(original, nullptr);
    restore(before, nullptr);
#endif
    *ctx = in;
    BrawlDirtyMemory::MarksCleared();  // every extent of the plan is compared in this pass
    RunPass(ctx);
    BrawlDirtyMemory::MarksCleared();
    ++compared;
    int logged = 0;
    bool same = true;
    if (std::memcmp(ctx, &afterOriginal, sizeof(CpuContext)) != 0) {
        same = false;
        for (size_t off = 0; off + 4 <= sizeof(CpuContext) && logged < 12; off += 4) {
            uint32_t o = 0, n = 0;
            std::memcpy(&o, reinterpret_cast<const uint8_t*>(&afterOriginal) + off, 4);
            std::memcpy(&n, reinterpret_cast<const uint8_t*>(ctx) + off, 4);
            if (o == n) continue;
            if (mismatches < 40) RT_LOGF(RT_TAG_MOD, "[gecko-verify] pass %llu: context word +%zu original %08X native %08X\n",
                                         static_cast<unsigned long long>(pass - 1), off, o, n);
            ++logged;
        }
    }
    for (int i = 0; i < 2; ++i) {
        const uint8_t* host = Memory::GetPointer(kRegions[i].base, kRegions[i].size);
        if (std::memcmp(host, original[i].data(), kRegions[i].size) != 0) {
            same = false;
            for (uint32_t at = 0; at < kRegions[i].size && logged < 24; at += 4) {
                if (std::memcmp(host + at, original[i].data() + at, 4) == 0) continue;
                if (mismatches < 40) {
                    uint32_t o = 0, n = 0;
                    std::memcpy(&o, original[i].data() + at, 4);
                    std::memcpy(&n, host + at, 4);
                    RT_LOGF(RT_TAG_MOD, "[gecko-verify] pass %llu: mem %08X original %08X native %08X (host order)\n",
                            static_cast<unsigned long long>(pass - 1), kRegions[i].base + at, o, n);
                }
                ++logged;
            }
        }
#if MKW_LEAN_MEMORY
        // Every block the port changed carries a mark.
        const uint8_t* dirty = marks(kRegions[i]);
        for (uint32_t block = 0; block < kRegions[i].size / 64; ++block) {
            if (dirty[block] != 0 || std::memcmp(host + block * 64, before[i].data() + block * 64, 64) == 0) continue;
            same = false;
            if (mismatches < 40 && logged++ < 24)
                RT_LOGF(RT_TAG_MOD, "[gecko-verify] pass %llu: block %08X changed without a dirty mark\n",
                        static_cast<unsigned long long>(pass - 1), kRegions[i].base + block * 64);
        }
#endif
    }
    if (!same) ++mismatches;
    if (compared <= 3 || (compared % 600) == 0 || (!same && mismatches <= 40))
        RT_LOGF(RT_TAG_MOD, "[gecko-verify] %llu passes compared, %llu mismatched\n",
                static_cast<unsigned long long>(compared), static_cast<unsigned long long>(mismatches));
    *ctx = afterOriginal;
#if MKW_LEAN_MEMORY
    restore(original, marksOriginal);
#else
    restore(original, nullptr);
#endif
}

} // namespace

MKW_NATIVE_BODY(0x800018A8u, BrawlGecko_CodeHandler) {
    if (g_originals) {
        func_800018A8_body(ctx);
        return;
    }
    if (g_verify) {
        VerifyPass(ctx);
        return;
    }
    if (g_statsOn) g_logChanges = g_stats.passes.load(std::memory_order_relaxed) + 1 == g_logChangesPass;
    RunPass(ctx);
    if (g_statsOn) {
        g_logChanges = false;
        g_stats.passes.fetch_add(1, std::memory_order_relaxed);
        ReportStats();
    }
}

// Test entry points (declared by the test itself). mode: 0 plan off, 1 plan on, 2 plan on without mark trust.
extern "C" void BrawlGecko_PlanControl(int mode) {
    GeckoPlan::g_plan.ResetAll();
    GeckoPlan::g_plan.haveBase = false;
    GeckoPlan::g_mode.store(mode <= 0 ? 0 : (mode == 1 ? 1 : 2));
    if (mode > 0) GeckoPlan::InstallHooks();
    else GeckoPlan::g_marksOn = false;
}

// Cumulative: passes, steps interpreted, runs replayed, runs recorded, runs dropped, key misses, extents compared,
// extents trusted, extents rewritten, verify checks, verify mismatches.
extern "C" void BrawlGecko_PlanCounters(uint64_t* out, size_t count) {
    const GeckoPlan::Counters& c = GeckoPlan::g_plan.c;
    const uint64_t values[11] = {c.passes, c.steps, c.replayed, c.recorded, c.dropped, c.keyMisses, c.compared,
                                 c.trusted, c.rewritten, c.verifyChecks, c.verifyMismatches};
    for (size_t i = 0; i < count && i < 11; ++i) out[i] = values[i];
}
