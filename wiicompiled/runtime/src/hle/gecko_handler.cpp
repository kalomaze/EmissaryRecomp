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
// BRAWL_HLE_VERIFY=1 (as hle/mtx.cpp): every pass runs the translated handler and this port from the same state and
// compares the register file and all of MEM1 and MEM2, and checks that every 64-byte block the port changed carries
// a dirty mark; the translated handler's results stay. BRAWL_GECKO_VERIFY_EVERY=N verifies every Nth pass only.
// BRAWL_GECKO_STATS=1: every 600 passes, what a pass does (codes, stores, stores that changed memory, string bytes).
// BRAWL_HLE_GECKO=0: the translated handler, for A/B measurements on one binary.

#include "abi_bridge.h"
#include "audio/ax_dsp.h"
#include "brawl_dirty_memory.h"
#include "brawl_mark_observer.h"
#include "memory.h"
#include "guest_code_patch.h"
#include "ppc_interpreter.h"
#include "runtime_log.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
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

// Inlined at each of the handler's stores (~3,900 a pass).
[[gnu::always_inline]] inline void Store8(uint32_t address, uint32_t value) {
    Count(g_stats.stores);
    if (InRam(address, 1) && FlatRead8(address) == static_cast<uint8_t>(value)) return;
    Count(g_stats.changed);
    if (g_logChanges) LogChange<uint8_t>(address, value);
    MemoryInline::FlatWrite8(address, static_cast<uint8_t>(value));
}
[[gnu::always_inline]] inline void Store16(uint32_t address, uint32_t value) {
    Count(g_stats.stores);
    if (InRam(address, 2) && FlatRead16(address) == static_cast<uint16_t>(value)) return;
    Count(g_stats.changed);
    if (g_logChanges) LogChange<uint16_t>(address, value);
    MemoryInline::FlatWrite16(address, static_cast<uint16_t>(value));
}
[[gnu::always_inline]] inline void Store32(uint32_t address, uint32_t value) {
    Count(g_stats.stores);
    if (InRam(address, 4) && FlatRead32(address) == value) return;
    Count(g_stats.changed);
    if (g_logChanges) LogChange<uint32_t>(address, value);
    MemoryInline::FlatWrite32(address, value);
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

// The handler resumes beyond the separately translated helper at 0x80001E88,
// so its native implementation depends on both guard intervals.
bool HandlerChanged() {
    return GuestCodePatch::AnyPatchedInRange(0x800018A8u, 0x800021B0u);
}

void RunHandler(CpuContext* ctx) {
    uint32_t r[32];
    std::memcpy(r, ctx->gpr, sizeof(r));
    uint32_t cr = ctx->cr, xer = ctx->xer, ctr = ctx->ctr;
    PPC_FPR f2 = ctx->fpr[2], f3 = ctx->fpr[3];
    Prefetcher prefetch;

    const auto bt = [&](int bit) { return (cr & (0x80000000u >> bit)) != 0; };
    const auto cmpw = [&](int field, uint32_t a, uint32_t b) {
        SetCRResident(cr, xer, field, static_cast<int32_t>(a), static_cast<int32_t>(b));
    };
    const auto cmplw = [&](int field, uint32_t a, uint32_t b) { SetCRResident(cr, xer, field, a, b); };
    const auto rec = [&](uint32_t value) {  // a record form's CR0
        SetCRResident(cr, xer, 0, static_cast<int32_t>(value), 0);
        return value;
    };
    const auto addic = [&](uint32_t value, int32_t imm) {
        const uint64_t sum = uint64_t{value} + static_cast<uint32_t>(imm);
        xer = (xer & ~kCa) | ((sum >> 32) != 0 ? kCa : 0u);
        return static_cast<uint32_t>(sum);
    };
    const auto writeBack = [&] {
        std::memcpy(ctx->gpr, r, sizeof(r));
        ctx->cr = cr;
        ctx->xer = xer;
        ctx->ctr = ctr;
        ctx->fpr[2] = f2;
        ctx->fpr[3] = f3;
    };
    const auto readBack = [&] {
        std::memcpy(r, ctx->gpr, sizeof(r));
        cr = ctx->cr;
        xer = ctx->xer;
        ctr = ctx->ctr;
        f2 = ctx->fpr[2];
        f3 = ctx->fpr[3];
    };
    // The rest of the pass in the interpreter, from the handler's instruction at pc with this state; it returns at the
    // handler's own blr, to the return address the frame holds.
    const auto bail = [&](uint32_t pc) {
        Count(g_stats.bails);
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
    Count(g_stats.codes);
    if (g_logChanges) g_logCode = r[15];
    prefetch.Advanced();
    prefetch.Run(r[15], r[6], r[16]);
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
        Store16(r[9] + r[12], r[4]);
        r[9] += 2;
    } else {
        Store8(r[9] + r[12], r[4]);
        r[9] += 1;
    }
    r[10] = rec(addic(r[10], -1));
    if (!bt(kLt0)) goto L_1A18;
    goto L_198C;
L_1A3C:
    r[12] &= ~3u;
    Store32(r[12], r[4]);
    goto L_198C;

L_1A48:  // 06: string write, copied from its last byte down
    r[9] = r[4];
    if (!bt(kEq7)) goto L_1F14;
    if (static_cast<int32_t>(r[9] - 1) >= 0 && AlreadyCopied(r[12], r[15], r[9])) {
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
    Store8(r[9] + r[12], r[5]);
    goto L_1A50;

L_1A64:  // 08: serial write
    r[15] += 8;
    if (!bt(kEq7)) goto L_198C;
    r[5] = FlatRead32(r[15] - 8);
    r[11] = FlatRead32(r[15] - 4);
    r[17] = r[5] & 0xFFFFu;
    r[10] = Rlwinm(r[5], 16, 20, 31);
    r[5] >>= 28;
L_1A80:
    cmpw(5, r[5], 1);
    if (bt(kEq5)) Store16(r[9] + r[12], r[4]);
    else if (bt(kGt5)) Store32(r[9] + r[12], r[4]);
    else Store8(r[9] + r[12], r[4]);
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
        r[9] = FlatRead32(r[7] + r[9]);
        r[4] += r[9];
    }
    if (!bt(kEq3)) r[4] = r[12] + r[4];
    cmpw(5, r[14], 1);
    if (!bt(kEq5)) r[4] = FlatRead32(r[4]);
    r[3] = rec(Rlwinm(r[3], 12, 31, 31));
    if (bt(kEq0)) goto L_1C18;
    r[4] += bt(kLt4) ? r[6] : r[16];
    goto L_1C18;
L_1BF4:
    r[5] = rec(Rlwinm(r[3], 20, 31, 31));
    if (!bt(kEq0)) {
        r[9] = FlatRead32(r[7] + r[9]);
        r[4] += r[9];
    }
    Store32(r[12] + r[4], bt(kLt4) ? r[6] : r[16]);
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
        Store32(r[7] + r[9], r[15]);  // gosub
    } else if (!bt(kEq0)) {
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
        Store32(r[5], r[15]);
        Store32(r[5] + 4, r[4]);
        goto L_198C;
    }
    r[9] = FlatRead32(r[5] + 4);  // execute repeat
    cmpw(0, r[9], 0);
    if (bt(kEq0)) goto L_198C;
    r[9] -= 1;
    Store32(r[5] + 4, r[9]);
    r[15] = FlatRead32(r[5]);
    goto L_198C;

L_1CB8:  // 80-8E: Gecko registers
    if (!bt(kEq7)) goto L_198C;
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
        Store32(r[4], r[9]);
        r[4] += 4;
    } else if (bt(kEq6)) {
        Store16(r[4], r[9]);
        r[4] += 2;
    } else {
        Store8(r[4], r[9]);
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
        xer = (xer & ~kCa) | (carry ? kCa : 0u);
        r[4] = PPC_Sraw(r[9], r[4]);
        goto L_1DFC;
    }
    r[4] = r[9] + r[4];
L_1DFC:
    Store32(r[26], r[4]);
    goto L_198C;
L_1E04:  // 86/88 operations 9 and 10: single-precision add and multiply
    cmpw(0, r[5], 10);
    if (bt(kGt0)) goto L_198C;
    f2.d = MemoryInline::FlatReadFloat32(r[26]);
    f3.d = MemoryInline::FlatReadFloat32(r[19]);
    if (bt(kEq0)) f2.d = PpcFmulsInline(f3.d, f2.d);
    else f2.d = static_cast<double>(PpcForceSingleValueInline(PpcFaddInline(f3.d, f2.d)));
    Store32(r[26], MemoryInline::ConvertPpcDoubleToSingleBits(f2.d));
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
    if (HandlerChanged()) return bail(0x80001E58u);
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
        Store8(r[5] + r[9], r[10]);
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
        writeBack();
        ctx->lr = 0x80001F14u;
        InvokeIndirectCpu(r[15], ctx);
        readBack();
        if (HandlerChanged()) return bail(0x80001F14u);
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
        Store32(r[12], r[5]);
        goto L_198C;
    }
    if (!bt(kEq7)) goto L_1F14;  // C2/C4: a branch to the code's instructions, and one back after them
    r[9] = r[15] - r[12];
    r[5] = Rlwimi(r[5], r[9], 0, 6, 29);
    Store32(r[12], r[5]);
    r[12] += 4;
    r[11] = r[15] + r[4];
    r[11] -= 4;
    r[9] = r[12] - r[11];
    r[5] = Rlwimi(r[5], r[9], 0, 6, 29);
    Store32(r[11], r[5]);
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
        Store32(r[15] - 4, r[26]);
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

L_1944:
    // The exit restores the caller and executes isync at 0x80001984. Run it
    // through the interpreter so code written by this pass is synchronized,
    // including changes to the handler's exit instructions themselves.
    return bail(0x80001944u);
}

// BRAWL_GECKO_STATS: one line per 600 passes.
void ReportStats() {
    static std::atomic<uint64_t> lastReported{0};
    const uint64_t passes = g_stats.passes.load(std::memory_order_relaxed);
    if (passes - lastReported.load(std::memory_order_relaxed) < 600) return;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static uint64_t last[8] = {};
    static const auto start = std::chrono::steady_clock::now();
    const uint64_t now[8] = {passes, g_stats.codes.load(), g_stats.stores.load(), g_stats.changed.load(),
                             g_stats.stringBytes.load(), g_stats.stringsSkipped.load(), g_stats.c0.load(),
                             g_stats.bails.load()};
    const double n = static_cast<double>(now[0] - last[0]);
    if (n <= 0) return;
    RT_LOGF(RT_TAG_MOD,
            "[gecko] %.1f s: %llu passes; per pass %.0f codes, %.0f stores (%.1f changed memory), %.0f string bytes "
            "(%.0f strings already there), %.1f C0 calls, %.2f interpreter hand-offs\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
            static_cast<unsigned long long>(now[0]), (now[1] - last[1]) / n, (now[2] - last[2]) / n,
            (now[3] - last[3]) / n, (now[4] - last[4]) / n, (now[5] - last[5]) / n, (now[6] - last[6]) / n,
            (now[7] - last[7]) / n);
    std::memcpy(last, now, sizeof(last));
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
    const BrawlDirtyMemory::ScopedMarkObservation observeMarks;
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
    RunHandler(ctx);
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
    if (g_originals || HandlerChanged()) {
        func_800018A8_body(ctx);
        return;
    }
    if (g_verify) {
        VerifyPass(ctx);
        return;
    }
    if (g_statsOn) g_logChanges = g_stats.passes.load(std::memory_order_relaxed) + 1 == g_logChangesPass;
    RunHandler(ctx);
    if (g_statsOn) {
        g_logChanges = false;
        g_stats.passes.fetch_add(1, std::memory_order_relaxed);
        ReportStats();
    }
}
