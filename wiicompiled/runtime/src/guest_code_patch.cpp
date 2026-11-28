#include "brawl_dirty_memory.h"
// guest_code_patch.cpp - routes translated functions whose guest code a mod changed at run time to the interpreter.
// See include/guest_code_patch.h.

#include "guest_code_patch.h"

#include "abi_bridge.h"
#include "guest_flat_memory.h"
#include "hz_perf.h"
#include "memory.h"
#include "ppc_interpreter.h"
#include "ppc_interpreter_internal.h"
#include "recomp_mod_loader.h"
#include "runtime_log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <utility>
#include <vector>

// Generated with the translation (guest_code_guard table): every translated function's start, ascending, and its
// "patched" byte.
extern "C" const uint32_t g_mkwGuardedFunctionStarts[];
extern "C" const uint32_t g_mkwGuardedFunctionCount;
// (leaf, caller) pairs of guard indices, by leaf: callers that hold an inlined copy of a small leaf function.
extern "C" const uint32_t g_mkwInlinedLeaves[];
extern "C" const uint32_t g_mkwInlinedLeafCount;

namespace {

constexpr uint32_t kPage = 4096;

std::atomic<bool> g_active{false};
std::atomic<bool> g_anyDirty{false};  // pending write notifications or retained edge pages
// Lock-free single-word fast path for lean-memory NoteWrite: 0 = clean, (wordAddr | (1ULL<<32)) = 1 dirty 8-byte
// aligned span (covers 1/2/4/8-byte writes within a page), ~0ULL = overflowed to page bitmap.
std::atomic<uint64_t> g_pendingFast{0};
// NoteWrite runs inside memory-fault recovery. Its lookup and notification must
// neither allocate nor wait for a lock held by the interrupted thread.
static_assert(std::atomic<uint32_t>::is_always_lock_free);
static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(std::atomic<bool>::is_always_lock_free);
constexpr uint32_t kPageWords = (1u << 20) / 32;
std::atomic<uint32_t> g_guardedPages[kPageWords]{};
std::atomic<uint32_t> g_pendingPages[kPageWords]{};
std::atomic<uint32_t> g_pendingGroups[kPageWords / 32]{};
std::atomic<uint32_t> g_pendingRoots[kPageWords / 32 / 32]{};

std::mutex g_mutex;
std::vector<std::pair<uint32_t, uint32_t>> g_ranges;                              // guarded code, [start, end)
std::unordered_map<uint32_t, std::unique_ptr<std::array<uint8_t, kPage>>> g_pristine;  // by page address
// Each compared page as of its last Sync (created from the pristine copy on first use, dropped with it by
// SnapshotLocked). A word equal here was already handled with this same value; data that permanently differs from the
// disc (a module's variables beside its code) then costs one page compare per Sync, not a lookup per word.
std::unordered_map<uint32_t, std::unique_ptr<std::array<uint8_t, kPage>>> g_seen;  // by page address
std::vector<uint32_t> g_dirty;                                                     // page addresses written since Sync
// 1-entry MRU caches for ComparePageLocked / CompareWordPairLocked (under g_mutex): avoids hash lookups + range scan
// when the same page/word is written repeatedly (e.g. a mod variable embedded inside a code page). Invalidated on
// SnapshotLocked / Protect / ResetPristine.
uint32_t g_lastComparedPage = UINT32_MAX;
const uint8_t* g_lastPristine = nullptr;
uint8_t* g_lastSeen = nullptr;
uint32_t g_lastSpanLo = 0, g_lastSpanHi = 0;
uint32_t g_lastPatchedWordAddr = UINT32_MAX;
uint32_t* g_lastPatchedSitePtr = nullptr;
uint8_t* g_lastPatchedSeenPtr = nullptr;
std::unordered_map<uint32_t, uint32_t> g_sites;  // every word a mod patched, and what it wrote there (guest order)
std::unordered_set<uint32_t> g_reported;                                           // functions logged as patched
std::unordered_set<uint32_t> g_loggedWords;                                        // BRAWL_PATCH_LOG words logged
std::vector<uint32_t> g_forced;                                                    // BRAWL_INTERPRET guard indices

// RedirectModule's home ranges, read without the lock by every patched call (start 0: a free slot).
struct Redirect {
    std::atomic<uint32_t> start{0}, end{0}, target{0};
};
Redirect g_redirects[16];
// The lowest start of any active redirect (or UINT32_MAX): a patched call below it skips the slot scan. Redirects
// cover module homes in MEM2; the patched calls are mostly main.dol and resident code in MEM1.
std::atomic<uint32_t> g_lowestRedirect{UINT32_MAX};

// RunPatched's answer for a patched call is a function of the address, the redirect slots and the variants' states.
// Per thread, the last answer that was a code variant is kept with the epoch it was computed under; every change that
// can alter an answer (a variant set back to unchecked, a redirect added or cleared) bumps the epoch first-to-last
// under g_mutex, so a kept entry is used only while nothing it depended on has changed. Epoch 0 is never current, so
// the zero-initialized slots never match.
struct VariantCallCacheSlot {
    uint32_t address;
    uint32_t epoch;
    void (*entry)(CpuContext*);
};
constexpr uint32_t kVariantCallCacheSlots = 1024;
thread_local VariantCallCacheSlot t_variantCallCache[kVariantCallCacheSlots];
std::atomic<uint32_t> g_variantCallEpoch{1};

void BumpVariantCallEpoch() {
    uint32_t next = g_variantCallEpoch.load(std::memory_order_relaxed) + 1;
    if (next == 0) next = 1;
    g_variantCallEpoch.store(next, std::memory_order_release);
}

void UpdateLowestRedirectLocked() {
    uint32_t lowest = UINT32_MAX;
    for (const Redirect& r : g_redirects) {
        const uint32_t start = r.start.load(std::memory_order_relaxed);
        if (start != 0 && start < lowest) lowest = start;
    }
    g_lowestRedirect.store(lowest, std::memory_order_release);
}

// BRAWL_PATCH_STATS=<file>: per presented frame (SyncFrame), one line of these counters, for finding what heavy frames
// spend on patched code. Bumped with relaxed load/store (no locked add): approximate under contention, ~free.
struct PatchStats {
    std::atomic<uint32_t> variantCalls{0}, interpreterCalls{0}, checks{0}, checkFailures{0}, noteWrites{0},
        pagesCompared{0}, pagesChanged{0};
};
PatchStats g_stats;
bool PatchStatsEnabled() {
    static const bool on = std::getenv("BRAWL_PATCH_STATS") != nullptr;
    return on;
}
inline void Bump(std::atomic<uint32_t>& counter, uint32_t n = 1) {
    if (__builtin_expect(PatchStatsEnabled(), 0))
        counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

// Code variants (guest_code_patch.h): per variant, whether guest memory still holds the words it was translated from.
// A change to any page holding one of its words sets it back to unchecked; the next patched call of its function
// checks it again. Read without the lock by every patched call.
enum : uint8_t { kVariantUnchecked = 0, kVariantMatches = 1, kVariantDiffers = 2 };
std::unique_ptr<std::atomic<uint8_t>[]> g_variantState;
std::unique_ptr<std::atomic<uint8_t>[]> g_variantLogged;                  // bit 0: logged matching, bit 1: differing
std::unordered_map<uint32_t, std::vector<uint32_t>> g_variantsByPage;  // page -> variants with a word there
std::vector<std::pair<uint32_t, uint32_t>> g_variantSlots;              // open-addressed: (address, first variant + 1)
uint32_t g_variantSlotMask = 0;
bool g_variantsEnabled = false;
std::once_flag g_variantsOnce;

uint32_t VariantSlot(uint32_t address) { return ((address >> 2) * 0x9E3779B1u) >> 8; }

// Once, at the first Protect. BRAWL_NO_CODE_VARIANTS=1 (and BRAWL_INTERPRET's check of the interpreter) keep every
// patched function in the interpreter.
void InitVariants() {
    const uint32_t count = g_mkwCodeVariantCount;
    g_variantsEnabled = count != 0 && std::getenv("BRAWL_NO_CODE_VARIANTS") == nullptr &&
                        std::getenv("BRAWL_INTERPRET") == nullptr;
    if (!g_variantsEnabled) return;
    g_variantState = std::make_unique<std::atomic<uint8_t>[]>(count);
    g_variantLogged = std::make_unique<std::atomic<uint8_t>[]>(count);
    uint32_t size = 16;
    while (size < count * 4) size <<= 1;
    g_variantSlots.assign(size, {0u, 0u});
    g_variantSlotMask = size - 1;
    // Bisecting a variant: BRAWL_CODE_VARIANTS_SKIP=<hex>,<hex>... and BRAWL_CODE_VARIANTS_RANGE=<hex lo>-<hex hi> leave
    // the functions outside them to the interpreter.
    std::vector<uint32_t> skip;
    for (const char* p = std::getenv("BRAWL_CODE_VARIANTS_SKIP"); p != nullptr && *p;) {
        char* end = nullptr;
        skip.push_back(static_cast<uint32_t>(std::strtoul(p, &end, 16)));
        if (end == p) break;
        p = *end ? end + 1 : end;
    }
    uint32_t rangeLo = 0, rangeHi = 0xFFFFFFFFu;
    if (const char* range = std::getenv("BRAWL_CODE_VARIANTS_RANGE")) {
        char* dash = nullptr;
        rangeLo = static_cast<uint32_t>(std::strtoul(range, &dash, 16));
        if (*dash == '-') rangeHi = static_cast<uint32_t>(std::strtoul(dash + 1, nullptr, 16));
    }
    for (uint32_t v = 0; v < count; ++v) {
        const MkwCodeVariant& variant = g_mkwCodeVariants[v];
        if (variant.address < rangeLo || variant.address > rangeHi ||
            std::find(skip.begin(), skip.end(), variant.address) != skip.end()) {
            continue;
        }
        if (v == 0 || g_mkwCodeVariants[v - 1].address != variant.address) {
            uint32_t slot = VariantSlot(variant.address) & g_variantSlotMask;
            while (g_variantSlots[slot].second != 0) slot = (slot + 1) & g_variantSlotMask;
            g_variantSlots[slot] = {variant.address, v + 1};
        }
        for (uint32_t g = variant.firstGuard; g < variant.firstGuard + variant.guardCount; ++g) {
            const uint32_t start = g_mkwCodeVariantGuards[g * 3], words = g_mkwCodeVariantGuards[g * 3 + 1];
            for (uint32_t page = start & ~(kPage - 1); page < start + words * 4; page += kPage) {
                auto& list = g_variantsByPage[page];
                if (list.empty() || list.back() != v) list.push_back(v);
            }
        }
    }
    RT_LOGF(RT_TAG_MOD, "[mods] %u code variant(s): patched functions whose code matches one run it translated\n",
            count);
}

// The memory outside translated code that variants were translated from (a code list's hook bodies, a plugin
// loader's hook trampolines): its pages holding a variant's words are write-trapped like guarded code, without marking
// any function patched, so a change there re-checks the variants. Only those pages: the regions also hold data.
void ProtectVariantRegions() {
    if (!g_variantsEnabled) return;
    std::vector<uint32_t> pages;
    for (const auto& [page, variants] : g_variantsByPage) {
        for (uint32_t r = 0; r < g_mkwCodeVariantRegionCount; ++r) {
            if (page + kPage > g_mkwCodeVariantRegions[r * 2] && page < g_mkwCodeVariantRegions[r * 2 + 1]) {
                pages.push_back(page);
                break;
            }
        }
    }
    std::sort(pages.begin(), pages.end());
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_lastComparedPage = UINT32_MAX;
        g_lastPatchedWordAddr = UINT32_MAX;
        for (const uint32_t page : pages) {
            if (!Memory::Contains(page, kPage)) continue;
            auto& slot = g_pristine[page];
            if (!slot) slot = std::make_unique<std::array<uint8_t, kPage>>();
            g_seen.erase(page);
            std::memcpy(slot->data(), Memory::GetPointer(page, kPage), kPage);
            const uint32_t index = page / kPage;
            g_guardedPages[index / 32].fetch_or(1u << (index % 32), std::memory_order_release);
        }
    }
    for (size_t i = 0; i < pages.size();) {  // consecutive pages as one range
        size_t j = i + 1;
        while (j < pages.size() && pages[j] == pages[j - 1] + kPage) ++j;
        RecompMod::RegisterExecutableRange(pages[i], pages[j - 1] + kPage, "code variant dependencies");
        i = j;
    }
    RT_LOGF(RT_TAG_MOD, "[mods] code variants: %zu page(s) of hook bodies write-trapped\n", pages.size());
}

void DirtyVariantsOnPageLocked(uint32_t page) {
    if (!g_variantsEnabled) return;
    const auto found = g_variantsByPage.find(page);
    if (found == g_variantsByPage.end()) return;
    for (const uint32_t v : found->second) g_variantState[v].store(kVariantUnchecked, std::memory_order_release);
    BumpVariantCallEpoch();
}

bool VariantMatchesLocked(uint32_t v) {
    const MkwCodeVariant& variant = g_mkwCodeVariants[v];
    for (uint32_t g = variant.firstGuard; g < variant.firstGuard + variant.guardCount; ++g) {
        const uint32_t start = g_mkwCodeVariantGuards[g * 3], words = g_mkwCodeVariantGuards[g * 3 + 1];
        const uint32_t* expected = g_mkwCodeVariantWords + g_mkwCodeVariantGuards[g * 3 + 2];
        if (!Memory::Contains(start, words * 4)) return false;
        for (uint32_t k = 0; k < words; ++k) {
            if (Memory::Read32(start + k * 4) != expected[k]) return false;
        }
    }
    return true;
}

// The variant to run for a patched call of `address`: one whose words guest memory holds, else null.
void (*MatchingVariant(uint32_t address))(CpuContext*) {
    if (!g_variantsEnabled) return nullptr;
    uint32_t first = 0;
    for (uint32_t slot = VariantSlot(address) & g_variantSlotMask;; slot = (slot + 1) & g_variantSlotMask) {
        const auto& [key, value] = g_variantSlots[slot];
        if (value == 0) return nullptr;
        if (key == address) {
            first = value - 1;
            break;
        }
    }
    for (uint32_t v = first; v < g_mkwCodeVariantCount && g_mkwCodeVariants[v].address == address; ++v) {
        uint8_t state = g_variantState[v].load(std::memory_order_acquire);
        if (state == kVariantUnchecked) {
            std::lock_guard<std::mutex> lock(g_mutex);
            state = g_variantState[v].load(std::memory_order_relaxed);
            if (state == kVariantUnchecked) {
                state = VariantMatchesLocked(v) ? kVariantMatches : kVariantDiffers;
                Bump(g_stats.checks);
                if (state == kVariantDiffers) Bump(g_stats.checkFailures);
                g_variantState[v].store(state, std::memory_order_release);
                const uint8_t bit = state == kVariantMatches ? 1 : 2;
                if ((g_variantLogged[v].fetch_or(bit, std::memory_order_relaxed) & bit) == 0) {
                    if (state == kVariantMatches) {
                        RT_LOGF(RT_TAG_MOD, "[mods] function 0x%08X runs its code variant (translated as patched)\n",
                                address);
                    } else {
                        RT_LOGF(RT_TAG_MOD, "[mods] function 0x%08X differs from its code variant: interpreted\n",
                                address);
                    }
                }
            }
        }
        if (state == kVariantMatches) return g_mkwCodeVariants[v].entry;
    }
    return nullptr;
}

// The cached (0x8/0x9) view: the trap also fires for the uncached alias of guarded code.
uint32_t Cached(uint32_t address) { return (address & 0xE0000000u) == 0xC0000000u ? address - 0x40000000u : address; }

bool IsGuardedLocked(uint32_t address) {
    return std::any_of(g_ranges.begin(), g_ranges.end(),
                       [&](const auto& range) { return address >= range.first && address < range.second; });
}

// A page holding some guarded code. An edge page (code and data, e.g. a module's first code page after its header)
// is guarded too: REX hooks sora_menu_sel_stage's constructor there, and unguarded the translated original ran.
bool IsGuardedPageLocked(uint32_t page) {
    return std::any_of(g_ranges.begin(), g_ranges.end(),
                       [&](const auto& range) { return page < range.second && page + kPage > range.first; });
}

bool IsEdgePageLocked(uint32_t page) {
    return IsGuardedPageLocked(page) &&
           std::none_of(g_ranges.begin(), g_ranges.end(),
                        [&](const auto& range) { return page >= range.first && page + kPage <= range.second; });
}

// The bytes of `page` a Sync compares, [first, second) from the page's start: the guarded code on it, or the whole page
// when no guarded range covers it (a code variant's hook bodies) or a code variant has words on it (their words may lie
// anywhere on the page). Words outside guarded code never mark a function patched, so only variants need the rest.
std::pair<uint32_t, uint32_t> CompareSpanLocked(uint32_t page) {
    if (g_variantsEnabled && g_variantsByPage.count(page) != 0) return {0, kPage};
    uint32_t lo = kPage, hi = 0;
    for (const auto& [start, end] : g_ranges) {
        if (end <= page || start >= page + kPage) continue;
        lo = std::min(lo, start > page ? (start - page) & ~3u : 0u);
        hi = std::max(hi, end < page + kPage ? ((end - page) + 3u) & ~3u : kPage);
    }
    if (lo >= hi) return {0, kPage};
    return {lo, hi};
}

// Drain notifications under g_mutex at an ordinary execution boundary. The two
// summary levels avoid scanning the entire guest address space on every Sync.
void CollectPendingPagesLocked() {
    for (uint32_t root = 0; root < std::size(g_pendingRoots); ++root) {
        if (g_pendingRoots[root].load(std::memory_order_relaxed) == 0) continue;
        uint32_t groups = g_pendingRoots[root].exchange(0, std::memory_order_acquire);
        while (groups) {
            const uint32_t group = root * 32 + std::countr_zero(groups);
            groups &= groups - 1;
            uint32_t words = g_pendingGroups[group].exchange(0, std::memory_order_acquire);
            while (words) {
                const uint32_t word = group * 32 + std::countr_zero(words);
                words &= words - 1;
                uint32_t pages = g_pendingPages[word].exchange(0, std::memory_order_acquire);
                while (pages) {
                    g_dirty.push_back((word * 32 + std::countr_zero(pages)) * kPage);
                    pages &= pages - 1;
                }
            }
        }
    }
}

inline void MarkPendingPage(uint32_t address) {
    const uint32_t page = address / kPage;
    const uint32_t word = page / 32;
    const uint32_t bit = 1u << (page % 32);
    g_pendingPages[word].fetch_or(bit, std::memory_order_relaxed);
    g_pendingGroups[word / 32].fetch_or(1u << (word % 32), std::memory_order_release);
    g_pendingRoots[word / 32 / 32].fetch_or(1u << ((word / 32) % 32), std::memory_order_release);
}

// The translated function holding `address`: the last start at or below it.
int64_t FunctionIndex(uint32_t address) {
    const uint32_t* begin = g_mkwGuardedFunctionStarts;
    const uint32_t* end = begin + g_mkwGuardedFunctionCount;
    const uint32_t* it = std::upper_bound(begin, end, address);
    return it == begin ? -1 : (it - begin) - 1;
}

// The callers a patched leaf was inlined into run their copy of the old code: they go to the interpreter too.
void MarkInliningCallers(uint32_t leaf) {
    const auto* pairs = reinterpret_cast<const std::array<uint32_t, 2>*>(g_mkwInlinedLeaves);
    const auto* end = pairs + g_mkwInlinedLeafCount;
    for (auto* it = std::lower_bound(pairs, end, leaf, [](const auto& pair, uint32_t value) { return pair[0] < value; });
         it != end && (*it)[0] == leaf; ++it) {
        const uint32_t caller = (*it)[1];
        if (g_mkwGuestCodePatched[caller]) continue;
        g_mkwGuestCodePatched[caller] = 1;
        RT_LOGF(RT_TAG_MOD, "[mods] function 0x%08X inlined 0x%08X: it now runs in the interpreter too\n",
                g_mkwGuardedFunctionStarts[caller], g_mkwGuardedFunctionStarts[leaf]);
    }
}

// A relinked range drops its interpreter marks, including those it carried only because it inlined a
// leaf patched elsewhere. That leaf is still patched, so its inlining callers in the range go back.
void RemarkInliningCallersIn(uint32_t start, uint32_t end) {
    const auto* pairs = reinterpret_cast<const std::array<uint32_t, 2>*>(g_mkwInlinedLeaves);
    for (uint32_t i = 0; i < g_mkwInlinedLeafCount; ++i) {
        const uint32_t leaf = pairs[i][0], caller = pairs[i][1];
        const uint32_t function = g_mkwGuardedFunctionStarts[caller];
        if (function < start || function >= end || !g_mkwGuestCodePatched[leaf] || g_mkwGuestCodePatched[caller]) continue;
        g_mkwGuestCodePatched[caller] = 1;
        RT_LOGF(RT_TAG_MOD, "[mods] function 0x%08X inlined patched 0x%08X: it runs in the interpreter again after a relink\n",
                function, g_mkwGuardedFunctionStarts[leaf]);
    }
}

// BRAWL_BAKED_PATCHES=<file of "ADDR VALUE" hex lines>: code words a mod writes that were baked into the
// translation (bake_patches.py). The pristine copy of their range holds the baked words, so once the mod has written
// them the code matches what was translated and runs translated; a word the mod writes differently afterwards
// differs from the pristine copy like any other patch, and its function goes to the interpreter. Until the code
// handler's first pass (ReleaseBaked) memory still holds the disc's words, and the baked words branch into hook
// bodies the handler has not copied in yet: every function holding one runs the disc's bytes in the interpreter.
const std::vector<std::pair<uint32_t, uint32_t>>& BakedWords() {
    static const std::vector<std::pair<uint32_t, uint32_t>> words = [] {
        std::vector<std::pair<uint32_t, uint32_t>> parsed;
        const char* path = std::getenv("BRAWL_BAKED_PATCHES");
        if (path == nullptr) return parsed;
        if (FILE* file = std::fopen(path, "r")) {
            unsigned address = 0, value = 0;
            while (std::fscanf(file, "%x %x", &address, &value) == 2) parsed.emplace_back(address, value);
            std::fclose(file);
        }
        std::sort(parsed.begin(), parsed.end());
        RT_LOGF(RT_TAG_MOD, "[mods] %zu baked code word(s) from %s\n", parsed.size(), path);
        return parsed;
    }();
    return words;
}

bool g_bakedReleased = false;
std::vector<uint32_t> g_bakedHeld;  // guard indices held in the interpreter until ReleaseBaked

// The baked words in [start, end), each as the guest stores it.
template <typename Fn>
void ForEachBaked(uint32_t start, uint32_t end, Fn&& fn) {
    const auto& words = BakedWords();
    for (auto it = std::lower_bound(words.begin(), words.end(), std::make_pair(start, 0u));
         it != words.end() && it->first < end; ++it) {
        fn(it->first, __builtin_bswap32(it->second));
    }
}

// Before the first code handler pass: the functions holding baked words in [start, end), and the callers that
// inlined one of them, run the disc's bytes in the interpreter.
void HoldBakedLocked(uint32_t start, uint32_t end);

void SnapshotLocked(uint32_t start, uint32_t end) {
    g_lastComparedPage = UINT32_MAX;
    g_lastPatchedWordAddr = UINT32_MAX;
    for (uint32_t page = start & ~(kPage - 1); page < end; page += kPage) {
        auto& slot = g_pristine[page];
        if (!slot) slot = std::make_unique<std::array<uint8_t, kPage>>();
        g_seen.erase(page);
        if (Memory::Contains(page, kPage)) std::memcpy(slot->data(), Memory::GetPointer(page, kPage), kPage);
    }
    // The pristine state of baked code is the baked word: what the translation was made from. Once the code handler
    // has run (released), freshly linked code gets the words in memory too, as the handler would write them again
    // at its next pass: memory then equals the pristine copy, so a different word the handler writes is noticed.
    ForEachBaked(start & ~(kPage - 1), end, [](uint32_t address, uint32_t stored) {
        const auto found = g_pristine.find(address & ~(kPage - 1));
        if (found != g_pristine.end()) std::memcpy(found->second->data() + (address & (kPage - 1)), &stored, 4);
        if (g_bakedReleased && Memory::Contains(address, 4)) std::memcpy(Memory::GetPointer(address, 4), &stored, 4);
    });
    if (!g_bakedReleased) HoldBakedLocked(start, end);
}

void HoldBakedLocked(uint32_t start, uint32_t end) {
    int64_t last = -1;
    ForEachBaked(start, end, [&](uint32_t address, uint32_t) {
        const int64_t index = FunctionIndex(address);
        if (index < 0 || index == last) return;
        last = index;
        if (!g_mkwGuestCodePatched[index]) {
            g_mkwGuestCodePatched[index] = 1;
            g_bakedHeld.push_back(static_cast<uint32_t>(index));
        }
        // Callers holding an inlined copy of this function's baked code wait for the release too.
        const auto* pairs = reinterpret_cast<const std::array<uint32_t, 2>*>(g_mkwInlinedLeaves);
        const auto* pairsEnd = pairs + g_mkwInlinedLeafCount;
        for (auto* it = std::lower_bound(pairs, pairsEnd, static_cast<uint32_t>(index),
                                         [](const auto& pair, uint32_t value) { return pair[0] < value; });
             it != pairsEnd && (*it)[0] == static_cast<uint32_t>(index); ++it) {
            if (!g_mkwGuestCodePatched[(*it)[1]]) {
                g_mkwGuestCodePatched[(*it)[1]] = 1;
                g_bakedHeld.push_back((*it)[1]);
            }
        }
    });
}

} // namespace

namespace GuestCodePatch {

bool Active() { return g_active.load(std::memory_order_acquire); }

// Every translated function starting in [start, end) runs in the interpreter from now on, through module relinks.
uint32_t InterpretRangeLocked(uint32_t start, uint32_t end) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < g_mkwGuardedFunctionCount; ++i) {
        const uint32_t function = g_mkwGuardedFunctionStarts[i];
        if (function < start || function >= end) continue;
        // A veneer is the translator's own stand-in for a far call; nothing executable sits at its address.
        const auto* info = TranslatedFunctionRegistry::FindByAddressPtr(function);
        if (info != nullptr && info->name != nullptr && std::strstr(info->name, "_veneer_") != nullptr) continue;
        g_mkwGuestCodePatched[i] = 1;
        g_forced.push_back(i);
        MarkInliningCallers(i);
        ++count;
    }
    return count;
}

void InterpretFromEnvironment() {
    const char* text = std::getenv("BRAWL_INTERPRET");
    if (text == nullptr) return;
    char* dash = nullptr;
    const uint32_t start = static_cast<uint32_t>(std::strtoul(text, &dash, 16));
    const uint32_t end = *dash == '-' ? static_cast<uint32_t>(std::strtoul(dash + 1, nullptr, 16)) : 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    const uint32_t count = InterpretRangeLocked(start, end);
    RT_LOGF(RT_TAG_MOD, "[mods] BRAWL_INTERPRET: %u translated function(s) in 0x%08X-0x%08X run in the interpreter\n",
            count, start, end);
}

void InterpretReplacedModule(uint32_t start, uint32_t end, const char* name) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const uint32_t count = InterpretRangeLocked(start, end);
    RT_LOGF(RT_TAG_MOD, "[mods] module %s replaced by a mod at 0x%08X-0x%08X: its %u translated function(s) now run "
            "the mod's code in the interpreter\n", name, start, end, count);
}

void InterpretForHooks(uint32_t address, const char* why) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const uint32_t count = InterpretRangeLocked(address, address + 1);
    RT_LOGF(RT_TAG_MOD, "[mods] 0x%08X runs in the interpreter (%s)%s\n", address, why,
            count == 1 ? "" : ": no translated function starts there");
}

// BRAWL_TRAP_STATS=1 (diagnostic): write notifications per guarded page, the busiest logged every 600 frames (in the
// flat build each one is a write trap: an exception and a page protection change, and another to guard it again).
// Project+ keeps a variable in sora_melee's code (0x807824D4, a word its hook jumps over): ~70 traps a frame.
struct TrapStat {
    std::atomic<uint32_t> count{0};
    std::atomic<uint32_t> last{0};  // the last address written there
    std::atomic<uint32_t> pc{0};    // the guest pc then (an interpreted block's start)
};
std::atomic<TrapStat*> g_trapStats{nullptr};  // one per guest page, allocated by Protect when asked for
bool TrapStats() {
    static const bool on = std::getenv("BRAWL_TRAP_STATS") != nullptr;
    return on;
}

void Protect(uint32_t start, uint32_t end, const char* name) {
    std::call_once(g_variantsOnce, [] {
        InitVariants();
        ProtectVariantRegions();
        if (!g_variantsEnabled) {
            PpcInterpreter::detail::PatchedHooks hooks;
            hooks.patchedByteFor = [](uint32_t address) -> const uint8_t* {
                const uint32_t* begin = g_mkwGuardedFunctionStarts;
                const uint32_t* end = begin + g_mkwGuardedFunctionCount;
                const uint32_t* it = std::lower_bound(begin, end, address);
                if (it == end || *it != address) return nullptr;
                return &g_mkwGuestCodePatched[static_cast<size_t>(it - begin)];
            };
            hooks.pendingFast = &g_pendingFast;
            hooks.anyDirty = &g_anyDirty;
            hooks.lowestRedirect = &g_lowestRedirect;
            hooks.sync = &Sync;
            PpcInterpreter::detail::RegisterPatchedHooks(hooks);
        }
    });
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (std::find(g_ranges.begin(), g_ranges.end(), std::make_pair(start, end)) != g_ranges.end()) return;
        g_ranges.emplace_back(start, end);
        // NoteWrite runs in fault recovery and must not allocate: the diagnostic's table is made here, once.
        if (TrapStats() && g_trapStats.load(std::memory_order_relaxed) == nullptr)
            g_trapStats.store(new TrapStat[kPageWords * 32], std::memory_order_release);
        SnapshotLocked(start, end);
        for (uint64_t page = start / kPage; page * kPage < end; ++page) {
            g_guardedPages[page / 32].fetch_or(1u << (page % 32), std::memory_order_release);
        }
        g_active.store(true, std::memory_order_release);
    }
    RecompMod::RegisterExecutableRange(start, end, name);
}

bool NoteWrite(uint32_t address) {
    if (!Active()) return false;
    address = Cached(address);
    const uint32_t page = address / kPage;
    const uint32_t word = page / 32;
    const uint32_t bit = 1u << (page % 32);
    if ((g_guardedPages[word].load(std::memory_order_acquire) & bit) == 0) return false;
    if (__builtin_expect(g_trapStats.load(std::memory_order_acquire) != nullptr, 0)) {
        TrapStat* const stats = g_trapStats.load(std::memory_order_acquire);
        stats[page].count.fetch_add(1, std::memory_order_relaxed);
        stats[page].last.store(address, std::memory_order_relaxed);
        if (const CpuContext* c = TryGetCpuContext()) stats[page].pc.store(c->pc, std::memory_order_relaxed);
    }
    // Lean-memory single-word fast path: when only one aligned 8-byte span inside a page is pending (e.g. a mod
    // variable embedded inside a code page), record its address directly and skip the 3-level bitmap and 4 KiB scan.
    if (!GuestFlat::IsActive() && ((address & (kPage - 1)) <= kPage - 8)) {
        const uint64_t tag = static_cast<uint64_t>(address & ~3u) | (uint64_t{1} << 32);
        uint64_t cur = g_pendingFast.load(std::memory_order_relaxed);
        if (cur == tag &&
            g_pendingFast.compare_exchange_strong(cur, tag, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            HZ_PERF_ADD(kHzNoteWrite, 1);
            Bump(g_stats.noteWrites);
            return true;
        }
        if (cur == 0 &&
            g_pendingFast.compare_exchange_strong(cur, tag, std::memory_order_release, std::memory_order_relaxed)) {
            HZ_PERF_ADD(kHzNoteWrite, 1);
            Bump(g_stats.noteWrites);
            return true;
        }
        if (cur != ~uint64_t{0} && cur != 0) {
            MarkPendingPage(static_cast<uint32_t>(cur));
        }
    }
    MarkPendingPage(address);
    g_anyDirty.store(true, std::memory_order_release);
    g_pendingFast.store(~uint64_t{0}, std::memory_order_release);
    HZ_PERF_ADD(kHzNoteWrite, 1);
    Bump(g_stats.noteWrites);
    return true;
}

namespace {

// An edge page (guarded code and data on one page) whose data stores would trap again after every Sync (and Sync
// follows every patched call): it stays open, its code compared at each Sync, until the frame's SyncFrame guards it
// again. Kept apart from the write notifications, so that a Sync with only open edge pages to look at is a compare of
// their code with what the last Sync saw (no notification scan, no set or allocation): with one open page that was
// 20% of a P+ match's game thread at ~1,700 patched calls a frame.
struct OpenEdge {
    uint32_t page;
    uint32_t lo, hi;  // CompareSpanLocked
};
std::vector<OpenEdge> g_openEdges;          // under g_mutex
std::atomic<uint32_t> g_openEdgeCount{0};   // g_openEdges.size(), read without the lock

void ForgetOpenEdgeLocked(uint32_t page) {
    g_openEdges.erase(std::remove_if(g_openEdges.begin(), g_openEdges.end(),
                                     [&](const OpenEdge& e) { return e.page == page; }),
                      g_openEdges.end());
    g_openEdgeCount.store(static_cast<uint32_t>(g_openEdges.size()), std::memory_order_release);
}

// Compares a written page's code with what the last Sync saw there and handles every changed word.
void ComparePageLocked(uint32_t page) {
    Bump(g_stats.pagesCompared);
    const uint8_t* was = nullptr;
    uint8_t* seen = nullptr;
    uint32_t spanStart = 0, spanEnd = 0;
    if (page == g_lastComparedPage && Memory::Contains(page, kPage)) {
        was = g_lastPristine;
        seen = g_lastSeen;
        spanStart = g_lastSpanLo;
        spanEnd = g_lastSpanHi;
    } else {
        const auto found = g_pristine.find(page);
        if (found == g_pristine.end() || !Memory::Contains(page, kPage)) return;
        was = found->second->data();
        auto& seenSlot = g_seen[page];
        if (!seenSlot) seenSlot = std::make_unique<std::array<uint8_t, kPage>>(*found->second);
        seen = seenSlot->data();
        // Only the page's code: an edge page's data (a module's variables after its last code, main.dol's data
        // after .text) changes all the time, and comparing it made every Sync walk the page word by word.
        const auto span = CompareSpanLocked(page);
        spanStart = span.first;
        spanEnd = span.second;
        g_lastComparedPage = page;
        g_lastPristine = was;
        g_lastSeen = seen;
        g_lastSpanLo = spanStart;
        g_lastSpanHi = spanEnd;
    }
    const uint8_t* now = Memory::GetPointer(page, kPage);
    if (std::memcmp(now + spanStart, seen + spanStart, spanEnd - spanStart) == 0) return;
    Bump(g_stats.pagesChanged);
    DirtyVariantsOnPageLocked(page);
    const auto handleWord = [&](uint32_t offset) {
        if (std::memcmp(now + offset, seen + offset, 4) == 0) return;
        std::memcpy(seen + offset, now + offset, 4);
        if (std::memcmp(now + offset, was + offset, 4) == 0) return;
        const uint32_t address = page + offset;
        if (!IsGuardedLocked(address)) return;
        std::memcpy(&g_sites[address], now + offset, 4);
        // BRAWL_PATCH_LOG: every changed word, once (the pristine copy stays the disc's code).
        static const bool logWords = std::getenv("BRAWL_PATCH_LOG") != nullptr;
        if (logWords && g_loggedWords.insert(address).second) {
            uint32_t before = 0, after = 0;
            std::memcpy(&before, was + offset, 4);
            std::memcpy(&after, now + offset, 4);
            RT_LOGF(RT_TAG_MOD, "[mods] word 0x%08X: %08X -> %08X\n", address, __builtin_bswap32(before),
                    __builtin_bswap32(after));
        }
        const int64_t index = FunctionIndex(address);
        if (index < 0 || g_mkwGuestCodePatched[index]) return;
        g_mkwGuestCodePatched[index] = 1;
        const uint32_t function = g_mkwGuardedFunctionStarts[index];
        if (g_reported.insert(function).second) {
            RT_LOGF(RT_TAG_MOD, "[mods] code at 0x%08X changed: function 0x%08X now runs in the interpreter\n",
                    address, function);
        }
        MarkInliningCallers(static_cast<uint32_t>(index));
    };
    uint32_t offset = spanStart;
    for (; offset + 8 <= spanEnd; offset += 8) {
        uint64_t a, b;
        std::memcpy(&a, now + offset, 8);
        std::memcpy(&b, seen + offset, 8);
        if (__builtin_expect(a == b, 1)) continue;
        handleWord(offset);
        handleWord(offset + 4);
    }
    for (; offset < spanEnd; offset += 4) handleWord(offset);
}

// Fast single-span compare for lean-memory NoteWrite: checks only the 8 bytes [wordAddr, wordAddr + 8) instead of
// scanning 4 KiB. Caches the last already-patched word so repeated writes to a mod variable inside code cost 2 compares.
void CompareWordPairLocked(uint32_t wordAddr) {
    Bump(g_stats.pagesCompared);
    if (__builtin_expect(wordAddr == g_lastPatchedWordAddr && !g_variantsEnabled && Memory::Contains(wordAddr, 8), 1)) {
        const uint8_t* const now = Memory::GetPointer(wordAddr, 8);
        uint32_t now0, now1, seen0, seen1;
        std::memcpy(&now0, now, 4);
        std::memcpy(&now1, now + 4, 4);
        std::memcpy(&seen0, g_lastPatchedSeenPtr, 4);
        std::memcpy(&seen1, g_lastPatchedSeenPtr + 4, 4);
        if (__builtin_expect(now1 == seen1, 1)) {
            if (now0 != seen0) {
                Bump(g_stats.pagesChanged);
                std::memcpy(g_lastPatchedSeenPtr, &now0, 4);
                std::memcpy(g_lastPatchedSitePtr, &now0, 4);
            }
            return;
        }
    }
    const uint32_t page = wordAddr & ~(kPage - 1);
    const uint32_t off = wordAddr & (kPage - 1);
    const uint8_t* was = nullptr;
    uint8_t* seen = nullptr;
    uint32_t spanStart = 0, spanEnd = 0;
    if (page == g_lastComparedPage && Memory::Contains(page, kPage)) {
        was = g_lastPristine;
        seen = g_lastSeen;
        spanStart = g_lastSpanLo;
        spanEnd = g_lastSpanHi;
    } else {
        const auto found = g_pristine.find(page);
        if (found == g_pristine.end() || !Memory::Contains(page, kPage)) return;
        was = found->second->data();
        auto& seenSlot = g_seen[page];
        if (!seenSlot) seenSlot = std::make_unique<std::array<uint8_t, kPage>>(*found->second);
        seen = seenSlot->data();
        const auto span = CompareSpanLocked(page);
        spanStart = span.first;
        spanEnd = span.second;
        g_lastComparedPage = page;
        g_lastPristine = was;
        g_lastSeen = seen;
        g_lastSpanLo = spanStart;
        g_lastSpanHi = spanEnd;
    }
    const uint8_t* now = Memory::GetPointer(page, kPage);
    if (std::memcmp(now + off, seen + off, 8) == 0) return;
    Bump(g_stats.pagesChanged);
    DirtyVariantsOnPageLocked(page);
    static const bool logWords = std::getenv("BRAWL_PATCH_LOG") != nullptr;
    for (uint32_t delta = 0; delta < 8; delta += 4) {
        const uint32_t offset = off + delta;
        if (offset < spanStart || offset >= spanEnd) continue;
        if (std::memcmp(now + offset, seen + offset, 4) == 0) continue;
        std::memcpy(seen + offset, now + offset, 4);
        if (std::memcmp(now + offset, was + offset, 4) == 0) continue;
        const uint32_t address = page + offset;
        if (!IsGuardedLocked(address)) continue;
        uint32_t& site = g_sites[address];
        std::memcpy(&site, now + offset, 4);
        if (logWords && g_loggedWords.insert(address).second) {
            uint32_t before = 0, after = 0;
            std::memcpy(&before, was + offset, 4);
            std::memcpy(&after, now + offset, 4);
            RT_LOGF(RT_TAG_MOD, "[mods] word 0x%08X: %08X -> %08X\n", address, __builtin_bswap32(before),
                    __builtin_bswap32(after));
        }
        const int64_t index = FunctionIndex(address);
        if (index < 0) continue;
        if (g_mkwGuestCodePatched[index]) {
            if (delta == 0 && !g_variantsEnabled && !logWords) {
                g_lastPatchedWordAddr = wordAddr;
                g_lastPatchedSitePtr = &site;
                g_lastPatchedSeenPtr = seen + offset;
            }
            continue;
        }
        g_mkwGuestCodePatched[index] = 1;
        const uint32_t function = g_mkwGuardedFunctionStarts[index];
        if (g_reported.insert(function).second) {
            RT_LOGF(RT_TAG_MOD, "[mods] code at 0x%08X changed: function 0x%08X now runs in the interpreter\n",
                    address, function);
        }
        MarkInliningCallers(static_cast<uint32_t>(index));
    }
}

void SyncPages(bool frame) {
    // Sync follows every patched call: plain loads keep the usual nothing-pending case off the atomic exchanges.
    const uint64_t fast = g_pendingFast.load(std::memory_order_acquire);
    const bool dirtyFlag = g_anyDirty.load(std::memory_order_acquire);
    const uint32_t openEdges = g_openEdgeCount.load(std::memory_order_acquire);
    if (fast == 0 && !dirtyFlag && openEdges == 0) return;
    // Fast single-word path (lean memory, no bitmap overflow, no open edges).
    if (!frame && fast != 0 && fast != ~uint64_t{0} && !dirtyFlag && openEdges == 0) {
        const uint64_t took = g_pendingFast.exchange(0, std::memory_order_acq_rel);
        if (took != 0 && took != ~uint64_t{0}) {
            std::lock_guard<std::mutex> lock(g_mutex);
            HZ_PERF_ADD(kHzSyncWork, 1);
            CompareWordPairLocked(static_cast<uint32_t>(took));
            HZ_PERF_ADD(kHzPagesCompared, 1);
            return;
        }
    }
    const uint64_t tookFast = fast != 0 ? g_pendingFast.exchange(0, std::memory_order_acq_rel) : 0;
    const bool pending =
        g_anyDirty.load(std::memory_order_acquire) && g_anyDirty.exchange(false, std::memory_order_acq_rel);
    const bool hasFastWord = tookFast != 0 && tookFast != ~uint64_t{0};
    if (!pending && !hasFastWord && g_openEdgeCount.load(std::memory_order_acquire) == 0) return;
    const bool flatActive = GuestFlat::IsActive();
    std::vector<uint32_t> reprotect;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        HZ_PERF_ADD(kHzSyncWork, 1);
        if (hasFastWord) {
            CompareWordPairLocked(static_cast<uint32_t>(tookFast));
            HZ_PERF_ADD(kHzPagesCompared, 1);
        }
        // Pages a notification brought (each compared here) and, below, the open edge pages it did not.
        uint32_t written[8];
        uint32_t writtenCount = 0;
        if (pending) {
            CollectPendingPagesLocked();
            for (const uint32_t page : g_dirty) {
                ComparePageLocked(page);
                if (writtenCount < std::size(written)) written[writtenCount++] = page;
                // Only where a store to guarded code traps (the flat build's page protection): a build that notices
                // every store without a trap (lean memory's checked stores) compares an edge page when it is written.
                if (flatActive) {
                    if (!frame && IsEdgePageLocked(page)) {
                        if (std::none_of(g_openEdges.begin(), g_openEdges.end(),
                                         [&](const OpenEdge& e) { return e.page == page; })) {
                            const auto [lo, hi] = CompareSpanLocked(page);
                            g_openEdges.push_back({page, lo, hi});
                        }
                    } else {
                        reprotect.push_back(page);
                    }
                }
            }
            HZ_PERF_ADD(kHzPagesCompared, g_dirty.size());
            g_dirty.clear();
        }
        if (flatActive || !g_openEdges.empty()) {
            const auto comparedAbove = [&](uint32_t page) {
                return std::find(written, written + writtenCount, page) != written + writtenCount;
            };
            if (frame) {
                for (const OpenEdge& e : g_openEdges) {
                    if (!comparedAbove(e.page)) ComparePageLocked(e.page);
                    if (flatActive) reprotect.push_back(e.page);
                }
                g_openEdges.clear();
            } else {
                for (const OpenEdge& e : g_openEdges) {
                    if (comparedAbove(e.page)) continue;
                    const auto seen = g_seen.find(e.page);
                    if (seen != g_seen.end() && seen->second && Memory::Contains(e.page, kPage) &&
                        std::memcmp(Memory::GetPointer(e.page, kPage) + e.lo, seen->second->data() + e.lo, e.hi - e.lo) == 0)
                        continue;
                    ComparePageLocked(e.page);
                }
            }
            g_openEdgeCount.store(static_cast<uint32_t>(g_openEdges.size()), std::memory_order_release);
            HZ_PERF_SET(kHzEdgeDirty, g_openEdges.size());
        }
    }
    if (flatActive) {
        for (const uint32_t page : reprotect) GuestFlat::ReprotectExecutablePage(page);
    }
}

} // namespace

void ReleaseBaked() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_bakedReleased) return;
    g_bakedReleased = true;
    if (g_bakedHeld.empty()) return;
    // Whether guest memory holds a function's pristine bytes (the baked words included) over its whole range.
    const auto matchesPristine = [](uint32_t index) {
        const uint32_t start = g_mkwGuardedFunctionStarts[index];
        const uint32_t end = index + 1 < g_mkwGuardedFunctionCount ? g_mkwGuardedFunctionStarts[index + 1] : start + 4;
        for (uint32_t address = start; address < end; address += 4) {
            const auto found = g_pristine.find(address & ~(kPage - 1));
            if (found == g_pristine.end() || !Memory::Contains(address, 4)) continue;
            if (std::memcmp(Memory::GetPointer(address, 4), found->second->data() + (address & (kPage - 1)), 4) != 0)
                return false;
        }
        return true;
    };
    std::vector<uint32_t> differing;
    for (const uint32_t index : g_bakedHeld) {
        if (!matchesPristine(index)) differing.push_back(index);
    }
    uint32_t released = 0;
    for (const uint32_t index : g_bakedHeld) {
        if (std::find(differing.begin(), differing.end(), index) != differing.end()) continue;
        if (std::find(g_forced.begin(), g_forced.end(), index) != g_forced.end()) continue;
        g_mkwGuestCodePatched[index] = 0;
        ++released;
    }
    for (const uint32_t index : differing) MarkInliningCallers(index);  // their inlining callers stay with them
    RT_LOGF(RT_TAG_MOD, "[mods] baked code: %u function(s) run translated, %zu differ from the bake and stay in the "
            "interpreter\n", released, differing.size());
    g_bakedHeld.clear();
}

void Sync() {
    HZ_PERF_ADD(kHzSync, 1);
    HZ_PERF_T0();
    SyncPages(false);
    HZ_PERF_SINCE(kHzSyncTicks);
}

void SyncFrame() {
#if MKW_INTERP_PROFILE
    PpcInterpreter::ProfileFrame();
#endif
    // A code variant handing guest code to the interpreter through a call that never returns (a hook whose code does
    // not come back to the call's lr) leaves its host frames behind each time and the game thread's stack creeps down
    // frame after frame until it overflows: say so once, while the log can still show which variants were running.
    if (g_variantsEnabled) {
        static thread_local uintptr_t firstFrame = 0;
        static std::atomic<bool> warned{false};
        const auto here = reinterpret_cast<uintptr_t>(__builtin_frame_address(0));
        if (firstFrame == 0) firstFrame = here;
        if (here + (1u << 20) < firstFrame && !warned.exchange(true)) {
            RT_LOGF(RT_TAG_MOD, "[mods] host stack grew %zu KiB across frames: a code variant's call into the "
                    "interpreter never returned\n", static_cast<size_t>((firstFrame - here) >> 10));
        }
    }
    // BRAWL_PATCH_STATS=<file>: "variant-calls interpreter-calls checks check-failures note-writes pages-compared
    // pages-changed interpreted-instructions" for the frame that ends here.
    static FILE* const stats = [] {
        const char* path = std::getenv("BRAWL_PATCH_STATS");
        return path ? std::fopen(path, "w") : nullptr;
    }();
    if (stats != nullptr) {
        static uint64_t lastInstructions = 0;
        const uint64_t instructions = PpcInterpreter::InstructionCount();
        auto take = [](std::atomic<uint32_t>& c) { return c.exchange(0, std::memory_order_relaxed); };
        std::fprintf(stats, "%u %u %u %u %u %u %u %llu\n", take(g_stats.variantCalls), take(g_stats.interpreterCalls),
                     take(g_stats.checks), take(g_stats.checkFailures), take(g_stats.noteWrites),
                     take(g_stats.pagesCompared), take(g_stats.pagesChanged),
                     static_cast<unsigned long long>(instructions - lastInstructions));
        lastInstructions = instructions;
    }
    if (TrapStat* const traps = g_trapStats.load(std::memory_order_acquire)) {
        static uint32_t frames = 0;
        if (++frames % 600 == 0) {
            std::vector<std::pair<uint32_t, uint32_t>> busiest;
            for (uint32_t page = 0; page < kPageWords * 32; ++page) {
                const uint32_t n = traps[page].count.exchange(0, std::memory_order_relaxed);
                if (n) busiest.emplace_back(n, page);
            }
            std::sort(busiest.rbegin(), busiest.rend());
            std::string line;
            for (size_t i = 0; i < busiest.size() && i < 12; ++i) {
                char item[64];
                const TrapStat& t = traps[busiest[i].second];
                std::snprintf(item, sizeof item, " %08X:%u(last %08X pc %08X)", busiest[i].second * kPage,
                              busiest[i].first, t.last.load(std::memory_order_relaxed),
                              t.pc.load(std::memory_order_relaxed));
                line += item;
            }
            RT_LOGF(RT_TAG_MOD, "[mods] trap stats, 600 frames: %zu pages,%s\n", busiest.size(), line.c_str());
        }
    }
    HZ_PERF_ADD(kHzFrames, 1);
    HZ_PERF_T0();
    SyncPages(true);
    HZ_PERF_SINCE(kHzSyncTicks);
}

void ResetPristine(uint32_t start, uint32_t end) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        SnapshotLocked(start, end);
        for (int64_t index = FunctionIndex(start); index >= 0 && index < static_cast<int64_t>(g_mkwGuardedFunctionCount); ++index) {
            const uint32_t function = g_mkwGuardedFunctionStarts[index];
            if (function >= end) break;
            if (function >= start) {
                g_mkwGuestCodePatched[index] = 0;
                g_reported.erase(function);  // a patch of the reloaded code is news again
            }
        }
        for (const uint32_t index : g_forced) {  // the interpreter check outlives a module's relink
            const uint32_t function = g_mkwGuardedFunctionStarts[index];
            if (function >= start && function < end) g_mkwGuestCodePatched[index] = 1;
        }
        RemarkInliningCallersIn(start, end);
        for (uint32_t page = start & ~(kPage - 1); page < end; page += kPage) {
            g_dirty.erase(std::remove(g_dirty.begin(), g_dirty.end(), page), g_dirty.end());
            ForgetOpenEdgeLocked(page);
            const uint32_t index = page / kPage;
            g_pendingPages[index / 32].fetch_and(~(1u << (index % 32)), std::memory_order_relaxed);
            DirtyVariantsOnPageLocked(page);  // the relink copied code without the write trap
        }
        for (auto it = g_loggedWords.begin(); it != g_loggedWords.end();) {
            it = *it >= start && *it < end ? g_loggedWords.erase(it) : std::next(it);
        }
    }
    // Loading the module opened its pages for good: guard them again, or a Gecko hook the code handler writes into
    // the fresh code later (Project+'s stock icon loader in the character select module) goes unnoticed and the
    // translated original runs instead.
    for (uint32_t page = start & ~(kPage - 1); page < end; page += kPage) GuestFlat::ReprotectExecutablePage(page);
}

void RecordPatch(uint32_t address, uint32_t value) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_sites[address] = value;
}

void Reapply(uint32_t start, uint32_t end) {
    std::vector<std::pair<uint32_t, uint32_t>> writes;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        for (const auto& [address, value] : g_sites) {
            if (address >= start && address < end) writes.emplace_back(address, value);
        }    }
    if (writes.empty()) return;
    for (const auto& [address, value] : writes) {  // through the write trap, as the code handler's own stores go
        BrawlDirtyMemory::Mark(address, 4);
        std::memcpy(Memory::GetPointer(address, 4), &value, 4);
        NoteWrite(address);
    }
    Sync();
}

void RedirectModule(uint32_t homeStart, uint32_t homeEnd, uint32_t target, const char* name) {
    std::lock_guard<std::mutex> lock(g_mutex);
    Redirect* slot = nullptr;
    for (Redirect& r : g_redirects) {
        const uint32_t start = r.start.load(std::memory_order_relaxed);
        if (start == homeStart || (slot == nullptr && start == 0)) slot = &r;
    }
    if (slot == nullptr) {
        RT_LOGF(RT_TAG_MOD, "[mods] module %s: no free redirect slot, its home code keeps running\n", name);
        return;
    }
    // Disabled while it changes, so a call never pairs the old target with the new range.
    slot->start.store(0, std::memory_order_release);
    slot->end.store(homeEnd, std::memory_order_relaxed);
    slot->target.store(target, std::memory_order_relaxed);
    if (homeStart < g_lowestRedirect.load(std::memory_order_relaxed)) {  // lowered before the slot is live
        g_lowestRedirect.store(homeStart, std::memory_order_release);
    }
    slot->start.store(homeStart, std::memory_order_release);
    UpdateLowestRedirectLocked();
    BumpVariantCallEpoch();
    uint32_t count = 0;
    for (int64_t index = std::max<int64_t>(FunctionIndex(homeStart), 0);
         index < static_cast<int64_t>(g_mkwGuardedFunctionCount); ++index) {
        const uint32_t function = g_mkwGuardedFunctionStarts[index];
        if (function >= homeEnd) break;
        if (function < homeStart || g_mkwGuestCodePatched[index]) continue;
        g_mkwGuestCodePatched[index] = 1;
        MarkInliningCallers(static_cast<uint32_t>(index));
        ++count;
    }
    RT_LOGF(RT_TAG_MOD, "[mods] module %s linked at 0x%08X: calls into its home 0x%08X-0x%08X (%u function(s)) run it "
            "there\n", name, target, homeStart, homeEnd, count);
}

void ClearRedirect(uint32_t homeStart) {
    std::lock_guard<std::mutex> lock(g_mutex);
    for (Redirect& r : g_redirects) {
        if (r.start.load(std::memory_order_relaxed) != homeStart) continue;
        const uint32_t end = r.end.load(std::memory_order_relaxed);
        r.start.store(0, std::memory_order_release);
        for (int64_t index = std::max<int64_t>(FunctionIndex(homeStart), 0);
             index < static_cast<int64_t>(g_mkwGuardedFunctionCount); ++index) {
            const uint32_t function = g_mkwGuardedFunctionStarts[index];
            if (function >= end) break;
            if (function >= homeStart) g_mkwGuestCodePatched[index] = 0;
        }
        for (const uint32_t index : g_forced) {
            const uint32_t function = g_mkwGuardedFunctionStarts[index];
            if (function >= homeStart && function < end) g_mkwGuestCodePatched[index] = 1;
        }
        RemarkInliningCallersIn(homeStart, end);
    }
    UpdateLowestRedirectLocked();  // raised only once the slot is off
    BumpVariantCallEpoch();
}

void (*ContinuationAt(uint32_t address))(CpuContext*) { return MatchingVariant(address); }

void ModuleLinkedInPlace(uint32_t start, uint32_t end) {
    if (!g_variantsEnabled || end <= start) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    for (uint32_t page = start & ~(kPage - 1); page < end; page += kPage) DirtyVariantsOnPageLocked(page);
}

namespace {

inline bool SyncNeeded() {
    return g_pendingFast.load(std::memory_order_acquire) != 0 || g_anyDirty.load(std::memory_order_acquire) ||
           g_openEdgeCount.load(std::memory_order_acquire) != 0;
}

MKW_PPC_NO_INLINE uint32_t ResolveRedirect(uint32_t address, bool* redirected = nullptr) {
    for (const Redirect& r : g_redirects) {
        const uint32_t start = r.start.load(std::memory_order_acquire);
        if (start != 0 && address >= start && address < r.end.load(std::memory_order_relaxed)) {
            if (redirected) *redirected = true;
            return address - start + r.target.load(std::memory_order_relaxed);
        }
    }
    return address;
}

MKW_PPC_NO_INLINE void RunPatchedWithVariantsSlow(CpuContext* ctx, uint32_t address, VariantCallCacheSlot& cached,
                                                  uint32_t epoch) {
    bool redirected = false;
    if (__builtin_expect(address >= g_lowestRedirect.load(std::memory_order_acquire), 0)) {
        address = ResolveRedirect(address, &redirected);
    }
    // A function patched exactly as the build saw the mod patch it runs that translation (code variants).
    if (!redirected) {
        if (const auto variant = MatchingVariant(address)) {
            // Under the epoch read before the lookup: a change racing it leaves the slot stale, never wrong.
            cached = {address, epoch, variant};
            Bump(g_stats.variantCalls);
            variant(ctx);
            if (__builtin_expect(SyncNeeded(), 0)) Sync();
            return;
        }
    }
    Bump(g_stats.interpreterCalls);
    HZ_PERF_ADD(kHzPatched, 1);
    HZ_PHASE(kHzPatchedTopTicks);
    {
        HZ_PERF_T0();
        if (!PpcInterpreter::detail::RunCompiledEntry(ctx, address)) PpcInterpreter::Run(ctx, address);
        HZ_PERF_SINCE(kHzPatchedTicks);
        HZ_PERF_SET(kHzInterpInstructions, PpcInterpreter::InstructionCount());
    }
    if (__builtin_expect(SyncNeeded(), 0)) Sync();
}

MKW_PPC_NO_INLINE void RunPatchedFallback(CpuContext* ctx, uint32_t address) {
    PpcInterpreter::Run(ctx, address);
}

}  // namespace

void RunPatched(CpuContext* ctx, uint32_t address) {
    if (g_variantsEnabled) {
        VariantCallCacheSlot& cached = t_variantCallCache[(address >> 2) & (kVariantCallCacheSlots - 1)];
        const uint32_t epoch = g_variantCallEpoch.load(std::memory_order_acquire);
        if (__builtin_expect(cached.address == address && cached.epoch == epoch, 1)) {
            Bump(g_stats.variantCalls);
            cached.entry(ctx);
            if (__builtin_expect(SyncNeeded(), 0)) Sync();
            return;
        }
        RunPatchedWithVariantsSlow(ctx, address, cached, epoch);
        return;
    }
    if (__builtin_expect(address >= g_lowestRedirect.load(std::memory_order_acquire), 0)) {
        address = ResolveRedirect(address);
    }
    Bump(g_stats.interpreterCalls);
    HZ_PERF_ADD(kHzPatched, 1);
    HZ_PHASE(kHzPatchedTopTicks);
    {
        HZ_PERF_T0();
        if (__builtin_expect(!PpcInterpreter::detail::RunCompiledEntry(ctx, address), 0)) {
            RunPatchedFallback(ctx, address);
        }
        HZ_PERF_SINCE(kHzPatchedTicks);
        HZ_PERF_SET(kHzInterpInstructions, PpcInterpreter::InstructionCount());
    }
    // A patched function is usually a mod hook, and what it called may have patched more code (a Gecko hook runs
    // the code handler): that takes effect now, as the handler's isync makes it on a Wii.
    if (__builtin_expect(SyncNeeded(), 0)) Sync();
}

} // namespace GuestCodePatch
