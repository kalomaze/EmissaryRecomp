#include "guest_memory_alias.h"
#include "brawl_dirty_memory.h"
// guest_code_patch.cpp - routes translated functions whose guest code a mod changed at run time to the interpreter.
// See include/guest_code_patch.h.

#include "guest_code_patch.h"
#include "guest_native_trampoline.h"

#include "abi_bridge.h"
#include "guest_flat_memory.h"
#include "hz_perf.h"
#include "memory.h"
#include "ppc_interpreter.h"
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

std::vector<GuestCodePatch::OriginalNativeTarget>& OriginalNatives() {
    static std::vector<GuestCodePatch::OriginalNativeTarget> entries;
    return entries;
}

std::atomic<bool> g_active{false};
std::atomic<bool> g_anyDirty{false};  // pending write notifications or retained edge pages
// NoteWrite runs inside memory-fault recovery. Its lookup and notification must
// neither allocate nor wait for a lock held by the interrupted thread.
static_assert(std::atomic<uint32_t>::is_always_lock_free);
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
std::unordered_set<uint32_t> g_dirty;                                              // page addresses written since Sync
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
inline void Bump(std::atomic<uint32_t>& counter, uint32_t n = 1) {
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
uint32_t Cached(uint32_t address) { return GuestMemoryAlias::Cached(address); }

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

// Drain notifications under g_mutex at an ordinary execution boundary. The two
// summary levels avoid scanning the entire guest address space on every Sync.
void CollectPendingPagesLocked() {
    for (uint32_t root = 0; root < std::size(g_pendingRoots); ++root) {
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
                    g_dirty.insert((word * 32 + std::countr_zero(pages)) * kPage);
                    pages &= pages - 1;
                }
            }
        }
    }
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
    for (uint32_t page = start & ~(kPage - 1); page < end; page += kPage) {
        auto& slot = g_pristine[page];
        const bool fresh = !slot;
        if (fresh) slot = std::make_unique<std::array<uint8_t, kPage>>();
        g_seen.erase(page);
        // A module relink can share an edge page with unrelated patched code. Only
        // refresh the requested bytes; never bless neighboring patches as pristine.
        const uint32_t first = fresh ? page : std::max(start, page);
        const uint32_t last = fresh ? page + kPage : std::min(end, page + kPage);
        if (Memory::Contains(first, last - first))
            std::memcpy(slot->data() + first - page, Memory::GetPointer(first, last - first), last - first);
    }
    // The pristine state of baked code is the baked word: what the translation was made from. Once the code handler
    // has run (released), freshly linked code gets the words in memory too, as the handler would write them again
    // at its next pass: memory then equals the pristine copy, so a different word the handler writes is noticed.
    ForEachBaked(start, end, [](uint32_t address, uint32_t stored) {
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

bool AnyPatchedInRange(uint32_t start, uint32_t end) {
    if (start >= end || g_mkwGuardedFunctionCount == 0) return false;
    const auto first = std::max<int64_t>(0, FunctionIndex(start));
    for (uint32_t i = static_cast<uint32_t>(first);
         i < g_mkwGuardedFunctionCount && g_mkwGuardedFunctionStarts[i] < end; ++i) {
        if (g_mkwGuestCodePatched[i]) return true;
    }
    return false;
}

void RegisterOriginalNative(uint32_t address, void (*entry)(CpuContext*)) {
    // Native hooks register during startup, before guest execution.
    OriginalNatives().push_back({address, entry});
}

OriginalNativeTarget ResolveOriginalNativeTrampoline(uint32_t trampoline) {
    static const bool enabled = [] {
        const char* value = std::getenv("MKW_NATIVE_TRAMPOLINES");
        return value == nullptr || *value != '0';
    }();
    if (!enabled || !Active() || OriginalNatives().empty() || (trampoline & 3u) || !Memory::Contains(trampoline, 8)) return {};
    // A displaced, location-independent stack prologue followed by b original+4. Do not shortcut arbitrary mod
    // functions or relocate PC-relative instructions. Re-read both words on every call, including cache hits above us.
    const uint32_t first = Memory::Read32(trampoline), branch = Memory::Read32(trampoline + 4);
    if ((first & 0xFFFF0000u) != 0x94210000u || (branch & 0xFC000003u) != 0x48000000u) return {};
    int32_t displacement = static_cast<int32_t>(branch & 0x03FFFFFCu);
    if (displacement & 0x02000000) displacement -= 0x04000000;
    const uint32_t original = trampoline + static_cast<uint32_t>(displacement);
    const auto found = std::find_if(OriginalNatives().begin(), OriginalNatives().end(),
                                    [original](const auto& entry) { return entry.address == original; });
    if (found == OriginalNatives().end()) return {};

    // An interpreted mod can patch an inlined leaf and call this trampoline
    // before returning to RunPatched. Refresh dependency bits at this boundary.
    Sync();
    std::lock_guard<std::mutex> lock(g_mutex);
    const int64_t index = FunctionIndex(original);
    if (index < 0 || index + 1 >= g_mkwGuardedFunctionCount || g_mkwGuardedFunctionStarts[index] != original ||
        std::find(g_forced.begin(), g_forced.end(), static_cast<uint32_t>(index)) != g_forced.end()) return {};
    // The entry detour makes this function's own patched bit unusable here. Its inlined
    // dependencies must still be pristine: otherwise this shortcut would bypass the
    // interpreter fallback installed by MarkInliningCallers. Index the immutable table
    // once rather than scanning every generated inline edge on every trampoline call.
    static const auto inlinedDependencies = [] {
        std::unordered_map<uint32_t, std::vector<uint32_t>> byCaller;
        for (uint32_t i = 0; i < g_mkwInlinedLeafCount; ++i)
            byCaller[g_mkwInlinedLeaves[2u * i + 1u]].push_back(g_mkwInlinedLeaves[2u * i]);
        return byCaller;
    }();
    if (const auto dependencies = inlinedDependencies.find(static_cast<uint32_t>(index));
        dependencies != inlinedDependencies.end()) {
        for (const uint32_t leaf : dependencies->second)
            if (g_mkwGuestCodePatched[leaf]) return {};
    }
    for (const auto& redirect : g_redirects) {
        const uint32_t start = redirect.start.load(std::memory_order_acquire);
        if (start != 0 && original >= start && original < redirect.end.load(std::memory_order_relaxed)) return {};
    }
    const uint32_t end = g_mkwGuardedFunctionStarts[index + 1];
    if (!IsGuardedLocked(original) || !IsGuardedLocked(end - 1)) return {};
    const auto pristine = g_pristine.find(original & ~(kPage - 1));
    if (pristine == g_pristine.end() ||
        std::memcmp(Memory::GetPointer(trampoline, 4), pristine->second->data() + (original & (kPage - 1)), 4)) return {};
    // The entry detour is intentionally excluded; every other body byte must still be the protected original.
    // Compare on each invocation so a later code patch cannot leave a stale native continuation enabled.
    for (uint32_t at = original + 4; at < end;) {
        const uint32_t page = at & ~(kPage - 1), offset = at - page;
        const uint32_t size = std::min(end - at, kPage - offset);
        const auto before = g_pristine.find(page);
        if (before == g_pristine.end() || !Memory::Contains(at, size) ||
            std::memcmp(Memory::GetPointer(at, size), before->second->data() + offset, size)) return {};
        at += size;
    }
    return *found;
}



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

void Protect(uint32_t start, uint32_t end, const char* name) {
    std::call_once(g_variantsOnce, [] {
        InitVariants();
        ProtectVariantRegions();
    });
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (std::find(g_ranges.begin(), g_ranges.end(), std::make_pair(start, end)) != g_ranges.end()) return;
        g_ranges.emplace_back(start, end);
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
    g_pendingPages[word].fetch_or(bit, std::memory_order_relaxed);
    g_pendingGroups[word / 32].fetch_or(1u << (word % 32), std::memory_order_release);
    g_pendingRoots[word / 32 / 32].fetch_or(1u << ((word / 32) % 32), std::memory_order_release);
    g_anyDirty.store(true, std::memory_order_release);
    HZ_PERF_ADD(kHzNoteWrite, 1);
    Bump(g_stats.noteWrites);
    return true;
}

namespace {

// An edge page's data stores would trap again after every Sync (and Sync follows every patched call): it stays open,
// compared at each Sync, until the frame's SyncFrame guards it again.
void SyncPages(bool frame) {
    // Sync follows every patched call: the plain load keeps the usual nothing-pending case off the atomic exchange.
    if (!g_anyDirty.load(std::memory_order_acquire) || !g_anyDirty.exchange(false, std::memory_order_acq_rel)) return;
    std::vector<uint32_t> pages;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        CollectPendingPagesLocked();
        if (g_dirty.empty()) return;
        pages.assign(g_dirty.begin(), g_dirty.end());
        g_dirty.clear();
        for (const uint32_t page : pages) {
            if (!frame && IsEdgePageLocked(page)) g_dirty.insert(page);
        }
        // Never clear a notification concurrently published by NoteWrite.
        if (!g_dirty.empty()) g_anyDirty.store(true, std::memory_order_release);
        HZ_PERF_ADD(kHzSyncWork, 1);
        HZ_PERF_ADD(kHzPagesCompared, pages.size());
        Bump(g_stats.pagesCompared, static_cast<uint32_t>(pages.size()));
        HZ_PERF_SET(kHzEdgeDirty, g_dirty.size());
        for (const uint32_t page : pages) {
            const auto found = g_pristine.find(page);
            if (found == g_pristine.end() || !Memory::Contains(page, kPage)) continue;
            const uint8_t* now = Memory::GetPointer(page, kPage);
            const uint8_t* was = found->second->data();
            auto& seenSlot = g_seen[page];
            if (!seenSlot) seenSlot = std::make_unique<std::array<uint8_t, kPage>>(*found->second);
            uint8_t* seen = seenSlot->data();
            if (std::memcmp(now, seen, kPage) == 0) continue;
            Bump(g_stats.pagesChanged);
            DirtyVariantsOnPageLocked(page);
            for (uint32_t offset = 0; offset < kPage; offset += 4) {
                if (std::memcmp(now + offset, seen + offset, 4) == 0) continue;
                std::memcpy(seen + offset, now + offset, 4);
                if (std::memcmp(now + offset, was + offset, 4) == 0) continue;
                const uint32_t address = page + offset;
                if (!IsGuardedLocked(address)) continue;
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
                if (index < 0 || g_mkwGuestCodePatched[index]) continue;
                g_mkwGuestCodePatched[index] = 1;
                const uint32_t function = g_mkwGuardedFunctionStarts[index];
                if (g_reported.insert(function).second) {
                    RT_LOGF(RT_TAG_MOD, "[mods] code at 0x%08X changed: function 0x%08X now runs in the interpreter\n",
                            address, function);
                }
                MarkInliningCallers(static_cast<uint32_t>(index));
            }
        }
        if (!frame) {
            pages.erase(std::remove_if(pages.begin(), pages.end(), [](uint32_t page) { return IsEdgePageLocked(page); }),
                        pages.end());
        }
    }
    for (const uint32_t page : pages) GuestFlat::ReprotectExecutablePage(page);
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
            if (page < start || end - page < kPage) {
                // Resetting this range must not discard unprocessed writes to its
                // neighbors. Recompare the edge page against the preserved bytes.
                g_dirty.insert(page);
                g_anyDirty.store(true, std::memory_order_release);
            } else {
                g_dirty.erase(page);
            }
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

void RunPatched(CpuContext* ctx, uint32_t address) {
    // The common case: the same code variant this thread last found for this address, nothing changed since.
    VariantCallCacheSlot& cached = t_variantCallCache[(address >> 2) & (kVariantCallCacheSlots - 1)];
    const uint32_t epoch = g_variantCallEpoch.load(std::memory_order_acquire);
    if (cached.address == address && cached.epoch == epoch) {
        Bump(g_stats.variantCalls);
        cached.entry(ctx);
        Sync();
        return;
    }
    bool redirected = false;
    if (address >= g_lowestRedirect.load(std::memory_order_acquire)) {
        for (const Redirect& r : g_redirects) {
            const uint32_t start = r.start.load(std::memory_order_acquire);
            if (start != 0 && address >= start && address < r.end.load(std::memory_order_relaxed)) {
                address = address - start + r.target.load(std::memory_order_relaxed);
                redirected = true;
                break;
            }
        }
    }
    // A function patched exactly as the build saw the mod patch it runs that translation (code variants).
    if (!redirected) {
        if (const auto variant = MatchingVariant(address)) {
            // Under the epoch read before the lookup: a change racing it leaves the slot stale, never wrong.
            cached = {address, epoch, variant};
            Bump(g_stats.variantCalls);
            variant(ctx);
            Sync();  // as after the interpreter below: what it called may have patched code
            return;
        }
    }
    Bump(g_stats.interpreterCalls);
    HZ_PERF_ADD(kHzPatched, 1);
    HZ_PHASE(kHzPatchedTopTicks);
    {
        HZ_PERF_T0();
        PpcInterpreter::Run(ctx, address);
        HZ_PERF_SINCE(kHzPatchedTicks);
        HZ_PERF_SET(kHzInterpInstructions, PpcInterpreter::InstructionCount());
    }
    // A patched function is usually a mod hook, and what it called may have patched more code (a Gecko hook runs
    // the code handler): that takes effect now, as the handler's isync makes it on a Wii.
    Sync();
}

} // namespace GuestCodePatch
