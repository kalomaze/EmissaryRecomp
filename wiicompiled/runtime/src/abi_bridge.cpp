#include "abi_bridge.h"
#include "indirect_call_site_hit.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "generated/RuntimeConfig.h"
#include "runtime_log.h"

#if MKW_HZ_PERF && !defined(__SWITCH__)
// The counter table Horizon builds define in platform/horizon.c, for an MKW_HZ_PERF=1 build elsewhere (hz_perf.h).
extern "C" volatile unsigned long long g_hzPerf[48] = {};
#endif

// Nothing reads CpuContext by offset; these asserts just canary accidental field reordering
// (the order is a cache-locality choice explained at the struct definition).
static_assert(offsetof(CpuContext, gpr) == 0);
static_assert(offsetof(CpuContext, crv) == 128);
static_assert(offsetof(CpuContext, lr) == 192);
static_assert(offsetof(CpuContext, ctr) == 196);
static_assert(offsetof(CpuContext, xer) == 200);
static_assert(offsetof(CpuContext, fpscr) == 204);
static_assert(offsetof(CpuContext, pc) == 208);
static_assert(offsetof(CpuContext, fpr) == 216);
static_assert(offsetof(CpuContext, gqr) == 472);
static_assert(offsetof(CpuContext, hid0) == 504);
static_assert(offsetof(CpuContext, hid1) == 508);
static_assert(offsetof(CpuContext, hid2) == 512);
static_assert(offsetof(CpuContext, srr0) == 516);
static_assert(offsetof(CpuContext, srr1) == 520);
static_assert(offsetof(CpuContext, msr) == 524);
static_assert(sizeof(CpuContext) == 528);

namespace {
std::vector<TranslatedFunctionInfo>& Registry() {
    static std::vector<TranslatedFunctionInfo> entries;
    return entries;
}

size_t& PriorityOverrideCount() {
    static size_t count = 0;
    return count;
}

std::mutex& RegistryMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<uint32_t, size_t>& AddressIndex() {
    static std::unordered_map<uint32_t, size_t> index;
    return index;
}

std::vector<RawDispatchRecord>& DynamicRawDispatchEntries() {
    static std::vector<RawDispatchRecord> entries;
    return entries;
}

const StaticIndirectDispatchTable*& GeneratedIndirectDispatchTable() {
    static const StaticIndirectDispatchTable* table = nullptr;
    return table;
}

std::unordered_map<uint32_t, std::vector<size_t>>& AddressEntries() {
    static std::unordered_map<uint32_t, std::vector<size_t>> entries;
    return entries;
}

struct HostEntryRef {
    uintptr_t addr = 0;
    size_t index = 0;
};

std::vector<HostEntryRef>& HostIndex() {
    static std::vector<HostEntryRef> index;
    return index;
}

std::atomic_bool& RegistryFrozen() {
    static std::atomic_bool frozen{false};
    return frozen;
}

const char* KindLabel(FunctionKind kind) {
    switch (kind) {
    case FunctionKind::BaseTranslated:
        return "base translated";
    case FunctionKind::ModTranslated:
        return "mod translated";
    case FunctionKind::Native:
        return "native";
    }
    return "unknown";
}

uint32_t DefaultPriorityForKind(FunctionKind kind) {
    switch (kind) {
    case FunctionKind::BaseTranslated:
        return kBaseTranslatedFunctionPriority;
    case FunctionKind::ModTranslated:
        return kModTranslatedFunctionPriorityBase;
    case FunctionKind::Native:
        return kNativeFunctionPriority;
    }
    return 0;
}

uint32_t EffectivePriority(const TranslatedFunctionInfo& info) {
    return info.priority != 0 ? info.priority : DefaultPriorityForKind(info.kind);
}

uint32_t KindTieBreaker(FunctionKind kind) {
    switch (kind) {
    case FunctionKind::BaseTranslated:
        return 0;
    case FunctionKind::ModTranslated:
        return 1;
    case FunctionKind::Native:
        return 2;
    }
    return 0;
}

bool IsBetterCandidate(const TranslatedFunctionInfo& candidate, const TranslatedFunctionInfo& current) {
    const uint32_t candidatePriority = EffectivePriority(candidate);
    const uint32_t currentPriority = EffectivePriority(current);
    if (candidatePriority != currentPriority) {
        return candidatePriority > currentPriority;
    }

    const uint32_t candidateKindRank = KindTieBreaker(candidate.kind);
    const uint32_t currentKindRank = KindTieBreaker(current.kind);
    if (candidateKindRank != currentKindRank) {
        return candidateKindRank > currentKindRank;
    }

    return candidate.moduleId > current.moduleId;
}

bool IsSameRegistration(const TranslatedFunctionInfo& a, const TranslatedFunctionInfo& b) {
    return a.address == b.address &&
           a.kind == b.kind &&
           EffectivePriority(a) == EffectivePriority(b) &&
           a.moduleId == b.moduleId;
}

void RebuildIndicesLocked() {
    auto& entries = Registry();
    auto& addrIndex = AddressIndex();
    auto& addressEntries = AddressEntries();
    auto& rawDispatchEntries = DynamicRawDispatchEntries();
    auto& hostIndex = HostIndex();

    addrIndex.clear();
    addressEntries.clear();
    rawDispatchEntries.clear();
    hostIndex.clear();
    addrIndex.reserve(entries.size());
    addressEntries.reserve(entries.size());

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        addressEntries[entry.address].push_back(i);
        auto addressIt = addrIndex.find(entry.address);
        if (addressIt == addrIndex.end() || IsBetterCandidate(entry, entries[addressIt->second])) {
            addrIndex[entry.address] = i;
        }
        if (entry.entryPoint) {
            hostIndex.push_back({reinterpret_cast<uintptr_t>(entry.entryPoint), i});
        }
    }

    std::sort(hostIndex.begin(), hostIndex.end(), [](const HostEntryRef& a, const HostEntryRef& b) {
        return a.addr < b.addr;
    });

    const bool translatedWinnersAreGenerated = GeneratedIndirectDispatchTable() != nullptr;
    rawDispatchEntries.reserve(translatedWinnersAreGenerated ? 256u : addrIndex.size());
    for (const auto& [address, index] : addrIndex) {
        const auto& entry = entries[index];
        if (!entry.rawCpuInvoker || !entry.mustRemainDynamicallyDispatchable) {
            continue;
        }
        if (translatedWinnersAreGenerated && entry.kind != FunctionKind::Native) {
            continue;
        }
        rawDispatchEntries.push_back(RawDispatchRecord{
            .address = address,
            .entry = entry.rawCpuInvoker,
            .nonvolatileFprWriteMask = NonvolatileFprGuardMaskFor(&entry),
            .preserveNonvolatileGprs = ShouldPreserveNonvolatileGprsForRawCpuCall(&entry),
        });
    }
    std::sort(rawDispatchEntries.begin(), rawDispatchEntries.end(),
              [](const RawDispatchRecord& a, const RawDispatchRecord& b) {
                  return a.address < b.address;
              });
}

std::string ValidateGeneratedIndirectDispatchLocked() {
    const auto* table = GeneratedIndirectDispatchTable();
    if (!table) {
        return {};
    }
    if (!table->profileName || !table->segments || !table->entries || table->entryCount == 0) {
        return "Invalid generated indirect dispatch table";
    }

    uint32_t previousAddress = 0;
    for (size_t i = 0; i < table->entryCount; ++i) {
        const auto& record = table->entries[i];
        if ((record.address & 3u) != 0 || !record.entry ||
            (i != 0 && record.address <= previousAddress) ||
            FindStaticIndirectDispatchEntry(table, record.address) != &record) {
            std::ostringstream message;
            message << "Malformed generated indirect dispatch entry " << i
                    << " for profile '" << table->profileName << "'";
            return message.str();
        }
        previousAddress = record.address;

        const auto winner = AddressIndex().find(record.address);
        if (winner == AddressIndex().end()) {
            std::ostringstream message;
            message << "Generated indirect dispatch target 0x" << std::hex << record.address
                    << " is not registered for profile '" << table->profileName << "'";
            return message.str();
        }
        const auto& info = Registry()[winner->second];
        if (info.kind == FunctionKind::Native || info.rawCpuInvoker != record.entry ||
            NonvolatileFprGuardMaskFor(&info) != record.nonvolatileFprWriteMask ||
            record.preserveNonvolatileGprs) {
            std::ostringstream message;
            message << "Stale generated indirect dispatch winner at 0x" << std::hex
                    << record.address << " for profile '" << table->profileName << "'";
            return message.str();
        }
    }

    for (const auto& [address, index] : AddressIndex()) {
        const auto& info = Registry()[index];
        if ((info.kind == FunctionKind::BaseTranslated || info.kind == FunctionKind::ModTranslated) &&
            info.rawCpuInvoker && !FindStaticIndirectDispatchEntry(table, address)) {
            std::ostringstream message;
            message << "Generated indirect dispatch profile '" << table->profileName
                    << "' omits translated winner 0x" << std::hex << address;
            return message.str();
        }
    }
    return {};
}
}

void RegisterStaticIndirectDispatchTable(const StaticIndirectDispatchTable* table) {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    if (RegistryFrozen().load(std::memory_order_acquire)) {
        RT_LOG(RT_TAG_RUNTIME) << "ERROR: Generated indirect dispatch table registered after finalization" << std::endl;
        ShowRuntimeFatalPopup("translated dispatch initialization failed",
                              "A generated indirect-dispatch table was registered after the function registry was finalized.");
        std::abort();
    }
    auto*& registered = GeneratedIndirectDispatchTable();
    if (registered && registered != table) {
        RT_LOG(RT_TAG_RUNTIME) << "ERROR: Multiple generated indirect dispatch profiles were linked" << std::endl;
        ShowRuntimeFatalPopup("translated dispatch initialization failed",
                              "Multiple generated indirect-dispatch profiles were linked into the same product.");
        std::abort();
    }
    registered = table;
    // Mirror into the header-visible atomic under the same lock, so the
    // inlined miss path and this owning static can never disagree.
    g_publishedStaticIndirectDispatchTable.store(table, std::memory_order_release);
}

void RegisterBulkTranslatedFunctions(const BulkTranslatedFunctionRecord* records, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const auto& record = records[i];
        TranslatedFunctionInfo info;
        info.address = record.address;
        info.name = record.name ? record.name : "";
        info.moduleId = record.moduleId;
        info.priority = record.priority;
        info.nonvolatileFprWriteMask = record.preservesNonvolatileFprs
            ? 0u
            : (record.nonvolatileFprWriteMask & kPpcAllNonvolatileFprMask);
        info.entryPoint = reinterpret_cast<void*>(record.entry);
        info.rawCpuInvoker = record.entry;
        info.mustRemainDynamicallyDispatchable = record.mustRemainDynamicallyDispatchable;
        info.kind = record.kind;
        TranslatedFunctionRegistry::Register(std::move(info));
    }
}

void TranslatedFunctionRegistry::Register(TranslatedFunctionInfo info) {
    std::lock_guard<std::mutex> lock(RegistryMutex());
    if (RegistryFrozen().load(std::memory_order_acquire)) {
        RT_LOG(RT_TAG_RUNTIME) << "ERROR: Attempted to register function after registry was finalized: 0x"
                  << std::hex << info.address << " '" << info.name << "'" << std::dec << std::endl;
        ShowRuntimeFatalPopup("translated function registration failed",
                              "A translated function was registered after guest execution had already been finalized.");
        std::abort();
    }
    info.priority = EffectivePriority(info);

    auto& entries = Registry();
    auto& addrIndex = AddressIndex();
    auto& addressEntries = AddressEntries();

    auto sameAddressIt = addressEntries.find(info.address);
    if (sameAddressIt != addressEntries.end()) {
        for (const size_t existingIndex : sameAddressIt->second) {
            const auto& existing = entries[existingIndex];
            if (IsSameRegistration(existing, info)) {
                RT_LOG(RT_TAG_RUNTIME) << "Duplicate " << KindLabel(info.kind) << " registration for " << info.name << " (0x"
                          << std::hex << info.address << ") ignored" << std::dec << std::endl;
                return;
            }
        }
    }

    entries.push_back(std::move(info));
    const size_t index = entries.size() - 1;
    addressEntries[entries[index].address].push_back(index);
    auto bestAddressIt = addrIndex.find(entries[index].address);
    if (bestAddressIt == addrIndex.end() || IsBetterCandidate(entries[index], entries[bestAddressIt->second])) {
        if (bestAddressIt != addrIndex.end()) {
            // Expected on every mod entry - a Retro Rewind boot has thousands.
            // Counted here and reported once by Finalize instead of per address.
            ++PriorityOverrideCount();
        }
        addrIndex[entries[index].address] = index;
    }
    if (entries[index].entryPoint) {
        HostIndex().push_back({reinterpret_cast<uintptr_t>(entries[index].entryPoint), index});
    }
}

void TranslatedFunctionRegistry::Finalize() {
    if (const size_t overrides = PriorityOverrideCount(); overrides != 0) {
        RT_LOG(RT_TAG_RUNTIME) << overrides
                  << " registration(s) took over an address by priority" << std::endl;
    }
    std::string validationError;
    {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        // Publication is one-way: cached pointers into the rebuilt vectors
        // remain valid for the lifetime of the process.
        if (RegistryFrozen().load(std::memory_order_acquire)) {
            return;
        }
        RebuildIndicesLocked();
        validationError = ValidateGeneratedIndirectDispatchLocked();
        if (validationError.empty()) {
            // Duplicate-registration tracking is initialization-only. AddressIndex
            // owns every published winner after this point, and Register rejects
            // later writes, so release the per-address candidate vectors before
            // the game starts.
            auto& addressEntries = AddressEntries();
            addressEntries.clear();
            addressEntries.rehash(0);
            RegistryFrozen().store(true, std::memory_order_release);
            lookupPublished_.store(true, std::memory_order_release);
        }
    }
    if (!validationError.empty()) {
        throw std::runtime_error("[" RT_TAG_RUNTIME "] " + validationError);
    }
}

const TranslatedFunctionInfo* TranslatedFunctionRegistry::FindByAddressPtrSlow(uint32_t address) {
    if (RegistryFrozen().load(std::memory_order_acquire)) {
        const auto it = AddressIndex().find(address);
        return it != AddressIndex().end() ? &Registry()[it->second] : nullptr;
    }

    std::lock_guard<std::mutex> lock(RegistryMutex());
    auto it = AddressIndex().find(address);
    if (it != AddressIndex().end()) {
        return &Registry()[it->second];
    }
    for (auto& entry : Registry()) {
        if (entry.address == address) {
            return &entry;
        }
    }
    return nullptr;
}

const RawDispatchRecord* TranslatedFunctionRegistry::FindRawByAddressPtrSlow(uint32_t address) {
    if (!RegistryFrozen().load(std::memory_order_acquire)) {
        return nullptr;
    }
    if (const auto* generated = FindStaticIndirectDispatchEntry(GeneratedIndirectDispatchTable(), address)) {
        return generated;
    }
    const auto& entries = DynamicRawDispatchEntries();
    const auto it = std::lower_bound(entries.begin(), entries.end(), address,
        [](const RawDispatchRecord& entry, uint32_t target) {
            return entry.address < target;
        });
    return it != entries.end() && it->address == address ? &*it : nullptr;
}

std::optional<TranslatedFunctionInfo> TranslatedFunctionRegistry::FindByHostAddress(uintptr_t hostAddr) {
    const bool frozen = RegistryFrozen().load(std::memory_order_acquire);
    if (!frozen) {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        RebuildIndicesLocked();
    }

    const auto& entries = Registry();
    const auto& sorted = HostIndex();
    if (sorted.empty()) {
        return std::nullopt;
    }

    auto it = std::upper_bound(sorted.begin(), sorted.end(), hostAddr,
                               [](uintptr_t addr, const HostEntryRef& ref) { return addr < ref.addr; });

    if (it == sorted.begin()) {
        return std::nullopt;
    }
    --it;

    uintptr_t funcStart = it->addr;
    uintptr_t funcEnd;
    auto nextIt = it + 1;
    if (nextIt != sorted.end()) {
        funcEnd = nextIt->addr;
    } else {
        funcEnd = funcStart + 256 * 1024;
    }

    if (hostAddr >= funcStart && hostAddr < funcEnd) {
        return entries[it->index];
    }
    return std::nullopt;
}

std::optional<TranslatedFunctionInfo> TranslatedFunctionRegistry::FindNearestByAddress(uint32_t address) {
    std::optional<TranslatedFunctionInfo> best;
    uint32_t bestDelta = UINT32_MAX;
    const auto scan = [&] {
        for (const auto& info : Registry()) {
            if (address < info.address) {
                continue;
            }
            const uint32_t delta = address - info.address;
            if (delta < bestDelta) {
                bestDelta = delta;
                best = info;
            }
        }
    };
    if (RegistryFrozen().load(std::memory_order_acquire)) {
        scan();
    } else {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        scan();
    }
    return best;
}

namespace {
CpuContext g_persistentCpu{};
}

CpuContext& GetPersistentCpuContext() {
    return g_persistentCpu;
}

void InitializePersistentCpuContext() {
    // Idempotent by contract: generated mod code calls this again before each of
    // its initializers, after the boot path has already established r1 and live
    // guest state in the persistent context. Never clear the context here; only
    // seed the PowerPC ABI environmental registers if no one has yet:
    // r2 = _SDA2_BASE_ (read-only small data), r13 = _SDA_BASE_.
    if (g_persistentCpu.gpr[2] == 0) {
        g_persistentCpu.gpr[2] = RuntimeConfig::SDA2_BASE;
    }
    if (g_persistentCpu.gpr[13] == 0) {
        g_persistentCpu.gpr[13] = RuntimeConfig::SDA1_BASE;
    }
}

#if MKW_INDIRECT_IC
// Slow paths of the per-site indirect caches (indirect_call_site.h): everything but an unguarded hit in the current
// context. Each one runs what InvokeIndirectCpu / InvokeIndirectJump run, with the site's record standing in for the
// lookup when it matches, and refills the site after a lookup.
namespace {
// MKW_INDIRECT_IC=0 in the environment: never fill a site, so every call takes the old lookup and dispatch.
const bool g_indirectSiteFillEnabled = [] {
    const char* value = std::getenv("MKW_INDIRECT_IC");
    return !(value && value[0] == '0');
}();

#if MKW_INDIRECT_IC_STATS
// Debug counters (-DMKW_INDIRECT_IC_STATS=1). Plain load/store increments: a lost count under a race is fine here.
struct IndirectSiteStats {
    std::atomic<uint64_t> guarded{0};         // lookup found a guarded record (kept in the guarded slot)
    std::atomic<uint64_t> unpacked{0};        // lookup found an unguarded record whose entry could not be packed
    std::atomic<uint64_t> matchedContext{0};  // site matched, call switches CpuContext
    std::atomic<uint64_t> empty{0};           // site never filled (first call, or only unregistered targets)
    std::atomic<uint64_t> mismatch{0};        // site held other targets
    std::atomic<uint64_t> unregistered{0};    // lookup found no record (interpreter, natives without raw record)
    std::atomic<uint64_t> memoHit{0};         // lookups the thread-local memo answered
    std::atomic<uint64_t> memoMiss{0};        // lookups that searched the generated table
    std::atomic<uint64_t> slowCalls{0};
};
IndirectSiteStats g_indirectSiteStats;

struct IndirectSiteInfo {
    MkwIndirectSite* site;
    uintptr_t caller;
    bool jump;
    uint64_t lastHits;
    uint64_t lastSlow;
    uint32_t lastFills;
    uint32_t lastTarget;
};
std::mutex g_indirectSitesMutex;
std::vector<IndirectSiteInfo> g_indirectSites;
std::chrono::steady_clock::time_point g_indirectSiteLastDump = std::chrono::steady_clock::now();
uint64_t g_indirectSiteLastTotals[9] = {};

void Bump(std::atomic<uint64_t>& counter) {
    counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

void DumpIndirectSiteStatsLocked() {
    auto& s = g_indirectSiteStats;
    const uint64_t totals[9] = {s.guarded.load(), s.matchedContext.load(), s.empty.load(), s.mismatch.load(),
                                s.unregistered.load(), s.memoHit.load(), s.memoMiss.load(), s.slowCalls.load(),
                                s.unpacked.load()};
    uint64_t d[9];
    for (int i = 0; i < 9; ++i) {
        d[i] = totals[i] - g_indirectSiteLastTotals[i];
        g_indirectSiteLastTotals[i] = totals[i];
    }
    uint64_t hits = 0, slow = 0, active = 0, polymorphic = 0, hitsJump = 0, slowJump = 0;
    struct Top { uint64_t slow, hits; uint32_t fills; size_t index; };
    std::vector<Top> top;
    for (size_t i = 0; i < g_indirectSites.size(); ++i) {
        auto& info = g_indirectSites[i];
        const uint64_t h = info.site->hits.load() - info.lastHits;
        const uint64_t sl = info.site->slow.load() - info.lastSlow;
        const uint32_t f = info.site->fills.load() - info.lastFills;
        info.lastHits += h;
        info.lastSlow += sl;
        info.lastFills += f;
        if (const uint64_t way = info.site->ways[0].load()) {
            info.lastTarget = static_cast<uint32_t>(way);
        }
        hits += h;
        slow += sl;
        if (info.jump) {
            hitsJump += h;
            slowJump += sl;
        }
        if (h + sl != 0) ++active;
        if (f >= 2) ++polymorphic;
        if (sl != 0) top.push_back({sl, h, f, i});
    }
    const uint64_t calls = hits + slow;
    std::fprintf(stderr,
                 "[indirect-ic] interval: %llu calls (%llu jumps), fast hits %.2f%%, slow %llu: matched in another "
                 "context %llu, empty %llu, mismatch %llu; unpackable entry %llu; lookups: memo hit %llu, memo miss "
                 "%llu, guarded record %llu, no record %llu; sites %zu known, %llu active, %llu refilled 2+ times\n",
                 (unsigned long long)calls, (unsigned long long)(hitsJump + slowJump),
                 calls ? 100.0 * hits / calls : 0.0, (unsigned long long)slow, (unsigned long long)d[1],
                 (unsigned long long)d[2], (unsigned long long)d[3], (unsigned long long)d[8], (unsigned long long)d[5],
                 (unsigned long long)d[6], (unsigned long long)d[0], (unsigned long long)d[4],
                 g_indirectSites.size(), (unsigned long long)active, (unsigned long long)polymorphic);
    std::sort(top.begin(), top.end(), [](const Top& a, const Top& b) { return a.slow > b.slow; });
    const uintptr_t anchor = reinterpret_cast<uintptr_t>(&MkwInvokeIndirectCpuSiteSlow);
    for (size_t k = 0; k < top.size() && k < 12; ++k) {
        const auto& info = g_indirectSites[top[k].index];
        const auto owner = TranslatedFunctionRegistry::FindByHostAddress(info.caller);
        std::fprintf(stderr,
                     "[indirect-ic]   site %s caller %s (anchor%+lld) last target %08X: slow %llu, hits %llu, "
                     "fills %u\n",
                     info.jump ? "jump" : "call", owner ? owner->name : "?",
                     static_cast<long long>(info.caller - anchor), info.lastTarget,
                     (unsigned long long)top[k].slow, (unsigned long long)top[k].hits, top[k].fills);
    }
    std::fflush(stderr);
}

enum class SlowKind { MatchedContext, Empty, Mismatch };

void RecordIndirectSiteSlow(MkwIndirectSite& site, bool jump, void* caller, SlowKind kind) {
    auto& s = g_indirectSiteStats;
    switch (kind) {
    case SlowKind::MatchedContext: Bump(s.matchedContext); break;
    case SlowKind::Empty: Bump(s.empty); break;
    case SlowKind::Mismatch: Bump(s.mismatch); break;
    }
    site.slow.store(site.slow.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    if (site.registered.exchange(1) == 0) {
        std::lock_guard<std::mutex> lock(g_indirectSitesMutex);
        g_indirectSites.push_back({&site, reinterpret_cast<uintptr_t>(caller), jump, 0, 0, 0, 0});
    }
    const uint64_t n = s.slowCalls.load(std::memory_order_relaxed) + 1;
    s.slowCalls.store(n, std::memory_order_relaxed);
    if ((n & 4095u) == 0 && std::chrono::steady_clock::now() - g_indirectSiteLastDump >= std::chrono::seconds(5)) {
        std::lock_guard<std::mutex> lock(g_indirectSitesMutex);
        g_indirectSiteLastDump = std::chrono::steady_clock::now();
        DumpIndirectSiteStatsLocked();
    }
}

void RecordIndirectSiteLookup(uint32_t target) {
    const auto& cached = g_indirectRawDispatchMemo[IndirectDispatchMemoIndex(target)];
    Bump(cached.valid && cached.address == target ? g_indirectSiteStats.memoHit : g_indirectSiteStats.memoMiss);
}

SlowKind ClassifyIndirectSiteSlow(const MkwIndirectSite& site, uint32_t target) {
    bool empty = true;
    for (const auto& w : site.ways) {
        const uint64_t way = w.load(std::memory_order_relaxed);
        if (MkwIndirectWayMatches(way, target)) return SlowKind::MatchedContext;
        empty = empty && way == 0;
    }
    const RawDispatchRecord* guarded = site.guarded.load(std::memory_order_relaxed);
    if (guarded != nullptr && guarded->address == target) return SlowKind::MatchedContext;
    return empty && guarded == nullptr ? SlowKind::Empty : SlowKind::Mismatch;
}
#endif

bool IsUnguardedRecord(const RawDispatchRecord* record) {
    return record->nonvolatileFprWriteMask == 0 && !record->preserveNonvolatileGprs;
}

// The target's record from the lookup (the thread-local memo, then the generated table), which then refills the
// site. The slow path runs for a miss, a call from another CpuContext (the site may hold the target then, and is left
// alone) and an untranslated target (no record, nothing kept). Only records with an entry and a nonzero address are
// kept. An unguarded record whose entry packs goes into way 0, the other ways shifting down (the last one's target
// drops out); any other record goes to the guarded slot (see indirect_call_site.h).
MKW_PPC_FORCE_INLINE const RawDispatchRecord* ResolveIndirectSite(uint32_t target, MkwIndirectSite& site) {
#if MKW_INDIRECT_IC_STATS
    RecordIndirectSiteLookup(target);
#endif
    const RawDispatchRecord* record = TranslatedFunctionRegistry::FindRawByAddressPtr(target);
    if (record == nullptr || record->entry == nullptr || record->address == 0 || !g_indirectSiteFillEnabled) {
#if MKW_INDIRECT_IC_STATS
        if (record == nullptr) Bump(g_indirectSiteStats.unregistered);
#endif
        return record;
    }
    const uint64_t packed = IsUnguardedRecord(record) ? MkwIndirectPackWay(record) : 0;
    if (packed != 0) {
        std::atomic<uint64_t>& shared = MkwIndirectSharedWay(target);
        if (shared.load(std::memory_order_relaxed) != packed) shared.store(packed, std::memory_order_release);
        uint64_t held[kMkwIndirectSiteWays];
        for (unsigned i = 0; i < kMkwIndirectSiteWays; ++i) {
            held[i] = site.ways[i].load(std::memory_order_relaxed);
            if (held[i] == packed) return record;  // already cached: a call from another context
        }
        for (unsigned i = kMkwIndirectSiteWays - 1; i > 0; --i) {
            site.ways[i].store(held[i - 1], std::memory_order_release);
        }
        site.ways[0].store(packed, std::memory_order_release);
    } else if (site.guarded.load(std::memory_order_relaxed) != record) {
        site.guarded.store(record, std::memory_order_release);
#if MKW_INDIRECT_IC_STATS
        Bump(IsUnguardedRecord(record) ? g_indirectSiteStats.unpacked : g_indirectSiteStats.guarded);
#endif
    } else {
        return record;
    }
#if MKW_INDIRECT_IC_STATS
    site.fills.store(site.fills.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
#endif
    return record;
}

// TryDispatchRawCpuTarget, with its unguarded case spelled out: for such a record the FPR and GPR guards do nothing,
// and constructing them is two out-of-line calls in the old path.
MKW_PPC_FORCE_INLINE bool DispatchIndirectSiteRecord(const RawDispatchRecord* record, CpuContext* cpu) {
    if (record != nullptr && record->entry != nullptr && IsUnguardedRecord(record)) {
        RecompMod::ScopedTranslatedExecutionAddress translatedExecution(record->address);
        if (TryGetCpuContext() != cpu) {
            CpuContextScope scope(cpu);
            record->entry(cpu);
        } else {
            record->entry(cpu);
        }
        return true;
    }
    return TryDispatchRawCpuTarget(record, cpu);
}
}  // namespace

std::atomic<uint64_t> g_mkwIndirectSharedWays[size_t{1} << kMkwIndirectSharedBits];

void MkwIndirectEntryAnchor() {
    // Never called: only its address is used, as the base of the ways' entry displacements. The body keeps identical
    // code folding from merging it with another function (its address must not be any record's entry).
    static volatile int anchorTouched = 0;
    anchorTouched = 1;
}

#if !MKW_INDIRECT_IC_INLINE
void MkwInvokeIndirectCpuSite(uint32_t target, CpuContext* ctx, MkwIndirectSite& site) {
    MkwIndirectSiteDispatch<false>(target, ctx, site);
}

void MkwInvokeIndirectJumpSite(uint32_t target, CpuContext* ctx, MkwIndirectSite& site) {
    MkwIndirectSiteDispatch<true>(target, ctx, site);
}
#endif

// TryDispatchRawCpuTarget for a guarded record in the current CpuContext, minus the execution address's restore (as
// the hit: diagnostics only).
void MkwIndirectSiteGuardedCall(const RawDispatchRecord* record, CpuContext* ctx) {
    HZ_EXCLUSIVE_PAUSE_IF_ACTIVE();
    RecompMod::g_currentTranslatedExecutionAddress = record->address;
    PpcNonvolatileGprGuard gprGuard(ctx, record->preserveNonvolatileGprs);
    PpcNonvolatileFprGuard fprGuard(ctx, record->nonvolatileFprWriteMask);
    record->entry(ctx);
}

// InvokeIndirectCpu with the site standing in for the lookup.
void MkwInvokeIndirectCpuSiteSlow(uint32_t target, CpuContext* ctx, MkwIndirectSite& site) {
    HZ_EXCLUSIVE_PAUSE_IF_ACTIVE();
    CpuContext* cpu = ctx ? ctx : &GetPersistentCpuContext();
    if (target == 0) {
        ReportMissingCpuTarget(target, cpu);
    }
#if MKW_INDIRECT_IC_STATS
    RecordIndirectSiteSlow(site, false, __builtin_return_address(0),
                           ClassifyIndirectSiteSlow(site, target));
#endif
    if (DispatchIndirectSiteRecord(ResolveIndirectSite(target, site), cpu)) {
        return;
    }
    if (TryDispatchResolvedCpuTarget(TranslatedFunctionRegistry::FindByAddressPtr(target), cpu)) {
        return;
    }
    RunUntranslatedOrFail(target, cpu);
}

// InvokeIndirectJump with the site standing in for the lookup. A target without a record goes to InvokeIndirectJump
// itself for its fallbacks (interpreter, fatal report); its own lookup of the target finds nothing again.
void MkwInvokeIndirectJumpSiteSlow(uint32_t target, CpuContext* ctx, MkwIndirectSite& site) {
    CpuContext* cpu = ctx ? ctx : &GetPersistentCpuContext();
#if MKW_INDIRECT_IC_STATS
    RecordIndirectSiteSlow(site, true, __builtin_return_address(0),
                           ClassifyIndirectSiteSlow(site, target));
#endif
    if (DispatchIndirectSiteRecord(ResolveIndirectSite(target, site), cpu)) {
        return;
    }
    InvokeIndirectJump(target, cpu);
}
#endif
