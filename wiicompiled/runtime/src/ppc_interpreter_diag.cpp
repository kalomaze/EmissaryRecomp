// BRAWL_JIT_CHURN and BRAWL_JIT_PROBE (ppc_interpreter_diag.h): diagnostics only.
#include "ppc_interpreter_diag.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory.h"
#include "memory_access.h"
#include "ppc_runtime.h"
#include "runtime_log.h"

#if defined(__linux__)
#include <unistd.h>
#endif

namespace PpcInterpreter::diag {

const bool g_churn = MKW_JIT_CHURN && [] {
    const char* v = std::getenv("BRAWL_JIT_CHURN");
    return v != nullptr && *v != '\0';
}();

namespace {

uint64_t g_counters[kCounters] = {};

struct Tables {
    std::mutex mutex;
    std::unordered_map<uint64_t, uint64_t> t[kTables];
};
Tables& T() {
    static Tables* const tables = new Tables;  // never destroyed: the game is stopped, not exited, by the bench
    return *tables;
}

uint64_t MonoNs() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

FILE* Out() {
    static FILE* const f = [] {
        const char* path = std::getenv("BRAWL_JIT_CHURN");
        std::string p = path != nullptr ? path : "";
        const size_t at = p.find("%p");  // %p: the pid, so concurrent or repeated runs keep their own files
        if (at != std::string::npos) {
#if defined(__linux__)
            p.replace(at, 2, std::to_string(static_cast<long>(getpid())));
#else
            p.replace(at, 2, "0");
#endif
        }
        FILE* file = p.empty() ? nullptr : std::fopen(p.c_str(), "w");
        if (file != nullptr) {
            std::fprintf(file, "# counters:");
            for (const char* n : kCounterNames) std::fprintf(file, " %s", n);
            for (const char* n : kCacheCounterNames) std::fprintf(file, " %s", n);
            std::fprintf(file, "\n");
        }
        return file;
    }();
    return f;
}

constexpr uint64_t kMapEvery = 250;

}  // namespace

uint64_t* CounterCell(Counter c) { return &g_counters[c]; }

void AddKey(Table t, uint64_t key, uint64_t n) {
    if (!g_churn) return;
    Tables& s = T();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.t[t][key] += n;
}

void Fallback(uint32_t pc, uint32_t kind, uint32_t w) {
    if (!g_churn) return;
    ++g_counters[kFallbackOps];
    const uint32_t key = kind == 0 ? 0x10000u | ((w >> 26) << 10) | ((w >> 1) & 0x3FFu) : kind;
    Tables& s = T();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.t[kTFallbackKind][key] += 1;
    s.t[kTFallbackSite][(uint64_t{pc} << 32) | key] += 1;
}

void Pass(const uint64_t (&cache)[kCacheCounters]) {
    if (!g_churn) return;
    FILE* const f = Out();
    if (f == nullptr) return;
    static uint64_t passes = 0;
    ++passes;
    const uint64_t now = MonoNs();
    std::fprintf(f, "P %" PRIu64 " %" PRIu64, now, passes);
    for (uint64_t v : g_counters) std::fprintf(f, " %" PRIu64, v);
    for (uint64_t v : cache) std::fprintf(f, " %" PRIu64, v);
    std::fprintf(f, "\n");
    if (passes % kMapEvery == 0) {
        Tables& s = T();
        std::lock_guard<std::mutex> lock(s.mutex);
        for (uint32_t t = 0; t < kTables; ++t) {
            std::fprintf(f, "M %" PRIu64 " %" PRIu64 " %s", now, passes, kTableNames[t]);
            for (const auto& [k, v] : s.t[t]) std::fprintf(f, " %" PRIx64 ":%" PRIu64, k, v);
            std::fprintf(f, "\n");
        }
    }
    if (passes % 64 == 0) std::fflush(f);
}

// --- BRAWL_JIT_PROBE ----------------------------------------------------------------------------------------------------

namespace {

struct ProbeItem {
    int reg = -1;          // -1: an absolute address
    uint32_t base = 0;     // the address, or the offset from the register
    bool deref = false;    // a second step: [first] + second
    uint32_t second = 0;
    std::string text;
};

struct ProbeConfig {
    std::vector<uint32_t> pcs;
    std::vector<ProbeItem> items;
    ProbeConfig() {
        for (const char* p = std::getenv("BRAWL_JIT_PROBE"); p != nullptr && *p;) {
            char* end = nullptr;
            const unsigned long pc = std::strtoul(p, &end, 16);
            if (end == p) break;
            pcs.push_back(static_cast<uint32_t>(pc));
            p = *end ? end + 1 : end;
        }
        const char* r = std::getenv("BRAWL_JIT_PROBE_READ");
        std::string list = r != nullptr ? r : "";
        size_t at = 0;
        while (at < list.size()) {
            size_t comma = list.find(',', at);
            if (comma == std::string::npos) comma = list.size();
            const std::string tok = list.substr(at, comma - at);
            at = comma + 1;
            if (tok.empty()) continue;
            ProbeItem item;
            item.text = tok;
            const char* s = tok.c_str();
            char* end = nullptr;
            if (*s == 'r') {
                item.reg = static_cast<int>(std::strtol(s + 1, &end, 10)) & 31;
                s = end;
                if (*s == ':') item.base = static_cast<uint32_t>(std::strtol(s + 1, &end, 16)), s = end;
            } else {
                item.base = static_cast<uint32_t>(std::strtoul(s, &end, 16));
                s = end;
                if (*s == ':') {
                    item.deref = true;
                    item.second = static_cast<uint32_t>(std::strtol(s + 1, &end, 16));
                    s = end;
                }
            }
            if (item.reg >= 0 && *s == ':') {
                item.deref = true;
                item.second = static_cast<uint32_t>(std::strtol(s + 1, &end, 16));
            }
            items.push_back(item);
        }
    }
};
const ProbeConfig& Probe() {
    static const ProbeConfig* const config = new ProbeConfig;
    return *config;
}

bool ReadWord(uint32_t a, uint32_t& out) {
    if (!Memory::Contains(a, 4)) return false;
    out = MemoryInline::FlatRead32(a);
    return true;
}

}  // namespace

bool ProbeAt(uint32_t pc) {
    for (uint32_t p : Probe().pcs)
        if (p == pc) return true;
    return false;
}

void ProbeHit(const CpuContext* c, uint32_t pairedMask) noexcept {
    try {
        ++g_counters[kJitProbeHits];
        static std::mutex mutex;
        static std::unordered_map<uint32_t, uint64_t> hits;
        std::lock_guard<std::mutex> lock(mutex);
        const uint64_t n = hits[c->pc]++;
        if (n >= 40 && (n & 4095u) != 0) return;
        std::string line;
        char item[96];
        std::snprintf(item, sizeof(item), "[probe] %08X #%" PRIu64 " t=%" PRIu64 " lr=%08X ctr=%08X paired=%08X", c->pc, n,
                      MonoNs(), c->lr, c->ctr, pairedMask);
        line += item;
        for (int r = 0; r < 32; ++r) {
            std::snprintf(item, sizeof(item), " r%d=%08X", r, c->gpr[r]);
            line += item;
        }
        for (int r : {0, 1, 2, 3, 31}) {
            std::snprintf(item, sizeof(item), " f%d=%016" PRIX64, r, c->fpr[r].raw);
            line += item;
        }
        for (const ProbeItem& it : Probe().items) {
            uint32_t a = it.reg >= 0 ? c->gpr[it.reg] + it.base : it.base, v = 0;
            bool ok = true;
            if (it.deref) {
                ok = ReadWord(a, v);
                a = v + it.second;
            }
            ok = ok && ReadWord(a, v);
            if (ok) std::snprintf(item, sizeof(item), " [%s]=%08X", it.text.c_str(), v);
            else std::snprintf(item, sizeof(item), " [%s]=?", it.text.c_str());
            line += item;
        }
        RT_LOGF(RT_TAG_RUNTIME, "%s\n", line.c_str());
    } catch (...) {
    }
}

}  // namespace PpcInterpreter::diag
