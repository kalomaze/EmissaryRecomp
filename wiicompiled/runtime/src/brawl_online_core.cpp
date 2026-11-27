// Brawl Online's core in-process, the runtime side (brawl_online_bridge.h): guest memory, the CPU at
// the main loop's hooks, the deterministic clock and the pads for the core, and the loop driver.

#include "brawl_online_core.h"
#include "profile_flush.h"
#include "brawl_online_bridge.h"

#include "abi_bridge.h"
#include "audio_backend.h"
#include "brawl_dirty_memory.h"
#include "brawl_snapshot_store.h"
#include "brawl_write_watch.h"
#include "det_clock.h"
#include "display_settings.h"
#include "frame_input.h"
#include "guest_code_patch.h"
#include "hle_stubs.h"
#include "hle/storage/sd_card.h"
#include "memory.h"
#include "memory_access.h"
#include "settings_overlay.h"
#include "aurora_events.h"
#include "hle_stubs.h"

#include <cstdio>
#include <exception>
#if defined(__APPLE__)
// The Objective-C runtime's two calls used here, declared rather than included: <objc/objc.h>'s BOOL is not the Wii
// SDK's (dolphin/types.h), and this file sees both.
extern "C" struct objc_selector* sel_registerName(const char* name);
extern "C" void objc_msgSend();
#endif
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>
#include <string>

#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_video.h>
#include <dolphin/pad.h>

void PAD_HLE_SampleLocalPads(FrameInput::GcPad* pads);
extern std::atomic_bool g_auroraFrameActive;
extern "C" void aurora_gx_present_last_display_copy();
extern "C" void aurora_gx_sync();  // aurora.cpp: wait for Aurora's GX thread

namespace {

thread_local CpuContext* t_ctx = nullptr;
std::mutex s_regs_mutex;
std::map<const void*, CpuContext> s_saved_regs;

struct CtxScope {
    CpuContext* previous;
    explicit CtxScope(CpuContext* ctx) : previous(t_ctx) { t_ctx = ctx; }
    ~CtxScope() { t_ctx = previous; }
};

CpuContext& Ctx() {
    if (!t_ctx) {
        if (g_currentCpuContext) return *g_currentCpuContext;
        std::fputs("[brawl-online] core hook without a CPU context\n", stderr);
        std::abort();
    }
    return *t_ctx;
}

// The core's snapshots: the region set packed in physical order, its delta store and write tracking.
struct Snapshots {
    struct Backing { size_t offset, end; uint8_t* data; };
    std::vector<Backing> backing;
    size_t total = 0;
    std::unique_ptr<BrawlRollback::SnapshotStore> store;
    std::unique_ptr<BrawlRollback::WriteWatch> watch;
    int anchor = 0;
    std::unordered_map<int, uint32_t> markSequences;

    template <bool Write> void Transfer(size_t& cursor, size_t offset, uint8_t* bytes, size_t count) {
        auto at = backing.begin() + cursor;
        if (offset < at->offset || offset >= at->end)
            at = std::lower_bound(backing.begin(), backing.end(), offset,
                                  [](const Backing& r, size_t o) { return r.end <= o; });
        while (count) {
            if (at == backing.end() || offset < at->offset) throw std::runtime_error("snapshot outside packed memory");
            const size_t n = std::min(count, at->end - offset);
            uint8_t* guest = at->data + (offset - at->offset);
            if constexpr (Write) std::memcpy(guest, bytes, n); else std::memcpy(bytes, guest, n);
            bytes += n; offset += n; count -= n;
            if (count) ++at;
        }
        cursor = size_t(at - backing.begin());
    }
};
Snapshots s_snap;

}  // namespace

namespace BrawlOnlineBridge {

void SnapshotRegions(const uint32_t* addrs, const uint32_t* sizes, size_t count) {
    std::fprintf(stderr, "[brawl-online] snapshot regions: %zu\n", count);
    s_snap.watch.reset();
    s_snap.store.reset();
    s_snap.backing.clear();
    s_snap.total = 0;
    s_snap.anchor = 0;
    s_snap.markSequences.clear();
    BrawlDirtyMemory::MarksCleared();
    if (!count) return;
    s_snap.store = std::make_unique<BrawlRollback::SnapshotStore>();
#if MKW_FLAT_MEMORY
    s_snap.watch = std::make_unique<BrawlRollback::WriteWatch>(reinterpret_cast<uintptr_t>(MKW_FLAT_GUEST_BASE));
#elif MKW_LEAN_MEMORY
    s_snap.watch = std::make_unique<BrawlRollback::WriteWatch>(reinterpret_cast<uintptr_t>(MemoryInline::g_leanBase));
#else
    s_snap.watch = std::make_unique<BrawlRollback::WriteWatch>();
#endif
    BrawlDirtyMemory::hosts.push_back({reinterpret_cast<uintptr_t>(Memory::GetPointer(Memory::kMem1CachedBase, Memory::kMem1BankSize)), Memory::kMem1BankSize, Memory::kMem1CachedBase});
    BrawlDirtyMemory::hosts.push_back({reinterpret_cast<uintptr_t>(Memory::GetPointer(Memory::kMem2CachedBase, Memory::kMem2Size)), Memory::kMem2Size, Memory::kMem2CachedBase});
    for (size_t i = 0; i < count; ++i) {
        uint8_t* host = Memory::GetPointer(addrs[i], sizes[i]);
        s_snap.backing.push_back({s_snap.total, s_snap.total + sizes[i], host});
        s_snap.watch->Add(host, sizes[i], s_snap.total, addrs[i]);
        s_snap.total += sizes[i];
    }
}

void SnapshotSave(int key) {
    auto& s = s_snap;
    if (!s.store) return;
    auto read = [cursor = size_t{0}](size_t offset, uint8_t* out, size_t n) mutable { s_snap.Transfer<false>(cursor, offset, out, n); };
    const std::vector<uint32_t> none;
    const bool initial = s.store->Empty();
    const auto& dirty = !initial ? s.watch->Capture(s.total, s.store->Jobs()) : none;
    s.store->Save(key, s.total, read, s.store->Empty() ? nullptr : &dirty, s.anchor);
    s.markSequences[key] = BrawlDirtyMemory::consumeSeq.load(std::memory_order_acquire);
    for (auto it = s.markSequences.begin(); it != s.markSequences.end();) {
        if (!s.store->Contains(it->first)) it = s.markSequences.erase(it);
        else ++it;
    }
    if (initial) s.watch->Enable();  // Capture already consumed subsequent saves' marks.
    s.anchor = key;
}

void SnapshotRestore(int key) {
    auto& s = s_snap;
    if (!s.store) return;
    // The restore rewrites guest memory that GX work already handed over may still read: the g3d native draw worker
    // replays deferred draws from resource display lists, vertex and matrix arrays in guest memory (g3d_native.cpp,
    // G3D_NATIVE_ASYNC=2), and only a GX sync orders those reads before the memory is reused. Without one, a rollback
    // every frame (the rollback benchmark's sync test) had the worker walk display lists of garbage length.
    // The store's own part of the restore goes first, while the GX thread may still be busy.
    s.store->PrepareRestore(key, s.anchor);
    aurora_gx_sync();
    BrawlDirtyMemory::MarksRestored(s.markSequences.at(key));
    const auto& dirty = s.watch->Capture(s.total, s.store->Jobs());
    auto read = [cursor = size_t{0}](size_t offset, uint8_t* out, size_t n) mutable { s_snap.Transfer<false>(cursor, offset, out, n); };
    auto write = [cursor = size_t{0}](size_t offset, const uint8_t* in, size_t n) mutable { s_snap.Transfer<true>(cursor, offset, const_cast<uint8_t*>(in), n); };
    s.store->Restore(key, s.anchor, dirty, read, write);
    // Capture consumed the marks; the direct restore copies do not set new ones.
    s.anchor = key;
}

uint8_t* GuestPointer(uint32_t addr, uint32_t size) {
    return Memory::Contains(addr, size) ? Memory::GetPointer(addr, size) : nullptr;
}
uint8_t* Mem1() { return Memory::GetPointer(Memory::kMem1CachedBase, Memory::kMem1Size); }
uint32_t Mem1Size() { return uint32_t(Memory::kMem1Size); }
uint8_t* Mem2() { return Memory::GetPointer(Memory::kMem2CachedBase, Memory::kMem2Size); }
uint32_t Mem2Size() { return uint32_t(Memory::kMem2Size); }
void MarkDirty(uint32_t addr, uint32_t size) { BrawlDirtyMemory::Mark(addr, size); }

uint32_t Gpr(int index) { return Ctx().gpr[index]; }
void SetGpr(int index, uint32_t value) { Ctx().gpr[index] = value; }
void SetLr(uint32_t value) { Ctx().lr = value; }
void SaveRegs(const void* key) {
    std::lock_guard lock(s_regs_mutex);
    s_saved_regs[key] = Ctx();
}
void RestoreRegs(const void* key) {
    std::lock_guard lock(s_regs_mutex);
    const auto it = s_saved_regs.find(key);
    if (it != s_saved_regs.end()) Ctx() = it->second;
}

void DumpGuestCpu() {
    const CpuContext* c = t_ctx ? t_ctx : g_currentCpuContext;
    if (!c) { std::fputs("[brawl-online] no guest CPU on this thread\n", stderr); return; }
    std::fprintf(stderr, "[brawl-online] guest pc %08x lr %08x ctr %08x cr %08x\n", c->pc, c->lr, c->ctr, PpcGetCr(c));
    for (int i = 0; i < 32; i += 8)
        std::fprintf(stderr, "[brawl-online]  r%-2d %08x %08x %08x %08x %08x %08x %08x %08x\n", i, c->gpr[i], c->gpr[i + 1],
                     c->gpr[i + 2], c->gpr[i + 3], c->gpr[i + 4], c->gpr[i + 5], c->gpr[i + 6], c->gpr[i + 7]);
    std::fflush(stderr);
}

uint64_t TimeBase() { return DetClock::Now(); }
void SetResimulating(bool active) {
    static bool s_active = false;
    if (s_active && !active) VI_HLE_NoteRollback();
    s_active = active;
    DetClock::SetRollbackResimulating(active);
    // Keep rerun frame boundaries/readbacks, but do not display their stale images.
    // The harness rollback path uses the same VI suppression.
    VI_HLE_SetPresentSuppressed(active);
}
void ResetPacing() { DetClock::ResetPacing(); }
void SetSpeed(double factor) { DetClock::SetSpeed(factor); }
void AdvancePacing(int64_t ns) { DetClock::AdvanceSchedule(std::chrono::nanoseconds(ns)); }
void BeginBurst() { DetClock::BeginRollbackBurst(); }
void SetLoadWait(bool waiting) {
    DetClock::SetLoadWait(waiting);
    Audio_HLE_SetLoadWait(waiting);
}
void SetAudioWaits(bool wait) { Audio_HLE_SetRollbackWaits(wait); }
void AddAudioPause(int64_t ticks) { Audio_HLE_AddRollbackPause(ticks); }
double SetAudioMuted(bool muted) {
    static uint64_t s_mutedBefore = 0;
    const AudioHleStats a = Audio_HLE_Stats();
    Audio_HLE_SetOutputMuted(muted);
    if (muted) {
        s_mutedBefore = a.muted;
        return 0.0;
    }
    return double((a.muted - s_mutedBefore) * a.blockTicks) / 60'750'000.0;
}
bool ScheduledHostTime(int64_t* steady_ns) {
    std::chrono::steady_clock::time_point due;
    if (!DetClock::ScheduledRealTime(&due)) return false;
    *steady_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(due.time_since_epoch()).count();
    return true;
}
int64_t TakeMaxBehindNs() { return DetClock::TakeMaxBehind().count(); }

namespace {
struct PacingMark {
    bool on = false;
    std::chrono::steady_clock::time_point real{};
    uint64_t ticks = 0, sleptNs = 0;
    DetClock::PacingStats stats;
    AudioHleStats audio;
    AudioBackend::Stats output;
};
PacingMark s_pacingMark;
std::chrono::nanoseconds PacedLead() { return std::chrono::nanoseconds(CorePacedSleepLeadNs()); }
}  // namespace

void SetPresentPacing(bool on, bool lead) {
    DetClock::SetPresentPacing(on, on && lead ? &PacedLead : nullptr);
    if (on && !s_pacingMark.on) {
        DetClock::ResetSpeedRange();
        AudioBackend::Instance().ResetRateRange();
        char discard[8];
        VI_HLE_DisplayReport(discard, sizeof discard, true);
        s_pacingMark = {true, std::chrono::steady_clock::now(), DetClock::Now(), DetClock::PacingNanoseconds(), DetClock::Stats(),
                        Audio_HLE_Stats(), AudioBackend::Instance().GetStats()};
    } else if (!on)
        s_pacingMark.on = false;
}

void DisplayReport(char* out, size_t size) { VI_HLE_DisplayReport(out, size, true); }

void PacingReport(char* out, size_t size) {
    // Since the match's pacing went on: the emulated time against the real time it took (a timeline running slow or
    // fast shows here), the time spent waiting for the schedule, and the pacing's own events.
    const auto& m = s_pacingMark;
    const auto s = DetClock::Stats();
    const double real = std::chrono::duration<double>(std::chrono::steady_clock::now() - m.real).count();
    // Rollback re-runs move the timeline on without real time of their own: they are left out of the comparison.
    const double rerun = double(s.resimTicks - m.stats.resimTicks) / 60'750'000.0;
    const double emulated = double(DetClock::Now() - m.ticks) / 60'750'000.0 - rerun;
    std::snprintf(out, size,
                  "recomp pacing: emulated %.3f s in %.3f s real (%.3f%%; re-runs %.3f s more), waited %.3f s for the "
                  "schedule, speed [%.4f, %.4f], schedule advances %llu, re-anchored %llu times (over 100 ms behind), "
                  "rollback bursts %llu, waits a coming rollback ended early %llu",
                  emulated, real, real > 0 ? 100.0 * emulated / real : 0.0, rerun,
                  double(DetClock::PacingNanoseconds() - m.sleptNs) / 1e9, s.speedMin, s.speedMax,
                  (unsigned long long)(s.advances - m.stats.advances), (unsigned long long)(s.reanchors - m.stats.reanchors),
                  (unsigned long long)(s.bursts - m.stats.bursts), (unsigned long long)(s.leadWakes - m.stats.leadWakes));
}

void AudioReport(char* out, size_t size) {
    // Since the match's pacing went on: the AI DMA blocks played against the shown timeline (the timeline less the
    // re-runs and the load waits: what the audio has to follow), the blocks held for re-runs against the re-runs' own
    // time, those played while a re-run ran, and what the output queue did (its latency, blocks it dropped as full,
    // the times the device ran dry and how much silence that was, the silence that filled the queue again after, and
    // the playback rate's range keeping its level: audio_backend.cpp).
    const auto& m = s_pacingMark;
    const AudioHleStats a = Audio_HLE_Stats();
    const AudioBackend::Stats o = AudioBackend::Instance().GetStats();
    const double tps = 60'750'000.0;
    const double block = double(a.blockTicks) / tps;
    const uint64_t played = a.played - m.audio.played, rerunsHeld = a.heldForReruns - m.audio.heldForReruns,
                   loadsHeld = a.heldInLoadWaits - m.audio.heldInLoadWaits;
    const double rerun = double(DetClock::Stats().resimTicks - m.stats.resimTicks) / tps;
    const double shown = double(DetClock::Now() - m.ticks) / tps - rerun - double(loadsHeld) * block;
    const uint64_t offered = o.blocks - m.output.blocks;
    const double perMs = o.bytesPerSecond ? 1000.0 / o.bytesPerSecond : 0.0;
    std::snprintf(out, size,
                  "recomp audio: played %llu blocks (%.3f s) for %.3f s shown (%.2f%%), held %llu for re-runs (%.3f s; "
                  "re-runs %.3f s), %llu in load waits, %llu played during re-runs, %llu muted; output queue avg %.1f ms, "
                  "max %.1f ms, %llu of %llu blocks dropped (full), ran dry %llu times (%.1f ms short), refilled %.1f ms, "
                  "rate [%.3f, %.3f]",
                  (unsigned long long)played, double(played) * block, shown,
                  shown > 0 ? 100.0 * double(played) * block / shown : 0.0, (unsigned long long)rerunsHeld,
                  double(rerunsHeld) * block, rerun, (unsigned long long)loadsHeld,
                  (unsigned long long)(a.playedInReruns - m.audio.playedInReruns),
                  (unsigned long long)(a.muted - m.audio.muted),
                  offered ? double(o.queuedBytesSum - m.output.queuedBytesSum) / double(offered) * perMs : 0.0,
                  double(o.queuedBytesMax) * perMs, (unsigned long long)(o.dropped - m.output.dropped),
                  (unsigned long long)offered, (unsigned long long)(o.ranDry - m.output.ranDry),
                  double(o.shortBytes - m.output.shortBytes) * perMs, double(o.refillBytes - m.output.refillBytes) * perMs,
                  double(o.rateMin), double(o.rateMax));
}

Pad LocalPad(int chan) {
    // A look (the core's read after a rollback): what the game would read now, changing nothing; while a test
    // script drives the pads, the latched read.
    FrameInput::GcPad pads[4]{};
    if (!FrameInput::PeekLocal(pads)) pads[chan & 3] = FrameInput::Get(uint32_t(chan & 3)).pad;
    const auto& p = pads[chan & 3];
    return {p.connected, p.button, p.stickX, p.stickY, p.substickX, p.substickY, p.triggerL, p.triggerR};
}

void ScreenMessage(const char* text, uint32_t, uint32_t) { std::fprintf(stderr, "[brawl-online] %s\n", text); }

namespace {
std::atomic<bool> s_leaveHeld{false};
std::atomic<int64_t> s_leaveReadNs{0};
int64_t SteadyNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

void StandStill() {
    // As netplay_ui's OnSessionStall: at most about 60 presents a second while nothing moves.
    static int64_t last = 0;
    const int64_t now = SteadyNs();
    if (now - last < 15'000'000) return;
    last = now;
    UpdateAuroraAndProcessEvents();
    // The controllers as the game would read them (the harness's pads while it drives the game, at
    // rest while an overlay owns input): the game's pad thread stands still with it, so they are read
    // here (SampleLocal notes START / B for LeaveButtonsHeld, as at every retrace).
    (void)FrameInput::SampleLocal();
    if (g_auroraFrameActive.load(std::memory_order_acquire)) {
        aurora_gx_present_last_display_copy();  // the last picture, not the empty frame buffer
        settings_overlay::Draw();
        VI_HLE_PresentFrame(/*presentedXfb=*/false, /*paceToRetrace=*/false);
    }
}

void NoteLeaveButtons(const FrameInput::Frame& frame) {
    bool held = false;
    for (const FrameInput::Remote& r : frame) {
        if (r.pad.connected && (r.pad.button & (0x1000u | 0x0200u)) != 0) held = true;  // START, B
        if (r.connected && (r.sample.hold & (0x0400u | 0x0010u)) != 0) held = true;     // B, PLUS
    }
    s_leaveHeld.store(held, std::memory_order_relaxed);
    s_leaveReadNs.store(SteadyNs(), std::memory_order_release);
}

bool LeaveButtonsHeld() {
    const int64_t at = s_leaveReadNs.load(std::memory_order_acquire);
    return at != 0 && SteadyNs() - at < 250'000'000 && s_leaveHeld.load(std::memory_order_relaxed);
}
float DrawAspect() { return DisplaySettings::PictureAspect(); }

bool SdCardImage(char* out, size_t size) {
    const std::string path = SdCard::PlainImagePath();
    if (path.empty() || path.size() >= size) return false;
    std::memcpy(out, path.c_str(), path.size() + 1);
    return true;
}

void StallReport(char* out, size_t size) {
    // Any thread: the deterministic clock is an atomic, the scheduler's words are guest memory. The
    // emulated time moves while guest threads wait in the scheduler (which paces it to real time) and
    // stands still while guest code runs or the game's thread is held in host code; the idle context
    // current means no guest thread could run. Each report says how far it moved since the last one.
    static std::atomic<uint64_t> s_lastTicks{0};
    static std::atomic<int64_t> s_lastNs{0};
    const auto be32 = [](uint32_t addr) -> uint32_t {
        const uint8_t* p = GuestPointer(addr, 4);
        return p ? (uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | uint32_t(p[3])) : 0;
    };
    const uint64_t ticks = DetClock::Now();
    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch()).count();
    const uint64_t lastTicks = s_lastTicks.exchange(ticks);
    const int64_t lastNs = s_lastNs.exchange(ns);
    constexpr double kTicksPerSecond = 60'750'000.0;
    char moved[96] = "";
    if (lastNs != 0 && ns - lastNs < 60'000'000'000LL && ticks >= lastTicks)
        std::snprintf(moved, sizeof moved, " (%.2f s in the last %.1f s)", double(ticks - lastTicks) / kTicksPerSecond,
                      double(ns - lastNs) / 1e9);
    const uint32_t current = be32(0x800000D4u);  // OSCurrentContext
    const uint32_t running = be32(0x800000E4u);  // the running thread's context
    const uint32_t queued = be32(0x805A06F8u);   // RunQueueBits
    std::snprintf(out, size, "recomp: emulated time %.3f s%s, %s, running thread %08x, run queue bits %08x",
                  double(ticks) / kTicksPerSecond, moved,
                  current == 0x804DD970u ? "the scheduler idles (no guest thread can run)" : "a guest thread runs",
                  running, queued);
}
void StopGame() {
    // As Dolphin's Core::Stop: the core's parts are told it stops (the stopping callbacks: replays finished, the
    // session closed), on a thread of its own so a caller holding one of the core's locks never waits on itself;
    // then the program ends without running static destructors. exit() from here raced aurora's GX and render
    // threads, which still ran, through the destructors (heap corruption, 0xc0000374, seen after a session ended).
    // The core's log flushes every line. If the stop hangs, the program ends after 5 seconds anyway.
    static std::atomic<bool> stopping{false};
    if (stopping.exchange(true)) return;
    std::fputs("[brawl-online] the core stops the game\n", stderr);
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::fputs("[brawl-online] the stop took over 5 s: ending now\n", stderr);
        std::fflush(nullptr);
        std::_Exit(0);
    }).detach();
    std::thread([] {
        BrawlOnlineCore::Shutdown();
        FlushTrainingProfile();
        std::fflush(nullptr);
        std::_Exit(0);
    }).detach();
}

}  // namespace BrawlOnlineBridge

namespace BrawlOnlineCore {

namespace B = BrawlOnlineBridge;

bool Enabled() {
    static const bool on = std::getenv("BRAWL_ONLINE_CORE") != nullptr;
    return on;
}

void Start() {
    if (!Enabled()) return;
    // An exception nothing catches (the snapshot store's own checks among them) says what it was.
    std::set_terminate([] {
        try {
            if (auto e = std::current_exception()) std::rethrow_exception(e);
            std::fputs("[brawl-online] terminate without an exception\n", stderr);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[brawl-online] uncaught exception: %s\n", e.what());
        } catch (...) {
            std::fputs("[brawl-online] uncaught exception of an unknown type\n", stderr);
        }
        std::fflush(stderr);
        std::abort();
    });
    DetClock::Enable();
    DetClock::SetWaitHook(&SampleForSuggestions);
    // The sound hooks act at pcs inside these two (SoundHook): they run in the interpreter, as a mod's patched code does.
    GuestCodePatch::InterpretForHooks(0x801C9950u, "Brawl Online's sound hooks: SoundArchivePlayer::detail_SetupSound");
    GuestCodePatch::InterpretForHooks(0x801BC684u, "Brawl Online's sound hooks: BasicSound::Stop");
    const char* comm = std::getenv("BRAWL_ONLINE_REPLAY_COMM");
    B::CoreStart(std::getenv("BRAWL_ONLINE_CORE"), comm ? comm : "");
}

void Shutdown() {
    if (Enabled()) B::CoreShutdown();
}

void RunLoop(CpuContext* ctx, Pass pass) {
    CtxScope scope(ctx);
    // In a match, the controllers again for the frame that starts here: the retrace latched them about 3 ms ago (the
    // game's other threads run between it and the loop top), and Dolphin's loop top takes its newest SI poll, about
    // 1 ms old. BRAWL_LOOPTOP_READ=0: the retrace's read, as before (docs/design/brawl-online-core.md §10.6).
    static const bool loopTopRead = [] {
        const char* env = std::getenv("BRAWL_LOOPTOP_READ");
        return env == nullptr || env[0] != '0';
    }();
    if (loopTopRead && DetClock::MatchPacing()) FrameInput::Relatch();
    for (;;) {
        uint32_t call = 0;
        switch (B::CoreLoopTop(&call)) {
        case B::Action::Normal:
            ctx->gpr[25] = 1;
            pass(ctx);
            return;
        case B::Action::Continue:
            pass(ctx);
            return;
        case B::Action::Skip: {
            B::CoreLoopEnd();
            // A skipped pass still lets time go on, as it does in Dolphin (whose CPU keeps running
            // the loop's hooks while the throttle paces it): one retrace, the game's other threads
            // and the deterministic clock run meanwhile. The loop's registers are kept.
            const CpuContext kept = *ctx;
            InvokeIndirectCpu(0x801E892Cu, ctx);  // VIWaitForRetrace
            *ctx = kept;
            return;
        }
        case B::Action::Call:
            // The call returns to the loop top (lr), where the core's hook runs again.
            InvokeIndirectCpu(call, ctx);
            continue;
        }
    }
}

bool FrameEnd(CpuContext* ctx) {
    static bool seen = false;
    if (!seen) { seen = true; std::fprintf(stderr, "[brawl-online] hook reached: frame end 0x80017504\n"); }
    CtxScope scope(ctx);
    return B::CoreFrameEnd();
}

void LoopEnd(CpuContext* ctx) {
    static bool seen = false;
    if (!seen) { seen = true; std::fprintf(stderr, "[brawl-online] hook reached: loop end 0x80017508\n"); }
    CtxScope scope(ctx);
    B::CoreLoopEnd();
}

bool DrivesLoop() { return B::CoreDrivesLoop(); }

uint32_t SoundHook(CpuContext* ctx, uint32_t pc) {
    static bool seenAlloc = false, seenAttach = false, seenStop = false;
    CtxScope scope(ctx);
    switch (pc) {
    case 0x801C9AACu:  // bl detail_AllocSeqSound
    case 0x801C9B54u:  // bl detail_AllocStrmSound
    case 0x801C9BFCu: {  // bl detail_AllocWaveSound
        if (!seenAlloc) { seenAlloc = true; std::fprintf(stderr, "[brawl-online] hook reached: sound allocation %08x\n", pc); }
        // r8 is the sound id at all three call sites, r28 the game's SoundHandle (detail_SetupSound's second argument).
        uint32_t sound = 0;
        switch (B::CoreSoundAlloc(ctx->gpr[8], ctx->gpr[28], &sound)) {
        case B::SoundAlloc::Reattach:
            // The sound an earlier run of this frame started still plays: no allocation and no setup; the success path
            // attaches the handle (r28) to it (r24) and returns START_SUCCESS.
            ctx->gpr[24] = sound;
            return 0x801C9C70u;
        case B::SoundAlloc::NoChannel:
            ctx->gpr[3] = 0;  // the pool's own "no channel" (null)
            return pc + 4;
        case B::SoundAlloc::Proceed:
            break;
        }
        return pc;
    }
    case 0x801C9C70u:  // the success path: handle r28 gets sound r24 (id r29)
        if (!seenAttach) { seenAttach = true; std::fprintf(stderr, "[brawl-online] hook reached: sound attach 801c9c70\n"); }
        B::CoreSoundAttached(ctx->gpr[28], ctx->gpr[24], ctx->gpr[29]);
        return pc;
    case 0x801BC684u:  // BasicSound::Stop(r3 sound, r4 fade frames)
        if (!seenStop) { seenStop = true; std::fprintf(stderr, "[brawl-online] hook reached: sound stop 801bc684\n"); }
        return B::CoreSoundStop(ctx->gpr[3]) ? (ctx->lr & ~3u) : pc;
    }
    return pc;
}

void StageCreate(CpuContext* ctx) {
    CtxScope scope(ctx);
    B::CoreStageCreate();
}
void StageFile(CpuContext* ctx) {
    CtxScope scope(ctx);
    B::CoreStageFile();
}
void PadRead(CpuContext* ctx) {
    static bool seen = false;
    if (!seen) { seen = true; std::fprintf(stderr, "[brawl-online] hook reached: pad read after updateLow\n"); }
    CtxScope scope(ctx);
    B::CorePadRead();
}
void HarnessPads(void* gc_pads) {
    if (!Enabled() || !B::CoreHarnessActive()) return;
    auto* pads = static_cast<FrameInput::GcPad*>(gc_pads);
    B::Pad p[4]{};
    B::CoreHarnessPads(p);
    for (int i = 0; i < 4; ++i)
        pads[i] = {p[i].connected, p[i].button, p[i].stickX, p[i].stickY, p[i].substickX, p[i].substickY, p[i].triggerL, p[i].triggerR};
    // A on the harness's pads lifts the startup cover, as a player's A does.
    for (int i = 0; i < 4; ++i)
        if (p[i].button & 0x0100) settings_overlay::NotifyStrapInputAccepted();
}
void HarnessPeekPads(void* gc_pads) {
    if (!Enabled() || !B::CoreHarnessActive()) return;
    auto* pads = static_cast<FrameInput::GcPad*>(gc_pads);
    B::Pad p[4]{};
    B::CoreHarnessPeekPads(p);
    for (int i = 0; i < 4; ++i)
        pads[i] = {p[i].connected, p[i].button, p[i].stickX, p[i].stickY, p[i].substickX, p[i].substickY, p[i].triggerL, p[i].triggerR};
}
void HostPadsRead(void* frame) {
    if (!Enabled()) return;
    auto& f = *static_cast<FrameInput::Frame*>(frame);
    B::Pad p[4]{};
    for (int i = 0; i < 4; ++i) {
        const FrameInput::GcPad& g = f[size_t(i)].pad;
        p[i] = {g.connected, g.button, g.stickX, g.stickY, g.substickX, g.substickY, g.triggerL, g.triggerR};
    }
    B::CoreHostPadsRead(p);
    for (int i = 0; i < 4; ++i)
        f[size_t(i)].pad = {p[i].connected, p[i].button, p[i].stickX, p[i].stickY, p[i].substickX, p[i].substickY,
                            p[i].triggerL, p[i].triggerR};
}
void TracePresent(std::chrono::steady_clock::time_point copy, std::chrono::steady_clock::time_point done) {
    if (!Enabled()) return;
    const auto ns = [](std::chrono::steady_clock::time_point t) {
        return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count());
    };
    B::CoreTracePresent(ns(copy), ns(done));
}
bool SampleForSuggestions() {
    if (!Enabled()) return false;
    const int chan = B::CoreSuggestChannel();
    if (chan < 0 || chan >= 4) return false;
    // About once a period (PPR_GPRB_SUGGEST_SAMPLE_HZ, 1000 by default), however the waits come.
    static const auto period = std::chrono::nanoseconds(B::CoreSuggestPeriodNs());
    static std::chrono::steady_clock::time_point last{};
    const auto start = std::chrono::steady_clock::now();
    if (start - last < period * 9 / 10) return true;
    FrameInput::GcPad pads[4]{};
    if (!FrameInput::PeekLocal(pads)) return false;
    const auto end = std::chrono::steady_clock::now();
    last = start;
    const FrameInput::GcPad& g = pads[chan];
    const auto ns = [](std::chrono::steady_clock::time_point t) {
        return int64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count());
    };
    B::CoreSuggestSample(chan, {g.connected, g.button, g.stickX, g.stickY, g.substickX, g.substickY, g.triggerL, g.triggerR},
                         ns(start), ns(end));
    return true;
}
void NoteLocalPads(const void* frame) {
    if (Enabled()) B::NoteLeaveButtons(*static_cast<const FrameInput::Frame*>(frame));
}
void DrawUi(float top) {
    if (Enabled()) B::CoreUiDraw(top);
}
bool UiBlocksInput() { return Enabled() && B::CoreUiBlocksInput(); }

namespace {
// Loaded on the first read, once the window is there (DInput's keyboard needs its HWND).
bool DolphinControlsLoaded() {
    static std::once_flag once;
    static bool loaded = false;
    std::call_once(once, [] {
        void* window = nullptr;
        int count = 0;
        if (SDL_Window** windows = SDL_GetWindows(&count)) {
#ifdef _WIN32
            if (count > 0)
                window = SDL_GetPointerProperty(SDL_GetWindowProperties(windows[0]), SDL_PROP_WINDOW_WIN32_HWND_POINTER,
                                                nullptr);
#elif defined(__APPLE__)
            // Dolphin's macOS keyboard and mouse (Quartz) take the window's content view.
            if (count > 0) {
                if (void* ns_window = SDL_GetPointerProperty(SDL_GetWindowProperties(windows[0]),
                                                             SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, nullptr)) {
                    window = reinterpret_cast<void* (*)(void*, objc_selector*)>(objc_msgSend)(
                        ns_window, sel_registerName("contentView"));
                }
            }
#endif
            SDL_free(windows);
        }
        B::DolphinControlsLoad(std::getenv("BRAWL_ONLINE_CORE"), window);
        loaded = true;
    });
    return loaded;
}
}  // namespace

void ApplyDolphinControls(PADStatus* statuses, bool peek) {
    if (!Enabled() || statuses == nullptr || !DolphinControlsLoaded()) return;
    B::Pad pads[4]{};
    int si[4]{};
    // Dolphin reads the controls only with its game window in front (Dolphin.ini [Input] BackgroundInput off).
    if (peek)
        B::DolphinControlsPeek(SDL_GetKeyboardFocus() != nullptr, pads, si);
    else
        B::DolphinControlsRead(SDL_GetKeyboardFocus() != nullptr, pads, si);
    if (!B::DolphinControlsActive()) return;
    for (int port = 0; port < 4; ++port) {
        PADStatus& s = statuses[port];
        if (si[port] == 12) continue;  // the GameCube adapter: the runtime's own pads
        s = {};
        if (si[port] != 6 || !pads[port].connected) {
            s.err = PAD_ERR_NO_CONTROLLER;
            continue;
        }
        const B::Pad& p = pads[port];
        s.err = PAD_ERR_NONE;
        s.button = p.button;
        s.stickX = p.stickX;
        s.stickY = p.stickY;
        s.substickX = p.substickX;
        s.substickY = p.substickY;
        s.triggerLeft = p.triggerL;
        s.triggerRight = p.triggerR;
    }
}

bool DolphinControlsRumble(int32_t chan, bool on) {
    if (!Enabled() || chan < 0 || chan >= 4 || !B::DolphinControlsActive() || B::DolphinControlsSiDevice(chan) != 6)
        return false;
    B::DolphinControlsRumble(chan, on ? 1.0 : 0.0);
    return true;
}

bool DolphinControlsActive() { return Enabled() && B::DolphinControlsActive(); }
int DolphinControlsSiDevice(int port) { return Enabled() ? B::DolphinControlsSiDevice(port) : 0; }
std::string DolphinControlsDevice(int port) {
    char text[256] = {};
    if (Enabled()) B::DolphinControlsDevice(port, text, sizeof(text));
    return text;
}
bool DolphinControlsUseKey(const char* key) { return Enabled() && B::DolphinControlsUseKey(key); }
bool DolphinControllersAvailable() { return Enabled() && B::DolphinControllersAvailable(); }
bool DolphinControllersShowing() { return Enabled() && B::DolphinControllersShowing(); }
bool OpenDolphinControllers() { return Enabled() && B::DolphinControllersOpen(); }
void PadThreadUpdated(CpuContext* ctx) {
    static bool seen = false;
    if (!seen) { seen = true; std::fprintf(stderr, "[brawl-online] hook reached: pad thread loop 0x8002BA84\n"); }
    CtxScope scope(ctx);
    B::CorePadThreadUpdated();
}

}  // namespace BrawlOnlineCore
