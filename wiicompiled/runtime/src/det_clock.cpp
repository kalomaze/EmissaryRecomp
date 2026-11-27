#include "det_clock.h"

#include "runtime_log.h"
#include "timebase_contract.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace DetClock {
namespace detail {
bool g_enabled = false;
} // namespace detail

namespace {

using Clock = std::chrono::steady_clock;

// About a microsecond of guest time per mftb read: enough for a loop that polls the time base to
// get anywhere, too little to matter for the few hundred reads a frame makes.
constexpr uint64_t kTicksPerGuestRead = 61;
// How far real time may run ahead of the timeline before pacing gives up catching up and starts
// over from the present (the backlog would otherwise play back as a burst of fast frames).
constexpr auto kMaxLag = std::chrono::milliseconds(100);

std::atomic<uint64_t> g_now{0};

// Real-time anchor of the pacing: virtual instant g_anchorTicks happens at real time g_anchorReal.
// Guest thread only.
bool g_anchored = false;
Clock::time_point g_anchorReal{};
uint64_t g_anchorTicks = 0;
bool g_resimulating = false;
uint64_t g_resimStartTicks = 0;
uint64_t g_pacingNanoseconds = 0;
// Time sync and frame pacing (det_clock.h, Brawl Online's Engine): the timeline's speed against real time, a rollback
// burst's due time, the load wait, the worst lateness since the session last asked, the paced wait's lead.
double g_speed = 1.0;
bool g_burstPending = false;
Clock::time_point g_burstDue{};
bool g_loadWait = false;
std::atomic<int64_t> g_maxBehindNs{0};
bool g_presentPacing = false;
// TakeFrameDue: the last frame boundary's due time, until taken.
bool g_frameDueValid = false;
Clock::time_point g_frameDue{};
LeadFn g_lead = nullptr;
WaitHook g_waitHook = nullptr;
PacingStats g_stats;

#if defined(__APPLE__)
// macOS: how long before a pacing deadline the wait stops sleeping and watches the clock. Not for precision (a
// sleep there lands within a millisecond or two): it keeps the game's thread on a performance core at its clock.
// macOS places a thread by how busy it has been, and the game's thread, about 1 ms of work in a 16.7 ms frame and
// asleep the rest, ran on the efficiency cores (or a performance core at a low clock) for about half its frames,
// where the same work took 3-9 ms instead of 1: the time from a frame's retrace to its present swung by more than a
// 120 Hz refresh, so the frames reached the screen unevenly (M4 Max, replay 004 of the corpus: ~1,000 of 6,558 frames
// over 20 ms apart, jitter 2.6 ms). QoS user-interactive and time-constraint (realtime) scheduling did not change
// where it ran. A 1 ms tail made it worse, 1.5 ms mostly fixed it; with 2 ms the work stays on a performance core
// (13-16 frames over 20 ms, jitter 0.16 ms), for a quarter to a third of a core more CPU (docs/frame-pacing.md).
// BRAWL_PACE_NO_SPIN=1: plain sleeps, as before (measurement).
Clock::duration SpinTail() {
    static const Clock::duration tail = std::getenv("BRAWL_PACE_NO_SPIN") != nullptr
                                            ? Clock::duration::zero()
                                            : Clock::duration(std::chrono::milliseconds(2));
    return tail;
}

inline void CpuRelax() {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}
#endif

void SleepUntil(Clock::time_point deadline) {
#if defined(_WIN32)
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
    // The high-resolution waitable timer lands within about half a millisecond; spin the rest so
    // frames keep an even cadence instead of the 1 ms scheduler sawtooth.
    constexpr auto kSpin = std::chrono::microseconds(500);
    struct HighResolutionTimer {
        HANDLE handle = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                               TIMER_MODIFY_STATE | SYNCHRONIZE);
        ~HighResolutionTimer() {
            if (handle != nullptr) {
                CloseHandle(handle);
            }
        }
    };
    static thread_local HighResolutionTimer timer;
    const auto now = Clock::now();
    if (deadline - now > kSpin && timer.handle != nullptr) {
        const auto wait = std::chrono::duration_cast<std::chrono::duration<int64_t, std::ratio<1, 10000000>>>(
            deadline - kSpin - now);
        LARGE_INTEGER due{};
        due.QuadPart = -std::max<int64_t>(wait.count(), 1);
        if (SetWaitableTimerEx(timer.handle, &due, 0, nullptr, nullptr, nullptr, 0) != FALSE) {
            WaitForSingleObject(timer.handle, INFINITE);
        }
    }
    while (Clock::now() < deadline) {
        YieldProcessor();
    }
#elif defined(__APPLE__)
    const auto tail = SpinTail();
    if (deadline - Clock::now() > tail) std::this_thread::sleep_until(deadline - tail);
    while (Clock::now() < deadline) {
        CpuRelax();
    }
#else
    // clock_nanosleep is precise to well under a millisecond on Android and Linux; no spin, which
    // would only cost a phone battery.
    std::this_thread::sleep_until(deadline);
#endif
}

// A slice of a wait the hook asked for: no spin (the hook looks again in about a millisecond, it needs no cadence).
void Nap(Clock::time_point until) {
#if defined(_WIN32)
    struct HighResolutionTimer {
        HANDLE handle = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                               TIMER_MODIFY_STATE | SYNCHRONIZE);
        ~HighResolutionTimer() {
            if (handle != nullptr) CloseHandle(handle);
        }
    };
    static thread_local HighResolutionTimer timer;
    const auto now = Clock::now();
    if (until > now && timer.handle != nullptr) {
        const auto wait =
            std::chrono::duration_cast<std::chrono::duration<int64_t, std::ratio<1, 10000000>>>(until - now);
        LARGE_INTEGER due{};
        due.QuadPart = -std::max<int64_t>(wait.count(), 1);
        if (SetWaitableTimerEx(timer.handle, &due, 0, nullptr, nullptr, nullptr, 0) != FALSE) {
            WaitForSingleObject(timer.handle, INFINITE);
            return;
        }
    }
#endif
    std::this_thread::sleep_until(until);
}

// SleepUntil(deadline), in slices of a millisecond while the wait hook wants them (it looks before each); the last
// stretch as SleepUntil's own precise sleep.
void HookedSleepUntil(Clock::time_point deadline) {
    constexpr auto kSlice = std::chrono::milliseconds(1);
#if defined(__APPLE__)
    // macOS's spin tail (SpinTail) is longer than the last slice: the hook still looks about once a millisecond in it.
    const Clock::duration last = std::max<Clock::duration>(kSlice + kSlice / 2, SpinTail());
#else
    const Clock::duration last = kSlice + kSlice / 2;
#endif
    while (g_waitHook != nullptr && g_waitHook()) {
        const auto now = Clock::now();
        if (deadline - now <= last) {
#if defined(__APPLE__)
            if (SpinTail() > Clock::duration::zero()) {
                auto looked = now;
                for (auto t = Clock::now(); t < deadline; t = Clock::now()) {
                    if (t - looked >= kSlice) {
                        looked = t;
                        if (!g_waitHook()) break;
                    }
                    CpuRelax();
                }
                SleepUntil(deadline);
                return;
            }
#endif
            break;
        }
        Nap(now + kSlice);
    }
    SleepUntil(deadline);
}

// The real time the virtual instant `ticks` is due at: the anchor, then the ticks since it at the timeline's speed.
Clock::time_point Due(uint64_t ticks) {
    const auto elapsed = TimeBaseContract::TicksToDuration(ticks - g_anchorTicks);
    if (g_speed == 1.0) return g_anchorReal + elapsed;
    return g_anchorReal + std::chrono::duration_cast<Clock::duration>(
                              std::chrono::duration<double, std::nano>(double(elapsed.count()) / g_speed));
}

void Anchor(Clock::time_point real, uint64_t ticks) {
    g_anchored = true;
    g_anchorReal = real;
    g_anchorTicks = ticks;
}

// Until `target` less the lead (SetPresentPacing), in slices, asking the lead again as the other player's inputs come
// in (CoreTiming::PacedSleepUntil): 1 ms OS sleeps, the last few ms as one precise sleep.
void PacedSleepUntil(Clock::time_point target) {
    constexpr auto kSlice = std::chrono::milliseconds(1);
    constexpr auto kFinal = std::chrono::milliseconds(3);
    for (;;) {
        const auto lead = g_lead();
        const auto wake = target - lead;
        const auto now = Clock::now();
        if (now >= wake) {
            if (lead.count() > 0 && now < target) ++g_stats.leadWakes;
            return;
        }
        if (wake - now <= kFinal) {
            HookedSleepUntil(wake);
            continue;
        }
        // While the wait hook looks at the controllers, a precise millisecond (sleep_for's lasts about two on Windows).
        if (g_waitHook != nullptr && g_waitHook()) Nap(Clock::now() + kSlice);
        else std::this_thread::sleep_for(kSlice);
    }
}

} // namespace

void Enable() noexcept {
    if (!detail::g_enabled) {
        detail::g_enabled = true;
        RT_LOGF(RT_TAG_RUNTIME, "deterministic timing on: guest clocks run on a virtual timeline\n");
    }
}

uint64_t Now() noexcept {
    return g_now.load(std::memory_order_relaxed);
}

uint64_t ReadForGuest() noexcept {
    return g_now.fetch_add(kTicksPerGuestRead, std::memory_order_relaxed);
}

void AdvanceTo(uint64_t ticks) noexcept {
    uint64_t now = g_now.load(std::memory_order_relaxed);
    while (ticks > now && !g_now.compare_exchange_weak(now, ticks, std::memory_order_relaxed)) {
    }
}

void PaceTo(uint64_t ticks, bool frameBoundary) {
    // BRAWL_UNTHROTTLED=1 (benchmarks): the timeline runs as fast as the game does, never waiting for real time, as
    // Dolphin's Core.EmulationSpeed = 0. Only when frames come to the screen changes, not what the game sees.
    static const bool unthrottled = [] {
        const char* v = std::getenv("BRAWL_UNTHROTTLED");
        return v != nullptr && *v != '\0' && *v != '0';  // BRAWL_UNTHROTTLED=0 (or empty) keeps real-time pacing
    }();
    if (g_resimulating || unthrottled) return;
    const auto now = Clock::now();
    if (!g_anchored || ticks < g_anchorTicks || g_loadWait) {
        // (While the session waits for loads, the timeline is not paced: it keeps re-anchoring, so the frames after the
        // wait are paced from then and do not race to make up for it.)
        Anchor(now, ticks);
        if (frameBoundary) g_frameDueValid = false;
        return;
    }
    const auto target = Due(ticks);
    if (frameBoundary) {
        g_frameDue = target;
        g_frameDueValid = true;
    }
    if (now >= target) {
        if (frameBoundary) {
            const int64_t behind = std::chrono::duration_cast<std::chrono::nanoseconds>(now - target).count();
            int64_t worst = g_maxBehindNs.load(std::memory_order_relaxed);
            while (behind > worst && !g_maxBehindNs.compare_exchange_weak(worst, behind, std::memory_order_relaxed)) {
            }
        }
        if (now - target > kMaxLag) {
            Anchor(now, ticks);
            ++g_stats.reanchors;
        }
        return;
    }
    const auto sleepStart = Clock::now();
    // (The lead only at the frame's own pacing point, as Dolphin's paced sleep is the first throttle after a shown frame.)
    if (frameBoundary && g_presentPacing && g_lead != nullptr) PacedSleepUntil(target);
    else HookedSleepUntil(target);
    g_pacingNanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - sleepStart).count();
}

uint64_t PacingNanoseconds() noexcept { return g_pacingNanoseconds; }

void SetRollbackResimulating(bool active) noexcept {
    if (!active && g_resimulating) g_stats.resimTicks += Now() - g_resimStartTicks;
    if (!active && g_burstPending) {
        // The presented frame after a rollback burst continues the schedule from where it was due when the burst began:
        // the re-run's ticks are skipped, the real time it took comes out of the frame's slack (CoreTiming).
        g_burstPending = false;
        g_resimulating = false;
        Anchor(g_burstDue, Now());
        return;
    }
    if (active == g_resimulating) return;
    if (active) g_resimStartTicks = Now();
    else if (g_anchored) g_anchorTicks += Now() - g_resimStartTicks;
    g_resimulating = active;
}

bool RollbackResimulating() noexcept { return g_resimulating; }

void ResetPacing() noexcept {
    // (Brawl Online's ResetThrottleToNow: a rollback burst still pending is dropped too.)
    g_burstPending = false;
    Anchor(Clock::now(), Now());
}

void SetSpeed(double factor) noexcept {
    if (!(factor > 0.0) || factor == g_speed) return;
    // The schedule goes on from the current instant's due time at the new speed (CoreTiming::UpdateSpeedLimit).
    if (g_anchored && !g_resimulating && !g_loadWait) {
        const uint64_t now = Now();
        if (now >= g_anchorTicks) Anchor(Due(now), now);
    }
    g_speed = factor;
    g_stats.speed = factor;
    g_stats.speedMin = std::min(g_stats.speedMin, factor);
    g_stats.speedMax = std::max(g_stats.speedMax, factor);
}

void ResetSpeedRange() noexcept { g_stats.speedMin = g_stats.speedMax = g_speed; }

double Speed() noexcept { return g_speed; }

void AdvanceSchedule(std::chrono::nanoseconds amount) noexcept {
    if (!g_anchored) return;
    g_anchorReal -= amount;
    // A rollback burst in progress resumes the schedule at its due time: that moves as well.
    if (g_burstPending) g_burstDue -= amount;
    ++g_stats.advances;
}

void BeginRollbackBurst() noexcept {
    if (g_burstPending || !g_anchored) return;
    // At the loop top, before the load: the virtual instant here is the end of the frame just presented, which the
    // pacing already waited for.
    g_burstDue = g_resimulating || g_loadWait ? Clock::now() : Due(Now());
    g_burstPending = true;
    ++g_stats.bursts;
}

void SetLoadWait(bool waiting) noexcept {
    if (waiting == g_loadWait) return;
    g_loadWait = waiting;
    if (!waiting && !g_resimulating) Anchor(Clock::now(), Now());
}

bool ScheduledRealTime(Clock::time_point* out) noexcept {
    if (!g_anchored || g_resimulating || g_loadWait) return false;
    *out = Due(Now());
    return true;
}

bool TakeFrameDue(Clock::time_point* out) noexcept {
    if (!std::exchange(g_frameDueValid, false)) return false;
    *out = g_frameDue;
    return true;
}

std::chrono::nanoseconds TakeMaxBehind() noexcept {
    return std::chrono::nanoseconds(g_maxBehindNs.exchange(0, std::memory_order_relaxed));
}

void SetWaitHook(WaitHook hook) noexcept { g_waitHook = hook; }

void SetPresentPacing(bool on, LeadFn lead) noexcept {
    g_presentPacing = on;
    g_lead = lead;
}

bool MatchPacing() noexcept { return g_presentPacing && !g_resimulating && !g_loadWait; }

PacingStats Stats() noexcept { return g_stats; }

Clock::time_point NowTimePoint() noexcept {
    return Clock::time_point{TimeBaseContract::TicksToDuration(Now())};
}

uint64_t TicksAtOrAfter(Clock::time_point point) noexcept {
    const auto sinceEpoch = std::chrono::duration_cast<std::chrono::nanoseconds>(point.time_since_epoch()).count();
    if (sinceEpoch <= 0) {
        return 0;
    }
    const uint64_t ns = static_cast<uint64_t>(sinceEpoch);
    constexpr uint64_t kNum = TimeBaseContract::kTickRatioNumerator;    // 243 ticks
    constexpr uint64_t kDen = TimeBaseContract::kTickRatioDenominator;  // per 4000 ns
    return (ns / kDen) * kNum + ((ns % kDen) * kNum + kDen - 1) / kDen;
}

} // namespace DetClock
