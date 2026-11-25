#pragma once
#include <cstdint>

namespace BrawlDirtyMemory {
// Differential verifiers also read untracked-page marks. While observing a guest call on this thread, keep code
// marks until the verifier captures them instead of letting the interpreter consume them during a cache refresh.
inline thread_local uint32_t markObservers = 0;
struct ScopedMarkObservation {
    ScopedMarkObservation() { ++markObservers; }
    ~ScopedMarkObservation() { --markObservers; }
    ScopedMarkObservation(const ScopedMarkObservation&) = delete;
    ScopedMarkObservation& operator=(const ScopedMarkObservation&) = delete;
};
}  // namespace BrawlDirtyMemory
