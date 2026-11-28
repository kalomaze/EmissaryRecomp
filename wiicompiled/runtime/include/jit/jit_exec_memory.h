#pragma once

// Memory the host runs as code, for the run-time compiler of guest code (src/jit). On Linux x86-64 it is mapped within a
// rel32 displacement of the executable's text while a 256 MiB reservation there lasts, so that emitted code calls translated
// functions and runtime helpers with `call rel32`; elsewhere it is mapped anywhere, and emitted code, which checks the
// distance of every such call, goes through an absolute address in a register (BRAWL_JIT_FAR_MEMORY=1 forces that).

#include <cstddef>
#include <vector>

namespace JitExecMemory {

// A mapping that is readable, writable and executable. Empty (base null) when the allocation failed.
struct Region {
    void* base = nullptr;
    size_t size = 0;  // bytes mapped, a whole number of pages
    explicit operator bool() const { return base != nullptr; }
};

// Whether the host hands out memory that is writable and executable at the same time (probed once, on first call).
// False keeps the runtime in the interpreter: other OSs, and hosts whose policy forbids such mappings.
bool Available();

// Rounds bytes up to whole pages. Callable from any thread.
Region Allocate(size_t bytes);

// Unmaps a region from Allocate. An empty region is ignored. Callable from any thread.
void Free(Region region);

// Hands out code space from chunks, one after another. Nothing is returned piecemeal: Reset and the destructor free
// every chunk. Used by one thread.
//
// On Linux a chunk of a huge page or more (the default) starts on a 2 MiB boundary and is advised MADV_HUGEPAGE, so that a
// cache of generated code is mapped by a few huge pages and the instruction TLB is not the limit; other hosts and smaller
// chunks (tests) get plain page mappings.
class Arena {
public:
#if defined(__linux__)
    static constexpr size_t kHugePageBytes = size_t{2} << 20;
    static constexpr size_t kDefaultChunkBytes = kHugePageBytes;
#else
    static constexpr size_t kHugePageBytes = 0;
    static constexpr size_t kDefaultChunkBytes = 1u << 20;
#endif

    explicit Arena(size_t chunkBytes = kDefaultChunkBytes);
    ~Arena();
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&& other) noexcept;
    Arena& operator=(Arena&& other) noexcept;

    // bytes of code space, aligned to align (a power of two). Null when memory is unavailable or bytes is 0. A request
    // larger than a chunk gets a chunk of its own.
    void* Allocate(size_t bytes, size_t align = 16);

    // Gives back the tail of the most recent allocation: p was returned by the last Allocate, which asked for oldBytes,
    // and only newBytes of it are kept (0 gives all of it back). For a compiler that reserves a worst case, emits and
    // keeps what it used. False, changing nothing, when p is not the last allocation.
    bool Shrink(void* p, size_t oldBytes, size_t newBytes);

    // Bytes handed out so far, without alignment padding.
    size_t BytesUsed() const { return used_; }
    // Bytes mapped in all chunks.
    size_t BytesReserved() const { return reserved_; }
    // The underlying executable chunks, in allocation order (append-only until Reset).
    const std::vector<Region>& Chunks() const { return chunks_; }

    // Frees every chunk; every pointer handed out is invalid afterwards.
    void Reset();

private:
    std::vector<Region> chunks_;
    size_t chunkBytes_;
    size_t offset_ = 0;  // into chunks_.back()
    size_t used_ = 0;
    size_t reserved_ = 0;
};

}  // namespace JitExecMemory
