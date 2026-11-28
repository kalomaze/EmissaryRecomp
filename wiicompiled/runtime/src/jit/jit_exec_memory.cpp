#include "jit/jit_exec_memory.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#define MKW_JIT_MEMORY_HOST 1
#if defined(__x86_64__)
// The linker's symbols for the executable's first byte and the end of its text (GNU ld and lld define both).
extern "C" char __executable_start[];
extern "C" char etext[];
#endif
#elif defined(_WIN32)
#include <windows.h>
#define MKW_JIT_MEMORY_HOST 1
#else
#define MKW_JIT_MEMORY_HOST 0
#endif

namespace JitExecMemory {
namespace {

#if MKW_JIT_MEMORY_HOST

size_t PageSize() {
    static const size_t size = [] {
#if defined(_WIN32)
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        return static_cast<size_t>(info.dwPageSize);
#else
        long value = sysconf(_SC_PAGESIZE);
        return value > 0 ? static_cast<size_t>(value) : static_cast<size_t>(4096);
#endif
    }();
    return size;
}

#if defined(__linux__) && defined(__x86_64__)
// Code space within a rel32 displacement of the executable's whole text, so that emitted code calls translated functions and
// runtime helpers with `call rel32` (the emitters check the distance and fall back to an absolute address in a register for
// memory that is not near). One PROT_NONE reservation at the top of the reachable window (far from the program break, which
// grows up from the end of the image), carved into 2 MiB slots that are made RWX when handed out and dropped when freed.
class NearSpace {
public:
    static constexpr size_t kSlot = size_t{2} << 20;
    static constexpr size_t kSlots = 128;  // 256 MiB of address space; past it allocations are mapped anywhere

    static NearSpace& Get() {
        static NearSpace space;
        return space;
    }

    // `size` bytes (a whole number of pages) of RWX memory in whole slots, or null when no near space is left.
    void* Map(size_t size, bool hugeAdvice) {
        if (base_ == 0 || size == 0 || size > kSlots * kSlot) return nullptr;
        const size_t n = (size + kSlot - 1) / kSlot;
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t first = 0; first + n <= kSlots; ++first) {
            size_t k = 0;
            while (k < n && !used_[first + k]) ++k;
            if (k != n) {
                first += k;
                continue;
            }
            void* const at = reinterpret_cast<void*>(base_ + first * kSlot);
            if (mprotect(at, n * kSlot, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) return nullptr;
#if defined(MADV_HUGEPAGE)
            if (hugeAdvice) madvise(at, n * kSlot, MADV_HUGEPAGE);
#else
            (void)hugeAdvice;
#endif
            for (size_t i = 0; i < n; ++i) used_[first + i] = true;
            return at;
        }
        return nullptr;
    }

    // Takes back a region Map made (false: not near memory, the caller unmaps it).
    bool Unmap(void* p, size_t size) {
        const uintptr_t at = reinterpret_cast<uintptr_t>(p);
        if (base_ == 0 || at < base_ || at >= base_ + kSlots * kSlot) return false;
        const size_t first = (at - base_) / kSlot, n = (size + kSlot - 1) / kSlot;
        std::lock_guard<std::mutex> lock(mutex_);
        madvise(p, n * kSlot, MADV_DONTNEED);  // the next owner gets zeroed pages, as from a fresh mapping
        mprotect(p, n * kSlot, PROT_NONE);
        for (size_t i = 0; i < n && first + i < kSlots; ++i) used_[first + i] = false;
        return true;
    }

private:
    NearSpace() {
        // BRAWL_JIT_FAR_MEMORY=1 (a diagnostic): code space anywhere, as on a host without the reservation
        if (const char* v = std::getenv("BRAWL_JIT_FAR_MEMORY"); v != nullptr && *v != '\0' && *v != '0') return;
        const uintptr_t textLo = reinterpret_cast<uintptr_t>(__executable_start);
        const uintptr_t textHi = reinterpret_cast<uintptr_t>(etext);
        constexpr uintptr_t kReach = (uintptr_t{1} << 31) - (uintptr_t{1} << 20);  // a margin under 2 GiB
        constexpr size_t kBytes = kSlots * kSlot;
        if (textHi <= textLo || textLo > UINTPTR_MAX - kReach) return;
        // Every address of [at, at + kBytes) reaches [textLo, textHi) and back: at + kBytes <= textLo + kReach and
        // at >= textHi - kReach. From the top of that window down, in 64 MiB steps, until a hint is free.
        const uintptr_t top = (textLo + kReach - kBytes) & ~(uintptr_t{kSlot} - 1);
        const uintptr_t bottom = textHi > kReach ? textHi - kReach : 0;
        for (uintptr_t at = top; at >= bottom && at >= (uintptr_t{1} << 30) && at <= top; at -= uintptr_t{64} << 20) {
            void* const p = mmap(reinterpret_cast<void*>(at), kBytes, PROT_NONE,
                                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
            if (p == MAP_FAILED) continue;
            if (reinterpret_cast<uintptr_t>(p) != at) {  // a kernel without MAP_FIXED_NOREPLACE took it as a hint
                munmap(p, kBytes);
                continue;
            }
            base_ = at;
            return;
        }
    }

    std::mutex mutex_;
    uintptr_t base_ = 0;
    bool used_[kSlots] = {};
};
#endif

Region MapRegion(size_t bytes) {
    Region region;
    if (bytes == 0) {
        return region;
    }
    const size_t page = PageSize();
    if (bytes > SIZE_MAX - page) {
        return region;
    }
    const size_t size = (bytes + page - 1) / page * page;
#if defined(__linux__) && defined(__x86_64__)
    if (void* const near = NearSpace::Get().Map(size, false)) {
        region.base = near;
        region.size = size;
        return region;
    }
#endif
#if defined(_WIN32)
    void* base = VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
    if (!base) {
        return region;
    }
#else
    void* base = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        return region;
    }
#endif
    region.base = base;
    region.size = size;
    return region;
}

#if defined(__linux__)
// A region of whole 2 MiB pages starting on a 2 MiB boundary, advised for transparent huge pages (the advice is a hint: a
// host that refuses it still has the mapping).
Region MapRegionHuge(size_t bytes) {
    Region region;
    constexpr size_t kHuge = Arena::kHugePageBytes;
    if (bytes == 0 || bytes > SIZE_MAX - 2 * kHuge) {
        return region;
    }
    const size_t size = (bytes + kHuge - 1) / kHuge * kHuge;
    // BRAWL_JIT_NO_HUGEPAGES=1 (a diagnostic for measuring): the same aligned chunks without the advice
    static const bool noAdvice = [] { const char* v = std::getenv("BRAWL_JIT_NO_HUGEPAGES"); return v != nullptr && *v != '\0' && *v != '0'; }();
#if defined(__x86_64__)
    static_assert(NearSpace::kSlot == kHuge, "near slots are huge pages");
    if (void* const near = NearSpace::Get().Map(size, !noAdvice)) {
        region.base = near;
        region.size = size;
        return region;
    }
#endif
    void* raw = mmap(nullptr, size + kHuge, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        return region;
    }
    const uintptr_t start = reinterpret_cast<uintptr_t>(raw);
    const uintptr_t aligned = (start + kHuge - 1) & ~(uintptr_t{kHuge} - 1);
    if (aligned > start) {
        munmap(raw, aligned - start);
    }
    const uintptr_t end = start + size + kHuge;
    if (end > aligned + size) {
        munmap(reinterpret_cast<void*>(aligned + size), end - (aligned + size));
    }
#if defined(MADV_HUGEPAGE)
    if (!noAdvice) madvise(reinterpret_cast<void*>(aligned), size, MADV_HUGEPAGE);
#endif
    region.base = reinterpret_cast<void*>(aligned);
    region.size = size;
    return region;
}
#endif

void UnmapRegion(Region region) {
#if defined(_WIN32)
    VirtualFree(region.base, 0, MEM_RELEASE);
#else
#if defined(__x86_64__)
    if (NearSpace::Get().Unmap(region.base, region.size)) return;
#endif
    munmap(region.base, region.size);
#endif
}

bool Probe() {
    Region region = MapRegion(1);
    if (!region) {
        return false;
    }
    // Run one instruction to see that the mapping really executes (a policy can allow the mapping and still fault).
    // x86-64: mov eax, 42; ret. Other hosts have no probe code, so they report unavailable.
#if defined(__x86_64__) || defined(_M_X64)
    static const uint8_t code[] = {0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3};
    std::memcpy(region.base, code, sizeof(code));
    using Fn = int (*)();
    const int result = reinterpret_cast<Fn>(region.base)();
    UnmapRegion(region);
    return result == 42;
#else
    UnmapRegion(region);
    return false;
#endif
}

#endif  // MKW_JIT_MEMORY_HOST

}  // namespace

bool Available() {
#if MKW_JIT_MEMORY_HOST
    static const bool available = Probe();
    return available;
#else
    return false;
#endif
}

Region Allocate(size_t bytes) {
#if MKW_JIT_MEMORY_HOST
    return MapRegion(bytes);
#else
    (void)bytes;
    return Region{};
#endif
}

void Free(Region region) {
#if MKW_JIT_MEMORY_HOST
    if (region.base) {
        UnmapRegion(region);
    }
#else
    (void)region;
#endif
}

Arena::Arena(size_t chunkBytes) : chunkBytes_(chunkBytes ? chunkBytes : kDefaultChunkBytes) {}

Arena::~Arena() { Reset(); }

Arena::Arena(Arena&& other) noexcept
    : chunks_(std::move(other.chunks_)),
      chunkBytes_(other.chunkBytes_),
      offset_(other.offset_),
      used_(other.used_),
      reserved_(other.reserved_) {
    other.chunks_.clear();
    other.offset_ = other.used_ = other.reserved_ = 0;
}

Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        Reset();
        chunks_ = std::move(other.chunks_);
        chunkBytes_ = other.chunkBytes_;
        offset_ = other.offset_;
        used_ = other.used_;
        reserved_ = other.reserved_;
        other.chunks_.clear();
        other.offset_ = other.used_ = other.reserved_ = 0;
    }
    return *this;
}

void* Arena::Allocate(size_t bytes, size_t align) {
    if (bytes == 0 || align == 0 || (align & (align - 1)) != 0) {
        return nullptr;
    }
    // Chunks start page-aligned, so alignment up to a page holds as an offset into one.
    if (align > 4096) {
        return nullptr;
    }
    if (!chunks_.empty()) {
        const Region& tail = chunks_.back();
        const size_t start = (offset_ + align - 1) & ~(align - 1);
        if (start <= tail.size && bytes <= tail.size - start) {
            offset_ = start + bytes;
            used_ += bytes;
            return static_cast<uint8_t*>(tail.base) + start;
        }
    }
    if (bytes > SIZE_MAX - align) {
        return nullptr;
    }
    const size_t want = bytes > chunkBytes_ ? bytes : chunkBytes_;
    Region chunk;
#if defined(__linux__)
    if (want >= kHugePageBytes) chunk = MapRegionHuge(want);
    if (!chunk)
#endif
        chunk = JitExecMemory::Allocate(want);
    if (!chunk) {
        return nullptr;
    }
    chunks_.push_back(chunk);
    reserved_ += chunk.size;
    // An oversized chunk is exact, so the chunk before it may still have room: keep filling that one.
    if (bytes > chunkBytes_ && chunks_.size() >= 2) {
        std::swap(chunks_[chunks_.size() - 1], chunks_[chunks_.size() - 2]);
        used_ += bytes;
        return chunks_[chunks_.size() - 2].base;
    }
    offset_ = bytes;
    used_ += bytes;
    return chunk.base;
}

bool Arena::Shrink(void* p, size_t oldBytes, size_t newBytes) {
    if (chunks_.empty() || p == nullptr || newBytes > oldBytes) {
        return false;
    }
    const Region& tail = chunks_.back();
    auto* const at = static_cast<uint8_t*>(p);
    auto* const base = static_cast<uint8_t*>(tail.base);
    if (at < base || at >= base + tail.size || static_cast<size_t>(at - base) + oldBytes != offset_) {
        return false;
    }
    offset_ -= oldBytes - newBytes;
    used_ -= oldBytes - newBytes;
    return true;
}

void Arena::Reset() {
    for (const Region& chunk : chunks_) {
        Free(chunk);
    }
    chunks_.clear();
    offset_ = 0;
    used_ = 0;
    reserved_ = 0;
}

}  // namespace JitExecMemory
