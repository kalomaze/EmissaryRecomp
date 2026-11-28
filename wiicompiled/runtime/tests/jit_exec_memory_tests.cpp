// Executable memory (include/jit/jit_exec_memory.h) and xbyak emitting into it, the way the run-time compiler will:
// a CodeGenerator over memory the Arena owns.

#include "jit/jit_exec_memory.h"

#include <xbyak/xbyak.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

#if defined(_WIN32)
constexpr bool kWin64 = true;
#else
constexpr bool kWin64 = false;
#endif

// The argument registers of the host's C calling convention.
const Xbyak::Reg64& Arg0() { return kWin64 ? Xbyak::util::rcx : Xbyak::util::rdi; }
const Xbyak::Reg64& Arg1() { return kWin64 ? Xbyak::util::rdx : Xbyak::util::rsi; }

int CalleeDouble(int x) { return x * 2 + 1; }

void TestAvailable() {
    assert(JitExecMemory::Available());
    assert(JitExecMemory::Available());
}

void TestRegion() {
    JitExecMemory::Region region = JitExecMemory::Allocate(100);
    assert(region && region.base);
    assert(region.size >= 100 && region.size % 4096 == 0);
    auto* bytes = static_cast<uint8_t*>(region.base);
    for (size_t i = 0; i < region.size; ++i) {
        bytes[i] = static_cast<uint8_t>(i);
    }
    JitExecMemory::Free(region);
    JitExecMemory::Free(JitExecMemory::Region{});  // an empty region is ignored
    assert(!JitExecMemory::Allocate(0));
}

void TestAdd() {
    JitExecMemory::Region region = JitExecMemory::Allocate(4096);
    assert(region);
    {
        Xbyak::CodeGenerator code(region.size, region.base);
        code.lea(Xbyak::util::eax, code.ptr[Arg0() + Arg1()]);
        code.ret();
        using Fn = int (*)(int, int);
        auto fn = code.getCode<Fn>();
        assert(fn(2, 3) == 5);
        assert(fn(-7, 7) == 0);
        assert(fn(0x7fffffff, 1) == static_cast<int>(0x80000000u));
    }
    JitExecMemory::Free(region);
}

void TestCallThroughRegister() {
    using namespace Xbyak::util;
    JitExecMemory::Arena arena;
    void* mem = arena.Allocate(256);
    assert(mem);
    Xbyak::CodeGenerator code(128, mem);
    // int fn(int (*callee)(int), int arg): the stack is 16 bytes aligned at the call, with Win64's shadow space.
    code.push(rbx);  // return address + 8 = aligned
    code.mov(rax, Arg0());
    code.mov(Arg0(), Arg1());
    if (kWin64) {
        code.sub(rsp, 32);
    }
    code.call(rax);
    if (kWin64) {
        code.add(rsp, 32);
    }
    code.pop(rbx);
    code.ret();
    // And the form the backend uses, with the target as an immediate.
    Xbyak::CodeGenerator imm(128, static_cast<uint8_t*>(mem) + 128);
    imm.push(rbx);
    if (kWin64) {
        imm.sub(rsp, 32);
    }
    imm.mov(rax, reinterpret_cast<uint64_t>(&CalleeDouble));
    imm.call(rax);
    if (kWin64) {
        imm.add(rsp, 32);
    }
    imm.pop(rbx);
    imm.ret();
    // The two share one 256 byte allocation: the first is shorter than 128 bytes.
    assert(code.getSize() < 128);
    using Fn = int (*)(int (*)(int), int);
    assert(code.getCode<Fn>()(&CalleeDouble, 20) == 41);
    using Fn1 = int (*)(int);
    assert(imm.getCode<Fn1>()(5) == 11);
}

void TestCalleeSavedUsed() {
    using namespace Xbyak::util;
    JitExecMemory::Arena arena;
    void* mem = arena.Allocate(512);
    assert(mem);
    Xbyak::CodeGenerator code(512, mem);
    // Saves, uses and restores every callee-saved register the compiled blocks use; returns their sum plus the argument.
    const Xbyak::Reg64 saved[] = {rbx, r12, r13, r14, r15};
    for (const auto& r : saved) {
        code.push(r);
    }
    code.mov(rax, Arg0());
    uint64_t v = 100;
    for (const auto& r : saved) {
        code.mov(r, v);
        code.add(rax, r);
        v *= 10;
    }
    for (int i = 4; i >= 0; --i) {
        code.pop(saved[i]);
    }
    code.ret();
    using Fn = uint64_t (*)(uint64_t);
    Fn fn = code.getCode<Fn>();
    // Hold known values in the callee-saved registers across the call through asm.
    uint64_t result = fn(7);
    assert(result == 7 + 100 + 1000 + 10000 + 100000 + 1000000);
}

void TestCalleeSavedPreserved() {
    // A function that saves, clobbers and restores rbx and r12-r15 leaves the caller's values in them.
    using namespace Xbyak::util;
    JitExecMemory::Arena arena;
    void* mem = arena.Allocate(256);
    assert(mem);
    Xbyak::CodeGenerator code(256, mem);
    code.push(rbx);
    code.push(r12);
    code.push(r13);
    code.push(r14);
    code.push(r15);
    code.xor_(ebx, ebx);
    code.mov(r12, 0x1111);
    code.mov(r13, 0x2222);
    code.mov(r14, 0x3333);
    code.mov(r15, 0x4444);
    code.pop(r15);
    code.pop(r14);
    code.pop(r13);
    code.pop(r12);
    code.pop(rbx);
    code.mov(eax, 1);
    code.ret();
    using Fn = int (*)();
    Fn fn = code.getCode<Fn>();
    uint64_t rbxBefore = 0xAAAA, r12Before = 0xBBBB, r13Before = 0xCCCC, r14Before = 0xDDDD, r15Before = 0xEEEE;
    uint64_t rbxAfter, r12After, r13After, r14After, r15After;
    int ret;
#if defined(__GNUC__)
    asm volatile(
        "mov %[b0], %%rbx\n\t"
        "mov %[b1], %%r12\n\t"
        "mov %[b2], %%r13\n\t"
        "mov %[b3], %%r14\n\t"
        "mov %[b4], %%r15\n\t"
        "call *%[fn]\n\t"
        "mov %%rbx, %[a0]\n\t"
        "mov %%r12, %[a1]\n\t"
        "mov %%r13, %[a2]\n\t"
        "mov %%r14, %[a3]\n\t"
        "mov %%r15, %[a4]\n\t"
        : [a0] "=m"(rbxAfter), [a1] "=m"(r12After), [a2] "=m"(r13After), [a3] "=m"(r14After), [a4] "=m"(r15After),
          "=a"(ret)
        : [fn] "r"(fn), [b0] "m"(rbxBefore), [b1] "m"(r12Before), [b2] "m"(r13Before), [b3] "m"(r14Before),
          [b4] "m"(r15Before)
        : "rbx", "r12", "r13", "r14", "r15", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "memory", "cc");
    assert(ret == 1);
    assert(rbxAfter == rbxBefore && r12After == r12Before && r13After == r13Before && r14After == r14Before &&
           r15After == r15Before);
#else
    assert(fn() == 1);
    (void)rbxBefore;
    (void)r12Before;
    (void)r13Before;
    (void)r14Before;
    (void)r15Before;
    (void)rbxAfter;
    (void)r12After;
    (void)r13After;
    (void)r14After;
    (void)r15After;
    (void)ret;
#endif
}

// Shrink gives back the tail of the last allocation only.
void TestArenaShrink() {
    using JitExecMemory::Arena;
    Arena arena(4096);
    auto* first = static_cast<uint8_t*>(arena.Allocate(1000, 16));
    assert(first);
    assert(arena.Shrink(first, 1000, 100));
    assert(arena.BytesUsed() == 100);
    auto* second = static_cast<uint8_t*>(arena.Allocate(500, 16));
    assert(second >= first + 100 && second < first + 128);  // right after the kept bytes
    assert(!arena.Shrink(first, 100, 50));                  // not the last allocation
    assert(arena.Shrink(second, 500, 0));                   // all of it back
    assert(arena.BytesUsed() == 100);
    assert(!arena.Shrink(second, 500, 600));                // cannot grow
    assert(!arena.Shrink(nullptr, 1, 0));
    // an allocation that opened a new chunk shrinks too, and the space is reused
    auto* big = static_cast<uint8_t*>(arena.Allocate(4000, 16));
    assert(big && arena.Shrink(big, 4000, 10));
    auto* next = static_cast<uint8_t*>(arena.Allocate(100, 16));
    assert(next >= big + 10 && next < big + 32);
}

void TestArena() {
    using JitExecMemory::Arena;
    {
        Arena arena(4096);
        assert(arena.BytesUsed() == 0);
        std::vector<std::pair<uint8_t*, size_t>> blocks;
        size_t total = 0;
        for (size_t i = 0; i < 2000; ++i) {
            const size_t bytes = 1 + (i * 37) % 200;
            const size_t align = size_t{1} << (i % 7);  // 1 .. 64
            auto* p = static_cast<uint8_t*>(arena.Allocate(bytes, align));
            assert(p);
            assert(reinterpret_cast<uintptr_t>(p) % align == 0);
            for (size_t b = 0; b < bytes; ++b) {
                p[b] = static_cast<uint8_t>(i);
            }
            blocks.emplace_back(p, bytes);
            total += bytes;
        }
        assert(arena.BytesUsed() == total);
        assert(arena.BytesReserved() > 4096 * 10);  // many chunks
        // Nothing overlaps and nothing was overwritten.
        for (size_t i = 0; i < blocks.size(); ++i) {
            for (size_t b = 0; b < blocks[i].second; ++b) {
                assert(blocks[i].first[b] == static_cast<uint8_t>(i));
            }
        }
        // Oversized: bigger than a chunk, in a chunk of its own.
        const size_t reservedBefore = arena.BytesReserved();
        auto* big = static_cast<uint8_t*>(arena.Allocate(100000, 16));
        assert(big);
        assert(arena.BytesReserved() >= reservedBefore + 100000);
        big[0] = 1;
        big[99999] = 2;
        // The small chunk before it keeps being filled.
        auto* small = static_cast<uint8_t*>(arena.Allocate(8, 8));
        assert(small && (small < big || small >= big + 100000));
        assert(arena.BytesUsed() == total + 100000 + 8);

        assert(arena.Allocate(0) == nullptr);
        assert(arena.Allocate(16, 3) == nullptr);  // not a power of two

        arena.Reset();
        assert(arena.BytesUsed() == 0 && arena.BytesReserved() == 0);
        auto* again = arena.Allocate(64);
        assert(again && arena.BytesUsed() == 64);
    }
    {
        // Default chunk size, and an arena moved into another.
        Arena a;
        void* p = a.Allocate(10);
        assert(p);
        assert(a.BytesReserved() == Arena::kDefaultChunkBytes);
        Arena b(std::move(a));
        assert(a.BytesReserved() == 0 && b.BytesReserved() == Arena::kDefaultChunkBytes);
        assert(b.BytesUsed() == 10);
    }
    {
        // Code emitted into an arena stays valid while later allocations cross chunks.
        using namespace Xbyak::util;
        Arena arena(4096);
        using Fn = int (*)();
        std::vector<Fn> fns;
        for (int i = 0; i < 300; ++i) {
            void* mem = arena.Allocate(64, 16);
            assert(mem);
            Xbyak::CodeGenerator code(64, mem);
            code.mov(eax, i);
            code.ret();
            fns.push_back(code.getCode<Fn>());
        }
        for (int i = 0; i < 300; ++i) {
            assert(fns[i]() == i);
        }
    }
}

#if defined(__linux__)
// Whether the kernel reports transparent huge pages possible for the mapping that holds `address` (THPeligible in smaps), or
// -1 when it cannot tell (no such field, no file).
int ThpEligible(const void* address) {
    FILE* f = std::fopen("/proc/self/smaps", "r");
    if (f == nullptr) return -1;
    const uintptr_t at = reinterpret_cast<uintptr_t>(address);
    char line[512];
    bool inside = false;
    int answer = -1;
    while (std::fgets(line, sizeof line, f) != nullptr) {
        unsigned long lo = 0, hi = 0;
        if (std::sscanf(line, "%lx-%lx ", &lo, &hi) == 2) {
            inside = at >= lo && at < hi;
        } else if (inside && std::strncmp(line, "THPeligible:", 12) == 0) {
            answer = std::atoi(line + 12);
            break;
        }
    }
    std::fclose(f);
    return answer;
}

bool ThpDisabledBySystem() {
    FILE* f = std::fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
    if (f == nullptr) return true;
    char text[128] = {};
    const bool read = std::fgets(text, sizeof text, f) != nullptr;
    std::fclose(f);
    return !read || std::strstr(text, "[never]") != nullptr;
}

// Linux: the default chunk starts on a 2 MiB boundary and is a whole number of huge pages; it is advised for huge pages only with
// BRAWL_JIT_HUGEPAGES=1 (THPeligible then).
void TestArenaHugePages() {
    using JitExecMemory::Arena;
    Arena arena;
    auto* first = static_cast<uint8_t*>(arena.Allocate(100, 16));
    assert(first);
    assert(reinterpret_cast<uintptr_t>(first) % Arena::kHugePageBytes == 0);
    assert(arena.BytesReserved() % Arena::kHugePageBytes == 0);
    first[0] = 1;
    const char* advise = std::getenv("BRAWL_JIT_HUGEPAGES");
    if (!ThpDisabledBySystem() && advise != nullptr && *advise == '1') {
        const int eligible = ThpEligible(first);
        assert(eligible == 1 || eligible == -1);
        (void)eligible;
    }
    // a second chunk is aligned too, and code in it runs
    using namespace Xbyak::util;
    for (int i = 0; i < 3; ++i) {
        auto* big = static_cast<uint8_t*>(arena.Allocate(Arena::kHugePageBytes - 64, 16));
        assert(big);
        Xbyak::CodeGenerator code(64, big);
        code.mov(eax, 7 + i);
        code.ret();
        assert(code.getCode<int (*)()>()() == 7 + i);
    }
    assert(arena.BytesReserved() >= 3 * Arena::kHugePageBytes);
    // a request larger than a chunk is a whole number of huge pages as well
    auto* huge = static_cast<uint8_t*>(arena.Allocate(Arena::kHugePageBytes + 4096, 16));
    assert(huge && reinterpret_cast<uintptr_t>(huge) % Arena::kHugePageBytes == 0);
    huge[Arena::kHugePageBytes + 4095] = 3;
    arena.Reset();
    assert(arena.BytesReserved() == 0);
}

#if defined(__x86_64__)
// Linux x86-64: code space is within a rel32 call of the executable's functions (unless BRAWL_JIT_FAR_MEMORY=1), and code
// there calls one that way.
void TestNearExecutable() {
    using namespace Xbyak::util;
    const char* far = std::getenv("BRAWL_JIT_FAR_MEMORY");
    const bool wantNear = far == nullptr || *far == '\0' || *far == '0';
    const auto reaches = [](const void* from, const void* to) {
        const int64_t d = reinterpret_cast<const uint8_t*>(to) - reinterpret_cast<const uint8_t*>(from);
        return d > INT32_MIN + 64 && d < INT32_MAX - 64;
    };
    JitExecMemory::Arena arena;
    for (int i = 0; i < 3; ++i) {
        void* mem = arena.Allocate(JitExecMemory::Arena::kHugePageBytes - 64, 16);
        assert(mem);
        assert(reaches(mem, reinterpret_cast<const void*>(&CalleeDouble)) == wantNear);
        if (!wantNear) continue;
        Xbyak::CodeGenerator code(64, mem);
        code.push(rbx);
        code.mov(Arg0().cvt32(), 20 + i);
        code.call(reinterpret_cast<const void*>(&CalleeDouble));  // rel32
        code.pop(rbx);
        code.ret();
        assert(code.getCode<int (*)()>()() == 41 + 2 * i);
    }
    JitExecMemory::Region r = JitExecMemory::Allocate(4096);
    assert(r && reaches(r.base, reinterpret_cast<const void*>(&CalleeDouble)) == wantNear);
    JitExecMemory::Free(r);
    // freed near space is handed out again: many more allocations than the reservation has slots
    for (int i = 0; i < 300; ++i) {
        JitExecMemory::Region t = JitExecMemory::Allocate(3u << 20);
        assert(t);
        static_cast<uint8_t*>(t.base)[(3u << 20) - 1] = 1;
        JitExecMemory::Free(t);
    }
}
#endif
#endif

void TestThreads() {
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([] {
            for (int i = 0; i < 200; ++i) {
                JitExecMemory::Region r = JitExecMemory::Allocate(1 + i * 50);
                assert(r);
                static_cast<uint8_t*>(r.base)[0] = 1;
                JitExecMemory::Free(r);
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
}

}  // namespace

int main() {
    TestAvailable();
    TestRegion();
    TestAdd();
    TestCallThroughRegister();
    TestCalleeSavedUsed();
    TestCalleeSavedPreserved();
    TestArena();
    TestArenaShrink();
#if defined(__linux__)
    TestArenaHugePages();
#if defined(__x86_64__)
    TestNearExecutable();
#endif
#endif
    TestThreads();
    std::puts("jit exec memory tests passed");
    return 0;
}
