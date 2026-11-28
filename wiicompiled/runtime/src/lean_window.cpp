// The lean window's OS calls (lean_window.h). Windows reserves it as one placeholder (VirtualAlloc2), so a range can
// later become private memory or a view of a shared section (MapViewOfFile3), as the flat build's reservation does
// (guest_flat_memory.cpp); POSIX maps over its PROT_NONE reservation; Horizon backs it with svcMapMemory
// (platform/horizon.c).
#include "lean_window.h"

#if MKW_LEAN_MEMORY

#include <cerrno>
#include <cstring>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__SWITCH__)
extern "C" void* HorizonLeanWindowReserve(uint64_t size);
extern "C" uint32_t HorizonLeanWindowBack(uint8_t* target, uint64_t size);
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#endif

namespace LeanWindow {

bool g_aliased = false;
bool g_storeView = false;

#if defined(_WIN32)
namespace {
// Placeholder flags, declared here so the build does not depend on the Windows SDK version that first shipped them.
constexpr DWORD kMemReplacePlaceholder = 0x00004000;
constexpr DWORD kMemReservePlaceholder = 0x00040000;
constexpr DWORD kMemPreservePlaceholder = 0x00000002;
using VirtualAlloc2Fn = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void*, ULONG);
using MapViewOfFile3Fn = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, void*, ULONG);
struct Placement {
    VirtualAlloc2Fn virtualAlloc2 = nullptr;
    MapViewOfFile3Fn mapViewOfFile3 = nullptr;
};
// Windows 10 1803 and newer (the flat build needs them too).
const Placement& Api() {
    static const Placement api = [] {
        Placement p;
        HMODULE kernelBase = GetModuleHandleW(L"kernelbase.dll");
        if (kernelBase == nullptr) kernelBase = LoadLibraryW(L"kernelbase.dll");
        if (kernelBase != nullptr) {
            p.virtualAlloc2 = reinterpret_cast<VirtualAlloc2Fn>(GetProcAddress(kernelBase, "VirtualAlloc2"));
            p.mapViewOfFile3 = reinterpret_cast<MapViewOfFile3Fn>(GetProcAddress(kernelBase, "MapViewOfFile3"));
        }
        return p;
    }();
    return api;
}
// Carves [at, at + size) out of the placeholder holding it, so a commit or a view replaces exactly that range. A
// placeholder of exactly that size cannot be split and stays as it is (the call fails harmlessly).
void Split(uint8_t* at, uint64_t size) {
    VirtualFree(at, static_cast<SIZE_T>(size), MEM_RELEASE | kMemPreservePlaceholder);
}
}  // namespace

uint8_t* Reserve(uint64_t size) {
    const Placement& api = Api();
    if (api.virtualAlloc2 == nullptr || api.mapViewOfFile3 == nullptr) return nullptr;
    return static_cast<uint8_t*>(api.virtualAlloc2(GetCurrentProcess(), nullptr, static_cast<SIZE_T>(size),
                                                   MEM_RESERVE | kMemReservePlaceholder, PAGE_NOACCESS, nullptr, 0));
}

uint32_t Commit(uint8_t* at, uint64_t size) {
    Split(at, size);
    void* const committed = Api().virtualAlloc2(GetCurrentProcess(), at, static_cast<SIZE_T>(size),
                                                MEM_RESERVE | MEM_COMMIT | kMemReplacePlaceholder, PAGE_READWRITE,
                                                nullptr, 0);
    return committed == at ? 0u : static_cast<uint32_t>(GetLastError());
}

bool MapSharedViews(uint64_t size, const View* views, int count, std::string* error) {
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, static_cast<DWORD>(size >> 32),
                                        static_cast<DWORD>(size & 0xFFFFFFFFu), nullptr);
    if (section == nullptr) {
        *error = "CreateFileMapping failed (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    for (int i = 0; i < count; ++i) {
        Split(views[i].at, views[i].length);
        void* const view = Api().mapViewOfFile3(section, GetCurrentProcess(), views[i].at, views[i].offset,
                                                static_cast<SIZE_T>(views[i].length), kMemReplacePlaceholder,
                                                PAGE_READWRITE, nullptr, 0);
        if (view != views[i].at) {
            *error = "MapViewOfFile3 failed (error " + std::to_string(GetLastError()) + ")";
            CloseHandle(section);
            return false;
        }
    }
    CloseHandle(section);  // the views keep the section
    return true;
}

uint32_t Protect(uint8_t*, uint64_t, bool) { return 1; }

#elif defined(__SWITCH__)

uint8_t* Reserve(uint64_t size) { return static_cast<uint8_t*>(HorizonLeanWindowReserve(size)); }

uint32_t Commit(uint8_t* at, uint64_t size) { return HorizonLeanWindowBack(at, size); }

bool MapSharedViews(uint64_t, const View*, int, std::string* error) {
    *error = "Horizon cannot map one memory object twice";
    return false;
}

uint32_t Protect(uint8_t*, uint64_t, bool) { return 1; }

#else

uint8_t* Reserve(uint64_t size) {
    void* const reserved = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return reserved == MAP_FAILED ? nullptr : static_cast<uint8_t*>(reserved);
}

uint32_t Commit(uint8_t* at, uint64_t size) {
    // The anonymous PROT_NONE reservation is already demand-zero memory: making it accessible backs it.
    return mprotect(at, size, PROT_READ | PROT_WRITE) == 0 ? 0u : static_cast<uint32_t>(errno);
}

bool MapSharedViews(uint64_t size, const View* views, int count, std::string* error) {
    const auto fail = [&](const char* what, int fd) {
        *error = std::string(what) + " failed (" + std::strerror(errno) + ")";
        if (fd >= 0) close(fd);
        return false;
    };
#if defined(__linux__) && !defined(MKW_LEAN_WINDOW_NO_MEMFD)
    const int fd = memfd_create("lean-guest-memory", MFD_CLOEXEC);
    if (fd < 0) return fail("memfd_create", fd);
#else
    // POSIX hosts without memfd_create (macOS): a POSIX shared memory object, unlinked at once, as Dolphin's
    // MemArenaUnix backs its views. A file would do too, but its pages are the disk's: paged out to the file under memory
    // pressure, and a SIGBUS when the disk is too full to give them blocks. A temporary file only if shm_open fails.
    static std::atomic<unsigned> s_objects{0};
    char name[32];  // macOS allows 31 characters
    std::snprintf(name, sizeof(name), "/lean-%d-%u", static_cast<int>(getpid()), s_objects.fetch_add(1));
    int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd >= 0) {
        shm_unlink(name);
    } else {
        const char* const tmp = std::getenv("TMPDIR");
        std::string path = std::string(tmp != nullptr && *tmp != '\0' ? tmp : "/tmp") + "/lean-guest-memory-XXXXXX";
        fd = mkstemp(path.data());
        if (fd < 0) return fail("shm_open and mkstemp", fd);
        unlink(path.c_str());
    }
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) return fail("preparing the shared memory", fd);
#endif
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) return fail("ftruncate", fd);
    for (int i = 0; i < count; ++i) {
        // MAP_FIXED replaces part of the window's own PROT_NONE reservation, nothing else.
        void* const view = mmap(views[i].at, views[i].length, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd,
                                static_cast<off_t>(views[i].offset));
        if (view != views[i].at) return fail("mmap", fd);
    }
    close(fd);  // the mappings keep the object
    return true;
}

uint32_t Protect(uint8_t* at, uint64_t size, bool writable) {
    return mprotect(at, size, writable ? PROT_READ | PROT_WRITE : PROT_READ) == 0 ? 0u : static_cast<uint32_t>(errno);
}

#endif

bool MapShared(uint64_t size, uint8_t* const* targets, int count, std::string* error) {
    constexpr int kMaxViews = 32;
    if (count < 0 || count > kMaxViews) {
        *error = "too many views";
        return false;
    }
    View views[kMaxViews];
    for (int i = 0; i < count; ++i) views[i] = View{targets[i], 0, size};
    return MapSharedViews(size, views, count, error);
}

}  // namespace LeanWindow

#endif  // MKW_LEAN_MEMORY
