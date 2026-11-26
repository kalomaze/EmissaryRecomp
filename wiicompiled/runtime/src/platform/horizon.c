// Horizon (libnx) platform glue.
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <switch.h>
#include <sys/iosupport.h>  // devoptab_t, devoptab_list: stdout/stderr -> HorizonDebugWrite

extern char _start[];
// Native thread_local. Clang reads the ELF thread pointer from tpidr_el0; libnx keeps it in its thread variables
// (GCC's -mtp=soft reads it through __aarch64_read_tp) and leaves tpidr_el0, which the kernel saves per thread,
// unused. So every thread copies libnx's thread pointer there before running clang-compiled code: the main thread
// first thing in the first initializer, every other thread in its entry (threadCreate is wrapped, which pthreads
// go through too). A thread_local access is then one mrs and an add, where emulated TLS was a call.
void* __aarch64_read_tp(void);
static inline void HorizonSetThreadPointer(void) {
    __asm__ volatile("msr tpidr_el0, %0" : : "r"(__aarch64_read_tp()));
}
typedef struct {
    ThreadFunc entry;
    void* arg;
} HorizonThreadStart;
void HzsRegisterThread(unsigned long long entryOffset);  // hzs telemetry
static void HorizonThreadEntry(void* value) {
    HorizonThreadStart start = *(HorizonThreadStart*)value;
    free(value);
    HorizonSetThreadPointer();
    s32 ideal = -1;
    u64 mask = 0;
    svcGetThreadCoreMask(&ideal, &mask, CUR_THREAD_HANDLE);
    s32 priority = -1;
    svcGetThreadPriority(&priority, CUR_THREAD_HANDLE);
    fprintf(stderr, "[horizon] thread entry +0x%llx: on core %u, ideal core %d, core mask 0x%llx, priority 0x%x\n",
            (unsigned long long)((uintptr_t)start.entry - (uintptr_t)_start), svcGetCurrentProcessorNumber(), ideal, (unsigned long long)mask, priority);
    HzsRegisterThread((unsigned long long)((uintptr_t)start.entry - (uintptr_t)_start));  // hzs telemetry
    start.entry(start.arg);
}
Result __real_threadCreate(Thread* t, ThreadFunc entry, void* arg, void* stack_mem, size_t stack_sz, int prio, int cpuid);
Result __wrap_threadCreate(Thread* t, ThreadFunc entry, void* arg, void* stack_mem, size_t stack_sz, int prio, int cpuid) {
    HorizonThreadStart* start = (HorizonThreadStart*)malloc(sizeof *start);
    if (!start) return MAKERESULT(Module_Libnx, LibnxError_OutOfMemory);
    start->entry = entry;
    start->arg = arg;
    const Result rc = __real_threadCreate(t, HorizonThreadEntry, start, stack_mem, stack_sz, prio, cpuid);
    if (R_FAILED(rc)) free(start);
    return rc;
}

// Every debug line also goes to sdmc:/WiiCompiled/debug.log (next to env.txt): on hardware nothing reads the kernel
// channel, and the runtime's console.log transcript needs pipes Horizon lacks. Flushed per write, so a crash keeps
// every line before it.
static FILE* g_debugLog;
// Buffered (64 KiB, HorizonDebugLogOpen), flushed at most once a second: thousands of boot lines cost a few SD writes
// instead of one each. A crash loses at most the last second.
static void HorizonDebugLogAppend(const char* text, size_t length, int endLine) {
    if (!g_debugLog) return;
    static u64 lastFlush;
    fwrite(text, 1, length, g_debugLog);
    if (endLine && (length == 0 || text[length - 1] != '\n')) fputc('\n', g_debugLog);
    (void)lastFlush;  // hzs: flushed by HzsDebugLogFlush on the sampler thread, never by the writer
}
void HzsDebugLogFlush(void) {
    if (g_debugLog) fflush(g_debugLog);
}
// A whole line: each kernel debug string is one line, so the file gets the newline the string lacks.
static void HorizonDebugWrite(const char* text, size_t length) {
    if (!g_debugLog) svcOutputDebugString(text, length);  // hzs: not when the file has it
    HorizonDebugLogAppend(text, length, 1);
}
// stdout/stderr: C++ streams hand over pieces of a line, so they go to the file exactly as written.
static ssize_t HorizonDebugDeviceWrite(struct _reent* r, void* fd, const char* ptr, size_t len) {
    (void)r;
    (void)fd;
    if (!g_debugLog) svcOutputDebugString(ptr, len);  // hzs: not when the file has it
    HorizonDebugLogAppend(ptr, len, 0);
    return (ssize_t)len;
}
static const devoptab_t g_debugDevice = {.name = "debuglog", .write_r = HorizonDebugDeviceWrite};

// stdout and stderr go to the kernel debug channel (Ryujinx logs it; on hardware a debugger reads it) and debug.log.
__attribute__((constructor(101))) static void HorizonDebugOutput(void) {
    HorizonSetThreadPointer();
    static const char kStart[] = "[horizon] debug output: kernel channel (+ sdmc:/WiiCompiled/debug.log with HZ_DEBUG_LOG=1)";
    HorizonDebugWrite(kStart, sizeof kStart - 1);
    devoptab_list[STD_OUT] = &g_debugDevice;
    devoptab_list[STD_ERR] = &g_debugDevice;
    // Real line buffers: with size 0 newlib writes every printf directive as its own debug string.
    static char outBuffer[4096], errBuffer[4096];
    setvbuf(stdout, outBuffer, _IOLBF, sizeof outBuffer);
    setvbuf(stderr, errBuffer, _IOLBF, sizeof errBuffer);
    fprintf(stderr, "[horizon] debug output: stderr\n");
}

// Horizon has no environment for a program to inherit, so KEY=VALUE lines from sdmc:/WiiCompiled/env.txt
// become it, before any other initializer reads one: every debug switch of the runtime, SDL, Aurora and
// Mesa then changes with a text edit instead of a rebuild.
__attribute__((constructor(102))) static void HorizonEnvironment(void) {
    // Mesa's disk shader cache (sdmc:/.mesa) is keyed by the driver version string, not its build: a rebuilt NVK
    // keeps loading shaders compiled by the previous build. Off until it is keyed by build; env.txt can override.
    setenv("MESA_SHADER_CACHE_DISABLE", "false", 0);  // hzs: keyed by nvk-build-id
    FILE* file = fopen("sdmc:/WiiCompiled/env.txt", "r");
    if (!file) return;
    // Lines of any length: a fixed 512-byte fgets buffer cut a long BRAWL_PAD_SCRIPT off silently, EXIT included,
    // and parsed the rest of the line as more lines. (newlib names POSIX getline __getline.)
    char* line = NULL;
    size_t capacity = 0;
    while (__getline(&line, &capacity, file) != -1) {
        char* end = line + strlen(line);
        while (end > line && (end[-1] == '\n' || end[-1] == '\r' || end[-1] == ' ')) *--end = '\0';
        char* equals = strchr(line, '=');
        if (line[0] == '#' || !equals || equals == line) continue;
        *equals = '\0';
        setenv(line, equals + 1, 1);
        fprintf(stderr, "[horizon] env %s=%s\n", line, equals + 1);
    }
    free(line);
    fclose(file);
}

// Hold ZR while launching: vanilla Brawl for this run (HZS_VANILLA=1 -> RuntimeConfigFile::ModsVanilla(): no Gecko
// codes, no SD card), with the same Config.toml, NAND and save. Read once, before the runtime reads its config.
__attribute__((constructor(104))) static void HorizonLaunchButtons(void) {
    if (R_FAILED(hidInitialize())) return;
    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    PadState pad;
    padInitializeDefault(&pad);
    // A new process's pad state starts filling only after input is configured, so one read can see nothing even with
    // ZR held: poll up to 0.5 s, counting ZR seen in any sample, and stop once a controller has reported a few samples.
    bool zr = false;
    int connectedSamples = 0;
    for (int i = 0; i < 50 && connectedSamples < 5; i++) {
        padUpdate(&pad);
        if (padIsConnected(&pad)) {
            connectedSamples++;
            if (padGetButtons(&pad) & HidNpadButton_ZR) zr = true;
        }
        svcSleepThread(10000000);  // 10 ms
    }
    fprintf(stderr, "[horizon] launch buttons: controller %s after %d samples, ZR %s\n",
            connectedSamples ? "seen" : "NOT seen", connectedSamples, zr ? "held" : "not held");
    if (zr) {
        setenv("HZS_VANILLA", "1", 1);
        fprintf(stderr, "[horizon] ZR held at launch: vanilla Brawl (no codes, no SD card)\n");
    }
}

// debug.log only on request (env.txt HZ_DEBUG_LOG=1): a flush per write costs a filesystem round trip on the console,
// and P+'s boot alone writes thousands of lines.
__attribute__((constructor(103))) static void HorizonDebugLogOpen(void) {
    const char* value = getenv("HZ_DEBUG_LOG");
    if (!value || value[0] != '1') return;
    {  // hzs: one numbered log per launch
        unsigned n = 0;
        FILE* c = fopen("sdmc:/WiiCompiled/launch_count.txt", "r");
        if (c) { if (fscanf(c, "%u", &n) != 1) n = 0; fclose(c); }
        ++n;
        c = fopen("sdmc:/WiiCompiled/launch_count.txt", "w");
        if (c) { fprintf(c, "%u\n", n); fclose(c); }
        char path[64];
        snprintf(path, sizeof path, "sdmc:/WiiCompiled/debug-%04u.log", n);
        g_debugLog = fopen(path, "w");
        if (g_debugLog) fprintf(g_debugLog, "=== launch %u ===\n", n);
    }
    static char buffer[64 * 1024];
    if (g_debugLog) setvbuf(g_debugLog, buffer, _IOFBF, sizeof buffer);
}

volatile unsigned long long g_hzPerf[48];
static u64 g_hzPeriodMs = 5000;  // the sampler's counter window (HZ_PERF_PERIOD_MS)

// BRAWL_PAD_SCRIPT's EXIT: end the process from inside, as HZ_EXIT_AFTER does.
void HorizonTrace(const char* text);
void HzDumpFallbackPages(void);  // memory.cpp: top guest pages taking the out-of-line store path
void HorizonExitNow(void) {
    HorizonTrace("[horizon] pad script EXIT: exiting");
    svcExitProcess();
}

// Emulated TLS without the pthread layer (linked with --wrap=__emutls_get_address), for any object still built with
// -femulated-tls (this build uses native TLS, above): every such thread_local access calls __emutls_get_address,
// which in libgcc goes through __gthread_getspecific -> pthread_getspecific -> __syscall_tls_get -> armGetTls. That
// chain was ~10% of the main thread's samples in the menus (the indirect-call scope guard alone touches a
// thread_local three times per call). Same contract as libgcc's emutls: each variable gets an index on first use,
// each thread an array of its instances (in one libnx TLS slot), each instance a copy of the variable's template
// (or zeros), freed with the thread. Every caller is clang-compiled code reaching this through the wrap; libgcc's
// own __emutls_get_address is never called, so the two index spaces never mix.
typedef struct {
    uintptr_t size;
    uintptr_t align;
    uintptr_t index;  // 0 until first use; then 1-based
    const void* templ;
} EmuTlsObject;
typedef struct {
    uintptr_t capacity;
    void* slots[];
} EmuTlsArray;
static Mutex g_emuTlsLock;
static s32 g_emuTlsSlot = -1;
static uintptr_t g_emuTlsCount;
static void EmuTlsFreeThread(void* value) {
    EmuTlsArray* array = (EmuTlsArray*)value;
    if (!array) return;
    for (uintptr_t i = 0; i < array->capacity; ++i) free(array->slots[i]);
    free(array);
}
static __attribute__((noinline)) void* EmuTlsSlow(EmuTlsObject* object) {
    mutexLock(&g_emuTlsLock);
    if (g_emuTlsSlot < 0) g_emuTlsSlot = threadTlsAlloc(EmuTlsFreeThread);
    uintptr_t index = object->index;
    if (index == 0) {
        index = ++g_emuTlsCount;
        __atomic_store_n(&object->index, index, __ATOMIC_RELEASE);
    }
    const s32 slot = g_emuTlsSlot;
    mutexUnlock(&g_emuTlsLock);
    EmuTlsArray* array = (EmuTlsArray*)threadTlsGet(slot);
    if (!array || index > array->capacity) {  // this thread's array, grown to hold the index
        uintptr_t capacity = array ? array->capacity * 2 : 32;
        if (capacity < index) capacity = index + 32;
        EmuTlsArray* grown = (EmuTlsArray*)realloc(array, sizeof(EmuTlsArray) + capacity * sizeof(void*));
        if (!grown) abort();
        const uintptr_t old = array ? grown->capacity : 0;
        memset(&grown->slots[old], 0, (capacity - old) * sizeof(void*));
        grown->capacity = capacity;
        threadTlsSet(slot, grown);
        array = grown;
    }
    void* instance = array->slots[index - 1];
    if (!instance) {
        const uintptr_t align = object->align > sizeof(void*) ? object->align : sizeof(void*);
        const uintptr_t size = (object->size + align - 1) & ~(align - 1);
        instance = aligned_alloc(align, size ? size : align);
        if (!instance) abort();
        if (object->templ) memcpy(instance, object->templ, object->size);
        else memset(instance, 0, object->size);
        array->slots[index - 1] = instance;
    }
    return instance;
}
void* __wrap___emutls_get_address(EmuTlsObject* object) {
    const uintptr_t index = __atomic_load_n(&object->index, __ATOMIC_ACQUIRE);
    if (__builtin_expect(index != 0, 1)) {
        const EmuTlsArray* array = (const EmuTlsArray*)threadTlsGet(g_emuTlsSlot);
        if (__builtin_expect(array != NULL && index <= array->capacity, 1)) {
            void* instance = array->slots[index - 1];
            if (__builtin_expect(instance != NULL, 1)) return instance;
        }
    }
    return EmuTlsSlow(object);
}

// A line on the kernel debug channel that does not depend on stdio.
void HorizonTrace(const char* text) {
    size_t length = 0;
    while (text[length]) ++length;
    HorizonDebugWrite(text, length);
}

// Lean guest memory (runtime/include/memory_access.h, MKW_LEAN_MEMORY): every guest access is
// window + (address & 0x3FFFFFFF) into one 1 GiB host window. Horizon has no lazily backed anonymous memory, so the
// window is address space reserved in the stack region (svcMapMemory's destination), and only the ranges
// memory.cpp's BackLeanWindow names are backed (as on every OS), with heap memory aliased in by svcMapMemory as libnx
// maps thread stacks.
#if MKW_LEAN_MEMORY
#define kLeanWindowBytes (1ull << 30)  // memory_access.h, kLeanWindowSize
static u8* g_leanWindow;

// The window's address space is reserved before any static initializer runs, while no thread has been created:
// libnx places each thread's stack at a random spot in the stack region, and a few of them can leave no 1 GiB hole
// (under Ryujinx the region is 0x78000000 bytes, shared with the code). The first free stretch of the region that
// holds it 2 MiB aligned, with 64 KiB of guard on each side (backing aligned alike can then be mapped with 2 MiB
// blocks). Memory::Init backs parts of it (HorizonLeanWindowBack).
__attribute__((constructor(103))) static void HorizonLeanWindowEarly(void) {
    const u64 block = 2u << 20, guard = 0x10000;
    u64 stackBase = 0, stackSize = 0, total = 0, used = 0;
    svcGetInfo(&stackBase, InfoType_StackRegionAddress, CUR_PROCESS_HANDLE, 0);
    svcGetInfo(&stackSize, InfoType_StackRegionSize, CUR_PROCESS_HANDLE, 0);
    const u64 stackEnd = stackBase + stackSize;
    virtmemLock();
    for (u64 address = stackBase; address < stackEnd && !g_leanWindow;) {
        MemoryInfo info;
        u32 pageInfo = 0;
        if (R_FAILED(svcQueryMemory(&info, &pageInfo, address)) || info.size == 0) break;
        const u64 from = info.addr > stackBase ? info.addr : stackBase;
        const u64 end = info.addr + info.size < stackEnd ? info.addr + info.size : stackEnd;
        const u64 start = (from + guard + block - 1) & ~(block - 1);
        // The guard after the window is reserved with it (memory_access.h, kLeanGuardSize): nothing else maps there.
        if (info.type == MemType_Unmapped && start + kLeanWindowBytes + guard <= end &&
            virtmemAddReservation((void*)start, kLeanWindowBytes + guard))
            g_leanWindow = (u8*)start;
        address = info.addr + info.size;
    }
    virtmemUnlock();
    svcGetInfo(&total, InfoType_TotalMemorySize, CUR_PROCESS_HANDLE, 0);
    svcGetInfo(&used, InfoType_UsedMemorySize, CUR_PROCESS_HANDLE, 0);
    fprintf(stderr, "[horizon] lean window %p+0x%llx in the stack region 0x%llx+0x%llx%s; process memory %llu of %llu MiB\n",
            (void*)g_leanWindow, (unsigned long long)kLeanWindowBytes, (unsigned long long)stackBase,
            (unsigned long long)stackSize, g_leanWindow ? "" : ": NO ROOM, its map:", (unsigned long long)(used >> 20),
            (unsigned long long)(total >> 20));
    for (u64 address = stackBase; !g_leanWindow && address < stackEnd;) {
        MemoryInfo info;
        u32 pageInfo = 0;
        if (R_FAILED(svcQueryMemory(&info, &pageInfo, address)) || info.size == 0) break;
        fprintf(stderr, "[horizon]   0x%010llx+0x%09llx type 0x%x perm %u\n", (unsigned long long)info.addr,
                (unsigned long long)info.size, (unsigned)info.type, (unsigned)info.perm);
        address = info.addr + info.size;
    }
}

// The window reserved above; NULL if there was no room.
void* HorizonLeanWindowReserve(u64 size) {
    return size <= kLeanWindowBytes + 0x10000 ? g_leanWindow : NULL;  // the window and its guard
}

// Backs [target, target + size) of the window, unbacked and in whole 64 KiB chunks (BackLeanWindow). The backing reads
// zero. 0, or the failing Result.
u32 HorizonLeanWindowBack(u8* target, u64 size) {
    const u64 block = 2u << 20;
    void* const heap = aligned_alloc(size >= block && ((uintptr_t)target & (block - 1)) == 0 ? block : 0x1000, size);
    if (!heap) return MAKERESULT(Module_Libnx, LibnxError_OutOfMemory);
    const Result rc = svcMapMemory(target, heap, size);  // the heap pages stay aliased here for good
    if (R_FAILED(rc)) {
        free(heap);
        return rc;
    }
    memset(target, 0, size);
    return 0;
}
#endif

// The display's size: the default VI layer's (1280x720 docked, the panel handheld). Returns 0 on failure.
int HorizonDisplaySize(unsigned* width, unsigned* height) {
    u32 w = 0, h = 0;
    if (R_FAILED(nwindowGetDimensions(nwindowGetDefault(), &w, &h)) || w == 0 || h == 0) return 0;
    *width = w;
    *height = h;
    return 1;
}

// Every static initializer, logged before and after it runs (linked with --wrap=__libc_init_array):
// a hang before main() names its initializer. Offsets match `readelf`'s .init_array and addr2line.
extern void (*__preinit_array_start[])(void);
extern void (*__preinit_array_end[])(void);
extern void (*__init_array_start[])(void);
extern void (*__init_array_end[])(void);
extern void _init(void);
// The image's load address: crt0's _start is at offset 0 and is reached PC-relative. (libnx's
// __start__ is an absolute 0, not the load address.)
extern char _start[];
#define IMAGE_OFFSET(address) ((u64)(uintptr_t)(address) - (u64)(uintptr_t)_start)
static void Append(char** out, const char* text);
static void AppendHex(char** out, u64 value);
static void TraceInitializer(const char* what, u64 index, u64 count, void (*function)(void)) {
    char line[128];
    char* out = line;
    Append(&out, "[horizon] init ");
    AppendHex(&out, index);
    Append(&out, "/");
    AppendHex(&out, count);
    Append(&out, " ");
    Append(&out, what);
    Append(&out, " +");
    AppendHex(&out, IMAGE_OFFSET(function));
    HorizonDebugWrite(line, (u64)(out - line));
}
void __wrap___libc_init_array(void) {
    for (void (**f)(void) = __preinit_array_start; f < __preinit_array_end; ++f) (*f)();
    _init();
    const u64 count = (u64)(__init_array_end - __init_array_start);
    for (u64 i = 0; i < count; ++i) {
        TraceInitializer("start", i, count, __init_array_start[i]);
        __init_array_start[i]();
        TraceInitializer("done", i, count, __init_array_start[i]);
    }
}

// Every file, directory, thread, wait and exit call, logged on entry and return with the calling
// thread (linked with --wrap for each): a hang or failure names the call and its arguments.
static void TraceCall(const char* call, const char* what, const char* text, u64 value) {
    char line[400];
    char* out = line;
    u64 thread = 0;
    svcGetThreadId(&thread, CUR_THREAD_HANDLE);
    Append(&out, "[horizon] t");
    AppendHex(&out, thread);
    Append(&out, " ");
    Append(&out, call);
    Append(&out, " ");
    Append(&out, what);
    if (text) {
        Append(&out, " ");
        for (int i = 0; text[i] && i < 200; ++i) *out++ = text[i];
    }
    Append(&out, " ");
    AppendHex(&out, value);
    HorizonDebugWrite(line, (u64)(out - line));
}

#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
FILE* __real_fopen(const char*, const char*);
FILE* __wrap_fopen(const char* path, const char* mode) {
    TraceCall("fopen", mode, path, 0);
    FILE* f = __real_fopen(path, mode);
    TraceCall("fopen", "=", path, (u64)(uintptr_t)f);
    return f;
}
int __real_open(const char*, int, ...);
int __wrap_open(const char* path, int flags, ...) {
    va_list args;
    va_start(args, flags);
    const int mode = va_arg(args, int);
    va_end(args);
    TraceCall("open", "", path, (u64)flags);
    const int fd = __real_open(path, flags, mode);
    TraceCall("open", "=", path, (u64)(s64)fd);
    return fd;
}
DIR* __real_opendir(const char*);
DIR* __wrap_opendir(const char* path) {
    TraceCall("opendir", "", path, 0);
    DIR* d = __real_opendir(path);
    TraceCall("opendir", "=", path, (u64)(uintptr_t)d);
    return d;
}
struct dirent* __real_readdir(DIR*);
struct dirent* __wrap_readdir(DIR* d) {
    struct dirent* e = __real_readdir(d);
    TraceCall("readdir", "=", e ? e->d_name : "(end)", (u64)(uintptr_t)d);
    return e;
}
int __real_stat(const char*, struct stat*);
int __wrap_stat(const char* path, struct stat* st) {
    const int r = __real_stat(path, st);
    TraceCall("stat", "=", path, (u64)(s64)r);
    return r;
}
int __real_mkdir(const char*, mode_t);
int __wrap_mkdir(const char* path, mode_t mode) {
    const int r = __real_mkdir(path, mode);
    TraceCall("mkdir", "=", path, (u64)(s64)r);
    return r;
}
int __real_rename(const char*, const char*);
int __wrap_rename(const char* from, const char* to) {
    const int r = __real_rename(from, to);
    TraceCall("rename", "=", from, (u64)(s64)r);
    (void)to;
    return r;
}
int __real_unlink(const char*);
int __wrap_unlink(const char* path) {
    const int r = __real_unlink(path);
    TraceCall("unlink", "=", path, (u64)(s64)r);
    return r;
}
int __real_pthread_create(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);
int HzsPthreadCreate(pthread_t*, const pthread_attr_t*, void* (*)(void*), void*);  // hzs telemetry
int __wrap_pthread_create(pthread_t* thread, const pthread_attr_t* attr, void* (*start)(void*), void* arg) {
    TraceCall("pthread_create", "start", NULL, IMAGE_OFFSET(start));
    const int r = HzsPthreadCreate(thread, attr, start, arg);  // hzs telemetry: labels the thread
    TraceCall("pthread_create", "=", NULL, (u64)(s64)r);
    return r;
}
int __real_pthread_join(pthread_t, void**);
int __wrap_pthread_join(pthread_t thread, void** value) {
    TraceCall("pthread_join", "", NULL, (u64)(uintptr_t)thread);
    const int r = __real_pthread_join(thread, value);
    TraceCall("pthread_join", "=", NULL, (u64)(s64)r);
    return r;
}
int __real_pthread_cond_wait(pthread_cond_t*, pthread_mutex_t*);
// Counted (perf slots condWait/nanosleep); traced only when a wait outlasts a second, so a stuck waiter is still named
// without a log line per audio-rate wakeup.
int __wrap_pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex) {
    g_hzPerf[12] += 1;
    const u64 start = armGetSystemTick();
    const int r = __real_pthread_cond_wait(cond, mutex);
    if (armGetSystemTick() - start > armGetSystemTickFreq()) TraceCall("cond_wait >1s", "=", NULL, (u64)(uintptr_t)cond);
    return r;
}
int __real_nanosleep(const struct timespec*, struct timespec*);
int __wrap_nanosleep(const struct timespec* duration, struct timespec* remaining) {
    g_hzPerf[13] += 1;
    if (duration->tv_sec >= 1) TraceCall("nanosleep", "ns", NULL, (u64)duration->tv_sec * 1000000000ull + (u64)duration->tv_nsec);
    return __real_nanosleep(duration, remaining);
}
void __real_exit(int) __attribute__((noreturn));
void __wrap_exit(int code) {
    TraceCall("exit", "", NULL, (u64)(s64)code);
    __real_exit(code);
}
void __real_abort(void) __attribute__((noreturn));
void __wrap_abort(void) {
    TraceCall("abort", "", NULL, 0);
    __real_abort();
}
// C++ throws: the exception's type name (a mangled typeinfo name).
struct HorizonTypeInfo { const void* vtable; const char* name; };
static bool Readable(u64 address);
void __real___cxa_throw(void*, void*, void (*)(void*)) __attribute__((noreturn));
void __wrap___cxa_throw(void* object, void* type, void (*destructor)(void*)) {
    const char* name = type ? ((const struct HorizonTypeInfo*)type)->name : "?";
    // The value is the throwing call site, as an image offset for addr2line.
    TraceCall("throw", "", name, IMAGE_OFFSET(__builtin_return_address(0)));
    // The chain, by frame pointers (no unwinder), written through to the SD card first: Ryujinx
    // dies on the throw that follows and loses the end of its own log.
    FILE* file = fopen("sdmc:/WiiCompiled/stacks.txt", "a");
    if (file) {
        fprintf(file, "throw %s:", name);
        u64 frame = (u64)(uintptr_t)__builtin_frame_address(0);
        for (int depth = 0; depth < 24 && frame && !(frame & 15) && Readable(frame); ++depth) {
            fprintf(file, " 0x%lx", (unsigned long)IMAGE_OFFSET(((const u64*)(uintptr_t)frame)[1]));
            frame = ((const u64*)(uintptr_t)frame)[0];
        }
        fputc('\n', file);
        fclose(file);
    }
    __real___cxa_throw(object, type, destructor);
}

// The console's controller, as an SDL gamepad: SDL has no Horizon joystick driver, so a virtual
// gamepad is attached once SDL's joystick subsystem is up and fed from libnx's pad state at 60 Hz.
// Buttons by label, not position: the runtime reads the GameCube A from SDL's south button (an Xbox
// pad's A), so the Switch's A goes there.
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_joystick.h>
#include <SDL3/SDL_gamepad.h>
static void FeedGamepad(void* unused) {
    (void)unused;
    while (!SDL_WasInit(SDL_INIT_JOYSTICK)) svcSleepThread(100000000ull);
    SDL_VirtualJoystickDesc desc;
    SDL_INIT_INTERFACE(&desc);
    desc.type = SDL_JOYSTICK_TYPE_GAMEPAD;
    desc.naxes = SDL_GAMEPAD_AXIS_COUNT;
    desc.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
    desc.name = "Nintendo Switch controller";
    const SDL_JoystickID id = SDL_AttachVirtualJoystick(&desc);
    SDL_Joystick* joystick = id ? SDL_OpenJoystick(id) : NULL;
    HorizonTrace(joystick ? "[horizon] gamepad: attached" : "[horizon] gamepad: SDL refused the virtual gamepad");
    if (!joystick) return;
    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    PadState pad;
    padInitializeDefault(&pad);
    static const struct { u64 nx; SDL_GamepadButton sdl; } kButtons[] = {
        {HidNpadButton_A, SDL_GAMEPAD_BUTTON_SOUTH}, {HidNpadButton_B, SDL_GAMEPAD_BUTTON_EAST},
        {HidNpadButton_X, SDL_GAMEPAD_BUTTON_NORTH}, {HidNpadButton_Y, SDL_GAMEPAD_BUTTON_WEST},
        {HidNpadButton_Plus, SDL_GAMEPAD_BUTTON_START}, {HidNpadButton_Minus, SDL_GAMEPAD_BUTTON_BACK},
        {HidNpadButton_L, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER}, {HidNpadButton_R, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER},
        {HidNpadButton_StickL, SDL_GAMEPAD_BUTTON_LEFT_STICK}, {HidNpadButton_StickR, SDL_GAMEPAD_BUTTON_RIGHT_STICK},
        {HidNpadButton_Up, SDL_GAMEPAD_BUTTON_DPAD_UP}, {HidNpadButton_Down, SDL_GAMEPAD_BUTTON_DPAD_DOWN},
        {HidNpadButton_Left, SDL_GAMEPAD_BUTTON_DPAD_LEFT}, {HidNpadButton_Right, SDL_GAMEPAD_BUTTON_DPAD_RIGHT},
    };
    for (;;) {
        padUpdate(&pad);
        const u64 held = padGetButtons(&pad);
        for (size_t i = 0; i < sizeof kButtons / sizeof kButtons[0]; ++i)
            SDL_SetJoystickVirtualButton(joystick, kButtons[i].sdl, (held & kButtons[i].nx) != 0);
        const HidAnalogStickState left = padGetStickPos(&pad, 0), right = padGetStickPos(&pad, 1);
        // libnx: up is positive; SDL: down is positive.
        SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_LEFTX, (Sint16)left.x);
        SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_LEFTY, (Sint16)(left.y == -32768 ? 32767 : -left.y));
        SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_RIGHTX, (Sint16)right.x);
        SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_RIGHTY, (Sint16)(right.y == -32768 ? 32767 : -right.y));
        SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, (held & HidNpadButton_ZL) ? 32767 : 0);
        SDL_SetJoystickVirtualAxis(joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, (held & HidNpadButton_ZR) ? 32767 : 0);
        svcSleepThread(16666667ull);
    }
}
static Thread g_gamepad;
__attribute__((constructor(104))) static void StartGamepad(void) {
    if (R_SUCCEEDED(threadCreate(&g_gamepad, FeedGamepad, NULL, NULL, 0x8000, 0x2C, 2))) threadStart(&g_gamepad);
}

// Where the main thread is, every 5 s: a self-sampling watchdog, since there is no debugger to ask.
// It pauses the main thread, reads its registers and frame-pointer chain, and prints them as offsets
// into the image (addr2line -f -C -e WiiCompiled <offset>). Formatting is by hand: the paused
// thread may hold the C library's locks.
static void Append(char** out, const char* text) { while (*text) *(*out)++ = *text++; }
static void AppendHex(char** out, u64 value) {
    Append(out, "0x");
    int digits = 1;
    while (digits < 16 && (value >> (digits * 4))) ++digits;
    while (digits--) *(*out)++ = "0123456789abcdef"[(value >> (digits * 4)) & 15];
}
static void AppendDec(char** out, u64 value) {
    char digits[20];
    int count = 0;
    do digits[count++] = (char)('0' + value % 10); while ((value /= 10) && count < 20);
    while (count--) *(*out)++ = digits[count];
}
static bool Readable(u64 address) {
    MemoryInfo info;
    u32 page;
    return R_SUCCEEDED(svcQueryMemory(&info, &page, address)) && (info.perm & Perm_R) &&
           address + 16 <= info.addr + info.size;
}
Handle HzsTargetThread(void);  // hzs telemetry
static void SampleMainThread(void* unused) {
    (void)unused;
    const u64 base = (u64)(uintptr_t)_start;
    HorizonTrace("[horizon] main-thread sampler: running");
    // HZ_EXIT_AFTER=<seconds>: the process ends itself then. A test run that kills the emulator from outside while it
    // renders has hung the host GPU (Xid 13/109); ended from inside, the emulator stops rendering first.
    const char* exitAfterText = getenv("HZ_EXIT_AFTER");
    const u64 exitAfter = exitAfterText ? strtoull(exitAfterText, NULL, 10) : 0;
    // HZ_PERF_PERIOD_MS: the counter window (default 5000); a short one gets a measurement out of a short run.
    const char* periodText = getenv("HZ_PERF_PERIOD_MS");
    g_hzPeriodMs = periodText ? strtoull(periodText, NULL, 10) : 5000;
    if (g_hzPeriodMs < 100) g_hzPeriodMs = 100;
    for (u64 elapsedMs = g_hzPeriodMs;; elapsedMs += g_hzPeriodMs) {
        const u64 seconds = elapsedMs / 1000;
        svcSleepThread(g_hzPeriodMs * 1000000ull);
        if (exitAfter && seconds >= exitAfter) {
            HorizonTrace("[horizon] HZ_EXIT_AFTER reached: exiting");
            svcExitProcess();
        }
        HorizonTrace("[horizon] main-thread sampler: woke");
        const Handle main = HzsTargetThread();  // hzs: the game thread
        const Result paused = svcSetThreadActivity(main, ThreadActivity_Paused);
        if (R_FAILED(paused)) {  // hzs: no log write while the main thread is paused (FILE lock deadlock)
            char note[64];
            char* end = note;
            Append(&end, "[horizon] main-thread sampler: pause failed ");
            AppendHex(&end, paused);
            HorizonDebugWrite(note, (u64)(end - note));
            continue;
        }
        ThreadContext context;
        const Result result = svcGetThreadContext3(&context, main);
        char line[600];
        char* out = line;
        Append(&out, "[horizon] main thread at ");
        AppendHex(&out, seconds);
        Append(&out, "s:");
        if (R_SUCCEEDED(result)) {
            Append(&out, " pc ");
            AppendHex(&out, context.pc.x - base);
            Append(&out, " lr ");
            AppendHex(&out, context.lr - base);
            Append(&out, " frames");
            u64 frame = context.fp;
            for (int depth = 0; depth < 24 && frame && !(frame & 15) && Readable(frame); ++depth) {
                const u64 returnAddress = ((const u64*)(uintptr_t)frame)[1];
                Append(&out, " ");
                AppendHex(&out, returnAddress - base);
                frame = ((const u64*)(uintptr_t)frame)[0];
            }
        } else {
            Append(&out, " no context");
        }
        svcSetThreadActivity(main, ThreadActivity_Runnable);
        HorizonDebugWrite(line, (u64)(out - line));
        // Cumulative hot-path counters (hz_perf.h); time slots are counter ticks at 19.2 MHz. 32 slots of
        // " name=0x..." outgrow the context line's buffer.
        char perfLine[1024];
        out = perfLine;
        Append(&out, "[horizon] perf");
        static const char* const kNames[] = {"sync", "syncWork", "pagesCmp", "syncTicks", "patched", "patchedTicks",
                                             "audioBlocks", "audioCbTicks", "audioTicks", "noteWrite", "edgeDirty",
                                             "frames", "condWait", "nanosleep", "interpInsns", "fifoWrites",
                                             "fifoTicks", "dlCalls", "dlTicks", "dlBytes", "dlProbe",
                                             "dlApply", "dlAurora", "dlSync", "audioTickT", "copyTexT",
                                             "copyDispT", "drawDoneT", "patchedTopT", "auDrawT", "auUnmerged",
                                             "auMerged", "auPipeT", "auBindT", "auUniformT", "auArrayT", "s36",
                                             "s37", "s38", "s39"};
        for (int i = 0; i < 40; ++i) {
            if (!g_hzPerf[i]) continue;
            Append(&out, " ");
            Append(&out, kNames[i]);
            Append(&out, "=");
            AppendHex(&out, g_hzPerf[i]);
        }
        HorizonDebugWrite(perfLine, (u64)(out - perfLine));
        // The same window as microseconds per frame: each frame phase's change since the last window, divided by
        // the frames in it. Phases nest (display lists inside FIFO writes and patched calls), so they do not sum.
        static unsigned long long previous[48];
        unsigned long long now[48];
        for (int i = 0; i < 48; ++i) now[i] = g_hzPerf[i];
        const unsigned long long frames = now[11] - previous[11];
        if (frames) {
            static const int kPhases[] = {16, 18, 22, 20, 21, 23, 24, 39, 7, 25, 26, 27, 28, 3, 29, 35, 32, 33, 34,
                                          36, 37, 38};
            static const char* const kPhaseNames[] = {"fifo", "dl", "dlAurora", "dlProbe", "dlApply", "dlSync",
                                                      "audio", "audioJoin", "audioCb", "copyTex", "copyDisp",
                                                      "drawDone", "patched", "sync", "auDraw", "auArrays",
                                                      "auPipeline", "auBind", "auUniform", "gecko", "interp", "hle"};
            const u64 freq = armGetSystemTickFreq();
            out = perfLine;
            Append(&out, "[horizon] frame-us frames=");
            AppendDec(&out, frames);
            Append(&out, " frame=");
            AppendDec(&out, g_hzPeriodMs * 1000ull / frames);
            // The main thread's own run time per frame (ticks it was scheduled on any core): far below frame= means
            // it spends the frame blocked (GPU fences, vsync, events), not computing.
            static u64 previousCpu;
            u64 cpu = 0;
            if (R_SUCCEEDED(svcGetInfo(&cpu, InfoType_ThreadTickCount, main, UINT64_MAX))) {
                Append(&out, " cpu=");
                AppendDec(&out, (cpu - previousCpu) * 1000000ull / freq / frames);
                previousCpu = cpu;
            }
            for (unsigned p = 0; p < sizeof kPhases / sizeof kPhases[0]; ++p) {
                const int slot = kPhases[p];
                Append(&out, " ");
                Append(&out, kPhaseNames[p]);
                Append(&out, "=");
                AppendDec(&out, (now[slot] - previous[slot]) * 1000000ull / freq / frames);
            }
            Append(&out, " fifoWrites=");
            AppendDec(&out, (now[15] - previous[15]) / frames);
            Append(&out, " dlCalls=");
            AppendDec(&out, (now[17] - previous[17]) / frames);
            Append(&out, " auUnmerged=");
            AppendDec(&out, (now[30] - previous[30]) / frames);
            Append(&out, " auMerged=");
            AppendDec(&out, (now[31] - previous[31]) / frames);
            Append(&out, " fbReads=");
            AppendDec(&out, (now[40] - previous[40]) / frames);
            Append(&out, " fbWrites=");
            AppendDec(&out, (now[41] - previous[41]) / frames);
            HorizonDebugWrite(perfLine, (u64)(out - perfLine));
        }
        for (int i = 0; i < 48; ++i) previous[i] = now[i];
        HzDumpFallbackPages();
    }
}
// HZ_PROF=1: a statistical profile of the main thread. Every 2 ms it is paused and its PC and LR recorded (as image
// offsets); every 5 s the 40 hottest of each are printed and the tables cleared. Symbolize offline:
// addr2line -f -C -e WiiCompiled <offset>. Off by default: the pauses cost the main thread a little time.
enum { kProfSlots = 8192 };
typedef struct {
    u64 key[kProfSlots];
    u32 count[kProfSlots];
} ProfTable;
static ProfTable g_profPc, g_profLr, g_profInc, g_profCond;
static u32 g_profSamples;
static void ProfAdd(ProfTable* table, u64 key) {
    u32 slot = (u32)(((key >> 2) * 0x9E3779B97F4A7C15ull) >> 51);
    for (u32 probe = 0; probe < kProfSlots; ++probe) {
        const u32 at = (slot + probe) & (kProfSlots - 1);
        if (table->key[at] == key) {
            ++table->count[at];
            return;
        }
        if (table->key[at] == 0) {
            table->key[at] = key;
            table->count[at] = 1;
            return;
        }
    }
}
static void ProfDump(ProfTable* table, const char* what, int top) {
    char line[600];
    for (int rank = 0, printed = 0; rank < top; ++rank) {
        u32 best = 0, bestAt = 0;
        for (u32 at = 0; at < kProfSlots; ++at) {
            if (table->count[at] > best) best = table->count[at], bestAt = at;
        }
        if (best == 0) break;
        if (printed % 8 == 0) {
            if (printed) HorizonDebugWrite(line, (u64)(strlen(line)));
            char* out = line;
            Append(&out, "[horizon] prof ");
            Append(&out, what);
            Append(&out, " of ");
            AppendHex(&out, g_profSamples);
            Append(&out, ":");
            *out = '\0';
        }
        char* out = line + strlen(line);
        Append(&out, " ");
        AppendHex(&out, table->key[bestAt]);
        Append(&out, "=");
        AppendHex(&out, best);
        *out = '\0';
        table->count[bestAt] = 0;  // taken; the table is cleared below anyway
        ++printed;
    }
    HorizonDebugWrite(line, (u64)(strlen(line)));
    memset(table, 0, sizeof *table);
}
Handle HzsTargetThread(void);  // hzs telemetry: the game thread once it presents a frame
static void ProfileMainThread(void* unused) {
    (void)unused;
    const u64 base = (u64)(uintptr_t)_start;
    const u64 period = armGetSystemTickFreq() * 5;
    u64 next = armGetSystemTick() + period;
    for (;;) {
        svcSleepThread(5000000ull);  // hzs: 200 Hz
        const Handle main = HzsTargetThread();  // hzs: the game thread
        if (R_FAILED(svcSetThreadActivity(main, ThreadActivity_Paused))) continue;
        ThreadContext context;
        if (R_SUCCEEDED(svcGetThreadContext3(&context, main))) {
            ProfAdd(&g_profPc, context.pc.x - base);
            ProfAdd(&g_profLr, context.lr - base);
            // Inclusive: every return address on the frame-pointer chain, once per sample. The chain is only
            // followed while it stays on this stack (aligned, increasing, within 8 MiB above sp).
            u64 seen[24];
            int depth = 0;
            // Samples inside a condvar syscall: their nearest callers, in their own table, name who signals and waits.
            const u64 pcNow = context.pc.x;
            const int inCondvar = (pcNow >= (u64)(uintptr_t)svcSignalProcessWideKey &&
                                   pcNow < (u64)(uintptr_t)svcSignalProcessWideKey + 16) ||
                                  (pcNow >= (u64)(uintptr_t)svcWaitProcessWideKeyAtomic &&
                                   pcNow < (u64)(uintptr_t)svcWaitProcessWideKeyAtomic + 16);
            if (inCondvar) ProfAdd(&g_profCond, context.lr - base);  // the svc stubs keep no frame of their own
            const u64 sp = context.sp;
            u64 fp = context.fp;
            for (int frame = 0; frame < 24 && fp >= sp && fp < sp + (8u << 20) && (fp & 15) == 0; ++frame) {
                const u64 next = ((const u64*)(uintptr_t)fp)[0], ret = ((const u64*)(uintptr_t)fp)[1];
                if (ret > base) {
                    int dup = 0;
                    for (int i = 0; i < depth; ++i) dup |= seen[i] == ret;
                    if (!dup) {
                        seen[depth++] = ret;
                        ProfAdd(&g_profInc, ret - base);
                        if (inCondvar && frame < 5) ProfAdd(&g_profCond, ret - base);
                    }
                }
                if (next <= fp) break;
                fp = next;
            }
            ++g_profSamples;
        }
        svcSetThreadActivity(main, ThreadActivity_Runnable);
        if (armGetSystemTick() >= next) {
            ProfDump(&g_profPc, "pc", 40);
            ProfDump(&g_profLr, "lr", 40);
            ProfDump(&g_profInc, "inc", 160);
            ProfDump(&g_profCond, "cond", 40);
            g_profSamples = 0;
            next += period;
        }
    }
}
static Thread g_profiler;
__attribute__((constructor(105))) static void StartProfiler(void) {
    const char* enabled = getenv("HZ_PROF");
    if (enabled && enabled[0] == '0') return;  // hzs: on by default
    if (R_SUCCEEDED(threadCreate(&g_profiler, ProfileMainThread, NULL, NULL, 0x4000, 0x20, 1))) threadStart(&g_profiler);
}

// HZ_CALLBENCH=1: ns per loop iteration of straight-line work, a direct leaf call, a call that itself calls, an
// indirect call and a load/store pair, run once at startup. The emulator's per-call cost against its straight-line
// cost says whether translated code pays for its call count.
__attribute__((noinline)) static u64 BenchLeaf(u64 x) { __asm__ volatile("" : "+r"(x)); return x + 1; }
__attribute__((noinline)) static u64 BenchNonLeaf(u64 x) { return BenchLeaf(x) + 1; }
static u64 (*volatile g_benchIndirect)(u64) = BenchLeaf;
static volatile u64 g_benchMemory[64];
static void BenchReport(const char* what, u64 ticks, u64 iterations) {
    char line[96];
    char* out = line;
    Append(&out, "[horizon] callbench ");
    Append(&out, what);
    Append(&out, " ps/iter=");
    AppendDec(&out, ticks * 1000000000ull / armGetSystemTickFreq() * 1000ull / iterations);
    HorizonDebugWrite(line, (u64)(out - line));
}
// One timed loop of each kind; each is its own function, called many times, so the emulator's JIT promotes it to
// optimized code as it would a game function.
__attribute__((noinline)) static u64 BenchRound(int kind, u64 x, u64 n) {
    switch (kind) {
    case 0: for (u64 i = 0; i < n; ++i) { x += i; __asm__ volatile("" : "+r"(x)); } break;
    case 1: for (u64 i = 0; i < n; ++i) x = BenchLeaf(x); break;
    case 2: for (u64 i = 0; i < n; ++i) x = BenchNonLeaf(x); break;
    case 3: for (u64 i = 0; i < n; ++i) x = g_benchIndirect(x); break;
    default: for (u64 i = 0; i < n; ++i) g_benchMemory[i & 63] = g_benchMemory[(i + 7) & 63] + x; break;
    }
    return x;
}
__attribute__((constructor(106))) static void CallBench(void) {
    const char* enabled = getenv("HZ_CALLBENCH");
    if (!enabled || enabled[0] != '1') return;
    static const char* const kNames[] = {"inline-add", "direct-leaf-call", "nonleaf-call(2 calls)", "indirect-call",
                                         "load+store"};
    u64 x = 0;
    for (int warm = 0; warm < 3; ++warm) {  // warm-up rounds, a pause for the background JIT, then the timed one
        for (int kind = 0; kind < 5; ++kind)
            for (int call = 0; call < 2000; ++call) x = BenchRound(kind, x, 64);
        svcSleepThread(300000000ull);
    }
    for (int kind = 0; kind < 5; ++kind) {
        enum { kCalls = 200, kPerCall = 50000 };
        const u64 t0 = armGetSystemTick();
        for (int call = 0; call < kCalls; ++call) x = BenchRound(kind, x, kPerCall);
        BenchReport(kNames[kind], armGetSystemTick() - t0, (u64)kCalls * kPerCall);
    }
    g_benchMemory[0] = x;
}

static Thread g_sampler;
__attribute__((constructor(103))) static void StartSampler(void) {
    char line[96];
    char* out = line;
    // Core 1, above the main thread's priority: Horizon does not time-slice equal priorities on one
    // core, so on the main thread's core a spinning main thread would starve the sampler.
    Result result = threadCreate(&g_sampler, SampleMainThread, NULL, NULL, 0x4000, 0x20, 1);
    if (R_SUCCEEDED(result)) result = threadStart(&g_sampler);
    Append(&out, "[horizon] main-thread sampler: result ");
    AppendHex(&out, result);
    HorizonDebugWrite(line, (u64)(out - line));
}

// POSIX calls that newlib declares but libnx does not implement. Horizon has no signals, child
// processes or pipes for an application, so those report failure the way POSIX allows and every
// caller already handles (SDL's quit signals, sqlite's file ownership, imgui's open-in-shell,
// netplay's restart and matchmaker helpers, the log transcript pipe).

int sigaction(int sig, const struct sigaction* action, struct sigaction* old) {
    (void)sig; (void)action;
    if (old) *old = (struct sigaction){0};
    return 0;  // nothing will ever be delivered
}

int pthread_sigmask(int how, const sigset_t* set, sigset_t* old) {
    (void)how; (void)set;
    if (old) sigemptyset(old);
    return 0;
}

int pipe(int fds[2]) { (void)fds; errno = ENOSYS; return -1; }
FILE* popen(const char* command, const char* mode) { (void)command; (void)mode; errno = ENOSYS; return NULL; }
int pclose(FILE* stream) { (void)stream; errno = ECHILD; return -1; }
int execv(const char* path, char* const argv[]) { (void)path; (void)argv; errno = ENOSYS; return -1; }
int execvp(const char* file, char* const argv[]) { (void)file; (void)argv; errno = ENOSYS; return -1; }
pid_t waitpid(pid_t pid, int* status, int options) { (void)pid; (void)status; (void)options; errno = ECHILD; return -1; }

long sysconf(int name) {
    switch (name) {
    case _SC_PAGESIZE: return 4096;
    case _SC_NPROCESSORS_CONF:
    case _SC_NPROCESSORS_ONLN: return 3;  // the cores an application gets
    default: errno = EINVAL; return -1;
    }
}

// libnx implements every newlib thread hook except detach, so pthread_detach (and std::thread::detach)
// fails with ENOSYS. A detached thread still has to be joined to free its stack and handle, so one
// reaper thread joins detached threads as they finish, with libnx's own join hook.
#include <sys/iosupport.h>  // newlib's thread hooks: __syscall_thread_join / _detach
static Mutex g_detachLock;
static CondVar g_detachReady;
static struct __pthread_t* g_detached[64];
static int g_detachedCount;
static Thread g_reaper;
static bool g_reaperStarted;
static void ReapDetached(void* unused) {
    (void)unused;
    for (;;) {
        mutexLock(&g_detachLock);
        while (g_detachedCount == 0) condvarWait(&g_detachReady, &g_detachLock);
        struct __pthread_t* thread = g_detached[--g_detachedCount];
        mutexUnlock(&g_detachLock);
        (void)__syscall_thread_join(thread);
    }
}
int __syscall_thread_detach(struct __pthread_t* thread) {
    mutexLock(&g_detachLock);
    if (!g_reaperStarted) {
        g_reaperStarted = R_SUCCEEDED(threadCreate(&g_reaper, ReapDetached, NULL, NULL, 0x4000, 0x3B, -2)) &&
                          R_SUCCEEDED(threadStart(&g_reaper));
    }
    if (!g_reaperStarted || g_detachedCount == (int)(sizeof g_detached / sizeof g_detached[0])) {
        mutexUnlock(&g_detachLock);
        return EAGAIN;
    }
    g_detached[g_detachedCount++] = thread;
    condvarWakeOne(&g_detachReady);
    mutexUnlock(&g_detachLock);
    return 0;
}

// Horizon's file systems have no POSIX permission bits, so there is nothing to change. libstdc++'s
// std::filesystem::copy_file sets the copy's mode with fchmod and fails the whole copy if it does.
int fchmod(int fd, mode_t mode) { (void)fd; (void)mode; return 0; }

uid_t geteuid(void) { return 0; }
int fchown(int fd, uid_t owner, gid_t group) { (void)fd; (void)owner; (void)group; return 0; }
int fdatasync(int fd) { return fsync(fd); }

// hw-vanilla.sh: a vanilla-only NRO; P+'s codes in the shared Config.toml crash it (0x8057F190, gfApplication::init).
__attribute__((constructor(104))) static void HorizonForceVanilla(void) {
    setenv("HZS_VANILLA", "1", 1);
    fprintf(stderr, "[horizon] vanilla-only build: vanilla Brawl (no codes, no SD card)\n");
}

// ---- hzs telemetry (appended by hzs_hw_tel.sh) ----
// HzsThreadCpuNanos: CPU time the calling thread has run (svcGetInfo ThreadTickCount, 13.0.0+; 0 if unavailable).
// vi.cpp uses it for the game thread's per-frame CPU time, which excludes the time it spends waiting.
unsigned long long HzsThreadCpuNanos(void) {
    u64 ticks = 0;
    if (R_FAILED(svcGetInfo(&ticks, InfoType_ThreadTickCount, CUR_THREAD_HANDLE, (u64)-1))) return 0;
    return armTicksToNs(ticks);
}


// Every runtime thread (HorizonThreadEntry) and the main thread register here; the sampler logs each one's CPU time.
#define HZS_MAX_THREADS 48
static Handle s_hzsThreads[HZS_MAX_THREADS];
static unsigned long long s_hzsThreadEntry[HZS_MAX_THREADS];  // entry offset from _start (0 = main thread)
static u64 s_hzsThreadLast[HZS_MAX_THREADS];
static u32 s_hzsThreadCount;
void HzsRegisterThread(unsigned long long entryOffset) {
    const u32 i = __atomic_fetch_add(&s_hzsThreadCount, 1, __ATOMIC_RELAXED);
    if (i >= HZS_MAX_THREADS) return;
    s_hzsThreadEntry[i] = entryOffset;
    __atomic_store_n(&s_hzsThreads[i], threadGetCurHandle(), __ATOMIC_RELEASE);
}
// pthreads all enter through libnx's __thread_entry, so HorizonThreadEntry registers that for every one of them. The
// runtime's __wrap_pthread_create routes through HzsPthreadCreate: the new thread relabels its own slot with the start
// routine it was given (addr2line -f -C -e WiiCompiled-test <offset> names it), then runs it.
typedef struct {
    void* (*start)(void*);
    void* arg;
} HzsPthreadStart;
// The memory mapping holding address a, if it is readable: [*lo, *hi).
static bool HzsReadableRange(u64 a, u64* lo, u64* hi) {
    MemoryInfo info;
    u32 page;
    if (R_FAILED(svcQueryMemory(&info, &page, a)) || !(info.perm & Perm_R)) return false;
    *lo = info.addr;
    *hi = info.addr + info.size;
    return true;
}
// std::thread hands every thread the same start routine (libstdc++'s execute_native_thread_routine) and the real code
// behind its argument: a std::thread::_State whose vtable's third entry (after the two destructors) is _M_run, the
// thread's own function. When the argument looks like that (readable, first word points at readable memory whose
// entry [2] lies in this image's code), the label is that _M_run; otherwise the start routine.
static unsigned long long HzsThreadLabel(void* (*start)(void*), void* arg) {
    const u64 base = (u64)(uintptr_t)_start;
    u64 codeLo, codeHi, lo, hi;
    if (arg && HzsReadableRange(base, &codeLo, &codeHi) && HzsReadableRange((u64)(uintptr_t)arg, &lo, &hi) &&
        (u64)(uintptr_t)arg + 8 <= hi) {
        const u64 vtable = *(const u64*)arg;
        if (vtable && !(vtable & 7) && HzsReadableRange(vtable, &lo, &hi) && vtable + 24 <= hi) {
            const u64 run = ((const u64*)(uintptr_t)vtable)[2];
            if (run >= codeLo && run < codeHi) {
                // std::thread(function pointer, args...): _M_run is one template shared by every such thread and the
                // function pointer sits in the state after the vptr (libstdc++ tuples lay elements out last-first, so
                // not always the first word): the first word of the next 8 that points into code names the thread.
                HzsReadableRange((u64)(uintptr_t)arg, &lo, &hi);
                for (int w = 1; w <= 8 && (u64)(uintptr_t)arg + 8 * (w + 1) <= hi; ++w) {
                    const u64 fn = ((const u64*)arg)[w];
                    if (fn >= codeLo && fn < codeHi) return fn - base;
                }
                return run - base;
            }
        }
    }
    return (unsigned long long)((uintptr_t)start - (uintptr_t)_start);
}
static void* HzsPthreadTrampoline(void* value) {
    const HzsPthreadStart s = *(HzsPthreadStart*)value;
    free(value);
    const Handle self = threadGetCurHandle();
    const unsigned long long label = HzsThreadLabel(s.start, s.arg);
    const u32 count = s_hzsThreadCount < HZS_MAX_THREADS ? s_hzsThreadCount : HZS_MAX_THREADS;
    for (u32 i = 0; i < count; i++)
        if (__atomic_load_n(&s_hzsThreads[i], __ATOMIC_ACQUIRE) == self) s_hzsThreadEntry[i] = label;
    return s.start(s.arg);
}
int HzsPthreadCreate(pthread_t* thread, const pthread_attr_t* attr, void* (*start)(void*), void* arg) {
    HzsPthreadStart* s = (HzsPthreadStart*)malloc(sizeof *s);
    if (!s) return __real_pthread_create(thread, attr, start, arg);
    s->start = start;
    s->arg = arg;
    const int r = __real_pthread_create(thread, attr, HzsPthreadTrampoline, s);
    if (r) free(s);
    return r;
}
// One line: CPU ms per second of wall time each registered thread ran since the last line (under 0.1 left out).
// Divided by the real interval: the sampler is lowest priority and can be late, so it is not always 1 s.
static void HzsLogThreads(void) {
    static u64 lastTick;
    const u64 nowTick = armGetSystemTick();
    const double perSecond = lastTick ? 1e9 / (double)armTicksToNs(nowTick - lastTick) : 0.0;
    lastTick = nowTick;
    char line[1024];
    int n = snprintf(line, sizeof line, "[hzs-thr] cpu ms/s:");
    const u32 count = s_hzsThreadCount < HZS_MAX_THREADS ? s_hzsThreadCount : HZS_MAX_THREADS;
    for (u32 i = 0; i < count && n < (int)sizeof line - 40; i++) {
        const Handle h = __atomic_load_n(&s_hzsThreads[i], __ATOMIC_ACQUIRE);
        u64 ticks = 0;
        if (!h || R_FAILED(svcGetInfo(&ticks, InfoType_ThreadTickCount, h, (u64)-1))) continue;
        const double ms = s_hzsThreadLast[i] ? armTicksToNs(ticks - s_hzsThreadLast[i]) / 1e6 * perSecond : 0.0;
        s_hzsThreadLast[i] = ticks;
        if (ms < 0.1) continue;
        if (s_hzsThreadEntry[i]) n += snprintf(line + n, sizeof line - n, " +0x%llx %.1f", s_hzsThreadEntry[i], ms);
        else n += snprintf(line + n, sizeof line - n, " main %.1f", ms);
    }
    snprintf(line + n, sizeof line - n, "\n");
    fputs(line, stderr);
}

// Once a second: the clocks the CPU, GPU and memory actually run at, both temperature sensors, battery and charger,
// and this process's memory. A field whose service refuses this process prints n/a. HZ_TELEMETRY=0 turns it off.
static void HzsHwSampler(void* arg) {
    (void)arg;
    ClkrstSession cpu, gpu, emc;
    const bool clk = R_SUCCEEDED(clkrstInitialize()) && R_SUCCEEDED(clkrstOpenSession(&cpu, PcvModuleId_CpuBus, 3)) &&
                     R_SUCCEEDED(clkrstOpenSession(&gpu, PcvModuleId_GPU, 3)) &&
                     R_SUCCEEDED(clkrstOpenSession(&emc, PcvModuleId_EMC, 3));
    TsSession tsIn, tsOut;
    const bool ts = R_SUCCEEDED(tsInitialize()) && R_SUCCEEDED(tsOpenSession(&tsIn, TsDeviceCode_LocationInternal)) &&
                    R_SUCCEEDED(tsOpenSession(&tsOut, TsDeviceCode_LocationExternal));
    const bool psm = R_SUCCEEDED(psmInitialize());
    // GPU load: the PMU's load in per-mille, the ioctl Status Monitor Overlay uses (NVGPU_GPU_IOCTL_PMU_GET_GPU_LOAD on
    // /dev/nvhost-ctrl-gpu). Sampled 10 times a second; each line has that second's average and peak.
    u32 gpuFd = 0;
    Result gpuResult = nvInitialize();
    if (R_SUCCEEDED(gpuResult)) gpuResult = nvOpen(&gpuFd, "/dev/nvhost-ctrl-gpu");
    if (R_SUCCEEDED(gpuResult)) {
        u32 probe = 0;
        gpuResult = nvIoctl(gpuFd, 0x80044715, &probe);
    }
    const bool gpuLoad = R_SUCCEEDED(gpuResult);
    fprintf(stderr, "[hzs-hw] sampler: clocks %s, temperatures %s, battery %s, gpu load %s (0x%x)\n", clk ? "ok" : "n/a",
            ts ? "ok" : "n/a", psm ? "ok" : "n/a", gpuLoad ? "ok" : "n/a", (unsigned)gpuResult);
    for (;;) {
        u32 loadSum = 0, loadMax = 0, loadCount = 0;
        for (int tick = 0; tick < 10; ++tick) {
            u32 load = 0;
            if (gpuLoad && R_SUCCEEDED(nvIoctl(gpuFd, 0x80044715, &load))) {
                loadSum += load;
                loadMax = load > loadMax ? load : loadMax;
                ++loadCount;
            }
            svcSleepThread(100000000ull);
        }
        char line[256];
        int n = snprintf(line, sizeof line, "[hzs-hw]");
        if (loadCount)
            n += snprintf(line + n, sizeof line - n, " gpu load %.1f%% max %.1f%% |", loadSum / (10.0 * loadCount),
                          loadMax / 10.0);
        if (clk) {
            u32 c = 0, g = 0, e = 0;
            clkrstGetClockRate(&cpu, &c);
            clkrstGetClockRate(&gpu, &g);
            clkrstGetClockRate(&emc, &e);
            n += snprintf(line + n, sizeof line - n, " cpu %u gpu %u emc %u MHz", c / 1000000, g / 1000000,
                          e / 1000000);
        }
        if (ts) {
            float tin = 0, tout = 0;
            tsSessionGetTemperature(&tsIn, &tin);
            tsSessionGetTemperature(&tsOut, &tout);
            n += snprintf(line + n, sizeof line - n, " | internal %.1fC external %.1fC", tin, tout);
        }
        if (psm) {
            u32 pct = 0;
            PsmChargerType charger = PsmChargerType_Unconnected;
            psmGetBatteryChargePercentage(&pct);
            psmGetChargerType(&charger);
            n += snprintf(line + n, sizeof line - n, " | battery %u%% charger %d", pct, (int)charger);
            PsmBatteryChargeInfoFields info;
            if (R_SUCCEEDED(psmGetBatteryChargeInfoFields(&info)))
                n += snprintf(line + n, sizeof line - n, " %u mV %.1fC", info.battery_charge_milli_voltage,
                              info.temperature_celcius / 1000.0);
        }
        u64 used = 0, total = 0;
        svcGetInfo(&used, InfoType_UsedMemorySize, CUR_PROCESS_HANDLE, 0);
        svcGetInfo(&total, InfoType_TotalMemorySize, CUR_PROCESS_HANDLE, 0);
        snprintf(line + n, sizeof line - n, " | memory %llu of %llu MiB\n", (unsigned long long)(used >> 20),
                 (unsigned long long)(total >> 20));
        fputs(line, stderr);
        HzsLogThreads();
        HzsDebugLogFlush();  // the once-a-second debug.log flush, here instead of on whichever thread logs
    }
}

__attribute__((constructor(105))) static void HzsHwSamplerStart(void) {
    const char* off = getenv("HZ_TELEMETRY");
    if (off && off[0] == '0') return;
    HzsRegisterThread(0);  // the main thread
    static Thread thread;
    // 0x3B: the lowest priority an application's NPDM normally allows (0x3F was refused, silently: no [hzs-hw] at all).
    // Falls back to the main thread's 0x2C; any core: one wake a second.
    Result r = threadCreate(&thread, HzsHwSampler, NULL, NULL, 0x8000, 0x3B, -2);
    if (R_FAILED(r)) r = threadCreate(&thread, HzsHwSampler, NULL, NULL, 0x8000, 0x2C, -2);
    if (R_SUCCEEDED(r)) r = threadStart(&thread);
    fprintf(stderr, "[hzs-hw] sampler thread: 0x%x\n", (unsigned)r);
}
// The game thread: the one that presents frames (vi.cpp calls HzsSetGameThread there). The profiler and the 5 s
// watchdog pause it to sample where it is; until the first frame they fall back to the main thread.
static Handle s_hzsGameThread;
void HzsSetGameThread(void) {
    if (__atomic_load_n(&s_hzsGameThread, __ATOMIC_RELAXED)) return;
    __atomic_store_n(&s_hzsGameThread, threadGetCurHandle(), __ATOMIC_RELEASE);
    fprintf(stderr, "[hzs] game thread registered for the profiler and watchdog\n");
}
Handle HzsTargetThread(void) {
    const Handle game = __atomic_load_n(&s_hzsGameThread, __ATOMIC_ACQUIRE);
    return game ? game : envGetMainThreadHandle();
}
// ---- end hzs telemetry ----
