#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

// MKW_LEAN_DIRTY_MARKS=0 (lean memory only): no guest store writes the dirty map (the per-store 16-bit mark, range
// marks, the helpers below). Its one reader that matters is rollback's WriteWatch, and brawl_rollback.cpp refuses to
// track writes in such a build. For the vanilla EABI translation, which runs neither rollback (its hooks need the
// interpreted, mod-patched main loop) nor Gecko codes, the marks are ~2-3 host instructions per guest store that
// nothing reads. Natives that write or compare marks themselves (g3d_calc, mtx BRAWL_HLE_VERIFY) still do, so their
// verify modes report mark differences in such a build: verify with marks on. Default on.
#ifndef MKW_LEAN_DIRTY_MARKS
#define MKW_LEAN_DIRTY_MARKS 1
#endif

// Portable guest write tracking. Aliases share a Page allocated by the match
// tracker. Publication/removal happens at the simulation's snapshot safe point.
namespace BrawlDirtyMemory {
struct Page {
    std::atomic<uint8_t> touched{0};
    std::atomic<uint8_t> blocks[1024]{};
};
inline std::atomic<bool> active{false};
inline Page* pages[65536]{};
// Bumped before anything clears marks or rewrites guest memory without marking it (a rollback or save-state load,
// a verifier putting memory back). Between bumps a mark only goes from clear to set, so code that matched guest
// memory in the current epoch and whose marks are still clear has not been written since: the interpreter's block
// cache checks that instead of comparing the code on every entry.
inline std::atomic<uint32_t> markEpoch{1};
inline void MarksCleared() { markEpoch.fetch_add(1, std::memory_order_acq_rel); }
// Rollback's WriteWatch consumes the marks of the pages it tracks (scans then clears them, writers quiescent) and
// restores those pages from snapshots; consumeSeq counts the consumes, and a snapshot keeps the count it was taken at.
// The interpreter's code caches set these hooks so that neither needs a new epoch: before a consume they retire the
// code under set marks (noting the count), and after a restore to a snapshot they retire the code written since it
// (and code built since). Without hooks, both start a new epoch.
inline std::atomic<uint32_t> consumeSeq{0};
inline void (*consumingMarks)(uint32_t seq) = nullptr;
inline void (*restoredMarks)(uint32_t seq) = nullptr;
inline void MarksConsuming() {
    const uint32_t seq = consumeSeq.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (consumingMarks) consumingMarks(seq); else MarksCleared();
}
inline void MarksRestored(uint32_t snapshotSeq) {
    if (restoredMarks) restoredMarks(snapshotSeq); else MarksCleared();
}
// Lean memory: the 64 KiB pages (of the 1 GiB canonical window) whose marks rollback's WriteWatch consumes. Nothing
// reads the marks of any other page, so the interpreter clears those of the code it checks there itself.
inline std::atomic<uint8_t> trackedPages[1u << 14]{};
// Mark/MarkHost run at every guest store in translated code. Their bodies stay out of line (and cold) so a store
// pays only the inline test of `active` while no match is tracked; an inlined call to the whole body made the
// compiler spill caller-saved registers around every store even with tracking off.
#if defined(_MSC_VER)
#define BRAWL_DIRTY_MEMORY_SLOW __declspec(noinline)
#else
#define BRAWL_DIRTY_MEMORY_SLOW __attribute__((noinline, cold))
#endif
BRAWL_DIRTY_MEMORY_SLOW inline void MarkSlow(uint32_t address,size_t size) {
    if(!active.load(std::memory_order_relaxed) || !size)return;
    while(size) {
        const size_t offset=address&65535u;
        const size_t count=size<65536-offset?size:65536-offset;
        if(auto* p=pages[address>>16]) {
            p->touched.store(1,std::memory_order_relaxed);
            for(size_t i=offset/64;i<=(offset+count-1)/64;++i)
                p->blocks[i].store(1,std::memory_order_relaxed);
        }
        address+=uint32_t(count);size-=count;
    }
}
#if MKW_LEAN_MEMORY
// The lean memory model (memory_access.h) gives every guest byte one canonical offset, `address & 0x3FFFFFFF`, in a
// 1 GiB window, so its marks are one flat byte per 64 guest bytes over that window (leanBlocks, allocated by
// Memory::Init), with no page table or alias pointers. Every store marks unconditionally, tracked or not: a shift
// and one 16-bit store of 0x0101, no branch. That store also marks the following block, which covers a write of up
// to 64 bytes that crosses into it; the scanner copies the extra block. WriteWatch scans and clears its pages' slices.
inline constexpr uint32_t kLeanDirtyMask=0x3FFFFFFFu;
inline constexpr size_t kLeanDirtyBlocks=(size_t{1}<<30)/64;
inline uint8_t* leanBlocks=nullptr;  // kLeanDirtyBlocks + 1 bytes (the last block's mark spills one byte)
inline void LeanMarkBlockPair(uint32_t canonical) {
    const uint16_t both=0x0101;
    std::memcpy(leanBlocks+(canonical>>6),&both,sizeof(both));
}
// Guarded code (the JIT's, src/jit/jit_x64.cpp): compiled blocks that lie wholly in 4 KiB pages registered as executable
// (where every guest store takes the checked path, which marks through Mark below) and that skip the dirty-mark test on
// their linked entries. Instead, a mark of one of their blocks drops them at once, before compiled code can follow a link
// slot to them: guardChunks has a byte per 64 KiB chunk and guardLines a bit per mark (one per 64 bytes) under guarded
// code, set by the interpreter's block cache, which installs guardedWritten (it drops the guarded blocks under the
// canonical range written). Every other writer that marks a byte of such a page marks through Mark or LeanMarkRange too
// (natives that write marks themselves test the store policy first, and the policy flags these pages' chunks).
inline uint8_t* guardChunks=nullptr;
inline uint64_t* guardLines=nullptr;
inline void (*guardedWritten)(uint32_t canonical,size_t size)=nullptr;
BRAWL_DIRTY_MEMORY_SLOW inline void GuardedLinesTest(uint32_t canonical,size_t size) {
    const size_t first=canonical>>6;
    size_t last=(size_t(canonical)+size-1)>>6;
    if(last>=kLeanDirtyBlocks) last=kLeanDirtyBlocks-1;
    for(size_t w=first>>6;w<=(last>>6);++w) {  // 64 marks per word
        uint64_t bits=guardLines[w];
        if(w==(first>>6)) bits&=~uint64_t{0}<<(first&63);
        if(w==(last>>6) && (last&63)!=63) bits&=(uint64_t{2}<<(last&63))-1u;
        if(bits!=0) { guardedWritten(canonical,size); return; }
    }
}
inline void NoteGuarded(uint32_t canonical,size_t size) {
    if(guardChunks!=nullptr && size!=0 && guardChunks[canonical>>16]!=0) [[unlikely]] GuardedLinesTest(canonical,size);
}
BRAWL_DIRTY_MEMORY_SLOW inline void LeanMarkRange(uint32_t canonical,size_t size) {
    const size_t first=canonical>>6;
    const size_t last=(size_t(canonical)+size-1)>>6;
    std::memset(leanBlocks+first,1,(last<kLeanDirtyBlocks?last:kLeanDirtyBlocks-1)-first+1);
    if(guardChunks!=nullptr)
        for(size_t c=first>>10;c<=(last>>10) && c<(size_t{1}<<14);++c)
            if(guardChunks[c]!=0) { GuardedLinesTest(canonical,size); break; }
}
#endif
inline void Mark(uint32_t address,size_t size) {
#if MKW_LEAN_MEMORY && !MKW_LEAN_DIRTY_MARKS
    (void)address;(void)size;
#elif MKW_LEAN_MEMORY
    if(size<=64) { LeanMarkBlockPair(address&kLeanDirtyMask); NoteGuarded(address&kLeanDirtyMask,size); }
    else if(size) LeanMarkRange(address&kLeanDirtyMask,size);
#else
    if(active.load(std::memory_order_relaxed))MarkSlow(address,size);
#endif
}
// Bulk helpers holding a host backing pointer use this at the actual write,
// never at pointer acquisition (which would also mark reads and miss caching).
struct HostSpan {uintptr_t start;size_t size;uint32_t guest;};
inline uintptr_t flatBase=0;
inline std::vector<HostSpan> hosts;
BRAWL_DIRTY_MEMORY_SLOW inline void MarkHostSlow(const void* pointer,size_t size) {
    if(!active.load(std::memory_order_relaxed) || !size)return;
    const uintptr_t start=reinterpret_cast<uintptr_t>(pointer);
    if(start>=flatBase && start-flatBase<=UINT32_MAX && flatBase) {
        Mark(uint32_t(start-flatBase),size);return;
    }
    for(const auto& h:hosts) {
        const uintptr_t lo=start>h.start?start:h.start;
        const uintptr_t hi=start+size<h.start+h.size?start+size:h.start+h.size;
        if(lo<hi)Mark(h.guest+uint32_t(lo-h.start),hi-lo);
    }
}
inline void MarkHost(const void* pointer,size_t size) {
#if MKW_LEAN_MEMORY  // flatBase is the lean window (Memory::Init): every guest host pointer lies inside it
    const uintptr_t offset=reinterpret_cast<uintptr_t>(pointer)-flatBase;
    if(offset<(uintptr_t{1}<<30)) Mark(uint32_t(offset),size);
#else
    if(active.load(std::memory_order_relaxed))MarkHostSlow(pointer,size);
#endif
}
}
