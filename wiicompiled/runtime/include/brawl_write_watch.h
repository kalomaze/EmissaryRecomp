#pragma once
#include "brawl_dirty_memory.h"
#include "brawl_snapshot_jobs.h"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

namespace BrawlRollback {
// Explicit portable instrumentation, not OS page protection. The simulation
// boundary must quiesce writers before Capture/Reset or destroying the tracker.
class WriteWatch {
    struct Span {uint32_t guest;size_t size,packed;};
    std::vector<Span> spans;
    std::map<uint32_t,std::unique_ptr<BrawlDirtyMemory::Page>> owned;
    // A tracked 64 KiB page's marks: its Page, or under MKW_LEAN_MEMORY its 1024-byte slice of the flat lean map,
    // which has no touched byte (touched == nullptr: the page is always scanned).
    struct ScanPage {uint32_t index;std::atomic<uint8_t>* touched;std::atomic<uint8_t>* blocks;};
    std::vector<ScanPage> scanPages;
    static constexpr size_t PagesPerBatch=4;
    std::vector<std::vector<uint32_t>> scanResults;
    std::vector<uint32_t> capturedKeys;
public:
    size_t metadataBytes=0;
    uint64_t resetNs=0,captureNs=0,captureJobsNs=0,captureWorkerNs=0,captureMergeNs=0;
    explicit WriteWatch(uintptr_t flatBase=0) {
        if(BrawlDirtyMemory::active.load())throw std::runtime_error("duplicate guest dirty tracker");
        BrawlDirtyMemory::flatBase=flatBase;
    }
    ~WriteWatch(){
        BrawlDirtyMemory::MarksCleared();
        for (const auto& page : scanPages)
            BrawlDirtyMemory::trackedPages[page.index & 0x3FFFu].store(0, std::memory_order_release);
        BrawlDirtyMemory::active.store(false,std::memory_order_relaxed);
        for(auto& p:BrawlDirtyMemory::pages)p=nullptr;
        BrawlDirtyMemory::hosts.clear();
    }
    WriteWatch(const WriteWatch&)=delete;
    WriteWatch& operator=(const WriteWatch&)=delete;
    void Add(const void* host,size_t size,size_t packed,uint32_t guest=0) {
        if(!size)return;
        const uint32_t physical=GuestMemoryAlias::Offset(guest);
        for(const auto& s:spans)if(uint64_t(physical)<uint64_t(s.guest)+s.size && uint64_t(s.guest)<uint64_t(physical)+size)
            throw std::runtime_error("overlapping selected snapshot regions");
        spans.push_back({physical,size,packed});
        std::sort(spans.begin(),spans.end(),[](const Span& a,const Span& b){return a.guest<b.guest;});
        for(size_t i=1;i<spans.size();++i)
            if(spans[i-1].packed+spans[i-1].size>spans[i].packed)
                throw std::runtime_error("snapshot packing must preserve physical address order");
        const auto hostStart=reinterpret_cast<uintptr_t>(host);
        bool registered=false;
        for(const auto& span:BrawlDirtyMemory::hosts) {
            if(hostStart<span.start || hostStart-span.start>span.size || size>span.size-(hostStart-span.start))continue;
            if(GuestMemoryAlias::Offset(span.guest+uint32_t(hostStart-span.start))!=physical)
                throw std::runtime_error("inconsistent guest mapping for host memory");
            registered=true;break;
        }
        if(!registered)BrawlDirtyMemory::hosts.push_back({hostStart,size,guest});
        for(uint64_t addr=uint64_t(guest)&~uint64_t(65535);addr<uint64_t(guest)+size;addr+=65536) {
            // MEM1/MEM2 physical/cached/uncached aliases share the same mark.
            const uint32_t canonical=GuestMemoryAlias::Offset(uint32_t(addr));
            auto& page=owned[canonical>>16];
#if MKW_LEAN_MEMORY  // the marks live in BrawlDirtyMemory::leanBlocks at the canonical offset: no Page, no aliases
            (void)page;
#else
            if(!page)page=std::make_unique<BrawlDirtyMemory::Page>();
            BrawlDirtyMemory::pages[uint32_t(addr)>>16]=page.get();
            if(canonical<0x20000000u) {
                BrawlDirtyMemory::pages[canonical>>16]=page.get();
                BrawlDirtyMemory::pages[(canonical|0x80000000u)>>16]=page.get();
                BrawlDirtyMemory::pages[(canonical|0xc0000000u)>>16]=page.get();
            }
#endif
        }
        scanPages.clear();
        for(const auto& [index,page]:owned) {
#if MKW_LEAN_MEMORY
            static_assert(sizeof(std::atomic<uint8_t>)==1,"dirty marks must be byte sized");
            scanPages.push_back({index,nullptr,
                reinterpret_cast<std::atomic<uint8_t>*>(BrawlDirtyMemory::leanBlocks+(size_t(index)<<10))});
            // before Enable's clear and base image: marks the interpreter cleared here until now are not needed
            BrawlDirtyMemory::trackedPages[index&0x3FFFu].store(1,std::memory_order_release);
#else
            scanPages.push_back({index,&page->touched,page->blocks});
#endif
        }
        scanResults.resize((scanPages.size()+PagesPerBatch-1)/PagesPerBatch);
    }
    // Capture consumes all marks. Enable after the initial base image; later
    // save/restore boundaries need no second walk over clean tracking pages.
    void Enable(){
        BrawlDirtyMemory::MarksConsuming();  // its clear drops whatever marks are set
#if MKW_LEAN_MEMORY  // lean stores mark at all times: drop what the base image already holds
        for(const auto& page:scanPages)std::memset(static_cast<void*>(page.blocks),0,1024);
#endif
        BrawlDirtyMemory::active.store(true,std::memory_order_relaxed);
        resetNs=0;
    }
    void Reset(){
        BrawlDirtyMemory::MarksConsuming();
        const auto start=std::chrono::steady_clock::now();
        // Writers are quiescent here. Capture already cleared consumed pages;
        // an exchange would still issue a locked write for every clean page.
        for(const auto& page:scanPages) {
            if(!page.touched) {std::memset(static_cast<void*>(page.blocks),0,1024);continue;}
            if(!page.touched->load(std::memory_order_relaxed))continue;
            for(size_t i=0;i<1024;++i)page.blocks[i].store(0,std::memory_order_relaxed);
            page.touched->store(0,std::memory_order_relaxed);
        }
        BrawlDirtyMemory::active.store(true,std::memory_order_relaxed);
        resetNs=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
    }
    // Valid until the next Capture; snapshot workers finish before that boundary.
    const std::vector<uint32_t>& Capture(size_t bytes,SnapshotJobs& jobs){
        BrawlDirtyMemory::MarksConsuming();  // it clears the marks it consumes
        const auto start=std::chrono::steady_clock::now();
        auto& dirty=capturedKeys;dirty.clear();dirty.reserve(8192);metadataBytes=0;
        // Same cache-line scan/skip and clear-as-consumed ordering as Dolphin's
        // JITDirtyBitmap::ScanAndClear. Writers are quiescent at this boundary,
        // so inspecting atomic byte representations cannot race a live store.
        static_assert(sizeof(std::atomic<uint8_t>)==1,"dirty marks must be byte sized");
        std::array<size_t,SnapshotJobs::Chunks> metadata{};
        std::array<uint64_t,SnapshotJobs::Chunks> workerNs{};
        const auto jobsStart=std::chrono::steady_clock::now();
        jobs.Run([&](size_t chunk) {
          const auto workerStart=std::chrono::steady_clock::now();
          size_t examined=0;
          // Spread clustered match heaps over all five workers. Each small
          // batch retains its own sorted output, merged in address order below.
          for(size_t batch=chunk;batch<scanResults.size();batch+=SnapshotJobs::Chunks) {
          auto dirty=std::move(scanResults[batch]);dirty.clear();
          const size_t begin=batch*PagesPerBatch;
          const size_t end=std::min(begin+PagesPerBatch,scanPages.size());
          auto spanCursor=spans.begin();
          if(begin<end)spanCursor=std::lower_bound(spans.begin(),spans.end(),uint64_t(scanPages[begin].index)<<16,
              [](const Span& span,uint64_t address){return uint64_t(span.guest)+span.size<=address;});
          for(size_t pageNumber=begin;pageNumber<end;++pageNumber) {
            const auto [pageIndex,touched,blocks]=scanPages[pageNumber];
            ++examined;
            if(touched && !touched->load(std::memory_order_relaxed))continue;
            const uint64_t pageStart=uint64_t(pageIndex)<<16;
            while(spanCursor!=spans.end() && uint64_t(spanCursor->guest)+spanCursor->size<=pageStart)++spanCursor;
            // Interior pages have one constant physical-to-packed offset.
            // Resolve it once per page, not once for every marked granule.
            const bool interior=spanCursor!=spans.end() && spanCursor->guest<=pageStart &&
                uint64_t(spanCursor->guest)+spanCursor->size>=pageStart+65536;
            const size_t packedStart=interior?spanCursor->packed+pageStart-spanCursor->guest:0;
            if(interior && (packedStart+65536+63)/64>(bytes+63)/64)
                throw std::runtime_error("dirty span outside packed memory");
            for(size_t first=0;first<1024;first+=64) {
                uint8_t marks[64];uint64_t words[8];
                std::memcpy(marks,static_cast<const void*>(blocks+first),sizeof(marks));examined+=sizeof(marks);
                std::memcpy(words,marks,sizeof(words));
                if(!(words[0]|words[1]|words[2]|words[3]|words[4]|words[5]|words[6]|words[7]))continue;
                for(size_t wordIndex=0;wordIndex<8;++wordIndex) {
                  auto word=words[wordIndex];
                  while(word) {
                    const size_t byte=std::countr_zero(word)/8;
                    word&=~(uint64_t{0xff}<<(byte*8));
                    const size_t i=wordIndex*8+byte;
                    blocks[first+i].store(0,std::memory_order_relaxed);
                    if(interior) {
                        const uint32_t key=uint32_t(packedStart/64+first+i);
                        if(dirty.empty() || dirty.back()!=key)dirty.push_back(key);
                        if(packedStart%64)dirty.push_back(key+1);
                        continue;
                    }
                    const uint64_t block=pageStart+(first+i)*64;
                    while(spanCursor!=spans.end() && uint64_t(spanCursor->guest)+spanCursor->size<=block)++spanCursor;
                    auto at=spanCursor;
                    for(;at!=spans.end() && at->guest<block+64;++at) {
                        const auto& span=*at;
                        const uint64_t loGuest=std::max(block,uint64_t(span.guest));
                        const uint64_t hiGuest=std::min(block+64,uint64_t(span.guest)+span.size);
                        const size_t lo=(span.packed+loGuest-span.guest)/64;
                        const size_t hi=(span.packed+hiGuest-span.guest+63)/64;
                        if(hi>(bytes+63)/64)throw std::runtime_error("dirty span outside packed memory");
                        for(size_t key=lo;key<hi;++key)
                            if(dirty.empty() || dirty.back()!=key)dirty.push_back(uint32_t(key));
                    }
                  }
                }
            }
            if(touched)touched->store(0,std::memory_order_relaxed);
        }
          scanResults[batch]=std::move(dirty);
          }
          metadata[chunk]=examined;
          workerNs[chunk]=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-workerStart).count();
        });
        captureJobsNs=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-jobsStart).count();
        captureWorkerNs=*std::max_element(workerNs.begin(),workerNs.end());
        const auto mergeStart=std::chrono::steady_clock::now();
        for(size_t chunk=0;chunk<scanResults.size();++chunk) {
            const auto& part=scanResults[chunk];
            auto begin=part.begin();
            if(begin!=part.end() && !dirty.empty() && dirty.back()==*begin)++begin;
            dirty.insert(dirty.end(),begin,part.end());
        }
        for(size_t examined:metadata)metadataBytes+=examined;
        captureMergeNs=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-mergeStart).count();
        captureNs=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
        return dirty;
    }
};
}
