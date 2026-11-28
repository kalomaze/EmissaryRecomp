#pragma once
#include "brawl_snapshot_jobs.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace BrawlRollback {
// Port of Brawl Online's RollbackManager / DeltaSaveSlot RAM snapshot path:
// raw 64-byte granules, retained slot buffers, asynchronous
// eviction into a base image, and backward source lookup on restore. Guest RAM
// is supplied by the recomp's selected-region reader/writer instead of Dolphin.
// Each saved granule also links to the previous saved copy of its block, so the
// restore finds a block's source from its first discarded copy instead of
// scanning every older slot.
class SnapshotStore {
public:
    static constexpr size_t BlockSize=64, Capacity=16;
private:
    struct Delta {
        std::unique_ptr<uint32_t[]> indices;
        std::unique_ptr<uint32_t[]> previous;  // each block's copy before this one (a Location), or kNone
        std::unique_ptr<uint8_t[]> bytes;
        size_t count=0,capacity=0;
        int frame=0;
        void Reserve(size_t blocks) {
            if(capacity>=blocks)return;
            // Like Dolphin's slot buffers, grow only when needed and overwrite
            // the new storage directly; old deltas in this slot are discarded.
            auto newIndices=std::make_unique_for_overwrite<uint32_t[]>(blocks);
            auto newPrevious=std::make_unique_for_overwrite<uint32_t[]>(blocks);
            auto newBytes=std::make_unique_for_overwrite<uint8_t[]>(blocks*BlockSize);
            indices=std::move(newIndices);previous=std::move(newPrevious);bytes=std::move(newBytes);capacity=blocks;
        }
        size_t ResidentBytes() const {return capacity*(2*sizeof(uint32_t)+BlockSize);}
    };
    // A saved copy of a block: its slot (top 4 bits) and its position in the slot. A location can outlive its copy
    // (the slot evicted and reused, or discarded by a restore), so Core::Valid checks one before it is followed.
    static constexpr uint32_t kNone=~uint32_t{0},kIndexBits=28;
    static_assert(Capacity<=(size_t{1}<<(32-kIndexBits)),"slot number must fit a location");
    static constexpr uint32_t Location(size_t slot,size_t index){return uint32_t(slot<<kIndexBits|index);}
    static constexpr size_t SlotOf(uint32_t at){return at>>kIndexBits;}
    static constexpr size_t IndexOf(uint32_t at){return at&((uint32_t{1}<<kIndexBits)-1);}
    struct Core {
        SnapshotJobs jobs;
        std::array<Delta,Capacity> slots;
        Delta evicted;
        job::Job* eviction=nullptr;
        std::vector<uint8_t> base,scratch,needsSource;
        std::vector<uint32_t> dirtyKeys,flagged;
        std::vector<uint32_t> flaggedSources;
        std::vector<uint32_t> latest;  // per block: its newest saved copy (a Location), or kNone (base only)
        size_t next=0,count=0;
        ~Core(){jobs.Wait(eviction);}
        size_t Oldest() const {return (next+Capacity-count)%Capacity;}
        // A slot's age in the retained history: 0 for the oldest, count-1 for the newest, Capacity if not retained.
        size_t Age(size_t slot) const {const size_t age=(slot+Capacity-Oldest())%Capacity;return age<count?age:Capacity;}
        // `at` still holds a copy of `key` saved before a slot of age `before`: retained, older, and that key.
        bool Valid(uint32_t at,uint32_t key,size_t before) const {
            if(at==kNone)return false;
            const size_t slot=SlotOf(at),index=IndexOf(at);
            return Age(slot)<before && index<slots[slot].count && slots[slot].indices[index]==key;
        }
        size_t Find(int frame) const {
            for(size_t n=0;n<count;++n){size_t i=(next+Capacity-1-n)%Capacity;if(slots[i].frame==frame)return i;}
            throw std::runtime_error("rollback: snapshot outside retained history");
        }
        static void Apply(std::vector<uint8_t>& image,const Delta& delta) {
            for(size_t i=0;i<delta.count;++i) {
                const size_t offset=size_t(delta.indices[i])*BlockSize;
                std::memcpy(image.data()+offset,delta.bytes.get()+i*BlockSize,std::min(BlockSize,image.size()-offset));
            }
        }
        size_t ResidentBytes() const {
            size_t bytes=jobs.ResidentBytes()+base.capacity()+scratch.capacity()+needsSource.capacity();
            bytes+=(dirtyKeys.capacity()+flagged.capacity()+flaggedSources.capacity()+latest.capacity())*sizeof(uint32_t);
            bytes+=evicted.ResidentBytes();
            for(const auto& slot:slots)bytes+=slot.ResidentBytes();
            return bytes;
        }
    };
    // Stable storage: an in-flight eviction must survive Session's move/reset.
    std::unique_ptr<Core> core;
    bool prepared=false;int preparedFrame=0,preparedAnchor=0;
    uint64_t preparedIndexNs=0,preparedEvictWaitNs=0;
    static void Flag(Core& c,uint32_t key,uint32_t source) {
        if(c.needsSource[key])return;
        c.needsSource[key]=1;c.flagged.push_back(key);c.flaggedSources.push_back(source);
    }
    void DropPrepared() {
        if(!prepared)return;
        prepared=false;
        for(uint32_t key:core->flagged)core->needsSource[key]=0;
        core->flagged.clear();core->flaggedSources.clear();
    }
public:
    size_t lastPayload=0,peakResident=0,lastReadBytes=0;
    size_t lastRestoreReadBytes=0,lastRestoreWriteBytes=0;
    using Clock=std::chrono::steady_clock;
    static uint64_t Elapsed(Clock::time_point start){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();}
    uint64_t lastIndexNs=0,lastJobsNs=0,lastWorkerNs=0,lastFinishNs=0,lastEvictWaitNs=0;
    SnapshotJobs& Jobs() {
        if(!core)throw std::runtime_error("rollback: worker pool requested before initial save");
        return core->jobs;
    }
    bool Empty() const {return !core || !core->count;}
    size_t Count() const {return core?core->count:0;}
    bool Contains(int frame) const {
        if(!core)return false;
        for(size_t n=0;n<core->count;++n)if(core->slots[(core->Oldest()+n)%Capacity].frame==frame)return true;
        return false;
    }
    std::vector<uint8_t> Materialize(int frame) const {
        if(!core)throw std::runtime_error("rollback: no snapshot");
        auto& c=*core;const size_t target=c.Find(frame);
        c.jobs.Wait(c.eviction);
        auto image=c.base;
        for(size_t n=0;n<c.count;++n) {
            const size_t slot=(c.Oldest()+n)%Capacity;
            Core::Apply(image,c.slots[slot]);
            if(slot==target)return image;
        }
        throw std::runtime_error("rollback: unreachable snapshot");
    }
    const std::vector<uint8_t>& View(int frame) {
        if(!core)throw std::runtime_error("rollback: no snapshot");
        core->scratch=Materialize(frame);return core->scratch;
    }
    template<class Reader> void Save(int frame,size_t size,Reader read,
                                    const std::vector<uint32_t>* dirty=nullptr,int anchor=0) {
        lastReadBytes=0;lastIndexNs=lastJobsNs=lastWorkerNs=lastFinishNs=lastEvictWaitNs=0;
        if(!core) {
            core=std::make_unique<Core>();auto& c=*core;
            c.base.resize(size);read(0,c.base.data(),size);
            c.needsSource.resize((size+BlockSize-1)/BlockSize);
            c.latest.assign(c.needsSource.size(),kNone);
            c.slots[0].frame=frame;c.count=1;c.next=1;
            lastReadBytes=lastPayload=size;Measure();return;
        }
        DropPrepared();
        auto& c=*core;
        if(size!=c.base.size())throw std::runtime_error("snapshot region layout changed during match");
        if(frame<=c.slots[(c.next+Capacity-1)%Capacity].frame)
            throw std::runtime_error("rollback: save must follow the loaded snapshot");
        if(dirty && anchor!=c.slots[(c.next+Capacity-1)%Capacity].frame)
            throw std::runtime_error("rollback: dirty tracker does not follow current snapshot");
        if(c.count==Capacity) {
            const auto waitStart=Clock::now();
            c.jobs.Wait(c.eviction);lastEvictWaitNs=Elapsed(waitStart);
            // Keep the old allocation for reuse after its eviction finishes.
            std::swap(c.evicted,c.slots[c.next]);
            c.eviction=c.jobs.Start([&c](job::JobTaskThread&,job::Job&){Core::Apply(c.base,c.evicted);});
        }
        auto& slot=c.slots[c.next];slot.frame=frame;slot.count=0;
        const auto indexStart=Clock::now();
        if(!dirty) {
            c.dirtyKeys.clear();
            for(size_t i=0;i<c.needsSource.size();++i)c.dirtyKeys.push_back(uint32_t(i));
        }
        const auto& keys=dirty?*dirty:c.dirtyKeys;
        if(std::adjacent_find(keys.begin(),keys.end(),[](uint32_t a,uint32_t b){return a>=b;})!=keys.end() ||
           (!keys.empty() && keys.back()>=c.needsSource.size()))
            throw std::runtime_error("rollback: invalid dirty block list");
        if(keys.size()>=(size_t{1}<<kIndexBits))throw std::runtime_error("rollback: too many dirty blocks for one save");
        slot.Reserve(keys.size());
        const size_t slotNumber=c.next;
        std::array<size_t,SnapshotJobs::SaveChunks> readBytes{};
        const size_t split=(keys.size()+SnapshotJobs::SaveChunks-1)/SnapshotJobs::SaveChunks;
        lastIndexNs=Elapsed(indexStart);
        std::array<uint64_t,SnapshotJobs::SaveChunks> workerNs{};
        const auto jobsStart=Clock::now();
        c.jobs.Run<SnapshotJobs::SaveChunks>([&,read](size_t chunk) mutable {
            const auto workerStart=Clock::now();
            const size_t begin=chunk*split,end=std::min(begin+split,keys.size());
            size_t examined=0;
            for(size_t i=begin;i<end;++i) {
                const uint32_t key=keys[i];const size_t offset=size_t(key)*BlockSize;
                const size_t count=std::min(BlockSize,size-offset);
                // Dolphin's gameplay region mode saves every marked granule
                // directly, without a shadow comparison or chunk compaction.
                read(offset,slot.bytes.get()+i*BlockSize,count);examined+=count;
                if(count<BlockSize)std::memset(slot.bytes.get()+i*BlockSize+count,0,BlockSize-count);
                slot.indices[i]=key;
                // keys are distinct, so each chunk owns its keys' links
                slot.previous[i]=c.latest[key];c.latest[key]=Location(slotNumber,i);
            }
            readBytes[chunk]=examined;workerNs[chunk]=Elapsed(workerStart);
        });
        lastJobsNs=Elapsed(jobsStart);lastWorkerNs=*std::max_element(workerNs.begin(),workerNs.end());
        const auto finishStart=Clock::now();
        slot.count=keys.size();
        for(size_t bytes:readBytes)lastReadBytes+=bytes;
        lastPayload=slot.count*(BlockSize+sizeof(uint32_t));
        c.next=(c.next+1)%Capacity;c.count=std::min(c.count+1,Capacity);Measure();lastFinishNs=Elapsed(finishStart);
    }
    // The part of Restore(frame, anchor, ...) that needs neither guest memory nor the dirty list, done ahead (while
    // something else finishes, e.g. the GX thread). The next Restore must be that one; a Save drops it.
    void PrepareRestore(int frame,int anchor) {
        if(!core)throw std::runtime_error("rollback: no snapshot");
        DropPrepared();
        auto& c=*core;const size_t target=c.Find(frame);
        if(anchor!=c.slots[(c.next+Capacity-1)%Capacity].frame)
            throw std::runtime_error("rollback: invalid restore anchor or dirty bitmap");
        const auto waitStart=Clock::now();c.jobs.Wait(c.eviction);preparedEvictWaitNs=Elapsed(waitStart);
        const auto indexStart=Clock::now();
        c.flagged.clear();c.flaggedSources.clear();
        // Only writes after the target can differ from its saved image: the blocks saved in the discarded slots
        // (here) and those written since the newest (Restore's dirty list). Including the target would rescan its
        // dirty list and needlessly restore blocks untouched since that snapshot. Each one's source is its newest
        // copy at or before the target: for a block saved after the target, the link of its first copy after it (the
        // discarded slots oldest first); for one only written since, its newest copy. A copy not retained means the
        // base image (evicted copies are applied there). The source is the block's newest copy from now on.
        const size_t framesBack=((c.next+Capacity-1)%Capacity+Capacity-target)%Capacity;
        for(size_t n=1;n<=framesBack;++n) {
            const auto& slot=c.slots[(target+n)%Capacity];
            for(size_t i=0;i<slot.count;++i)Flag(c,slot.indices[i],slot.previous[i]);
        }
        preparedIndexNs=Elapsed(indexStart);
        prepared=true;preparedFrame=frame;preparedAnchor=anchor;
    }
    template<class Reader,class Writer> void Restore(int frame,int anchor,const std::vector<uint32_t>& dirty,Reader read,Writer write) {
        if(!prepared || preparedFrame!=frame || preparedAnchor!=anchor)PrepareRestore(frame,anchor);
        prepared=false;
        auto& c=*core;const size_t target=c.Find(frame);
        lastIndexNs=preparedIndexNs;lastEvictWaitNs=preparedEvictWaitNs;lastJobsNs=lastWorkerNs=lastFinishNs=0;
        const size_t framesBack=((c.next+Capacity-1)%Capacity+Capacity-target)%Capacity;
        std::array<size_t,SnapshotJobs::Chunks> reads{},writes{};
        auto prepare=[&] {
            const auto indexStart=Clock::now();
            for(uint32_t key:dirty) {
                if(key>=c.needsSource.size())throw std::runtime_error("rollback: dirty block outside memory");
                Flag(c,key,c.latest[key]);
            }
            lastIndexNs+=Elapsed(indexStart);
        };
        std::array<uint64_t,SnapshotJobs::Chunks> workerNs{};
        const size_t through=c.Age(target)+1;
        const auto jobsStart=Clock::now();
        c.jobs.RunPrepared(prepare,[&,read,write](size_t chunk) mutable {
            const auto workerStart=Clock::now();
            const size_t begin=c.flagged.size()*chunk/SnapshotJobs::Chunks;
            const size_t end=c.flagged.size()*(chunk+1)/SnapshotJobs::Chunks;
            size_t examined=0,written=0;
            for(size_t n=begin;n<end;++n) {
                // flagged keys are distinct: each chunk owns its keys' entries
                const uint32_t key=c.flagged[n];
                uint32_t at=c.flaggedSources[n];
                if(!c.Valid(at,key,through))at=kNone;
                c.needsSource[key]=0;c.latest[key]=at;
                const size_t offset=size_t(key)*BlockSize,count=std::min(BlockSize,c.base.size()-offset);
                const uint8_t* saved=at==kNone?c.base.data()+offset:c.slots[SlotOf(at)].bytes.get()+IndexOf(at)*BlockSize;
                uint8_t now[BlockSize];read(offset,now,count);examined+=count;
                if(std::memcmp(now,saved,count)){write(offset,saved,count);written+=count;}
            }
            reads[chunk]=examined;writes[chunk]=written;workerNs[chunk]=Elapsed(workerStart);
        });
        lastJobsNs=Elapsed(jobsStart);lastWorkerNs=*std::max_element(workerNs.begin(),workerNs.end());
        const auto finishStart=Clock::now();
        lastRestoreReadBytes=lastRestoreWriteBytes=0;
        for(size_t i=0;i<reads.size();++i){lastRestoreReadBytes+=reads[i];lastRestoreWriteBytes+=writes[i];}
        // Discard the predicted future, exactly as LoadFrame rewinds Dolphin's ring.
        c.next=(target+1)%Capacity;c.count-=framesBack;Measure();lastFinishNs=Elapsed(finishStart);
    }
    size_t DeltaBytes() const {
        if(!core)return 0;size_t n=0;
        for(size_t i=0;i<core->count;++i)n+=core->slots[(core->Oldest()+i)%Capacity].count*(BlockSize+sizeof(uint32_t));
        return n;
    }
    size_t ResidentBytes() const {return core?core->ResidentBytes():0;}
    void Measure(){peakResident=std::max(peakResident,ResidentBytes());}
};
}
