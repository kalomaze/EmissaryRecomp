// Gameplay rollback prototype. Pad anchoring and region discovery ported from
// Brawl Online GameplayRollback.cpp / gp-v29 (GPL-2.0-or-later).
#include "brawl_rollback.h"
#include "profile_flush.h"
#include "abi_bridge.h"
#include "axfx_async.h"
#include "det_clock.h"
#include "brawl_snapshot_store.h"
#include "brawl_write_watch.h"
#include <memory>
#include "brawl_rollback_udp.h"
#include "brawl_online_ipc.h"
#include "brawl_online_core.h"
#include "memory.h"
#include "ppc_runtime.h"
#include "save_states.h"
#include "gekkonet.h"
#include "hle_stubs.h"
#include "g3d_native.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <random>
#include <thread>
#include <unordered_map>
#include <map>
#include <tuple>
#include <iterator>
#if defined(__x86_64__)
#include <nmmintrin.h>
#endif


namespace BrawlRollback {
namespace {
struct Range { uint32_t addr, size; };
struct BackingRange {size_t offset,end;uint8_t* data;};
#include "brawl_rollback_regions.inc"
using Pads = std::array<uint8_t, 256>;
struct Snapshot {
    int frame = -999;
    CpuContext cpu{};
    std::vector<uint8_t> bytes;
    uint32_t markSeq = 0;  // BrawlDirtyMemory::consumeSeq once taken: guest memory as of that consume
};
struct Session {
    bool configured = false, active = false, finished = false, sawMatchReset = false;
    bool live = false, resimulating = false, stopRequested = false;
    unsigned localPad=0, matchIndex=0, mailbox=0, mailboxSize=0;
    int gameSetFrame=-1;
    unsigned mask = 0, cursor = 0, distance = 0, start = 240, limit = 600;
    unsigned udpPort=0, peerPort=0, localSlot=0, delayMs=0;
    uint64_t saves = 0, loads = 0, advances = 0, mismatches = 0, regionDifferences = 0;
    std::vector<Pads> input;
    Pads current{}, recordedCountdown{};
    bool hasRecordedCountdown=false;
    std::vector<Range> ranges;
    std::vector<BackingRange> backing;
    std::array<Snapshot, 16> snapshots;
    SnapshotStore incremental;
    std::unique_ptr<WriteWatch> writeWatch;
    int dirtyAnchor=0;
    bool trackWrites=false,regionHash=false;
    bool fullSnapshots=false,verifySnapshots=false;
    uint64_t snapshotPayload=0,snapshotDeltaSaves=0,snapshotSaveNs=0,snapshotLoadNs=0;
    FILE* snapshotMetrics=nullptr;
    std::array<int, 4> handles{{-1,-1,-1,-1}};
    GekkoSession* gekko = nullptr;
    FILE* trace = nullptr;
};
Session session;
bool crossplayStageSeen=false;
bool matchSeeded=false;  // Brawl Online's did_frame0: this match's RNGs are seeded
std::array<uint8_t,64> latestLocalPad{};
uint32_t Read(uint32_t p) { return Memory::Contains(p,4) ? Memory::Read32(p) : 0; }
unsigned Env(const char* key, unsigned fallback) {
    const char* value = std::getenv(key);
    return value ? static_cast<unsigned>(std::stoul(value)) : fallback;
}
// GameplaySession::RunIoWait: retire mid-match resource loads before hashing
// the completed frame or saving it. In particular, Zelda/Sheik must finish
// loading the other fighter form between the same two frames on both peers.
void WaitForResourceLoads(CpuContext* ctx) {
    auto pending=[] {
        uint32_t count=0;
        const auto manager=Read(0x8059FFF4);
        if(manager) for(uint32_t offset : {8u,12u}) {
            const auto queue=Read(manager+offset);
            const auto entries=queue?Read(queue+0x18):0;
            if(entries) count+=Read(entries)&0xFFFF;
        }
        const auto snd=Read(0x805A01D0);
        const auto loader=snd?Read(snd+0xB98):0;
        if(loader) {
            const auto queued=Read(loader+0xCC8),loading=Read(loader+0xCCC);
            if(queued<=0x95 && loading<=0x95) count+=queued+loading;
        }
        return count;
    };
    if(!pending()) return;
    CpuContext call=*ctx;
    CpuContextScope scope(&call);
    unsigned retraces=0;
    while(pending()) {
        if(retraces==600) throw std::runtime_error("rollback: resource loads did not finish within 600 retraces");
        call.gpr[3]=Read(0x8059FFF4);
        InvokeIndirectCpu(0x80022F84,&call); // gfFileIOManager::update
        if(!pending()) break;
        InvokeIndirectCpu(0x801E892C,&call); // VIWaitForRetrace: let loader threads run
        ++retraces;
    }
    std::fprintf(stderr,"[rollback] resource wait game_frame=%u retraces=%u\n",Read(0x901812A4),retraces);
}
void Add(uint32_t addr, uint32_t size) {
    if (size && Memory::Contains(addr,size)) session.ranges.push_back({addr,size});
}
void DiscoverRegions() {
    for (unsigned i=1;i<0x48;++i) {
        const auto entry=0x80494958u+16*i, name=Read(entry);
        if (!Memory::Contains(name,40)) continue;
        const char* text=reinterpret_cast<const char*>(Memory::GetPointer(name));
        std::string heap(text,strnlen(text,40));
        if (std::find(std::begin(kHeaps),std::end(kHeaps),heap)!=std::end(kHeaps))
            Add(Read(entry+4),Read(entry+8));
    }
    std::vector<uint32_t> seen;
    for (uint32_t m=Read(0x800030C8);m && seen.size()<256;m=Read(m+4)) {
        if (std::find(seen.begin(),seen.end(),m)!=seen.end()) break;
        seen.push_back(m);
        const auto count=Read(m+0xC), table=Read(m+0x10);
        if (count>64) throw std::runtime_error("rollback: invalid module table");
        for (unsigned i=0;i<count;++i) {
            const auto addr=Read(table+8*i),size=Read(table+8*i+4);
            if (!(addr&1)) Add(addr,size);
        }
    }
    const auto snd=Read(0x805A01D0),hs=Read(snd+0x2D0);
    if (Memory::Contains(hs,0x504)) {
        Add(hs,0x258);
        for (unsigned i=0;i<12;++i) {
            Add(hs+0x258+0x2C*i+0x1C,0x10);
            const auto base=Read(hs+0x474+4*i),size=Read(hs+0x4A4+4*i);
            if (size<=0x1000000) Add(base,size);
        }
        Add(hs+0x468,0x504-0x468);
    }
    const auto table=Read(snd+0x18C+0x14),count=Read(table);
    if (count && count<=0x10000) Add(table,4+8*count);
    for (auto r:kAdded) Add(r.addr,r.size);
    Add(0x901812A0,0x18); // simulation and persistent frame counters
    Add(0x8059C668,4); // object serial numbers
    const auto app=Read(0x8059FFAC);
    if (app) Add(app+0x100,4);
    std::sort(session.ranges.begin(),session.ranges.end(),[](auto a,auto b){return a.addr<b.addr;});
    std::vector<Range> merged;
    for (auto r:session.ranges) {
        if (!merged.empty() && uint64_t(merged.back().addr)+merged.back().size>=r.addr)
            merged.back().size=std::max(uint64_t(merged.back().addr)+merged.back().size,uint64_t(r.addr)+r.size)-merged.back().addr;
        else merged.push_back(r);
    }
    for (auto holes:{std::vector<Range>(std::begin(kRemoved),std::end(kRemoved)),
                     std::vector<Range>(std::begin(kExcluded),std::end(kExcluded)),
                     session.mailboxSize ? std::vector<Range>{{session.mailbox,session.mailboxSize}} : std::vector<Range>{}}) {
        for (auto h:holes) {
            std::vector<Range> out;
            for (auto r:merged) {
                const uint64_t end=uint64_t(r.addr)+r.size, he=uint64_t(h.addr)+h.size;
                if (end<=h.addr || r.addr>=he) out.push_back(r);
                else {
                    if (r.addr<h.addr) out.push_back({r.addr,h.addr-r.addr});
                    if (end>he) out.push_back({uint32_t(he),uint32_t(end-he)});
                }
            }
            merged=std::move(out);
        }
    }
    session.ranges=std::move(merged);
    session.backing.clear();
    size_t packedOffset=0;
    for(const auto r:session.ranges) {
        session.backing.push_back({packedOffset,packedOffset+r.size,Memory::GetPointer(r.addr,r.size)});
        packedOffset+=r.size;
    }
    if(session.trackWrites) {
#if MKW_LEAN_MEMORY && !MKW_LEAN_DIRTY_MARKS
        // Guest stores in this build do not mark the dirty map (brawl_dirty_memory.h), so WriteWatch would see no
        // writes and every snapshot would miss them. Refuse rather than desync.
        std::fprintf(stderr,"[rollback] built with MKW_LEAN_DIRTY_MARKS=0: no write tracking, rollback unavailable\n");
        std::abort();
#endif
#if MKW_FLAT_MEMORY
        session.writeWatch=std::make_unique<WriteWatch>(reinterpret_cast<uintptr_t>(MKW_FLAT_GUEST_BASE));size_t packed=0;
#elif MKW_LEAN_MEMORY  // marks are indexed by the canonical window offset (brawl_dirty_memory.h)
        session.writeWatch=std::make_unique<WriteWatch>(reinterpret_cast<uintptr_t>(MemoryInline::g_leanBase));size_t packed=0;
#else
        session.writeWatch=std::make_unique<WriteWatch>();size_t packed=0;  // heap memory: the registered hosts map it
#endif
        // Register the two RAM backing allocations once for retained HLE pointers.
        // WriteWatch::Add reuses these mappings for their selected subranges.
        BrawlDirtyMemory::hosts.push_back({reinterpret_cast<uintptr_t>(Memory::GetPointer(Memory::kMem1CachedBase,Memory::kMem1BankSize)),Memory::kMem1BankSize,Memory::kMem1CachedBase});
        BrawlDirtyMemory::hosts.push_back({reinterpret_cast<uintptr_t>(Memory::GetPointer(Memory::kMem2CachedBase,Memory::kMem2Size)),Memory::kMem2Size,Memory::kMem2CachedBase});
        for(auto r:session.ranges){
            session.writeWatch->Add(Memory::GetPointer(r.addr,r.size),r.size,packed,r.addr);
            packed+=r.size;
        }
    }
    size_t total=0;for(auto r:session.ranges) total+=r.size;
    if (total<1000000) throw std::runtime_error("rollback: match heaps not ready");
    std::fprintf(stderr,"[rollback] %zu ranges, %zu bytes, distance %u\n",session.ranges.size(),total,session.distance);
}
void Configure() {
    session.configured=true;
    const char* mode=std::getenv("BRAWL_SNAPSHOT_MODE");
    if(mode && std::strcmp(mode,"full") && std::strcmp(mode,"scan") && std::strcmp(mode,"delta"))
        throw std::runtime_error("invalid BRAWL_SNAPSHOT_MODE; expected full, scan, or delta");
    session.fullSnapshots=mode && std::strcmp(mode,"full")==0;
    session.trackWrites=!session.fullSnapshots && !(mode && std::strcmp(mode,"scan")==0);
    session.regionHash=Env("BRAWL_ROLLBACK_REGION_HASH",0)!=0;
    session.verifySnapshots=Env("BRAWL_SNAPSHOT_VERIFY",0)!=0;
    const char* path=std::getenv("BRAWL_ROLLBACK_INPUT");
    if (!path && !session.live) return;
    uint32_t count=0;
    if(path) {
    std::ifstream in(path,std::ios::binary);
    char magic[8];
    in.read(magic,8);in.read(reinterpret_cast<char*>(&count),4);
    in.read(reinterpret_cast<char*>(&session.mask),4);
    if (!in || std::memcmp(magic,"BRPAD001",8) || count>1000000 || !count || !session.mask || session.mask>15)
        throw std::runtime_error("rollback: invalid BRPAD input");
    session.input.resize(count);
    in.read(reinterpret_cast<char*>(session.input.data()),count*sizeof(Pads));
    if (!in || in.peek()!=EOF) throw std::runtime_error("rollback: input length mismatch");
    } else {session.mask=3;count=1000000000;}
    if(const char* countdown=std::getenv("BRAWL_REPLAY_COUNTDOWN")) {
        // Stage constructors consume this seed before recorded start-state synchronization.
        const char* index=std::getenv("BRAWL_REPLAY_MATCH_INDEX");
        if(!index)throw std::runtime_error("rollback: recorded countdown requires its recorded match index");
        session.matchIndex=static_cast<unsigned>(std::stoul(index));
        std::ifstream in(countdown,std::ios::binary);
        in.read(reinterpret_cast<char*>(session.recordedCountdown.data()),session.recordedCountdown.size());
        if(!in || in.peek()!=EOF)throw std::runtime_error("rollback: recorded countdown must contain exactly four gfPadStatus records");
        session.hasRecordedCountdown=true;
    }
    session.distance=Env("BRAWL_ROLLBACK_DISTANCE",0);
    session.udpPort=Env("BRAWL_ROLLBACK_PORT",0);
    session.peerPort=Env("BRAWL_ROLLBACK_PEER_PORT",0);
    session.localSlot=Env("BRAWL_ROLLBACK_SLOT",0);
    session.delayMs=Env("BRAWL_ROLLBACK_DELAY_MS",0);
    if(session.udpPort && (session.mask!=3 || session.localSlot>1 || !session.peerPort))
        throw std::runtime_error("rollback: UDP mode requires ports 1/2, slot 0/1, and peer port");
    if (session.distance>7) throw std::runtime_error("rollback: distance must be 0..7");
    session.start=Env("BRAWL_ROLLBACK_START",240);
    session.limit=session.live?count:std::min(count,Env("BRAWL_ROLLBACK_FRAMES",600));
    const char* trace=std::getenv("BRAWL_ROLLBACK_TRACE");
    std::string tracePath=trace?trace:"rollback-trace.csv";
    if(session.live) tracePath+=".match-"+std::to_string(session.matchIndex+1)+".csv";
    session.trace=std::fopen(tracePath.c_str(),"w");
    session.snapshotMetrics=std::fopen((tracePath+".snapshots.csv").c_str(),"w");
    if(!session.snapshotMetrics)throw std::runtime_error("cannot open snapshot metrics");
    std::fprintf(session.snapshotMetrics,"event,frame,selected_ram_bytes,state_payload_bytes,delta_history_bytes,resident_bytes,duration_ns,verified,ram_bytes_examined,dirty_metadata_bytes,tracking_capture_ns,tracking_reset_ns,index_ns,jobs_ns,max_worker_ns,finish_ns,eviction_wait_ns,tracking_jobs_ns,tracking_worker_ns,tracking_merge_ns\n");
    if (!session.trace) throw std::runtime_error("rollback: cannot open trace");
    std::fprintf(session.trace,"session_frame,game_frame,rng,rng_other,libc_rng,region_hash,resimulation,gameplay_hash,fighters,input_hash,expected_input_hash,completed_ns,advance_ns,pace_ns\n");
    std::fprintf(stderr,"[rollback] %s input, limit %u\n",path?path:"live controller",count);
}
uint32_t Hash() {
    AxfxAsync::Join();  // the ranges may include the AXFX jobs' buffers and lines (axfx_async.h)
    uint32_t h=2166136261u;
    for (auto r:session.ranges) {
        auto* p=Memory::GetPointer(r.addr,r.size);
        unsigned i=0;
#if defined(__x86_64__)
        for(;i+8<=r.size;i+=8) {uint64_t word;std::memcpy(&word,p+i,8);h=_mm_crc32_u64(h,word);}
        for(;i<r.size;++i) h=_mm_crc32_u8(h,p[i]);
#else
        for(;i<r.size;++i) h=(h^p[i])*16777619u;
#endif
    }
    return h;
}
// Same gameplay checksum contract as Brawl Online GameplayRollback/GameplaySession.
using Fighters = std::array<std::array<uint32_t,6>,4>;
Fighters FighterState() {
    Fighters out{};for(auto& f:out) f[0]=0xff;
    const auto entries=Read(0x80624780), mm=Read(Read(0x805A00E0)+8);
    unsigned ports=0,players=0;
    for(unsigned p=0;p<4;++p) if(Memory::Contains(mm+0x99+p*0x5C) && Memory::Read8(mm+0x99+p*0x5C)<=1) {ports|=1u<<p;++players;}
    unsigned seen=0;
    for(unsigned k=0;k<std::min({Read(0x80624784),9u,players});++k) {
        const auto e=entries+k*0x244, p=Read(e+0x58);
        if(Read(e+4)==0xffffffff || p>=4 || !(ports&(1u<<p)) || (seen&(1u<<p))) continue;
        seen|=1u<<p;auto& f=out[p];f[0]=Memory::Read8(e+0xA);
        const auto data=Read(Read(e+0x28));
        f[1]=Read(data+0x24);f[2]=Read(data+0x34);
        if(f[0]<4) {
            const auto en=Read(Read(Read(e+0x34+8*f[0])+0x60)+0xD8);
            const auto posture=Read(en+0xC),status=Read(en+0x70);
            f[3]=Read(posture+0xC);f[4]=Read(posture+0x10);f[5]=Read(status+0x34);
        }
    }
    return out;
}
uint32_t Crc(const void* data,size_t length) {
    uint32_t crc=~0u;auto* p=static_cast<const uint8_t*>(data);
    for(size_t i=0;i<length;++i) {crc^=p[i];for(unsigned bit=0;bit<8;++bit) crc=(crc>>1)^(0xEDB88320u&-(crc&1));}
    return ~crc;
}
uint32_t GameplayHash() {
    const auto f=FighterState();
    const uint32_t words[]={Read(0x901812A4),Read(0x805A00BC),Read(0x805A0424),Read(0x8059FF58),Crc(f.data(),sizeof(f))};
    return Crc(words,sizeof(words));
}
// Each worker owns its range cursor. Most consecutive granules belong to the
// same allocation; crossing a selected-region boundary resolves the next one.
template<bool Write> MKW_PPC_FORCE_INLINE void TransferPacked(size_t& cursor,size_t offset,uint8_t* bytes,size_t count) {
    if(!count)return;
    const auto& current=session.backing.at(cursor);
    if(count==64 && offset>=current.offset && offset<current.end && current.end-offset>=64) {
        auto* guest=current.data+offset-current.offset;
        if constexpr(Write)std::memcpy(guest,bytes,64);else std::memcpy(bytes,guest,64);
        return;
    }
    auto at=session.backing.begin()+cursor;
    if(offset<current.offset || offset>=current.end)
        at=std::lower_bound(session.backing.begin(),session.backing.end(),offset,
            [](const BackingRange& r,size_t o){return r.end<=o;});
    while(count) {
        if(at==session.backing.end() || offset<at->offset)throw std::runtime_error("snapshot outside packed memory");
        const size_t n=std::min(count,at->end-offset);
        auto* guest=at->data+offset-at->offset;
        if(n==64) {
            if constexpr(Write)std::memcpy(guest,bytes,64);else std::memcpy(bytes,guest,64);
        } else {
            if constexpr(Write)std::memcpy(guest,bytes,n);else std::memcpy(bytes,guest,n);
        }
        bytes+=n;offset+=n;count-=n;
        if(count)++at;
    }
    cursor=size_t(at-session.backing.begin());
}
void Save(int frame,CpuContext* ctx) {
    AxfxAsync::Join();  // no AXFX job still writing guest memory (axfx_async.h)
    const auto started=std::chrono::steady_clock::now();
    auto& s=session.snapshots[(frame+16)%16];
    size_t total=0;for(auto r:session.ranges)total+=r.size;
    if(s.frame==frame && !session.udpPort && (session.verifySnapshots || (session.regionHash && (session.fullSnapshots || session.incremental.Contains(frame))))) {
        const auto& old=(session.fullSnapshots || session.verifySnapshots)?s.bytes:session.incremental.View(frame);
        size_t pos=0;unsigned differences=0;
        for(auto r:session.ranges) {
            if(std::memcmp(Memory::GetPointer(r.addr,r.size),old.data()+pos,r.size))++differences;
            pos+=r.size;
        }
        if(differences)++session.regionDifferences;
    }
    s.frame=frame;s.cpu=*ctx;
    if(session.fullSnapshots || session.verifySnapshots) {
        s.bytes.clear();s.bytes.reserve(total);
        for(auto r:session.ranges) {
            auto* p=Memory::GetPointer(r.addr,r.size);s.bytes.insert(s.bytes.end(),p,p+r.size);
        }
    }
    // Capture clears the tracked marks before an incremental save. Only the
    // initial base image (or a full-copy save without Capture) needs Enable's
    // reset; the subsequent snapshot copies do not mark guest memory.
    const bool resetWriteWatch=session.fullSnapshots || session.incremental.Empty();
    size_t payload=total+sizeof(CpuContext),resident=0;
    if(!session.fullSnapshots) {
        auto read=[cursor=size_t{0}](size_t offset,uint8_t* out,size_t count) mutable {TransferPacked<false>(cursor,offset,out,count);};
        const std::vector<uint32_t> empty;
        const auto& dirty=session.writeWatch && !session.incremental.Empty()?
            session.writeWatch->Capture(total,session.incremental.Jobs()):empty;
        session.incremental.Save(frame,total,read,session.writeWatch?&dirty:nullptr,session.dirtyAnchor);
        s.markSeq=BrawlDirtyMemory::consumeSeq.load(std::memory_order_acquire);
        payload=session.incremental.lastPayload+sizeof(CpuContext);
        resident=session.incremental.ResidentBytes()+sizeof(session.snapshots);
        if(session.incremental.Count()>1){session.snapshotPayload+=payload;++session.snapshotDeltaSaves;}
        if(session.verifySnapshots && session.incremental.View(frame)!=s.bytes) {
            const auto& image=session.incremental.View(frame);size_t packed=0;
            for(auto r:session.ranges) {
                for(size_t i=0;i<r.size;++i)if(image[packed+i]!=s.bytes[packed+i]) {
                    std::fprintf(stderr,"[dirty-miss] save frame=%d guest=%08x stored=%02x live=%02x\n",frame,r.addr+unsigned(i),image[packed+i],s.bytes[packed+i]);
                    throw std::runtime_error("incremental save differs from full-copy reference");
                }
                packed+=r.size;
            }
        }
    } else for(const auto& snapshot:session.snapshots)resident+=snapshot.bytes.capacity()+sizeof(snapshot);
    if(session.writeWatch && resetWriteWatch)session.writeWatch->Enable();session.dirtyAnchor=frame;
    ++session.saves;
    const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
    session.snapshotSaveNs+=ns;
    std::fprintf(session.snapshotMetrics,"save,%d,%zu,%zu,%zu,%zu,%lld,%d,%zu,%zu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",frame,total,payload,
        session.incremental.DeltaBytes(),resident,(long long)ns,session.verifySnapshots,session.fullSnapshots?total:session.incremental.lastReadBytes,
        session.writeWatch?session.writeWatch->metadataBytes:0,
        (unsigned long long)(session.writeWatch?session.writeWatch->captureNs:0),
        (unsigned long long)(session.writeWatch?session.writeWatch->resetNs:0),
        (unsigned long long)session.incremental.lastIndexNs,(unsigned long long)session.incremental.lastJobsNs,
        (unsigned long long)session.incremental.lastWorkerNs,(unsigned long long)session.incremental.lastFinishNs,
        (unsigned long long)session.incremental.lastEvictWaitNs,
        (unsigned long long)(session.writeWatch?session.writeWatch->captureJobsNs:0),
        (unsigned long long)(session.writeWatch?session.writeWatch->captureWorkerNs:0),
        (unsigned long long)(session.writeWatch?session.writeWatch->captureMergeNs:0));
}
void Load(int frame,CpuContext* ctx) {
    AxfxAsync::Join();  // no AXFX job still writing guest memory (axfx_async.h)
    const auto started=std::chrono::steady_clock::now();
    auto& s=session.snapshots[(frame+16)%16];
    if(s.frame!=frame) throw std::runtime_error("rollback: snapshot outside ring");
    size_t total=0;for(auto r:session.ranges)total+=r.size;
    size_t examined=total,written=total;
    // The restore rewrites guest memory without marks: what was written since the snapshot (all marked and consumed
    // since, or by the Capture below), or everything.
    if(session.fullSnapshots || !session.writeWatch) BrawlDirtyMemory::MarksCleared();
    else BrawlDirtyMemory::MarksRestoring(s.markSeq);
    // the code caches compare what the restore wrote once it is done, also when it throws (MarksRestoring's contract)
    struct RestoreWritten { ~RestoreWritten() { BrawlDirtyMemory::MarksRestoreWritten(); } } restoreWritten;
    if(!session.fullSnapshots) {
        std::vector<uint32_t> all;
        if(!session.writeWatch)for(size_t i=0;i<(total+63)/64;++i)all.push_back(uint32_t(i));
        const auto& dirty=session.writeWatch?session.writeWatch->Capture(total,session.incremental.Jobs()):all;
        auto read=[cursor=size_t{0}](size_t offset,uint8_t* out,size_t count) mutable {TransferPacked<false>(cursor,offset,out,count);};
        auto write=[cursor=size_t{0}](size_t offset,const uint8_t* in,size_t count) mutable {TransferPacked<true>(cursor,offset,const_cast<uint8_t*>(in),count);};
        session.incremental.Restore(frame,session.dirtyAnchor,dirty,read,write);
        examined=session.incremental.lastRestoreReadBytes;written=session.incremental.lastRestoreWriteBytes;
    } else {
        const auto& bytes=session.fullSnapshots?s.bytes:session.incremental.View(frame);
        size_t pos=0;
        for(auto r:session.ranges){std::memcpy(Memory::GetPointer(r.addr,r.size),bytes.data()+pos,r.size);pos+=r.size;}
    }
    if(session.verifySnapshots) {
        size_t pos=0;
        for(auto r:session.ranges){
            if(std::memcmp(Memory::GetPointer(r.addr,r.size),s.bytes.data()+pos,r.size))
                throw std::runtime_error("restored guest memory differs from full-copy reference");
            pos+=r.size;
        }
    }
    *ctx=s.cpu;++session.loads;
    // Capture consumed the incremental path's marks; TransferPacked restores
    // bytes directly without setting new marks. Full-copy loads skip Capture.
    if(session.writeWatch && session.fullSnapshots)session.writeWatch->Enable();session.dirtyAnchor=frame;
    const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
    session.snapshotLoadNs+=ns;
    std::fprintf(session.snapshotMetrics,"load,%d,%zu,%zu,%zu,%zu,%lld,%d,%zu,%zu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",frame,total,written,session.incremental.DeltaBytes(),
        session.incremental.ResidentBytes()+sizeof(session.snapshots),(long long)ns,session.verifySnapshots,examined,session.writeWatch?session.writeWatch->metadataBytes:0,
        (unsigned long long)(session.writeWatch?session.writeWatch->captureNs:0),
        (unsigned long long)(session.writeWatch?session.writeWatch->resetNs:0),
        (unsigned long long)session.incremental.lastIndexNs,(unsigned long long)session.incremental.lastJobsNs,
        (unsigned long long)session.incremental.lastWorkerNs,(unsigned long long)session.incremental.lastFinishNs,
        (unsigned long long)session.incremental.lastEvictWaitNs,
        (unsigned long long)(session.writeWatch?session.writeWatch->captureJobsNs:0),
        (unsigned long long)(session.writeWatch?session.writeWatch->captureWorkerNs:0),
        (unsigned long long)(session.writeWatch?session.writeWatch->captureMergeNs:0));
}
} // namespace

// The gameplay checksum outside a session, for the determinism diagnostics (det_hash.cpp, NSMBW_DET_HASH).
uint32_t CurrentGameplayHash() { return GameplayHash(); }

// Called at the loop's IPC safe point, before entering a new match.
void BeginOnlineMatch(unsigned index,unsigned pad,unsigned mailbox,unsigned mailboxSize) {
    if(session.active) throw std::runtime_error("online: previous rollback match is still active");
    if(session.trace) std::fclose(session.trace);
    if(session.snapshotMetrics)std::fclose(session.snapshotMetrics);
    if(session.gekko) gekko_destroy(&session.gekko);
    CloseUdp();session=Session{};session.live=true;session.matchIndex=index;
    session.localPad=pad;session.mailbox=mailbox;session.mailboxSize=mailboxSize;
    crossplayStageSeen=false;matchSeeded=false;latestLocalPad={};latestLocalPad[0x38]=0xff;
    session.current[2*64+0x38]=0xff;
    session.current[3*64+0x38]=0xff;
}
unsigned OnlineHold() {
    const auto* pad = latestLocalPad.data();
    const bool wii = pad[0x3c] == 0 && pad[0x3d] == 0 && pad[0x3e] == 0 &&
                     pad[0x3f] >= 1 && pad[0x3f] <= 3;
    const unsigned type = wii ? unsigned(pad[0x3f]) << 12 : 0;
    return type | (pad[0x38] ? 0 : ((unsigned(pad[2]) << 8 | pad[3]) & 0xf7fu));
}
void SetOnlineCountdown(const std::array<unsigned, 4>& holds) {
    if (!session.live || session.active)
        throw std::runtime_error("countdown outside match setup");
    session.recordedCountdown = {};
    for (unsigned p = 0; p < 4; ++p) {
        auto* pad = session.recordedCountdown.data() + 64 * p;
        if (p >= 2) {
            pad[0x38] = 0xff;
            continue;
        }
        for (unsigned off : {0u, 4u}) {
            pad[off + 2] = (holds[p] & 0xf7f) >> 8;
            pad[off + 3] = holds[p] & 0xff;
        }
        pad[0x3f] = (holds[p] & 0x3000) >> 12;
    }
    session.hasRecordedCountdown = true;
}
void StopOnlineMatch(){session.stopRequested=true;if(!session.active){session.live=false;session.finished=true;}}
std::string OnlineStatus() {
    return std::string(session.active?"running":session.finished?"ended":session.live?"armed":"idle")+" "+
        std::to_string(session.cursor)+" "+std::to_string(session.loads)+" "+
        std::to_string(session.advances)+" "+std::to_string(session.mismatches);
}

void UpdateSystem(CpuContext* ctx,GuestFunction update) {
    update(ctx);
}

void AnchorPads() {
    // Brawl Online's core takes the pad thread's read (GameplayRollback's pads).
    // The pad thread's call of updateLow returns to 0x8002BA84 (P+'s own code at updateLow's end has
    // run): Dolphin's PadsOnPadRead and GprbPadThreadLoopHook, in that order.
    if(BrawlOnlineCore::Enabled()) {BrawlOnlineCore::PadRead(g_currentCpuContext);BrawlOnlineCore::PadThreadUpdated(g_currentCpuContext);return;}
    if(!session.resimulating) std::memcpy(latestLocalPad.data(),Memory::GetPointer(0x805BAD00+64*session.localPad,64),64);
    if (!session.active && !((session.live || session.hasRecordedCountdown) && crossplayStageSeen && !session.finished)) return;
    std::array<uint8_t,64> absent{};absent[0x38]=0xff;
    for(unsigned p=0;p<4;++p) {
        const auto& slots=(!session.active && session.hasRecordedCountdown) ? session.recordedCountdown : session.current;
        const auto* rec=slots.data()+64*p;
        const bool wii=rec[0x3C]==0 && rec[0x3D]==0 && rec[0x3E]==0 && rec[0x3F]>=1 && rec[0x3F]<=3;
        auto empty=absent;empty[0x3F]=wii?0:2;
        BrawlDirtyMemory::Mark(0x805BAD00+64*p,64);
        BrawlDirtyMemory::Mark(0x805BAD00+64*(p+4),64);
        std::memcpy(Memory::GetPointer(0x805BAD00+64*(p+(wii?4:0)),64),rec,64);
        std::memcpy(Memory::GetPointer(0x805BAD00+64*(p+(wii?0:4)),64),empty.data(),64);
    }
    auto pad=Read(0x805A0040);
    if (Memory::Contains(pad+0x34)) Memory::Write8(pad+0x34,Memory::Read8(pad+0x34)&~8u);
}

void WriteMatchSeeds(const char* seed,const char* when,bool serial=true) {
    std::seed_seq seq{uint32_t(std::strtoul(seed,nullptr,0)),session.matchIndex,0x50502b31u};
    uint32_t out[3];seq.generate(out,out+3);out[0]&=0x7fffffff;out[1]&=0x7fffffff;
    Memory::Write32(0x805A00BC,out[0]);Memory::Write32(0x805A0424,out[1]);Memory::Write32(0x8059FF58,out[2]);
    if(serial) Memory::Write32(0x8059C668,0x10000);
    std::fprintf(stderr,"[crossplay] seeded %08x %08x %08x%s\n",out[0],out[1],out[2],when);
}

// Brawl Online's IsWaitingRoom: the online waiting room is an scMelee on the training stage with
// fewer than two players; it is not a match.
bool IsWaitingRoom() {
    const uint32_t gg=Read(0x805A00E0);
    if(!Memory::Contains(gg+8,4)) return false;
    const uint32_t mm=Read(gg+8);
    if(!Memory::Contains(mm,0x98+4*0x5C)) return false;
    if(Memory::Read16(mm+0x1A)!=0x37) return false;
    int players=0;
    for(unsigned i=0;i<4;++i) if(Memory::Read8(mm+0x98+i*0x5C+0x01)<=1) ++players;
    return players<2;
}

// Brawl Online's IsNextSceneMelee: gfSceneManager +0x8 is the scene a pending change switches to.
bool NextSceneIsMelee() {
    const uint32_t manager=Read(0x805A0060);
    if(!Memory::Contains(manager+8,4)) return false;
    const uint32_t next=Read(manager+8);
    if(!Memory::Contains(next,4)) return false;
    const uint32_t name=Read(next);
    if(!Memory::Contains(name,8)) return false;
    return std::strncmp(reinterpret_cast<const char*>(Memory::GetPointer(name,8)),"scMelee",8)==0 && !IsWaitingRoom();
}

// Brawl Online seeds the match's RNGs once, at the first loop top that sees the scene change to
// scMelee pending, or else the first one in scMelee: before the match scene loads anything. A
// dual-load stage (Lylat Cruise) draws its second stage file from g_mtRand while loading, before
// the stage's constructor.
void SeedMatchAtLoopTop() {
    const char* seed=std::getenv("BRAWL_PROTOCOL_SEED");
    if(!seed || matchSeeded || !(session.live || session.hasRecordedCountdown)) return;
    const bool melee=SaveStates::SceneName()=="scMelee" && !IsWaitingRoom();
    if(!melee && !NextSceneIsMelee()) return;
    matchSeeded=true;
    WriteMatchSeeds(seed,melee?" in scMelee":" before the scene change");
}

// Controlled setup for the cross-engine harness; ordinary prototype runs are unchanged.
void SeedCrossplayStage() {
    if(BrawlOnlineCore::Enabled()) {BrawlOnlineCore::StageCreate(g_currentCpuContext);return;}
    BrawlOnline::Stage();
    const char* seed=std::getenv("BRAWL_PROTOCOL_SEED");
    if(!seed) seed=std::getenv("BRAWL_CROSSPLAY_SEED");if(!seed) return;
    crossplayStageSeen=true;
    // Seeded at the loop top already: Brawl Online seeds the RNGs again here, so the stage and
    // everything built after it start from the same RNG state on both machines. Objects made
    // since the loop top keep their serials.
    if(matchSeeded) {WriteMatchSeeds(seed," again at the stage's construction",false);return;}
    WriteMatchSeeds(seed,"");
}
void CrossplaySetup() {
    if(!std::getenv("BRAWL_CROSSPLAY_SEED")) return;
    const auto scene=SaveStates::SceneName();const auto gg=Read(0x805A00E0);
    if(scene=="scSelStage") {
        const auto rule=Read(gg+0x1C),rec=Read(gg+0x24);
        if(Memory::Contains(rule,16)) {
            Memory::Write8(rule+2,(Memory::Read8(rule+2)&0xF8)|1);
            Memory::Write8(rule+4,4);Memory::Write8(rule+8,8);
        }
        if(Memory::Contains(rec+0x810)) Memory::Write8(rec+0x810,0);
    }
    static bool synced=false;
    const auto stage=Read(0x80B8A428);
    if(!synced && crossplayStageSeen && scene=="scMelee" && Read(0x901812A4)==1 && Memory::Contains(stage+0x1B4,16)) {
        SeedCrossplayStage();
        for(unsigned i=0;i<4;++i) Memory::Write32(stage+0x1B4+4*i,i);
        Memory::Write32(0x901812B4,2);Memory::Write32(Read(0x8059FFAC)+0x100,1000);
        synced=true;std::fprintf(stderr,"[crossplay] initial barrier; stage=%u\n",Memory::Read16(Read(gg+8)+0x1A));
    }
}
// Brawl Online GameplaySession::ResimHide. Effects must still draw because
// nw4r::ef frees dead effects during drawing; only model layers and shadows hide.
class ResimHide {
    struct Cleared {uint32_t address;uint8_t before,written;};
    std::array<Cleared,10> cleared{};
    size_t count=0;
public:
    explicit ResimHide(bool hide) {
        if(!hide)return;
        const uint32_t root=Read(0x8059FF80);
        if(!Memory::Contains(root+0x399,0x18*9) || !Memory::Contains(root+0x464))
            throw std::runtime_error("rollback: scene root unavailable during resimulation");
        auto clear=[&](uint32_t address,uint8_t bit) {
            const auto before=Memory::Read8(address);
            if(!(before&bit))return;
            const auto written=uint8_t(before&~bit);
            cleared[count++]={address,before,written};
            Memory::Write8(address,written);
        };
        for(unsigned layer=0;layer<9;++layer)clear(root+0x399+0x18*layer,0x80);
        clear(root+0x464,0x40);
    }
    ~ResimHide() {
        for(size_t i=0;i<count;++i) {
            const auto& c=cleared[i];
            if(Memory::Read8(c.address)==c.written)Memory::Write8(c.address,c.before);
        }
    }
};
// BRAWL_ROLLBACK_PHASES=1: a uprobe target for profiling rollback sessions. perf reads the game thread's cycles at each
// call ({probe,cycles}:S), which splits them into phases: the argument is a phase code (kPhase*; an advance's also
// carries its index and the event batch's advance count), or the guest address of a call the main loop body makes
// (LoopCall; the loop body itself after it returns), with kPhaseResim set while resimulating. Never inlined, so the
// symbol and the call stay.
extern "C" [[gnu::noinline]] [[gnu::used]] void BrawlRollbackPhaseMarker(uint64_t phase) {
    asm volatile("" : : "r"(phase) : "memory");
}
namespace {
constexpr uint64_t kPhaseGekko=1,kPhaseLoad=2,kPhaseSave=3,kPhaseAdvance=4,kPhaseAdvanceEnd=5,kPhaseIdle=6;
constexpr uint64_t kPhaseResim=uint64_t(1)<<32;
constexpr uint32_t kLoopBody=0x800171B4u,kRender=0x8001789Cu;
// BRAWL_ROLLBACK_PHASES: 1 marks RunLoop's events and the main phases of an advance (update, render, renderPre,
// XFB copy + present, draw pass: ~10 uprobe hits per advance); 2 only RunLoop's events; 3 every call of the loop body.
unsigned PhaseLevel() {
    static const unsigned level=Env("BRAWL_ROLLBACK_PHASES",0);
    return level;
}
bool PhaseMarkers() {return PhaseLevel()==1 || PhaseLevel()==3;}
bool MajorCall(uint64_t target) {
    return target==0x80017618u || target==kRender || target==0x8002E79Cu || target==0x80023AE4u || target==0x8002E844u;
}
void Phase(uint64_t phase) {
    const unsigned level=PhaseLevel();
    if(!level) return;
    const bool call=phase>=0x80000000u;  // calls are guest addresses
    if(call && !(level==3 || (level==1 && MajorCall(phase)))) return;
    BrawlRollbackPhaseMarker(phase|(session.resimulating?kPhaseResim:0));
}
// BRAWL_RESIM_DIFF=<file> (with BRAWL_SNAPSHOT_VERIFY's full copies): at each re-save of a frame, the 32-byte granules
// of the snapshot ranges that differ from that frame's previous save, counted per granule and by what produced the
// new state (kind: 0 a real advance, 1 the last resimulated advance of its batch, 2 an earlier resimulated advance,
// 3 a load); written to <file> with the heap table when the session completes. A diagnostic for comparing the state
// render-side work leaves in saved memory.
struct ResaveDiff {
    std::array<std::unordered_map<uint32_t,uint32_t>,4> granules;
    std::array<uint64_t,4> resaves{},differing{};
    unsigned stateKind=0;  // what produced the current state (kind above)
};
ResaveDiff resaveDiff;
const char* ResaveDiffPath() {
    static const char* path=std::getenv("BRAWL_RESIM_DIFF");
    return path;
}
void DiffResave(int frame,unsigned kind) {
    if(!ResaveDiffPath()) return;
    AxfxAsync::Join();  // compares live memory (axfx_async.h)
    const auto& s=session.snapshots[(frame+16)%16];
    if(s.frame!=frame || s.bytes.empty()) return;
    ++resaveDiff.resaves[kind];
    size_t pos=0;bool any=false;
    for(auto r:session.ranges) {
        const auto* live=Memory::GetPointer(r.addr,r.size);
        const auto* old=s.bytes.data()+pos;
        if(std::memcmp(live,old,r.size)) for(uint32_t i=0;i<r.size;i+=32) {
            const uint32_t n=std::min<uint32_t>(32,r.size-i);
            if(std::memcmp(live+i,old+i,n)) {++resaveDiff.granules[kind][(r.addr+i)&~31u];any=true;}
        }
        pos+=r.size;
    }
    if(any) ++resaveDiff.differing[kind];
}
// BRAWL_RESIM_PAGEHASH=<file> (diagnostics; BRAWL_RESIM_PAGEHASH_STRIDE, default 8): after every advance of a session
// frame divisible by the stride, one record: u32 advance number, u32 session frame, u32 kind (0 real, 1 last replay,
// 2 earlier replay), u32 page count, then a CRC32C of each 4 KiB of the snapshot ranges in their packed order (scripts:
// pagecmp.py). Two runs' files show which memory differs after which advances.
void PageHashRecord(int frame,unsigned kind) {
    AxfxAsync::Join();  // hashes live memory (axfx_async.h)
    static FILE* out=[] {
        const char* path=std::getenv("BRAWL_RESIM_PAGEHASH");
        return path?std::fopen(path,"wb"):nullptr;
    }();
    static const unsigned stride=std::max(1u,Env("BRAWL_RESIM_PAGEHASH_STRIDE",8));
    if(!out || frame<0 || unsigned(frame)%stride) return;
    static bool rangesWritten=false;
    if(!rangesWritten) {  // <file>.ranges: the snapshot ranges (guest address, size) in packed order, and the heaps
        rangesWritten=true;
        if(FILE* f=std::fopen((std::string(std::getenv("BRAWL_RESIM_PAGEHASH"))+".ranges").c_str(),"w")) {
            for(auto r:session.ranges) std::fprintf(f,"range %08x %08x\n",r.addr,r.size);
            for(unsigned i=1;i<0x48;++i) {
                const auto entry=0x80494958u+16*i,name=Read(entry);
                if(!Memory::Contains(name,40)) continue;
                const char* text=reinterpret_cast<const char*>(Memory::GetPointer(name));
                std::fprintf(f,"heap %.*s %08x %08x\n",int(strnlen(text,40)),text,Read(entry+4),Read(entry+8));
            }
            std::fclose(f);
        }
    }
    std::vector<uint32_t> crcs;
    uint32_t crc=0;size_t inPage=0;
    for(auto r:session.ranges) {
        const auto* p=Memory::GetPointer(r.addr,r.size);
        for(uint32_t i=0;i<r.size;) {
            const size_t n=std::min<size_t>(4096-inPage,r.size-i);
#if defined(__x86_64__)
            size_t j=0;
            for(;j+8<=n;j+=8) {uint64_t w;std::memcpy(&w,p+i+j,8);crc=uint32_t(_mm_crc32_u64(crc,w));}
            for(;j<n;++j) crc=_mm_crc32_u8(crc,p[i+j]);
#else
            for(size_t j=0;j<n;++j) crc=(crc^p[i+j])*16777619u;
#endif
            i+=uint32_t(n);inPage+=n;
            if(inPage==4096) {crcs.push_back(crc);crc=0;inPage=0;}
        }
    }
    if(inPage) crcs.push_back(crc);
    const uint32_t head[4]={uint32_t(session.advances),uint32_t(frame),kind,uint32_t(crcs.size())};
    std::fwrite(head,4,4,out);std::fwrite(crcs.data(),4,crcs.size(),out);std::fflush(out);
}
void WriteResaveDiff() {
    if(!ResaveDiffPath()) return;
    FILE* f=std::fopen(ResaveDiffPath(),"w");
    if(!f) return;
    for(unsigned k=0;k<4;++k)
        std::fprintf(f,"# kind %u resaves %llu differing %llu granules %zu\n",k,(unsigned long long)resaveDiff.resaves[k],
            (unsigned long long)resaveDiff.differing[k],resaveDiff.granules[k].size());
    for(unsigned i=1;i<0x48;++i) {
        const auto entry=0x80494958u+16*i,name=Read(entry);
        if(!Memory::Contains(name,40)) continue;
        const char* text=reinterpret_cast<const char*>(Memory::GetPointer(name));
        std::fprintf(f,"heap %.*s %08x %08x\n",int(strnlen(text,40)),text,Read(entry+4),Read(entry+8));
    }
    for(unsigned k=0;k<4;++k) {
        std::vector<std::pair<uint32_t,uint32_t>> rows(resaveDiff.granules[k].begin(),resaveDiff.granules[k].end());
        std::sort(rows.begin(),rows.end());
        for(auto [address,count]:rows) std::fprintf(f,"%u %08x %u\n",k,address,count);
    }
    std::fclose(f);
}
} // namespace
// Resimulation skips (BRAWL_RESIM_SKIP, default 1; 0 replays frames as before). No replayed advance is presented
// (VI_HLE_SetPresentSuppressed): each advance's render first copies the EFB the previous advance drew, so the real
// advance presents the last replayed advance's image, and the batch's earlier presents only flashed stale frames (the
// hidden advance's, without models). Their Aurora frames still end and submit (aurora_end_frame_unpresented): ending
// them later moved Aurora's frame-latent EFB readbacks and changed the shown image (41 pixels). The guest's
// copy, draw-sync and XFB bookkeeping run unchanged.
// The last replayed advance must still draw (the real advance presents its image). An earlier ("hidden") one draws an
// image nobody sees, but its draw pass is not render-only: gfSceneRoot::render also draws the effects, and skipping it
// leaves saved state that differs into the following advances (the Effect heap, its allocator header included, and
// the camera EffectManager copies at each effect draw; BRAWL_RESIM_DIFF). BRAWL_RESIM_SKIP_CALLS (hex
// guest addresses of main-loop or render calls, comma-separated) skips calls in hidden advances, for experiments only.
std::vector<uint32_t> HexList(const char* text) {
    std::vector<uint32_t> out;
    for(const char* p=text?text:"";*p;) {
        char* end=nullptr;
        const auto value=std::strtoul(p,&end,16);
        if(end==p) {++p;continue;}
        out.push_back(uint32_t(value));
        p=end;
    }
    return out;
}
struct ResimSkips {
    bool on=Env("BRAWL_RESIM_SKIP",1)!=0;
    std::vector<uint32_t> hidden=HexList(std::getenv("BRAWL_RESIM_SKIP_CALLS"));
    // BRAWL_RESIM_INLINE (hex list, diagnostics): functions the loop body calls that are interpreted inline, so their
    // own calls reach LoopCall too (phase markers, attribution, skips).
    std::vector<uint32_t> inlined=HexList(std::getenv("BRAWL_RESIM_INLINE"));
};
const ResimSkips& Skips() {
    static const ResimSkips skips;
    return skips;
}
bool Listed(const std::vector<uint32_t>& list,uint32_t target) {
    return std::find(list.begin(),list.end(),target)!=list.end();
}
// Light draws: an advance whose models are not drawn but whose guest state is a full draw's. Everything runs as usual,
// renderPre and the draw pass included, except the leaf that draws a model, nw4r::g3d::DrawResMdlDirectly (the native
// path, g3d_native.cpp), which only does the guest-visible part of its work (its Z sorts, which advance MSL std::sort's
// shared pivot generator, and G3DState's PosNrm array; g3dn::SetDrawSuppressed). The rest of the draw pass has to run:
// its writes persist (BRAWL_RESIM_ATTRIB + BRAWL_RESIM_PAGEHASH): nw4r::ef's particle draws with their
// begin/end and the emitters' restore walk, the scene graph's calc, child insert/remove and the models' draw callbacks
// (a Tmp-heap word fighters read, stage state), the task scheduler's phase; so does renderPre's fighter model calc,
// which later effects follow. Skipping the scene graph's draw calls instead (or hiding layers, as ResimHide does, which
// also drops their calc) leaves state that later frames read.
// BRAWL_RESIM_HIDE: how replays before the last one of their batch draw: 1 with ResimHide (model layers and shadows
// hidden, their calc too: the replay's state differs from a full one, 39 pages of Effect heap, stage and System at
// distance 3, unseen by the checksums); 2 light (exactly a full draw's state); 0 in full (diagnostics).
unsigned HideLevel() {
    static const unsigned level=Env("BRAWL_RESIM_HIDE",2);
    return level;
}
// BRAWL_RESIM_LIGHT_LAST=1 (default 0; needs BRAWL_RESIM_SKIP): the last replay of a batch draws light too, and the real
// advance after it shows the image already in the display copy, the frame that was mispredicted, for that one frame
// (its own copy only clears the EFB). The guest state is that of a full replay; only the picture differs.
bool LightLast() {
    static const bool on=Env("BRAWL_RESIM_LIGHT_LAST",0)!=0 && Env("BRAWL_RESIM_SKIP",1)!=0;
    return on;
}
bool hiddenAdvance=false;  // the current advance is a hidden replay (RunLoop)
bool inAdvance=false;      // RunLoop is running an advance's gameProc

// BRAWL_RESIM_ATTRIB=<file of hex 32-byte granule addresses> (diagnostics): during real advances, which call of the
// loop body (or of a function it runs inline) changes each listed granule. A call's leaf execution runs between its
// LoopCall and LoopCallDone; interpreted code between two such events is charged to "interp after <call>". Written to
// <file>.attrib before the session's COMPLETE line: granule, call (bit 31 clear: interpreted segment), count.
struct Attribution {
    std::vector<uint32_t> granules;
    std::vector<uint8_t> shadow;  // the granules' contents at the previous event
    std::map<std::tuple<uint32_t,uint32_t,uint32_t>,uint32_t> writes;  // (granule, segment, major phase) -> count
    uint32_t last=0;              // the previous event's call
    uint32_t phase=0;             // the latest major call (MajorCall) of the advance: update, renderPre, draw pass...
    bool active=false;
};
Attribution& Attrib() {
    static Attribution a=[] {
        Attribution out;
        if(const char* path=std::getenv("BRAWL_RESIM_ATTRIB")) {
            std::ifstream in(path);
            std::string text((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
            for(auto g:HexList(text.c_str())) if(Memory::Contains(g&~31u,32)) out.granules.push_back(g&~31u);
            std::sort(out.granules.begin(),out.granules.end());
            out.granules.erase(std::unique(out.granules.begin(),out.granules.end()),out.granules.end());
            out.shadow.resize(out.granules.size()*32);
            out.active=!out.granules.empty();
            std::fprintf(stderr,"[resim-attrib] watching %zu granules\n",out.granules.size());
        }
        return out;
    }();
    return a;
}
// Charges every watched granule that changed since the previous event to `segment`, and re-reads the shadow.
void AttribEvent(uint32_t segment) {
    auto& a=Attrib();
    if(!a.active || !inAdvance || session.resimulating) return;
    for(size_t i=0;i<a.granules.size();++i) {
        const auto* live=Memory::GetPointer(a.granules[i],32);
        auto* old=a.shadow.data()+32*i;
        if(std::memcmp(live,old,32)) {++a.writes[{a.granules[i],segment,a.phase}];std::memcpy(old,live,32);}
    }
}
void AttribReset() {  // at an advance's start: the shadow is the state the advance begins from
    auto& a=Attrib();
    if(!a.active) return;
    for(size_t i=0;i<a.granules.size();++i) std::memcpy(a.shadow.data()+32*i,Memory::GetPointer(a.granules[i],32),32);
    a.last=kLoopBody;
    a.phase=kLoopBody;
}
void WriteAttrib() {
    auto& a=Attrib();
    const char* path=std::getenv("BRAWL_RESIM_ATTRIB");
    if(!a.active || !path) return;
    FILE* f=std::fopen((std::string(path)+".attrib").c_str(),"w");
    if(!f) return;
    for(const auto& [key,count]:a.writes)
        std::fprintf(f,"%08x %08x %u %08x\n",std::get<0>(key),std::get<1>(key),count,std::get<2>(key));
    std::fclose(f);
}
// BRAWL_RESIM_CALLTRACE=<n> (diagnostics): log every call LoopCall sees during the n-th real advance (target, call
// site, depth of inline frames is not tracked), as "[resim-call] phase target site".
bool CallTraceActive() {
    static const unsigned n=Env("BRAWL_RESIM_CALLTRACE",0);
    static unsigned realAdvances=0;
    static uint64_t lastAdvance=~0ull;
    if(!n || session.resimulating) return false;
    if(session.advances!=lastAdvance) {lastAdvance=session.advances;++realAdvances;}
    return realAdvances==n;
}
LoopCallAction LoopCall(CpuContext* ctx,uint32_t target) {
    if(!session.active) return LoopCallAction::Invoke;
    if(CallTraceActive()) std::fprintf(stderr,"[resim-call] %08x %08x %08x\n",Attrib().phase,target,ctx->lr-4);
    Phase(target);
    AttribEvent(Attrib().last&0x7FFFFFFFu);  // interpreted code since the previous event
    Attrib().last=target;
    if(MajorCall(target) && target!=kRender) Attrib().phase=target;
    if(hiddenAdvance && Skips().on && Listed(Skips().hidden,target)) {
        LoopCallDone(target);
        return LoopCallAction::Skipped;
    }
    if(Listed(Skips().inlined,target)) return LoopCallAction::Inline;
    if(target==kRender && (PhaseMarkers() || Attrib().active || (hiddenAdvance && Skips().on && !Skips().hidden.empty())))
        return LoopCallAction::Inline;  // the render's own calls come through here
    return LoopCallAction::Invoke;
}
void LoopCallDone(uint32_t target) {
    if(!session.active) return;
    if(PhaseLevel()==3 || (PhaseLevel()==1 && MajorCall(target)))  // back in the caller (charged as loop-body glue)
        BrawlRollbackPhaseMarker(kLoopBody|(session.resimulating?kPhaseResim:0));
    AttribEvent(target|0x80000000u);
}
bool DrivesLoop() {return BrawlOnlineCore::Enabled()?BrawlOnlineCore::DrivesLoop():session.active;}
void Run(CpuContext* ctx,GuestFunction gameProc) {gameProc(ctx);}
void RunLoop(CpuContext* ctx,GuestFunction gameProc) {
    if(BrawlOnlineCore::Enabled()) {BrawlOnlineCore::RunLoop(ctx,gameProc);return;}
    SeedMatchAtLoopTop();
    BrawlOnline::Loop(crossplayStageSeen);
    CrossplaySetup();
    if (!session.configured) Configure();
    if ((!session.live && session.input.empty()) || session.finished) {gameProc(ctx);return;}
    if (!session.active) {
        if(session.live && !crossplayStageSeen){gameProc(ctx);return;}
        const bool melee=SaveStates::SceneName()=="scMelee";
        const auto frame=Read(0x901812A4);
        if(melee && frame<10) session.sawMatchReset=true;
        if (!melee || !session.sawMatchReset || frame<session.start) {
            if(session.live && crossplayStageSeen) AnchorPads();
            gameProc(ctx);
            if(session.live && crossplayStageSeen && melee && session.trace) {
                std::fprintf(session.trace,"-1,%u,%08x,%08x,%08x,00000000,0,%08x,",
                    Read(0x901812A4),Read(0x805A00BC),Read(0x805A0424),Read(0x8059FF58),GameplayHash());
                for(auto f:FighterState()) for(auto word:f) std::fprintf(session.trace,"%08x",word);
                std::fprintf(session.trace,",%08x,00000000\n",Crc(session.current.data(),256));
                std::fflush(session.trace);
            }
            return;
        }
        WaitForResourceLoads(ctx);
        DiscoverRegions();
        if (!gekko_create(&session.gekko,session.udpPort?GekkoGameSession:GekkoStressSession)) throw std::runtime_error("rollback: Gekko creation failed");
        GekkoConfig cfg{};
        for(unsigned p=0;p<4;++p) if(session.mask&(1u<<p)) ++cfg.num_players;
        cfg.input_size=64;cfg.state_size=4;cfg.input_prediction_window=7;
        cfg.check_distance=session.distance;cfg.desync_detection=true;cfg.limited_saving=false;
        gekko_start(session.gekko,&cfg);
        if(session.udpPort) {
            gekko_net_adapter_set(session.gekko,OpenUdp(session.udpPort,session.peerPort,session.delayMs));
            gekko_set_disconnect_timeout(session.gekko,60000);
            if(!session.live) gekko_set_frame_limit(session.gekko,session.limit);
        }
        for(unsigned p=0;p<4;++p) if(session.mask&(1u<<p)) {
            const bool remote=session.udpPort && p!=session.localSlot;
            session.handles[p]=gekko_add_actor(session.gekko,remote?GekkoRemotePlayer:GekkoLocalPlayer,remote?PeerAddress():nullptr);
        }
        session.active=true;
    }
    const CpuContext entry=*ctx;
    for(unsigned p=0;p<4;++p) if(session.cursor<session.limit && session.handles[p]>=0 && (!session.udpPort || p==session.localSlot)) {
        auto* pad=session.live?latestLocalPad.data():session.input[session.cursor].data()+64*p;
        if(session.udpPort && session.cursor>0) {
            if(!gekko_add_local_input_at(session.gekko,session.handles[p],session.cursor,pad))
                throw std::runtime_error("rollback: Gekko rejected sequential local input");
        }
        else gekko_add_local_input(session.gekko,session.handles[p],pad);
    }
    int count=0;
    GekkoGameEvent** events=nullptr;
    Phase(kPhaseGekko);
    const auto waitStart=std::chrono::steady_clock::now();
    do {
        BrawlOnline::Poll();
        if(session.stopRequested) break;
        if(session.udpPort) gekko_network_poll(session.gekko);
        events=gekko_update_session(session.gekko,&count);
        if(count || !session.udpPort || (session.cursor>=session.limit && gekko_confirmed_frame(session.gekko)>=int(session.limit)-1)) break;
        if(std::chrono::steady_clock::now()-waitStart>std::chrono::seconds(60))
            throw std::runtime_error("rollback: peer did not advance within 60 seconds");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while(true);
    bool advanced=false,ended=session.stopRequested;
    unsigned advanceCount=0,advanceIndex=0;
    bool lightLastBefore=false;  // the previous advance was a light last replay (BRAWL_RESIM_LIGHT_LAST)
    for(int i=0;i<count;++i)if(events[i]->type==GekkoAdvanceEvent)++advanceCount;
    for(int i=0;i<count;++i) {
        auto& e=*events[i];
        if(e.type==GekkoLoadEvent) {Phase(kPhaseLoad);Load(e.data.load.frame,ctx);resaveDiff.stateKind=3;}
        else if(e.type==GekkoSaveEvent) {
            Phase(kPhaseSave);
            DiffResave(e.data.save.frame,resaveDiff.stateKind);
            Save(e.data.save.frame,ctx);
            *e.data.save.state_len=4;
            std::memcpy(e.data.save.state,&e.data.save.frame,4);
            *e.data.save.checksum=GameplayHash();
        } else if(e.type==GekkoAdvanceEvent) {
            if(session.live) {
                const auto sc=Read(Read(0x805A0060)+4),rule=Read(sc+0x54),info=Read(sc+0x68);
                const bool gameSet=(Memory::Contains(rule+0x74)&&Memory::Read8(rule+0x74)) ||
                    (Memory::Contains(info+0x11B)&&(Memory::Read8(info+0x11B)&0x30));
                const int previous=e.data.adv.frame-1;
                if(gameSet && (session.gameSetFrame<0 || previous<session.gameSetFrame))session.gameSetFrame=previous;
                else if(!gameSet && previous<=session.gameSetFrame)session.gameSetFrame=-1;
                if(session.gameSetFrame>=0 && !e.data.adv.rolling_back && e.data.adv.frame>=session.gameSetFrame+19){ended=true;break;}
            }
            advanced=true;
            if(session.udpPort) session.cursor=std::max(session.cursor,unsigned(e.data.adv.frame+1));
            session.current={};
            unsigned slot=0;
            for(unsigned p=0;p<4;++p) {
                if(session.handles[p]>=0) std::memcpy(session.current.data()+64*p,e.data.adv.inputs+64*slot++,64);
                else session.current[64*p+0x38]=0xff;
            }
            BrawlOnline::WaitForGo();
            session.resimulating=e.data.adv.rolling_back;
            const auto pacingStart=DetClock::PacingNanoseconds();
            const auto advanceStart=std::chrono::steady_clock::now();
            DetClock::SetRollbackResimulating(session.resimulating);
            hiddenAdvance=session.resimulating && advanceIndex+2<advanceCount;
            const bool lightLast=session.resimulating && advanceIndex+2==advanceCount && LightLast();
            const bool light=(hiddenAdvance && HideLevel()==2) || lightLast;  // models not drawn (see Light draws)
            VI_HLE_SetPresentSuppressed(session.resimulating && Skips().on);
            VI_HLE_SetKeepDisplayCopy(!session.resimulating && lightLastBefore);
            Phase(kPhaseAdvance|uint64_t(advanceIndex&0xff)<<8|uint64_t(advanceCount&0xff)<<16);
            inAdvance=true;AttribReset();
            g3dn::SetDrawSuppressed(light);
            {
                ResimHide hide(hiddenAdvance && HideLevel()==1);
                AnchorPads();gameProc(ctx);
                WaitForResourceLoads(ctx);
            }
            g3dn::SetDrawSuppressed(false);
            lightLastBefore=lightLast;
            VI_HLE_SetKeepDisplayCopy(false);
            AttribEvent(Attrib().last&0x7FFFFFFFu);inAdvance=false;
            resaveDiff.stateKind=!session.resimulating?0:advanceIndex+2<advanceCount?2:1;
            PageHashRecord(e.data.adv.frame,resaveDiff.stateKind);
            ++advanceIndex;++session.advances;
            DetClock::SetRollbackResimulating(false);
            VI_HLE_SetPresentSuppressed(false);
            hiddenAdvance=false;
            session.resimulating=false;
            Phase(kPhaseAdvanceEnd);
            const auto advanceEnd=std::chrono::steady_clock::now();
            std::fprintf(session.trace,"%d,%u,%08x,%08x,%08x,%08x,%d,%08x,",e.data.adv.frame,
                Read(0x901812A4),Read(0x805A00BC),Read(0x805A0424),Read(0x8059FF58),session.regionHash?Hash():0,e.data.adv.rolling_back,GameplayHash());
            for(auto f:FighterState()) for(auto word:f) std::fprintf(session.trace,"%08x",word);
            std::fprintf(session.trace,",%08x,%08x,%llu,%llu,%llu\n",Crc(session.current.data(),256),session.live?0:Crc(session.input.at(e.data.adv.frame).data(),256),
                (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(advanceEnd.time_since_epoch()).count(),
                (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(advanceEnd-advanceStart).count(),
                (unsigned long long)(DetClock::PacingNanoseconds()-pacingStart));
        }
    }
    Phase(kPhaseIdle);
    auto** status=gekko_session_events(session.gekko,&count);
    for(int i=0;i<count;++i) if(status[i]->type==GekkoDesyncDetected) ++session.mismatches;
    std::fflush(session.trace);
    if(advanced && !session.udpPort) ++session.cursor;
    if(ended || (session.cursor>=session.limit && (!session.udpPort || gekko_confirmed_frame(session.gekko)>=int(session.limit)-1))) {
        WriteResaveDiff();  // before COMPLETE: the harness stops the game when it reads that line
        WriteAttrib();
        FlushTrainingProfile();
        std::fprintf(stderr,"[rollback] COMPLETE frames=%u saves=%llu loads=%llu advances=%llu desyncs=%llu region_differences=%llu\n",
            session.cursor,(unsigned long long)session.saves,(unsigned long long)session.loads,
            (unsigned long long)session.advances,(unsigned long long)session.mismatches,(unsigned long long)session.regionDifferences);
        std::fprintf(stderr,"[snapshot] mode=%s delta_saves=%llu mean_payload=%llu peak_resident=%zu save_ns=%llu load_ns=%llu verify=%d\n",
            session.fullSnapshots?"full":session.trackWrites?"dirty-delta":"scan-delta",(unsigned long long)session.snapshotDeltaSaves,
            (unsigned long long)(session.snapshotDeltaSaves?session.snapshotPayload/session.snapshotDeltaSaves:0),
            session.incremental.peakResident,(unsigned long long)session.snapshotSaveNs,(unsigned long long)session.snapshotLoadNs,session.verifySnapshots);
        if(session.snapshotMetrics){std::fclose(session.snapshotMetrics);session.snapshotMetrics=nullptr;}
        std::fclose(session.trace);session.trace=nullptr;session.finished=true;session.active=false;
        if(session.udpPort) {
            gekko_send_local_inputs(session.gekko);
            const auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(session.delayMs+100);
            while(std::chrono::steady_clock::now()<until) {
                gekko_network_poll(session.gekko);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        gekko_destroy(&session.gekko);
        CloseUdp();
    }
}
} // namespace BrawlRollback
