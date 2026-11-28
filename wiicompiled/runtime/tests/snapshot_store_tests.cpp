#include "brawl_snapshot_store.h"
#include "brawl_write_watch.h"
#include <cassert>
#include <iostream>
#include <random>
#include <utility>

using BrawlRollback::SnapshotStore;

static void TestOnlyLaterWritesNeedRestoring() {
    SnapshotStore store;
    std::vector<uint8_t> live(2 * SnapshotStore::BlockSize + 7);
    const auto read = [&](size_t at, uint8_t* out, size_t size) { std::memcpy(out, live.data() + at, size); };
    const auto write = [&](size_t at, const uint8_t* in, size_t size) { std::memcpy(live.data() + at, in, size); };
    store.Save(0, live.size(), read);
    live[0] = 1;
    std::vector<uint32_t> dirty{0};
    store.Save(1, live.size(), read, &dirty, 0);
    const auto frame1 = live;
    live[64] = 2;
    dirty = {1};
    store.Save(2, live.size(), read, &dirty, 1);
    store.Restore(1, 2, {}, read, write);
    assert(live == frame1);
    // The target's block 0 was never touched afterwards. Reading it serves no restore purpose.
    assert(store.lastRestoreReadBytes == 64);
    store.Restore(1, 1, {}, read, write);
    assert(live == frame1 && store.lastRestoreReadBytes == 0);
    live[130] = 3;
    store.Restore(1, 1, {2}, read, write);
    assert(live == frame1 && store.lastRestoreReadBytes == 7);
}

// One block in `density` is written per step; sparse writes leave most blocks' newest copy in the base image.
static void TestHistoryAgainstFullImages(unsigned density, uint32_t seed) {
    SnapshotStore store;
    std::mt19937 random(seed);
    std::vector<uint8_t> live(257 * SnapshotStore::BlockSize - 13);
    std::vector<std::pair<int, std::vector<uint8_t>>> history{{0, live}};
    const auto read = [&](size_t at, uint8_t* out, size_t size) { std::memcpy(out, live.data() + at, size); };
    const auto write = [&](size_t at, const uint8_t* in, size_t size) { std::memcpy(live.data() + at, in, size); };
    store.Save(0, live.size(), read);
    int nextFrame = 1;
    unsigned restores = 0, evictions = 0;
    for (unsigned step = 0; step < 3000; ++step) {
        std::vector<uint32_t> dirty;
        for (uint32_t block = 0; block < 257; ++block) {
            if (random() % density != 0) continue;
            dirty.push_back(block);
            const size_t at = block * SnapshotStore::BlockSize;
            const size_t size = std::min(SnapshotStore::BlockSize, live.size() - at);
            for (size_t i = 0; i < size; ++i) live[at + i] = static_cast<uint8_t>(random());
        }
        // Sometimes prepared ahead (PrepareRestore), sometimes prepared for a save that drops it.
        const unsigned prepare = random() % 3;
        if (history.size() > 1 && random() % 7 == 0) {
            const size_t target = random() % history.size();
            if (prepare == 0) store.PrepareRestore(history[target].first, history.back().first);
            if (prepare == 1) store.PrepareRestore(history[random() % history.size()].first, history.back().first);
            store.Restore(history[target].first, history.back().first, dirty, read, write);
            assert(live == history[target].second);
            history.resize(target + 1);
            ++restores;
        } else {
            const int frame = nextFrame++;
            if (prepare == 0 && history.size() > 1) store.PrepareRestore(history[random() % history.size()].first, history.back().first);
            store.Save(frame, live.size(), read, &dirty, history.back().first);
            history.emplace_back(frame, live);
            if (history.size() > SnapshotStore::Capacity) { history.erase(history.begin()); ++evictions; }
            assert(store.Materialize(frame) == live);
        }
    }
    assert(restores > 100 && evictions > 100);
    // Traverse all retained snapshots backwards, including oldest/base fallback and repeated current loads.
    while (!history.empty()) {
        const auto expected = history.back();
        store.Restore(expected.first, expected.first, {}, read, write);
        assert(live == expected.second);
        history.pop_back();
        if (!history.empty()) {
            store.Restore(history.back().first, expected.first, {}, read, write);
            assert(live == history.back().second);
        }
    }
    std::cout << "snapshot history (1 in " << density << "): 3000 transitions, " << restores << " restores, " << evictions << " evictions\n";
}

// Model the production incremental boundaries: Enable once for the base,
// then Capture consumes marks before saves and unmarked restore writes.
static void TestConsumedMarksAcrossSnapshotBoundaries() {
#if MKW_LEAN_MEMORY
    std::vector<uint8_t> marks(BrawlDirtyMemory::kLeanDirtyBlocks + 1, 1);
    BrawlDirtyMemory::leanBlocks = marks.data();
#endif
    SnapshotStore store;
    constexpr uint32_t guest = 0x80010013u;
    std::vector<uint8_t> live(2 * 65536 + 117);
    BrawlRollback::WriteWatch watch;
    watch.Add(live.data(), live.size(), 0, guest);
    const auto read = [&](size_t at, uint8_t* out, size_t size) { std::memcpy(out, live.data() + at, size); };
    const auto write = [&](size_t at, const uint8_t* in, size_t size) { std::memcpy(live.data() + at, in, size); };
    store.Save(0, live.size(), read);
    watch.Enable();
    std::vector<std::pair<int, std::vector<uint8_t>>> history{{0, live}};
    std::mt19937 random(0x431c0a5);
    int nextFrame = 1;
    for (unsigned step = 0; step < 1200; ++step) {
        for (unsigned n = 0; n < 23; ++n) {
            const size_t at = random() % live.size();
            const size_t count = std::min(size_t(1 + random() % 97), live.size() - at);
            std::fill_n(live.data() + at, count, uint8_t(random()));
            BrawlDirtyMemory::Mark(guest + uint32_t(at), count);
        }
        const auto& dirty = watch.Capture(live.size(), store.Jobs());
        if (history.size() > 1 && random() % 3 == 0) {
            const size_t target = random() % history.size();
            store.Restore(history[target].first, history.back().first, dirty, read, write);
            assert(live == history[target].second);
            history.resize(target + 1);
        } else {
            store.Save(nextFrame, live.size(), read, &dirty, history.back().first);
            assert(store.Materialize(nextFrame) == live);
            history.emplace_back(nextFrame++, live);
            if (history.size() > SnapshotStore::Capacity) history.erase(history.begin());
        }
        // No Enable/reset after either operation: all tracked marks are clean,
        // including bytes outside the selected span in its boundary pages.
        assert(watch.Capture(live.size(), store.Jobs()).empty());
    }
    std::cout << "consumed write marks: 1200 snapshot transitions ok\n";
}

static void TestEvictedSourcesAndSlotReuse() {
    SnapshotStore store;
    std::vector<uint8_t> live(4 * SnapshotStore::BlockSize + 7);
    std::vector<std::pair<int, std::vector<uint8_t>>> history;
    const auto read = [&](size_t at, uint8_t* out, size_t size) { std::memcpy(out, live.data() + at, size); };
    const auto write = [&](size_t at, const uint8_t* in, size_t size) { std::memcpy(live.data() + at, in, size); };
    int serial = 0, anchor = 0;
    store.Save(0, live.size(), read);
    history.emplace_back(0, live);
    const auto save = [&](const std::vector<uint32_t>& dirty, bool scan = false) {
        const int frame = ++serial;
        store.Save(frame, live.size(), read, scan ? nullptr : &dirty, anchor);
        anchor = frame;
        history.emplace_back(frame, live);
        if (history.size() > SnapshotStore::Capacity) history.erase(history.begin());
    };
    const auto restore = [&](size_t index, const std::vector<uint32_t>& dirty) {
        const auto expected = history[index];
        store.Restore(expected.first, anchor, dirty, read, write);
        assert(live == expected.second);
        anchor = expected.first;
        history.resize(index + 1);
    };
    // Block zero's source is evicted while its contents stay live; stale references must use the base.
    live[0] = 17;
    save({0});
    for (int i = 0; i < 80; ++i) {
        live[64] = uint8_t(i);
        save({1});
    }
    live[0] = 99;
    restore(0, {0});
    assert(live[0] == 17);
    // Reuse discarded slots at different generations, with partial blocks and scan-mode saves.
    for (int round = 0; round < 80; ++round) {
        for (int i = 0; i < 20; ++i) {
            live[64] = uint8_t(round + i);
            live.back() = uint8_t(round ^ i);
            save({1, 4}, i % 5 == 0);
        }
        restore(history.size() - 7, {});
        live[128] ^= 0x5a;
        save({2});
        live[0] ^= 0xff;
        restore(history.size() - 1, {0});
        assert(store.Materialize(anchor) == live);
    }
    std::cout << "eviction fallback, slot reuse and scan saves: ok\n";
}

static void TestBatchCallbacksAndFailures() {
    BrawlRollback::SnapshotJobs jobs;
    std::array<unsigned,10> output{};
    bool prepared=false;
    const auto work=[&,counter=41u](size_t index) mutable {
        assert(prepared);
        output[index]=++counter; // Each chunk must own an independent mutable callback.
    };
    jobs.RunPrepared<10>([&]{prepared=true;},work);
    for(auto value:output)assert(value==42);

    output.fill(0);
    bool caught=false;
    try {
        jobs.RunPrepared<10>([]{throw 19;},[&](size_t index){++output[index];});
    } catch(int error) {assert(error==19);caught=true;}
    assert(caught);
    for(auto value:output)assert(value==0);

    caught=false;
    try {
        jobs.Run<10>([&](size_t index){
            output[index]=1;
            if(index==2 || index==7)throw int(index);
        });
    } catch(int error) {assert(error==2);caught=true;}
    assert(caught); // Failures still join every chunk, then report in chunk order.
    for(auto value:output)assert(value==1);

    std::atomic<unsigned> background{0};
    auto* pending=jobs.Start([&](job::JobTaskThread&,job::Job&){
        for(unsigned i=0;i<10000;++i)background.fetch_add(1,std::memory_order_relaxed);
    });
    for(unsigned round=0;round<1000;++round) {
        output.fill(0);
        jobs.Run<10>([&,counter=round](size_t index) mutable {output[index]=++counter;});
        for(auto value:output)assert(value==round+1);
    }
    jobs.Wait(pending);
    assert(pending==nullptr && background.load()==10000);
    std::cout << "batch callback isolation, failures and asynchronous overlap: ok\n";
}

int main() {
    TestBatchCallbacksAndFailures();
    TestEvictedSourcesAndSlotReuse();
    TestOnlyLaterWritesNeedRestoring();
    TestHistoryAgainstFullImages(9, 0x70a11bac);
    TestHistoryAgainstFullImages(61, 0x5ca77e2);
    TestConsumedMarksAcrossSnapshotBoundaries();
    std::cout << "snapshot store: ok\n";
}
