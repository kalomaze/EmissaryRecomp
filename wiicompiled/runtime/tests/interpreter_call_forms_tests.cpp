#include "abi_bridge.h"
#include "ppc_interpreter.h"
#include "memory_access.h"
#include "isa/ppc_isa_fpenv.h"
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include "brawl_dirty_memory.h"
#include "brawl_write_watch.h"
#include "brawl_snapshot_store.h"
#include <iostream>

// Single loads preserve subnormal values and NaN payloads even in NI mode.
// Olimar's script getter uses lfsx to load 0x0040000b; flushing it changes
// script conditions and consumes an extra RNG value.
static void TestSingleLoads() {
    const uint32_t savedControl = MkwGetHostFpControl();
    struct Case { uint32_t single; uint64_t widened; };
    const Case cases[] = {
        {0x00000001, 0x36a0000000000000ULL},
        {0x0040000b, 0x38000002c0000000ULL},
        {0x8040000b, 0xb8000002c0000000ULL},
        {0x00800000, 0x3810000000000000ULL},
        {0x80000000, 0x8000000000000000ULL},
        {0x7f800000, 0x7ff0000000000000ULL},
        {0x7f800001, 0x7ff0000020000000ULL},
    };
    Memory::Write32(0x80004000, (48u << 26) | (2u << 21) | (3u << 16)); // lfs f2,0(r3)
    Memory::Write32(0x80004004, (31u << 26) | (4u << 21) | (3u << 16) | (535u << 1)); // lfsx f4,r3,r0
    Memory::Write32(0x80004008, 0x4e800020); // blr
    for (uint32_t fpscr : {0u, 4u}) {
        for (const auto& c : cases) {
            Memory::Write32(0x80005000, c.single);
            MkwApplyHostNiMode(fpscr);
            assert(PpcBitCastToU64Inline(MemoryInline::FlatReadFloat32(0x80005000)) == c.widened);
            auto* host = MemoryInline::ResolveRangeHost(0x80005000, 0, 4, false);
            assert(PpcBitCastToU64Inline(MemoryInline::ReadResolvedFloat32(host, 0, 0x80005000)) == c.widened);
            CpuContext ctx{};
            ctx.fpscr = fpscr;
            ctx.gpr[3] = 0x80005000;
            ctx.lr = 0x80006000;
            PpcInterpreter::Run(&ctx, 0x80004000);
            assert(PpcBitCastToU64Inline(ctx.fpr[2].d) == c.widened);
            assert(PpcBitCastToU64Inline(ctx.fpr[4].d) == c.widened);
        }
    }
    MkwRestoreHostMxcsr(savedControl);
}

// P+'s jump patch holds scalar f3 across a posture-scale call, then uses it
// as the multiplier in ps_muls0. Exercise the actual interpreter/registry
// boundary, including a paired value and the scalar f1 return.
static void ReturnScale(CpuContext* ctx) { ctx->fpr[1].d = 0.5; }
// Discontiguous followed branches need independent code checks. Exercise both
// the common inline checks and every overflow check, with edits in each segment
// and epoch/consume invalidation after the cache has been warmed.
static void TestDiscontiguousCodeChecks() {
    constexpr uint32_t base = 0x80008000, stride = 0x400;
    const auto branch = [](uint32_t from, uint32_t to) { return 0x48000000u | ((to - from) & 0x03FFFFFCu); };
    for (unsigned segments : {1u, 2u, 3u, 6u}) {
        for (bool reverse : {false, true}) {
            uint32_t addresses[6];
            for (unsigned i = 0; i < segments; ++i)
                addresses[i] = base + (reverse ? segments - 1 - i : i) * stride;
            for (unsigned i = 0; i < segments; ++i) {
                Memory::Write32(addresses[i], 0x38630001); // addi r3,r3,1
                const uint32_t next = i + 1 < segments ? branch(addresses[i] + 4, addresses[i + 1]) : 0x4e800020;
                Memory::Write32(addresses[i] + 4, next); // next segment, or blr
            }
            const auto run = [&](uint32_t expected) {
                CpuContext ctx{};
                ctx.lr = 0x80006000;
                PpcInterpreter::Run(&ctx, addresses[0]);
                assert(ctx.gpr[3] == expected);
            };
            for (unsigned i = 0; i < 3; ++i) run(segments);
            for (unsigned i = 0; i < segments; ++i) {
                Memory::Write32(addresses[i], 0x38630012); // addi r3,r3,18
                for (unsigned repeat = 0; repeat < 3; ++repeat) run(segments + 17 * (i + 1));
            }
            BrawlDirtyMemory::MarksCleared();
            run(18 * segments);
            BrawlDirtyMemory::MarksConsuming();
            run(18 * segments);
            BrawlDirtyMemory::MarksRestored(0);
            run(18 * segments);
            if (segments == 6) {
                // Rebuild enough blocks to cross arena chunks, retaining all
                // segment checks when overflow storage starts in a new chunk.
                uint32_t values[6] = {18, 18, 18, 18, 18, 18};
                uint32_t expected = 18 * segments;
                for (unsigned iteration = 0; iteration < 2048; ++iteration) {
                    const unsigned i = iteration % segments;
                    const uint32_t value = 1 + iteration % 100;
                    expected = expected - values[i] + value;
                    values[i] = value;
                    Memory::Write32(addresses[i], 0x38630000u | value);
                    run(expected);
                }
            }
        }
    }
}

// Removing followed jump dispatches must retain the PC and word for calls,
// generic Exec, compare/branch fusion and every original source-code check.
static void TestCompactedJumpSources() {
    constexpr uint32_t entry = 0x8000c000, first = 0x8000c400, second = 0x8000c800;
    constexpr uint32_t helper = 0x8000cc00, fail = 0x8000ce00, alternate = 0x8000cf00;
    const auto branch = [](uint32_t from, uint32_t to, bool link = false) {
        return 0x48000000u | ((to - from) & 0x03fffffcu) | uint32_t(link);
    };
    Memory::Write32(entry, 0x7d4802a6); // mflr r10
    Memory::Write32(entry + 4, branch(entry + 4, first));
    Memory::Write32(first, 0x38630001); // addi r3,r3,1
    Memory::Write32(first + 4, branch(first + 4, second));
    Memory::Write32(second, 0x30a60001); // addic r5,r6,1 (Generic)
    Memory::Write32(second + 4, branch(second + 4, helper, true));
    Memory::Write32(second + 8, 0x2c030003); // cmpwi r3,3 (fused with bne)
    Memory::Write32(second + 12, 0x40820000u | ((fail - second - 12) & 0xfffcu));
    Memory::Write32(second + 16, 0x3880000b); // li r4,11
    Memory::Write32(second + 20, 0x7d4803a6); // mtlr r10
    Memory::Write32(second + 24, 0x4e800020);
    Memory::Write32(fail, 0x38800016); // li r4,22
    Memory::Write32(fail + 4, 0x7d4803a6);
    Memory::Write32(fail + 8, 0x4e800020);
    Memory::Write32(helper, 0x38630002); // addi r3,r3,2
    Memory::Write32(helper + 4, 0x4e800020);
    Memory::Write32(alternate, 0x38630004); // addi r3,r3,4
    Memory::Write32(alternate + 4, branch(alternate + 4, second));
    const auto run = [&](uint32_t start, uint32_t increment, uint32_t immediate = 1) {
        for (uint32_t input : {0u, 7u}) for (uint32_t operand : {0u, 0xffffffffu}) {
            CpuContext ctx{};
            ctx.lr = 0x80006000;
            ctx.gpr[3] = input;
            ctx.gpr[6] = operand;
            PpcSetCr(&ctx, 0x5a123456);
            ctx.xer = 0x80000000;
            PpcInterpreter::Run(&ctx, start);
            const uint32_t result = input + increment;
            assert(ctx.gpr[3] == result);
            assert(ctx.gpr[4] == (result == 3 ? 11u : 22u));
            assert(ctx.gpr[5] == operand + immediate);
            const uint32_t carry = operand == 0xffffffffu ? 0x20000000u : 0u;
            assert(ctx.xer == (0x80000000u | carry));
            const uint32_t nibble = (result < 3 ? 8u : result > 3 ? 4u : 2u) | 1u;
            assert(PpcGetCr(&ctx) == (0x0a123456u | (nibble << 28)));
            assert(ctx.lr == 0x80006000);
        }
    };
    for (unsigned repeat = 0; repeat < 3; ++repeat) run(entry, 3);
    // Mutate only an omitted jump, then mutate a generic word after two such jumps.
    Memory::Write32(entry + 4, branch(entry + 4, alternate));
    run(entry, 6);
    Memory::Write32(entry + 4, branch(entry + 4, first));
    Memory::Write32(second, 0x30a60003); // addic r5,r6,3
    run(entry, 3, 3);
    BrawlDirtyMemory::MarksCleared();
    run(entry, 3, 3);

    // A followed jump in the last source slot leaves only End in its target
    // segment; execution must resume there with the original LR intact.
    constexpr uint32_t boundary = 0x8000d000;
    for (unsigned i = 0; i < 31; ++i) Memory::Write32(boundary + 4*i, 0x60000000); // nop
    Memory::Write32(boundary + 124, branch(boundary + 124, helper));
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        CpuContext ctx{};
        ctx.lr = 0x80006000;
        PpcInterpreter::Run(&ctx, boundary);
        assert(ctx.gpr[3] == 2 && ctx.lr == 0x80006000);
    }
    // The callee edits a later op in the calling block. AfterCall must restart
    // at the original return PC, not an index in the compacted instruction list.
    Memory::Write32(helper + 4, 0x91090000); // stw r8,0(r9)
    Memory::Write32(helper + 8, 0x4e800020);
    CpuContext ctx{};
    ctx.lr = 0x80006000;
    ctx.gpr[8] = 0x2c030004; // cmpwi r3,4: result 3 now takes the bne
    ctx.gpr[9] = second + 8;
    PpcInterpreter::Run(&ctx, entry);
    assert(ctx.gpr[3] == 3 && ctx.gpr[4] == 22 && ctx.lr == 0x80006000);
    assert(Memory::Read32(second + 8) == 0x2c030004);
}


// Compare this deterministic state digest in separate cached/classic/verify
// processes. Verify mode alone deliberately unfuses operations, so cannot test
// the combined handlers. Repeated writes change second/third source slots in
// already cached blocks; register choices include dependencies and aliases.
static void TestSequencePairs() {
    constexpr uint32_t code = 0x8000b000, data = 0x8000c000;
    uint32_t rng = 0x1289ab73;
    const auto random = [&]() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; };
    const auto d = [](uint32_t op, uint32_t dest, uint32_t base, uint32_t imm) {
        return (op << 26) | (dest << 21) | (base << 16) | (imm & 65535);
    };
    uint64_t digest = 14695981039346656037ULL;
    const auto hash = [&](uint64_t value) {
        for (unsigned i = 0; i < 8; ++i) { digest ^= uint8_t(value >> (i * 8)); digest *= 1099511628211ULL; }
    };
    unsigned runs = 0, taken = 0, fallthrough = 0;
    for (unsigned kind = 0; kind < 19; ++kind) for (unsigned sample = 0; sample < 512; ++sample) {
        CpuContext initial{};
        for (auto& r : initial.gpr) r = data + 0x80 + (random() & 0x3c);
        for (auto& f : initial.fpr) f.raw = (uint64_t(random()) << 32) | random();
        for (auto& q : initial.gqr) q = random();
        PpcSetCr(&initial, random()); initial.xer = random(); initial.ctr = random();
        initial.fpscr = sample & 4; initial.lr = initial.gpr[30] = 0x80006000;
        initial.hid0 = random(); initial.hid1 = random(); initial.hid2 = random();
        initial.srr0 = random(); initial.srr1 = random(); initial.msr = random();
        const uint32_t a = 1 + random() % 26, b = 1 + random() % 26;
        const uint32_t c = sample % 3 ? a : 1 + random() % 26;
        const uint32_t e = sample % 4 ? b : 1 + random() % 26;
        const uint32_t off = (random() & 7) * 4 - 16, imm = random() & 65535;
        uint32_t words[12]{}; unsigned n = 2;
        switch (kind) {
        case 0: words[0]=d(32,a,b,off); words[1]=d(32,c,e,off); break;
        case 1: words[0]=d(14,a,b,imm); words[1]=d(14,c,e,imm ^ 0xa5a5); break;
        case 2: words[0]=d(32,a,b,off); words[1]=d(36,c,e,off); break;
        case 3: words[0]=d(36,a,b,off); words[1]=d(32,c,e,off); break;
        case 4: words[0]=d(36,a,b,off); words[1]=d(36,c,e,off); break;
        case 5: words[0]=d(14,a,0,imm); words[1]=d(38,c,28,off); break;
        case 6: words[0]=(31u<<26)|(a<<21)|(b<<16)|(c<<11)|(444<<1);
                words[1]=(31u<<26)|(c<<21)|(e<<16)|(a<<11)|(444<<1); break;
        case 7: words[0]=d(32,a,b,off); words[1]=0x7c0903a6|(c<<21); break;
        case 8: words[0]=d(32,a,b,off); words[1]=0x7c0803a6|(c<<21); break;
        case 9: words[0]=0x7c0803a6|(a<<21); words[1]=d(14,c,e,imm); break;
        case 10: words[0]=d(40,a,b,off); words[1]=d(24,c,e,imm); break;
        case 11: words[0]=d(38,a,b,off); words[1]=d(40,c,e,off); break;
        case 12: words[0]=d(32,a,b,off); words[1]=d(14,c,e,imm); break;
        case 13: words[0]=d(14,a,b,imm); words[1]=d(11,0,c,imm); break;
        case 14: words[0]=d(32,a,b,off); words[1]=d(11,0,c,imm); break;
        case 15: words[0]=d(34,a,b,off); words[1]=d(28,c,e,sample & 1 ? 0 : imm); break;
        case 16: words[0]=d(34,a,b,off);
                 words[1]=(21u<<26)|(c<<21)|(e<<16)|((random()&31)<<11)|((random()&31)<<6)|((random()&31)<<1)|1; break;
        case 17: words[0]=d(28,a,b,sample & 1 ? 0 : imm); n=1; break;
        case 18: words[0]=d(29,a,b,sample & 1 ? 0 : imm); n=1; break;
        }
        const bool branches = kind >= 13;
        if (branches) {
            words[n++] = sample & 2 ? 0x4182000c : 0x4082000c; // beq/bne CR0,+12
            words[n++] = d(14,27,0,0x1111);
            words[n++] = 0x48000008;
            words[n++] = d(14,27,0,0x2222);
        }
        words[n++]=0x7c0802a6|(29u<<21); // capture intermediate LR
        words[n++]=0x7c0803a6|(30u<<21); // restore return address
        words[n++]=0x4e800020;
        for (unsigned i=0;i<n;++i) Memory::Write32(code+4*i,words[i]);
        for (unsigned repeat=0;repeat<2;++repeat) {
            for (unsigned i=0;i<0x200;i+=4) Memory::Write32(data+i,data+0x80+(random()&0x3c));
#if MKW_LEAN_MEMORY
            // Real mark reset, followed by the invalidation notification.
            std::memset(BrawlDirtyMemory::leanBlocks+(data&0x3fffffff)/64,0,9);
#endif
            BrawlDirtyMemory::MarksCleared();
            CpuContext ctx=initial;
            PpcInterpreter::Run(&ctx,code);
            for(auto r:ctx.gpr) hash(r);
            for(auto f:ctx.fpr) hash(f.raw);
            for(auto q:ctx.gqr) hash(q);
            for(auto r:{PpcGetCr(&ctx),ctx.xer,ctx.lr,ctx.ctr,ctx.fpscr,ctx.hid0,ctx.hid1,ctx.hid2,ctx.srr0,ctx.srr1,ctx.msr}) hash(r);
            // pc is an internal cursor and differs between cached/classic execution.
            for(unsigned i=0;i<0x200;++i) hash(Memory::Read8(data+i));
#if MKW_LEAN_MEMORY
            for(unsigned i=0;i<9;++i) hash(BrawlDirtyMemory::leanBlocks[(data&0x3fffffff)/64+i]);
#endif
            if(branches) { assert(ctx.gpr[27]==0x1111 || ctx.gpr[27]==0x2222); taken+=ctx.gpr[27]==0x2222; fallthrough+=ctx.gpr[27]==0x1111; }
            ++runs;
        }
    }
    // Edits confined to the second/third word must invalidate a warm fused block.
    const auto runEdited = [&](uint32_t expected3, uint32_t expected5, uint32_t expected27) {
        CpuContext ctx{}; ctx.lr=0x80006000; ctx.gpr[4]=7;
        PpcInterpreter::Run(&ctx,code);
        assert(ctx.gpr[3]==expected3 && ctx.gpr[5]==expected5 && ctx.gpr[27]==expected27);
    };
    Memory::Write32(code,d(14,3,4,1));
    Memory::Write32(code+4,d(14,5,3,2));
    Memory::Write32(code+8,0x4e800020);
    runEdited(8,10,0); runEdited(8,10,0);
    Memory::Write32(code+4,d(14,5,3,0xfffe));
    runEdited(8,6,0);
    // Enter the second source slot independently of its fused predecessor.
    { CpuContext ctx{}; ctx.lr=0x80006000; ctx.gpr[3]=100;
      PpcInterpreter::Run(&ctx,code+4); assert(ctx.gpr[5]==98); }
    Memory::Write32(code+4,d(11,0,3,8));
    Memory::Write32(code+8,0x4182000c);
    Memory::Write32(code+12,d(14,27,0,0x1111));
    Memory::Write32(code+16,0x48000008);
    Memory::Write32(code+20,d(14,27,0,0x2222));
    Memory::Write32(code+24,0x4e800020);
    runEdited(8,0,0x2222); runEdited(8,0,0x2222);
    Memory::Write32(code+8,0x4082000c);
    runEdited(8,0,0x1111);
    Memory::Write32(code+4,d(11,0,3,9));
    runEdited(8,0,0x2222);
    assert(taken && fallthrough);
    std::cout << "sequence differential: runs=" << runs << " taken=" << taken << " fallthrough=" << fallthrough
              << " hash=" << std::hex << digest << std::dec << "\n";
}

// Exercise the online adapter's consume/save/restore ordering against real cached code.
static void TestSnapshotCodeRestoration() {
    constexpr uint32_t code = 0x80008000, bytes = 4096;
    BrawlRollback::WriteWatch watch;
    auto* host = Memory::GetPointer(code, bytes);
    watch.Add(host, bytes, 0, code);
    BrawlRollback::SnapshotStore store;
    const auto read = [&](size_t at, uint8_t* out, size_t n) { std::memcpy(out, host + at, n); };
    const auto write = [&](size_t at, const uint8_t* in, size_t n) { std::memcpy(host + at, in, n); };
    const auto run = [&](uint32_t expected) {
        CpuContext ctx{}; ctx.lr = 0x80006000;
        PpcInterpreter::Run(&ctx, code);
        assert(ctx.gpr[3] == expected);
    };
    Memory::Write32(code, 0x38600001); // li r3,1
    Memory::Write32(code + 4, 0x4e800020);
    store.Save(0, bytes, read);
    watch.Enable();
    const auto baseSeq = BrawlDirtyMemory::consumeSeq.load();
    run(1); run(1);
    Memory::Write32(code, 0x38600012); // li r3,18
    const auto& changed = watch.Capture(bytes, store.Jobs());
    store.Save(1, bytes, read, &changed, 0);
    run(18); run(18);
    BrawlDirtyMemory::MarksRestored(baseSeq);
    const auto& pending = watch.Capture(bytes, store.Jobs());
    store.Restore(0, 1, pending, read, write);
    run(1); run(1);
    Memory::Write32(code, 0x3860001d); // unsaved change, already executed
    run(29);
    BrawlDirtyMemory::MarksRestored(baseSeq);
    const auto& unsaved = watch.Capture(bytes, store.Jobs());
    store.Restore(0, 0, unsaved, read, write);
    run(1); run(1);
}

// Rollback's consume over code spread across tracked pages (two 64 KiB pages a WriteWatch covers, several lines of the mark map
// each) and an untracked page: every block whose code a store changed runs the new code after the consume cleared its marks,
// also when the block was compiled long before and its neighbours under the same line were not written.
static void TestConsumeAcrossPages() {
    constexpr uint32_t tracked = 0x80020000, bytes = 0x20000, untracked = 0x80040000;
    BrawlRollback::WriteWatch watch;
    auto* host = Memory::GetPointer(tracked, bytes);
    watch.Add(host, bytes, 0, tracked);
    BrawlRollback::SnapshotStore store;
    const auto read = [&](size_t at, uint8_t* out, size_t n) { std::memcpy(out, host + at, n); };
    std::vector<uint32_t> sites;
    for (uint32_t off = 0; off < bytes; off += 0x1340) sites.push_back(tracked + off);  // lines and pages apart, some shared
    for (uint32_t off = 0; off < 0x4000; off += 0x1100) sites.push_back(untracked + off);
    for (uint32_t i = 0; i < sites.size(); ++i) {
        Memory::Write32(sites[i], 0x38600000u | (i + 1));  // li r3,i+1
        Memory::Write32(sites[i] + 4, 0x4e800020);
    }
    const auto run = [&](uint32_t at, uint32_t expected) {
        for (int k = 0; k < 3; ++k) {  // past the compile threshold
            CpuContext ctx{}; ctx.lr = 0x80006000;
            PpcInterpreter::Run(&ctx, at);
            assert(ctx.gpr[3] == expected);
        }
    };
    store.Save(0, bytes, read);
    watch.Enable();
    for (uint32_t i = 0; i < sites.size(); ++i) run(sites[i], i + 1);
    for (uint32_t round = 1; round <= 3; ++round)
        for (uint32_t i = round % 2; i < sites.size(); i += 2) {
            const uint32_t value = 0x100u * round + i;
            Memory::Write32(sites[i], 0x38600000u | value);
            watch.Capture(bytes, store.Jobs());  // consumes: the marks of the tracked pages are clear afterwards
            run(sites[i], value);
            if (i + 1 < sites.size()) {  // a neighbour that was not written keeps running its code
                const uint32_t w = Memory::Read32(sites[i + 1]);
                run(sites[i + 1], w & 0xFFFFu);
            }
        }
}

namespace InterpreterTest { extern unsigned frameEnds, loopEnds, soundHooks; }

static void TestOnlineHooksAndContinuation() {
    // A branch into a compiled variant must retain the variant's continuation path.
    Memory::Write32(0x8000e000, 0x4bfff004); // b 0x8000d004
    Memory::Write32(0x8000d004, 0x38630001); // would give a different result if interpreted
    Memory::Write32(0x8000d008, 0x4e800020);
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        CpuContext ctx{}; ctx.lr = 0x80006000;
        PpcInterpreter::Run(&ctx, 0x8000e000);
        assert(ctx.gpr[3] == 0x55);
    }
    if (!std::getenv("BRAWL_TEST_ONLINE_CORE")) return;
    for (uint32_t hook : {0x801C9AACu, 0x801C9B54u, 0x801C9BFCu, 0x801C9C70u, 0x801BC684u}) {
        Memory::Write32(hook - 4, 0x38630001); // addi r3,r3,1
        Memory::Write32(hook, 0x38840001); // hook skips this addi r4,r4,1
        Memory::Write32(hook + 4, 0x4e800020);
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
            CpuContext ctx{}; ctx.lr = 0x80006000;
            PpcInterpreter::Run(&ctx, hook - 4);
            assert(ctx.gpr[3] == 1 && ctx.gpr[4] == 0);
        }
    }
    assert(InterpreterTest::soundHooks == 15);
    Memory::Write32(0x800171B4, 0x48000350); // branch to frame-end hook
    Memory::Write32(0x80017504, 0x38840001); // hook must skip this instruction
    Memory::Write32(0x80017508, 0x4e800020);
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        CpuContext ctx{}; ctx.lr = 0x80006000;
        bool completed = false;
        try { PpcInterpreter::Run(&ctx, 0x800171B4); }
        catch (const std::runtime_error& error) {
            assert(std::string_view(error.what()) == "test loop completed");
            completed = true;
        }
        assert(completed && ctx.gpr[4] == 0);
    }
    assert(InterpreterTest::frameEnds == 3 && InterpreterTest::loopEnds == 3);

}

// Translated code's indirect calls of guarded records (abi_bridge.cpp's site slow path): callees that may write every
// non-volatile FPR (the fixed copy), f20-f31, f31 alone, a sparse set (the register loop), and a native (GPR guard too, the
// general dispatch). Each writes f1-f31 and r14-r31 (and throws when r3 is odd): what its guard covers comes back, the rest keeps
// what the callee wrote, also when the callee throws.
static void ClobberGuardTest(CpuContext* ctx) {
    for (int i = 1; i < 32; ++i) ctx->fpr[i].d = -1000.0 - i;
    for (int i = 14; i < 32; ++i) ctx->gpr[i] = 0xDEAD0000u + static_cast<uint32_t>(i);
    if (ctx->gpr[3] & 1u) throw std::runtime_error("guarded callee threw");
}
static void RegisterGuardTestCallee(uint32_t address, uint32_t fprMask, FunctionKind kind) {
    TranslatedFunctionInfo info{};
    info.address = address;
    info.name = "ClobberGuardTest";
    info.rawCpuInvoker = ClobberGuardTest;
    info.entryPoint = reinterpret_cast<void*>(ClobberGuardTest);
    info.nonvolatileFprWriteMask = fprMask;
    info.kind = kind;
    TranslatedFunctionRegistry::Register(std::move(info));
}
constexpr uint32_t kGuardAll = 0x80002100, kGuardRun = 0x80002200, kGuardNative = 0x80002300, kGuardF31 = 0x80002400,
                   kGuardSparse = 0x80002500;
static void TestIndirectSiteGuards() {
#if MKW_INDIRECT_IC
    struct Case { uint32_t target; uint32_t fprKept; bool gprsKept; };
    const Case cases[] = {{kGuardAll, 0xFFFFC000u, false}, {kGuardRun, 0xFFF00000u, false}, {kGuardNative, 0xFFFFC000u, true},
                          {kGuardF31, 0x80000000u, false}, {kGuardSparse, 0x50024000u, false}};
    for (const Case& c : cases)
        for (uint32_t r3 : {2u, 3u}) {
            for (int round = 0; round < 3; ++round) {  // the site's slots empty, then holding the record
                CpuContext ctx{};
                for (int i = 0; i < 32; ++i) ctx.fpr[i].d = 1.0 + i;
                for (int i = 0; i < 32; ++i) ctx.gpr[i] = 0x1000u + static_cast<uint32_t>(i);
                ctx.gpr[3] = r3;
                CpuContextScope scope(&ctx);
                bool threw = false;
                try {
                    MKW_INDIRECT_CALL(c.target, &ctx);
                } catch (const std::runtime_error&) {
                    threw = true;
                }
                assert(threw == ((r3 & 1u) != 0));
                for (int i = 1; i < 32; ++i)
                    assert(ctx.fpr[i].d == (((c.fprKept >> i) & 1u) ? 1.0 + i : -1000.0 - i));
                for (int i = 14; i < 32; ++i)
                    assert(ctx.gpr[i] == (c.gprsKept ? 0x1000u + static_cast<uint32_t>(i) : 0xDEAD0000u + static_cast<uint32_t>(i)));
            }
        }
#endif
}

int main() {
    Memory::Init(0x200000);
    RegisterGuardTestCallee(kGuardAll, 0xFFFFC000u, FunctionKind::BaseTranslated);
    RegisterGuardTestCallee(kGuardRun, 0xFFF00000u, FunctionKind::BaseTranslated);
    RegisterGuardTestCallee(kGuardNative, 0xFFFFC000u, FunctionKind::Native);
    RegisterGuardTestCallee(kGuardF31, 0x80000000u, FunctionKind::BaseTranslated);
    RegisterGuardTestCallee(kGuardSparse, 0x50024000u, FunctionKind::BaseTranslated);
    TestSingleLoads();
    TranslatedFunctionInfo callee{};
    callee.address = 0x80002000;
    callee.name = "ReturnScale";
    callee.rawCpuInvoker = ReturnScale;
    callee.entryPoint = reinterpret_cast<void*>(ReturnScale);
    callee.kind = FunctionKind::Native;
    TranslatedFunctionRegistry::Register(std::move(callee));
    const uint32_t words[] = {
        (63u<<26)|(3u<<21)|(1u<<11)|(72u<<1), // fmr f3,f1
        (4u<<26)|(6u<<21)|(2u<<16)|(2u<<11)|(21u<<1), // ps_add f6,f2,f2
        0x48000ff9, // bl 0x80002000 from 0x80001008
        (4u<<26)|(4u<<21)|(2u<<16)|(3u<<6)|(12u<<1), // ps_muls0 f4,f2,f3
        (4u<<26)|(5u<<21)|(2u<<16)|(1u<<6)|(12u<<1), // ps_muls0 f5,f2,f1
        (4u<<26)|(7u<<21)|(6u<<16)|(3u<<6)|(12u<<1), // ps_muls0 f7,f6,f3
        0x7fc803a6, // mtlr r30
        0x4e800020, // blr
    };
    for (size_t i=0;i<sizeof(words)/sizeof(*words);++i) Memory::Write32(0x80001000+4*i,words[i]);
    CpuContext ctx{};
    ctx.lr=ctx.gpr[30]=0x80003000;
    ctx.fpr[1].d=1.0;
    ctx.fpr[2].d=2.455000877380371;
    PpcInterpreter::Run(&ctx,0x80001000);
    // These destinations were written by paired instructions.
    assert(PPC_PsToScalarInline(ctx.fpr[4].d)==2.455000877380371);
    assert(PPC_PsToScalarInline(ctx.fpr[5].d)==1.2275004386901855);
    assert(PPC_PsToScalarInline(ctx.fpr[7].d)==4.910001754760742);
    assert(ctx.fpr[4].paired.ps1 == 2.455000877380371f);
    assert(ctx.fpr[5].paired.ps1 == 1.2275004386901855f);
    assert(ctx.fpr[7].paired.ps1 == 4.910001754760742f);
    TranslatedFunctionRegistry::Finalize();
    TestIndirectSiteGuards();
    TestOnlineHooksAndContinuation();
    TestSnapshotCodeRestoration();
    TestConsumeAcrossPages();
    TestDiscontiguousCodeChecks();
    TestCompactedJumpSources();
    TestSequencePairs();
    std::cout << "interpreter call forms: ok\n";
}
