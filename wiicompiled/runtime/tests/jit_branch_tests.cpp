// The branch group of the JIT (src/jit/jit_x64_branch.cpp) against the interpreter, on the forms the general differential
// test (jit_differential_tests.cpp) does not enumerate: every BO x BI of bc and bclr with ctr at 0, 1, 2 and 0xFFFFFFFF (each CR
// bit set and clear, every field value as a table value and as a compare's difference, and raw 64-bit patterns: jit_test_cr.h); the
// fused compare+bc kinds on every pairing of compare field, bit and BO class (including a bc of another field than the compare
// wrote, and the SO bit); blr and bclr to the frame's return address (with low bits set) and to other blocks; f1 normalised on a
// return; bctr to untranslated code and to a registered function, alternating in one block; bl and bctrl to leaf code and to
// registered functions; a callee that rewrites its caller's next instruction; b to a registered function (a tail call) and to
// a continuation. Programs run under the classic interpreter (Exec, the reference) and the JIT (threshold 0 and 2); registers,
// CR field values (bytewise: the JIT stores the interpreter's exact values), FPR bits, the sandbox and the code must come out equal. A last part shows that the branch kinds are compiled natively (not
// silently fallbacks) and that hot loops stay in compiled code.
#include "jit_test_env.h"
#include "jit_test_cr.h"
#include "abi_bridge.h"
#include "brawl_dirty_memory.h"
#include "memory.h"
#include "ppc_interpreter.h"
#include "../src/jit/jit_x64.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <utility>
#include <string>
#include <utility>
#include <vector>

namespace GuestCodePatch {  // the test double (interpreter_test_support.cpp)
void TestSetInterpreted(uint32_t address, uint32_t at);
uint32_t TestInterpretedAt(uint32_t address);
unsigned TestSyncCount();
void Sync();
}  // namespace GuestCodePatch

namespace {

using PpcInterpreter::Engine;
using PpcInterpreter::detail::JitX64GetStats;
using PpcInterpreter::detail::JitX64KindOps;
using PpcInterpreter::detail::K;

constexpr uint32_t kDataBase = 0x80030000u;
constexpr uint32_t kDataSize = 0x2000u;
constexpr uint32_t kReturn = 0x80020000u;
constexpr uint32_t kFuncBase = 0x80010000u;
constexpr uint32_t kSlotBase = 0x80100000u;
constexpr uint32_t kSlotBytes = 0x2000u;
constexpr uint32_t kSlotCount = 64;
constexpr uint32_t kContinuation = 0x8000d004u;  // interpreter_test_support.cpp's GuestCodePatch::ContinuationAt answers here

unsigned g_failures = 0, g_cases = 0;
uint64_t g_runs = 0;

std::string Fmt(const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof buffer, format, args);
    va_end(args);
    return buffer;
}

// --- encoders and a small assembler ---------------------------------------------------------------------------------
namespace ppc {
constexpr uint32_t D(uint32_t op, uint32_t rt, uint32_t ra, int32_t imm) {
    return (op << 26) | (rt << 21) | (ra << 16) | (static_cast<uint32_t>(imm) & 0xFFFFu);
}
constexpr uint32_t X(uint32_t rt, uint32_t ra, uint32_t rb, uint32_t xo, uint32_t rc = 0) {
    return (31u << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (xo << 1) | rc;
}
constexpr uint32_t addi(uint32_t rt, uint32_t ra, int32_t i) { return D(14, rt, ra, i); }
constexpr uint32_t li(uint32_t rt, int32_t i) { return addi(rt, 0, i); }
constexpr uint32_t lis(uint32_t rt, int32_t i) { return D(15, rt, 0, i); }
constexpr uint32_t ori(uint32_t ra, uint32_t rs, uint32_t u) { return D(24, rs, ra, static_cast<int32_t>(u)); }
constexpr uint32_t cmpwi(uint32_t crf, uint32_t ra, int32_t i) { return D(11, crf << 2, ra, i); }
constexpr uint32_t cmplwi(uint32_t crf, uint32_t ra, uint32_t u) { return D(10, crf << 2, ra, static_cast<int32_t>(u)); }
constexpr uint32_t cmpw(uint32_t crf, uint32_t ra, uint32_t rb) { return X(crf << 2, ra, rb, 0); }
constexpr uint32_t cmplw(uint32_t crf, uint32_t ra, uint32_t rb) { return X(crf << 2, ra, rb, 32); }
constexpr uint32_t add(uint32_t rt, uint32_t ra, uint32_t rb) { return X(rt, ra, rb, 266); }
constexpr uint32_t rlwinm(uint32_t ra, uint32_t rs, uint32_t sh, uint32_t mb, uint32_t me, uint32_t rc = 0) {
    return (21u << 26) | (rs << 21) | (ra << 16) | (sh << 11) | (mb << 6) | (me << 1) | rc;
}
constexpr uint32_t mfspr(uint32_t rt, uint32_t spr) { return X(rt, spr & 31u, spr >> 5, 339); }
constexpr uint32_t mtspr(uint32_t spr, uint32_t rs) { return X(rs, spr & 31u, spr >> 5, 467); }
constexpr uint32_t mflr(uint32_t rt) { return mfspr(rt, 8); }
constexpr uint32_t mtlr(uint32_t rs) { return mtspr(8, rs); }
constexpr uint32_t mtctr(uint32_t rs) { return mtspr(9, rs); }
constexpr uint32_t bclr(uint32_t bo, uint32_t bi, uint32_t lk = 0) { return (19u << 26) | (bo << 21) | (bi << 16) | (16u << 1) | lk; }
constexpr uint32_t bcctr(uint32_t bo, uint32_t bi, uint32_t lk = 0) { return (19u << 26) | (bo << 21) | (bi << 16) | (528u << 1) | lk; }
constexpr uint32_t blr() { return bclr(20, 0); }
constexpr uint32_t bctr() { return bcctr(20, 0); }
constexpr uint32_t bctrl() { return bcctr(20, 0, 1); }
constexpr uint32_t nop() { return ori(0, 0, 0); }
constexpr uint32_t lwz(uint32_t rt, int32_t d, uint32_t ra) { return D(32, rt, ra, d); }
constexpr uint32_t stw(uint32_t rs, int32_t d, uint32_t ra) { return D(36, rs, ra, d); }
constexpr uint32_t fx(uint32_t op, uint32_t d, uint32_t a, uint32_t b, uint32_t xo10, uint32_t rc = 0) {
    return (op << 26) | (d << 21) | (a << 16) | (b << 11) | (xo10 << 1) | rc;
}
constexpr uint32_t ps_merge10(uint32_t d, uint32_t a, uint32_t b) { return fx(4, d, a, b, 592); }
}  // namespace ppc

struct Label {
    int id = -1;
};

struct Image {
    struct Reloc {
        enum Kind { BAbs, LabelAddr } kind;
        size_t index;
        uint32_t value;
        bool link;
    };
    std::vector<uint32_t> words;
    std::vector<Reloc> relocs;
    std::vector<uint32_t> Resolve(uint32_t base) const {
        std::vector<uint32_t> out = words;
        for (const Reloc& r : relocs) {
            if (r.kind == Reloc::BAbs) {
                const uint32_t pc = base + 4u * static_cast<uint32_t>(r.index);
                out[r.index] = 0x48000000u | ((r.value - pc) & 0x03FFFFFCu) | (r.link ? 1u : 0u);
            } else {
                const uint32_t address = base + 4u * r.value;
                out[r.index] |= address >> 16;
                out[r.index + 1] |= address & 0xFFFFu;
            }
        }
        return out;
    }
};

class Asm {
public:
    Label NewLabel() {
        labels_.push_back(-1);
        return Label{static_cast<int>(labels_.size()) - 1};
    }
    void Bind(Label l) { labels_[static_cast<size_t>(l.id)] = static_cast<int>(words_.size()); }
    size_t Here() const { return words_.size(); }
    Asm& operator<<(uint32_t word) {
        words_.push_back(word);
        return *this;
    }
    void Org(size_t index) {
        while (words_.size() < index) words_.push_back(0);
    }
    void B(Label l, bool link = false) {
        fixes_.push_back({true, words_.size(), l.id});
        words_.push_back(0x48000000u | (link ? 1u : 0u));
    }
    void Bc(uint32_t bo, uint32_t bi, Label l, bool link = false) {
        fixes_.push_back({false, words_.size(), l.id});
        words_.push_back((16u << 26) | (bo << 21) | (bi << 16) | (link ? 1u : 0u));
    }
    void BAbs(uint32_t target, bool link = false) {
        relocs_.push_back({Image::Reloc::BAbs, words_.size(), target, link});
        words_.push_back(0);
    }
    Asm& LiLabel(uint32_t rt, Label l) {
        relocs_.push_back({Image::Reloc::LabelAddr, words_.size(), static_cast<uint32_t>(l.id), false});
        words_.push_back(ppc::lis(rt, 0));
        words_.push_back(ppc::ori(rt, rt, 0));
        return *this;
    }
    Asm& Li32(uint32_t rt, uint32_t value) {
        words_.push_back(ppc::lis(rt, static_cast<int16_t>(value >> 16)));
        words_.push_back(ppc::ori(rt, rt, value & 0xFFFFu));
        return *this;
    }
    Image Finish() {
        Image image;
        image.words = words_;
        for (const Fix& f : fixes_) {
            const int32_t to = labels_[static_cast<size_t>(f.label)];
            if (to < 0) Fail("unbound label");
            const int32_t bytes = 4 * (to - static_cast<int32_t>(f.index));
            if (f.b) image.words[f.index] |= static_cast<uint32_t>(bytes) & 0x03FFFFFCu;
            else image.words[f.index] |= static_cast<uint32_t>(bytes) & 0xFFFCu;
        }
        image.relocs = relocs_;
        for (Image::Reloc& r : image.relocs) {
            if (r.kind != Image::Reloc::LabelAddr) continue;
            if (labels_[r.value] < 0) Fail("unbound label");
            r.value = static_cast<uint32_t>(labels_[r.value]);
        }
        if (image.words.size() * 4 > kSlotBytes) Fail("program larger than a slot");
        return image;
    }

private:
    struct Fix {
        bool b;
        size_t index;
        int label;
    };
    [[noreturn]] static void Fail(const char* what) {
        std::fprintf(stderr, "jit branch tests: assembler: %s\n", what);
        std::exit(2);
    }
    std::vector<uint32_t> words_;
    std::vector<int> labels_;
    std::vector<Fix> fixes_;
    std::vector<Image::Reloc> relocs_;
};

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint32_t Next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return static_cast<uint32_t>((s * 0x2545F4914F6CDD1Dull) >> 32);
    }
    uint32_t Below(uint32_t n) { return Next() % n; }
};

constexpr uint32_t kEdges[] = {0u, 1u, 2u, 3u, 5u, 6u, 7u, 8u, 0x7FFFFFFFu, 0x80000000u, 0x80000001u, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFDu, 0x1234u};
uint32_t Edge(Rng& r) { return kEdges[r.Below(sizeof kEdges / sizeof *kEdges)]; }

CpuContext BaseState(Rng& r) {
    CpuContext c{};
    for (auto& g : c.gpr) g = r.Below(3) == 0 ? r.Next() : Edge(r);
    c.gpr[1] = kDataBase + 0x1800u;
    c.gpr[29] = kDataBase + 0x800u;
    c.gpr[30] = kDataBase + 0x1000u;
    JitTestCr::Randomize(c, [&] { return r.Next(); });
    c.xer = (r.Below(4) != 0 ? 0u : 0x80000000u) | (r.Next() & 0x6000007Fu);  // SO set: compiled code does not run (JitBlocksMayRun)
    c.ctr = Edge(r);
    c.lr = kReturn;
    for (auto& f : c.fpr) f.raw = (static_cast<uint64_t>(r.Next()) << 32) | r.Next();
    c.fpr[1].d = 1.5;
    return c;
}

// --- registered functions ---------------------------------------------------------------------------------------------
void FnModify(CpuContext* c) {
    c->gpr[3] = c->gpr[3] * 3u + c->gpr[4];
    c->gpr[5] ^= 0x13579BDFu;
    c->gpr[0] = 0x70u;
    c->gpr[12] = c->lr;
    c->ctr += 5u;
    c->crv[0] = PpcCrvFromNibble(c->gpr[3] >> 28);
    c->fpr[1].d = c->fpr[1].d * 2.0 + 1.0;
}
void FnWrite(CpuContext* c) {  // writes r4 to the guest address in r3, as a guest store does
    Memory::Write32(c->gpr[3], c->gpr[4]);
    c->gpr[7] = c->gpr[3] + 4u;
}
constexpr uint32_t kFnModify = kFuncBase + 0x00, kFnWrite = kFuncBase + 0x10;

void RegisterFunction(uint32_t address, const char* name, void (*fn)(CpuContext*)) {
    TranslatedFunctionInfo info{};
    info.address = address;
    info.name = name;
    info.rawCpuInvoker = fn;
    info.entryPoint = reinterpret_cast<void*>(fn);
    info.kind = FunctionKind::Native;
    TranslatedFunctionRegistry::Register(std::move(info));
}

// --- running and comparing ----------------------------------------------------------------------------------------------
struct Snap {
    CpuContext ctx{};
    bool threw = false;
    unsigned syncs = 0;  // GuestCodePatch::Sync calls (the test double's count) during the run
    std::vector<uint8_t> data, code;
};

struct EngineConfig {
    const char* name;
    Engine engine;
    uint32_t threshold;
};
const EngineConfig kReference = {"classic", Engine::Classic, 0};
const EngineConfig kConfigs[] = {{"jit-0", Engine::Jit, 0}, {"jit-2", Engine::Jit, 2}};

// What one run of a scenario is: a context and the words to write (guest stores, so the code is invalidated) before it.
struct RunSpec {
    CpuContext ctx;
    std::vector<std::pair<uint32_t, uint32_t>> writes;  // (offset from the program's base, word)
    uint32_t entryOffset = 0;
    std::function<void()> before;  // run before the program (the patch double's state)
};
struct Scenario {
    Image image;
    std::function<void(uint32_t base, std::vector<RunSpec>& runs)> runs;
};

std::vector<Snap> Execute(const EngineConfig& config, const Scenario& s, uint32_t base) {
    std::vector<RunSpec> runs;
    s.runs(base, runs);
    const std::vector<uint32_t> words = s.image.Resolve(base);
    for (size_t i = 0; i < words.size(); ++i) Memory::Write32(base + 4u * static_cast<uint32_t>(i), words[i]);
    PpcInterpreter::SetEngineOverride(config.engine);
    PpcInterpreter::SetJitThreshold(config.threshold);
    std::vector<Snap> out;
    for (const RunSpec& spec : runs) {
        for (uint32_t off = 0; off < kDataSize; off += 4) Memory::Write32(kDataBase + off, 0x9E3779B1u * (off + 1));
        for (const auto& [offset, word] : spec.writes) Memory::Write32(base + offset, word);
        Snap snap;
        CpuContext ctx = spec.ctx;
        ctx.pc = 0;
        if (spec.before) spec.before();
        const unsigned syncsBefore = GuestCodePatch::TestSyncCount();
        {
            CpuContextScope scope(&ctx);
            try {
                // Alternately the entry RunPatched makes (the lean one, which declines what it cannot take) and Run itself
                if ((g_runs & 1u) == 0 || !PpcInterpreter::detail::RunCompiledEntry(&ctx, base + spec.entryOffset))
                    PpcInterpreter::Run(&ctx, base + spec.entryOffset);
            } catch (...) {
                snap.threw = true;
            }
        }
        ctx.pc = 0;
        snap.ctx = ctx;
        snap.syncs = GuestCodePatch::TestSyncCount() - syncsBefore;
        const uint8_t* d = Memory::GetPointer(kDataBase, kDataSize);
        snap.data.assign(d, d + kDataSize);
        const uint8_t* c = Memory::GetPointer(base, static_cast<uint32_t>(words.size() * 4));
        snap.code.assign(c, c + words.size() * 4);
        out.push_back(std::move(snap));
        ++g_runs;
    }
    PpcInterpreter::ClearEngineOverride();
    return out;
}

std::string Diff(const Snap& a, const Snap& b) {
    std::string out;
    if (a.threw != b.threw) out += "    exception differs\n";
    if (a.syncs != b.syncs) out += Fmt("    patch syncs: classic=%u got=%u\n", a.syncs, b.syncs);
    for (int i = 0; i < 32; ++i)
        if (a.ctx.gpr[i] != b.ctx.gpr[i]) out += Fmt("    r%d: classic=%08X got=%08X\n", i, a.ctx.gpr[i], b.ctx.gpr[i]);
    const auto word = [&](const char* n, uint32_t x, uint32_t y) {
        if (x != y) out += Fmt("    %s: classic=%08X got=%08X\n", n, x, y);
    };
    for (uint32_t i = 0; i < 8; ++i)  // bytewise: the JIT stores the interpreter's exact field values (jit_test_cr.h)
        if (a.ctx.crv[i] != b.ctx.crv[i])
            out += Fmt("    crv[%u]: classic=%016llX (reads %X) got=%016llX (reads %X)\n", i, static_cast<unsigned long long>(a.ctx.crv[i]),
                       PpcCrvNibble(a.ctx.crv[i]), static_cast<unsigned long long>(b.ctx.crv[i]), PpcCrvNibble(b.ctx.crv[i]));
    word("lr", a.ctx.lr, b.ctx.lr);
    word("ctr", a.ctx.ctr, b.ctx.ctr);
    word("xer", a.ctx.xer, b.ctx.xer);
    for (int i = 0; i < 32; ++i)
        if (a.ctx.fpr[i].raw != b.ctx.fpr[i].raw)
            out += Fmt("    f%d: classic=%016llX got=%016llX\n", i, static_cast<unsigned long long>(a.ctx.fpr[i].raw),
                       static_cast<unsigned long long>(b.ctx.fpr[i].raw));
    if (a.data != b.data) out += "    sandbox differs\n";
    if (a.code != b.code) out += "    code differs\n";
    return out;
}

uint32_t NextBase() {
    static uint32_t next = 0;
    return kSlotBase + (next++ % kSlotCount) * kSlotBytes;
}

unsigned g_reported = 0;
void RunCase(const std::string& name, const Scenario& s) {
    ++g_cases;
    static const char* const filter = std::getenv("JIT_BRANCH_FILTER");
    if (filter != nullptr && name.find(filter) == std::string::npos) return;
    for (const EngineConfig& config : kConfigs) {
        const uint32_t base = NextBase();
        const std::vector<Snap> ref = Execute(kReference, s, base);
        const std::vector<Snap> got = Execute(config, s, base);
        for (size_t i = 0; i < ref.size(); ++i) {
            const std::string diff = Diff(ref[i], got[i]);
            if (diff.empty()) continue;
            ++g_failures;
            if (g_reported++ < 8)
                std::fprintf(stderr, "FAIL %s: engine %s, run %zu of %zu, base %08X\n%s", name.c_str(), config.name, i + 1, ref.size(),
                             base, diff.c_str());
            break;
        }
    }
}

// A scenario with one program run from each state, twice over (the second run takes the linked paths).
void AddProgram(const std::string& name, Asm& a, const std::vector<CpuContext>& states) {
    Scenario s;
    s.image = a.Finish();
    s.runs = [states](uint32_t, std::vector<RunSpec>& runs) {
        for (int round = 0; round < 2; ++round)
            for (const CpuContext& c : states) runs.push_back({c, {}, 0, {}});
    };
    RunCase(name, s);
}

// --- the programs -----------------------------------------------------------------------------------------------------------
using namespace ppc;

constexpr uint32_t kCtrs[] = {0u, 1u, 2u, 0xFFFFFFFFu};

// Every BO and BI of bc; the result says whether it was taken (r4) or not (r3), and ctr shows the decrement.
void BcMatrix() {
    Rng rng(1);
    for (uint32_t bo = 0; bo < 32; ++bo)
        for (uint32_t bi = 0; bi < 32; ++bi) {
            Asm a;
            Label taken = a.NewLabel();
            a.Bc(bo, bi, taken);
            a << addi(3, 3, 1) << blr();
            a.Bind(taken);
            a << addi(4, 4, 1) << blr();
            std::vector<CpuContext> states;
            for (uint32_t ctr : kCtrs) {
                for (uint32_t cr : {0u, 0xFFFFFFFFu, 0x80000000u >> bi, ~(0x80000000u >> bi)})
                    for (bool alternate : {false, true}) {  // the nibbles as table values, and as compare differences that read the same
                        CpuContext c = BaseState(rng);
                        c.ctr = ctr;
                        JitTestCr::SetPacked(c, cr, alternate, [&] { return rng.Next(); });
                        states.push_back(c);
                    }
                CpuContext c = BaseState(rng);  // fields of every kind, raw 64-bit patterns included
                c.ctr = ctr;
                states.push_back(c);
            }
            AddProgram(Fmt("bc bo=%u bi=%u", bo, bi), a, states);
        }
}

// bclr to the frame's return address (lr as the harness set it, with low bits), and to a block of the program.
void BclrMatrix() {
    Rng rng(2);
    for (uint32_t bo = 0; bo < 32; ++bo)
        for (uint32_t bi : {0u, 1u, 2u, 3u, 14u, 31u}) {
            if ((bo & 0x14u) == 0x14u) continue;  // the unconditional form is blr
            for (int elsewhere = 0; elsewhere < 2; ++elsewhere) {
                Asm a;
                Label other = a.NewLabel();
                a << mflr(6);
                if (elsewhere) a.LiLabel(5, other) << mtlr(5);
                a << bclr(bo, bi) << addi(3, 3, 1) << mtlr(6) << blr();
                a.Bind(other);
                a << addi(4, 4, 1) << mtlr(6) << blr();
                std::vector<CpuContext> states;
                for (uint32_t ctr : kCtrs)
                    for (uint32_t cr : {0u, 0xFFFFFFFFu, 0x80000000u >> bi, ~(0x80000000u >> bi), rng.Next()})
                        for (bool alternate : {false, true}) {
                            CpuContext c = BaseState(rng);
                            c.ctr = ctr;
                            JitTestCr::SetPacked(c, cr, alternate, [&] { return rng.Next(); });
                            if (!elsewhere) c.lr = kReturn | (rng.Below(4));
                            states.push_back(c);
                        }
                AddProgram(Fmt("bclr bo=%u bi=%u %s", bo, bi, elsewhere ? "elsewhere" : "return"), a, states);
            }
        }
}

// The five fused kinds, a compare field and a bc field (the same or not), every bit, and the BO classes.
void FusedMatrix() {
    Rng rng(3);
    constexpr uint32_t bos[] = {12, 4, 8, 0, 10, 2, 16, 18, 20};
    for (int kind = 0; kind < 5; ++kind)
        for (uint32_t cmpField : {0u, 3u, 7u})
            for (uint32_t bcField : {0u, 3u, 7u})
                for (uint32_t bit = 0; bit < 4; ++bit)
                    for (uint32_t bo : bos) {
                        const uint32_t imm = kind == 1 || kind == 3 ? (rng.Below(2) ? 7u : 0xFFFFu) : (rng.Below(2) ? 7u : 0xFFFDu);
                        Asm a;
                        Label taken = a.NewLabel();
                        switch (kind) {
                        case 0: a << cmpwi(cmpField, 3, static_cast<int16_t>(imm)); break;
                        case 1: a << cmplwi(cmpField, 3, imm & 0xFFFFu); break;
                        case 2: a << cmpw(cmpField, 3, 4); break;
                        case 3: a << cmplw(cmpField, 3, 4); break;
                        default: a << rlwinm(5, 3, rng.Below(32), rng.Below(32), rng.Below(32), 1); break;
                        }
                        a.Bc(bo, bcField * 4 + bit, taken);
                        a << addi(8, 8, 1) << blr();
                        a.Bind(taken);
                        a << addi(9, 9, 1) << blr();
                        std::vector<CpuContext> states;
                        for (uint32_t x : {0u, 5u, 6u, 7u, 8u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFDu, 0xFFFFFFFFu, 0xFFFFu}) {
                            for (uint32_t y : {0u, 7u, 0xFFFFFFFFu}) {
                                CpuContext c = BaseState(rng);
                                c.gpr[3] = x;
                                c.gpr[4] = kind == 2 || kind == 3 ? y : x;
                                c.ctr = kCtrs[rng.Below(4)];
                                states.push_back(c);
                            }
                        }
                        AddProgram(Fmt("fused kind=%d cmp=cr%u bc=cr%u.%u bo=%u", kind, cmpField, bcField, bit, bo), a, states);
                    }
}

// A compare followed by a bc in the middle of a block, and several of them in a row.
void FusedChains() {
    Rng rng(4);
    Asm a;
    Label l1 = a.NewLabel(), l2 = a.NewLabel(), l3 = a.NewLabel(), end = a.NewLabel();
    a << cmpwi(1, 3, 5);
    a.Bc(12, 4 * 1 + 0, l1);   // cr1 lt
    a << cmplwi(2, 4, 100);
    a.Bc(4, 4 * 2 + 1, l2);    // not cr2 gt
    a << rlwinm(6, 3, 3, 28, 31, 1);
    a.Bc(12, 2, l3);           // cr0 eq
    a << addi(7, 7, 1);
    a.B(end);
    a.Bind(l1);
    a << addi(7, 7, 10);
    a.B(end);
    a.Bind(l2);
    a << addi(7, 7, 100);
    a.B(end);
    a.Bind(l3);
    a << addi(7, 7, 1000);
    a.Bind(end);
    a << blr();
    std::vector<CpuContext> states;
    for (int i = 0; i < 60; ++i) states.push_back(BaseState(rng));
    AddProgram("fused chain", a, states);
}

// f1 held in paired form (ps_merge10) when the frame returns, through blr and through a taken bclr.
void ReturnNormalisation() {
    Rng rng(5);
    for (int kind = 0; kind < 3; ++kind) {
        Asm a;
        a << ps_merge10(1, 2, 3);
        if (kind == 0) a << blr();
        else if (kind == 1) a << bclr(12, 2) << ps_merge10(1, 4, 5) << blr();
        else a << nop() << cmpwi(0, 3, 0) << bclr(4, 2) << addi(3, 3, 1) << blr();
        std::vector<CpuContext> states;
        for (int i = 0; i < 24; ++i) states.push_back(BaseState(rng));
        AddProgram(Fmt("return f1 normalised %d", kind), a, states);
    }
}

// A jump table through bctr: the case comes from r3's low bits, with runs that revisit cases (the inline cache).
void JumpTables() {
    Rng rng(6);
    for (int cases : {1, 2, 3, 4, 8}) {
        Asm a;
        Label table = a.NewLabel(), join = a.NewLabel();
        a.LiLabel(4, table);
        a << rlwinm(5, 3, 3, 26, 28) << add(4, 4, 5) << mtctr(4) << bctr();
        a.Bind(table);
        for (int c = 0; c < 8; ++c) {
            a << addi(6, 6, c + 1);
            a.B(join);
        }
        a.Bind(join);
        a << addi(7, 7, 1) << blr();
        std::vector<CpuContext> states;
        for (int i = 0; i < 200; ++i) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = rng.Below(static_cast<uint32_t>(cases)) + (rng.Below(3) == 0 ? 8u * rng.Below(4) : 0u);
            c.gpr[6] = 0;
            states.push_back(c);
        }
        AddProgram(Fmt("jump table %d cases", cases), a, states);
    }
}

// bctr whose target is a local label, or the registered function (a tail call), in the order the run list gives: the
// registered target must always be a tail call, however often the local one was cached before it.
void BctrTranslatedAndNot() {
    Rng rng(7);
    for (int pattern = 0; pattern < 4; ++pattern) {
        Asm a;
        Label local = a.NewLabel(), local2 = a.NewLabel();
        a << cmpwi(0, 3, 0);
        a.Bc(12, 2, local);  // r3 == 0: a local target
        a << cmpwi(0, 3, 1);
        a.Bc(12, 2, local2);
        a.Li32(4, kFnModify);
        a << mtctr(4) << bctr();
        Label t1 = a.NewLabel(), t2 = a.NewLabel();
        a.Bind(local);
        a.LiLabel(4, t1);
        a << mtctr(4) << bctr() << nop();
        a.Bind(t1);
        a << addi(6, 6, 1) << blr();
        a.Bind(local2);
        a.LiLabel(4, t2);
        a << mtctr(4) << bctr() << nop();
        a.Bind(t2);
        a << addi(6, 6, 2) << blr();
        std::vector<CpuContext> states;
        static const unsigned orders[4][12] = {{0, 0, 2, 0, 2, 2, 0, 1, 2, 1, 0, 2}, {2, 2, 0, 0, 0, 2, 1, 1, 2, 0, 1, 2},
                                               {0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 2, 2}, {2, 0, 2, 1, 2, 0, 2, 1, 0, 0, 1, 1}};
        for (unsigned which : orders[pattern]) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = which;
            states.push_back(c);
        }
        AddProgram(Fmt("bctr local/registered pattern %d", pattern), a, states);
    }
}

// b and bctr to the registered function (a tail call), with lr the frame's return or another address (the call then goes on
// there), and b to a continuation (the test support's ContinuationAt answers for kContinuation).
void TailCalls() {
    Rng rng(8);
    for (int kind = 0; kind < 3; ++kind)
        for (int lrIsLocal = 0; lrIsLocal < 2; ++lrIsLocal) {
            Asm a;
            Label after = a.NewLabel();
            a << mflr(6);
            if (lrIsLocal) a.LiLabel(5, after) << mtlr(5);
            if (kind == 0) a.BAbs(kFnModify);
            else if (kind == 1) a.Li32(4, kFnModify), a << mtctr(4) << bctr();
            else a.BAbs(kContinuation);
            a.Bind(after);
            a << addi(10, 10, 1) << mtlr(6) << blr();
            std::vector<CpuContext> states;
            for (int i = 0; i < 6; ++i) states.push_back(BaseState(rng));
            AddProgram(Fmt("tail call kind=%d lr %s", kind, lrIsLocal ? "local" : "frame"), a, states);
        }
}

// Calls: bl to leaf code, nested, bl .+4, to a registered function, bctrl to leaf code and to a registered one in turn.
void Calls() {
    Rng rng(9);
    {
        Asm a;
        Label leaf = a.NewLabel(), leaf2 = a.NewLabel();
        a << mflr(31);
        a.B(leaf, true);
        a << addi(3, 3, 1);
        a.B(leaf, true);
        a << cmpwi(0, 3, 4);
        a.Bc(12, 0, leaf2, true);  // bcl: a conditional call (not a Bl op)
        a << addi(3, 3, 100) << mtlr(31) << blr();
        a.Bind(leaf);
        a << addi(4, 4, 1) << blr();
        a.Bind(leaf2);
        a << mflr(30) << addi(5, 5, 7);
        a.B(leaf, true);
        a << mtlr(30) << blr();
        std::vector<CpuContext> states;
        for (int i = 0; i < 12; ++i) states.push_back(BaseState(rng));
        AddProgram("leaf calls, nested and conditional", a, states);
    }
    {
        Asm a;
        a << mflr(31) << 0x48000005u /* bl .+4 */ << mflr(6) << addi(3, 3, 2) << mtlr(31) << blr();
        std::vector<CpuContext> states;
        for (int i = 0; i < 4; ++i) states.push_back(BaseState(rng));
        AddProgram("bl .+4", a, states);
    }
    {
        Asm a;
        a << mflr(31);
        a.BAbs(kFnModify, true);
        a << addi(3, 3, 1);
        a.BAbs(kFnModify, true);
        a << addi(3, 3, 1) << cmpwi(0, 3, 100) << mtlr(31) << blr();
        std::vector<CpuContext> states;
        for (int i = 0; i < 12; ++i) states.push_back(BaseState(rng));
        AddProgram("calls into a registered function", a, states);
    }
    for (int order = 0; order < 3; ++order) {  // bctrl to leaf code and to a registered function in one block
        Asm a;
        Label leaf = a.NewLabel();
        a << mflr(31);
        a.LiLabel(4, leaf) << mtctr(4) << bctrl() << addi(3, 3, 1);
        a.Li32(4, kFnModify);
        a << mtctr(4) << bctrl() << addi(3, 3, 1);
        a.LiLabel(4, leaf);
        a << mtctr(4) << bctrl() << mtlr(31) << blr();
        a.Bind(leaf);
        a << addi(5, 5, 1) << blr();
        std::vector<CpuContext> states;
        for (int i = 0; i < 8; ++i) states.push_back(BaseState(rng));
        AddProgram(Fmt("bctrl leaf and registered %d", order), a, states);
    }
}

// A callee that rewrites the caller's next instruction: the block must leave after the call and run the new word.
void SelfModifying() {
    for (int kind = 0; kind < 3; ++kind) {  // bl to a registered writer, bctrl to it, bl to a leaf that stores through r3
        Asm b;
        Label v = b.NewLabel(), lf = b.NewLabel();
        b << mflr(31);
        b.LiLabel(3, v);
        b << ori(4, 12, 0);  // the word to write is in r12
        if (kind == 0) b.BAbs(kFnWrite, true);
        else if (kind == 1) b.Li32(8, kFnWrite), b << mtctr(8) << bctrl();
        else b.B(lf, true);
        b.Bind(v);
        b << addi(6, 6, 1) << addi(6, 6, 2) << mtlr(31) << blr();
        b.Bind(lf);
        b << stw(4, 0, 3) << blr();
        Scenario t;
        t.image = b.Finish();
        const size_t vi = static_cast<size_t>(std::find(t.image.words.begin(), t.image.words.end(), addi(6, 6, 1)) - t.image.words.begin());
        t.runs = [=](uint32_t, std::vector<RunSpec>& runs) {
            Rng local(11 + kind);
            for (int i = 0; i < 10; ++i) {
                CpuContext c = BaseState(local);
                c.gpr[6] = 0;
                c.gpr[12] = i % 3 == 0 ? addi(6, 6, 1) : addi(6, 6, 1000 + i);
                // every run starts with the original word in place: the call changes it, the next run restores it
                runs.push_back({c, {{static_cast<uint32_t>(vi * 4), addi(6, 6, 1)}}, 0, {}});
            }
        };
        RunCase(Fmt("callee rewrites the next instruction %d", kind), t);
    }
}

// --- thin calls --------------------------------------------------------------------------------------
// Calls into translated functions with every guard configuration the registry can express (the FPR write mask: empty, one
// register, runs, scattered; GPR preservation, which only natives have), every float ABI the test support's table holds, and
// callees that clobber everything and, for some inputs, throw. The reference is the classic interpreter's Call and
// TryDispatchRawCpuTarget; the JIT runs the same programs through its per-target shims.
constexpr uint32_t kThinBase = 0x80040000u, kThinCount = 64, kThinStride = 0x10u;
constexpr uint32_t kPatchedBase = 0x80041000u, kPatchedCount = 24, kPatchedStride = 0x10u;
constexpr uint32_t kThrower = 0x80042000u, kReenter = 0x80042010u, kPlain = 0x80042030u;
constexpr uint32_t kMasks[] = {0u, 0xffffc000u, 0x03f00000u, 0x80000000u, 0xc0000000u, 0x80028000u, 0xffffffffu};
constexpr uint32_t kMaskCount = sizeof kMasks / sizeof *kMasks;

uint32_t Mix(uint32_t x) {
    x ^= x >> 15;
    x *= 0x2C1B3C6Du;
    x ^= x >> 12;
    x *= 0x297A2D39u;
    x ^= x >> 15;
    return x;
}

// A callee that writes everything it may: the volatile and non-volatile GPRs and every FPR, as raw bits; not r1, r2, r13, lr.
void ClobberAll(CpuContext* c, uint32_t salt) {
    const uint32_t h = Mix(c->gpr[3] ^ salt ^ (c->gpr[4] * 7u));
    for (int r = 0; r < 32; ++r) {
        if (r == 1 || r == 2 || r == 13) continue;
        c->gpr[r] = Mix(h + static_cast<uint32_t>(r));
    }
    for (int r = 0; r < 32; ++r) c->fpr[r].raw = (static_cast<uint64_t>(Mix(h + 100u + r)) << 32) | Mix(h + 200u + r);
    for (uint32_t i = 0; i < 8; ++i)
        c->crv[i] = JitTestCr::FieldValue(Mix(h + 300u + 3u * i), Mix(h + 301u + 3u * i), Mix(h + 302u + 3u * i));
    c->ctr = Mix(h + 400u);
}
void FnClobber(CpuContext* c) { ClobberAll(c, 1); }
void FnClobberThrow(CpuContext* c) {
    const bool fail = (c->gpr[3] & 1u) != 0;
    ClobberAll(c, 2);
    if (fail) throw std::runtime_error("callee threw");
}
void FnThrower(CpuContext* c) {
    ClobberAll(c, 5);
    throw std::runtime_error("callee threw");
}
void FnReenter(CpuContext* c) {  // a registered C++ function that runs guest code at r5 in a new Run
    c->gpr[0] = 0x99;
    PpcInterpreter::Run(c, c->gpr[5]);
    c->gpr[6] ^= 0x5A5A5A5Au;
}
void FnPlain(CpuContext* c) {
    c->gpr[3] = Mix(c->gpr[3]) | 1u;
    c->fpr[1].d = c->fpr[1].d * 0.5 + 1.0;
}

// The patched functions: what RunPatched does for a function whose guard byte is set, with the test double's answers.
void PatchedBody(CpuContext* c, uint32_t address, uint32_t k) {
    const uint32_t at = GuestCodePatch::TestInterpretedAt(address);
    if (at != 0) {
        // RunPatched's last step: the lean entry when it applies, the general one otherwise
        if (!PpcInterpreter::detail::RunCompiledEntry(c, at)) PpcInterpreter::Run(c, at);
        GuestCodePatch::Sync();
        return;
    }
    ClobberAll(c, 10u + k);  // the translation's body
    c->gpr[3] = (c->gpr[3] & 0xFFFFu) + 3u;
}
template <uint32_t K>
void Patched(CpuContext* c) {
    PatchedBody(c, kPatchedBase + kPatchedStride * K, K);
}
template <uint32_t... K>
constexpr std::array<void (*)(CpuContext*), sizeof...(K)> PatchedTable(std::integer_sequence<uint32_t, K...>) {
    return {&Patched<K>...};
}
constexpr auto kPatchedFns = PatchedTable(std::make_integer_sequence<uint32_t, kPatchedCount>{});

void RegisterFunctionEx(uint32_t address, const char* name, void (*fn)(CpuContext*), FunctionKind kind, uint32_t mask) {
    TranslatedFunctionInfo info{};
    info.address = address;
    info.name = name;
    info.rawCpuInvoker = fn;
    info.entryPoint = reinterpret_cast<void*>(fn);
    info.kind = kind;
    info.nonvolatileFprWriteMask = mask;
    TranslatedFunctionRegistry::Register(std::move(info));
}

void RegisterCallTestFunctions() {
    static const FunctionKind kinds[3] = {FunctionKind::BaseTranslated, FunctionKind::ModTranslated, FunctionKind::Native};
    for (uint32_t i = 0; i < kThinCount; ++i)
        RegisterFunctionEx(kThinBase + kThinStride * i, "ThinCallee", i % 4 == 3 ? FnClobberThrow : FnClobber,
                           kinds[i % 3], kMasks[(i / 3) % kMaskCount]);
    for (uint32_t k = 0; k < kPatchedCount; ++k) {
        RegisterFunctionEx(kPatchedBase + kPatchedStride * k, "PatchedCallee", kPatchedFns[k], FunctionKind::BaseTranslated,
                           kMasks[k % kMaskCount]);
        GuestCodePatch::TestSetInterpreted(kPatchedBase + kPatchedStride * k, 0);
    }
    RegisterFunctionEx(kThrower, "Thrower", FnThrower, FunctionKind::BaseTranslated, kMasks[2]);
    RegisterFunctionEx(kReenter, "Reenter", FnReenter, FunctionKind::BaseTranslated, kMasks[1]);
    RegisterFunctionEx(kPlain, "Plain", FnPlain, FunctionKind::BaseTranslated, kMasks[3]);
}

constexpr uint32_t stfd(uint32_t fs, int32_t d, uint32_t ra) { return ppc::D(54, fs, ra, d); }
constexpr uint32_t lfd(uint32_t fd, int32_t d, uint32_t ra) { return ppc::D(50, fd, ra, d); }
constexpr uint32_t stwu(uint32_t rs, int32_t d, uint32_t ra) { return ppc::D(37, rs, ra, d); }

// f1-f13 into paired form (so a float argument normalisation has something to do), and their values stored as scalars (which
// shows each register's form to the next engine step).
void PairedSetup(Asm& a, uint32_t salt) {
    for (uint32_t k = 1; k <= 13; ++k) a << ppc::ps_merge10(k, 14 + (k + salt) % 18, 14 + (k * 3 + salt) % 18);
}
void ObserveFprs(Asm& a, int32_t offset) {
    for (uint32_t k = 1; k <= 13; ++k) a << stfd(k, offset + static_cast<int32_t>(8 * k), 1);
}

const char* const kCallStyle[] = {"bl", "bctrl", "b", "bctr", "bl twice"};

// Every callee of the thin-call table through the four call forms, with paired f1-f13 beforehand and the result observed after.
void ThinCalls() {
    Rng rng(21);
    for (uint32_t i = 0; i < kThinCount; ++i) {
        const uint32_t target = kThinBase + kThinStride * i;
        for (int style = 0; style < 5; ++style) {
            Asm a;
            const bool tail = style == 2 || style == 3;
            Label after = a.NewLabel();
            a << mflr(0) << stw(0, 0, 1);
            PairedSetup(a, i);
            if (tail) a.LiLabel(5, after) << mtlr(5);  // the callee returns to `after`; the frame's own lr is in its stack slot
            auto call = [&] {
                if (style == 0 || style == 4) a.BAbs(target, true);
                else if (style == 1) a.Li32(8, target), a << mtctr(8) << bctrl();
                else if (style == 2) a.BAbs(target);
                else a.Li32(8, target), a << mtctr(8) << bctr();
            };
            call();
            if (style == 4) {
                a << addi(10, 10, 1);
                call();
            }
            a.Bind(after);
            ObserveFprs(a, 0x40);
            a << lwz(0, 0, 1) << mtlr(0) << blr();
            std::vector<CpuContext> states;
            for (int s = 0; s < 6; ++s) {
                CpuContext c = BaseState(rng);
                c.gpr[3] = rng.Next();
                states.push_back(c);
            }
            AddProgram(Fmt("thin %s -> function %u", kCallStyle[style], i), a, states);
        }
    }
}

// One bctrl site and one bctr site whose target changes: registered functions of different configurations, a patched one
// (translated, or, with `patched`, interpreted at guest code of the program), and untranslated code, in the order the table in
// the program's image says (written per run).
void IndirectSites(bool patched) {
    Rng rng(22);
    for (int style = 0; style < 2; ++style)
        for (int pattern = 0; pattern < 6; ++pattern) {
            Asm a;
            Label table = a.NewLabel(), loop = a.NewLabel(), done = a.NewLabel(), leaf = a.NewLabel();
            // r4 walks the table (in the image), r5 counts; the callees clobber everything, so the walk's state lives in the
            // stack frame
            a << mflr(0) << stw(0, 0, 1);
            a.LiLabel(4, table) << stw(4, 4, 1) << li(5, 12) << stw(5, 8, 1);
            a.Bind(loop);
            a << lwz(5, 8, 1) << cmpwi(0, 5, 0);
            a.Bc(12, 2, done);
            a << addi(5, 5, -1) << stw(5, 8, 1) << lwz(4, 4, 1) << lwz(6, 0, 4) << addi(4, 4, 4) << stw(4, 4, 1);
            PairedSetup(a, 5);
            if (style == 0) {
                a << mtctr(6) << bctrl() << addi(10, 10, 1);
                a.B(loop);
            } else {
                Label back = a.NewLabel();
                a.LiLabel(7, back) << mtlr(7) << mtctr(6) << bctr();
                a.Bind(back);
                a.B(loop);
            }
            a.Bind(done);
            ObserveFprs(a, 0x40);
            a << lwz(0, 0, 1) << mtlr(0) << blr();
            a.Bind(leaf);
            a << addi(3, 3, 17) << ori(12, 3, 0x30) << blr();
            const size_t x0At = a.Here();
            a << addi(3, 3, 31) << ppc::ps_merge10(1, 2, 3) << blr();
            const size_t x1At = a.Here();
            a << mflr(9) << addi(3, 3, 41) << blr();
            a.Bind(table);
            const size_t tableIndex = a.Here();
            for (int e = 0; e < 14; ++e) a << 0u;
            Scenario s;
            s.image = a.Finish();
            const size_t leafAt = static_cast<size_t>(std::find(s.image.words.begin(), s.image.words.end(), addi(3, 3, 17)) - s.image.words.begin());
            s.runs = [=](uint32_t base, std::vector<RunSpec>& runs) {
                Rng local(30 + pattern * 7 + style + (patched ? 100 : 0));
                for (int round = 0; round < 3; ++round) {
                    RunSpec spec;
                    spec.ctx = BaseState(local);
                    spec.ctx.gpr[3] = local.Next();
                    uint32_t f[3];
                    if (patched) {
                        f[0] = kPatchedBase;
                        f[1] = kPatchedBase + kPatchedStride;
                        f[2] = kPatchedBase + kPatchedStride * 2;
                    } else {
                        f[0] = kThinBase + kThinStride * local.Below(kThinCount);
                        f[1] = kThinBase + kThinStride * local.Below(kThinCount);
                        f[2] = kPatchedBase + kPatchedStride * (3 + local.Below(kPatchedCount - 3));
                    }
                    const uint32_t leafAddress = base + 4u * static_cast<uint32_t>(leafAt);
                    static const int plans[6][12] = {{0, 0, 0, 1, 1, 0, 3, 3, 2, 0, 3, 1}, {0, 1, 0, 1, 0, 1, 2, 2, 0, 0, 3, 3},
                                                     {3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1}, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
                                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 2, 3, 1, 0, 2, 3, 1, 0, 2, 3}};
                    for (int e = 0; e < 12; ++e) {
                        const int w = plans[pattern][e];
                        spec.writes.push_back({static_cast<uint32_t>(4 * (tableIndex + e)), w == 3 ? leafAddress : f[w]});
                    }
                    if (patched) {  // functions 0 and 1 are interpreted (at guest code of this program), function 2 runs translated
                        const uint32_t a0 = base + 4u * static_cast<uint32_t>(x0At), a1 = base + 4u * static_cast<uint32_t>(x1At);
                        spec.before = [a0, a1] {
                            for (uint32_t k = 0; k < kPatchedCount; ++k) GuestCodePatch::TestSetInterpreted(kPatchedBase + kPatchedStride * k, 0);
                            GuestCodePatch::TestSetInterpreted(kPatchedBase, a0);
                            GuestCodePatch::TestSetInterpreted(kPatchedBase + kPatchedStride, a1);
                        };
                    }
                    runs.push_back(std::move(spec));
                }
            };
            RunCase(Fmt("%s indirect %s pattern %d", patched ? "patched" : "thin", style == 0 ? "bctrl" : "bctr", pattern), s);
        }
}

// The programs that call patched functions. Level k of a chain is guest code interpreted for a call of function k (kPatchedBase + 16 k): it
// opens a frame, writes non-volatile GPRs and FPRs, leaves f1 paired, calls function k + 1 (or itself, counting r4 down, in the
// recursion), and returns by blr, by a conditional return, by a tail call of a translated function, or by a tail call of a
// patched function (interpreted at a leaf of its own).
constexpr uint32_t PatchedAddress(uint32_t k) { return kPatchedBase + kPatchedStride * k; }

void ResetPatched() {
    for (uint32_t k = 0; k < kPatchedCount; ++k) GuestCodePatch::TestSetInterpreted(PatchedAddress(k), 0);
}

void PatchedChain(int depth, int retStyle, bool recursion) {
    Rng rng(40 + depth * 4 + retStyle);
    Asm a;
    std::vector<size_t> at(static_cast<size_t>(depth));
    a << mflr(0) << stw(0, 0, 1);
    PairedSetup(a, static_cast<uint32_t>(depth));
    a.BAbs(PatchedAddress(0), true);
    a << addi(10, 10, 1);
    ObserveFprs(a, 0x40);
    a << lwz(0, 0, 1) << mtlr(0) << blr();
    for (int k = 0; k < depth; ++k) {
        at[static_cast<size_t>(k)] = a.Here();
        const uint32_t u = static_cast<uint32_t>(k);
        a << stwu(1, -32, 1) << mflr(0) << stw(0, 36, 1);
        a << addi(14 + u % 8, 3, static_cast<int32_t>(u)) << lfd(14 + (u * 3) % 10, 0x40, 1) << lfd(20 + u % 8, 0x48, 1);
        a << ppc::ps_merge10(1, 2 + u % 6, 3 + u % 5) << addi(3, 3, 1 + static_cast<int32_t>(u));
        if (!recursion) {
            if (k + 1 < depth) a.BAbs(PatchedAddress(u + 1), true);
        } else {
            Label skip = a.NewLabel();
            a << cmpwi(0, 4, 0);
            a.Bc(12, 2, skip);
            a << addi(4, 4, -1);
            a.BAbs(PatchedAddress(0), true);
            a.Bind(skip);
        }
        a << addi(11, 11, static_cast<int32_t>(u)) << ppc::ps_merge10(1, 2, 4);
        a << lwz(0, 36, 1) << mtlr(0) << addi(1, 1, 32);
        switch (retStyle) {
        case 0: a << blr(); break;
        case 1: a << rlwinm(0, 3, 0, 31, 31, 1) << bclr(12, 2) << addi(3, 3, 7) << blr(); break;
        case 2: a.BAbs(kPlain); break;
        default: a.BAbs(PatchedAddress(kPatchedCount - 1)); break;
        }
    }
    const size_t leafAt = a.Here();
    a << addi(3, 3, 5) << ppc::ps_merge10(1, 3, 3) << blr();
    Scenario s;
    s.image = a.Finish();
    s.runs = [=](uint32_t base, std::vector<RunSpec>& runs) {
        Rng local(50 + depth + retStyle);
        for (int round = 0; round < 3; ++round)
            for (uint32_t r4 : recursion ? std::vector<uint32_t>{0, 1, 4, 8, 9, 20} : std::vector<uint32_t>{0}) {
                RunSpec spec;
                spec.ctx = BaseState(local);
                spec.ctx.gpr[4] = r4;
                spec.before = [=] {
                    ResetPatched();
                    for (int k = 0; k < depth; ++k)
                        GuestCodePatch::TestSetInterpreted(PatchedAddress(static_cast<uint32_t>(k)),
                                                           base + 4u * static_cast<uint32_t>(at[static_cast<size_t>(k)]));
                    GuestCodePatch::TestSetInterpreted(PatchedAddress(kPatchedCount - 1), base + 4u * static_cast<uint32_t>(leafAt));
                };
                runs.push_back(std::move(spec));
            }
    };
    RunCase(Fmt("patched %s depth %d return style %d", recursion ? "recursion" : "chain", depth, retStyle), s);
}

// An exception thrown by a registered C++ function called from level `at` of a chain (an even deeper callee, a re-entered Run in
// between for the second kind): the registers after it, the guards of every level restored, must equal the reference's, and the next
// run (the table's every second state does not throw) must work.
void PatchedExceptions() {
    for (int depth : {1, 2, 3, 4, 6, 10})
        for (int kind = 0; kind < 2; ++kind) {
            Asm a;
            Label viaReenter = a.NewLabel();
            std::vector<size_t> at(static_cast<size_t>(depth));
            a << mflr(0) << stw(0, 0, 1);
            PairedSetup(a, 7);
            a.BAbs(PatchedAddress(0), true);
            ObserveFprs(a, 0x40);
            a << lwz(0, 0, 1) << mtlr(0) << blr();
            for (int k = 0; k < depth; ++k) {
                at[static_cast<size_t>(k)] = a.Here();
                const uint32_t u = static_cast<uint32_t>(k);
                a << stwu(1, -32, 1) << mflr(0) << stw(0, 36, 1) << addi(15 + u % 8, 3, 9) << lfd(16 + u % 9, 0x48, 1);
                if (k + 1 < depth) {
                    a.BAbs(PatchedAddress(u + 1), true);
                } else {
                    Label skip = a.NewLabel();
                    a << rlwinm(0, 12, 0, 31, 31, 1);  // r12's low bit: this run throws
                    a.Bc(12, 2, skip);
                    if (kind == 0) {
                        a.BAbs(kThrower, true);
                    } else {  // through a registered function that re-enters Run, whose guest code throws
                        a.LiLabel(5, viaReenter);
                        a.BAbs(kReenter, true);
                    }
                    a.Bind(skip);
                }
                a << addi(11, 11, 1) << lwz(0, 36, 1) << mtlr(0) << addi(1, 1, 32) << blr();
            }
            a.Bind(viaReenter);
            const size_t reenterAt = a.Here();
            a << mflr(9);
            a.BAbs(kThrower, true);
            a << blr();
            Scenario s;
            s.image = a.Finish();
            s.runs = [=](uint32_t base, std::vector<RunSpec>& runs) {
                Rng local(70 + depth + kind);
                for (int round = 0; round < 3; ++round)
                    for (int e = 0; e < 6; ++e) {
                        RunSpec spec;
                        spec.ctx = BaseState(local);
                        spec.ctx.gpr[12] = static_cast<uint32_t>(e) & 1u ? 0u : 1u;
                        spec.before = [=] {
                            ResetPatched();
                            for (int k = 0; k < depth; ++k)
                                GuestCodePatch::TestSetInterpreted(PatchedAddress(static_cast<uint32_t>(k)),
                                                                   base + 4u * static_cast<uint32_t>(at[static_cast<size_t>(k)]));
                        };
                        (void)reenterAt;
                        runs.push_back(std::move(spec));
                    }
            };
            RunCase(Fmt("patched exception depth %d kind %d", depth, kind), s);
        }
}

// A patched callee that calls a registered function that runs guest code (a new Run), which itself calls a patched function.
void PatchedReenter() {
    Asm a;
    Label inner = a.NewLabel();
    a << mflr(0) << stw(0, 0, 1);
    PairedSetup(a, 3);
    a.BAbs(PatchedAddress(0), true);
    ObserveFprs(a, 0x40);
    a << lwz(0, 0, 1) << mtlr(0) << blr();
    const size_t level0 = a.Here();
    a << stwu(1, -32, 1) << mflr(0) << stw(0, 36, 1) << addi(16, 3, 1);
    a.LiLabel(5, inner);
    a.BAbs(kReenter, true);
    a.BAbs(PatchedAddress(1), true);
    a << lwz(0, 36, 1) << mtlr(0) << addi(1, 1, 32) << blr();
    a.Bind(inner);
    const size_t innerAt = a.Here();
    a << stwu(1, -32, 1) << mflr(0) << stw(0, 36, 1) << addi(17, 3, 2);
    a.BAbs(PatchedAddress(2), true);
    a << lwz(0, 36, 1) << mtlr(0) << addi(1, 1, 32) << blr();
    const size_t level1 = a.Here();
    a << addi(3, 3, 11) << ppc::ps_merge10(1, 2, 3) << blr();
    const size_t level2 = a.Here();
    a << addi(3, 3, 13) << li(18, 5) << blr();
    Scenario s;
    s.image = a.Finish();
    s.runs = [=](uint32_t base, std::vector<RunSpec>& runs) {
        Rng local(80);
        for (int round = 0; round < 4; ++round) {
            RunSpec spec;
            spec.ctx = BaseState(local);
            (void)innerAt;
            spec.before = [=] {
                ResetPatched();
                GuestCodePatch::TestSetInterpreted(PatchedAddress(0), base + 4u * static_cast<uint32_t>(level0));
                GuestCodePatch::TestSetInterpreted(PatchedAddress(1), base + 4u * static_cast<uint32_t>(level1));
                GuestCodePatch::TestSetInterpreted(PatchedAddress(2), base + 4u * static_cast<uint32_t>(level2));
            };
            runs.push_back(std::move(spec));
        }
    };
    RunCase("patched callee re-enters Run", s);
}

// The patched callee writes the instruction after the call site: the caller's block must leave through its validity test and run
// the new word.
void PatchedSelfModifying() {
    for (int kind = 0; kind < 2; ++kind) {
        Asm b;
        Label v = b.NewLabel();
        b << mflr(31);
        b.LiLabel(3, v);
        b << ori(4, 12, 0);
        if (kind == 0) {
            b.BAbs(PatchedAddress(0), true);
        } else {
            b.Li32(8, PatchedAddress(0));
            b << mtctr(8) << bctrl();
        }
        b.Bind(v);
        b << addi(6, 6, 1) << addi(6, 6, 2) << mtlr(31) << blr();
        const size_t writerAt = b.Here();
        b << stw(4, 0, 3) << addi(5, 5, 1) << blr();
        Scenario t;
        t.image = b.Finish();
        const size_t vi = static_cast<size_t>(std::find(t.image.words.begin(), t.image.words.end(), addi(6, 6, 1)) - t.image.words.begin());
        t.runs = [=](uint32_t base, std::vector<RunSpec>& runs) {
            Rng local(90 + kind);
            for (int i = 0; i < 10; ++i) {
                RunSpec spec;
                spec.ctx = BaseState(local);
                spec.ctx.gpr[6] = 0;
                spec.ctx.gpr[12] = i % 3 == 0 ? addi(6, 6, 1) : addi(6, 6, 1000 + i);
                spec.writes.push_back({static_cast<uint32_t>(vi * 4), addi(6, 6, 1)});
                spec.before = [=] {
                    ResetPatched();
                    GuestCodePatch::TestSetInterpreted(PatchedAddress(0), base + 4u * static_cast<uint32_t>(writerAt));
                };
                runs.push_back(std::move(spec));
            }
        };
        RunCase(Fmt("patched callee rewrites the next instruction %d", kind), t);
    }
}

// --- subcall frames: patched functions whose guest code is at their own address -------------------------------------------
// The real RunPatched's case (no redirect): the wrapper of a patched function X runs X's own guest code in a Run of its own. With
// the patch hooks registered (main), the JIT calls such a function from a jit_enter_direct or subcall frame in a subcall frame
// linked to X's block (BlockEmitter::CallSubFrame; jit_subcall for a bctrl) and must leave what the wrapper leaves: the guard byte,
// the redirect bound and RunPatched's Sync test decide as the wrapper does, run by run.
constexpr uint32_t kSubBase = 0x80060000u, kSubCount = 8, kSubStride = 0x400u;
constexpr uint32_t kSubThrowLean = 0x80042040u, kSubSetPending = 0x80042050u, kSubSetDirty = 0x80042060u, kLeanPlain = 0x80042070u;
constexpr uint32_t kSubThrowGuarded = 0x80042080u;  // a native (GPR guard): called through JitThinCall, its exception stored and rethrown
uint8_t g_subGuard[kSubCount];
std::atomic<uint64_t> g_subPendingFast{0};
std::atomic<bool> g_subAnyDirty{false};
std::atomic<uint32_t> g_subLowestRedirect{UINT32_MAX};
uint64_t g_subWrapperRuns = 0;      // patched calls that went through a wrapper (not a subcall frame)
uint64_t g_subTranslationRuns = 0;  // calls of an unpatched function (its translation)
constexpr uint32_t SubAddress(uint32_t k) { return kSubBase + kSubStride * k; }

void SubSync() {
    g_subPendingFast.store(0);
    g_subAnyDirty.store(false);
    GuestCodePatch::Sync();
}
template <uint32_t K>
void SubWrapper(CpuContext* c) {
    if (g_subGuard[K] != 0) {  // MKW_GUEST_CODE_GUARD, then RunPatched (the redirect bound only routes the JIT here)
        ++g_subWrapperRuns;
        if (!PpcInterpreter::detail::RunCompiledEntry(c, SubAddress(K))) PpcInterpreter::Run(c, SubAddress(K));
        if (g_subPendingFast.load() != 0 || g_subAnyDirty.load()) SubSync();
        return;
    }
    ++g_subTranslationRuns;
    ClobberAll(c, 40u + K);  // the translation's body
    c->gpr[3] = (c->gpr[3] & 0xFFFFu) + 5u;
}
template <uint32_t... K>
constexpr std::array<void (*)(CpuContext*), sizeof...(K)> SubTable(std::integer_sequence<uint32_t, K...>) {
    return {&SubWrapper<K>...};
}
constexpr auto kSubFns = SubTable(std::make_integer_sequence<uint32_t, kSubCount>{});
void FnSetPending(CpuContext* c) { g_subPendingFast.store(1); c->gpr[3] += 0x100u; }
void FnSetDirty(CpuContext* c) { g_subAnyDirty.store(true); c->gpr[3] += 0x200u; }
void FnThrowLean(CpuContext* c) {
    c->gpr[3] ^= 0x77u;
    if ((c->gpr[12] & 1u) != 0) throw std::runtime_error("lean callee threw");
}

void RegisterSubFrameFunctions() {
    for (uint32_t k = 0; k < kSubCount; ++k)
        RegisterFunctionEx(SubAddress(k), "SubCallee", kSubFns[k], FunctionKind::BaseTranslated, 0);
    RegisterFunctionEx(kSubThrowLean, "SubThrowLean", FnThrowLean, FunctionKind::BaseTranslated, 0);
    RegisterFunctionEx(kSubThrowGuarded, "SubThrowGuarded", FnThrowLean, FunctionKind::Native, 0);
    RegisterFunctionEx(kSubSetPending, "SubSetPending", FnSetPending, FunctionKind::BaseTranslated, 0);
    RegisterFunctionEx(kSubSetDirty, "SubSetDirty", FnSetDirty, FunctionKind::BaseTranslated, 0);
    RegisterFunctionEx(kLeanPlain, "LeanPlain", FnPlain, FunctionKind::BaseTranslated, 0);
}

// The hooks GuestCodePatch::Protect registers, for the functions above: before any call target record is made (main).
void RegisterSubFrameHooks() {
    PpcInterpreter::detail::PatchedHooks hooks;
    hooks.patchedByteFor = [](uint32_t address) -> const uint8_t* {
        if (address < kSubBase || address >= kSubBase + kSubStride * kSubCount || (address - kSubBase) % kSubStride != 0) return nullptr;
        return &g_subGuard[(address - kSubBase) / kSubStride];
    };
    hooks.pendingFast = &g_subPendingFast;
    hooks.anyDirty = &g_subAnyDirty;
    hooks.lowestRedirect = &g_subLowestRedirect;
    hooks.sync = &SubSync;
    PpcInterpreter::detail::RegisterPatchedHooks(hooks);
}

// What a run sets before the program: the guard bytes (bit k of `unpatched`: function k runs its translation), the redirect bound,
// the hook flags.
void SubState(uint32_t unpatched, bool redirect) {
    for (uint32_t k = 0; k < kSubCount; ++k) g_subGuard[k] = (unpatched >> k) & 1u ? 0 : 1;
    g_subLowestRedirect.store(redirect ? SubAddress(0) : UINT32_MAX);
    g_subPendingFast.store(0);
    g_subAnyDirty.store(false);
}

void WriteImage(const Image& image, uint32_t at) {
    const std::vector<uint32_t> words = image.Resolve(at);
    for (size_t i = 0; i < words.size(); ++i) Memory::Write32(at + 4u * static_cast<uint32_t>(i), words[i]);
}

// Function k of a chain (at its own address): opens a frame, writes non-volatile GPRs and FPRs, leaves f1 paired, calls function
// k + 1 (or itself while r4 counts down, `recursion`), and leaves by `style`: 0 blr, 1 a conditional return, 2 a tail call of a
// translated function with an FPR guard, 3 of a lean one (a direct tail call), 4 bctr to a lean one, 5 a tail call of the next
// patched function, 6 blr to code of its own that then returns (a jump through lr inside the subcall frame), 7 a lean callee that
// throws on odd r12, 8 a callee that sets the hooks' pending write (RunPatched's Sync), 9 the dirty flag, 10 a bctrl of the next,
// 11 a guarded callee that throws on odd r12 (the exception stored in the subcall frame and rethrown by its continuation), 12 the
// same by bctrl.
Image SubLevel(uint32_t k, uint32_t depth, int style, bool recursion) {
    Asm a;
    a << stwu(1, -32, 1) << mflr(0) << stw(0, 36, 1);
    a << addi(14 + k % 8, 3, static_cast<int32_t>(k)) << lfd(14 + (k * 3) % 10, 0x40, 1) << lfd(20 + k % 8, 0x48, 1);
    a << ppc::ps_merge10(1, 2 + k % 6, 3 + k % 5) << addi(3, 3, 1 + static_cast<int32_t>(k));
    const bool next = k + 1 < depth;
    if (recursion) {
        Label skip = a.NewLabel();
        a << cmpwi(0, 4, 0);
        a.Bc(12, 2, skip);
        a << addi(4, 4, -1);
        a.BAbs(SubAddress(k), true);
        a.Bind(skip);
    } else if (next && style == 10) {
        a.Li32(8, SubAddress(k + 1)), a << mtctr(8) << bctrl();
    } else if (next) {
        a.BAbs(SubAddress(k + 1), true);
    }
    if (style == 7) a.BAbs(kSubThrowLean, true);
    if (style == 11) a.BAbs(kSubThrowGuarded, true);
    if (style == 12) a.Li32(8, kSubThrowGuarded), a << mtctr(8) << bctrl();
    if (style == 8 && k + 1 == depth) a.BAbs(kSubSetPending, true);
    if (style == 9 && k + 1 == depth) a.BAbs(kSubSetDirty, true);
    a << addi(11, 11, static_cast<int32_t>(k)) << ppc::ps_merge10(1, 2, 4);
    a << lwz(0, 36, 1) << mtlr(0) << addi(1, 1, 32);
    Label own = a.NewLabel();
    switch (style) {
    case 1: a << rlwinm(0, 3, 0, 31, 31, 1) << bclr(12, 2) << addi(3, 3, 7) << blr(); break;
    case 2: a.BAbs(kPlain); break;
    case 3: a.BAbs(kLeanPlain); break;
    case 4: a.Li32(8, kLeanPlain), a << mtctr(8) << bctr(); break;
    case 5: a.BAbs(next ? SubAddress(k + 1) : kLeanPlain); break;
    case 6:
        a << mflr(9);
        a.LiLabel(5, own) << mtlr(5) << blr();
        a.Bind(own);
        a << addi(3, 3, 99) << mtlr(9) << blr();
        break;
    default: a << blr(); break;
    }
    return a.Finish();
}

void SubFrameChain(uint32_t depth, int style, bool recursion) {
    Asm a;
    a << mflr(0) << stw(0, 0, 1);
    PairedSetup(a, depth);
    if (style == 10) a.Li32(8, SubAddress(0)), a << mtctr(8) << bctrl();
    else a.BAbs(SubAddress(0), true);
    a << addi(10, 10, 1);
    ObserveFprs(a, 0x40);
    a << lwz(0, 0, 1) << mtlr(0) << blr();
    Scenario s;
    s.image = a.Finish();
    std::vector<Image> levels;
    for (uint32_t k = 0; k < depth; ++k) levels.push_back(SubLevel(k, depth, style, recursion));
    s.runs = [=](uint32_t, std::vector<RunSpec>& runs) {
        for (uint32_t k = 0; k < depth; ++k) WriteImage(levels[k], SubAddress(k));
        Rng local(300 + depth * 16 + static_cast<uint32_t>(style) + (recursion ? 1000 : 0));
        // mostly patched (the subcall frames), now and then a function unpatched, or the redirect bound below them
        static const uint32_t kUnpatched[] = {0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0xFF, 0, 4, 0};
        for (int round = 0; round < 2; ++round)
            for (size_t i = 0; i < sizeof kUnpatched / sizeof *kUnpatched; ++i) {
                RunSpec spec;
                spec.ctx = BaseState(local);
                spec.ctx.gpr[4] = recursion ? local.Below(3) == 0 ? 40u : local.Below(6) : 0u;
                spec.ctx.gpr[12] = local.Below(4) == 0 ? 1u : 0u;
                const uint32_t unpatched = kUnpatched[i];
                const bool redirect = i == 3;
                spec.before = [unpatched, redirect] { SubState(unpatched, redirect); };
                runs.push_back(std::move(spec));
            }
    };
    // the blocks are compiled with the guard bytes set: the call sites are subcalls
    SubState(0, false);
    RunCase(Fmt("subcall frame %s depth %u style %d", recursion ? "recursion" : "chain", depth, style), s);
}

void SubFrameCalls() {
    for (uint32_t depth : {1u, 2u, 3u, 5u})
        for (int style = 0; style <= 12; ++style) SubFrameChain(depth, style, false);
    for (int style : {0, 1, 3, 5, 6, 7, 8, 11, 12}) SubFrameChain(1, style, true);
    SubState(0, false);
}


// Float arguments held paired, one register at a time (every position f1-f13, named by the callee's ABI or not): after the call the
// named one is a scalar, every other stays paired. Callees of every ABI and guard configuration, by bl, bctrl and b.
void ThinArgPositions() {
    Rng rng(24);
    for (uint32_t i = 0; i < 12; ++i) {
        const uint32_t target = kThinBase + kThinStride * i;
        for (uint32_t position = 1; position <= 13; ++position)
            for (int style = 0; style < 3; ++style) {
                Asm a;
                Label after = a.NewLabel();
                a << mflr(0) << stw(0, 0, 1);
                for (uint32_t k = 1; k <= 13; ++k) a << lfd(k, 0x100 + static_cast<int32_t>(8 * k), 30);  // scalar form
                a << ppc::ps_merge10(position, 14 + position % 18, 15 + position % 17);                      // the one paired register
                if (style == 2) a.LiLabel(5, after) << mtlr(5);
                if (style == 0) a.BAbs(target, true);
                else if (style == 1) a.Li32(8, target), a << mtctr(8) << bctrl();
                else a.BAbs(target);
                a.Bind(after);
                ObserveFprs(a, 0x40);
                a << lwz(0, 0, 1) << mtlr(0) << blr();
                std::vector<CpuContext> states;
                for (int st = 0; st < 3; ++st) {
                    CpuContext c = BaseState(rng);
                    c.gpr[3] = rng.Next();
                    states.push_back(c);
                }
                AddProgram(Fmt("thin arg f%u of function %u, %s", position, i, kCallStyle[style]), a, states);
            }
    }
}

// One bctrl or bctr site cycling through more targets than a cell holds entries: translated functions of every kind, patched ones
// (translated, and with `patched` interpreted at guest code), leaf code, in several orders (a short run of one target, a cycle of
// five, of eight, one that returns to the first after a long gap).
void IndirectCycles(bool patched) {
    Rng rng(26);
    for (int style = 0; style < 2; ++style)
        for (int pattern = 0; pattern < 5; ++pattern) {
            Asm a;
            Label table = a.NewLabel(), loop = a.NewLabel(), done = a.NewLabel(), leaf = a.NewLabel();
            a << mflr(0) << stw(0, 0, 1);
            a.LiLabel(4, table) << stw(4, 4, 1) << li(5, 24) << stw(5, 8, 1);
            a.Bind(loop);
            a << lwz(5, 8, 1) << cmpwi(0, 5, 0);
            a.Bc(12, 2, done);
            a << addi(5, 5, -1) << stw(5, 8, 1) << lwz(4, 4, 1) << lwz(6, 0, 4) << addi(4, 4, 4) << stw(4, 4, 1);
            PairedSetup(a, 7);
            if (style == 0) {
                a << mtctr(6) << bctrl() << addi(10, 10, 1);
                a.B(loop);
            } else {
                Label back = a.NewLabel();
                a.LiLabel(7, back) << mtlr(7) << mtctr(6) << bctr();
                a.Bind(back);
                a.B(loop);
            }
            a.Bind(done);
            ObserveFprs(a, 0x40);
            a << lwz(0, 0, 1) << mtlr(0) << blr();
            a.Bind(leaf);
            a << addi(3, 3, 19) << ori(12, 3, 0x70) << blr();
            const size_t x0At = a.Here();
            a << addi(3, 3, 33) << ppc::ps_merge10(1, 2, 3) << blr();
            const size_t x1At = a.Here();
            a << mflr(9) << addi(3, 3, 43) << blr();
            a.Bind(table);
            const size_t tableIndex = a.Here();
            for (int e = 0; e < 26; ++e) a << 0u;
            Scenario s;
            s.image = a.Finish();
            const size_t leafAt = static_cast<size_t>(std::find(s.image.words.begin(), s.image.words.end(), addi(3, 3, 19)) - s.image.words.begin());
            s.runs = [=](uint32_t base, std::vector<RunSpec>& runs) {
                Rng local(70 + pattern * 5 + style + (patched ? 100 : 0));
                for (int round = 0; round < 3; ++round) {
                    RunSpec spec;
                    spec.ctx = BaseState(local);
                    spec.ctx.gpr[3] = local.Next();
                    // eight distinct targets: functions 0-4 translated (several kinds and masks), two patched (kPatchedBase 0, 1),
                    // and leaf code; index 5 and 6 are the patched ones
                    const uint32_t leafAddress = base + 4u * static_cast<uint32_t>(leafAt);
                    const uint32_t targets[8] = {kThinBase + kThinStride * (1 + local.Below(20)), kThinBase + kThinStride * (21 + local.Below(20)),
                                                 kThinBase + kThinStride * 2, kThinBase + kThinStride * 3, kThinBase + kThinStride * 4,
                                                 PatchedAddress(0), PatchedAddress(1), leafAddress};
                    static const int plans[5][24] = {
                        {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6},
                        {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3},
                        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
                        {0, 7, 7, 7, 7, 7, 7, 7, 1, 7, 7, 7, 7, 7, 7, 0, 7, 7, 7, 7, 7, 7, 7, 1},
                        {5, 6, 5, 6, 7, 5, 6, 4, 5, 6, 3, 5, 6, 2, 5, 6, 1, 5, 6, 0, 7, 7, 5, 6}};
                    for (int e = 0; e < 24; ++e)
                        spec.writes.push_back({static_cast<uint32_t>(4 * (tableIndex + e)), targets[plans[pattern][e]]});
                    if (patched) {  // the patched functions are interpreted (at guest code of the program), the third runs translated
                        const uint32_t a0 = base + 4u * static_cast<uint32_t>(x0At), a1 = base + 4u * static_cast<uint32_t>(x1At);
                        spec.before = [a0, a1] {
                            ResetPatched();
                            GuestCodePatch::TestSetInterpreted(PatchedAddress(0), a0);
                            GuestCodePatch::TestSetInterpreted(PatchedAddress(1), a1);
                        };
                    } else {
                        spec.before = [] { ResetPatched(); };
                    }
                    runs.push_back(std::move(spec));
                }
            };
            RunCase(Fmt("%s indirect cycles %s pattern %d", patched ? "patched" : "thin", style == 0 ? "bctrl" : "bctr", pattern), s);
        }
}

// The lean entry (RunPatched's) and the general one agree: functions entered from C++ again and again (the harness alternates the
// two), at a pc the lean entry declines (a hook pc of the dispatcher: 0x800171B4 is the rollback loop's, which does nothing in a
// process without a rollback session) and at ordinary ones, leaving through links, through a tail call, through an exception.
void LeanEntryHooks() {
    Rng rng(28);
    for (int style = 0; style < 3; ++style) {
        Asm a;
        a << mflr(0) << stw(0, 0, 1) << addi(3, 3, 1) << cmpwi(0, 3, 1000);
        Label skip = a.NewLabel();
        a.Bc(12, 0, skip);
        a << addi(3, 3, 5);
        a.Bind(skip);
        if (style == 0) {
            a.BAbs(kFnModify, true);
        } else if (style == 1) {
            a << addi(4, 4, 3) << lwz(0, 0, 1) << mtlr(0);
            a.BAbs(kFnModify);  // a tail call
        } else {
            a.BAbs(kThrower, true);
        }
        a << lwz(0, 0, 1) << mtlr(0) << blr();
        Scenario s;
        s.image = a.Finish();
        s.runs = [=](uint32_t, std::vector<RunSpec>& runs) {
            Rng local(90 + style);
            for (int round = 0; round < 6; ++round) {
                RunSpec spec;
                spec.ctx = BaseState(local);
                spec.ctx.gpr[3] = local.Below(2000);
                runs.push_back(std::move(spec));
            }
        };
        RunCase(Fmt("lean entry, ordinary pc, style %d", style), s);
        // the same program placed at the dispatcher's hook pc
        ++g_cases;
        for (const EngineConfig& config : kConfigs) {
            const uint32_t base = 0x800171B4u;
            const std::vector<Snap> ref = Execute(kReference, s, base);
            const std::vector<Snap> got = Execute(config, s, base);
            for (size_t i = 0; i < ref.size(); ++i) {
                const std::string diff = Diff(ref[i], got[i]);
                if (diff.empty()) continue;
                ++g_failures;
                std::fprintf(stderr, "FAIL lean entry at a hook pc, style %d: engine %s, run %zu\n%s", style, config.name, i + 1, diff.c_str());
                break;
            }
        }
    }
}

void CallTests() {
    ThinCalls();
    IndirectSites(false);
    IndirectSites(true);
    for (int depth : {1, 2, 3, 7, 8, 9, 12})
        for (int ret = 0; ret < 4; ++ret) PatchedChain(depth, ret, false);
    for (int ret = 0; ret < 4; ++ret) PatchedChain(3, ret, true);
    ThinArgPositions();
    IndirectCycles(false);
    IndirectCycles(true);
    LeanEntryHooks();
    PatchedExceptions();
    PatchedReenter();
    PatchedSelfModifying();
    SubFrameCalls();
}

// Loops: counted bdnz loops of several lengths, and a compare-and-branch loop, from several trip counts.
void Loops() {
    Rng rng(12);
    for (int body : {0, 1, 5, 20, 40}) {
        Asm a;
        Label loop = a.NewLabel();
        a << mtctr(3);
        a.Bind(loop);
        for (int i = 0; i < body; ++i) a << addi(6, 6, 1);
        a << addi(4, 4, 1);
        a.Bc(16, 0, loop);
        a << blr();
        std::vector<CpuContext> states;
        for (uint32_t n : {1u, 2u, 3u, 10u, 33u, 100u}) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = n;
            c.gpr[4] = 0;
            states.push_back(c);
        }
        AddProgram(Fmt("bdnz loop, body %d", body), a, states);
    }
    {  // a b back edge (a block end that stays a B: its target is code of the same block)
        Asm a;
        Label loop = a.NewLabel(), done = a.NewLabel();
        a.Bind(loop);
        a << addi(4, 4, 1) << cmplw(0, 4, 3);
        a.Bc(12, 0 + 1, done);
        a.B(loop);
        a.Bind(done);
        a << blr();
        std::vector<CpuContext> states;
        for (uint32_t n : {1u, 5u, 40u}) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = n;
            c.gpr[4] = 0;
            states.push_back(c);
        }
        AddProgram("b back edge loop", a, states);
    }
    {
        Asm a;
        Label loop = a.NewLabel();
        a << li(5, 0);
        a.Bind(loop);
        a << addi(5, 5, 3) << cmplw(1, 5, 3);
        a.Bc(12, 4 * 1 + 0, loop);
        a << blr();
        std::vector<CpuContext> states;
        for (uint32_t n : {1u, 2u, 30u, 300u}) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = n;
            states.push_back(c);
        }
        AddProgram("compare loop", a, states);
    }
}

// --- the native-code and linking checks ---------------------------------------------------------------------------------
int g_checks = 0;
#define CHECK(cond)                                                                                          \
    do {                                                                                                     \
        if (!(cond)) {                                                                                       \
            std::fprintf(stderr, "jit branch: line %d: check failed: %s\n", __LINE__, #cond);                \
            ++g_failures;                                                                                    \
        }                                                                                                    \
        ++g_checks;                                                                                          \
    } while (0)

uint32_t RunJit(uint32_t entry, CpuContext& ctx) {
    PpcInterpreter::SetEngineOverride(Engine::Jit);
    PpcInterpreter::SetJitThreshold(0);
    CpuContextScope scope(&ctx);
    PpcInterpreter::Run(&ctx, entry);
    PpcInterpreter::ClearEngineOverride();
    return ctx.gpr[3];
}

// The calls between patched functions are made in subcall frames: a chain entered through the lean entry (a jit_enter_direct frame)
// or through Run (a jit_enter frame) calls no wrapper once its blocks are compiled and linked, while the guard bytes stay set; an
// unpatched function and a redirect bound below the targets go through the wrappers again.
void SubFramesUsed() {
    const uint32_t base = kSlotBase + (kSlotCount + 4) * kSlotBytes;
    for (int style : {0, 10}) {
        Asm a;
        a << mflr(0) << stw(0, 0, 1);
        if (style == 10) a.Li32(8, SubAddress(0)), a << mtctr(8) << bctrl();
        else a.BAbs(SubAddress(0), true);
        a << lwz(0, 0, 1) << mtlr(0) << blr();
        WriteImage(a.Finish(), base);
        for (uint32_t k = 0; k < 3; ++k) WriteImage(SubLevel(k, 3, style, false), SubAddress(k));
        PpcInterpreter::SetEngineOverride(Engine::Jit);
        PpcInterpreter::SetJitThreshold(0);
        const auto runs = [&](int n, uint32_t unpatched, bool redirect, bool lean) {
            const uint64_t before = g_subWrapperRuns;
            for (int i = 0; i < n; ++i) {
                SubState(unpatched, redirect);
                CpuContext ctx{};
                ctx.lr = kReturn;
                ctx.gpr[1] = kDataBase + 0x1800u;
                CpuContextScope scope(&ctx);
                if (!lean || !PpcInterpreter::detail::RunCompiledEntry(&ctx, base)) PpcInterpreter::Run(&ctx, base);
            }
            return g_subWrapperRuns - before;
        };
        runs(4, 0, false, true);  // compiled and linked
        CHECK(runs(10, 0, false, true) == 0);
        const uint64_t translations = g_subTranslationRuns;
        CHECK(runs(10, 2, false, true) == 0);    // function 1 unpatched: its translation, function 2 not called
        CHECK(g_subTranslationRuns - translations == 10);
        CHECK(runs(10, 0, true, true) == 30);    // the redirect bound: every call through its wrapper
        CHECK(runs(10, 0, false, false) == 0);   // Run's frame (a jit_enter frame) calls in subcall frames too
        CHECK(runs(10, 0, false, true) == 0);
        PpcInterpreter::ClearEngineOverride();
    }
    SubState(0, false);
}

void NativeAndLinking() {
    using PpcInterpreter::detail::JitX64Stats;
    // Every kind of the group compiles natively in a program that holds it (and the sum of fallbacks of these kinds stays
    // where the differential tests left it: no fallback is needed for a plain program of this kind).
    const uint32_t base = kSlotBase + (kSlotCount + 2) * kSlotBytes;
    Asm a;
    Label loop = a.NewLabel(), leaf = a.NewLabel(), done = a.NewLabel(), next1 = a.NewLabel(), next2 = a.NewLabel(),
          next3 = a.NewLabel(), next4 = a.NewLabel();
    a << mflr(31) << li(3, 0) << li(4, 50) << mtctr(4);
    a.Bind(loop);
    a.B(leaf, true);                      // Bl to untranslated code (a leaf: the inline case)
    a << cmpwi(0, 3, 1000);
    a.Bc(12, 1, done);                    // CmpwiBc
    a.Bind(next1);
    a << cmplwi(1, 3, 5);
    a.Bc(12, 4 + 2, next2);               // CmplwiBc (to the next word either way)
    a.Bind(next2);
    a << cmpw(0, 3, 4);
    a.Bc(12, 2, next3);                   // CmpwBc
    a.Bind(next3);
    a << cmplw(0, 3, 4);
    a.Bc(12, 2, next4);                   // CmplwBc
    a.Bind(next4);
    a << rlwinm(5, 3, 0, 0, 31, 1);
    a.Bc(12, 2, done);                    // RlwinmRcBc: r3 is never zero here
    a.Bc(16, 0, loop);                    // Bc: bdnz
    a.Bind(done);
    a << mtlr(31) << blr();               // Blr
    a.Bind(leaf);
    a << addi(3, 3, 1) << blr();
    Scenario s;
    s.image = a.Finish();
    const std::vector<uint32_t> words = s.image.Resolve(base);
    for (size_t i = 0; i < words.size(); ++i) Memory::Write32(base + 4u * static_cast<uint32_t>(i), words[i]);
    CpuContext ctx{};
    ctx.lr = kReturn;
    ctx.gpr[1] = kDataBase + 0x1800u;
    // Bc (not fused) with a bdnz: the ctr loop ends the program
    CpuContext c0 = ctx;
    const uint32_t r3 = RunJit(base, c0);
    CHECK(r3 == 50);
    for (K k : {K::Bl, K::Blr, K::CmpwiBc, K::CmplwiBc, K::CmpwBc, K::CmplwBc, K::RlwinmRcBc})
        CHECK(JitX64KindOps(k, true) > 0);
    // And none of the group's kinds was compiled as a fallback anywhere in this process (the differential tests ran first in
    // a different process; here every kind the programs above used must have had native code).
    for (K k : {K::Bc, K::B, K::Bl, K::Blr, K::Bclr, K::Bctr, K::Bctrl, K::CmpwiBc, K::CmplwiBc, K::CmpwBc, K::CmplwBc, K::RlwinmRcBc}) {
        const uint64_t native = JitX64KindOps(k, true), fallback = JitX64KindOps(k, false);
        std::printf("  %-10s compiled native %6llu  as fallback %4llu\n", PpcInterpreter::detail::kKindNames[static_cast<unsigned>(k)],
                    static_cast<unsigned long long>(native), static_cast<unsigned long long>(fallback));
        CHECK(native > 0);
    }
}

// A hot loop must not return to the dispatcher per iteration, and calls and returns inside it must not either.
void HotLoopsStayCompiled() {
    using PpcInterpreter::detail::JitX64Stats;
    const uint32_t base = kSlotBase + (kSlotCount + 4) * kSlotBytes;
    struct Case {
        const char* name;
        std::function<void(Asm&)> build;
        uint32_t maxEnters;
    };
    const Case cases[] = {
        {"counted loop",
         [](Asm& a) {
             Label loop = a.NewLabel();
             a << li(4, 0) << mtctr(3);
             a.Bind(loop);
             a << addi(4, 4, 1);
             a.Bc(16, 0, loop);
             a << blr();
         },
         4},
        {"call loop (bl to a leaf, blr)",
         [](Asm& a) {
             Label loop = a.NewLabel(), leaf = a.NewLabel();
             a << mflr(31) << li(4, 0) << mtctr(3);
             a.Bind(loop);
             a.B(leaf, true);
             a.Bc(16, 0, loop);
             a << mtlr(31) << blr();
             a.Bind(leaf);
             a << addi(4, 4, 1) << blr();
         },
         8},
        {"call loop (bctrl to two leaves, blr)",
         [](Asm& a) {
             Label loop = a.NewLabel(), leaf = a.NewLabel(), other = a.NewLabel(), skip = a.NewLabel();
             a << mflr(31) << li(4, 0) << li(5, 0);
             a.Bind(loop);
             a.LiLabel(7, leaf);
             a << rlwinm(0, 3, 0, 31, 31, 1);
             a.Bc(12, 2, skip);
             a.LiLabel(7, other);
             a.Bind(skip);
             a << mtctr(7) << bctrl() << addi(3, 3, -1) << cmpwi(0, 3, 0);
             a.Bc(4, 2, loop);
             a << mtlr(31) << blr();
             a.Bind(leaf);
             a << addi(4, 4, 1) << blr();
             a.Bind(other);
             a << addi(5, 5, 1) << blr();
         },
         8},
        {"jump table loop (bctr, 4 targets)",
         [](Asm& a) {
             Label loop = a.NewLabel(), table = a.NewLabel(), join = a.NewLabel();
             a << li(4, 0) << mtctr(3);
             a.Bind(loop);
             a.LiLabel(7, table);
             a << rlwinm(5, 3, 3, 27, 28) << add(7, 7, 5) << addi(3, 3, 1) << mtctr(7) << bctr();
             a.Bind(table);
             for (int c = 0; c < 4; ++c) {
                 a << addi(4, 4, c + 1);
                 a.B(join);
             }
             a.Bind(join);
             a << cmplwi(0, 3, 4000);
             a.Bc(12, 0, loop);
             a << blr();
         },
         8},
    };
    uint32_t slot = 0;
    for (const Case& c : cases) {
        Asm a;
        c.build(a);
        const std::vector<uint32_t> words = a.Finish().Resolve(base + slot * kSlotBytes);
        for (size_t i = 0; i < words.size(); ++i) Memory::Write32(base + slot * kSlotBytes + 4u * static_cast<uint32_t>(i), words[i]);
        CpuContext ctx{};
        ctx.lr = kReturn;
        ctx.gpr[1] = kDataBase + 0x1800u;
        ctx.gpr[3] = 1000;
        CpuContext again = ctx;
        CpuContext classic = ctx;
        RunJit(base + slot * kSlotBytes, ctx);  // compiles and links
        const JitX64Stats before = JitX64GetStats();
        RunJit(base + slot * kSlotBytes, again);
        const JitX64Stats after = JitX64GetStats();
        {  // the same run under the reference
            PpcInterpreter::SetEngineOverride(Engine::Classic);
            CpuContextScope scope(&classic);
            PpcInterpreter::Run(&classic, base + slot * kSlotBytes);
            PpcInterpreter::ClearEngineOverride();
        }
        CHECK(classic.gpr[3] == again.gpr[3] && classic.gpr[4] == again.gpr[4] && classic.ctr == again.ctr);
        std::printf("    result r3=%u r4=%u r5=%x r7=%x ctr=%x\n", again.gpr[3], again.gpr[4], again.gpr[5], again.gpr[7], again.ctr);
        const uint64_t enters = after.enters - before.enters;
        std::printf("  %-36s %llu dispatcher enters for the run, inline cache %llu hits %llu misses\n", c.name,
                    static_cast<unsigned long long>(enters), static_cast<unsigned long long>(after.cacheHits - before.cacheHits),
                    static_cast<unsigned long long>(after.cacheMisses - before.cacheMisses));
        CHECK(enters <= c.maxEnters);
        ++slot;
    }
}

// Regression tests for intra-block GPR caching (subf aliasing o.d == o.a; fused cmpw+bc with both useCtr and useCond
// when the CR field is dead and the second compare operand is read later) and direct jmp rel32 link/retire.
void GprCacheAndDirectLinkRegressions() {
    Rng rng(1337);
    {
        Asm a;
        Label skip = a.NewLabel();
        // subf r3, r3, r4 -> r3 = r4 - r3 (d == a, both cached), followed by cmpw cr3, r5, r6 + bc 10, 14 (useCtr + useCond)
        // with cr3 overwritten immediately after (dead) and r6 read afterwards.
        a << addi(3, 3, 1) << addi(4, 4, 2) << X(3, 3, 4, 40) << addi(5, 5, 3) << cmpw(3, 5, 6);
        a.Bc(10, 14, skip);
        a << cmpwi(3, 7, 0) << add(3, 3, 6) << add(3, 3, 5) << blr();
        a.Bind(skip);
        a << add(3, 3, 6) << blr();
        std::vector<CpuContext> states;
        for (uint32_t ctr : {1u, 2u, 3u}) {
            for (int eq = 0; eq < 2; ++eq) {
                CpuContext c = BaseState(rng);
                c.ctr = ctr;
                c.gpr[5] = 10;
                c.gpr[6] = eq ? 13u : 99u;  // after addi(5,5,3), r5 == 13
                states.push_back(c);
            }
        }
        AddProgram("subf d==a and fused cmpw+bc useCtr+useCond GPR cache", a, states);
    }
    {
        // Direct jmp rel32 link and retire: run a linked jump, rewrite the target block to trigger Retire, run again.
        const uint32_t base = kSlotBase + (kSlotCount + 8) * kSlotBytes;
        const uint32_t targetPc = base + 0x40u;
        Asm entryAsm;
        entryAsm << addi(3, 3, 1);
        entryAsm.BAbs(targetPc);
        const std::vector<uint32_t> entryWords = entryAsm.Finish().Resolve(base);
        for (size_t i = 0; i < entryWords.size(); ++i) Memory::Write32(base + 4u * static_cast<uint32_t>(i), entryWords[i]);
        Memory::Write32(targetPc + 0, addi(3, 3, 10));
        Memory::Write32(targetPc + 4, blr());
        CpuContext ctx{};
        ctx.lr = kReturn;
        ctx.gpr[1] = kDataBase + 0x1800u;
        ctx.gpr[3] = 5;
        CHECK(RunJit(base, ctx) == 16);
        ctx.gpr[3] = 5;
        CHECK(RunJit(base, ctx) == 16);  // linked direct jmp rel32
        // Rewrite targetPc so Retire() must unpatch the direct jmp rel32 back to its stub
        Memory::Write32(targetPc + 0, addi(3, 3, 100));
        ctx.gpr[3] = 5;
        CHECK(RunJit(base, ctx) == 106);
        ctx.gpr[3] = 5;
        CHECK(RunJit(base, ctx) == 106);  // re-linked to rebuilt target
    }
    for (int lone = 0; lone < 2; ++lone) {
        // Direct conditional link (jcc rel32): a taken bc (fused with its compare, or alone after an unrelated op) jumps straight to
        // the target's block; rewriting the target must unpatch the jcc (Retire) and the next run links the rebuilt block.
        const uint32_t base = kSlotBase + (kSlotCount + 9 + static_cast<uint32_t>(lone)) * kSlotBytes;
        const uint32_t targetPc = base + 0x40u;
        const auto bc = [](uint32_t at, uint32_t to) { return (16u << 26) | (12u << 21) | (2u << 16) | ((to - at) & 0xFFFCu); };
        uint32_t at = base;
        Memory::Write32(at, addi(3, 3, 1)), at += 4;
        Memory::Write32(at, cmpwi(0, 4, 0)), at += 4;
        if (lone) Memory::Write32(at, addi(5, 5, 1)), at += 4;
        Memory::Write32(at, bc(at, targetPc)), at += 4;
        Memory::Write32(at, addi(3, 3, 2)), at += 4;
        Memory::Write32(at, blr());
        Memory::Write32(targetPc + 0, addi(3, 3, 10));
        Memory::Write32(targetPc + 4, blr());
        const auto run = [&](uint32_t r3, uint32_t r4) {
            CpuContext ctx{};
            ctx.lr = kReturn;
            ctx.gpr[1] = kDataBase + 0x1800u;
            ctx.gpr[3] = r3;
            ctx.gpr[4] = r4;
            return RunJit(base, ctx);
        };
        CHECK(run(5, 0) == 16);
        CHECK(run(5, 0) == 16);  // the taken bc linked: jcc rel32 to the target block
        CHECK(run(5, 1) == 8);
        Memory::Write32(targetPc + 0, addi(3, 3, 100));
        CHECK(run(5, 0) == 106);
        CHECK(run(5, 0) == 106);  // re-linked to the rebuilt target
        CHECK(run(5, 1) == 8);
    }
}

}  // namespace

int main() {
    // JIT_BRANCH_DIRECT=1 (mkw_jit_branch_direct_tests): without BRAWL_JIT_STATS, which keeps the lean entry off jit_enter_direct, so
    // that the programs run in direct and subcall frames as in the game; the parts that read the statistics are left out.
    const char* direct = std::getenv("JIT_BRANCH_DIRECT");
    const bool stats = direct == nullptr || *direct == '\0' || *direct == '0';
    if (stats) setenv("BRAWL_JIT_STATS", "1", 1);  // the inline cache's hit and miss counters are emitted only with it
    Memory::Init(0x200000);
    RegisterFunction(kFnModify, "BranchModify", FnModify);
    RegisterFunction(kFnWrite, "BranchWrite", FnWrite);
    RegisterSubFrameHooks();
    RegisterCallTestFunctions();
    RegisterSubFrameFunctions();
    TranslatedFunctionRegistry::Finalize();
    if (!PpcInterpreter::JitAvailable()) {
        std::printf("jit branch: no JIT backend on this host, nothing to test\n");
        return 0;
    }
    BcMatrix();
    BclrMatrix();
    FusedMatrix();
    FusedChains();
    ReturnNormalisation();
    JumpTables();
    BctrTranslatedAndNot();
    TailCalls();
    Calls();
    CallTests();
    SelfModifying();
    Loops();
    if (stats) {
        NativeAndLinking();
        HotLoopsStayCompiled();
    } else {
        SubFramesUsed();
    }
    GprCacheAndDirectLinkRegressions();
    std::printf("jit branch: %u programs, %llu guest runs, %d checks, %u failures\n", g_cases, static_cast<unsigned long long>(g_runs),
                g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
