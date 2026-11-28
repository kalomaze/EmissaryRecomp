// The guest-code engines against each other: the same guest programs run under the classic interpreter (Exec, one
// instruction at a time: the reference), the cached interpreter and the JIT (threshold 0, so a block is compiled before
// its first run, and a small threshold, so its first runs are interpreted and the later ones compiled) in one process,
// from the same registers and the same guest memory, and every register, every CR field's 64-bit value (bytewise: the JIT stores
// the interpreter's exact values, jit_test_cr.h), every FPR's raw bits and the guest memory the programs may write must come out
// equal. Where this build has no JIT backend (PpcInterpreter::JitAvailable), Engine::Jit
// runs the cached interpreter and the test says so; it needs no change when a backend exists.
//
// Layout of the file: instruction encoders and a tiny assembler; the machine state, the sandbox and the comparison; the
// registered functions and the support code programs call; the directed programs (integer, memory, floating point,
// paired singles), control flow, calls into registered functions, stale code; the random programs; main.
//
// Every program runs several times per engine without anything cleared between the runs (the block cache and the
// compiled code persist, so second and later runs take the cached and linked paths), each from a fresh register state
// and a fresh sandbox. A program is placed at a code address no earlier program of that engine used (the slots recycle
// only after many programs), so a block's first runs really are its first.
#include "abi_bridge.h"
#include "brawl_dirty_memory.h"
#include "brawl_write_watch.h"
#include "isa/ppc_isa_fpenv.h"
#include "jit_test_cr.h"
#include "lean_window.h"
#include "memory.h"
#include "ppc_interpreter.h"
#include "../src/ppc_interpreter_internal.h"
#include "../src/jit/jit_x64.h"
#include "recomp_mod_loader.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using PpcInterpreter::Engine;

// --- Guest memory layout (MEM1 is 2 MiB here) ---------------------------------------------------------------------
//
// The sandbox is the only data memory a program may write: its lower 4 KiB is input, the upper 4 KiB holds probes (the
// directed programs store each result there so a wrong value is seen even when a later instruction overwrites its
// register). Code never shares a 64-byte dirty-mark block with the sandbox. No code lies at a runtime hook address
// (0x80017xxx, 0x801BC684, 0x801C9xxx): those are not the subject here.
constexpr uint32_t kDataBase = 0x80030000u;
constexpr uint32_t kDataSize = 0x2000u;
constexpr uint32_t kProbeOffset = 0x1000u;      // r30 holds kDataBase + kProbeOffset in the directed programs
constexpr uint32_t kReturn = 0x80020000u;       // the link address programs return to: no code there
constexpr uint32_t kFuncBase = 0x80010000u;     // registered C++ functions, one every 0x10 bytes (no code there)
constexpr uint32_t kSupportPlain = 0x80012000u; // guest code a registered function runs re-entrantly
constexpr uint32_t kSupportThrow = 0x80012100u;
constexpr uint32_t kSlotBase = 0x80100000u;     // program images, one per slot
constexpr uint32_t kSlotBytes = 0x2000u;
constexpr uint32_t kSlotCount = 80;

// The JIT threshold the interpreter starts with (PpcInterpreter has no getter, and the test sets every threshold it uses).
constexpr uint32_t kDefaultJitThreshold = 2;

struct EngineConfig {
    const char* name;
    Engine engine;
    uint32_t threshold;
};
const EngineConfig kReference = {"classic", Engine::Classic, 0};
const EngineConfig kConfigs[] = {
    {"cached", Engine::Cached, 0},
    {"jit-threshold-0", Engine::Jit, 0},
    {"jit-default-threshold", Engine::Jit, kDefaultJitThreshold},
};

// --- PowerPC instruction encoders ---------------------------------------------------------------------------------
namespace ppc {

constexpr uint32_t D(uint32_t op, uint32_t rt, uint32_t ra, int32_t imm) {
    return (op << 26) | (rt << 21) | (ra << 16) | (static_cast<uint32_t>(imm) & 0xFFFFu);
}
// Primary opcode 31: the first register at bit 21, the second at 16, the third at 11.
constexpr uint32_t X(uint32_t rt, uint32_t ra, uint32_t rb, uint32_t xo, uint32_t rc = 0) {
    return (31u << 26) | (rt << 21) | (ra << 16) | (rb << 11) | (xo << 1) | rc;
}

constexpr uint32_t addi(uint32_t rt, uint32_t ra, int32_t i) { return D(14, rt, ra, i); }
constexpr uint32_t addis(uint32_t rt, uint32_t ra, int32_t i) { return D(15, rt, ra, i); }
constexpr uint32_t addic(uint32_t rt, uint32_t ra, int32_t i) { return D(12, rt, ra, i); }
constexpr uint32_t addic_(uint32_t rt, uint32_t ra, int32_t i) { return D(13, rt, ra, i); }
constexpr uint32_t subfic(uint32_t rt, uint32_t ra, int32_t i) { return D(8, rt, ra, i); }
constexpr uint32_t mulli(uint32_t rt, uint32_t ra, int32_t i) { return D(7, rt, ra, i); }
constexpr uint32_t li(uint32_t rt, int32_t i) { return addi(rt, 0, i); }
constexpr uint32_t lis(uint32_t rt, int32_t i) { return addis(rt, 0, i); }
constexpr uint32_t ori(uint32_t ra, uint32_t rs, uint32_t u) { return D(24, rs, ra, static_cast<int32_t>(u)); }
constexpr uint32_t oris(uint32_t ra, uint32_t rs, uint32_t u) { return D(25, rs, ra, static_cast<int32_t>(u)); }
constexpr uint32_t xori(uint32_t ra, uint32_t rs, uint32_t u) { return D(26, rs, ra, static_cast<int32_t>(u)); }
constexpr uint32_t xoris(uint32_t ra, uint32_t rs, uint32_t u) { return D(27, rs, ra, static_cast<int32_t>(u)); }
constexpr uint32_t andi_(uint32_t ra, uint32_t rs, uint32_t u) { return D(28, rs, ra, static_cast<int32_t>(u)); }
constexpr uint32_t andis_(uint32_t ra, uint32_t rs, uint32_t u) { return D(29, rs, ra, static_cast<int32_t>(u)); }
constexpr uint32_t cmpwi(uint32_t crf, uint32_t ra, int32_t i) { return D(11, crf << 2, ra, i); }
constexpr uint32_t cmplwi(uint32_t crf, uint32_t ra, uint32_t u) { return D(10, crf << 2, ra, static_cast<int32_t>(u)); }
constexpr uint32_t cmpw(uint32_t crf, uint32_t ra, uint32_t rb) { return X(crf << 2, ra, rb, 0); }
constexpr uint32_t cmplw(uint32_t crf, uint32_t ra, uint32_t rb) { return X(crf << 2, ra, rb, 32); }
// XO-form arithmetic: xo is the 9-bit extended opcode, OE the bit above it.
constexpr uint32_t arith(uint32_t xo, uint32_t rt, uint32_t ra, uint32_t rb, uint32_t oe, uint32_t rc) {
    return X(rt, ra, rb, xo | (oe << 9), rc);
}
constexpr uint32_t mr(uint32_t ra, uint32_t rs) { return X(rs, ra, rs, 444); }
// Logical and shift forms: the source register rs is at bit 21, the destination ra at 16.
constexpr uint32_t logic(uint32_t xo, uint32_t ra, uint32_t rs, uint32_t rb, uint32_t rc = 0) { return X(rs, ra, rb, xo, rc); }
constexpr uint32_t srawi(uint32_t ra, uint32_t rs, uint32_t sh, uint32_t rc = 0) { return X(rs, ra, sh, 824, rc); }
constexpr uint32_t cntlzw(uint32_t ra, uint32_t rs, uint32_t rc = 0) { return X(rs, ra, 0, 26, rc); }
constexpr uint32_t extsb(uint32_t ra, uint32_t rs, uint32_t rc = 0) { return X(rs, ra, 0, 954, rc); }
constexpr uint32_t extsh(uint32_t ra, uint32_t rs, uint32_t rc = 0) { return X(rs, ra, 0, 922, rc); }
constexpr uint32_t rlwinm(uint32_t ra, uint32_t rs, uint32_t sh, uint32_t mb, uint32_t me, uint32_t rc = 0) {
    return (21u << 26) | (rs << 21) | (ra << 16) | (sh << 11) | (mb << 6) | (me << 1) | rc;
}
constexpr uint32_t rlwimi(uint32_t ra, uint32_t rs, uint32_t sh, uint32_t mb, uint32_t me, uint32_t rc = 0) {
    return (20u << 26) | (rs << 21) | (ra << 16) | (sh << 11) | (mb << 6) | (me << 1) | rc;
}
constexpr uint32_t rlwnm(uint32_t ra, uint32_t rs, uint32_t rb, uint32_t mb, uint32_t me, uint32_t rc = 0) {
    return (23u << 26) | (rs << 21) | (ra << 16) | (rb << 11) | (mb << 6) | (me << 1) | rc;
}
constexpr uint32_t mfspr(uint32_t rt, uint32_t spr) { return X(rt, spr & 31u, spr >> 5, 339); }
constexpr uint32_t mtspr(uint32_t spr, uint32_t rs) { return X(rs, spr & 31u, spr >> 5, 467); }
constexpr uint32_t mflr(uint32_t rt) { return mfspr(rt, 8); }
constexpr uint32_t mtlr(uint32_t rs) { return mtspr(8, rs); }
constexpr uint32_t mfctr(uint32_t rt) { return mfspr(rt, 9); }
constexpr uint32_t mtctr(uint32_t rs) { return mtspr(9, rs); }
constexpr uint32_t mfxer(uint32_t rt) { return mfspr(rt, 1); }
constexpr uint32_t mtxer(uint32_t rs) { return mtspr(1, rs); }
constexpr uint32_t mfcr(uint32_t rt) { return X(rt, 0, 0, 19); }
constexpr uint32_t mtcrf(uint32_t crm, uint32_t rs) { return (31u << 26) | (rs << 21) | (crm << 12) | (144u << 1); }
constexpr uint32_t mcrf(uint32_t crfd, uint32_t crfs) { return (19u << 26) | (crfd << 23) | (crfs << 18); }
constexpr uint32_t mcrxr(uint32_t crf) { return X(crf << 2, 0, 0, 512); }
constexpr uint32_t mfmsr(uint32_t rt) { return X(rt, 0, 0, 83); }
constexpr uint32_t mtmsr(uint32_t rs) { return X(rs, 0, 0, 146); }
constexpr uint32_t crop(uint32_t xo, uint32_t bt, uint32_t ba, uint32_t bb) {
    return (19u << 26) | (bt << 21) | (ba << 16) | (bb << 11) | (xo << 1);
}
// Branches through lr and ctr (bo, bi as the instruction's fields).
constexpr uint32_t bclr(uint32_t bo, uint32_t bi, uint32_t lk = 0) { return (19u << 26) | (bo << 21) | (bi << 16) | (16u << 1) | lk; }
constexpr uint32_t bcctr(uint32_t bo, uint32_t bi, uint32_t lk = 0) { return (19u << 26) | (bo << 21) | (bi << 16) | (528u << 1) | lk; }
constexpr uint32_t blr() { return bclr(20, 0); }
constexpr uint32_t blrl() { return bclr(20, 0, 1); }
constexpr uint32_t bctr() { return bcctr(20, 0); }
constexpr uint32_t bctrl() { return bcctr(20, 0, 1); }
constexpr uint32_t nop() { return ori(0, 0, 0); }
// Loads and stores: D-form by primary opcode, X-form by extended opcode.
constexpr uint32_t ld(uint32_t op, uint32_t rt, int32_t d, uint32_t ra) { return D(op, rt, ra, d); }
constexpr uint32_t ldx(uint32_t xo, uint32_t rt, uint32_t ra, uint32_t rb) { return X(rt, ra, rb, xo); }
constexpr uint32_t lwz(uint32_t rt, int32_t d, uint32_t ra) { return D(32, rt, ra, d); }
constexpr uint32_t stw(uint32_t rs, int32_t d, uint32_t ra) { return D(36, rs, ra, d); }
constexpr uint32_t stfd(uint32_t fs, int32_t d, uint32_t ra) { return D(54, fs, ra, d); }
constexpr uint32_t lswi(uint32_t rt, uint32_t ra, uint32_t nb) { return X(rt, ra, nb, 597); }
constexpr uint32_t stswi(uint32_t rs, uint32_t ra, uint32_t nb) { return X(rs, ra, nb, 725); }
constexpr uint32_t lswx(uint32_t rt, uint32_t ra, uint32_t rb) { return X(rt, ra, rb, 533); }
constexpr uint32_t stswx(uint32_t rs, uint32_t ra, uint32_t rb) { return X(rs, ra, rb, 661); }
constexpr uint32_t dcbz(uint32_t ra, uint32_t rb) { return X(0, ra, rb, 1014); }
constexpr uint32_t dcbz_l(uint32_t ra, uint32_t rb) { return (4u << 26) | (ra << 16) | (rb << 11) | (1014u << 1); }
// Floating point: A-form (fa) and the X-form family (fx), by primary opcode 59, 63 or 4 (paired singles).
constexpr uint32_t fa(uint32_t op, uint32_t d, uint32_t a, uint32_t b, uint32_t c, uint32_t xo5, uint32_t rc = 0) {
    return (op << 26) | (d << 21) | (a << 16) | (b << 11) | (c << 6) | (xo5 << 1) | rc;
}
constexpr uint32_t fx(uint32_t op, uint32_t d, uint32_t a, uint32_t b, uint32_t xo10, uint32_t rc = 0) {
    return (op << 26) | (d << 21) | (a << 16) | (b << 11) | (xo10 << 1) | rc;
}
constexpr uint32_t stfiwx(uint32_t fs, uint32_t ra, uint32_t rb) { return X(fs, ra, rb, 983); }
constexpr uint32_t mffs(uint32_t fd, uint32_t rc = 0) { return fx(63, fd, 0, 0, 583, rc); }
constexpr uint32_t mtfsf(uint32_t fm, uint32_t fb, uint32_t rc = 0) { return (63u << 26) | (fm << 17) | (fb << 11) | (711u << 1) | rc; }
constexpr uint32_t mtfsfi(uint32_t crf, uint32_t imm, uint32_t rc = 0) { return (63u << 26) | (crf << 23) | (imm << 12) | (134u << 1) | rc; }
constexpr uint32_t mtfsb0(uint32_t bt, uint32_t rc = 0) { return (63u << 26) | (bt << 21) | (70u << 1) | rc; }
constexpr uint32_t mtfsb1(uint32_t bt, uint32_t rc = 0) { return (63u << 26) | (bt << 21) | (38u << 1) | rc; }
constexpr uint32_t mcrfs(uint32_t crfd, uint32_t crfs) { return (63u << 26) | (crfd << 23) | (crfs << 18) | (64u << 1); }
constexpr uint32_t fcmpu(uint32_t crf, uint32_t a, uint32_t b) { return (63u << 26) | (crf << 23) | (a << 16) | (b << 11); }
constexpr uint32_t fcmpo(uint32_t crf, uint32_t a, uint32_t b) { return (63u << 26) | (crf << 23) | (a << 16) | (b << 11) | (32u << 1); }
constexpr uint32_t ps_cmp(uint32_t xo10, uint32_t crf, uint32_t a, uint32_t b) {
    return (4u << 26) | (crf << 23) | (a << 16) | (b << 11) | (xo10 << 1);
}
// psq_l (56), psq_lu (57), psq_st (60), psq_stu (61): 12-bit displacement, W at bit 15, GQR index at 12.
constexpr uint32_t psq(uint32_t op, uint32_t fr, int32_t d, uint32_t ra, uint32_t w, uint32_t i) {
    return (op << 26) | (fr << 21) | (ra << 16) | (w << 15) | (i << 12) | (static_cast<uint32_t>(d) & 0xFFFu);
}
// psq_lx (6), psq_stx (7), psq_lux (38), psq_stux (39).
constexpr uint32_t psqx(uint32_t xo6, uint32_t fr, uint32_t ra, uint32_t rb, uint32_t w, uint32_t i) {
    return (4u << 26) | (fr << 21) | (ra << 16) | (rb << 11) | (w << 10) | (i << 7) | (xo6 << 1);
}

}  // namespace ppc

// --- A small assembler --------------------------------------------------------------------------------------------
//
// Programs are built position-independent where they can be: branches to labels are relative and resolved when the
// program is finished. What depends on where a program is placed (a bl to a registered function, an address held in a
// register) is a relocation, resolved by Image::Resolve(base) at placement.
struct Label {
    int id = -1;
};

struct Image {
    struct Reloc {
        enum Kind { BAbs, LabelAddr } kind;
        size_t index;    // the word (LabelAddr: the lis, with the ori after it)
        uint32_t value;  // BAbs: the target address; LabelAddr: the label's word index
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
    // Zero words up to the word index (data, or room between pieces of code that a branch skips).
    void Org(size_t index) {
        while (words_.size() < index) words_.push_back(0);
    }
    void B(Label l, bool link = false) {
        fixes_.push_back({Fix::B, words_.size(), l.id, -1});
        words_.push_back(0x48000000u | (link ? 1u : 0u));
    }
    void Bc(uint32_t bo, uint32_t bi, Label l, bool link = false) {
        fixes_.push_back({Fix::Bc, words_.size(), l.id, -1});
        words_.push_back((16u << 26) | (bo << 21) | (bi << 16) | (link ? 1u : 0u));
    }
    // b / bl to a fixed address (a registered function), resolved at placement.
    void BAbs(uint32_t target, bool link = false) {
        relocs_.push_back({Image::Reloc::BAbs, words_.size(), target, link});
        words_.push_back(0);
    }
    // addi rt, ra, (address of label - address of from): the distance between two labels, for code that read its own
    // address with bl .+4 or bcl 20,31,.+4.
    void AddiDiff(uint32_t rt, uint32_t ra, Label to, Label from) {
        fixes_.push_back({Fix::Diff, words_.size(), to.id, from.id});
        words_.push_back(ppc::addi(rt, ra, 0));
    }
    // A data word: the distance in bytes from one label to another (a jump table's entry).
    void WordDiff(Label to, Label from) {
        fixes_.push_back({Fix::Word, words_.size(), to.id, from.id});
        words_.push_back(0);
    }
    // The address of a label in a register (lis, ori), resolved at placement.
    void LiLabel(uint32_t rt, Label l) {
        relocs_.push_back({Image::Reloc::LabelAddr, words_.size(), static_cast<uint32_t>(l.id), false});
        words_.push_back(ppc::lis(rt, 0));
        words_.push_back(ppc::ori(rt, rt, 0));
    }
    // A 32-bit constant in a register.
    void Li32(uint32_t rt, uint32_t value) {
        words_.push_back(ppc::lis(rt, static_cast<int16_t>(value >> 16)));
        words_.push_back(ppc::ori(rt, rt, value & 0xFFFFu));
    }
    // Stores a result where the comparison sees it: a GPR (word) or an FPR (double) at the next probe slot off r30.
    void Probe(uint32_t rt) {
        words_.push_back(ppc::stw(rt, static_cast<int32_t>(probe_), 30));
        probe_ += 4;
    }
    void ProbeF(uint32_t fr) {
        probe_ = (probe_ + 7u) & ~7u;
        words_.push_back(ppc::stfd(fr, static_cast<int32_t>(probe_), 30));
        probe_ += 8;
    }
    // The prologue and epilogue of programs that change lr: r31 keeps the caller's.
    void Save() { words_.push_back(ppc::mflr(31)); }
    void Done() {
        words_.push_back(ppc::mtlr(31));
        words_.push_back(ppc::blr());
    }

    Image Finish() {
        Image image;
        image.words = words_;
        for (const Fix& f : fixes_) {
            const int32_t to = labels_[static_cast<size_t>(f.label)];
            if (to < 0) fail("unbound label");
            const int32_t from = f.label2 >= 0 ? labels_[static_cast<size_t>(f.label2)] : static_cast<int32_t>(f.index);
            if (f.label2 >= 0 && from < 0) fail("unbound label");
            const int32_t bytes = 4 * (to - from);
            switch (f.kind) {
            case Fix::B: image.words[f.index] |= static_cast<uint32_t>(bytes) & 0x03FFFFFCu; break;
            case Fix::Bc: image.words[f.index] |= static_cast<uint32_t>(bytes) & 0xFFFCu; break;
            case Fix::Diff: image.words[f.index] |= static_cast<uint32_t>(bytes) & 0xFFFFu; break;
            case Fix::Word: image.words[f.index] = static_cast<uint32_t>(bytes); break;
            }
        }
        image.relocs = relocs_;
        for (Image::Reloc& r : image.relocs) {
            if (r.kind != Image::Reloc::LabelAddr) continue;
            if (labels_[r.value] < 0) fail("unbound label");
            r.value = static_cast<uint32_t>(labels_[r.value]);  // the label id becomes its word index
        }
        if (probe_ > kDataSize - kProbeOffset - 0x100) fail("too many probes");
        if (image.words.size() * 4 > kSlotBytes) fail("program larger than a slot");
        return image;
    }

private:
    struct Fix {
        enum Kind { B, Bc, Diff, Word } kind;
        size_t index;
        int label;
        int label2;
    };
    [[noreturn]] static void fail(const char* what) {
        std::fprintf(stderr, "jit differential: assembler: %s\n", what);
        std::exit(2);
    }
    std::vector<uint32_t> words_;
    std::vector<int> labels_;
    std::vector<Fix> fixes_;
    std::vector<Image::Reloc> relocs_;
    uint32_t probe_ = 0;
};

// --- Random numbers and the edge values ---------------------------------------------------------------------------
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
    bool Chance(uint32_t percent) { return Below(100) < percent; }
    template <typename T, size_t N>
    T Pick(const T (&a)[N]) { return a[Below(static_cast<uint32_t>(N))]; }
};

constexpr uint32_t kIntEdges[] = {0u, 1u, 2u, 3u, 31u, 32u, 33u, 63u, 64u, 0x7FFFFFFFu, 0x80000000u, 0x80000001u,
                                  0xFFFFFFFFu, 0xFFFFFFFEu, 0x12345678u, 0xFFFF0000u, 0x0000FFFFu, 0x00008000u};

// Doubles as raw bits: zeros, ordinary values, large exponents, subnormals, infinities, quiet and signalling NaNs with
// payloads, and values that round differently in single and double precision or sit at the single range's edges.
constexpr uint64_t kDoubleEdges[] = {
    0x0000000000000000ull, 0x8000000000000000ull, 0x3FF0000000000000ull, 0xBFF0000000000000ull,
    0x3FE0000000000000ull, 0x4008000000000000ull, 0x7FE0000000000000ull, 0xFFE0000000000000ull,
    0x7FEFFFFFFFFFFFFFull, 0x0010000000000000ull, 0x0000000000000001ull, 0x000FFFFFFFFFFFFFull,
    0x8000000000000001ull, 0x7FF0000000000000ull, 0xFFF0000000000000ull, 0x7FF8000000000000ull,
    0x7FF8DEAD00000000ull, 0x7FF0000000000001ull, 0xFFF4000000000123ull, 0x380FFFFFE0000000ull,
    0x47EFFFFFE0000000ull, 0x47EFFFFFF0000000ull, 0x3FF0000010000000ull, 0x4170000010000000ull,
    0x3FB999999999999Aull, 0xC0590000A0000000ull, 0x369FFFFFFFFFFFFFull, 0x7FFFFFFFFFFFFFFFull,
};

// Singles as raw bits.
constexpr uint32_t kSingleEdges[] = {
    0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u, 0x3F000000u, 0x40400000u, 0x00000001u, 0x007FFFFFu,
    0x00800000u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7FA00001u, 0xFFC12345u,
    0x4B800000u, 0x4B800001u, 0x7F000000u, 0x01000000u, 0x80000001u, 0x3EAAAAABu, 0xC2C80000u,
};

uint32_t RandInt(Rng& r) { return r.Chance(60) ? r.Pick(kIntEdges) : r.Next(); }
uint64_t RandDouble(Rng& r) {
    switch (r.Below(4)) {
    case 0: return r.Pick(kDoubleEdges);
    case 1: return (static_cast<uint64_t>(r.Next()) << 32) | r.Next();
    case 2: {  // a single widened to double
        const uint32_t s = r.Pick(kSingleEdges);
        const uint32_t sign = s & 0x80000000u, exp = (s >> 23) & 0xFF, frac = s & 0x7FFFFFu;
        if (exp == 0 && frac != 0) {  // a single subnormal: normalise
            int shift = 0;
            uint32_t f = frac;
            while ((f & 0x800000u) == 0) { f <<= 1; ++shift; }
            return (static_cast<uint64_t>(sign) << 32) | (static_cast<uint64_t>(1023 - 126 - shift) << 52) |
                   (static_cast<uint64_t>(f & 0x7FFFFFu) << 29);
        }
        const uint64_t e = exp == 0 ? 0 : exp == 255 ? 2047 : exp + 896u;
        return (static_cast<uint64_t>(sign) << 32) | (e << 52) | (static_cast<uint64_t>(frac) << 29);
    }
    default: return r.Pick(kDoubleEdges) ^ (r.Chance(30) ? 1ull << r.Below(52) : 0ull);
    }
}
// A GQR with valid quantization types (the reserved ones abort the runtime) and scales.
uint32_t RandGqr(Rng& r) {
    constexpr uint32_t types[] = {0, 0, 4, 5, 6, 7};
    constexpr uint32_t scales[] = {0, 0, 0, 1, 5, 8, 31, 32, 33, 63};
    return (r.Pick(scales) << 24) | (r.Pick(types) << 16) | (r.Pick(scales) << 8) | r.Pick(types);
}

// The state programs start from. r1 is never zero (the float helpers refuse a zero stack pointer), r29 points into the
// sandbox's input, r30 at its probes.
CpuContext BaseState(Rng& r, uint32_t fpscr = 0) {
    CpuContext c{};
    for (auto& g : c.gpr) g = RandInt(r);
    c.gpr[1] = kDataBase + 0x1800u;
    c.gpr[29] = kDataBase + 0x800u;
    c.gpr[30] = kDataBase + kProbeOffset;
    JitTestCr::Randomize(c, [&] { return r.Next(); });
    // XER[SO] set in one state in eight: compiled code does not run while it is set (JitBlocksMayRun), so the others run there
    c.xer = (r.Chance(80) ? (r.Next() & 0x6000007Fu) : (r.Next() & 0x7FFFFFFFu)) | (r.Chance(12) ? 0x80000000u : 0u);
    c.ctr = RandInt(r);
    c.lr = kReturn;
    c.fpscr = fpscr;
    for (auto& f : c.fpr) f.raw = RandDouble(r);
    for (auto& q : c.gqr) q = RandGqr(r);
    c.msr = r.Next();
    c.srr0 = r.Next();
    c.srr1 = r.Next();
    return c;
}

// --- Running a program, and what is compared ----------------------------------------------------------------------
struct Snap {
    CpuContext init{}, ctx{};
    uint32_t entry = 0;
    bool threw = false;
    std::string what;
    std::vector<std::pair<uint32_t, uint8_t>> memory;  // sandbox bytes that differ from the sandbox the run started with
    std::vector<std::pair<uint32_t, uint8_t>> code;    // watched code bytes that differ from when the program was placed
};

unsigned g_failures = 0, g_reported = 0;
uint64_t g_runs = 0;
std::string g_current = "starting";
constexpr unsigned kMaxReports = 8;

std::string Fmt(const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof buffer, format, args);
    va_end(args);
    return buffer;
}

class Runner {
public:
    explicit Runner(const EngineConfig& config) : config_(config) {}
    const EngineConfig& config() const { return config_; }
    std::vector<Snap>& trace() { return trace_; }

    // The sandbox for the runs that follow: random words with edge values among them (floats, small integers), from a seed.
    void SetSandbox(uint32_t seed) {
        Rng r(0xC0FFEEu + seed);
        for (uint32_t off = 0; off < kDataSize; off += 4) {
            uint32_t w;
            switch (r.Below(10)) {
            case 0: case 1: case 2: case 3: w = r.Next(); break;
            case 4: case 5: w = r.Pick(kSingleEdges); break;
            case 6: w = r.Below(300); break;
            case 7: w = 0; break;
            case 8: w = 0xFFFFFFFFu; break;
            default: w = static_cast<uint32_t>(RandDouble(r) >> (r.Chance(50) ? 32 : 0)); break;
            }
            Memory::Write32(kDataBase + off, w);
        }
        sandbox_.assign(Memory::GetPointer(kDataBase, kDataSize), Memory::GetPointer(kDataBase, kDataSize) + kDataSize);
    }

    // Writes a program's words (resolved for base) as the guest stores would, and watches that range for changes.
    void Place(const Image& image, uint32_t base) {
        const std::vector<uint32_t> words = image.Resolve(base);
        for (size_t i = 0; i < words.size(); ++i) Memory::Write32(base + 4u * static_cast<uint32_t>(i), words[i]);
        placed_.push_back({base, words});
        WatchCode(base, static_cast<uint32_t>(words.size() * 4));
    }
    void Write32(uint32_t address, uint32_t word) {
        Memory::Write32(address, word);
        for (auto& [base, words] : placed_)
            if (address >= base && address < base + 4u * words.size()) words[(address - base) / 4] = word;
    }
    // The code bytes (as guest memory holds them) the snapshots report differences in, against now.
    void WatchCode(uint32_t base, uint32_t bytes) {
        watchBase_ = base;
        watchBytes_ = bytes;
        const uint8_t* p = Memory::GetPointer(base, bytes);
        watch_.assign(p, p + bytes);
    }

    void Run(uint32_t entry, const CpuContext& init) {
        if (sandbox_.empty()) SetSandbox(1);
        std::memcpy(Memory::GetPointer(kDataBase, kDataSize), sandbox_.data(), kDataSize);
        PpcInterpreter::SetEngineOverride(config_.engine);
        PpcInterpreter::SetJitThreshold(config_.threshold);
        Snap s;
        s.init = init;
        s.entry = entry;
        CpuContext ctx = init;
        ctx.pc = 0;
        {
            CpuContextScope scope(&ctx);  // the context the float and quantization helpers read, and NI mode
            try {
                PpcInterpreter::Run(&ctx, entry);
            } catch (const std::exception& e) {
                s.threw = true;
                s.what = e.what();
            } catch (...) {
                s.threw = true;
                s.what = "unknown exception";
            }
        }
        ctx.pc = 0;
        s.ctx = ctx;
        const uint8_t* mem = Memory::GetPointer(kDataBase, kDataSize);
        for (uint32_t i = 0; i < kDataSize; ++i)
            if (mem[i] != sandbox_[i]) s.memory.push_back({i, mem[i]});
        if (watchBytes_ != 0) {
            const uint8_t* code = Memory::GetPointer(watchBase_, watchBytes_);
            for (uint32_t i = 0; i < watchBytes_; ++i)
                if (code[i] != watch_[i]) s.code.push_back({i, code[i]});
        }
        ++g_runs;
        trace_.push_back(std::move(s));
    }

    void Dump(std::string& out) const {
        for (const auto& [base, words] : placed_) {
            out += Fmt("  program at 0x%08X (%zu words):\n", base, words.size());
            const size_t limit = std::min<size_t>(words.size(), 320);
            for (size_t i = 0; i < limit; i += 4) {
                out += Fmt("    %08X:", base + 4u * static_cast<uint32_t>(i));
                for (size_t k = i; k < std::min(limit, i + 4); ++k) out += Fmt(" %08X", words[k]);
                out += "\n";
            }
            if (limit < words.size()) out += "    ...\n";
        }
    }

private:
    EngineConfig config_;
    std::vector<Snap> trace_;
    std::vector<uint8_t> sandbox_;
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> placed_;
    uint32_t watchBase_ = 0, watchBytes_ = 0;
    std::vector<uint8_t> watch_;
};

// Each difference between a reference snapshot and another engine's, one per line; empty when they are equal.
std::string Diff(const Snap& ref, const Snap& got) {
    std::string out;
    const auto word = [&](const char* name, uint32_t a, uint32_t b) {
        if (a != b) out += Fmt("    %s: classic=%08X got=%08X\n", name, a, b);
    };
    if (ref.threw != got.threw)
        out += Fmt("    exception: classic=%s got=%s\n", ref.threw ? ref.what.c_str() : "none", got.threw ? got.what.c_str() : "none");
    else if (ref.what != got.what) out += Fmt("    exception text: classic='%s' got='%s'\n", ref.what.c_str(), got.what.c_str());
    for (int i = 0; i < 32; ++i) word(Fmt("r%d", i).c_str(), ref.ctx.gpr[i], got.ctx.gpr[i]);
    for (uint32_t i = 0; i < 8; ++i)
        if (ref.ctx.crv[i] != got.ctx.crv[i])
            out += Fmt("    crv[%u]: classic=%016llX (reads %X) got=%016llX (reads %X)\n", i, static_cast<unsigned long long>(ref.ctx.crv[i]),
                       PpcCrvNibble(ref.ctx.crv[i]), static_cast<unsigned long long>(got.ctx.crv[i]), PpcCrvNibble(got.ctx.crv[i]));
    word("lr", ref.ctx.lr, got.ctx.lr);
    word("ctr", ref.ctx.ctr, got.ctx.ctr);
    word("xer", ref.ctx.xer, got.ctx.xer);
    word("fpscr", ref.ctx.fpscr, got.ctx.fpscr);
    for (int i = 0; i < 8; ++i) word(Fmt("gqr%d", i).c_str(), ref.ctx.gqr[i], got.ctx.gqr[i]);
    word("hid0", ref.ctx.hid0, got.ctx.hid0);
    word("hid1", ref.ctx.hid1, got.ctx.hid1);
    word("hid2", ref.ctx.hid2, got.ctx.hid2);
    word("srr0", ref.ctx.srr0, got.ctx.srr0);
    word("srr1", ref.ctx.srr1, got.ctx.srr1);
    word("msr", ref.ctx.msr, got.ctx.msr);
    for (int i = 0; i < 32; ++i)
        if (ref.ctx.fpr[i].raw != got.ctx.fpr[i].raw)
            out += Fmt("    f%d: classic=%016llX got=%016llX\n", i, static_cast<unsigned long long>(ref.ctx.fpr[i].raw),
                       static_cast<unsigned long long>(got.ctx.fpr[i].raw));
    const auto bytes = [&](const char* what, const std::vector<std::pair<uint32_t, uint8_t>>& a,
                           const std::vector<std::pair<uint32_t, uint8_t>>& b) {
        if (a == b) return;
        // The offsets either side changed, and each side's value (or "unchanged").
        std::map<uint32_t, std::pair<int, int>> merged;
        for (const auto& [off, v] : a) merged[off] = {v, -1};
        for (const auto& [off, v] : b) {
            auto it = merged.find(off);
            if (it == merged.end()) merged[off] = {-1, v};
            else it->second.second = v;
        }
        unsigned shown = 0, total = 0;
        for (const auto& [off, vals] : merged) {
            if (vals.first == vals.second) continue;
            ++total;
            if (shown < 8) {
                ++shown;
                out += Fmt("    %s +0x%04X: classic=%s got=%s\n", what, off,
                           vals.first < 0 ? "unchanged" : Fmt("%02X", vals.first).c_str(),
                           vals.second < 0 ? "unchanged" : Fmt("%02X", vals.second).c_str());
            }
        }
        if (total > shown) out += Fmt("    %s: %u more bytes differ\n", what, total - shown);
    };
    bytes("sandbox", ref.memory, got.memory);
    bytes("code", ref.code, got.code);
    return out;
}

// What the cases ran, by category, for the summary.
struct CategoryStats {
    unsigned cases = 0;
    uint64_t runs = 0;
};
std::map<std::string, CategoryStats> g_categories;

uint32_t NextBase() {
    static uint32_t next = 0;
    const uint32_t base = kSlotBase + (next++ % kSlotCount) * kSlotBytes;
    return base;
}

// A scenario places code and runs it from some states, whatever the engine; it is deterministic, so the reference and
// the compared engines make the same sequence of runs.
using Scenario = std::function<void(Runner&, uint32_t base)>;

void ReportFailure(const std::string& category, const std::string& name, const EngineConfig& config, size_t run,
                   size_t runs, const Runner& reference, const Snap& got, const std::string& diff) {
    ++g_failures;
    if (g_reported >= kMaxReports) return;
    ++g_reported;
    std::string out = Fmt("FAIL [%s] %s: engine %s, run %zu of %zu, entry 0x%08X\n", category.c_str(), name.c_str(),
                          config.name, run + 1, runs, got.entry);
    reference.Dump(out);
    out += "  initial state:";
    for (int i = 0; i < 32; ++i) {
        if (i % 8 == 0) out += "\n   ";
        out += Fmt(" r%d=%X", i, got.init.gpr[i]);
    }
    out += Fmt("\n    cr=%08X lr=%08X ctr=%08X xer=%08X fpscr=%08X\n    crv:", PpcGetCr(&got.init), got.init.lr, got.init.ctr, got.init.xer, got.init.fpscr);
    for (int i = 0; i < 8; ++i) out += Fmt(" %016llX", static_cast<unsigned long long>(got.init.crv[i]));
    out += "\n   ";
    for (int i = 0; i < 14; ++i) out += Fmt(" f%d=%llX", i, static_cast<unsigned long long>(got.init.fpr[i].raw));
    out += "\n  differences:\n" + diff;
    std::fputs(out.c_str(), stderr);
    if (g_reported == kMaxReports) std::fputs("(further failures are counted, not printed)\n", stderr);
}

// Runs a scenario under the reference and under each compared engine, each at a code address of its own, and compares.
void RunCase(const std::string& category, const std::string& name, const Scenario& scenario) {
    g_current = category + ": " + name;
    static const char* const filter = std::getenv("JIT_DIFF_FILTER");  // only the cases whose "category: name" contains this
    if (filter != nullptr && g_current.find(filter) == std::string::npos) return;
    CategoryStats& stats = g_categories[category];
    ++stats.cases;
    for (const EngineConfig& config : kConfigs) {
        const uint32_t base = NextBase();
        Runner reference(kReference);
        scenario(reference, base);
        Runner other(config);
        scenario(other, base);
        PpcInterpreter::ClearEngineOverride();
        static const bool inject = std::getenv("JIT_DIFF_INJECT") != nullptr;  // flips r5 in the compared engine: shows what a failure looks like
        if (inject && !other.trace().empty()) other.trace().back().ctx.gpr[5] ^= 0x10u;
        stats.runs += other.trace().size();
        if (reference.trace().size() != other.trace().size()) {
            ++g_failures;
            std::fprintf(stderr, "FAIL [%s] %s: engine %s made %zu runs, classic %zu\n", category.c_str(), name.c_str(),
                         config.name, other.trace().size(), reference.trace().size());
            continue;
        }
        for (size_t i = 0; i < reference.trace().size(); ++i) {
            const std::string diff = Diff(reference.trace()[i], other.trace()[i]);
            if (!diff.empty()) {
                ReportFailure(category, name, config, i, reference.trace().size(), reference, other.trace()[i], diff);
                break;  // one report per case and engine: later runs differ for the same reason
            }
        }
    }
}

using States = std::vector<CpuContext>;

// A program from an assembler, run from each state in turn (the sandbox seeds in turn too).
void AddProgram(const char* category, const std::string& name, Asm& a, const States& states, unsigned sandboxSeeds = 1) {
    const Image image = a.Finish();
    RunCase(category, name, [image, states, sandboxSeeds](Runner& r, uint32_t base) {
        r.Place(image, base);
        for (unsigned seed = 1; seed <= sandboxSeeds; ++seed) {
            r.SetSandbox(seed);
            for (const CpuContext& s : states) r.Run(base, s);
        }
    });
}

// --- Functions registered the way translated code is, and guest code they run ----------------------------------------
//
// They change registers (volatile ones only), write guest memory, run guest code themselves, and throw. They do the same
// whatever engine called them.
void FnModify(CpuContext* c) {
    c->gpr[3] = c->gpr[3] * 3u + c->gpr[4];
    c->gpr[5] ^= 0x13579BDFu;
    c->gpr[0] = 0x70u;
    c->gpr[12] = c->lr;  // the link address the callee was given
    c->ctr += 5u;
    c->crv[0] = PpcCrvFromNibble(c->gpr[3] >> 28);
    c->xer ^= 0x20000000u;
    c->fpr[1].d = c->fpr[1].d * 2.0 + 1.0;
}
void FnFloat(CpuContext* c) {
    c->fpr[1].d = c->fpr[1].d + c->fpr[2].d;
    c->fpr[2].raw = c->fpr[2].raw * 0x9E3779B97F4A7C15ull + 1ull;
    c->fpr[13].raw = c->fpr[3].raw;
    c->gpr[3] += 1u;
}
// Writes r4 to the guest address in r3 (data or code) as a guest store does.
void FnWrite(CpuContext* c) {
    Memory::Write32(c->gpr[3], c->gpr[4]);
    if (c->gpr[3] - kDataBase < kDataSize - 16u) Memory::Write8(c->gpr[3] + 9u, 0x5Au);
    c->gpr[7] = c->gpr[3] + 4u;
}
// Runs guest code in the same context (the plain support function, or the one that calls FnThrow).
void FnReenter(CpuContext* c) {
    const uint32_t saved = c->lr;
    c->lr = kReturn;
    PpcInterpreter::Run(c, (c->gpr[5] & 1u) ? kSupportThrow : kSupportPlain);
    c->lr = saved;
    c->gpr[3] += 1u;
}
// Sets XER[SO] when r3 is odd (a callee that leaves it set: compiled code must not go on with it set).
void FnSetSo(CpuContext* c) {
    if (c->gpr[3] & 1u) c->xer |= 0x80000000u;
    c->gpr[3] += 2u;
}
// Throws when r6 is odd, after changing a register and guest memory.
void FnThrow(CpuContext* c) {
    c->gpr[3] ^= 0xDEAD0000u;
    Memory::Write32(kDataBase + 0x1FF0u, c->gpr[3]);
    if (c->gpr[6] & 1u) throw std::runtime_error("thrown from a registered function");
    c->gpr[7] = 0x77u;
}

constexpr uint32_t kFnModify = kFuncBase + 0x00, kFnFloat = kFuncBase + 0x10, kFnWrite = kFuncBase + 0x20,
                   kFnReenter = kFuncBase + 0x30, kFnThrow = kFuncBase + 0x40, kFnRewrite = kFuncBase + 0x50,
                   kFnRestore = kFuncBase + 0x60, kFnSetSo = kFuncBase + 0x70, kFnRestoreWritten = kFuncBase + 0x80;

void RegisterFunction(uint32_t address, const char* name, void (*fn)(CpuContext*)) {
    TranslatedFunctionInfo info{};
    info.address = address;
    info.name = name;
    info.rawCpuInvoker = fn;
    info.entryPoint = reinterpret_cast<void*>(fn);
    info.kind = FunctionKind::Native;
    TranslatedFunctionRegistry::Register(std::move(info));
}

// The guest functions FnReenter runs: addi, a store, a local call to a leaf; and one that calls FnThrow.
void WriteSupportCode() {
    using namespace ppc;
    const uint32_t plain[] = {mflr(0), addi(3, 3, 100), stw(3, 0x7F0, 29), 0x48000011u /* bl .+16 */, mtlr(0), blr(),
                              nop(), nop(), addi(4, 4, 1), blr()};
    for (size_t i = 0; i < sizeof plain / sizeof *plain; ++i) Memory::Write32(kSupportPlain + 4u * static_cast<uint32_t>(i), plain[i]);
    const uint32_t callThrow = kSupportThrow + 8u;
    const uint32_t thrower[] = {mflr(0), addi(3, 3, 200),
                                0x48000001u | ((kFnThrow - callThrow) & 0x03FFFFFCu), mtlr(0), blr()};
    for (size_t i = 0; i < sizeof thrower / sizeof *thrower; ++i) Memory::Write32(kSupportThrow + 4u * static_cast<uint32_t>(i), thrower[i]);
}

// --- Checks of the assembler against words known from the architecture and from the interpreter's decoder ----------
bool CheckEncodings() {
    using namespace ppc;
    struct Known {
        const char* what;
        uint32_t got, expected;
    };
    const Known known[] = {
        {"blr", blr(), 0x4E800020u}, {"bctr", bctr(), 0x4E800420u}, {"bctrl", bctrl(), 0x4E800421u},
        {"blrl", blrl(), 0x4E800021u}, {"mflr r0", mflr(0), 0x7C0802A6u}, {"mtlr r10", mtlr(10), 0x7D4803A6u},
        {"mtctr r9", mtctr(9), 0x7D2903A6u}, {"li r3,1", li(3, 1), 0x38600001u}, {"addi r3,r3,1", addi(3, 3, 1), 0x38630001u},
        {"cmpwi r3,3", cmpwi(0, 3, 3), 0x2C030003u}, {"cmplwi cr1,r3,5", cmplwi(1, 3, 5), 0x28830005u},
        {"stw r8,0(r9)", stw(8, 0, 9), 0x91090000u}, {"lwz r0,4(r1)", lwz(0, 4, 1), 0x80010004u},
        {"slwi r0,r0,2", rlwinm(0, 0, 2, 0, 29), 0x5400103Au}, {"mulhw r3,r4,r5", arith(75, 3, 4, 5, 0, 0), 0x7C642896u},
        {"fadd f1,f2,f3", fa(63, 1, 2, 3, 0, 21), 0xFC22182Au}, {"fmadd f1,f2,f3,f4", fa(63, 1, 2, 4, 3, 29), 0xFC2220FAu},
        {"psq_l f1,0(r3),0,0", psq(56, 1, 0, 3, 0, 0), 0xE0230000u}, {"mfcr r5", mfcr(5), 0x7CA00026u},
        {"lfs f2,0(r3)", ld(48, 2, 0, 3), (48u << 26) | (2u << 21) | (3u << 16)},
        {"ps_add f6,f2,f2", fa(4, 6, 2, 2, 0, 21), (4u << 26) | (6u << 21) | (2u << 16) | (2u << 11) | (21u << 1)},
        {"mfmsr r3", mfmsr(3), 0x7C6000A6u}, {"dcbz r0,r3", dcbz(0, 3), 0x7C001FECu},
        {"mtcrf 0xff,r3", mtcrf(0xFF, 3), 0x7C6FF120u}, {"bdnz .+0", 0x42000000u, 0x42000000u},
    };
    bool ok = true;
    for (const Known& k : known)
        if (k.got != k.expected) {
            std::fprintf(stderr, "encoder mismatch: %s is %08X, expected %08X\n", k.what, k.got, k.expected);
            ok = false;
        }
    return ok;
}

// === Directed programs: integer ===================================================================================

constexpr uint32_t kXerCa = 0x20000000u, kXerSo = 0x80000000u;
const uint32_t kXerVariants[] = {0u, kXerCa, kXerSo, 0xE0000000u};

// The registers and the shift counts the arithmetic and logical forms run on: every pair of edge values, with the XER
// carry and overflow state cycling.
States PairStates(Rng& rng, bool varyXer = true) {
    States states;
    size_t n = 0;
    for (uint32_t a : kIntEdges)
        for (uint32_t b : kIntEdges) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = a;
            c.gpr[4] = b;
            if (varyXer) c.xer = kXerVariants[n % 4] | (c.xer & 0x7F);
            ++n;
            states.push_back(c);
        }
    return states;
}

void AddIntegerArithmetic() {
    Rng rng(11);
    struct ArithOp { const char* name; uint32_t xo; bool unary; };
    const ArithOp ops[] = {{"add", 266, false}, {"addc", 10, false}, {"adde", 138, false}, {"addme", 234, true},
                           {"addze", 202, true}, {"subf", 40, false}, {"subfc", 8, false}, {"subfe", 136, false},
                           {"subfme", 232, true}, {"subfze", 200, true}, {"neg", 104, true}, {"mullw", 235, false},
                           {"mulhw", 75, false}, {"mulhwu", 11, false}, {"divw", 491, false}, {"divwu", 459, false}};
    const States states = PairStates(rng);
    for (const ArithOp& op : ops)
        for (uint32_t oe = 0; oe < 2; ++oe)
            for (uint32_t rc = 0; rc < 2; ++rc) {
                using namespace ppc;
                Asm a;
                a << arith(op.xo, 5, 3, op.unary ? 0 : 4, oe, rc);
                a.Probe(5);
                a << mfcr(6);
                a.Probe(6);
                a << mfxer(7);
                a.Probe(7);
                a << arith(op.xo, 3, 3, op.unary ? 0 : 3, oe, rc) << mfcr(8);  // the destination is a source, and both are
                a.Probe(8);
                a << blr();
                AddProgram("integer", Fmt("%s oe=%u rc=%u", op.name, oe, rc), a, states);
            }
}

void AddIntegerLogic() {
    Rng rng(12);
    struct LogicOp { const char* name; uint32_t xo; bool unary; };
    const LogicOp ops[] = {{"and", 28, false}, {"andc", 60, false}, {"or", 444, false}, {"orc", 412, false},
                           {"xor", 316, false}, {"nand", 476, false}, {"nor", 124, false}, {"eqv", 284, false},
                           {"slw", 24, false}, {"srw", 536, false}, {"sraw", 792, false}, {"cntlzw", 26, true},
                           {"extsb", 954, true}, {"extsh", 922, true}};
    const States states = PairStates(rng);
    for (const LogicOp& op : ops)
        for (uint32_t rc = 0; rc < 2; ++rc) {
            using namespace ppc;
            Asm a;
            a << logic(op.xo, 5, 3, op.unary ? 0 : 4, rc);
            a.Probe(5);
            a << mfcr(6);
            a.Probe(6);
            a << mfxer(7);
            a.Probe(7);
            a << logic(op.xo, 3, 3, op.unary ? 0 : 3, rc) << logic(op.xo, 4, 3, op.unary ? 0 : 4, rc) << mfcr(8);
            a.Probe(8);
            a << blr();
            AddProgram("integer", Fmt("%s rc=%u", op.name, rc), a, states);
        }
    for (uint32_t sh : {0u, 1u, 5u, 15u, 16u, 30u, 31u})
        for (uint32_t rc = 0; rc < 2; ++rc) {
            using namespace ppc;
            Asm a;
            a << srawi(5, 3, sh, rc);
            a.Probe(5);
            a << mfcr(6);
            a.Probe(6);
            a << mfxer(7);
            a.Probe(7);
            a << blr();
            AddProgram("integer", Fmt("srawi %u rc=%u", sh, rc), a, PairStates(rng));
        }
}

void AddIntegerImmediates() {
    Rng rng(13);
    using namespace ppc;
    States states;
    for (uint32_t v : kIntEdges)
        for (uint32_t x : kXerVariants) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = v;
            c.xer = x | (c.xer & 0x7F);
            states.push_back(c);
        }
    for (int32_t imm : {0, 1, -1, 0x7FFF, -0x8000, 0x1234, -0x1234, 0x5A5A}) {
        const uint32_t u = static_cast<uint32_t>(imm) & 0xFFFFu;
        Asm a;
        a << addi(5, 3, imm) << li(6, imm) << addis(7, 3, imm) << lis(8, imm) << addic(9, 3, imm) << addic_(10, 3, imm)
          << subfic(11, 3, imm) << mulli(12, 3, imm) << ori(13, 3, u) << oris(14, 3, u) << xori(15, 3, u)
          << xoris(16, 3, u) << andi_(17, 3, u) << andis_(18, 3, u);
        for (uint32_t r = 5; r <= 18; ++r) a.Probe(r);
        a << mfcr(19);
        a.Probe(19);
        a << mfxer(20);
        a.Probe(20);
        a << blr();
        AddProgram("integer", Fmt("immediates %d", imm), a, states);
    }
}

void AddIntegerCompares() {
    Rng rng(14);
    using namespace ppc;
    States states;
    size_t n = 0;
    for (uint32_t x : kIntEdges)
        for (uint32_t y : kIntEdges) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = x;
            c.gpr[4] = y;
            c.xer = ((n++ & 1) ? kXerSo : 0u) | (c.xer & 0x7FFFFFFu);
            states.push_back(c);
        }
    for (int kind = 0; kind < 2; ++kind) {
        Asm a;
        for (uint32_t crf = 0; crf < 8; ++crf)
            a << (kind == 0 ? cmpw(crf, crf & 1 ? 4 : 3, crf & 1 ? 3 : 4) : cmplw(crf, crf & 1 ? 4 : 3, crf & 1 ? 3 : 4));
        a << blr();
        AddProgram("integer", kind == 0 ? "cmpw every field" : "cmplw every field", a, states);
    }
    for (int32_t imm : {0, 1, -1, 0x7FFF, -0x8000, 0x1234, 31, 64}) {
        Asm a, b;
        for (uint32_t crf = 0; crf < 8; ++crf) {
            a << cmpwi(crf, crf & 1 ? 4 : 3, imm);
            b << cmplwi(crf, crf & 1 ? 4 : 3, static_cast<uint32_t>(imm) & 0xFFFFu);
        }
        a << blr();
        b << blr();
        AddProgram("integer", Fmt("cmpwi every field, %d", imm), a, states);
        AddProgram("integer", Fmt("cmplwi every field, %d", imm), b, states);
    }
}

void AddIntegerRotates() {
    Rng rng(15);
    using namespace ppc;
    const uint32_t bounds[] = {0, 1, 8, 15, 16, 24, 30, 31};  // mb and me: all pairs, so masks wrap and cover both ends
    States states;
    for (uint32_t v : {0x00000000u, 0xFFFFFFFFu, 0x80000001u, 0x12345678u, 0xA5A5A5A5u, 0x0F0F0F0Fu, 0x7FFFFFFFu, 0x00000001u}) {
        CpuContext c = BaseState(rng);
        c.gpr[3] = v;
        c.gpr[5] = rng.Next();
        states.push_back(c);
    }
    for (int kind = 0; kind < 2; ++kind)
        for (uint32_t sh : {0u, 1u, 7u, 16u, 31u})
            for (uint32_t rc = 0; rc < 2; ++rc) {
                Asm a;
                for (uint32_t mb : bounds)
                    for (uint32_t me : bounds) {
                        a << (kind == 0 ? rlwinm(5, 3, sh, mb, me, rc) : rlwimi(5, 3, sh, mb, me, rc));
                        a.Probe(5);
                        if (rc) {
                            a << mfcr(6);
                            a.Probe(6);
                        }
                    }
                a << blr();
                AddProgram("integer", Fmt("%s sh=%u rc=%u", kind == 0 ? "rlwinm" : "rlwimi", sh, rc), a, states);
            }
    States shifts;
    for (uint32_t v : {0x00000000u, 0xFFFFFFFFu, 0x80000001u, 0x12345678u})
        for (uint32_t count : {0u, 1u, 4u, 16u, 31u, 32u, 33u, 63u, 0xFFFFFFE1u}) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = v;
            c.gpr[4] = count;
            c.gpr[5] = rng.Next();
            shifts.push_back(c);
        }
    for (uint32_t rc = 0; rc < 2; ++rc) {
        Asm a;
        for (uint32_t mb : bounds)
            for (uint32_t me : bounds) {
                a << rlwnm(5, 3, 4, mb, me, rc);
                a.Probe(5);
                if (rc) {
                    a << mfcr(6);
                    a.Probe(6);
                }
            }
        a << blr();
        AddProgram("integer", Fmt("rlwnm rc=%u", rc), a, shifts);
    }
}

void AddConditionRegisterPrograms() {
    Rng rng(16);
    using namespace ppc;
    States states;
    const uint32_t patterns[] = {0u, 0xFFFFFFFFu, 0xAAAAAAAAu, 0x55555555u, 0x0F0F0F0Fu, 0xF0F0F0F0u, 0x80000000u, 0x00000001u};
    for (unsigned i = 0; i < 64; ++i) {
        CpuContext c = BaseState(rng);
        if (i < 16) JitTestCr::SetPacked(c, patterns[i & 7], i >= 8, [&] { return rng.Next(); });  // as table values, then as compare differences
        c.gpr[3] = rng.Next();
        c.xer = rng.Next() & 0xF0000000u;
        states.push_back(c);
    }
    const struct { const char* name; uint32_t xo; } logicals[] = {
        {"crand", 257}, {"crandc", 129}, {"creqv", 289}, {"crnand", 225}, {"crnor", 33}, {"cror", 449}, {"crorc", 417}, {"crxor", 193}};
    const uint32_t triples[][3] = {{0, 1, 2}, {31, 0, 5}, {7, 7, 7}, {3, 3, 4}, {4, 4, 4}, {8, 9, 10}, {12, 13, 12},
                                   {30, 29, 28}, {2, 2, 2}, {1, 17, 18}, {20, 21, 22}, {5, 6, 7}, {16, 3, 28}, {9, 9, 8}};
    for (const auto& op : logicals) {
        Asm a;
        for (const auto& t : triples) {
            a << crop(op.xo, t[0], t[1], t[2]) << mfcr(5);
            a.Probe(5);
        }
        a << blr();
        AddProgram("integer", op.name, a, states);
    }
    {
        Asm a;
        const uint32_t pairs[][2] = {{0, 1}, {7, 0}, {3, 3}, {5, 2}, {1, 6}};
        for (const auto& p : pairs) {
            a << mcrf(p[0], p[1]) << mfcr(5);
            a.Probe(5);
        }
        for (uint32_t crm : {0xFFu, 0x80u, 0x01u, 0xA5u, 0x00u, 0x5Au, 0x18u}) {
            a << mtcrf(crm, 3) << mfcr(6);
            a.Probe(6);
        }
        for (uint32_t crf = 0; crf < 8; ++crf) {
            a << mcrxr(crf) << mfcr(7) << mfxer(8);
            a.Probe(7);
            a.Probe(8);
        }
        a << blr();
        AddProgram("integer", "mcrf mtcrf mcrxr mfcr", a, states);
    }
}

void AddSpecialRegisterPrograms() {
    Rng rng(17);
    using namespace ppc;
    States states;
    for (unsigned i = 0; i < 24; ++i) {
        CpuContext c = BaseState(rng);
        c.gpr[3] = i < 6 ? kIntEdges[i * 3] : rng.Next();
        states.push_back(c);
    }
    const uint32_t sprs[] = {1, 8, 9, 26, 27, 912, 913, 914, 915, 916, 917, 918, 919, 920, 1008, 1009};
    Asm a;
    a.Save();
    for (uint32_t spr : sprs) {
        a << mtspr(spr, 3) << mfspr(5, spr);
        a.Probe(5);
        if (spr == 1) {
            a << arith(202, 6, 4, 0, 0, 1) << mfcr(7);  // addze. reads the carry mtxer set
            a.Probe(6);
            a.Probe(7);
        }
    }
    a << mtctr(3) << mfctr(5) << mtspr(22, 3);  // ctr again as its own kind; the decrementer's write has no effect
    a.Probe(5);
    a << mtmsr(3) << mfmsr(6);
    a.Probe(6);
    a.Done();
    AddProgram("integer", "mtspr mfspr mtmsr", a, states);

    Asm m;
    m << (3u << 26 | 31u << 21 | 3u << 16)           // twi 31,r3,0: traps are ignored
      << ppc::X(31, 3, 4, 4)                           // tw 31,r3,r4
      << (19u << 26 | 150u << 1)                       // isync
      << ppc::X(0, 0, 0, 598) << ppc::X(0, 0, 0, 854)  // sync, eieio
      << ppc::X(0, 3, 4, 86) << ppc::X(0, 3, 4, 470) << ppc::X(0, 3, 4, 54) << ppc::X(0, 3, 4, 278)
      << ppc::X(0, 3, 4, 246) << ppc::X(0, 3, 4, 982)  // dcbf, dcbi, dcbst, dcbt, dcbtst, icbi
      << ppc::X(3, 0, 0, 210) << ppc::X(3, 0, 4, 242)  // mtsr, mtsrin
      << addi(5, 3, 1) << blr();
    AddProgram("integer", "cache and sync no-ops", m, states);
}

// === Directed programs: memory =====================================================================================

// Displacements from r29 (the middle of the sandbox's input) that cover every alignment and both ends.
const int32_t kDisps[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -8, 0x40, -0x40, 0x7C, 0x100, -0x100, 0x7F0, -0x7F0, 0x7FC, -0x7FC};

States OffsetStates(Rng& rng, bool absolute) {
    States states;
    const int32_t offsets[] = {0, 1, 2, 3, 4, 7, 0x10, -0x10, 0x3FF, -0x3FF, 0x7F0, -0x7F0, 0x7FC, 0x20};
    for (int32_t off : offsets) {
        CpuContext c = BaseState(rng);
        c.gpr[4] = absolute ? kDataBase + 0x800u + static_cast<uint32_t>(off) : static_cast<uint32_t>(off);
        states.push_back(c);
    }
    return states;
}

void AddMemoryPrograms() {
    Rng rng(21);
    using namespace ppc;
    struct MemOp { const char* name; uint32_t op; bool update; bool fp; };
    const MemOp loads[] = {{"lwz", 32, 0, 0}, {"lwzu", 33, 1, 0}, {"lbz", 34, 0, 0}, {"lbzu", 35, 1, 0}, {"lhz", 40, 0, 0},
                           {"lhzu", 41, 1, 0}, {"lha", 42, 0, 0}, {"lhau", 43, 1, 0}, {"lfs", 48, 0, 1}, {"lfsu", 49, 1, 1},
                           {"lfd", 50, 0, 1}, {"lfdu", 51, 1, 1}};
    const MemOp stores[] = {{"stw", 36, 0, 0}, {"stwu", 37, 1, 0}, {"stb", 38, 0, 0}, {"stbu", 39, 1, 0}, {"sth", 44, 0, 0},
                            {"sthu", 45, 1, 0}, {"stfs", 52, 0, 1}, {"stfsu", 53, 1, 1}, {"stfd", 54, 0, 1}, {"stfdu", 55, 1, 1}};
    States few;
    for (int i = 0; i < 3; ++i) few.push_back(BaseState(rng));
    for (const MemOp& op : loads) {
        Asm a;
        for (int32_t d : kDisps) {
            if (op.update) a << addi(28, 29, 0);
            a << ld(op.op, 5, d, op.update ? 28 : 29);
            if (op.fp) a.ProbeF(5);
            else a.Probe(5);
            if (op.update) a.Probe(28);
        }
        a << blr();
        AddProgram("memory", Fmt("%s every displacement", op.name), a, few, 3);
    }
    States values;
    for (int i = 0; i < 12; ++i) {
        CpuContext c = BaseState(rng);
        c.gpr[3] = i < 6 ? kIntEdges[i * 3] : rng.Next();
        c.fpr[3].raw = RandDouble(rng);
        values.push_back(c);
    }
    for (const MemOp& op : stores) {
        Asm a;
        for (int32_t d : kDisps) {
            if (op.update) a << addi(28, 29, 0);
            a << ld(op.op, 3, d, op.update ? 28 : 29);
            if (op.update) a.Probe(28);
        }
        a << blr();
        AddProgram("memory", Fmt("%s every displacement", op.name), a, values, 2);
    }

    // Indexed forms, with rA a register and with rA = 0 (the address is rB alone).
    const MemOp xloads[] = {{"lwzx", 23, 0, 0}, {"lwzux", 55, 1, 0}, {"lbzx", 87, 0, 0}, {"lbzux", 119, 1, 0},
                            {"lhzx", 279, 0, 0}, {"lhzux", 311, 1, 0}, {"lhax", 343, 0, 0}, {"lhaux", 375, 1, 0},
                            {"lwbrx", 534, 0, 0}, {"lhbrx", 790, 0, 0}, {"lwarx", 20, 0, 0}, {"lfsx", 535, 0, 1},
                            {"lfsux", 567, 1, 1}, {"lfdx", 599, 0, 1}, {"lfdux", 631, 1, 1}};
    const MemOp xstores[] = {{"stwx", 151, 0, 0}, {"stwux", 183, 1, 0}, {"stbx", 215, 0, 0}, {"stbux", 247, 1, 0},
                             {"sthx", 407, 0, 0}, {"sthux", 439, 1, 0}, {"stwbrx", 662, 0, 0}, {"sthbrx", 918, 0, 0},
                             {"stfsx", 663, 0, 1}, {"stfsux", 695, 1, 1}, {"stfdx", 727, 0, 1}, {"stfdux", 759, 1, 1},
                             {"stfiwx", 983, 0, 1}};
    for (const MemOp& op : xloads)
        for (int absolute = 0; absolute < (op.update ? 1 : 2); ++absolute) {
            Asm a;
            a << addi(28, 29, 0) << ldx(op.op, 5, absolute ? 0 : (op.update ? 28 : 29), 4);
            if (op.fp) a.ProbeF(5);
            else a.Probe(5);
            if (op.update) a.Probe(28);
            a << blr();
            AddProgram("memory", Fmt("%s%s", op.name, absolute ? " rA=0" : ""), a, OffsetStates(rng, absolute), 2);
        }
    for (const MemOp& op : xstores)
        for (int absolute = 0; absolute < (op.update ? 1 : 2); ++absolute) {
            Asm a;
            a << addi(28, 29, 0) << X(op.fp ? 3 : 3, absolute ? 0 : (op.update ? 28 : 29), 4, op.op);
            if (op.update) a.Probe(28);
            a << blr();
            States states = OffsetStates(rng, absolute);
            for (CpuContext& c : states) {
                c.gpr[3] = rng.Next();
                c.fpr[3].raw = RandDouble(rng);
            }
            AddProgram("memory", Fmt("%s%s", op.name, absolute ? " rA=0" : ""), a, states, 2);
        }
    {
        Asm a;  // stwcx. always succeeds here and sets cr0 from XER[SO]
        a << X(3, 29, 4, 150, 1) << mfcr(5);
        a.Probe(5);
        a << blr();
        AddProgram("memory", "stwcx.", a, OffsetStates(rng, false), 2);
    }

    // Load and store multiple, and string moves.
    {
        Asm a;
        for (uint32_t first : {3u, 20u, 28u, 31u})
            for (int32_t d : {0, 3, -0x80}) a << ld(47, first, d, 29);  // stmw
        a << blr();
        AddProgram("memory", "stmw", a, few, 2);
    }
    for (uint32_t first : {20u, 28u, 31u})
        for (int32_t d : {0, 3, -0x40}) {
            Asm a;  // lmw clobbers r29..r31 for the first two: the program ends there
            a << addi(19, 29, -0x100) << ld(46, first, d, 19) << blr();
            AddProgram("memory", Fmt("lmw r%u,%d", first, d), a, few, 3);
        }
    {
        Asm a;
        for (uint32_t n : {0u, 1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 13u, 16u, 31u}) {
            a << lswi(3, 29, n);
            for (uint32_t r = 3; r <= 10; ++r) a.Probe(r);
        }
        a << blr();
        AddProgram("memory", "lswi", a, few, 3);
        Asm b;
        for (uint32_t n : {0u, 1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 13u, 16u, 31u}) b << addi(28, 29, static_cast<int32_t>(n * 5)) << stswi(3, 28, n);
        b << blr();
        AddProgram("memory", "stswi", b, few, 2);
    }
    {
        States states;
        for (uint32_t count : {0u, 1u, 3u, 4u, 5u, 9u, 17u, 31u, 32u})
            for (uint32_t off : {0u, 1u, 3u}) {
                CpuContext c = BaseState(rng);
                c.xer = count | ((count & 1) ? kXerCa : 0u);
                c.gpr[4] = off;
                states.push_back(c);
            }
        Asm a;
        a << lswx(8, 29, 4);
        for (uint32_t r = 8; r <= 15; ++r) a.Probe(r);
        a << addi(28, 29, 0x200) << stswx(8, 28, 4) << blr();
        AddProgram("memory", "lswx stswx", a, states, 2);
    }
    {
        States states = OffsetStates(rng, false);
        Asm a;
        a << dcbz(29, 4) << dcbz_l(29, 4) << addi(28, 29, 0x40) << dcbz(28, 4) << blr();
        AddProgram("memory", "dcbz dcbz_l", a, states, 2);
        States absolute = OffsetStates(rng, true);
        Asm b;
        b << dcbz(0, 4) << blr();
        AddProgram("memory", "dcbz rA=0", b, absolute, 2);
    }
}

// === Directed programs: floating point =============================================================================
//
// Float programs run with FPSCR[NI] clear and set (ctx.fpscr bit 2: the host's flush-to-zero mode, which the context
// scope applies). Every operation is followed by mffs and a store of FPSCR (and, for the Rc forms, by mfcr), so the
// status bits each operation raised are seen, not only the sum of them at the end.

constexpr uint32_t kDoubleEdgeCount = sizeof kDoubleEdges / sizeof *kDoubleEdges;

struct FpOp {
    const char* name;
    uint32_t op, xo;
    enum Shape { AB, AC, ABC, B, X10 } shape;  // which operands the instruction reads; X10: an X-form opcode (xo is 10 bits)
};
uint32_t EncodeFp(const FpOp& o, uint32_t d, uint32_t a, uint32_t b, uint32_t c, uint32_t rc) {
    switch (o.shape) {
    case FpOp::AB: return ppc::fa(o.op, d, a, b, 0, o.xo, rc);
    case FpOp::AC: return ppc::fa(o.op, d, a, 0, c, o.xo, rc);
    case FpOp::ABC: return ppc::fa(o.op, d, a, b, c, o.xo, rc);
    case FpOp::B: return ppc::fa(o.op, d, 0, b, 0, o.xo, rc);
    case FpOp::X10: return ppc::fx(o.op, d, 0, b, o.xo, rc);
    }
    return 0;
}
// fmadd and its relatives take (a, c, b): the product's operands are frA and frC, the addend frB.
const FpOp kScalarOps[] = {
    {"fadd", 63, 21, FpOp::AB}, {"fsub", 63, 20, FpOp::AB}, {"fmul", 63, 25, FpOp::AC}, {"fdiv", 63, 18, FpOp::AB},
    {"fmadd", 63, 29, FpOp::ABC}, {"fmsub", 63, 28, FpOp::ABC}, {"fnmadd", 63, 31, FpOp::ABC}, {"fnmsub", 63, 30, FpOp::ABC},
    {"fsel", 63, 23, FpOp::ABC}, {"fadds", 59, 21, FpOp::AB}, {"fsubs", 59, 20, FpOp::AB}, {"fmuls", 59, 25, FpOp::AC},
    {"fdivs", 59, 18, FpOp::AB}, {"fmadds", 59, 29, FpOp::ABC}, {"fmsubs", 59, 28, FpOp::ABC}, {"fnmadds", 59, 31, FpOp::ABC},
    {"fnmsubs", 59, 30, FpOp::ABC}, {"fres", 59, 24, FpOp::B}, {"frsqrte", 63, 26, FpOp::B}, {"frsp", 63, 12, FpOp::X10},
    {"fctiw", 63, 14, FpOp::X10}, {"fctiwz", 63, 15, FpOp::X10}, {"fneg", 63, 40, FpOp::X10}, {"fmr", 63, 72, FpOp::X10},
    {"fnabs", 63, 136, FpOp::X10}, {"fabs", 63, 264, FpOp::X10},
};

// Operand states: f1, f2, f3 from the edge classes, FPSCR[NI] alternating.
States FloatPairStates(Rng& rng, bool full) {
    States states;
    size_t n = 0;
    for (uint32_t i = 0; i < kDoubleEdgeCount; ++i)
        for (uint32_t j = 0; j < kDoubleEdgeCount; ++j) {
            if (!full && rng.Below(4) != 0) continue;
            for (uint32_t ni : {0u, 4u}) {
                CpuContext c = BaseState(rng, ni);
                c.fpr[1].raw = kDoubleEdges[i];
                c.fpr[2].raw = kDoubleEdges[j];
                c.fpr[3].raw = (n++ % 3 == 0) ? kDoubleEdges[(i * 7 + j) % kDoubleEdgeCount] : RandDouble(rng);
                states.push_back(c);
            }
        }
    return states;
}
States FloatTripleStates(Rng& rng, unsigned count) {
    States states;
    for (unsigned i = 0; i < count; ++i) {
        CpuContext c = BaseState(rng, (i & 1) ? 4u : 0u);
        for (uint32_t r = 1; r <= 3; ++r) c.fpr[r].raw = rng.Chance(70) ? rng.Pick(kDoubleEdges) : RandDouble(rng);
        states.push_back(c);
    }
    return states;
}

void AddScalarFloatPrograms() {
    Rng rng(31);
    using namespace ppc;
    const States full = FloatPairStates(rng, true), sparse = FloatPairStates(rng, false), triples = FloatTripleStates(rng, 2000);
    for (const FpOp& op : kScalarOps)
        for (uint32_t rc = 0; rc < 2; ++rc) {
            Asm a;
            const uint32_t forms[][4] = {{8, 1, 2, 3}, {9, 2, 2, 1}, {1, 1, 1, 1}, {10, 3, 1, 3}};
            for (const auto& f : forms) {
                a << EncodeFp(op, f[0], f[1], f[2], f[3], rc);
                if (rc) {
                    a << mfcr(6);
                    a.Probe(6);
                }
                a << mffs(0);
                a.ProbeF(0);
                a.ProbeF(f[0]);
            }
            a << blr();
            const bool three = op.shape == FpOp::ABC;
            AddProgram("float", Fmt("%s rc=%u", op.name, rc), a, three ? triples : (rc ? sparse : full));
        }
    // Compares into every CR field, ordered and unordered.
    {
        Asm a;
        for (uint32_t crf = 0; crf < 8; ++crf) {
            const uint32_t x = crf % 3 + 1, y = (crf + 1) % 3 + 1;
            a << (crf & 1 ? fcmpo(crf, x, y) : fcmpu(crf, x, y));
        }
        a << mfcr(5);
        a.Probe(5);
        a << mffs(0);
        a.ProbeF(0);
        a << blr();
        AddProgram("float", "fcmpu fcmpo every field", a, FloatTripleStates(rng, 1500));
    }
    // A compare and the bc right after it on each bit of its field, taken when set and when clear (the JIT branches on the
    // compare's flags), with the field read after the branch on both paths or overwritten on the path that goes on (so the
    // block writes it on the taken path only), and FPRs written before the compare (the bc stores them first).
    {
        const States st = FloatTripleStates(rng, 600);
        for (uint32_t bit = 0; bit < 4; ++bit)
            for (uint32_t want = 0; want < 2; ++want)
                for (uint32_t dead = 0; dead < 2; ++dead) {
                    Asm a;
                    const Label taken = a.NewLabel();
                    a << li(5, 0);
                    a << fx(63, 4, 0, 1, 72);  // fmr f4, f1: a dirty FPR cache slot at the branch
                    a << (bit & 1 ? fcmpo(1, 1, 2) : fcmpu(1, 1, 2));
                    a.Bc(want ? 12u : 4u, 4u + bit, taken);
                    a << li(5, 1);
                    if (dead) a << fcmpu(1, 2, 3);
                    a.Bind(taken);
                    a << mfcr(6);
                    a.Probe(6);
                    a.Probe(5);
                    a.ProbeF(4);
                    a << blr();
                    AddProgram("float", Fmt("fcmpu bc bit %u %s %s", bit, want ? "set" : "clear", dead ? "field dead" : "field live"), a, st);
                }
    }
    // FPSCR moves. The register image and the status bits go through the helpers every engine shares.
    States fpscrStates;
    const uint32_t fpscrs[] = {0u, 4u, 0xFFFFFFFFu, 0xA5A5A5A5u, 0x000000F8u, 0xFFFFE0FFu, 0x0000F000u, 0x80000000u, 0x1F80F000u};
    for (unsigned i = 0; i < 90; ++i) {
        CpuContext c = BaseState(rng, i < 9 ? fpscrs[i] : rng.Pick(fpscrs) ^ (rng.Chance(50) ? rng.Next() : 0u));
        c.fpr[1].raw = (static_cast<uint64_t>(rng.Next()) << 32) | (rng.Chance(30) ? rng.Pick(kIntEdges) : rng.Next());
        c.fpr[2].raw = RandDouble(rng);
        c.fpr[3].raw = RandDouble(rng);
        fpscrStates.push_back(c);
    }
    for (uint32_t rc = 0; rc < 2; ++rc) {
        Asm a;
        for (uint32_t fm : {0xFFu, 0x01u, 0x80u, 0x0Fu, 0xF0u, 0xA5u, 0x00u, 0x3Cu}) {
            a << mtfsf(fm, 1, rc);
            if (rc) {
                a << mfcr(6);
                a.Probe(6);
            }
            a << mffs(0);
            a.ProbeF(0);
        }
        a << blr();
        AddProgram("float", Fmt("mtfsf rc=%u", rc), a, fpscrStates);
        Asm b;
        for (uint32_t crf = 0; crf < 8; ++crf) {
            b << mtfsfi(crf, (crf * 5 + 3) & 15, rc);
            if (rc) {
                b << mfcr(6);
                b.Probe(6);
            }
            b << mffs(0);
            b.ProbeF(0);
        }
        b << blr();
        AddProgram("float", Fmt("mtfsfi rc=%u", rc), b, fpscrStates);
        Asm c;
        for (uint32_t bit = 0; bit < 32; ++bit) {
            c << mtfsb1(bit, rc) << mffs(0);
            c.ProbeF(0);
        }
        for (uint32_t bit = 0; bit < 32; ++bit) {
            c << mtfsb0(bit, rc) << mffs(0);
            c.ProbeF(0);
        }
        c << blr();
        AddProgram("float", Fmt("mtfsb1 mtfsb0 every bit rc=%u", rc), c, fpscrStates);
    }
    {
        Asm a;
        for (uint32_t crfd : {0u, 3u, 7u})
            for (uint32_t crfs : {0u, 1u, 3u, 4u, 5u, 7u}) {
                a << mcrfs(crfd, crfs) << mfcr(5) << mffs(0);
                a.Probe(5);
                a.ProbeF(0);
            }
        a << blr();
        AddProgram("float", "mcrfs", a, fpscrStates);
    }
    // NI switched on and off inside the program, over operands that differ between the modes.
    {
        Asm a;
        a << fa(59, 8, 1, 0, 2, 25) << fa(63, 9, 1, 2, 0, 21) << mtfsb1(29) << fa(59, 10, 1, 0, 2, 25) << fa(63, 11, 1, 2, 0, 21)
          << fx(63, 12, 0, 1, 12) << fa(59, 13, 1, 2, 0, 21) << mtfsb0(29) << fa(59, 14, 1, 0, 2, 25) << fa(63, 15, 1, 2, 0, 21)
          << fx(63, 16, 0, 1, 12) << mffs(0);
        a.ProbeF(0);
        a << blr();
        AddProgram("float", "NI toggled inside", a, FloatPairStates(rng, false));
    }
}

// === Directed programs: paired singles ==============================================================================

// f1, f2 and f3 become packed pairs of floats built from r3..r8 (ps0 first), as psq_l with GQR0 = 0 loads them.
void LoadPairs(Asm& a) {
    using namespace ppc;
    for (uint32_t i = 0; i < 6; ++i) a << stw(3 + i, 0x700 + 4 * static_cast<int32_t>(i), 29);
    a << psq(56, 1, 0x700, 29, 0, 0) << psq(56, 2, 0x708, 29, 0, 0) << psq(56, 3, 0x710, 29, 0, 0);
}
States PairedStates(Rng& rng, unsigned count) {
    States states;
    for (unsigned i = 0; i < count; ++i) {
        CpuContext c = BaseState(rng, (i & 1) ? 4u : 0u);
        for (uint32_t r = 3; r <= 8; ++r) c.gpr[r] = rng.Chance(75) ? rng.Pick(kSingleEdges) : rng.Next();
        c.gqr[0] = 0;
        states.push_back(c);
    }
    return states;
}

struct PsOp { const char* name; uint32_t xo; enum Shape { AB, AC, ABC, B, X10 } shape; };
const PsOp kPsOps[] = {
    {"ps_add", 21, PsOp::AB}, {"ps_sub", 20, PsOp::AB}, {"ps_div", 18, PsOp::AB}, {"ps_mul", 25, PsOp::AC},
    {"ps_muls0", 12, PsOp::AC}, {"ps_muls1", 13, PsOp::AC}, {"ps_madd", 29, PsOp::ABC}, {"ps_msub", 28, PsOp::ABC},
    {"ps_nmadd", 31, PsOp::ABC}, {"ps_nmsub", 30, PsOp::ABC}, {"ps_madds0", 14, PsOp::ABC}, {"ps_madds1", 15, PsOp::ABC},
    {"ps_sum0", 10, PsOp::ABC}, {"ps_sum1", 11, PsOp::ABC}, {"ps_sel", 23, PsOp::ABC}, {"ps_res", 24, PsOp::B},
    {"ps_rsqrte", 26, PsOp::B}, {"ps_neg", 40, PsOp::X10}, {"ps_mr", 72, PsOp::X10}, {"ps_nabs", 136, PsOp::X10},
    {"ps_abs", 264, PsOp::X10}, {"ps_merge00", 528, PsOp::AB}, {"ps_merge01", 560, PsOp::AB},
    {"ps_merge10", 592, PsOp::AB}, {"ps_merge11", 624, PsOp::AB},
};

uint32_t EncodePs(const PsOp& op, uint32_t d, uint32_t x, uint32_t y, uint32_t z, uint32_t rc) {
    using namespace ppc;
    switch (op.shape) {
    case PsOp::AB: return op.xo >= 512 ? fx(4, d, x, y, op.xo, rc) : fa(4, d, x, y, 0, op.xo, rc);
    case PsOp::AC: return fa(4, d, x, 0, z, op.xo, rc);
    case PsOp::ABC: return fa(4, d, x, y, z, op.xo, rc);
    case PsOp::B: return fa(4, d, 0, y, 0, op.xo, rc);
    case PsOp::X10: return fx(4, d, 0, y, op.xo, rc);
    }
    return 0u;
}

void AddPairedSinglePrograms() {
    Rng rng(32);
    using namespace ppc;
    const States states = PairedStates(rng, 500);
    for (const PsOp& op : kPsOps)
        for (uint32_t rc = 0; rc < 2; ++rc) {
            Asm a;
            LoadPairs(a);
            const uint32_t forms[][4] = {{8, 1, 2, 3}, {9, 2, 2, 1}, {1, 1, 1, 1}, {10, 3, 1, 3}};
            for (const auto& f : forms) {
                a << EncodePs(op, f[0], f[1], f[2], f[3], rc);
                if (rc) {
                    a << mfcr(6);
                    a.Probe(6);
                }
                a << mffs(0);
                a.ProbeF(0);
            }
            a << blr();
            AddProgram("paired", Fmt("%s rc=%u", op.name, rc), a, states);
        }
    {
        Asm a;
        LoadPairs(a);
        uint32_t crf = 0;
        for (uint32_t xo : {0u, 32u, 64u, 96u})
            for (uint32_t pair = 0; pair < 2; ++pair) {
                a << ps_cmp(xo, crf, 1 + pair, 2 - pair) << mfcr(5);
                a.Probe(5);
                crf = (crf + 3) & 7;
            }
        a << mffs(0);
        a.ProbeF(0);
        a << blr();
        AddProgram("paired", "ps_cmpu0 ps_cmpo0 ps_cmpu1 ps_cmpo1", a, PairedStates(rng, 400));
    }

    // Quantized loads and stores: every GQR (types 0, 4..7, several scales), W = 0 and 1, update and indexed forms.
    {
        States qstates;
        for (unsigned i = 0; i < 64; ++i) {
            CpuContext c = BaseState(rng, (i & 1) ? 4u : 0u);
            for (auto& q : c.gqr) q = RandGqr(rng);
            if (i < 6) {  // each plain type with scale 0, then a scaled one
                const uint32_t types[] = {0, 4, 5, 6, 7, 5};
                for (auto& q : c.gqr) q = (types[i] << 16) | types[i] | (i == 5 ? (3u << 24) | (3u << 8) : 0u);
            }
            c.gpr[4] = (rng.Below(64)) * 2;
            qstates.push_back(c);
        }
        Asm a;
        for (uint32_t i = 0; i < 8; ++i)
            for (uint32_t w = 0; w < 2; ++w) {
                a << psq(56, 8 + i + 8 * w, static_cast<int32_t>(0x20 * i + 0x300 * w + (i & 3)), 29, w, i);
            }
        for (uint32_t i = 0; i < 8; ++i)
            for (uint32_t w = 0; w < 2; ++w) a << psq(60, 8 + i + 8 * w, static_cast<int32_t>(0x40 * i + 0x400 * w + 2 * i), 30, w, i);
        for (uint32_t i = 0; i < 8; ++i) {  // stores of scalar-form registers (f1, f2: arbitrary doubles) through each GQR
            a << psq(60, 1, 0x7C0 - static_cast<int32_t>(0x20 * i), 29, i & 1, i) << psq(60, 2, 0x700 - static_cast<int32_t>(0x20 * i), 29, 0, i);
        }
        for (uint32_t i = 0; i < 8; ++i) {
            a << addi(28, 29, 0x100) << psq(57, 24, 8, 28, i & 1, i);
            a.Probe(28);
            a << addi(28, 29, 0x100) << psq(61, 24, 8, 28, (i + 1) & 1, i);
            a.Probe(28);
        }
        for (uint32_t i = 0; i < 8; ++i) {
            a << psqx(6, 25, 29, 4, i & 1, i) << psqx(7, 25, 30, 4, (i + 1) & 1, i);
            a << addi(28, 29, 0x80) << psqx(38, 26, 28, 4, i & 1, i);
            a.Probe(28);
            a << addi(28, 30, 0x80) << psqx(39, 26, 28, 4, (i + 1) & 1, i);
            a.Probe(28);
        }
        a << blr();
        AddProgram("paired", "psq_l psq_st (u, x, ux) every GQR", a, qstates, 3);
    }

    // The same register used as a scalar and as a pair, in both directions (the form conversions): hand-written mixes.
    const States mixed = PairedStates(rng, 300);
    {
        Asm a;
        LoadPairs(a);
        a << fa(63, 8, 1, 2, 0, 21) << fa(4, 9, 8, 8, 0, 21) << fx(63, 10, 0, 9, 72) << fa(63, 11, 10, 10, 0, 21);
        a.ProbeF(8);
        a.ProbeF(9);
        a.ProbeF(10);
        a.ProbeF(11);
        a << li(27, 0x20) << psq(60, 10, 0x20, 30, 0, 0) << ld(52, 9, 0x30, 30) << stfiwx(11, 30, 27) << fx(63, 14, 0, 11, 15);
        a.ProbeF(14);
        a << blr();
        AddProgram("paired", "scalar-paired-scalar chain", a, mixed);
    }
    {
        Asm a;
        a << ld(48, 4, 0x10, 29) << fa(4, 5, 4, 1, 0, 21) << fx(4, 6, 4, 5, 528) << fa(59, 7, 6, 5, 0, 21) << fx(4, 8, 7, 4, 592)
          << fx(4, 9, 0, 8, 72) << fx(63, 10, 0, 4, 72) << fx(4, 11, 0, 10, 72) << fx(4, 12, 0, 1, 72) << fx(63, 13, 0, 5, 72)
          << fa(63, 14, 5, 5, 0, 21) << fa(4, 15, 14, 14, 0, 21) << fa(59, 16, 15, 4, 0, 20) << fx(63, 17, 0, 16, 14)
          << fx(63, 18, 0, 8, 12);
        for (uint32_t r = 5; r <= 18; ++r) a.ProbeF(r);
        a << blr();
        AddProgram("paired", "form copies (fmr, ps_mr) and conversions", a, mixed);
    }
    {
        Asm a;
        a << ld(50, 4, 0x18, 29) << ld(48, 5, 0x24, 29) << fa(4, 6, 4, 5, 0, 20) << fa(63, 7, 6, 4, 0, 18) << fx(4, 8, 6, 7, 560)
          << fa(4, 9, 8, 6, 0, 18) << li(27, 0x28) << stfiwx(9, 30, 27) << fx(63, 10, 0, 9, 264) << fx(63, 11, 0, 8, 136) << fx(4, 12, 0, 8, 40)
          << fx(63, 13, 0, 12, 40);
        for (uint32_t r = 6; r <= 13; ++r) a.ProbeF(r);
        a << psq(60, 8, 0x40, 30, 0, 0) << psq(60, 9, 0x50, 30, 1, 0) << ld(54, 8, 0x60, 30) << ld(52, 8, 0x68, 30)
          << ld(54, 12, 0x70, 30) << blr();
        AddProgram("paired", "loads, arithmetic and stores in both forms", a, mixed, 2);
    }
}

// === Control flow ==================================================================================================
//
// Programs that change lr start with Save (mflr r31) and end with Done (mtlr r31, blr). Labels make the branches
// relative, so a program means the same wherever it is placed.

States CrCtrStates(Rng& rng) {
    States states;
    const uint32_t patterns[] = {0u, 0xFFFFFFFFu, 0xAAAAAAAAu, 0x55555555u, 0x0F0F0F0Fu, 0xF0F0F0F0u, 0x12345678u, 0x87654321u};
    for (uint32_t cr : patterns)
        for (uint32_t ctr : {0u, 1u, 2u, 3u, 0x80000000u, 0xFFFFFFFFu}) {
            for (bool alternate : {false, true}) {  // the nibbles as table values, and as compare differences that read the same
                CpuContext c = BaseState(rng);
                JitTestCr::SetPacked(c, cr, alternate, [&] { return rng.Next(); });
                c.ctr = ctr;
                states.push_back(c);
            }
        }
    return states;
}

// bc, bclr, bcctr and their link forms, every BO, a few BI, taken and not, with ctr counting down.
void AddBranchForms() {
    Rng rng(41);
    using namespace ppc;
    const States states = CrCtrStates(rng);
    for (uint32_t bo = 0; bo < 32; ++bo)
        for (uint32_t bi : {0u, 2u, 5u, 31u}) {
            for (int backward = 0; backward < 2; ++backward) {
                Asm a;
                Label taken = a.NewLabel(), end = a.NewLabel(), main = a.NewLabel();
                a.Save();
                if (backward) a.B(main);
                if (!backward) a.Bc(bo, bi, taken);
                if (backward) {
                    a.Bind(taken);
                    a << li(5, 2);
                    a.B(end);
                    a.Bind(main);
                    a.Bc(bo, bi, taken);
                }
                a << li(5, 1);
                if (!backward) {
                    a.B(end);
                    a.Bind(taken);
                    a << li(5, 2);
                }
                a.Bind(end);
                a.Done();
                AddProgram("control", Fmt("bc BO=%u BI=%u %s", bo, bi, backward ? "backward" : "forward"), a, states);
            }
            // Through lr and ctr, set to the taken label with bl .+4 to read the pc.
            for (int kind = 0; kind < 4; ++kind) {
                Asm a;
                Label get = a.NewLabel(), taken = a.NewLabel(), end = a.NewLabel();
                a.Save();
                a.B(get, true);
                a.Bind(get);
                a << mflr(6);
                a.AddiDiff(6, 6, taken, get);
                a << (kind < 2 ? mtlr(6) : mtctr(6));
                const bool link = kind & 1;
                a << (kind < 2 ? bclr(bo, bi, link) : bcctr(bo, bi, link)) << li(5, 1);
                a.B(end);
                a.Bind(taken);
                a << li(5, 2) << mflr(8);
                a.Probe(8);
                a.Bind(end);
                a.Done();
                const char* names[] = {"bclr", "bclrl", "bcctr", "bcctrl"};
                AddProgram("control", Fmt("%s BO=%u BI=%u", names[kind], bo, bi), a, states);
            }
        }
}

// A compare directly followed by a conditional branch (the interpreter runs the pair in one dispatch), on the compare's
// field and on another, with every BO that tests a bit or ctr.
void AddFusedPairs() {
    Rng rng(42);
    using namespace ppc;
    States states;
    const uint32_t values[] = {0u, 1u, 2u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu, 5u, 0xFFFFFFFBu};
    for (uint32_t x : values)
        for (uint32_t y : values)
            for (uint32_t ctr : {0u, 1u, 5u}) {
                CpuContext c = BaseState(rng);
                c.gpr[3] = x;
                c.gpr[4] = y;
                c.ctr = ctr;
                c.xer = (c.xer & ~kXerSo) | ((x ^ y) & 1 ? kXerSo : 0u);
                states.push_back(c);
            }
    for (int kind = 0; kind < 5; ++kind)
        for (uint32_t crf : (kind == 4 ? std::vector<uint32_t>{0u} : std::vector<uint32_t>{0u, 1u, 6u})) {
            const uint32_t other = ((crf + 3) & 7) * 4 + 2;
            const uint32_t options[][2] = {{12, crf * 4}, {12, crf * 4 + 1}, {12, crf * 4 + 2}, {4, crf * 4 + 2}, {4, crf * 4 + 1},
                                           {12, other}, {16, crf * 4 + 2}, {18, crf * 4 + 2}, {8, crf * 4 + 2}, {0, crf * 4},
                                           {10, crf * 4 + 2}, {2, crf * 4 + 1}, {20, 0}, {13, crf * 4 + 3}};
            for (const auto& o : options) {
                Asm a;
                Label taken = a.NewLabel(), end = a.NewLabel();
                a.Save();
                switch (kind) {
                case 0: a << cmpwi(crf, 3, 2); break;
                case 1: a << cmplwi(crf, 3, 2); break;
                case 2: a << cmpw(crf, 3, 4); break;
                case 3: a << cmplw(crf, 3, 4); break;
                default: a << rlwinm(5, 3, 4, 0, 27, 1); break;
                }
                a.Bc(o[0], o[1], taken);
                a << li(6, 1);
                a.B(end);
                a.Bind(taken);
                a << li(6, 2);
                a.Bind(end);
                a.Done();
                const char* names[] = {"cmpwi", "cmplwi", "cmpw", "cmplw", "rlwinm."};
                AddProgram("control", Fmt("%s cr%u + bc BO=%u BI=%u", names[kind], crf, o[0], o[1]), a, states);
            }
        }
}

// A compare or record form and the bc that tests its field, not adjacent (a followed b between) or adjacent but not a fused kind (the
// record forms): compiled as one pair. The result register read later or not (cached or not), in place or not, the field read or
// overwritten after the branch, the ctr forms.
void AddSplitPairs() {
    Rng rng(43);
    using namespace ppc;
    States states;
    const uint32_t values[] = {0u, 1u, 0x40u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu, 0x8001u, 0xF0F00000u};
    for (uint32_t x : values)
        for (uint32_t y : {0u, 0x40u, 0x80000000u, 0xFFFFFFFFu}) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = x;
            c.gpr[4] = y;
            c.ctr = rng.Below(3);
            states.push_back(c);
        }
    struct First { const char* name; uint32_t word; uint32_t field; };
    const First firsts[] = {
        {"andi. r5,r3,0x8001", andi_(5, 3, 0x8001), 0}, {"andi. r3,r3,0xf0", andi_(3, 3, 0xF0), 0},
        {"andis. r5,r3,0x8000", andis_(5, 3, 0x8000), 0}, {"andis. r5,r3,0xf0", andis_(5, 3, 0xF0), 0},
        {"rlwinm. r5,r3,26,31,31", rlwinm(5, 3, 26, 31, 31, 1), 0}, {"rlwinm. r5,r3,4,0,27", rlwinm(5, 3, 4, 0, 27, 1), 0},
        {"rlwinm. r3,r3,0,16,31", rlwinm(3, 3, 0, 16, 31, 1), 0}, {"rlwinm. r5,r3,0,0,31", rlwinm(5, 3, 0, 0, 31, 1), 0},
        {"rlwinm. r5,r3,8,0,31", rlwinm(5, 3, 8, 0, 31, 1), 0}, {"and. r5,r3,r4", logic(28, 5, 3, 4, 1), 0},
        {"and. r3,r3,r4", logic(28, 3, 3, 4, 1), 0}, {"and. r4,r3,r4", logic(28, 4, 3, 4, 1), 0},
        {"or. r5,r3,r4", logic(444, 5, 3, 4, 1), 0}, {"mr. r5,r3", logic(444, 5, 3, 3, 1), 0},
        {"mr. r3,r3", logic(444, 3, 3, 3, 1), 0}, {"cmpwi cr2,r3,0x40", cmpwi(2, 3, 0x40), 2},
        {"cmplwi cr0,r3,0x8001", cmplwi(0, 3, 0x8001), 0}, {"cmpw cr7,r3,r4", cmpw(7, 3, 4), 7},
        {"cmplw cr0,r4,r3", cmplw(0, 4, 3), 0}};
    // pre: a float op before the pair whose form write is pending in the FPR cache (ps_merge00: Paired, an `or` of the form
    // mask when stored; frsp: Scalar, an `and`), stored before the branch: not between a record form and the branch on its flags
    const uint32_t pres[] = {0u, fx(4, 5, 1, 2, 528), fx(63, 6, 0, 1, 12)};
    for (const First& f : firsts)
        for (uint32_t pre = 0; pre < 3; ++pre)
        for (int split = 0; split < 2; ++split)
            for (int after = 0; after < 3; ++after)
                for (const uint32_t bo : {12u, 4u, 8u, 0u})
                    for (uint32_t bit = 0; bit < 3; ++bit) {
                        if ((bo == 8u || bo == 0u) && (bit != 2 || after != 0)) continue;
                        if (pre != 0 && after != 0) continue;
                        Asm a;
                        Label next = a.NewLabel(), taken = a.NewLabel(), end = a.NewLabel();
                        a.Save();
                        if (pre != 0) a << pres[pre];
                        a << f.word;
                        if (split) {
                            a.B(next);
                            a.Bind(next);
                        }
                        a.Bc(bo, f.field * 4 + bit, taken);
                        if (after == 0) a << addi(7, 5, 1) << addi(9, 3, 1);  // the result read after the branch
                        else if (after == 1) a << mfcr(8);                    // the field read
                        else a << cmpwi(f.field, 4, 3);                       // the field overwritten
                        a << li(6, 1);
                        a.B(end);
                        a.Bind(taken);
                        a << li(6, 2) << mfcr(8);
                        a.Bind(end);
                        a.Done();
                        AddProgram("control", Fmt("%s%s%s; bc %u,%u; after %d", pre == 1 ? "ps_merge00 f5,f1,f2; " : pre == 2 ? "frsp f6,f1; " : "",
                                                  f.name, split ? "; b .+4" : "", bo, f.field * 4 + bit, after), a, states);
                    }
}

void AddLoops() {
    Rng rng(43);
    using namespace ppc;
    States counts;
    for (uint32_t n : {1u, 2u, 3u, 5u, 64u, 1000u}) {
        CpuContext c = BaseState(rng);
        c.gpr[3] = n;
        c.gpr[7] = 1 + n % 3;
        c.gpr[4] = rng.Below(20);
        counts.push_back(c);
    }
    {
        Asm a;
        Label loop = a.NewLabel();
        a << mtctr(3);
        a.Bind(loop);
        a << addi(4, 4, 3) << logic(316, 5, 5, 4) << arith(266, 6, 6, 5, 0, 0);
        a.Bc(16, 0, loop);
        a << blr();
        AddProgram("control", "bdnz loop", a, counts);
    }
    {
        Asm a;  // a loop of loops: ctr inside, a counter register outside
        Label outer = a.NewLabel(), inner = a.NewLabel();
        a << mr(8, 7);
        a.Bind(outer);
        a << mtctr(3);
        a.Bind(inner);
        a << addi(4, 4, 1) << arith(235, 5, 4, 4, 0, 0);
        a.Bc(16, 0, inner);
        a << addic_(8, 8, -1);
        a.Bc(4, 2, outer);
        a << blr();
        States nested = counts;
        for (CpuContext& c : nested) c.gpr[3] = 1 + c.gpr[3] % 33;
        AddProgram("control", "nested loops", a, nested);
    }
    {
        Asm a;  // a while loop closed by a backward b, left by a conditional branch in the middle of the body
        Label loop = a.NewLabel(), skip = a.NewLabel(), exit = a.NewLabel();
        a.Bind(loop);
        a << addi(4, 4, 7) << cmpw(0, 4, 3);
        a.Bc(4, 0, exit);
        a << andi_(6, 4, 3);
        a.Bc(12, 2, skip);
        a << addi(5, 5, 1);
        a.Bind(skip);
        a << xori(7, 7, 0x55);
        a.B(loop);
        a.Bind(exit);
        a << blr();
        States whiles;
        for (uint32_t k : {1u, 10u, 100u, 1000u}) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = k;
            c.gpr[4] = rng.Below(20);
            whiles.push_back(c);
        }
        AddProgram("control", "while loop closed by b", a, whiles);
    }
    for (uint32_t bo : {0u, 8u}) {  // bdnzf / bdnzt: the counter and a CR bit together
        Asm a;
        Label loop = a.NewLabel();
        a << mtctr(3);
        a.Bind(loop);
        a << addi(4, 4, 1) << cmpwi(0, 4, 5);
        a.Bc(bo, 2, loop);
        a << blr();
        States st;
        for (uint32_t n : {1u, 3u, 10u, 100u})
            for (uint32_t r4 : {0u, 1u, 4u, 5u, 6u}) {
                CpuContext c = BaseState(rng);
                c.gpr[3] = n;
                c.gpr[4] = r4;
                st.push_back(c);
            }
        AddProgram("control", Fmt("bdnz%s loop", bo ? "t" : "f"), a, st);
    }
    // Loop bodies and straight-line runs around the block length limit (32 instructions) and its multiples.
    const auto body = [&](Asm& a, unsigned n) {
        for (unsigned i = 0; i < n; ++i) {
            switch (i % 4) {
            case 0: a << addi(4, 4, static_cast<int32_t>(i + 1)); break;
            case 1: a << arith(266, 5, 5, 4, 0, 0); break;
            case 2: a << rlwinm(5, 5, (i % 29) + 1, 0, 31); break;
            default: a << logic(316, 6, 6, 5); break;
            }
        }
    };
    for (unsigned n : {1u, 2u, 15u, 28u, 29u, 30u, 31u, 32u, 33u, 34u, 35u, 36u, 62u, 63u, 64u, 65u, 66u, 95u, 96u, 97u, 130u}) {
        Asm a;
        Label loop = a.NewLabel();
        a << mtctr(3);
        a.Bind(loop);
        body(a, n);
        a.Bc(16, 0, loop);
        a << blr();
        States st;
        for (uint32_t k : {1u, 3u}) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = k;
            st.push_back(c);
        }
        AddProgram("control", Fmt("loop body of %u instructions", n), a, st);
        Asm b;
        body(b, n);
        b << blr();
        AddProgram("control", Fmt("straight run of %u instructions", n), b, st);
        // A conditional branch in each position near the limit.
        if (n >= 29 && n <= 36)
            for (unsigned at : {29u, 30u, 31u, 32u, 33u, 34u}) {
                Asm c;
                Label done = c.NewLabel();
                for (unsigned i = 0; i < n; ++i) {
                    if (i == at) {
                        c << cmpwi(0, 3, 1);
                        c.Bc(12, 2, done);
                    }
                    c << addi(4, 4, static_cast<int32_t>(i * 3 + 1));
                }
                c.Bind(done);
                c << blr();
                AddProgram("control", Fmt("%u instructions, bc at %u", n, at), c, st);
            }
    }
    {
        Asm a;  // leaves the block at each of many conditional branches in turn
        Label done = a.NewLabel(), done2 = a.NewLabel();
        for (uint32_t i = 0; i < 12; ++i) {
            a << cmpwi(0, 3, static_cast<int32_t>(i));
            a.Bc(12, 2, done);
            a << addi(4, 4, static_cast<int32_t>(i * 3 + 1));
            a << andi_(6, 3, 1u << i);
            a.Bc(4, 2, done2);
            a << addi(5, 5, static_cast<int32_t>(i + 1));
        }
        a.Bind(done);
        a << li(7, 1);
        a.Bind(done2);
        a << blr();
        States st;
        for (int32_t k = -1; k <= 13; ++k) {
            CpuContext c = BaseState(rng);
            c.gpr[3] = static_cast<uint32_t>(k);
            st.push_back(c);
        }
        AddProgram("control", "bc leaving a block at each branch", a, st);
    }
    {
        Asm a;  // conditional returns, taken when lr is the frame's own return address
        a << cmpwi(0, 3, 0);
        a << bclr(12, 2) << addi(4, 4, 1) << cmpwi(1, 3, 1) << bclr(4, 6) << addi(5, 5, 1) << bclr(16, 0) << addi(6, 6, 1)
          << bclr(18, 0) << addi(7, 7, 1) << bclr(12, 0) << addi(8, 8, 1) << blr();
        States st;
        for (int32_t k : {0, 1, 2, -1})
            for (uint32_t ctr : {0u, 1u, 2u, 100u}) {
                CpuContext c = BaseState(rng);
                c.gpr[3] = static_cast<uint32_t>(k);
                c.ctr = ctr;
                st.push_back(c);
            }
        AddProgram("control", "conditional returns", a, st);
    }
}

// b chains: more followed segments than a block holds, forward and backward, with other branches in the pieces.
void AddChains() {
    Rng rng(44);
    using namespace ppc;
    States st;
    for (int i = 0; i < 4; ++i) st.push_back(BaseState(rng));
    for (unsigned n : {2u, 3u, 5u, 6u, 7u, 8u, 12u})
        for (int order = 0; order < 3; ++order) {
            // Piece k starts at word position[k]; piece 0 at 0.
            std::vector<size_t> position(n);
            for (unsigned k = 0; k < n; ++k)
                position[k] = order == 0 ? 8 * k : order == 1 ? (k == 0 ? 0 : 8 * (n - k) + 8) : (k == 0 ? 0 : 8 * ((k * 5) % n + 1));
            if (order == 2) {  // pieces must not share a position
                std::vector<size_t> seen;
                bool clash = false;
                for (size_t p : position) {
                    if (std::find(seen.begin(), seen.end(), p) != seen.end()) clash = true;
                    seen.push_back(p);
                }
                if (clash) continue;
            }
            std::vector<unsigned> byPosition(n);
            for (unsigned k = 0; k < n; ++k) byPosition[k] = k;
            std::sort(byPosition.begin(), byPosition.end(), [&](unsigned x, unsigned y) { return position[x] < position[y]; });
            Asm a;
            std::vector<Label> piece(n);
            for (auto& l : piece) l = a.NewLabel();
            for (unsigned k : byPosition) {
                a.Org(position[k]);
                a.Bind(piece[k]);
                a << addi(3, 3, static_cast<int32_t>(k + 1)) << xori(4, 4, k * 3 + 1);
                if (k % 3 == 1) {
                    a << cmpwi(0, 3, 1000);
                    a.Bc(12, 2, piece[n - 1]);  // never taken for the values here: a branch inside a followed piece
                }
                if (k + 1 < n) a.B(piece[k + 1]);
                else a << blr();
            }
            AddProgram("control", Fmt("b chain of %u pieces, order %d", n, order), a, st);
        }
}

// bl to local code (a leaf returns into the middle of the caller), nested calls, bl .+4, bcl, jump tables through ctr and
// lr, calls through ctr and lr.
void AddLocalCalls() {
    Rng rng(45);
    using namespace ppc;
    States st;
    for (int i = 0; i < 8; ++i) {
        CpuContext c = BaseState(rng);
        c.gpr[3] = i < 4 ? static_cast<uint32_t>(i) : rng.Next();
        st.push_back(c);
    }
    {
        Asm a;
        Label leaf = a.NewLabel();
        a.Save();
        a << li(3, 1);
        a.B(leaf, true);
        a << addi(4, 3, 0);
        a.B(leaf, true);
        a << mflr(5);
        a.Probe(5);
        a.Done();
        a.Bind(leaf);
        a << addi(3, 3, 10) << blr();
        AddProgram("control", "bl to a local leaf", a, st);
    }
    {
        Asm a;
        Label b = a.NewLabel(), c = a.NewLabel();
        a.Save();
        a.B(b, true);
        a << addi(9, 3, 0);
        a.Done();
        a.Bind(b);
        a << mflr(10) << addi(3, 3, 1);
        a.B(c, true);
        a << mtlr(10) << blr();
        a.Bind(c);
        a << rlwinm(3, 3, 1, 0, 30) << blr();
        AddProgram("control", "nested local calls", a, st);
    }
    {
        Asm a;
        Label l1 = a.NewLabel(), l2 = a.NewLabel(), l3 = a.NewLabel();
        a.Save();
        a.B(l1, true);
        a.Bind(l1);
        a << mflr(5);
        a.Probe(5);
        a.Bc(20, 31, l2, true);
        a.Bind(l2);
        a << mflr(6);
        a.Probe(6);
        a.Bc(12, 2, l3, true);  // a conditional call: lr changes only when it is taken
        a.Bind(l3);
        a << mflr(7);
        a.Probe(7);
        a.Done();
        States cr = st;
        uint32_t lcg = 12345;
        for (size_t i = 0; i < cr.size(); ++i)
            JitTestCr::SetPacked(cr[i], (i & 1) ? 0x20000000u : 0u, (i & 2) != 0, [&] { return lcg = lcg * 1664525u + 1013904223u; });
        AddProgram("control", "bl .+4, bcl 20,31, conditional bcl", a, cr);
    }
    {
        Asm a;
        Label fn = a.NewLabel();
        a.Save();
        a.LiLabel(6, fn);
        a << mtctr(6) << bctrl() << addi(4, 3, 0);
        a.LiLabel(6, fn);
        a << mtlr(6) << blrl() << mflr(8);
        a.Probe(8);
        a.Done();
        a.Bind(fn);
        a << addi(3, 3, 5) << blr();
        AddProgram("control", "bctrl and blrl to local code", a, st);
    }
    {
        Asm a;
        Label target = a.NewLabel();
        a.Save();
        a.LiLabel(6, target);
        a << mtlr(6) << blr();  // a jump through lr: not the frame's return
        a << li(9, 99);
        a.Bind(target);
        a << mtlr(31) << addi(4, 3, 4) << blr();
        AddProgram("control", "blr that is a jump", a, st);
    }
    for (int viaCall = 0; viaCall < 2; ++viaCall) {
        Asm a;
        Label l0 = a.NewLabel(), table = a.NewLabel(), end = a.NewLabel();
        Label c[4];
        for (auto& l : c) l = a.NewLabel();
        a.Save();
        a.B(l0, true);
        a.Bind(l0);
        a << mflr(6);
        a.AddiDiff(6, 6, table, l0);
        a << rlwinm(7, 3, 2, 28, 29) << ldx(23, 8, 6, 7) << arith(266, 8, 8, 6, 0, 0) << mtctr(8) << (viaCall ? bctrl() : bctr());
        if (viaCall) {
            a.B(end);
        }
        a.Bind(table);
        for (auto& l : c) a.WordDiff(l, table);
        for (int i = 0; i < 4; ++i) {
            a.Bind(c[i]);
            a << li(5, 10 * (i + 1)) << addi(4, 4, i);
            if (viaCall) a << blr();
            else a.B(end);
        }
        a.Bind(end);
        a.Done();
        AddProgram("control", viaCall ? "jump table through bctrl" : "jump table through bctr", a, st);
    }
    {
        Asm a;  // sc, rfi, twi and tw
        Label target = a.NewLabel();
        a.Save();
        a << li(5, 1) << 0x44000002u << li(6, 2);
        a.LiLabel(7, target);
        a << mtspr(26, 7) << mtspr(27, 3) << 0x4C000064u << li(8, 99);  // rfi
        a.Bind(target);
        a << mfmsr(9) << mfspr(10, 27);
        a.Done();
        States few(st.begin(), st.begin() + 3);
        AddProgram("control", "sc and rfi", a, few);
    }
}

// === Calls into registered functions ===============================================================================

States CallStates(Rng& rng, bool write) {
    States states;
    for (unsigned i = 0; i < 14; ++i) {
        CpuContext c = BaseState(rng, (i & 1) ? 4u : 0u);
        for (uint32_t r = 3; r <= 8; ++r) c.gpr[r] = rng.Chance(75) ? rng.Pick(kSingleEdges) : rng.Next();
        if (write) {
            c.gpr[3] = kDataBase + 0x1C00u + 4u * rng.Below(16);
            c.gpr[4] = rng.Next();
        }
        c.gqr[0] = 0;
        c.lr = kReturn;
        states.push_back(c);
    }
    return states;
}

void AddRegisteredCalls() {
    Rng rng(51);
    using namespace ppc;
    const struct { const char* name; uint32_t address; bool write; } fns[] = {
        {"modifies registers", kFnModify, false}, {"modifies floats", kFnFloat, false}, {"writes guest memory", kFnWrite, true},
        {"runs guest code", kFnReenter, false}, {"throws", kFnThrow, false}};
    for (const auto& fn : fns) {
        const States states = CallStates(rng, fn.write);
        // After the call, float registers are read in both forms (f1 as the scalar return, f2 and f3 as they were left).
        const auto prologue = [](Asm& a) { LoadPairs(a); };
        const auto epilogue = [](Asm& a) {
            a << fa(63, 9, 1, 2, 0, 21) << fa(4, 10, 3, 3, 0, 21) << fx(63, 11, 0, 2, 72);
        };
        {
            Asm a;
            a.Save();
            prologue(a);
            a << addi(9, 3, 1);
            a.BAbs(fn.address, true);
            a << addi(10, 3, 2) << mflr(11);
            epilogue(a);
            a.Probe(11);
            a.Done();
            AddProgram("calls", Fmt("bl to a function that %s", fn.name), a, states);
        }
        {
            Asm a;
            a.Save();
            prologue(a);
            a.Li32(12, fn.address);
            a << mtctr(12) << bctrl() << addi(10, 3, 2) << mfctr(11);
            epilogue(a);
            a.Probe(11);
            a.Done();
            AddProgram("calls", Fmt("bctrl to a function that %s", fn.name), a, states);
        }
        {
            Asm a;
            prologue(a);
            a << addi(9, 3, 1);
            a.BAbs(fn.address, false);  // the callee returns to the link address this frame was entered with
            AddProgram("calls", Fmt("tail b to a function that %s", fn.name), a, states);
        }
        {
            Asm a;
            prologue(a);
            a.Li32(12, fn.address);
            a << mtctr(12) << bctr();
            AddProgram("calls", Fmt("tail bctr to a function that %s", fn.name), a, states);
        }
        {
            Asm a;  // a tail call made by local code that was itself called: the callee returns into this frame
            Label local = a.NewLabel();
            a.Save();
            prologue(a);
            a.B(local, true);
            a << addi(10, 3, 3);
            epilogue(a);
            a.Done();
            a.Bind(local);
            a << addi(3, 3, 7);
            a.BAbs(fn.address, false);
            AddProgram("calls", Fmt("tail b from a local callee to a function that %s", fn.name), a, states);
        }
        {
            Asm a;  // conditional calls through ctr and through lr
            a.Save();
            prologue(a);
            a.Li32(12, fn.address);
            a << mtctr(12) << cmpwi(0, 4, 0) << bcctr(12, 2, 1) << addi(10, 3, 2);
            a << mtlr(12) << cmpwi(1, 5, 0) << bclr(4, 6, 1) << addi(9, 3, 4);
            epilogue(a);
            a.Done();
            AddProgram("calls", Fmt("conditional calls to a function that %s", fn.name), a, states);
        }
        {
            Asm a;  // many calls from a loop
            Label loop = a.NewLabel();
            a.Save();
            prologue(a);
            a << li(27, 40);
            a.Bind(loop);
            a.BAbs(fn.address, true);
            a << addic_(27, 27, -1);
            a.Bc(4, 2, loop);
            epilogue(a);
            a.Done();
            States few(states.begin(), states.begin() + 6);
            AddProgram("calls", Fmt("a loop calling a function that %s", fn.name), a, few);
        }
    }
}

// === XER[SO] ========================================================================================================
// Compiled code stores compare results without the SO copy and runs only while XER[SO] is clear: what sets it inside a block
// (mtxer, an OE form, a callee) must hand the rest to the dispatcher, which interprets until it is clear again (mcrxr, mtxer).
// Each program compares and branches after the point that may set it, in the same block, in a loop and after calls.
void AddSummaryOverflow() {
    Rng rng(71);
    using namespace ppc;
    States states;
    for (unsigned i = 0; i < 16; ++i) {
        CpuContext c = BaseState(rng);
        c.xer = (i % 8 == 7 ? 0x80000000u : 0u) | (rng.Next() & 0x6000007Fu);  // SO already set in two
        c.gpr[3] = (i & 1 ? 0x80000000u : 0u) | (i & 2 ? 1u : 0u) | (rng.Next() & 0x7FFFFFFEu & (i & 4 ? ~0u : 0xFFu));
        c.gpr[4] = rng.Chance(50) ? rng.Pick(kIntEdges) : rng.Next();
        c.gpr[5] = rng.Chance(50) ? c.gpr[4] : rng.Next();
        c.gpr[12] = (i & 1) ? 0x7FFFFFFFu : 0x10u;  // the OE add overflows with r13 = 1
        c.gpr[13] = 1;
        c.lr = kReturn;
        states.push_back(c);
    }
    {
        Asm a;
        Label skip = a.NewLabel();
        a << cmpwi(1, 4, 5) << mtspr(1, 3) << cmpwi(0, 4, 7) << cmplw(2, 4, 5) << addi(6, 4, 1) << cmpwi(6, 6, 0);
        a.Bc(12, 26, skip);
        a << li(7, 1) << cmpw(0, 5, 4);
        a.Bind(skip);
        a << mfcr(8);
        a.Probe(8);
        a << mfxer(9);
        a.Probe(9);
        a << blr();
        AddProgram("xer so", "mtxer, then compares and a compare and bc", a, states);
    }
    {
        Asm a;
        Label loop = a.NewLabel(), out = a.NewLabel();
        a << li(10, 4) << mtctr(10);
        a.Bind(loop);
        a << cmpw(3, 4, 5) << arith(266, 14, 12, 13, 1, 0) << cmpwi(0, 14, 0) << addi(12, 14, 0);
        a.Bc(12, 2, out);
        a.Bc(16, 0, loop);
        a.Bind(out);
        a << mfcr(8);
        a.Probe(8);
        a << mfxer(9);
        a.Probe(9);
        a << blr();
        AddProgram("xer so", "addo in a loop of compares", a, states);
    }
    {
        Asm a;
        Label next = a.NewLabel();
        a << mcrxr(7) << cmpwi(1, 4, 0);
        a.B(next);
        a.Bind(next);
        a << cmpw(0, 4, 5) << mfcr(8);
        a.Probe(8);
        a << blr();
        AddProgram("xer so", "mcrxr clears it, then compares", a, states);
    }
    {
        Asm a;
        a.Save();
        a << cmpwi(1, 4, 9);
        a.BAbs(kFnSetSo, true);
        a << cmpwi(0, 4, 3) << cmpw(2, 4, 5) << mfcr(8);
        a.Probe(8);
        a.Done();
        AddProgram("xer so", "bl to a function that sets it, then compares", a, states);
    }
    {
        Asm a;
        Label skip = a.NewLabel();
        a.Save();
        a.Li32(12, kFnSetSo);
        a << mtctr(12) << bctrl() << cmpwi(0, 4, 3);
        a.Bc(4, 2, skip);
        a << cmplw(5, 4, 5);
        a.Bind(skip);
        a << mfcr(8);
        a.Probe(8);
        a.Done();
        AddProgram("xer so", "bctrl to a function that sets it, then a compare and bc", a, states);
    }
    {
        Asm a;
        Label local = a.NewLabel();
        a.Save();
        a.B(local, true);
        a << cmpwi(0, 4, 3) << cmpw(1, 4, 5) << mfcr(8);
        a.Probe(8);
        a.Done();
        a.Bind(local);
        a << cmpwi(6, 5, 1);
        a.BAbs(kFnSetSo, false);
        AddProgram("xer so", "tail b from a local callee to a function that sets it", a, states);
    }
}

// === Stale code ====================================================================================================

// Writes a word into guest memory as it is held (the host's form of the guest word), without a dirty mark.
void RawWrite(uint32_t address, uint32_t word) {
    constexpr uint32_t scratch = kFuncBase + 0x800u;
    Memory::Write32(scratch, word);
    uint8_t bytes[4];
    std::memcpy(bytes, Memory::GetPointer(scratch, 4), 4);
    std::memcpy(Memory::GetPointer(address, 4), bytes, 4);
}

// Valid instructions to patch in (none returns: where lr is the patched address, blr would jump to itself).
const uint32_t kPatchWords[] = {
    ppc::addi(4, 4, 1), ppc::addi(4, 4, 9), ppc::nop(), ppc::li(4, -1), ppc::mulli(4, 4, 3), ppc::logic(444, 4, 4, 3),
    ppc::cmpwi(0, 4, 0), ppc::arith(40, 4, 3, 4, 0, 0), ppc::xori(5, 5, 1), 0x48000008u /* b .+8 */,
    ppc::addic_(4, 4, -1), ppc::rlwinm(4, 4, 3, 0, 28), ppc::mflr(4), ppc::addi(4, 4, 1)};

// A registered function that rewrites guest memory as a save-state load or a rollback restore does in the middle of a call:
// r4 to the address in r3 without a dirty mark, then a new epoch. Code that was linked before must not run after it without
// a check of its words.
void FnRewrite(CpuContext* c) {
    RawWrite(c->gpr[3], c->gpr[4]);
    BrawlDirtyMemory::MarksCleared();
    c->gpr[7] = c->gpr[3] + 8u;
}
// The same as a rollback restore tells it, with no new epoch: the block cache drops what it built since the snapshot (all of
// it, for snapshot 0), which the compiled code's own epoch tests must see.
void FnRestore(CpuContext* c) {
    RawWrite(c->gpr[3], c->gpr[4]);
    BrawlDirtyMemory::MarksRestored(0);
    c->gpr[7] = c->gpr[3] + 12u;
}
// A restore in the order rollback makes it: the notification (MarksRestoring), the write without marks, then MarksRestoreWritten,
// which drops the blocks of this thread's cache whose words the write changed and keeps the others. The caller's block holds the
// rewritten word unless it is the one already there.
void FnRestoreWritten(CpuContext* c) {
    BrawlDirtyMemory::MarksRestoring(0);
    RawWrite(c->gpr[3], c->gpr[4]);
    BrawlDirtyMemory::MarksRestoreWritten();
    c->gpr[7] = c->gpr[3] + 16u;
}

// A word rewritten without marks by a call the running chain of compiled blocks makes (FnRewrite), reached after the call
// through a link slot or an inline cache, which do not test the target's epoch (Block::jitLinked). The call is made three
// ways: a bl to the registered function (the block's own check follows), a blrl (a Generic op: the interpreter's handler makes
// the call, JitFallbackOp), and a tail call from interpreted code of the frame (the continuation goes through the inline cache:
// ExitDynamicAfterCall). The rewritten word is a later block's first, or the word the tail call returns to.
void AddLinkEpochPrograms() {
    using namespace ppc;
    Rng rng(62);
    // A registered function rewrites the instruction after its own call, as a restore does (no mark, no new epoch, the caller's
    // block dropped): the caller's test after the call must fail on the epoch its code holds.
    {
        Asm a;
        Label patched = a.NewLabel();
        a.Save();
        a.LiLabel(3, patched);
        a << mr(4, 8);
        a.BAbs(kFnRestore, true);
        a.Bind(patched);
        a << addi(10, 4, 0) << xori(11, 11, 2);
        a.Done();
        States st;
        for (int round = 0; round < 2; ++round)
            for (uint32_t w : kPatchWords)
                for (int repeat = 0; repeat < 3; ++repeat) {
                    CpuContext c = BaseState(rng);
                    c.gpr[8] = w;
                    st.push_back(c);
                }
        AddProgram("stale", "a call rewrites the next instruction without marks, restore notification", a, st);
    }
    for (int how = 0; how < 3; ++how) {
        Asm a;
        Label later = a.NewLabel(), patched = a.NewLabel(), leaf = a.NewLabel(), end = a.NewLabel();
        a.Save();
        a.LiLabel(3, how == 2 ? patched : later);
        a << mr(4, 8);
        if (how == 0) {
            a.BAbs(kFnRewrite, true);
        } else if (how == 1) {
            a.Li32(12, kFnRewrite);
            a << mtlr(12) << blrl();
        } else {
            a.B(leaf, true);
        }
        a.Bind(patched);
        a << addi(10, 4, 0) << xori(11, 11, 2);
        if (how != 2) {
            a.LiLabel(12, later);
            a << mtctr(12) << bctr();
            a << nop();
            a.Bind(later);
            a << addi(4, 4, 1) << xori(5, 5, 0x33);
        }
        a.B(end);
        a << nop();
        a.Bind(leaf);
        a.BAbs(kFnRewrite, false);
        a.Bind(end);
        a.Done();
        // Each word three runs in a row: the rewritten block is compiled and linked while its word stays, so that the next word
        // meets a slot or cache entry aimed at it.
        States st;
        for (int round = 0; round < 2; ++round)
            for (uint32_t w : kPatchWords)
                for (int repeat = 0; repeat < 3; ++repeat) {
                    CpuContext c = BaseState(rng);
                    c.gpr[8] = w;
                    st.push_back(c);
                }
        static const char* const kHow[] = {"bl", "blrl (Generic)", "tail call back to interpreted code"};
        AddProgram("stale", Fmt("a call rewrites code without marks, new epoch (%s)", kHow[how]), a, st);
    }
}

// The restore-notification program of AddLinkEpochPrograms with the restore written (FnRestoreWritten), in a stage of its own after
// the others (the slots the stages before use stay as they were: the store view lag stage notices its fault in one program only);
// `twice`: a second such call writes the word back.
void AddRestoreWrittenPrograms() {
    Rng rng(63);
    using namespace ppc;
    for (int twice = 0; twice < 2; ++twice) {
        Asm a;
        Label patched = a.NewLabel();
        a.Save();
        a.LiLabel(3, patched);
        a << mr(4, 8);
        a.BAbs(kFnRestoreWritten, true);
        a.Bind(patched);
        a << addi(10, 4, 0) << xori(11, 11, 2);
        if (twice) {
            a.LiLabel(3, patched);
            a.Li32(4, addi(10, 4, 0));
            a.BAbs(kFnRestoreWritten, true);
            a << addi(10, 10, 3);
        }
        a.Done();
        States st;
        for (int round = 0; round < 2; ++round)
            for (uint32_t w : kPatchWords)
                for (int repeat = 0; repeat < 3; ++repeat) {
                    CpuContext c = BaseState(rng);
                    c.gpr[8] = w;
                    st.push_back(c);
                }
        AddProgram("stale", twice ? "a call rewrites the next instruction and puts it back, restore written"
                                  : "a call rewrites the next instruction without marks, restore written",
                   a, st);
    }
}

// The restore-written programs with the restore's compare left out on purpose (g_jitEpochFault 6: every block the restore would
// drop kept, its words changed or not): they must notice. With BRAWL_JIT_RESTORE_PENDING=0 every such block is dropped before the
// restore writes and nothing waits for the compare: there is no fault to notice.
void RestoreWrittenFault() {
    if (!PpcInterpreter::JitAvailable()) return;
    const char* const pendingEnv = std::getenv("BRAWL_JIT_RESTORE_PENDING");
    if (pendingEnv != nullptr && *pendingEnv == '0') return;
    const unsigned failures = g_failures, reported = g_reported;
    PpcInterpreter::detail::g_jitEpochFault = 6;
    g_reported = kMaxReports;
    AddRestoreWrittenPrograms();
    const unsigned noticed = g_failures - failures;
    PpcInterpreter::detail::g_jitEpochFault = 0;
    g_failures = failures;
    g_reported = reported;
    std::printf("  restore written fault: %u failed\n", noticed);
    if (noticed == 0) {
        std::fprintf(stderr, "FAIL: the deliberate restore fault (the blocks a restore changed kept) was not noticed\n");
        ++g_failures;
    }
}

// The same programs with the link epoch made wrong on purpose four ways (g_jitEpochFault): each must make a program fail.
void LinkEpochFaults() {
    if (!PpcInterpreter::JitAvailable()) return;
    const unsigned failures = g_failures, reported = g_reported;
    int notNoticed = 0;
    std::printf("  link epoch faults:");
    for (int fault = 1; fault <= 4; ++fault) {
        PpcInterpreter::detail::g_jitEpochFault = fault;
        const unsigned before = g_failures;
        g_reported = kMaxReports;  // the failures expected here are not printed
        AddLinkEpochPrograms();
        const unsigned noticed = g_failures - before;
        std::printf(" %d: %u failed;", fault, noticed);
        if (noticed == 0) ++notNoticed;
    }
    PpcInterpreter::detail::g_jitEpochFault = 0;
    g_failures = failures;
    g_reported = reported;
    std::printf("\n");
    if (notNoticed != 0) {
        std::fprintf(stderr, "FAIL: %d of the 4 deliberate link epoch faults were not noticed\n", notNoticed);
        ++g_failures;
    }
}

void AddStaleCode() {
    Rng rng(61);
    using namespace ppc;
    const auto branch = [](uint32_t from, uint32_t to) { return 0x48000000u | ((to - from) & 0x03FFFFFCu); };

    // The invalidation paths of the block cache, over code split in 1 to 8 pieces joined by b: edits to each piece, as guest
    // stores (marks) and as writes without marks followed by a new epoch, then the consume and restore notifications.
    for (unsigned pieces : {1u, 2u, 3u, 6u, 8u})
        for (int reverse = 0; reverse < 2; ++reverse) {
            RunCase("stale", Fmt("%u pieces, %s", pieces, reverse ? "reverse" : "forward"), [=](Runner& r, uint32_t base) {
                constexpr uint32_t stride = 0x180;
                uint32_t at[8];
                for (unsigned i = 0; i < pieces; ++i) at[i] = base + (reverse ? pieces - 1 - i : i) * stride;
                Image image;
                image.words.assign(stride / 4 * pieces, 0);
                for (unsigned i = 0; i < pieces; ++i) {
                    const size_t w = (at[i] - base) / 4;
                    image.words[w] = addi(3, 3, 1);
                    image.words[w + 1] = i + 1 < pieces ? branch(at[i] + 4, at[i + 1]) : blr();
                }
                r.Place(image, base);
                Rng local(7 + pieces);
                const auto run = [&] {
                    CpuContext c = BaseState(local);
                    c.gpr[3] = 0;
                    r.Run(at[0], c);
                };
                for (int i = 0; i < 3; ++i) run();
                for (unsigned i = 0; i < pieces; ++i) {
                    r.Write32(at[i], addi(3, 3, 0x12));
                    for (int k = 0; k < 3; ++k) run();
                }
                BrawlDirtyMemory::MarksCleared();
                run();
                BrawlDirtyMemory::MarksConsuming();
                run();
                BrawlDirtyMemory::MarksRestored(0);
                run();
                BrawlDirtyMemory::MarksRestoring(0);
                BrawlDirtyMemory::MarksRestoreWritten();
                run();
                for (unsigned i = 0; i < pieces; ++i) {  // restores that change a piece's first word, and put it back
                    BrawlDirtyMemory::MarksRestoring(0);
                    RawWrite(at[i], addi(3, 3, 0x33));
                    BrawlDirtyMemory::MarksRestoreWritten();
                    run();
                    run();
                    BrawlDirtyMemory::MarksRestoring(0);
                    RawWrite(at[i], addi(3, 3, 1));
                    BrawlDirtyMemory::MarksRestoreWritten();
                    run();
                }
                for (unsigned i = 0; i < pieces; ++i) {  // writes that set no mark: only the new epoch tells
                    RawWrite(at[i], addi(3, 3, 0x21));
                    BrawlDirtyMemory::MarksCleared();
                    run();
                    run();
                    r.Write32(at[i], addi(3, 3, 1));
                    run();
                }
                for (unsigned i = 0; i < pieces; ++i) {  // edit the branch of a piece: it goes to the end, or back
                    if (i + 1 < pieces) {
                        r.Write32(at[i] + 4, blr());
                        run();
                        r.Write32(at[i] + 4, branch(at[i] + 4, at[i + 1]));
                        run();
                    }
                }
                // Enough rebuilds to fill arena chunks, edits cycling through the pieces.
                for (unsigned iteration = 0; iteration < 300; ++iteration) {
                    r.Write32(at[iteration % pieces], addi(3, 3, static_cast<int32_t>(1 + iteration % 100)));
                    run();
                    if (iteration % 50 == 49) {
                        BrawlDirtyMemory::MarksCleared();
                        run();
                    }
                }
            });
        }

    // A program that patches an instruction of a later block of itself, and reaches it five different ways. The words it
    // writes cycle through valid instructions, including the original (restoring it) and ones that branch.
    for (int how = 0; how < 5; ++how) {
        Asm a;
        Label patch = a.NewLabel(), end = a.NewLabel();
        a.Save();
        a.LiLabel(7, patch);
        a << stw(6, 0, 7) << addi(9, 9, 1);
        switch (how) {
        case 0: a << mtctr(7) << bctr(); break;                     // a jump through ctr
        case 1: a.Bc(20, 0, patch); break;                           // an always-taken bc
        case 2: a << mtlr(7) << blr(); break;                        // a jump through lr
        case 3: a << mtctr(7) << bctrl() << addi(10, 4, 0); a.B(end); break;  // a call through ctr
        default: a.B(patch, true); a << addi(10, 4, 0); a.B(end); break;      // a direct call
        }
        a << nop();
        a.Bind(patch);
        a << addi(4, 4, 1) << xori(5, 5, 0x33);
        if (how >= 3) a << blr();
        else a.Done();
        a.Bind(end);
        a.Done();
        States st;
        // A patched-in blr returns the frame where lr is still its own (ways 0 and 1), or the call (3 and 4); in way 2 lr
        // is the patched address itself, so it is left out.
        std::vector<uint32_t> words(std::begin(kPatchWords), std::end(kPatchWords));
        if (how != 2) words.push_back(ppc::blr());
        for (int round = 0; round < 3; ++round)
            for (uint32_t w : words) {
                CpuContext c = BaseState(rng);
                c.gpr[6] = w;
                st.push_back(c);
            }
        // The patch lands in the program's own image: the watched code range sees it.
        AddProgram("stale", Fmt("program patches a later block (way %d)", how), a, st);
    }

    // A registered function patches code: the instruction after its own call (the calling block must go on with the new
    // word), and a later block.
    for (int target = 0; target < 3; ++target) {
        Asm a;
        Label patched = a.NewLabel(), later = a.NewLabel(), end = a.NewLabel();
        a.Save();
        if (target == 0) a.LiLabel(3, patched);
        else a.LiLabel(3, later);
        a << mr(4, 8);  // the new word, from the state
        a.BAbs(kFnWrite, true);
        a.Bind(patched);
        a << addi(10, 4, 0) << xori(11, 11, 2);
        if (target != 0) {
            a.LiLabel(12, later);
            a << mtctr(12) << bctr();
            a << nop();
            a.Bind(later);
            a << addi(4, 4, 1) << xori(5, 5, 0x33);
            if (target == 2) {
                a.B(end);
                a << nop();
            }
        }
        a.Bind(end);
        a.Done();
        States st;
        for (int round = 0; round < 3; ++round)
            for (uint32_t w : kPatchWords) {
                CpuContext c = BaseState(rng);
                c.gpr[8] = w;
                st.push_back(c);
            }
        AddProgram("stale", Fmt("a registered function patches code (case %d)", target), a, st);
    }

    // Interpreted guest code (not a registered function) stores into the calling block's later operation, then returns.
    {
        Asm a;
        Label patched = a.NewLabel(), helper = a.NewLabel();
        a.Save();
        a.LiLabel(9, patched);
        a.B(helper, true);
        a.Bind(patched);
        a << addi(4, 4, 1) << xori(5, 5, 0x33);
        a.Done();
        a.Bind(helper);
        a << stw(8, 0, 9) << blr();
        States st;
        for (int round = 0; round < 3; ++round)
            for (uint32_t w : kPatchWords) {
                CpuContext c = BaseState(rng);
                c.gpr[8] = w;
                st.push_back(c);
            }
        AddProgram("stale", "guest callee patches the caller's next operation", a, st);
    }
    AddLinkEpochPrograms();
}

// === Random programs ===============================================================================================
//
// Programs from a seeded generator. They stay inside the sandbox and always terminate:
//   r1            never written (the float helpers refuse a zero stack pointer)
//   r23           the caller's lr, restored before the final blr
//   r24           base of the memory operands (the middle of the sandbox's input), never written
//   r25           a walking pointer for the update forms (each is followed by an instruction that puts it back)
//   r26           a scratch: an index for the indexed forms, an address for calls and jumps, set just before use
//   r27           the counter of the loops that do not use ctr
// Memory operands are small displacements or indexes from r24 or r25. Branches go to labels inside the program, forward,
// or backward only in loops whose counter is r27 or ctr. Calls go to leaf functions emitted after the final blr.
struct GenCtx {
    bool ctrLocked = false;  // a ctr loop is running (or this is a leaf called from one): nothing may change ctr
    bool lrLocked = false;   // inside a leaf: nothing may change lr
    bool simple = false;     // no control flow (leaf bodies)
    bool inLoop = false;     // loops do not nest
    int depth = 0;
};

class Gen {
public:
    explicit Gen(Rng& rng) : r_(rng) {}

    Image Build(unsigned items) {
        using namespace ppc;
        a_ << mflr(23);
        a_.Li32(24, kDataBase + 0x800u);
        a_.Li32(25, kDataBase + 0x1000u);
        GenCtx g;
        Body(items, g);
        a_ << mtlr(23) << blr();
        for (Label leaf : leaves_) {
            a_.Bind(leaf);
            GenCtx l;
            l.simple = l.lrLocked = l.ctrLocked = true;
            Body(1 + r_.Below(5), l);
            a_ << blr();
        }
        return a_.Finish();
    }

private:
    uint32_t Gpr() {
        static const uint32_t pool[] = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 28, 29, 30, 31};
        return r_.Pick(pool);
    }
    uint32_t Fpr() { return r_.Below(12); }
    int32_t Imm() {
        static const int32_t edges[] = {0, 1, -1, 0x7FFF, -0x8000, 2, 31, 32, 0x1234, -2, 0x7FF, 4};
        return r_.Chance(60) ? r_.Pick(edges) : static_cast<int16_t>(r_.Next());
    }
    // A small offset from r24 for the indexed forms.
    int32_t Offset() { return static_cast<int32_t>(r_.Below(0x600)) - 0x300; }

    void Body(unsigned n, const GenCtx& g) {
        for (unsigned i = 0; i < n && a_.Here() < 1200; ++i) {
            if (g.simple || g.depth >= 2 || r_.Below(100) < 82) SimpleOp(g);
            else Control(g);
        }
    }

    void SimpleOp(const GenCtx& g) {
        const uint32_t k = r_.Below(100);
        if (k < 42) IntOp(g);
        else if (k < 68) MemOp();
        else if (k < 84) FloatOp();
        else PairedOp();
    }

    void IntOp(const GenCtx& g) {
        using namespace ppc;
        const uint32_t d = Gpr(), s = Gpr(), t = Gpr(), rc = r_.Below(2);
        switch (r_.Below(18)) {
        case 0: case 1: case 2: {
            static const uint32_t xos[] = {266, 10, 138, 234, 202, 40, 8, 136, 232, 200, 104, 235, 75, 11, 491, 459};
            const uint32_t xo = r_.Pick(xos);
            const bool unary = xo == 234 || xo == 202 || xo == 232 || xo == 200 || xo == 104;
            a_ << arith(xo, d, s, unary ? 0 : t, r_.Below(2), rc);
            break;
        }
        case 3: case 4: {
            static const uint32_t xos[] = {28, 60, 444, 412, 316, 476, 124, 284, 24, 536, 792};
            a_ << logic(r_.Pick(xos), d, s, t, rc);
            break;
        }
        case 5:
            switch (r_.Below(4)) {
            case 0: a_ << cntlzw(d, s, rc); break;
            case 1: a_ << extsb(d, s, rc); break;
            case 2: a_ << extsh(d, s, rc); break;
            default: a_ << srawi(d, s, r_.Below(32), rc); break;
            }
            break;
        case 6: case 7: {
            const int32_t imm = Imm();
            const uint32_t u = static_cast<uint32_t>(imm) & 0xFFFFu;
            switch (r_.Below(14)) {
            case 0: a_ << addi(d, s, imm); break;
            case 1: a_ << addis(d, s, imm); break;
            case 2: a_ << addic(d, s, imm); break;
            case 3: a_ << addic_(d, s, imm); break;
            case 4: a_ << subfic(d, s, imm); break;
            case 5: a_ << mulli(d, s, imm); break;
            case 6: a_ << ori(d, s, u); break;
            case 7: a_ << oris(d, s, u); break;
            case 8: a_ << xori(d, s, u); break;
            case 9: a_ << xoris(d, s, u); break;
            case 10: a_ << andi_(d, s, u); break;
            case 11: a_ << andis_(d, s, u); break;
            case 12: a_ << li(d, imm); break;
            default: a_ << lis(d, imm); break;
            }
            break;
        }
        case 8: case 9:
            switch (r_.Below(3)) {
            case 0: a_ << rlwinm(d, s, r_.Below(32), r_.Below(32), r_.Below(32), rc); break;
            case 1: a_ << rlwimi(d, s, r_.Below(32), r_.Below(32), r_.Below(32), rc); break;
            default: a_ << rlwnm(d, s, t, r_.Below(32), r_.Below(32), rc); break;
            }
            break;
        case 10: case 11: {
            const uint32_t crf = r_.Below(8);
            switch (r_.Below(4)) {
            case 0: a_ << cmpw(crf, s, t); break;
            case 1: a_ << cmplw(crf, s, t); break;
            case 2: a_ << cmpwi(crf, s, Imm()); break;
            default: a_ << cmplwi(crf, s, static_cast<uint32_t>(Imm()) & 0xFFFFu); break;
            }
            break;
        }
        case 12: {
            static const uint32_t xos[] = {257, 129, 289, 225, 33, 449, 417, 193};
            a_ << crop(r_.Pick(xos), r_.Below(32), r_.Below(32), r_.Below(32));
            break;
        }
        case 13:
            switch (r_.Below(5)) {
            case 0: a_ << mfcr(d); break;
            case 1: a_ << mtcrf(r_.Below(256), s); break;
            case 2: a_ << mcrf(r_.Below(8), r_.Below(8)); break;
            case 3: a_ << mcrxr(r_.Below(8)); break;
            default: a_ << mr(d, s); break;
            }
            break;
        case 14:
            switch (r_.Below(9)) {
            case 0: a_ << mflr(d); break;
            case 1: if (!g.lrLocked) a_ << mtlr(s); break;
            case 2: a_ << mfctr(d); break;
            case 3: if (!g.ctrLocked) a_ << mtctr(s); break;
            case 4: a_ << mfxer(d); break;
            case 5: a_ << mtxer(s); break;
            case 6: a_ << mfspr(d, 912 + r_.Below(8)); break;
            case 7:
                a_.Li32(26, RandGqr(r_));  // a valid one: a reserved quantization type aborts
                a_ << mtspr(912 + r_.Below(8), 26);
                break;
            default: a_ << mfspr(d, r_.Pick(kOtherSprs)); break;
            }
            break;
        case 15:
            switch (r_.Below(6)) {
            case 0: a_ << mfmsr(d); break;
            case 1: a_ << mtmsr(s); break;
            case 2: a_ << mtspr(r_.Pick(kOtherSprs), s); break;
            case 3: a_ << X(0, 0, 0, 598); break;  // sync
            case 4: a_ << X(0, s, t, 86); break;   // dcbf
            default: a_ << nop(); break;
            }
            break;
        default: a_ << addi(d, s, Imm()) << arith(266, t, d, s, 0, rc); break;
        }
    }
    static constexpr uint32_t kOtherSprs[] = {26, 27, 920, 1008, 1009};

    void MemOp() {
        using namespace ppc;
        const int32_t d = Offset();
        const uint32_t rt = Gpr(), fr = Fpr(), w = r_.Below(2), i = r_.Below(8);
        const auto index = [&](int32_t off) { a_ << li(26, off); };
        switch (r_.Below(15)) {
        case 0: case 1: {
            static const uint32_t ops[] = {32, 34, 40, 42};
            a_ << ld(r_.Pick(ops), rt, d, 24);
            break;
        }
        case 2: {
            static const uint32_t ops[] = {36, 38, 44};
            a_ << ld(r_.Pick(ops), Gpr(), d, 24);
            break;
        }
        case 3: {
            static const uint32_t ops[] = {48, 50, 52, 54};
            a_ << ld(r_.Pick(ops), fr, d, 24);
            break;
        }
        case 4: {  // update forms put the walking pointer back
            static const uint32_t ops[] = {33, 35, 41, 43, 37, 39, 45, 49, 51, 53, 55};
            const uint32_t op = r_.Pick(ops);
            a_ << ld(op, op >= 48 ? fr : rt, d, 25) << addi(25, 25, -d);
            break;
        }
        case 5: case 6: {
            static const uint32_t ints[] = {23, 87, 279, 343, 534, 790, 20, 151, 215, 407, 662, 918};
            static const uint32_t floats[] = {535, 599, 663, 727, 983};
            index(Offset());
            if (r_.Chance(70)) a_ << ldx(r_.Pick(ints), rt, 24, 26);
            else a_ << ldx(r_.Pick(floats), fr, 24, 26);
            break;
        }
        case 7:  // rA = 0: the address is rB alone
            a_.Li32(26, kDataBase + 0x800u + static_cast<uint32_t>(Offset()));
            a_ << ldx(r_.Chance(50) ? 23u : 151u, rt, 0, 26);
            break;
        case 8: {  // update indexed
            static const uint32_t ints[] = {55, 119, 311, 375, 183, 247, 439};
            static const uint32_t floats[] = {567, 631, 695, 759};
            index(Offset());
            if (r_.Chance(70)) a_ << ldx(r_.Pick(ints), rt, 25, 26);
            else a_ << ldx(r_.Pick(floats), fr, 25, 26);
            a_ << arith(40, 25, 26, 25, 0, 0);
            break;
        }
        case 9:
            if (r_.Chance(50)) a_ << ld(47, 3 + r_.Below(29), d, 24);   // stmw
            else a_ << ld(46, 28 + r_.Below(4), d, 24);                 // lmw: only registers the program may lose
            break;
        case 10:
            switch (r_.Below(4)) {
            case 0: a_ << lswi(3 + r_.Below(10), 24, r_.Below(32)); break;
            case 1: a_ << stswi(3 + r_.Below(16), 24, r_.Below(32)); break;
            case 2:
                a_ << li(26, static_cast<int32_t>(r_.Below(32))) << mtxer(26) << li(26, Offset() & 0xFF) << lswx(3 + r_.Below(10), 24, 26);
                break;
            default:
                a_ << li(26, static_cast<int32_t>(r_.Below(32))) << mtxer(26) << li(26, Offset() & 0xFF) << stswx(3 + r_.Below(16), 24, 26);
                break;
            }
            break;
        case 11:
            index(Offset());
            a_ << (r_.Chance(70) ? dcbz(24, 26) : dcbz_l(24, 26));
            break;
        case 12: {
            const int32_t d12 = d / 2;
            a_ << psq(r_.Chance(50) ? 56 : 60, fr, d12, 24, w, i);
            break;
        }
        case 13: {
            const int32_t d12 = d / 2;
            a_ << psq(r_.Chance(50) ? 57 : 61, fr, d12, 25, w, i) << addi(25, 25, -d12);
            break;
        }
        default:
            index(Offset());
            switch (r_.Below(4)) {
            case 0: a_ << psqx(6, fr, 24, 26, w, i); break;
            case 1: a_ << psqx(7, fr, 24, 26, w, i); break;
            case 2: a_ << psqx(38, fr, 25, 26, w, i) << arith(40, 25, 26, 25, 0, 0); break;
            default: a_ << psqx(39, fr, 25, 26, w, i) << arith(40, 25, 26, 25, 0, 0); break;
            }
            break;
        }
    }

    void FloatOp() {
        using namespace ppc;
        const uint32_t rc = r_.Below(5) == 0;
        switch (r_.Below(8)) {
        case 0: case 1: case 2: case 3: case 4:
            a_ << EncodeFp(r_.Pick(kScalarOps), Fpr(), Fpr(), Fpr(), Fpr(), rc);
            break;
        case 5: a_ << (r_.Chance(50) ? fcmpu(r_.Below(8), Fpr(), Fpr()) : fcmpo(r_.Below(8), Fpr(), Fpr())); break;
        default:
            switch (r_.Below(6)) {
            case 0: a_ << mffs(Fpr(), rc); break;
            case 1: a_ << mtfsf(r_.Below(256), Fpr(), rc); break;
            case 2: a_ << mtfsfi(r_.Below(8), r_.Below(16), rc); break;
            case 3: a_ << mtfsb0(r_.Below(32), rc); break;
            case 4: a_ << mtfsb1(r_.Below(32), rc); break;
            default: a_ << mcrfs(r_.Below(8), r_.Below(8)); break;
            }
            break;
        }
    }

    void PairedOp() { a_ << EncodePs(r_.Pick(kPsOps), Fpr(), Fpr(), Fpr(), Fpr(), r_.Below(5) == 0); }

    Label LeafLabel() {
        if (leaves_.size() < 4 && (leaves_.empty() || r_.Chance(40))) leaves_.push_back(a_.NewLabel());
        return leaves_[r_.Below(static_cast<uint32_t>(leaves_.size()))];
    }

    void Control(const GenCtx& g) {
        GenCtx c = g;
        ++c.depth;
        switch (r_.Below(10)) {
        case 0: case 1: case 2: case 3: If(g, c); break;
        case 4: case 5: if (!g.inLoop) Loop(g, c); else If(g, c); break;
        case 6: case 7: Call(g); break;
        case 8: if (!g.ctrLocked) JumpTable(c); else If(g, c); break;
        default:
            if (!g.lrLocked && r_.Chance(50)) {
                Label l = a_.NewLabel();
                a_.LiLabel(26, l);
                a_ << ppc::mtlr(26) << ppc::blr() << ppc::nop();  // a jump through lr, which is not this frame's return
                a_.Bind(l);
            } else {
                Label l = a_.NewLabel();  // a forward b that skips code, two of them often a chain
                a_.B(l);
                Body(1 + r_.Below(2), c);
                a_.Bind(l);
            }
            break;
        }
    }

    void If(const GenCtx& g, const GenCtx& c) {
        using namespace ppc;
        Label skip = a_.NewLabel();
        const uint32_t crf = r_.Below(8);
        uint32_t field = crf;
        switch (r_.Below(6)) {
        case 0: a_ << cmpwi(crf, Gpr(), Imm()); break;
        case 1: a_ << cmplwi(crf, Gpr(), static_cast<uint32_t>(Imm()) & 0xFFFFu); break;
        case 2: a_ << cmpw(crf, Gpr(), Gpr()); break;
        case 3: a_ << cmplw(crf, Gpr(), Gpr()); break;
        case 4: a_ << rlwinm(Gpr(), Gpr(), r_.Below(32), r_.Below(32), r_.Below(32), 1); field = 0; break;
        default: a_ << fcmpu(crf, Fpr(), Fpr()); break;
        }
        if (r_.Chance(25)) SimpleOp(c);  // something between the compare and the branch
        static const uint32_t counted[] = {16, 18, 8, 0, 10, 2};
        const uint32_t bo = (!g.ctrLocked && r_.Chance(25)) ? r_.Pick(counted) : (r_.Chance(50) ? 12u : 4u);
        a_.Bc(bo, field * 4 + r_.Below(4), skip);
        Body(1 + r_.Below(4), c);
        if (r_.Chance(30)) {
            Label end = a_.NewLabel();
            a_.B(end);
            a_.Bind(skip);
            Body(1 + r_.Below(3), c);
            a_.Bind(end);
        } else {
            a_.Bind(skip);
        }
    }

    void Loop(const GenCtx& g, const GenCtx& c) {
        using namespace ppc;
        (void)g;
        GenCtx in = c;
        in.inLoop = true;
        const int32_t n = r_.Chance(90) ? 1 + static_cast<int32_t>(r_.Below(6)) : 1 + static_cast<int32_t>(r_.Below(40));
        Label loop = a_.NewLabel();
        switch (r_.Below(3)) {
        case 0:  // ctr
            if (!g.ctrLocked) {
                in.ctrLocked = true;
                a_ << li(27, n) << mtctr(27);
                a_.Bind(loop);
                Body(1 + r_.Below(5), in);
                a_.Bc(16, 0, loop);
                break;
            }
            [[fallthrough]];
        case 1:  // a counter register, tested by addic. and bne
            a_ << li(27, n);
            a_.Bind(loop);
            Body(1 + r_.Below(5), in);
            a_ << addic_(27, 27, -1);
            a_.Bc(4, 2, loop);
            break;
        default: {  // tested at the top, left by a conditional branch, closed by a backward b
            Label exit = a_.NewLabel();
            a_ << li(27, n);
            a_.Bind(loop);
            a_ << addic_(27, 27, -1);
            a_.Bc(12, 2, exit);
            Body(1 + r_.Below(5), in);
            a_.B(loop);
            a_.Bind(exit);
            break;
        }
        }
    }

    void Call(const GenCtx& g) {
        using namespace ppc;
        Label leaf = LeafLabel();
        switch (r_.Below(3)) {
        case 0: a_.B(leaf, true); break;
        case 1:
            if (!g.ctrLocked) {
                a_.LiLabel(26, leaf);
                a_ << mtctr(26) << bctrl();
            } else {
                a_.B(leaf, true);
            }
            break;
        default:
            a_.LiLabel(26, leaf);
            a_ << mtlr(26) << blrl();
            break;
        }
    }

    void JumpTable(const GenCtx& c) {
        using namespace ppc;
        Label table = a_.NewLabel(), end = a_.NewLabel();
        Label cases[4];
        for (auto& l : cases) l = a_.NewLabel();
        const uint32_t rt = Gpr(), ri = Gpr();
        a_.LiLabel(26, table);
        a_ << rlwinm(rt, ri, 2, 28, 29) << ldx(23, rt, 26, rt) << arith(266, rt, rt, 26, 0, 0) << mtctr(rt) << bctr();
        a_.Bind(table);
        for (Label l : cases) a_.WordDiff(l, table);
        for (int i = 0; i < 4; ++i) {
            a_.Bind(cases[i]);
            Body(1 + r_.Below(2), c);
            if (i < 3) a_.B(end);
        }
        a_.Bind(end);
    }

    Rng& r_;
    Asm a_;
    std::vector<Label> leaves_;
};

void AddRandomPrograms(unsigned count, uint64_t firstSeed) {
    for (unsigned i = 0; i < count; ++i) {
        const uint64_t seed = firstSeed + i;
        Rng rng(seed);
        Gen gen(rng);
        const Image image = gen.Build(8 + rng.Below(80));
        States states;
        for (int k = 0; k < 4; ++k) states.push_back(BaseState(rng, rng.Chance(25) ? 4u : 0u));
        RunCase("random", Fmt("seed %llu", static_cast<unsigned long long>(seed)), [image, states, seed](Runner& r, uint32_t base) {
            r.Place(image, base);
            r.SetSandbox(1 + static_cast<uint32_t>(seed % 5));
            for (const CpuContext& s : states) r.Run(base, s);
        });
    }
}

}  // namespace


namespace {

// --- Known results: the assembler, the harness and the registered functions do what the programs assume -------------
//
// The differential comparison could pass for the wrong reason (programs that all return at once, a harness that sees
// nothing), so a few programs are checked against values worked out by hand, under every engine.
bool SanityChecks() {
    using namespace ppc;
    bool ok = true;
    const auto expect = [&](const EngineConfig& config, const char* what, bool condition) {
        if (!condition) {
            std::fprintf(stderr, "sanity check failed under %s: %s\n", config.name, what);
            ok = false;
        }
    };
    Rng rng(99);
    std::vector<EngineConfig> engines = {kReference};
    for (const EngineConfig& c : kConfigs) engines.push_back(c);
    for (const EngineConfig& config : engines) {
        Runner r(config);
        r.SetSandbox(1);
        const auto run = [&](Asm& a, const std::function<void(CpuContext&)>& setup) -> Snap {
            const Image image = a.Finish();
            const uint32_t base = NextBase();
            r.Place(image, base);
            CpuContext c = BaseState(rng);
            setup(c);
            r.Run(base, c);
            return r.trace().back();
        };
        {
            Asm a;
            a << li(3, 5) << addi(3, 3, 7) << blr();
            const Snap s = run(a, [](CpuContext&) {});
            expect(config, "li, addi: r3 = 12", s.ctx.gpr[3] == 12 && !s.threw);
        }
        {
            Asm a;
            Label loop = a.NewLabel();
            a << mtctr(3);
            a.Bind(loop);
            a << addi(4, 4, 3);
            a.Bc(16, 0, loop);
            a << blr();
            for (int round = 0; round < 3; ++round) {
                const Snap s = run(a, [](CpuContext& c) { c.gpr[3] = 10; c.gpr[4] = 0; });
                expect(config, "bdnz loop of 10: r4 = 30, ctr = 0", s.ctx.gpr[4] == 30 && s.ctx.ctr == 0);
            }
        }
        {
            Asm a;
            a << arith(10, 5, 3, 4, 0, 0) << blr();  // addc
            const Snap s = run(a, [](CpuContext& c) { c.gpr[3] = 0xFFFFFFFFu; c.gpr[4] = 1; c.xer = 0; });
            expect(config, "addc 0xFFFFFFFF + 1: r5 = 0, carry set", s.ctx.gpr[5] == 0 && (s.ctx.xer & kXerCa));
        }
        {
            Asm a;
            Label l0 = a.NewLabel(), table = a.NewLabel(), end = a.NewLabel();
            Label c[4];
            for (auto& l : c) l = a.NewLabel();
            a.Save();
            a.B(l0, true);
            a.Bind(l0);
            a << mflr(6);
            a.AddiDiff(6, 6, table, l0);
            a << rlwinm(7, 3, 2, 28, 29) << ldx(23, 8, 6, 7) << arith(266, 8, 8, 6, 0, 0) << mtctr(8) << bctr();
            a.Bind(table);
            for (auto& l : c) a.WordDiff(l, table);
            for (int i = 0; i < 4; ++i) {
                a.Bind(c[i]);
                a << li(5, 10 * (i + 1));
                a.B(end);
            }
            a.Bind(end);
            a.Done();
            for (uint32_t i = 0; i < 4; ++i) {
                const Snap s = run(a, [i](CpuContext& cx) { cx.gpr[3] = i; });
                expect(config, "jump table: r5 = 10 * (index + 1)", s.ctx.gpr[5] == 10 * (i + 1) && s.ctx.lr == kReturn);
            }
        }
        {
            Asm a;
            a.Save();
            a.BAbs(kFnModify, true);
            a.Done();
            const Snap s = run(a, [](CpuContext& c) { c.gpr[3] = 7; c.gpr[4] = 5; });
            expect(config, "bl to FnModify: r3 = 7 * 3 + 5, r12 = the return address",
                   s.ctx.gpr[3] == 26 && (s.ctx.gpr[12] & 0x1FFFu) == 0x0008u && s.ctx.lr == kReturn);
        }
        {
            Asm a;
            a.Save();
            a.BAbs(kFnThrow, true);
            a.Done();
            const Snap thrown = run(a, [](CpuContext& c) { c.gpr[3] = 1; c.gpr[6] = 1; });
            expect(config, "FnThrow with r6 odd throws", thrown.threw && thrown.what == "thrown from a registered function");
            const Snap fine = run(a, [](CpuContext& c) { c.gpr[3] = 1; c.gpr[6] = 2; });
            expect(config, "FnThrow with r6 even does not, after a throw", !fine.threw && fine.ctx.gpr[7] == 0x77);
        }
        {
            Asm a;
            a.Save();
            a.BAbs(kFnReenter, true);
            a.Done();
            const Snap s = run(a, [](CpuContext& c) { c.gpr[3] = 1; c.gpr[4] = 0; c.gpr[5] = 0; });
            expect(config, "FnReenter runs guest code: r3 = 1 + 100 + 1, r4 = 1", s.ctx.gpr[3] == 102 && s.ctx.gpr[4] == 1 && s.ctx.lr == kReturn);
            const Snap t = run(a, [](CpuContext& c) { c.gpr[3] = 1; c.gpr[5] = 1; c.gpr[6] = 1; });
            expect(config, "an exception through two runs", t.threw);
        }
        {
            Asm a;
            a << fa(63, 8, 1, 2, 0, 21) << fa(4, 9, 1, 2, 0, 21) << blr();
            const Snap s = run(a, [](CpuContext& c) { c.fpr[1].d = 1.0; c.fpr[2].d = 2.0; });
            expect(config, "fadd 1 + 2 = 3", s.ctx.fpr[8].d == 3.0);
            expect(config, "ps_add of two splatted singles: ps0 = ps1 = 3", s.ctx.fpr[9].paired.ps0 == 3.0f && s.ctx.fpr[9].paired.ps1 == 3.0f);
        }
        {
            Asm a;
            a << addi(3, 3, 1) << blr();
            const Image image = a.Finish();
            const uint32_t base = NextBase();
            r.Place(image, base);
            CpuContext c = BaseState(rng);
            c.gpr[3] = 0;
            r.Run(base, c);
            r.Run(base, c);
            r.Write32(base, addi(3, 3, 9));
            r.Run(base, c);
            expect(config, "a guest store over code is run: r3 = 9", r.trace().back().ctx.gpr[3] == 9);
        }
    }
    return ok;
}

// A writes into code that sets no dirty mark and starts no new epoch cannot be noticed by a cache that trusts its marks:
// the cached engine is expected to run the old code, the classic one the new. Printed for information (it shows the
// comparison tells the engines apart); a cache that rechecks every entry would pass this too, so it is not an error.
void NegativeControl() {
    using namespace ppc;
    Asm a;
    a << addi(3, 3, 1) << blr();
    const Image image = a.Finish();
    const uint32_t base = NextBase();
    Rng rng(98);
    uint32_t result[2] = {0, 0};
    for (int i = 0; i < 2; ++i) {
        Runner r(i == 0 ? kReference : kConfigs[0]);
        r.SetSandbox(1);
        r.Place(image, base);
        CpuContext c = BaseState(rng);
        c.gpr[3] = 0;
        r.Run(base, c);
        r.Run(base, c);
        RawWrite(base, addi(3, 3, 5));
        r.Run(base, c);
        result[i] = r.trace().back().ctx.gpr[3];
    }
    PpcInterpreter::ClearEngineOverride();
    BrawlDirtyMemory::MarksCleared();
    Memory::Write32(base, addi(3, 3, 1));
    std::printf("  negative control: an unmarked write over code, classic r3 = %u, cached r3 = %u (%s)\n", result[0], result[1],
                result[0] != result[1] ? "the comparison tells the engines apart" : "the cache rechecks every entry");
}


// A guest store over the next instruction of the block that is running. Hardware without an instruction cache flush
// is free to run either word, so the engines may differ here (the cached interpreter checks a block when it enters it);
// no program of this test depends on it, and the result is printed for information only.
void SameBlockPatchProbe() {
    using namespace ppc;
    Asm a;
    Label patch = a.NewLabel();
    a.LiLabel(7, patch);
    a << stw(6, 0, 7);
    a.Bind(patch);
    a << addi(4, 4, 1) << blr();
    const Image image = a.Finish();
    std::printf("  probe: a store over the next instruction of the running block, r4 after each of three runs:");
    for (const EngineConfig& config : {kReference, kConfigs[0], kConfigs[2]}) {
        Runner r(config);
        r.SetSandbox(1);
        const uint32_t base = NextBase();
        r.Place(image, base);
        Rng rng(97);
        std::printf(" %s", config.name);
        for (int i = 0; i < 3; ++i) {
            CpuContext c = BaseState(rng);
            c.gpr[4] = 0;
            c.gpr[6] = addi(4, 4, i == 1 ? 9 : 1);
            r.Run(base, c);
            std::printf(" %u", r.trace().back().ctx.gpr[4]);
        }
        std::printf(";");
    }
    PpcInterpreter::ClearEngineOverride();
    std::printf("\n");
}

}  // namespace

// === main ==========================================================================================================

namespace {
using Clock = std::chrono::steady_clock;
double Since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// === Guarded code ==================================================================================================
//
// The program slots and the support code registered as executable pages (their 64 KiB chunks flagged in the store policy), as
// the game's own code is: the JIT then compiles their blocks as guarded code (BlockEmitter::Guarded: no dirty-mark test on a
// linked entry, the epoch alone after a call), which relies on every write under them dropping them at once
// (BrawlDirtyMemory::guardedWritten). Run after everything else: the pages stay registered (executable ranges are only ever
// added), as the guarded blocks compiled while they are require.
void SetGuardedCode() {
    using namespace RecompMod;
    g_executableWriteGuardEnabled.store(true);
    const auto add = [](uint32_t start, uint32_t end) {
        for (uint32_t a = start; a < end; a += 1u << kExecutableWriteGuardPageShift) {
            g_executableWriteGuardPages[a >> kExecutableWriteGuardPageShift].store(1);
            g_executableWriteGuardMidPages[a >> kExecutableWriteGuardMidPageShift].store(1);
            g_executableWriteGuardCoarsePages[a >> kExecutableWriteGuardCoarsePageShift].store(1);
        }
    };
    add(kSlotBase, kSlotBase + kSlotCount * kSlotBytes);
    add(kSupportPlain & ~0xFFFu, (kSupportThrow & ~0xFFFu) + 0x1000u);
    Memory::RefreshWritableFastPathsForExecutableRanges();
}

uint64_t GuardedCompiles() { return PpcInterpreter::detail::JitX64GetStats().guarded; }
// BRAWL_JIT_GUARD=0: no block is compiled as guarded code (every line tested); the stages that expect guarded blocks expect none.
bool GuardOff() {
    const char* v = std::getenv("BRAWL_JIT_GUARD");
    return v != nullptr && *v == '0';
}

void GuardedStages(unsigned randomCount, uint64_t firstSeed);

// The guarded programs with the guard made wrong on purpose (g_jitEpochFault 5: a write under guarded code drops nothing): the
// stale-code programs must notice.
void GuardFault() {
    if (!PpcInterpreter::JitAvailable() || GuardOff()) return;
    const unsigned failures = g_failures, reported = g_reported;
    PpcInterpreter::detail::g_jitEpochFault = 5;
    g_reported = kMaxReports;
    AddStaleCode();
    const unsigned noticed = g_failures - failures;
    PpcInterpreter::detail::g_jitEpochFault = 0;
    g_failures = failures;
    g_reported = reported;
    std::printf("  guard fault: %u failed\n", noticed);
    if (noticed == 0) {
        std::fprintf(stderr, "FAIL: the deliberate guard fault (writes under guarded code drop nothing) was not noticed\n");
        ++g_failures;
    }
}

// Guarded code under the store view (lean_window.h): a compiled store has no policy test of its own, and a store into a guarded
// line reaches the checked path, whose mark drops the guarded blocks there, only because the store view's copy of a policy-flagged
// chunk is read-only (the store faults: jit_x64_mem.cpp). The stale-code programs store into guarded lines of other blocks and of
// their callers: they must take that fault path, and with the copy of the guarded chunks made writable on purpose (the protection
// lagging behind the store policy) they must notice.
uint64_t StoreFaults() { return PpcInterpreter::detail::JitX64GetFaultStats().faults; }
void GuardedStaleCode() {
    const uint64_t faults = StoreFaults();
    AddStaleCode();
    const uint64_t took = StoreFaults() - faults;
    std::printf("  guarded stores that faulted: %llu (store view %s)\n", static_cast<unsigned long long>(took),
                LeanWindow::g_storeView ? "on" : "off");
    if (PpcInterpreter::JitAvailable() && LeanWindow::g_storeView && took == 0) {
        std::fprintf(stderr, "FAIL: no store into guarded code took the store view's fault path\n");
        ++g_failures;
    }
}
void StoreViewLag() {
    if (!PpcInterpreter::JitAvailable() || !LeanWindow::g_storeView || GuardOff()) return;
    // the cached copy of the store view (one GiB up from the aliased view's +2 GiB: lean_window.h) over the guarded chunks
    const auto protect = [](bool writable) {
        const auto range = [&](uint32_t start, uint32_t end) {
            constexpr uint32_t kChunk = 1u << MemoryInline::kLeanPolicyShift;
            const uint32_t first = start & ~(kChunk - 1u), last = (end + kChunk - 1u) & ~(kChunk - 1u);
            uint8_t* const copy = MemoryInline::g_leanBase + LeanWindow::kStoreViewOffset + 0x80000000ull + (first & MemoryInline::kLeanMask);
            if (LeanWindow::Protect(copy, last - first, writable) != 0) {
                std::fprintf(stderr, "FAIL: could not change the store view's protection\n");
                ++g_failures;
            }
        };
        range(kSlotBase, kSlotBase + kSlotCount * kSlotBytes);
        range(kSupportPlain, kSupportThrow + 0x100u);
    };
    const unsigned failures = g_failures, reported = g_reported;
    g_reported = kMaxReports;
    protect(true);
    AddStaleCode();
    protect(false);  // as StoreViewSync left it (the policy flags these chunks)
    const unsigned noticed = g_failures - failures;
    g_failures = failures;
    g_reported = reported;
    std::printf("  store view lag: %u failed\n", noticed);
    if (noticed == 0) {
        std::fprintf(stderr, "FAIL: guarded code under a store view writable over it on purpose was not noticed\n");
        ++g_failures;
    }
}

void Stage(const char* name, void (*fn)()) {
    const auto start = Clock::now();
    const unsigned failuresBefore = g_failures;
    fn();
    std::printf("  %-34s %6.2f s%s\n", name, Since(start), g_failures != failuresBefore ? "  FAILED" : "");
    std::fflush(stdout);
}
void GuardedStages(unsigned randomCount, uint64_t firstSeed) {
    SetGuardedCode();
    const uint64_t before = GuardedCompiles();
    Stage("guarded: branch forms", AddBranchForms);
    Stage("guarded: loops and block lengths", AddLoops);
    Stage("guarded: b chains", AddChains);
    Stage("guarded: local calls and jumps", AddLocalCalls);
    Stage("guarded: registered functions", AddRegisteredCalls);
    Stage("guarded: stale code", GuardedStaleCode);
    Stage("guarded: guard fault", GuardFault);
    Stage("guarded: store view lag", StoreViewLag);
    // Last of the stages that place the stale-code programs: link epoch fault 3 is compiled into the blocks made while it is on
    // (their continuations after a call left without the epoch guard), and a later stage that placed the same program in the same
    // slot would find those blocks matching its words and run them (a failure that is the fault's, not the code's).
    Stage("guarded: link epoch faults", LinkEpochFaults);
    {
        const auto t = Clock::now();
        const unsigned failed = g_failures;
        AddRandomPrograms(randomCount / 4 + 1, firstSeed + randomCount);
        std::printf("  %-34s %6.2f s%s\n", "guarded: random programs", Since(t), g_failures != failed ? "  FAILED" : "");
    }
    const uint64_t guarded = GuardedCompiles() - before;
    std::printf("  guarded blocks compiled: %llu\n", static_cast<unsigned long long>(guarded));
    if (PpcInterpreter::JitAvailable() && (guarded == 0) != GuardOff()) {
        std::fprintf(stderr, "FAIL: %s\n", GuardOff() ? "blocks were compiled as guarded code with BRAWL_JIT_GUARD=0"
                                                      : "no block was compiled as guarded code in the guarded stages");
        ++g_failures;
    }
}
}  // namespace

int main() {
    std::signal(SIGABRT, [](int) {
        std::fprintf(stderr, "jit differential: the runtime aborted while running [%s]\n", g_current.c_str());
        std::signal(SIGABRT, SIG_DFL);
        std::abort();
    });
    if (!CheckEncodings()) return 2;
    Memory::Init(0x200000);
    RegisterFunction(kFnModify, "DiffModify", FnModify);
    RegisterFunction(kFnFloat, "DiffFloat", FnFloat);
    RegisterFunction(kFnWrite, "DiffWrite", FnWrite);
    RegisterFunction(kFnReenter, "DiffReenter", FnReenter);
    RegisterFunction(kFnThrow, "DiffThrow", FnThrow);
    RegisterFunction(kFnRewrite, "DiffRewrite", FnRewrite);
    RegisterFunction(kFnRestore, "DiffRestore", FnRestore);
    RegisterFunction(kFnRestoreWritten, "DiffRestoreWritten", FnRestoreWritten);
    RegisterFunction(kFnSetSo, "DiffSetSo", FnSetSo);
    TranslatedFunctionRegistry::Finalize();
    WriteSupportCode();
    if (!SanityChecks()) return 2;

    unsigned randomCount = 8000;
    uint64_t firstSeed = 1;
    if (const char* text = std::getenv("JIT_DIFF_RANDOM")) randomCount = static_cast<unsigned>(std::strtoul(text, nullptr, 10));
    if (const char* text = std::getenv("JIT_DIFF_SEED")) firstSeed = std::strtoull(text, nullptr, 10);

    std::printf("jit differential: JIT backend %s; Engine::Jit %s\n",
                PpcInterpreter::JitAvailable() ? "available" : "not available",
                PpcInterpreter::JitAvailable() ? "compiles blocks" : "runs the cached interpreter");
    std::printf("  engines compared with classic: cached, jit threshold 0, jit threshold %u\n", kDefaultJitThreshold);
    std::printf("  random programs: seeds %llu..%llu\n", static_cast<unsigned long long>(firstSeed),
                static_cast<unsigned long long>(firstSeed + randomCount - 1));

    const auto start = Clock::now();
    NegativeControl();
    SameBlockPatchProbe();
    const uint32_t savedControl = MkwGetHostFpControl();
    Stage("integer arithmetic", AddIntegerArithmetic);
    Stage("integer logic and shifts", AddIntegerLogic);
    Stage("integer immediates", AddIntegerImmediates);
    Stage("integer compares", AddIntegerCompares);
    Stage("rotates", AddIntegerRotates);
    Stage("condition register", AddConditionRegisterPrograms);
    Stage("special registers", AddSpecialRegisterPrograms);
    Stage("memory", AddMemoryPrograms);
    Stage("scalar floating point", AddScalarFloatPrograms);
    Stage("paired singles", AddPairedSinglePrograms);
    Stage("branch forms", AddBranchForms);
    Stage("compare and branch pairs", AddFusedPairs);
    Stage("split compare and branch pairs", AddSplitPairs);
    Stage("loops and block lengths", AddLoops);
    Stage("b chains", AddChains);
    Stage("local calls and jumps", AddLocalCalls);
    Stage("registered functions", AddRegisteredCalls);
    Stage("XER[SO] set in compiled code", AddSummaryOverflow);
    Stage("stale code", AddStaleCode);
    Stage("link epoch faults", LinkEpochFaults);
    {
        const auto t = Clock::now();
        const unsigned before = g_failures;
        AddRandomPrograms(randomCount, firstSeed);
        std::printf("  %-34s %6.2f s%s\n", "random programs", Since(t), g_failures != before ? "  FAILED" : "");
    }
    GuardedStages(randomCount, firstSeed);
    Stage("restore written", AddRestoreWrittenPrograms);
    Stage("restore written fault", RestoreWrittenFault);
    MkwRestoreHostMxcsr(savedControl);
    PpcInterpreter::ClearEngineOverride();

    std::printf("\n  %-14s %8s %10s\n", "category", "programs", "runs");
    unsigned cases = 0;
    uint64_t runs = 0;
    for (const auto& [name, stats] : g_categories) {
        std::printf("  %-14s %8u %10llu\n", name.c_str(), stats.cases, static_cast<unsigned long long>(stats.runs));
        cases += stats.cases;
        runs += stats.runs;
    }
    std::printf("  %-14s %8u %10llu  (compared runs; every one also ran under classic)\n", "total", cases, static_cast<unsigned long long>(runs));
    std::printf("jit differential: %.1f s, %llu guest runs and %llu guest instructions in all (reference included), %u failure%s\n",
                Since(start), static_cast<unsigned long long>(g_runs),
                static_cast<unsigned long long>(PpcInterpreter::InstructionCount()), g_failures, g_failures == 1 ? "" : "s");
    if (g_failures != 0) {
        std::printf("jit differential: FAILED\n");
        return 1;
    }
    std::printf("jit differential: ok\n");
    return 0;
}
