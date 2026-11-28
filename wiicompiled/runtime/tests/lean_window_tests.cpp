// LeanWindow (lean_window.h): the lean window's reservation, private backing and shared views, with the layout
// Memory::Init gives MEM1's mirrors (memory.cpp, MapLeanMem1Mirrors): MEM1 at window offset 0 and at 0x02000000,
// 0x04000000 and 0x06000000, and its slice of the dirty map at 0x3E000000 plus each offset / 64.
#define MKW_LEAN_MEMORY 1
#include "../src/lean_window.cpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int g_failures = 0;
void Check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}
}  // namespace

int main() {
    constexpr uint64_t kWindow = uint64_t{1} << 30, kGuard = 0x10000;
    constexpr uint64_t kMem1 = 24u << 20, kMarks = kMem1 >> 6, kDirty = 0x3E000000u;
    constexpr uint64_t kMirrors[] = {0x02000000u, 0x04000000u, 0x06000000u};
    uint8_t* const window = LeanWindow::Reserve(kWindow + kGuard);
    Check(window != nullptr, "reserve the window");
    if (window == nullptr) return 1;

    uint8_t* data[4] = {window, window + kMirrors[0], window + kMirrors[1], window + kMirrors[2]};
    uint8_t* marks[4];
    for (int i = 0; i < 4; ++i) marks[i] = window + kDirty + (static_cast<uint64_t>(data[i] - window) >> 6);
    std::string error;
    Check(LeanWindow::MapShared(kMem1, data, 4, &error), error.c_str());
    Check(LeanWindow::MapShared(kMarks, marks, 4, &error), error.c_str());

    // MEM2's place and a mark spill chunk: private memory, next to the shared views.
    uint8_t* const mem2 = window + 0x10000000u;
    Check(LeanWindow::Commit(mem2, 0x20000) == 0, "commit private memory");
    Check(LeanWindow::Commit(window + kDirty + kMarks, 0x10000) == 0, "commit the chunk after MEM1's marks");

    // Fresh views read zero.
    bool zero = true;
    for (int i = 0; i < 4; ++i)
        zero = zero && data[i][0] == 0 && data[i][kMem1 - 1] == 0 && marks[i][0] == 0 && marks[i][kMarks - 1] == 0;
    Check(zero, "shared views read zero");

    // The address Brawl reads through a mirror (0x8348B5C8: MEM1 0x0148B5C8), written through MEM1 (0x8148B5C8).
    const uint64_t at = 0x0148B5C8u;
    const uint32_t value = 0xDEADBEEFu;
    std::memcpy(data[0] + at, &value, 4);
    for (int i = 1; i < 4; ++i) {
        uint32_t read = 0;
        std::memcpy(&read, data[i] + at, 4);
        Check(read == value, "a MEM1 store reads back through each mirror");
    }
    // A store through a mirror reaches MEM1 and the other mirrors, its mark too.
    data[3][0x10] = 0x77;
    marks[3][0x10 >> 6] = 1;
    Check(data[0][0x10] == 0x77 && data[1][0x10] == 0x77 && data[2][0x10] == 0x77, "a mirror store reaches MEM1");
    Check(marks[0][0] == 1 && marks[1][0] == 1 && marks[2][0] == 1, "a mirror's mark is MEM1's mark");
    // The last bytes of each view, and private memory unaffected.
    data[1][kMem1 - 1] = 0x42;
    Check(data[0][kMem1 - 1] == 0x42, "the last byte aliases");
    mem2[0] = 0x11;
    Check(data[0][0] == 0 && mem2[0] == 0x11, "private memory is separate");
    window[kDirty + kMarks] = 1;  // a pair mark's spill byte past MEM1's slice: private, next to the shared slice
    Check(marks[0][kMarks - 1] == 0 && marks[2][kMarks - 1] == 0, "the spill byte is not part of the shared marks");

    // Views of parts of one object (the aliased view's tail): a second reservation, a 128 KiB object shown whole at 0 and
    // 0x40000, and its first 64 KiB at 0x80000.
    uint8_t* const other = LeanWindow::Reserve(0x100000);
    Check(other != nullptr, "reserve a second range");
    if (other != nullptr) {
        const LeanWindow::View views[] = {{other, 0, 0x20000}, {other + 0x40000, 0, 0x20000}, {other + 0x80000, 0, 0x10000}};
        Check(LeanWindow::MapSharedViews(0x20000, views, 3, &error), error.c_str());
        other[0x1234] = 0x5C;
        other[0x40000 + 0x1FFFF] = 0x6D;
        Check(other[0x40000 + 0x1234] == 0x5C && other[0x80000 + 0x1234] == 0x5C, "a partial view shows the object's bytes");
        Check(other[0x1FFFF] == 0x6D, "the views share the last byte");
        other[0x80000 + 0xFFFF] = 0x7E;
        Check(other[0xFFFF] == 0x7E && other[0x40000 + 0xFFFF] == 0x7E, "a store through the partial view reaches the others");
    }

    if (g_failures == 0) std::puts("lean window tests passed");
    return g_failures == 0 ? 0 : 1;
}
