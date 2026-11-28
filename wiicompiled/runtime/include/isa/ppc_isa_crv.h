#pragma once

#include <cstdint>

// The condition register, one 64-bit value per field (CpuContext::crv), as Dolphin keeps it
// (Source/Core/Core/PowerPC/ConditionRegister.h), so a compare is a subtraction and a call boundary
// stores only the fields a function wrote, instead of rebuilding the packed 32-bit word at every
// compare. For a field value v:
//   LT  bit 62 of v
//   GT  v > 0 (signed)
//   EQ  the low 32 bits of v are 0
//   SO  bit 59 of v differs from bit 62
// A 32-bit compare's operands, sign- or zero-extended and subtracted in 64 bits, give a value in
// (-2^32, 2^32): its LT, GT and EQ are the compare's and its SO is clear, since bits 32-63 of such a
// value are all equal. (Dolphin's SO is bit 59 alone, so its negative results read SO set; this
// encoding keeps the architectural SO exact, as the packed word did.) Any field value from a
// nibble: PpcCrvEncode. The packed word (mfcr, mtcrf, OSContext, the interpreter's CR logic):
// PpcCrvPack / PpcCrvUnpack.

#ifndef MKW_LEAN_FLAGS
#define MKW_LEAN_FLAGS 0  // compares leave SO clear (translated code never sets XER[SO])
#endif

#if defined(_WIN32)
#define MKW_CRV_INLINE __forceinline
#else
#define MKW_CRV_INLINE __attribute__((always_inline)) inline
#endif

// The field value of a 4-bit CR field (LT 8, GT 4, EQ 2, SO 1).
constexpr int64_t PpcCrvEncode(uint32_t nibble) noexcept {
    uint64_t v = 0;
    if ((nibble & 0x2u) == 0) v |= 1u;                    // EQ clear: a nonzero low word
    if ((nibble & 0x4u) != 0) {
        if ((nibble & 0x2u) != 0) v |= uint64_t{1} << 32; // GT and EQ: positive with a zero low word
    } else {
        v |= uint64_t{1} << 63;                           // GT clear: negative
    }
    if ((nibble & 0x8u) != 0) v |= uint64_t{1} << 62;
    if (((nibble & 0x1u) != 0) != ((nibble & 0x8u) != 0)) v |= uint64_t{1} << 59;
    return static_cast<int64_t>(v);
}

// PpcCrvEncode of every nibble, for values only known at run time (one load instead of the bit work).
inline constexpr int64_t kPpcCrvTable[16] = {
    PpcCrvEncode(0), PpcCrvEncode(1), PpcCrvEncode(2), PpcCrvEncode(3), PpcCrvEncode(4), PpcCrvEncode(5),
    PpcCrvEncode(6), PpcCrvEncode(7), PpcCrvEncode(8), PpcCrvEncode(9), PpcCrvEncode(10), PpcCrvEncode(11),
    PpcCrvEncode(12), PpcCrvEncode(13), PpcCrvEncode(14), PpcCrvEncode(15)};
MKW_CRV_INLINE int64_t PpcCrvFromNibble(uint32_t nibble) noexcept { return kPpcCrvTable[nibble & 0xFu]; }

constexpr int64_t kPpcCrvLt = PpcCrvEncode(0x8u);
constexpr int64_t kPpcCrvGt = PpcCrvEncode(0x4u);
constexpr int64_t kPpcCrvEq = PpcCrvEncode(0x2u);
constexpr int64_t kPpcCrvSo = PpcCrvEncode(0x1u);

MKW_CRV_INLINE constexpr bool PpcCrvLt(int64_t v) noexcept { return ((static_cast<uint64_t>(v) >> 62) & 1u) != 0; }
MKW_CRV_INLINE constexpr bool PpcCrvGt(int64_t v) noexcept { return v > 0; }
MKW_CRV_INLINE constexpr bool PpcCrvEq(int64_t v) noexcept { return static_cast<uint32_t>(v) == 0; }
MKW_CRV_INLINE constexpr bool PpcCrvSo(int64_t v) noexcept {
    return (((static_cast<uint64_t>(v) >> 59) ^ (static_cast<uint64_t>(v) >> 62)) & 1u) != 0;
}

// The field's 4-bit value.
MKW_CRV_INLINE constexpr uint32_t PpcCrvNibble(int64_t v) noexcept {
    return (uint32_t{PpcCrvLt(v)} << 3) | (uint32_t{PpcCrvGt(v)} << 2) | (uint32_t{PpcCrvEq(v)} << 1) |
           uint32_t{PpcCrvSo(v)};
}

// One bit of a field, numbered as in the CR (0 LT, 1 GT, 2 EQ, 3 SO).
// A constant bit (translated code) is one test; a run-time one (the interpreter's bc) is the nibble's bit, no branches.
MKW_CRV_INLINE constexpr bool PpcCrvBit(int64_t v, uint32_t bit) noexcept {
    if (__builtin_constant_p(bit)) {
        switch (bit & 3u) {
            case 0: return PpcCrvLt(v);
            case 1: return PpcCrvGt(v);
            case 2: return PpcCrvEq(v);
            default: return PpcCrvSo(v);
        }
    }
    return ((PpcCrvNibble(v) >> (3u - (bit & 3u))) & 1u) != 0;
}

MKW_CRV_INLINE int64_t PpcCrvWithBit(int64_t v, uint32_t bit, bool set) noexcept {
    const uint32_t mask = 0x8u >> (bit & 3u);
    const uint32_t nibble = PpcCrvNibble(v);
    return PpcCrvFromNibble(set ? (nibble | mask) : (nibble & ~mask));
}

// A field value with SO set: XER[SO] copied into the field.
[[gnu::cold, gnu::noinline]] inline int64_t PpcCrvWithSo(int64_t v) noexcept { return PpcCrvFromNibble(PpcCrvNibble(v) | 1u); }

// PpcCrvWithSo of a 32-bit compare's value v (in (-2^32, 2^32), so bits 32-63 are equal and SO is clear): flipping bit
// 59 sets SO and keeps LT, GT and EQ, except for v == 0, whose GT would appear; EQ with SO is bit 63 (GT clear) and bit
// 59. A few instructions and no call: the compare's never-taken arm (translated code never sets XER[SO]) then clobbers
// no register, so it costs the function nothing outside its out-of-line block.
MKW_CRV_INLINE int64_t PpcCrvCompareWithSo(int64_t v) noexcept {
    return v == 0 ? static_cast<int64_t>(0x8800000000000000ull) : (v ^ (int64_t{1} << 59));
}
static_assert(0x8800000000000000ull == static_cast<uint64_t>(PpcCrvEncode(0x3u)), "EQ|SO encoding");

// cmp/cmpi/cmpl/cmpli and the record forms (a result against 0): the field value, SO from XER[SO].
// XER[SO] is never set by translated code (MKW_LEAN_FLAGS), so its copy is a branch that is never taken.
MKW_CRV_INLINE int64_t PpcCrvCompare(int32_t a, int32_t b, uint32_t xer) noexcept {
    const int64_t v = static_cast<int64_t>(a) - static_cast<int64_t>(b);
    if (!MKW_LEAN_FLAGS && (xer & 0x80000000u) != 0) [[unlikely]]
        return PpcCrvCompareWithSo(v);
    return v;
}
MKW_CRV_INLINE int64_t PpcCrvCompare(uint32_t a, uint32_t b, uint32_t xer) noexcept {
    const int64_t v = static_cast<int64_t>(static_cast<uint64_t>(a) - static_cast<uint64_t>(b));
    if (!MKW_LEAN_FLAGS && (xer & 0x80000000u) != 0) [[unlikely]]
        return PpcCrvCompareWithSo(v);
    return v;
}

// fcmpu/fcmpo/ps_cmp*: unordered (a NaN operand) is SO alone.
MKW_CRV_INLINE int64_t PpcCrvCompareFloat(double a, double b) noexcept {
    return a < b ? kPpcCrvLt : a > b ? kPpcCrvGt : a == b ? kPpcCrvEq : kPpcCrvSo;
}

// The packed 32-bit CR (cr0 in the top nibble) from its eight fields, and back.
MKW_CRV_INLINE constexpr uint32_t PpcCrvPack(int64_t f0, int64_t f1, int64_t f2, int64_t f3, int64_t f4, int64_t f5,
                                             int64_t f6, int64_t f7) noexcept {
    return (PpcCrvNibble(f0) << 28) | (PpcCrvNibble(f1) << 24) | (PpcCrvNibble(f2) << 20) |
           (PpcCrvNibble(f3) << 16) | (PpcCrvNibble(f4) << 12) | (PpcCrvNibble(f5) << 8) |
           (PpcCrvNibble(f6) << 4) | PpcCrvNibble(f7);
}
MKW_CRV_INLINE constexpr uint32_t PpcCrvPack(const int64_t (&f)[8]) noexcept {
    return PpcCrvPack(f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
}
MKW_CRV_INLINE void PpcCrvUnpack(uint32_t cr, int64_t (&f)[8]) noexcept {
    for (uint32_t i = 0; i < 8; ++i) f[i] = PpcCrvFromNibble(cr >> ((7u - i) * 4u));
}

// A whole-CR destination in translated code: `PpcCrView(f0, ..., f7) = packed;` sets the eight fields
// (field locals or CpuContext's), so the IR's whole-register CR writes (mtcrf, mcrf, crxor, ...) need no
// form of their own.
struct PpcCrView {
    int64_t& f0; int64_t& f1; int64_t& f2; int64_t& f3; int64_t& f4; int64_t& f5; int64_t& f6; int64_t& f7;
    MKW_CRV_INLINE PpcCrView(int64_t& a0, int64_t& a1, int64_t& a2, int64_t& a3, int64_t& a4, int64_t& a5,
                             int64_t& a6, int64_t& a7) noexcept
        : f0(a0), f1(a1), f2(a2), f3(a3), f4(a4), f5(a5), f6(a6), f7(a7) {}
    MKW_CRV_INLINE void operator=(uint32_t cr) const noexcept {
        f0 = PpcCrvFromNibble(cr >> 28);
        f1 = PpcCrvFromNibble(cr >> 24);
        f2 = PpcCrvFromNibble(cr >> 20);
        f3 = PpcCrvFromNibble(cr >> 16);
        f4 = PpcCrvFromNibble(cr >> 12);
        f5 = PpcCrvFromNibble(cr >> 8);
        f6 = PpcCrvFromNibble(cr >> 4);
        f7 = PpcCrvFromNibble(cr);
    }
    MKW_CRV_INLINE operator uint32_t() const noexcept { return PpcCrvPack(f0, f1, f2, f3, f4, f5, f6, f7); }
};

// crand/cror/crxor/... (op as PpcCrLogicalResident's: 0 crnor, 1 crandc, 2 crxor, 3 crnand, 4 crand,
// 5 creqv, 6 crorc, 7 cror) on two field bits, the result for the field of bt.
MKW_CRV_INLINE constexpr bool PpcCrvLogic(uint32_t op, bool a, bool b) noexcept {
    switch (op & 7u) {
        case 0: return !(a || b);
        case 1: return a && !b;
        case 2: return a != b;
        case 3: return !(a && b);
        case 4: return a && b;
        case 5: return a == b;
        case 6: return a || !b;  // crorc complements CRBB.
        default: return a || b;
    }
}
