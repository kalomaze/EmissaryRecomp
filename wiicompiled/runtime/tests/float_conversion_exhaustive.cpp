// Exhaustive bit-pattern tests for the inline single<->double conversion fast paths in isa/ppc_isa_float.h and
// isa/ppc_isa_quantized.h, checked against the reference (manual) algorithms, under the default MXCSR and under the
// runtime's FPSCR[NI] mode (FTZ + DAZ). Standalone:
//   clang++ -O2 -std=gnu++17 -march=x86-64-v3 -I wiicompiled/runtime/include \
//       wiicompiled/runtime/tests/float_conversion_exhaustive.cpp -o /tmp/float_conversion_exhaustive && /tmp/float_conversion_exhaustive
// Also with -DMKW_ACCURATE_NANS=1 (NaN results then exact too).
// (a minute or two: 2^32 words per conversion check per MXCSR mode, 2^25 operand sets per arithmetic form).
#include "isa/ppc_isa_float.h"
#include "isa/ppc_isa_quantized.h"

#include <cstdio>
#include <cstdlib>
#include <xmmintrin.h>

namespace {

bool IsNormalOrZero(uint32_t value)
{
    const uint32_t exponent = (value >> 23) & 0xFFu;
    return (exponent != 0 && exponent != 0xFFu) || (value & 0x7FFFFFFFu) == 0;
}

int failures = 0;

void Fail(const char* what, uint64_t input, uint64_t got, uint64_t want)
{
    if (++failures <= 20)
        std::printf("FAIL %s: input %016llx got %016llx want %016llx\n", what, static_cast<unsigned long long>(input),
                    static_cast<unsigned long long>(got), static_cast<unsigned long long>(want));
}

void CheckWidenPredicate()
{
    uint32_t value = 0;
    do
    {
        if (PpcSingleWidensByCastInline(value) != IsNormalOrZero(value))
            Fail("PpcSingleWidensByCastInline", value, PpcSingleWidensByCastInline(value), IsNormalOrZero(value));
    } while (++value != 0);
}

void CheckWiden(const char* mode)
{
    uint32_t value = 0;
    do
    {
        const uint64_t got = PpcConvertToDoubleBitsInline(value);
        const uint64_t want = PpcConvertToDoubleBitsSlow(value);  // the manual's algorithm, for every word
        if (got != want)
            Fail(mode, value, got, want);
    } while (++value != 0);
}

// psq_st's scalar store word: the cast-only flush against the generic per-lane rule, for every single pattern widened
// exactly, for each with its low 29 double fraction bits set to a pseudo-random pattern (values the cast rounds), and
// for 2^28 random doubles.
void CheckPsqScalarStore(const char* mode)
{
    auto check = [&](uint64_t bits) {
        const float single = static_cast<float>(PpcBitCastToDoubleInline(bits));
        const uint32_t got = PpcStorePsqCastFloatBitsInline(single);
        const uint32_t want = PpcStorePsqFloatBitsInline(PpcBitCastToU32Inline(single));
        if (got != want)
            Fail(mode, bits, got, want);
    };
    uint32_t value = 0;
    uint64_t state = 0x9E3779B97F4A7C15ull;
    do
    {
        const uint64_t widened = PpcConvertToDoubleBitsSlow(value);
        check(widened);
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        check(widened ^ (state >> 35));  // the low 29 bits
    } while (++value != 0);
    for (uint32_t i = 0; i < (1u << 28); ++i)
    {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        check(state);
    }
}

// The stack psq_st W=1 float word: the integer per-lane rule (PpcStoreSinglePsqFloatStackBitsInline) against the
// widen-then-FTZ body it replaced at the hot wrapper, for every ps0 word, with a pseudo-random ps1.
void CheckPsqStackScalarStore(const char* mode)
{
    uint32_t value = 0;
    uint64_t state = 0x13198A2E03707344ull;
    do
    {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const double pair = PpcPackPairedBitsInline(value, static_cast<uint32_t>(state >> 32));
        const uint32_t got = PpcStoreSinglePsqFloatStackBitsInline(pair);
        const uint32_t want = PpcConvertToSingleFTZInline(
            PpcBitCastToU64Inline(static_cast<double>(PpcGetPs0Inline(pair))));
        if (got != want)
            Fail(mode, value, got, want);
    } while (++value != 0);
}

// Informational: the words whose plain host cast differs from the manual's widening. Under FTZ+DAZ (the game thread's
// MXCSR once the guest sets FPSCR[NI]) these include every subnormal (DAZ reads them as zero), so the lfs fast-path test
// cannot shrink to an exponent-255-only check.
void CountCastMismatches(const char* mode)
{
    uint64_t nonNan = 0, nan = 0;
    uint32_t value = 0;
    do
    {
        float single;
        std::memcpy(&single, &value, sizeof(single));
        if (PpcBitCastToU64Inline(static_cast<double>(single)) != PpcConvertToDoubleBitsSlow(value))
            ++(((value >> 23) & 0xFFu) == 0xFFu ? nan : nonNan);
    } while (++value != 0);
    std::printf("host cast != manual widening (%s): %llu exponent-255 words, %llu others\n", mode,
                static_cast<unsigned long long>(nan), static_cast<unsigned long long>(nonNan));
}

// fmuls and the fused single family as floats (PpcFmulsFloatInline, PpcFusedSingleFloatInline) against the double
// entry points, whose results must be exactly their widenings: 2^25 operand triples per form, drawn from a mix of
// widened singles (any word, so subnormals, infinities and NaNs of both kinds), single-precision products' typical
// magnitudes, and arbitrary doubles.
uint64_t g_state = 0x243F6A8885A308D3ull;
uint64_t Next()
{
    g_state = g_state * 6364136223846793005ull + 1442695040888963407ull;
    return g_state ^ (g_state >> 29);
}

double Operand(bool single)
{
    const uint64_t r = Next();
    if (single || (r & 3) != 0)
        return PpcBitCastToDoubleInline(PpcConvertToDoubleBitsSlow(static_cast<uint32_t>(r >> 16)));
    return PpcBitCastToDoubleInline(Next());
}

// Without MKW_ACCURATE_NANS (the product default) a NaN result is the host's: for fmsubs/fnmsubs with a NaN B, its sign
// depends on whether Clang folds fma(a, c, -b) into vfmsub (B's sign) or negates B first, which it decides per inlined
// copy, for the double entry points as much as for these. Such results only have to agree up to that sign bit there.
bool SameResult(double got, double want)
{
    const uint64_t g = PpcBitCastToU64Inline(got), w = PpcBitCastToU64Inline(want);
    if (g == w) return true;
#if !MKW_ACCURATE_NANS
    if (std::isnan(got) && std::isnan(want) && (g ^ w) == 0x8000000000000000ull) return true;
#endif
    return false;
}

void CheckEqual(const char* what, double a, double c, double b, double got, double want)
{
    if (!SameResult(got, want))
    {
        if (++failures <= 20)
            std::printf("FAIL %s: a %016llx c %016llx b %016llx got %016llx want %016llx\n", what,
                        static_cast<unsigned long long>(PpcBitCastToU64Inline(a)),
                        static_cast<unsigned long long>(PpcBitCastToU64Inline(c)),
                        static_cast<unsigned long long>(PpcBitCastToU64Inline(b)),
                        static_cast<unsigned long long>(PpcBitCastToU64Inline(got)),
                        static_cast<unsigned long long>(PpcBitCastToU64Inline(want)));
    }
}

void CheckSingleProducts()
{
    for (uint32_t i = 0; i < (1u << 25); ++i)
    {
        const double a = Operand(false), c = Operand(false), b = Operand(false), cs = Operand(true);
        CheckEqual("fmuls", a, c, 0, static_cast<double>(PpcFmulsFloatInline<false>(a, c)), PpcFmulsBody(a, c));
        CheckEqual("fmuls single C", a, cs, 0, static_cast<double>(PpcFmulsFloatInline<true>(a, cs)), PpcFmulsSingleCInline(a, cs));
        CheckEqual("fmadds", a, c, b, static_cast<double>(PpcFusedSingleFloatInline<false, false, false>(a, c, b)), PpcFmaddsBody(a, c, b));
        CheckEqual("fmsubs", a, c, b, static_cast<double>(PpcFusedSingleFloatInline<true, false, false>(a, c, b)), PpcFmsubsBody(a, c, b));
        CheckEqual("fnmadds", a, c, b, static_cast<double>(PpcFusedSingleFloatInline<false, true, false>(a, c, b)), PpcFnmaddsBody(a, c, b));
        CheckEqual("fnmsubs", a, c, b, static_cast<double>(PpcFusedSingleFloatInline<true, true, false>(a, c, b)), PpcFnmsubsBody(a, c, b));
        CheckEqual("fmadds single C", a, cs, b, static_cast<double>(PpcFusedSingleFloatInline<false, false, true>(a, cs, b)), PPC_FmaddsSingleC(a, cs, b));
        CheckEqual("fmsubs single C", a, cs, b, static_cast<double>(PpcFusedSingleFloatInline<true, false, true>(a, cs, b)), PPC_FmsubsSingleC(a, cs, b));
        CheckEqual("fnmadds single C", a, cs, b, static_cast<double>(PpcFusedSingleFloatInline<false, true, true>(a, cs, b)), PPC_FnmaddsSingleC(a, cs, b));
        CheckEqual("fnmsubs single C", a, cs, b, static_cast<double>(PpcFusedSingleFloatInline<true, true, true>(a, cs, b)), PPC_FnmsubsSingleC(a, cs, b));
    }
}

}  // namespace

int main()
{
    CheckWidenPredicate();
    const unsigned int defaultCsr = _mm_getcsr();
    CheckWiden("PpcConvertToDoubleBitsInline (default MXCSR)");
    CheckPsqScalarStore("PpcStorePsqCastFloatBitsInline (default MXCSR)");
    CheckPsqStackScalarStore("PpcStoreSinglePsqFloatStackBitsInline (default MXCSR)");
    CountCastMismatches("default MXCSR");
    CheckSingleProducts();
    _mm_setcsr(defaultCsr | 0x8040u);  // FTZ | DAZ, as the runtime sets them with FPSCR[NI]
    CheckWiden("PpcConvertToDoubleBitsInline (FTZ+DAZ)");
    CheckPsqScalarStore("PpcStorePsqCastFloatBitsInline (FTZ+DAZ)");
    CheckPsqStackScalarStore("PpcStoreSinglePsqFloatStackBitsInline (FTZ+DAZ)");
    CountCastMismatches("FTZ+DAZ");
    CheckSingleProducts();
    _mm_setcsr(defaultCsr);
    if (failures != 0)
    {
        std::printf("%d failures\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("float conversion exhaustive tests passed\n");
    return EXIT_SUCCESS;
}
