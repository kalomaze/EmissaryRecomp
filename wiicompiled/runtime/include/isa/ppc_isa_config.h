#pragma once

#include <atomic>
#include <cstdint>

#define MKW_RESTRICT __restrict
#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#else
#error "ppc_isa_config.h has no SIMD intrinsics header for this architecture"
#endif

inline constexpr bool MkwStateFreeAbiEnabled(uint32_t) noexcept
{
    return true;
}

#if defined(_WIN32)
#define MKW_PPC_FORCE_INLINE __forceinline
#define MKW_PPC_NO_INLINE __declspec(noinline)
#define MKW_PPC_INTERNAL_CALL __regcall
#else
// __forceinline/__declspec are MS-extension keywords Clang only recognizes when targeting
// Windows (MSVC or mingw); native Linux Clang needs the GNU-attribute spellings instead.
// __regcall has no portable non-Windows equivalent worth chasing here - the extra register
// args it saves matter for the hot PPC interpreter loop on Windows, but plain calls are fine
// elsewhere.
#define MKW_PPC_FORCE_INLINE __attribute__((always_inline)) inline
#define MKW_PPC_NO_INLINE __attribute__((noinline))
#define MKW_PPC_INTERNAL_CALL
#endif
#define MKW_PPC_ALWAYS_INLINE_BODY __attribute__((always_inline))
#define MKW_PPC_COLD __attribute__((cold))

// MKW_OUTLINE_GUEST_OPS=1 (opt-in; Clang, lean memory, exact floats): the guest operations translated code repeats at
// every site (lean stores, exact single-precision loads/stores/arithmetic, compares into CR) and their cold fallbacks
// are out-of-line calls with Clang's preserve_all convention, MKW_GUEST_OP_CC. The callee saves every register it
// touches, XMM/vector ones included, so a call site is its argument moves and the call: no spill of the caller's live
// values (x86-64 has no callee-saved XMM registers, so a plain call spilled every live float). Each operation is the
// same code as before, compiled once with the same floating-point options (src/guest_ops_outlined.cpp): results are
// bit-identical; translated code shrinks. Off by default: the game thread ran ~11% more cycles per frame with it on
// (vanilla Training, Battlefield: 13.62M vs 12.14M), so smaller code is not faster code here.
#ifndef MKW_OUTLINE_GUEST_OPS
#define MKW_OUTLINE_GUEST_OPS 0
#endif
#if MKW_OUTLINE_GUEST_OPS
#define MKW_GUEST_OP_CC __attribute__((preserve_all))
#else
#define MKW_GUEST_OP_CC
#endif

// MKW_COLD_OP_CC: the cold out-of-line arms of inline guest operations (lfs's exact widening of a subnormal or NaN,
// stfs's subnormal narrowing, psq's generic GQR fallbacks) use preserve_all whatever MKW_OUTLINE_GUEST_OPS says. Under
// the default convention each of those never-taken calls was a call site to the register allocator: every live XMM
// value (x86-64 has no callee-saved XMM registers) and caller-saved GPR had to survive it, so a hot body around an lfs
// or stfs saved callee-saved GPRs in its prologue and spilled floats around the call. The callees are the same code;
// only who saves registers changes, so results are identical. Clang on x86-64 only.
#if defined(__clang__) && defined(__x86_64__)
#define MKW_COLD_OP_CC __attribute__((preserve_all))
#else
#define MKW_COLD_OP_CC MKW_GUEST_OP_CC
#endif


using MkwStateFreeResult2 = uint64_t __attribute__((ext_vector_type(2)));

// MKW_EABI_REGS register-passing entries with two results (r3/r4/f1/f2 in that order, FPRs as their raw bits):
// two integers, so SysV x86-64 returns them in rax:rdx.
struct MkwEabiPair {
    uint64_t first;
    uint64_t second;
};

// Three or four results: returned through memory (the caller's frame), as CpuContext would have carried them.
struct MkwEabiQuad {
    uint64_t v[4];
};
