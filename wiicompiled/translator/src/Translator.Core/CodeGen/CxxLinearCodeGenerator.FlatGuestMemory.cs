namespace Translator.Core.CodeGen;

/// <summary>
/// Selects the runtime helper family for guest memory accesses. Checked helpers resolve through the 1 MiB page
/// table each access; flat helpers use the 4 GiB guest reservation, so an access is just a byte swap with
/// MMIO/EFB/executable-write interception moved to host page protections. Both families behave identically.
/// </summary>
public sealed partial class CxxLinearCodeGenerator
{
    private enum FlatMemoryHelperFamily
    {
        Read,
        Write,
        WriteRam
    }

    // Index order is 8, 16, 32, 64 bits. Keep the literal spellings in one
    // table: the runtime deliberately exposes separate guarded and proven-RAM
    // write families, while Bits(sizeBytes) supplies the same normalization at
    // every call site.
    private static readonly string[] FlatHelpers =
    {
        "MemoryInline::FlatRead8", "MemoryInline::FlatRead16",
        "MemoryInline::FlatRead32", "MemoryInline::FlatRead64",
        "MemoryInline::FlatReadFloat8", "MemoryInline::FlatReadFloat16",
        "MemoryInline::FlatReadFloat32", "MemoryInline::FlatReadFloat64",
        "MemoryInline::FlatWrite8", "MemoryInline::FlatWrite16",
        "MemoryInline::FlatWrite32", "MemoryInline::FlatWrite64",
        "MemoryInline::FlatWriteFloat8", "MemoryInline::FlatWriteFloat16",
        "MemoryInline::FlatWriteFloat32", "MemoryInline::FlatWriteFloat64",
        "MemoryInline::FlatWriteRam8", "MemoryInline::FlatWriteRam16",
        "MemoryInline::FlatWriteRam32", "MemoryInline::FlatWriteRam64",
        "MemoryInline::FlatWriteRamFloat8", "MemoryInline::FlatWriteRamFloat16",
        "MemoryInline::FlatWriteRamFloat32", "MemoryInline::FlatWriteRamFloat64"
    };

    private static int FlatHelperIndex(int bits) => bits switch
    {
        8 => 0,
        16 => 1,
        64 => 3,
        _ => 2
    };

    private static string FlatHelper(FlatMemoryHelperFamily family, int bits, bool isFloat)
    {
        var familyIndex = (int)family;
        if ((uint)familyIndex >= 3u)
            throw new ArgumentOutOfRangeException(nameof(family), family, null);
        return FlatHelpers[((familyIndex * 2) + (isFloat ? 1 : 0)) * 4 + FlatHelperIndex(bits)];
    }

    /// <summary>
    /// Helper spellings are a closed set of widths and families. Selection is
    /// table-driven but remains allocation-free at each emitted access.
    /// </summary>
    private static string FlatReadHelper(int bits, bool isFloat) =>
        FlatHelper(FlatMemoryHelperFamily.Read, bits, isFloat);

    private static string FlatWriteHelper(int bits, bool isFloat) =>
        FlatHelper(FlatMemoryHelperFamily.Write, bits, isFloat);

    /// <summary>
    /// The exact predicate <c>MemoryInline::FlatWriteNeedsPolicy</c> uses:
    /// 0xCC000000..0xCDFFFFFF. Kept in sync by construction - the two must never
    /// disagree, because that is what makes the check-free family sound.
    /// </summary>
    private static bool FlatWriteNeedsPolicy(uint address) =>
        (address & 0xFE000000u) == 0xCC000000u;

    /// <summary>
    /// A translate-time-constant effective address whose complete span sits
    /// outside the MMIO policy window, and which therefore provably takes the
    /// plain-store arm of every <c>FlatWrite*</c> helper. Lean memory
    /// (MKW_LEAN_MEMORY) also needs the span outside the guest code the runtime
    /// guards and outside every alias of the MMIO window: there the check-free
    /// helper skips the store policy byte, which is what sends a store to the
    /// code write trap and to the MMIO handlers.
    /// </summary>
    private static bool IsProvenRamStore(uint address, int sizeBytes)
    {
        if (sizeBytes <= 0 || sizeBytes > 8) return false;
        return IsProvenRamRange(address, sizeBytes);
    }

    /// <summary>
    /// True when the entire byte span <c>[address, address + length)</c> is provably ordinary guest RAM:
    /// does not wrap, does not intersect the MMIO window or any lean-memory alias of it, and does not overlap
    /// any guarded guest code range.
    /// </summary>
    private static bool IsProvenRamRange(uint address, int length)
    {
        if (length <= 0) return false;
        var last = unchecked(address + (uint)(length - 1));
        if (last < address) return false; // wraps past the end of the address space
        var canonicalFirst = address & 0x3FFFFFFFu;
        var canonicalLast = last & 0x3FFFFFFFu;
        if (canonicalLast < canonicalFirst) return false; // crosses 1 GiB alias boundary
        if (canonicalFirst < 0x0E000000u && canonicalLast >= 0x0C000000u) return false; // MMIO / lean MMIO alias
        return !FlatWriteNeedsPolicy(address) && !FlatWriteNeedsPolicy(last) &&
               !OverlapsGuardedCode(address, last);
    }

    /// <summary>
    /// Lean memory folds every view onto <c>addr &amp; 0x3FFFFFFF</c>, so the MMIO window (+0x0C000000..+0x0DFFFFFF)
    /// is reached from 0x0C/0x0D, 0x4C/0x4D, 0x8C/0x8D and 0xCC/0xCD alike.
    /// </summary>
    private static bool IsLeanMmioAlias(uint address) =>
        (address & 0x3FFFFFFFu) is >= 0x0C000000u and < 0x0E000000u;

    /// <summary>
    /// Guest code that a constant-address store must not reach check-free: main.dol's own text sections and the
    /// REL's executable sections (Brawl's module homes), in their cached and uncached views. Filled by
    /// <see cref="RegisterGuardedCode"/> when the CLI loads the program image, before any translation.
    /// </summary>
    private static Loading.AddressRange[] _guardedCode = Array.Empty<Loading.AddressRange>();

    public static void RegisterGuardedCode(IEnumerable<Loading.AddressRange> ranges)
    {
        var all = new List<Loading.AddressRange>(_guardedCode);
        foreach (var range in ranges)
        {
            if (range.End <= range.Start) continue;
            all.Add(range);
            if ((range.Start >> 28) is 0x8u or 0x9u && (range.End - 1) >> 28 == range.Start >> 28)
                all.Add(new Loading.AddressRange(range.Start + 0x40000000u, range.End + 0x40000000u)); // uncached view
        }
        _guardedCode = all.ToArray();
    }

    private static bool OverlapsGuardedCode(uint first, uint last)
    {
        foreach (var range in _guardedCode)
        {
            if (first < range.End && last >= range.Start) return true;
        }
        return false;
    }

    private static string GuestLoadHelper(int sizeBytes, bool isFloat) =>
        FlatReadHelper(Bits(sizeBytes), isFloat);

    private static string GuestStoreHelper(int sizeBytes, bool isFloat) =>
        FlatWriteHelper(Bits(sizeBytes), isFloat);

    /// <summary>
    /// Stack slots are ordinary guest memory; the checked variants only skipped
    /// the executable-write guard because a stack frame can never hold code. The
    /// flat path has no per-access tables at all, so both collapse to one helper.
    /// </summary>
    private static string StackLoadHelper(int sizeBytes, bool isFloat) =>
        FlatReadHelper(Bits(sizeBytes), isFloat);

    /// <summary>
    /// Check-free flat store family. Selected only where the caller has already
    /// proven the effective address is ordinary guest RAM; see the contract on
    /// <c>MemoryInline::FlatWriteRam*</c> in runtime/include/memory_access.h.
    /// </summary>
    private static string FlatWriteRamHelper(int bits, bool isFloat) =>
        FlatHelper(FlatMemoryHelperFamily.WriteRam, bits, isFloat);

    private static string StackStoreHelper(int sizeBytes, bool isFloat)
    {
        // A stack slot is `r1 + constant` with the whole chain back to r1
        // visible in this body (StackAddressFacts). Guest stack frames
        // live in MEM1/MEM2 and can never be MMIO, so the per-store MMIO
        // policy test is provably dead here - and 45% of all emitted flat
        // stores have this shape.
        return FlatWriteRamHelper(Bits(sizeBytes), isFloat);
    }
}
