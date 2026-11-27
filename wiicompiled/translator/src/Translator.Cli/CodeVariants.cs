using System;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Translator.Core.Analysis;
using Translator.Core.CodeGen;
using Translator.Core.Loading;
using Translator.Core.Translation;

/// <summary>
/// Code variants (translate-recursive --code-variants SPEC --code-variants-out DIR): extra translations of functions as
/// a mod leaves them, beside the base translation, for a mod whose code patches are fixed per release (Project+'s Gecko
/// code lists). The spec gives the mod's patched code words, the memory its hooks branch into (the code lists, as
/// installed) and the functions to translate. Each variant is the function decoded from that patched image, with the
/// hook bodies decoded as its own blocks (a C2 hook branches into the list and back), named func_X_pp. The runtime
/// (guest_code_patch.cpp) runs a variant instead of the interpreter only while guest memory holds exactly the words it
/// was translated from (its manifest lists them), so another mod, another release or a code that changes later falls
/// back to the interpreter. The base translation is untouched: a game without the mod never runs a variant.
///
/// Spec lines: "F addr" (translate this function), "W addr word" (the patched image's word at addr), "R start end file"
/// (the region's bytes from a memory dump, path relative to the spec; branches may enter it), "X pc" / "C target" (a PC
/// or call target the runtime's interpreter treats specially: a function holding or calling one stays interpreted).
/// Hex, no 0x.
/// Output: DIR/variants.txt ("V addr symbol" then "G start count words..." lines: the guest words it requires) and
/// DIR/symbol.cpp per variant; DIR/rejected.txt lists functions left to the interpreter and why.
/// </summary>
internal static class CodeVariants
{
    internal sealed record Spec(
        IReadOnlyList<uint> Functions,
        IReadOnlyDictionary<uint, uint> Words,
        IReadOnlyList<(AddressRange Range, byte[] Bytes)> Regions,
        IReadOnlySet<uint> InterpreterOnlyPcs,
        IReadOnlySet<uint> InterpreterOnlyCallTargets);

    internal static Spec ReadSpec(string path)
    {
        var functions = new List<uint>();
        var words = new Dictionary<uint, uint>();
        var regions = new List<(AddressRange, byte[])>();
        var interpreterPcs = new HashSet<uint>();
        var interpreterCalls = new HashSet<uint>();
        var directory = Path.GetDirectoryName(Path.GetFullPath(path))!;
        foreach (var raw in File.ReadLines(path))
        {
            var line = raw.Trim();
            if (line.Length == 0 || line[0] == '#') continue;
            var f = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            switch (f[0])
            {
                case "F":
                    functions.Add(Hex(f[1]));
                    break;
                case "W":
                    words[Hex(f[1])] = Hex(f[2]);
                    break;
                case "X":
                    interpreterPcs.Add(Hex(f[1]));
                    break;
                case "C":
                    interpreterCalls.Add(Hex(f[1]));
                    break;
                case "R":
                {
                    var start = Hex(f[1]);
                    var end = Hex(f[2]);
                    var bytes = File.ReadAllBytes(Path.Combine(directory, f[3]));
                    if (bytes.Length < end - start)
                        throw new InvalidDataException($"{f[3]}: {bytes.Length} bytes, region 0x{start:X8}-0x{end:X8} needs {end - start}");
                    regions.Add((new AddressRange(start, end), bytes[..(int)(end - start)]));
                    break;
                }
                default:
                    throw new InvalidDataException($"{path}: unknown line '{line}'");
            }
        }

        return new Spec(functions.Distinct().OrderBy(static a => a).ToArray(), words, regions, interpreterPcs,
            interpreterCalls);
    }

    static uint Hex(string text) => uint.Parse(text, NumberStyles.HexNumber, CultureInfo.InvariantCulture);

    /// <summary>
    /// Whether the ctr a bctr at <paramref name="bctr"/> jumps through is a pointer loaded from a fixed offset (a
    /// virtual call or function pointer): going back over the straight run of decoded instructions before it to its
    /// "mtctr rX", the first instruction naming rX in either register field is "lwz rX, d(rY)". An indexed load
    /// (lwzx: a switch table) or anything else counts as a computed jump (conservative).
    /// </summary>
    static bool CtrLoadedFromMemory(IReadOnlyDictionary<uint, Translator.Core.Disassembly.PpcInstruction> decoded, uint bctr)
    {
        uint? register = null;
        for (var at = bctr - 4; at + 16 * 4 > bctr && decoded.TryGetValue(at, out var ins); at -= 4)
        {
            var word = ins.Word;
            if (register is null)
            {
                if ((word & 0xFC1FFFFFu) == 0x7C0903A6u) register = (word >> 21) & 31;  // mtctr rX
                continue;
            }
            var rd = (word >> 21) & 31;
            var ra = (word >> 16) & 31;
            if (rd != register && ra != register) continue;
            return rd == register && word >> 26 == 32;
        }
        return false;
    }

    /// <summary>
    /// Whether code at <paramref name="target"/> starts like a subroutine that returns to its caller's lr: along its
    /// straight line (conditional branches fall through), a leaf's blr or a prologue's "mflr rX; stw rX, d(r1)" comes
    /// before any other use of lr or any unconditional branch or call, or a plain branch to a known function entry (a
    /// stub that sets arguments and tail-calls: Project+'s 0x8056AEF0, Syriinge's call-original stubs), which returns
    /// to that lr in its place.
    /// </summary>
    static bool LooksLikeSubroutine(ProgramImage image, uint target, Func<uint, bool> isEntry)
    {
        uint Word(uint at) => image.Contains(at, 4)
            ? BinaryPrimitives.ReadUInt32BigEndian(image.Memory.AsSpan(image.GetOffset(at, 4), 4))
            : 0u;
        var visited = new HashSet<uint>();
        for (var (at, steps) = (target, 0); steps < 128 && visited.Add(at); at += 4, steps++)
        {
            var word = Word(at);
            if (word == 0x4E800020u) return true;  // blr
            // A conditional return (bnelr, beqlr...: bclr without link) returns to the caller's lr as well; a leaf
            // often starts with one (Project+'s fighter modules, PPOnline).
            if (word >> 26 == 19 && ((word >> 1) & 0x3FF) == 16 && (word & 1) == 0) return true;
            if ((word & 0xFC1FFFFFu) == 0x7C0802A6u)  // mflr rD: a prologue saves it to the stack
            {
                var rd = (word >> 21) & 31;
                for (var next = at + 4; next < at + 4 * 4; next += 4)
                {
                    var use = Word(next);
                    if (use >> 26 is 36 or 37 && ((use >> 21) & 31) == rd && ((use >> 16) & 31) == 1) return true;
                }
                return false;
            }
            if ((word & 0xFC1FFFFFu) == 0x7C0803A6u) return false;           // mtlr
            if (word >> 26 == 18)                                            // b, bl
            {
                var disp = word & 0x03FFFFFCu;
                var branchTarget = unchecked(at + ((disp & 0x02000000u) != 0 ? disp - 0x04000000u : disp));
                if ((word & 3u) != 0) return false;                          // bl (and absolute forms)
                if (isEntry(branchTarget)) return true;                      // a tail call
                if (branchTarget > at && branchTarget - at < 0x1000)         // a jump further into its own body
                {
                    at = branchTarget - 4;
                    continue;
                }
                return false;
            }
            if (word >> 26 == 16 && (word & 1) != 0) return false;           // bcl
            // An import veneer: "lis rX, hi; addi rX, rX, lo (or ori); mtctr rX; bctr" to a known function is a tail
            // call to it (PPOnline reaches main.dol's functions through such stubs).
            if (word == 0x4E800420u && ConstantCtrTarget(Word, at) is { } veneerTarget) return isEntry(veneerTarget);
            if (word >> 26 == 19 && ((word >> 1) & 0x3FF) is 16 or 528) return false;  // other bclr/bcctr forms
        }
        return false;
    }

    // The constant a "lis rX, hi; addi rX, rX, lo (or ori rX, rX, lo); mtctr rX" right before `bctr` puts in ctr.
    static uint? ConstantCtrTarget(Func<uint, uint> word, uint bctr)
    {
        var mtctr = word(bctr - 4);
        if ((mtctr & 0xFC1FFFFFu) != 0x7C0903A6u) return null;
        var register = (mtctr >> 21) & 31;
        var low = word(bctr - 8);
        var high = word(bctr - 12);
        if (high >> 26 != 15 || ((high >> 21) & 31) != register || ((high >> 16) & 31) != 0) return null;  // lis
        if (((low >> 21) & 31) != register || ((low >> 16) & 31) != register) return null;
        var hi = (high & 0xFFFFu) << 16;
        return (low >> 26) switch
        {
            14 => unchecked(hi + (uint)(short)(low & 0xFFFFu)),  // addi
            24 => hi | (low & 0xFFFFu),                          // ori
            _ => null,
        };
    }

    /// <summary>The base image with the spec's words and regions written over it.</summary>
    internal static ProgramImage BuildOverlay(ProgramImage image, Spec spec)
    {
        var memory = (byte[])image.Memory.Clone();
        foreach (var (address, word) in spec.Words)
        {
            BinaryPrimitives.WriteUInt32BigEndian(memory.AsSpan(image.GetOffset(address, 4), 4), word);
        }

        foreach (var (range, bytes) in spec.Regions)
        {
            bytes.CopyTo(memory.AsSpan(image.GetOffset(range.Start, bytes.Length), bytes.Length));
        }

        return new ProgramImage(memory, image.UsedRange, image.DolRange, image.RelRange, image.Sha256 + "+variants",
            image.MemoryBase);
    }

    /// <summary>
    /// Translates every spec function from the patched image. <paramref name="callContracts"/> are the base
    /// translation's interprocedural contracts: calls to unpatched functions use them (their translated bodies are the
    /// base ones); calls to patched functions get none (a full register synchronization), since the code that runs
    /// there is the mod's. Leaf inlining is off: an inlined copy would outlive a later patch of the leaf.
    /// </summary>
    internal static void Translate(
        Spec spec,
        ProgramImage baseImage,
        IGuestFunctionAbiProvider abiProvider,
        IReadOnlySet<uint> knownFunctionEntryPoints,
        IReadOnlyDictionary<uint, GuestAbiContract> callContracts,
        bool allowUnsupportedInstructions,
        string outDir,
        int threads,
        uint? sda1Base = null,
        uint? sda2Base = null)
    {
        Directory.CreateDirectory(outDir);
        foreach (var stale in Directory.EnumerateFiles(outDir, "func_*_pp.cpp")) File.Delete(stale);
        var overlay = BuildOverlay(baseImage, spec);
        var translator = new FunctionTranslator(overlay, abiProvider);
        var patched = spec.Functions.ToHashSet();
        // The spec's functions inside its regions are entries too: a mod's code linked in place (a module or plugin) has
        // no base translation, and a branch from one of its functions (or from a hook trampoline) to another is a tail
        // call, not a jump into the caller's body. Calls to them are emitted as InvokeDirectCpu<target>, which reaches
        // their variants at run time (abi_bridge.h, RunUntranslatedOrFail). Entries inside base functions (a plugin
        // loader's continuations at a function's second instruction) stay out: a hook body's branch back there is the
        // rest of the hooked function, decoded with it as before.
        var regionList = spec.Regions.Select(static region => region.Range).ToArray();
        var known = new HashSet<uint>(knownFunctionEntryPoints);
        known.UnionWith(patched.Where(a => regionList.Any(r => a >= r.Start && a < r.End)));
        knownFunctionEntryPoints = known;
        var contracts = callContracts
            .Where(pair => !patched.Contains(pair.Key))
            .ToDictionary(static pair => pair.Key, static pair => pair.Value);
        var regions = spec.Regions.Select(static region => region.Range).ToArray();
        var results = new ConcurrentDictionary<uint, (string Symbol, string Source, List<(uint Start, uint[] Words)> Guards)>();
        var rejected = new ConcurrentDictionary<uint, string>();

        Parallel.ForEach(spec.Functions, new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, threads) }, address =>
        {
            var symbol = $"func_{address:X8}_pp";
            var options = new TranslationOptions(
                PreferredName: symbol,
                AllowUnsupportedInstructions: allowUnsupportedInstructions,
                KnownFunctionEntryPoints: knownFunctionEntryPoints,
                GuestAbiContracts: contracts,
                EnableLeafAbiSpillElision: true,
                EnableLeafInlining: false,
                InlineCodeRegions: regions,
                Sda1Base: sda1Base,
                Sda2Base: sda2Base);
            FunctionTranslationResult result;
            try
            {
                result = translator.Translate(address, options);
            }
            catch (Exception ex)
            {
                rejected[address] = $"translation failed: {ex.GetType().Name}: {ex.Message.Split('\n')[0]}";
                return;
            }

            var decoded = result.Instructions.Select(static i => i.Address).ToHashSet();
            bool InRegion(uint a) => regions.Any(r => a >= r.Start && a < r.End);
            // The entry and the unconditional branches it chains through (a plugin loader's hook: the function's first
            // instruction branches to a trampoline that branches to the replacement): nothing on that path changes lr.
            var byAddress = result.Instructions.ToDictionary(static i => i.Address);
            var entryChain = new HashSet<uint>();
            for (var at = address; byAddress.TryGetValue(at, out var link) && entryChain.Add(at);)
            {
                if (!link.IsUnconditionalBranch || link.BranchTargets.Count != 1) break;
                at = link.BranchTargets[0];
            }
            foreach (var ins in result.Instructions)
            {
                // The runtime's interpreter does more at some PCs and around some calls than the guest code says (the
                // rollback main loop of hle/brawl_rollback.cpp): a function reaching them stays interpreted.
                if (spec.InterpreterOnlyPcs.Contains(ins.Address))
                {
                    rejected[address] = $"holds 0x{ins.Address:X8}, which only the interpreter handles";
                    return;
                }
                if (ins.IsCall && ins.BranchTargets.Any(spec.InterpreterOnlyCallTargets.Contains))
                {
                    rejected[address] = $"calls 0x{ins.BranchTargets[0]:X8} at 0x{ins.Address:X8}, which only the interpreter wraps";
                    return;
                }
                // A call into a hook region runs there in the interpreter until a blr returns to the call's lr. Code
                // that is no subroutine (it jumps through lr some other way, such as the "bl to a blrl" address
                // trick) would never come back, and the interpreter would carry on with the rest of the game inside
                // that call: keep such functions interpreted.
                if (ins.IsCall && ins.BranchTargets.Count == 1 && InRegion(ins.BranchTargets[0]) &&
                    !LooksLikeSubroutine(overlay, ins.BranchTargets[0], known.Contains))
                {
                    rejected[address] = $"calls 0x{ins.BranchTargets[0]:X8} at 0x{ins.Address:X8}, which is no plain subroutine";
                    return;
                }
                if (ins.IsCall) continue;
                foreach (var target in ins.BranchTargets)
                {
                    if (decoded.Contains(target)) continue;
                    if (!ins.IsConditionalBranch && knownFunctionEntryPoints.Contains(target)) continue;  // tail call
                    // A branch on the entry chain (a mod replacing the whole function) leaves lr as the caller set
                    // it, so a tail call to untranslated code returns exactly where the guest's does.
                    if (!ins.IsConditionalBranch && entryChain.Contains(ins.Address)) continue;
                    rejected[address] = $"branch at 0x{ins.Address:X8} leaves the function for 0x{target:X8}";
                    return;
                }
                // A bctr without recognized targets is an indirect tail call. That is right for a function pointer
                // (a virtual call: "lwz r12,d(r12); mtctr r12; bctr"), but a target computed from immediates jumps back
                // into some function's middle (fn_801C8010's hook picks 0x801C80A8 or 0x801C80B0), and an indexed load
                // is a switch table the variant failed to recognize (0x8078249C's hook moved the switch's bounds check
                // into its body): the interpreter would enter a case label with a stale lr and never come back.
                if (ins.BranchTargets.Count == 0 &&
                    (ins.Mnemonic.Equals("bctr", StringComparison.OrdinalIgnoreCase) ||
                     ins.Mnemonic.StartsWith("bcctr", StringComparison.OrdinalIgnoreCase)) &&
                    !CtrLoadedFromMemory(byAddress, ins.Address))
                {
                    rejected[address] = $"computed jump at 0x{ins.Address:X8}";
                    return;
                }
            }

            if (result.Metrics.UnsupportedInstructionCount > 0 && !allowUnsupportedInstructions)
            {
                rejected[address] = "unsupported instructions";
                return;
            }

            // The generator follows a body not named func_<address> with that alias calling it
            // (CxxLinearCodeGenerator); the base translation owns func_<address>.
            var nl = Environment.NewLine;
            var alias = $"extern \"C\" void func_{address:X8}(CpuContext* MKW_RESTRICT ctx){nl}{{{nl}    {symbol}(ctx);{nl}}}{nl}";
            var source = result.CxxCode.Replace(alias, string.Empty, StringComparison.Ordinal);
            if (source.Contains($"void func_{address:X8}(", StringComparison.Ordinal))
            {
                rejected[address] = "its source defines the base symbol";
                return;
            }

            // Guarded words: every decoded instruction, nothing else. A word no decoded path executes cannot change what
            // the variant does, and Project+ keeps data in such words inside the functions it hooks and rewrites them
            // during play (fn_800B9984's dead code; 0x8078249C's hook stores into 0x807824D4, which its own patch
            // branches over): guarding them sent those functions back to the interpreter at every store.
            var guarded = new SortedSet<uint>(decoded);
            var guards = new List<(uint Start, uint[] Words)>();
            var run = new List<uint>();
            var runStart = 0u;
            var last = 0u;
            foreach (var a in guarded)
            {
                if (run.Count > 0 && a != last + 4)
                {
                    guards.Add((runStart, run.ToArray()));
                    run.Clear();
                }
                if (run.Count == 0) runStart = a;
                run.Add(BinaryPrimitives.ReadUInt32BigEndian(overlay.Memory.AsSpan(overlay.GetOffset(a, 4), 4)));
                last = a;
            }
            if (run.Count > 0) guards.Add((runStart, run.ToArray()));
            results[address] = (symbol, source, guards);
        });

        var manifest = new StringBuilder();
        manifest.AppendLine("# code variants: D start end (memory outside translated code they depend on: the runtime write-traps it),");
        manifest.AppendLine("# V address symbol, then G start count words (the guest words it was translated from)");
        foreach (var region in regions) manifest.AppendLine($"D {region.Start:X8} {region.End:X8}");
        foreach (var (address, (symbol, source, guards)) in results.OrderBy(static pair => pair.Key))
        {
            File.WriteAllText(Path.Combine(outDir, symbol + ".cpp"), source);
            manifest.AppendLine($"V {address:X8} {symbol}");
            foreach (var (start, words) in guards)
            {
                manifest.Append($"G {start:X8} {words.Length}");
                foreach (var word in words) manifest.Append($" {word:X8}");
                manifest.AppendLine();
            }
        }

        File.WriteAllText(Path.Combine(outDir, "variants.txt"), manifest.ToString());
        File.WriteAllText(Path.Combine(outDir, "rejected.txt"), string.Concat(rejected
            .OrderBy(static pair => pair.Key)
            .Select(static pair => $"{pair.Key:X8} {pair.Value}\n")));
        Console.WriteLine(
            $"[translator] Code variants: {results.Count:N0} translated, {rejected.Count:N0} left to the interpreter " +
            $"({outDir}).");
    }
}
