using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Translator.Core.Analysis;
using Translator.Core.Ir;

namespace Translator.Core.CodeGen;

public sealed partial class CxxLinearCodeGenerator
{
    /// <summary>
    /// MKW_SINGLE_C=0 turns this off. fmuls and the fused single-precision family round their C operand to a 25-bit
    /// significand first (runtime isa/ppc_isa_float.h); that rounding is the identity on a value that is already
    /// single-precision, so where C provably is one the emitter calls the variants without it
    /// (PpcFmulsSingleCInline, PPC_FmaddsSingleC, ...).
    /// </summary>
    private static readonly bool SingleCOperands = Environment.GetEnvironmentVariable("MKW_SINGLE_C") != "0";

    /// <summary>The body being emitted: its FPR SSA values proven single-precision (<see cref="ComputeSingleFprValues"/>).</summary>
    [ThreadStatic]
    private static HashSet<string>? _singleFprValues;

    /// <summary>
    /// The body being emitted stores psq through PPC_PsqStSegmentInline (MKW_LEAN_RANGES), not PPC_PsqStResolvedInline:
    /// off where a state-free clone is derived from the body's text, whose rewrite knows only the resolved spelling.
    /// </summary>
    [ThreadStatic]
    private static bool _segmentPsqStores;

    private static readonly HashSet<string> SinglePrecisionProducers = new(StringComparer.OrdinalIgnoreCase)
    {
        "PPC_Fadds", "PPC_Fsubs", "PPC_Fmuls", "PPC_Fdivs", "PPC_Fmadds", "PPC_Fmsubs", "PPC_Fnmadds", "PPC_Fnmsubs",
        "PPC_FaddsNoNi", "PPC_FsubsNoNi", "PPC_FmulsNoNi", "PPC_FdivsNoNi", "PPC_Fres"
    };

    private static readonly HashSet<string> FusedSingleTargets = new(StringComparer.OrdinalIgnoreCase)
    {
        "PPC_Fmadds", "PPC_Fmsubs", "PPC_Fnmadds", "PPC_Fnmsubs"
    };

    /// <summary>The body being emitted: FPR SSA values whose C++ local still holds them here (<see cref="FprIntactness"/>).</summary>
    [ThreadStatic]
    private static HashSet<string>? _intactFprs;

    /// <summary>
    /// Whether an instruction may leave a C++ FPR local holding something its SSA name does not describe. The IR does
    /// not version the registers a guest call clobbers: after `bl`, the same name still stands for the value before the
    /// call while the emitted code reloads the local from CpuContext (the callee's f1, say). Guest and indirect calls
    /// (inline guest thunks included) and helpers that may write an FPR or need a complete context are such points.
    /// </summary>
    private static bool IsFprBarrier(IrInstruction instruction) => instruction switch
    {
        IrIndirectCall or IrIndirectJump => true,
        IrCall call => !IsPureFprHelperCall(call),
        _ => false
    };

    private static bool IsPureFprHelperCall(IrCall call)
    {
        if (string.IsNullOrWhiteSpace(call.Target) || TryParseAddress(call.Target, out _) || IsGuestCallTarget(call.Target))
            return false;
        if (InlineScalarCallSpecs.ContainsKey(call.Target) || InlinePairedCallSpecs.ContainsKey(call.Target))
            return true;
        var effect = Translator.Core.Analysis.GuestEabi.HelperEffect(call);
        return effect.FprWriteMask == 0 && effect.BoundaryFlags == GuestCallBoundaryFlags.None;
    }

    /// <summary>
    /// Which FPR SSA values are intact (their C++ local holds them) at each block's entry: defined once, and no
    /// <see cref="IsFprBarrier"/> on any path since the definition. A must analysis (intersection over predecessors);
    /// a phi is intact where each of its sources is intact at the end of its predecessor.
    /// </summary>
    private sealed class FprIntactness
    {
        public required Dictionary<string, HashSet<string>> BlockIn { get; init; }
        public required HashSet<string> IntactPhis { get; init; }
        public required HashSet<string> SingleDefinitionFprs { get; init; }

        /// <summary>Advances <paramref name="intact"/> over <paramref name="instruction"/>.</summary>
        public void Step(HashSet<string> intact, IrInstruction instruction)
        {
            if (IsFprBarrier(instruction))
                intact.Clear();
            if (instruction is IrPhi phi)
            {
                if (IntactPhis.Contains(phi.Destination)) intact.Add(phi.Destination);
                else intact.Remove(phi.Destination);
                return;
            }
            foreach (var defined in IrRegisterDataFlow.Definitions(instruction))
                if (SingleDefinitionFprs.Contains(defined))
                    intact.Add(defined);
        }

        public HashSet<string> EntryOf(string label) =>
            BlockIn.TryGetValue(label, out var entry) ? new HashSet<string>(entry, StringComparer.Ordinal)
                                                      : new HashSet<string>(StringComparer.Ordinal);
    }

    private static FprIntactness ComputeFprIntactness(IrFunction function)
    {
        var definitions = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var block in function.Blocks)
            foreach (var instruction in block.Instructions)
                foreach (var defined in IrRegisterDataFlow.Definitions(instruction))
                    definitions[defined] = definitions.GetValueOrDefault(defined) + 1;
        var singleDefinition = new HashSet<string>(
            definitions.Where(pair => pair.Value == 1 && IsFloatRegister(pair.Key)).Select(pair => pair.Key),
            StringComparer.Ordinal);
        var cfg = Translator.Core.Analysis.Ssa.IrCfg.Build(function);
        var blockIn = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        var blockOut = new Dictionary<string, HashSet<string>?>(StringComparer.Ordinal);  // null: not reached yet (top)
        var intactPhis = new HashSet<string>(singleDefinition, StringComparer.Ordinal);  // optimistic, shrinks
        var analysis = new FprIntactness { BlockIn = blockIn, IntactPhis = intactPhis, SingleDefinitionFprs = singleDefinition };
        foreach (var block in function.Blocks) blockOut[block.Label] = null;
        bool changed;
        var rounds = 0;
        do
        {
            changed = false;
            foreach (var block in function.Blocks)
            {
                HashSet<string>? entry = null;
                if (block.Label != function.EntryLabel)
                {
                    foreach (var pred in cfg.Predecessors(block.Label))
                    {
                        if (blockOut.GetValueOrDefault(pred) is not { } predOut) continue;
                        if (entry is null) entry = new HashSet<string>(predOut, StringComparer.Ordinal);
                        else entry.IntersectWith(predOut);
                    }
                    if (entry is null) continue;  // not reached (yet)
                }
                entry ??= new HashSet<string>(StringComparer.Ordinal);
                foreach (var instruction in block.Instructions)
                {
                    if (instruction is not IrPhi phi || !intactPhis.Contains(phi.Destination)) continue;
                    foreach (var (pred, source) in phi.Sources)
                        if (blockOut.GetValueOrDefault(pred) is { } predOut && !predOut.Contains(source))
                        {
                            intactPhis.Remove(phi.Destination);
                            changed = true;
                            break;
                        }
                }
                blockIn[block.Label] = entry;
                var running = new HashSet<string>(entry, StringComparer.Ordinal);
                foreach (var instruction in block.Instructions)
                    analysis.Step(running, instruction);
                var previous = blockOut[block.Label];
                if (previous is null || !previous.SetEquals(running))
                {
                    blockOut[block.Label] = running;
                    changed = true;
                }
            }
        } while (changed && ++rounds < 1000);
        if (changed)
        {
            // Did not settle (never seen): nothing is intact anywhere.
            foreach (var label in blockIn.Keys.ToList()) blockIn[label] = new HashSet<string>(StringComparer.Ordinal);
            intactPhis.Clear();
        }
        return analysis;
    }

    /// <summary>
    /// The FPR SSA values that hold a single-precision value wherever they are intact: the results of lfs (a 4-byte
    /// load into an FPR, alone or as an element of a load pair: the exact widening of a single), of frsp and of the
    /// single-precision arithmetic (each rounds its result to float), and fmr / fneg / fabs / fsel / phis of such
    /// values whose sources are intact there (a move or a sign change of a single is one). The greatest fixed point:
    /// every member's rule holds with the other members, so cycles through phis qualify only when every value entering
    /// them does. A use must also find the value intact (<see cref="_intactFprs"/>).
    /// </summary>
    private static HashSet<string> ComputeSingleFprValues(IrFunction function, FprIntactness intactness)
    {
        var unconditional = new HashSet<string>(StringComparer.Ordinal);
        var conditional = new Dictionary<string, string[]>(StringComparer.Ordinal);
        var brokenSources = new HashSet<string>(StringComparer.Ordinal);  // a conditional member with a source not intact at its definition
        foreach (var block in function.Blocks)
        {
            var intact = intactness.EntryOf(block.Label);
            foreach (var instruction in block.Instructions)
            {
                string[]? sources = null;
                switch (instruction)
                {
                    case IrCall { Destination: { } dest } call when IsFloatRegister(dest):
                        if (SinglePrecisionProducers.Contains(call.Target))
                            unconditional.Add(dest);
                        else if (call.Target.Equals("PPC_Fsel", StringComparison.OrdinalIgnoreCase) &&
                                 call.Arguments.Count == 3 &&
                                 call.Arguments[1] is { Kind: "register", RegisterName: { } b } &&
                                 call.Arguments[2] is { Kind: "register", RegisterName: { } c })
                            conditional[dest] = sources = new[] { b, c };
                        break;
                    case IrLoad { SizeBytes: 4 } load when IsFloatRegister(load.Destination):
                        unconditional.Add(load.Destination);
                        break;
                    case IrResolvedLoad { SizeBytes: 4 } rload when IsFloatRegister(rload.Destination):
                        unconditional.Add(rload.Destination);
                        break;
                    // A pair of 4-byte loads (GuestMemoryRangeLowering): each float element is an lfs.
                    case IrResolvedLoadPair { ElementSizeBytes: 4 } pair:
                        if (IsFloatRegister(pair.FirstDestination)) unconditional.Add(pair.FirstDestination);
                        if (IsFloatRegister(pair.SecondDestination)) unconditional.Add(pair.SecondDestination);
                        break;
                    case IrBinary { Op: "frsp" } frsp when IsFloatRegister(frsp.Destination):
                        unconditional.Add(frsp.Destination);
                        break;
                    case IrBinary { Op: "fneg" or "fabs", Left: { Kind: "register", RegisterName: { } source } } sign
                        when IsFloatRegister(sign.Destination) && IsFloatRegister(source):
                        conditional[sign.Destination] = sources = new[] { source };
                        break;
                    case IrAssign { Value: { Kind: "register", RegisterName: { } source } } move
                        when IsFloatRegister(move.Destination) && IsFloatRegister(source):
                        conditional[move.Destination] = sources = new[] { source };
                        break;
                    case IrPhi phi when IsFloatRegister(phi.Destination):
                        conditional[phi.Destination] = phi.Sources.Values.ToArray();
                        if (!intactness.IntactPhis.Contains(phi.Destination)) brokenSources.Add(phi.Destination);
                        break;
                }
                if (sources is not null && !sources.All(intact.Contains))
                    brokenSources.Add(IrRegisterDataFlow.Definitions(instruction).First());
                intactness.Step(intact, instruction);
            }
        }

        var singles = new HashSet<string>(unconditional, StringComparer.Ordinal);
        singles.UnionWith(conditional.Keys);
        singles.RemoveWhere(name => !intactness.SingleDefinitionFprs.Contains(name) || brokenSources.Contains(name));
        bool changed;
        do
        {
            changed = false;
            foreach (var (name, sources) in conditional)
                if (singles.Contains(name) && !sources.All(singles.Contains))
                {
                    singles.Remove(name);
                    changed = true;
                }
        } while (changed);
        return singles;
    }

    /// <summary>
    /// MKW_RAW_SINGLE_STORES=0 turns this off. An stfs narrows its FPR with Gekko's truncating rule
    /// (ConvertPpcDoubleToSingleBits), which is the exact inverse of lfs's widening for every one of the 2^32 single
    /// patterns (normals, subnormals, zeros, infinities and NaNs, signalling ones included) and of the host's widening
    /// of a float that single-precision arithmetic produced. So where the stored value is such a single, its 32 bits are
    /// already known before the widening: the word lfs loaded, or the float the arithmetic rounded to. The emitter keeps
    /// them in a uint32_t local (<see cref="RawSingleBitsLocal"/>) beside the FPR and stores that word with the integer
    /// store of the same family: no narrowing, and for a copy (lfs then stfs) no widening either once the FPR is dead.
    /// Only within one block and with no <see cref="IsFprBarrier"/> between the definition and the store, so the FPR's
    /// local provably still holds that single (no call reloaded it, no edge normalized its representation).
    /// </summary>
    private static readonly bool RawSingleStores = Environment.GetEnvironmentVariable("MKW_RAW_SINGLE_STORES") != "0";

    /// <summary>
    /// MKW_SINGLE_ADD_CAST=0 turns this off. fadds/fsubs of two single-precision operands and frsp of one round with a
    /// plain double-to-float conversion instead of PpcForceSingleValueInline. The NI mask flushes a pre-round magnitude
    /// below 2^-126; the host conversion under FTZ (set exactly when NI is) flushes a tiny result itself, so the two
    /// differ only on values in [2^-126 - 2^-150, 2^-126) that need rounding. A sum or difference of two singles, and a
    /// single itself, is an exact multiple of 2^-149, and every such multiple below 2^-126 is an exact single subnormal:
    /// none needs rounding there. (Products and quotients of singles can land in that window, so fmuls, fdivs and the
    /// fused family keep PpcForceSingleValueInline.)
    /// </summary>
    private static readonly bool SingleAddCast = Environment.GetEnvironmentVariable("MKW_SINGLE_ADD_CAST") != "0";

    /// <summary>The body being emitted: the 4-byte FPR stores that store raw single bits, per element (null: narrow).</summary>
    [ThreadStatic]
    private static Dictionary<IrInstruction, string?[]>? _rawSingleStores;

    /// <summary>The body being emitted: the FPR definitions that assign a raw single bits local.</summary>
    [ThreadStatic]
    private static HashSet<string>? _rawSingleRoots;

    private static string RawSingleBitsLocal(string root) => "fbits_" + root;

    /// <summary>
    /// The rounded single-precision scalar arithmetic whose emission computes a float and widens it
    /// (<see cref="ScalarInlineKind.RoundedHelper"/>: fadds, fsubs, fdivs).
    /// </summary>
    private static bool IsRoundedSingleCall(IrCall call) =>
        !string.IsNullOrWhiteSpace(call.Destination) && IsFloatRegister(call.Destination) &&
        InlineScalarCallSpecs.TryGetValue(call.Target, out var spec) && spec.Kind == ScalarInlineKind.RoundedHelper &&
        !spec.NoNiVariant && call.Arguments.Count == 2;

    /// <summary>
    /// The 4-byte FPR stores whose value's single bits are available as an integer (see <see cref="RawSingleStores"/>):
    /// the value is defined in the same block by a 4-byte FPR load (lfs, a resolved or segment lfs, an element of a lean
    /// segment pair), by fadds/fsubs/fdivs or frsp, or by an fmr chain from one, and nothing in between is a barrier.
    /// </summary>
    private static (Dictionary<IrInstruction, string?[]> Stores, HashSet<string> Roots) ComputeRawSingleStores(
        IrFunction function, FprIntactness intactness)
    {
        var stores = new Dictionary<IrInstruction, string?[]>(ReferenceEqualityComparer.Instance);
        var roots = new HashSet<string>(StringComparer.Ordinal);
        if (!RawSingleStores)
            return (stores, roots);
        foreach (var block in function.Blocks)
        {
            var intact = intactness.EntryOf(block.Label);
            // Names defined in this block whose single bits are known: the root each traces back to.
            var rootOf = new Dictionary<string, string>(StringComparer.Ordinal);
            void Root(string name)
            {
                if (IsFloatRegister(name) && intactness.SingleDefinitionFprs.Contains(name))
                    rootOf[name] = name;
            }
            string? Bits(IrValue value, int sizeBytes)
            {
                if (sizeBytes != 4 || value is not { Kind: "register", RegisterName: { } name } || !intact.Contains(name) ||
                    !rootOf.TryGetValue(name, out var root))
                    return null;
                roots.Add(root);
                return RawSingleBitsLocal(root);
            }
            foreach (var instruction in block.Instructions)
            {
                switch (instruction)
                {
                    case IrStore store when Bits(store.Source, store.SizeBytes) is { } bits:
                        stores[store] = new string?[] { bits };
                        break;
                    case IrResolvedStore store when Bits(store.Source, store.SizeBytes) is { } bits:
                        stores[store] = new string?[] { bits };
                        break;
                    case IrResolvedStorePair pair:
                        var first = Bits(pair.FirstSource, pair.ElementSizeBytes);
                        var second = Bits(pair.SecondSource, pair.ElementSizeBytes);
                        if (first is not null || second is not null)
                            stores[pair] = new[] { first, second };
                        break;
                }
                // The definition's own rule, checked against what was intact before it.
                switch (instruction)
                {
                    case IrAssign { Value: { Kind: "register", RegisterName: { } source } } move
                        when IsFloatRegister(move.Destination) && intact.Contains(source) &&
                             rootOf.TryGetValue(source, out var moveRoot) &&
                             intactness.SingleDefinitionFprs.Contains(move.Destination):
                        rootOf[move.Destination] = moveRoot;
                        break;
                }
                if (IsFprBarrier(instruction))
                    rootOf.Clear();
                switch (instruction)
                {
                    case IrLoad { SizeBytes: 4 } load:
                        Root(load.Destination);
                        break;
                    case IrResolvedLoad { SizeBytes: 4 } load:
                        Root(load.Destination);
                        break;
                    case IrResolvedLoadPair { ElementSizeBytes: 4 } pair when LeanRanges:
                        Root(pair.FirstDestination);
                        Root(pair.SecondDestination);
                        break;
                    case IrCall call when IsRoundedSingleCall(call):
                        Root(call.Destination);
                        break;
                    case IrBinary { Op: "frsp" } frsp:
                        Root(frsp.Destination);
                        break;
                }
                intactness.Step(intact, instruction);
            }
        }
        return (stores, roots);
    }

    /// <summary>The raw single bits local element <paramref name="element"/> of 4-byte FPR store <paramref name="store"/> stores, if any.</summary>
    private static bool TryGetRawStoreBits(IrInstruction store, int element, out string local)
    {
        local = string.Empty;
        if (_rawSingleStores?.TryGetValue(store, out var elements) != true || element >= elements!.Length ||
            elements[element] is not { } bits)
            return false;
        local = bits;
        return true;
    }

    /// <summary>Whether FPR SSA value <paramref name="destination"/> assigns a raw single bits local where it is defined.</summary>
    private static bool TryGetRawSingleRoot(string destination, out string local)
    {
        local = string.Empty;
        if (_rawSingleRoots?.Contains(destination) != true)
            return false;
        local = RawSingleBitsLocal(destination);
        return true;
    }

    /// <summary>
    /// Every raw single bits local a store reads must have been assigned by its definition's emission: a definition
    /// emitted by a path that does not know about it would leave the store writing a stale word. Fail the translation
    /// instead.
    /// </summary>
    private static void VerifyRawSingleBitsAssigned(StringBuilder body, HashSet<string> roots, string function)
    {
        if (roots.Count == 0)
            return;
        var text = body.ToString();
        foreach (var root in roots)
        {
            var local = RawSingleBitsLocal(root);
            var assigned = text.Contains(local + " = ", StringComparison.Ordinal);
            var mentions = 0;
            for (var at = text.IndexOf(local, StringComparison.Ordinal); at >= 0;
                 at = text.IndexOf(local, at + local.Length, StringComparison.Ordinal))
            {
                var end = at + local.Length;
                if (end < text.Length && (char.IsLetterOrDigit(text[end]) || text[end] == '_'))
                    continue;  // a longer name (fbits_f1_1 inside fbits_f1_12)
                mentions++;
            }
            // The declaration is one mention; any other without an assignment is a read of a word never written.
            if (!assigned && mentions > 1)
                throw new InvalidOperationException(
                    $"{function}: raw single bits local {local} is read but its definition never assigns it");
        }
    }

    /// <summary>
    /// Assigns a single-precision result given as a float expression: through its raw bits local when it has one,
    /// widened into the FPR either way.
    /// </summary>
    private static void EmitSingleResult(StringBuilder sb, string pad, string destination, string destinationExpression,
        string floatExpression)
    {
        if (TryGetRawSingleRoot(destination, out var local))
        {
            sb.AppendLine($"{pad}{local} = PpcBitCastToU32Inline({floatExpression});");
            sb.AppendLine($"{pad}{destinationExpression} = static_cast<double>(PpcBitCastToFloatInline({local}));");
            return;
        }
        sb.AppendLine($"{pad}{destinationExpression} = static_cast<double>({floatExpression});");
    }

    /// <summary>The float a 4-byte load's word widens to, assigned through its raw bits local when it has one.</summary>
    private static void EmitSingleLoad(StringBuilder sb, string pad, string destination, string destinationExpression,
        string wordExpression)
    {
        if (TryGetRawSingleRoot(destination, out var local))
        {
            sb.AppendLine($"{pad}{local} = {wordExpression};");
            sb.AppendLine($"{pad}{destinationExpression} = PpcBitCastToDoubleInline(PpcConvertToDoubleBitsInline({local}));");
            return;
        }
        sb.AppendLine($"{pad}{destinationExpression} = PpcBitCastToDoubleInline(PpcConvertToDoubleBitsInline({wordExpression}));");
    }

    /// <summary>
    /// fadds/fsubs (two single-precision operands) and frsp (one): the plain conversion (see <see cref="SingleAddCast"/>);
    /// otherwise PpcForceSingleValueInline.
    /// </summary>
    private static string RoundToSingleExpression(string doubleExpression, bool operandsSingle) =>
        SingleAddCast && operandsSingle
            ? $"static_cast<float>({doubleExpression})"
            : $"PpcForceSingleValueInline({doubleExpression})";

    /// <summary>
    /// Whether <paramref name="value"/> is an FPR operand holding a single-precision value here (no MKW_SINGLE_C gate):
    /// a paired register's PS0, or a member of <see cref="_singleFprValues"/> still intact.
    /// </summary>
    private static bool IsSingleFprValue(IrValue value, IReadOnlyDictionary<string, bool> localPaired)
    {
        if (value is not { Kind: "register", RegisterName: { } name } || !IsFloatRegister(name))
            return false;
        if (localPaired.TryGetValue(GetRegisterBaseName(name), out var paired) && paired)
            return true;
        return _singleFprValues?.Contains(name) == true && _intactFprs?.Contains(name) == true;
    }

    /// <summary>Whether <paramref name="value"/> is an FPR operand holding a single-precision value here.</summary>
    private static bool IsSingleFprOperand(IrValue value, IReadOnlyDictionary<string, bool> localPaired)
    {
        if (!SingleCOperands || value is not { Kind: "register", RegisterName: { } name } || !IsFloatRegister(name))
            return false;
        // A paired register's scalar view is its PS0, a float (ToScalarFloatExpression: PPC_PsToScalarInline).
        if (localPaired.TryGetValue(GetRegisterBaseName(name), out var paired) && paired)
            return true;
        return _singleFprValues?.Contains(name) == true && _intactFprs?.Contains(name) == true;
    }
}
