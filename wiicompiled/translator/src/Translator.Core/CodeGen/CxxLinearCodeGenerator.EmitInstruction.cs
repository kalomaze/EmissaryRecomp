using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Translator.Core.Analysis.Ssa;
using Translator.Core.Ir;
using Translator.Core.Analysis.Representation;
using Translator.Core.Analysis;
using Translator.Core.Representation;

namespace Translator.Core.CodeGen;

public sealed partial class CxxLinearCodeGenerator
{
    private static void EmitInstruction(string currentLabel, int directCallOrdinal, IrInstruction ins, StringBuilder sb, int bufferBaseLength, int indent,
        IrCfg cfg, Dictionary<string, string> labelNames, RepresentationEnvironment types, FunctionAbiClassification signature,
        Dictionary<string, bool> localPaired, IGuestFunctionAbiProvider guestAbiProvider,
        IReadOnlyDictionary<string, uint> knownConstants,
        IReadOnlyDictionary<string, uint> localConstants,
        LinkedAddressRemap? linkedAddressRemap,
        IReadOnlySet<uint> nonReturningCallTargets,
        IReadOnlySet<uint> lrContinuationCallTargets,
        StackAddressFacts stackFacts,
        bool inlineGuestThunkStackBase,
        uint? localFallthroughLr,
        IReadOnlyDictionary<uint, GuestAbiContract> guestAbiContracts,
        IReadOnlyDictionary<uint, GuestAbiContract> stateFreeAbiContracts,
        IReadOnlyDictionary<uint, string> stateFreeCallSymbols,
        IReadOnlyDictionary<GuestStateFreeCallSiteKey, GuestStateFreeCallVariant> stateFreeCallSiteVariants,
        IReadOnlySet<uint> modOverridableCallTargets,
        bool shareLrContinuationDispatch)
    {
        if (ins is IrPhi)
        {
            return; // Phi nodes are eliminated before codegen.
        }

        var pad = IndentPad(indent);
        switch (ins)
        {
            case IrAssign assign:
                {
                    var dest = ToDestination(assign.Destination, types);
                    var src = ToRegisterAssignmentExpression(assign.Destination, assign.Value, types);
                    sb.AppendLine($"{pad}{dest} = {src};");
                    if (IsFloatRegister(assign.Destination))
                    {
                        EmitRawSingleCopy(sb, pad, assign.Destination);
                        var paired = false;
                        if (assign.Value.Kind == "register" &&
                            assign.Value.RegisterName != null &&
                            IsFloatRegister(assign.Value.RegisterName))
                        {
                            var srcBase = GetRegisterBaseName(assign.Value.RegisterName);
                            paired = localPaired.TryGetValue(srcBase, out var isPaired) && isPaired;
                        }
                        localPaired[GetRegisterBaseName(assign.Destination)] = paired;
                    }
                }
                break;
            case IrBinary bin:
                {
                    var dest = ToDestination(bin.Destination, types);
                    var scalarFloatOperation = IsFloatRegister(bin.Destination);
                    string expr;
                    if (!scalarFloatOperation && TryFoldSdaBinary(bin, out var foldedSda))
                    {
                        expr = foldedSda;
                    }
                    else if (scalarFloatOperation && bin.Op == "frsp")
                    {
                        // frsp: a single operand rounds by the plain conversion, and the result's bits may be kept for
                        // a store (CxxLinearCodeGenerator.SingleFpr.cs).
                        var source = ToScalarFloatExpression(bin.Left, types, localPaired);
                        EmitSingleResult(sb, pad, bin.Destination, dest,
                            RoundToSingleExpression(source, IsSingleFprValue(bin.Left, localPaired)));
                        localPaired[GetRegisterBaseName(bin.Destination)] = false;
                        break;
                    }
                    else
                    {
                        var left = scalarFloatOperation
                            ? ToScalarFloatExpression(bin.Left, types, localPaired)
                            : ToExpression(bin.Left, types);
                        var right = scalarFloatOperation
                            ? ToScalarFloatExpression(bin.Right, types, localPaired)
                            : ToExpression(bin.Right, types);
                        expr = BinaryExpressionWithOperands(bin, left, right);
                    }
                    sb.AppendLine($"{pad}{dest} = {expr};");
                    if (IsFloatRegister(bin.Destination))
                    {
                        localPaired[GetRegisterBaseName(bin.Destination)] = false;
                    }
                }
                break;
            case IrLoad load:
                {
                    var dest = ToDestination(load.Destination, types);
                    var destType = types.Get(load.Destination);
                    var isFloatDest = (destType is ValueRepresentation pt && pt.IsFloat) ||
                                      IsFloatRegister(load.Destination);

                    if (TryEmitStackLoad(load, dest, isFloatDest, sb, pad, types, stackFacts))
                    {
                    }
                    else if (isFloatDest && load.SizeBytes == 4 && TryGetRawSingleRoot(load.Destination, out _))
                    {
                        // lfs is the 32-bit load and the exact widening (FlatReadFloat32Body); the word is kept for a store.
                        EmitSingleLoad(sb, pad, load.Destination, dest,
                            $"{GuestLoadHelper(4, isFloat: false)}({RemapAddress(load.Address, types, localConstants, linkedAddressRemap)})");
                    }
                    else if (isFloatDest && load.SizeBytes == 4)
                    {
                        sb.AppendLine($"{pad}{dest} = {GuestLoadHelper(4, isFloat: true)}({RemapAddress(load.Address, types, localConstants, linkedAddressRemap)});");
                    }
                    else if (isFloatDest && load.SizeBytes == 8)
                    {
                        sb.AppendLine($"{pad}{dest} = {GuestLoadHelper(8, isFloat: true)}({RemapAddress(load.Address, types, localConstants, linkedAddressRemap)});");
                    }
                    else
                    {
                        sb.AppendLine($"{pad}{dest} = {GuestLoadHelper(load.SizeBytes, isFloat: false)}({RemapAddress(load.Address, types, localConstants, linkedAddressRemap)});");
                    }
                    if (IsFloatRegister(load.Destination))
                    {
                        localPaired[GetRegisterBaseName(load.Destination)] = false;
                    }
                }
                break;
            case IrStore store:
                {
                    var srcExpr = ToExpression(store.Source, types);
                    var isFloatStore = store.Source.Kind == "register" && 
                                       store.Source.RegisterName != null &&
                                       IsFloatRegister(store.Source.RegisterName);
                    if (isFloatStore &&
                        store.SizeBytes == 4 &&
                        store.Source.RegisterName != null &&
                        localPaired.TryGetValue(GetRegisterBaseName(store.Source.RegisterName), out var sourceIsPaired) &&
                        sourceIsPaired)
                    {
                        srcExpr = $"PPC_PsToScalarInline({srcExpr})";
                    }

                    if (TryEmitKnownGpuFifoStore(store, srcExpr, isFloatStore, sb, pad, knownConstants))
                    {
                        break;
                    }

                    // A translate-time-constant address outside the MMIO window (gather-pipe already handled
                    // above) always takes the flat helper's plain-store arm, so the runtime MMIO test is provably dead.
                    var provenRamStore =
                        TryGetKnownEffectiveAddress(store, store.Address, knownConstants, out var knownStoreAddress) &&
                        IsProvenRamStore(knownStoreAddress, store.SizeBytes);
                    string StoreHelper(int sizeBytes, bool isFloat) => provenRamStore
                        ? FlatWriteRamHelper(Bits(sizeBytes), isFloat)
                        : GuestStoreHelper(sizeBytes, isFloat);

                    // stfs of a value whose single bits are known: the integer store of the same family (SingleFpr.cs).
                    if (isFloatStore && store.SizeBytes == 4 && TryGetRawStoreBits(store, 0, out var rawBits))
                    {
                        if (!TryEmitStackStore(store, rawBits, isFloatStore: false, sb, pad, types, stackFacts))
                            sb.AppendLine($"{pad}{StoreHelper(4, isFloat: false)}({RemapAddress(store.Address, types, localConstants, linkedAddressRemap)}, {rawBits});");
                    }
                    else if (TryEmitStackStore(store, srcExpr, isFloatStore, sb, pad, types, stackFacts))
                    {
                    }
                    else if (isFloatStore && store.SizeBytes == 4)
                    {
                        sb.AppendLine($"{pad}{StoreHelper(4, isFloat: true)}({RemapAddress(store.Address, types, localConstants, linkedAddressRemap)}, {srcExpr});");
                    }
                    else if (isFloatStore && store.SizeBytes == 8)
                    {
                        sb.AppendLine($"{pad}{StoreHelper(8, isFloat: true)}({RemapAddress(store.Address, types, localConstants, linkedAddressRemap)}, {srcExpr});");
                    }
                    else
                    {
                        // Memory::Write8/Write16 take uint8_t/uint16_t, so cast to
                        // the exact parameter type instead of widening to
                        // uint32_t and letting the call narrow it again. The
                        // stored bytes are identical either way. Write64 keeps a
                        // 32-bit cast: it has no 64-bit integer producer here.
                        var integerSource = store.SizeBytes switch
                        {
                            4 => srcExpr,
                            1 or 2 => $"static_cast<uint{Bits(store.SizeBytes)}_t>({srcExpr})",
                            _ => $"static_cast<uint32_t>({srcExpr})"
                        };
                        sb.AppendLine($"{pad}{StoreHelper(store.SizeBytes, isFloat: false)}({RemapAddress(store.Address, types, localConstants, linkedAddressRemap)}, {integerSource});");
                    }
                }
                break;
            case IrResolveGuestMemoryRange resolve:
                {
                    var start = RemapAddress(
                        new IrAddress(resolve.Base.RegisterName!, resolve.MinOffset),
                        types, localConstants, linkedAddressRemap);
                    // MKW_LEAN_RANGES: segment ranges (memory_access.h, ResolveSegmentRange) and their accessors. A write
                    // range off a proven stack address or proven RAM SDA range skips the policy byte, as
                    // FlatStoreRam does: ResolveSegmentStackRange.
                    var provenWriteRamRange = LeanRanges && resolve.NeedsWriteAccess &&
                        (stackFacts.TryResolve(resolve, new IrAddress(resolve.Base.RegisterName!, resolve.MinOffset), out _) ||
                         (TryGetFixedSdaBase(resolve.Base.RegisterName, out var sdaRangeBase) &&
                          IsProvenRamRange(unchecked(sdaRangeBase + (uint)resolve.MinOffset), resolve.Length)));
                    if (provenWriteRamRange)
                    {
                        sb.AppendLine($"{pad}{resolve.Destination} = MemoryInline::ResolveSegmentStackRange({start}, 0, {resolve.Length}u);");
                        break;
                    }
                    var resolver = LeanRanges ? "ResolveSegmentRange" : "ResolveRangeHost";
                    sb.AppendLine($"{pad}{resolve.Destination} = MemoryInline::{resolver}({start}, 0, {resolve.Length}u, {(resolve.NeedsWriteAccess ? "true" : "false")});");
                }
                break;
            case IrResolvedLoad load:
                {
                    var dest = ToDestination(load.Destination, types);
                    var destType = types.Get(load.Destination);
                    var isFloatDest = (destType is ValueRepresentation pt && pt.IsFloat) || IsFloatRegister(load.Destination);
                    var address = RemapAddress(load.OriginalAddress, types, localConstants, linkedAddressRemap);
                    var helper = isFloatDest ? $"Float{Bits(load.SizeBytes)}" : Bits(load.SizeBytes).ToString();
                    if (isFloatDest && load.SizeBytes == 4 && TryGetRawSingleRoot(load.Destination, out _))
                        EmitSingleLoad(sb, pad, load.Destination, dest,
                            $"MemoryInline::{(LeanRanges ? "SegRead" : "ReadResolved")}32({load.Range}, {load.RangeOffset}u, {address})");
                    else
                        sb.AppendLine($"{pad}{dest} = MemoryInline::{(LeanRanges ? "SegRead" : "ReadResolved")}{helper}({load.Range}, {load.RangeOffset}u, {address});");
                    if (IsFloatRegister(load.Destination))
                        localPaired[GetRegisterBaseName(load.Destination)] = false;
                }
                break;
            case IrResolvedStore store:
                {
                    var srcExpr = ToExpression(store.Source, types);
                    var isFloatStore = store.Source.Kind == "register" && store.Source.RegisterName is not null &&
                                       IsFloatRegister(store.Source.RegisterName);
                    if (isFloatStore && store.SizeBytes == 4 && store.Source.RegisterName is not null &&
                        localPaired.TryGetValue(GetRegisterBaseName(store.Source.RegisterName), out var sourceIsPaired) && sourceIsPaired)
                        srcExpr = $"PPC_PsToScalarInline({srcExpr})";
                    var address = RemapAddress(store.OriginalAddress, types, localConstants, linkedAddressRemap);
                    var helper = isFloatStore ? $"Float{Bits(store.SizeBytes)}" : Bits(store.SizeBytes).ToString();
                    var value = isFloatStore || store.SizeBytes == 4 ? srcExpr : $"static_cast<uint{Bits(store.SizeBytes)}_t>({srcExpr})";
                    if (isFloatStore && store.SizeBytes == 4 && TryGetRawStoreBits(store, 0, out var rawStoreBits))
                    {
                        helper = "32";
                        value = rawStoreBits;
                    }
                    sb.AppendLine($"{pad}MemoryInline::{(LeanRanges ? "SegWrite" : "WriteResolved")}{helper}({store.Range}, {store.RangeOffset}u, {address}, {value});");
                }
                break;
            case IrResolvedPsqLoad load:
                {
                    var helper = load.KnownGqr.HasValue
                        ? "PPC_PsqLKnownResolvedInline"
                        : "PPC_PsqLResolvedInline";
                    var known = load.KnownGqr.HasValue ? $", 0x{load.KnownGqr.Value:X8}u" : string.Empty;
                    var template = $"<{load.W}u, {load.I}u{known}>";
                    var address = ToExpression(load.OriginalAddress, types);
                    var knownCall = $"{helper}{template}(ctx, {load.Range}, {load.RangeOffset}u, {address})";
                    if (load.GuardKnownGqr && load.KnownGqr.HasValue)
                    {
                        var fallback = $"PPC_PsqLResolvedInline<{load.W}u, {load.I}u>(ctx, {load.Range}, {load.RangeOffset}u, {address})";
                        sb.AppendLine($"{pad}if ({GqrEntryGuardName(load.I, load.KnownGqr.Value)}) {{");
                        sb.AppendLine($"{pad}    {PairedAssignment(load.Destination, knownCall, types)}");
                        sb.AppendLine($"{pad}}} else {{");
                        sb.AppendLine($"{pad}    {PairedAssignment(load.Destination, fallback, types)}");
                        sb.AppendLine($"{pad}}}");
                    }
                    else sb.AppendLine($"{pad}{PairedAssignment(load.Destination, knownCall, types)}");
                    if (IsFloatRegister(load.Destination))
                        localPaired[GetRegisterBaseName(load.Destination)] = true;
                }
                break;
            case IrResolvedPsqStore store:
                {
                    var helper = store.KnownGqr.HasValue
                        ? "PPC_PsqStKnownResolvedInline"
                        : _segmentPsqStores ? "PPC_PsqStSegmentInline" : "PPC_PsqStResolvedInline";
                    var known = store.KnownGqr.HasValue ? $", 0x{store.KnownGqr.Value:X8}u" : string.Empty;
                    var template = $"<{store.W}u, {store.I}u{known}>";
                    var value = ToPairedFloatExpression(store.Source, types, localPaired);
                    var address = ToExpression(store.OriginalAddress, types);
                    if (_segmentPsqStores && !store.KnownGqr.HasValue && store.W == 0 &&
                        !IsPairedFloatValue(store.Source, localPaired))
                    {
                        sb.AppendLine($"{pad}PPC_PsqStScalarSegmentInline<{store.I}u>(ctx, {store.Range}, {store.RangeOffset}u, {address}, {ToExpression(store.Source, types)});");
                        break;
                    }
                    var knownCall = $"{helper}{template}(ctx, {store.Range}, {store.RangeOffset}u, {address}, {value})";
                    if (store.GuardKnownGqr && store.KnownGqr.HasValue)
                    {
                        sb.AppendLine($"{pad}if ({GqrEntryGuardName(store.I, store.KnownGqr.Value)}) {{");
                        sb.AppendLine($"{pad}    {knownCall};");
                        sb.AppendLine($"{pad}}} else {{");
                        sb.AppendLine($"{pad}    PPC_PsqStResolvedInline<{store.W}u, {store.I}u>(ctx, {store.Range}, {store.RangeOffset}u, {address}, {value});");
                        sb.AppendLine($"{pad}}}");
                    }
                    else sb.AppendLine($"{pad}{knownCall};");
                }
                break;
            case IrResolvedLoadPair pair:
                {
                    bool IsFloat(string destination) =>
                        (types.Get(destination) is ValueRepresentation primitive && primitive.IsFloat) || IsFloatRegister(destination);
                    var firstFloat = IsFloat(pair.FirstDestination);
                    var secondFloat = IsFloat(pair.SecondDestination);
                    var firstAddress = RemapAddress(pair.FirstOriginalAddress, types, localConstants, linkedAddressRemap);
                    var secondAddress = RemapAddress(pair.SecondOriginalAddress, types, localConstants, linkedAddressRemap);
                    var readPrefix = LeanRanges ? "SegRead" : "ReadResolved";
                    var pairHelper = pair.ElementSizeBytes == 2
                        ? $"MemoryInline::{readPrefix}Pair16({pair.Range}, {pair.RangeOffset}u)"
                        : $"MemoryInline::{readPrefix}Pair32({pair.Range}, {pair.RangeOffset}u)";
                    // A float word is an lfs: the exact widening (FlatReadFloat32Body), which a float->double
                    // conversion is not under host DAZ (FPSCR[NI]). Segment pairs always take this arm.
                    string FastValue(string member, bool isFloat) => !isFloat
                        ? $"resolved_pair.{member}"
                        : LeanRanges
                            ? $"PpcBitCastToDoubleInline(PpcConvertToDoubleBitsInline(resolved_pair.{member}))"
                            : $"PpcBitCastToFloatInline(resolved_pair.{member})";
                    string SlowHelper(bool isFloat) => isFloat ? $"Float{Bits(pair.ElementSizeBytes)}" : Bits(pair.ElementSizeBytes).ToString();
                    var firstMember = pair.Descending ? "second" : "first";
                    var secondMember = pair.Descending ? "first" : "second";
                    var firstOffset = pair.Descending ? pair.RangeOffset + pair.ElementSizeBytes : pair.RangeOffset;
                    var secondOffset = pair.Descending ? pair.RangeOffset : pair.RangeOffset + pair.ElementSizeBytes;
                    // A float word whose bits a store reuses goes through its raw single bits local (SingleFpr.cs):
                    // `word` is the element's 32-bit word, `value` its usual expression.
                    void Element(string destination, bool isFloat, string word, string value)
                    {
                        var target = ToDestination(destination, types);
                        if (isFloat && pair.ElementSizeBytes == 4 && LeanRanges && TryGetRawSingleRoot(destination, out _))
                            EmitSingleLoad(sb, pad + "        ", destination, target, word);
                        else
                            sb.AppendLine($"{pad}        {target} = {value};");
                    }
                    sb.AppendLine($"{pad}{{");
                    sb.AppendLine($"{pad}    const auto resolved_pair = {pairHelper};");
                    sb.AppendLine($"{pad}    if (resolved_pair.valid) {{");
                    Element(pair.FirstDestination, firstFloat, $"resolved_pair.{firstMember}", FastValue(firstMember, firstFloat));
                    Element(pair.SecondDestination, secondFloat, $"resolved_pair.{secondMember}", FastValue(secondMember, secondFloat));
                    sb.AppendLine($"{pad}    }} else {{");
                    Element(pair.FirstDestination, firstFloat, $"MemoryInline::{readPrefix}32({pair.Range}, {firstOffset}u, {firstAddress})",
                        $"MemoryInline::{readPrefix}{SlowHelper(firstFloat)}({pair.Range}, {firstOffset}u, {firstAddress})");
                    Element(pair.SecondDestination, secondFloat, $"MemoryInline::{readPrefix}32({pair.Range}, {secondOffset}u, {secondAddress})",
                        $"MemoryInline::{readPrefix}{SlowHelper(secondFloat)}({pair.Range}, {secondOffset}u, {secondAddress})");
                    sb.AppendLine($"{pad}    }}");
                    sb.AppendLine($"{pad}}}");
                    if (IsFloatRegister(pair.FirstDestination)) localPaired[GetRegisterBaseName(pair.FirstDestination)] = false;
                    if (IsFloatRegister(pair.SecondDestination)) localPaired[GetRegisterBaseName(pair.SecondDestination)] = false;
                }
                break;
            case IrResolvedStorePair pair:
                {
                    bool IsFloat(IrValue source) => source.RegisterName is { } name && IsFloatRegister(name);
                    string Source(IrValue source)
                    {
                        var expression = ToExpression(source, types);
                        if (pair.ElementSizeBytes == 4 && source.RegisterName is { } name && IsFloatRegister(name) &&
                            localPaired.TryGetValue(GetRegisterBaseName(name), out var paired) && paired)
                            expression = $"PPC_PsToScalarInline({expression})";
                        return expression;
                    }
                    var firstFloat = IsFloat(pair.FirstSource);
                    var secondFloat = IsFloat(pair.SecondSource);
                    var first = Source(pair.FirstSource);
                    var second = Source(pair.SecondSource);
                    // A float word whose single bits are known is stored as that word (SingleFpr.cs).
                    string? firstBits = null, secondBits = null;
                    if (pair.ElementSizeBytes == 4 && firstFloat && TryGetRawStoreBits(pair, 0, out var firstRaw))
                        firstBits = firstRaw;
                    if (pair.ElementSizeBytes == 4 && secondFloat && TryGetRawStoreBits(pair, 1, out var secondRaw))
                        secondBits = secondRaw;
                    // A float word is an stfs: Gekko's narrowing (ConvertPpcDoubleToSingleBits), not the host's rounding
                    // cast, for a value left in double precision. Segment pairs always store through this arm.
                    string Raw(string value, bool isFloat) => !isFloat
                        ? $"static_cast<uint32_t>({value})"
                        : LeanRanges
                            ? $"MemoryInline::ConvertPpcDoubleToSingleBits({value})"
                            : $"PpcBitCastToU32Inline(static_cast<float>({value}))";
                    var lowSource = pair.Descending ? secondBits ?? second : firstBits ?? first;
                    var highSource = pair.Descending ? firstBits ?? first : secondBits ?? second;
                    var lowFloat = pair.Descending ? secondFloat && secondBits is null : firstFloat && firstBits is null;
                    var highFloat = pair.Descending ? firstFloat && firstBits is null : secondFloat && secondBits is null;
                    var packed = pair.ElementSizeBytes == 2
                        ? $"((static_cast<uint32_t>(static_cast<uint16_t>({lowSource})) << 16) | static_cast<uint16_t>({highSource}))"
                        : $"((static_cast<uint64_t>({Raw(lowSource, lowFloat)}) << 32) | {Raw(highSource, highFloat)})";
                    var firstAddress = RemapAddress(pair.FirstOriginalAddress, types, localConstants, linkedAddressRemap);
                    var secondAddress = RemapAddress(pair.SecondOriginalAddress, types, localConstants, linkedAddressRemap);
                    var writePrefix = LeanRanges ? "SegWrite" : "WriteResolved";
                    var fast = pair.ElementSizeBytes == 2
                        ? $"MemoryInline::{writePrefix}Pair16({pair.Range}, {pair.RangeOffset}u, {packed})"
                        : $"MemoryInline::{writePrefix}Pair32({pair.Range}, {pair.RangeOffset}u, {packed})";
                    string Helper(bool isFloat) => isFloat ? $"Float{Bits(pair.ElementSizeBytes)}" : Bits(pair.ElementSizeBytes).ToString();
                    var firstOffset = pair.Descending ? pair.RangeOffset + pair.ElementSizeBytes : pair.RangeOffset;
                    var secondOffset = pair.Descending ? pair.RangeOffset : pair.RangeOffset + pair.ElementSizeBytes;
                    sb.AppendLine($"{pad}if (!{fast}) {{");
                    sb.AppendLine($"{pad}    MemoryInline::{writePrefix}{Helper(firstFloat && firstBits is null)}({pair.Range}, {firstOffset}u, {firstAddress}, {firstBits ?? first});");
                    sb.AppendLine($"{pad}    MemoryInline::{writePrefix}{Helper(secondFloat && secondBits is null)}({pair.Range}, {secondOffset}u, {secondAddress}, {secondBits ?? second});");
                    sb.AppendLine($"{pad}}}");
                }
                break;
            case IrCall call:
                if (TryParseAddress(call.Target, out var addr))
                {
                    if (TryEmitInlineGuestThunk(addr, sb, pad, inlineGuestThunkStackBase))
                    {
                        break;
                    }

                    // Guest call boundary: CpuContext becomes authoritative for the callee. Argument normalization
                    // and state-free marshalling both read/write CpuContext directly, so the flush choice below
                    // can't be narrowed when either is present; normalization is built into its own buffer first
                    // so the emit order stays fixed.
                    var argumentNormalization = new StringBuilder();
                    EmitNormalizeGuestCallFloatArguments(
                        call.Target, call.Arguments, argumentNormalization, pad, localPaired, guestAbiProvider);
                    var callSiteKey = new GuestStateFreeCallSiteKey(currentLabel, addr, directCallOrdinal);
                    var hasCallSiteVariant = stateFreeCallSiteVariants.TryGetValue(callSiteKey, out var callSiteVariant);
                    var stateFreeSymbol = hasCallSiteVariant
                        ? callSiteVariant!.Symbol
                        : stateFreeCallSymbols.GetValueOrDefault(addr);
                    var stateFreeContract = hasCallSiteVariant
                        ? callSiteVariant!.Contract
                        : stateFreeAbiContracts.GetValueOrDefault(addr);
                    var usesStateFreeCall = stateFreeSymbol is not null && stateFreeContract is not null &&
                                            IsStateFreeCallSafe(stateFreeContract);
                    var resumesThroughLinkRegister =
                        nonReturningCallTargets.Contains(addr) ||
                        (lrContinuationCallTargets.Contains(addr) && localFallthroughLr.HasValue);
                    // The explicit-state signature carries everything the callee reads/writes, so a resident caller
                    // can pass host registers instead of spilling to CpuContext. Excluded: paired-single arg/return
                    // normalization rewrites ctx->fpr[N] directly, and LR-continuation sites dispatch on ctx->lr
                    // after the callee writes it, so those keep the full boundary.
                    var marshalsStateFreeThroughResidency =
                        usesStateFreeCall &&
                        _activeResidency is not null &&
                        argumentNormalization.Length == 0 &&
                        !resumesThroughLinkRegister &&
                        !IsGuestPairedScalarFloatReturn(guestAbiProvider, call.Target);
                    if (marshalsStateFreeThroughResidency)
                    {
                        EmitResidentStateFreeCall(
                            sb, pad, bufferBaseLength, addr, stateFreeSymbol!, stateFreeContract!,
                            localFallthroughLr, stateFreeAbiContracts, stateFreeCallSymbols);
                        // Emits nothing for this shape (the paired-scalar return
                        // case is excluded above); it still owns the f1
                        // representation tag every guest call boundary applies.
                        EmitNormalizeGuestCallFloatReturn(call.Target, sb, pad, localPaired, guestAbiProvider);
                        break;
                    }
                    // MKW_EABI_REGS register-passing call. Same exclusions as the state-free form: paired-single
                    // argument/return normalization and LR-continuation sites rewrite CpuContext directly.
                    if (_activeResidency is not null && !usesStateFreeCall &&
                        _eabiVariants?.TryGetValue(addr, out var calleeVariant) == true &&
                        argumentNormalization.Length == 0 &&
                        !resumesThroughLinkRegister &&
                        !modOverridableCallTargets.Contains(addr) &&
                        !IsGuestPairedScalarFloatReturn(guestAbiProvider, call.Target))
                    {
                        EmitEabiVariantCall(sb, pad, bufferBaseLength, calleeVariant!, guestAbiContracts.GetValueOrDefault(addr));
                        EmitNormalizeGuestCallFloatReturn(call.Target, sb, pad, localPaired, guestAbiProvider);
                        break;
                    }
                    var boundarySync = usesStateFreeCall || argumentNormalization.Length != 0
                        ? ResidencyBoundarySync.Full.WithEabiPreservation(guestAbiContracts.GetValueOrDefault(addr))
                        : ResolveDirectCallBoundarySync(addr, guestAbiContracts, modOverridableCallTargets);
                    if (!boundarySync.IsFull &&
                        IsGuestPairedScalarFloatReturn(guestAbiProvider, call.Target))
                    {
                        // The return normalization below rewrites ctx->fpr[1]
                        // directly, so f1 has to be part of this boundary even if
                        // the callee contract does not mention it.
                        boundarySync = boundarySync.WithFpr(1);
                    }
                    // Unconditional on both sides: the state-free arms below only touch CpuContext, and the reload
                    // after them runs on every path (LR-continuation bodies get no flush plan at all).
                    AppendFlush(sb, pad, boundarySync, unconditional: true);
                    sb.Append(argumentNormalization);
                    // Only the resident path (EmitResidentStateFreeCall) takes MKW_LEAN_CALLS: this context-spelled form
                    // is also the input the state-free clone rewriter recognizes by its if/else shape, so it keeps it.
                    if (usesStateFreeCall)
                    {
                        sb.AppendLine($"{pad}if ({StateFreeAvailabilityCondition(addr, stateFreeAbiContracts, stateFreeCallSymbols)}) {{");
                        if (StateFreeHasOutputs(stateFreeContract!))
                        {
                            // Uniquified by the call site's position in the
                            // translation unit. The body is built in its own
                            // buffer, so the offset of that buffer inside the
                            // unit has to be added back.
                            var stateFreeResult = $"state_free_result_{addr:X8}_{bufferBaseLength + sb.Length:X}";
                            sb.AppendLine($"{pad}    const auto {stateFreeResult} = {stateFreeSymbol}({StateFreeCallArguments(stateFreeContract!, localFallthroughLr)});");
                            AppendStateFreeResultStores(sb, pad + "    ", stateFreeContract!, stateFreeResult);
                        }
                        else
                        {
                            sb.AppendLine($"{pad}    {stateFreeSymbol}({StateFreeCallArguments(stateFreeContract!, localFallthroughLr)});");
                        }
                        sb.AppendLine($"{pad}}} else {{");
                        RecordEmittedDirectCallTarget(addr);
                        sb.AppendLine($"{pad}    InvokeDirectCpu<0x{addr:X8}u>(ctx);");
                        sb.AppendLine($"{pad}}}");
                    }
                    else
                    {
                        RecordEmittedDirectCallTarget(addr);
                        sb.AppendLine($"{pad}InvokeDirectCpu<0x{addr:X8}u>(ctx);");
                    }
                    if (resumesThroughLinkRegister)
                    {
                        // The callee may resume this body through a local LR
                        // continuation label, so the locals must be refreshed
                        // from the context it just wrote. The exit paths below
                        // return without writing locals back.
                        AppendReload(sb, pad, boundarySync);
                        var fallbackPad = pad;
                        if (localFallthroughLr.HasValue)
                        {
                            sb.AppendLine($"{pad}if (ctx->lr != 0x{localFallthroughLr.Value:X8}u) {{");
                            fallbackPad = IndentPad(indent + 1);
                        }

                        if (shareLrContinuationDispatch)
                        {
                            sb.AppendLine($"{fallbackPad}goto lr_continuation_dispatch;");
                        }
                        else
                        {
                            EmitLocalLrContinuationDispatch(sb, fallbackPad, labelNames);
                            sb.AppendLine($"{fallbackPad}if (TranslatedFunctionRegistry::FindByAddressPtr(ctx->lr) != nullptr) {{");
                            sb.AppendLine($"{fallbackPad}    InvokeIndirectCpu(ctx->lr, ctx);");
                            sb.AppendLine($"{fallbackPad}}}");
                            sb.AppendLine($"{fallbackPad}{ReturnStatement(fromContext: true)}");
                        }
                        if (localFallthroughLr.HasValue)
                        {
                            sb.AppendLine($"{pad}}}");
                        }
                        break;
                    }
                    EmitNormalizeGuestCallFloatReturn(call.Target, sb, pad, localPaired, guestAbiProvider);
                    AppendReload(sb, pad, boundarySync, unconditional: true);
                }
                else if (IsGuestCallTarget(call.Target))
                {
                    var targetSymbol = FormatSymbol(call.Target);
                    AppendFlush(sb, pad, ResidencyBoundarySync.Full, unconditional: true);
                    EmitNormalizeGuestCallFloatArguments(call.Target, call.Arguments, sb, pad, localPaired, guestAbiProvider);
                    sb.AppendLine($"{pad}{targetSymbol}(ctx);");
                    EmitNormalizeGuestCallFloatReturn(call.Target, sb, pad, localPaired, guestAbiProvider);
                    AppendReload(sb, pad, ResidencyBoundarySync.Full.WithEabiPreservation(null), unconditional: true);
                }
                else
                {
                    // Try inline expansion first for performance
                    if (TryEmitInlinePpc(call, sb, pad, types, localPaired, stackFacts))
                    {
                        // Inline code was emitted - update paired tracking
                        if (!string.IsNullOrWhiteSpace(call.Destination) && IsFloatRegister(call.Destination))
                        {
                            localPaired[GetRegisterBaseName(call.Destination)] = IsPairedProducerTarget(call.Target);
                        }
                        break;
                    }

                    // Helper function call (e.g., PPC_Ps*, memory helpers)
                    var isPairedConsumer = IsPairedConsumerTarget(call.Target);
                    var isSingleConsumer = IsSinglePrecisionConsumerTarget(call.Target);
                    var isScalarConsumer = IsScalarFloatConsumerTarget(call.Target);
                    var args = new List<string>(call.Arguments.Count);
                    for (var argIndex = 0; argIndex < call.Arguments.Count; argIndex++)
                    {
                        var argVal = call.Arguments[argIndex];
                        var expr = isPairedConsumer
                            ? ToPairedFloatExpression(argVal, types, localPaired)
                            : (isSingleConsumer || isScalarConsumer)
                                ? ToScalarFloatExpression(argVal, types, localPaired)
                                : ToExpression(argVal, types);
                        args.Add(expr);
                    }
                    // Helpers can implicitly touch guest state through CpuContext (PPC_Stwcx writes CR0, PPC_Lswi
                    // writes GPRs, OSSystemCall is a full boundary). Resident locals can't observe that, so the
                    // cataloged effect becomes a mini boundary: flush what it may read, reload what it may write.
                    var helperEffect = AnalyzeGuestHelperEffect(call);
                    var helperSync = _activeResidency is not null
                        ? ResidencyBoundarySync.FromHelperEffect(helperEffect)
                        : null;
                    // MKW_EABI_REGS: an uncataloged (complete-boundary) helper is not trusted with this body's
                    // nonvolatiles' CpuContext copies, which may be stale; it does not change them either.
                    if (helperSync is not null && helperEffect.BoundaryFlags.HasFlag(GuestCallBoundaryFlags.RequiresCompleteContext))
                        helperSync = helperSync.WithEabiPreservation(null);
                    if (helperSync is not null && !helperSync.IsEmpty)
                    {
                        AppendFlush(sb, pad, helperSync, unconditional: true);
                    }
                    // The fused single family with a single-precision C (multiplier) skips C's 25-bit rounding
                    // (CxxLinearCodeGenerator.SingleFpr.cs).
                    var callSymbol = FusedSingleTargets.Contains(call.Target) && call.Arguments.Count == 3 &&
                                     IsSingleFprOperand(call.Arguments[1], localPaired)
                        ? FormatSymbol(call.Target) + "SingleC"
                        : FormatSymbol(call.Target);
                    var callExpr = $"{callSymbol}({string.Join(", ", args)})";
                    if (FusedSingleTargets.Contains(call.Target) && call.Arguments.Count == 3 &&
                        !string.IsNullOrWhiteSpace(call.Destination) && TryGetRawSingleRoot(call.Destination, out _))
                    {
                        // A fused single whose result's bits a store reuses: computed as its float (SingleFpr.cs).
                        var subtract = call.Target.EndsWith("subs", StringComparison.OrdinalIgnoreCase) ? "true" : "false";
                        var negate = call.Target.StartsWith("PPC_Fn", StringComparison.OrdinalIgnoreCase) ? "true" : "false";
                        var singleC = callSymbol.EndsWith("SingleC", StringComparison.Ordinal) ? "true" : "false";
                        EmitSingleResult(sb, pad, call.Destination, ToDestination(call.Destination, types),
                            $"PpcFusedSingleFloatInline<{subtract}, {negate}, {singleC}>({string.Join(", ", args)})");
                    }
                    else if (!string.IsNullOrWhiteSpace(call.Destination))
                    {
                        sb.AppendLine(IsPairedProducerTarget(call.Target)
                            ? $"{pad}{PairedAssignment(call.Destination, callExpr, types)}"
                            : $"{pad}{ToDestination(call.Destination, types)} = {callExpr};");
                    }
                    else
                    {
                        sb.AppendLine($"{pad}{callExpr};");
                    }
                    if (helperSync is not null && !helperSync.IsEmpty)
                    {
                        // An explicit destination carries the helper's return
                        // value in the local; reloading it from CpuContext would
                        // clobber that result with stale state.
                        var reloadSync = ExcludeDestinationFromReload(helperSync, call.Destination);
                        AppendReload(sb, pad, reloadSync, unconditional: true);
                    }
                }
                if (!string.IsNullOrWhiteSpace(call.Destination) && IsFloatRegister(call.Destination))
                {
                    localPaired[GetRegisterBaseName(call.Destination)] = IsPairedProducerTarget(call.Target);
                }
                break;
            case IrIndirectCall icall:
                {
                    var indirectTarget = ToExpression(icall.Target, types);
                    // Gecko code handler (0x800018A8) keeps a full reload: P+ C0 codes mutate r15 across its bctrl
                    // to switch code lists to BOOST.GCT. Code variants (_variantOwnCode != null) also keep a full
                    // reload: P+ C2 hooks spliced into code variants invoke custom non-EABI ASM helpers (e.g.
                    // 0x8058549C returning in r28, 0x80585750 returning in f29, 0x800018A8) via bctrl. Normal base
                    // functions flush all dirty locals (keeping CpuContext 100% architectural for PoyoJIT, C2 hooks,
                    // and OSSaveContext) and reload only PowerPC EABI volatiles, keeping r1, r2, r13, r14-r31,
                    // f14-f31, and cr2-cr4 in host locals.
                    var indirectSync = GuestEabi.Enabled
                        ? ResidencyBoundarySync.EabiUnknownCall
                        : (GuestAbiContractAnalyzer.ModSafeIndirectAbi &&
                           _variantOwnCode is null &&
                           _currentEntryPoint != GuestAbiContractAnalyzer.GeckoCodeHandlerAddress
                            ? ResidencyBoundarySync.ModSafeIndirectCall
                            : ResidencyBoundarySync.Full);
                    AppendFlush(sb, pad, indirectSync, unconditional: true);
                    if (IndirectCallProfile.Instrument && _currentGuestSite != 0)
                    {
                        sb.AppendLine($"{pad}MKW_INDIRECT_PROFILE_COUNT(0x{_currentEntryPoint:X8}u, 0x{_currentGuestSite:X8}u, {indirectTarget});");
                    }
                    // Profile-guided promotion (IndirectCallProfile): exact compares against the site's dominant
                    // targets, each an ordinary direct call, under the same flush and reload as the indirect call.
                    // One line: the shard emitter drops an arm it would not lower to MKW_STATIC_TRANSLATED_CALL.
                    var promotedTargets = _currentEntryPoint == GuestAbiContractAnalyzer.GeckoCodeHandlerAddress
                        ? Array.Empty<uint>()
                        : IndirectCallProfile.PromotedTargets(_currentEntryPoint, _currentGuestSite);
                    if (promotedTargets.Count != 0)
                    {
                        foreach (var promoted in promotedTargets) RecordEmittedDirectCallTarget(promoted);
                        sb.AppendLine(IndirectCallProfile.PromotedLine(
                            pad, indirectTarget, promotedTargets, $"MKW_INDIRECT_CALL({indirectTarget}, ctx);"));
                    }
                    else
                    {
                        sb.AppendLine($"{pad}MKW_INDIRECT_CALL({indirectTarget}, ctx);");
                    }
                    AppendReload(sb, pad, indirectSync, unconditional: true);
                    // f1 is the conventional scalar floating-point return.
                    // Do not retag the other volatile FPRs: a callee that leaves
                    // one untouched also leaves its packed payload untouched.
                    localPaired["f1"] = false;
                }
                break;
            case IrIndirectJump ijump:
                {
                    var jumpTarget = ToExpression(ijump.Target, types);
                    // Tail dispatch: the callee inherits authoritative context
                    // and this frame never observes registers again, so the
                    // locals must not be written back afterwards.
                    AppendFlush(sb, pad);
                    EmitVariantIndirectJumpCheck(sb, pad, jumpTarget, currentLabel);
                    if (IndirectCallProfile.Instrument && _currentGuestSite != 0)
                    {
                        sb.AppendLine($"{pad}MKW_INDIRECT_PROFILE_JUMP(0x{_currentEntryPoint:X8}u, 0x{_currentGuestSite:X8}u, {jumpTarget});");
                    }
                    // Promotion as for calls: a direct call to the tail jump's dominant target, then the return below.
                    var promotedJumpTargets = _currentEntryPoint == GuestAbiContractAnalyzer.GeckoCodeHandlerAddress
                        ? Array.Empty<uint>()
                        : IndirectCallProfile.PromotedTargets(_currentEntryPoint, _currentGuestSite, jump: true);
                    if (promotedJumpTargets.Count != 0)
                    {
                        foreach (var promoted in promotedJumpTargets) RecordEmittedDirectCallTarget(promoted);
                        sb.AppendLine(IndirectCallProfile.PromotedLine(
                            pad, jumpTarget, promotedJumpTargets, $"MKW_INDIRECT_JUMP({jumpTarget}, ctx);"));
                    }
                    else
                    {
                        sb.AppendLine($"{pad}MKW_INDIRECT_JUMP({jumpTarget}, ctx);");
                    }
                    sb.AppendLine($"{pad}{ReturnStatement(fromContext: true)}");
                }
                break;
            case IrComment comment:
                {
                    // Emit as a C++ comment - these are legitimate comments like nop, mtcrf no-op, psq operations
                    sb.AppendLine($"{pad}// {EscapeForCxxLiteral(comment.Text)}");
                }
                break;
            case IrTracePpc trace:
                {
                }
                break;
            case IrUndefined undef:
                {
                    var details = undef.Reason == null
                        ? $"{undef.Disassembly}"
                        : $"{undef.Disassembly} | {undef.Reason}";
                    var escaped = EscapeForCxxLiteral(details);
                    sb.AppendLine($"{pad}UNDEFINED(0x{undef.Address:X8}u, 0x{undef.RawInstruction:X8}u, \"{escaped}\");");
                }
                break;
            case IrSetCrField setCr:
                {
                    var leftExpr = ToExpression(setCr.Left, types);
                    var rightExpr = ToExpression(setCr.Right, types);
                    // The field's value (isa/ppc_isa_crv.h): an integer compare is the operands' difference, so a
                    // compare whose field a call or a return publishes is a subtraction and, at that boundary, one
                    // store. XER is localized too, so the summary-overflow copy reads a host register instead of
                    // reloading CpuContext after every potentially aliasing guest store.
                    var crField = CrFieldExpression(setCr.FieldIndex & 7, written: true);
                    if (IsFloatValue(setCr.Left, types) || IsFloatValue(setCr.Right, types))
                    {
                        leftExpr = ToScalarFloatExpression(setCr.Left, types, localPaired);
                        rightExpr = ToScalarFloatExpression(setCr.Right, types, localPaired);
                        sb.AppendLine($"{pad}{crField} = PpcCrvCompareFloat({leftExpr}, {rightExpr});");
                    }
                    else
                    {
                        var leftCast = setCr.IsUnsigned
                            ? $"static_cast<uint32_t>({leftExpr})"
                            : $"static_cast<int32_t>({leftExpr})";
                        var rightCast = setCr.IsUnsigned
                            ? $"static_cast<uint32_t>({rightExpr})"
                            : $"static_cast<int32_t>({rightExpr})";
                        var xer = _activeResidency is null ? "ctx->xer" : _activeResidency.Xer(written: false);
                        sb.AppendLine($"{pad}{crField} = PpcCrvCompare({leftCast}, {rightCast}, {xer});");
                    }
                }
                break;
        }
    }

    private static bool TryEmitStackLoad(
        IrLoad load,
        string dest,
        bool isFloatDest,
        StringBuilder sb,
        string pad,
        RepresentationEnvironment types,
        StackAddressFacts stackFacts)
    {
        if (!stackFacts.TryResolve(load, load.Address, out _))
        {
            return false;
        }

        var address = Address(load.Address, types);
        if (isFloatDest && load.SizeBytes == 4 && TryGetRawSingleRoot(load.Destination, out _))
        {
            EmitSingleLoad(sb, pad, load.Destination, dest, $"{StackLoadHelper(4, isFloat: false)}({address})");
            return true;
        }
        if (isFloatDest && load.SizeBytes == 4)
        {
            sb.AppendLine($"{pad}{dest} = {StackLoadHelper(4, isFloat: true)}({address});");
            return true;
        }
        if (isFloatDest && load.SizeBytes == 8)
        {
            sb.AppendLine($"{pad}{dest} = {StackLoadHelper(8, isFloat: true)}({address});");
            return true;
        }

        sb.AppendLine($"{pad}{dest} = {StackLoadHelper(load.SizeBytes, isFloat: false)}({address});");
        return true;
    }

    private static bool TryEmitStackStore(
        IrStore store,
        string srcExpr,
        bool isFloatStore,
        StringBuilder sb,
        string pad,
        RepresentationEnvironment types,
        StackAddressFacts stackFacts)
    {
        if (!stackFacts.TryResolve(store, store.Address, out _))
        {
            return false;
        }

        var address = Address(store.Address, types);
        if (isFloatStore && store.SizeBytes == 4)
        {
            sb.AppendLine($"{pad}{StackStoreHelper(4, isFloat: true)}({address}, {srcExpr});");
            return true;
        }
        if (isFloatStore && store.SizeBytes == 8)
        {
            sb.AppendLine($"{pad}{StackStoreHelper(8, isFloat: true)}({address}, {srcExpr});");
            return true;
        }

        var bits = Bits(store.SizeBytes);
        var integerSource = bits == 32 ? srcExpr : $"static_cast<uint{bits}_t>({srcExpr})";
        sb.AppendLine($"{pad}{StackStoreHelper(store.SizeBytes, isFloat: false)}({address}, {integerSource});");
        return true;
    }

    private static bool TryEmitKnownGpuFifoStore(
        IrStore store,
        string srcExpr,
        bool isFloatStore,
        StringBuilder sb,
        string pad,
        IReadOnlyDictionary<string, uint> knownConstants)
    {
        if (!TryGetKnownGpuFifoStore(store, isFloatStore, knownConstants, out var fifoStore))
        {
            return false;
        }

        if (isFloatStore)
        {
            if (_activeGpuFifoBurstSlot is { } floatSlot)
            {
                EmitGpuFifoBurstSlotStore(sb, pad, floatSlot, srcExpr);
                return true;
            }

            sb.AppendLine($"{pad}GX_HLE_FIFO_WriteFloat(static_cast<float>({srcExpr}));");
            return true;
        }

        if (_activeGpuFifoBurstSlot is { } slot)
        {
            EmitGpuFifoBurstSlotStore(sb, pad, slot, srcExpr);
            return true;
        }

        var helper = fifoStore.ByteLength switch
        {
            1 => "GX_HLE_FIFO_Write8",
            2 => "GX_HLE_FIFO_Write16",
            4 => "GX_HLE_FIFO_Write32",
            _ => throw new InvalidOperationException($"Unexpected GPU FIFO width {fifoStore.ByteLength}.")
        };
        sb.AppendLine($"{pad}{helper}(static_cast<uint{fifoStore.ByteLength * 8}_t>({srcExpr}));");
        return true;
    }

    private static void EmitLocalLrContinuationDispatch(
        StringBuilder sb,
        string pad,
        IReadOnlyDictionary<string, string> labelNames)
    {
        var localContinuations = labelNames
            .Where(static pair => TryParseAddress(pair.Key, out _))
            .Select(static pair =>
            {
                TryParseAddress(pair.Key, out var address);
                return (Address: address, Label: pair.Value);
            })
            .OrderBy(static item => item.Address)
            .ToArray();

        if (localContinuations.Length == 0)
        {
            return;
        }

        sb.AppendLine($"{pad}switch (ctx->lr) {{");
        foreach (var (address, label) in localContinuations)
        {
            RecordEmittedLocalLabel(label);
            sb.AppendLine($"{pad}case 0x{address:X8}u:");
            sb.AppendLine($"{pad}    goto {label};");
        }

        sb.AppendLine($"{pad}default:");
        sb.AppendLine($"{pad}    break;");
        sb.AppendLine($"{pad}}}");
    }

    /// <summary>
    /// The translate-time-constant address <paramref name="at"/> accesses, when its base is a known constant there:
    /// not where the base is read across a guest call that may have changed it (CallClobberedValues).
    /// </summary>
    private static bool TryGetKnownEffectiveAddress(
        IrInstruction at,
        IrAddress address,
        IReadOnlyDictionary<string, uint> knownConstants,
        out uint effectiveAddress)
    {
        if (!CallClobberedValues.IsStaleUse(at, address.Base) &&
            knownConstants.TryGetValue(address.Base, out var baseAddress))
        {
            effectiveAddress = unchecked(baseAddress + (uint)address.Offset);
            return true;
        }

        effectiveAddress = 0;
        return false;
    }

    private static bool IsGpuFifoAddress(uint address) =>
        address >= 0xCC008000u && address < 0xCC009000u;

    private static void EmitNormalizeGuestCallFloatArguments(
        string target,
        IReadOnlyList<IrValue> arguments,
        StringBuilder sb,
        string pad,
        Dictionary<string, bool> localPaired,
        IGuestFunctionAbiProvider guestAbiProvider)
    {
        foreach (var arg in arguments)
        {
            if (arg.Kind != "register" || arg.RegisterName == null)
            {
                continue;
            }

            var baseName = GetRegisterBaseName(arg.RegisterName);
            if (!localPaired.TryGetValue(baseName, out var isPaired) ||
                !isPaired ||
                !IsGuestScalarFloatArgument(guestAbiProvider, target, baseName))
            {
                continue;
            }

            var regIndex = ParseFloatRegisterIndex(baseName);
            sb.AppendLine($"{pad}ctx->fpr[{regIndex}].d = PPC_PsToScalarInline(ctx->fpr[{regIndex}].d);");
            localPaired[baseName] = false;
        }
    }

    private static void EmitNormalizeGuestCallFloatReturn(
        string target,
        StringBuilder sb,
        string pad,
        Dictionary<string, bool> localPaired,
        IGuestFunctionAbiProvider guestAbiProvider)
    {
        if (IsGuestPairedScalarFloatReturn(guestAbiProvider, target))
        {
            sb.AppendLine($"{pad}ctx->fpr[1].d = PPC_PsToScalarInline(ctx->fpr[1].d);");
        }

        // f1 is the conventional scalar floating-point return. Preserve the
        // representation tags for f2-f13: ABI volatility does not mean that an
        // untouched packed payload has magically become a scalar value.
        localPaired["f1"] = false;
    }
}
