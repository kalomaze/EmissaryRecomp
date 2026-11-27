using System;
using System.Collections.Generic;
using System.Linq;
using Translator.Core.Ir;

namespace Translator.Core.Analysis;

[Flags]
public enum GuestCallBoundaryFlags
{
    None = 0,
    RequiresCompleteContext = 1 << 0,
    CanSuspend = 1 << 1,
    CanSwitchThreads = 1 << 2,
    InvokesGuestCode = 1 << 3
}

public sealed record GuestAbiContract(
    uint GprReadBeforeWriteMask,
    uint GprPossibleWriteMask,
    uint GprReturnMask,
    uint FprReadBeforeWriteMask,
    uint FprPossibleWriteMask,
    uint FprReturnMask,
    byte CrReadBeforeWriteMask,
    byte CrPossibleWriteMask,
    bool ReadsXerBeforeWrite,
    bool MayWriteXer,
    bool ReadsCtrBeforeWrite,
    bool MayWriteCtr,
    bool ReadsLrBeforeWrite,
    bool MayWriteLr,
    GuestCallBoundaryFlags BoundaryFlags,
    IReadOnlyList<uint> DirectCallTargets)
{
    public bool ReadsFpscrBeforeWrite { get; init; }
    public bool MayWriteFpscr { get; init; }
    public byte GqrReadBeforeWriteMask { get; init; }
    public byte GqrPossibleWriteMask { get; init; }
    public byte HidReadBeforeWriteMask { get; init; }
    public byte HidPossibleWriteMask { get; init; }
    public uint GprDefiniteWriteMask { get; init; }
    public uint FprDefiniteWriteMask { get; init; }
    public byte CrDefiniteWriteMask { get; init; }
    public bool DefinitelyWritesXer { get; init; }
    public bool DefinitelyWritesCtr { get; init; }
    public bool DefinitelyWritesLr { get; init; }
    public bool DefinitelyWritesFpscr { get; init; }
    public byte GqrDefiniteWriteMask { get; init; }
    public byte HidDefiniteWriteMask { get; init; }

    /// <summary>
    /// MKW_EABI_REGS: CR fields read before written, not counting whole-register reads by mfcr/mtcrf (saves and
    /// field merges). Only computed in EABI mode; the masked contract keeps cr1 from it (varargs FP flag).
    /// </summary>
    public byte EabiCrReadBeforeWriteMask { get; init; }

    /// <summary>
    /// MKW_EABI_REGS: GPR/FPR/CTR values read before written by an instruction (not an SSA phi, which only merges
    /// values) or by a callee's own such reads, over the CFG with fall-through edges. The exact masks above also count
    /// phi sources, so a volatile register merged unchanged into the exit value looks like an input there.
    /// </summary>
    public uint EabiGprReadMask { get; init; }
    public uint EabiFprReadMask { get; init; }
    public bool EabiReadsCtr { get; init; }

    /// <summary>
    /// MKW_EABI_REGS: this contract is the EABI view of a function verified to follow the PowerPC EABI
    /// (<see cref="GuestEabi"/>): callers pass only argument registers and see only r3/r4/f1/f2 change.
    /// </summary>
    public bool EabiConformant { get; init; }

    public bool HasFullSynchronizationFence =>
        (BoundaryFlags & GuestCallBoundaryFlags.RequiresCompleteContext) != 0;
}

public static class GuestAbiContractAnalyzer
{
    public static GuestAbiContract Analyze(
        IrFunction function,
        IReadOnlyDictionary<uint, GuestAbiContract>? calleeContracts = null)
    {
        // Block labels are replaced by ordinals for the duration of the
        // analysis: the fixpoint below runs on every function several times per
        // interprocedural round, and string-keyed dictionaries dominated it.
        var blockList = function.Blocks;
        var blockCount = blockList.Count;
        var ordinals = new Dictionary<string, int>(blockCount, StringComparer.Ordinal);
        for (var index = 0; index < blockCount; ++index)
        {
            ordinals.Add(blockList[index].Label, index);
        }

        var predecessors = new List<int>[blockCount];
        for (var index = 0; index < blockCount; ++index) predecessors[index] = new List<int>();
        for (var index = 0; index < blockCount; ++index)
        {
            var successorCount = Successors(blockList[index], out var first, out var second);
            if (successorCount >= 1 && ordinals.TryGetValue(first, out var firstOrdinal))
                predecessors[firstOrdinal].Add(index);
            if (successorCount == 2 && ordinals.TryGetValue(second, out var secondOrdinal))
                predecessors[secondOrdinal].Add(index);
            if (!GuestEabi.Enabled) continue;
            // MKW_EABI_REGS also follows the implicit sequential edge and jump table cases, which the edges above
            // omit (a block reached only that way otherwise starts with nothing written).
            var instructions = blockList[index].Instructions;
            var terminator = instructions.Count == 0 ? null : instructions[instructions.Count - 1];
            if (terminator is IrJumpTable table)
            {
                foreach (var jumpCase in table.Cases)
                    if (ordinals.TryGetValue(jumpCase.TargetLabel, out var caseOrdinal) && !predecessors[caseOrdinal].Contains(index))
                        predecessors[caseOrdinal].Add(index);
            }
            else if (successorCount == 0 && index + 1 < blockCount &&
                     terminator is not (IrReturn or IrUndefined or IrIndirectJump))
            {
                predecessors[index + 1].Add(index);
            }
        }

        var modSafeIndirect = !GuestEabi.Enabled && IsModSafeIndirectFunction(function);

        var definiteWritesByBlock = new GuestStateMask[blockCount];
        for (var index = 0; index < blockCount; ++index)
        {
            var instructions = blockList[index].Instructions;
            var set = GuestStateMask.Empty;
            for (var position = 0; position < instructions.Count; ++position)
            {
                Writes(instructions[position], calleeContracts, modSafeIndirect, out _, out var definiteWrite);
                set = set.Union(definiteWrite);
            }
            definiteWritesByBlock[index] = set;
        }

        var definiteIn = new GuestStateMask[blockCount];
        var definiteOut = new GuestStateMask[blockCount];
        if (GuestEabi.Enabled)
        {
            // MKW_EABI_REGS: the greatest fixpoint of this must-analysis. Starting every block at "nothing written"
            // keeps a loop header at nothing forever (its back edge intersects in the empty set), so every register
            // a loop reads looked read-before-write even when the entry block wrote it. Every block starts at
            // everything (the entry and predecessor-less blocks then take nothing as input): from the top every value
            // only shrinks, so the iteration ends. Starting those blocks at nothing instead mixed rising and falling
            // values, and a code variant's CFG (hook bodies spliced in) kept it cycling forever.
            for (var index = 0; index < blockCount; ++index)
                definiteOut[index] = FullState;
        }

        bool changed;
        do
        {
            changed = false;
            for (var index = 0; index < blockCount; ++index)
            {
                var blockPredecessors = predecessors[index];
                GuestStateMask incoming;
                if (blockList[index].Label == function.EntryLabel || blockPredecessors.Count == 0)
                {
                    incoming = GuestStateMask.Empty;
                }
                else
                {
                    incoming = definiteOut[blockPredecessors[0]];
                    for (var position = 1; position < blockPredecessors.Count; ++position)
                        incoming = incoming.Intersect(definiteOut[blockPredecessors[position]]);
                }

                var outgoing = incoming.Union(definiteWritesByBlock[index]);
                if (incoming != definiteIn[index] || outgoing != definiteOut[index])
                {
                    definiteIn[index] = incoming;
                    definiteOut[index] = outgoing;
                    changed = true;
                }
            }
        } while (changed);

        var readBeforeWrite = GuestStateMask.Empty;
        var possibleWrites = GuestStateMask.Empty;
        var directTargets = new SortedSet<uint>();
        var boundaryFlags = GuestCallBoundaryFlags.None;
        byte eabiCrReads = 0;
        uint eabiGprReads = 0, eabiFprReads = 0;
        var eabiCtrRead = false;
        for (var index = 0; index < blockCount; ++index)
        {
            var written = definiteIn[index];
            var instructions = blockList[index].Instructions;
            for (var position = 0; position < instructions.Count; ++position)
            {
                var instruction = instructions[position];
                var reads = Reads(instruction, calleeContracts);
                readBeforeWrite = readBeforeWrite.Union(reads.Except(written));
                if (GuestEabi.Enabled && instruction is not IrPhi)
                {
                    if (!GuestEabi.IsWholeCrSaveOrMerge(instruction))
                        eabiCrReads |= (byte)(reads.Cr & ~written.Cr);
                    var eabiRead = instruction is IrCall eabiCall &&
                                   GuestTargetParser.TryParseAddress(eabiCall.Target, out var eabiTarget) &&
                                   calleeContracts?.TryGetValue(eabiTarget, out var eabiCallee) == true
                        ? new GuestStateMask(eabiCallee.EabiGprReadMask, eabiCallee.EabiFprReadMask, 0, false,
                            eabiCallee.EabiReadsCtr, false, false, 0, 0)
                        : reads;
                    eabiGprReads |= eabiRead.Gpr & ~written.Gpr;
                    eabiFprReads |= eabiRead.Fpr & ~written.Fpr;
                    eabiCtrRead |= eabiRead.Ctr && !written.Ctr;
                }
                Writes(instruction, calleeContracts, modSafeIndirect, out var possibleInstructionWrites, out var definiteInstructionWrites);
                possibleWrites = possibleWrites.Union(possibleInstructionWrites);
                written = written.Union(definiteInstructionWrites);
                if (instruction is IrCall call && GuestTargetParser.TryParseAddress(call.Target, out var target))
                {
                    if (!TryGetInlineThunk(target, out _, out _))
                        directTargets.Add(target);
                    if (calleeContracts?.TryGetValue(target, out var callee) == true)
                        boundaryFlags |= callee.BoundaryFlags;
                }
                else if (instruction is IrCall helperCall)
                {
                    boundaryFlags |= GuestEabi.HelperEffect(helperCall).BoundaryFlags;
                }
                if (instruction is IrUndefined ||
                    (!GuestEabi.Enabled && !modSafeIndirect && instruction is IrIndirectCall or IrIndirectJump))
                {
                    boundaryFlags |= GuestCallBoundaryFlags.RequiresCompleteContext;
                }
                else if (instruction is IrIndirectCall or IrIndirectJump)
                {
                    boundaryFlags |= GuestCallBoundaryFlags.InvokesGuestCode;
                }
            }
        }

        var definiteWrites = GuestStateMask.Empty;
        var sawExitBlock = false;
        for (var index = 0; index < blockCount; ++index)
        {
            var successorCount = Successors(blockList[index], out var first, out var second);
            if ((successorCount >= 1 && ordinals.ContainsKey(first)) ||
                (successorCount == 2 && ordinals.ContainsKey(second)))
                continue;
            definiteWrites = sawExitBlock ? definiteWrites.Intersect(definiteOut[index]) : definiteOut[index];
            sawExitBlock = true;
        }

        if (StackPreservedContracts && !GuestEabi.Enabled)
        {
            // Writes only: callers stop reloading mechanically proven stack-preserved registers (r1, r14-r31,
            // f14-f31, cr2-cr4) after the call while keeping ReadBeforeWrite intact so CpuContext stays
            // architecturally flushed on entry to callees, PoyoJIT/interpreter patches, and OSSaveContext.
            var preserved = GuestEabi.AnalyzePreservedFrameState(
                function,
                calleeContracts,
                trustIndirectCalls: modSafeIndirect,
                allowPsqFprPreservation: false);
            if (GuestEabi.IsNonLocalExit(function, preserved))
            {
                // C++ exception unwinder context restore (0x803F2948) jumps directly to the catch landing pad
                // and never returns to its caller; keep its ReadBeforeWrite intact so inputs flush, while not
                // polluting every function that can reach a throw (__dynamic_cast) with r1/r2/r13/NV writes.
                possibleWrites = GuestStateMask.Empty;
                definiteWrites = GuestStateMask.Empty;
            }
            else
            {
                // Writes only: callers stop reloading mechanically proven stack-preserved registers (r1, r14-r31,
                // f14-f31, cr2-cr4) while ReadBeforeWrite stays intact so CpuContext remains architecturally
                // flushed on entry to callees, Project+ C2 hooks, PoyoJIT patches, and OSSaveContext.
                possibleWrites = possibleWrites with
                {
                    Gpr = possibleWrites.Gpr & ~preserved.Gprs,
                    Fpr = possibleWrites.Fpr & ~preserved.Fprs,
                    Cr = (byte)(possibleWrites.Cr & ~preserved.CrFields)
                };
                definiteWrites = definiteWrites with
                {
                    Gpr = definiteWrites.Gpr & ~preserved.Gprs,
                    Fpr = definiteWrites.Fpr & ~preserved.Fprs,
                    Cr = (byte)(definiteWrites.Cr & ~preserved.CrFields)
                };
            }
        }

        const uint gprReturnMask = (1u << 3) | (1u << 4);
        const uint fprReturnMask = 1u << 1;
        return new GuestAbiContract(
            readBeforeWrite.Gpr,
            possibleWrites.Gpr,
            possibleWrites.Gpr & gprReturnMask,
            readBeforeWrite.Fpr,
            possibleWrites.Fpr,
            possibleWrites.Fpr & fprReturnMask,
            readBeforeWrite.Cr,
            possibleWrites.Cr,
            readBeforeWrite.Xer,
            possibleWrites.Xer,
            readBeforeWrite.Ctr,
            possibleWrites.Ctr,
            readBeforeWrite.Lr,
            possibleWrites.Lr,
            boundaryFlags,
            directTargets.ToArray())
        {
            ReadsFpscrBeforeWrite = readBeforeWrite.Fpscr,
            MayWriteFpscr = possibleWrites.Fpscr,
            GqrReadBeforeWriteMask = readBeforeWrite.Gqr,
            GqrPossibleWriteMask = possibleWrites.Gqr,
            HidReadBeforeWriteMask = readBeforeWrite.Hid,
            HidPossibleWriteMask = possibleWrites.Hid,
            GprDefiniteWriteMask = definiteWrites.Gpr,
            FprDefiniteWriteMask = definiteWrites.Fpr,
            CrDefiniteWriteMask = definiteWrites.Cr,
            DefinitelyWritesXer = definiteWrites.Xer,
            DefinitelyWritesCtr = definiteWrites.Ctr,
            DefinitelyWritesLr = definiteWrites.Lr,
            DefinitelyWritesFpscr = definiteWrites.Fpscr,
            GqrDefiniteWriteMask = definiteWrites.Gqr,
            HidDefiniteWriteMask = definiteWrites.Hid,
            EabiCrReadBeforeWriteMask = eabiCrReads,
            EabiGprReadMask = eabiGprReads,
            EabiFprReadMask = eabiFprReads,
            EabiReadsCtr = eabiCtrRead
        };
    }

    /// <summary>
    /// Successor count (0, 1, or 2) and labels for <paramref name="block"/>; count-plus-out-params
    /// keeps the CFG walk allocation-free.
    /// </summary>
    private static int Successors(IrBasicBlock block, out string first, out string second)
    {
        var instructions = block.Instructions;
        var terminator = instructions.Count == 0 ? null : instructions[instructions.Count - 1];
        if (terminator is IrBranch branch)
        {
            first = branch.TrueLabel;
            second = branch.FalseLabel;
            return 2;
        }

        if (terminator is IrJump jump)
        {
            first = jump.TargetLabel;
            second = string.Empty;
            return 1;
        }

        first = string.Empty;
        second = string.Empty;
        return 0;
    }

    private static GuestStateMask Reads(
        IrInstruction instruction,
        IReadOnlyDictionary<uint, GuestAbiContract>? calleeContracts)
    {
        var result = GuestStateMask.Empty;
        switch (instruction)
        {
            case IrAssign value: AddValue(ref result, value.Value); break;
            case IrBinary value: AddValue(ref result, value.Left); AddValue(ref result, value.Right); break;
            case IrLoad value: Add(ref result, value.Address.Base); break;
            case IrStore value: Add(ref result, value.Address.Base); AddValue(ref result, value.Source); break;
            case IrResolveGuestMemoryRange value: AddValue(ref result, value.Base); break;
            case IrResolvedLoad value: Add(ref result, value.OriginalAddress.Base); break;
            case IrResolvedStore value: Add(ref result, value.OriginalAddress.Base); AddValue(ref result, value.Source); break;
            case IrResolvedPsqLoad value:
                AddValue(ref result, value.OriginalAddress);
                if (value.KnownGqr is null || value.GuardKnownGqr) AddGqr(ref result, value.I);
                break;
            case IrResolvedPsqStore value:
                AddValue(ref result, value.OriginalAddress); AddValue(ref result, value.Source);
                if (value.KnownGqr is null || value.GuardKnownGqr) AddGqr(ref result, value.I);
                break;
            case IrResolvedLoadPair value:
                Add(ref result, value.FirstOriginalAddress.Base);
                Add(ref result, value.SecondOriginalAddress.Base);
                break;
            case IrResolvedStorePair value:
                Add(ref result, value.FirstOriginalAddress.Base);
                Add(ref result, value.SecondOriginalAddress.Base);
                AddValue(ref result, value.FirstSource);
                AddValue(ref result, value.SecondSource);
                break;
            case IrCall value:
                if (GuestTargetParser.TryParseAddress(value.Target, out var directTarget) &&
                    calleeContracts?.TryGetValue(directTarget, out var callee) == true)
                {
                    // A decoded call carries the volatile ABI registers as generic IR arguments;
                    // once the callee has a contract, only state it actually consumes is a real read.
                    result = result.Union(GuestStateMask.FromContractReads(callee));
                }
                else if (GuestTargetParser.TryParseAddress(value.Target, out var readThunkAddress) &&
                    TryGetInlineThunk(readThunkAddress, out var readThunk, out var readStart) &&
                    readThunk is InlineThunkKind.SaveGpr or InlineThunkKind.SaveFpr)
                {
                    AddRegisterRangeToTop(ref result, readThunk == InlineThunkKind.SaveGpr, readStart);
                }
                else if (!GuestTargetParser.TryParseAddress(value.Target, out _))
                {
                    foreach (var argument in value.Arguments) AddValue(ref result, argument);
                    result = result.Union(HiddenReads(GuestEabi.HelperEffect(value)));
                }
                else if (GuestEabi.Enabled)
                {
                    // MKW_EABI_REGS: an unresolved target is an EABI function like an indirect one.
                    result = result.Union(GuestEabi.UnknownCallReads);
                }
                else
                {
                    // An unresolved direct/native target has no stronger
                    // contract, so its explicitly modelled ABI arguments remain
                    // observable inputs.
                    foreach (var argument in value.Arguments) AddValue(ref result, argument);
                }
                break;
            case IrIndirectCall value:
                AddValue(ref result, value.Target);
                // MKW_EABI_REGS: the target is an EABI function (or tail-calls one) that reads only its arguments
                // (the IR's generic argument list also names f9-f13).
                if (GuestEabi.Enabled)
                {
                    result = result.Union(GuestEabi.UnknownCallReads);
                    break;
                }
                foreach (var argument in value.Arguments) AddValue(ref result, argument);
                result = result.Union(FullState);
                break;
            case IrIndirectJump value:
                AddValue(ref result, value.Target);
                result = result.Union(GuestEabi.Enabled ? GuestEabi.UnknownCallReads : FullState);
                break;
            case IrSetCrField value:
                AddValue(ref result, value.Left);
                AddValue(ref result, value.Right);
                // Integer comparisons copy XER.SO into the CR field.
                result = result with { Xer = true };
                break;
            case IrPhi value: foreach (var source in value.Sources.Values) Add(ref result, source); break;
            case IrBranch value:
                Add(ref result, value.ConditionRegister);
                result = result.Union(SpecialRegistersInRawExpression(value.ConditionRegister));
                break;
            case IrJumpTable value: Add(ref result, value.Selector); break;
            case IrReturn { Value: { } value }: AddValue(ref result, value); break;
        }
        return result;
    }

    /// <summary>
    /// Classifies both write flavours of an instruction in one pass: <paramref name="definite"/> is
    /// guaranteed, <paramref name="possible"/> is what it may write. Only call cases differ.
    /// </summary>
    private static void Writes(
        IrInstruction instruction,
        IReadOnlyDictionary<uint, GuestAbiContract>? calleeContracts,
        bool modSafeIndirect,
        out GuestStateMask possible,
        out GuestStateMask definite)
    {
        var shared = GuestStateMask.Empty;
        var possibleOnly = GuestStateMask.Empty;
        var definiteOnly = GuestStateMask.Empty;
        switch (instruction)
        {
            case IrAssign value: Add(ref shared, value.Destination); break;
            case IrBinary value: Add(ref shared, value.Destination); break;
            case IrLoad value: Add(ref shared, value.Destination); break;
            case IrResolveGuestMemoryRange value: Add(ref shared, value.Destination); break;
            case IrResolvedLoad value: Add(ref shared, value.Destination); break;
            case IrResolvedPsqLoad value: Add(ref shared, value.Destination); break;
            case IrResolvedLoadPair value:
                Add(ref shared, value.FirstDestination);
                Add(ref shared, value.SecondDestination);
                break;
            case IrCall value:
                var helperEffect = !GuestTargetParser.TryParseAddress(value.Target, out _)
                    ? GuestEabi.HelperEffect(value)
                    : null;
                var destination = GuestStateMask.FromRegister(value.Destination);
                // CR helpers return the complete packed CR only because that is
                // the runtime helper ABI. Architecturally they change the fields
                // described by the helper contract, not all eight fields.
                if (!(destination.Cr == byte.MaxValue && helperEffect?.CrWriteMask != 0))
                    shared = shared.Union(destination);
                if (GuestTargetParser.TryParseAddress(value.Target, out var directTarget) &&
                    calleeContracts?.TryGetValue(directTarget, out var callee) == true)
                {
                    possibleOnly = GuestStateMask.FromContractWrites(callee);
                    definiteOnly = GuestStateMask.FromContractDefiniteWrites(callee);
                }
                else if (GuestTargetParser.TryParseAddress(value.Target, out var writeThunkAddress) &&
                    TryGetInlineThunk(writeThunkAddress, out var writeThunk, out var writeStart) &&
                    writeThunk is InlineThunkKind.RestGpr or InlineThunkKind.RestFpr)
                {
                    AddRegisterRangeToTop(ref shared, writeThunk == InlineThunkKind.RestGpr, writeStart);
                }
                else if (helperEffect is not null)
                {
                    // Hidden helper state is a possible write only: the definite
                    // flavour deliberately ignores non-guest call targets.
                    possibleOnly = HiddenWrites(helperEffect);
                }
                else if (GuestEabi.Enabled &&
                         !(GuestTargetParser.TryParseAddress(value.Target, out var saveThunkAddress) &&
                           TryGetInlineThunk(saveThunkAddress, out _, out _)))
                {
                    // MKW_EABI_REGS: an unresolved target may change any volatile register.
                    possibleOnly = GuestEabi.UnknownCallWrites;
                }
                break;
            case IrIndirectCall value:
                Add(ref shared, value.Destination);
                if (GuestEabi.Enabled) possibleOnly = GuestEabi.UnknownCallWrites;
                else if (modSafeIndirect) possibleOnly = ModSafeIndirectWrites;
                else shared = shared.Union(FullState);
                break;
            case IrIndirectJump when GuestEabi.Enabled:
                // A tail dispatch: the target's results (r3/r4/f1/f2, volatile scratch) become this function's.
                possibleOnly = GuestEabi.UnknownCallWrites;
                break;
            case IrIndirectJump when modSafeIndirect:
                possibleOnly = ModSafeIndirectWrites;
                break;
            case IrSetCrField value:
                shared = shared with { Cr = (byte)(shared.Cr | (1 << value.FieldIndex)) };
                break;
            // A phi only merges values its incoming edges already wrote; counting it as a write made merges
            // of untouched r2/r13/cr/non-volatiles look like spurious writes of those registers.
            case IrPhi:
                break;
            case IrUndefined:
                shared = FullState;
                break;
        }

        possible = shared.Union(possibleOnly);
        definite = shared.Union(definiteOnly);
    }

    /// <summary>
    /// Unions <c>rN..r31</c> or <c>fN..f31</c>, the register window an inline
    /// save/restore thunk touches.
    /// </summary>
    private static void AddRegisterRangeToTop(ref GuestStateMask set, bool general, int startRegister)
    {
        var mask = uint.MaxValue << startRegister;
        set = general
            ? set with { Gpr = set.Gpr | mask }
            : set with { Fpr = set.Fpr | mask };
    }

    private static void Add(ref GuestStateMask set, string? name) => set = set.Union(GuestStateMask.FromRegister(name));

    private static void AddValue(ref GuestStateMask set, IrValue value)
    {
        if (value.Kind == "register") Add(ref set, value.RegisterName);
    }

    private static void AddGqr(ref GuestStateMask set, uint index)
    {
        // Mirrors GuestStateMask.FromRegister("gqr{index}"): only the eight architectural
        // quantization registers exist, anything else names nothing.
        if (index <= 7) set = set with { Gqr = (byte)(set.Gqr | (1 << (int)index)) };
    }

    private static GuestStateMask HiddenReads(GuestHelperEffect effect) =>
        new(effect.GprReadMask, effect.FprReadMask, effect.CrReadMask,
            effect.ReadsXer, effect.ReadsCtr, effect.ReadsLr, false, 0, 0);

    private static GuestStateMask HiddenWrites(GuestHelperEffect effect) =>
        new(effect.GprWriteMask, effect.FprWriteMask, effect.CrWriteMask,
            effect.WritesXer, effect.WritesCtr, effect.WritesLr, false, 0, 0);

    private static GuestStateMask FullState =>
        new(uint.MaxValue, uint.MaxValue, byte.MaxValue, true, true, true, true, byte.MaxValue, byte.MaxValue);

    /// <summary>
    /// Mod-safe and JIT-safe possible writes of an indirect call or tail jump: all PowerPC EABI volatile
    /// registers, FPSCR, and GQRs, leaving r1, r2, r13, r14-r31, f14-f31 and cr2-cr4 preserved.
    /// </summary>
    private static GuestStateMask ModSafeIndirectWrites =>
        new(GuestEabi.VolatileGprMask, GuestEabi.VolatileFprMask, GuestEabi.VolatileCrFields,
            true, true, true, true, byte.MaxValue, 0);

    private static GuestStateMask SpecialRegistersInRawExpression(string expression) =>
        new(
            0,
            0,
            expression.Contains("ctx->cr", StringComparison.Ordinal) ||
            expression.Contains("GetCRBit(ctx", StringComparison.Ordinal) ? byte.MaxValue : (byte)0,
            expression.Contains("ctx->xer", StringComparison.Ordinal),
            expression.Contains("ctx->ctr", StringComparison.Ordinal),
            expression.Contains("ctx->lr", StringComparison.Ordinal),
            expression.Contains("ctx->fpscr", StringComparison.Ordinal),
            0,
            (byte)((expression.Contains("ctx->hid0", StringComparison.Ordinal) ? 1 : 0) |
                   (expression.Contains("ctx->hid1", StringComparison.Ordinal) ? 2 : 0) |
                   (expression.Contains("ctx->hid2", StringComparison.Ordinal) ? 4 : 0)));

    // MKW_STACK_PRESERVED_CONTRACTS (default ON, set =0 to opt out): non-volatile registers (r1, r14-r31,
    // f14-f31, cr2-cr4) mechanically proven to be stack-saved and restored before every reachable exit are
    // removed from possible/definite writes so callers stop reloading them after direct calls.
    public static readonly bool StackPreservedContracts =
        Environment.GetEnvironmentVariable("MKW_STACK_PRESERVED_CONTRACTS") != "0";

    // MKW_MODSAFE_INDIRECT_ABI (default ON, set =0 to opt out): indirect calls/jumps keep architectural
    // full-context reads/flushes (safe for PoyoJIT, C2 hooks, and OSSaveContext) while treating r1, r2, r13,
    // r14-r31, f14-f31, and cr2-cr4 as preserved across indirect boundaries. The Gecko code handler
    // (0x800018A8) is excluded because P+ C0 Gecko codes mutate r15 across its indirect call.
    public static readonly bool ModSafeIndirectAbi =
        Environment.GetEnvironmentVariable("MKW_MODSAFE_INDIRECT_ABI") != "0";

    internal const uint GeckoCodeHandlerAddress = 0x800018A8u;

    internal static bool IsGeckoCodeHandler(IrFunction function)
    {
        if (GuestTargetParser.TryParseAddress(function.EntryLabel, out var entryAddr) &&
            entryAddr == GeckoCodeHandlerAddress)
            return true;
        return GuestTargetParser.TryParseAddress(function.Name, out var nameAddr) &&
               nameAddr == GeckoCodeHandlerAddress;
    }

    internal static bool IsModSafeIndirectFunction(IrFunction function) =>
        ModSafeIndirectAbi && !IsGeckoCodeHandler(function);

    internal enum InlineThunkKind { SaveGpr, RestGpr, SaveFpr, RestFpr }

    internal static bool TryGetInlineThunk(uint address, out InlineThunkKind kind, out int startRegister)
    {
        var thunks = GuestSaveRestoreThunks.Current;
        if (TryRange(address, thunks.SaveGpr, out startRegister)) { kind = InlineThunkKind.SaveGpr; return true; }
        if (TryRange(address, thunks.RestGpr, out startRegister)) { kind = InlineThunkKind.RestGpr; return true; }
        if (TryRange(address, thunks.SaveFpr, out startRegister)) { kind = InlineThunkKind.SaveFpr; return true; }
        if (TryRange(address, thunks.RestFpr, out startRegister)) { kind = InlineThunkKind.RestFpr; return true; }
        kind = default;
        startRegister = -1;
        return false;

        static bool TryRange(uint candidate, GuestSaveRestoreThunkRange? range, out int start)
        {
            if (range is null)
            {
                start = -1;
                return false;
            }

            var (baseAddress, firstRegister, lastStartRegister) = range;
            var lastAddress = baseAddress + checked((uint)((lastStartRegister - firstRegister) * 4));
            if (candidate < baseAddress || candidate > lastAddress || ((candidate - baseAddress) & 3u) != 0)
            {
                start = -1;
                return false;
            }
            start = firstRegister + checked((int)((candidate - baseAddress) / 4));
            return true;
        }
    }
}
