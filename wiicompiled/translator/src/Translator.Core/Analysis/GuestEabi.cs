using System;
using System.Collections.Generic;
using System.Linq;
using Translator.Core.Ir;

namespace Translator.Core.Analysis;

/// <summary>
/// MKW_EABI_REGS=1: trust the PowerPC EABI at translated call boundaries. VANILLA ONLY: mod code (P+ Gecko codes)
/// breaks the EABI (it switches code lists by changing r15 across an indirect call), so the mode is opt-in.
/// <para>A function is <em>conformant</em> when its own body shows that it restores r1 and every nonvolatile register
/// it (or a non-conformant callee) writes, reads a nonvolatile register only to save it in its frame, never reads
/// r0/r11/r12/f0/f9-f13/CTR before writing them, and never writes r2/r13, touches supervisor state, makes a system call
/// or contains an untranslatable instruction. Its callers then pass only r1, r2, r13, r3-r10, f1-f8 (and CR when it
/// reads cr1, the varargs FP-argument flag) and see only r3, r4, f1, f2 change; every other register stays in their
/// host locals across the call. Non-conformant functions ("violators") keep their exact contract, minus the
/// nonvolatile registers they provably restore.</para>
/// <para>Consequence: CpuContext's copies of r14-r31, f14-f31 and cr2-cr4 stop being architectural. No call site
/// reloads a register its callee preserves, so nothing translated reads them back; a native or guest routine that
/// reads the whole register file (OSSaveContext) records stale values there.</para>
/// </summary>
/// <summary>
/// MKW_EABI_REGS register-passing entry of a conformant function: <c>Symbol(CpuContext* ctx, inputs...)</c> takes
/// r1/r3-r10 and f1-f8 as host arguments and returns the r3/r4/f1/f2 values its direct callers read (one: a uint64_t;
/// two: a MkwEabiPair in rax:rdx; three or four: a MkwEabiQuad in the caller's frame). Everything else (r2/r13, CR,
/// XER, LR, FPSCR, GQRs) stays in CpuContext.
/// </summary>
public sealed record GuestEabiVariant(
    uint Address, string Symbol, uint InputGprMask, uint InputFprMask, uint OutputGprMask, uint OutputFprMask)
{
    public int OutputCount => System.Numerics.BitOperations.PopCount(OutputGprMask) +
                              System.Numerics.BitOperations.PopCount(OutputFprMask);
}

public static class GuestEabi
{
    public static readonly bool Enabled = Environment.GetEnvironmentVariable("MKW_EABI_REGS") == "1";

    /// <summary>Volatile non-argument registers read at entry are EABI inputs (see <see cref="Mask"/>).</summary>
    public static readonly bool ScratchInputs = Environment.GetEnvironmentVariable("MKW_EABI_SCRATCH_INPUTS") != "0";

    /// <summary>MKW_EABI_VARIANTS=0 keeps MKW_EABI_REGS to EABI contracts only (no register-passing entries).</summary>
    public static readonly bool VariantsEnabled = Enabled && Environment.GetEnvironmentVariable("MKW_EABI_VARIANTS") != "0";

    public const uint ArgumentGprMask = 0x7F8u;                                  // r3..r10
    public const uint InputGprMask = ArgumentGprMask | (1u << 1) | (1u << 2) | (1u << 13);
    public const uint ArgumentFprMask = 0x1FEu;                                  // f1..f8
    public const uint OutputGprMask = (1u << 3) | (1u << 4);
    public const uint OutputFprMask = (1u << 1) | (1u << 2);
    public const uint NonvolatileGprMask = 0xFFFFC000u;                          // r14..r31
    public const uint NonvolatileFprMask = 0xFFFFC000u;                          // f14..f31
    /// <summary>r1, r2, r13 and r14-r31: what every conformant call leaves unchanged in the caller.</summary>
    public const uint PreservedGprMask = NonvolatileGprMask | (1u << 1) | (1u << 2) | (1u << 13);
    public const byte NonvolatileCrFields = 0x1C;                                // cr2..cr4
    /// <summary>cr2..cr4 as bits of the packed 32-bit CR (cr0 is the top nibble).</summary>
    public const uint NonvolatileCrBits = 0x00FFF000u;
    public const byte VarargCrField = 0x02;                                      // cr1
    public const uint ScratchGprMask = 1u | (1u << 11) | (1u << 12);             // r0, r11, r12
    public const uint ScratchFprMask = 1u | (0x1Fu << 9);                        // f0, f9..f13
    /// <summary>What an EABI call may change: every volatile register.</summary>
    public const uint VolatileGprMask = 1u | 0x1FF8u;                            // r0, r3..r12
    public const uint VolatileFprMask = 0x3FFFu;                                 // f0..f13
    public const byte VolatileCrFields = 0xE3;                                   // cr0, cr1, cr5..cr7

    /// <summary>
    /// Natives that replace the register file. OSLoadContext (0x801D80B4) loads all GPRs from an OSContext and jumps
    /// to its SRR0; every other native reads its arguments and writes r3/r4/f1 (checked by mask below).
    /// </summary>
    private static readonly HashSet<uint> ViolatingNatives = new() { 0x801D80B4u };

    /// <summary>Translated routines that leave into another frame (see <see cref="Classify"/>).</summary>
    private static readonly HashSet<uint> NonLocalExits = new() { 0x803F2948u };

    /// <summary>mfcr (a CR save or bool extraction) and mtcrf's merge read: not genuine CR field inputs.</summary>
    public static bool IsWholeCrSaveOrMerge(IrInstruction instruction) =>
        instruction switch
        {
            IrAssign { Value: { Kind: "register", RegisterName: { } source } } =>
                RegisterNameUtils.StripNumericSuffix(source) == "cr",
            IrBinary binary => binary.Destination.Contains("_mtcrf_preserve", StringComparison.Ordinal),
            _ => false
        };

    /// <summary>
    /// The helper effect catalog's answer, except that MKW_EABI_REGS knows ps_mr (PPC_PsMr, a register move whose
    /// operands are explicit) is pure. The catalog leaves it out, so each of its call sites counted as a complete
    /// boundary that reads and writes every register, which made every ancestor look like it read r0/r11/r12/CTR and
    /// the caller's nonvolatiles and wrote r2/r13. Codegen paths outside the contracts keep the catalog's answer.
    /// </summary>
    public static GuestHelperEffect HelperEffect(IrCall call) =>
        Enabled && string.Equals(call.Target, "PPC_PsMr", StringComparison.OrdinalIgnoreCase)
            ? PureHelper
            : GuestHelperEffectCatalog.Analyze(call);

    private static readonly GuestHelperEffect PureHelper = new();

    /// <summary>Reads of an indirect call or jump in EABI mode: the argument registers, r1/r2/r13 and cr1.</summary>
    public static GuestStateMask UnknownCallReads =>
        new(InputGprMask, ArgumentFprMask, VarargCrField, false, false, true, false, 0, 0);

    /// <summary>Possible writes of an indirect call in EABI mode: every volatile register.</summary>
    public static GuestStateMask UnknownCallWrites =>
        new(VolatileGprMask, VolatileFprMask, VolatileCrFields, true, true, true, false, 0, 0);

    /// <summary>The EABI view of a conformant function's contract.</summary>
    public static GuestAbiContract Mask(GuestAbiContract precise) => precise with
    {
        // Phi-free reads: an argument register merged unchanged into the exit value is not an input. Volatile
        // non-argument registers read at entry (r12 for __ptmf_scall, the old f10/f12 that ps_sum0 keeps in PS1,
        // CTR) are inputs too, so callers publish whatever they hold there, as the CPU would.
        GprReadBeforeWriteMask = precise.EabiGprReadMask & (InputGprMask | ScratchGprMask),
        EabiGprReadMask = precise.EabiGprReadMask & (InputGprMask | ScratchGprMask),
        GprPossibleWriteMask = precise.GprPossibleWriteMask & OutputGprMask,
        GprReturnMask = precise.GprPossibleWriteMask & OutputGprMask,
        FprReadBeforeWriteMask = precise.EabiFprReadMask & (ArgumentFprMask | ScratchFprMask),
        EabiFprReadMask = precise.EabiFprReadMask & (ArgumentFprMask | ScratchFprMask),
        FprPossibleWriteMask = precise.FprPossibleWriteMask & OutputFprMask,
        FprReturnMask = precise.FprPossibleWriteMask & OutputFprMask,
        CrReadBeforeWriteMask = (byte)(precise.EabiCrReadBeforeWriteMask & VarargCrField),
        CrPossibleWriteMask = 0,
        ReadsXerBeforeWrite = false,
        MayWriteXer = false,
        ReadsCtrBeforeWrite = precise.EabiReadsCtr,
        MayWriteCtr = false,
        BoundaryFlags = precise.BoundaryFlags & ~GuestCallBoundaryFlags.RequiresCompleteContext,
        GprDefiniteWriteMask = precise.GprDefiniteWriteMask & OutputGprMask,
        FprDefiniteWriteMask = precise.FprDefiniteWriteMask & OutputFprMask,
        CrDefiniteWriteMask = 0,
        DefinitelyWritesXer = false,
        DefinitelyWritesCtr = false,
        EabiConformant = true
    };

    /// <summary>
    /// Checks every hand-written declaration of a register-passing entry under <paramref name="runtimeSourceDirectory"/>
    /// (<c>RET func_XXXXXXXX_eabi(CpuContext*, uint32_t..., PPC_FPR...)</c>, comments and disabled blocks ignored)
    /// against the entry the translation emits: it must exist, return void / uint64_t / MkwEabiPair for 0 / 1 / 2
    /// results (a MkwEabiQuad travels through memory, which no hand-written caller expects) and list as many uint32_t
    /// and then PPC_FPR parameters as it has GPR and FPR inputs. Throws on any mismatch, so a resized entry fails the
    /// translation instead of miscompiling its native caller (g3d_calc.cpp, mtx.cpp and game_natives.cpp call six).
    /// </summary>
    public static int CheckHandWrittenPrototypes(
        string runtimeSourceDirectory, IReadOnlyDictionary<uint, GuestEabiVariant> variants)
    {
        var problems = new List<string>();
        var checkedCount = 0;
        foreach (var source in NativeSourceParsing.ReadDirectory(runtimeSourceDirectory))
        {
            foreach (System.Text.RegularExpressions.Match match in System.Text.RegularExpressions.Regex.Matches(
                         source.Content,
                         @"\b(?<ret>void|uint64_t|MkwEabiPair|MkwEabiQuad)\s+func_(?<addr>[0-9A-Fa-f]{8})_eabi\s*\((?<params>[^)]*)\)"))
            {
                ++checkedCount;
                var address = Convert.ToUInt32(match.Groups["addr"].Value, 16);
                var where = $"{source.RelativePath}: func_{address:X8}_eabi";
                if (!variants.TryGetValue(address, out var variant))
                {
                    problems.Add($"{where}: the translation emits no such entry");
                    continue;
                }
                var types = match.Groups["params"].Value.Split(',')
                    .Select(static parameter => parameter.Trim())
                    .Select(static parameter => parameter.StartsWith("CpuContext", StringComparison.Ordinal) ? "CpuContext*"
                        : parameter.StartsWith("uint32_t", StringComparison.Ordinal) ? "uint32_t"
                        : parameter.StartsWith("PPC_FPR", StringComparison.Ordinal) ? "PPC_FPR"
                        : parameter)
                    .ToArray();
                var expected = new[] { "CpuContext*" }
                    .Concat(Enumerable.Repeat("uint32_t", System.Numerics.BitOperations.PopCount(variant.InputGprMask)))
                    .Concat(Enumerable.Repeat("PPC_FPR", System.Numerics.BitOperations.PopCount(variant.InputFprMask)))
                    .ToArray();
                var expectedReturn = variant.OutputCount switch { 0 => "void", 1 => "uint64_t", 2 => "MkwEabiPair", _ => "MkwEabiQuad" };
                if (match.Groups["ret"].Value != expectedReturn)
                    problems.Add($"{where}: returns {match.Groups["ret"].Value}, the entry returns {expectedReturn}");
                if (!types.SequenceEqual(expected))
                    problems.Add($"{where}: parameters ({string.Join(", ", types)}), the entry takes ({string.Join(", ", expected)})");
                if (variant.OutputCount > 2)
                    problems.Add($"{where}: the entry returns three or four results through memory");
            }
        }
        if (problems.Count != 0)
            throw new InvalidOperationException(
                "Hand-written callers of register-passing entries no longer match them:\n  " + string.Join("\n  ", problems));
        return checkedCount;
    }

    /// <summary>
    /// The least contract covering both: reads and possible writes unioned, definite writes intersected, conformant
    /// only if both are. Conservative for every consumer (callers sync a superset, never assume a write happened).
    /// </summary>
    public static GuestAbiContract Join(GuestAbiContract a, GuestAbiContract b) => b with
    {
        GprReadBeforeWriteMask = a.GprReadBeforeWriteMask | b.GprReadBeforeWriteMask,
        GprPossibleWriteMask = a.GprPossibleWriteMask | b.GprPossibleWriteMask,
        GprReturnMask = a.GprReturnMask | b.GprReturnMask,
        FprReadBeforeWriteMask = a.FprReadBeforeWriteMask | b.FprReadBeforeWriteMask,
        FprPossibleWriteMask = a.FprPossibleWriteMask | b.FprPossibleWriteMask,
        FprReturnMask = a.FprReturnMask | b.FprReturnMask,
        CrReadBeforeWriteMask = (byte)(a.CrReadBeforeWriteMask | b.CrReadBeforeWriteMask),
        CrPossibleWriteMask = (byte)(a.CrPossibleWriteMask | b.CrPossibleWriteMask),
        ReadsXerBeforeWrite = a.ReadsXerBeforeWrite || b.ReadsXerBeforeWrite,
        MayWriteXer = a.MayWriteXer || b.MayWriteXer,
        ReadsCtrBeforeWrite = a.ReadsCtrBeforeWrite || b.ReadsCtrBeforeWrite,
        MayWriteCtr = a.MayWriteCtr || b.MayWriteCtr,
        ReadsLrBeforeWrite = a.ReadsLrBeforeWrite || b.ReadsLrBeforeWrite,
        MayWriteLr = a.MayWriteLr || b.MayWriteLr,
        BoundaryFlags = a.BoundaryFlags | b.BoundaryFlags,
        ReadsFpscrBeforeWrite = a.ReadsFpscrBeforeWrite || b.ReadsFpscrBeforeWrite,
        MayWriteFpscr = a.MayWriteFpscr || b.MayWriteFpscr,
        GqrReadBeforeWriteMask = (byte)(a.GqrReadBeforeWriteMask | b.GqrReadBeforeWriteMask),
        GqrPossibleWriteMask = (byte)(a.GqrPossibleWriteMask | b.GqrPossibleWriteMask),
        HidReadBeforeWriteMask = (byte)(a.HidReadBeforeWriteMask | b.HidReadBeforeWriteMask),
        HidPossibleWriteMask = (byte)(a.HidPossibleWriteMask | b.HidPossibleWriteMask),
        GprDefiniteWriteMask = a.GprDefiniteWriteMask & b.GprDefiniteWriteMask,
        FprDefiniteWriteMask = a.FprDefiniteWriteMask & b.FprDefiniteWriteMask,
        CrDefiniteWriteMask = (byte)(a.CrDefiniteWriteMask & b.CrDefiniteWriteMask),
        DefinitelyWritesXer = a.DefinitelyWritesXer && b.DefinitelyWritesXer,
        DefinitelyWritesCtr = a.DefinitelyWritesCtr && b.DefinitelyWritesCtr,
        DefinitelyWritesLr = a.DefinitelyWritesLr && b.DefinitelyWritesLr,
        DefinitelyWritesFpscr = a.DefinitelyWritesFpscr && b.DefinitelyWritesFpscr,
        GqrDefiniteWriteMask = (byte)(a.GqrDefiniteWriteMask & b.GqrDefiniteWriteMask),
        HidDefiniteWriteMask = (byte)(a.HidDefiniteWriteMask & b.HidDefiniteWriteMask),
        EabiCrReadBeforeWriteMask = (byte)(a.EabiCrReadBeforeWriteMask | b.EabiCrReadBeforeWriteMask),
        EabiGprReadMask = a.EabiGprReadMask | b.EabiGprReadMask,
        EabiFprReadMask = a.EabiFprReadMask | b.EabiFprReadMask,
        EabiReadsCtr = a.EabiReadsCtr || b.EabiReadsCtr,
        EabiConformant = a.EabiConformant && b.EabiConformant
    };

    /// <summary>Classification result for one translated function; <see cref="Reason"/> is null when conformant.</summary>
    public sealed record Classification(GuestAbiContract Contract, string? Reason);

    /// <summary>
    /// Classifies <paramref name="function"/> from its exact contract (computed against <paramref name="callees"/>,
    /// whose conformant members are already masked) and returns the contract its callers should see.
    /// </summary>
    public static Classification Classify(
        IrFunction function, GuestAbiContract precise, IReadOnlyDictionary<uint, GuestAbiContract>? callees,
        string? previousReason = null)
    {
        var facts = FrameFacts.Analyze(function, callees);
        var violatorContract = precise with
        {
            GprPossibleWriteMask = precise.GprPossibleWriteMask & ~facts.PreservedGprs,
            GprDefiniteWriteMask = precise.GprDefiniteWriteMask & ~facts.PreservedGprs,
            GprReturnMask = precise.GprReturnMask & ~facts.PreservedGprs,
            FprPossibleWriteMask = precise.FprPossibleWriteMask & ~facts.PreservedFprs,
            FprDefiniteWriteMask = precise.FprDefiniteWriteMask & ~facts.PreservedFprs,
            FprReturnMask = precise.FprReturnMask & ~facts.PreservedFprs,
            // mtcrf in compiled code restores the cr2-cr4 fields its prologue saved with mfcr.
            CrPossibleWriteMask = facts.HasMtcrf
                ? (byte)(precise.CrPossibleWriteMask & ~NonvolatileCrFields)
                : precise.CrPossibleWriteMask,
            // Nonvolatiles a violator reads only to save them in its frame are no input its callers must supply.
            EabiGprReadMask = precise.EabiGprReadMask & ~facts.SaveOnlyGprs,
            EabiFprReadMask = precise.EabiFprReadMask & ~facts.SaveOnlyFprs
        };

        string? reason = facts.IntrinsicViolation;
        // MKW_EABI_SCRATCH_INPUTS=0: reading r0/r11/r12/f0/f9-f13/CTR at entry makes a violator, as originally.
        if (!ScratchInputs && reason is null && (precise.EabiGprReadMask & ScratchGprMask) != 0)
            reason = $"reads scratch GPR(s) {Gprs(precise.EabiGprReadMask & ScratchGprMask)} at entry";
        if (!ScratchInputs && reason is null && (precise.EabiFprReadMask & ScratchFprMask) != 0)
            reason = $"reads scratch FPR(s) {Fprs(precise.EabiFprReadMask & ScratchFprMask)} at entry";
        if (!ScratchInputs && reason is null && precise.EabiReadsCtr)
            reason = "reads CTR at entry";
        if (reason is null && (precise.GprPossibleWriteMask & ((1u << 2) | (1u << 13))) != 0)
            reason = "writes r2/r13";
        if (reason is null && (precise.GprPossibleWriteMask & (1u << 1)) != 0 && (facts.PreservedGprs & (1u << 1)) == 0)
            reason = "does not restore r1";
        var clobberedGprs = precise.GprPossibleWriteMask & NonvolatileGprMask & ~facts.PreservedGprs;
        if (reason is null && clobberedGprs != 0)
            reason = $"does not restore {Gprs(clobberedGprs)}";
        var clobberedFprs = precise.FprPossibleWriteMask & NonvolatileFprMask & ~facts.PreservedFprs;
        if (reason is null && clobberedFprs != 0)
            reason = $"does not restore {Fprs(clobberedFprs)}";
        var genuineGprReads = precise.EabiGprReadMask & NonvolatileGprMask & ~facts.SaveOnlyGprs;
        if (reason is null && genuineGprReads != 0)
            reason = $"reads caller's {Gprs(genuineGprReads)}";
        var genuineFprReads = precise.EabiFprReadMask & NonvolatileFprMask & ~facts.SaveOnlyFprs;
        if (reason is null && genuineFprReads != 0)
            reason = $"reads caller's {Fprs(genuineFprReads)}";
        if (reason is null && (precise.CrPossibleWriteMask & NonvolatileCrFields) != 0 && !facts.HasMtcrf)
            reason = "writes cr2-cr4 without mtcrf";
        reason ??= previousReason;

        // A non-local exit: the C++ unwinder's context restore (0x803F2948) loads r1, r2/r13, the nonvolatiles and LR
        // from a record and returns into the landing pad's frame, so no caller resumes after it with its registers.
        // Publishing its writes made every function that can reach a throw (__dynamic_cast) write r2/r13. Only this
        // routine: __init_registers (0x800041A8) also loads r1/r2/r13 but returns to __start, which needs them.
        if (NonLocalExits.Contains(function.Name is { } name && GuestTargetParser.TryParseAddress(name, out var self) ? self : 0) &&
            (facts.PreservedGprs & (1u << 1)) == 0 && facts.WritesSdaBase)
        {
            return new Classification(violatorContract with
            {
                EabiConformant = false,
                GprPossibleWriteMask = 0, GprDefiniteWriteMask = 0, GprReturnMask = 0,
                FprPossibleWriteMask = 0, FprDefiniteWriteMask = 0, FprReturnMask = 0,
                CrPossibleWriteMask = 0, CrDefiniteWriteMask = 0,
                MayWriteXer = false, MayWriteCtr = false, DefinitelyWritesXer = false, DefinitelyWritesCtr = false
            }, "non-local exit (loads r1 and r2/r13)");
        }

        if (DebugAddresses.Count != 0 && GuestTargetParser.TryParseAddress(function.Name, out var debugAddress) &&
            DebugAddresses.Contains(debugAddress))
        {
            Console.WriteLine(
                $"[eabi-debug] {function.Name}: {reason ?? "conformant"}; gprRead {precise.GprReadBeforeWriteMask:X8} " +
                $"eabiGprRead {precise.EabiGprReadMask:X8} gprWrite {precise.GprPossibleWriteMask:X8} fprWrite {precise.FprPossibleWriteMask:X8} " +
                $"crWrite {precise.CrPossibleWriteMask:X2} preserved {facts.PreservedGprs:X8}/{facts.PreservedFprs:X8} " +
                $"saveOnly {facts.SaveOnlyGprs:X8} mtcrf {facts.HasMtcrf} callees " + string.Join(" ", precise.DirectCallTargets
                    .Select(target => callees?.TryGetValue(target, out var c) == true
                        ? $"{target:X8}({(c.EabiConformant ? "ok" : "V")} w{c.GprPossibleWriteMask:X8} cr{c.CrPossibleWriteMask:X2} r{c.EabiGprReadMask:X8})"
                        : $"{target:X8}(?)")));
        }

        return reason is null
            ? new Classification(Mask(precise), null)
            : new Classification(violatorContract with { EabiConformant = false }, reason);
    }

    /// <summary>MKW_EABI_DEBUG=ADDR,ADDR: print the classification inputs of these functions (every round).</summary>
    private static readonly HashSet<uint> DebugAddresses =
        (Environment.GetEnvironmentVariable("MKW_EABI_DEBUG") ?? string.Empty)
        .Split(',', StringSplitOptions.RemoveEmptyEntries)
        .Select(static text => Convert.ToUInt32(text.Trim(), 16)).ToHashSet();

    /// <summary>
    /// The EABI view of a registered native's contract. Natives are host functions: they read their arguments from
    /// CpuContext and write their results back. Ones whose contract is exact but reaches past the EABI keep it.
    /// </summary>
    public static (GuestAbiContract Contract, string? Reason) ClassifyNative(uint address, GuestAbiContract precise)
    {
        // A native's body analysis has no phis: its exact reads are its genuine reads.
        precise = precise with
        {
            EabiGprReadMask = precise.GprReadBeforeWriteMask,
            EabiFprReadMask = precise.FprReadBeforeWriteMask,
            EabiReadsCtr = precise.ReadsCtrBeforeWrite,
            EabiCrReadBeforeWriteMask = precise.CrReadBeforeWriteMask
        };
        if (ViolatingNatives.Contains(address)) return (precise, "replaces the register file");
        if (!precise.HasFullSynchronizationFence)
        {
            if ((precise.GprReadBeforeWriteMask & ~InputGprMask) != 0 ||
                (precise.FprReadBeforeWriteMask & ~ArgumentFprMask) != 0)
                return (precise, "reads non-argument registers");
            if ((precise.GprPossibleWriteMask & ~VolatileGprMask) != 0 ||
                (precise.FprPossibleWriteMask & ~VolatileFprMask) != 0)
                return (precise, "writes nonvolatile registers");
        }
        return (Mask(precise), null);
    }

    /// <summary>
    /// Mod-safe and JIT-safe view of a conservative registered native's contract when <see cref="Enabled"/> is off.
    /// Every registered native except <c>OSLoadContext</c> (0x801D80B4) preserves r1, r2, r13, r14-r31, f14-f31, and
    /// cr2-cr4 (runtime dispatch guards non-volatile GPRs/FPRs; fast-path mutex/IRQ overrides write only r3 and fiber
    /// thread switches save/restore the complete CpuContext). Keeping full architectural reads ensures callers still
    /// flush all dirty state before suspension, logging, callbacks, or PoyoJIT entry, while keeping
    /// <see cref="GuestCallBoundaryFlags.InvokesGuestCode"/> keeps GQR hoisting and memory range lowering opaque.
    /// </summary>
    public static GuestAbiContract ClassifyModSafeNative(uint address, GuestAbiContract precise)
    {
        if (!precise.HasFullSynchronizationFence || ViolatingNatives.Contains(address))
            return precise;
        return precise with
        {
            GprReadBeforeWriteMask = uint.MaxValue,
            GprPossibleWriteMask = VolatileGprMask,
            GprReturnMask = (1u << 3) | (1u << 4),
            GprDefiniteWriteMask = 0,
            FprReadBeforeWriteMask = uint.MaxValue,
            FprPossibleWriteMask = VolatileFprMask,
            FprReturnMask = 1u << 1,
            FprDefiniteWriteMask = 0,
            CrReadBeforeWriteMask = byte.MaxValue,
            CrPossibleWriteMask = VolatileCrFields,
            CrDefiniteWriteMask = 0,
            ReadsXerBeforeWrite = true,
            MayWriteXer = true,
            DefinitelyWritesXer = false,
            ReadsCtrBeforeWrite = true,
            MayWriteCtr = true,
            DefinitelyWritesCtr = false,
            ReadsLrBeforeWrite = true,
            MayWriteLr = true,
            DefinitelyWritesLr = false,
            ReadsFpscrBeforeWrite = true,
            MayWriteFpscr = true,
            DefinitelyWritesFpscr = false,
            GqrReadBeforeWriteMask = byte.MaxValue,
            GqrPossibleWriteMask = byte.MaxValue,
            GqrDefiniteWriteMask = 0,
            HidReadBeforeWriteMask = byte.MaxValue,
            HidPossibleWriteMask = byte.MaxValue,
            HidDefiniteWriteMask = 0,
            BoundaryFlags = (precise.BoundaryFlags & ~GuestCallBoundaryFlags.RequiresCompleteContext) |
                            GuestCallBoundaryFlags.InvokesGuestCode
        };
    }

    /// <summary>
    /// MKW_EABI_FRAME_SAVES=elide (opt-in): conformant bodies drop their nonvolatile frame saves (see
    /// <see cref="ElideFrameSaves"/>). Off by default: the save slots' bytes change (they keep older stack contents),
    /// and vanilla Training's character select diverged with it on (the cursor never reached
    /// Mario), consistent with the game reading uninitialized stack locals that overlap earlier frames' save slots.
    /// </summary>
    public static readonly bool FrameSaveElision =
        Enabled && Environment.GetEnvironmentVariable("MKW_EABI_FRAME_SAVES") == "elide";

    /// <summary>
    /// MKW_EABI_CR_PARTIAL=1 (opt-in): calls to conformant callees and indirect calls publish only cr1 (the varargs flag,
    /// a conformant callee's only CR input) instead of the packed CR. Off by default: a thread switch inside such a call
    /// saves CpuContext's CR into the thread's OSContext, and with this on the saved cr0 nibble is the stale one
    /// (vanilla boot: OSContext.cr words at 0x90DE6E68, 0x804C2090, 0x804DD5D8, 0x805BA188 differed from the
    /// base build; dead register state, gameplay identical, but MEM2 no longer matched the main tree's hashes).
    /// </summary>
    public static readonly bool PartialCrFlush = Enabled && Environment.GetEnvironmentVariable("MKW_EABI_CR_PARTIAL") == "1";

    /// <summary>Name of the host local that carries a nonvolatile register's entry value (<see cref="ElideFrameSaves"/>).</summary>
    public static string FrameSavedName(int register) =>
        register < 32 ? $"mkw_frame_saved_r{register}" : $"mkw_frame_saved_f{register - 32}";

    /// <summary>
    /// MKW_EABI_REGS, conformant bodies (the caller checks): the prologue's frame saves of r14-r31/f14-f31 become a
    /// host local. A save is a store of a register's entry value to a stack slot (the entry r1's affine form, as
    /// <see cref="Classify"/> recognizes them); each such store goes, the entry block opens with
    /// <c>saved = entry value</c>, and each restore (a load of the same size into the same register class from that
    /// slot) becomes <c>dest = saved</c>. Every register therefore holds exactly what it held before, given what the
    /// classification already assumes (nothing else writes a save slot between its save and restore; checked here for
    /// every access off the entry stack pointer). Only the save slots' memory differs: they are not written. In this
    /// mode they already held CpuContext's stale nonvolatile copies (zeros in a register-passing entry), never a
    /// caller's register, and a conformant body's restored nonvolatiles are never published, so the stores were pure
    /// overhead (~7 host instructions each: address, mask, policy, dirty mark, byte swap). A paired-single save
    /// (psq_st) goes only when every psq_l of its slot is dead (the canonical epilogue's psq_l is overwritten by the
    /// lfd after it); those dead loads go too. A register is left alone when anything else reads or writes its slot;
    /// the whole body is when it calls a save/restore thunk, hands a stack address to a helper, or branches back to
    /// its entry block.
    /// </summary>
    public static IrFunction ElideFrameSaves(IrFunction function, Action<string, bool> declareLocal, out int elided)
    {
        elided = 0;
        foreach (var block in function.Blocks)
        {
            var last = block.Instructions.Count == 0 ? null : block.Instructions[^1];
            var targetsEntry = last switch
            {
                IrBranch branch => branch.TrueLabel == function.EntryLabel || branch.FalseLabel == function.EntryLabel,
                IrJump jump => jump.TargetLabel == function.EntryLabel,
                IrJumpTable table => table.Cases.Any(c => c.TargetLabel == function.EntryLabel),
                _ => false
            };
            if (targetsEntry) return function;
        }

        var facts = FrameFacts.AddressesOf(function);
        var uses = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var block in function.Blocks)
            foreach (var instruction in block.Instructions)
                foreach (var used in IrRegisterDataFlow.Uses(instruction))
                    uses[used] = uses.GetValueOrDefault(used) + 1;

        // Every access off the entry stack pointer: (register or -1, slot kind or '\0', block, index, offset, size).
        var accesses = new List<(int Register, char Kind, bool IsStore, int Block, int Index, long Offset, int Size)>();
        var entryNames = new Dictionary<int, string>();
        for (var b = 0; b < function.Blocks.Count; ++b)
        {
            var instructions = function.Blocks[b].Instructions;
            for (var i = 0; i < instructions.Count; ++i)
            {
                switch (instructions[i])
                {
                    case IrStore store:
                    {
                        var (root, offset) = facts.SlotOf(store.Address);
                        if (root != "") break;
                        var register = store.Source is { Kind: "register", RegisterName: { } stored } && facts.IsEntryValue(stored)
                            ? ArchIndex(stored) : -1;
                        var isSave = register is >= 14 and < 32 && store.SizeBytes == 4 ||
                                     register is >= 46 and < 64 && store.SizeBytes == 8;
                        if (isSave) entryNames[register] = store.Source.RegisterName!;
                        accesses.Add((isSave ? register : -1, isSave ? (register < 32 ? 'w' : 'd') : '\0', true, b, i,
                            offset, store.SizeBytes));
                        break;
                    }
                    case IrLoad load:
                    {
                        var (root, offset) = facts.SlotOf(load.Address);
                        if (root == "") accesses.Add((-1, '\0', false, b, i, offset, load.SizeBytes));
                        break;
                    }
                    case IrCall call when call.Target is "PPC_PsqSt" or "PPC_PsqL":
                    {
                        if (call.Arguments.Count < 2 || call.Arguments[0] is not { Kind: "register", RegisterName: { } address })
                            return function;
                        var (root, offset) = facts.FormOf(address);
                        if (root != "") break;
                        var isStore = call.Target == "PPC_PsqSt";
                        var register = isStore && call.Arguments[1] is { Kind: "register", RegisterName: { } source } &&
                                       facts.IsEntryValue(source) ? ArchIndex(source) : -1;
                        var isSave = register is >= 46 and < 64;
                        if (isSave) entryNames[register] = ((IrCall)instructions[i]).Arguments[1].RegisterName!;
                        // A quantized access is at most 8 bytes (two 32-bit singles).
                        accesses.Add((isSave ? register : -1, isSave ? 'p' : '\0', isStore, b, i, offset, 8));
                        break;
                    }
                    case IrCall call:
                    {
                        if (GuestTargetParser.TryParseAddress(call.Target, out var target))
                        {
                            if (TryGetFrameThunk(target, out _, out _)) return function;
                            break;
                        }
                        // A helper handed a stack address may read or write the frame anywhere.
                        foreach (var argument in call.Arguments)
                            if (argument is { Kind: "register", RegisterName: { } name } && facts.FormOf(name).Root == "")
                                return function;
                        break;
                    }
                }
            }
        }

        static bool Overlaps(long a, int aSize, long b, int bSize) => a < b + bSize && b < a + aSize;
        var saves = accesses.Where(static a => a.Register >= 0).ToList();
        if (saves.Count == 0) return function;

        // Stack addresses used other than as a load/store address (passed to a callee, stored, compared): the frame
        // offsets they point at may be read or written by other code.
        var escaping = new List<long>();
        foreach (var block in function.Blocks)
            foreach (var instruction in block.Instructions)
                foreach (var used in IrRegisterDataFlow.Uses(instruction))
                {
                    var addressUse = instruction switch
                    {
                        IrStore store => store.Address.Base == used && store.Source.RegisterName != used,
                        IrLoad load => load.Address.Base == used,
                        IrCall { Target: "PPC_PsqSt" or "PPC_PsqL" } psq =>
                            psq.Arguments.Count > 0 && psq.Arguments[0].RegisterName == used &&
                            psq.Arguments.Skip(1).All(argument => argument.RegisterName != used),
                        // Address arithmetic (addi/mr): its result is checked as a value of its own.
                        IrBinary { Op: "add", Right.Kind: "const" } add => add.Left.RegisterName == used,
                        IrAssign { Value.Kind: "register" } => true,
                        _ => false
                    };
                    if (addressUse) continue;
                    var (root, offset) = facts.FormOf(used);
                    if (root == "") escaping.Add(offset);
                }
        // A guest register holding a stack address is read by callees without an IR use (their contracts): any GPR
        // definition pointing into the frame counts as escaping where it points.
        foreach (var block in function.Blocks)
            foreach (var instruction in block.Instructions)
                foreach (var defined in IrRegisterDataFlow.Definitions(instruction))
                    if (ArchIndex(defined) is >= 0 and < 32 and not 1 && facts.FormOf(defined) is ("", var pointed))
                        escaping.Add(pointed);
        if (escaping.Count != 0)
            saves.RemoveAll(save => escaping.Any(offset => offset >= save.Offset && offset < save.Offset + save.Size));
        if (saves.Count == 0) return function;
        var removals = new HashSet<(int, int)>();
        var replacements = new Dictionary<(int, int), IrInstruction>();
        var saved = new SortedSet<int>();
        foreach (var register in saves.Select(static s => s.Register).Distinct())
        {
            var mine = saves.Where(s => s.Register == register).ToList();
            var ok = true;
            var registerRemovals = new List<(int, int)>();
            var registerReplacements = new List<((int, int) Key, IrInstruction Instruction)>();
            foreach (var access in accesses)
            {
                var slot = mine.FirstOrDefault(s => Overlaps(s.Offset, s.Size, access.Offset, access.Size));
                if (slot.Kind == '\0') continue;  // touches none of this register's slots
                if (access.Register == register && access.Offset == slot.Offset && access.Kind == slot.Kind)
                {
                    registerRemovals.Add((access.Block, access.Index));  // one of its saves
                    continue;
                }
                if (access.IsStore) { ok = false; break; }
                var instruction = function.Blocks[access.Block].Instructions[access.Index];
                if (access.Offset != slot.Offset) { ok = false; break; }
                if (slot.Kind == 'p')
                {
                    // A paired restore: dead (overwritten by the scalar restore after it) or the register stays.
                    if (instruction is IrCall { Target: "PPC_PsqL", Destination: var psqDest } &&
                        (string.IsNullOrEmpty(psqDest) || uses.GetValueOrDefault(psqDest) == 0))
                    {
                        registerRemovals.Add((access.Block, access.Index));
                        continue;
                    }
                    ok = false;
                    break;
                }
                var gpr = slot.Kind == 'w';
                if (instruction is IrLoad load && load.SizeBytes == slot.Size &&
                    ArchIndex(load.Destination) is var destination && (gpr ? destination is >= 0 and < 32 : destination >= 32))
                {
                    registerReplacements.Add(((access.Block, access.Index),
                        new IrAssign(load.Destination, IrValue.Register(FrameSavedName(register)))));
                    continue;
                }
                ok = false;
                break;
            }
            if (!ok) continue;
            // A paired save without a scalar one is no CodeWarrior pattern: leave it.
            if (mine.All(static s => s.Kind == 'p')) continue;
            saved.Add(register);
            foreach (var key in registerRemovals) removals.Add(key);
            foreach (var (key, instruction) in registerReplacements) replacements[key] = instruction;
        }
        if (saved.Count == 0) return function;
        // Each save local is assigned only if a restore reads it; registers without a restore just lose the store.
        var restoredFrom = replacements.Values.OfType<IrAssign>()
            .Select(static a => a.Value.RegisterName!).ToHashSet(StringComparer.Ordinal);

        var blocks = new List<IrBasicBlock>(function.Blocks.Count);
        for (var b = 0; b < function.Blocks.Count; ++b)
        {
            var block = function.Blocks[b];
            var rewritten = new List<IrInstruction>(block.Instructions.Count + saved.Count);
            if (block.Label == function.EntryLabel)
            {
                foreach (var register in saved)
                {
                    if (!restoredFrom.Contains(FrameSavedName(register))) continue;
                    declareLocal(FrameSavedName(register), register < 32);
                    rewritten.Add(new IrAssign(FrameSavedName(register), IrValue.Register(entryNames[register])));
                }
            }
            for (var i = 0; i < block.Instructions.Count; ++i)
            {
                if (removals.Contains((b, i))) { ++elided; continue; }
                rewritten.Add(replacements.TryGetValue((b, i), out var replacement) ? replacement : block.Instructions[i]);
            }
            blocks.Add(new IrBasicBlock(block.Label, rewritten));
        }
        return function with { Blocks = blocks };
    }

    private enum ThunkKind { SaveGpr, RestGpr, SaveFpr, RestFpr }

    /// <summary>A CodeWarrior save/restore routine entry (see <see cref="GuestSaveRestoreThunks.FrameAnalysis"/>).</summary>
    private static bool TryGetFrameThunk(uint address, out ThunkKind kind, out int startRegister)
    {
        var thunks = GuestSaveRestoreThunks.FrameAnalysis;
        (GuestSaveRestoreThunkRange? Range, ThunkKind Kind)[] families =
        {
            (thunks.SaveGpr, ThunkKind.SaveGpr), (thunks.RestGpr, ThunkKind.RestGpr),
            (thunks.SaveFpr, ThunkKind.SaveFpr), (thunks.RestFpr, ThunkKind.RestFpr)
        };
        foreach (var (range, familyKind) in families)
        {
            if (range is null) continue;
            var last = range.BaseAddress + (uint)((range.LastRegister - range.FirstRegister) * 4);
            if (address < range.BaseAddress || address > last || ((address - range.BaseAddress) & 3u) != 0) continue;
            kind = familyKind;
            startRegister = range.FirstRegister + (int)((address - range.BaseAddress) / 4);
            return true;
        }
        kind = default;
        startRegister = -1;
        return false;
    }

    private static string Gprs(uint mask) =>
        string.Join(",", Enumerable.Range(0, 32).Where(r => (mask & (1u << r)) != 0).Select(r => $"r{r}"));

    private static string Fprs(uint mask) =>
        string.Join(",", Enumerable.Range(0, 32).Where(r => (mask & (1u << r)) != 0).Select(r => $"f{r}"));

    /// <summary>
    /// Register index of an architectural GPR (0-31) or FPR (32-63) name, ignoring only an SSA version suffix
    /// (named temporaries such as r1_psq_tmp are not r1); -1 otherwise.
    /// </summary>
    private static int ArchIndex(string? name)
    {
        if (string.IsNullOrEmpty(name)) return -1;
        var stripped = RegisterNameUtils.StripNumericSuffixSpan(name.AsSpan());
        if (stripped.Length < 2 || (stripped[0] != 'r' && stripped[0] != 'f')) return -1;
        if (!int.TryParse(stripped[1..], out var index) || index < 0 || index > 31) return -1;
        return stripped[0] == 'r' ? index : 32 + index;
    }

    /// <summary>
    /// Mechanically proven preserved non-volatile registers for one function (stack-saved and restored along every
    /// reachable exit path, or untouched along every reachable path).
    /// </summary>
    internal readonly record struct PreservedFrameState(
        uint Gprs,
        uint Fprs,
        byte CrFields,
        bool WritesSdaBase);

    internal static PreservedFrameState AnalyzePreservedFrameState(
        IrFunction function,
        IReadOnlyDictionary<uint, GuestAbiContract>? callees,
        bool trustIndirectCalls,
        bool allowPsqFprPreservation)
    {
        var facts = FrameFacts.Analyze(
            function,
            callees,
            strictUnknownCallees: true,
            trustIndirectCalls: trustIndirectCalls,
            allowPsqFprPreservation: allowPsqFprPreservation);
        return new PreservedFrameState(
            facts.PreservedGprs,
            facts.PreservedFprs,
            facts.PreservedCrFields,
            facts.WritesSdaBase);
    }

    internal static bool IsNonLocalExit(IrFunction function, PreservedFrameState preserved)
    {
        if ((preserved.Gprs & (1u << 1)) != 0 || !preserved.WritesSdaBase)
            return false;
        if (GuestTargetParser.TryParseAddress(function.EntryLabel, out var entryAddr) &&
            NonLocalExits.Contains(entryAddr))
            return true;
        return GuestTargetParser.TryParseAddress(function.Name, out var nameAddr) &&
               NonLocalExits.Contains(nameAddr);
    }

    /// <summary>
    /// Frame save/restore facts for r1, r14-r31, f14-f31 and cr2-cr4 from SSA. Every address is an affine form
    /// (root, offset): root "" is the entry stack pointer, followed through mr/addi and through a load of the back
    /// chain the prologue stored (lwz rX,0(r1) of an aligned stwux frame); any other definition is its own root. A
    /// save is a store of a register's entry value to a slot (stw/stmw/stfd/psq_st, mfcr+stw, or a _savegpr/_savefpr
    /// thunk at r11's form) whose byte span is never overwritten by another frame store; a forward may-clobber
    /// dataflow treats a definition whose value (through copies) was loaded from that register's save slot
    /// (lwz/lmw/lfd/psq_l, lwz+mtcrf, or a _restgpr/_restfpr thunk) as a restore.
    /// </summary>
    private sealed class FrameFacts
    {
        public string? IntrinsicViolation;
        public uint PreservedGprs;
        public uint PreservedFprs;
        public byte PreservedCrFields;
        public uint SaveOnlyGprs;
        public uint SaveOnlyFprs;
        public bool HasMtcrf;
        /// <summary>The body itself defines r2 or r13 (not through a callee).</summary>
        public bool WritesSdaBase;

        private const ulong Tracked = 0xFFFFC000_FFFFC002ul;  // r1, r14-r31, f14-f31
        private const int CrRegisterSlot = -2;

        private readonly Dictionary<string, IrInstruction> _definitions = new(StringComparer.Ordinal);
        private readonly Dictionary<string, (string Root, long Offset)> _affine = new(StringComparer.Ordinal);
        private readonly HashSet<(string, long)> _backChainSlots = new();
        private readonly Dictionary<int, HashSet<(string, long, char)>> _saves = new();
        private readonly Dictionary<string, bool> _possiblyEntry = new(StringComparer.Ordinal);
        private readonly Dictionary<string, bool> _holdsEntryCr = new(StringComparer.Ordinal);

        private bool IsEntryName(string name) => !_definitions.ContainsKey(name);

        private (string Root, long Offset) Affine(string name, int depth = 0)
        {
            if (_affine.TryGetValue(name, out var known)) return known;
            (string, long) result = (name, 0);
            if (depth < 64)
            {
                if (!_definitions.TryGetValue(name, out var definition))
                    result = ArchIndex(name) == 1 ? ("", 0) : (name, 0);
                else
                    result = definition switch
                    {
                        IrAssign { Value: { Kind: "register", RegisterName: { } source } } => Affine(source, depth + 1),
                        IrBinary { Op: "add", Left: { Kind: "register", RegisterName: { } left },
                            Right: { Kind: "const", Constant: { } k } } => Shift(Affine(left, depth + 1), k),
                        IrLoad { SizeBytes: 4 } load when _backChainSlots.Contains(Slot(load.Address, depth)) => ("", 0),
                        IrResolvedLoad { SizeBytes: 4 } rload when _backChainSlots.Contains(Slot(rload.OriginalAddress, depth)) => ("", 0),
                        IrResolvedLoadPair { ElementSizeBytes: 4 } rpair when
                            (rpair.FirstDestination == name && _backChainSlots.Contains(Slot(rpair.FirstOriginalAddress, depth))) ||
                            (rpair.SecondDestination == name && _backChainSlots.Contains(Slot(rpair.SecondOriginalAddress, depth))) => ("", 0),
                        _ => (name, 0)
                    };
            }
            _affine[name] = result;
            return result;

            static (string, long) Shift((string Root, long Offset) form, long k) => (form.Root, form.Offset + k);
        }

        private (string, long) Slot(IrAddress address, int depth = 0)
        {
            var (root, offset) = Affine(address.Base, depth + 1);
            return (root, offset + address.Offset);
        }

        private void AddSave(int register, (string Root, long Offset) slot, char kind)
        {
            if (!_saves.TryGetValue(register, out var set)) _saves[register] = set = new HashSet<(string, long, char)>();
            set.Add((slot.Root, slot.Offset, kind));
        }

        private bool IsSave(int register, (string Root, long Offset) slot, char kind) =>
            _saves.TryGetValue(register, out var set) && set.Contains((slot.Root, slot.Offset, kind));

        private static bool IsWholeCrName(string? name) =>
            !string.IsNullOrEmpty(name) &&
            RegisterNameUtils.StripNumericSuffixSpan(name.AsSpan()).Equals("cr", StringComparison.OrdinalIgnoreCase);

        private bool IsEntryCrBeforeLoads(string name, int depth = 0)
        {
            if (depth > 64) return false;
            if (!_definitions.TryGetValue(name, out var definition)) return IsWholeCrName(name);
            return definition is IrAssign { Value: { Kind: "register", RegisterName: { } source } } &&
                   IsEntryCrBeforeLoads(source, depth + 1);
        }

        private bool HoldsEntryCr(string name, int depth = 0)
        {
            if (_holdsEntryCr.TryGetValue(name, out var known)) return known;
            if (depth > 64) return false;
            _holdsEntryCr[name] = false; // cycle guard
            bool result;
            if (!_definitions.TryGetValue(name, out var definition))
            {
                result = IsWholeCrName(name);
            }
            else
            {
                result = definition switch
                {
                    IrAssign { Value: { Kind: "register", RegisterName: { } source } } => HoldsEntryCr(source, depth + 1),
                    IrLoad { SizeBytes: 4 } load => IsSave(CrRegisterSlot, Slot(load.Address), 'w'),
                    IrResolvedLoad { SizeBytes: 4 } rload => IsSave(CrRegisterSlot, Slot(rload.OriginalAddress), 'w'),
                    IrResolvedLoadPair { ElementSizeBytes: 4 } rpair =>
                        (rpair.FirstDestination == name && IsSave(CrRegisterSlot, Slot(rpair.FirstOriginalAddress), 'w')) ||
                        (rpair.SecondDestination == name && IsSave(CrRegisterSlot, Slot(rpair.SecondOriginalAddress), 'w')),
                    _ => false
                };
            }
            _holdsEntryCr[name] = result;
            return result;
        }

        private bool TryMatchMtcrf(IrBinary orInstruction, out byte touchedCrFields, out string? sourceRegister)
        {
            touchedCrFields = 0;
            sourceRegister = null;
            if (!string.Equals(orInstruction.Op, "or", StringComparison.Ordinal) ||
                !IsWholeCrName(orInstruction.Destination) ||
                orInstruction.Left.Kind != "register" || orInstruction.Right.Kind != "register" ||
                orInstruction.Left.RegisterName is not { } leftName ||
                orInstruction.Right.RegisterName is not { } rightName)
            {
                return false;
            }

            bool TryInspect(string preserveCandidate, string srcCandidate, out byte fields, out string? src)
            {
                fields = 0;
                src = null;
                if (!preserveCandidate.Contains("_mtcrf_preserve", StringComparison.Ordinal) ||
                    !srcCandidate.Contains("_mtcrf_src", StringComparison.Ordinal) ||
                    !_definitions.TryGetValue(preserveCandidate, out var preserveDef) ||
                    !_definitions.TryGetValue(srcCandidate, out var srcDef) ||
                    preserveDef is not IrBinary { Op: "and", Left: { Kind: "register", RegisterName: { } crName }, Right: { Kind: "const", Constant: { } preserveMask } } ||
                    !IsWholeCrName(crName) ||
                    srcDef is not IrBinary { Op: "and", Left: { Kind: "register", RegisterName: { } valueName }, Right: { Kind: "const", Constant: { } srcMask } })
                {
                    return false;
                }

                var mask = unchecked((uint)srcMask);
                if (unchecked((uint)preserveMask) != ~mask) return false;
                fields = FieldsOfCrMask(mask);
                src = valueName;
                return true;
            }

            return TryInspect(leftName, rightName, out touchedCrFields, out sourceRegister) ||
                   TryInspect(rightName, leftName, out touchedCrFields, out sourceRegister);
        }

        private static byte FieldsOfCrMask(uint mask)
        {
            byte fields = 0;
            for (var field = 0; field < 8; ++field)
                if (((mask >> ((7 - field) * 4)) & 0xFu) != 0)
                    fields |= (byte)(1 << field);
            return fields;
        }

        /// <summary>The value of <paramref name="name"/> is <paramref name="register"/>'s entry value, restored.</summary>
        private bool HoldsEntryValue(string name, int register, bool allowPsqFprPreservation, int depth = 0)
        {
            if (depth > 64) return false;
            if (register == 1) return Affine(name) == ("", 0);
            if (!_definitions.TryGetValue(name, out var definition)) return ArchIndex(name) == register;
            var gpr = register < 32;
            return definition switch
            {
                IrAssign { Value: { Kind: "register", RegisterName: { } source } } =>
                    HoldsEntryValue(source, register, allowPsqFprPreservation, depth + 1),
                IrLoad load => (gpr ? load.SizeBytes == 4 : load.SizeBytes == 8) &&
                               IsSave(register, Slot(load.Address), gpr ? 'w' : 'd'),
                IrResolvedLoad rload => (gpr ? rload.SizeBytes == 4 : rload.SizeBytes == 8) &&
                                        IsSave(register, Slot(rload.OriginalAddress), gpr ? 'w' : 'd'),
                IrResolvedLoadPair rpair => (gpr ? rpair.ElementSizeBytes == 4 : rpair.ElementSizeBytes == 8) &&
                    ((rpair.FirstDestination == name && IsSave(register, Slot(rpair.FirstOriginalAddress), gpr ? 'w' : 'd')) ||
                     (rpair.SecondDestination == name && IsSave(register, Slot(rpair.SecondOriginalAddress), gpr ? 'w' : 'd'))),
                IrCall { Target: "PPC_PsqL" } psql when allowPsqFprPreservation && !gpr && psql.Arguments.Count >= 1 &&
                    psql.Arguments[0] is { Kind: "register", RegisterName: { } address } =>
                    IsSave(register, Affine(address), 'p'),
                _ => false
            };
        }

        /// <summary>An entry value, or a phi that may merge one in.</summary>
        private bool PossiblyEntry(string name, int depth = 0)
        {
            if (_possiblyEntry.TryGetValue(name, out var known)) return known;
            if (IsEntryName(name)) return true;
            if (depth > 64 || _definitions[name] is not IrPhi phi) return false;
            _possiblyEntry[name] = false;  // cycle guard
            var result = phi.Sources.Values.Any(source => PossiblyEntry(source, depth + 1));
            _possiblyEntry[name] = result;
            return result;
        }

        private static string? ReachingR11(IReadOnlyList<IrInstruction> instructions, int position)
        {
            for (var i = position - 1; i >= 0; --i)
                foreach (var name in IrRegisterDataFlow.Definitions(instructions[i]))
                    if (ArchIndex(name) == 11) return name;
            return null;
        }

        /// <summary>The affine address forms of <paramref name="function"/> (definitions and back-chain slots) only.</summary>
        public static FrameFacts AddressesOf(IrFunction function)
        {
            var facts = new FrameFacts();
            foreach (var block in function.Blocks)
                foreach (var instruction in block.Instructions)
                    foreach (var name in IrRegisterDataFlow.Definitions(instruction))
                        facts._definitions[name] = instruction;
            foreach (var block in function.Blocks)
                foreach (var instruction in block.Instructions)
                    if (instruction is IrStore { SizeBytes: 4, Source: { Kind: "register", RegisterName: { } stored } } store &&
                        ArchIndex(stored) == 1 && facts.IsEntryName(stored))
                        facts._backChainSlots.Add(facts.Slot(store.Address));
            facts._affine.Clear();
            return facts;
        }

        public bool IsEntryValue(string name) => IsEntryName(name);
        public (string Root, long Offset) SlotOf(IrAddress address) => Slot(address);
        public (string Root, long Offset) FormOf(string name) => Affine(name);

        private readonly record struct RecordedStore(string Root, long Offset, int Size, int SavedRegister, char Kind);

        public static FrameFacts Analyze(
            IrFunction function,
            IReadOnlyDictionary<uint, GuestAbiContract>? callees,
            bool strictUnknownCallees = false,
            bool trustIndirectCalls = true,
            bool allowPsqFprPreservation = true)
        {
            var facts = new FrameFacts();
            foreach (var block in function.Blocks)
                foreach (var instruction in block.Instructions)
                    foreach (var name in IrRegisterDataFlow.Definitions(instruction))
                        facts._definitions[name] = instruction;

            // The back chain first: stores of the entry stack pointer (stwu/stwux). Loads from those slots then
            // read the entry stack pointer, which the affine forms below rely on.
            foreach (var block in function.Blocks)
                foreach (var instruction in block.Instructions)
                {
                    if (instruction is IrStore { SizeBytes: 4, Source: { Kind: "register", RegisterName: { } stored } } store &&
                        ArchIndex(stored) == 1 && facts.IsEntryName(stored))
                        facts._backChainSlots.Add(facts.Slot(store.Address));
                    else if (instruction is IrResolvedStore { SizeBytes: 4, Source: { Kind: "register", RegisterName: { } rstored } } rstore &&
                             ArchIndex(rstored) == 1 && facts.IsEntryName(rstored))
                        facts._backChainSlots.Add(facts.Slot(rstore.OriginalAddress));
                    else if (instruction is IrResolvedStorePair { ElementSizeBytes: 4 } rpair)
                    {
                        if (rpair.FirstSource is { Kind: "register", RegisterName: { } first } &&
                            ArchIndex(first) == 1 && facts.IsEntryName(first))
                            facts._backChainSlots.Add(facts.Slot(rpair.FirstOriginalAddress));
                        if (rpair.SecondSource is { Kind: "register", RegisterName: { } second } &&
                            ArchIndex(second) == 1 && facts.IsEntryName(second))
                            facts._backChainSlots.Add(facts.Slot(rpair.SecondOriginalAddress));
                    }
                }
            facts._affine.Clear();

            // Entry values of r14-r31/f14-f31 used other than as a frame save, and nonvolatiles read by violator callees.
            var otherUses = 0ul;
            var violatorReads = 0ul;
            var recordedStores = new List<RecordedStore>();
            var candidateSaves = new List<RecordedStore>();

            void RecordSingleStore(IrValue source, IrAddress address, int sizeBytes, out string? savedSource)
            {
                savedSource = null;
                var slot = facts.Slot(address);
                if (source is { Kind: "register", RegisterName: { } regName })
                {
                    if (sizeBytes is 4 or 8 && facts.IsEntryName(regName) && ArchIndex(regName) is var regIndex and >= 0)
                    {
                        var kind = sizeBytes == 8 ? 'd' : 'w';
                        var entry = new RecordedStore(slot.Item1, slot.Item2, sizeBytes, regIndex, kind);
                        recordedStores.Add(entry);
                        candidateSaves.Add(entry);
                        savedSource = regName;
                        return;
                    }
                    if (sizeBytes == 4 && facts.IsEntryCrBeforeLoads(regName))
                    {
                        var entry = new RecordedStore(slot.Item1, slot.Item2, 4, CrRegisterSlot, 'w');
                        recordedStores.Add(entry);
                        candidateSaves.Add(entry);
                        return;
                    }
                }
                recordedStores.Add(new RecordedStore(slot.Item1, slot.Item2, sizeBytes, -1, '\0'));
            }

            foreach (var block in function.Blocks)
            {
                var instructions = block.Instructions;
                for (var position = 0; position < instructions.Count; ++position)
                {
                    var instruction = instructions[position];
                    string? saveSource = null;
                    string? secondSaveSource = null;
                    switch (instruction)
                    {
                        case IrUndefined:
                            facts.IntrinsicViolation ??= "untranslatable instruction";
                            break;
                        case IrCall { Target: "OSSystemCall" }:
                            facts.IntrinsicViolation ??= "system call";
                            break;
                        case IrBinary binary when binary.Destination.Contains("_mtcrf_", StringComparison.Ordinal):
                            facts.HasMtcrf = true;
                            break;
                    }
                    if (instruction is IrStore store)
                    {
                        RecordSingleStore(store.Source, store.Address, store.SizeBytes, out saveSource);
                    }
                    else if (instruction is IrResolvedStore rstore)
                    {
                        RecordSingleStore(rstore.Source, rstore.OriginalAddress, rstore.SizeBytes, out saveSource);
                    }
                    else if (instruction is IrResolvedStorePair rpair)
                    {
                        RecordSingleStore(rpair.FirstSource, rpair.FirstOriginalAddress, rpair.ElementSizeBytes, out saveSource);
                        RecordSingleStore(rpair.SecondSource, rpair.SecondOriginalAddress, rpair.ElementSizeBytes, out secondSaveSource);
                    }
                    else if (instruction is IrCall { Target: "PPC_PsqSt" } psq && psq.Arguments.Count >= 2 &&
                             psq.Arguments[0] is { Kind: "register", RegisterName: { } psqAddress })
                    {
                        var slot = facts.Affine(psqAddress);
                        if (allowPsqFprPreservation &&
                            psq.Arguments[1] is { Kind: "register", RegisterName: { } psqSource } &&
                            facts.IsEntryName(psqSource) && ArchIndex(psqSource) is var psqRegister and >= 32)
                        {
                            var recorded = new RecordedStore(slot.Root, slot.Offset, 8, psqRegister, 'p');
                            recordedStores.Add(recorded);
                            candidateSaves.Add(recorded);
                            saveSource = psqSource;
                        }
                        else
                        {
                            recordedStores.Add(new RecordedStore(slot.Root, slot.Offset, 8, -1, '\0'));
                        }
                    }
                    else if (instruction is IrResolvedPsqStore rpsq &&
                             rpsq.OriginalAddress is { Kind: "register", RegisterName: { } rpsqAddress })
                    {
                        var slot = facts.Affine(rpsqAddress);
                        recordedStores.Add(new RecordedStore(slot.Root, slot.Offset, 8, -1, '\0'));
                    }
                    else if (instruction is IrCall thunkCall && GuestTargetParser.TryParseAddress(thunkCall.Target, out var thunkAddress) &&
                             TryGetFrameThunk(thunkAddress, out var thunkKind, out var start))
                    {
                        // A save thunk reads its registers only to store them in the frame: no other use. Restore
                        // thunks are handled by the dataflow below.
                        var gpr = thunkKind == ThunkKind.SaveGpr;
                        if ((gpr || thunkKind == ThunkKind.SaveFpr) &&
                            ReachingR11(instructions, position) is { } r11Name)
                        {
                            var (root, offset) = facts.Affine(r11Name);
                            var stride = gpr ? 4 : 8;
                            var kind = gpr ? 'w' : 'd';
                            for (var r = start; r < 32; ++r)
                            {
                                var recorded = new RecordedStore(root, offset - stride * (32 - r), stride, gpr ? r : 32 + r, kind);
                                recordedStores.Add(recorded);
                                candidateSaves.Add(recorded);
                            }
                        }
                    }
                    else if (instruction is IrCall call)
                    {
                        if (GuestTargetParser.TryParseAddress(call.Target, out var callee))
                        {
                            if (callees?.TryGetValue(callee, out var calleeContract) == true)
                            {
                                if (!calleeContract.EabiConformant)
                                {
                                    if (strictUnknownCallees)
                                    {
                                        var gprRead = calleeContract.HasFullSynchronizationFence
                                            ? NonvolatileGprMask
                                            : calleeContract.GprReadBeforeWriteMask & NonvolatileGprMask;
                                        var fprRead = calleeContract.HasFullSynchronizationFence
                                            ? NonvolatileFprMask
                                            : calleeContract.FprReadBeforeWriteMask & NonvolatileFprMask;
                                        violatorReads |= gprRead | ((ulong)fprRead << 32);
                                    }
                                    else
                                    {
                                        violatorReads |= calleeContract.EabiGprReadMask & NonvolatileGprMask;
                                        violatorReads |= (ulong)(calleeContract.EabiFprReadMask & NonvolatileFprMask) << 32;
                                    }
                                }
                            }
                            else if (strictUnknownCallees)
                            {
                                violatorReads |= NonvolatileGprMask | ((ulong)NonvolatileFprMask << 32);
                            }
                        }
                        else if (strictUnknownCallees)
                        {
                            var effect = HelperEffect(call);
                            if (effect.BoundaryFlags.HasFlag(GuestCallBoundaryFlags.RequiresCompleteContext))
                            {
                                violatorReads |= NonvolatileGprMask | ((ulong)NonvolatileFprMask << 32);
                            }
                            else
                            {
                                violatorReads |= (effect.GprReadMask & NonvolatileGprMask) |
                                                 ((ulong)(effect.FprReadMask & NonvolatileFprMask) << 32);
                            }
                        }
                    }
                    else if (instruction is IrUndefined ||
                             (!trustIndirectCalls && instruction is IrIndirectCall or IrIndirectJump))
                    {
                        violatorReads |= NonvolatileGprMask | ((ulong)NonvolatileFprMask << 32);
                    }

                    foreach (var used in IrRegisterDataFlow.Uses(instruction))
                    {
                        var strippedUse = RegisterNameUtils.StripNumericSuffix(used);
                        if (strippedUse is "msr" or "srr0" or "srr1" or "dar" or "dsisr")
                            facts.IntrinsicViolation ??= $"reads {strippedUse}";
                        // Phis only merge values; a real use of a phi that may carry the entry value reads it.
                        if (instruction is IrPhi) continue;
                        var index = ArchIndex(used);
                        if (index < 0 || !facts.PossiblyEntry(used)) continue;
                        if (used == saveSource) { saveSource = null; continue; }
                        if (used == secondSaveSource) { secondSaveSource = null; continue; }
                        otherUses |= 1ul << index;
                    }
                    foreach (var name in IrRegisterDataFlow.Definitions(instruction))
                    {
                        var stripped = RegisterNameUtils.StripNumericSuffix(name);
                        if (stripped is "msr" or "srr0" or "srr1" or "dar" or "dsisr")
                            facts.IntrinsicViolation ??= $"writes {stripped}";
                        if (instruction is not IrPhi && ArchIndex(name) is 2 or 13) facts.WritesSdaBase = true;
                    }
                }
            }

            // Retain only save slots whose byte span is never overwritten by a conflicting store off the same base.
            if (candidateSaves.Count != 0)
            {
                var storesByRoot = new Dictionary<string, List<RecordedStore>>(StringComparer.Ordinal);
                foreach (var st in recordedStores)
                {
                    if (!storesByRoot.TryGetValue(st.Root, out var list))
                        storesByRoot[st.Root] = list = new List<RecordedStore>();
                    list.Add(st);
                }

                foreach (var save in candidateSaves)
                {
                    var conflicted = false;
                    foreach (var other in storesByRoot[save.Root])
                    {
                        if (other.Offset < save.Offset + save.Size && save.Offset < other.Offset + other.Size)
                        {
                            if (other.SavedRegister != save.SavedRegister ||
                                other.Offset != save.Offset ||
                                other.Size != save.Size ||
                                other.Kind != save.Kind)
                            {
                                conflicted = true;
                                break;
                            }
                        }
                    }
                    if (!conflicted)
                        facts.AddSave(save.SavedRegister, (save.Root, save.Offset), save.Kind);
                }
            }

            // Forward may-clobber dataflow over the tracked registers (bit 1: r1 differs from its entry value)
            // and cr2-cr4.
            var blocks = function.Blocks;
            var ordinals = new Dictionary<string, int>(blocks.Count, StringComparer.Ordinal);
            for (var i = 0; i < blocks.Count; ++i) ordinals[blocks[i].Label] = i;
            var successors = new List<int>[blocks.Count];
            var exits = new List<int>();
            for (var i = 0; i < blocks.Count; ++i)
            {
                successors[i] = new List<int>();
                var instructions = blocks[i].Instructions;
                var terminator = instructions.Count == 0 ? null : instructions[^1];
                switch (terminator)
                {
                    case IrBranch branch:
                        if (ordinals.TryGetValue(branch.TrueLabel, out var t)) successors[i].Add(t);
                        if (ordinals.TryGetValue(branch.FalseLabel, out var f)) successors[i].Add(f);
                        break;
                    case IrJump jump:
                        if (ordinals.TryGetValue(jump.TargetLabel, out var j)) successors[i].Add(j);
                        break;
                    case IrJumpTable table:
                        foreach (var jumpCase in table.Cases)
                            if (ordinals.TryGetValue(jumpCase.TargetLabel, out var c)) successors[i].Add(c);
                        break;
                    case IrReturn or IrIndirectJump:
                        exits.Add(i);
                        break;
                    case IrUndefined:
                        break;
                    default:
                        if (i + 1 < blocks.Count) successors[i].Add(i + 1);
                        else exits.Add(i);
                        break;
                }
            }
            var entry = ordinals.TryGetValue(function.EntryLabel, out var entryIndex) ? entryIndex : 0;
            var clobberIn = new ulong[blocks.Count];
            var clobberOut = new ulong[blocks.Count];
            var crIn = new byte[blocks.Count];
            var crOut = new byte[blocks.Count];
            var reached = new bool[blocks.Count];
            if (blocks.Count > 0) reached[entry] = true;

            (ulong State, byte CrState) Transfer(int blockIndex, ulong state, byte crState)
            {
                var instructions = blocks[blockIndex].Instructions;
                for (var position = 0; position < instructions.Count; ++position)
                {
                    var instruction = instructions[position];
                    if (instruction is IrPhi) continue;
                    if (instruction is IrUndefined)
                    {
                        state |= Tracked;
                        crState |= NonvolatileCrFields;
                        continue;
                    }
                    if (instruction is IrIndirectCall or IrIndirectJump)
                    {
                        if (!trustIndirectCalls)
                        {
                            state |= Tracked;
                            crState |= NonvolatileCrFields;
                        }
                        // An indirect call/jump can still have an explicit IR destination (e.g. lr).
                    }
                    if (instruction is IrSetCrField setCr)
                    {
                        crState |= (byte)((1 << (setCr.FieldIndex & 7)) & NonvolatileCrFields);
                        continue;
                    }
                    if (instruction is IrBinary mtcrfOr && facts.TryMatchMtcrf(mtcrfOr, out var touchedCrFields, out var mtcrfSource))
                    {
                        var nonvolTouched = (byte)(touchedCrFields & NonvolatileCrFields);
                        if (nonvolTouched != 0)
                        {
                            if (mtcrfSource is not null && facts.HoldsEntryCr(mtcrfSource))
                                crState &= (byte)~nonvolTouched;
                            else
                                crState |= nonvolTouched;
                        }
                        continue;
                    }
                    var helperHandledCr = false;
                    if (instruction is IrCall call)
                    {
                        if (GuestTargetParser.TryParseAddress(call.Target, out var target))
                        {
                            if (TryGetFrameThunk(target, out var kind, out var start))
                            {
                                if (kind is ThunkKind.RestGpr or ThunkKind.RestFpr)
                                {
                                    var gpr = kind == ThunkKind.RestGpr;
                                    var r11Name = ReachingR11(instructions, position);
                                    var (root, offset) = r11Name is null ? ("?", 0L) : facts.Affine(r11Name);
                                    for (var r = start; r < 32; ++r)
                                    {
                                        var register = gpr ? r : 32 + r;
                                        if (r11Name is not null &&
                                            facts.IsSave(register, (root, offset - (gpr ? 4 : 8) * (32 - r)), gpr ? 'w' : 'd'))
                                            state &= ~(1ul << register);
                                        else
                                            state |= 1ul << register;
                                    }
                                }
                                continue;
                            }
                            if (callees?.TryGetValue(target, out var contract) == true)
                            {
                                if (!contract.EabiConformant)
                                {
                                    if (contract.HasFullSynchronizationFence)
                                    {
                                        state |= Tracked;
                                        crState |= NonvolatileCrFields;
                                    }
                                    else
                                    {
                                        state |= contract.GprPossibleWriteMask & (NonvolatileGprMask | (1u << 1));
                                        state |= (ulong)(contract.FprPossibleWriteMask & NonvolatileFprMask) << 32;
                                        crState |= (byte)(contract.CrPossibleWriteMask & NonvolatileCrFields);
                                    }
                                }
                            }
                            else if (strictUnknownCallees)
                            {
                                state |= Tracked;
                                crState |= NonvolatileCrFields;
                            }
                        }
                        else
                        {
                            var effect = HelperEffect(call);
                            if (effect.BoundaryFlags.HasFlag(GuestCallBoundaryFlags.RequiresCompleteContext))
                            {
                                state |= Tracked;
                                crState |= NonvolatileCrFields;
                            }
                            else
                            {
                                state |= effect.GprWriteMask & (NonvolatileGprMask | (1u << 1));
                                state |= (ulong)(effect.FprWriteMask & NonvolatileFprMask) << 32;
                                crState |= (byte)(effect.CrWriteMask & NonvolatileCrFields);
                                if (effect.CrWriteMask != 0) helperHandledCr = true;
                            }
                        }
                    }
                    foreach (var name in IrRegisterDataFlow.Definitions(instruction))
                    {
                        if (IsWholeCrName(name))
                        {
                            if (!helperHandledCr) crState |= NonvolatileCrFields;
                            continue;
                        }
                        var register = ArchIndex(name);
                        if (register < 0 || ((Tracked >> register) & 1) == 0) continue;
                        if (facts.HoldsEntryValue(name, register, allowPsqFprPreservation)) state &= ~(1ul << register);
                        else state |= 1ul << register;
                    }
                }
                return (state, crState);
            }

            bool changed;
            do
            {
                changed = false;
                for (var i = 0; i < blocks.Count; ++i)
                {
                    if (!reached[i]) continue;
                    var (outgoing, outgoingCr) = Transfer(i, clobberIn[i], crIn[i]);
                    if (outgoing != clobberOut[i] || outgoingCr != crOut[i])
                    {
                        clobberOut[i] = outgoing;
                        crOut[i] = outgoingCr;
                        changed = true;
                    }
                    foreach (var successor in successors[i])
                    {
                        var merged = clobberIn[successor] | outgoing;
                        var mergedCr = (byte)(crIn[successor] | outgoingCr);
                        if (!reached[successor] || merged != clobberIn[successor] || mergedCr != crIn[successor])
                        {
                            reached[successor] = true;
                            clobberIn[successor] = merged;
                            crIn[successor] = mergedCr;
                            changed = true;
                        }
                    }
                }
            } while (changed);

            var clobberedAtExit = 0ul;
            byte clobberedCrAtExit = 0;
            var anyReachableExit = false;
            foreach (var exit in exits)
            {
                if (!reached[exit]) continue;
                anyReachableExit = true;
                clobberedAtExit |= clobberOut[exit];
                clobberedCrAtExit |= crOut[exit];
            }
            if (!anyReachableExit && strictUnknownCallees)
            {
                facts.PreservedGprs = 0;
                facts.PreservedFprs = 0;
                facts.PreservedCrFields = 0;
            }
            else
            {
                var preserved = Tracked & ~clobberedAtExit;
                facts.PreservedGprs = (uint)preserved;
                facts.PreservedFprs = (uint)(preserved >> 32);
                facts.PreservedCrFields = (byte)(NonvolatileCrFields & ~clobberedCrAtExit);
            }
            var saveOnly = ~(otherUses | violatorReads);
            facts.SaveOnlyGprs = (uint)saveOnly & NonvolatileGprMask;
            facts.SaveOnlyFprs = (uint)(saveOnly >> 32) & NonvolatileFprMask;
            return facts;
        }
    }
}
