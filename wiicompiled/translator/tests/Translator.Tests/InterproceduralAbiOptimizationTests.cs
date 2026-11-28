using System;
using System.Collections.Generic;
using Translator.Core.Analysis;
using Translator.Core.Analysis.Representation;
using Translator.Core.Analysis.Ssa;
using Translator.Core.CodeGen;
using Translator.Core.Ir;
using Translator.Core.Representation;
using Xunit;

namespace Translator.Tests;

public sealed class InterproceduralAbiOptimizationTests
{
    private static string Emit(
        IrFunction function,
        uint entryPoint = 0x80005000u,
        IReadOnlyDictionary<uint, GuestAbiContract>? guestAbiContracts = null,
        IReadOnlySet<uint>? modOverridableCallTargets = null,
        uint? sda1Base = null,
        uint? sda2Base = null,
        uint[]? variantOwnCode = null)
    {
        var ssa = new SsaTransformer().Convert(function);
        var types = new RepresentationClassifier().Classify(ssa.Function);
        var signature = new FunctionAbiClassifier().Classify(function.Name, ssa.Function, types);
        return new CxxLinearCodeGenerator().EmitWithFacts(
            entryPoint,
            ssa,
            signature,
            types,
            guestAbiContracts: guestAbiContracts,
            modOverridableCallTargets: modOverridableCallTargets,
            variantOwnCode: variantOwnCode,
            sda1Base: sda1Base,
            sda2Base: sda2Base).Code;
    }

    private static string FunctionBody(string code, string name)
    {
        var header = $"extern \"C\" void {name}(CpuContext* MKW_RESTRICT ctx)";
        var start = code.IndexOf(header, StringComparison.Ordinal);
        Assert.True(start >= 0, $"Missing {header} in:\n{code}");
        var open = code.IndexOf('{', start);
        var depth = 0;
        for (var i = open; i < code.Length; ++i)
        {
            if (code[i] == '{') ++depth;
            else if (code[i] == '}' && --depth == 0) return code[open..(i + 1)];
        }
        throw new InvalidOperationException($"Unclosed body for {name}");
    }

    [Fact]
    public void StackPreservedGprsAndR1ExcludeRestoredNonvolatilesFromWritesWhileKeepingReads()
    {
        var raw = new IrFunction("0x80005000", "0x80005000", new[]
        {
            new IrBasicBlock("0x80005000", new IrInstruction[]
            {
                new IrStore(new IrAddress("r1", -32), IrValue.Register("r1"), 4),
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(-32), "add"),
                new IrStore(new IrAddress("r1", 24), IrValue.Register("r30"), 4),
                new IrStore(new IrAddress("r1", 28), IrValue.Register("r31"), 4),
                new IrAssign("r30", IrValue.Register("r3")),
                new IrBinary("r31", IrValue.Register("r30"), IrValue.Imm(1), "add"),
                new IrAssign("r3", IrValue.Register("r31")),
                new IrLoad("r30", new IrAddress("r1", 24), 4),
                new IrLoad("r31", new IrAddress("r1", 28), 4),
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(32), "add"),
                new IrReturn(null)
            })
        });
        var ssa = new SsaTransformer().Convert(raw).Function;
        var contract = GuestAbiContractAnalyzer.Analyze(ssa);

        Assert.False(contract.HasFullSynchronizationFence);
        Assert.Equal(0u, contract.GprPossibleWriteMask & ((1u << 1) | (1u << 30) | (1u << 31)));
        Assert.Equal(1u << 3, contract.GprPossibleWriteMask);
        // Reads stay intact so CpuContext stays architecturally flushed on entry to callees, P+ C2 hooks,
        // PoyoJIT patches, and OSSaveContext.
        Assert.NotEqual(0u, contract.GprReadBeforeWriteMask & (1u << 1));
        Assert.NotEqual(0u, contract.GprReadBeforeWriteMask & (1u << 30));
        Assert.NotEqual(0u, contract.GprReadBeforeWriteMask & (1u << 31));

        // Return flush omits r30/r31 because neither any non-return flush nor any callee touched ctx->gpr[30..31],
        // keeping CpuContext 100% invariant for PoyoJIT/native callers while letting Clang dead-eliminate the restores.
        var body = FunctionBody(Emit(raw, entryPoint: 0x80005000u), "func_80005000");
        Assert.DoesNotContain("ctx->gpr[30] =", body, StringComparison.Ordinal);
        Assert.DoesNotContain("ctx->gpr[31] =", body, StringComparison.Ordinal);
    }

    [Fact]
    public void StackPreservedProofHandlesBackChainR1AndResolvedPairs()
    {
        var function = new IrFunction("0x80005100", "0x80005100", new[]
        {
            new IrBasicBlock("0x80005100", new IrInstruction[]
            {
                new IrStore(new IrAddress("r1_0", -32), IrValue.Register("r1_0"), 4),
                new IrBinary("r1_1", IrValue.Register("r1_0"), IrValue.Imm(-32), "add"),
                new IrResolvedStorePair(
                    "guest_range_0",
                    new IrAddress("r1_1", 24),
                    new IrAddress("r1_1", 28),
                    24,
                    IrValue.Register("r30_0"),
                    IrValue.Register("r31_0"),
                    4),
                new IrAssign("r30_1", IrValue.Register("r3_0")),
                new IrAssign("r31_1", IrValue.Register("r4_0")),
                new IrBinary("r3_1", IrValue.Register("r30_1"), IrValue.Register("r31_1"), "add"),
                new IrResolvedLoadPair(
                    "r30_2",
                    "r31_2",
                    "guest_range_1",
                    new IrAddress("r1_1", 24),
                    new IrAddress("r1_1", 28),
                    24,
                    4),
                new IrLoad("r11_1", new IrAddress("r1_1", 0), 4),
                new IrAssign("r1_2", IrValue.Register("r11_1")),
                new IrReturn(null)
            })
        });

        var contract = GuestAbiContractAnalyzer.Analyze(function);

        Assert.Equal(0u, contract.GprPossibleWriteMask & ((1u << 1) | (1u << 30) | (1u << 31)));
        Assert.NotEqual(0u, contract.GprPossibleWriteMask & (1u << 3));
    }

    [Fact]
    public void ConflictingStoreToSaveSlotPreventsPreservationProof()
    {
        var raw = new IrFunction("0x80005200", "0x80005200", new[]
        {
            new IrBasicBlock("0x80005200", new IrInstruction[]
            {
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(-16), "add"),
                new IrStore(new IrAddress("r1", 12), IrValue.Register("r31"), 4),
                new IrAssign("r31", IrValue.Register("r3")),
                // Clobbers the save slot before restore.
                new IrStore(new IrAddress("r1", 12), IrValue.Register("r4"), 4),
                new IrLoad("r31", new IrAddress("r1", 12), 4),
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(16), "add"),
                new IrReturn(null)
            })
        });
        var ssa = new SsaTransformer().Convert(raw).Function;
        var contract = GuestAbiContractAnalyzer.Analyze(ssa);

        Assert.NotEqual(0u, contract.GprPossibleWriteMask & (1u << 31));
        Assert.Equal(0u, contract.GprPossibleWriteMask & (1u << 1));
    }

    [Fact]
    public void StackPreservedFprsRequireBitExactDoubleSaveRestoreNotPsqOnly()
    {
        var exactRaw = new IrFunction("0x80005300", "0x80005300", new[]
        {
            new IrBasicBlock("0x80005300", new IrInstruction[]
            {
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(-16), "add"),
                new IrStore(new IrAddress("r1", 8), IrValue.Register("f31"), 8),
                new IrAssign("f31", IrValue.Register("f1")),
                new IrAssign("f1", IrValue.Register("f31")),
                new IrLoad("f31", new IrAddress("r1", 8), 8),
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(16), "add"),
                new IrReturn(null)
            })
        });
        var exactContract = GuestAbiContractAnalyzer.Analyze(new SsaTransformer().Convert(exactRaw).Function);
        Assert.Equal(0u, exactContract.FprPossibleWriteMask & (1u << 31));
        Assert.NotEqual(0u, exactContract.FprReadBeforeWriteMask & (1u << 31));
        // Emitting exactRaw elides ctx->fpr[31] return flush and marks preservesNonvolatileFprs=true for PoyoJIT!
        var exactEmitted = Emit(exactRaw, entryPoint: 0x80005300u);
        Assert.DoesNotContain("ctx->fpr[31] =", FunctionBody(exactEmitted, "func_80005300"), StringComparison.Ordinal);
        Assert.Contains("preserves=true fpr_mask=0x00000000", exactEmitted, StringComparison.Ordinal);

        var psqOnlyRaw = new IrFunction("0x80005340", "0x80005340", new[]
        {
            new IrBasicBlock("0x80005340", new IrInstruction[]
            {
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(-16), "add"),
                new IrCall(string.Empty, "PPC_PsqSt", new[] { IrValue.Register("r1"), IrValue.Register("f31"), IrValue.Imm(0), IrValue.Imm(0) }),
                new IrAssign("f31", IrValue.Register("f1")),
                new IrCall("f31", "PPC_PsqL", new[] { IrValue.Register("r1"), IrValue.Imm(0), IrValue.Imm(0) }),
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(16), "add"),
                new IrReturn(null)
            })
        });
        var psqOnlyContract = GuestAbiContractAnalyzer.Analyze(new SsaTransformer().Convert(psqOnlyRaw).Function);
        Assert.NotEqual(0u, psqOnlyContract.FprPossibleWriteMask & (1u << 31));
    }

    [Fact]
    public void StackPreservedNonvolatileCrFieldsAreExcludedWhenRestoredWithMtcrf()
    {
        // mfcr r12; stw r12, 8(r1); cmpwi cr2, r3, 0; lwz r12, 8(r1); mtcrf 0x38, r12 (cr2..cr4 mask 0x00FFF000)
        var raw = new IrFunction("0x80005400", "0x80005400", new[]
        {
            new IrBasicBlock("0x80005400", new IrInstruction[]
            {
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(-16), "add"),
                new IrAssign("r12", IrValue.Register("cr")),
                new IrStore(new IrAddress("r1", 8), IrValue.Register("r12"), 4),
                new IrSetCrField(2, IrValue.Register("r3"), IrValue.Imm(0), false),
                new IrLoad("r12", new IrAddress("r1", 8), 4),
                new IrBinary("r12_mtcrf_src", IrValue.Register("r12"), IrValue.Imm(0x00FFF000), "and"),
                new IrBinary("r12_mtcrf_preserve", IrValue.Register("cr"), IrValue.Imm(unchecked((int)~0x00FFF000u)), "and"),
                new IrBinary("cr", IrValue.Register("r12_mtcrf_preserve"), IrValue.Register("r12_mtcrf_src"), "or"),
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(16), "add"),
                new IrReturn(null)
            })
        });
        var contract = GuestAbiContractAnalyzer.Analyze(new SsaTransformer().Convert(raw).Function);
        Assert.Equal(0, contract.CrPossibleWriteMask & GuestEabi.NonvolatileCrFields);

        // Without mtcrf restore, cr2 stays in CrPossibleWriteMask.
        var unrestoredRaw = new IrFunction("0x80005440", "0x80005440", new[]
        {
            new IrBasicBlock("0x80005440", new IrInstruction[]
            {
                new IrSetCrField(2, IrValue.Register("r3"), IrValue.Imm(0), false),
                new IrReturn(null)
            })
        });
        var unrestoredContract = GuestAbiContractAnalyzer.Analyze(new SsaTransformer().Convert(unrestoredRaw).Function);
        Assert.Equal(1 << 2, unrestoredContract.CrPossibleWriteMask & GuestEabi.NonvolatileCrFields);
    }

    [Fact]
    public void PsMrIsPureAndDoesNotFenceFunction()
    {
        var function = new IrFunction("0x80005500", "0x80005500", new[]
        {
            new IrBasicBlock("0x80005500", new IrInstruction[]
            {
                new IrCall("f1", "PPC_PsMr", new[] { IrValue.Register("f2") }),
                new IrReturn(null)
            })
        });
        var contract = GuestAbiContractAnalyzer.Analyze(function);
        Assert.False(contract.HasFullSynchronizationFence);
        Assert.Equal(0u, contract.GprPossibleWriteMask);
        Assert.Equal(1u << 1, contract.FprPossibleWriteMask);
        Assert.Equal(1u << 2, contract.FprReadBeforeWriteMask);
    }

    [Fact]
    public void RegisteredNativeContractWinsOverReplacedFencedPpcBodyInInterproceduralAnalysis()
    {
        const uint caller = 0x80006000u;
        const uint nativeAddr = 0x801DCF10u; // e.g., OSDisableInterrupts whose PPC body has mfmsr -> IrUndefined
        var functions = new Dictionary<uint, IrFunction>
        {
            [caller] = new("caller", "0x80006000", new[]
            {
                new IrBasicBlock("0x80006000", new IrInstruction[]
                {
                    new IrCall("lr", $"0x{nativeAddr:X8}", Array.Empty<IrValue>()),
                    new IrReturn(null)
                })
            }),
            [nativeAddr] = new("replaced_ppc", $"0x{nativeAddr:X8}", new[]
            {
                new IrBasicBlock($"0x{nativeAddr:X8}", new IrInstruction[]
                {
                    new IrUndefined(nativeAddr, 0x7C6000A6u, "mfmsr r3"),
                    new IrReturn(null)
                })
            })
        };
        var externalContracts = new Dictionary<uint, GuestAbiContract>
        {
            [nativeAddr] = new(
                0u, 1u << 3, 1u << 3,
                0u, 0u, 0u,
                0, 0,
                false, false, false, false, false, false,
                GuestCallBoundaryFlags.None,
                Array.Empty<uint>())
        };

        var result = GuestAbiInterproceduralAnalyzer.Analyze(functions, externalContracts);
        var callerContract = result.Contracts[caller];
        Assert.False(callerContract.HasFullSynchronizationFence);
        Assert.Equal(1u << 3, callerContract.GprPossibleWriteMask);
    }

    [Fact]
    public void IndirectCallAndJumpPreserveNonvolatilesWhileKeepingArchitecturalFlushExceptGeckoHandler()
    {
        var normal = new IrFunction("normal_indirect", "0x80007000", new[]
        {
            new IrBasicBlock("0x80007000", new IrInstruction[]
            {
                new IrAssign("r31", IrValue.Register("r3")),
                new IrAssign("f20", IrValue.Register("f1")),
                new IrSetCrField(2, IrValue.Register("r3"), IrValue.Imm(0), false),
                new IrBinary("r3", IrValue.Register("r3"), IrValue.Register("r4"), "add"),
                new IrIndirectCall(string.Empty, IrValue.Register("r12"), Array.Empty<IrValue>()),
                new IrBinary("r3", IrValue.Register("r31"), IrValue.Register("r3"), "add"),
                new IrAssign("f1", IrValue.Register("f20")),
                new IrBranch("beq", "taken", "done", "cr2")
            }),
            new IrBasicBlock("taken", new IrInstruction[] { new IrReturn(null) }),
            new IrBasicBlock("done", new IrInstruction[] { new IrReturn(null) })
        });

        var normalContract = GuestAbiContractAnalyzer.Analyze(new SsaTransformer().Convert(normal).Function);
        Assert.False(normalContract.HasFullSynchronizationFence);
        // Full architectural reads stay intact so P+ C2 hooks reading ancestor non-volatiles observe CpuContext
        // (only r31 is written by the entry block before the indirect call).
        Assert.Equal(uint.MaxValue & ~(1u << 31), normalContract.GprReadBeforeWriteMask);
        Assert.Equal(0u, normalContract.GprPossibleWriteMask & ((1u << 1) | (1u << 2) | (1u << 13) | (1u << 15) | (1u << 29)));
        Assert.NotEqual(0u, normalContract.GprPossibleWriteMask & (1u << 31)); // written by body itself without restore
        Assert.Equal(GuestEabi.VolatileCrFields | (1 << 2), normalContract.CrPossibleWriteMask);

        var normalBody = FunctionBody(Emit(normal, entryPoint: 0x80007000u), "normal_indirect");
        var icallIdx = normalBody.IndexOf("MKW_INDIRECT_CALL(", StringComparison.Ordinal);
        Assert.True(icallIdx > 0);
        var before = normalBody[..icallIdx];
        var after = normalBody[icallIdx..];
        // Dirty non-volatiles MUST flush before base indirect calls (so P+ C2 hooks reading caller r29/r31 see
        // them in CpuContext), and are NOT reloaded afterwards.
        Assert.Contains("ctx->gpr[31] = r31;", before, StringComparison.Ordinal);
        Assert.Contains("ctx->fpr[20] = f20;", before, StringComparison.Ordinal);
        Assert.Contains("ctx->crv[2] = crv2;", before, StringComparison.Ordinal);
        Assert.Contains("r3 = ctx->gpr[3];", after, StringComparison.Ordinal);
        Assert.Contains("r4 = ctx->gpr[4];", after, StringComparison.Ordinal);
        Assert.DoesNotContain("r31 = ctx->gpr[31];", after, StringComparison.Ordinal);
        Assert.DoesNotContain("f20 = ctx->fpr[20];", after, StringComparison.Ordinal);
        Assert.DoesNotContain("crv2 = ctx->crv[2];", after, StringComparison.Ordinal);

        // Gecko code handler (0x800018A8) MUST keep full sync and reload r15 across its C0 bctrl!
        var gecko = new IrFunction("gecko_handler", "0x800018A8", new[]
        {
            new IrBasicBlock("0x800018A8", new IrInstruction[]
            {
                new IrAssign("r15", IrValue.Register("r3")),
                new IrIndirectCall(string.Empty, IrValue.Register("r15"), Array.Empty<IrValue>()),
                new IrAssign("r3", IrValue.Register("r15")),
                new IrReturn(null)
            })
        });
        var geckoContract = GuestAbiContractAnalyzer.Analyze(new SsaTransformer().Convert(gecko).Function);
        Assert.True(geckoContract.HasFullSynchronizationFence);
        Assert.Equal(uint.MaxValue, geckoContract.GprPossibleWriteMask);

        var geckoBody = FunctionBody(Emit(gecko, entryPoint: 0x800018A8u), "gecko_handler");
        var geckoCallIdx = geckoBody.IndexOf("MKW_INDIRECT_CALL(", StringComparison.Ordinal);
        Assert.True(geckoCallIdx > 0);
        Assert.Contains("r15 = ctx->gpr[15];", geckoBody[geckoCallIdx..], StringComparison.Ordinal);
    }

    [Fact]
    public void SdaBaseFoldingEliminatesR13AndR2LocalsAndFoldsConstantAddresses()
    {
        const uint sda1 = 0x805A4420u;
        const uint sda2 = 0x805A9320u;

        var readerWriter = new IrFunction("sda_user", "0x80008000", new[]
        {
            new IrBasicBlock("0x80008000", new IrInstruction[]
            {
                new IrLoad("r3", new IrAddress("r13", -17216), 4),     // 0x805A00E0
                new IrStore(new IrAddress("r13", -17212), IrValue.Register("r3"), 4), // 0x805A00E4
                new IrLoad("f1", new IrAddress("r2", -24544), 4),      // 0x805A3340
                new IrBinary("r4", IrValue.Register("r13"), IrValue.Imm(-17216), "add"),
                new IrReturn(null)
            })
        });

        var foldedBody = FunctionBody(
            Emit(readerWriter, entryPoint: 0x80008000u, sda1Base: sda1, sda2Base: sda2),
            "sda_user");

        Assert.DoesNotContain("uint32_t r13 =", foldedBody, StringComparison.Ordinal);
        Assert.DoesNotContain("uint32_t r2 =", foldedBody, StringComparison.Ordinal);
        Assert.Contains("MemoryInline::FlatRead32(0x805A00E0u)", foldedBody, StringComparison.Ordinal);
        Assert.Contains("MemoryInline::FlatWriteRam32(0x805A00E4u, r3)", foldedBody, StringComparison.Ordinal);
        Assert.Contains("MemoryInline::FlatReadFloat32(0x805A3340u)", foldedBody, StringComparison.Ordinal);
        Assert.Contains("r4 = 0x805A00E0u;", foldedBody, StringComparison.Ordinal);

        // When SDA bases are omitted (default), r13 and r2 stay ordinary localized registers.
        var unfoldedBody = FunctionBody(Emit(readerWriter, entryPoint: 0x80008000u), "sda_user");
        Assert.Contains("uint32_t r13 = ctx->gpr[13];", unfoldedBody, StringComparison.Ordinal);
        Assert.Contains("uint32_t r2 = ctx->gpr[2];", unfoldedBody, StringComparison.Ordinal);

        // A function that defines r13 itself (like __init_registers) must NOT fold r13.
        var initRegs = new IrFunction("init_regs", "0x800041A8", new[]
        {
            new IrBasicBlock("0x800041A8", new IrInstruction[]
            {
                new IrAssign("r13", IrValue.Imm(unchecked((int)sda1))),
                new IrLoad("r3", new IrAddress("r13", -17216), 4),
                new IrReturn(null)
            })
        });
        var initBody = FunctionBody(
            Emit(initRegs, entryPoint: 0x800041A8u, sda1Base: sda1, sda2Base: sda2),
            "init_regs");
        Assert.Contains("uint32_t r13 = ctx->gpr[13];", initBody, StringComparison.Ordinal);
        Assert.Contains("ctx->gpr[13] = r13;", initBody, StringComparison.Ordinal);
    }

    [Fact]
    public void IndirectTailJumpPreservesStackRestoredNonvolatiles()
    {
        var tail = new IrFunction("0x80009000", "0x80009000", new[]
        {
            new IrBasicBlock("0x80009000", new IrInstruction[]
            {
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(-16), "add"),
                new IrStore(new IrAddress("r1", 12), IrValue.Register("r31"), 4),
                new IrAssign("r31", IrValue.Register("r3")),
                new IrLoad("ctr", new IrAddress("r31", 0), 4),
                new IrLoad("r31", new IrAddress("r1", 12), 4),
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(16), "add"),
                new IrIndirectJump(IrValue.Register("ctr"))
            })
        });

        var contract = GuestAbiContractAnalyzer.Analyze(new SsaTransformer().Convert(tail).Function);
        Assert.False(contract.HasFullSynchronizationFence);
        Assert.Equal(0u, contract.GprPossibleWriteMask & ((1u << 1) | (1u << 2) | (1u << 13) | (1u << 31)));
        Assert.Equal(GuestEabi.VolatileGprMask, contract.GprPossibleWriteMask);
    }

    [Fact]
    public void MultiStoreSdaRangeUsesCheckFreeSegmentStackRange()
    {
        const uint sda1 = 0x805A4420u;
        var multiStore = new IrFunction("sda_multi_store", "0x80009100", new[]
        {
            new IrBasicBlock("0x80009100", new IrInstruction[]
            {
                new IrStore(new IrAddress("r13", -17216), IrValue.Register("r3"), 4), // 0x805A00E0
                new IrStore(new IrAddress("r13", -17212), IrValue.Register("r4"), 4), // 0x805A00E4
                new IrReturn(null)
            })
        });

        var body = FunctionBody(
            Emit(multiStore, entryPoint: 0x80009100u, sda1Base: sda1),
            "sda_multi_store");

        Assert.DoesNotContain("uint32_t r13 =", body, StringComparison.Ordinal);
        Assert.Contains("MemoryInline::ResolveSegmentStackRange(0x805A00E0u, 0, 8u);", body, StringComparison.Ordinal);
    }

    [Fact]
    public void ModSafeNativeClassificationPreservesNonvolatilesWhileKeepingFullFlushAndKeepingOsLoadContextFenced()
    {
        const uint callerMutex = 0x80009200u;
        const uint callerLoadCtx = 0x80009240u;
        const uint osLockMutex = 0x801DEBB4u;
        const uint osLoadContext = 0x801D80B4u;

        static GuestAbiContract CompleteNative(GuestCallBoundaryFlags extra = GuestCallBoundaryFlags.None) =>
            new(uint.MaxValue, uint.MaxValue, (1u << 3) | (1u << 4),
                uint.MaxValue, uint.MaxValue, 1u << 1,
                byte.MaxValue, byte.MaxValue, true, true, true, true, true, true,
                extra | GuestCallBoundaryFlags.RequiresCompleteContext, Array.Empty<uint>());

        var functions = new Dictionary<uint, IrFunction>
        {
            [callerMutex] = new("caller_mutex", "0x80009200", new[]
            {
                new IrBasicBlock("0x80009200", new IrInstruction[]
                {
                    new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(-16), "add"),
                    new IrStore(new IrAddress("r1", 12), IrValue.Register("r31"), 4),
                    new IrAssign("r31", IrValue.Register("r3")),
                    new IrCall("lr", $"0x{osLockMutex:X8}", Array.Empty<IrValue>()),
                    new IrLoad("r3", new IrAddress("r13", -17216), 4),
                    new IrBinary("r3", IrValue.Register("r3"), IrValue.Register("r31"), "add"),
                    new IrLoad("r31", new IrAddress("r1", 12), 4),
                    new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(16), "add"),
                    new IrReturn(null)
                })
            }),
            [callerLoadCtx] = new("caller_loadctx", "0x80009240", new[]
            {
                new IrBasicBlock("0x80009240", new IrInstruction[]
                {
                    new IrCall("lr", $"0x{osLoadContext:X8}", Array.Empty<IrValue>()),
                    new IrReturn(null)
                })
            })
        };

        var externalContracts = new Dictionary<uint, GuestAbiContract>
        {
            [osLockMutex] = CompleteNative(GuestCallBoundaryFlags.CanSwitchThreads),
            [osLoadContext] = CompleteNative(GuestCallBoundaryFlags.InvokesGuestCode)
        };

        var ssaFunctions = new Dictionary<uint, IrFunction>();
        foreach (var (addr, fn) in functions)
            ssaFunctions[addr] = new SsaTransformer().Convert(fn).Function;

        var result = GuestAbiInterproceduralAnalyzer.Analyze(ssaFunctions, externalContracts);

        var mutexCallerContract = result.Contracts[callerMutex];
        Assert.False(mutexCallerContract.HasFullSynchronizationFence);
        Assert.True(mutexCallerContract.BoundaryFlags.HasFlag(GuestCallBoundaryFlags.CanSwitchThreads));
        Assert.True(mutexCallerContract.BoundaryFlags.HasFlag(GuestCallBoundaryFlags.InvokesGuestCode));
        Assert.Equal(0u, mutexCallerContract.GprPossibleWriteMask & ((1u << 1) | (1u << 2) | (1u << 13) | (1u << 31)));

        // OSLoadContext (0x801D80B4) MUST remain fully fenced.
        var loadCtxCallerContract = result.Contracts[callerLoadCtx];
        Assert.True(loadCtxCallerContract.HasFullSynchronizationFence);
        Assert.Equal(uint.MaxValue, loadCtxCallerContract.GprPossibleWriteMask);

        // Emitting caller_mutex with classified native contract: flushes dirty r1 and r31 before OSLockMutex
        // (keeping CpuContext architectural for thread suspension / PoyoJIT / C2 hooks), does NOT reload r1/r13/r31
        // afterwards, restores ctx->gpr[31] at return because it was flushed before OSLockMutex, and folds r13!
        var emitContracts = new Dictionary<uint, GuestAbiContract>(result.Contracts)
        {
            [osLockMutex] = GuestEabi.ClassifyModSafeNative(osLockMutex, externalContracts[osLockMutex])
        };
        var body = FunctionBody(
            Emit(functions[callerMutex], entryPoint: callerMutex, guestAbiContracts: emitContracts, sda1Base: 0x805A4420u),
            "caller_mutex");
        var callIdx = body.IndexOf("0x801DEBB4u", StringComparison.Ordinal);
        Assert.True(callIdx > 0);
        Assert.Contains("ctx->gpr[1] = r1;", body[..callIdx], StringComparison.Ordinal);
        Assert.Contains("ctx->gpr[31] = r31;", body[..callIdx], StringComparison.Ordinal);
        Assert.DoesNotContain("r31 = ctx->gpr[31];", body[callIdx..], StringComparison.Ordinal);
        Assert.DoesNotContain("r1 = ctx->gpr[1];", body[callIdx..], StringComparison.Ordinal);
        Assert.Contains("ctx->gpr[31] = r31;", body[callIdx..], StringComparison.Ordinal);
        Assert.DoesNotContain("uint32_t r13 =", body, StringComparison.Ordinal);
        Assert.Contains("MemoryInline::FlatRead32(0x805A00E0u)", body, StringComparison.Ordinal);

        // Even if r31 is clean (unmodified) before calling a clobbering/fenced callee, the callee can overwrite
        // ctx->gpr[31], so restoring r31 from stack afterwards MUST still flush ctx->gpr[31] at return!
        var callsClobberClean = new IrFunction("calls_clobber_clean", "0x80009280", new[]
        {
            new IrBasicBlock("0x80009280", new IrInstruction[]
            {
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(-16), "add"),
                new IrStore(new IrAddress("r1", 12), IrValue.Register("r31"), 4),
                // r31 is NOT modified before the fenced call, so pre-call flush does not store ctx->gpr[31],
                // yet the callee's reload mask includes r31 (callee may clobber ctx->gpr[31]).
                new IrCall("lr", $"0x{osLoadContext:X8}", Array.Empty<IrValue>()),
                new IrLoad("r31", new IrAddress("r1", 12), 4),
                new IrBinary("r1", IrValue.Register("r1"), IrValue.Imm(16), "add"),
                new IrReturn(null)
            })
        });
        var fencedEmitContracts = new Dictionary<uint, GuestAbiContract>
        {
            [osLoadContext] = CompleteNative(GuestCallBoundaryFlags.InvokesGuestCode)
        };
        var callsFencedBody = FunctionBody(
            Emit(callsClobberClean, entryPoint: 0x80009280u, guestAbiContracts: fencedEmitContracts),
            "calls_clobber_clean");
        var fencedCallIdx = callsFencedBody.IndexOf("0x801D80B4u", StringComparison.Ordinal);
        Assert.True(fencedCallIdx > 0);
        Assert.DoesNotContain("ctx->gpr[31] = r31;", callsFencedBody[..fencedCallIdx], StringComparison.Ordinal);
        Assert.Contains("ctx->gpr[31] = r31;", callsFencedBody[fencedCallIdx..], StringComparison.Ordinal);
    }

    [Fact]
    public void NonLocalExitUnwinderDoesNotPolluteCallersWithSdaOrNonvolatileWrites()
    {
        const uint unwinder = 0x803F2948u;
        const uint throwCaller = 0x80009300u;

        var functions = new Dictionary<uint, IrFunction>
        {
            [unwinder] = new($"0x{unwinder:X8}", $"0x{unwinder:X8}", new[]
            {
                new IrBasicBlock($"0x{unwinder:X8}", new IrInstruction[]
                {
                    new IrLoad("r1", new IrAddress("r3", 0), 4),
                    new IrLoad("r2", new IrAddress("r3", 4), 4),
                    new IrLoad("r13", new IrAddress("r3", 8), 4),
                    new IrLoad("r31", new IrAddress("r3", 12), 4),
                    new IrLoad("f31", new IrAddress("r3", 16), 8),
                    new IrSetCrField(2, IrValue.Register("r3"), IrValue.Imm(0), false),
                    new IrReturn(null)
                })
            }),
            [throwCaller] = new("throw_caller", "0x80009300", new[]
            {
                new IrBasicBlock("0x80009300", new IrInstruction[]
                {
                    new IrBranch("beq", "do_throw", "normal", "cr0")
                }),
                new IrBasicBlock("do_throw", new IrInstruction[]
                {
                    new IrCall("lr", $"0x{unwinder:X8}", Array.Empty<IrValue>()),
                    new IrReturn(null)
                }),
                new IrBasicBlock("normal", new IrInstruction[]
                {
                    new IrLoad("r3", new IrAddress("r13", -17216), 4),
                    new IrReturn(null)
                })
            })
        };

        var ssaFunctions = new Dictionary<uint, IrFunction>();
        foreach (var (addr, fn) in functions)
            ssaFunctions[addr] = new SsaTransformer().Convert(fn).Function;

        var result = GuestAbiInterproceduralAnalyzer.Analyze(ssaFunctions);
        var unwinderContract = result.Contracts[unwinder];
        Assert.Equal(0u, unwinderContract.GprPossibleWriteMask);
        Assert.Equal(0u, unwinderContract.FprPossibleWriteMask);
        Assert.Equal(0, unwinderContract.CrPossibleWriteMask);
        Assert.NotEqual(0u, unwinderContract.GprReadBeforeWriteMask & (1u << 3));

        var callerContract = result.Contracts[throwCaller];
        Assert.False(callerContract.HasFullSynchronizationFence);
        Assert.Equal(0u, callerContract.GprPossibleWriteMask & ((1u << 1) | (1u << 2) | (1u << 13) | (1u << 31)));
    }

    [Fact]
    public void CodeVariantIndirectCallsKeepFullNonvolatileReloadAndDisableSdaFoldingAcrossIndirectCall()
    {
        // Models a P+ C2 hook spliced into a code variant calling a custom non-EABI ASM helper via bctrl
        // that returns results in r28 and f29 (such as 0x8058549C / 0x80585750).
        var variantFn = new IrFunction("func_80867420_pp", "0x80867420", new[]
        {
            new IrBasicBlock("0x80867420", new IrInstruction[]
            {
                new IrAssign("r28", IrValue.Register("r3")),
                new IrAssign("f29", IrValue.Register("f1")),
                new IrIndirectCall(string.Empty, IrValue.Register("r24"), Array.Empty<IrValue>()),
                new IrStore(new IrAddress("r2", 16), IrValue.Register("r28"), 4),
                new IrAssign("f1", IrValue.Register("f29")),
                new IrReturn(null)
            })
        });

        var variantBody = FunctionBody(
            Emit(
                variantFn,
                entryPoint: 0x80867420u,
                sda1Base: 0x805A4420u,
                sda2Base: 0x805A9320u,
                variantOwnCode: new[] { 0x80867420u, 0x80867500u }),
            "func_80867420_pp");

        var icallIdx = variantBody.IndexOf("MKW_INDIRECT_CALL(", StringComparison.Ordinal);
        Assert.True(icallIdx > 0);
        var after = variantBody[icallIdx..];
        Assert.Contains("r28 = ctx->gpr[28];", after, StringComparison.Ordinal);
        Assert.Contains("f29 = ctx->fpr[29];", after, StringComparison.Ordinal);
        Assert.Contains("uint32_t r2 = ctx->gpr[2];", variantBody, StringComparison.Ordinal);
        Assert.Contains("r2 = ctx->gpr[2];", after, StringComparison.Ordinal);
    }
}
