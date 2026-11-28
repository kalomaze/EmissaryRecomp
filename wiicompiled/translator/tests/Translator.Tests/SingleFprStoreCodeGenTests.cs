using System;
using System.Collections.Generic;
using Translator.Core.Analysis.Representation;
using Translator.Core.Analysis.Ssa;
using Translator.Core.CodeGen;
using Translator.Core.Ir;
using Translator.Core.Representation;
using Xunit;

namespace Translator.Tests;

/// <summary>
/// stfs of a value whose single bits are known (an lfs, fadds/fsubs/fdivs, frsp, or an fmr of one) stores those bits
/// with the integer store, and fadds/fsubs of two singles round by the plain conversion
/// (CxxLinearCodeGenerator.SingleFpr.cs).
/// </summary>
public sealed class SingleFprStoreCodeGenTests
{
    [Fact]
    public void LfsThenStfsStoresTheLoadedWord()
    {
        var code = Emit(
            new IrLoad("f1", new IrAddress("r4", 0), 4),
            new IrStore(new IrAddress("r5", 8), IrValue.Register("f1"), 4));

        Assert.Contains("= MemoryInline::FlatRead32(r4);", code, StringComparison.Ordinal);
        Assert.Contains("PpcConvertToDoubleBitsInline(fbits_f1", code, StringComparison.Ordinal);
        Assert.Contains("MemoryInline::FlatWrite32((r5 + 8), fbits_f1", code, StringComparison.Ordinal);
        Assert.DoesNotContain("FlatWriteFloat32", code, StringComparison.Ordinal);
        Assert.DoesNotContain("ConvertPpcDoubleToSingleBits", code, StringComparison.Ordinal);
    }

    [Fact]
    public void FmrOfAnLfsStoresTheLoadedWord()
    {
        var code = Emit(
            new IrLoad("f1", new IrAddress("r4", 0), 4),
            new IrAssign("f2", IrValue.Register("f1")),
            new IrStore(new IrAddress("r5", 0), IrValue.Register("f2"), 4));

        // The move copies the word into the move's own bits local (CxxLinearCodeGenerator.SingleFpr.cs).
        Assert.Contains("fbits_f2_0 = fbits_f1_0;", code, StringComparison.Ordinal);
        Assert.Contains("MemoryInline::FlatWrite32(r5, fbits_f2_0);", code, StringComparison.Ordinal);
        Assert.DoesNotContain("FlatWriteFloat32", code, StringComparison.Ordinal);
    }

    [Fact]
    public void StfsOfAnUnknownValueKeepsTheNarrowing()
    {
        var code = Emit(new IrStore(new IrAddress("r5", 0), IrValue.Register("f1"), 4));

        Assert.Contains("MemoryInline::FlatWriteFloat32(r5, f1.d);", code, StringComparison.Ordinal);
        Assert.DoesNotContain("fbits_", code, StringComparison.Ordinal);
    }

    [Fact]
    public void FaddsOfTwoSinglesRoundsByTheConversionAndKeepsItsBitsForAStore()
    {
        var code = Emit(
            new IrLoad("f1", new IrAddress("r4", 0), 4),
            new IrLoad("f2", new IrAddress("r4", 4), 4),
            new IrCall("f3", "PPC_Fadds", new[] { IrValue.Register("f1"), IrValue.Register("f2") }),
            new IrStore(new IrAddress("r5", 0), IrValue.Register("f3"), 4));

        Assert.Contains("= PpcBitCastToU32Inline(static_cast<float>(PpcFaddInline(", code, StringComparison.Ordinal);
        Assert.Contains("f3.d = static_cast<double>(PpcBitCastToFloatInline(fbits_f3", code, StringComparison.Ordinal);
        Assert.Contains("MemoryInline::FlatWrite32(r5, fbits_f3", code, StringComparison.Ordinal);
        Assert.DoesNotContain("PpcForceSingleValueInline", code, StringComparison.Ordinal);
    }

    [Fact]
    public void FaddsOfAnUnknownOperandKeepsTheNiMask()
    {
        var code = Emit(
            new IrLoad("f1", new IrAddress("r4", 0), 4),
            new IrCall("f3", "PPC_Fadds", new[] { IrValue.Register("f1"), IrValue.Register("f2") }));

        Assert.Contains("PpcForceSingleValueInline(PpcFaddInline(", code, StringComparison.Ordinal);
    }

    [Fact]
    public void FdivsOfTwoSinglesKeepsTheNiMask()
    {
        // A quotient of singles can fall in [2^-126 - 2^-150, 2^-126), where the mask and the conversion differ.
        var code = Emit(
            new IrLoad("f1", new IrAddress("r4", 0), 4),
            new IrLoad("f2", new IrAddress("r4", 4), 4),
            new IrCall("f3", "PPC_Fdivs", new[] { IrValue.Register("f1"), IrValue.Register("f2") }),
            new IrStore(new IrAddress("r5", 0), IrValue.Register("f3"), 4));

        Assert.Contains("PpcBitCastToU32Inline(PpcForceSingleValueInline(PpcFdivInline(", code, StringComparison.Ordinal);
        Assert.Contains("MemoryInline::FlatWrite32(r5, fbits_f3", code, StringComparison.Ordinal);
    }

    [Fact]
    public void FrspOfASingleRoundsByTheConversion()
    {
        var code = Emit(
            new IrLoad("f1", new IrAddress("r4", 0), 4),
            new IrBinary("f2", IrValue.Register("f1"), IrValue.Imm(0), "frsp"),
            new IrBinary("f3", IrValue.Register("f4"), IrValue.Imm(0), "frsp"));

        Assert.Contains("f2.d = static_cast<double>(static_cast<float>(f1.d));", code, StringComparison.Ordinal);
        Assert.Contains("f3.d = static_cast<double>(PpcForceSingleValueInline(f4.d));", code, StringComparison.Ordinal);
    }

    [Fact]
    public void AGuestCallBetweenTheLoadAndTheStoreKeepsTheNarrowing()
    {
        // The IR keeps f1's name across the call, but the emitted code reloads f1 from CpuContext (the callee's value).
        var code = Emit(
            new IrLoad("f1", new IrAddress("r4", 0), 4),
            new IrCall(string.Empty, "0x80001234", Array.Empty<IrValue>()),
            new IrStore(new IrAddress("r5", 0), IrValue.Register("f1"), 4));

        Assert.Contains("MemoryInline::FlatWriteFloat32(r5, f1.d);", code, StringComparison.Ordinal);
        Assert.DoesNotContain("FlatWrite32(r5", code, StringComparison.Ordinal);
    }

    [Fact]
    public void AGuestCallEndsSinglePrecisionProvenance()
    {
        var code = Emit(
            new IrLoad("f1", new IrAddress("r4", 0), 4),
            new IrLoad("f2", new IrAddress("r4", 8), 4),
            new IrCall(string.Empty, "0x80001234", Array.Empty<IrValue>()),
            new IrCall("f3", "PPC_Fadds", new[] { IrValue.Register("f1"), IrValue.Register("f2") }),
            new IrCall("f4", "PPC_Fmuls", new[] { IrValue.Register("f2"), IrValue.Register("f1") }));

        Assert.Contains("PpcForceSingleValueInline(PpcFaddInline(", code, StringComparison.Ordinal);
        Assert.Contains("PpcFmulsInline(", code, StringComparison.Ordinal);
        Assert.DoesNotContain("SingleC", code, StringComparison.Ordinal);
    }

    [Fact]
    public void FmulsOfAnIntactSingleCKeepsTheSingleCForm()
    {
        var code = Emit(
            new IrLoad("f1", new IrAddress("r4", 0), 4),
            new IrCall("f4", "PPC_Fmuls", new[] { IrValue.Register("f2"), IrValue.Register("f1") }));

        Assert.Contains("PpcFmulsSingleCInline(", code, StringComparison.Ordinal);
    }

    [Fact]
    public void FmulsThenStfsStoresTheProductsWord()
    {
        var code = Emit(
            new IrLoad("f1", new IrAddress("r4", 0), 4),
            new IrCall("f3", "PPC_Fmuls", new[] { IrValue.Register("f2"), IrValue.Register("f1") }),
            new IrStore(new IrAddress("r5", 0), IrValue.Register("f3"), 4));

        Assert.Contains("= PpcBitCastToU32Inline(PpcFmulsFloatInline<true>(f2.d, f1.d));", code, StringComparison.Ordinal);
        Assert.Contains("f3.d = static_cast<double>(PpcBitCastToFloatInline(fbits_f3", code, StringComparison.Ordinal);
        Assert.Contains("MemoryInline::FlatWrite32(r5, fbits_f3", code, StringComparison.Ordinal);
        Assert.DoesNotContain("FlatWriteFloat32", code, StringComparison.Ordinal);
    }

    [Fact]
    public void FnmsubsThenStfsStoresTheResultsWord()
    {
        var code = Emit(
            new IrCall("f3", "PPC_Fnmsubs", new[] { IrValue.Register("f1"), IrValue.Register("f2"), IrValue.Register("f4") }),
            new IrStore(new IrAddress("r5", 0), IrValue.Register("f3"), 4));

        Assert.Contains("PpcFusedSingleFloatInline<true, true, false>(f1.d, f2.d, f4.d)", code, StringComparison.Ordinal);
        Assert.Contains("MemoryInline::FlatWrite32(r5, fbits_f3", code, StringComparison.Ordinal);
    }

    [Fact]
    public void FmulsWithoutAStoreKeepsTheDoubleEntryPoint()
    {
        var code = Emit(
            new IrCall("f3", "PPC_Fmuls", new[] { IrValue.Register("f2"), IrValue.Register("f1") }));

        Assert.Contains("f3.d = PpcFmulsInline(f2.d, f1.d);", code, StringComparison.Ordinal);
        Assert.DoesNotContain("fbits_", code, StringComparison.Ordinal);
    }

    [Fact]
    public void ANonvolatileSingleKeepsItsBitsAcrossAnIndirectCall()
    {
        // Indirect calls reload only the EABI volatiles (MKW_MODSAFE_INDIRECT_ABI): f20's local survives the call, so
        // its loaded word is still the value to store, and it is still single for fmuls.
        var code = Emit(
            new IrLoad("f20", new IrAddress("r4", 0), 4),
            new IrIndirectCall(string.Empty, IrValue.Register("r12"), Array.Empty<IrValue>()),
            new IrStore(new IrAddress("r5", 0), IrValue.Register("f20"), 4),
            new IrCall("f4", "PPC_Fmuls", new[] { IrValue.Register("f2"), IrValue.Register("f20") }));

        Assert.Contains("MemoryInline::FlatWrite32(r5, fbits_f20", code, StringComparison.Ordinal);
        Assert.DoesNotContain("FlatWriteFloat32", code, StringComparison.Ordinal);
        Assert.Contains("PpcFmulsSingleCInline(", code, StringComparison.Ordinal);
    }

    [Fact]
    public void AVolatileSingleLosesItsBitsAcrossAnIndirectCall()
    {
        var code = Emit(
            new IrLoad("f1", new IrAddress("r4", 0), 4),
            new IrIndirectCall(string.Empty, IrValue.Register("r12"), Array.Empty<IrValue>()),
            new IrStore(new IrAddress("r5", 0), IrValue.Register("f1"), 4));

        Assert.Contains("MemoryInline::FlatWriteFloat32(r5, f1.d);", code, StringComparison.Ordinal);
    }

    [Fact]
    public void ASingleKeepsItsBitsIntoALaterBlock()
    {
        // Defined in the entry block, stored in both arms and after the merge: each store reuses the loaded word.
        var code = EmitBlocks(
            new IrBasicBlock("entry", new List<IrInstruction>
            {
                new IrLoad("f20", new IrAddress("r4", 0), 4),
                new IrAssign("f21", IrValue.Register("f20")),
                new IrBranch("ne", "left", "right")
            }),
            new IrBasicBlock("left", new List<IrInstruction>
            {
                new IrStore(new IrAddress("r5", 0), IrValue.Register("f20"), 4),
                new IrJump("join")
            }),
            new IrBasicBlock("right", new List<IrInstruction>
            {
                new IrIndirectCall(string.Empty, IrValue.Register("r12"), Array.Empty<IrValue>()),
                new IrJump("join")
            }),
            new IrBasicBlock("join", new List<IrInstruction>
            {
                new IrStore(new IrAddress("r5", 4), IrValue.Register("f21"), 4),
                new IrReturn(null)
            }));

        Assert.Contains("MemoryInline::FlatWrite32(r5, fbits_f20", code, StringComparison.Ordinal);
        Assert.Contains("fbits_f21_", code, StringComparison.Ordinal);
        Assert.Contains("MemoryInline::FlatWrite32((r5 + 4), fbits_f21", code, StringComparison.Ordinal);
        Assert.DoesNotContain("FlatWriteFloat32", code, StringComparison.Ordinal);
    }

    [Fact]
    public void AVolatileSingleLosesItsBitsOnAPathThroughACall()
    {
        var code = EmitBlocks(
            new IrBasicBlock("entry", new List<IrInstruction>
            {
                new IrLoad("f3", new IrAddress("r4", 0), 4),
                new IrBranch("ne", "left", "join")
            }),
            new IrBasicBlock("left", new List<IrInstruction>
            {
                new IrIndirectCall(string.Empty, IrValue.Register("r12"), Array.Empty<IrValue>()),
                new IrJump("join")
            }),
            new IrBasicBlock("join", new List<IrInstruction>
            {
                new IrStore(new IrAddress("r5", 0), IrValue.Register("f3"), 4),
                new IrReturn(null)
            }));

        Assert.Contains("MemoryInline::FlatWriteFloat32(r5, f3.d);", code, StringComparison.Ordinal);
    }

    private static string Emit(params IrInstruction[] instructions)
    {
        var body = new List<IrInstruction>(instructions) { new IrReturn(null) };
        var function = new IrFunction("single_store_probe", "entry", new[] { new IrBasicBlock("entry", body) });
        var types = new RepresentationEnvironment(new Dictionary<string, ValueRepresentation>
        {
            ["r4"] = ValueRepresentation.UInt32,
            ["r5"] = ValueRepresentation.UInt32,
            ["f1"] = ValueRepresentation.Float64,
            ["f2"] = ValueRepresentation.Float64,
            ["f3"] = ValueRepresentation.Float64,
            ["f4"] = ValueRepresentation.Float64,
            ["f20"] = ValueRepresentation.Float64,
            ["r12"] = ValueRepresentation.UInt32
        });
        var signature = new FunctionAbiClassification("single_store_probe", ValueRepresentation.Void);
        return new CxxLinearCodeGenerator().Emit(0x80040000, new SsaTransformer().Convert(function), signature, types);
    }

    private static string EmitBlocks(params IrBasicBlock[] blocks)
    {
        var function = new IrFunction("single_store_probe", blocks[0].Label, blocks);
        var types = new RepresentationEnvironment(new Dictionary<string, ValueRepresentation>
        {
            ["r4"] = ValueRepresentation.UInt32,
            ["r5"] = ValueRepresentation.UInt32,
            ["r12"] = ValueRepresentation.UInt32,
            ["f3"] = ValueRepresentation.Float64,
            ["f20"] = ValueRepresentation.Float64,
            ["f21"] = ValueRepresentation.Float64
        });
        var signature = new FunctionAbiClassification("single_store_probe", ValueRepresentation.Void);
        return new CxxLinearCodeGenerator().Emit(0x80040000, new SsaTransformer().Convert(function), signature, types);
    }
}
