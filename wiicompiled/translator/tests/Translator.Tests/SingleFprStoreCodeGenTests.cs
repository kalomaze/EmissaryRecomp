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

        Assert.Contains("MemoryInline::FlatWrite32(r5, fbits_f1", code, StringComparison.Ordinal);
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
            ["f4"] = ValueRepresentation.Float64
        });
        var signature = new FunctionAbiClassification("single_store_probe", ValueRepresentation.Void);
        return new CxxLinearCodeGenerator().Emit(0x80040000, new SsaTransformer().Convert(function), signature, types);
    }
}
