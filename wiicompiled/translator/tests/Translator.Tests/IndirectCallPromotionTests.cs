using System;
using System.Collections.Generic;
using System.Linq;
using Translator.Core.Analysis.Representation;
using Translator.Core.Analysis.Ssa;
using Translator.Core.CodeGen;
using Translator.Core.Ir;
using Translator.Core.Representation;
using Xunit;

namespace Translator.Tests;

public class IndirectCallPromotionTests
{
    // Entries no other test uses: the promotion table is process-wide.
    private const uint Entry = 0x80F01000u;
    private const uint OtherEntry = 0x80F02000u;
    private const uint JumpEntry = 0x80F03000u;

    static IndirectCallPromotionTests()
    {
        IndirectCallProfile.SetForTesting(new Dictionary<(uint Function, uint Site), uint[]>
        {
            [(Entry, 0x80F01010u)] = new[] { 0x80005000u },
            [(OtherEntry, 0x80F02010u)] = new[] { 0x80005000u, 0x80006000u },
        }, new Dictionary<(uint Function, uint Site), uint[]>
        {
            [(JumpEntry, 0x80F03020u)] = new[] { 0x80005000u },
        });
    }

    [Fact]
    public void SelectKeepsDominantTargets()
    {
        var sites = IndirectCallProfile.Parse(new[]
        {
            "# comment",
            "J 80001000 80001010 80007000 99999",
            "80001000 80001010 80005000 900",
            "80001000 80001010 80006000 100",
            "80002000 80002010 80005000 50",
            "80002000 80002010 80006000 50",
            "80003000 80003010 80005000 10",
            "80004000 80004010 80005000 40",
            "80004000 80004010 80006000 30",
            "80004000 80004010 80007000 30",
        });
        var one = IndirectCallProfile.Select(sites, minCount: 64, minShare: 0.3, maxTargets: 1, minCover: 0.8);
        Assert.Equal(new[] { 0x80005000u }, one[(0x80001000u, 0x80001010u)]);
        Assert.False(one.ContainsKey((0x80002000u, 0x80002010u)));  // 50% < cover
        Assert.False(one.ContainsKey((0x80003000u, 0x80003010u)));  // too few calls
        var two = IndirectCallProfile.Select(sites, minCount: 64, minShare: 0.3, maxTargets: 2, minCover: 0.8);
        Assert.Equal(new[] { 0x80005000u }, two[(0x80001000u, 0x80001010u)]);  // 10% < share
        Assert.Equal(new[] { 0x80005000u, 0x80006000u }, two[(0x80002000u, 0x80002010u)]);
        Assert.False(two.ContainsKey((0x80004000u, 0x80004010u)));  // two targets cover 70%
        var jumps = IndirectCallProfile.Select(IndirectCallProfile.Parse(new[] { "J 80001000 80001010 80007000 99999" }, jumps: true),
            minCount: 64, minShare: 0.3, maxTargets: 1, minCover: 0.8);
        Assert.Equal(new[] { 0x80007000u }, jumps[(0x80001000u, 0x80001010u)]);
    }

    [Fact]
    public void PromotedTailJumpCallsDirectlyThenReturns()
    {
        var body = new IrInstruction[]
        {
            new IrAssign("r3", IrValue.Imm(7)),
            new IrTracePpc(0x80F03020u, "bctr", "0x4E800420"),
            new IrIndirectJump(IrValue.Register("ctr")),
        };
        var function = new IrFunction("promoted_jump", "entry", new[] { new IrBasicBlock("entry", body) });
        var types = new Dictionary<string, ValueRepresentation>(StringComparer.OrdinalIgnoreCase)
        {
            ["r3"] = ValueRepresentation.UInt32,
            ["ctr"] = ValueRepresentation.UInt32,
        };
        var code = new CxxLinearCodeGenerator().Emit(
            JumpEntry,
            new SsaTransformer().Convert(function),
            new FunctionAbiClassification("promoted_jump", ValueRepresentation.Void),
            new RepresentationEnvironment(types));
        Assert.Contains(
            "if ((ctr) == 0x80005000u) [[likely]] { InvokeDirectCpu<0x80005000u>(ctx); } else { MKW_INDIRECT_JUMP(ctr, ctx); }",
            code);
    }

    [Fact]
    public void PromotedSiteComparesAndCallsDirectlyInsideTheIndirectBoundary()
    {
        var code = Emit(Entry, 0x80F01014u);
        var line = "if ((ctr) == 0x80005000u) [[likely]] { InvokeDirectCpu<0x80005000u>(ctx); } else { MKW_INDIRECT_CALL(ctr, ctx); }";
        var at = code.IndexOf(line, StringComparison.Ordinal);
        Assert.True(at >= 0, code);
        // The flush comes before the compare, the reload after it: both arms share the indirect call's boundary.
        Assert.True(code.IndexOf("ctx->gpr[3] = r3;", StringComparison.Ordinal) is var flush && flush >= 0 && flush < at, code);
        Assert.Contains("r3 = ctx->gpr[3];", code[at..]);
    }

    [Fact]
    public void TwoTargetSiteChainsCompares()
    {
        var code = Emit(OtherEntry, 0x80F02014u);
        Assert.Contains(
            "if ((ctr) == 0x80005000u) [[likely]] { InvokeDirectCpu<0x80005000u>(ctx); } else if ((ctr) == 0x80006000u) { InvokeDirectCpu<0x80006000u>(ctx); } else { MKW_INDIRECT_CALL(ctr, ctx); }",
            code);
    }

    [Fact]
    public void UnprofiledSiteStaysIndirect()
    {
        var code = Emit(Entry, 0x80F01024u);
        Assert.Contains("MKW_INDIRECT_CALL(ctr, ctx);", code);
        Assert.DoesNotContain("InvokeDirectCpu", code);
    }

    [Fact]
    public void SiteAddressComesFromTheReturnAddressOrTrace()
    {
        var withLr = new IrInstruction[]
        {
            new IrAssign("lr", IrValue.Imm(unchecked((int)0x80F01014u))),
            new IrIndirectCall(string.Empty, IrValue.Register("ctr"), Array.Empty<IrValue>()),
        };
        Assert.Equal(0x80F01010u, CxxLinearCodeGenerator.FindGuestSiteAddress(withLr, 1));
        var withTrace = new IrInstruction[]
        {
            new IrIndirectCall(string.Empty, IrValue.Register("ctr"), Array.Empty<IrValue>()),
            new IrTracePpc(0x80F01010u, "bctrl", "0x4E800421"),
        };
        Assert.Equal(0x80F01010u, CxxLinearCodeGenerator.FindGuestSiteAddress(withTrace, 0));
        var jump = new IrInstruction[]
        {
            new IrTracePpc(0x80F01020u, "bctr", "0x4E800420"),
            new IrIndirectJump(IrValue.Register("ctr")),
        };
        Assert.Equal(0x80F01020u, CxxLinearCodeGenerator.FindGuestSiteAddress(jump, 1));
    }

    private static string Emit(uint entry, uint returnAddress)
    {
        var body = new IrInstruction[]
        {
            new IrAssign("r3", IrValue.Imm(7)),
            new IrAssign("lr", IrValue.Imm(unchecked((int)returnAddress))),
            new IrIndirectCall(string.Empty, IrValue.Register("ctr"), new[] { IrValue.Register("r3") }),
            new IrAssign("r4", IrValue.Register("r3")),
            new IrReturn(null),
        };
        var function = new IrFunction("promoted", "entry", new[] { new IrBasicBlock("entry", body) });
        var types = new Dictionary<string, ValueRepresentation>(StringComparer.OrdinalIgnoreCase)
        {
            ["r3"] = ValueRepresentation.UInt32,
            ["r4"] = ValueRepresentation.UInt32,
            ["ctr"] = ValueRepresentation.UInt32,
            ["lr"] = ValueRepresentation.UInt32,
        };
        return new CxxLinearCodeGenerator().Emit(
            entry,
            new SsaTransformer().Convert(function),
            new FunctionAbiClassification("promoted", ValueRepresentation.Void),
            new RepresentationEnvironment(types));
    }
}
