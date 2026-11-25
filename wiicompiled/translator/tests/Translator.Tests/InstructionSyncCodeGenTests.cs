using Translator.Core.Analysis.Ssa;
using Translator.Core.Analysis.Representation;
using Translator.Core.CodeGen;
using Translator.Core.Ir;
using Translator.Core.Representation;
using Xunit;

namespace Translator.Tests;

public class InstructionSyncCodeGenTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FullContextInstructionSyncInterpretsFromEntry(bool mod)
    {
        var function = new IrFunction("func_80001000", "0x80001000", new[]
        {
            new IrBasicBlock("0x80001000", new IrInstruction[]
            {
                new IrAssign("r3", IrValue.Imm(77)),
                new IrComment("isync @ 0x80001004 (no-op)"),
                new IrCall(string.Empty, "0x80002000", Array.Empty<IrValue>()),
                new IrReturn(null)
            })
        });
        var types = new RepresentationEnvironment(new Dictionary<string, ValueRepresentation>
        {
            ["r3"] = ValueRepresentation.UInt32
        });
        var result = new CxxLinearCodeGenerator().EmitWithFacts(0x80001000,
            new SsaTransformer().Convert(function),
            new FunctionAbiClassification("func_80001000", ValueRepresentation.Void), types,
            emitModRegistration: mod);

        Assert.Contains("#include \"ppc_interpreter.h\"", result.Code);
        Assert.Contains("PpcInterpreter::Run(ctx, 0x80001000u);", result.Code);
        Assert.DoesNotContain("r3 = 77;", result.Code);
        Assert.DoesNotContain("InvokeDirectCpu<0x80002000u>", result.Code);
        Assert.Empty(result.EmittedDirectCallTargets);
        Assert.Empty(result.EmittedLocalLabelAddresses);
    }
}
