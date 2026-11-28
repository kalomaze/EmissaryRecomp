using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Translator.Core.Build;
using Xunit;

namespace Translator.Tests;

public sealed class TranslatedBuildShardEmitterTests
{
    [Theory]
    [InlineData(0x809435F0u, "brawl_rollback.h", "BrawlRollback::SeedCrossplayStage();")]
    [InlineData(0x80949BC0u, "brawl_online_ipc.h", "BrawlOnline::Stage();")]
    public void FreshStageShardsContainOnlineSetupGateAfterCodeGuard(uint address, string header, string hook)
    {
        var root = Path.Combine(Path.GetTempPath(), $"translator-stage-gate-{Guid.NewGuid():N}");
        var functions = Path.Combine(root, "functions");
        var native = Path.Combine(root, "runtime");
        var output = Path.Combine(root, "out");
        Directory.CreateDirectory(functions);
        Directory.CreateDirectory(native);
        try
        {
            var stage = $$"""
                extern "C" void func_{{address:X8}}(CpuContext* MKW_RESTRICT ctx) {
                    ctx->gpr[3] = 7;
                }
                // RECOMP_REGISTRATION base 0x{{address:X8}} func_{{address:X8}} preserves=true fpr_mask=0x00000000
                """;
            Write(Path.Combine(functions, $"func_{address:X8}.cpp"), stage);
            var metadata = Path.Combine(root, "base_output.json");
            BaseTranslationOutputMetadataFile.WriteIfChangedAtomic(metadata,
                BaseTranslationOutputMetadata.Create([
                    Metadata($"func_{address:X8}.cpp", address, stage)
                ], TranslationQualityMetadata.Clean, "stage-gate"));
            var options = new TranslatedBuildShardOptions(metadata, functions, output, native,
                BaseShardCount: 1, RegistrationShardCount: 1);
            for (var generation = 0; generation < 2; generation++)
            {
                TranslatedBuildShardEmitter.Emit(options);
                var shard = File.ReadAllText(Assert.Single(Directory.GetFiles(
                    Path.Combine(output, "base_common"), "shard_*.cpp")));
                Assert.Contains($"#include \"{header}\"", shard);
                Assert.Equal(1, shard.Split(hook).Length - 1);
                Assert.True(shard.IndexOf("MKW_GUEST_CODE_GUARD(", StringComparison.Ordinal) <
                            shard.IndexOf(hook, StringComparison.Ordinal));
                Assert.True(shard.IndexOf(hook, StringComparison.Ordinal) <
                            shard.IndexOf("ctx->gpr[3] = 7", StringComparison.Ordinal));
                Directory.Delete(output, true);
            }
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [Fact]
    public void NativeBodyMarkerRoutesTheWrapperAndKeepsTheBodyAndStaticCalls()
    {
        var root = Path.Combine(Path.GetTempPath(), $"translator-native-body-{Guid.NewGuid():N}");
        var functions = Path.Combine(root, "functions");
        var native = Path.Combine(root, "runtime", "src");
        var output = Path.Combine(root, "out");
        Directory.CreateDirectory(functions);
        Directory.CreateDirectory(native);
        try
        {
            const string caller = """
                #include "abi_bridge.h"
                extern "C" void func_80001000(CpuContext* MKW_RESTRICT ctx) {
                    InvokeDirectCpu<0x80005000u>(ctx);
                }
                // RECOMP_REGISTRATION base 0x80001000 func_80001000 preserves=true fpr_mask=0x00000000
                """;
            const string target = """
                #include "abi_bridge.h"
                extern "C" void func_80005000(CpuContext* MKW_RESTRICT ctx) {
                    ctx->gpr[3] = 5;
                }
                // RECOMP_REGISTRATION base 0x80005000 func_80005000 preserves=true fpr_mask=0x00000000
                """;
            Write(Path.Combine(functions, "func_80001000.cpp"), caller);
            Write(Path.Combine(functions, "func_80005000.cpp"), target);
            // A marker in a comment is not one.
            Write(Path.Combine(native, "native.cpp"),
                "MKW_NATIVE_BODY(0x80005000u, Native_80005000);\n// MKW_NATIVE_BODY(0x80001000u, Commented);\n");
            var metadataPath = Path.Combine(root, "base_output.json");
            BaseTranslationOutputMetadataFile.WriteIfChangedAtomic(metadataPath,
                BaseTranslationOutputMetadata.Create([
                    Metadata("func_80001000.cpp", 0x80001000u, caller),
                    Metadata("func_80005000.cpp", 0x80005000u, target)
                ], TranslationQualityMetadata.Clean, "native-body"));
            TranslatedBuildShardEmitter.Emit(new TranslatedBuildShardOptions(
                metadataPath, functions, output, native, BaseShardCount: 1, RegistrationShardCount: 1));
            var shard = File.ReadAllText(Assert.Single(Directory.GetFiles(
                Path.Combine(output, "base_common"), "shard_*.cpp")));
            Assert.Contains("extern \"C\" void Native_80005000(CpuContext* ctx);", shard);
            Assert.Contains("[[clang::musttail]] return Native_80005000(ctx);", shard);
            Assert.Contains("extern \"C\" void func_80005000_body(CpuContext* MKW_RESTRICT ctx)", shard);
            Assert.Contains("ctx->gpr[3] = 5;", shard);
            Assert.Contains("[[clang::musttail]] return func_80001000_body(ctx);", shard);
            Assert.DoesNotContain("Commented", shard);
            // Not a registration: callers keep the static call into the (guarded) wrapper.
            Assert.Contains("MKW_STATIC_TRANSLATED_CALL(0x80005000u, func_80005000, ctx);", shard);
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [Fact]
    public void InlineProfileInlinesPromotedIndirectArms()
    {
        var root = Path.Combine(Path.GetTempPath(), $"translator-inline-{Guid.NewGuid():N}");
        var functions = Path.Combine(root, "functions");
        var output = Path.Combine(root, "out");
        Directory.CreateDirectory(functions);
        Directory.CreateDirectory(Path.Combine(root, "native"));
        try
        {
            const string caller = """
                #include "abi_bridge.h"
                extern "C" void func_80001000(CpuContext* MKW_RESTRICT ctx) {
                    uint32_t r3 = ctx->gpr[3];
                    uint32_t ctr = ctx->ctr;
                    ctx->lr = 0x80001008u;
                    ctx->gpr[3] = r3;
                    ctx->ctr = ctr;
                    if ((ctr) == 0x80005000u) [[likely]] { InvokeDirectCpu<0x80005000u>(ctx); } else if ((ctr) == 0x80006000u) { InvokeDirectCpu<0x80006000u>(ctx); } else { MKW_INDIRECT_CALL(ctr, ctx); }
                    r3 = ctx->gpr[3];
                    ctx->gpr[3] = r3;
                }
                // RECOMP_REGISTRATION base 0x80001000 func_80001000 preserves=true fpr_mask=0x00000000
                """;
            const string target = """
                #include "abi_bridge.h"
                extern "C" void func_80005000(CpuContext* MKW_RESTRICT ctx) {
                    ctx->gpr[3] = ctx->gpr[3] + 5;
                }
                // RECOMP_REGISTRATION base 0x80005000 func_80005000 preserves=true fpr_mask=0x00000000
                """;
            const string other = """
                #include "abi_bridge.h"
                extern "C" void func_80006000(CpuContext* MKW_RESTRICT ctx) {
                    ctx->gpr[3] = ctx->gpr[3] + 6;
                }
                // RECOMP_REGISTRATION base 0x80006000 func_80006000 preserves=true fpr_mask=0x00000000
                """;
            Write(Path.Combine(functions, "func_80001000.cpp"), caller);
            Write(Path.Combine(functions, "func_80005000.cpp"), target);
            Write(Path.Combine(functions, "func_80006000.cpp"), other);
            var profile = Path.Combine(root, "inline_profile.txt");
            Write(profile, "# test\n80001000 80005000 100\n");
            var metadataPath = Path.Combine(root, "base_output.json");
            BaseTranslationOutputMetadataFile.WriteIfChangedAtomic(metadataPath,
                BaseTranslationOutputMetadata.Create([
                    Metadata("func_80001000.cpp", 0x80001000u, caller),
                    Metadata("func_80005000.cpp", 0x80005000u, target),
                    Metadata("func_80006000.cpp", 0x80006000u, other)
                ], TranslationQualityMetadata.Clean, "inline"));
            var previous = Environment.GetEnvironmentVariable("MKW_INLINE_STATIC");
            Environment.SetEnvironmentVariable("MKW_INLINE_STATIC", "0");
            try
            {
                TranslatedBuildShardEmitter.Emit(new TranslatedBuildShardOptions(
                    metadataPath, functions, output, Path.Combine(root, "native"), BaseShardCount: 1, RegistrationShardCount: 1,
                    InlineProfilePath: profile));
            }
            finally
            {
                Environment.SetEnvironmentVariable("MKW_INLINE_STATIC", previous);
            }
            var shard = File.ReadAllText(Assert.Single(Directory.GetFiles(
                Path.Combine(output, "base_common"), "shard_*.cpp")));
            // The promoted line becomes an if/else chain; the selected arm is the guarded inline form, the others keep
            // their call, and every arm repeats the flush and the reload.
            Assert.Contains("    if ((ctr) == 0x80005000u) [[likely]] {\n        if (__builtin_expect(g_mkwGuestCodePatched[1u], 0)) {", shard);
            Assert.Contains("            func_80005000_inl(ctx);", shard);
            Assert.Contains("    } else if ((ctr) == 0x80006000u) {\n        ctx->lr = 0x80001008u;", shard);
            Assert.Contains("        MKW_STATIC_TRANSLATED_CALL(0x80006000u, func_80006000, ctx);\n        r3 = ctx->gpr[3];", shard);
            Assert.Contains("    } else {\n        ctx->lr = 0x80001008u;\n        ctx->gpr[3] = r3;\n        ctx->ctr = ctr;\n        MKW_INDIRECT_CALL(ctr, ctx);\n        r3 = ctx->gpr[3];\n    }", shard);
            Assert.Equal(4, shard.Split("ctx->lr = 0x80001008u;").Length - 1);
            Assert.DoesNotContain("func_80006000_inl", shard);
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [Fact]
    public void InlineProfileGuardsInlinedCopiesAndSkipsNativeBodies()
    {
        var root = Path.Combine(Path.GetTempPath(), $"translator-inline-{Guid.NewGuid():N}");
        var functions = Path.Combine(root, "functions");
        var native = Path.Combine(root, "runtime", "src");
        var output = Path.Combine(root, "out");
        Directory.CreateDirectory(functions);
        Directory.CreateDirectory(native);
        try
        {
            const string caller = """
                #include "abi_bridge.h"
                extern "C" void func_80001000(CpuContext* MKW_RESTRICT ctx) {
                    uint32_t r3 = ctx->gpr[3];
                    ctx->lr = 0x80001008u;
                    ctx->gpr[3] = r3;
                    InvokeDirectCpu<0x80005000u>(ctx);
                    r3 = ctx->gpr[3];
                    InvokeDirectCpu<0x80006000u>(ctx);
                    ctx->gpr[3] = r3;
                }
                // RECOMP_REGISTRATION base 0x80001000 func_80001000 preserves=true fpr_mask=0x00000000
                """;
            const string target = """
                #include "abi_bridge.h"
                extern "C" void func_80005000(CpuContext* MKW_RESTRICT ctx) {
                    ctx->gpr[3] = ctx->gpr[3] + 5;
                }
                // RECOMP_REGISTRATION base 0x80005000 func_80005000 preserves=true fpr_mask=0x00000000
                """;
            const string nativeTarget = """
                #include "abi_bridge.h"
                extern "C" void func_80006000(CpuContext* MKW_RESTRICT ctx) {
                    ctx->gpr[3] = 6;
                }
                // RECOMP_REGISTRATION base 0x80006000 func_80006000 preserves=true fpr_mask=0x00000000
                """;
            Write(Path.Combine(functions, "func_80001000.cpp"), caller);
            Write(Path.Combine(functions, "func_80005000.cpp"), target);
            Write(Path.Combine(functions, "func_80006000.cpp"), nativeTarget);
            Write(Path.Combine(native, "native.cpp"), "MKW_NATIVE_BODY(0x80006000u, Native_80006000);\n");
            var profile = Path.Combine(root, "inline_profile.txt");
            Write(profile, "# test\n80001000 80005000 100\n80001000 80006000 90\n80001000 80001000 5\n");
            var metadataPath = Path.Combine(root, "base_output.json");
            BaseTranslationOutputMetadataFile.WriteIfChangedAtomic(metadataPath,
                BaseTranslationOutputMetadata.Create([
                    Metadata("func_80001000.cpp", 0x80001000u, caller),
                    Metadata("func_80005000.cpp", 0x80005000u, target),
                    Metadata("func_80006000.cpp", 0x80006000u, nativeTarget)
                ], TranslationQualityMetadata.Clean, "inline"));
            TranslatedBuildShardEmitter.Emit(new TranslatedBuildShardOptions(
                metadataPath, functions, output, native, BaseShardCount: 1, RegistrationShardCount: 1,
                InlineProfilePath: profile));
            var shard = File.ReadAllText(Assert.Single(Directory.GetFiles(
                Path.Combine(output, "base_common"), "shard_*.cpp")));
            // The copy is the callee's body; the site tests the callee's guard byte (index 1: sorted addresses).
            Assert.Contains("static inline __attribute__((always_inline)) void func_80005000_inl(CpuContext* MKW_RESTRICT ctx)\n{", shard);
            Assert.Contains("if (__builtin_expect(g_mkwGuestCodePatched[1u], 0)) {", shard);
            Assert.Contains("        func_80005000(ctx);", shard);
            Assert.Contains("        func_80005000_inl(ctx);", shard);
            // Flush and reload are repeated in both arms.
            Assert.Equal(2, shard.Split("        ctx->lr = 0x80001008u;").Length - 1);
            Assert.Equal(2, shard.Split("        r3 = ctx->gpr[3];").Length - 1);
            // The callee keeps its own guarded wrapper and body.
            Assert.Contains("[[clang::musttail]] return func_80005000_body(ctx);", shard);
            // A native body is never inlined, nor is a function into itself.
            Assert.Contains("MKW_STATIC_TRANSLATED_CALL(0x80006000u, func_80006000, ctx);", shard);
            Assert.DoesNotContain("func_80006000_inl", shard);
            Assert.DoesNotContain("func_80001000_inl", shard);
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [Fact]
    public void RemainingWeightPartitionDoesNotCreateOversizedTailBin()
    {
        var weights = Enumerable.Repeat(300L, 10).ToArray();
        var first = TranslatedBuildShardEmitter.PartitionWeightedIndices(weights, 4);
        var second = TranslatedBuildShardEmitter.PartitionWeightedIndices(weights, 4);

        Assert.Equal(new[] { 2, 3, 2, 3 }, first.Select(static group => group.Count));
        Assert.Equal(
            first.Select(static group => string.Join(",", group)),
            second.Select(static group => string.Join(",", group)));
        Assert.All(first, group => Assert.InRange(group.Sum(index => weights[index]), 600, 900));
    }

    [Fact]
    public void EmitsDeterministicSmallTraitShardsWithoutCodeMap()
    {
        var root = Path.Combine(Path.GetTempPath(), $"translator-build-shards-{Guid.NewGuid():N}");
        var functions = Path.Combine(root, "functions");
        var native = Path.Combine(root, "runtime", "src");
        var output = Path.Combine(root, "out");
        Directory.CreateDirectory(functions);
        Directory.CreateDirectory(native);
        try
        {
            const string caller = """
                #include "abi_bridge.h"
                extern "C" void func_80001000(CpuContext* ctx) {
                    InvokeDirectCpu<0x80002000u>(ctx);
                }
                // RECOMP_REGISTRATION base 0x80001000 func_80001000 preserves=true fpr_mask=0x00000000
                """;
            const string target = """
                #include "abi_bridge.h"
                extern "C" void func_80002000(CpuContext*) {}
                // RECOMP_REGISTRATION base 0x80002000 func_80002000 preserves=true fpr_mask=0x00000000
                """;
            const string runtimeOverridden = """
                #include "abi_bridge.h"
                extern "C" void func_80003000(CpuContext*) {}
                // RECOMP_REGISTRATION base 0x80003000 func_80003000 preserves=true fpr_mask=0x00000000
                """;
            const string nativeOverridden = """
                #include "abi_bridge.h"
                extern "C" void func_80004000(CpuContext*) {}
                // RECOMP_REGISTRATION base 0x80004000 func_80004000 preserves=true fpr_mask=0x00000000
                """;
            Write(Path.Combine(functions, "func_80001000.cpp"), caller);
            Write(Path.Combine(functions, "func_80002000.cpp"), target);
            Write(Path.Combine(functions, "func_80003000.cpp"), runtimeOverridden);
            Write(Path.Combine(functions, "func_80004000.cpp"), nativeOverridden);
            Write(Path.Combine(native, "override.cpp"),
                "REGISTER_TRANSLATED_FUNCTION(0x80003000, RuntimeOverride_80003000);\n" +
                "REGISTER_NATIVE_FUNCTION(0x80004000, NativeOverride_80004000);\n");
            var metadataPath = Path.Combine(root, "base_output.json");
            BaseTranslationOutputMetadataFile.WriteIfChangedAtomic(metadataPath,
                BaseTranslationOutputMetadata.Create([
                    Metadata("func_80001000.cpp", 0x80001000u, caller),
                    Metadata("func_80002000.cpp", 0x80002000u, target),
                    Metadata("func_80003000.cpp", 0x80003000u, runtimeOverridden),
                    Metadata("func_80004000.cpp", 0x80004000u, nativeOverridden)
                ], TranslationQualityMetadata.Clean, "identity"));
            var options = new TranslatedBuildShardOptions(
                metadataPath, functions, output, native,
                BaseShardCount: 2, RegistrationShardCount: 1);
            var first = TranslatedBuildShardEmitter.Emit(options);
            Assert.Equal(2, first.SharedBaseFunctionCount);
            Assert.DoesNotContain("RETRO", File.ReadAllText(first.CMakeManifestPath));
            var shardText = string.Concat(Directory.GetFiles(Path.Combine(output, "base_common"), "*.cpp").Select(File.ReadAllText));
            Assert.Contains("#include \"abi_bridge.h\"", shardText);
            Assert.DoesNotContain("#include \"" + Path.GetFullPath(Path.Combine(functions, "func_80001000.cpp")).Replace('\\', '/'), shardText);
            Assert.Contains("MKW_STATIC_TRANSLATED_CALL(0x80002000u, func_80002000, ctx);", shardText);

            var traits = string.Concat(Directory.GetFiles(Path.Combine(output, "base_common"), "*_traits.h").Select(File.ReadAllText));
            Assert.Contains("MKW_TRANSLATED_TRAIT(80002000, func_80002000,", traits);
            var baseDispatchText = string.Concat(Directory.GetFiles(Path.Combine(output, "base_dispatch"), "*.cpp").Select(File.ReadAllText));
            Assert.Contains("const StaticIndirectDispatchSegment kSegments[256]", baseDispatchText);
            Assert.Contains("{0x80001000u, &func_80001000, 0x00000000u, false}", baseDispatchText);
            Assert.Contains("{0x80002000u, &func_80002000, 0x00000000u, false}", baseDispatchText);
            Assert.Contains("{0x80003000u, &RuntimeOverride_80003000, 0xFFFFC000u, false}", baseDispatchText);
            Assert.DoesNotContain("0x80004000u", baseDispatchText);
            // A function claimed by a native registration is excluded from the
            // translated graph entirely rather than emitted and then overridden.
            Assert.DoesNotContain(
                "func_80003000",
                Directory.GetFiles(output, "*.cpp", SearchOption.AllDirectories)
                    .Select(File.ReadAllText)
                    .Aggregate(string.Concat));

            var timestamps = Directory.GetFiles(output, "*", SearchOption.AllDirectories)
                .ToDictionary(Path.GetFullPath, File.GetLastWriteTimeUtc, StringComparer.OrdinalIgnoreCase);
            TranslatedBuildShardEmitter.Emit(options);
            foreach (var (path, timestamp) in timestamps)
                Assert.Equal(timestamp, File.GetLastWriteTimeUtc(path));
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void KeepsFprWritingCallsDynamicallyGuarded()
    {
        var root = Path.Combine(Path.GetTempPath(), $"translator-same-tu-calls-{Guid.NewGuid():N}");
        var functions = Path.Combine(root, "functions");
        var native = Path.Combine(root, "runtime", "src");
        var output = Path.Combine(root, "out");
        Directory.CreateDirectory(functions);
        Directory.CreateDirectory(native);
        try
        {
            // One target preserves FPRs, the other writes nonvolatile FPRs.
            const string commonCaller = """
                #include "abi_bridge.h"
                extern "C" void func_80001000(CpuContext* ctx) {
                    InvokeDirectCpu<0x80002000u>(ctx);
                    InvokeDirectCpu<0x80003000u>(ctx);
                }
                // RECOMP_REGISTRATION base 0x80001000 func_80001000 preserves=true fpr_mask=0x00000000
                """;
            const string invariantTarget = """
                #include "abi_bridge.h"
                extern "C" void func_80002000(CpuContext*) {}
                // RECOMP_REGISTRATION base 0x80002000 func_80002000 preserves=true fpr_mask=0x00000000
                """;
            const string fprWritingTarget = """
                #include "abi_bridge.h"
                extern "C" void func_80003000(CpuContext*) {}
                // RECOMP_REGISTRATION base 0x80003000 func_80003000 preserves=false fpr_mask=0x00004000
                """;
            Write(Path.Combine(functions, "func_80001000.cpp"), commonCaller);
            Write(Path.Combine(functions, "func_80002000.cpp"), invariantTarget);
            Write(Path.Combine(functions, "func_80003000.cpp"), fprWritingTarget);

            var metadataPath = Path.Combine(root, "base_output.json");
            BaseTranslationOutputMetadataFile.WriteIfChangedAtomic(metadataPath,
                BaseTranslationOutputMetadata.Create([
                    Metadata("func_80001000.cpp", 0x80001000u, commonCaller),
                    Metadata("func_80002000.cpp", 0x80002000u, invariantTarget),
                    Metadata("func_80003000.cpp", 0x80003000u, fprWritingTarget)
                ], TranslationQualityMetadata.Clean, "identity"));
            var options = new TranslatedBuildShardOptions(
                metadataPath, functions, output, native,
                BaseShardCount: 1, RegistrationShardCount: 1);
            TranslatedBuildShardEmitter.Emit(options);

            var commonShard = File.ReadAllText(
                Directory.GetFiles(Path.Combine(output, "base_common"), "*.cpp").Single());
            Assert.Contains("MKW_STATIC_TRANSLATED_CALL(0x80002000u, func_80002000, ctx);", commonShard);
            // A nonvolatile-FPR writer keeps the generic dynamically guarded call.
            Assert.Contains("InvokeDirectCpu<0x80003000u>(ctx);", commonShard);
            // The lowering choice participates in shard identity, so a rebuild is
            // byte-identical and file identities are stable.
            var timestamps = Directory.GetFiles(output, "*", SearchOption.AllDirectories)
                .ToDictionary(Path.GetFullPath, File.GetLastWriteTimeUtc, StringComparer.OrdinalIgnoreCase);
            TranslatedBuildShardEmitter.Emit(options);
            foreach (var (path, timestamp) in timestamps)
                Assert.Equal(timestamp, File.GetLastWriteTimeUtc(path));
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void FrozenBaseCommonBoundariesKeepUnchangedShardIdentityAcrossAFunctionBodyChange()
    {
        var root = Path.Combine(Path.GetTempPath(), $"translator-frozen-shards-{Guid.NewGuid():N}");
        var functions = Path.Combine(root, "functions");
        var native = Path.Combine(root, "runtime", "src");
        var output = Path.Combine(root, "out");
        Directory.CreateDirectory(functions);
        Directory.CreateDirectory(native);
        try
        {
            // Six equally weighted profile-neutral functions pack 3/3 across two
            // base_common shards.
            var addresses = new uint[] { 0x80001000, 0x80002000, 0x80003000, 0x80004000, 0x80005000, 0x80006000 };
            static string Body(uint address, string filler) =>
                "#include \"abi_bridge.h\"\n" +
                $"extern \"C\" void func_{address:X8}(CpuContext*) {{}}\n" +
                $"// {filler}\n" +
                $"// RECOMP_REGISTRATION base 0x{address:X8} func_{address:X8} preserves=true fpr_mask=0x00000000\n";

            var metadataPath = Path.Combine(root, "base_output.json");
            void WriteInputs(string lastFunctionFiller)
            {
                var entries = new List<BaseTranslationFunctionMetadata>();
                foreach (var address in addresses)
                {
                    var filler = address == addresses[^1] ? lastFunctionFiller : "";
                    var source = Body(address, filler);
                    var relative = $"func_{address:X8}.cpp";
                    Write(Path.Combine(functions, relative), source);
                    entries.Add(Metadata(relative, address, source));
                }
                BaseTranslationOutputMetadataFile.WriteIfChangedAtomic(
                    metadataPath, BaseTranslationOutputMetadata.Create(
                        entries, TranslationQualityMetadata.Clean, "identity"));

            }

            string[] BaseCommonShards() =>
                Directory.GetFiles(Path.Combine(output, "base_common"), "*.cpp")
                    .Select(static path => Path.GetFileName(path)!)
                    .Order(StringComparer.Ordinal)
                    .ToArray();

            var options = new TranslatedBuildShardOptions(
                metadataPath, functions, output, native,
                BaseShardCount: 2, RegistrationShardCount: 1);

            WriteInputs("");
            var first = TranslatedBuildShardEmitter.Emit(options);
            Assert.False(first.BaseCommonBoundariesReused);
            Assert.NotNull(first.BaseCommonShardMapPath);
            var mapPath = first.BaseCommonShardMapPath!;
            Assert.Equal("base_common_shard_map.json", Path.GetFileName(mapPath));
            Assert.True(File.Exists(mapPath));
            Assert.NotNull(first.BaseCommonBalance);
            Assert.Equal(2, first.BaseCommonBalance!.ShardCount);
            var frozenMap = File.ReadAllText(mapPath);
            Assert.Contains("\"0x80001000\"", frozenMap);
            Assert.Contains("\"0x80004000\"", frozenMap);
            var before = BaseCommonShards();
            Assert.Equal(2, before.Length);

            // Only the last function's body changes, and it changes enough to
            // move every greedy cut point behind it. Frozen membership keeps the
            // shard that owns the first three functions byte-identical, so its
            // content-addressed name - and its object file - survive.
            WriteInputs(new string('x', 4096));
            var second = TranslatedBuildShardEmitter.Emit(options);
            Assert.True(second.BaseCommonBoundariesReused);
            Assert.Equal(frozenMap, File.ReadAllText(mapPath));
            var after = BaseCommonShards();
            Assert.Equal(2, after.Length);
            Assert.Single(before.Intersect(after, StringComparer.Ordinal));

            // Discarding the recorded table is the only thing that moves a boundary.
            File.Delete(mapPath);
            var repacked = TranslatedBuildShardEmitter.Emit(options);
            Assert.False(repacked.BaseCommonBoundariesReused);
            var repackedMap = File.ReadAllText(mapPath);
            Assert.NotEqual(frozenMap, repackedMap);
            Assert.Contains("\"0x80006000\"", repackedMap);
            Assert.Empty(after.Intersect(BaseCommonShards(), StringComparer.Ordinal));
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void PromotedIndirectArmsKeepOnlyTranslatedTargets()
    {
        var root = Path.Combine(Path.GetTempPath(), $"translator-promoted-indirect-{Guid.NewGuid():N}");
        var functions = Path.Combine(root, "functions");
        var native = Path.Combine(root, "runtime", "src");
        var output = Path.Combine(root, "out");
        Directory.CreateDirectory(functions);
        Directory.CreateDirectory(native);
        try
        {
            // 0x80005000 is translated; 0x80007000 has no translation, so its arm goes.
            const string caller = """
                #include "abi_bridge.h"
                extern "C" void func_80001000(CpuContext* MKW_RESTRICT ctx) {
                    uint32_t ctr = ctx->ctr;
                    if ((ctr) == 0x80005000u) [[likely]] { InvokeDirectCpu<0x80005000u>(ctx); } else if ((ctr) == 0x80007000u) { InvokeDirectCpu<0x80007000u>(ctx); } else { MKW_INDIRECT_CALL(ctr, ctx); }
                    if ((ctr) == 0x80007000u) [[likely]] { InvokeDirectCpu<0x80007000u>(ctx); } else { MKW_INDIRECT_CALL(ctr, ctx); }
                }
                // RECOMP_REGISTRATION base 0x80001000 func_80001000 preserves=true fpr_mask=0x00000000
                """;
            const string target = """
                #include "abi_bridge.h"
                extern "C" void func_80005000(CpuContext* MKW_RESTRICT ctx) {
                    ctx->gpr[3] = 5;
                }
                // RECOMP_REGISTRATION base 0x80005000 func_80005000 preserves=true fpr_mask=0x00000000
                """;
            Write(Path.Combine(functions, "func_80001000.cpp"), caller);
            Write(Path.Combine(functions, "func_80005000.cpp"), target);
            var metadataPath = Path.Combine(root, "base_output.json");
            BaseTranslationOutputMetadataFile.WriteIfChangedAtomic(metadataPath,
                BaseTranslationOutputMetadata.Create([
                    Metadata("func_80001000.cpp", 0x80001000u, caller),
                    Metadata("func_80005000.cpp", 0x80005000u, target)
                ], TranslationQualityMetadata.Clean, "promoted-indirect"));
            TranslatedBuildShardEmitter.Emit(new TranslatedBuildShardOptions(
                metadataPath, functions, output, native, BaseShardCount: 1, RegistrationShardCount: 1));
            var shard = File.ReadAllText(Assert.Single(Directory.GetFiles(
                Path.Combine(output, "base_common"), "shard_*.cpp")));
            Assert.Contains(
                "if ((ctr) == 0x80005000u) [[likely]] { MKW_STATIC_TRANSLATED_CALL(0x80005000u, func_80005000, ctx); } else { MKW_INDIRECT_CALL(ctr, ctx); }",
                shard);
            Assert.DoesNotContain("0x80007000u", shard);
            Assert.Equal(2, shard.Split("MKW_INDIRECT_CALL(ctr, ctx);").Length - 1);
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    private static BaseTranslationFunctionMetadata Metadata(string path, uint address, string content)
    {
        var bytes = Encoding.UTF8.GetBytes(content);
        return new BaseTranslationFunctionMetadata(
            path, bytes.Length, Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(), address, []);
    }

    private static void Write(string path, string content)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content, new UTF8Encoding(false));
    }
}
