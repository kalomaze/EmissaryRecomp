using System;
using Translator.Core.Ir;
using Translator.Core.Analysis.Representation;
using Translator.Core.Representation;

namespace Translator.Core.CodeGen;

public sealed partial class CxxLinearCodeGenerator
{
    private static string BinaryExpressionWithOperands(IrBinary bin, string left, string right)
    {
        if (bin.Op == "sext")
        {
            var widthBits = bin.Right.Constant.GetValueOrDefault(8);
            return widthBits switch
            {
                16 => $"(static_cast<int32_t>(static_cast<int16_t>({left})))",
                8 => $"(static_cast<int32_t>(static_cast<int8_t>({left})))",
                _ => $"(static_cast<int32_t>({left}))",
            };
        }

        if (IsFloatRegister(bin.Destination))
        {
            var helper = bin.Op switch
            {
                "add" => "PpcFaddInline",
                "sub" => "PpcFsubInline",
                "mul" => "PpcFmulInline",
                "fdiv" => "PpcFdivInline",
                _ => null
            };
            if (helper is not null)
                return $"{helper}({left}, {right})";
        }

        return bin.Op switch
        {
            "add" => $"({left} + {right})",
            "sub" => $"({left} - {right})",
            "and" => $"({left} & {right})",
            "andc" => $"({left} & ~{right})",
            "or" => $"({left} | {right})",
            "orc" => $"({left} | ~{right})",
            "fdiv" => $"({left} / {right})",
            "nor" => $"~({left} | {right})",
            "nand" => $"~({left} & {right})",
            "eqv" => $"~({left} ^ {right})",
            "xor" => $"({left} ^ {right})",
            "shl" => $"({left} << {right})",
            "rotl" => $"PpcRotl32Inline(static_cast<uint32_t>({left}), static_cast<uint32_t>({right}))",
            "shr" => $"({left} >> {right})",
            "ppc_slw" => $"PPC_Slw(static_cast<uint32_t>({left}), static_cast<uint32_t>({right}))",
            "ppc_srw" => $"PPC_Srw(static_cast<uint32_t>({left}), static_cast<uint32_t>({right}))",
            "ppc_sraw" => $"PPC_Sraw(static_cast<uint32_t>({left}), static_cast<uint32_t>({right}))",
            "fabs" => $"std::fabs({left})",
            "fneg" => $"(-({left}))",
            "frsp" => $"static_cast<double>(PpcForceSingleValueInline({left}))",
            "frsp_noni" => $"static_cast<double>(static_cast<float>({left}))",
            "fctiw" => $"PPC_Fctiw({left})",
            "fctiwz" => $"PPC_Fctiwz({left})",
            "fpr_low_word" => $"PPC_FprLowWordInline({left})",
            "sar" => $"(static_cast<int32_t>({left}) >> {right})",
            "mul" => $"({left} * {right})",
            "mulhwu" => $"(static_cast<uint32_t>((static_cast<uint64_t>({left}) * static_cast<uint64_t>({right})) >> 32))",
            "mulhw" => $"(static_cast<int32_t>((static_cast<int64_t>(static_cast<int32_t>({left})) * static_cast<int64_t>(static_cast<int32_t>({right}))) >> 32))",
            "divu" => $"PPC_Divwu(static_cast<uint32_t>({left}), static_cast<uint32_t>({right}))",
            "div" => $"PPC_Divw(static_cast<int32_t>({left}), static_cast<int32_t>({right}))",
            "sext16" => $"(static_cast<int32_t>(static_cast<int16_t>({left})))",
            "not" => $"~({left})",
            "sub_u" => $"(CompareUnsigned(static_cast<uint32_t>({left}), static_cast<uint32_t>({right})))",
            _ => throw new ArgumentOutOfRangeException(nameof(bin.Op), bin.Op, null)
        };
    }

    // Guest physical address window: MEM1 at 0x80000000 and MEM2 up to
    // 0x94000000. A negative IR immediate whose unsigned form lands here is a
    // code or data address that was only negative because IrValue carries a
    // signed int.
    private const uint GuestAddressWindowStart = 0x80000000u;
    private const uint GuestAddressWindowEnd = 0x94000000u;

    /// <summary>
    /// Formats an assignment RHS. Negative <c>int</c> immediates that are really guest addresses (return
    /// addresses, <c>lis/ori</c> constants) print as hex when the destination is an unsigned 32-bit guest
    /// register, for readability only; small signed immediates like <c>-16</c> stay decimal.
    /// </summary>
    private static string ToRegisterAssignmentExpression(
        string destination,
        IrValue value,
        RepresentationEnvironment types)
    {
        if (value.Kind == "const" &&
            value.Constant is { } constant &&
            constant < 0 &&
            IsUnsigned32BitGuestRegister(destination))
        {
            var unsigned = unchecked((uint)constant);
            if (unsigned is >= GuestAddressWindowStart and < GuestAddressWindowEnd)
            {
                return $"0x{unsigned:X8}u";
            }
        }

        return ToExpression(value, types);
    }

    private static bool IsUnsigned32BitGuestRegister(string name)
    {
        if (!IsCpuRegister(name))
        {
            return false;
        }

        var baseName = BaseRegister(name);
        if (baseName.Length > 1 && baseName[0] == 'r' &&
            int.TryParse(baseName.AsSpan(1), out var gpr) && gpr is >= 0 and < 32)
        {
            return true;
        }

        // CR field destinations lower to ordinary locals whose host type comes
        // from representation classification, so they are deliberately excluded.
        return baseName.Equals("lr", StringComparison.OrdinalIgnoreCase) ||
               baseName.Equals("ctr", StringComparison.OrdinalIgnoreCase);
    }

    private static string ToExpression(IrValue val, RepresentationEnvironment types)
    {
        if (val.Kind == "const" && val.Constant.HasValue)
        {
            return val.Constant.Value.ToString();
        }

        if (val.Kind != "register" || val.RegisterName == null)
        {
            return "0";
        }

        var name = val.RegisterName;
        if (IsCpuRegister(name))
        {
            return RegisterToCtxExpression(name, written: false);
        }

        return name;
    }

    private static bool TryGetFixedSdaBase(string? name, out uint value)
    {
        if (!string.IsNullOrEmpty(name) && IsCpuRegister(name))
        {
            var baseName = BaseRegisterSpan(name);
            if (_fixedSda1Base.HasValue && baseName.Equals("r13", StringComparison.OrdinalIgnoreCase))
            {
                value = _fixedSda1Base.Value;
                return true;
            }
            if (_fixedSda2Base.HasValue && baseName.Equals("r2", StringComparison.OrdinalIgnoreCase))
            {
                value = _fixedSda2Base.Value;
                return true;
            }
        }
        value = 0;
        return false;
    }

    private static bool TryFoldSdaBinary(IrBinary bin, out string expression)
    {
        if (string.Equals(bin.Op, "add", StringComparison.Ordinal))
        {
            if (bin.Left.Kind == "register" && bin.Right.Kind == "const" && bin.Right.Constant.HasValue &&
                TryGetFixedSdaBase(bin.Left.RegisterName, out var leftBase))
            {
                expression = $"0x{unchecked(leftBase + (uint)bin.Right.Constant.Value):X8}u";
                return true;
            }
            if (bin.Right.Kind == "register" && bin.Left.Kind == "const" && bin.Left.Constant.HasValue &&
                TryGetFixedSdaBase(bin.Right.RegisterName, out var rightBase))
            {
                expression = $"0x{unchecked(rightBase + (uint)bin.Left.Constant.Value):X8}u";
                return true;
            }
        }
        expression = string.Empty;
        return false;
    }

    private static string RegisterToCtxExpression(string name, bool written)
    {
        var baseName = BaseRegisterSpan(name);
        var residency = _activeResidency;

        if (baseName.Length > 1 && baseName[0] == 'r' && int.TryParse(baseName[1..], out var gpr) && gpr is >= 0 and < 32)
        {
            if (!written)
            {
                if (gpr == 13 && _fixedSda1Base.HasValue) return $"0x{_fixedSda1Base.Value:X8}u";
                if (gpr == 2 && _fixedSda2Base.HasValue) return $"0x{_fixedSda2Base.Value:X8}u";
            }
            return residency is null ? CtxGprName(gpr) : residency.Gpr(gpr, written: written);
        }

        if (baseName.Length > 1 && baseName[0] == 'f' && int.TryParse(baseName[1..], out var fpr) && fpr is >= 0 and < 32)
        {
            return residency is null ? CtxFprScalarName(fpr) : residency.FprScalar(fpr, written: written);
        }

        if (baseName.Length == 3 && baseName.StartsWith("cr".AsSpan(), StringComparison.OrdinalIgnoreCase) && char.IsDigit(baseName[2]))
        {
            // Field writes are handled separately; a nibble extraction is not an lvalue.
            if (written) return name;
            var crField = baseName[2] - '0';
            if (crField is >= 0 and <= 7)
            {
                return $"PpcCrvNibble({CrFieldExpression(crField, written: false)})";
            }
        }

        if (SpanEqualsIgnoreCase(baseName, "lr")) return "ctx->lr";
        if (SpanEqualsIgnoreCase(baseName, "ctr")) return residency is null ? "ctx->ctr" : residency.Ctr(written: written);
        if (SpanEqualsIgnoreCase(baseName, "cr")) return CrWholeExpression(written);
        if (SpanEqualsIgnoreCase(baseName, "xer")) return residency is null ? "ctx->xer" : residency.Xer(written: written);
        if (SpanEqualsIgnoreCase(baseName, "srr0")) return "ctx->srr0";
        if (SpanEqualsIgnoreCase(baseName, "srr1")) return "ctx->srr1";
        if (SpanEqualsIgnoreCase(baseName, "msr")) return "ctx->msr";
        if (SpanEqualsIgnoreCase(baseName, "hid0")) return "ctx->hid0";
        if (SpanEqualsIgnoreCase(baseName, "hid1")) return "ctx->hid1";
        if (SpanEqualsIgnoreCase(baseName, "hid2")) return "ctx->hid2";
        if (baseName.StartsWith("gqr".AsSpan(), StringComparison.OrdinalIgnoreCase))
        {
            return int.TryParse(baseName[3..], out var gqrIdx) && gqrIdx is >= 0 and < 8
                ? $"ctx->gqr[{gqrIdx}]"
                : written ? name : "0";
        }

        return written ? name : "0";
    }

    /// <summary>
    /// One CR field (an <c>int64_t</c> field value, isa/ppc_isa_crv.h): its resident local, or CpuContext's. Every
    /// emitted CR access goes through this or <see cref="CrWholeExpression"/>, so the text forms are exactly
    /// <c>crvN</c> and <c>ctx-&gt;crv[N]</c> (the leaf register cache rewrites the latter).
    /// </summary>
    private static string CrFieldExpression(int field, bool written) =>
        _activeResidency is null ? $"ctx->crv[{field & 7}]" : _activeResidency.CrField(field, written);

    /// <summary>The whole CR: read, its packed word; written, a <c>PpcCrView</c> that takes one.</summary>
    private static string CrWholeExpression(bool written)
    {
        if (_activeResidency is not null) return _activeResidency.CrWhole(written);
        var fields = "ctx->crv[0], ctx->crv[1], ctx->crv[2], ctx->crv[3], ctx->crv[4], ctx->crv[5], ctx->crv[6], ctx->crv[7]";
        return $"{(written ? "PpcCrView" : "PpcCrvPack")}({fields})";
    }

    /// <summary>The IR's whole condition register (mtcrf, mcrf, crxor, ... and helpers that return it) as a destination.</summary>
    private static bool IsWholeCrDestination(string destination) =>
        GetRegisterBaseName(destination).Equals("cr", StringComparison.OrdinalIgnoreCase);

    /// <summary>A branch test of one CR bit (mask 8 LT, 4 GT, 2 EQ, 1 SO) of a field value.</summary>
    private static string CrFieldBitTest(string field, int mask, bool set) => mask switch
    {
        8 => set ? $"PpcCrvLt({field})" : $"!PpcCrvLt({field})",
        4 => set ? $"({field} > 0)" : $"({field} <= 0)",
        2 => set ? $"(static_cast<uint32_t>({field}) == 0u)" : $"(static_cast<uint32_t>({field}) != 0u)",
        _ => set ? $"PpcCrvSo({field})" : $"!PpcCrvSo({field})",
    };

    private static readonly System.Text.RegularExpressions.Regex GetCrBitTextRegex = new(
        @"GetCRBit\(\s*ctx\s*,\s*(?<field>\d+)\s*,\s*(?<bit>\d+)\s*\)",
        System.Text.RegularExpressions.RegexOptions.Compiled | System.Text.RegularExpressions.RegexOptions.CultureInvariant);

    private static string ToDestination(string name, RepresentationEnvironment types)
    {
        if (IsCpuRegister(name))
        {
            return RegisterToCtxExpression(name, written: true);
        }

        return name;
    }

    private static (int Mask, bool Set) CrFieldConditionTest(string condition)
    {
        if (EqualsIgnoreCase(condition, "bne")) return (2, false);
        if (EqualsIgnoreCase(condition, "beq")) return (2, true);
        if (EqualsIgnoreCase(condition, "bgt")) return (4, true);
        if (EqualsIgnoreCase(condition, "ble")) return (4, false);
        if (EqualsIgnoreCase(condition, "blt")) return (8, true);
        if (EqualsIgnoreCase(condition, "bge")) return (8, false);
        if (EqualsIgnoreCase(condition, "bso")) return (1, true);
        if (EqualsIgnoreCase(condition, "bns")) return (1, false);
        return (0, false);
    }

    private static bool EqualsIgnoreCase(string value, string candidate) =>
        string.Equals(value, candidate, StringComparison.OrdinalIgnoreCase);

    private static string ToCondition(string condition, string conditionRegister, RepresentationEnvironment types)
    {
        var conditionBase = BaseRegisterSpan(conditionRegister);
        if (conditionBase.Length == 3 &&
            conditionBase.StartsWith("cr".AsSpan(), StringComparison.OrdinalIgnoreCase) &&
            conditionBase[2] is >= '0' and <= '7')
        {
            var field = conditionBase[2] - '0';
            var (mask, set) = CrFieldConditionTest(condition);
            if (mask != 0)
            {
                return $"({CrFieldBitTest(CrFieldExpression(field, written: false), mask, set)})";
            }
        }

        var reg = IsCpuRegister(conditionRegister)
            ? RegisterToCtxExpression(conditionRegister, written: false)
            : ResidentRawCondition(conditionRegister);

        if (EqualsIgnoreCase(condition, "bdnz")) return $"({reg} != 0)";
        if (EqualsIgnoreCase(condition, "bdz")) return $"({reg} == 0)";
        if (EqualsIgnoreCase(condition, "bne")) return $"(({reg} & 0x2) == 0)";
        if (EqualsIgnoreCase(condition, "beq")) return $"(({reg} & 0x2) != 0)";
        if (EqualsIgnoreCase(condition, "bgt")) return $"(({reg} & 0x4) != 0)";
        if (EqualsIgnoreCase(condition, "blt")) return $"(({reg} & 0x8) != 0)";
        if (EqualsIgnoreCase(condition, "bge")) return $"(({reg} & 0x8) == 0)";
        if (EqualsIgnoreCase(condition, "ble")) return $"(({reg} & 0x4) == 0)";
        if (EqualsIgnoreCase(condition, "bso")) return $"(({reg} & 0x1) != 0)";
        if (EqualsIgnoreCase(condition, "bns")) return $"(({reg} & 0x1) == 0)";
        return reg;
    }

    /// <summary>
    /// Maps the raw branch-condition text <c>PpcLifter</c> builds for <c>bc</c>/<c>bcctr</c> onto resident
    /// locals. This is the only emitted register access that bypasses the expression layer, so missing a
    /// rewrite here means reading a stale architectural CR/CTR instead of the live local.
    /// </summary>
    private static string ResidentRawCondition(string conditionText)
    {
        if (string.IsNullOrEmpty(conditionText))
        {
            return conditionText;
        }

        // GetCRBit(ctx, F, B) becomes a read of field F (its local, or CpuContext's) in every body, so the leaf
        // register cache sees CR only as ctx->crv[N].
        var rewritten = conditionText;
        if (rewritten.Contains("GetCRBit(", StringComparison.Ordinal))
        {
            rewritten = GetCrBitTextRegex.Replace(rewritten, match =>
                $"PpcCrvBit({CrFieldExpression(int.Parse(match.Groups["field"].Value) & 7, written: false)}, " +
                $"{int.Parse(match.Groups["bit"].Value) & 3}u)");
            if (rewritten.Contains("GetCRBit(", StringComparison.Ordinal))
                throw new InvalidOperationException($"Unparsed CR bit test in branch condition: {conditionText}");
        }

        var residency = _activeResidency;
        if (residency is null)
        {
            return rewritten;
        }

        if (rewritten.Contains("ctx->ctr", StringComparison.Ordinal))
        {
            rewritten = rewritten.Replace("ctx->ctr", residency.Ctr(written: false), StringComparison.Ordinal);
        }

        return rewritten;
    }

    private static string Address(IrAddress addr, RepresentationEnvironment types)
    {
        if (TryGetFixedSdaBase(addr.Base, out var sdaBase))
        {
            return $"0x{unchecked(sdaBase + (uint)addr.Offset):X8}u";
        }

        var baseExpr = IsCpuRegister(addr.Base)
            ? RegisterToCtxExpression(addr.Base, written: false)
            : addr.Base;

        return addr.Offset == 0
            ? baseExpr
            : $"({baseExpr} + {addr.Offset})";
    }

    private static string RemapAddress(
        IrAddress addr,
        RepresentationEnvironment types,
        IReadOnlyDictionary<string, uint> localConstants,
        LinkedAddressRemap? remap)
    {
        if (remap is not null &&
            (localConstants.TryGetValue(addr.Base, out var baseValue) ||
             localConstants.TryGetValue(GetRegisterBaseName(addr.Base), out baseValue)))
        {
            var effective = unchecked(baseValue + (uint)addr.Offset);
            var linkEnd = checked(remap.LinkBase + remap.CodeSize);
            if (effective >= remap.LinkBase && effective < linkEnd)
            {
                var relocated = checked(remap.GuestBase + (effective - remap.LinkBase));
                return $"0x{relocated:X8}u";
            }
        }

        return Address(addr, types);
    }

    private static string BaseRegister(string name)
    {
        var idx = name.IndexOf('_');
        return idx < 0 ? name : name[..idx];
    }

    /// <summary>
    /// Allocation-free <see cref="BaseRegister"/>. The expression layer only
    /// inspects the prefix, so the substring it used to build per access was
    /// pure garbage.
    /// </summary>
    private static ReadOnlySpan<char> BaseRegisterSpan(string name)
    {
        var idx = name.IndexOf('_');
        return idx < 0 ? name.AsSpan() : name.AsSpan(0, idx);
    }

    private static int Bits(int sizeBytes) => sizeBytes switch
    {
        1 => 8,
        2 => 16,
        8 => 64,
        _ => 32
    };
}
