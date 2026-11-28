using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace Translator.Core.CodeGen;

/// <summary>
/// Profile-guided promotion of indirect calls (bctrl/blrl) to guarded direct calls.
///
/// The profile (translate-recursive --indirect-profile, projects/brawl/indirect_profile.txt) is the target histogram of
/// every indirect call site a run reached: lines "FUNCTION SITE TARGET COUNT" (hex guest addresses, decimal count),
/// FUNCTION the entry of the translated function holding the bctrl/blrl at SITE; "J FUNCTION SITE TARGET COUNT" for an
/// indirect tail jump (bctr), promoted the same way (the direct call is followed by the jump's return). A profiling build
/// (MKW_INDIRECT_PROFILE_SITES=1 in the translator's environment) emits MKW_INDIRECT_PROFILE_COUNT before every
/// indirect call, and the runtime writes those lines to $BRAWL_INDIRECT_PROFILE (indirect_call_site.h).
///
/// A site whose dominant targets cover enough of its calls gets, for each such target T,
/// <c>if (target == T) { InvokeDirectCpu&lt;T&gt;(ctx); } else ... MKW_INDIRECT_CALL(target, ctx);</c>: an exact
/// compare, then the ordinary direct call (lowered by the shard emitter to MKW_STATIC_TRANSLATED_CALL, which enters
/// func_T and its mod-patch guard). The shard emitter drops an arm whose target it would not lower (no translated
/// base winner, a native winner, a nonvolatile-FPR-writing callee), so only a plain translated call is ever promoted.
///
/// Thresholds (environment of the translate step): MKW_INDIRECT_PROMOTE_MIN_COUNT (calls at the site, default 64),
/// MKW_INDIRECT_PROMOTE_MIN_SHARE (share of the site's calls a promoted target must have, default 0.15),
/// MKW_INDIRECT_PROMOTE_MAX_TARGETS (default 2), MKW_INDIRECT_PROMOTE_MIN_COVER (share the promoted targets must
/// cover together, default 0.60), MKW_INDIRECT_PROMOTE_JUMPS (default 1; 0 leaves indirect jumps alone).
/// </summary>
public static class IndirectCallProfile
{
    /// <summary>MKW_INDIRECT_PROFILE_SITES=1: emit the per-site counting hook (profiling builds only).</summary>
    public static readonly bool Instrument = Environment.GetEnvironmentVariable("MKW_INDIRECT_PROFILE_SITES") == "1";

    private static Dictionary<(uint Function, uint Site), uint[]> _promotions = new();
    private static Dictionary<(uint Function, uint Site), uint[]> _jumpPromotions = new();

    public static int PromotedSiteCount => _promotions.Count;

    public static int PromotedTargetCount => _promotions.Values.Sum(static targets => targets.Length);

    public static int PromotedJumpSiteCount => _jumpPromotions.Count;

    private static double EnvDouble(string name, double fallback) =>
        double.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Float, CultureInfo.InvariantCulture,
            out var value) ? value : fallback;

    /// <summary>Reads the histogram and keeps each site's promoted targets (most frequent first).</summary>
    public static void Load(string path)
    {
        var minCount = (long)EnvDouble("MKW_INDIRECT_PROMOTE_MIN_COUNT", 64);
        var minShare = EnvDouble("MKW_INDIRECT_PROMOTE_MIN_SHARE", 0.15);
        var maxTargets = (int)EnvDouble("MKW_INDIRECT_PROMOTE_MAX_TARGETS", 2);
        var minCover = EnvDouble("MKW_INDIRECT_PROMOTE_MIN_COVER", 0.60);
        var lines = File.ReadLines(path).ToArray();
        _promotions = Select(Parse(lines), minCount, minShare, maxTargets, minCover);
        _jumpPromotions = Environment.GetEnvironmentVariable("MKW_INDIRECT_PROMOTE_JUMPS") == "0"
            ? new()
            : Select(Parse(lines, jumps: true), minCount, minShare, maxTargets, minCover);
    }

    /// <summary>For tests: replaces the promotions.</summary>
    public static void SetForTesting(
        Dictionary<(uint Function, uint Site), uint[]>? promotions,
        Dictionary<(uint Function, uint Site), uint[]>? jumpPromotions = null)
    {
        _promotions = promotions ?? new();
        _jumpPromotions = jumpPromotions ?? new();
    }

    /// <summary>The call sites' histograms, or with <paramref name="jumps"/> the tail jumps' ("J ..." lines).</summary>
    public static Dictionary<(uint Function, uint Site), Dictionary<uint, long>> Parse(IEnumerable<string> lines,
        bool jumps = false)
    {
        var sites = new Dictionary<(uint, uint), Dictionary<uint, long>>();
        foreach (var raw in lines)
        {
            var line = raw.Trim();
            if (line.Length == 0 || line[0] == '#') continue;
            var f = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            var jump = f[0] == "J";
            if (jump != jumps) continue;
            if (jump) f = f[1..];
            if (f.Length < 4) throw new InvalidDataException($"indirect profile: bad line '{line}'");
            var key = (Hex(f[0]), Hex(f[1]));
            var target = Hex(f[2]);
            var count = long.Parse(f[3], CultureInfo.InvariantCulture);
            if (!sites.TryGetValue(key, out var histogram)) sites[key] = histogram = new Dictionary<uint, long>();
            histogram[target] = histogram.GetValueOrDefault(target) + count;
        }
        return sites;
    }

    public static Dictionary<(uint Function, uint Site), uint[]> Select(
        Dictionary<(uint Function, uint Site), Dictionary<uint, long>> sites,
        long minCount, double minShare, int maxTargets, double minCover)
    {
        var result = new Dictionary<(uint, uint), uint[]>();
        foreach (var (key, histogram) in sites)
        {
            var total = histogram.Values.Sum();
            if (total < minCount || total <= 0) continue;
            var chosen = histogram
                .Where(static pair => pair.Key != 0)
                .OrderByDescending(static pair => pair.Value).ThenBy(static pair => pair.Key)
                .Take(Math.Max(0, maxTargets))
                .Where(pair => pair.Value >= minShare * total)
                .ToArray();
            if (chosen.Length == 0 || chosen.Sum(static pair => pair.Value) < minCover * total) continue;
            result[key] = chosen.Select(static pair => pair.Key).ToArray();
        }
        return result;
    }

    public static IReadOnlyList<uint> PromotedTargets(uint function, uint site, bool jump = false) =>
        site != 0 && (jump ? _jumpPromotions : _promotions).TryGetValue((function, site), out var targets)
            ? targets
            : Array.Empty<uint>();

    /// <summary>
    /// The promoted form of a site: one line, <c>if ((target) == 0xT) [[likely]] { InvokeDirectCpu&lt;0xT&gt;(ctx); }
    /// else ... { fallback }</c> (TranslatedBuildShardEmitter's PromotedIndirectArmRegex reads it back).
    /// </summary>
    public static string PromotedLine(string pad, string target, IReadOnlyList<uint> targets, string fallback)
    {
        var line = new System.Text.StringBuilder(pad);
        for (var t = 0; t < targets.Count; t++)
        {
            line.Append($"if (({target}) == 0x{targets[t]:X8}u){(t == 0 ? " [[likely]]" : "")} {{ InvokeDirectCpu<0x{targets[t]:X8}u>(ctx); }} else ");
        }
        return line.Append($"{{ {fallback} }}").ToString();
    }

    private static uint Hex(string text) =>
        uint.Parse(text.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ? text[2..] : text,
            NumberStyles.HexNumber, CultureInfo.InvariantCulture);
}
