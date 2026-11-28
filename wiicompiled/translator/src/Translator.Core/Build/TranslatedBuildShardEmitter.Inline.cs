using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Translator.Core.Build;

/// <summary>
/// Call-site inlining of translated functions at C++ level (emit-build-shards --inline-profile).
///
/// A selected direct call edge (caller, callee) gets, in the caller's shard, a <c>static inline always_inline</c> copy
/// of the callee's translated body (<c>func_X_inl</c>: the text of func_X_body, unchanged), and its call sites become
///     if (g_mkwGuestCodePatched[callee]) [[unlikely]] { flush; func_X(ctx); reload; } else { flush; func_X_inl(ctx); reload; }
/// The flush (the caller's ctx stores just before the call) and the reload (its ctx loads just after) are repeated
/// in both arms, so in the inlined arm Clang forwards the stores into the callee's entry loads and drops the ones the
/// callee overwrites. The copy is the callee's own C++, so the translated behaviour is identical; the guard keeps a
/// mod's patch of the callee winning at run time (the patched arm is the plain call into the guarded wrapper, as
/// before) without sending the caller to the interpreter: unlike the IR-level leaf inliner, no guard pair is added.
///
/// Selection, two rules united: the profile's edges by descending count until <c>MKW_INLINE_COVERAGE</c> of all profiled
/// direct calls is considered (default 0.98), taking callees up to <c>MKW_INLINE_MAX_CALLEE</c> statements (default 80);
/// and every static call of a callee with at most <c>MKW_INLINE_STATIC</c> statements (default 30; no profile needed;
/// 0 = off, and coverage 0 turns the profile rule off). An edge with more than <c>MKW_INLINE_MAX_SITES</c> call sites
/// in its caller (default 8) keeps them all as calls. Measured on the 3-replay benchmark: large callees lost what they
/// saved (each caller's copy is colder than the one shared body), small hot callees carry the gain.
/// <c>MKW_INLINE_DEPTH</c> (default 1) lets the copies inline their own selected edges (copies at level d are
/// func_X_inl{d}, so nesting is bounded by construction).
/// <c>MKW_INLINE=0</c> turns it off.
/// <c>MKW_INLINE_COLD_PATCHED=1</c> moves the patched arm's flush into a cold out-of-line closure so the inline arm's
/// stores are not merged and hoisted above the guard; measured neutral (the inlined bodies keep those ctx fields live)
/// for +2% text, so it is off.
/// </summary>
public static partial class TranslatedBuildShardEmitter
{
    /// <summary>The profile: per line "caller callee count" (hex guest addresses); a caller ending in "v" is a code
    /// variant's (Translator.Cli/CodeVariants.cs). '#' comments.</summary>
    public sealed record InlineProfileEdge(uint Caller, bool VariantCaller, uint Callee, long Count);

    public static IReadOnlyList<InlineProfileEdge> ReadInlineProfile(string path)
    {
        var edges = new List<InlineProfileEdge>();
        foreach (var raw in File.ReadLines(path))
        {
            var line = raw.Trim();
            if (line.Length == 0 || line[0] == '#') continue;
            var f = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (f.Length < 3) throw new InvalidDataException($"Bad inline profile line '{raw}' in '{path}'.");
            var variant = f[0].EndsWith('v');
            edges.Add(new InlineProfileEdge(
                ParseHexUInt32(variant ? f[0][..^1] : f[0]), variant, ParseHexUInt32(f[1]),
                long.Parse(f[2], CultureInfo.InvariantCulture)));
        }
        return edges;
    }

    internal sealed record InlineSettings(double Coverage, int MaxCalleeStatements, int MaxSitesPerEdge, int StaticMaxStatements, int Depth)
    {
        public static InlineSettings FromEnvironment()
        {
            static string? Env(string name) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } v ? v : null;
            return new InlineSettings(
                Env("MKW_INLINE_COVERAGE") is { } c ? double.Parse(c, CultureInfo.InvariantCulture) : 0.98,
                Env("MKW_INLINE_MAX_CALLEE") is { } m ? int.Parse(m, CultureInfo.InvariantCulture) : 80,
                Env("MKW_INLINE_MAX_SITES") is { } s ? int.Parse(s, CultureInfo.InvariantCulture) : 8,
                Env("MKW_INLINE_STATIC") is { } t ? int.Parse(t, CultureInfo.InvariantCulture) : 30,
                Env("MKW_INLINE_DEPTH") is { } d ? Math.Max(1, int.Parse(d, CultureInfo.InvariantCulture)) : 1);
        }
    }

    /// <summary>The selected edges, by caller (base functions; code variant callers keyed separately).</summary>
    internal sealed class InlinePlan
    {
        public required IReadOnlyDictionary<uint, IReadOnlySet<uint>> BaseEdges { get; init; }
        public required IReadOnlyDictionary<uint, IReadOnlySet<uint>> VariantEdges { get; init; }
        public required IReadOnlyDictionary<uint, int> GuardIndex { get; init; }
        public required IReadOnlyDictionary<uint, FunctionRecordView> Callees { get; init; }
        public required int Depth { get; init; }
        public required int MaxSitesPerEdge { get; init; }
        public bool ColdPatchedArm { get; init; }
        public static readonly InlinePlan Empty = new()
        {
            BaseEdges = new Dictionary<uint, IReadOnlySet<uint>>(),
            VariantEdges = new Dictionary<uint, IReadOnlySet<uint>>(),
            GuardIndex = new Dictionary<uint, int>(),
            Callees = new Dictionary<uint, FunctionRecordView>(),
            Depth = 1,
            MaxSitesPerEdge = 0
        };
        public bool IsEmpty => BaseEdges.Count == 0 && VariantEdges.Count == 0;
    }

    /// <summary>What a copy needs of its callee's record.</summary>
    internal sealed record FunctionRecordView(uint Address, string Fingerprint, string Source, IReadOnlyList<uint> DirectCalls);

    private static readonly uint[] NeverInlinedCallees =
    {
        0x80029438u, // MKW_STATIC_TRANSLATED_CALL runs BrawlRollback::AnchorPads() after it
        0x809435F0u, 0x80949BC0u, // entry hooks (InsertGuestCodeGuardAtEntries)
    };

    private static InlinePlan BuildInlinePlan(
        string? profilePath,
        IReadOnlyList<FunctionRecord> functions,
        IReadOnlyDictionary<uint, Trait> traits,
        IReadOnlyDictionary<uint, int> guardIndex,
        IReadOnlyDictionary<uint, string> nativeBodies)
    {
        if (Environment.GetEnvironmentVariable("MKW_INLINE") == "0") return InlinePlan.Empty;
        var settings = InlineSettings.FromEnvironment();
        var profile = profilePath is not null && File.Exists(profilePath) ? ReadInlineProfile(profilePath) : null;
        // Only with --inline-profile (build.py passes it when projects/brawl/inline_profile.txt exists), so other
        // users of the emitter keep their output unchanged.
        if (profile is null) return InlinePlan.Empty;

        var byAddress = functions.ToDictionary(static f => f.Address);
        var statements = new Dictionary<uint, int>();
        int Statements(FunctionRecord f)
        {
            lock (statements)
            {
                if (statements.TryGetValue(f.Address, out var n)) return n;
                n = (f.SourceText ?? File.ReadAllText(f.SourcePath)).Count(static c => c == ';');
                statements[f.Address] = n;
                return n;
            }
        }
        string? Reject(uint callee, int maxStatements)
        {
            if (!byAddress.TryGetValue(callee, out var record)) return "not a translated base function";
            if (!guardIndex.ContainsKey(callee)) return "no guard";
            if (nativeBodies.ContainsKey(callee) || NativeHookSlot(callee) >= 0) return "native body/hook";
            if (Array.IndexOf(NeverInlinedCallees, callee) >= 0) return "call/entry hook";
            if (!traits.TryGetValue(callee, out var trait) || !trait.Available || trait.NonvolatileFprWriteMask != 0 ||
                trait.MustRemainDynamicallyDispatchable || trait.WinnerSymbol != $"func_{callee:X8}" ||
                RequiresGenericDirectCall(callee))
                return "not a static translated call";
            if (Statements(record) > maxStatements) return "too large";
            return null;
        }

        var baseEdges = new Dictionary<uint, HashSet<uint>>();
        var variantEdges = new Dictionary<uint, HashSet<uint>>();
        var reasons = new Dictionary<string, long>();
        long total = 0, taken = 0, considered = 0;
        if (settings.StaticMaxStatements > 0)
        {
            foreach (var f in functions)
            {
                foreach (var callee in f.DirectCalls)
                {
                    if (callee == f.Address || Reject(callee, settings.StaticMaxStatements) is not null) continue;
                    if (!baseEdges.TryGetValue(f.Address, out var set)) baseEdges[f.Address] = set = new HashSet<uint>();
                    set.Add(callee);
                }
            }
        }
        // Both may be on: the profile's hot edges and the static small callees, united.
        if (profile is not null && settings.Coverage > 0)
        {
            total = profile.Sum(static e => e.Count);
            foreach (var edge in profile!.OrderByDescending(static e => e.Count).ThenBy(static e => e.Caller).ThenBy(static e => e.Callee))
            {
                if (considered >= settings.Coverage * total) break;
                considered += edge.Count;
                var reason = edge.Callee == edge.Caller ? "recursive"
                    : !edge.VariantCaller && !byAddress.ContainsKey(edge.Caller) ? "caller not translated"
                    : Reject(edge.Callee, settings.MaxCalleeStatements);
                if (reason is not null)
                {
                    reasons[reason] = reasons.GetValueOrDefault(reason) + edge.Count;
                    continue;
                }
                var map = edge.VariantCaller ? variantEdges : baseEdges;
                if (!map.TryGetValue(edge.Caller, out var set)) map[edge.Caller] = set = new HashSet<uint>();
                set.Add(edge.Callee);
                taken += edge.Count;
            }
        }

        if (profile is not null)
        {
            total = profile.Sum(static e => e.Count);
            taken = profile.Where(e => (e.VariantCaller ? variantEdges : baseEdges).TryGetValue(e.Caller, out var s) && s.Contains(e.Callee))
                .Sum(static e => e.Count);
        }
        var callees = baseEdges.Values.Concat(variantEdges.Values).SelectMany(static s => s).ToHashSet();
        // Nested copies inline their own selected edges: their callees need copies too.
        var frontier = new Queue<uint>(callees);
        for (var level = 1; level < settings.Depth && frontier.Count != 0; level++)
        {
            var next = new Queue<uint>();
            while (frontier.Count != 0)
            {
                var c = frontier.Dequeue();
                if (!baseEdges.TryGetValue(c, out var set)) continue;
                foreach (var y in set)
                    if (callees.Add(y)) next.Enqueue(y);
            }
            frontier = next;
        }
        Console.WriteLine(
            $"[translator] Inline plan: {baseEdges.Sum(static p => p.Value.Count):N0} base edge(s), " +
            $"{variantEdges.Sum(static p => p.Value.Count):N0} code variant edge(s), {callees.Count:N0} callee(s), depth {settings.Depth}" +
            (settings.StaticMaxStatements > 0 ? $", static <= {settings.StaticMaxStatements} statements" : "") +
            (profile is not null && settings.Coverage > 0 ? $", coverage target {settings.Coverage:0.###}, callee <= {settings.MaxCalleeStatements} statements" : "") +
            (total > 0 ? $"; profiled calls covered {100.0 * taken / total:0.0}%" : ""));
        foreach (var (reason, count) in reasons.OrderByDescending(static p => p.Value))
            Console.WriteLine($"[translator] Inline plan: not taken ({reason}): {100.0 * count / Math.Max(1, total):0.0}% of profiled calls");

        return new InlinePlan
        {
            BaseEdges = baseEdges.ToDictionary(static p => p.Key, static p => (IReadOnlySet<uint>)p.Value),
            VariantEdges = variantEdges.ToDictionary(static p => p.Key, static p => (IReadOnlySet<uint>)p.Value),
            GuardIndex = guardIndex,
            Callees = callees.ToDictionary(static a => a, a =>
            {
                var r = byAddress[a];
                return new FunctionRecordView(a, r.SourceFingerprint, r.SourceText ?? File.ReadAllText(r.SourcePath), r.DirectCalls);
            }),
            Depth = settings.Depth,
            MaxSitesPerEdge = settings.MaxSitesPerEdge,
            ColdPatchedArm = Environment.GetEnvironmentVariable("MKW_INLINE_COLD_PATCHED") == "1"
        };
    }

    [GeneratedRegex(@"^(?<indent>[ \t]*)MKW_STATIC_TRANSLATED_CALL\(0x(?<address>[0-9A-F]{8})u, func_\k<address>, ctx\);$", RegexOptions.CultureInvariant)]
    private static partial Regex StaticCallLineRegex();

    // A flush store just before a call: one ctx field set from an expression without calls.
    [GeneratedRegex(@"^[ \t]*ctx->[A-Za-z0-9_\[\]\.]+ = [^;(]*(\([^;()]*\)[^;(]*)*;$", RegexOptions.CultureInvariant)]
    private static partial Regex FlushLineRegex();

    // A reload just after a call: one local set from one ctx field.
    [GeneratedRegex(@"^[ \t]*[A-Za-z_][A-Za-z0-9_]* = ctx->[A-Za-z0-9_\[\]\.]+;$", RegexOptions.CultureInvariant)]
    private static partial Regex ReloadLineRegex();

    // A promoted indirect call or tail jump (CodeGen/IndirectCallProfile.PromotedLine): one line testing the target
    // against the promoted addresses, each arm one call statement, the last arm the indirect call/jump.
    [GeneratedRegex(@"^(?<indent>[ \t]*)if \(\((?<target>[A-Za-z_][A-Za-z0-9_]*)\) == 0x(?<address>[0-9A-F]{8})u\) \[\[likely\]\] \{ (?<call>[^{}]+;) \}(?: else if \(\(\k<target>\) == 0x(?<address>[0-9A-F]{8})u\) \{ (?<call>[^{}]+;) \})* else \{ (?<fallback>MKW_INDIRECT_(?:CALL|JUMP)\([^{}]+\);) \}$", RegexOptions.CultureInvariant)]
    private static partial Regex PromotedLineRegex();

    [GeneratedRegex(@"^MKW_STATIC_TRANSLATED_CALL\(0x(?<address>[0-9A-F]{8})u, func_\k<address>, ctx\);$", RegexOptions.CultureInvariant)]
    private static partial Regex StaticCallStatementRegex();

    /// <summary>
    /// Rewrites the call sites of <paramref name="caller"/>'s selected edges in <paramref name="source"/> (already
    /// lowered by <see cref="LowerStableDirectCalls"/>) into the guarded inline form calling func_X_inl{level}.
    /// Plain call lines and the direct arms of promoted indirect calls/jumps are both sites; a promoted line with a
    /// selected arm is split into an if/else chain whose arms each repeat the flush and the reload around their call
    /// (the target is a local the flush does not change, so testing it before the flush is the same test).
    /// Returns the callees whose copies the rewritten text uses.
    /// </summary>
    internal static string RewriteInlineCalls(
        string source, IReadOnlySet<uint>? edges, InlinePlan plan, int level, ISet<uint> used)
    {
        if (edges is null || edges.Count == 0 || !source.Contains("MKW_STATIC_TRANSLATED_CALL(", StringComparison.Ordinal))
            return source;
        var lines = source.Split('\n');
        // An edge with more sites than the limit keeps them all as calls (the profile does not say which is hot).
        var sites = new Dictionary<uint, int>();
        void Count(string address) => sites[ParseHexUInt32(address)] = sites.GetValueOrDefault(ParseHexUInt32(address)) + 1;
        foreach (var line in lines)
        {
            var trimmed = line.TrimEnd('\r');
            var m = StaticCallLineRegex().Match(trimmed);
            if (m.Success) { Count(m.Groups["address"].Value); continue; }
            var p = PromotedLineRegex().Match(trimmed);
            if (!p.Success) continue;
            foreach (Capture call in p.Groups["call"].Captures)
                if (StaticCallStatementRegex().Match(call.Value) is { Success: true } c) Count(c.Groups["address"].Value);
        }
        int Selected(string statement)
        {
            var m = StaticCallStatementRegex().Match(statement);
            if (!m.Success) return -1;
            var callee = ParseHexUInt32(m.Groups["address"].Value);
            return edges.Contains(callee) && plan.Callees.ContainsKey(callee) && sites[callee] <= plan.MaxSitesPerEdge &&
                   plan.GuardIndex.TryGetValue(callee, out var index) ? index : -1;
        }
        var output = new List<string>(lines.Length + 16);
        var suffix = level == 1 ? "_inl" : $"_inl{level}";
        // One call with its flush before and its reload after, at indent.
        void Site(string indent, List<string> flush, List<string> reload, string statement)
        {
            var index = Selected(statement);
            if (index < 0)
            {
                output.AddRange(flush.Select(l => indent + l));
                output.Add(indent + statement);
                output.AddRange(reload.Select(l => indent + l));
                return;
            }
            var callee = ParseHexUInt32(StaticCallStatementRegex().Match(statement).Groups["address"].Value);
            var name = $"func_{callee:X8}";
            output.Add($"{indent}if (__builtin_expect(g_mkwGuestCodePatched[{index}u], 0)) {{");
            if (plan.ColdPatchedArm && flush.Count != 0)
            {
                // The flush happens only inside an out-of-line cold closure (values captured by copy), so the inline
                // arm's copy of the stores is its own: Clang cannot merge and hoist the two copies above the guard.
                output.Add($"{indent}    [=]() __attribute__((cold, noinline)) {{");
                output.AddRange(flush.Select(l => indent + "        " + l));
                output.Add($"{indent}        {name}(ctx);");
                output.Add($"{indent}    }}();");
            }
            else
            {
                output.AddRange(flush.Select(l => indent + "    " + l));
                output.Add($"{indent}    {name}(ctx);");
            }
            output.AddRange(reload.Select(l => indent + "    " + l));
            output.Add($"{indent}}} else {{");
            output.AddRange(flush.Select(l => indent + "    " + l));
            output.Add($"{indent}    {name}{suffix}(ctx);");
            output.AddRange(reload.Select(l => indent + "    " + l));
            output.Add($"{indent}}}");
            used.Add(callee);
        }
        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i].TrimEnd('\r');
            var match = StaticCallLineRegex().Match(line);
            var promoted = match.Success ? null : PromotedLineRegex().Match(line);
            var calls = match.Success
                ? new List<string> { line.TrimStart() }
                : promoted!.Success ? promoted.Groups["call"].Captures.Select(static c => c.Value).ToList() : null;
            if (calls is null || !calls.Any(c => Selected(c) >= 0))
            {
                output.Add(lines[i]);
                continue;
            }
            var flushStart = output.Count;
            while (flushStart > 0 && FlushLineRegex().IsMatch(output[flushStart - 1].TrimEnd('\r'))) flushStart--;
            var flush = output.GetRange(flushStart, output.Count - flushStart).Select(static l => l.TrimEnd('\r').TrimStart()).ToList();
            output.RemoveRange(flushStart, output.Count - flushStart);
            var reload = new List<string>();
            while (i + 1 < lines.Length && ReloadLineRegex().IsMatch(lines[i + 1].TrimEnd('\r'))) reload.Add(lines[++i].TrimEnd('\r').TrimStart());
            var indent = (match.Success ? match : promoted!).Groups["indent"].Value;
            if (match.Success)
            {
                Site(indent, flush, reload, calls[0]);
                continue;
            }
            var target = promoted!.Groups["target"].Value;
            var addresses = promoted.Groups["address"].Captures;
            for (var arm = 0; arm < calls.Count; arm++)
            {
                output.Add(arm == 0
                    ? $"{indent}if (({target}) == 0x{addresses[arm].Value}u) [[likely]] {{"
                    : $"{indent}}} else if (({target}) == 0x{addresses[arm].Value}u) {{");
                Site(indent + "    ", flush, reload, calls[arm]);
            }
            output.Add($"{indent}}} else {{");
            Site(indent + "    ", flush, reload, promoted.Groups["fallback"].Value);
            output.Add($"{indent}}}");
        }
        return string.Join('\n', output);
    }

    /// <summary>
    /// The copies a shard needs (func_X_inl{level} for each callee its functions' rewritten sites use, and theirs up
    /// to the plan's depth), as C++ text: forward declarations first, then the definitions.
    /// </summary>
    private static string BuildInlineCopies(
        ISet<uint> level1, InlinePlan plan, IReadOnlyDictionary<uint, Trait> traits, ISet<uint> dependencyCalls,
        StringBuilder identity)
    {
        if (level1.Count == 0) return string.Empty;
        var declarations = new StringBuilder();
        var definitions = new StringBuilder();
        var current = level1.Order().ToList();
        for (var level = 1; current.Count != 0 && level <= plan.Depth; level++)
        {
            var next = new SortedSet<uint>();
            foreach (var callee in current)
            {
                var view = plan.Callees[callee];
                var name = $"func_{callee:X8}{(level == 1 ? "_inl" : $"_inl{level}")}";
                identity.Append(CultureInfo.InvariantCulture, $"inl:{callee:X8}:{level}:{view.Fingerprint}\n");
                foreach (var call in view.DirectCalls) dependencyCalls.Add(call);
                var text = RegistrationMarkerRegex().Replace(view.Source, string.Empty);
                text = LowerStableDirectCalls(text, traits);
                var used = new SortedSet<uint>();
                if (level < plan.Depth)
                {
                    text = RewriteInlineCalls(text, plan.BaseEdges.GetValueOrDefault(callee), plan, level + 1, used);
                    foreach (var u in used)
                    {
                        identity.Append(CultureInfo.InvariantCulture, $"inl-edge:{callee:X8}:{u:X8}:{level}\n");
                        next.Add(u);
                    }
                }
                text = GuardStateFreeCalls(text, plan.GuardIndex);
                var entry = GuardedEntryRegex().Match(text);
                var end = text.LastIndexOf('}');
                if (!entry.Success || end < entry.Index + entry.Length)
                    throw new InvalidDataException($"Cannot find the body of 0x{callee:X8} to inline.");
                declarations.Append(CultureInfo.InvariantCulture,
                    $"static inline __attribute__((always_inline)) void {name}(CpuContext* MKW_RESTRICT ctx);\n");
                definitions.Append(CultureInfo.InvariantCulture,
                    $"static inline __attribute__((always_inline)) void {name}(CpuContext* MKW_RESTRICT ctx)\n{{");
                definitions.Append(text, entry.Index + entry.Length, end - (entry.Index + entry.Length));
                definitions.Append("}\n");
            }
            current = next.ToList();
        }
        return declarations.ToString() + definitions;
    }
}
