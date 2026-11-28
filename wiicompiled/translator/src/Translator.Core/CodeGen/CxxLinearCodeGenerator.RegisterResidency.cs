using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Translator.Core.Analysis;

namespace Translator.Core.CodeGen;

public sealed partial class CxxLinearCodeGenerator
{
    /// <summary>
    /// Guest register residency for one emitted function body: C++ locals are authoritative between call boundaries (host regs), synced via flush/reload at each one.
    /// Emitted twice, a recording pass then the real pass, so any register reference outside the recorded set fails loud instead of silently reading stale <c>ctx-&gt;</c> state.
    /// </summary>
    private sealed class RegisterResidency
    {
        private readonly bool _recording;
        private readonly SortedSet<int> _gprs = new();
        private readonly SortedSet<int> _fprs = new();
        private readonly SortedSet<int> _writtenGprs = new();
        private readonly SortedSet<int> _writtenFprs = new();
        // CR fields held in locals (bit f: crv<f>, isa/ppc_isa_crv.h), and the ones written.
        private byte _crFields;
        private bool _ctr;
        private bool _xer;
        private byte _writtenCrFields;
        private bool _writesCtr;
        private bool _writesXer;

        // Dirty-since-sync tracking. The recording pass logs, per block and in emission order, every local
        // write and every flush/reload; SolveFlushPlan turns that into the set of locals that can differ from
        // CpuContext at each flush, and the real pass stores only those. Without a plan every flush stores the
        // whole write set, as before.
        private enum SyncEventKind { Write, Flush, Reload }
        private readonly record struct SyncEvent(SyncEventKind Kind, UInt128 Mask, bool Unconditional);
        private readonly List<string> _blockOrder = new();
        private readonly Dictionary<string, List<SyncEvent>> _blockEvents = new(StringComparer.OrdinalIgnoreCase);
        private string? _block;
        private bool _eventOutsideBlock;
        private int _flushOrdinal;
        private FlushPlan? _flushPlan;

        private sealed record FlushPlan(UInt128[] DirtyAtFlush, string[] BlockAtFlush, UInt128 StoredUnion);

        /// <summary>
        /// Every register some flush of this body may store into CpuContext (dirty there and in that flush's mask),
        /// from the solved plan; null without one (then each flush stores its whole mask of the write set).
        /// </summary>
        public UInt128? PossiblyStoredByFlushes => _flushPlan?.StoredUnion;

        private const int CtrKey = 65;
        private const int XerKey = 66;
        private const int CrFieldKeyBase = 67;  // 67..74: CR fields 0..7, each synced on its own
        private static int CrFieldKey(int field) => CrFieldKeyBase + field;

        private RegisterResidency(bool recording) => _recording = recording;

        public static RegisterResidency CreateRecorder() => new(true);

        public RegisterResidency Snapshot()
        {
            var snapshot = new RegisterResidency(false);
            snapshot._gprs.UnionWith(_gprs);
            snapshot._fprs.UnionWith(_fprs);
            snapshot._writtenGprs.UnionWith(_writtenGprs);
            snapshot._writtenFprs.UnionWith(_writtenFprs);
            snapshot._crFields = _crFields;
            snapshot._ctr = _ctr;
            snapshot._xer = _xer;
            snapshot._writtenCrFields = _writtenCrFields;
            snapshot._writesCtr = _writesCtr;
            snapshot._writesXer = _writesXer;
            return snapshot;
        }

        public bool IsEmpty => _gprs.Count == 0 && _fprs.Count == 0 && _crFields == 0 && !_ctr && !_xer;

        public static string GprLocal(int index) => GprRegisterName(index);
        public static string FprLocal(int index) => FprRegisterName(index);
        public static string CrFieldLocal(int field) => $"crv{field}";
        public const string CtrLocal = "ctr";
        public const string XerLocal = "xer";

        public IEnumerable<string> DeclaredLocalNames
        {
            get
            {
                foreach (var gpr in _gprs) yield return GprLocal(gpr);
                foreach (var fpr in _fprs) yield return FprLocal(fpr);
                for (var field = 0; field < 8; ++field)
                    if ((_crFields & (1 << field)) != 0) yield return CrFieldLocal(field);
                if (_ctr) yield return CtrLocal;
                if (_xer) yield return XerLocal;
            }
        }

        public string Gpr(int index, bool written)
        {
            Track(_gprs, _writtenGprs, index, written, "r");
            if (_recording && written) RecordWrite(index);
            return GprLocal(index);
        }

        /// <summary>Scalar (PS0) view of a localized FPR.</summary>
        public string FprScalar(int index, bool written)
        {
            Track(_fprs, _writtenFprs, index, written, "f");
            if (_recording && written) RecordWrite(32 + index);
            return FprScalarLocalName(index);
        }

        /// <summary>Whole <c>PPC_FPR</c> value of a localized FPR.</summary>
        public string FprStorage(int index, bool written)
        {
            Track(_fprs, _writtenFprs, index, written, "f");
            if (_recording && written) RecordWrite(32 + index);
            return FprLocal(index);
        }

        /// <summary>One CR field's local (an <c>int64_t</c> field value, isa/ppc_isa_crv.h).</summary>
        public string CrField(int field, bool written)
        {
            field &= 7;
            var bit = (byte)(1 << field);
            if (_recording)
            {
                _crFields |= bit;
                if (written)
                {
                    _writtenCrFields |= bit;
                    RecordWrite(CrFieldKey(field));
                }
            }
            else if ((_crFields & bit) == 0 || (written && (_writtenCrFields & bit) == 0))
            {
                throw new InvalidOperationException(
                    $"Register residency did not localize cr{field} for this function body.");
            }

            return CrFieldLocal(field);
        }

        /// <summary>
        /// The whole CR: read, the packed word of the eight field locals; written, a <c>PpcCrView</c> of them that
        /// takes a packed word.
        /// </summary>
        public string CrWhole(bool written)
        {
            var fields = new string[8];
            for (var field = 0; field < 8; ++field) fields[field] = CrField(field, written);
            return $"{(written ? "PpcCrView" : "PpcCrvPack")}({string.Join(", ", fields)})";
        }

        public string Ctr(bool written)
        {
            if (_recording)
            {
                _ctr = true;
                _writesCtr |= written;
                if (written) RecordWrite(CtrKey);
            }
            else if (!_ctr || (written && !_writesCtr))
            {
                throw new InvalidOperationException(
                    "Register residency did not localize ctr for this function body.");
            }

            return CtrLocal;
        }

        /// <summary>
        /// XER, localized as one value (not just summary-overflow) because the carry helpers
        /// read/modify/write it and every integer compare reads XER.SO into its CR field.
        /// </summary>
        public string Xer(bool written)
        {
            if (_recording)
            {
                _xer = true;
                _writesXer |= written;
                if (written) RecordWrite(XerKey);
            }
            else if (!_xer || (written && !_writesXer))
            {
                throw new InvalidOperationException(
                    "Register residency did not localize xer for this function body.");
            }

            return XerLocal;
        }

        private void Track(SortedSet<int> referenced, SortedSet<int> written, int index, bool isWrite, string prefix)
        {
            if (_recording)
            {
                referenced.Add(index);
                if (isWrite) written.Add(index);
                return;
            }

            if (!referenced.Contains(index) || (isWrite && !written.Contains(index)))
            {
                throw new InvalidOperationException(
                    $"Register residency did not localize {prefix}{index} for this function body.");
            }
        }

        /// <summary>
        /// Entry loads. Every referenced register is loaded unconditionally:
        /// liveness-narrowed initialization is a later optimization and reading
        /// an architectural register is always well defined.
        /// </summary>
        public void AppendEntryLoads(StringBuilder sb)
        {
            if (_recording) return;
            foreach (var gpr in _gprs)
                sb.Append("    uint32_t ").Append(GprLocal(gpr)).Append(" = ctx->gpr[").Append(gpr).AppendLine("];");
            foreach (var fpr in _fprs)
                sb.Append("    PPC_FPR ").Append(FprLocal(fpr)).Append(" = ctx->fpr[").Append(fpr).AppendLine("];");
            AppendCrFieldEntryLoads(sb);
            if (_ctr) sb.AppendLine($"    uint32_t {CtrLocal} = ctx->ctr;");
            if (_xer) sb.AppendLine($"    uint32_t {XerLocal} = ctx->xer;");
        }

        private void AppendCrFieldEntryLoads(StringBuilder sb)
        {
            for (var field = 0; field < 8; ++field)
                if ((_crFields & (1 << field)) != 0)
                    sb.AppendLine($"    int64_t {CrFieldLocal(field)} = ctx->crv[{field}];");
        }

        /// <summary>
        /// MKW_EABI_REGS register-passing entry: its argument and result registers must be locals of the body even
        /// when only a callee reads or writes them, and flushable (written) since the arguments arrive dirty.
        /// </summary>
        public void RequireLocals(uint gprs, uint fprs)
        {
            if (!_recording) throw new InvalidOperationException("Only a recording residency can require locals.");
            for (var gpr = 0; gpr < 32; ++gpr)
                if ((gprs & (1u << gpr)) != 0) { _gprs.Add(gpr); _writtenGprs.Add(gpr); }
            for (var fpr = 0; fpr < 32; ++fpr)
                if ((fprs & (1u << fpr)) != 0) { _fprs.Add(fpr); _writtenFprs.Add(fpr); }
        }

        /// <summary>
        /// Entry of a register-passing body: arguments from the host parameters, r14-r31/f14-f31 from nothing
        /// (EABI: their entry values are only ever saved to the frame, and CpuContext's copies are stale anyway),
        /// everything else from CpuContext.
        /// </summary>
        public void AppendVariantEntryLoads(StringBuilder sb, GuestEabiVariant variant)
        {
            if (_recording) return;
            foreach (var gpr in _gprs)
            {
                sb.Append("    uint32_t ").Append(GprLocal(gpr));
                if ((variant.InputGprMask & (1u << gpr)) != 0) sb.Append(" = eabi_r").Append(gpr).AppendLine(";");
                else if ((GuestEabi.NonvolatileGprMask & (1u << gpr)) != 0) sb.AppendLine(" = 0;");
                else sb.Append(" = ctx->gpr[").Append(gpr).AppendLine("];");
            }
            foreach (var fpr in _fprs)
            {
                sb.Append("    PPC_FPR ").Append(FprLocal(fpr));
                if ((variant.InputFprMask & (1u << fpr)) != 0) sb.Append(" = eabi_f").Append(fpr).AppendLine(";");
                else if ((GuestEabi.NonvolatileFprMask & (1u << fpr)) != 0) sb.AppendLine("{};");
                else sb.Append(" = ctx->fpr[").Append(fpr).AppendLine("];");
            }
            AppendCrFieldEntryLoads(sb);
            if (_ctr) sb.AppendLine($"    uint32_t {CtrLocal} = ctx->ctr;");
            if (_xer) sb.AppendLine($"    uint32_t {XerLocal} = ctx->xer;");
        }

        /// <summary>
        /// Locals of a register-passing body that CpuContext must still receive: the arguments (r1 excepted: the
        /// caller publishes it before the call). The nonvolatiles started at zero differ from CpuContext too, but its
        /// copies of them are stale in this mode anyway and no call site reads them back.
        /// </summary>
        public UInt128 VariantEntryDirty(GuestEabiVariant variant)
        {
            var dirty = UInt128.Zero;
            foreach (var gpr in _gprs)
                if (gpr != 1 && (variant.InputGprMask & (1u << gpr)) != 0)
                    dirty |= UInt128.One << gpr;
            foreach (var fpr in _fprs)
                if ((variant.InputFprMask & (1u << fpr)) != 0)
                    dirty |= UInt128.One << (32 + fpr);
            return dirty;
        }

        /// <summary>
        /// Stores the possibly-written localized registers back into
        /// <c>CpuContext</c>. Storing an unmodified register writes the same
        /// value back, so the conservative write set stays correct.
        /// </summary>
        public void AppendFlush(StringBuilder sb, string pad) => AppendFlush(sb, pad, ResidencyBoundarySync.Full);

        /// <summary>
        /// Contract-narrowed flush. Only the registers the boundary can observe
        /// or overwrite are published; see <see cref="ResidencyBoundarySync"/>
        /// for why the write set has to participate in the flush mask.
        /// </summary>
        public void AppendFlush(StringBuilder sb, string pad, ResidencyBoundarySync sync) =>
            AppendFlush(sb, pad, sync, unconditional: false);

        /// <summary>
        /// <paramref name="unconditional"/>: the flush runs on every path through this point (not in one arm of
        /// an emitted <c>if</c>, not before a <c>return</c>), so afterwards the flushed locals equal CpuContext.
        /// Only such flushes let later ones skip a register; a conditional one still stores what is dirty.
        /// </summary>
        public void AppendFlush(StringBuilder sb, string pad, ResidencyBoundarySync sync, bool unconditional)
        {
            if (_recording)
            {
                RecordEvent(new SyncEvent(SyncEventKind.Flush, FlushMask(sync), unconditional));
                return;
            }
            var dirty = NextFlushDirtySet();
            foreach (var gpr in _writtenGprs)
                if (sync.FlushesGpr(gpr) && IsDirty(dirty, gpr))
                    sb.Append(pad).Append("ctx->gpr[").Append(gpr).Append("] = ").Append(GprLocal(gpr)).AppendLine(";");
            foreach (var fpr in _writtenFprs)
                if (sync.FlushesFpr(fpr) && IsDirty(dirty, 32 + fpr))
                    sb.Append(pad).Append("ctx->fpr[").Append(fpr).Append("] = ").Append(FprLocal(fpr)).AppendLine(";");
            for (var field = 0; field < 8; ++field)
                if ((_writtenCrFields & (1 << field)) != 0 && sync.FlushesCrField(field) && IsDirty(dirty, CrFieldKey(field)))
                    sb.Append(pad).AppendLine($"ctx->crv[{field}] = {CrFieldLocal(field)};");
            if (_writesCtr && sync.FlushesCtr && IsDirty(dirty, CtrKey)) sb.Append(pad).AppendLine($"ctx->ctr = {CtrLocal};");
            if (_writesXer && sync.FlushesXer && IsDirty(dirty, XerKey)) sb.Append(pad).AppendLine($"ctx->xer = {XerLocal};");
        }

        private static bool IsDirty(UInt128 dirty, int key) => ((dirty >> key) & UInt128.One) != UInt128.Zero;

        private static UInt128 FlushMask(ResidencyBoundarySync sync) =>
            (UInt128)sync.FlushGprMask | ((UInt128)sync.FlushFprMask << 32) |
            CrFieldKeys(sync.FlushedCrFields) |
            (sync.FlushesCtr ? UInt128.One << CtrKey : UInt128.Zero) |
            (sync.FlushesXer ? UInt128.One << XerKey : UInt128.Zero);

        private static UInt128 ReloadMask(ResidencyBoundarySync sync) =>
            (UInt128)sync.ReloadGprMask | ((UInt128)sync.ReloadFprMask << 32) |
            CrFieldKeys(sync.ReloadedCrFields) |
            (sync.ReloadsCtr ? UInt128.One << CtrKey : UInt128.Zero) |
            (sync.ReloadsXer ? UInt128.One << XerKey : UInt128.Zero);

        private static UInt128 CrFieldKeys(byte fields)
        {
            var keys = UInt128.Zero;
            for (var field = 0; field < 8; ++field)
                if ((fields & (1 << field)) != 0) keys |= UInt128.One << CrFieldKey(field);
            return keys;
        }

        /// <summary>Starts one emission of the body: flushes are numbered from zero again.</summary>
        public void BeginEmission()
        {
            _block = null;
            _flushOrdinal = 0;
            if (!_recording) return;
            _blockOrder.Clear();
            _blockEvents.Clear();
            _eventOutsideBlock = false;
        }

        public void BeginBlock(string label)
        {
            _block = label;
            if (_recording && !_blockEvents.ContainsKey(label))
            {
                _blockOrder.Add(label);
                _blockEvents.Add(label, new List<SyncEvent>());
            }
        }

        /// <summary>
        /// Ends the per-block part of an emission. Anything emitted after this (the shared LR-continuation
        /// dispatch) is reached outside the block CFG, so an event there disables the plan; the real pass also
        /// checks it emitted exactly the flushes the plan was solved for.
        /// </summary>
        public void EndBlocks()
        {
            _block = null;
            if (!_recording && _flushPlan is not null && _flushOrdinal != _flushPlan.DirtyAtFlush.Length)
            {
                throw new InvalidOperationException(
                    $"Register residency emitted {_flushOrdinal} flushes but its plan has {_flushPlan.DirtyAtFlush.Length}.");
            }
        }

        private void RecordWrite(int key) => RecordEvent(new SyncEvent(SyncEventKind.Write, UInt128.One << key, true));

        // Instruction capture (recording pass only): the FPR locals one instruction's emission writes or reloads.
        private bool _capturing;
        private uint _capturedFprs;

        /// <summary>Starts collecting the FPR locals the next emission writes or reloads (recording pass).</summary>
        public void BeginFprCapture()
        {
            _capturing = _recording;
            _capturedFprs = 0;
        }

        /// <summary>
        /// Ends a capture: every FPR local written (through the residency accessors) or reloaded from CpuContext since
        /// <see cref="BeginFprCapture"/>, or null when this is not a recording residency.
        /// </summary>
        public uint? EndFprCapture()
        {
            var captured = _capturing ? _capturedFprs : (uint?)null;
            _capturing = false;
            _capturedFprs = 0;
            return captured;
        }

        private void RecordEvent(SyncEvent syncEvent)
        {
            if (_capturing && syncEvent.Kind != SyncEventKind.Flush)
                _capturedFprs |= (uint)((syncEvent.Mask >> 32) & uint.MaxValue);
            if (_block is null || !_blockEvents.TryGetValue(_block, out var events))
            {
                _eventOutsideBlock = true;
                return;
            }
            events.Add(syncEvent);
        }

        private UInt128 WrittenMask()
        {
            var mask = UInt128.Zero;
            foreach (var gpr in _writtenGprs) mask |= UInt128.One << gpr;
            foreach (var fpr in _writtenFprs) mask |= UInt128.One << (32 + fpr);
            mask |= CrFieldKeys(_writtenCrFields);
            if (_writesCtr) mask |= UInt128.One << CtrKey;
            if (_writesXer) mask |= UInt128.One << XerKey;
            return mask;
        }

        private UInt128 NextFlushDirtySet()
        {
            if (_flushPlan is null) return UInt128.MaxValue;
            var ordinal = _flushOrdinal++;
            if (ordinal >= _flushPlan.DirtyAtFlush.Length ||
                !string.Equals(_flushPlan.BlockAtFlush[ordinal], _block, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException(
                    $"Register residency flush #{ordinal} in block {_block ?? "<none>"} does not match the recorded emission.");
            }
            return _flushPlan.DirtyAtFlush[ordinal];
        }

        /// <summary>
        /// Solves which locals can differ from CpuContext at every flush the recording pass saw. A forward
        /// may-analysis over <paramref name="successors"/>: the entry loads make every local equal CpuContext,
        /// a write makes it dirty, and an unconditional flush or reload makes the synced locals clean again.
        /// Blocks without a predecessor other than the entry start fully dirty, in case the emitter reaches them
        /// some way the CFG does not show. Leaves <paramref name="target"/> without a plan (every flush stores the
        /// whole write set, the old behaviour) when the recording saw an event outside a block or a CFG edge to a
        /// block it did not emit.
        /// </summary>
        public void SolveFlushPlan(RegisterResidency target, Func<string, IReadOnlyList<string>> successors, string entryLabel,
            UInt128 entryDirty = default)
        {
            if (!_recording) throw new InvalidOperationException("Only a recording residency can solve a flush plan.");
            if (_eventOutsideBlock) return;
            var count = _blockOrder.Count;
            var index = new Dictionary<string, int>(count, StringComparer.OrdinalIgnoreCase);
            for (var i = 0; i < count; ++i) index[_blockOrder[i]] = i;
            var predecessors = new List<int>[count];
            for (var i = 0; i < count; ++i) predecessors[i] = new List<int>();
            for (var i = 0; i < count; ++i)
                foreach (var successor in successors(_blockOrder[i]))
                {
                    if (!index.TryGetValue(successor, out var to)) return;  // an edge to an unemitted block: no plan
                    predecessors[to].Add(i);
                }

            var all = WrittenMask();
            var entry = index.TryGetValue(entryLabel, out var entryIndex) ? entryIndex : -1;
            if (entry < 0) return;
            var dirtyIn = new UInt128[count];
            var dirtyOut = new UInt128[count];
            var storedUnion = UInt128.Zero;
            UInt128 Transfer(int block, UInt128 dirty, List<(UInt128 Dirty, string Block)>? flushes)
            {
                foreach (var syncEvent in _blockEvents[_blockOrder[block]])
                {
                    switch (syncEvent.Kind)
                    {
                        case SyncEventKind.Write:
                            dirty |= syncEvent.Mask;
                            break;
                        case SyncEventKind.Flush:
                            if (flushes is not null) storedUnion |= dirty & syncEvent.Mask;
                            flushes?.Add((dirty, _blockOrder[block]));
                            if (syncEvent.Unconditional) dirty &= ~syncEvent.Mask;
                            break;
                        case SyncEventKind.Reload:
                            if (syncEvent.Unconditional) dirty &= ~syncEvent.Mask;
                            break;
                    }
                }
                return dirty;
            }

            for (var i = 0; i < count; ++i)
                dirtyIn[i] = i != entry && predecessors[i].Count == 0 ? all : UInt128.Zero;
            // A register-passing body starts with its arguments only in locals.
            dirtyIn[entry] |= entryDirty & all;
            bool changed;
            do
            {
                changed = false;
                for (var i = 0; i < count; ++i)
                {
                    var incoming = dirtyIn[i];
                    foreach (var predecessor in predecessors[i]) incoming |= dirtyOut[predecessor];
                    var outgoing = Transfer(i, incoming, null);
                    if (incoming != dirtyIn[i] || outgoing != dirtyOut[i])
                    {
                        dirtyIn[i] = incoming;
                        dirtyOut[i] = outgoing;
                        changed = true;
                    }
                }
            } while (changed);

            // Flush ordinals follow emission order, which is block order then event order.
            var plan = new List<(UInt128 Dirty, string Block)>();
            for (var i = 0; i < count; ++i) Transfer(i, dirtyIn[i], plan);
            target._flushPlan = new FlushPlan(
                plan.Select(static item => item.Dirty).ToArray(),
                plan.Select(static item => item.Block).ToArray(),
                storedUnion & all);
        }

        /// <summary>
        /// Reloads the full localized set after a boundary returns, unconditionally: a callee can
        /// change any volatile register and a nonvolatile one too if it restores it itself.
        /// </summary>
        public void AppendReload(StringBuilder sb, string pad) => AppendReload(sb, pad, ResidencyBoundarySync.Full);

        /// <summary>Contract-narrowed reload.</summary>
        public void AppendReload(StringBuilder sb, string pad, ResidencyBoundarySync sync) =>
            AppendReload(sb, pad, sync, unconditional: false);

        /// <summary><paramref name="unconditional"/> as for <see cref="AppendFlush(StringBuilder, string, ResidencyBoundarySync, bool)"/>.</summary>
        public void AppendReload(StringBuilder sb, string pad, ResidencyBoundarySync sync, bool unconditional)
        {
            if (_recording)
            {
                RecordEvent(new SyncEvent(SyncEventKind.Reload, ReloadMask(sync), unconditional));
                return;
            }
            foreach (var gpr in _gprs)
                if (sync.ReloadsGpr(gpr))
                    sb.Append(pad).Append(GprLocal(gpr)).Append(" = ctx->gpr[").Append(gpr).AppendLine("];");
            foreach (var fpr in _fprs)
                if (sync.ReloadsFpr(fpr))
                    sb.Append(pad).Append(FprLocal(fpr)).Append(" = ctx->fpr[").Append(fpr).AppendLine("];");
            // MKW_EABI_REGS: ReloadKeepCrBits' fields the callee preserves in these locals; CpuContext's copy may be stale.
            for (var field = 0; field < 8; ++field)
                if ((_crFields & (1 << field)) != 0 && sync.ReloadsCrField(field))
                    sb.Append(pad).AppendLine($"{CrFieldLocal(field)} = ctx->crv[{field}];");
            if (_ctr && sync.ReloadsCtr) sb.Append(pad).AppendLine($"{CtrLocal} = ctx->ctr;");
            if (_xer && sync.ReloadsXer) sb.Append(pad).AppendLine($"{XerLocal} = ctx->xer;");
        }
    }

    /// <summary>
    /// Which localized registers one call boundary must sync. Flush mask is <c>read-before-write | possible-write</c>, not just callee reads, or an unflushed possibly-written register reloads stale.
    /// Reload mask is the callee's possible-write mask (transitive union, not net effect); safe even for a spill/restore, which just re-reads a value already restored to what this frame flushed.
    /// Falls back to <see cref="Full"/> without a precise contract (missing, full-fence, indirect, or CpuContext-touching marshalling).
    /// </summary>
    internal sealed record ResidencyBoundarySync(
        uint FlushGprMask,
        uint FlushFprMask,
        bool FlushesCr,
        bool FlushesCtr,
        uint ReloadGprMask,
        uint ReloadFprMask,
        bool ReloadsCr,
        bool ReloadsCtr,
        bool FlushesXer = false,
        bool ReloadsXer = false,
        uint ReloadKeepCrBits = 0,
        uint FlushCrBits = uint.MaxValue,
        byte FlushCrFieldMask = 0xFF,
        byte ReloadCrFieldMask = 0xFF)
    {
        /// <summary>
        /// The CR fields this boundary publishes and reloads. CpuContext keeps CR per field (isa/ppc_isa_crv.h), so a
        /// call stores only the fields its callee can read or write, not the whole register.
        /// </summary>
        public byte FlushedCrFields => FlushesCr ? (byte)(FlushCrFieldMask & FieldsOfBits(FlushCrBits)) : (byte)0;
        public byte ReloadedCrFields => ReloadsCr ? (byte)(ReloadCrFieldMask & ~FieldsOfBits(ReloadKeepCrBits)) : (byte)0;
        public bool FlushesCrField(int field) => (FlushedCrFields & (1 << field)) != 0;
        public bool ReloadsCrField(int field) => (ReloadedCrFields & (1 << field)) != 0;

        /// <summary>The fields with any bit in a packed-CR mask (cr0 in the top nibble).</summary>
        private static byte FieldsOfBits(uint bits)
        {
            byte fields = 0;
            for (var field = 0; field < 8; ++field)
                if (((bits >> ((7 - field) * 4)) & 0xFu) != 0) fields |= (byte)(1 << field);
            return fields;
        }

        /// <summary>
        /// MKW_EABI_REGS + MKW_EABI_CR_PARTIAL=1 (GuestEabi.PartialCrFlush; otherwise every bit): the CR bits an EABI-conformant callee (or an indirect target, assumed conformant) reads: cr1
        /// only, the varargs FP-argument flag (GuestEabi.Mask keeps no other CR input). A flush with these bits publishes
        /// just them, so a compare whose CR field only a branch reads need not be materialized for the call.
        /// </summary>
        public static uint EabiVarargCrBits => GuestEabi.PartialCrFlush ? 0x0F000000u : uint.MaxValue;
        public static ResidencyBoundarySync Full { get; } = new(
            uint.MaxValue, uint.MaxValue, true, true,
            uint.MaxValue, uint.MaxValue, true, true,
            FlushesXer: true, ReloadsXer: true);

        /// <summary>
        /// MKW_EABI_REGS: an indirect call. The target reads its arguments (r3-r10, f1-f8, cr1 for varargs) and the
        /// stack/SDA pointers, and returns r3/r4/f1/f2; everything else this body holds survives in its locals.
        /// </summary>
        public static ResidencyBoundarySync EabiUnknownCall { get; } = new(
            FlushGprMask: GuestEabi.InputGprMask,
            FlushFprMask: GuestEabi.ArgumentFprMask,
            FlushesCr: true,
            FlushesCtr: true,
            ReloadGprMask: GuestEabi.OutputGprMask,
            ReloadFprMask: GuestEabi.OutputFprMask,
            ReloadsCr: false,
            ReloadsCtr: false,
            FlushesXer: true,
            ReloadsXer: false,
            FlushCrBits: EabiVarargCrBits);

        /// <summary>
        /// MKW_EABI_REGS: the return of a conformant body. Callers see only r3/r4/f1/f2, and r1 must be back at
        /// its entry value in CpuContext because an earlier flush in this body stored this frame's stack pointer.
        /// </summary>
        public static ResidencyBoundarySync EabiReturn { get; } = new(
            FlushGprMask: GuestEabi.OutputGprMask | (1u << 1),
            FlushFprMask: GuestEabi.OutputFprMask,
            FlushesCr: false,
            FlushesCtr: false,
            ReloadGprMask: 0,
            ReloadFprMask: 0,
            ReloadsCr: false,
            ReloadsCtr: false);

        /// <summary>
        /// MKW_EABI_REGS: the CpuContext side of a register-passing call. The callee starts from CpuContext's r1 (its
        /// own context calls flush only what they wrote), XER (interrupt handlers serviced mid-call read SO from it,
        /// as for <see cref="StateFreeResidentOuter"/>) and, for a varargs callee, CR.
        /// </summary>
        public static ResidencyBoundarySync EabiVariantOuter(bool flushesCr, bool flushesCtr) => new(
            FlushGprMask: 1u << 1,
            FlushFprMask: 0u,
            FlushesCr: flushesCr,
            FlushesCtr: flushesCtr,
            ReloadGprMask: 0u,
            ReloadFprMask: 0u,
            ReloadsCr: false,
            ReloadsCtr: false,
            FlushesXer: true,
            ReloadsXer: false,
            FlushCrBits: EabiVarargCrBits);  // register-passing entries exist for conformant functions only

        /// <summary>MKW_EABI_REGS: the return of a register-passing body: results go back as values, r1 to CpuContext.</summary>
        public static ResidencyBoundarySync EabiVariantReturn { get; } = new(
            FlushGprMask: 1u << 1,
            FlushFprMask: 0u,
            FlushesCr: false,
            FlushesCtr: false,
            ReloadGprMask: 0u,
            ReloadFprMask: 0u,
            ReloadsCr: false,
            ReloadsCtr: false);

        /// <summary>
        /// MKW_EABI_REGS: never reload what the callee preserves. CpuContext's copies of r14-r31, f14-f31 and
        /// cr2-cr4 are stale in this mode (a conformant callee works on its own copies and never writes them back),
        /// so a reload would clobber the caller's live value. <paramref name="callee"/> null: an indirect target,
        /// assumed conformant. A violator's contract already omits the nonvolatiles it provably restores.
        /// </summary>
        public ResidencyBoundarySync WithEabiPreservation(GuestAbiContract? callee)
        {
            if (!GuestEabi.Enabled) return this;
            var conformant = callee is null || callee.EabiConformant;
            var keepGpr = conformant ? GuestEabi.PreservedGprMask : GuestEabi.PreservedGprMask & ~callee!.GprPossibleWriteMask;
            var keepFpr = conformant ? GuestEabi.NonvolatileFprMask : GuestEabi.NonvolatileFprMask & ~callee!.FprPossibleWriteMask;
            var keepCr = conformant || (callee!.CrPossibleWriteMask & GuestEabi.NonvolatileCrFields) == 0
                ? GuestEabi.NonvolatileCrBits
                : 0u;
            return this with
            {
                ReloadGprMask = ReloadGprMask & ~keepGpr,
                ReloadFprMask = ReloadFprMask & ~keepFpr,
                ReloadKeepCrBits = ReloadsCr ? keepCr : 0u
            };
        }

        /// <summary>
        /// Fast arm of a state-free call site (explicit-state ABI, locals carry args/results) still flushes r1 and XER unconditionally, since a guest gather-pipe store can trigger GX HLE mid-call servicing a VI retrace or OS alarm whose handlers read them architecturally.
        /// Flush-only; the slow arm (<c>InvokeDirectCpu</c>) keeps a full <see cref="Full"/> pair.
        /// </summary>
        public static ResidencyBoundarySync StateFreeResidentOuter { get; } = new(
            FlushGprMask: 1u << 1,
            FlushFprMask: 0u,
            FlushesCr: false,
            FlushesCtr: false,
            ReloadGprMask: 0u,
            ReloadFprMask: 0u,
            ReloadsCr: false,
            ReloadsCtr: false,
            FlushesXer: true,
            ReloadsXer: false);

        public bool FlushesGpr(int index) => (FlushGprMask & (1u << index)) != 0;
        public bool FlushesFpr(int index) => (FlushFprMask & (1u << index)) != 0;
        public bool ReloadsGpr(int index) => (ReloadGprMask & (1u << index)) != 0;
        public bool ReloadsFpr(int index) => (ReloadFprMask & (1u << index)) != 0;

        public bool IsFull =>
            FlushGprMask == uint.MaxValue && FlushFprMask == uint.MaxValue && FlushesCr && FlushesCtr &&
            ReloadGprMask == uint.MaxValue && ReloadFprMask == uint.MaxValue && ReloadsCr && ReloadsCtr &&
            FlushesXer && ReloadsXer;

        /// <summary>
        /// Adds a register class back to both masks, for call sites that emit their own architectural
        /// <c>CpuContext</c> traffic (paired-single return normalization) the callee contract omits.
        /// </summary>
        public ResidencyBoundarySync WithFpr(int index) => this with
        {
            FlushFprMask = FlushFprMask | (1u << index),
            ReloadFprMask = ReloadFprMask | (1u << index)
        };

        public bool IsEmpty =>
            FlushGprMask == 0 && FlushFprMask == 0 && !FlushesCr && !FlushesCtr &&
            ReloadGprMask == 0 && ReloadFprMask == 0 && !ReloadsCr && !ReloadsCtr &&
            !FlushesXer && !ReloadsXer;

        /// <summary>
        /// Boundary set for a non-guest helper's implicit <c>CpuContext</c> effects, from
        /// <see cref="Translator.Core.Analysis.GuestHelperEffectCatalog"/>. Explicit argument/destination registers
        /// need no sync; unknown helpers catalog as complete boundaries and arrive as full masks; a pure helper is empty.
        /// </summary>
        public static ResidencyBoundarySync FromHelperEffect(Translator.Core.Analysis.GuestHelperEffect effect) => new(
            FlushGprMask: effect.GprReadMask | effect.GprWriteMask,
            FlushFprMask: effect.FprReadMask | effect.FprWriteMask,
            FlushesCr: (effect.CrReadMask | effect.CrWriteMask) != 0,
            FlushesCtr: effect.ReadsCtr || effect.WritesCtr,
            ReloadGprMask: effect.GprWriteMask,
            ReloadFprMask: effect.FprWriteMask,
            ReloadsCr: effect.CrWriteMask != 0,
            ReloadsCtr: effect.WritesCtr,
            FlushesXer: effect.ReadsXer || effect.WritesXer,
            ReloadsXer: effect.WritesXer,
            FlushCrFieldMask: (byte)(effect.CrReadMask | effect.CrWriteMask),
            ReloadCrFieldMask: effect.CrWriteMask);

        /// <summary>
        /// r1 flushes unconditionally: typed native stubs derive their contract from a host signature and can't see the body re-enter guest code via HLE interrupt delivery, whose callback chains build frames from <c>ctx-&gt;gpr[1]</c>.
        /// A stale r1 once let such a callback overwrite a live frame's save area (return address landed in nw4r::ef's saved r29 at postVRetrace, 0x80210050, as a wild read). Flush-only; caller keeps its stack pointer.
        /// </summary>
        private const uint StackPointerFlushMask = 1u << 1;

        public static ResidencyBoundarySync FromCalleeContract(GuestAbiContract contract)
        {
            if (contract.HasFullSynchronizationFence) return Full;
            return new ResidencyBoundarySync(
                FlushGprMask: contract.GprReadBeforeWriteMask | contract.GprPossibleWriteMask |
                              StackPointerFlushMask,
                FlushFprMask: contract.FprReadBeforeWriteMask | contract.FprPossibleWriteMask,
                // CR per field, as the contract tracks it: the fields the callee may read or write.
                FlushesCr: (contract.CrReadBeforeWriteMask | contract.CrPossibleWriteMask) != 0,
                FlushesCtr: contract.ReadsCtrBeforeWrite || contract.MayWriteCtr,
                ReloadGprMask: contract.GprPossibleWriteMask,
                ReloadFprMask: contract.FprPossibleWriteMask,
                ReloadsCr: contract.CrPossibleWriteMask != 0,
                ReloadsCtr: contract.MayWriteCtr,
                // Flushed unconditionally for the same reason as r1: interrupt handlers read XER.SO out
                // of the architectural copy. Reload stays contract-driven so a handler can't clobber it.
                FlushesXer: true,
                ReloadsXer: contract.MayWriteXer,
                // MKW_EABI_REGS: a conformant callee's CR input is cr1 at most (GuestEabi.Mask).
                FlushCrBits: contract.EabiConformant ? EabiVarargCrBits : uint.MaxValue,
                FlushCrFieldMask: (byte)(contract.CrReadBeforeWriteMask | contract.CrPossibleWriteMask),
                ReloadCrFieldMask: contract.CrPossibleWriteMask);
        }
    }

    /// <summary>
    /// Ambient residency for the body being emitted. Thread-local, not an explicit parameter, since threading one through 30+ static formatting helpers risks a missed path silently reading stale <c>ctx-&gt;gpr[N]</c>; also keeps parallel per-function emission independent.
    /// </summary>
    [ThreadStatic]
    private static RegisterResidency? _activeResidency;

    /// <summary>MKW_EABI_REGS: the body being emitted belongs to an EABI-conformant function.</summary>
    [ThreadStatic]
    private static bool _eabiConformantBody;

    /// <summary>
    /// The flush before a return. A conformant body publishes only r1 and the EABI results; its callers keep
    /// everything else in their own locals. Other bodies publish every written local.
    /// </summary>
    private static void AppendReturnFlush(StringBuilder sb, string pad)
    {
        if (_eabiVariantBody is not null) AppendFlush(sb, pad, ResidencyBoundarySync.EabiVariantReturn);
        else if (_eabiConformantBody) AppendFlush(sb, pad, ResidencyBoundarySync.EabiReturn);
        else AppendFlush(sb, pad);
    }

    /// <summary>MKW_EABI_REGS: the register-passing entry being emitted, if any (see <see cref="GuestEabiVariant"/>).</summary>
    [ThreadStatic]
    private static GuestEabiVariant? _eabiVariantBody;

    /// <summary>MKW_EABI_REGS: every register-passing entry, for this body's direct call sites.</summary>
    [ThreadStatic]
    private static IReadOnlyDictionary<uint, GuestEabiVariant>? _eabiVariants;

    /// <summary>
    /// The statement that leaves the body: <c>return;</c>, or in a register-passing entry its results, from the
    /// locals or (after a tail dispatch that left them in CpuContext) from <paramref name="fromContext"/>.
    /// Order: r3, r4, f1, f2, matching <see cref="EmitEabiVariantCall"/>.
    /// </summary>
    private static string ReturnStatement(bool fromContext = false)
    {
        var variant = _eabiVariantBody;
        if (variant is null || variant.OutputCount == 0) return "return;";
        var locals = !fromContext && _activeResidency is not null;
        var values = new List<string>(2);
        for (var gpr = 3; gpr <= 4; ++gpr)
            if ((variant.OutputGprMask & (1u << gpr)) != 0)
                values.Add(locals ? $"static_cast<uint64_t>({RegisterResidency.GprLocal(gpr)})" : $"static_cast<uint64_t>(ctx->gpr[{gpr}])");
        for (var fpr = 1; fpr <= 2; ++fpr)
            if ((variant.OutputFprMask & (1u << fpr)) != 0)
                values.Add(locals ? $"{RegisterResidency.FprLocal(fpr)}.raw" : $"ctx->fpr[{fpr}].raw");
        return values.Count switch
        {
            1 => $"return {values[0]};",
            2 => $"return MkwEabiPair{{{values[0]}, {values[1]}}};",
            _ => $"return MkwEabiQuad{{{{{string.Join(", ", values)}}}}};"
        };
    }

    private static string EabiVariantReturnType(GuestEabiVariant variant) =>
        variant.OutputCount switch { 0 => "void", 1 => "uint64_t", 2 => "MkwEabiPair", _ => "MkwEabiQuad" };

    private static string EabiVariantPrototype(GuestEabiVariant variant, bool includeNames)
    {
        var parameters = new List<string> { includeNames ? "CpuContext* MKW_RESTRICT ctx" : "CpuContext*" };
        for (var gpr = 0; gpr < 32; ++gpr)
            if ((variant.InputGprMask & (1u << gpr)) != 0) parameters.Add(includeNames ? $"uint32_t eabi_r{gpr}" : "uint32_t");
        for (var fpr = 0; fpr < 32; ++fpr)
            if ((variant.InputFprMask & (1u << fpr)) != 0) parameters.Add(includeNames ? $"PPC_FPR eabi_f{fpr}" : "PPC_FPR");
        return $"extern \"C\" MKW_PPC_NO_INLINE {EabiVariantReturnType(variant)} {variant.Symbol}({string.Join(", ", parameters)})";
    }

    /// <summary>
    /// A direct call through a register-passing entry: the arguments are this body's locals, the results land in
    /// its locals, and nothing is reloaded (the callee preserves everything else per the EABI). CpuContext gets
    /// only what <see cref="ResidencyBoundarySync.EabiVariantOuter"/> publishes.
    /// </summary>
    private static void EmitEabiVariantCall(
        StringBuilder sb, string pad, int bufferBaseLength, GuestEabiVariant variant, GuestAbiContract? callee)
    {
        var residency = _activeResidency ?? throw new InvalidOperationException(
            $"Register-passing call to 0x{variant.Address:X8} without an active residency.");
        AppendFlush(sb, pad, ResidencyBoundarySync.EabiVariantOuter(
            (callee?.CrReadBeforeWriteMask ?? 0) != 0, callee?.ReadsCtrBeforeWrite ?? false), unconditional: true);
        var arguments = new List<string> { "ctx" };
        for (var gpr = 0; gpr < 32; ++gpr)
            if ((variant.InputGprMask & (1u << gpr)) != 0) arguments.Add(residency.Gpr(gpr, written: false));
        for (var fpr = 0; fpr < 32; ++fpr)
            if ((variant.InputFprMask & (1u << fpr)) != 0) arguments.Add(residency.FprStorage(fpr, written: false));
        var call = $"{variant.Symbol}({string.Join(", ", arguments)})";
        if (variant.OutputCount == 0)
        {
            sb.AppendLine($"{pad}{call};");
            return;
        }
        var result = $"eabi_result_{variant.Address:X8}_{bufferBaseLength + sb.Length:X}";
        sb.AppendLine($"{pad}const auto {result} = {call};");
        var index = 0;
        string Next() => variant.OutputCount switch
        {
            1 => result,
            2 => index++ == 0 ? $"{result}.first" : $"{result}.second",
            _ => $"{result}.v[{index++}]"
        };
        for (var gpr = 3; gpr <= 4; ++gpr)
            if ((variant.OutputGprMask & (1u << gpr)) != 0)
                sb.AppendLine($"{pad}{residency.Gpr(gpr, written: true)} = static_cast<uint32_t>({Next()});");
        for (var fpr = 1; fpr <= 2; ++fpr)
            if ((variant.OutputFprMask & (1u << fpr)) != 0)
                sb.AppendLine($"{pad}{residency.FprStorage(fpr, written: true)}.raw = {Next()};");
    }

    /// <summary>
    /// Flush only the locals written since the last sync (<see cref="RegisterResidency.SolveFlushPlan"/>).
    /// MKW_RESIDENCY_DIRTY_FLUSH=0 restores storing the whole write set at every flush, for A/B comparison.
    /// </summary>
    private static readonly bool ResidencyDirtyFlushEnabled =
        Environment.GetEnvironmentVariable("MKW_RESIDENCY_DIRTY_FLUSH") != "0";

    /// <summary>MKW_LEAN_RANGES=0: the earlier range-lowering thresholds (see the call in EmitWithFacts).</summary>
    private static readonly bool LeanRanges = Environment.GetEnvironmentVariable("MKW_LEAN_RANGES") != "0";
    private static void AppendFlush(StringBuilder sb, string pad)
    {
        if (_activeResidency is { } residency) residency.AppendFlush(sb, pad);
    }

    private static void AppendReload(StringBuilder sb, string pad)
    {
        if (_activeResidency is { } residency) residency.AppendReload(sb, pad);
    }

    private static void AppendFlush(StringBuilder sb, string pad, ResidencyBoundarySync sync, bool unconditional = false)
    {
        if (_activeResidency is { } residency) residency.AppendFlush(sb, pad, sync, unconditional);
    }

    private static void AppendReload(StringBuilder sb, string pad, ResidencyBoundarySync sync, bool unconditional = false)
    {
        if (_activeResidency is { } residency) residency.AppendReload(sb, pad, sync, unconditional);
    }

    /// <summary>
    /// Resolves the boundary sync set for a direct guest call, falling back to <see cref="ResidencyBoundarySync.Full"/> when narrowing is disabled, residency is inactive, the callee has no contract or a full-sync fence, or a mod could replace the callee with a re-translation this body's contract doesn't describe.
    /// </summary>
    private static ResidencyBoundarySync ResolveDirectCallBoundarySync(
        uint target,
        IReadOnlyDictionary<uint, GuestAbiContract> guestAbiContracts,
        IReadOnlySet<uint> modOverridableCallTargets)
    {
        guestAbiContracts.TryGetValue(target, out var known);
        if (_activeResidency is null || modOverridableCallTargets.Contains(target))
        {
            return ResidencyBoundarySync.Full.WithEabiPreservation(known);
        }
        if (known is not { } contract)
        {
            return ResidencyBoundarySync.Full.WithEabiPreservation(null);
        }

        return ResidencyBoundarySync.FromCalleeContract(contract).WithEabiPreservation(contract);
    }

    /// <summary>
    /// Removes an explicit helper destination register from a helper-effect
    /// reload mask. The destination local holds the helper's return value; the
    /// implicit-context reload must not overwrite it.
    /// </summary>
    private static ResidencyBoundarySync ExcludeDestinationFromReload(
        ResidencyBoundarySync sync,
        string? destination)
    {
        if (string.IsNullOrWhiteSpace(destination))
        {
            return sync;
        }

        var baseName = GetRegisterBaseName(destination);
        if (baseName.Length > 1 && baseName[0] is 'r' or 'R' &&
            int.TryParse(baseName.AsSpan(1), out var gpr) && gpr is >= 0 and < 32)
        {
            return sync with { ReloadGprMask = sync.ReloadGprMask & ~(1u << gpr) };
        }

        if (baseName.Length > 1 && baseName[0] is 'f' or 'F' &&
            int.TryParse(baseName.AsSpan(1), out var fpr) && fpr is >= 0 and < 32)
        {
            return sync with { ReloadFprMask = sync.ReloadFprMask & ~(1u << fpr) };
        }

        if (baseName.Equals("cr", StringComparison.OrdinalIgnoreCase))
        {
            return sync with { ReloadsCr = false };
        }

        if (baseName.Equals("ctr", StringComparison.OrdinalIgnoreCase))
        {
            return sync with { ReloadsCtr = false };
        }

        if (baseName.Equals("xer", StringComparison.OrdinalIgnoreCase))
        {
            return sync with { ReloadsXer = false };
        }

        return sync;
    }

    /// <summary>
    /// Rejects a residency whose local names would shadow an emitted temporary. A guard against a
    /// future naming change, not an expected condition, since <see cref="IsCpuRegisterNoLocalNeeded"/>
    /// already keeps machine register names out of the ordinary local set.
    /// </summary>
    private static void RequireResidencyNamesAreFree(
        RegisterResidency residency,
        uint entryPoint,
        IEnumerable<string> emittedLocalNames)
    {
        var taken = new HashSet<string>(emittedLocalNames, StringComparer.Ordinal);
        var collisions = residency.DeclaredLocalNames.Where(taken.Contains).ToArray();
        if (collisions.Length != 0)
        {
            throw new InvalidOperationException(
                $"Register residency local name(s) {string.Join(", ", collisions)} collide with an existing " +
                $"local in 0x{entryPoint:X8}.");
        }
    }
}
