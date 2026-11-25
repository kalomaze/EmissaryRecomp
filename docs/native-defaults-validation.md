# Native default safety validation

Production changes: `9f19a26`. Expanded differential fuzzers: `c66c5f0`.

## Default contract

The full-context mod build keeps code guards and falls back to translated bodies when a native cannot preserve guest-visible state. Native drawing remains opt-in because its complete register and stack-memory contract is not established. Native skinning, CalcView_LC, FVS readers and light-slot setters remain disabled. Approximate/asynchronous AXFX modes are opt-in. See [runtime policy](../wiicompiled/README.md#native-fast-paths-and-mods).

## Correctness evidence

Snapshot verification compares the incremental store against a full-copy reference of the **selected rollback regions**. It does not cover all guest RAM or all host-side runtime state. The repeated-save differences discussed below likewise refer to those selected regions. Gameplay checksums cover the recorded fighter/RNG/input fields; agreement does not establish equality of every guest-visible value.

- Seven native tests pass in each of lean and flat builds, including cached/classic/verification interpreter and native-trampoline tests.
- Safety checkpoint: recordings 001/064/128, 300 output frames each, one redo, on both x86 and ARM; original checksums pass, no reported native differential failures, and all 5,364 snapshot operations verify.
- Expanded fuzzing: five calculation/copy helpers with 1,000 trials each, alongside the four existing PSMTX fuzzers. x86 recording 001 and ARM recordings 001/064/128 pass 120-frame one-redo windows and all 1,416 snapshot operations.
- Fuzz trials include conservative fallbacks. Passing recordings and randomized checks are regression evidence, not exhaustive proof for all possible mods. Platform HLE accuracy and explicit EABI/approximate modes are outside this full-context replacement contract.

## Matched x86 Dolphin comparison

Measured November 25, 2026 on EPYC 9575F, CPU affinity 96–127. Recomp uses lean memory and compiled mods without PGO; Dolphin is a Release NoGUI build of Brawl Online 0.2.8 source `83463a29dd0c53b0301deeee10c96ef915ea2553`, with preserved snapshot/frame-marker hooks and the previously tested one-redo diagnostic hook. Dolphin uses OpenGL and recomp Vulkan.

Twenty-four serial captures: three identical recordings, two repeats per engine/depth, counterbalanced engine order; 1,200 output frames and 960 interior frame-marker intervals per capture. Three excluded Dolphin one-redo pilots warm the shared shader cache. Recomp retains its warm cache. All checksum/work-count/affinity/sample gates pass. Local reanalysis reproduces the remote report exactly.

| Depth | Statistic | Dolphin ms | Recomp ms | Recomp time reduction |
|---|---|---:|---:|---:|
| 0f | mean | 2.096314 | 1.889063 | 9.89% |
| 0f | p75 | 2.155085 | 1.971289 | 8.53% |
| 0f | p90 | 2.402646 | 2.093100 | 12.88% |
| 0f | p95 | 2.759448 | 2.182689 | 20.90% |
| 0f | p99 | 3.624121 | 2.385375 | 34.18% |
| 1f | mean | 4.041519 | 3.966811 | 1.85% |
| 1f | p75 | 4.212268 | 4.137323 | 1.78% |
| 1f | p90 | 4.610995 | 4.343280 | 5.81% |
| 1f | p95 | 4.928042 | 4.464381 | 9.41% |
| 1f | p99 | 5.811273 | 4.717386 | 18.82% |

Values are equal-weight averages of per-run statistics, not pooled percentiles. The aggregate 1f mean lead is small and does not establish a universal win. **Recording 128 at 1f is slower:** Dolphin 3.861890 ms versus recomp 4.035384 ms (4.49% more time). The requirement to outperform Dolphin on every tested workload remains unmet.

Frame markers differ in phase, but intervals include the selected update/save/load workload. These are host throughput results, not display latency, network-play latency, or Switch projections.

Binary SHA-256:

- Dolphin: `502cee6955573dc65e22758d9213545ce72ca03535e2a47008f51637ae843bd0`
- Recomp: `39a1057a77563f0883d0c4fb3007d78e5ad893bacafd3ccb822b65dec1234f02`

Detailed accepted-run statistics and input/sample hashes: [native-defaults-timing.json](native-defaults-timing.json). The recomp timing binary is the safety checkpoint; c66c5f0 changes only opt-in diagnostics and documentation.

## Trampoline dependency regression

The original-body trampoline shortcut must also respect invalidation of inlined callees. Checking the original body's bytes alone is insufficient: replacing a separately located leaf marks its translated callers for interpreter fallback while leaving their body bytes unchanged. The trampoline resolver now checks the generated inline dependencies' patched flags before accepting the native continuation.

A synthetic replacement regression failed in cached, classic and verification interpreter modes before the fix. After the fix, all seven native/interpreter tests pass in both lean and flat x86 builds. The regression checks that the shortcut is declined and that the mod wrapper still executes through interpreter fallback. This covers dependency invalidation delivered through the existing code-patch machinery; it does not establish exhaustive compatibility with arbitrary mods.

Three x86 one-redo replays (001/064/128, 120 output frames each) also pass original recording checksums and native differential checks, with all 1,062 snapshot operations verified. Tested binary SHA-256: `d53de4dd4f2346d1b90fc770f5b217ac8455d952a6d182e26d064a489dfb7dc3`. These are correctness runs, not new performance measurements.

ARM verification of this additional change is pending: the previously used GB10 SSH endpoint refused connections during validation. Earlier ARM results above apply to the earlier safety checkpoint and helper fuzzers.

## Partial-page code restoration regression

Code protection and module relinking previously refreshed pristine bytes for an entire page even when only part of that page belonged to the requested range. That could adopt a neighboring patch as pristine, and a partial-page reset could discard a pending notification for neighboring code. Pristine refreshes now preserve bytes outside the range on already-known pages; partial-page resets retain a pending comparison. Baked-word restoration is also limited to the requested range.

Regression coverage includes an ordinary code write plus `Sync`, rejecting the native trampoline while its inline dependency is patched, restoring that dependency with `ResetPristine`, accepting the trampoline again, and detecting a pending patch to another function on the same page after the reset. All seven tests pass in lean and flat x86 builds. This additional change still requires ARM validation.

Three x86 one-redo recordings (001/064/128, 120 frames each) pass original checksums with no reported native differential failures, and all 1,062 snapshot operations verify. Tested binary SHA-256: `32a8e01f6c02fefbe339ea2ab0a24fd4a1fe009dc9eb1d46e259c27e5fc4216a`. These runs do not establish a performance change.

## Current-head timing confirmation

A fresh matched 24-capture batch tests `6afee6a` using the same host, affinity, recordings, timing windows, renderer backends and warm-up procedure above. All checksums and work-count gates pass; local reanalysis is byte-identical to the remote report. Lean memory and compiled mods are enabled, without PGO.

| Depth | Statistic | Dolphin ms | Recomp ms | Recomp time reduction |
|---|---|---:|---:|---:|
| 0f | mean | 2.086071 | 1.960166 | 6.04% |
| 0f | p99 | 3.770361 | 2.411720 | 36.03% |
| 1f | mean | 4.040294 | 3.990298 | 1.24% |
| 1f | p99 | 6.447443 | 4.717122 | 26.84% |

Recording 128 remains slower at 1f: Dolphin 3.899161 ms, recomp 4.036787 ms (3.53% more time). The aggregate 1f lead remains small; this second batch does not meet the requirement to win every tested recording. Differences between batches do not isolate the cost of the guard fixes. Detailed results: [current-head timing](native-defaults-timing-current.json).

## Sparse snapshot source lookup

Snapshot restore now uses binary search in sorted delta indices when few blocks still need a source, retaining the scan for dense cases. Both paths select the newest saved source at or before the target; unresolved blocks still come from the base image. No guest-state checks or snapshot coverage are removed.

The candidate matches every source decision in a captured 1,197-restore workload. Existing randomized history/eviction tests and a targeted sparse-hit/base-fallback test pass on x86 and as AArch64 binaries under QEMU, in default and lean configurations. Three x86 recordings (001/064/128, 300 frames each, one redo) pass checksums and native differential checks with all 2,682 snapshot operations verified.

In a 16-capture paired experiment on recording 128, mean restore time falls from 122.96 to 101.79 microseconds. Mean 1f frame time is 3.970035 versus 3.897731 ms; 0f is 1.988631 versus 1.997024 ms. Restore cost improves in every candidate repeat, but total frame-time variation exceeds the small aggregate difference. This is evidence for the narrower lookup optimization, not proof of an across-the-board Dolphin win. [Paired evidence](snapshot-lookup-validation.json).

The two preceding guard fixes also pass all seven native/interpreter tests in both lean and flat AArch64 builds under QEMU (14 tests). Native ARM replay and timing validation remains pending because the GB10 endpoint refuses SSH. The emulated results are additional CPU evidence, not a replacement for native validation.

## Trampoline dependency synchronization

A mod can write an inlined dependency and call an original-function trampoline before returning through `RunPatched`, whose synchronization happens after the mod returns. The resolver previously checked stale dependency flags in that interval. It now synchronizes pending code writes after recognizing a registered original target and before acquiring the guard mutex or inspecting dependency flags. Unrecognized trampolines do not pay this synchronization cost.

The regression test marks a dependency write and asks the resolver to reject the shortcut before an explicit caller synchronization. It fails in cached, classic and verifying interpreter modes before the fix, and passes afterward. All seven native/interpreter tests pass in both x86 lean and flat configurations (14 tests). Both AArch64 configurations pass the same seven tests plus snapshot-store tests under QEMU (16 tests). Native ARM validation remains outstanding; emulation does not establish native GPU, replay or timing behavior.

A separate experiment registered the translated drawing body as an original trampoline with native drawing disabled. Its 16 timing runs passed checksum gates and three 300-frame one-redo recordings passed the snapshot replay harness. The experiment did not establish a consistent frame-time benefit, so that registration change is not retained. Those replay runs preceded the synchronization fix and must not be cited as replay validation of this fix.

Post-fix x86 validation of `808aeb2`: recordings 001/064/128 pass 120-frame one-redo windows, with all 1,062 snapshot operations verified, completed fuzz trials and no reported native differential failures. Binary SHA-256: `e84c2a17420c0a8c3fdeeb117f15d3b7cabad6d442d68405883ba47e6de34081`. These runs validate the retained default stack; the trampoline-specific pending-write behavior is covered by the synthetic regression above. [Replay evidence](trampoline-sync-validation.json).

## Fill-memory candidate not retained

An exact-register candidate for `__fill_mem` passed three x86 120-frame one-redo recordings, 1,062 snapshot checks, and 2,000 randomized fill-memory trials per recording. A standalone test using the generated reference body also passed 32,896 full-context and destination/guard-byte cases on x86 and AArch64 under QEMU. Native ARM replay testing remains unavailable.

The 16-run paired recording-128 experiment did not establish a consistent end-to-end benefit: 0f mean changed from 1.965455 to 1.981333 ms, and 1f from 3.986572 to 3.947026 ms, with substantial repeat variability. Most live fill calls still required checked-store fallback. The candidate was removed; the default remains translated fill-memory execution. [Experiment receipt](fillmem-experiment.json).

## Interpreter instruction synchronization

The interpreter previously treated `isync` as a no-op. An interpreted mod could perform a code-store/cache-synchronization sequence and then call a registered translated function while its guard flags still described the old code. A cached interpreter block could also retain decoded instructions after `isync` when the mod had just replaced those instructions.

`isync` now synchronizes pending guest-code writes and ends the decoded block. The next block is resolved and validated after that boundary. Two regressions cover a patched guarded callee and a replacement instruction later in the same straight-line sequence. The CPU harness supplies the executable-write notification explicitly because its platform write hook is stubbed; this test does not validate that hook itself.

Negative controls establish both requirements: removing synchronization fails the guarded-callee test in cached, classic and verifying modes; retaining synchronization but removing the block boundary fails self-modification in cached and verifying modes. With both changes, all seven tests pass in lean and flat x86 builds (14 tests), and all eight tests pass in each AArch64/QEMU configuration (16 tests, including snapshot-store tests). Three x86 120-frame one-redo recordings pass checksums and native differential checks, with all 1,062 snapshot operations verified. Native ARM validation remains pending.

Sixteen paired recording-128 captures of `cb9887a` pass all checksum/work-count gates; local reanalysis reproduces the remote results. Mean frame time changes from 1.952756 to 1.965064 ms at 0f, and from 3.918873 to 3.945213 ms at 1f. These small differences sit within the repeat variability and do not establish either a speedup or unchanged performance. The fix is retained for the proven correctness failures. [Replay and timing evidence](isync-validation.json).

## SDK instruction-cache invalidation

`ICInvalidateRange` and `ICFlashInvalidate` now synchronize pending code writes instead of sharing an empty cache-maintenance stub. This lets a mod's subsequent translated/native call observe updated code guards at the SDK invalidation boundary. Cache enable/disable stubs retain their existing behavior.

The full x86 build and three 120-frame one-redo recordings pass, with all 1,062 snapshot operations verified and no reported native differential failures. These recordings exercise the integrated default stack, not every possible cache-invalidation call sequence. [Validation receipt](icache-hle-validation.json). Native ARM validation remains pending. The compiled inline `isync` gap is addressed separately below.

## Compiled instruction synchronization

A translated function could execute a stale host body after an inline `isync`, even if entry guards were updated. Default full-context functions containing `isync` now interpret from their entry, so the interpreter synchronization and decoded-block boundary apply before subsequent guest instructions execute. Removed static call and interior-label facts prevent callers from treating the replaced body as emitted code. Explicit EABI/state-free modes retain their separate weaker contract.

Both new generator regressions (base and mod registration) fail before the change and pass after it. The related generator/shard/decoder selection has 237 passing, 11 failing and 4 skipped tests; the 11 failures also occur on the parent implementation. The production build was regenerated with the normal build tool. Three x86 120-frame one-redo recordings pass checksums with 1,062 verified snapshot operations. These bounded P+ runs are integrated regression evidence, not direct coverage of every self-modifying mod. Native ARM execution and foreign-mod validation remain pending. [Validation receipt](inline-isync-validation.json).

The paired 16-run recording-128 timing comparison passes checksum/work-count gates and reproduces under local reanalysis: 0f mean 1.964185 to 1.968256 ms; one-redo mean 3.958097 to 3.968097 ms. These differences are within repeat variability (run-mean standard deviations 0.027–0.036 ms); this does not establish unchanged performance or cover other recordings.

## Code guards after snapshot restoration

Gameplay rollback and the online-core snapshot callback now synchronize pending executable writes after restoring guest bytes, before resimulation resumes. Ordinary save states already had this boundary. Without it, a host restore can replace a guarded callee's instructions while its translated entry still sees the old guard state.

A focused host-copy regression fails without synchronization in cached, classic and verifying interpreter modes and passes with it. The CPU harness explicitly supplies the executable-write notification because its platform hook is stubbed. All 14 x86 lean/flat checks and 16 AArch64/QEMU checks pass. The combined production build passes three 120-frame one-redo recordings and 1,062 snapshot checks. These recordings cover gameplay rollback, not direct execution of the online-core callback. This fix synchronizes restored instruction bytes; it does not establish restoration of every host-side module or patch metadata structure. Native ARM and REX gameplay remain pending. [Validation receipt](restore-code-guard-validation.json).


### Gecko exit synchronization and REX

The native Gecko handler previously restored registers in C++ and returned without executing the guest exit's `isync` at `0x80001984`. It now resumes the interpreter at `0x80001944`, preserving the exit instructions and synchronizing code written by the pass before its caller resumes. This also allows the exit itself to reflect guest edits.

A focused regression writes a guarded callee through a Gecko 04 code and immediately calls it. The old exit fails in cached, classic and verifying interpreter modes; the fixed exit passes those modes with both lean and flat memory on x86 and AArch64 under QEMU. The CPU harness explicitly supplies the platform's pending executable-write notification.

REX Release 0½ Hotfix 1 boots into gameplay. With this fix, a 300-output-frame one-redo run completed 297 resimulations and verified all 894 snapshot save/load operations. The last Gecko report records 2,400 comparisons with zero mismatches; no native/reference mismatch was reported by the enabled differential checks. Full-memory replay differences persist in 297 repeated saves, even though the earlier 0f/1f runs agreed on all recorded gameplay, fighter, RNG and input fields. These memory differences remain unresolved, so this is not a claim of complete REX rollback determinism. Native ARM remains pending; QEMU results are CPU-level evidence only. [Validation receipt](gecko-exit-validation.json).

The same production binary also passes the 120-frame one-redo replay checks for Project+ recordings 001, 064 and 128, with all 1,062 snapshot operations verified and no reported native/reference mismatch.


### Gecko continuation guard coverage

A synthetic guest callback demonstrates an additional edge case: a C0 code can rewrite and synchronize the Gecko handler while its native implementation is running. The native continuation previously ignored those edits. This was not an observed REX or Project+ failure. The production guard table also splits the handler at the separately translated helper `0x80001E88`; checking only its entry interval misses changes to later handler code.

The native body now checks every guard interval intersecting its implementation (`0x800018A8..0x800021B0`) at entry and after either guest call. If the handler changed, the entry uses the original body or the active call resumes at its exact guest continuation in the interpreter. Unchanged handlers retain the native path.

The fixture reflects the production guard split and verifies both a modified continuation and fallback on the next entry. It fails against the old handler in cached/classic/verifying modes, then passes all 12 checks across x86 and AArch64/QEMU, lean and flat. A REX 300-frame one-redo differential run verifies all 894 snapshot operations with no reported native/reference mismatch. Native ARM remains pending.

A separate REX control with native Gecko disabled still reports 297 full-memory differences. Its 597 trace rows exactly match the native run's recorded gameplay, fighter, RNG and input fields. That excludes native Gecko as the sole explanation, but does not clear other native paths or establish full-memory determinism. [Validation receipt](gecko-continuation-validation.json).

The same binary passes Project+ recordings 001/064/128 for 120 output frames with one redo, including all 1,062 snapshot operations and no reported native differential failures.


### Timing after both Gecko fixes

A paired 16-capture batch compares `4bfba01` (before both Gecko changes) with `f6a159d` (interpreted exit plus continuation guards), using recording 128, lean memory and compiled mods without PGO on the same x86 host. Four runs per build/depth use counterbalanced order and 960 interior marker intervals each. All checksum/work-count gates pass; local reanalysis exactly reproduces all 16 reports.

| Depth | Mean before / after (ms) | p99 before / after (ms) | Run-mean SD before / after (ms) |
|---|---:|---:|---:|
| 0f | 1.952174 / 1.956363 | 2.428352 / 2.416683 | 0.052221 / 0.038752 |
| 1f | 3.900894 / 3.926366 | 4.731870 / 4.734326 | 0.152292 / 0.115571 |

The measured mean differences are +0.004189 ms (+0.21%) and +0.025471 ms (+0.65%). They are small relative to between-run variation, but this batch does not establish zero overhead. Percentiles are averages of per-run percentiles. This covers one recording and is not a new Dolphin, Switch/Horizon or native ARM comparison. [Run statistics and identities](gecko-safety-timing.json).


### REX control with handwritten helper families disabled

A further 300-output-frame REX one-redo control disables the native Gecko, matrix, tangent, G3D and AXFX helper families. All 597 trace rows match the native comparator on the recorded gameplay, fighter, RNG and input fields; all 894 snapshot operations verify. Both runs still report 297 full-memory differences. This control does not disable compiled mod variants, cached interpretation, lean memory or the platform HLE, and does not establish full-memory determinism. The native comparator also enabled render attribution instrumentation; these are correctness observations, not timing results.

This short, limited-input matchup is basic compatibility evidence. No REX-triggered native optimization fault has been established; the instruction-synchronization and continuation failures above were demonstrated by synthetic fixtures. Broader real combat and roster/stage coverage remain outstanding. [Control receipt](rex-reference-helpers-validation.json).


### Longer REX combat window

A 1,200-output-frame run adds signed-stick movement, A/B attacks, jumps, shielding, crouching and reversals using the actual `gfPadStatus` layout. The trace confirms movement, damage reaching about 24.96%, and a stock loss; this is more than a stationary compatibility check. It still covers only the existing matchup/stage.

The corresponding one-redo run enables native/reference helper verification. All 2,397 trace rows (including resimulations) match their corresponding 0f frames on gameplay, fighter, RNG and input fields. All 4,794 snapshot save/load operations across both runs verify, and enabled helper checks report no nonzero mismatch counters. The runtime binary is unchanged from `f6a159d`. Resource-open warnings also remain in the log; this is not a claim of complete audiovisual correctness.

Full-memory differences remain in all 1,197 repeated saves. Successful snapshot restoration and matching gameplay fields do not resolve those differences. This is stronger real-combat compatibility evidence, not complete rollback determinism, broad roster coverage, a timing comparison or native ARM validation. [Combat receipt](rex-combat-validation.json).


### Additional REX fighters and stage

Mewtwo versus Master Chief on Dracula's Castle completes a second 1,200-frame combat window at 0f and with one redo. Selection/gameplay screenshots and loaded `Ft_Mewtwo.rel`, `ft_chief.rel` and `st_mansion.rel` assets confirm the changed matchup and stage. The trace records damage on both fighters and a stock loss. An earlier alternate menu sequence failed to reach stage selection and is excluded from gameplay evidence.

All 2,397 rollback trace rows match the corresponding 0f frames on recorded gameplay, fighter, RNG and input fields. All 4,794 snapshot operations verify against the full-copy reference of the selected regions. Enabled native/reference checks report no mismatch; the receipt includes their last reported counters. The runtime remains `f6a159d`.

All 1,197 repeated saves still differ somewhere within the selected regions. These checks establish additional real-mod compatibility, not complete state determinism, audiovisual correctness or arbitrary-mod coverage. Current native ARM validation remains outstanding. [Roster/stage receipt](rex-roster-validation.json).


### Third REX matchup

Sceptile versus Nu-13 on Midgar completes 1,200 output frames at 0f and with one redo. Screenshots identify both fighters and the stage; loaded `ft_sceptile.rel`, `ft_nu.rel`, `st_palutena.rel` and `Midgar.param` assets support that identification. This window records movement, action changes and a stock loss, but no damage exchanges, so it is not additional evidence of damage handling.

All 2,397 rollback trace rows match their corresponding 0f frames on recorded gameplay, fighter, RNG and input fields. All 4,794 snapshot operations verify for the selected rollback regions. Enabled helper comparisons report no mismatches, and the log contains no reported out-of-declared-range verification writes or nonzero unverifiable counters. All 1,197 repeated saves still have region differences. These unresolved differences and pending native ARM validation limit the result. [Third-match receipt](rex-roster-c-validation.json).


### Connected REX Dolphin compatibility

A real connected gameplay session now pairs Brawl Online Dolphin 0.2.8 with the recomp runtime from `f6a159d`, using REX Release 0½ Hotfix 1, Ivysaur versus Sora on Final Destination. The harness performs initial selection, substage agreement and the existing setup synchronization, then passes gameplay packets through a bridge with 60 ms delay each way. This exercises the actual Gekko gameplay session, not production matchmaking. Dolphin uses the standalone materialized REX SD image; the recomp's sparse virtual-backed SD file cannot be used directly by Dolphin.

The accepted combat run compares all 1,200 corrected frames with no missing or mismatched frames and no reported client desyncs. Dolphin performs 160 rollbacks and 645 resimulated frames; recomp performs 235 loads and 2,149 total advances. Both fighters move and change actions; recorded damage reaches 37.36% and 37.00%, respectively. Neither loses a stock. Dolphin receives real controller inputs that approach the opponent using observed positions; the native player uses a recorded input tape. Consumed Dolphin pad records and controller decisions are retained. No gameplay memory is edited after setup synchronization.

Two earlier runs match 600 and 1,200 corrected frames, respectively, with rollbacks on both clients, but neither includes damage exchanges. They are movement/action parity evidence only; the longer run explicitly fails the combat-coverage gate. Earlier attempts with an idle Dolphin player are excluded from the accepted both-client rollback evidence.

Comparison covers three RNG words, each fighter's instance/damage/stocks/position/action fields, and the derived gameplay CRC. Client desync counters also remain zero. Full-region hashing is disabled for this connected test, so this does not resolve the selected-region repeated-save differences above or establish equality of all guest-visible state. It validates one connected x86 matchup, not arbitrary mods, audiovisual equivalence, native ARM, matchmaking, or performance. [Connected-run receipt and artifact hashes](rex-connected-validation.json).
