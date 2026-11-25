# Native default safety validation

Production changes: `9f19a26`. Expanded differential fuzzers: `c66c5f0`.

## Default contract

The full-context mod build keeps code guards and falls back to translated bodies when a native cannot preserve guest-visible state. Native drawing remains opt-in because its complete register and stack-memory contract is not established. Native skinning, CalcView_LC, FVS readers and light-slot setters remain disabled. Approximate/asynchronous AXFX modes are opt-in. See [runtime policy](../wiicompiled/README.md#native-fast-paths-and-mods).

## Correctness evidence

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
