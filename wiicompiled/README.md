# Brawl translation and runtime

This directory contains the PowerPC translator, Wii runtime, and Aurora renderer
used by Brawl. Build instructions are in the [repository README](../README.md).

The code derives from [WiiCompiled](https://github.com/patchzyy/Wiicompiled) and its
NSMBW and Super Paper Mario derivatives. Their standalone projects and installers
are not part of this fork. See [LICENSE](LICENSE) and
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for attribution and licenses.

## Native fast paths and mods

The default mod build retains guest code guards and full-context calls. Native replacements must preserve the
translated function's guest-visible registers, memory writes and rollback marks; ABI return values or matching
images alone are insufficient because a mod wrapper can inspect volatile registers and audio buffers.

Native drawing is currently **opt-in** (`G3D_NATIVE=1`). It matches the tested graphics command streams but does
not preserve the complete CPU register contract. Its subtree guards do not fix that distinction. `G3D_NATIVE=2`
now checks the CPU results as well as graphics state and reports these differences; it still does not compare
all guest memory, including dead stack bytes. It is not an arbitrary-mod certification switch.

The incomplete native skinning, CalcView_LC, FVS animation readers and light-slot setter use their translated
bodies. Retained calculation helpers preserve volatile registers and decline unsupported cases. Matrix helpers,
the Gecko handler and exact native AXFX processing remain enabled. AXFX runs synchronously and uses Broadway
floating-point operations by default. `AXFX_FAST=1` and `AXFX_ASYNC=1` explicitly opt into weaker contracts;
`AXFX_EXACT=1` and `BRAWL_HLE_VERIFY=1` override approximate audio processing.

For differential validation, use `G3D_CALC=2 BRAWL_HLE_VERIFY=1 BRAWL_HLE_FUZZ=1000` with recorded replays and
full-snapshot rollback checks. The fuzzers exercise the actual generated matrix, memcpy, camera, environment-matrix and light-setting-copy
bodies, including overlaps, cached/uncached aliases, empty counts and unusual floating-point values. Unsupported
cases may fall back instead of running natively; trial counts are not native-comparison counts. These are
regression checks, not exhaustive proof for every mod. EABI-specialized
translations (`MKW_EABI_REGS=1`) and explicit approximate modes are outside the default full-context contract.
