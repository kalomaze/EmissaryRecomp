# PoyoJIT: the JIT for untranslated and patched guest code

Guest code without a usable translation (Gecko code bodies, plugin code, replaced modules, and every translated
function a mod patches) ran in the cached interpreter (`wiicompiled/runtime/src/ppc_interpreter.cpp`). The JIT compiles
that same code to host code at run time, for any mod, with no build-time knowledge of the mod. The cached interpreter
stays: it is the reference the JIT is tested against, and the engine wherever the JIT is not built or is turned off.

## Rules

1. **Same results, bit for bit.** A JIT run leaves the registers, FPR forms and guest memory the interpreter leaves.
   Replays must keep matching their checksums. Floating point reuses the runtime's own helpers
   (`isa/ppc_isa_float.h`, `isa/ppc_isa_quantized.h`); an operation is inlined only with a test that shows the same
   bits for every input class (NaNs, subnormals, NI mode).
2. **Nothing about the code is assumed.** A compiled block runs only while guest memory holds the words it was
   compiled from. The JIT reuses the block cache's check (`Block::epoch`, the dirty marks under its code) and never
   adds a second invalidation scheme: a link slot is aimed at a block only while the block cache holds it current in the
   current epoch, so a linked entry tests only the marks (below).
3. **The interpreter is the fallback for everything.** Any op the backend has no native code for is compiled as a call
   to the interpreter's handler for it.
4. **No C++ exception and no `longjmp` unwinds JIT frames by table.** Every C++ function JIT code calls is a
   `noexcept` shim that catches, stores the exception in the frame and makes the block exit; the dispatcher rethrows.
5. **Code is never freed while it may be on a stack.** JIT code lives and dies with the `BlockCache` that owns its
   blocks (`BlockCache::Clear`, only with no `Run` active on the thread).

## Scope

- Host: x86-64 with lean memory (`MKW_LEAN_MEMORY=1`), Linux and Windows. CMake option `MKW_JIT`, on by default on
  both. Linux is tested natively and on the rollback benchmark. Windows (Win64 calling convention, `VirtualAlloc`) has
  only run as the test executables cross-built with llvm-mingw under Wine, which all pass; it has not run in the game
  or on Windows itself. There is no AArch64 backend; Horizon has no executable memory. Those hosts keep the
  interpreter.
- Run time: `BRAWL_JIT=0` turns it off. `BRAWL_INTERP=classic`, `BRAWL_INTERP_VERIFY`, `BRAWL_INTERP_STATS`,
  `BRAWL_TRACE_PC` and the profile build keep the interpreter, as they need every op dispatched.

## Shape

```
PpcInterpreter::Run (dispatcher, C++)
  hook pcs, rollback loop frame, logging        unchanged
  cache->Find(pc)  -> Block (decoded ops, validity data)   unchanged
  block has code?  -> the backend's enter(frame, block): runs it and every block it is linked to
                      returns {status, next pc, link slot}
  else             -> RunBlock (interpreter)
```

A compiled block is entered by `jmp` and leaves by `jmp`: to a linked block, or to a common exit stub that returns
from the enter trampoline. Blocks share one host frame, made by the trampoline, except that a block may call a patched
function's block with a host `call` (below, "Calls from compiled code"). A block is compiled once it has been entered
`SetJitThreshold` times (two by default).

### Registers (x86-64)

| Register | Holds |
|---|---|
| `rbx` | `CpuContext*` plus 0x80, so the GPRs, the eight CR field values, lr, ctr, xer, fpscr, pc and the first four FPRs have one-byte displacements |
| `r12` | lean window base (`g_leanBase`); dirty marks at `r12 + kLeanDirtyOffset`, store policy at `r12 + kLeanPolicyOffset` |
| `r13` | `Frame*` (FPR forms, return address, entry, pending exception) |
| `r14` | the address of `BrawlDirtyMemory::markEpoch`, for the entry check |
| `r15` | guest instructions run since the trampoline was entered (added to `Frame::executed` on the way out) |
| `rbp`, `r8`-`r11` | the GPR cache within a block (guest GPRs, zero-extended to 64 bits; `rbp` callee-saved) |
| `xmm6`-`xmm15` (SysV), `xmm4`-`xmm5` (Win64) | the FPR cache within a block (a guest FPR's 64 raw bits) |
| `rax`, `rcx`, `rdx`, `rsi`, `rdi`, `xmm0`-`xmm3` | scratch (emitted code leaves `xmm4`-`xmm5` alone on both ABIs) |

`rsp` is 16-byte aligned inside blocks. On Linux blocks run at *block level*, 16 bytes below the trampoline's frame:
`[rsp]` is a return address into whatever entered that level (the trampoline enters the first block with `push; call`, a
calling block with `push; call`) and `[rsp+8]` the return slot (a calling block's saved Frame word); one DWARF FDE per code
chunk describes every block (CFA = rsp + 16, return address at CFA - 16), so C++ exceptions unwind through blocks and host
calls between them. On Windows the trampoline's frame holds the 32-byte shadow space and blocks run at its level. Helper calls
go through one emitter routine that hides the SysV / Win64 argument registers. Guest registers live in memory between
blocks; within a block the two caches may hold them (stored before anything that calls or leaves the block, `jit_x64.h`).

### A block's code

```
hot code (one part of each code chunk, blocks one after another):
linked:    the block's dirty marks clear (but its guarded lines')      else -> stale (far)
entry:     r15 += count
           op 0 ... op n-1           native, or a call to the fallback; slow paths jump to far code
exits:     static target -> jmp rel32, patched to the target's `linked` (or its unlinked stub)
           dynamic target -> an inline cache keyed by guest pc, else the exit stub
           hook pc -> exit stub
           the block's own start, when no op before can write memory or run C++ -> entry (no test)
far code (another part):
checked:   epoch == markEpoch                                          else -> plain stale exit; then jmp linked
stale:     through the stale cell: counted, and past a threshold a stub that compares the block's words in place
plain:     exit {Continue, start, no link}
           the fallback exit, the ops' cold paths, inline-cache misses, unlinked stubs
data (a third part): link slots, the stale cell, inline caches, call cells (written at run time)
```

- Links and inline caches go to `linked`; entries from outside compiled code (the dispatcher's fast entries, `jit_subcall`) to
  `checked`; the dispatcher, having run `Find`, enters at `entry`.
- A slot is aimed at a block only while its epoch is the current one (`Block::jitLinked`). Whatever drops a block (epoch 0:
  rollback's consume and restore notifications, a mark cleared under it) suspends its incoming slots first (they leave through
  their unlinked stubs, staying on its list); the refresh that finds its words unchanged aims them again. When
  `markEpoch` moves (memory rewritten without marks), every slot is suspended before compiled code can follow one: the
  interpreter does it when it first validates a block in the new epoch, and every way compiled code goes on after C++ ran
  either tests its block's epoch (`EmitRecheck`, after a call), compares the link epoch (`ExitDynamicAfterCall`, after a tail
  call), or calls `JitNoteEpoch` before returning (the fallback, the store shims).
- After a call that returns into the block, the block's own check (epoch and marks) runs again; failing it goes on at the
  next pc through a link slot to the block there while the link epoch is the current one, else through the dispatcher.
- Guarded lines: a block's 64-byte lines that lie in 4 KiB pages registered as executable (RecompMod's write guard; the store
  policy flags their 64 KiB chunks, so every guest store there takes the checked path) are not tested, neither on the linked
  entry nor after a call (`JitGuardedLine`). The block cache puts them in two tables (`BrawlDirtyMemory::guardChunks`, a byte per
  64 KiB chunk, and `guardLines`, a bit per mark), and `BrawlDirtyMemory::Mark` and `LeanMarkRange`, which every write that marks
  such a line goes through (the checked store path, the natives that mark; the natives that write marks themselves test the
  store policy first, a translated stack store is never in an executable page), call its hook for a marked guarded line: it
  drops every guarded block with code there (`BlockCache::GuardedWritten`: epoch 0, slots suspended) before compiled code goes
  on. A block whose lines are all guarded (most of the game's own patched functions) has no test on its linked entry and only
  the epoch after a call. A start dropped that way four times is built again with every line tested (`BlockCache::unguarded`).
  Executable ranges are only ever added, so a line stays guarded. `BRAWL_JIT_GUARD=0` tests every line.
- A link is a data slot (`jmp [slot]`), first aimed at its own unlinked stub, which exits with the slot's address. The
  dispatcher finds the target block and stores its `linked` address in the slot. A block that is replaced resets its
  incoming slots. A target that is a hook pc is never linked.
- The three parts keep hot code dense (no cold path, stub or data between two blocks' main paths) and keep every line written
  at run time (slots, cells, counters) out of the lines that execute; a block that does not fit the room left in a part starts
  a new chunk.
- Exit status: `Continue` (next pc), `Return` (the frame's function returned), `Pending` (an exception is stored in
  the frame).
- A block whose dirty marks stay set though its words do not change (a mod's variable inside a function's code) gets,
  after repeated failures of its check, a stub that compares its words in place and enters it when they are equal.

### Condition register values

`CpuContext::crv[8]` holds one 64-bit value per CR field (`isa/ppc_isa_crv.h`: LT is bit 62, GT is a positive value, EQ a
zero low word, SO bit 59 differing from bit 62). The JIT stores exactly the value the interpreter stores, so a block
leaves the same bytes in the context whichever engine ran it (a write the liveness analysis skips is one that a later write
of the field in the same block overwrites):

- a compare stores the 64-bit difference of its operands, sign- or zero-extended (`movsxd` or a 32-bit load, `sub`, one
  8-byte store); a record form is the same with the result against 0. The interpreter copies a set XER[SO] into the value
  (`PpcCrvWithSo`); compiled code never runs while XER[SO] is set (`JitBlocksMayRun`: the dispatcher interprets until it is
  clear), so its values need no copy. What can set it inside compiled code leaves for the dispatcher when it did: a call
  into guest code (`EmitRecheck`), a fallback op, `mtxer`, a tail call that goes on at lr (`BlockEmitter::JumpIfSo`);
- `fcmpu` stores one of the four constants (`kPpcCrvLt`, `kPpcCrvGt`, `kPpcCrvEq`, `kPpcCrvSo`) chosen by `cmov`;
- a `bc` tests one field value in memory: LT is `test byte [field + 7], 0x40`, GT `cmp qword [field], 0` and a signed
  condition, EQ `cmp dword [field], 0`, SO bit 59 xor bit 62 (a shift, an xor and `bt`);
- a fused compare and `bc` branches on the host flags of the compare's own 64-bit difference (its signed conditions are
  the compare's for both signednesses); the value and every dirty GPR slot are stored before the branch, so that the taken
  path is one `jcc rel32` patched to the target block like a `jmp` link (`ExitLinkedIf`);
- `mcrf` copies the 8-byte value, `mtcrf` stores `kPpcCrvTable[nibble]` for each field its mask names, `stwcx.` stores
  the value of nibble 2 or 3 by XER[SO]; `mfcr` and the CR logic ops call `noexcept` helpers that run the interpreter's own
  code (`PpcGetCr`, `PpcCrvWithBit`, `PpcCrvLogic`).

A NaN result of a float op is the one the interpreter's own code computes: with `MKW_ACCURATE_NANS` off the host picks
which operand's NaN a result is, and that depends on the operand order the compiler gave each copy of the expression, so
native code and the helpers it calls send any NaN result to the interpreter's handler.

### Fallback

`JitFallbackOp(frame, block, index)` runs one op exactly as `RunBlock` does (`RunOp`, then `Exec` for
`Act::Exec`) and returns what to do next. Every decoded op kind and the recurring `Generic` instructions have native
code; what is left (rare instruction forms, overflow-enable forms) takes this path.

### Analyses per block

- Condition register (`jit_x64_cr.cpp`): a compare or record form whose field is overwritten before anything can read
  it or leave the block writes nothing; a fused compare and branch tests the host flags of the compare and writes the
  field only where it is needed. Every kind that reads or writes fields is classified: a compare, a record form and
  `fcmpu` overwrite one field in full; `bc`, `bclr`, `mfcr` (all eight fields), `mcrf`, `mtcrf` and the CR logic ops (which
  read up to two fields and rewrite one from its own old value) keep every field live, as do the ops that may leave the
  block or throw.
- FPR forms (`jit_x64_forms.cpp`): `Frame::form[]` says whether a register holds a double or a pair. Within a block the
  form is known after a register's first write or tested use; native float code tests the byte only when it is not
  known, and converts through the same instructions the interpreter's helpers use.
- FPR cache (`jit_x64_fpr.cpp`): float ops read and write guest FPRs in xmm slots within a block, and keep the form bits they
  write (`Frame::pairedMask`) pending until the next flush. Every slot is volatile: a call or exit stores and empties the
  cache, a `bc` stores the dirty slots and pending forms before it branches (a record form paired with its `bc` stores them
  before its flags, which the form store would clobber), and the snapshots slow paths store and reload carry it.
- Store groups (`jit_x64_mem.cpp`): integer D-form stores off one base value (followed through `addi`, `mr` and update
  forms) with no call between them share one store-policy test and one set of dirty marks, made by the first of them for
  the bytes all of them write (at most 129); the others are the bare byte-swapped store. A leader whose range needs the
  checked path stores through it and either goes on (the range is plain RAM) or leaves the block before the others.
- Loads (and grouped stores) on Linux use the aliased view of the lean window (`lean_window.h`): `[r12 + base +
  disp]` with no mask, the window being mapped five times plus its first 64 KiB once more past 5 GiB.
- Stores on Linux go through the store view (`lean_window.h`, `g_storeView`): `[r12 + 1 GiB + base + disp]`, whose copy for
  the cached view is writable exactly where the store policy is clear (`memory.cpp` keeps the protection in step with the
  policy) and whose other copies are read-only. A store has no policy test: where the policy would send it to the checked
  path it faults, and the fault handler (`jit_x64_mem.cpp`) resumes it at its slow entry in the op's cold code (the policy
  test, then the store through the window or the checked path's routine) and patches the store into a jump there. Loads use
  the same copies. A group's leader tests no range: its members fault for themselves. Marks of a store that went through are
  made from its unmasked address. `BRAWL_LEAN_STORE_VIEW=0` (and every other host) keeps the policy test in line.

## Calls from compiled code

- A `bl`, `bctrl`, `b` or `bctr` to a translated function calls its entry directly when the record needs no guard, else
  goes through `JitThinCall` / `JitThinTail` with a per-target `JitCallTarget` (the record's entry and guards, the float ABI),
  settled when the block is compiled. A call site that knows its callee tests the callee's float argument registers in line.
  `bctrl` and `bctr` keep their last eight targets in a cell of the block's data (`kCellWays`), each with what the site calls
  for it (`JitFastCallFor`: the entry, or a shared routine); an entry hit often moves to the front. Untranslated targets of
  either go through the block's inline cache 1. The rollback loop frame always takes the general path (`Call`).
- A call to a patched function (a translated function whose guard byte is set, which `GuestCodePatch::RunPatched` runs in
  a `Run` of its own) from a `bl`, while the guard byte stays set and no redirect applies (Linux), is a host `call` of the
  callee's block in the caller's `Frame`. The nested `Run`'s per-call state is only `pairedMask` (0) and `returnAddress`
  (pc + 4), side by side in `Frame`: the caller pushes its pair (the return slot), stores the callee's with bit 0 of
  `returnAddress` set (the tag; every reader masks the field with ~3), and calls through a *call link* (`CallLinked`: a
  `call rel32` patched like a jump link to the callee block's linked entry). The callee's `blr` compares lr & ~3 with the
  field: 0 is the frame's own function's return (the exit), 1 the tagged return, which puts the caller's pair back and
  returns with `ret` (predicted by the return stack; the caller then drops the slot: `ret 8` measured slower); anything else is a jump. Then the caller makes `RunPatched`'s `Sync`
  test and its validity test. A callee whose code jumps away from its return address keeps running with its own pair, as in
  the nested `Run`, until a `blr` to that address. An exit to the dispatcher from inside such a callee (a stale block, an
  unlinked call link, a cache miss, an exception) goes through `lift` (`JitLiftContinue`): the callee gets a `Frame` of its
  own from the shared state and the dispatcher runs it (`JitDirectContinue`) until it returns; then the tagged return. A
  `bctrl` does the same through `jit_subcall` (fast-entry table). Otherwise the wrapper runs `RunPatched`.
- `bctrl` and `bctr` sites whose cells miss look in the code cache's table of translated targets (`TargetTable`, one for
  calls and one for tail calls, filled from the cells) before their shims. A `bctr` after `lwzx; mtctr` (a switch's table)
  probes the inline cache first. `ctx->pc` (a diagnostic) is not stored before direct calls of translated code.
- `GuestCodePatch::RunPatched` enters compiled code through `RunCompiledEntry`, which does what `Run` does for a first
  block that has code, is current and is no hook pc, and hands anything else to the general entry. With code variants off
  it tail-calls `RunPatchedJit`, which makes the redirect test, `RunCompiledEntry`'s fast path and the `Sync` test in one
  function. `Run` itself tries `RunCompiledEntry` first, so translated code's calls of untranslated code (a mod's module,
  through `RunUntranslatedOrFail`) enter through `jit_enter_direct` too.
- Linux x86-64: code space is mapped within a rel32 displacement of the executable's text (a reservation at the top of the
  reachable window), so calls of translated functions and runtime helpers are `call rel32`.

## Switches

Read once, at first use.

| Variable | Effect |
|---|---|
| `BRAWL_JIT=0` | turn the JIT off; the cached interpreter runs everything |
| `BRAWL_JIT_STATS=1` | counts at intervals and at cache destruction; the inline cache's hit and miss counters exist only with it |
| `BRAWL_JIT_PERF_MAP=1` | names compiled blocks for `perf` in `/tmp/perf-<pid>.map` |
| `BRAWL_JIT_DUMP=<hex pc,...>` | a block's code bytes to `jit_<pc>.bin` |
| `BRAWL_JIT_FALLBACK=<list>` | ops to the interpreter's handler: `int`, `mem`, `branch`, `fp`, a kind name or `all` (`..._forms_fallback_tests`) |
| `BRAWL_JIT_GENERIC_STATS=1` | which `Generic` words still fall back |
| `BRAWL_JIT_CODE_LIMIT=<bytes>` | the code cache is dropped at the next outermost `Run` once it holds more (the core, forms and stale tests set it) |
| `BRAWL_JIT_NO_HUGEPAGES=1` | code chunks without the huge-page advice |
| `BRAWL_HUGE_TEXT=0` | the executable's text stays on small pages |
| `BRAWL_JIT_GENERIC=0` | `Generic` words always take the fallback call (`mkw_jit_generic_fallback_tests`) |
| `BRAWL_JIT_FP_HELPERS=1` | every float arithmetic kind calls its helper (`mkw_jit_fp_helper_tests`, `mkw_jit_forms_helper_tests`) |
| `BRAWL_JIT_FP_COUNTERS=1` | per-kind counters of what the float code did (`jit_fp_tests` sets it) |
| `BRAWL_JIT_NO_FORMS=1` | the FPR form analysis off: every form unknown (a diagnostic; `jit_forms_tests` passes with it, no ctest entry runs it) |
| `BRAWL_JIT_NO_CR_LIVENESS=1` | every condition register field live (a diagnostic; `jit_quality_tests` passes with it, no ctest entry runs it) |
| `BRAWL_JIT_NO_FUSED_FLAGS=1` | fused compare and bc write the field and branch on it (`mkw_jit_quality_bookkeeping_tests`) |
| `BRAWL_JIT_NO_FUSED_FCMP=1` | an `fcmpu` always writes its field and the `bc` after it reads the field in memory |
| `BRAWL_JIT_NO_FPR_CACHE=1` | no FPR cache: every float op loads and stores its registers (the pending forms stay) |
| `BRAWL_LEAN_ALIAS=0` | the lean window without its aliased view (Linux): loads and grouped stores mask their address |
| `BRAWL_LEAN_STORE_VIEW=0` | no store view (Linux): stores test the store policy in line (`mkw_jit_mem_policy_tests`) |
| `BRAWL_JIT_CHURN=<file>`, `BRAWL_JIT_PROBE=<pc,...>` | block cache and JIT churn counters (only in a build with `MKW_JIT_CHURN=1`); the registers logged before the op at each pc (`ppc_interpreter_diag.h`) |
| `BRAWL_JIT_FORMS_FAULT=<1-4>`, `BRAWL_JIT_STALE_FAULT=<n>` | make the form analysis or the stale stub wrong on purpose, to show the tests notice |
| `BRAWL_JIT_STALE_STUB=<n>` | stale paths of a block before it gets a stub; 0 never (`jit_stale_tests` sets 8) |
| `BRAWL_JIT_FAR_JUMPS=1` | jumps to the shared routines always take the long form (`mkw_jit_quality_far_tests`) |
| `BRAWL_JIT_FAR_MEMORY=1` | code space mapped anywhere, not near the executable (`mkw_jit_branch_direct_far_tests`) |
| `BRAWL_JIT_GUARD=0` | every line of a block tested, guarded ones too (see "Guarded lines") |
| `BRAWL_JIT_EXEC_MEM=1`, `BRAWL_JIT_BLOCK_PC=1` | blocks add their count to `frame.executed` in memory; blocks store `ctx->pc` at entry |

## Files

| Path (under `wiicompiled/runtime`) | What |
|---|---|
| `src/ppc_interpreter.cpp` | dispatcher, block cache, handlers, the shims compiled code calls |
| `src/ppc_interpreter_internal.h` | the types the interpreter and the JIT share (`Frame`, `Op`, `K`, `Block`, ...) and the backend interface |
| `include/jit/jit_exec_memory.h`, `src/jit/jit_exec_memory.cpp` | executable memory (2 MiB chunks on huge pages where the host grants them) |
| `src/jit/jit_x64.h`, `src/jit/jit_x64.cpp` | backend core: trampoline, block prologue and check, exits, links, inline caches, stale stubs |
| `src/jit/jit_x64_int.cpp`, `jit_x64_mem.cpp`, `jit_x64_branch.cpp`, `jit_x64_fp.cpp`, `jit_x64_generic.cpp` | native code per op group |
| `src/jit/jit_x64_cr.cpp`, `jit_x64_forms.cpp` | the per-block analyses |
| `third_party/xbyak` | the x86-64 emitter (BSD-3-Clause) |
| `include/platform/huge_text.h`, `src/platform/huge_text.cpp` | the executable's text on 2 MiB pages at startup (Linux; `BRAWL_HUGE_TEXT=0` off) |
| `tests/jit_*_tests.cpp`, `tests/jit_test_env.h`, `tests/huge_text_tests.cpp` | tests |
| `tests/interpreter_test_support.cpp` | what the interpreter tests link instead of the runtime: the float ABI table, a patch machinery stand-in |

## Tests

Built with `-DMKW_BUILD_INTERPRETER_TESTS=ON`; `ctest -R "jit|interpreter|guest_memory_map|huge_text"`.

- `mkw_jit_differential_tests`: guest programs (directed and random; `JIT_DIFF_RANDOM=<n>` for more) run under the
  classic interpreter, the cached interpreter and the JIT in one process (`PpcInterpreter::SetEngineOverride`);
  registers and a guest memory sandbox must be equal. Covers every op kind, block ends, links, stale code, calls into
  registered functions, exceptions through a call.
- Per group: `jit_int`, `jit_mem` (also compares dirty marks and exceptions from stores), `jit_fp` (every float kind
  over NaN, denormal and rounding operand tables, both forms), `jit_forms`, `jit_generic`, `jit_branch` (every branch
  form, calls of every guard class, patched callees, host calls between them (longjmp-like returns, exits from inside
  them); `mkw_jit_branch_direct_tests` runs them entered through
  `jit_enter_direct`), `jit_quality` (condition-register liveness, the entry check),
  `jit_stale`, `jit_core` (links, retire, the code limit), `jit_exec_memory`.
- The existing interpreter tests, run with the JIT as the engine.
- The rollback benchmark (`tools/rollback_bench`): every run must match the replay's checksums, with code variants off
  (`BRAWL_NO_CODE_VARIANTS=1`) so mod code actually runs in the JIT.

## Measured

Rollback benchmark, three Project+ replays, four passes an update, every run matching its replay's checksums, mean
update time: interpreter 6.49 ms, JIT 5.55 ms (14% less), JIT with ahead-of-time code variants 5.01 ms, Brawl Online's
Dolphin 4.96 ms. Repeat campaigns of the same build differ by a percent or two; compare within one campaign.

What separates the JIT from the ahead-of-time variants is about half the calls into and out of compiled code and about
half the code itself (roughly twice the cycles per guest instruction). Four further designs were built and measured
on the same benchmark and gave nothing, so they are not in the tree: compiling a hot function's blocks as one region
with direct jumps between them, keeping guest registers in host registers across such a region, compiling a
patched function as a host function called the way a variant is, and running a patched callee in the calling block's host
frame instead of a nested `Run` (5.35 ms against 5.32 ms with the nested call, twelve runs each, paired by replay and pass).
