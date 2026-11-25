#!/usr/bin/env python3
"""Check retained helper boundaries against a production translation manifest.

Metadata-only audit: this does not prove native instruction semantics or runtime
invalidation delivery. Run with the private wiicompiled/generated directory.
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import re

HELPERS = {
    0x80004338: "memcpy", 0x80162C04: "Mtx44Copy",
    0x801A7DBC: "CameraMtx", 0x801A7FC8: "EnvTexMtx",
    0x801B6708: "LightSettingCopy", 0x801EC184: "PSMTXCopy",
    0x801EC1B8: "PSMTXConcat", 0x801EC284: "PSMTXConcatArray",
    0x801EC41C: "PSMTXInverse", 0x80202A58: "AXFXReverbHiExpCallback",
    0x80203A10: "AXFXDelayCallback",
}
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("generated", type=Path)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
manifest_path = a.generated / "base_translation_output.json"
guard_paths = list((a.generated / "build_shards/base_dispatch").glob("guest_code_guard*.cpp"))
if len(guard_paths) != 1:
    raise SystemExit("Expected exactly one production guard table")
guard_path = guard_paths[0]
manifest = json.loads(manifest_path.read_text())
text = guard_path.read_text()
body = text.split("g_mkwGuardedFunctionStarts[] = {", 1)[1].split("};", 1)[0]
starts = [int(x, 16) for x in re.findall(r"0x([0-9A-Fa-f]+)u", body)]
if starts != sorted(set(starts)):
    raise SystemExit("Guard starts are not strictly increasing")
functions = {f["entryPoint"]: f for f in manifest["functions"]}
rows = []
for address, name in HELPERS.items():
    f = functions[address]
    index = bisect.bisect_left(starts, address)
    if index + 1 >= len(starts) or starts[index] != address:
        raise SystemExit(f"Missing guard for {name}")
    end = starts[index + 1]
    # These retained bodies are contiguous; instruction count is independent
    # metadata to check that the next guard is exactly at their boundary.
    instructions = f["build"]["guestInstructions"]
    labels = f.get("localLabelAddresses", [])
    passed = (end - address == 4 * instructions
              and all(address <= label < end for label in labels)
              and not f["interiorToOtherTranslation"])
    rows.append({"helper": name, "start": f"{address:08x}",
                 "next_guard": f"{end:08x}", "guest_instructions": instructions,
                 "direct_calls": [f"{v:08x}" for v in f["build"]["directCallDependencies"]],
                 "passed": passed})
result = {"scope": "Eleven retained non-Gecko helper entry intervals; metadata only",
          "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (manifest_path, guard_path)},
          "guard_count": len(starts), "helpers": rows,
          "passed": all(row["passed"] for row in rows)}
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"passed": result["passed"], "helpers": len(rows)}))
raise SystemExit(0 if result["passed"] else 1)
