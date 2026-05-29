#!/usr/bin/env python3
"""Sync ITEM_/ABILITY_ numeric constants in pokepy/ with id_mappings.json."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPINGS_PATH = ROOT / "pokepy" / "data" / "extracted" / "id_mappings.json"
POKEPY_DIR = ROOT / "pokepy"

CONST_RE = re.compile(
    r"^(\s*)((?:ITEM|ABILITY|_ITEM|_ABILITY)_([A-Z0-9_]+))\s*=\s*(\d+)\s*$"
)


def _lookup_id(
    prefix: str, name_part: str, mappings: dict[str, dict[str, int]]
) -> int | None:
    key = name_part.lower().replace("_", "")
    if prefix.endswith("ITEM"):
        return mappings["item_id_to_idx"].get(key)
    return mappings["ability_id_to_idx"].get(key)


def sync_file(path: Path, mappings: dict[str, dict[str, int]]) -> int:
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    changed = 0
    out: list[str] = []

    for line in lines:
        match = CONST_RE.match(line.rstrip("\n"))
        if not match:
            out.append(line)
            continue

        indent, full_name, name_part, old_val = match.groups()
        prefix = full_name.split("_", 1)[0]
        if full_name.startswith("_"):
            prefix = full_name.split("_", 2)[1]  # _ITEM_FOO -> ITEM

        new_id = _lookup_id(prefix, name_part, mappings)
        if new_id is None or str(new_id) == old_val:
            out.append(line)
            continue

        newline = "\n" if line.endswith("\n") else ""
        out.append(f"{indent}{full_name} = {new_id}{newline}")
        changed += 1

    if changed:
        path.write_text("".join(out))
    return changed


def main() -> None:
    mappings = json.loads(MAPPINGS_PATH.read_text())
    total = 0
    for path in sorted(POKEPY_DIR.rglob("*.py")):
        total += sync_file(path, mappings)
    print(f"Updated {total} constant assignments")


if __name__ == "__main__":
    main()
