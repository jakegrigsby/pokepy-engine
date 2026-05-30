#!/usr/bin/env python3
"""Extract per-gen pokepy data bundles from metamon showdown_dex static JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse array writers from the main extractor.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from extract_ps_data import create_numpy_arrays  # noqa: E402


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def extract_gen_from_dex(gen: int, repo_root: Path, output_dir: Path) -> None:
    static = repo_root / "metamon" / "backend" / "showdown_dex" / "static"
    pokedex = _load_json(static / "pokemon" / f"gen{gen}pokedex.json")
    moves = _load_json(static / "moves" / f"gen{gen}moves.json")
    typechart = _load_json(static / "typechart" / f"gen{gen}typechart.json")

    # Gen1-2 have no abilities/items in dex JSON; reuse empty stubs compatible
    # with create_numpy_arrays (abilities/items from gen9 root if present).
    ps_path = repo_root / "server" / "pokemon-showdown"
    abilities_path = ps_path / "data" / "abilities.ts"
    items_path = ps_path / "data" / "items.ts"
    from extract_ps_data import extract_abilities, extract_items

    abilities = extract_abilities(ps_path) if abilities_path.exists() else {}
    items = extract_items(ps_path) if items_path.exists() else {}

    output_dir.mkdir(parents=True, exist_ok=True)
    create_numpy_arrays(
        pokedex,
        moves,
        abilities,
        items,
        typechart,
        output_dir,
    )
    print(f"Wrote gen{gen} bundle to {output_dir}")


def main():
    repo_root = Path(__file__).resolve().parents[4]
    pokepy_data = Path(__file__).resolve().parents[1] / "pokepy" / "data" / "extracted"
    parser = argparse.ArgumentParser(
        description="Extract gen-N data from showdown_dex JSON"
    )
    parser.add_argument("--gen", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir (default pokepy/data/extracted/genN)",
    )
    args = parser.parse_args()
    out = args.output or (pokepy_data / f"gen{args.gen}")
    extract_gen_from_dex(args.gen, repo_root, out)


if __name__ == "__main__":
    main()
