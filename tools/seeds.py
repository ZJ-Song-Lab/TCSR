"""List the seed records (draws) per checkpoint pair.

Reads manifests/seeds.json and prints the calibration-draw seeds of the
selected pairs.  The paper's convention splits each draw into a
calibration seed (image-list sampling) and an optimization seed
(Stage-1/2 optimization); use ``--kind`` to select which one to emit.

Examples::

    python tools/seeds.py --pair HR-02
    python tools/seeds.py --setting hrsid_rtdetr --kind optimization
    python tools/seeds.py --format csv --output seeds_all.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SEEDS_PATH = os.path.join(REPO_ROOT, "manifests", "seeds.json")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pair", action="append", default=None,
                   help="checkpoint pair id (repeatable), e.g. HR-01")
    p.add_argument("--setting", default=None,
                   help="filter by setting name (e.g. hrsid_rtdetr)")
    p.add_argument("--kind", choices=("draw", "calibration", "optimization"),
                   default="draw",
                   help="draw: the raw seeds; calibration: same; "
                        "optimization: seed + offset (default: draw)")
    p.add_argument("--format", choices=("list", "json", "csv"),
                   default="list")
    p.add_argument("--output", default=None, help="write to this path")
    return p.parse_args(argv)


def select_pairs(manifest, pair_filter, setting):
    pairs = manifest["pairs"]
    if pair_filter:
        missing = [p for p in pair_filter if p not in pairs]
        if missing:
            raise SystemExit(f"unknown pair(s): {', '.join(missing)} "
                             f"(available: {', '.join(sorted(pairs))})")
        return {p: pairs[p] for p in pair_filter}
    if setting:
        return {p: e for p, e in pairs.items() if e["setting"] == setting}
    return dict(pairs)


def expand(entry, kind, opt_offset):
    conv = opt_offset if kind == "optimization" else 0
    seeds = list(range(entry["seeds_start"], entry["seeds_end"] + 1))
    return [s + conv for s in seeds]


def main(argv=None):
    args = parse_args(argv)
    with open(SEEDS_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    opt_offset = manifest["convention"]["optimization_seed_offset"]
    pairs = select_pairs(manifest, args.pair, args.setting)
    if not pairs:
        raise SystemExit("no pairs selected")

    rows = []  # (pair, setting, kind, seeds)
    for pair in sorted(pairs):
        entry = pairs[pair]
        rows.append((pair, entry["setting"], args.kind,
                     expand(entry, args.kind, opt_offset)))

    lines = []
    if args.format == "list":
        for pair, setting, kind, seeds in rows:
            lines.append(f"{pair} ({setting}, {kind} seeds): "
                         + " ".join(str(s) for s in seeds))
    elif args.format == "json":
        lines.append(json.dumps(
            {pair: seeds for pair, _, _, seeds in rows}, indent=2))
    else:  # csv
        lines.append("pair,setting,kind,seed")
        for pair, setting, kind, seeds in rows:
            lines.extend(f"{pair},{setting},{kind},{s}" for s in seeds)

    text = "\n".join(lines) + "\n"
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)),
                    exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[seeds] wrote {args.output}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
