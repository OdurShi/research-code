#!/usr/bin/env python3
"""
Minimal CLI wrapper around EntityCache for use from bash scripts.

Subcommands
-----------
lookup  --cache-dir DIR --subject STR [--dataset-dir DIR]
  Prints "HIT <proposal_dir>" or "MISS" to stdout.
  Exit code: 0 = HIT, 1 = MISS.

store   --cache-dir DIR --subject STR --proposal-dir DIR
                        [--dataset-dir DIR]
  Appends an entry to entity_index.json.
  Exit code: always 0.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[_\-]", " ", s.lower()).split())


def _load(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"entries": []}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def cmd_lookup(args: argparse.Namespace) -> int:
    index_path = Path(args.cache_dir) / "entity_index.json"
    data = _load(index_path)
    dataset_str = str(Path(args.dataset_dir).resolve()) if args.dataset_dir else None
    query_norm = _norm(args.subject)

    for entry in data.get("entries", []):
        if dataset_str and entry.get("source_dataset") != dataset_str:
            continue
        if _norm(entry.get("subject", "")) != query_norm:
            continue
        proposal_dir = Path(entry.get("proposal_dir", ""))
        if not proposal_dir.exists():
            continue
        print(f"HIT {entry['proposal_dir']}")
        print(f"[cache] HIT  '{args.subject}'", file=sys.stderr)
        return 0

    print("MISS")
    print(f"[cache] MISS '{args.subject}'", file=sys.stderr)
    return 1


def cmd_store(args: argparse.Namespace) -> int:
    index_path = Path(args.cache_dir) / "entity_index.json"
    data = _load(index_path)
    dataset_str = str(Path(args.dataset_dir).resolve()) if args.dataset_dir else ""
    data.setdefault("entries", []).append({
        "subject":        args.subject,
        "proposal_dir":   str(Path(args.proposal_dir).resolve()),
        "source_dataset": dataset_str,
    })
    _save(index_path, data)
    print(f"[cache] STORED '{args.subject}'  → {args.proposal_dir}", file=sys.stderr)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    lk = sub.add_parser("lookup")
    lk.add_argument("--cache-dir",   required=True)
    lk.add_argument("--subject",     required=True)
    lk.add_argument("--dataset-dir", default=None)

    st = sub.add_parser("store")
    st.add_argument("--cache-dir",    required=True)
    st.add_argument("--subject",      required=True)
    st.add_argument("--proposal-dir", required=True)
    st.add_argument("--dataset-dir",  default=None)

    args = parser.parse_args()
    if args.cmd == "lookup":
        sys.exit(cmd_lookup(args))
    else:
        sys.exit(cmd_store(args))


if __name__ == "__main__":
    main()
