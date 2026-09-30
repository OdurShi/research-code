#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


BACKBONE_CONFIGS = {
    "qwen3-8b": "configs/default.yaml",
    "qwen2.5-7b": "configs/qwen25_7b.yaml",
    "llama3.1-8b": "configs/llama31_8b.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an SBLS grid sequentially on one GPU")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONE_CONFIGS))
    parser.add_argument("--events", nargs="+", default=["add", "split", "prior_drift", "covariate_drift", "anomaly"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = []
    for dataset_dir in sorted(path for path in args.prepared_root.iterdir() if path.is_dir()):
        for event in args.events:
            event_dir = dataset_dir / event
            taxonomy = event_dir / "initial_taxonomy.json"
            stream = event_dir / "stream.jsonl"
            if not taxonomy.exists() or not stream.exists():
                continue
            for backbone in args.backbones:
                if backbone not in BACKBONE_CONFIGS:
                    raise ValueError(f"Unknown backbone {backbone!r}; choose from {sorted(BACKBONE_CONFIGS)}")
                output = args.runs_root / dataset_dir.name / event / backbone
                command = [
                    "sbls",
                    "run",
                    "--config",
                    BACKBONE_CONFIGS[backbone],
                    "--taxonomy",
                    str(taxonomy),
                    "--stream",
                    str(stream),
                    "--output",
                    str(output),
                ]
                jobs.append(command)
    if not jobs:
        raise RuntimeError("No prepared event directories were found")
    for index, command in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
