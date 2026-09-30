#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot false-edit probability versus detection delay")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-column", default="backbone")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    required = {args.method_column, "false_edit_probability", "mean_detection_delay"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for method, group in frame.groupby(args.method_column):
        ordered = group.sort_values("false_edit_probability")
        axis.plot(
            ordered["false_edit_probability"],
            ordered["mean_detection_delay"],
            marker="o",
            label=str(method),
        )
    axis.set_xlabel("No-edit false-edit probability (%)")
    axis.set_ylabel("Mean detection delay (observations)")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
