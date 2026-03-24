#!/usr/bin/env python3
"""
Utility script to convert the held-out AIT validation CSV into a JSONL file
that mirrors the structure expected from LLM labeling.

Instead of calling an LLM, we copy the ground-truth `is_attack` flag directly.
This is useful when you simply need a `val_labeled.jsonl` placeholder to keep
training scripts happy or to benchmark downstream models against the official
AIT labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate val_labeled.jsonl directly from ait_val_subset.csv."
    )
    parser.add_argument(
        "--input",
        default="data/processed/ait_val_subset.csv",
        type=Path,
        help="CSV file containing the held-out validation split.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/val_labeled.jsonl",
        type=Path,
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--reasoning-template",
        default="Label copied from AIT ground truth for {testbed}/{service}.",
        help="Short explanation inserted into each record.",
    )
    return parser.parse_args()


def to_bool(value: Any) -> bool:
    """Best-effort conversion that handles bools, ints, and common strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"true", "1", "yes"}
    return False


def build_record(row: pd.Series, reasoning_template: str) -> Dict[str, Any]:
    attack_flag = to_bool(row.get("is_attack", False))
    label = "Attack" if attack_flag else "Normal"

    reasoning = reasoning_template.format(
        testbed=row.get("testbed", "unknown"),
        service=row.get("service", "unknown"),
    )

    record = {
        "label": label,
        "mitre_t_code": None,
        "mitre_technique": None,
        "reasoning": reasoning,
        "testbed": row.get("testbed"),
        "service": row.get("service"),
        "raw": row.get("raw"),
        "source_file": row.get("source_file"),
        "relative_path": row.get("relative_path"),
        "timestamp": row.get("timestamp"),
        "line_number": int(row["line_number"]) if "line_number" in row else None,
        "is_attack": attack_flag,
        "time_label": row.get("time_label"),
        "similarity_label": row.get("similarity_label"),
    }

    # Drop keys whose values are pandas.NA or NaN to keep JSON clean.
    return {k: v for k, v in record.items() if pd.notna(v)}


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            record = build_record(row, args.reasoning_template)
            handle.write(json.dumps(record) + "\n")

    print(f"Wrote {len(df)} records to {args.output}")


if __name__ == "__main__":
    main()
