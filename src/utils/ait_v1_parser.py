"""
Parser utilities for the AIT-LDS v1.1 dataset.

The dataset that ships with the thesis repo uses the following layout:

data/ait_v1.1/
├── data/      # server + user hosts (logs of various services)
└── labels/    # mirrors server tree with per-line ground-truth labels

This module focuses on providing a light-weight parser that can:
1. List log files that belong to a given testbed (mail.cup.com, ...).
2. Load line-aligned labels when they are available (only for server logs).
3. Convert log files into pandas DataFrames enriched with metadata that
   downstream notebooks/models can consume (testbed, service, etc.).
4. Produce convenience helpers for creating train/validation splits.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd


class AITV1Parser:
    """Parser for the AIT-LDS v1.1 log corpus."""

    ATTACK_DATE = "2020-03-04"

    TESTBEDS = [
        "mail.cup.com",
        "mail.spiral.com",
        "mail.insect.com",
        "mail.onion.com",
    ]

    USER_SERVER_MAP: Dict[str, List[str]] = {
        "mail.cup.com": ["user-0", "user-1", "user-2", "user-6"],
        "mail.spiral.com": ["user-3", "user-5", "user-8"],
        "mail.insect.com": ["user-4", "user-9"],
        "mail.onion.com": ["user-7", "user-10"],
    }

    ISO_TIMESTAMP_RE = re.compile(
        r"^(?P<iso>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)"
    )
    SYSLOG_TIMESTAMP_RE = re.compile(r"^(?P<syslog>[A-Za-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})")
    APACHE_TIMESTAMP_RE = re.compile(
        r"\[(?P<apache>\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\]"
    )

    def __init__(self, dataset_root: str):
        self.root = Path(dataset_root).expanduser().resolve()
        self.data_dir = self.root / "data"
        self.labels_dir = self.root / "labels"

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Could not find data directory: {self.data_dir}")

        self._user_lookup = self._build_user_lookup()

    def _build_user_lookup(self) -> Dict[str, str]:
        reverse = {}
        for testbed, users in self.USER_SERVER_MAP.items():
            for user in users:
                reverse[user] = testbed
        return reverse

    # ------------------------------------------------------------------
    # File discovery helpers
    # ------------------------------------------------------------------
    def list_log_files(
        self,
        testbed: str,
        include_users: bool = False,
        log_types: Optional[Iterable[str]] = None,
    ) -> List[Path]:
        """
        List all log files for a given testbed.

        Args:
            testbed: One of the testbeds found in TESTBEDS.
            include_users: Whether to include host logs (user-* directories)
                that map to the chosen testbed.
            log_types: Optional iterable of substrings/filenames to filter by.
                Example: ["access.log", "fast.log"].
        """
        if testbed not in self.TESTBEDS:
            raise ValueError(f"Unknown testbed '{testbed}'. Expected one of {self.TESTBEDS}")

        paths: List[Path] = []
        base_dir = self.data_dir / testbed
        if base_dir.exists():
            paths.extend(self._collect_logs(base_dir, log_types))

        if include_users:
            for user_dirname in self.USER_SERVER_MAP.get(testbed, []):
                user_dir = self.data_dir / user_dirname
                if user_dir.exists():
                    paths.extend(self._collect_logs(user_dir, log_types))

        return sorted(paths)

    def _collect_logs(self, directory: Path, log_types: Optional[Iterable[str]]) -> List[Path]:
        files: List[Path] = []
        for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
            if log_types and not any(substr in file_path.name for substr in log_types):
                continue
            files.append(file_path)
        return files

    # ------------------------------------------------------------------
    # Label loading
    # ------------------------------------------------------------------
    LabelValue = Union[int, str, None]

    def load_labels(self, log_file: Path) -> Dict[int, Tuple[LabelValue, LabelValue]]:
        """
        Load labels for a given log file.

        Returns a dictionary mapping `line_number -> (time_label, similarity_label)`.
        If no labels are available (e.g., for user host logs), an empty dict is returned.
        """
        relative_path = self._relative_to_data(log_file)
        label_path = self.labels_dir / relative_path

        if not label_path.exists():
            # User logs and auxiliary files simply do not have labels.
            return {}

        labels: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        with label_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_no, raw in enumerate(handle, start=1):
                cleaned = raw.strip()
                if not cleaned:
                    continue
                parts = [p.strip() for p in cleaned.split(",")]
                time_label = self._parse_label_value(parts[0]) if parts else None
                similarity_label = self._parse_label_value(parts[1]) if len(parts) > 1 else None
                labels[line_no] = (time_label, similarity_label)
        return labels

    @staticmethod
    def _parse_label_value(value: str) -> "AITV1Parser.LabelValue":
        if value == "":
            return None
        try:
            return int(value)
        except ValueError:
            return value or None

    # ------------------------------------------------------------------
    # Main parsing routines
    # ------------------------------------------------------------------
    def parse_log_file(self, log_file: Path, include_labels: bool = True) -> pd.DataFrame:
        """
        Parse a single log file into a pandas DataFrame.

        Columns returned:
            - raw:        original log line
            - line_number
            - source_file
            - relative_path
            - testbed
            - service (apache2/suricata/user-*)
            - node_type ("server" or "user")
            - timestamp (best-effort ISO string)
            - time_label / similarity_label / is_attack (if labels available)
        """
        if not log_file.exists():
            raise FileNotFoundError(log_file)

        labels = self.load_labels(log_file) if include_labels else {}
        rows = []
        relative_path = self._relative_to_data(log_file)
        service = self._extract_service(relative_path)
        testbed = self._extract_testbed(relative_path)
        node_type = "user" if relative_path.parts[0].startswith("user-") else "server"

        with log_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                raw_line = raw_line.rstrip("\n")
                if raw_line == "":
                    continue

                timestamp = self._extract_timestamp(raw_line, log_file.suffix)
                time_label, similarity_label = labels.get(line_number, (None, None))
                is_attack = self._is_attack_label(time_label, similarity_label)

                rows.append(
                    {
                        "raw": raw_line,
                        "timestamp": timestamp,
                        "line_number": line_number,
                        "source_file": log_file.name,
                        "relative_path": str(relative_path),
                        "testbed": testbed,
                        "service": service,
                        "node_type": node_type,
                        "time_label": time_label,
                        "similarity_label": similarity_label,
                        "is_attack": is_attack,
                    }
                )

        return pd.DataFrame(rows)

    def parse_testbed(
        self,
        testbed: str,
        include_users: bool = False,
        log_types: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        """
        Parse every log file belonging to a testbed.

        Args:
            testbed: e.g., "mail.cup.com".
            include_users: also ingest logs from mapped user-* hosts.
            log_types: optional filename filters.
        """
        frames: List[pd.DataFrame] = []
        for log_file in self.list_log_files(testbed, include_users=include_users, log_types=log_types):
            try:
                frames.append(self.parse_log_file(log_file))
            except Exception as exc:
                # Keep the pipeline resilient; a single corrupt file should not abort everything.
                print(f"[WARN] Failed to parse {log_file}: {exc}")

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def create_train_val_split(
        self,
        train_testbeds: Optional[List[str]] = None,
        val_testbed: str = "mail.onion.com",
        include_users: bool = False,
        log_types: Optional[Iterable[str]] = None,
        sample_normal: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a train/validation split by reserving one testbed for validation.

        Args mirror the plan documented in thesis_execution_plan_v2.md.
        """
        if train_testbeds is None:
            train_testbeds = [tb for tb in self.TESTBEDS if tb != val_testbed]

        train_frames = [
            self.parse_testbed(tb, include_users=include_users, log_types=log_types)
            for tb in train_testbeds
        ]
        train_df = pd.concat([df for df in train_frames if not df.empty], ignore_index=True)

        val_df = self.parse_testbed(val_testbed, include_users=include_users, log_types=log_types)

        if sample_normal and not train_df.empty:
            attack_logs = train_df[train_df["is_attack"]]
            normal_logs = train_df[~train_df["is_attack"]]
            normal_sample = normal_logs.sample(
                n=min(sample_normal, len(normal_logs)), random_state=42, replace=False
            )
            train_df = pd.concat([attack_logs, normal_sample]).sample(frac=1.0, random_state=42)

        return train_df, val_df

    # ------------------------------------------------------------------
    # Stats utilities
    # ------------------------------------------------------------------
    @staticmethod
    def get_attack_statistics(df: pd.DataFrame) -> Dict[str, object]:
        if df.empty:
            return {
                "total_logs": 0,
                "attack_logs": 0,
                "normal_logs": 0,
                "attack_ratio": 0.0,
                "unique_sources": 0,
                "testbeds": [],
            }

        total = len(df)
        attack_logs = int(df["is_attack"].sum())
        normal_logs = total - attack_logs
        attack_ratio = float(attack_logs) / total if total else 0.0

        return {
            "total_logs": total,
            "attack_logs": attack_logs,
            "normal_logs": normal_logs,
            "attack_ratio": attack_ratio,
            "unique_sources": df["source_file"].nunique(),
            "testbeds": sorted(df["testbed"].dropna().unique().tolist()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _relative_to_data(self, path: Path) -> Path:
        try:
            return path.resolve().relative_to(self.data_dir)
        except ValueError:
            raise ValueError(f"{path} is not inside dataset data directory {self.data_dir}")

    def _extract_service(self, relative_path: Path) -> Optional[str]:
        if not relative_path.parts:
            return None
        if relative_path.parts[0].startswith("user-"):
            # User directories contain only flat log files.
            return relative_path.parts[0]
        if len(relative_path.parts) >= 2:
            return relative_path.parts[1]
        return None

    def _extract_testbed(self, relative_path: Path) -> Optional[str]:
        if not relative_path.parts:
            return None
        first = relative_path.parts[0]
        if first in self.TESTBEDS:
            return first
        return self._user_lookup.get(first)

    @staticmethod
    def _is_attack_label(
        time_label: LabelValue,
        similarity_label: LabelValue,
    ) -> bool:
        def _label_flag(label: AITV1Parser.LabelValue) -> bool:
            if label is None:
                return False
            if isinstance(label, int):
                return label != 0
            return label.strip() != ""

        return _label_flag(time_label) or _label_flag(similarity_label)

    def _extract_timestamp(self, raw_line: str, file_suffix: str) -> Optional[str]:
        """Best-effort timestamp extraction for user logs + Apache/Suricata formats."""
        match = self.ISO_TIMESTAMP_RE.match(raw_line)
        if match:
            iso_candidate = match.group("iso")
            try:
                # Normalise to ISO 8601
                parsed = datetime.fromisoformat(iso_candidate.replace(" ", "T"))
                return parsed.isoformat()
            except ValueError:
                return iso_candidate

        match = self.APACHE_TIMESTAMP_RE.search(raw_line)
        if match:
            ts = match.group("apache")
            try:
                parsed = datetime.strptime(ts, "%d/%b/%Y:%H:%M:%S %z")
                return parsed.isoformat()
            except ValueError:
                return ts

        match = self.SYSLOG_TIMESTAMP_RE.match(raw_line)
        if match:
            return match.group("syslog")

        if file_suffix == ".json" and raw_line.startswith("{"):
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                return None
            return payload.get("timestamp") or payload.get("@timestamp") or payload.get("time")

        return None


__all__ = ["AITV1Parser"]
