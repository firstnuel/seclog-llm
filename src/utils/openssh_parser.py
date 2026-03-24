"""
Parser utilities for the Loghub OpenSSH dataset.

The dataset ships as a single syslog-style text file (`SSH.log`) in
`data/openssh/`. Each line is a classic syslog entry that records SSH
authentication events (invalid user, failed password, accepted password, etc.).

This module offers:
    1. A thin parser that extracts timestamp, hostname, service/pid, and message.
    2. Pattern-based feature extraction for security-relevant events
       (invalid users, brute force attempts, connection closes, etc.).
    3. Convenience helpers to compute dataset statistics and produce balanced
       evaluation samples for downstream modeling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


SYSLOG_RE = re.compile(
    r"""
    ^(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})   # e.g., Dec 10 06:55:46
    \s+
    (?P<hostname>\S+)                                    # host identifier
    \s+
    (?P<service>\S+?)(?:\[(?P<pid>\d+)\])?:              # service and optional pid
    \s+
    (?P<message>.+)$                                     # rest of line
    """,
    re.VERBOSE,
)


SECURITY_PATTERNS: Dict[str, re.Pattern] = {
    "invalid_user": re.compile(r"Invalid user (\S+) from (\S+)", re.IGNORECASE),
    "failed_password": re.compile(
        r"Failed password for (?:invalid user )?(\S+) from (\S+)",
        re.IGNORECASE,
    ),
    "accepted_password": re.compile(r"Accepted password for (\S+) from (\S+)", re.IGNORECASE),
    "accepted_publickey": re.compile(r"Accepted publickey for (\S+) from (\S+)", re.IGNORECASE),
    "connection_closed": re.compile(r"Connection closed by (\S+)", re.IGNORECASE),
    "did_not_receive": re.compile(
        r"Did not receive identification string from (\S+)",
        re.IGNORECASE,
    ),
    "bad_protocol": re.compile(r"Bad protocol version identification.*from (\S+)", re.IGNORECASE),
    "reverse_mapping": re.compile(r"reverse mapping checking.*\[(\S+)\]", re.IGNORECASE),
    "break_in_attempt": re.compile(r"POSSIBLE BREAK-IN ATTEMPT", re.IGNORECASE),
}


@dataclass
class OpenSSHParser:
    """Simple parser for SSH logs stored in syslog format."""

    data_path: Path

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        if self.data_path.is_dir():
            self.log_file = self.data_path / "SSH.log"
        else:
            self.log_file = self.data_path

        if not self.log_file.exists():
            raise FileNotFoundError(f"OpenSSH log file not found: {self.log_file}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parse_file(self, max_lines: Optional[int] = None) -> pd.DataFrame:
        """
        Parse the log file into a pandas DataFrame.

        Columns:
            - timestamp, hostname, service, pid, message, raw
            - derived: event_type, username, source_ip, is_suspicious,
                       potential_brute_force
        """
        rows: List[Dict[str, str]] = []
        with self.log_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_idx, line in enumerate(handle):
                if max_lines is not None and line_idx >= max_lines:
                    break
                parsed = self._parse_line(line.rstrip("\n"))
                if parsed:
                    parsed["line_number"] = line_idx + 1
                    rows.append(parsed)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return self._extract_security_features(df)

    def create_evaluation_sample(
        self,
        df: pd.DataFrame,
        n_suspicious: int = 5000,
        n_normal: int = 5000,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Build a balanced evaluation slice of suspicious vs normal events.
        Useful for downstream generalization tests in the thesis pipeline.
        """
        suspicious = df[df["is_suspicious"]]
        normal = df[~df["is_suspicious"]]

        sampled_suspicious = suspicious.sample(
            min(n_suspicious, len(suspicious)),
            random_state=random_state,
        )
        sampled_normal = normal.sample(
            min(n_normal, len(normal)),
            random_state=random_state,
        )

        sample = pd.concat([sampled_suspicious, sampled_normal]).sample(
            frac=1.0,
            random_state=random_state,
        )
        sample = sample.reset_index(drop=True)
        return sample

    def get_statistics(self, df: pd.DataFrame) -> Dict[str, object]:
        """Summarize the parsed dataset."""
        if df.empty:
            return {
                "total_logs": 0,
                "suspicious_logs": 0,
                "suspicious_ratio": 0.0,
                "unique_ips": 0,
                "unique_users": 0,
                "event_type_distribution": {},
                "potential_brute_force_logs": 0,
            }

        event_counts = df["event_type"].value_counts().to_dict()
        suspicious_logs = int(df["is_suspicious"].sum())
        ratio = suspicious_logs / len(df)
        brute_force = int(df.get("potential_brute_force", pd.Series()).sum()) if "potential_brute_force" in df else 0

        return {
            "total_logs": len(df),
            "suspicious_logs": suspicious_logs,
            "suspicious_ratio": ratio,
            "unique_ips": df["source_ip"].nunique(),
            "unique_users": df["username"].nunique(),
            "event_type_distribution": event_counts,
            "potential_brute_force_logs": brute_force,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parse_line(self, line: str) -> Optional[Dict[str, str]]:
        if not line:
            return None

        match = SYSLOG_RE.match(line)
        if not match:
            return {
                "raw": line,
                "timestamp": None,
                "hostname": None,
                "service": None,
                "pid": None,
                "message": line,
            }

        data = match.groupdict()
        data["raw"] = line
        return data

    def _extract_security_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["event_type"] = "other"
        df["username"] = None
        df["source_ip"] = None
        df["is_suspicious"] = False

        for idx, message in df["message"].items():
            if not isinstance(message, str):
                continue

            for event_type, pattern in SECURITY_PATTERNS.items():
                match = pattern.search(message)
                if match:
                    df.at[idx, "event_type"] = event_type
                    groups = match.groups()

                    if event_type in {"invalid_user", "failed_password"}:
                        df.at[idx, "username"] = groups[0] if len(groups) > 0 else None
                        df.at[idx, "source_ip"] = groups[1] if len(groups) > 1 else None
                        df.at[idx, "is_suspicious"] = True
                    elif event_type.startswith("accepted_"):
                        df.at[idx, "username"] = groups[0] if len(groups) > 0 else None
                        df.at[idx, "source_ip"] = groups[1] if len(groups) > 1 else None
                    elif event_type in {"break_in_attempt"}:
                        df.at[idx, "is_suspicious"] = True
                    elif groups:
                        df.at[idx, "source_ip"] = groups[-1]
                    break

        # simple brute-force heuristic: >= 5 suspicious events from same IP
        suspicious_df = df[df["event_type"].isin({"invalid_user", "failed_password"})]
        ip_counts = suspicious_df["source_ip"].value_counts()
        brute_force_ips = ip_counts[ip_counts >= 5].index
        df["potential_brute_force"] = df["source_ip"].isin(brute_force_ips)
        df.loc[df["potential_brute_force"], "is_suspicious"] = True
        return df


__all__ = ["OpenSSHParser"]


