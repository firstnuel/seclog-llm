from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


JsonLike = Dict[str, Any]


@dataclass
class CsvWindowConfig:
    """Sliding-window settings when building sequences from CSV rows."""

    window_size: int = 32
    stride: int = 32
    drop_last: bool = True
    log_column: str = "raw"
    label_column: str = "is_attack"
    metadata_columns: Optional[Sequence[str]] = None


class SecLogDataset(Dataset):
    """
    Flexible dataset that can ingest either:
        1. JSONL samples where each record already contains a list of log lines.
        2. Flat CSV rows that are grouped into sliding windows.

    Each sample yields:
        {
            "log_sequence": List[str],
            "label": str,
            "label_id": int,
            "metadata": Dict[str, Any]
        }
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        sequence_keys: Sequence[str] = ("log_sequence", "logs", "raw_logs"),
        label_field: str = "label",
        reasoning_field: Optional[str] = "reasoning",
        csv_window_config: Optional[CsvWindowConfig] = None,
        label_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self.path = Path(path)
        self.sequence_keys = sequence_keys
        self.label_field = label_field
        self.reasoning_field = reasoning_field
        self.csv_window_config = csv_window_config or CsvWindowConfig()

        self.samples: List[Dict[str, Any]] = []
        self.label_to_id: Dict[str, int] = label_map.copy() if label_map else {}
        self._load_file()

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_file(self) -> None:
        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            records = self._load_jsonl()
        elif suffix in {".json"}:
            records = self._load_json()
        elif suffix in {".csv", ".tsv"}:
            records = self._load_csv()
        else:
            raise ValueError(f"Unsupported dataset format: {self.path}")

        for record in records:
            label_text = str(record["label"])
            if label_text not in self.label_to_id:
                self.label_to_id[label_text] = len(self.label_to_id)
            record["label_id"] = self.label_to_id[label_text]
            self.samples.append(record)

    def _load_jsonl(self) -> Iterable[Dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                yield self._normalize_json_record(record)

    def _load_json(self) -> Iterable[Dict[str, Any]]:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = [payload]
        for record in payload:
            yield self._normalize_json_record(record)

    def _load_csv(self) -> Iterable[Dict[str, Any]]:
        cfg = self.csv_window_config
        df = pd.read_csv(self.path)
        log_values = df[cfg.log_column].astype(str).tolist()

        if cfg.label_column in df:
            label_values = df[cfg.label_column].tolist()
        else:
            # Default to normal if label column missing
            label_values = [False] * len(df)

        metadata_cols = list(cfg.metadata_columns or [])

        for start in range(0, len(df), cfg.stride):
            end = start + cfg.window_size
            window_logs = log_values[start:end]
            window_labels = label_values[start:end]

            if len(window_logs) < cfg.window_size and cfg.drop_last:
                break
            if not window_logs:
                continue

            label = "Attack" if any(bool(val) for val in window_labels) else "Normal"
            metadata = {
                "source": "csv",
                "start_index": start,
                "end_index": min(end, len(df)),
            }
            for col in metadata_cols:
                if col in df:
                    metadata[col] = df[col].iloc[start:end].tolist()

            yield {
                "log_sequence": window_logs,
                "label": label,
                "metadata": metadata,
            }

    def _normalize_json_record(self, record: JsonLike) -> Dict[str, Any]:
        for key in self.sequence_keys:
            if key in record and record[key]:
                seq = record[key]
                if isinstance(seq, str):
                    seq = [seq]
                seq = [str(item) for item in seq if str(item).strip()]
                if seq:
                    break
        else:
            # Fall back to a singular "raw" entry when present
            raw = record.get("raw") or record.get("message")
            if raw is None:
                raise ValueError(f"Record missing log sequence in {self.path}")
            seq = [str(raw)]

        label_text = record.get(self.label_field, "Normal")
        metadata = {
            "source": "json",
        }
        if self.reasoning_field and self.reasoning_field in record:
            metadata["reasoning"] = record[self.reasoning_field]
        for key, value in record.items():
            if key in self.sequence_keys or key in {self.label_field}:
                continue
            metadata.setdefault(key, value)

        return {
            "log_sequence": seq,
            "label": str(label_text),
            "metadata": metadata,
        }


class SecLogCollator:
    """
    Collate function that keeps log sequences as raw text for the encoder while
    converting labels to tensors.
    """

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        log_sequences = [item["log_sequence"] for item in batch]
        label_ids = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
        labels = [item["label"] for item in batch]
        metadata = [item.get("metadata", {}) for item in batch]

        return {
            "log_sequences": log_sequences,
            "label_ids": label_ids,
            "labels": labels,
            "metadata": metadata,
        }
