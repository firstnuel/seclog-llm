"""
LLM-powered labeling pipeline for AIT logs.

This module converts raw log rows (CSV/JSONL) into structured annotations by
calling an OpenAI-compatible model. It focuses on:
    1. Consistent prompting so each response contains Attack/Normal, MITRE code,
       and short reasoning.
    2. Streaming pandas rows through the API with tqdm feedback.
    3. Saving results as JSONL so downstream stages (SFT, evaluation) can load
       them directly.

API keys are read from the `OPENAI_API_KEY` environment variable. When running
inside notebooks, prefer loading that variable via `python-dotenv`:

```python
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env.llm")
# ensures os.environ["OPENAI_API_KEY"] is set before instantiating LLMLabeler
```
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


RESPONSE_SCHEMA = {
    "label": "Attack or Normal",
    "mitre_t_code": "MITRE ATT&CK technique ID (e.g., T1110) or null",
    "mitre_technique": "Human-readable technique name or null",
    "reasoning": "One sentence explaining evidence",
}


@dataclass
class LabelingConfig:
    """Lightweight configuration bundle."""

    model: str = "gpt-5-nano"
    temperature: float = 1.0
    max_retries: int = 3
    retry_sleep: float = 2.0  # seconds


class LLMLabeler:
    """Wraps OpenAI responses for repeatable log labeling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LabelingConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Load it via env or dotenv before use.")

        self.client = OpenAI(api_key=api_key)
        self.config = config or LabelingConfig()
        self.system_prompt = system_prompt or self._default_system_prompt()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def label_log(
        self,
        log_line: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Label a single log line and return the parsed JSON response."""
        metadata = metadata or {}
        prompt = self._build_prompt(log_line, metadata)

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    temperature=self.config.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                parsed = json.loads(response.choices[0].message.content)
                return self._attach_metadata(parsed, metadata)
            except Exception as exc:  # pragma: no cover - relies on remote API
                if attempt == self.config.max_retries:
                    raise
                print(f"[WARN] labeling failed (attempt {attempt}/{self.config.max_retries}): {exc}")
                time.sleep(self.config.retry_sleep)

        raise RuntimeError("Labeling failed after max retries.")

    def label_dataframe(
        self,
        df: pd.DataFrame,
        log_column: str = "raw",
        metadata_cols: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        tqdm_desc: str = "LLM labeling",
    ) -> List[Dict[str, str]]:
        """
        Label each row of a DataFrame and optionally stream results to JSONL.

        Args:
            df: DataFrame containing log lines.
            log_column: Column name with the raw log text.
            metadata_cols: Additional columns passed into the prompt.
            output_path: When provided, write JSONL incrementally.
        """
        metadata_cols = metadata_cols or []
        results: List[Dict[str, str]] = []

        output_file = None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_file = output_path.open("w", encoding="utf-8")

        try:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=tqdm_desc):
                log_line = str(row[log_column])
                metadata = {col: str(row[col]) for col in metadata_cols if col in row and pd.notna(row[col])}
                labeled = self.label_log(log_line, metadata)
                results.append(labeled)
                if output_file:
                    output_file.write(json.dumps(labeled) + "\n")
        finally:
            if output_file:
                output_file.close()

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_prompt(self, log_line: str, metadata: Dict[str, str]) -> str:
        meta_bits = [f"{key}: {value}" for key, value in metadata.items()]
        meta_block = "\n".join(meta_bits)
        schema_lines = "\n".join(f'- "{k}": {v}' for k, v in RESPONSE_SCHEMA.items())

        return f"""
Analyze the following security log and respond with JSON.

Context:
{meta_block or 'None'}

Log:
\"\"\"{log_line}\"\"\"

Ensure the JSON contains:
{schema_lines}
"""

    @staticmethod
    def _attach_metadata(payload: Dict[str, str], metadata: Dict[str, str]) -> Dict[str, str]:
        enriched = dict(payload)
        enriched.update(metadata)
        return enriched

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are an experienced SOC analyst. "
            "Given a single log entry, state whether it indicates an Attack or Normal behavior. "
            "If Attack, cite the most relevant MITRE ATT&CK technique."
        )


def load_dataframe(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Utility for loading CSV/JSONL while selecting relevant columns."""
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(path)
    if columns:
        df = df[columns]
    return df


def save_jsonl(records: Iterable[Dict[str, str]], path: Path) -> None:
    """Write iterable of dicts to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
