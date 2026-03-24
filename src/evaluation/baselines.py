"""Baselines used for comparing Sec-LogLLM performance."""

from __future__ import annotations

from typing import Iterable, Sequence

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score


class TFIDFRandomForestBaseline:
    """Simple TF-IDF + Random Forest baseline for log labels."""

    def __init__(
        self,
        *,
        max_features: int = 5000,
        n_estimators: int = 100,
        max_depth: int = 10,
        n_jobs: int = -1,
        random_state: int = 42
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def train(self, logs: Sequence[str], labels: Sequence[str]) -> "TFIDFRandomForestBaseline":
        """Train the TF-IDF vectorizer and Random Forest classifier."""
        features = self.vectorizer.fit_transform(logs)
        self.classifier.fit(features, labels)
        return self

    def predict(self, logs: Sequence[str]) -> Sequence[str]:
        """Predict labels for a batch of log messages."""
        features = self.vectorizer.transform(logs)
        return self.classifier.predict(features)

    def evaluate(
        self,
        logs: Sequence[str],
        labels: Sequence[str],
        positive_label: str = "Attack"
    ) -> dict:
        """Return standard classification metrics for the supplied data."""
        predictions = self.predict(logs)
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        return {
            "classification_report": report,
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
            "f1_attack": f1_score(labels, predictions, pos_label=positive_label, zero_division=0)
        }
