import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


class EvaluationMetrics:
    """Shared evaluation utilities for Sec-LogLLM baselines."""

    @staticmethod
    def compute_classification_metrics(
        predictions: Sequence[str],
        ground_truth: Sequence[str],
        labels: Sequence[str] = ("Attack", "Normal"),
    ) -> Dict[str, Any]:
        """Compute standard classification metrics."""
        report = classification_report(
            ground_truth,
            predictions,
            labels=list(labels),
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(
            ground_truth,
            predictions,
            labels=list(labels),
        ).tolist()
        return {
            "classification_report": report,
            "f1_macro": f1_score(ground_truth, predictions, average="macro", zero_division=0),
            "f1_weighted": f1_score(ground_truth, predictions, average="weighted", zero_division=0),
            "confusion_matrix": matrix,
        }

    @staticmethod
    def compute_fidelity_score(
        model: Any,
        log_sequences: Iterable[List[str]],
        *,
        max_examples: Optional[int] = None,
    ) -> float:
        """Measure fidelity by removing the highest-alpha log and comparing predictions."""
        fidelity_scores: List[float] = []
        for idx, sequence in enumerate(log_sequences):
            if max_examples is not None and idx >= max_examples:
                break
            original_output, alphas = model.generate(sequence)
            parsed = json.loads(original_output)
            original_label = parsed.get("label")
            if original_label is None:
                continue

            alpha_array = np.array(alphas)
            max_alpha_idx = int(np.nanargmax(alpha_array))

            masked_sequence = sequence[:max_alpha_idx] + sequence[max_alpha_idx + 1 :]
            if not masked_sequence:
                continue

            masked_output, _ = model.generate(masked_sequence)
            masked_label = json.loads(masked_output).get("label")

            fidelity_scores.append(1.0 if masked_label != original_label else 0.0)

        if not fidelity_scores:
            return 0.0

        return float(np.mean(fidelity_scores))

    @staticmethod
    def compute_hallucination_rate(
        predictions: Iterable[Dict[str, Any]],
        *,
        expected_null_mitre: bool = True,
        mitre_field: str = "mitre_t_code",
    ) -> float:
        """Rate at which the model hallucinates MITRE codes on non-security data."""
        total = 0
        hallucinations = 0
        for record in predictions:
            total += 1
            mitre_code = record.get(mitre_field)
            if expected_null_mitre and mitre_code:
                hallucinations += 1

        if total == 0:
            return 0.0

        return hallucinations / total
