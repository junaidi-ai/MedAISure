"""Metric calculator implementation for MEDDSAI benchmark.

Adds simple built-in medical metrics:
- clinical_correctness: normalized string match between predicted answer and
  reference answer
- diagnostic_accuracy: alias of accuracy for diagnosis classification
- reasoning_quality: token-overlap F1 between predicted and reference
  rationale/explanation
- rouge_l: ROUGE-L F1 between predicted summary and reference summary
- clinical_relevance: token Jaccard between predicted summary and source
  note/context
- factual_consistency: token-overlap F1 between predicted summary and reference
  summary

These are lightweight heuristics to enable basic evaluations for QA, diagnostic, and
summarization tasks without introducing heavy dependencies (except rouge-score).
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for metric calculation results."""

    metric_name: str
    value: float
    metadata: Optional[Dict[str, Any]] = None


class MetricCalculator:
    """Calculates evaluation metrics for model predictions."""

    # Built-in metrics registry
    _METRICS_REGISTRY = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "roc_auc": roc_auc_score,
        "average_precision": average_precision_score,
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "r2": r2_score,
    }

    def __init__(self) -> None:
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self._register_builtin_metrics()
        self._register_medical_metrics()

    def _register_builtin_metrics(self) -> None:
        """Register all built-in metrics."""
        for name, func in self._METRICS_REGISTRY.items():
            self.register_metric(name, func)

    def register_metric(
        self,
        name: str,
        metric_func: Callable[..., Union[float, Tuple[float, Dict[str, Any]]]],
        **default_kwargs: Any,
    ) -> None:
        """Register a custom metric function.

        Args:
            name: Name to register the metric under
            metric_func: Function that calculates the metric
            **default_kwargs: Default arguments to pass to the metric function
        """
        self.metrics[name] = {"function": metric_func, "default_kwargs": default_kwargs}

    # ----------------------------
    # Medical metrics registration
    # ----------------------------
    def _register_medical_metrics(self) -> None:
        """Register additional medical-specific metrics with simple heuristics."""
        # Diagnostic accuracy is simply accuracy over diagnosis labels
        self.register_metric("diagnostic_accuracy", accuracy_score)

        # Clinical correctness: normalized exact/substring match between
        # prediction and reference answer
        self.register_metric("clinical_correctness", self._metric_clinical_correctness)

        # Reasoning quality: token overlap F1 between predicted and reference
        # rationale/explanation
        self.register_metric("reasoning_quality", self._metric_text_f1)

        # ROUGE-L for summaries (implemented via rouge-score)
        self.register_metric("rouge_l", self._metric_rouge_l)

        # Clinical relevance: Jaccard similarity between predicted summary and
        # source note/context
        self.register_metric("clinical_relevance", self._metric_jaccard)

        # Factual consistency: token-overlap F1 between predicted and reference summary
        self.register_metric("factual_consistency", self._metric_text_f1)

    # ----------------------------
    # Helpers for text normalization
    # ----------------------------
    @staticmethod
    def _normalize_text(s: Any) -> str:
        if s is None:
            return ""
        text = str(s)
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        # remove simple punctuation
        text = re.sub(r"[\.,!?;:\-\(\)\[\]\{\}\'\"]", "", text)
        return text

    @staticmethod
    def _tokenize(s: Any) -> List[str]:
        norm = MetricCalculator._normalize_text(s)
        return norm.split() if norm else []

    # ----------------------------
    # Metric implementations
    # ----------------------------
    def _metric_clinical_correctness(
        self, y_true: List[Any], y_pred: List[Any], **kwargs: Any
    ) -> float:
        """Binary correctness via normalized exact or substring match of answers."""
        scores: List[float] = []
        for t, p in zip(y_true, y_pred):
            t_norm = self._normalize_text(t)
            p_norm = self._normalize_text(p)
            if not t_norm and not p_norm:
                scores.append(1.0)
            elif not t_norm or not p_norm:
                scores.append(0.0)
            elif t_norm == p_norm or (t_norm in p_norm) or (p_norm in t_norm):
                scores.append(1.0)
            else:
                scores.append(0.0)
        return float(sum(scores) / len(scores)) if scores else float("nan")

    def _metric_text_f1(
        self, y_true: List[Any], y_pred: List[Any], **kwargs: Any
    ) -> float:
        """Simple token-overlap F1 score averaged across examples."""

        def f1_score_text(a: Any, b: Any) -> float:
            a_tokens = self._tokenize(a)
            b_tokens = self._tokenize(b)
            if not a_tokens and not b_tokens:
                return 1.0
            if not a_tokens or not b_tokens:
                return 0.0
            common = 0
            b_count: Dict[str, int] = {}
            for tok in b_tokens:
                b_count[tok] = b_count.get(tok, 0) + 1
            for tok in a_tokens:
                if b_count.get(tok, 0) > 0:
                    common += 1
                    b_count[tok] -= 1
            precision = common / len(b_tokens)
            recall = common / len(a_tokens)
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)

        scores = [f1_score_text(t, p) for t, p in zip(y_true, y_pred)]
        return float(sum(scores) / len(scores)) if scores else float("nan")

    def _metric_jaccard(
        self, y_true: List[Any], y_pred: List[Any], **kwargs: Any
    ) -> float:
        """Token Jaccard similarity averaged across examples."""
        scores: List[float] = []
        for t, p in zip(y_true, y_pred):
            set_t = set(self._tokenize(t))
            set_p = set(self._tokenize(p))
            if not set_t and not set_p:
                scores.append(1.0)
            elif not set_t or not set_p:
                scores.append(0.0)
            else:
                inter = len(set_t & set_p)
                union = len(set_t | set_p)
                scores.append(inter / union if union else 0.0)
        return float(sum(scores) / len(scores)) if scores else float("nan")

    def _metric_rouge_l(
        self, y_true: List[Any], y_pred: List[Any], **kwargs: Any
    ) -> float:
        """Compute average ROUGE-L F1 using rouge-score. Returns NaN if unavailable."""
        try:
            from rouge_score import rouge_scorer
        except Exception as e:  # pragma: no cover - soft dependency
            logger.error(f"rouge-score not available: {e}")
            return float("nan")

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores: List[float] = []
        for t, p in zip(y_true, y_pred):
            t_s = str(t) if t is not None else ""
            p_s = str(p) if p is not None else ""
            try:
                res = scorer.score(t_s, p_s)
                scores.append(float(res["rougeL"].fmeasure))
            except Exception:
                scores.append(0.0)
        return float(sum(scores) / len(scores)) if scores else float("nan")

    def calculate_metrics(
        self,
        task_id: str,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        metric_names: Optional[List[str]] = None,
        **metric_kwargs: Any,
    ) -> Dict[str, MetricResult]:
        """Calculate metrics for the given predictions and references.

        Args:
            task_id: ID of the task being evaluated
            predictions: List of model predictions
            references: List of reference/ground truth values
            metric_names: List of metric names to calculate
                (None for all registered metrics)
            **metric_kwargs: Additional arguments to pass to metric functions

        Returns:
            Dictionary mapping metric names to MetricResult objects
        """
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty")

        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        # Filter metrics if specific ones are requested
        metrics_to_use: Dict[str, Dict[str, Any]] = self.metrics
        if metric_names is not None:
            metrics_to_use = {
                name: self.metrics[name]
                for name in metric_names
                if name in self.metrics
            }

        results: Dict[str, MetricResult] = {}

        for metric_name, metric_info in metrics_to_use.items():
            try:
                # Get metric function and default kwargs
                metric_info["function"]
                default_kwargs: Dict[str, Any] = metric_info.get("default_kwargs", {})

                # Merge default kwargs with any provided kwargs
                kwargs: Dict[str, Any] = {
                    **default_kwargs,
                    **metric_kwargs.get(metric_name, {}),
                }

                # Extract relevant data for this metric
                y_true, y_pred = self._prepare_data_for_metric(
                    metric_name, predictions, references, **kwargs
                )

                # Calculate the metric
                result = self._calculate_single_metric(
                    metric_name, metric_info, y_true, y_pred, **kwargs
                )

                results[metric_name] = result

            except Exception as e:
                logger.error(f"Error calculating metric {metric_name}: {str(e)}")
                results[metric_name] = MetricResult(
                    metric_name=metric_name,
                    value=float("nan"),
                    metadata={"error": str(e)},
                )

        return results

    def _calculate_single_metric(
        self,
        metric_name: str,
        metric_info: Dict[str, Any],
        y_true: List[Any],
        y_pred: List[Any],
        **kwargs: Any,
    ) -> MetricResult:
        """Calculate a single metric.

        Args:
            metric_name: Name of the metric
            metric_info: Dictionary containing the metric function and default kwargs
            y_true: List of true labels/values
            y_pred: List of predicted labels/values
            **kwargs: Additional keyword arguments for the metric function

        Returns:
            MetricResult object containing the metric value and any metadata
        """
        try:
            metric_func = metric_info["function"]
            default_kwargs = metric_info.get("default_kwargs", {}).copy()

            # Update default kwargs with any provided kwargs
            kwargs = {**default_kwargs, **kwargs}

            # Special handling for classification metrics
            if metric_name in ["precision", "recall", "f1"]:
                # For classification metrics, ensure we have the
                # correct average parameter
                if "average" not in kwargs:
                    # Default to 'binary' for binary classification,
                    # 'macro' for multi-class
                    num_classes = len(set(y_true + y_pred))
                    kwargs["average"] = "binary" if num_classes <= 2 else "macro"
                    avg = kwargs["average"]
                    logger.debug(
                        f"Using average='{avg}' for {metric_name} "
                        f"with {num_classes} classes"
                    )

            # Calculate the metric
            logger.debug(f"Calculating {metric_name} with kwargs: {kwargs}")
            result = metric_func(y_true, y_pred, **kwargs)

            # Handle metrics that return a tuple (value, additional_info)
            if isinstance(result, tuple) and len(result) == 2:
                value, additional_info = result
                return MetricResult(
                    metric_name=metric_name,
                    value=float(value),
                    metadata=(
                        additional_info
                        if isinstance(additional_info, dict)
                        else {"info": additional_info}
                    ),
                )

            # Handle single return value
            return MetricResult(
                metric_name=metric_name,
                value=float(result),
            )

        except Exception as e:
            logger.error(
                f"Error calculating metric {metric_name}: {str(e)}", exc_info=True
            )
            return MetricResult(
                metric_name=metric_name,
                value=float("nan"),
                metadata={"error": str(e)},
            )

    def _prepare_data_for_metric(
        self,
        metric_name: str,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Tuple[List[Any], List[Any]]:
        """Prepare data for metric calculation.

        Args:
            metric_name: Name of the metric being calculated
            predictions: List of prediction dictionaries
            references: List of reference dictionaries
            **kwargs: Additional arguments that might be needed for data preparation

        Returns:
            Tuple of (y_true, y_pred) for the metric function
        """
        # Default behavior: extract 'label' from references and handle both
        # 'label' and 'prediction' from predictions. Some metrics override
        # fields to better fit task types (QA, diagnostic, summarization).
        try:
            # Choose reference and prediction fields based on metric
            ref_field = "label"
            pred_field_candidates = ["label", "prediction", "text", "summary"]

            if metric_name in ["clinical_correctness"]:
                ref_field = "answer"
                pred_field_candidates = ["label", "prediction", "answer", "text"]
            elif metric_name in ["diagnostic_accuracy"]:
                # by default use 'label', fallback to 'diagnosis'
                ref_field = "label"
            elif metric_name in ["rouge_l", "factual_consistency"]:
                # compare against reference summary
                ref_field = "summary"
                pred_field_candidates = ["summary", "prediction", "text", "label"]
            elif metric_name in ["clinical_relevance"]:
                # compare predicted summary to source note/context
                ref_field = "note"
                pred_field_candidates = ["summary", "prediction", "text", "label"]
            elif metric_name in ["reasoning_quality"]:
                ref_field = "rationale"
                pred_field_candidates = [
                    "rationale",
                    "explanation",
                    "prediction",
                    "label",
                    "text",
                ]

            # Extract true values from references with fallback chain
            y_true: List[Any] = []
            for ref in references:
                if isinstance(ref, dict):
                    val = ref.get(ref_field)
                    if val is None and ref_field == "label":
                        val = ref.get("diagnosis")
                    if val is None and ref_field == "rationale":
                        val = ref.get("explanation")
                    if val is None and ref_field == "summary":
                        val = ref.get("reference_summary")
                    y_true.append(val)
                else:
                    y_true.append(None)

            # Extract predictions using candidate fields in order
            y_pred: List[Any] = []
            for pred in predictions:
                if isinstance(pred, dict):
                    chosen = None
                    for f in pred_field_candidates:
                        if f in pred and pred.get(f) not in (None, ""):
                            chosen = pred.get(f)
                            break
                    y_pred.append(chosen)
                else:
                    y_pred.append(pred)

            # Handle binary classification metrics that require probabilities
            if metric_name in ["roc_auc", "average_precision"]:
                # For binary classification, we expect probabilities in the prediction
                if all(isinstance(p, dict) and "score" in p for p in predictions):
                    y_pred = [p.get("score") for p in predictions]

                # Ensure binary labels are 0/1 for scikit-learn metrics
                unique_labels = list(set(y_true))
                if len(unique_labels) == 2 and set(y_true) != {0, 1}:
                    unique_labels = sorted(unique_labels)
                    y_true = [1 if label == unique_labels[1] else 0 for label in y_true]

            # Ensure all labels are numeric for classification metrics only
            if (
                metric_name
                in ["accuracy", "precision", "recall", "f1", "diagnostic_accuracy"]
                and y_true
                and y_pred
            ):
                # Convert string labels to integers if needed
                if isinstance(y_true[0], str) or isinstance(y_pred[0], str):
                    all_labels = sorted(
                        set(
                            [x for x in y_true if x is not None]
                            + [x for x in y_pred if x is not None]
                        )
                    )
                    label_to_int = {label: i for i, label in enumerate(all_labels)}
                    y_true = [
                        label_to_int.get(label, -1) if label is not None else -1
                        for label in y_true
                    ]
                    y_pred = [
                        label_to_int.get(label, -1) if label is not None else -1
                        for label in y_pred
                    ]

            return y_true, y_pred

        except Exception as e:
            logger.warning(f"Error preparing data for metric {metric_name}: {str(e)}")
            return [], []

    def aggregate_metrics(
        self, metric_results: List[Dict[str, MetricResult]], aggregation: str = "mean"
    ) -> Dict[str, MetricResult]:
        """Aggregate metrics across multiple evaluation runs.

        Args:
            metric_results: List of metric result dictionaries
            aggregation: Aggregation method ('mean', 'sum', 'max', 'min')

        Returns:
            Dictionary of aggregated metric results
        """
        if not metric_results:
            return {}

        # Get all unique metric names
        all_metric_names: set[str] = set()
        for result in metric_results:
            all_metric_names.update(result.keys())

        aggregated_metrics: Dict[str, MetricResult] = {}

        for metric_name in all_metric_names:
            # Collect all values for this metric
            values: List[float] = []
            weights: List[int] = []
            metadata_list: List[Dict[str, Any]] = []

            for result in metric_results:
                if metric_name in result:
                    metric_result = result[metric_name]
                    values.append(metric_result.value)
                    weights.append(
                        metric_result.metadata.get("num_samples", 1)
                        if metric_result.metadata
                        else 1
                    )
                    if metric_result.metadata:
                        metadata_list.append(metric_result.metadata)

            if not values:
                continue

            # Calculate aggregated value
            agg_value: float
            if aggregation == "mean":
                if all(w == 1 for w in weights):
                    # Simple average if all weights are 1
                    agg_value = sum(values) / len(values)
                else:
                    # Weighted average
                    total_weight = sum(weights)
                    if total_weight > 0:
                        agg_value = (
                            sum(v * w for v, w in zip(values, weights)) / total_weight
                        )
                    else:
                        agg_value = float("nan")
            elif aggregation == "sum":
                agg_value = sum(values)
            elif aggregation == "max":
                agg_value = max(values)
            elif aggregation == "min":
                agg_value = min(values)
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregation}")

            # Calculate total samples across all runs
            total_samples = sum(
                meta.get("num_samples", 1) for meta in metadata_list if meta is not None
            )

            # Create aggregated metadata
            agg_metadata: Dict[str, Any] = {
                "aggregation": aggregation,
                "num_runs": len(values),
                "total_samples": total_samples,
                "original_metadata": metadata_list,
            }

            # Add min/max/mean if we have multiple values
            if len(values) > 1:
                mean_value = sum(values) / len(values)
                agg_metadata.update(
                    {
                        "min_value": min(values),
                        "max_value": max(values),
                        "mean_value": mean_value,
                        "std_dev": (
                            (sum((x - mean_value) ** 2 for x in values) / len(values))
                            ** 0.5
                            if len(values) > 1
                            else 0.0
                        ),
                    }
                )

            aggregated_metrics[metric_name] = MetricResult(
                metric_name=metric_name,
                value=float(agg_value),
                metadata=agg_metadata,
            )

        return aggregated_metrics
