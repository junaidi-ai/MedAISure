"""Metric calculator implementation for MEDDSAI benchmark."""

import logging
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
        # 'label' and 'prediction' from predictions
        try:
            # Extract true labels from references
            y_true: List[Any] = [ref.get("label") for ref in references]

            # Extract predictions - try 'label' first, then fall back to 'prediction'
            y_pred: List[Any] = []
            for pred in predictions:
                if isinstance(pred, dict):
                    if "label" in pred:
                        y_pred.append(pred["label"])
                    elif "prediction" in pred:
                        y_pred.append(pred["prediction"])
                    else:
                        y_pred.append(None)
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

            # Ensure all labels are numeric for classification metrics
            if (
                metric_name not in ["roc_auc", "average_precision"]
                and y_true
                and y_pred
            ):
                # Convert string labels to integers if needed
                if isinstance(y_true[0], str) or isinstance(y_pred[0], str):
                    all_labels = sorted(set(y_true + y_pred))
                    label_to_int = {label: i for i, label in enumerate(all_labels)}
                    y_true = [label_to_int.get(label, -1) for label in y_true]
                    y_pred = [label_to_int.get(label, -1) for label in y_pred]

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
