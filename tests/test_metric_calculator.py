"""Tests for the MetricCalculator component."""
import pytest
import numpy as np
from bench.evaluation.metric_calculator import MetricCalculator, MetricResult

@pytest.fixture
def metric_calculator():
    """Create a MetricCalculator instance for testing."""
    return MetricCalculator()

def test_register_metric(metric_calculator):
    """Test registering a custom metric."""
    def custom_metric(y_true, y_pred, **kwargs):
        return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    
    metric_calculator.register_metric("custom_accuracy", custom_metric)
    assert "custom_accuracy" in metric_calculator.metrics
    assert metric_calculator.metrics["custom_accuracy"]["function"] == custom_metric

def test_calculate_metrics_basic(metric_calculator):
    """Test basic metric calculation."""
    # Test data - using numerical labels for simplicity
    task_id = "test_task"
    predictions = [
        {"label": 1, "score": 0.9},  # Correct (1 == 1)
        {"label": 0, "score": 0.8},  # Incorrect (0 != 1)
        {"label": 1, "score": 0.7}   # Correct (1 == 1)
    ]
    references = [
        {"label": 1},  # Positive class
        {"label": 1},  # Positive class (incorrect prediction)
        {"label": 1}   # Positive class
    ]
    
    # Calculate metrics
    results = metric_calculator.calculate_metrics(
        task_id=task_id,
        predictions=predictions,
        references=references,
        metric_names=["accuracy", "precision", "recall", "f1"]
    )
    
    # Check results
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    
    # Check accuracy (2 out of 3 correct)
    assert abs(results["accuracy"].value - 0.6667) < 0.0001

def test_metric_aggregation(metric_calculator):
    """Test aggregating metrics across multiple runs."""
    # Create sample metric results
    metric_results = [
        {"accuracy": MetricResult("accuracy", 0.8, {"num_samples": 100})},
        {"accuracy": MetricResult("accuracy", 0.9, {"num_samples": 200})},
        {"accuracy": MetricResult("accuracy", 0.7, {"num_samples": 300})},
    ]
    
    # Aggregate metrics
    aggregated = metric_calculator.aggregate_metrics(metric_results)
    
    # Check results
    assert "accuracy" in aggregated
    agg_result = aggregated["accuracy"]
    assert abs(agg_result.value - 0.7833) < 0.0001  # Weighted average
    assert agg_result.metadata["num_runs"] == 3
    assert agg_result.metadata["total_samples"] == 600
    assert "min_value" in agg_result.metadata
    assert "max_value" in agg_result.metadata
    assert "std_dev" in agg_result.metadata

def test_prepare_data_for_metric(metric_calculator):
    """Test data preparation for metric calculation."""
    # Test data with numerical labels for simplicity
    predictions = [
        {"label": 1, "score": 0.9},  # Positive class
        {"label": 0, "score": 0.8},  # Negative class
        {"label": 1, "score": 0.7}   # Positive class
    ]
    references = [
        {"label": 1},  # Positive
        {"label": 0},  # Negative
        {"label": 0}   # Negative (different from prediction)
    ]
    
    # Test with different metric types
    for metric_name in ["accuracy", "f1", "roc_auc"]:
        y_true, y_pred = metric_calculator._prepare_data_for_metric(
            metric_name=metric_name,
            predictions=predictions,
            references=references
        )
        
        if metric_name == "roc_auc":
            # For roc_auc, we expect scores for the positive class
            assert len(y_true) == 3
            assert len(y_pred) == 3
            assert all(isinstance(score, float) for score in y_pred)
        else:
            # For classification metrics, we expect class labels
            assert len(y_true) == 3
            assert len(y_pred) == 3
            assert all(isinstance(label, (int, float)) for label in y_pred)
