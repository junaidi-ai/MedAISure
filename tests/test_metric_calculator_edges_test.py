import math
from typing import Any, Dict, List


from bench.evaluation.metric_calculator import MetricCalculator


def _mk_refs(labels: List[Any]) -> List[Dict[str, Any]]:
    return [{"label": label} for label in labels]


def _mk_scores(scores: List[Any]) -> List[Dict[str, Any]]:
    return [{"score": s} for s in scores]


class TestMetricCalculatorEdgeCases:
    def setup_method(self) -> None:
        self.calc = MetricCalculator()

    def test_roc_auc_single_class_returns_nan_with_warning(self):
        refs = _mk_refs([1, 1, 1, 1])
        preds = _mk_scores([0.9, 0.8, 0.7, 0.6])

        res = self.calc.calculate_metrics(
            task_id="t1",
            predictions=preds,
            references=refs,
            metric_names=["roc_auc"],
        )["roc_auc"]

        assert math.isnan(res.value)
        assert "warning" in (res.metadata or {})

    def test_average_precision_empty_after_filter_nan(self):
        refs = _mk_refs([None, float("nan")])
        preds = _mk_scores([None, float("nan")])

        res = self.calc.calculate_metrics(
            task_id="t2",
            predictions=preds,
            references=refs,
            metric_names=["average_precision"],
        )["average_precision"]

        assert math.isnan(res.value)
        assert (res.metadata or {}).get("warning") == "empty data after filtering"

    def test_roc_auc_two_class_valid(self):
        # Balanced binary labels with probabilities
        refs = _mk_refs([0, 1, 0, 1])
        preds = _mk_scores([0.1, 0.9, 0.2, 0.8])

        res = self.calc.calculate_metrics(
            task_id="t3",
            predictions=preds,
            references=refs,
            metric_names=["roc_auc"],
        )["roc_auc"]

        assert 0.0 <= res.value <= 1.0
