import pytest

from bench.evaluation.metrics import Metric
from bench.evaluation.metrics.clinical import ClinicalAccuracyMetric


def test_metric_is_abstract():
    with pytest.raises(TypeError):
        # type: ignore[abstract]
        Metric()  # Cannot instantiate abstract class


def test_validate_inputs_type_and_shape_checks():
    # Not lists
    with pytest.raises(TypeError):
        Metric.validate_inputs({}, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Metric.validate_inputs([], {})  # type: ignore[arg-type]

    # Empty lists
    with pytest.raises(ValueError):
        Metric.validate_inputs([], [])

    # Length mismatch
    with pytest.raises(ValueError):
        Metric.validate_inputs([{}], [{}, {}])

    # Non-dict elements
    with pytest.raises(TypeError):
        Metric.validate_inputs([{}], ["bad"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        Metric.validate_inputs(["bad"], [{}])  # type: ignore[list-item]


def test_concrete_metric_uses_validation():
    m = ClinicalAccuracyMetric()
    # Expect ValueError due to empty lists
    with pytest.raises(ValueError):
        m.calculate([], [])  # type: ignore[arg-type]
