import math

import pytest

from bench.evaluation.metrics import (
    MetricRegistry,
    get_default_registry,
)


def sample_io():
    expected = [
        {"diagnosis": "mi", "specialty": "cardiology", "note": "chest pain troponin"},
        {
            "diagnosis": "sepsis",
            "specialty": "infectious_disease",
            "note": "fever tachycardia",
        },
    ]
    outputs = [
        {
            "prediction": "myocardial infarction",
            "summary": "chest pain with elevated troponin",
        },
        {"prediction": "flu", "summary": "fever and tachycardia present"},
    ]
    return expected, outputs


def test_registry_register_and_duplicate_validation():
    reg = MetricRegistry()

    # register built-ins via helper
    default_reg = get_default_registry()
    # re-register one into fresh registry to test duplicate handling
    for name in ["clinical_accuracy", "reasoning_quality"]:
        reg.register_metric(default_reg.get_metric(name))

    with pytest.raises(ValueError):
        # duplicate
        reg.register_metric(default_reg.get_metric("clinical_accuracy"))


def test_calculate_metrics_and_missing_metric_error():
    reg = get_default_registry()
    e, o = sample_io()

    with pytest.raises(KeyError):
        reg.calculate_metrics(["nonexistent"], e, o)

    res = reg.calculate_metrics(
        ["clinical_accuracy", "diagnostic_accuracy", "clinical_relevance"], e, o
    )
    assert set(res.keys()) == {
        "clinical_accuracy",
        "diagnostic_accuracy",
        "clinical_relevance",
    }
    for v in res.values():
        assert isinstance(v, float) and not math.isnan(v)


def test_cache_behavior_and_info():
    reg = get_default_registry()
    e, o = sample_io()

    reg.clear_cache()
    assert reg.cache_info()["entries"] == 0

    names = ["clinical_accuracy", "diagnostic_accuracy"]
    res1 = reg.calculate_metrics(names, e, o, use_cache=True)
    c1 = reg.cache_info()["entries"]
    assert c1 >= 1

    # Second call should hit cache; results equal
    res2 = reg.calculate_metrics(names, e, o, use_cache=True)
    assert res1 == res2


def test_parallel_execution_yields_same_results():
    reg = get_default_registry()
    e, o = sample_io()
    names = [
        "clinical_accuracy",
        "diagnostic_accuracy",
        "clinical_relevance",
        "reasoning_quality",
    ]

    seq = reg.calculate_metrics(names, e, o, parallel=False, use_cache=False)
    par = reg.calculate_metrics(
        names, e, o, parallel=True, max_workers=4, use_cache=False
    )
    assert seq == par


def test_aggregation_and_serialization_roundtrip():
    # aggregation
    a = {"m1": 0.5, "m2": 1.0}
    b = {"m1": 0.7, "m2": 0.0}
    mean = MetricRegistry.aggregate_mean([a, b])
    assert pytest.approx(mean["m1"], rel=1e-6) == 0.6
    assert pytest.approx(mean["m2"], rel=1e-6) == 0.5

    # serialization
    payload = MetricRegistry.serialize_results(a)
    out = MetricRegistry.deserialize_results(payload)
    assert out == a
