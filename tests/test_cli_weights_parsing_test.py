import pytest

from bench.cli_typer import _parse_weights, _validate_weights


def test_parse_weights_json():
    s = '{"diagnostics":0.4, "safety":0.3, "communication":0.2, "summarization":0.1}'
    out = _parse_weights(s)
    assert out == {
        "diagnostics": 0.4,
        "safety": 0.3,
        "communication": 0.2,
        "summarization": 0.1,
    }


def test_parse_weights_key_value():
    s = "diagnostics=0.4,safety=0.3, communication=0.2 , summarization=0.1"
    out = _parse_weights(s)
    assert out == {
        "diagnostics": 0.4,
        "safety": 0.3,
        "communication": 0.2,
        "summarization": 0.1,
    }


def test_parse_weights_invalid_token():
    with pytest.raises(Exception):
        _parse_weights("diagnostics0.4")  # missing equals


def test_parse_weights_non_numeric():
    with pytest.raises(Exception):
        _parse_weights("diagnostics=abc")


def test_validate_weights_sum_one_ok():
    w = {"a": 0.5, "b": 0.5}
    out = _validate_weights(w)
    assert out == {"a": 0.5, "b": 0.5}


def test_validate_weights_sum_one_required_error():
    with pytest.raises(Exception):
        _validate_weights({"a": 0.6, "b": 0.5})


def test_validate_weights_non_negative():
    with pytest.raises(Exception):
        _validate_weights({"a": -0.1, "b": 1.1})
