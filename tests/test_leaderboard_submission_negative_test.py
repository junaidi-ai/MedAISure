import pytest

from bench.leaderboard.submission import validate_submission


def test_missing_top_level_fields():
    # Missing model_id and submissions
    bad = {
        "schema_version": 1,
        "run_id": "r1",
        "created_at": "2025-01-01T00:00:00Z",
    }
    with pytest.raises(Exception):
        validate_submission(bad)


def test_empty_submissions():
    bad = {
        "schema_version": 1,
        "run_id": "r1",
        "model_id": "m1",
        "created_at": "2025-01-01T00:00:00Z",
        "submissions": [],
    }
    with pytest.raises(Exception):
        validate_submission(bad)


def test_missing_items_array():
    bad = {
        "schema_version": 1,
        "run_id": "r1",
        "model_id": "m1",
        "created_at": "2025-01-01T00:00:00Z",
        "submissions": [{"task_id": "t1"}],
    }
    with pytest.raises(Exception):
        validate_submission(bad)


def test_item_missing_prediction():
    bad = {
        "schema_version": 1,
        "run_id": "r1",
        "model_id": "m1",
        "created_at": "2025-01-01T00:00:00Z",
        "submissions": [
            {
                "task_id": "t1",
                "items": [{"input_id": "i1"}],
            }
        ],
    }
    with pytest.raises(Exception):
        validate_submission(bad)


def test_item_prediction_wrong_type():
    bad = {
        "schema_version": 1,
        "run_id": "r1",
        "model_id": "m1",
        "created_at": "2025-01-01T00:00:00Z",
        "submissions": [
            {
                "task_id": "t1",
                "items": [{"input_id": "i1", "prediction": []}],
            }
        ],
    }
    with pytest.raises(Exception):
        validate_submission(bad)


def test_reasoning_wrong_type_when_present():
    bad = {
        "schema_version": 1,
        "run_id": "r1",
        "model_id": "m1",
        "created_at": "2025-01-01T00:00:00Z",
        "submissions": [
            {
                "task_id": "t1",
                "items": [{"input_id": "i1", "prediction": {"y": 1}, "reasoning": 123}],
            }
        ],
    }
    with pytest.raises(Exception):
        validate_submission(bad)
