from unittest.mock import patch

import pytest

from bench.evaluation.task_loader import TaskLoader


class FakeResp:
    def __init__(
        self, status: int = 200, text: str = "", ctype: str = "application/json"
    ):
        self.status_code = status
        self.text = text
        self.headers = {"Content-Type": ctype}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            from requests import HTTPError

            raise HTTPError(f"status {self.status_code}")


def test_url_bad_content_type_raises_value_error():
    loader = TaskLoader(tasks_dir="tasks")

    with patch("bench.evaluation.task_loader.requests.get") as mget:
        mget.return_value = FakeResp(
            status=200, text="<html></html>", ctype="text/html"
        )
        with pytest.raises(ValueError):
            loader.load_task("https://example.com/taskfile")


def test_url_404_propagates_http_error():
    loader = TaskLoader(tasks_dir="tasks")

    with patch("bench.evaluation.task_loader.requests.get") as mget:
        mget.return_value = FakeResp(
            status=404, text="not found", ctype="application/json"
        )
        with pytest.raises(Exception):  # requests.HTTPError
            loader.load_task("https://example.com/task.json")


def test_url_network_exception_propagates():
    loader = TaskLoader(tasks_dir="tasks")

    with patch("bench.evaluation.task_loader.requests.get") as mget:
        import requests

        mget.side_effect = requests.RequestException("boom")
        with pytest.raises(requests.RequestException):
            loader.load_task("https://example.com/task.yaml")


def test_malformed_yaml_raises_value_error():
    loader = TaskLoader(tasks_dir="tasks")

    bad_yaml = "name: test\nmetrics: [accuracy\n"  # missing closing bracket
    with patch("bench.evaluation.task_loader.requests.get") as mget:
        mget.return_value = FakeResp(
            status=200, text=bad_yaml, ctype="application/x-yaml"
        )
        with pytest.raises(ValueError):
            loader.load_task("https://example.com/task.yaml")


def test_malformed_json_raises_value_error():
    loader = TaskLoader(tasks_dir="tasks")

    bad_json = "{"  # invalid json
    with patch("bench.evaluation.task_loader.requests.get") as mget:
        mget.return_value = FakeResp(
            status=200, text=bad_json, ctype="application/json"
        )
        with pytest.raises(ValueError):
            loader.load_task("https://example.com/task.json")
