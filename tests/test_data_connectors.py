from __future__ import annotations

import csv
import json
from pathlib import Path


from bench.data import (
    JSONDataset,
    CSVDataset,
    DataPreprocessor,
    SecureDataHandler,
)


def test_json_dataset_loading(tmp_path: Path):
    data = [{"a": "1", "b": 2}, {"a": "3", "b": 4}]
    f = tmp_path / "data.json"
    f.write_text(json.dumps(data), encoding="utf-8")

    ds = JSONDataset(f)
    items = list(ds.load_data())

    assert items == data
    meta = ds.get_metadata()
    assert meta["format"] == "json"
    assert meta["source"].endswith("data.json")


def test_csv_dataset_loading(tmp_path: Path):
    f = tmp_path / "data.csv"
    rows = [
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
    ]
    with f.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["a", "b"])
        writer.writeheader()
        writer.writerows(rows)

    ds = CSVDataset(f)
    items = list(ds.load_data())

    assert items == rows
    meta = ds.get_metadata()
    assert meta["format"] == "csv"
    assert meta["source"].endswith("data.csv")


def test_preprocessor_steps():
    p = DataPreprocessor()

    def step1(d):
        d = dict(d)
        d["x"] = d.get("x", 0) + 1
        return d

    def step2(d):
        d = dict(d)
        d["y"] = d.get("x", 0) * 2
        return d

    p.add_step(step1)
    p.add_step(step2)

    out = p.process({"x": 1})
    assert out == {"x": 2, "y": 4}

    out_batch = p.process_batch([{"x": 0}, {"x": 5}])
    assert out_batch == [{"x": 1, "y": 2}, {"x": 6, "y": 12}]


def test_secure_data_handler_roundtrip():
    handler = SecureDataHandler("secret-key")
    src = {"a": "hello", "b": 5}
    enc = handler.encrypt_data(src)
    assert enc["b"] == 5 and enc["a"] != "hello"
    dec = handler.decrypt_data(enc)
    assert dec == src
