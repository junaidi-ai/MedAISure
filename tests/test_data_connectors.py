from __future__ import annotations

import csv
import json
import gzip
import zipfile
import io
from pathlib import Path


from bench.data import (
    JSONDataset,
    CSVDataset,
    DataPreprocessor,
    SecureDataHandler,
    ValidationError,
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


def test_json_dataset_gzip_and_zip(tmp_path: Path):
    data = [{"id": 1, "t": "a"}, {"id": 2, "t": "b"}]

    # gzip
    gz = tmp_path / "data.json.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    ds_gz = JSONDataset(gz, required_keys=["id", "t"])
    items_gz = list(ds_gz.load_data())
    assert items_gz == data

    # zip
    zf = tmp_path / "archive.zip"
    with zipfile.ZipFile(zf, "w") as zipf:
        zipf.writestr("nested.json", json.dumps(data))
    ds_zip = JSONDataset(zf, required_keys=["id", "t"])
    items_zip = list(ds_zip.load_data())
    assert items_zip == data


def test_csv_dataset_gzip_and_zip(tmp_path: Path):
    rows = [
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
    ]

    # gzip
    gz = tmp_path / "data.csv.gz"
    with gzip.open(gz, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["a", "b"])
        writer.writeheader()
        writer.writerows(rows)
    ds_gz = CSVDataset(gz, required_keys=["a", "b"])
    items_gz = list(ds_gz.load_data())
    assert items_gz == rows

    # zip
    zf = tmp_path / "archive_csv.zip"
    # Prepare CSV content in-memory
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["a", "b"])
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(zf, "w") as zipf:
        zipf.writestr("nested.csv", buf.getvalue())
    ds_zip = CSVDataset(zf, required_keys=["a", "b"])
    items_zip = list(ds_zip.load_data())
    assert items_zip == rows


def test_required_keys_validation(tmp_path: Path):
    # JSON missing key
    data = [{"id": 1, "t": "a"}, {"t": "missing-id"}]
    f = tmp_path / "data.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    ds = JSONDataset(f, required_keys=["id", "t"])
    try:
        list(ds.load_data())
        assert False, "Expected ValidationError for missing key in JSON"
    except ValidationError:
        pass

    # CSV missing key
    fcsv = tmp_path / "data.csv"
    with fcsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["a", "b"])
        writer.writeheader()
        writer.writerow({"a": "1", "b": "2"})
        writer.writerow({"a": "3"})  # missing 'b'
    dsc = CSVDataset(fcsv, required_keys=["a", "b"])
    try:
        list(dsc.load_data())
        assert False, "Expected ValidationError for missing key in CSV"
    except ValidationError:
        pass


def test_iter_batches(tmp_path: Path):
    data = [{"id": i} for i in range(5)]
    f = tmp_path / "data.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    ds = JSONDataset(f, required_keys=["id"])

    batches = list(ds.iter_batches(2))
    # Expect 3 batches: [0,1], [2,3], [4]
    assert len(batches) == 3
    assert [x["id"] for x in batches[0]] == [0, 1]
    assert [x["id"] for x in batches[1]] == [2, 3]
    assert [x["id"] for x in batches[2]] == [4]
