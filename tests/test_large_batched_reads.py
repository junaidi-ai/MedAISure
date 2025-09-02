from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pytest

from bench.data.local import JSONDataset, CSVDataset


def _make_large_json(path: Path, n: int = 10000) -> None:
    data = [{"id": i, "text": f"t{i}"} for i in range(n)]
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_large_csv(path: Path, n: int = 8000) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "text"])
        w.writeheader()
        w.writerows({"id": str(i), "text": f"t{i}"} for i in range(n))


@pytest.mark.parametrize("batch_size", [1, 64, 512, 2048])
@pytest.mark.parametrize("variant", ["plain", "gzip", "zip"])  # compression variants
def test_large_json_iter_batches(tmp_path: Path, batch_size: int, variant: str):
    n = 10000
    src = tmp_path / "large.json"
    _make_large_json(src, n=n)

    # Possibly wrap in gzip/zip
    path = src
    if variant == "gzip":
        import gzip as _gzip

        gz = tmp_path / "large.json.gz"
        with _gzip.open(gz, "wt", encoding="utf-8") as fh:
            fh.write(src.read_text(encoding="utf-8"))
        path = gz
    elif variant == "zip":
        import zipfile as _zip

        zf = tmp_path / "large.zip"
        with _zip.ZipFile(zf, "w") as z:
            z.writestr("nested.json", src.read_text(encoding="utf-8"))
        path = zf

    ds = JSONDataset(path, required_keys=["id", "text"])

    t0 = time.perf_counter()
    total = 0
    for batch in ds.iter_batches(batch_size):
        assert 1 <= len(batch) <= batch_size
        total += len(batch)
    dt = time.perf_counter() - t0

    assert total == n
    # Modest upper bound that is CI-safe but tight enough to catch regressions
    assert dt < 5.0


@pytest.mark.parametrize("batch_size", [1, 128, 1024])
@pytest.mark.parametrize("variant", ["plain", "gzip", "zip"])  # compression variants
def test_large_csv_iter_batches(tmp_path: Path, batch_size: int, variant: str):
    n = 8000
    src = tmp_path / "large.csv"
    _make_large_csv(src, n=n)

    path = src
    if variant == "gzip":
        import gzip as _gzip

        gz = tmp_path / "large.csv.gz"
        with _gzip.open(gz, "wt", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "text"])
            w.writeheader()
            w.writerows({"id": str(i), "text": f"t{i}"} for i in range(n))
        path = gz
    elif variant == "zip":
        import zipfile as _zip

        zf = tmp_path / "large_csv.zip"
        # write CSV content to nested file
        from io import StringIO

        buf = StringIO()
        w = csv.DictWriter(buf, fieldnames=["id", "text"])
        w.writeheader()
        w.writerows({"id": str(i), "text": f"t{i}"} for i in range(n))
        with _zip.ZipFile(zf, "w") as z:
            z.writestr("nested.csv", buf.getvalue())
        path = zf

    ds = CSVDataset(path, required_keys=["id", "text"])

    t0 = time.perf_counter()
    total = 0
    for batch in ds.iter_batches(batch_size):
        assert 1 <= len(batch) <= batch_size
        total += len(batch)
    dt = time.perf_counter() - t0

    assert total == n
    assert dt < 5.0
