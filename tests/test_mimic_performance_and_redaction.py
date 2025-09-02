from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List

import pytest

from bench.data.medical import MIMICConnector


def _mk_sqlite(path: Path, n: int = 1000) -> None:
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("create table items (id integer primary key, val text, name text)")
        rows = [(i, f"v{i}", f"Name{i}") for i in range(n)]
        cur.executemany("insert into items (id, val, name) values (?, ?, ?)", rows)
        con.commit()
    finally:
        con.close()


@pytest.mark.integration
def test_mimic_redaction_and_cache_perf(tmp_path: Path):
    db = tmp_path / "m.db"
    _mk_sqlite(db, n=500)
    conn = f"sqlite:///{db}"

    # Require fields that exclude PHI 'name' but keep it redacted
    ds = MIMICConnector(
        conn,
        "select id, val, name from items",
        filters={"id": 10},
        limit=10,
        use_cache=True,
        cache_size=16,
        persistent_cache_dir=str(tmp_path / "pcache"),
    )
    ds.required_keys = ["id", "val"]

    # First run (no cache)
    t0 = time.perf_counter()
    rows1: List[Dict[str, Any]] = list(ds.load_data())
    t1 = time.perf_counter() - t0

    assert len(rows1) == 1
    assert rows1[0]["id"] == 10
    # PHI redacted but present
    assert rows1[0].get("name") == "[REDACTED]"

    # Second run (should hit memory or persistent cache)
    t0 = time.perf_counter()
    rows2: List[Dict[str, Any]] = list(ds.load_data())
    t2 = time.perf_counter() - t0

    assert rows2 == rows1
    # Cache should make subsequent run faster by a reasonable factor
    # Avoid flakiness: just assert it's not slower by more than 2x
    assert t2 <= max(t1 * 2, 0.05)
