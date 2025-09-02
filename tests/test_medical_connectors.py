from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict


import pytest

from bench.data.medical import MIMICConnector, PubMedConnector


def _mk_sqlite(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("create table items (id integer primary key, val text, name text)")
        cur.executemany(
            "insert into items (id, val, name) values (?, ?, ?)",
            [(1, "a", "Alice"), (2, "b", "Bob"), (3, "c", "Carol")],
        )
        con.commit()
    finally:
        con.close()


def test_mimic_sqlite_basic(tmp_path: Path):
    db = tmp_path / "m.db"
    _mk_sqlite(db)
    conn = f"sqlite:///{db}"
    # Base query with filter and limit
    ds = MIMICConnector(
        conn, "select id, val, name from items", filters={"id": 2}, limit=5
    )
    # Require id & val, not name (which is PHI redacted)
    ds.required_keys = ["id", "val"]

    rows = list(ds.load_data())
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == 2 and r["val"] == "b"
    # PHI field 'name' should be redacted, not removed
    assert r.get("name") == "[REDACTED]"

    meta = ds.get_metadata()
    assert meta["source"] == "MIMIC"
    assert "sqlite:///*/" in meta["connection"]


class _Resp:
    def __init__(self, data: Dict[str, Any], status: int = 200):
        self._data = data
        self.status_code = status

    def json(self) -> Dict[str, Any]:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_pubmed_connector_mock(monkeypatch: pytest.MonkeyPatch):
    # Make deterministic mock for requests.get
    def fake_get(url: str, params: Dict[str, Any], timeout: float = 30):
        if "esearch.fcgi" in url:
            return _Resp({"esearchresult": {"idlist": ["111", "222"]}})
        if "esummary.fcgi" in url:
            data = {
                "result": {
                    "uids": ["111", "222"],
                    "111": {
                        "pmid": "111",
                        "title": "T1",
                        "pubdate": "2020",
                        "source": "J1",
                        "authors": [{"name": "Doe J"}],
                    },
                    "222": {
                        "pmid": "222",
                        "title": "T2",
                        "pubdate": "2021",
                        "source": "J2",
                        "authors": [{"name": "Roe J"}],
                    },
                }
            }
            return _Resp(data)
        raise AssertionError("Unexpected URL")

    import requests as _requests

    monkeypatch.setattr(_requests, "get", fake_get)

    ds = PubMedConnector(
        ["cancer", "imaging"], max_results=5, api_key=None, min_interval=0.0
    )
    # Expected fields present; none in required_keys to avoid redaction issues
    ds.required_keys = ["pmid", "title"]

    items = list(ds.load_data())
    assert len(items) == 2
    assert {i["pmid"] for i in items} == {"111", "222"}
    assert all("title" in i for i in items)

    meta = ds.get_metadata()
    assert meta["source"] == "PubMed"
    assert meta["uses_api_key"] is False
