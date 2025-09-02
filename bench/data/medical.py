from __future__ import annotations

from typing import Dict, Iterator, Any, List, Optional, Iterable, Tuple
import time
import threading
import sqlite3
import re
from collections import OrderedDict
import requests
import os
import json
import hashlib
import logging
from pathlib import Path

from .base import DatasetConnector


class MIMICConnector(DatasetConnector):
    """
    Minimal, secure MIMIC-like connector with stdlib SQLite support.

    Features:
    - Parameterized queries with optional filters to avoid SQL injection
    - PHI redaction via configured denylist of fields
    - SQLite connection (sqlite:///path.db). Other schemes raise informative errors
    """

    def __init__(
        self,
        connection_string: str,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        phi_denylist: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        use_cache: bool = True,
        cache_size: int = 128,
        persistent_cache_dir: Optional[str] = None,
    ):
        # Do NOT store raw credentials in metadata; only store safe host/db info.
        self.connection_string = connection_string
        self.query = query
        self.filters = dict(filters or {})
        self.phi_denylist = set(phi_denylist or DEFAULT_PHI_DENYLIST)
        self.limit = limit
        self._use_cache = bool(use_cache)
        self._cache = _LRUCache(cache_size) if self._use_cache else None
        self._pcache_dir: Optional[Path] = (
            Path(persistent_cache_dir).expanduser() if persistent_cache_dir else None
        )
        if self._pcache_dir:
            self._pcache_dir.mkdir(parents=True, exist_ok=True)
        self._log = logging.getLogger(__name__)

    def _connect_sqlite(self) -> sqlite3.Connection:
        if not self.connection_string.startswith("sqlite:///"):
            raise ValueError(
                "Only sqlite:/// paths are supported in OSS build. Use a secure driver in production."
            )
        path = self.connection_string.replace("sqlite:///", "", 1)
        # Isolation, read-only mode when possible
        uri = f"file:{path}?mode=ro"
        return sqlite3.connect(uri, uri=True, check_same_thread=False)

    def _build_param_query_sqlite(self) -> Tuple[str, Tuple[Any, ...]]:
        # Append WHERE conditions from filters safely
        sql = self.query.strip()
        params: List[Any] = []
        if self.filters:
            # Determine if existing WHERE present
            if re.search(r"\bwhere\b", sql, re.IGNORECASE):
                sql += " AND "
            else:
                sql += " WHERE "
            clauses = []
            for k, v in self.filters.items():
                # simple equality filters; extend as needed
                clauses.append(f"{k} = ?")
                params.append(v)
            sql += " AND ".join(clauses)
        if self.limit is not None and self.limit > 0:
            sql += " LIMIT ?"
            params.append(int(self.limit))
        return sql, tuple(params)

    def _build_param_query_sa(self) -> Tuple[str, Dict[str, Any]]:
        sql = self.query.strip()
        params: Dict[str, Any] = {}
        if self.filters:
            if re.search(r"\bwhere\b", sql, re.IGNORECASE):
                sql += " AND "
            else:
                sql += " WHERE "
            clauses = []
            for k, v in self.filters.items():
                pname = re.sub(r"[^a-zA-Z0-9_]", "_", k)
                clauses.append(f"{k} = :{pname}")
                params[pname] = v
            sql += " AND ".join(clauses)
        if self.limit is not None and self.limit > 0:
            sql += " LIMIT :_limit"
            params["_limit"] = int(self.limit)
        return sql, params

    def _connect_sqlalchemy(self):
        try:
            from sqlalchemy import create_engine, text
        except Exception as e:
            raise ImportError(
                "SQLAlchemy is required for non-sqlite connections. Install 'sqlalchemy' and the appropriate DB driver."
            ) from e
        connect_args: Dict[str, Any] = {}
        # Basic, env-driven TLS hints (may vary by driver)
        sslmode = os.getenv("DB_SSLMODE") or os.getenv("PGSSLMODE")
        sslroot = os.getenv("DB_SSLROOTCERT") or os.getenv("PGSSLROOTCERT")
        if sslmode:
            connect_args["sslmode"] = sslmode
        if sslroot and os.path.exists(sslroot):
            # For psycopg2 (Postgres). MySQL drivers differ; users can override via DSN/query params.
            connect_args["sslrootcert"] = sslroot
        engine = create_engine(
            self.connection_string,
            pool_pre_ping=True,
            future=True,
            connect_args=connect_args,
        )
        return engine, text

    # ----------------------
    # Persistent cache utils
    # ----------------------
    def _key_to_path(self, key: Any) -> Optional[Path]:
        if not self._pcache_dir:
            return None
        h = hashlib.sha256(repr(key).encode()).hexdigest()
        return self._pcache_dir / f"mimic_{h}.json"

    def _read_persistent(self, key: Any) -> Optional[List[Dict[str, Any]]]:
        p = self._key_to_path(key)
        if not p or not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            self._log.debug("Failed to read persistent cache %s: %s", p, e)
        return None

    def _write_persistent(self, key: Any, rows: List[Dict[str, Any]]) -> None:
        p = self._key_to_path(key)
        if not p:
            return
        try:
            with p.open("w", encoding="utf-8") as f:
                json.dump(rows, f)
        except Exception as e:
            self._log.debug("Failed to write persistent cache %s: %s", p, e)

    def load_data(self) -> Iterator[Dict[str, Any]]:
        # Cache key independent of driver style
        # Warning: query string changes affect cache hit; this is by design.
        key: Any = None
        try:
            if self.connection_string.startswith("sqlite:///"):
                pass  # keep consistent
        except Exception:
            pass
        # Build SQL and params per backend
        if self.connection_string.startswith("sqlite:///"):
            sql, params = self._build_param_query_sqlite()
            key = ("sqlite", self.connection_string, sql, params)
            # Memory cache
            if self._use_cache and self._cache:
                cached = self._cache.get(key)
                if cached is not None:
                    self._log.debug("MIMIC memory cache hit (sqlite)")
                    for item in cached:
                        yield item
                    return
            # Persistent cache
            if self._pcache_dir:
                pc = self._read_persistent(key)
                if pc is not None:
                    self._log.debug("MIMIC persistent cache hit (sqlite)")
                    for item in pc:
                        yield item
                    # also warm memory cache
                    if self._use_cache and self._cache:
                        self._cache.set(key, pc)
                    return
            conn = self._connect_sqlite()
            rows: List[Dict[str, Any]] = []
            try:
                cur = conn.cursor()
                cur.execute(sql, params)
                col_names = [d[0] for d in cur.description or []]
                for row in cur:
                    item = {col_names[i]: row[i] for i in range(len(col_names))}
                    item = redact_phi(item, denylist=self.phi_denylist)
                    self.validate_item(item)
                    rows.append(item)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
            if self._use_cache and self._cache:
                self._cache.set(key, rows)
            if self._pcache_dir:
                self._write_persistent(key, rows)
            for it in rows:
                yield it
        else:
            # SQLAlchemy path (Postgres/MySQL/etc.)
            engine, text_fn = self._connect_sqlalchemy()
            sql, params = self._build_param_query_sa()
            key = ("sa", self.connection_string, sql, tuple(sorted(params.items())))
            # Memory cache
            if self._use_cache and self._cache:
                cached = self._cache.get(key)
                if cached is not None:
                    self._log.debug("MIMIC memory cache hit (SA)")
                    for item in cached:
                        yield item
                    return
            # Persistent cache
            if self._pcache_dir:
                pc = self._read_persistent(key)
                if pc is not None:
                    self._log.debug("MIMIC persistent cache hit (SA)")
                    for item in pc:
                        yield item
                    if self._use_cache and self._cache:
                        self._cache.set(key, pc)
                    return
            rows: List[Dict[str, Any]] = []
            with engine.connect() as conn:
                res = conn.execute(text_fn(sql), params)
                col_names = list(res.keys())
                for row in res:
                    # row is RowMapping when future=True
                    item = {col_names[i]: row[i] for i in range(len(col_names))}
                    item = redact_phi(item, denylist=self.phi_denylist)
                    self.validate_item(item)
                    rows.append(item)
            if self._use_cache and self._cache:
                self._cache.set(key, rows)
            if self._pcache_dir:
                self._write_persistent(key, rows)
            for it in rows:
                yield it

    def get_metadata(self) -> Dict[str, Any]:
        # Only expose non-sensitive parts
        conn_display = (
            self.connection_string.replace("sqlite:///", "sqlite:///*/")
            if self.connection_string.startswith("sqlite:///")
            else "unsupported"
        )
        return {
            "source": "MIMIC",
            "query": self.query[:200],
            "connection": conn_display,
            "filters": list(self.filters.keys()),
            "limit": self.limit,
        }


class PubMedConnector(DatasetConnector):
    """
    PubMed (NCBI eutils) connector using requests with caching and rate limiting.

    - Respects eutils rate limits (default 3 rps; with api_key ~10 rps)
    - Simple in-memory LRU cache keyed by (query, retmax)
    - Retries with exponential backoff on transient HTTP errors
    - Supports API key authentication via `api_key`
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def __init__(
        self,
        search_terms: List[str],
        max_results: int = 100,
        *,
        api_key: Optional[str] = None,
        min_interval: Optional[float] = None,
        cache_size: int = 16,
        phi_denylist: Optional[Iterable[str]] = None,
    ):
        self.search_terms = list(search_terms)
        self.max_results = max_results
        self.api_key = api_key
        # Rate limiting
        default_interval = 0.1 if api_key else 0.34  # ~10 rps with key, ~3 rps without
        self.min_interval = (
            float(min_interval) if min_interval is not None else default_interval
        )
        self._last_call = 0.0
        self._lock = threading.Lock()
        # Cache
        self._cache = _LRUCache(cache_size)
        self.phi_denylist = set(phi_denylist or DEFAULT_PHI_DENYLIST)
        self._log = logging.getLogger(__name__)

    def _rate_limit(self) -> None:
        with self._lock:
            now = time.time()
            wait = self.min_interval - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def _request(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        # Append common params
        q = dict(params)
        q.setdefault("db", "pubmed")
        q.setdefault("retmode", "json")
        if self.api_key:
            q["api_key"] = self.api_key
        # Retry with backoff
        backoff = 0.5
        for attempt in range(5):
            self._rate_limit()
            try:
                self._log.debug(
                    "PubMed request %s params=%s attempt=%d",
                    url,
                    {k: ("***" if k == "api_key" else v) for k, v in q.items()},
                    attempt + 1,
                )
                resp = requests.get(url, params=q, timeout=30)
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"Server error: {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                return data
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
                if attempt == 4:
                    raise
                time.sleep(backoff)
                backoff *= 2

    def _build_query(self) -> str:
        # Combine terms with AND; escape simple quotes
        terms = [t.strip() for t in self.search_terms if t and t.strip()]
        if not terms:
            return ""
        return " AND ".join([f"({t})" for t in terms])

    def _fetch_pmids(self, query: str, retmax: int) -> List[str]:
        cache_key = ("pmids", query, retmax)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._log.debug("PubMed cache hit: pmids for '%s' retmax=%d", query, retmax)
            return cached  # type: ignore[return-value]
        data = self._request(
            self.ESEARCH_URL,
            {"term": query, "retmax": int(retmax), "retstart": 0},
        )
        ids = data.get("esearchresult", {}).get("idlist", [])
        self._cache.set(cache_key, ids)
        return ids

    def _fetch_summaries(self, pmids: List[str]) -> List[Dict[str, Any]]:
        if not pmids:
            return []
        # Batch pmids to avoid very long URLs; eutils supports comma-separated ids
        ids_str = ",".join(pmids)
        cache_key = ("summary", ids_str)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._log.debug("PubMed cache hit: summaries for %d ids", len(pmids))
            return cached  # type: ignore[return-value]
        data = self._request(self.ESUMMARY_URL, {"id": ids_str})
        result = data.get("result", {})
        # 'uids' lists order; each uid maps to fields
        uids: List[str] = result.get("uids", [])
        out: List[Dict[str, Any]] = []
        for uid in uids:
            rec = dict(result.get(uid, {}))
            rec["pmid"] = uid
            # Normalize likely useful fields
            item = {
                "pmid": rec.get("pmid") or uid,
                "title": rec.get("title"),
                "pubdate": rec.get("pubdate"),
                "source": rec.get("source"),
                "authors": [
                    a.get("name") for a in rec.get("authors", []) if isinstance(a, dict)
                ],
            }
            item = redact_phi(item, denylist=self.phi_denylist)
            out.append(item)
        self._cache.set(cache_key, out)
        return out

    def load_data(self) -> Iterator[Dict[str, Any]]:
        query = self._build_query()
        if not query:
            return iter(())  # type: ignore[return-value]
        pmids = self._fetch_pmids(query, min(max(self.max_results, 1), 10000))
        summaries = self._fetch_summaries(pmids)
        for item in summaries:
            self.validate_item(item)
            yield item

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source": "PubMed",
            "search_terms": self.search_terms,
            "max_results": self.max_results,
            "uses_api_key": bool(self.api_key),
            "min_interval": self.min_interval,
        }


# -----------------
# PHI/PII sanitizers
# -----------------

DEFAULT_PHI_DENYLIST = {
    "name",
    "full_name",
    "first_name",
    "last_name",
    "ssn",
    "social_security_number",
    "mrn",
    "medical_record_number",
    "phone",
    "email",
    "address",
}


def redact_phi(
    item: Dict[str, Any],
    *,
    denylist: Iterable[str],
    regex_keys: Optional[Iterable[str]] = None,
    max_depth: int = 2,
) -> Dict[str, Any]:
    """Redact PHI/PII fields by replacing their values.

    - Exact-key redaction via denylist
    - Optional regex-based key matching
    - Supports nested dicts/lists up to `max_depth`
    """
    if not item:
        return item
    patterns = [re.compile(p) for p in (regex_keys or [])]

    def _red(obj: Any, depth: int) -> Any:
        if depth < 0:
            return obj
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                if k in denylist or any(p.search(k) for p in patterns):
                    out[k] = "[REDACTED]" if v is not None else v
                else:
                    out[k] = _red(v, depth - 1)
            return out
        if isinstance(obj, list):
            return [_red(x, depth - 1) for x in obj]
        return obj

    return _red(dict(item), max_depth)


# -------------
# Simple LRU
# -------------


class _LRUCache:
    def __init__(self, capacity: int = 16):
        self.capacity = max(1, int(capacity))
        self._data: "OrderedDict[Any, Any]" = OrderedDict()

    def get(self, key: Any) -> Any:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: Any, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)
