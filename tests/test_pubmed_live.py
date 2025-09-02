import os
from typing import Dict, Any, List

import pytest

from bench.data.medical import PubMedConnector

RUN_LIVE = os.getenv("RUN_PUBMED_LIVE") == "1"

pytestmark = pytest.mark.skipif(not RUN_LIVE, reason="RUN_PUBMED_LIVE not set")


def test_pubmed_live_smoke():
    api_key = os.getenv("NCBI_API_KEY")
    # Use conservative settings to be friendly to eUtils
    ds = PubMedConnector(["hypertension"], max_results=5, api_key=api_key)
    # Ensure outputs contain minimal required keys
    ds.required_keys = ["pmid", "title"]

    items: List[Dict[str, Any]] = list(ds.load_data())
    assert 0 < len(items) <= 5
    assert all("pmid" in it and "title" in it for it in items)
    # Basic sanity: pmid looks numeric
    assert all(str(it["pmid"]).isdigit() for it in items)
