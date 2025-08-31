from __future__ import annotations

from typing import Dict, Iterator, Any, List

from .base import DatasetConnector


class MIMICConnector(DatasetConnector):
    """
    Placeholder connector for MIMIC database access.

    In production, implement secure DB connection handling and PHI safeguards.
    """

    def __init__(self, connection_string: str, query: str):
        # Do NOT store raw credentials in metadata; only store safe host/db info.
        self.connection_string = connection_string
        self.query = query

    def load_data(self) -> Iterator[Dict[str, Any]]:
        # Implement DB access here (e.g., SQLAlchemy/psycopg2) with least privilege.
        # Ensure all PHI handling complies with your org's security policies.
        raise NotImplementedError("MIMICConnector.load_data is not implemented")

    def get_metadata(self) -> Dict[str, Any]:
        host_part = (
            self.connection_string.split("@")[-1]
            if "@" in self.connection_string
            else self.connection_string
        )
        return {
            "source": "MIMIC",
            "query": self.query,
            "connection": host_part,
        }


class PubMedConnector(DatasetConnector):
    """
    Placeholder connector for PubMed API access.

    In production, use e.g., Entrez/eutils or an HTTP client and respect rate limits.
    """

    def __init__(self, search_terms: List[str], max_results: int = 100):
        self.search_terms = list(search_terms)
        self.max_results = max_results

    def load_data(self) -> Iterator[Dict[str, Any]]:
        # Implement PubMed API access here.
        # Yield dicts with fields like {"pmid": str, "title": str, "abstract": str, ...}
        raise NotImplementedError("PubMedConnector.load_data is not implemented")

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "source": "PubMed",
            "search_terms": self.search_terms,
            "max_results": self.max_results,
        }
