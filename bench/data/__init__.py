"""Data connectors and preprocessing utilities for MedAISure.

Modules:
- base: Abstract base classes and common errors/utilities
- local: Local file-based dataset connectors (JSON, CSV)
- medical: Placeholders for medical datasets/APIs (MIMIC, PubMed)
- preprocess: Simple, composable preprocessing pipeline
- security: Optional encryption/decryption helpers (Fernet)
"""

from .base import DatasetConnector, DatasetError, ValidationError
from .local import JSONDataset, CSVDataset
from .medical import MIMICConnector, PubMedConnector
from .preprocess import DataPreprocessor
from .security import SecureDataHandler

__all__ = [
    "DatasetConnector",
    "DatasetError",
    "ValidationError",
    "JSONDataset",
    "CSVDataset",
    "MIMICConnector",
    "PubMedConnector",
    "DataPreprocessor",
    "SecureDataHandler",
]
