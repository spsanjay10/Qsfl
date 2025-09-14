"""
QSFL-CAAD: Quantum-Safe Federated Learning with Client Anomaly and Attack Detection

Main package initialization and public API exports.
"""

from .system import QSFLSystem
from .client import QSFLClient

__version__ = "1.0.0"
__author__ = "QSFL-CAAD Team"

__all__ = [
    "QSFLSystem",
    "QSFLClient"
]