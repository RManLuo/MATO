"""
Dataset registry for the MATO open-source release.

Importing this module registers all included dataset implementations.
"""

from .base_dataset import BaseDataset, DatasetRegistry
from .hh_rlhf import HHRLHF
from .multifaceted import Multifaceted


__all__ = [
    "BaseDataset",
    "DatasetRegistry",
    "HHRLHF",
    "Multifaceted",
]
