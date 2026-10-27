"""
Base classes and registry for datasets.

This module provides a lightweight registry to enable plug-in style dataset
selection, similar to reward model registration.
"""

from abc import ABC
from typing import Any, Dict, List, Type


class DatasetRegistry:
    """Registry for dataset classes."""

    _registry: Dict[str, Type["BaseDataset"]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a dataset class under a name."""
        def decorator(dataset_class: Type["BaseDataset"]):
            cls._registry[name] = dataset_class
            return dataset_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type["BaseDataset"]:
        if name not in cls._registry:
            raise KeyError(
                f"Dataset '{name}' not found in registry. "
                f"Available datasets: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._registry.keys())


class BaseDataset(ABC):
    """Base class for datasets."""

    data = None

    @classmethod
    def get_name(cls) -> str:
        for name, registered_class in DatasetRegistry._registry.items():
            if registered_class == cls:
                return name
        return cls.__name__
