"""
Reward model implementations.

Importing this module registers all included reward models with the
RewardModelRegistry.
"""

from .helpful import HelpfulRewardModel
from .humor import HumorRewardModel

__all__ = [
    "HelpfulRewardModel",
    "HumorRewardModel",
]
