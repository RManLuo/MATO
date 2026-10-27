"""
Base classes for reward models.

This module defines the abstract base class for all reward models and provides
a registry mechanism for automatic discovery and registration of reward model types.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from .utils import load_reward_model


class RewardModelRegistry:
    """Registry for reward model classes."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a reward model class.

        Args:
            name: The name of the reward model (e.g., 'harmless', 'helpful').

        Usage:
            @RewardModelRegistry.register('harmless')
            class HarmlessRewardModel(BaseRewardModel):
                ...
        """

        def decorator(reward_class):
            cls._registry[name] = reward_class
            return reward_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """
        Get a reward model class by name.

        Args:
            name: The name of the reward model.

        Returns:
            The reward model class.

        Raises:
            KeyError: If the reward model name is not registered.
        """
        if name not in cls._registry:
            raise KeyError(
                f"Reward model '{name}' not found in registry. "
                f"Available models: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available reward model names."""
        return list(cls._registry.keys())

    @classmethod
    def get_default_model_path(cls, name: str) -> str:
        """
        Get the default model path for a reward model.

        Args:
            name: The name of the reward model.

        Returns:
            The default model path.
        """
        reward_class = cls.get(name)
        return reward_class.get_default_model_path()


class BaseRewardModel(ABC):
    """
    Abstract base class for all reward models.

    Each reward model subclass should implement:
    - get_default_model_path(): Return the default HuggingFace model path
    - build_input(): Build tokenized input from query-response pairs
    - compute_reward(): Compute reward scores from model outputs
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str,
        norm_reward: bool = False,
        reward_mean_std: Optional[
            Union[Tuple[float, float], Dict[str, Tuple[float, float]]]
        ] = None,
    ):
        """
        Initialize the reward model.

        Args:
            model_path: Path to the reward model.
            tokenizer_path: Path to the tokenizer.
            device: Device to load the model on (e.g., 'cuda:0').
            norm_reward: Whether to normalize the reward.
            reward_mean_std: Optional tuple of (mean, std) for reward normalization.
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.norm_reward = norm_reward
        self.reward_mean_std = reward_mean_std

        self.model = None  # Model will be loaded lazily in compute_reward
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._max_length = min(self.tokenizer.model_max_length, 1024)

    @property
    def max_length(self) -> int:
        """Get the maximum input length for the model."""
        return self._max_length

    def load_model(self):
        """
        Load the reward model and assign it to self.model.
        """
        if self.model is None:
            self.model = load_reward_model(self.model_path, self.device)

    @classmethod
    @abstractmethod
    def get_default_model_path(cls) -> str:
        """
        Get the default HuggingFace model path for this reward type.

        Returns:
            The default model path.
        """
        pass

    @abstractmethod
    def build_input(self, items: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        """
        Build tokenized inputs for the reward model.

        Args:
            items: List of evaluation items. Each item contains 'input' and 'output'.
                   The 'input' can be a string prompt or a list of messages.

        Returns:
            List of tokenized inputs ready for model inference.
        """
        pass

    def compute_reward(
        self, texts_for_rewards: List[Dict[str, torch.Tensor]]
    ) -> Union[List[float], Dict[str, List[float]]]:
        """
        Compute reward scores from tokenized inputs.

        Args:
            texts_for_rewards: List of tokenized inputs.

        Returns:
            List of reward scores, or a dict of named reward score lists.
        """

        # Load model
        self.load_model()

        # Get sub_position for models with multiple outputs
        sub_position = self.get_sub_position()
        rewards = []

        with torch.no_grad():
            pbar = tqdm(total=len(texts_for_rewards))
            for inputs in texts_for_rewards:
                if sub_position != 0:  # for multiple output
                    rewards.append(
                        self.model(**(inputs.to(self.device))).logits[0][sub_position]
                    )
                else:
                    rewards.append(self.model(**(inputs.to(self.device))).logits[0])
                pbar.update(1)
        if self.norm_reward:
            return self._normalize_reward_output(rewards)
        else:
            return [self._to_scalar(r) for r in rewards]

    def _normalize_reward_output(
        self, rewards: Union[List[Any], Dict[str, List[Any]]]
    ) -> Union[List[float], Dict[str, List[float]]]:
        if isinstance(rewards, dict):
            normalized: Dict[str, List[float]] = {}
            for name, values in rewards.items():
                mean_std = None
                if isinstance(self.reward_mean_std, dict):
                    mean_std = self.reward_mean_std.get(name)
                elif isinstance(self.reward_mean_std, tuple):
                    mean_std = self.reward_mean_std
                normalized[name] = self._normalize_reward_list(values, mean_std)
            return normalized

        mean_std = None
        if isinstance(self.reward_mean_std, tuple):
            mean_std = self.reward_mean_std
        return self._normalize_reward_list(rewards, mean_std)

    def _normalize_reward_list(
        self, rewards: List[Any], mean_std: Optional[Tuple[float, float]]
    ) -> List[float]:
        if mean_std is None:
            # Compute mean and std from the rewards if not provided
            mean_reward = np.mean([self._to_scalar(r) for r in rewards])
            std_reward = np.std([self._to_scalar(r) for r in rewards])
        else:
            mean_reward, std_reward = mean_std
        return [
            np.round((self._to_scalar(r) - mean_reward) / std_reward, 1)
            for r in rewards
        ]

    def _to_scalar(self, value: Any) -> float:
        if hasattr(value, "detach"):
            return value.detach().cpu().item()
        return float(value)

    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    def get_sub_position(self) -> int:
        """
        Get the sub-position for models with multiple output logits.

        Returns:
            The sub-position index (default: 0).
        """
        return 0

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of this reward model.

        Returns:
            The reward model name.
        """
        # Find the name from registry
        for name, registered_class in RewardModelRegistry._registry.items():
            if registered_class == cls:
                return name
        return cls.__name__
