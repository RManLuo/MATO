"""
Multi-reward model manager for evaluating multiple reward models simultaneously.

This module provides a unified interface for managing and evaluating multiple
reward models using the BaseRewardModel framework.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union

from .base_reward_model import RewardModelRegistry
from . import reward_models  # Import to register all models


class MultiRewardModel:
    """
    Manager for multiple reward models.
    
    This class handles loading and evaluating multiple reward models in parallel,
    with support for reward normalization and caching of tokenized inputs.
    """
    
    def __init__(
        self,
        reward_names: List[str],
        reward_model_path_list: List[str],
        rm_tokenizer_path_list: List[str],
        gpu_id_list,
        reward_stats_path: Optional[str] = None
    ):
        """
        Initialize multiple reward models.
        
        Args:
            reward_names: List of reward model names (e.g., ['harmless', 'helpful']).
            reward_model_path_list: List of model paths.
            rm_tokenizer_path_list: List of tokenizer paths.
            gpu_id_list: GPU ID(s) to use. Can be a single ID or list of IDs.
            reward_stats_path: Optional path to reward statistics for normalization.
        """
        assert len(reward_names) == len(reward_model_path_list) == len(rm_tokenizer_path_list)
        
        self.reward_names = reward_names
        self.reward_model_path_list = reward_model_path_list
        self.rm_tokenizer_path_list = rm_tokenizer_path_list
        self.num_rewards = len(reward_names)
        
        # Load reward statistics if provided
        self.reward_stats = np.load(reward_stats_path) if reward_stats_path is not None else None
        
        # Handle GPU ID list
        if type(gpu_id_list) != list:
            gpu_id_list = [gpu_id_list] * self.num_rewards
        assert len(gpu_id_list) == self.num_rewards
        self.gpu_id_list = gpu_id_list
        
        # Initialize reward model instances
        self.reward_models = []
        print('Loading reward models .....')
        for i, name in enumerate(reward_names):
            # Get reward mean/std if available
            reward_mean_std = self._get_reward_mean_std(i, name)
            
            # Get the reward model class from registry
            reward_class = RewardModelRegistry.get(name)
            
            # Instantiate the reward model
            reward_model = reward_class(
                model_path=reward_model_path_list[i],
                tokenizer_path=rm_tokenizer_path_list[i],
                device=gpu_id_list[i],
                reward_mean_std=reward_mean_std
            )
            self.reward_models.append(reward_model)
            print(f"  Loaded {name} reward model from {reward_model_path_list[i]}")
    
    def _get_reward_mean_std(
        self,
        index: int,
        reward_name: str
    ) -> Optional[Union[Tuple[float, float], Dict[str, Tuple[float, float]]]]:
        """
        Get reward mean and standard deviation for normalization.
        
        Args:
            index: Index of the reward model.
            
        Returns:
            Tuple of (mean, std) or None if not available.
        """
        if self.reward_stats is None:
            return None
        
        if isinstance(self.reward_stats, dict):
            return self.reward_stats.get(reward_name)

        if type(self.reward_stats) == list or len(self.reward_stats) == 2 * self.num_rewards:
            return (self.reward_stats[2*index], self.reward_stats[2*index+1])
        else:
            return self.reward_stats[index]
    
    def get_reward_model_scores(
        self,
        items: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """
        Compute reward scores for all reward models.
        
        Args:
            items: List of evaluation items.
            
        Returns:
            Dictionary mapping reward names to reward score lists.
        """
        rewards: Dict[str, List[float]] = {}
        
        for reward_model in self.reward_models:
            
            print(f"Evaluating with reward model: {reward_model.get_name()} ...")
            
            # Build inputs using the reward model's specific method
            texts_for_rewards = reward_model.build_input(items)
            
            # Compute rewards
            temp_reward = reward_model.compute_reward(texts_for_rewards)
            if isinstance(temp_reward, dict):
                for reward_name, reward_values in temp_reward.items():
                    if reward_name in rewards:
                        raise ValueError(
                            f"Duplicate reward name '{reward_name}' from model '{reward_model.get_name()}'."
                        )
                    rewards[reward_name] = reward_values
            else:
                reward_name = reward_model.get_name()
                if reward_name in rewards:
                    raise ValueError(
                        f"Duplicate reward name '{reward_name}' from model '{reward_model.get_name()}'."
                    )
                rewards[reward_name] = temp_reward
            
            # Free up model resources after computing rewards
            reward_model.cleanup()
        
        return rewards


# Backward compatibility: keep the old class name as alias
RewardModels = MultiRewardModel
            
