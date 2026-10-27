"""
Multi-reward evaluation module.

This module provides functionality to evaluate model outputs using multiple
reward models and compute average scores.
"""

import numpy as np
import json

from .multi_reward_models import MultiRewardModel
from .base_reward_model import RewardModelRegistry
from . import reward_models  # Import to register all models


def eval(
    result_file: str,
    reward_names: str | None,
    device: str,
    return_item_rewards: bool = True,
) -> dict | tuple[dict, list[dict]]:
    """
    Evaluate model outputs using multiple reward models.
    
    Args:
        result_file: Path to the JSON file containing model outputs.
        reward_names: Comma-separated list of reward model names.
        device: Device to use (e.g., 'cuda:0').
        return_item_rewards: If True, also return per-item rewards merged
            into each input sample.
        
    Returns:
                Dictionary containing individual reward scores and average.
                If return_item_rewards=True, returns a tuple of:
                    (summary_dict, results_with_reward)
    """
    if reward_names is None:
        # Use all registered reward models if none specified
        reward_names_list = RewardModelRegistry.list_available()
    else:
        # Parse reward names
        reward_names_list = [x.strip() for x in reward_names.split(',')]
        print(f"Evaluating with reward models: {reward_names_list}")
    
    # Get default model paths from registry
    reward_model_path_list = []
    rm_tokenizer_path_list = []
    
    for name in reward_names_list:
        try:
            model_path = RewardModelRegistry.get_default_model_path(name)
            reward_model_path_list.append(model_path)
            rm_tokenizer_path_list.append(model_path)  # Use same path for tokenizer
        except KeyError as e:
            available = RewardModelRegistry.list_available()
            raise ValueError(
                f"Unknown reward model '{name}'. "
                f"Available models: {available}"
            ) from e
    
    # Initialize multi-reward model
    reward_models = MultiRewardModel(
        reward_names=reward_names_list,
        reward_model_path_list=reward_model_path_list,
        rm_tokenizer_path_list=rm_tokenizer_path_list,
        gpu_id_list=device,
        reward_stats_path=None
    )
    
    # Load results
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # Compute rewards
    rewards_dict = reward_models.get_reward_model_scores(results)
    
    # Print and collect results
    result_dict = {}
    avg_rewards = []
    for name, scores in rewards_dict.items():
        score = np.mean(scores)
        score = np.round(score, 4)
        result_dict[name] = score
        avg_rewards.append(score)
        print(f"{name}: {score}")
    
    result_dict["average"] = np.mean(avg_rewards)
    print(f"Average reward: {result_dict['average']}")
    
    if not return_item_rewards:
        return result_dict

    results_with_reward = []
    for idx, item in enumerate(results):
        item_with_reward = dict(item)
        item_with_reward["rewards"] = {
            reward_name: float(np.round(reward_scores[idx], 4))
            for reward_name, reward_scores in rewards_dict.items()
        }
        results_with_reward.append(item_with_reward)

    return result_dict, results_with_reward
