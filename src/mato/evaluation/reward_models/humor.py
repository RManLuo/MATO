"""Humor reward model."""

from typing import List, Dict, Any
import torch

from ..base_reward_model import BaseRewardModel, RewardModelRegistry


@RewardModelRegistry.register('humor')
class HumorRewardModel(BaseRewardModel):
    """
    Humor detection reward model.
    
    Uses only response: tokenizer(response).
    Returns sub_position=1 from logits (binary classification).
    """
    
    @classmethod
    def get_default_model_path(cls) -> str:
        """Get the default model path."""
        return 'mohameddhiab/humor-no-humor'
    
    def build_input(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Build tokenized inputs using only the response.
        
        Args:
            items: List of evaluation items.
            
        Returns:
            List of tokenized inputs.
        """
        return [
            self.tokenizer(
                item['output'],
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length
            )
            for item in items
        ]
    
    def get_sub_position(self) -> int:
        """Get the sub-position for binary classification output."""
        return 1
