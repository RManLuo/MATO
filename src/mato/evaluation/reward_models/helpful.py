"""Helpful reward model."""

from typing import List, Dict, Any
import torch

from ..base_reward_model import BaseRewardModel, RewardModelRegistry

class Instructions:
    response_split = "\n\nAssistant:"
    input_split = "\n\nHuman:"
    score_split = "<rm_score>"
    input_template = "\n\nHuman: {input}\n\nAssistant:"
    output_template = "{response}"

    @staticmethod
    def get_input(query):
        if isinstance(query, list):
            prompt = ""
            for q in query:
                if q['role'] == 'user':
                    prompt += Instructions.input_split + ' ' + q['content'] + " "
                elif q['role'] == 'assistant':
                    prompt += Instructions.response_split + ' ' + q['content']
            prompt += Instructions.response_split
            return prompt
        else:
            return Instructions.input_template.format(input=query) + ' ' + Instructions.response_split

        
    @staticmethod
    def get_response(response):
        return Instructions.output_template.format(response=response)

@RewardModelRegistry.register('helpful')
class HelpfulRewardModel(BaseRewardModel):
    """
    Helpful reward model using GPT-2 Large.
    
    Uses standard query-response format: tokenizer(query, response).
    """
    
    @classmethod
    def get_default_model_path(cls) -> str:
        """Get the default model path."""
        return 'Ray2333/gpt2-large-helpful-reward_model'
    
    def build_input(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Build tokenized inputs using standard query-response format.
        
        Args:
            items: List of evaluation items.
            
        Returns:
            List of tokenized inputs.
        """
        return [
            self.tokenizer(
                Instructions.get_input(item['input']),
                Instructions.get_response(item['output']),
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length
            )
            for item in items
        ]
