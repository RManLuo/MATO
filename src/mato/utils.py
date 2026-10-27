from transformers import AutoTokenizer
from datasets import Dataset

def convert_to_chat_message(tokenizer: AutoTokenizer, dataset: Dataset, system_prompt: str = "", enable_thinking: bool = False) -> Dataset:
    def convert(sample):
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample["prompt"]},
            ]
        else:
            messages = [{"role": "user", "content": sample["prompt"]}]
        messages = tokenizer.apply_chat_template(messages, tokenize=True,add_generation_prompt=True, return_dict=True, enable_thinking=enable_thinking)
        return messages
    
    data = dataset.map(convert).select_columns(["input_ids", "attention_mask"])
    return data.with_format("torch")

def set_up_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)