import os
from peft import PeftModel
from transformers import AutoModelForSequenceClassification
import torch

def load_reward_model(reward_peft_path, gpu_id):
    num_labels = 2 if ('humor' in reward_peft_path or 'faithful' in reward_peft_path) else 1
    reward_model = AutoModelForSequenceClassification.from_pretrained(
                    reward_peft_path,
                    num_labels=num_labels, dtype=torch.bfloat16,
                    device_map=gpu_id,
                    )
    if check_lora_in_model_path(reward_model, reward_peft_path):
        reward_model = PeftModel.from_pretrained(reward_model, reward_peft_path)
    if hasattr(reward_model, 'merge_and_unload'):
        reward_model = reward_model.merge_and_unload() # merge lora weights
    return reward_model.to(gpu_id)


def check_lora_in_model_path(model, path):
    if os.path.exists(path):
        dirnames = os.listdir(path)
        if 'adapter_config.json' in dirnames:
            return True
        
        state_dict_keys = model.state_dict().keys()
        for key in state_dict_keys:
            if 'lora' in key:
                return True
    return False
