import json
import os
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from mato.evaluation.multi_reward_eval import eval

from mato.utils import set_up_seed  # type: ignore[import]

set_up_seed(42)

@dataclass
class ScriptArguments:
    result_file: str
    reward_names: str | None = field(default=None, metadata={"help": "Comma-separated reward model names for evaluation. If None, defaults to the full set of available reward models."})
    device: str = field(
        default="cuda:0",
        metadata={"help": "The device to use for inference."}
    )

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    result_dict, results_with_reward = eval(
        result_file=args.result_file,
        reward_names=args.reward_names,
        device=args.device,
        return_item_rewards=True,
    )
    output_path = os.path.dirname(args.result_file)
    # Save reward evaluation results
    with open(os.path.join(output_path, "reward_eval_results.json"), "w") as f:
        json.dump(result_dict, f, indent=2)

    # Save per-sample rewards
    with open(os.path.join(output_path, "results_with_reward.json"), "w") as f:
        json.dump(results_with_reward, f, indent=2)