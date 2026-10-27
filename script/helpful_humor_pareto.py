"""
Helpful + Humor Pareto alignment with MATO.

For one (principles, weights) configuration, generate responses on the HH-RLHF
test split and score them with the helpful and humor reward models. Sweeping
weights across runs produces an empirical Pareto front for the two objectives.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List

import torch
from tqdm import tqdm
from transformers import (  # type: ignore[import]
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
)

from mato.data import DatasetRegistry  # type: ignore[import]
from mato.data.multifaceted_prompts import (  # type: ignore[import]
    BASE_SYSTEM_PROMPT,
    build_individual_principle_prompts,
    build_weighted_system_prompt,
)
from mato.evaluation.multi_reward_eval import eval as evaluate_rewards  # type: ignore[import]
from mato.inference.mato import MATODecoder, MATODecodingConfig  # type: ignore[import]
from mato.utils import set_up_seed  # type: ignore[import]

set_up_seed(42)


@dataclass
class ScriptArguments:
    save_dir: str = field(
        default="./output", metadata={"help": "Directory to save results."}
    )
    exp_name: str = field(default="default", metadata={"help": "Experiment name."})
    model_name_or_path: str = field(
        default="Qwen/Qwen3-8B",
        metadata={"help": "Pre-trained model identifier or local path."},
    )
    attn_implementation: str = field(default="sdpa")
    dtype: str = field(default="auto")
    device: str = field(default="auto")
    dataset_name: str = field(
        default="hh-rlhf", metadata={"help": "Dataset key in the registry."}
    )
    dataset_size: int | None = field(
        default=100,
        metadata={"help": "Subset size; pass null/-1 to use the registered default."},
    )
    batch_size: int = field(default=1)
    principles: List[str] = field(
        default_factory=list,
        metadata={"help": "Per-objective principle descriptions."},
    )
    weights: List[float] = field(
        default_factory=list,
        metadata={"help": "Preference weight per principle (same length as --principles)."},
    )
    enable_thinking: bool = field(default=False)
    max_new_tokens: int = field(default=2048)
    do_sample: bool = field(default=True)
    temperature: float = field(default=0.7)
    top_k: int = field(default=20)
    top_p: float = field(default=0.8)
    min_p: float = field(default=0.0)
    # MATO-specific
    iteration_num: int = field(default=60)
    eta: int = field(default=10)
    alpha: float = field(default=0.5)
    player_lambda: float = field(default=1.0)
    rho: float = field(default=1.0)
    find_weight: bool = field(default=True)
    batch_inference: bool = field(
        default=False,
        metadata={"help": "Evaluate all system prompts together in one model batch."},
    )
    # Reward eval
    reward_names: str | None = field(
        default="helpful,humor",
        metadata={"help": "Comma-separated reward names registered in the reward model registry."},
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    assert len(args.principles) == len(args.weights), (
        "The number of --principles entries must match --weights."
    )
    assert len(args.principles) > 0, "Must provide at least one --principles entry."

    model_short = args.model_name_or_path.split("/")[-1]
    weight_tag = "_".join(str(w) for w in args.weights)
    output_path = os.path.join(
        args.save_dir,
        args.dataset_name,
        model_short,
        "mato",
        args.exp_name,
        weight_tag,
    )
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model and tokenizer.
    cfg = AutoConfig.from_pretrained(args.model_name_or_path)
    if getattr(cfg, "is_encoder_decoder", False):
        raise NotImplementedError(
            "MATO is implemented for decoder-only causal LMs in this release."
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        dtype=args.dtype,
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Dataset.
    dataset_kwargs = {}
    if args.dataset_size is not None and args.dataset_size > 0:
        dataset_kwargs["size"] = args.dataset_size
    dataset_class = DatasetRegistry.get(args.dataset_name)
    dataset = dataset_class(**dataset_kwargs)

    # Build system prompts: [base, weighted_combined, individual_1, ..., individual_N].
    individual_prompts = build_individual_principle_prompts(args.principles)
    weighted_prompt = build_weighted_system_prompt(args.principles, args.weights)
    system_prompts = [BASE_SYSTEM_PROMPT, weighted_prompt] + individual_prompts

    decoder = MATODecoder(
        model=model,
        tokenizer=tokenizer,
        system_prompts=system_prompts,
        weights=args.weights,
        config=MATODecodingConfig(
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=None if args.top_k <= 0 else args.top_k,
            top_p=None if args.top_p >= 1.0 else args.top_p,
            min_p=None if args.min_p <= 0.0 or args.min_p >= 1.0 else args.min_p,
            iteration_num=args.iteration_num,
            eta=args.eta,
            alpha=args.alpha,
            player_lambda=args.player_lambda,
            rho=args.rho,
            find_weight=args.find_weight,
            batch_inference=args.batch_inference,
        ),
        device=None if args.device == "auto" else args.device,
    )

    # Generate.
    output_data = []
    for idx, sample in enumerate(tqdm(dataset.data, desc="MATO inference")):
        text = decoder.generate_one(sample["prompt"])  # type: ignore[index]
        output_data.append(
            {"id": sample.get("id", idx), "input": sample["prompt"], "output": text}
        )
    results_path = os.path.join(output_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Free GPU memory before running reward models.
    del model
    del decoder
    torch.cuda.empty_cache()

    # Reward evaluation.
    result_dict, results_with_reward = evaluate_rewards(
        result_file=results_path,
        reward_names=args.reward_names,
        device=args.device,
    )
    with open(os.path.join(output_path, "reward_eval_results.json"), "w") as f:
        json.dump(result_dict, f, indent=2)
    with open(os.path.join(output_path, "results_with_reward.json"), "w") as f:
        json.dump(results_with_reward, f, indent=2)

    print(f"Saved outputs under {output_path}")


if __name__ == "__main__":
    main()
