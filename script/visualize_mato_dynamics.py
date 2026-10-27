"""
Visualize MATO dynamic weights and cumulative rewards during generation.

Generates text for a small number of prompts with dynamics tracking enabled,
then plots per-step cumulative rewards and adaptive weights to show how MATO
dynamically adjusts the multi-objective trade-off during token generation.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
)

from mato.data.multifaceted_prompts import (
    BASE_SYSTEM_PROMPT,
    build_individual_principle_prompts,
    build_weighted_system_prompt,
)
from mato.inference.mato import MATODecoder, MATODecodingConfig, MATODynamics
from mato.utils import set_up_seed

set_up_seed(42)


@dataclass
class VisArgs:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "Pre-trained model to use."},
    )
    principles: List[str] = field(
        default_factory=list,
        metadata={"help": "Per-objective principle descriptions."},
    )
    weights: List[float] = field(
        default_factory=list,
        metadata={"help": "Preference weight for each principle."},
    )
    principle_names: List[str] = field(
        default_factory=list,
        metadata={"help": "Short labels for each principle (used in plot legends)."},
    )
    prompts: List[str] = field(
        default_factory=list,
        metadata={"help": "User prompts to generate and visualize (1-3 recommended)."},
    )
    save_dir: str = field(
        default="./output/dynamics_vis",
        metadata={"help": "Directory to save plots and generated text."},
    )
    # Model loading
    attn_implementation: str = field(default="sdpa")
    dtype: str = field(default="auto")
    device: str = field(default="auto")
    # Decoding config
    max_new_tokens: int = field(default=256)
    enable_thinking: bool = field(default=False)
    do_sample: bool = field(default=True)
    temperature: float = field(default=0.7)
    top_k: int = field(default=20)
    top_p: float = field(default=0.8)
    min_p: float = field(default=0.0)
    # MATO config
    iteration_num: int = field(default=60)
    eta: int = field(default=10)
    alpha: float = field(default=0.5)
    player_lambda: float = field(default=1)
    rho: float = field(default=1.0)
    find_weight: bool = field(default=True)
    batch_inference: bool = field(
        default=False,
        metadata={"help": "Evaluate all system prompts together in one model batch."},
    )


def plot_dynamics(
    dynamics: MATODynamics,
    principle_names: List[str],
    tokenizer,
    save_path: str,
    title: str = "",
):
    """Plot cumulative rewards and weights over generation steps."""
    N = len(principle_names)
    rewards = torch.stack(dynamics.cumulative_rewards).numpy()  # [T, N]
    weights = torch.stack(dynamics.weights).numpy()             # [T, N]
    T = len(rewards)
    steps = np.arange(T)

    # Token labels for x-axis
    token_strs = [tokenizer.decode([tid]) for tid in dynamics.tokens]
    # Clean up token strings for display
    # token_strs = [t.replace("\\u0120", " ").replace("\\u010a", "\\n") if t else "" for t in token_strs]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, min(20, 0.15 * T)), 8), sharex=True)
    fig.suptitle(title, fontsize=12, y=0.98)

    colors = plt.cm.tab10(np.linspace(0, 1, max(N, 10)))[:N]

    # Top: Cumulative rewards
    for i in range(N):
        ax1.plot(steps, rewards[:, i], label=principle_names[i], color=colors[i], linewidth=2)
    ax1.set_ylabel("Cumulative Reward", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Cumulative Differential Reward per Objective", fontsize=11)

    # Bottom: Weights
    for i in range(N):
        ax2.plot(steps, weights[:, i], label=principle_names[i], color=colors[i], linewidth=2)
    ax2.set_ylabel("Weight", fontsize=11)
    ax2.set_xlabel("Generation Step", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Dynamic Weight per Objective", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)

    # Token labels on x-axis (sparse for readability)
    if T > 0:
        label_interval = max(1, T // 30)
        tick_positions = list(range(0, T, label_interval))
        ax2.set_xticks(tick_positions)
        tick_labels = []
        for pos in tick_positions:
            if pos < len(token_strs):
                tick_labels.append(f"{pos}: {token_strs[pos]}")
            else:
                tick_labels.append(str(pos))
        ax2.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path + ".pdf", bbox_inches="tight", dpi=150)
    fig.savefig(save_path + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved plots to {save_path}.pdf and {save_path}.png")


def main():
    parser = HfArgumentParser(VisArgs)
    args = parser.parse_args_into_dataclasses()[0]

    assert len(args.principles) == len(args.weights), (
        "Number of --principles must match --weights."
    )
    assert len(args.prompts) > 0, "Must provide at least one --prompts entry."

    if not args.principle_names:
        # Default: use first word of each principle as label
        args.principle_names = [p.split(":")[0] if ":" in p else p[:20] for p in args.principles]
    assert len(args.principle_names) == len(args.principles), (
        "Number of --principle_names must match --principles."
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # Load model and tokenizer
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        dtype=args.dtype,
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Build decoder
    device_for_decoder = None if args.device == "auto" else args.device
    individual_prompts = build_individual_principle_prompts(args.principles)
    weighted_prompt = build_weighted_system_prompt(args.principles, args.weights)
    system_prompts = [BASE_SYSTEM_PROMPT, weighted_prompt] + individual_prompts

    config = MATODecodingConfig(
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
    )

    decoder = MATODecoder(
        model=model,
        tokenizer=tokenizer,
        system_prompts=system_prompts,
        weights=args.weights,
        config=config,
        device=device_for_decoder,
    )

    results = []
    for i, prompt in enumerate(args.prompts):
        print(f"\n--- Sample {i} ---")
        print(f"Prompt: {prompt}")
        dynamics = decoder.generate_one(prompt, track_dynamics=True)
        print(f"Generated ({len(dynamics.tokens)} tokens): {dynamics.text[:200]}...")

        # Plot
        plot_dynamics(
            dynamics,
            args.principle_names,
            tokenizer,
            os.path.join(args.save_dir, f"sample_{i}"),
            title=f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}",
        )

        # Decode each token id to its surface string for later plotting
        token_strings = [
            tokenizer.decode([tid], skip_special_tokens=False)
            for tid in dynamics.tokens
        ]
        results.append({
            "id": i,
            "prompt": prompt,
            "output": dynamics.text,
            "num_tokens": len(dynamics.tokens),
            "tokens": dynamics.tokens,
            "token_strings": token_strings,
            "principle_names": args.principle_names,
            "weights": [w.tolist() for w in dynamics.weights],
            "cumulative_rewards": [r.tolist() for r in dynamics.cumulative_rewards],
        })

    # Save results
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
