"""
Multifaceted-Bench inference with MATO.

For each Multifaceted-Bench sample we read its `preference_set` and use MATO to
decode a single response that trades off the per-sample preferences. The script
also runs the LLM-as-judge evaluation (helpfulness + per-rubric scores)
immediately after generation.
"""

import json
import os
from dataclasses import dataclass, field
from multiprocessing.dummy import Pool
from threading import Lock

import dotenv
import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)

from mato.data import DatasetRegistry
from mato.data.multifaceted_prompts import (
    BASE_SYSTEM_PROMPT,
    build_individual_preference_prompts,
    build_preference_prompt,
)
from mato.evaluation.llm_judge import JudgeConfig, LLMJudge
from mato.inference.mato import MATODecoder, MATODecodingConfig
from mato.utils import set_up_seed

dotenv.load_dotenv()
set_up_seed(42)


@dataclass
class ScriptArguments:
    save_dir: str = field(default="./output")
    exp_name: str = field(default="default")
    model_name_or_path: str = field(default="Qwen/Qwen3-8B")
    attn_implementation: str = field(default="sdpa")
    dtype: str = field(default="auto")
    device: str = field(default="auto")
    dataset_name: str = field(default="multifaceted")
    dataset_size: int | None = field(
        default=None,
        metadata={"help": "Limit dataset to this many samples (debugging)."},
    )
    enable_thinking: bool = field(default=False)
    max_new_tokens: int = field(default=2048)
    do_sample: bool = field(default=True)
    temperature: float = field(default=0.7)
    top_k: int = field(default=20)
    top_p: float = field(default=0.8)
    min_p: float = field(default=0.0)
    # MATO-specific.
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
    # Checkpointing.
    resume: bool = field(default=True)
    save_interval: int = field(default=50)
    # LLM-as-judge.
    run_judge: bool = field(default=True)
    judge_provider: str = field(default="openai")
    judge_model: str = field(default="gpt-4o")
    judge_num_workers: int = field(default=10)


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    model_short = args.model_name_or_path.split("/")[-1]
    output_path = os.path.join(
        args.save_dir, args.dataset_name, model_short, "mato", args.exp_name
    )
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

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

    dataset_kwargs = {}
    if args.dataset_size is not None and args.dataset_size > 0:
        dataset_kwargs["size"] = args.dataset_size
    dataset_class = DatasetRegistry.get(args.dataset_name)
    dataset = dataset_class(**dataset_kwargs)

    # Build a single decoder once; per-sample we call update_prompts(...).
    decoder = MATODecoder(
        model=model,
        tokenizer=tokenizer,
        system_prompts=[BASE_SYSTEM_PROMPT, BASE_SYSTEM_PROMPT],
        weights=[1.0],
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

    results_path = os.path.join(output_path, "results.json")
    results: list[dict] = []
    completed_ids: set = set()
    if args.resume and os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        completed_ids = {r["id"] for r in results}
        print(f"Resuming: {len(results)} already completed samples.")

    new_count = 0
    for idx, sample in enumerate(tqdm(dataset.data, desc="MATO inference")):
        sample_id = sample.get("id", idx)
        if sample_id in completed_ids:
            continue
        preference_set = sample["preference_set"]
        individual_prompts = build_individual_preference_prompts(preference_set)
        combined = build_preference_prompt(preference_set)
        system_prompts = [BASE_SYSTEM_PROMPT, combined] + individual_prompts
        weights = [1.0] * len(preference_set)

        decoder.update_prompts(system_prompts, weights)
        text = decoder.generate_one(sample["prompt"])  # type: ignore[index]

        results.append(
            {
                "id": sample_id,
                "input": sample["prompt"],
                "output": text,
                "system_prompts": system_prompts,
                "weights": weights,
                "preference_set": preference_set,
                "rubrics": sample.get("rubrics", []),
                "source": sample.get("source", ""),
            }
        )
        new_count += 1
        if new_count % args.save_interval == 0:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Checkpoint: saved {len(results)} results")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {results_path}")

    # Free GPU memory.
    del model
    del decoder
    torch.cuda.empty_cache()

    if not args.run_judge:
        return

    print(f"\nRunning LLM-as-judge ({args.judge_model})...")
    judge = LLMJudge(
        JudgeConfig(
            provider=args.judge_provider,
            model=args.judge_model,
            temperature=0.0,
            max_tokens=1024,
            max_retries=3,
        )
    )

    safe_model = args.judge_model.replace("/", "-").replace(" ", "-")
    judge_file = os.path.join(output_path, f"judge_eval_results.{safe_model}.json")

    evaluated: dict = {}
    if os.path.exists(judge_file):
        with open(judge_file, "r") as f:
            evaluated = {item["id"]: item for item in json.load(f)}
        print(f"Resuming judge eval: {len(evaluated)} already evaluated.")

    samples_to_judge = [s for s in results if s["id"] not in evaluated]

    def evaluate_sample(sample):
        try:
            res = judge.evaluate_sample(
                prompt=sample["input"],
                response=sample["output"],
                rubrics=sample.get("rubrics", []),
            )
        except Exception as exc:
            print(f"Judge error on sample {sample['id']}: {exc}", flush=True)
            n = len(sample.get("rubrics", []))
            res = {
                "preference_scores": [-1] * n,
                "helpfulness_score": -1,
                "preference_justifications": [f"ERROR: {exc}"] * n,
                "helpfulness_justification": f"ERROR: {exc}",
            }
        return {
            "id": sample["id"],
            "source": sample.get("source", ""),
            "input": sample["input"],
            "output": sample["output"],
            "preference_set": sample.get("preference_set", []),
            "rubrics": sample.get("rubrics", []),
            **res,
        }

    judge_results = list(evaluated.values())
    lock = Lock()
    if samples_to_judge:
        with Pool(processes=args.judge_num_workers) as pool:
            for item in tqdm(
                pool.imap(evaluate_sample, samples_to_judge),
                total=len(samples_to_judge),
                desc="Judge eval",
            ):
                with lock:
                    judge_results.append(item)
        with open(judge_file, "w") as f:
            json.dump(judge_results, f, indent=2)
        print(f"Saved {len(judge_results)} judge results to {judge_file}")
    else:
        print("All samples already evaluated by judge.")


if __name__ == "__main__":
    main()
