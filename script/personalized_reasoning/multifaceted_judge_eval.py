"""
Judge evaluation runner for the Multifaceted-Bench experiment.

Reads inference results and evaluates each sample using LLM-as-judge
against the sample's rubrics + a helpfulness rubric.

Supports multi-threaded evaluation for faster processing.
"""

import json
import os
from multiprocessing.dummy import Pool
from dataclasses import dataclass, field
from threading import Lock

from tqdm import tqdm
from transformers import HfArgumentParser

from mato.evaluation.llm_judge import LLMJudge, JudgeConfig
import dotenv

dotenv.load_dotenv()  # Load environment variables from .env file


@dataclass
class ScriptArguments:
    result_file: str = field(
        metadata={"help": "Path to results.json from inference."},
    )
    judge_provider: str = field(
        default="openai",
        metadata={"help": "Judge LLM provider: 'openai', 'azure_openai', or 'anthropic'."},
    )
    judge_model: str = field(
        default="gpt-4o",
        metadata={"help": "Judge model name (e.g., 'gpt-4o', 'claude-sonnet-4-20250514'). "
                  "For Azure OpenAI this is the deployment name."},
    )
    azure_api_version: str = field(
        default="2024-08-01-preview",
        metadata={"help": "Azure OpenAI API version. Only used when judge_provider='azure_openai'. "
                  "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are read from environment."},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for judge LLM."},
    )
    max_tokens: int = field(
        default=1024,
        metadata={"help": "Max tokens for judge response."},
    )
    max_retries: int = field(
        default=3,
        metadata={"help": "Max retries for API calls."},
    )
    resume: bool = field(
        default=True,
        metadata={"help": "Resume from partial results if available."},
    )
    output_suffix: str = field(
        default="",
        metadata={"help": "Extra suffix appended to the output filename after the judge model name, "
                  "e.g. '.temp1.0' produces judge_eval_results.gpt-4o.temp1.0.json."},
    )
    save_interval: int = field(
        default=10,
        metadata={"help": "Save partial results every N samples."},
    )
    num_workers: int = field(
        default=10,
        metadata={"help": "Number of parallel worker threads for evaluation."},
    )


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Load results
    with open(args.result_file, "r") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} samples from {args.result_file}")

    # Initialize judge
    judge = LLMJudge(
        JudgeConfig(
            provider=args.judge_provider,
            model=args.judge_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )
    )

    # Output path: distinguish results by judge model name
    output_dir = os.path.dirname(args.result_file)
    safe_model = args.judge_model.replace("/", "-").replace(" ", "-")
    eval_file = os.path.join(output_dir, f"judge_eval_results.{safe_model}{args.output_suffix}.json")

    # Resume support: load existing partial results
    evaluated = {}
    if args.resume and os.path.exists(eval_file):
        with open(eval_file, "r") as f:
            existing = json.load(f)
        evaluated = {item["id"]: item for item in existing}
        print(f"Resuming: found {len(evaluated)} already evaluated samples")

    # Helper function for evaluating a single sample (runs in thread)
    def evaluate_single_sample(sample):
        sample_id = sample["id"]
        try:
            judge_result = judge.evaluate_sample(
                prompt=sample["input"],
                response=sample["output"],
                rubrics=sample["rubrics"],
            )
        except Exception as e:
            print(f"Error evaluating sample {sample_id}: {e}", flush=True)
            # Record error but continue
            judge_result = {
                "preference_scores": [-1] * len(sample["rubrics"]),
                "helpfulness_score": -1,
                "preference_justifications": [f"ERROR: {e}"] * len(sample["rubrics"]),
                "helpfulness_justification": f"ERROR: {e}",
            }

        eval_item = {
            "id": sample_id,
            "source": sample.get("source", ""),
            "input": sample["input"],
            "output": sample["output"],
            "preference_set": sample["preference_set"],
            "rubrics": sample["rubrics"],
            "preference_scores": judge_result["preference_scores"],
            "helpfulness_score": judge_result["helpfulness_score"],
            "preference_justifications": judge_result["preference_justifications"],
            "helpfulness_justification": judge_result["helpfulness_justification"],
        }
        return eval_item

    # Identify samples that need evaluation
    samples_to_evaluate = [s for s in results if s["id"] not in evaluated]
    print(f"Need to evaluate {len(samples_to_evaluate)} new samples")

    # Multi-threaded evaluation using multiprocessing.dummy.Pool
    eval_results = list(evaluated.values())  # Start with already evaluated samples
    results_lock = Lock()
    completed_count = 0

    with Pool(processes=args.num_workers) as pool:
        for eval_item in tqdm(
            pool.imap(evaluate_single_sample, samples_to_evaluate),
            total=len(samples_to_evaluate),
            desc="Judge evaluation",
        ):
            # Thread-safe result collection
            with results_lock:
                eval_results.append(eval_item)
                completed_count += 1

                # Periodic save for fault tolerance
                if completed_count % args.save_interval == 0:
                    with open(eval_file, "w") as f:
                        json.dump(eval_results, f, indent=2, ensure_ascii=False)
                    print(f"  Saved {len(eval_results)} results so far", flush=True)

    # Final save
    with open(eval_file, "w") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

    # Print summary
    valid_results = [r for r in eval_results if r["helpfulness_score"] > 0]
    if valid_results:
        avg_helpfulness = sum(r["helpfulness_score"] for r in valid_results) / len(
            valid_results
        )
        all_pref_scores = []
        for r in valid_results:
            all_pref_scores.extend(
                s for s in r["preference_scores"] if s > 0
            )
        avg_pref = sum(all_pref_scores) / len(all_pref_scores) if all_pref_scores else 0

        print(f"\nEvaluation complete: {len(eval_results)} samples")
        print(f"  Valid: {len(valid_results)}, Errors: {len(eval_results) - len(valid_results)}")
        print(f"  Avg Helpfulness: {avg_helpfulness:.2f}")
        print(f"  Avg Preference Score: {avg_pref:.2f}")
    else:
        print(f"\nEvaluation complete: {len(eval_results)} samples (all errors)")

    print(f"Results saved to {eval_file}  (judge: {args.judge_model})")


if __name__ == "__main__":
    main()
