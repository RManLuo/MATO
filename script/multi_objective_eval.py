from mato.evaluation.multi_objective_eval import (
    compute_hypervolume,
    compute_inner_product,
)
import numpy as np
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List
import os
import json


@dataclass
class ScriptArguments:
    result_path: str
    reward_names: str = field(
        default="harmless,helpful,humor",
        metadata={
            "help": "Comma-separated names of the rewards. need to match the order of weights."
        },
    )
    weights: Optional[List[float]] = field(
        default_factory=list,
        metadata={
            "help": "Weights for each objective. Must be provided if the result file does not contain them."
        },
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    result_path = args.result_path
    reward_names = args.reward_names.split(",")
    # Check if the result path contains subdirectories
    subdirs = [
        d
        for d in os.listdir(result_path)
        if os.path.isdir(os.path.join(result_path, d))
    ]
    if len(subdirs) > 0:
        all_solutions = []
        all_preferences = []
        for subdir in subdirs:
            subdir_path = os.path.join(result_path, subdir)
            # Read args
            if os.path.exists(os.path.join(subdir_path, "args.json")):
                with open(os.path.join(subdir_path, "args.json"), "r") as f:
                    run_args = json.load(f)
                weights = run_args.get("weights", args.weights)
                 # Print result for debugging
                exp_name = run_args.get("exp_name", "N/A")
            else:
                weights = [float(w) for w in subdir.split("_")]
                exp_name = result_path.split("/")[-1]
            # Read results
            with open(os.path.join(subdir_path, "reward_eval_results.json"), "r") as f:
                results = json.load(f)

            print(
                f"{exp_name} - Weights: {weights} - Results: {', '.join(f'{r}: {results[r]:.4f}' for r in reward_names)}"
            )

            solutions = []
            for reward_name in reward_names:
                if reward_name not in results:
                    raise ValueError(
                        f"Reward name {reward_name} not found in results of {subdir_path}."
                    )
                solutions.append(results[reward_name])
            all_solutions.append(solutions)
            all_preferences.append(weights)
        all_solutions = np.array(all_solutions)
        all_preferences = np.array(all_preferences)
        result_file_name = f"summary_result_{args.reward_names.replace(',', '_')}.json"
    else:
        # Read args
        with open(os.path.join(result_path, "args.json"), "r") as f:
            run_args = json.load(f)
        weights = run_args.get("weights", args.weights)
        if not weights:
            raise ValueError(
                "Weights must be provided either in args or in the args.json file."
            )
        # Read results
        with open(os.path.join(result_path, "reward_eval_results.json"), "r") as f:
            results = json.load(f)
        solutions = []
        for reward_name in reward_names:
            if reward_name not in results:
                raise ValueError(f"Reward name {reward_name} not found in results.")
            solutions.append(results[reward_name])
        all_solutions = np.array([solutions])
        all_preferences = np.array([weights])
        result_file_name = f"summary_result_{args.reward_names.replace(',', '_')}_{'_'.join(map(str, weights))}.json"

    # Sort all_solutions and all_preferences based on first value of all_preferences weights
    sorted_indices = np.argsort(-all_preferences[:, 0])
    all_solutions = all_solutions[sorted_indices]
    all_preferences = all_preferences[sorted_indices]

    ip_result = compute_inner_product(all_solutions, all_preferences)
    hv_result = compute_hypervolume(all_solutions)

    ip_result = np.round(ip_result, 4)
    hv_result = np.round(hv_result, 4)

    print(f"Hypervolume Result: {hv_result}")
    print(f"Inner Product Result: {ip_result}")

    # Dump the results for later pareto front drawing
    summary_result = {
        "reward_names": reward_names,
        "solutions": all_solutions.tolist(),
        "preferences": all_preferences.tolist(),
        "hypervolume": hv_result,
        "inner_product": ip_result,
    }
    with open(os.path.join(result_path, result_file_name), "w") as f:
        json.dump(summary_result, f, indent=1)
    print(f"Summary results saved to {os.path.join(result_path, result_file_name)}")
