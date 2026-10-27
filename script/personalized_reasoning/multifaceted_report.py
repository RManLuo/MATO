"""
Report generation for the Multifaceted-Bench experiment.

Reads judge evaluation results and computes aggregate metrics:
- Overall metrics (Helpfulness, ACS, AMR, APS, Variance, Worst Score)
- By data source
- By preference dimension and subdimension

Supports single-method reporting or multi-method comparison.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser

from mato.evaluation.multifaceted_metrics import (
    compute_all_metrics,
    compute_metrics_by_group,
    compute_metrics_by_dimension,
)

# Mapping from fine-grained source values to the 5 high-level benchmark datasets.
# Sources that map directly by name: alpaca→AlpacaEval 2.0, koala→Koala,
# mt-bench→MT-Bench, self-instruct→Self-Instruct.
# Everything else originates from FLASK (which aggregates many NLP benchmarks).
SOURCE_TO_DATASET = {
    "alpaca": "AlpacaEval 2.0",
    "koala": "Koala",
    "mt-bench": "MT-Bench",
    "self-instruct": "Self-Instruct",
}
FLASK_DATASET = "FLASK"


def map_source_to_dataset(source: str) -> str:
    return SOURCE_TO_DATASET.get(source, FLASK_DATASET)


@dataclass
class ScriptArguments:
    eval_dir: str = field(
        metadata={
            "help": "Path to a judge_eval_results.*.json file, or a directory containing multiple method outputs. "
            "If a directory, searches for judge_eval_results files within."
        },
    )
    judge_model: str = field(
        default="gpt-4o",
        metadata={
            "help": "Judge model name used to select which eval results to load "
            "(e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022'). "
            "Matches the prefix in judge_eval_results.{judge_model}{output_suffix}.json. "
            "Required when eval_dir is a directory and multiple judge results exist."
        },
    )
    output_suffix: str = field(
        default="",
        metadata={
            "help": "Extra suffix used to locate eval files, matching the suffix passed "
            "to multifaceted_judge_eval.py (e.g. '.temp1.0' looks for "
            "judge_eval_results.{judge_model}.temp1.0.json). Also appended to output report filenames."
        },
    )
    methods: str = field(
        default="",
        metadata={
            "help": "Comma-separated list of methods to include (e.g., 'mato,la,mod'). "
            "If empty, all methods are included."
        },
    )
    exp_names: str = field(
        default="",
        metadata={
            "help": "Comma-separated list of experiment names to include (e.g., 'default-100,default-200'). "
            "If empty, all experiments are included. Results are grouped by exp_name."
        },
    )
    tau: float = field(
        default=3.0,
        metadata={"help": "ACS threshold (score >= tau is considered satisfied)."},
    )
    output_format: str = field(
        default="json",
        metadata={"help": "Output format: 'json' or 'csv'."},
    )


def load_eval_results(
    path: str,
    judge_model: str = "",
    output_suffix: str = "",
    methods_filter: list[str] | None = None,
    exp_names_filter: list[str] | None = None,
) -> dict[str, dict[str, list[dict]]]:
    """Load evaluation results from a file or directory, grouped by experiment name.

    Supports any method name (not limited to predefined list). Expects directory structure:
    - Single file: path/to/METHOD/EXP_NAME/judge_eval_results.JUDGE_MODEL.json
    - Directory tree: scans recursively for judge_eval_results files

    Args:
        path: Path to a judge_eval_results.*.json file, or a directory containing multiple method outputs.
        judge_model: If provided, only load files whose name matches
            ``judge_eval_results.{safe_model}{output_suffix}.json``.
        output_suffix: Extra suffix appended after the judge model.
        methods_filter: If provided, only include these methods (case-sensitive).
        exp_names_filter: If provided, only include these experiment names.

    Returns:
        {exp_name: {method_name: [eval_results]}}
    """
    safe_model = judge_model.replace("/", "-").replace(" ", "-") if judge_model else ""
    results = {}  # {exp_name: {method_name: [eval_results]}}

    if os.path.isfile(path):
        # Single file -- use it directly regardless of selectors
        with open(path, "r") as f:
            data = json.load(f)
        parts = path.replace("\\", "/").split("/")

        # Extract method_name and exp_name from path: .../METHOD/EXP_NAME/judge_eval_results.*.json
        method_name = "unknown"
        exp_name = "unknown"

        for i in range(len(parts) - 1, -1, -1):
            if parts[i].startswith("judge_eval_results."):
                # Found eval file, previous dirs are exp_name and method_name
                if i >= 1:
                    exp_name = parts[i - 1]
                if i >= 2:
                    method_name = parts[i - 2]
                break

        # Apply filters
        if methods_filter and method_name not in methods_filter:
            return results
        if exp_names_filter and exp_name not in exp_names_filter:
            return results

        if exp_name not in results:
            results[exp_name] = {}
        results[exp_name][method_name] = data

    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            # Collect matching eval files in this directory
            if safe_model:
                target = f"judge_eval_results.{safe_model}{output_suffix}.json"
                candidates = [f for f in files if f == target]
            else:
                candidates = sorted(
                    f for f in files
                    if f.startswith("judge_eval_results.") and f.endswith(".json")
                )

            if not candidates:
                continue

            if len(candidates) > 1:
                print(
                    f"Warning: multiple judge result files in {root}; "
                    f"using '{candidates[0]}'. Use --judge_model to be explicit."
                )

            fpath = os.path.join(root, candidates[0])
            with open(fpath, "r") as f:
                data = json.load(f)

            # Parse method_name and exp_name from path structure
            # Expect: path/METHOD/EXP_NAME/judge_eval_results.*.json
            rel = os.path.relpath(root, path)
            parts = rel.replace("\\", "/").split("/")

            # Get method_name and exp_name from relative path
            method_name = parts[0] if parts and parts[0] != "." else "unknown"
            exp_name = parts[1] if len(parts) > 1 and parts[1] != "." else "unknown"

            # Apply filters
            if methods_filter and method_name not in methods_filter:
                continue
            if exp_names_filter and exp_name not in exp_names_filter:
                continue

            if exp_name not in results:
                results[exp_name] = {}
            results[exp_name][method_name] = data
            print(f"Loaded {len(data)} results for method '{method_name}' exp '{exp_name}' from {candidates[0]}")
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    return results


def generate_report(eval_results: list[dict], tau: float) -> dict:
    """Generate a full report for a single method."""
    # Filter out error samples
    valid = [r for r in eval_results if r.get("helpfulness_score", -1) > 0]

    # Attach high-level dataset label for grouping
    for r in valid:
        r["_dataset"] = map_source_to_dataset(r.get("source", ""))

    report = {
        "overall": compute_all_metrics(valid, tau=tau),
        "by_dataset": compute_metrics_by_group(valid, "_dataset", tau=tau),
    }

    dim_breakdown = compute_metrics_by_dimension(valid, tau=tau)
    report["by_dimension"] = dim_breakdown["by_dimension"]
    report["by_subdimension"] = dim_breakdown["by_subdimension"]

    return report


def print_report(method: str, report: dict):
    """Print a formatted report to stdout."""
    print(f"\n{'='*60}")
    print(f"  Method: {method}")
    print(f"{'='*60}")

    overall = report["overall"]
    print(f"\n  Overall Metrics (n={overall['num_samples']}, tau={overall['tau']}):")
    print(f"    Helpfulness:          {overall['helpfulness']:.4f}")
    print(f"    AMR:                  {overall['amr']:.4f}")
    print(f"    ACS:                  {overall['acs']:.4f}")
    print(f"    APS:                  {overall['aps']:.4f}")
    print(f"    Pref Variance:        {overall['preference_variance']:.4f}")
    print(f"    Worst Pref Score:     {overall['worst_preference']:.4f}")

    if report.get("by_dataset"):
        print(f"\n  By Dataset (5 benchmarks):")
        for dataset, metrics in sorted(report["by_dataset"].items()):
            print(
                f"    {dataset:20s}  n={metrics['num_samples']:4d}  "
                f"Help={metrics['helpfulness']:.2f}  AMR={metrics['amr']:.2f}  "
                f"ACS={metrics['acs']:.2f}  APS={metrics['aps']:.2f}  "
                f"Var={metrics['preference_variance']:.2f}  "
                f"Worst={metrics['worst_preference']:.2f}"
            )



    if report.get("by_dimension"):
        print(f"\n  By Dimension:")
        for dim, stats in sorted(report["by_dimension"].items()):
            print(
                f"    {dim:30s}  n={stats['count']:5d}  "
                f"mean={stats['mean']:.2f}  satisfied_rate={stats['satisfied_rate']:.2f}"
            )


def print_comparison_table(all_reports: dict[str, dict]):
    """Print a comparison table across all methods."""
    if len(all_reports) <= 1:
        return

    methods = sorted(all_reports.keys())

    # --- Overall ---
    print(f"\n{'='*100}")
    print(f"  COMPARISON TABLE — Overall")
    print(f"{'='*100}")
    header = f"{'Method':20s} | {'Help':>6s} | {'AMR':>6s} | {'ACS':>6s} | {'APS':>6s} | {'Var':>6s} | {'Worst':>6s} | {'N':>5s}"
    print(f"\n  {header}")
    print(f"  {'-'*len(header)}")
    for method in methods:
        o = all_reports[method]["overall"]
        print(
            f"  {method:20s} | {o['helpfulness']:6.2f} | {o['amr']:6.2f} | {o['acs']:6.2f} | "
            f"{o['aps']:6.2f} | {o['preference_variance']:6.2f} | "
            f"{o['worst_preference']:6.2f} | {o['num_samples']:5d}"
        )

    # --- By Dataset ---
    all_datasets = sorted({d for r in all_reports.values() for d in r.get("by_dataset", {})})
    if all_datasets:
        METRIC_KEYS = ["helpfulness", "amr", "acs", "aps", "preference_variance", "worst_preference", "num_samples"]
        METRIC_LABELS = ["Help", "AMR", "ACS", "APS", "Var", "Worst", "N"]
        col_w = 7
        m_w = 20
        header = f"  {'Method':{m_w}s}" + "".join(f" | {lbl:>{col_w}s}" for lbl in METRIC_LABELS)
        sep = f"  {'-'*m_w}" + "".join(f"-+-{'-'*col_w}" for _ in METRIC_LABELS)

        for ds in all_datasets:
            print(f"\n{'='*100}")
            print(f"  COMPARISON TABLE — By Dataset: {ds}")
            print(f"{'='*100}")
            print(header)
            print(sep)
            for method in methods:
                m_data = all_reports[method].get("by_dataset", {}).get(ds, {})
                if m_data:
                    row = f"  {method:{m_w}s}" + "".join(
                        f" | {m_data[k]:>{col_w}d}" if k == "num_samples" else f" | {m_data[k]:>{col_w}.2f}"
                        for k in METRIC_KEYS
                    )
                else:
                    row = f"  {method:{m_w}s}" + "".join(f" | {'N/A':>{col_w}s}" for _ in METRIC_KEYS)
                print(row)

    # --- By Dimension ---
    all_dims = sorted({d for r in all_reports.values() for d in r.get("by_dimension", {})})
    if all_dims:
        print(f"\n{'='*100}")
        print(f"  COMPARISON TABLE — By Dimension")
        print(f"{'='*100}")
        dim_w = 25
        col_w = 6
        hdr = f"  {'Dimension':>{dim_w}s}"
        for m in methods:
            hdr += f" | {m[:col_w*2]:>{col_w*2}s}"
        print(hdr)
        metric_hdr = f"  {'':>{dim_w}s}"
        for _ in methods:
            metric_hdr += f" | {'mean':>{col_w}s}{'sat%':>{col_w}s}"
        print(metric_hdr)
        print(f"  {'-' * (dim_w + (col_w * 2 + 3) * len(methods))}")
        for dim in all_dims:
            row = f"  {dim:>{dim_w}s}"
            for m in methods:
                stats = all_reports[m].get("by_dimension", {}).get(dim, {})
                if stats:
                    row += f" | {stats['mean']:>{col_w}.2f}{stats['satisfied_rate']:>{col_w}.2f}"
                else:
                    row += f" | {'N/A':>{col_w}s}{'N/A':>{col_w}s}"
            print(row)


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Parse method and exp_name filters
    methods_filter = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None
    exp_names_filter = [e.strip() for e in args.exp_names.split(",") if e.strip()] if args.exp_names else None

    # Load results (now grouped by exp_name)
    all_eval_results_by_exp = load_eval_results(
        args.eval_dir,
        judge_model=args.judge_model,
        output_suffix=args.output_suffix,
        methods_filter=methods_filter,
        exp_names_filter=exp_names_filter,
    )

    if not all_eval_results_by_exp:
        hints = []
        if args.judge_model:
            hints.append(f"judge_model='{args.judge_model}'")
        if args.output_suffix:
            hints.append(f"output_suffix='{args.output_suffix}'")
        if args.methods:
            hints.append(f"methods='{args.methods}'")
        if args.exp_names:
            hints.append(f"exp_names='{args.exp_names}'")
        hint_str = f" ({', '.join(hints)})" if hints else ""
        print(f"No evaluation results found{hint_str}.")
        return

    # Suffix for output files to distinguish by judge model + output_suffix
    safe_model = args.judge_model.replace("/", "-").replace(" ", "-") if args.judge_model else ""
    file_suffix = (f".{safe_model}" if safe_model else "") + args.output_suffix

    # Output directory
    if os.path.isfile(args.eval_dir):
        output_dir = os.path.dirname(args.eval_dir)
    else:
        output_dir = args.eval_dir

    # Process each experiment separately
    for exp_name in sorted(all_eval_results_by_exp.keys()):
        all_eval_results = all_eval_results_by_exp[exp_name]
        print(f"\n{'='*80}")
        print(f"  Processing experiment: {exp_name}")
        print(f"{'='*80}")

        # Generate reports for this experiment
        all_reports = {}
        for method, eval_results in sorted(all_eval_results.items()):
            report = generate_report(eval_results, tau=args.tau)
            all_reports[method] = report
            print_report(method, report)

        # Comparison table for this experiment
        print_comparison_table(all_reports)

        # Save individual reports for this experiment
        for method, report in all_reports.items():
            if os.path.isfile(args.eval_dir):
                report_path = os.path.join(output_dir, f"multifaceted_report{file_suffix}.json")
            else:
                method_dir = os.path.join(output_dir, method, exp_name)
                report_path = os.path.join(method_dir, f"multifaceted_report{file_suffix}.json")

            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Report saved: {report_path}")

        # Save combined comparison for this experiment
        if len(all_reports) > 1:
            comparison = {
                method: {
                    "overall": report["overall"],
                    "by_dataset": report.get("by_dataset", {}),
                    "by_dimension": report.get("by_dimension", {}),
                    "by_subdimension": report.get("by_subdimension", {}),
                }
                for method, report in all_reports.items()
            }
            # Include exp_name in comparison filename to distinguish multiple experiments
            exp_suffix = f"_{exp_name}" if exp_name != "unknown" else ""
            comparison_path = os.path.join(output_dir, f"comparison{file_suffix}{exp_suffix}.json")
            os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
            with open(comparison_path, "w") as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            print(f"Comparison saved: {comparison_path}")


if __name__ == "__main__":
    main()
