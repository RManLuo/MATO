"""
Metrics computation for the Multifaceted-Bench experiment.

Computes: Helpfulness, ACS, AMR, APS, Preference Score Variance, Worst Preference Score.
- Helpfulness: Average helpfulness score
- ACS: Average Constraint Satisfaction (mean count of preferences >= tau)
- AMR: All-Match Rate (proportion of samples satisfying ALL preferences >= tau)
- APS: Average Preference Score
- Pref Variance: Average variance of preference scores per sample
- Worst Pref Score: Average minimum preference score per sample

Supports grouping by source, dimension, and subdimension.
"""

import numpy as np


def compute_helpfulness(eval_results: list[dict]) -> float:
    """Average helpfulness score across all samples."""
    scores = [r["helpfulness_score"] for r in eval_results]
    return float(np.mean(scores))


def compute_acs(eval_results: list[dict], tau: float = 4.0) -> float:
    """Average Constraint Satisfaction.

    For each sample, count how many of the 4 preference scores >= tau.
    Return the mean count. Range: [0, 4].
    """
    counts = []
    for r in eval_results:
        count = sum(1 for s in r["preference_scores"] if s >= tau)
        counts.append(count)
    return float(np.mean(counts))


def compute_aps(eval_results: list[dict]) -> float:
    """Average Preference Score. Mean of all preference dimension scores."""
    all_scores = []
    for r in eval_results:
        all_scores.extend(r["preference_scores"])
    return float(np.mean(all_scores))


def compute_preference_variance(eval_results: list[dict]) -> float:
    """Average variance of preference scores across dimensions per sample."""
    variances = []
    for r in eval_results:
        variances.append(float(np.var(r["preference_scores"])))
    return float(np.mean(variances))


def compute_worst_preference(eval_results: list[dict]) -> float:
    """Average of the minimum preference score per sample."""
    worst_scores = []
    for r in eval_results:
        worst_scores.append(min(r["preference_scores"]))
    return float(np.mean(worst_scores))


def compute_amr(eval_results: list[dict], tau: float = 4.0) -> float:
    """All-Match Rate: Strict satisfaction rate.

    For each sample, check if ALL preference scores >= tau.
    Return the proportion of samples that satisfy all preferences.
    Range: [0, 1].
    """
    amr_scores = []
    for r in eval_results:
        all_satisfied = all(s >= tau for s in r["preference_scores"])
        amr_scores.append(1.0 if all_satisfied else 0.0)
    return float(np.mean(amr_scores))


def compute_all_metrics(eval_results: list[dict], tau: float = 4.0) -> dict:
    """Compute all metrics for a set of evaluation results."""
    return {
        "helpfulness": round(compute_helpfulness(eval_results), 4),
        "acs": round(compute_acs(eval_results, tau=tau), 4),
        "amr": round(compute_amr(eval_results, tau=tau), 4),
        "aps": round(compute_aps(eval_results), 4),
        "preference_variance": round(compute_preference_variance(eval_results), 4),
        "worst_preference": round(compute_worst_preference(eval_results), 4),
        "tau": tau,
        "num_samples": len(eval_results),
    }


def compute_metrics_by_group(
    eval_results: list[dict], group_key: str, tau: float = 4.0
) -> dict:
    """Compute metrics grouped by a field (e.g., 'source')."""
    groups = {}
    for r in eval_results:
        key = r.get(group_key, "unknown")
        groups.setdefault(key, []).append(r)
    return {k: compute_all_metrics(v, tau=tau) for k, v in sorted(groups.items())}


def compute_metrics_by_dimension(
    eval_results: list[dict], tau: float = 4.0
) -> dict:
    """Compute per-dimension and per-subdimension score breakdowns.

    Uses the preference_set metadata to map each score to its dimension/subdimension.

    Returns:
        {
            "by_dimension": {"style": {"mean": ..., "count": ..., "satisfied_rate": ...}, ...},
            "by_subdimension": {"style/conciseness": {"mean": ..., "count": ..., "satisfied_rate": ...}, ...},
        }
    """
    dimension_scores = {}
    subdimension_scores = {}

    for r in eval_results:
        for i, pref in enumerate(r["preference_set"]):
            dim = pref["dimension"]
            subdim = pref.get("subdimension", "unknown")
            score = r["preference_scores"][i]

            dimension_scores.setdefault(dim, []).append(score)
            key = f"{dim}/{subdim}"
            subdimension_scores.setdefault(key, []).append(score)

    def _summarize(scores, tau):
        arr = np.array(scores)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "count": len(scores),
            "satisfied_rate": round(float(np.mean(arr >= tau)), 4),
        }

    return {
        "by_dimension": {
            k: _summarize(v, tau) for k, v in sorted(dimension_scores.items())
        },
        "by_subdimension": {
            k: _summarize(v, tau) for k, v in sorted(subdimension_scores.items())
        },
    }
