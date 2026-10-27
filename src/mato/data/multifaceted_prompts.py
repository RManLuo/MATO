"""
Prompt construction utilities for MATO experiments.

Used by both the Multifaceted-Bench experiment (where preferences come from the
dataset) and the helpful/humor Pareto experiment (where principles + weights
are supplied on the command line).
"""

BASE_SYSTEM_PROMPT = "You are an AI assistant to provide responses to user questions."


def build_preference_prompt(preference_set: list[dict]) -> str:
    """Build a single concatenated preference prompt.

    Each entry of ``preference_set`` is a dict with at least
    ``"preference"`` (short name) and ``"description"`` (rubric description).

    Returns a single system prompt that lists all preferences without weights.
    """
    pref_lines = []
    for pref in preference_set:
        pref_lines.append(
            f"Name: {pref['preference']} Description: {pref['description']}"
        )
    return (
        f"{BASE_SYSTEM_PROMPT} Your responses should follow the following preferences:\n"
        + "\n".join(pref_lines)
    )


def build_individual_preference_prompts(preference_set: list[dict]) -> list[str]:
    """Build one system prompt per preference, each pairing the base prompt
    with a single preference description.
    """
    prompts = []
    for pref in preference_set:
        prompts.append(
            f"{BASE_SYSTEM_PROMPT} Your responses should follow the following preference:\n"
            f"Name: {pref['preference']} Description: {pref['description']}"
        )
    return prompts


def build_weighted_system_prompt(
    principles: list[str],
    weights: list[float],
) -> str:
    """Combine the base system prompt with multiple weighted principles.

    Each principle is tagged with a float weight inline so a single-prompt LLM
    can attend to the weighting at inference time.
    """
    if len(principles) != len(weights):
        raise ValueError(
            f"principles and weights length mismatch: {len(principles)} vs {len(weights)}"
        )

    header = (
        f"{BASE_SYSTEM_PROMPT} Your response should follow the multiple principles "
        "listed below. Each principle is tagged with a weight indicating its relative "
        "importance (higher weight = higher priority). When generating your response, "
        "attend to each principle proportionally to its weight and trade off between "
        "them accordingly."
    )
    parts = [header]
    for idx, (p, w) in enumerate(zip(principles, weights), start=1):
        parts.append(f"\n--- Principle {idx} (weight: {float(w):.2f}) ---\n{p}")
    return "\n".join(parts)


def build_individual_principle_prompts(principles: list[str]) -> list[str]:
    """Build one system prompt per principle, pairing the base prompt with each principle."""
    prompts = []
    for p in principles:
        prompts.append(
            f"{BASE_SYSTEM_PROMPT} Your response should follow the following principle:\n"
            f"{p}"
        )
    return prompts
