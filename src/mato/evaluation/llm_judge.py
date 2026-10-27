"""
LLM-as-judge evaluation for the Multifaceted-Bench experiment.

Supports OpenAI (GPT-4o, etc.), Azure OpenAI, and Anthropic (Claude) as judge backends.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional


JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. You will be given a user's instruction and an AI assistant's response. You will also be given a specific evaluation criterion with detailed score descriptions.

Please evaluate the response based ONLY on the given criterion.

## User Instruction
{prompt}

## AI Assistant's Response
{response}

## Evaluation Criterion
{criteria}

## Score Descriptions
- Score 1: {score_1}
- Score 2: {score_2}
- Score 3: {score_3}
- Score 4: {score_4}
- Score 5: {score_5}

Based on the criterion above, assign a score from 1 to 5. Provide a brief justification, then state your final score.

Output your final score in the format: [[score]]
"""

HELPFULNESS_RUBRIC = {
    "criteria": "Is the response helpful, relevant, and useful for addressing the user's instruction?",
    "score_descriptions": {
        "1": "The response is completely unhelpful, irrelevant, or fails to address the instruction.",
        "2": "The response is minimally helpful, addressing only a small part of the instruction with significant gaps.",
        "3": "The response is moderately helpful, addressing the main aspects of the instruction but with some gaps or inaccuracies.",
        "4": "The response is helpful, addressing the instruction well with only minor gaps or areas for improvement.",
        "5": "The response is exceptionally helpful, thoroughly and accurately addressing all aspects of the instruction.",
    },
}


def parse_score(text: str) -> int:
    """Extract score from [[score]] pattern in judge response."""
    match = re.search(r"\[\[(\d)\]\]", text)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return score
    # Fallback: look for "Score: N" pattern
    match = re.search(r"[Ss]core:\s*(\d)", text)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return score
    # Fallback: look for standalone digit after common patterns
    match = re.search(r"(?:final score|rating|grade)[:\s]*(\d)", text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return score
    raise ValueError(f"Could not parse score from judge response: {text[:300]}")


@dataclass
class JudgeConfig:
    provider: str = "openai"  # "openai", "azure_openai", or "anthropic"
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 3
    retry_delay: float = 2.0
    # Azure OpenAI specific (read from env vars AZURE_OPENAI_ENDPOINT /
    # AZURE_OPENAI_API_KEY if not set here)
    azure_endpoint: Optional[str] = None
    azure_api_version: str = "2025-01-01-preview"


class LLMJudge:
    """LLM-based judge for rubric evaluation."""

    def __init__(self, config: Optional[JudgeConfig] = None):
        if config is None:
            config = JudgeConfig()
        self.config = config

        if config.provider == "openai":
            from openai import OpenAI

            self.client = OpenAI()
        elif config.provider == "azure_openai":
            from openai import AzureOpenAI

            self.client = AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_version=config.azure_api_version,
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            )
        elif config.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic()
        else:
            raise ValueError(
                f"Unknown provider: {config.provider}. Must be 'openai', 'azure_openai', or 'anthropic'."
            )

    @staticmethod
    def _parse_sse_response(raw: str) -> str:
        """Parse raw SSE (Server-Sent Events) text into plain text content."""
        text_parts = []
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if data.get("type") == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_parts.append(delta["text"])
                except (json.JSONDecodeError, KeyError):
                    pass
        return "".join(text_parts)

    def _call_api(self, judge_prompt: str) -> str:
        """Call the LLM API with retries."""
        for attempt in range(self.config.max_retries):
            try:
                if self.config.provider in ("openai", "azure_openai"):
                    response = self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": judge_prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    return response.choices[0].message.content

                elif self.config.provider == "anthropic":
                    api_kwargs = dict(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        messages=[{"role": "user", "content": judge_prompt}],
                    )
                    # Only pass temperature if non-zero (some proxies reject 0.0)
                    if self.config.temperature > 0:
                        api_kwargs["temperature"] = self.config.temperature
                    response = self.client.messages.create(**api_kwargs)
                    # Handle various SDK response formats
                    if isinstance(response, str):
                        # Some environments return raw SSE text
                        return self._parse_sse_response(response) or response
                    return response.content[0].text

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    wait = self.config.retry_delay * (2**attempt)
                    print(
                        f"API call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"API call failed after {self.config.max_retries} attempts: {e}"
                    ) from e

    def _format_judge_prompt(self, prompt: str, response: str, rubric: dict) -> str:
        """Format the judge prompt from a rubric."""
        score_descs = rubric["score_descriptions"]
        return JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt,
            response=response,
            criteria=rubric["criteria"],
            score_1=score_descs["1"],
            score_2=score_descs["2"],
            score_3=score_descs["3"],
            score_4=score_descs["4"],
            score_5=score_descs["5"],
        )

    def evaluate_single(
        self, prompt: str, response: str, rubric: dict
    ) -> dict:
        """Evaluate a single (prompt, response) against a single rubric.

        Returns: {"score": int, "justification": str}
        """
        judge_prompt = self._format_judge_prompt(prompt, response, rubric)
        judge_response = self._call_api(judge_prompt)

        try:
            score = parse_score(judge_response)
        except ValueError:
            # Retry once with explicit instruction
            retry_prompt = (
                judge_prompt
                + "\n\nIMPORTANT: You must end your response with your score in the format [[score]], "
                "where score is an integer from 1 to 5."
            )
            judge_response = self._call_api(retry_prompt)
            score = parse_score(judge_response)

        return {"score": score, "justification": judge_response}

    def evaluate_sample(
        self, prompt: str, response: str, rubrics: list[dict]
    ) -> dict:
        """Evaluate against all preference rubrics + helpfulness rubric.

        Args:
            prompt: The user instruction.
            response: The AI assistant's response.
            rubrics: List of rubric dicts from the dataset (one per preference dimension).

        Returns:
            {
                "preference_scores": [score1, score2, score3, score4],
                "helpfulness_score": int,
                "preference_justifications": [str, str, str, str],
                "helpfulness_justification": str,
            }
        """
        preference_scores = []
        preference_justifications = []
        for rubric in rubrics:
            result = self.evaluate_single(prompt, response, rubric)
            preference_scores.append(result["score"])
            preference_justifications.append(result["justification"])

        # Helpfulness evaluation
        help_result = self.evaluate_single(prompt, response, HELPFULNESS_RUBRIC)

        return {
            "preference_scores": preference_scores,
            "helpfulness_score": help_result["score"],
            "preference_justifications": preference_justifications,
            "helpfulness_justification": help_result["justification"],
        }
