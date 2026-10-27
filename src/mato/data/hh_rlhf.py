"""HH-RLHF dataset wrapper for MATO inference."""

import re

import datasets

from .base_dataset import BaseDataset, DatasetRegistry


@DatasetRegistry.register("hh-rlhf")
class HHRLHF(BaseDataset):
    """Subset of Anthropic's HH-RLHF (test split) for multi-objective inference.

    Each example exposes a `prompt` field that is the parsed multi-turn chat
    history (a list of {"role", "content"} messages) so it can be fed directly
    through a tokenizer's chat template.
    """

    def __init__(
        self,
        dataset_name_or_path: str = "Anthropic/hh-rlhf",
        split: str = "test",
        size: int | None = 100,
    ):
        raw_data = datasets.load_dataset(dataset_name_or_path, split=split)
        if size is not None:
            if size > len(raw_data):
                raise ValueError(
                    f"Requested size {size} exceeds dataset size {len(raw_data)}."
                )
            raw_data = raw_data.train_test_split(test_size=size, seed=42)["test"]

        def process(sample):
            text = sample.get("chosen") or ""
            if not isinstance(text, str):
                text = str(text)

            # Split while preserving the delimiters 'Human:' and 'Assistant:'
            tokens = re.split(r"(Human:|Assistant:)", text)
            messages = []
            for i in range(1, len(tokens), 2):
                role_token = tokens[i].rstrip(":").strip()
                content = tokens[i + 1].strip()
                if not content:
                    continue
                role = "user" if role_token.lower().startswith("human") else "assistant"
                messages.append({"role": role, "content": content})

            # Drop any trailing assistant message (the response we want to generate)
            if messages and messages[-1]["role"] == "assistant":
                messages.pop()

            return {"prompt": messages}

        self.data = raw_data.map(
            process,
            batched=False,
            load_from_cache_file=False,
            remove_columns=raw_data.column_names,
        )


if __name__ == "__main__":
    hh_rlhf = HHRLHF(size=100)
    print(len(hh_rlhf.data))
    print(hh_rlhf.data[0])
