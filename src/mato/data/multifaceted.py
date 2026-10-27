import datasets

from .base_dataset import BaseDataset, DatasetRegistry


@DatasetRegistry.register("multifaceted")
class Multifaceted(BaseDataset):
    def __init__(self, dataset_name_or_path: str = "kaist-ai/Multifaceted-Bench", split: str = "train", size=None):
        raw_data = datasets.load_dataset(dataset_name_or_path, split=split)

        if size is not None:
            if size > len(raw_data):
                raise ValueError(f"Requested size {size} exceeds dataset size {len(raw_data)}.")
            raw_data = raw_data.train_test_split(test_size=size, seed=42)['test']

        # Add explicit ID column
        raw_data = raw_data.map(lambda example, idx: {"id": idx}, with_indices=True)
        self.data = raw_data
