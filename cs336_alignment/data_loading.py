import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase


class PackedSFTDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            dataset_path: str | os.PathLike,
            seq_length: int,
            shuffle: bool
        ):
        # Load data
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = [json.loads(line) for line in f]
        
        # Format all samples
        samples = []
        for ex in raw_data:
            sample = (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{ex['prompt']}\n\n### Response:\n{ex['response']}"
        )
            samples.append(sample)
        
        if shuffle:
            import random
            random.shuffle(samples)
        
        # Concatenate all samples with special tokens
        all_samples = "<|end_of_text|><|begin_of_text|>".join(samples)
        
        # Tokenize
        all_samples_encoded = tokenizer.encode(all_samples)
        
        num_samples = (len(all_samples_encoded) - 1) // seq_length
        self.inputs = [all_samples_encoded[seq_length * i:seq_length * (i + 1)] for i in range(num_samples)]
        self.outputs = [all_samples_encoded[seq_length * i + 1:seq_length * (i + 1) + 1] for i in range(num_samples)]
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, i):
        if not (i < self.__len__()):
            raise IndexError(f"Index {i} is out of range for dataset of length {len(self)}.")
        return {
            "input_ids": torch.tensor(self.inputs[i], dtype=torch.long),
            "labels": torch.tensor(self.outputs[i], dtype=torch.long)
        }


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )