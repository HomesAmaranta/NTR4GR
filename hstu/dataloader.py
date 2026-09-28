from functools import partial

import torch
from torch.utils.data import DataLoader

try:
    from .dataset import IGNORE_INDEX
except ImportError:
    from dataset import IGNORE_INDEX


class GenRecDataLoader(DataLoader):
    """Right-pad SID sequences while preserving per-token NTP labels."""

    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=partial(
                self.collate_fn, pad_token=dataset.pad_token
            ),
        )

    @staticmethod
    def collate_fn(batch, pad_token=0):
        max_length = max(len(sample["input_ids"]) for sample in batch)
        input_ids = []
        attention_masks = []
        labels = []

        for sample in batch:
            ids = sample["input_ids"]
            pad_length = max_length - len(ids)
            input_ids.append(ids + [pad_token] * pad_length)
            attention_masks.append([1] * len(ids) + [0] * pad_length)
            if "labels" in sample:
                labels.append(
                    sample["labels"] + [IGNORE_INDEX] * pad_length
                )

        output = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "lengths": torch.tensor(
                [len(sample["input_ids"]) for sample in batch], dtype=torch.long
            ),
        }
        if labels:
            output["labels"] = torch.tensor(labels, dtype=torch.long)
        if "target" in batch[0]:
            output["target"] = torch.tensor(
                [sample["target"] for sample in batch], dtype=torch.long
            )
        return output
