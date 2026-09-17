import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def item2code(code_path, codebook_size=256):
    data = np.load(code_path, allow_pickle=True)
    item_to_code = {}
    code_to_item = {}
    for index, code in enumerate(data):
        offsets = [int(c) + phase * codebook_size + 1 for phase, c in enumerate(code)]
        item_to_code[index + 1] = offsets
        code_to_item[tuple(offsets)] = index + 1
    return item_to_code, code_to_item


def load_item_embeddings(item_emb_path):
    data = pd.read_parquet(item_emb_path)

    def to_array(embedding):
        arr = np.asarray(embedding)
        if arr.dtype == object:
            arr = np.stack(embedding)
        return arr.astype(np.float32)

    return {
        int(row.ItemID): to_array(row.embedding)
        for row in data.itertuples(index=False)
    }


class SourceToNextSidDataset(Dataset):
    """Use one source item's embedding to predict a later item's 4-code SID."""

    def __init__(
        self,
        dataset_path,
        code_path,
        item_emb_path,
        mode,
        source_offset=0,
        max_source_offset=2,
        codebook_size=256,
    ):
        if mode not in {"train", "evaluation"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if source_offset < 0:
            raise ValueError(f"source_offset must be >= 0, got {source_offset}")
        if max_source_offset < source_offset:
            raise ValueError(
                "max_source_offset must be >= source_offset so all ablations "
                "can share the same filtered samples"
            )

        self.dataset_path = dataset_path
        self.mode = mode
        self.source_offset = source_offset
        self.max_source_offset = max_source_offset
        self.item_to_code, self.code_to_item = item2code(
            code_path, codebook_size=codebook_size
        )
        self.item_embeddings = load_item_embeddings(item_emb_path)
        self.embedding_dim = len(next(iter(self.item_embeddings.values())))
        self.data = self._prepare_data()

    def _prepare_data(self):
        raw = pd.read_parquet(self.dataset_path)
        samples = []
        min_target_index = self.max_source_offset + 1

        for row in raw.itertuples(index=False):
            sequence = list(row.history) + [row.target]
            if len(sequence) <= min_target_index:
                continue

            target_indices = (
                range(min_target_index, len(sequence))
                if self.mode == "train"
                else [len(sequence) - 1]
            )
            for target_index in target_indices:
                source_index = target_index - 1 - self.source_offset
                if source_index < 0:
                    continue
                source_item = int(sequence[source_index])
                target_item = int(sequence[target_index])
                if source_item not in self.item_embeddings:
                    raise KeyError(f"Missing source item embedding: {source_item}")
                if target_item not in self.item_to_code:
                    raise KeyError(f"Missing target SID code: {target_item}")
                samples.append(
                    {
                        "source_item": source_item,
                        "target_item": target_item,
                        "source_emb": self.item_embeddings[source_item],
                        "target_code": self.item_to_code[target_item],
                    }
                )
        return samples

    def __getitem__(self, index):
        item = self.data[index]
        return {
            "source_item": torch.tensor(item["source_item"], dtype=torch.long),
            "target_item": torch.tensor(item["target_item"], dtype=torch.long),
            "source_emb": torch.tensor(item["source_emb"], dtype=torch.float32),
            "target_code": torch.tensor(item["target_code"], dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)
