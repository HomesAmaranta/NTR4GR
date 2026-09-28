import numpy as np
import pandas as pd
from torch.utils.data import Dataset


IGNORE_INDEX = -100


def item2code(code_path, codebook_size=256):
    """Map one-based item IDs to disjoint SID-token IDs."""
    codes = np.load(code_path, allow_pickle=True)
    if codes.ndim != 2:
        raise ValueError(f"Expected a 2-D SID code array, got shape {codes.shape}")

    item_to_code = {}
    code_to_item = {}
    for index, code in enumerate(codes):
        if any(int(value) < 0 or int(value) >= codebook_size for value in code):
            raise ValueError(
                f"ItemID={index + 1} contains a code outside "
                f"[0, {codebook_size})"
            )
        offsets = tuple(
            int(value) + level * codebook_size + 1
            for level, value in enumerate(code)
        )
        item_id = index + 1
        item_to_code[item_id] = offsets
        code_to_item[offsets] = item_id
    return item_to_code, code_to_item


def _flatten_codes(item_ids, item_to_code):
    tokens = []
    for item_id in item_ids:
        item_id = int(item_id)
        if item_id not in item_to_code:
            raise KeyError(f"ItemID={item_id} is missing from the SID codebook")
        tokens.extend(item_to_code[item_id])
    return tokens


def process_data(file_path, mode, max_len):
    """Build one sample per parquet row without expanding sequence prefixes."""
    if mode not in {"train", "valid", "test", "evaluation"}:
        raise ValueError("mode must be 'train', 'valid', 'test', or 'evaluation'")
    if max_len <= 0:
        raise ValueError("max_len must be positive")

    data = pd.read_parquet(file_path)
    required = {"history", "target"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Parquet file is missing columns: {sorted(missing)}")

    processed_data = []
    for row in data.itertuples(index=False):
        history = [int(item_id) for item_id in row.history][-max_len:]
        target = int(row.target)

        if mode == "train":
            sequence = (history + [target])[-max_len:]
            if len(sequence) < 2:
                continue
            processed_data.append({"sequence": sequence})
        else:
            if not history:
                continue
            processed_data.append({"history": history, "target": target})
    return processed_data


class GenRecDataset(Dataset):
    """SID-level next-token dataset with one sample per user sequence."""

    def __init__(
        self,
        dataset_path,
        code_path,
        mode,
        max_len,
        pad_token=0,
        codebook_size=256,
    ):
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.mode = "test" if mode == "evaluation" else mode
        self.max_len = max_len
        self.pad_token = pad_token
        self.codebook_size = codebook_size
        self.item_to_code, self.code_to_item = item2code(
            code_path, codebook_size=codebook_size
        )
        if not self.item_to_code:
            raise ValueError("SID codebook is empty")
        self.code_length = len(next(iter(self.item_to_code.values())))
        self.vocab_size = self.code_length * codebook_size + 1
        self.data = self._prepare_data()

    def _prepare_data(self):
        rows = process_data(self.dataset_path, self.mode, self.max_len)
        samples = []
        for row in rows:
            if self.mode == "train":
                input_ids = _flatten_codes(row["sequence"], self.item_to_code)
                labels = [IGNORE_INDEX] * self.code_length + input_ids[self.code_length :]
                samples.append({"input_ids": input_ids, "labels": labels})
                continue

            history = _flatten_codes(row["history"], self.item_to_code)
            target = list(self.item_to_code[row["target"]])
            if self.mode == "valid":
                input_ids = history + target
                labels = [IGNORE_INDEX] * len(history) + target
                samples.append(
                    {
                        "input_ids": input_ids,
                        "labels": labels,
                        "target": target,
                    }
                )
            else:
                samples.append({"input_ids": history, "target": target})
        return samples

    def candidate_codes(self):
        return sorted(set(self.item_to_code.values()))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = GenRecDataset(
        "../data/Beauty/train.parquet",
        "../data/Beauty/Beauty_t5_rqvae.npy",
        mode="train",
        max_len=20,
    )
    print("Number of user sequences:", len(dataset))
    print("First sample:", dataset[0])
