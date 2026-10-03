from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


HSTU_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HSTU_DIR))

from dataloader import GenRecDataLoader
from dataset import GenRecDataset, IGNORE_INDEX
from generation_trie import Trie, prefix_allowed_tokens_fn
from model_hstu import HSTURec


def _dataset(tmp_path, monkeypatch, mode):
    codes = np.asarray(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3],
        ],
        dtype=np.int64,
    )
    code_path = tmp_path / "codes.npy"
    np.save(code_path, codes)
    frame = pd.DataFrame(
        [{"user": 1, "history": [1, 2, 3], "target": 4}]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: frame)
    return GenRecDataset(
        "unused.parquet",
        code_path,
        mode=mode,
        max_len=20,
        codebook_size=4,
    )


def test_train_keeps_one_sequence_and_supervises_b_c_d(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch, mode="train")
    assert len(dataset) == 1
    sample = dataset[0]
    assert len(sample["input_ids"]) == 16
    assert sample["labels"][:4] == [IGNORE_INDEX] * 4
    assert sample["labels"][4:] == sample["input_ids"][4:]
    assert sum(label != IGNORE_INDEX for label in sample["labels"]) == 12


def test_train_keeps_all_80_items_and_supervises_after_first(
    tmp_path, monkeypatch
):
    codes = np.asarray(
        [[item_id % 4] * 4 for item_id in range(80)],
        dtype=np.int64,
    )
    code_path = tmp_path / "long_codes.npy"
    np.save(code_path, codes)
    frame = pd.DataFrame(
        [{"user": 1, "history": list(range(1, 80)), "target": 80}]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: frame)

    dataset = GenRecDataset(
        "unused.parquet",
        code_path,
        mode="train",
        max_len=20,
        codebook_size=4,
    )
    sample = dataset[0]
    assert len(sample["input_ids"]) == 80 * 4
    assert sample["labels"][:4] == [IGNORE_INDEX] * 4
    assert sum(label != IGNORE_INDEX for label in sample["labels"]) == 79 * 4


def test_validation_only_supervises_final_item(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch, mode="valid")
    sample = dataset[0]
    assert len(sample["input_ids"]) == 16
    assert sample["labels"][:12] == [IGNORE_INDEX] * 12
    assert sample["labels"][12:] == sample["target"]


def test_collator_right_pads_and_preserves_labels():
    batch = GenRecDataLoader.collate_fn(
        [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "labels": [-100, 5]},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]]


def _model():
    torch.manual_seed(0)
    return HSTURec(
        vocab_size=17,
        max_seq_len=20,
        code_length=4,
        embedding_dim=8,
        num_blocks=2,
        num_heads=2,
        dqk=4,
        dv=4,
        dropout_rate=0.0,
    )


def test_model_is_causal_and_lm_head_is_tied():
    model = _model().eval()
    first = torch.tensor([[1, 5, 9, 13, 2, 6]])
    second = first.clone()
    second[0, -1] = 7
    mask = torch.ones_like(first)
    with torch.no_grad():
        first_hidden = model.encode(first, mask)
        second_hidden = model.encode(second, mask)
    torch.testing.assert_close(first_hidden[:, :-1], second_hidden[:, :-1])
    assert model.lm_head.weight.data_ptr() == model.sid_emb.weight.data_ptr()


def test_attention_is_limited_to_item_history_window():
    torch.manual_seed(0)
    model = HSTURec(
        vocab_size=17,
        max_seq_len=4,
        code_length=1,
        max_history_items=2,
        embedding_dim=8,
        num_blocks=1,
        num_heads=2,
        dqk=4,
        dv=4,
        dropout_rate=0.0,
    ).eval()
    first = torch.tensor([[1, 2, 3]])
    changed_outside_window = torch.tensor([[4, 2, 3]])
    mask = torch.ones_like(first)
    with torch.no_grad():
        first_hidden = model.encode(first, mask)
        changed_hidden = model.encode(changed_outside_window, mask)
    torch.testing.assert_close(first_hidden[:, -1], changed_hidden[:, -1])


def test_ntp_loss_and_trie_generation():
    model = _model()
    input_ids = torch.tensor(
        [[1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]]
    )
    labels = input_ids.clone()
    labels[:, :4] = IGNORE_INDEX
    mask = torch.ones_like(input_ids)
    loss, logits = model(input_ids, mask, labels)
    assert loss.isfinite()
    assert logits.shape == (1, 12, 17)
    loss.backward()

    candidates = [(1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15)]
    trie = Trie([list(code) for code in candidates])
    predictions, scores = model.eval().generate(
        input_ids[:, :4],
        mask[:, :4],
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn(trie),
        num_beams=3,
    )
    assert predictions.shape == (1, 3, 4)
    assert scores.shape == (1, 3)
    assert all(tuple(code) in candidates for code in predictions[0].tolist())
