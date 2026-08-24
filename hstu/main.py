import argparse
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_hstu import HSTURec
from sampled_softmax import SampledSoftmaxLoss


class HSTUSequenceDataset(Dataset):
    def __init__(self, data_path, max_len, item_num, mode):
        df = pd.read_parquet(data_path)
        if "user" in df.columns:
            df = df.groupby("user", sort=False).tail(1)
        self.max_len = max_len
        self.item_num = item_num
        self.mode = mode
        self.rows = []
        for row in df.itertuples(index=False):
            seq = [int(x) for x in row.history] + [int(row.target)]
            seq = seq[-max_len:]
            if len(seq) >= 2:
                self.rows.append(seq)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        seq = self.rows[idx]
        if self.mode == "train":
            return seq
        return seq[:-1], seq[-1]


class HSTUDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers=8):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        if isinstance(batch[0], tuple):
            max_len = max(len(x[0]) for x in batch)
            seqs, targets = [], []
            for seq, target in batch:
                seqs.append(torch.tensor([0] * (max_len - len(seq)) + seq, dtype=torch.long))
                targets.append(target)
            input_ids = torch.stack(seqs)
            return {
                "input_ids": input_ids,
                "attention_mask": (input_ids != 0).long(),
                "target": torch.tensor(targets, dtype=torch.long),
            }

        max_len = max(len(x) for x in batch)
        seqs = [torch.tensor([0] * (max_len - len(x)) + x, dtype=torch.long) for x in batch]
        ids = torch.stack(seqs)
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


def infer_item_num(dataset_path, data):
    item_num = 0
    for split in ["train", "valid", "test"]:
        df = pd.read_parquet(os.path.join(dataset_path, f"{split}{data}"))
        for row in df.itertuples(index=False):
            if len(row.history):
                item_num = max(item_num, max(int(x) for x in row.history))
            item_num = max(item_num, int(row.target))
    return item_num


def train_epoch(model, loss_fn, loader, optimizer, device):
    model.train()
    total = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        optimizer.zero_grad()
        hidden = model.encode(ids, mask)
        supervision_ids = ids[:, 1:]
        supervision_weights = (supervision_ids != 0).float()
        loss = loss_fn(hidden[:, :-1, :], supervision_ids, supervision_weights, model.item_emb)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def valid_loss(model, loss_fn, loader, device):
    model.eval()
    total = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        hidden = model.encode(ids, mask)
        supervision_ids = ids[:, 1:]
        supervision_weights = (supervision_ids != 0).float()
        loss = loss_fn(hidden[:, :-1, :], supervision_ids, supervision_weights, model.item_emb)
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, topk_list):
    model.eval()
    recalls = {f"Recall@{k}": 0.0 for k in topk_list}
    ndcgs = {f"NDCG@{k}": 0.0 for k in topk_list}
    total = 0
    max_k = max(topk_list)
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        target = batch["target"].to(device)
        scores = model.score_all(ids, mask)
        seen = ids > 0
        rows = torch.arange(ids.size(0), device=device).unsqueeze(1).expand_as(ids)
        seen_items = ids.clamp(min=1) - 1
        scores[rows[seen], seen_items[seen]] = -float("inf")
        topk = torch.topk(scores, k=max_k, dim=-1).indices + 1
        total += ids.size(0)
        for k in topk_list:
            hit = topk[:, :k] == target.unsqueeze(1)
            recalls[f"Recall@{k}"] += hit.any(dim=1).float().sum().item()
            weights = 1.0 / torch.log2(torch.arange(2, k + 2, device=device).float())
            ndcgs[f"NDCG@{k}"] += (hit.float() * weights).sum(dim=1).sum().item()
    return {k: v / total for k, v in recalls.items()}, {k: v / total for k, v in ndcgs.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--data", type=str, default=".parquet")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--infer_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=101)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dqk", type=int, default=50)
    parser.add_argument("--dv", type=int, default=50)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--num_negatives", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--item_l2_norm", action="store_true", default=True)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--topk_list", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument("--ckpt_path", type=str, default="None")
    parser.add_argument("--save_path", type=str, default="./ckpt/hstu.pth")
    parser.add_argument("--log_path", type=str, default="./logs/hstu.log")
    args = parser.parse_args()

    if args.dataset_path is None:
        args.dataset_path = f"../data/{args.dataset}"
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    logging.basicConfig(filename=args.log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Config: {vars(args)}")
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    item_num = infer_item_num(args.dataset_path, args.data)
    model = HSTURec(
        item_num=item_num,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dqk=args.dqk,
        dv=args.dv,
        dropout_rate=args.dropout_rate,
    ).to(device)
    loss_fn = SampledSoftmaxLoss(args.num_negatives, args.temperature, args.item_l2_norm).to(device)
    logging.info(model.n_parameters)
    print(model.n_parameters)
    if args.ckpt_path and args.ckpt_path != "None":
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    train_ds = HSTUSequenceDataset(os.path.join(args.dataset_path, f"train{args.data}"), args.max_len, item_num, "train")
    valid_ds = HSTUSequenceDataset(os.path.join(args.dataset_path, f"valid{args.data}"), args.max_len, item_num, "train")
    test_ds = HSTUSequenceDataset(os.path.join(args.dataset_path, f"test{args.data}"), args.max_len, item_num, "eval")
    train_loader = HSTUDataLoader(train_ds, args.batch_size, True)
    valid_loader = HSTUDataLoader(valid_ds, args.infer_size, False)
    test_loader = HSTUDataLoader(test_ds, args.infer_size, False)
    print(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}, Item: {item_num}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.type == "test":
        recalls, ndcgs = evaluate(model, test_loader, device, args.topk_list)
        print(f"Test Recalls: {recalls}")
        print(f"Test NDCGs: {ndcgs}")
    else:
        best_loss, best_epoch, patience = float("inf"), 0, 0
        for epoch in tqdm(range(args.num_epochs)):
            loss = train_epoch(model, loss_fn, train_loader, optimizer, device)
            logging.info(f"Epoch {epoch + 1}/{args.num_epochs}, Train Loss: {loss:.4f}")
            val = valid_loss(model, loss_fn, valid_loader, device)
            logging.info(f"Valid Loss: {val:.4f}")
            if val < best_loss:
                best_loss, best_epoch, patience = val, epoch, 0
                torch.save(model.state_dict(), args.save_path)
                logging.info(f"Best model saved: {best_loss:.4f}")
            else:
                patience += 1
                if args.early_stop >= 0 and patience >= args.early_stop:
                    logging.info("Early stopping triggered.")
                    break
        if args.early_stop < 0:
            torch.save(model.state_dict(), args.save_path)
            best_epoch = epoch
        else:
            model.load_state_dict(torch.load(args.save_path, map_location=device))
        recalls, ndcgs = evaluate(model, test_loader, device, args.topk_list)
        logging.info(f"Final Epoch: {best_epoch + 1}")
        logging.info(f"Final Test Recalls: {recalls}")
        logging.info(f"Final Test NDCGs: {ndcgs}")
        print(f"Final Epoch: {best_epoch + 1}")
        print(f"Final Test Recalls: {recalls}")
        print(f"Final Test NDCGs: {ndcgs}")
