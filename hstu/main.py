import argparse
import logging
import os
import random

import numpy as np
import torch
from tqdm import tqdm

try:
    from .dataloader import GenRecDataLoader
    from .dataset import GenRecDataset, IGNORE_INDEX
    from .generation_trie import Trie, prefix_allowed_tokens_fn
    from .model_hstu import HSTURec
except ImportError:
    from dataloader import GenRecDataLoader
    from dataset import GenRecDataset, IGNORE_INDEX
    from generation_trie import Trie, prefix_allowed_tokens_fn
    from model_hstu import HSTURec


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_loss_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if training:
                optimizer.zero_grad()
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if training:
                loss.backward()
                optimizer.step()

            token_count = labels[:, 1:].ne(IGNORE_INDEX).sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("No supervised SID tokens were found")
    return total_loss / total_tokens


@torch.no_grad()
def evaluate(
    model,
    loader,
    prefix_constraint,
    device,
    topk_list,
    beam_size,
):
    model.eval()
    if beam_size < max(topk_list):
        raise ValueError("beam_size must be at least max(topk_list)")

    hits = {k: 0.0 for k in topk_list}
    ndcgs = {k: 0.0 for k in topk_list}
    total = 0
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device)
        predictions, _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_allowed_tokens_fn=prefix_constraint,
            num_beams=beam_size,
        )
        matches = predictions.eq(targets[:, None, :]).all(dim=-1)
        total += targets.size(0)
        for k in topk_list:
            topk_matches = matches[:, :k]
            hits[k] += topk_matches.any(dim=1).float().sum().item()
            ranks = torch.arange(
                1, topk_matches.size(1) + 1, device=device
            )
            discounts = 1.0 / torch.log2(ranks.float() + 1.0)
            ndcgs[k] += (
                topk_matches.float() * discounts.unsqueeze(0)
            ).sum().item()

    if total == 0:
        raise ValueError("Test dataset is empty")
    recalls = {f"Recall@{k}": hits[k] / total for k in topk_list}
    ndcg_results = {f"NDCG@{k}": ndcgs[k] / total for k in topk_list}
    return recalls, ndcg_results


def build_dataset(args, split, mode):
    return GenRecDataset(
        dataset_path=os.path.join(args.dataset_path, f"{split}{args.data}"),
        code_path=args.code_path,
        mode=mode,
        max_len=args.max_len,
        pad_token=args.pad_token_id,
        codebook_size=args.codebook_size,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--data", type=str, default=".parquet")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--code_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--infer_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=101)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max_len",
        type=int,
        default=20,
        help=(
            "Maximum item-level attention history; validation and test "
            "histories are also capped to this length"
        ),
    )
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dqk", type=int, default=32)
    parser.add_argument("--dv", type=int, default=32)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument(
        "--topk_list", type=int, nargs="+", default=[5, 10, 20]
    )
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--type", type=str, choices=["train", "test"], default="train"
    )
    parser.add_argument("--ckpt_path", type=str, default="None")
    parser.add_argument("--save_path", type=str, default="./ckpt/hstu.pth")
    parser.add_argument("--log_path", type=str, default="./logs/hstu.log")
    args = parser.parse_args()

    if args.dataset_path is None:
        args.dataset_path = f"../data/{args.dataset}"
    if args.code_path is None:
        args.code_path = os.path.join(
            args.dataset_path, f"{args.dataset}_t5_rqvae.npy"
        )
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    logging.basicConfig(
        filename=args.log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Configuration: {vars(args)}")
    set_seed(args.seed)

    train_ds = build_dataset(args, "train", "train")
    valid_ds = build_dataset(args, "valid", "valid")
    test_ds = build_dataset(args, "test", "test")
    if not train_ds.item_to_code:
        raise ValueError("SID codebook is empty")
    if train_ds.code_length != 4:
        raise ValueError(
            f"Expected four SID levels, got {train_ds.code_length}"
        )

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    max_seq_len = max(
        len(sample["input_ids"])
        for dataset in (train_ds, valid_ds, test_ds)
        for sample in dataset.data
    )
    model = HSTURec(
        vocab_size=train_ds.vocab_size,
        max_seq_len=max_seq_len,
        code_length=train_ds.code_length,
        max_history_items=args.max_len,
        embedding_dim=args.embedding_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dqk=args.dqk,
        dv=args.dv,
        dropout_rate=args.dropout_rate,
        pad_token_id=args.pad_token_id,
    ).to(device)
    logging.info(model.n_parameters)
    print(model.n_parameters)

    checkpoint_path = (
        args.save_path
        if args.ckpt_path in {"", "None", None}
        else args.ckpt_path
    )
    if args.type == "test":
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif args.ckpt_path not in {"", "None", None}:
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))

    train_loader = GenRecDataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = GenRecDataLoader(
        valid_ds,
        batch_size=args.infer_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = GenRecDataLoader(
        test_ds,
        batch_size=args.infer_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(
        f"Train users: {len(train_ds)}, Valid users: {len(valid_ds)}, "
        f"Test users: {len(test_ds)}, SID vocab: {train_ds.vocab_size}"
    )

    item_trie = Trie([list(code) for code in test_ds.candidate_codes()])
    prefix_constraint = prefix_allowed_tokens_fn(item_trie)
    if args.type == "train":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        best_loss, best_epoch, patience = float("inf"), 0, 0
        for epoch in tqdm(range(args.num_epochs)):
            train_loss = run_loss_epoch(
                model, train_loader, device, optimizer=optimizer
            )
            valid_loss = run_loss_epoch(model, valid_loader, device)
            logging.info(
                f"Epoch {epoch + 1}/{args.num_epochs}, "
                f"Train NTP Loss: {train_loss:.6f}, "
                f"Valid NTP Loss: {valid_loss:.6f}"
            )
            if valid_loss < best_loss:
                best_loss, best_epoch, patience = valid_loss, epoch, 0
                torch.save(model.state_dict(), args.save_path)
                logging.info(f"Best model saved: {best_loss:.6f}")
            else:
                patience += 1
                if args.early_stop >= 0 and patience >= args.early_stop:
                    logging.info("Early stopping triggered.")
                    break

        if args.early_stop < 0:
            torch.save(model.state_dict(), args.save_path)
            best_epoch = epoch
        else:
            model.load_state_dict(
                torch.load(args.save_path, map_location=device)
            )
    else:
        best_epoch = -1

    recalls, ndcgs = evaluate(
        model,
        test_loader,
        prefix_constraint,
        device,
        args.topk_list,
        args.beam_size,
    )
    if best_epoch >= 0:
        logging.info(f"Final Epoch: {best_epoch + 1}")
        print(f"Final Epoch: {best_epoch + 1}")
    logging.info(f"Final Test Recalls: {recalls}")
    logging.info(f"Final Test NDCGs: {ndcgs}")
    print(f"Final Test Recalls: {recalls}")
    print(f"Final Test NDCGs: {ndcgs}")


if __name__ == "__main__":
    main()
