import argparse
import logging
import math
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SourceToNextSidDataset
from model import SourceToNextSidModel


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_topk(value):
    if isinstance(value, list):
        return value
    return [int(x) for x in value.split(",") if x.strip()]


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).float()


def ndcg_at_k(pos_index, k):
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=pos_index.device)
    gains = torch.where(pos_index, 1.0 / torch.log2(ranks + 1), 0.0)
    return gains[:, :k].sum(dim=1).float()


def calculate_pos_index(preds, labels, maxk):
    pos_index = torch.zeros((preds.size(0), maxk), dtype=torch.bool, device=preds.device)
    for rank in range(maxk):
        pos_index[:, rank] = (preds[:, rank, :] == labels).all(dim=1)
    return pos_index


def update_sid_stats(stats, preds, labels, ks):
    for k in ks:
        cur_preds = preds[:, :k, :]
        batch_size = labels.size(0)
        for phase in range(4):
            layer_hit = (cur_preds[:, :, phase] == labels[:, None, phase]).any(dim=1)
            stats["layer_num"][k][phase] += layer_hit.sum().item()
            stats["layer_den"][k][phase] += batch_size

            prefix_hit = (
                cur_preds[:, :, : phase + 1] == labels[:, None, : phase + 1]
            ).all(dim=2).any(dim=1)
            if phase == 0:
                cond_den = torch.ones(batch_size, dtype=torch.bool, device=labels.device)
            else:
                cond_den = (
                    cur_preds[:, :, :phase] == labels[:, None, :phase]
                ).all(dim=2).any(dim=1)
            stats["cond_num"][k][phase] += prefix_hit[cond_den].sum().item()
            stats["cond_den"][k][phase] += cond_den.sum().item()


def finalize_sid_stats(stats):
    metrics = {}
    for k in sorted(stats["layer_num"]):
        for phase, (num, den) in enumerate(
            zip(stats["layer_num"][k], stats["layer_den"][k]), start=1
        ):
            metrics[f"SID{phase}_HR@{k}"] = num / den if den else float("nan")
        for phase, (num, den) in enumerate(
            zip(stats["cond_num"][k], stats["cond_den"][k]), start=1
        ):
            metrics[f"SID{phase}_CondHR@{k}"] = num / den if den else float("nan")
    return metrics


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        source_emb = batch["source_emb"].to(device)
        target_code = batch["target_code"].to(device)
        optimizer.zero_grad()
        loss, _ = model(source_emb, target_code)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="valid-loss", leave=False):
        source_emb = batch["source_emb"].to(device)
        target_code = batch["target_code"].to(device)
        loss, _ = model(source_emb, target_code)
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, topk_list, beam_size, device):
    model.eval()
    recalls = {f"Recall@{k}": [] for k in topk_list}
    ndcgs = {f"NDCG@{k}": [] for k in topk_list}
    ce_losses = []
    sid_topks = [k for k in (5, 10) if k <= beam_size]
    sid_stats = {
        name: {k: [0.0, 0.0, 0.0, 0.0] for k in sid_topks}
        for name in ("layer_num", "layer_den", "cond_num", "cond_den")
    }

    for batch in tqdm(loader, desc="eval", leave=False):
        source_emb = batch["source_emb"].to(device)
        target_code = batch["target_code"].to(device)
        loss, _ = model(source_emb, target_code)
        ce_losses.append(loss.item())
        preds = model.generate(source_emb, beam_size=beam_size)
        pos_index = calculate_pos_index(preds, target_code, beam_size)
        update_sid_stats(sid_stats, preds, target_code, sid_topks)
        for k in topk_list:
            recalls[f"Recall@{k}"].append(recall_at_k(pos_index, k).mean().item())
            ndcgs[f"NDCG@{k}"].append(ndcg_at_k(pos_index, k).mean().item())

    avg_recalls = {k: float(np.mean(v)) for k, v in recalls.items()}
    avg_ndcgs = {k: float(np.mean(v)) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs, finalize_sid_stats(sid_stats), float(np.mean(ce_losses))


def build_dataset(config, split, mode):
    return SourceToNextSidDataset(
        dataset_path=os.path.join(config.dataset_path, f"{split}.parquet"),
        code_path=config.code_path,
        item_emb_path=config.item_emb_path,
        mode=mode,
        source_offset=config.source_offset,
        max_source_offset=config.max_source_offset,
        codebook_size=config.codebook_size,
    )


def log_and_print(message):
    print(message)
    logging.info(message)


def main():
    parser = argparse.ArgumentParser(description="Single-source embedding to next SID ablation")
    parser.add_argument("--dataset_path", type=str, default="../data/Beauty")
    parser.add_argument("--code_path", type=str, default="../data/Beauty/Beauty_t5_rqvae.npy")
    parser.add_argument(
        "--item_emb_path",
        type=str,
        default="../data/Beauty/item_emb_rqvae_quantized_latent.parquet",
    )
    parser.add_argument("--source_offset", type=int, default=0)
    parser.add_argument("--max_source_offset", type=int, default=2)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=1025)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--infer_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=30)
    parser.add_argument("--topk_list", type=str, default="5,10,20")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, choices=["train", "evaluation"], default="train")
    parser.add_argument("--save_path", type=str, default="./ckpt/abl.pth")
    parser.add_argument("--log_path", type=str, default="./logs/abl.log")
    config = parser.parse_args()
    config.topk_list = parse_topk(config.topk_list)

    os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    logging.basicConfig(
        filename=config.log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    set_seed(config.seed)
    log_and_print(f"Configuration: {vars(config)}")

    train_dataset = build_dataset(config, "train", "train")
    valid_dataset = build_dataset(config, "valid", "evaluation")
    test_dataset = build_dataset(config, "test", "evaluation")
    log_and_print(
        "Dataset sizes: "
        f"train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}"
    )
    if min(len(train_dataset), len(valid_dataset), len(test_dataset)) == 0:
        raise ValueError("At least one split is empty after max_source_offset filtering.")

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config.infer_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.infer_size, shuffle=False)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = SourceToNextSidModel(
        input_dim=train_dataset.embedding_dim,
        hidden_dim=config.hidden_dim,
        mlp_dim=config.mlp_dim,
        vocab_size=config.vocab_size,
        codebook_size=config.codebook_size,
        dropout=config.dropout,
    ).to(device)
    log_and_print(model.n_parameters.strip())

    if config.mode == "evaluation":
        model.load_state_dict(torch.load(config.save_path, map_location=device))
        recalls, ndcgs, sid_hr, ce_loss = evaluate(
            model, test_loader, config.topk_list, config.beam_size, device
        )
        log_and_print(
            f"Test CE Loss: {ce_loss:.4f}, Recalls: {recalls}, "
            f"NDCGs: {ndcgs}, SID_HR: {sid_hr}"
        )
        return

    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    best_valid = math.inf
    patience = 0

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_loss = validate_loss(model, valid_loader, device)
        log_and_print(
            f"Epoch {epoch}/{config.num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
        )
        if valid_loss < best_valid:
            best_valid = valid_loss
            patience = 0
            torch.save(model.state_dict(), config.save_path)
            log_and_print(f"Saved best model to {config.save_path}")
        else:
            patience += 1
            if patience >= config.early_stop:
                log_and_print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(config.save_path, map_location=device))
    recalls, ndcgs, sid_hr, ce_loss = evaluate(
        model, test_loader, config.topk_list, config.beam_size, device
    )
    log_and_print(
        f"Final Test CE Loss: {ce_loss:.4f}, Recalls: {recalls}, "
        f"NDCGs: {ndcgs}, SID_HR: {sid_hr}"
    )


if __name__ == "__main__":
    main()
