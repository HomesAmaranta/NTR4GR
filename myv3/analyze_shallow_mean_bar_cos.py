import argparse
import ast
from pathlib import Path

import torch
import torch.nn.functional as F

from causal_tiger import CausalTIGER
from dataloader import GenRecDataLoader
from dataset import GenRecDataset


def parse_config(log_path: Path):
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Configuration:" not in line:
            continue
        raw = line.split("Configuration:", 1)[1].strip()
        return ast.literal_eval(raw)
    raise ValueError(f"Configuration not found in {log_path}")


def item_attention_mask(attention_mask: torch.Tensor, code_per_item: int):
    batch_size, seq_len = attention_mask.shape
    if seq_len % code_per_item != 0:
        raise ValueError(f"attention length {seq_len} is not divisible by {code_per_item}")
    grouped = attention_mask.view(batch_size, seq_len // code_per_item, code_per_item)
    return grouped[:, :, 0].bool()


def shallow_targets(model, input_ids, attention_mask, layer, align_item):
    old_layer = model.shallow_layer
    old_align_item = model.align_item
    model.shallow_layer = layer
    model.align_item = align_item
    try:
        return model._shallow_align_targets(input_ids, attention_mask)
    finally:
        model.shallow_layer = old_layer
        model.align_item = old_align_item


def mean_cos(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    pred = pred[valid]
    target = target[valid].to(pred.dtype)
    if pred.numel() == 0:
        return None
    return F.cosine_similarity(pred, target, dim=-1).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path",
        type=str,
        default=(
            "./logs/0908_v3_base/"
            "causal_tiger_Beauty_parallel_b80_s60_bs128_lr5e-4_headmlp_seed42.log"
        ),
    )
    parser.add_argument("--num_sequences", type=int, default=100)
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    config = parse_config(log_path)
    device = torch.device(args.device)

    dataset = GenRecDataset(
        dataset_path=f"{config['dataset_path']}/{args.split}.parquet",
        code_path=config["code_path"],
        mode="train_parallel",
        max_len=config["max_len"],
        PAD_TOKEN=config["pad_token_id"],
        item_emb_path=None,
        align_item="next",
        block_items=config["block_items"],
        stride_items=config["stride_items"],
    )
    loader = GenRecDataLoader(
        dataset,
        batch_size=min(config["batch_size"], args.num_sequences),
        shuffle=False,
        num_workers=0,
    )

    model = CausalTIGER(config).to(device)
    state = torch.load(config["save_path"], map_location=device)
    model.load_state_dict(state)
    model.eval()

    sums = {(layer, item): 0.0 for layer in (1, 2, 3) for item in ("current", "next")}
    counts = {(layer, item): 0 for layer in (1, 2, 3) for item in ("current", "next")}
    seen = 0
    code_per_item = model.code_per_item

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            take = min(input_ids.size(0), args.num_sequences - seen)
            input_ids = input_ids[:take]
            attention_mask = attention_mask[:take]
            seen += take

            hidden_states = model._encode_tokens(input_ids, attention_mask)
            pred = model.hidden_to_item_emb(hidden_states)
            valid = item_attention_mask(attention_mask, code_per_item)

            for layer in (1, 2, 3):
                for align_item in ("current", "next"):
                    target = shallow_targets(model, input_ids, attention_mask, layer, align_item)
                    value = mean_cos(pred, target, valid)
                    if value is not None:
                        n_valid = int(valid.sum().item())
                        sums[(layer, align_item)] += value * n_valid
                        counts[(layer, align_item)] += n_valid

            if seen >= args.num_sequences:
                break

    print(f"log_path: {log_path}")
    print(f"ckpt: {config['save_path']}")
    print(f"split: {args.split}")
    print(f"num_sequences: {seen}")
    print("| shallow_layer | align_item | mean_bar_cos | valid_items |")
    print("| --- | --- | --- | --- |")
    for layer in (1, 2, 3):
        for align_item in ("current", "next"):
            count = counts[(layer, align_item)]
            value = sums[(layer, align_item)] / count if count else float("nan")
            print(f"| {layer} | {align_item} | {value:.6f} | {count} |")


if __name__ == "__main__":
    main()
