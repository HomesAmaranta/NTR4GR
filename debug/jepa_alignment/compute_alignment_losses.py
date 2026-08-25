import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


TIGER_ROOT = Path(__file__).resolve().parents[2]
GR_JEPA_DIR = TIGER_ROOT / "gr-jepa"
if str(GR_JEPA_DIR) not in sys.path:
    sys.path.insert(0, str(GR_JEPA_DIR))

from causal_tiger import CausalTIGER  # noqa: E402
from dataloader import GenRecDataLoader  # noqa: E402
from dataset import GenRecDataset  # noqa: E402


def build_config(args):
    return {
        "num_layers": args.num_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_heads": args.num_heads,
        "d_kv": args.d_kv,
        "dropout_rate": args.dropout_rate,
        "vocab_size": args.vocab_size,
        "pad_token_id": args.pad_token_id,
        "eos_token_id": args.eos_token_id,
        "decoder_start_token_id": args.pad_token_id,
        "feed_forward_proj": args.feed_forward_proj,
        "item_emb_dim": args.item_emb_dim,
        "mse_loss_weight": 1.0,
        "mse_loss_mode": args.mse_loss_mode,
        "align_loss_type": "mse",
    }


def target_hidden_states(model, input_ids, attention_mask, labels):
    batch_size = input_ids.size(0)
    start_tokens = torch.full(
        (batch_size, 1),
        model.decoder_start_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    decoder_input_ids = torch.cat([start_tokens, labels[:, :-1]], dim=1)
    model_input_ids = torch.cat([input_ids, decoder_input_ids], dim=1)

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    decoder_attention_mask = torch.ones_like(decoder_input_ids)
    model_attention_mask = torch.cat([attention_mask, decoder_attention_mask], dim=1)

    hidden_states = model._encode_tokens(model_input_ids, model_attention_mask)
    return hidden_states[:, -labels.size(1) :, :]


def compute_losses(model, hidden_states, target_item_emb, mse_loss_mode):
    if mse_loss_mode == "mean":
        pred_item_emb = model.hidden_to_item_emb(hidden_states.mean(dim=1))
        target_emb = target_item_emb[:, 0, :].to(pred_item_emb.dtype)
    else:
        pred_item_emb = model.hidden_to_item_emb(hidden_states)
        target_emb = target_item_emb.to(pred_item_emb.dtype)

    mse = F.mse_loss(pred_item_emb, target_emb, reduction="none").mean(dim=-1)
    cos_sim = F.cosine_similarity(pred_item_emb, target_emb, dim=-1)
    cos_loss = 1.0 - cos_sim

    if mse_loss_mode == "token":
        mse_per_sample = mse.mean(dim=1)
        cos_loss_per_sample = cos_loss.mean(dim=1)
        cos_sim_per_sample = cos_sim.mean(dim=1)
    else:
        mse_per_sample = mse
        cos_loss_per_sample = cos_loss
        cos_sim_per_sample = cos_sim

    return mse_per_sample, cos_loss_per_sample, cos_sim_per_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(TIGER_ROOT / "gr" / "ckpt" / "causal_tiger_Beauty.pth"),
    )
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mse_loss_mode", type=str, default="mean", choices=["token", "mean"])

    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int, default=1025)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--eos_token_id", type=int, default=0)
    parser.add_argument("--feed_forward_proj", type=str, default="relu")
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--item_emb_dim", type=int, default=768)
    args = parser.parse_args()

    data_dir = TIGER_ROOT / "data" / args.dataset
    dataset_path = data_dir / f"{args.split}.parquet"
    code_path = data_dir / f"{args.dataset}_t5_rqvae.npy"
    item_emb_path = data_dir / "item_emb.parquet"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = build_config(args)
    model = CausalTIGER(config)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    dataset = GenRecDataset(
        str(dataset_path),
        str(code_path),
        mode="evaluation" if args.split in {"valid", "test"} else "train",
        max_len=args.max_len,
        item_emb_path=str(item_emb_path),
    )
    loader = GenRecDataLoader(dataset, batch_size=args.num_samples, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    input_ids = batch["history"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["target"].to(device)
    target_item_emb = batch["target_item_emb"].to(device)

    with torch.no_grad():
        hidden_states = target_hidden_states(model, input_ids, attention_mask, labels)
        mse, cos_loss, cos_sim = compute_losses(
            model,
            hidden_states,
            target_item_emb,
            args.mse_loss_mode,
        )

    print(f"checkpoint_path: {args.checkpoint_path}")
    print(f"dataset_path: {dataset_path}")
    print(f"num_samples: {mse.numel()}")
    print(f"mse_loss_mode: {args.mse_loss_mode}")
    print(f"device: {device}")
    print(f"missing_keys: {missing_keys}")
    print(f"unexpected_keys: {unexpected_keys}")
    if "hidden_to_item_emb.weight" in missing_keys:
        print("warning: checkpoint has no hidden_to_item_emb.weight; projection layer is randomly initialized.")
    print("")
    for idx, (mse_i, cos_loss_i, cos_sim_i) in enumerate(
        zip(mse.detach().cpu(), cos_loss.detach().cpu(), cos_sim.detach().cpu())
    ):
        print(
            f"sample {idx}: "
            f"mse={mse_i.item():.8f}, "
            f"cos_loss={cos_loss_i.item():.8f}, "
            f"cos_sim={cos_sim_i.item():.8f}"
        )
    print("")
    print(f"avg_mse: {mse.mean().item():.8f}")
    print(f"avg_cos_loss: {cos_loss.mean().item():.8f}")
    print(f"avg_cos_sim: {cos_sim.mean().item():.8f}")


if __name__ == "__main__":
    main()
