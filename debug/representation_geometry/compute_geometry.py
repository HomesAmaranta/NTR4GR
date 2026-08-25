import argparse
import importlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration


TIGER_ROOT = Path(__file__).resolve().parents[2]


def import_from_model_dir(model_dir: Path, architecture: str):
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    dataset_mod = importlib.import_module("dataset")
    dataloader_mod = importlib.import_module("dataloader")
    CausalTIGER = None
    if architecture == "causal":
        causal_tiger = importlib.import_module("causal_tiger")
        CausalTIGER = causal_tiger.CausalTIGER
    return CausalTIGER, dataset_mod.GenRecDataset, dataloader_mod.GenRecDataLoader


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
        "mse_loss_weight": args.mse_loss_weight,
        "mse_loss_mode": args.mse_loss_mode,
        "align_loss_type": args.align_loss_type,
        "nitp_layer": args.nitp_layer,
    }


def build_t5_model(args):
    t5config = T5Config(
        num_layers=args.num_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        d_kv=args.d_kv,
        dropout_rate=args.dropout_rate,
        vocab_size=args.vocab_size,
        pad_token_id=args.pad_token_id,
        eos_token_id=args.eos_token_id,
        decoder_start_token_id=args.pad_token_id,
        feed_forward_proj=args.feed_forward_proj,
    )
    return T5ForConditionalGeneration(t5config)


def strip_t5_wrapper_prefix(state_dict):
    return {
        key[len("model.") :] if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }


def collect_first_item_token_hidden(model, batch, device, code_len, architecture):
    input_ids = batch["history"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    if architecture == "t5":
        hidden_states = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
    else:
        hidden_states = model._encode_tokens(input_ids, attention_mask)

    seq_len = input_ids.size(1)
    first_token_positions = torch.arange(seq_len, device=device) % code_len == 0
    valid_first_token_mask = attention_mask.bool() & first_token_positions.view(1, -1)
    return hidden_states[valid_first_token_mask]


def effective_rank(representations, eps):
    centered = representations - representations.mean(dim=0, keepdim=True)
    if centered.size(0) <= 1:
        raise ValueError("Need at least two valid token representations.")
    covariance = centered.T @ centered / (centered.size(0) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance.float()).clamp_min(0)
    total = eigenvalues.sum()
    if total <= eps:
        return torch.tensor(0.0, device=representations.device), eigenvalues
    probs = eigenvalues / total
    entropy = -(probs * torch.log(probs + eps)).sum()
    return torch.exp(entropy), eigenvalues


def average_pairwise_cosine(representations, num_pairs, seed):
    if representations.size(0) <= 1:
        raise ValueError("Need at least two valid token representations.")
    generator = torch.Generator(device=representations.device)
    generator.manual_seed(seed)
    normalized = F.normalize(representations.float(), p=2, dim=-1)
    idx1 = torch.randint(0, normalized.size(0), (num_pairs,), generator=generator, device=representations.device)
    idx2 = torch.randint(0, normalized.size(0), (num_pairs,), generator=generator, device=representations.device)
    same = idx1 == idx2
    if same.any():
        idx2[same] = (idx2[same] + 1) % normalized.size(0)
    cosine = (normalized[idx1] * normalized[idx2]).sum(dim=-1)
    return cosine.mean(), cosine.std(unbiased=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=str(TIGER_ROOT / "gr-nitp"))
    parser.add_argument("--architecture", type=str, default="causal", choices=["causal", "t5"])
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(TIGER_ROOT / "gr-nitp" / "ckpt" / "causal_tiger_Beauty_nitpL5_mse0.1_token.pth"),
    )
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--num_sequences", type=int, default=256)
    parser.add_argument("--num_pairs", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--code_len", type=int, default=4)

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
    parser.add_argument("--item_emb_dim", type=int, default=0)
    parser.add_argument("--mse_loss_weight", type=float, default=0.0)
    parser.add_argument("--mse_loss_mode", type=str, default="token")
    parser.add_argument("--align_loss_type", type=str, default="mse")
    parser.add_argument("--nitp_layer", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = Path(args.model_dir).resolve()
    CausalTIGER, GenRecDataset, GenRecDataLoader = import_from_model_dir(model_dir, args.architecture)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Pass --device cpu to run on CPU.")
    device = torch.device(args.device)

    data_dir = TIGER_ROOT / "data" / args.dataset
    dataset_path = data_dir / f"{args.split}.parquet"
    code_path = data_dir / f"{args.dataset}_t5_rqvae.npy"
    mode = "evaluation" if args.split in {"valid", "test"} else "train"

    dataset = GenRecDataset(
        str(dataset_path),
        str(code_path),
        mode=mode,
        max_len=args.max_len,
    )
    loader = GenRecDataLoader(dataset, batch_size=args.num_sequences, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    model = build_t5_model(args) if args.architecture == "t5" else CausalTIGER(build_config(args))
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    if args.architecture == "t5":
        state_dict = strip_t5_wrapper_prefix(state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        representations = collect_first_item_token_hidden(
            model,
            batch,
            device,
            args.code_len,
            args.architecture,
        )
        eff_rank, eigenvalues = effective_rank(representations, eps=1e-12)
        avg_cosine, std_cosine = average_pairwise_cosine(
            representations,
            num_pairs=args.num_pairs,
            seed=args.seed,
        )

    print(f"model_dir: {model_dir}")
    print(f"architecture: {args.architecture}")
    print(f"checkpoint_path: {args.checkpoint_path}")
    print(f"dataset_path: {dataset_path}")
    print(f"split: {args.split}")
    print(f"num_sequences: {args.num_sequences}")
    print(f"valid_first_item_tokens: {representations.size(0)}")
    print(f"hidden_dim: {representations.size(1)}")
    print(f"code_len: {args.code_len}")
    print(f"device: {device}")
    print(f"missing_keys: {missing_keys}")
    print(f"unexpected_keys: {unexpected_keys}")
    print("")
    print(f"effective_rank: {eff_rank.item():.8f}")
    print(f"avg_cosine_similarity: {avg_cosine.item():.8f}")
    print(f"std_cosine_similarity: {std_cosine.item():.8f}")
    print(f"top_10_eigenvalues: {[round(x, 8) for x in eigenvalues.flip(0)[:10].detach().cpu().tolist()]}")


if __name__ == "__main__":
    main()
