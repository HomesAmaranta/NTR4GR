import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.rqvae import RQVAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_emb_path", type=str, default="../data/Beauty/item_emb.parquet")
    parser.add_argument("--rqvae_ckpt", type=str, default="./ckpt/Beauty/Jun-17-2025_15-21-52/best_collision_model.pth")
    parser.add_argument("--output_dir", type=str, default="../data/Beauty")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--target_code_len", type=int, default=4)
    return parser.parse_args()


def save_table(path, item_ids, embeddings):
    df = pd.DataFrame(
        {
            "ItemID": item_ids.astype(np.int64),
            "embedding": [row.astype(np.float32).tolist() for row in embeddings],
        }
    )
    df.to_parquet(path, index=False)
    print(f"saved {path}, shape={embeddings.shape}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(args.item_emb_path).sort_values("ItemID").reset_index(drop=True)
    item_ids = df["ItemID"].astype(int).to_numpy()
    raw_emb = np.stack(df["embedding"].to_numpy()).astype(np.float32)

    ckpt = torch.load(args.rqvae_ckpt, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]
    model = RQVAE(
        in_dim=raw_emb.shape[-1],
        num_emb_list=ckpt_args.num_emb_list,
        e_dim=ckpt_args.e_dim,
        layers=ckpt_args.layers,
        dropout_prob=ckpt_args.dropout_prob,
        bn=ckpt_args.bn,
        loss_type=ckpt_args.loss_type,
        quant_loss_weight=ckpt_args.quant_loss_weight,
        kmeans_init=ckpt_args.kmeans_init,
        kmeans_iters=ckpt_args.kmeans_iters,
        sk_epsilons=ckpt_args.sk_epsilons,
        sk_iters=ckpt_args.sk_iters,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(raw_emb)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    encoder_latents = []
    quantized_latents = []
    codebook_latents = []
    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="encoding"):
            batch = batch.to(device)
            x_e = model.encoder(batch)
            x_q, _, indices = model.rq(x_e, use_sk=False)
            codebooks = model.rq.get_codebook().to(device)
            layer_ids = torch.arange(indices.size(1), device=device).view(1, -1).expand_as(indices)
            selected_codebook = codebooks[layer_ids, indices]
            if selected_codebook.size(1) < args.target_code_len:
                pad = torch.zeros(
                    selected_codebook.size(0),
                    args.target_code_len - selected_codebook.size(1),
                    selected_codebook.size(2),
                    dtype=selected_codebook.dtype,
                    device=selected_codebook.device,
                )
                selected_codebook = torch.cat([selected_codebook, pad], dim=1)
            elif selected_codebook.size(1) > args.target_code_len:
                selected_codebook = selected_codebook[:, : args.target_code_len, :]
            encoder_latents.append(x_e.cpu().numpy())
            quantized_latents.append(x_q.cpu().numpy())
            codebook_latents.append(selected_codebook.cpu().numpy())

    encoder_latents = np.concatenate(encoder_latents, axis=0)
    quantized_latents = np.concatenate(quantized_latents, axis=0)
    codebook_latents = np.concatenate(codebook_latents, axis=0)

    save_table(
        os.path.join(args.output_dir, "item_emb_rqvae_encoder_latent.parquet"),
        item_ids,
        encoder_latents,
    )
    save_table(
        os.path.join(args.output_dir, "item_emb_rqvae_quantized_latent.parquet"),
        item_ids,
        quantized_latents,
    )
    save_table(
        os.path.join(args.output_dir, "item_emb_rqvae_codebook.parquet"),
        item_ids,
        codebook_latents,
    )


if __name__ == "__main__":
    main()
