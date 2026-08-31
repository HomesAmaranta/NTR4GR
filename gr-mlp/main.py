import torch
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
import os
import random
import pandas as pd
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from dataset import GenRecDataset
from dataloader import GenRecDataLoader
from generation_trie import Trie, prefix_allowed_tokens_fn
from causal_tiger import CausalTIGER


class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
            num_layers=config["num_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            num_heads=config["num_heads"],
            d_kv=config["d_kv"],
            dropout_rate=config["dropout_rate"],
            vocab_size=config["vocab_size"],
            pad_token_id=config["pad_token_id"],
            eos_token_id=config["eos_token_id"],
            decoder_start_token_id=config["pad_token_id"],
            feed_forward_proj=config["feed_forward_proj"],
        )
        # Initialize T5 model with the specified configuration
        self.model = T5ForConditionalGeneration(t5config)

    @property
    def n_parameters(self):
        """Calculates the number of trainable parameters in the model.

        Returns:
            str: A string containing the number of embedding parameters,
            non-embedding parameters, and total trainable parameters.
        """
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.model.get_input_embeddings().parameters())
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Forward pass of the model. Returns the output logits and the loss value.

        Args:
            batch (dict): A dictionary containing the input data for the model.

        Returns:
            outputs (ModelOutput):
                The output of the model, which includes:
                - loss (torch.Tensor)
                - logits (torch.Tensor)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        self.last_target_hidden_states = outputs.decoder_hidden_states[-1].detach()
        return outputs.loss, outputs.logits

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 20,
        **kwargs,
    ):
        """Generate recommendations using the model.

        Args:
            input_ids (torch.Tensor): Input tensor for the model.
            attention_mask (Optional[torch.Tensor]): Attention mask for the input.
            max_length (int): Maximum length of the generated sequence.
            num_beams (int): Number of beams for beam search.

        Returns:
            torch.Tensor: Generated output tensor.
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs,
        )


def calculate_pos_index(preds, labels, maxk=20):
    """Calculate the position index of the ground truth items.

    Args:
        preds: The predicted token sequences, of shape
        (batch_size, maxk, seq_len).
        labels: The ground truth token sequences, of shape (batch_size, seq_len).

    Returns:
        A boolean tensor of shape (batch_size, maxk) indicating whether the
        prediction at each position is correct.
    """
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert preds.shape[1] == maxk, f"preds.shape[1] = {preds.shape[1]} != {maxk}"

    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True
                break
    return pos_index


def recall_at_k(pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()


def ndcg_at_k(pos_index, k):
    # Assume only one ground truth item per example
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(
        pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device)
    )
    return dcg[:, :k].sum(dim=1).cpu().float()


def compute_single_geometry(reps: torch.Tensor) -> Dict[str, float]:
    reps = reps.detach().float()
    if reps.size(0) < 2:
        return {"effective_rank": 0.0, "avg_cosine": 0.0}

    centered = reps - reps.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(centered.size(0) - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    probs = eigvals / eigvals.sum()
    effective_rank = torch.exp(-(probs * probs.log()).sum()).item()

    normed = F.normalize(reps, dim=-1)
    sim = normed @ normed.T
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    avg_cosine = sim[mask].mean().item()
    return {"effective_rank": effective_rank, "avg_cosine": avg_cosine}


def compute_geometry_metrics(hidden_states: torch.Tensor) -> Dict[str, float]:
    # Track the supervised positions, including h_PAD for the first code token.
    token_count = min(4, hidden_states.size(1))
    metrics = {}
    ranks = []
    cosines = []
    for token_idx in range(token_count):
        cur = compute_single_geometry(hidden_states[:, token_idx, :])
        metrics[f"effective_rank_token{token_idx + 1}"] = cur["effective_rank"]
        metrics[f"avg_cosine_token{token_idx + 1}"] = cur["avg_cosine"]
        ranks.append(cur["effective_rank"])
        cosines.append(cur["avg_cosine"])
    metrics["effective_rank"] = sum(ranks) / len(ranks) if ranks else 0.0
    metrics["avg_cosine"] = sum(cosines) / len(cosines) if cosines else 0.0
    return metrics


def train(model, train_loader, optimizer, device):
    model.train()
    totals = {
        "total": 0.0,
        "ce": 0.0,
        "align": 0.0,
        "effective_rank": 0.0,
        "avg_cosine": 0.0,
        "effective_rank_token1": 0.0,
        "effective_rank_token2": 0.0,
        "effective_rank_token3": 0.0,
        "effective_rank_token4": 0.0,
        "avg_cosine_token1": 0.0,
        "avg_cosine_token2": 0.0,
        "avg_cosine_token3": 0.0,
        "avg_cosine_token4": 0.0,
    }
    for batch in train_loader:
        input_ids = batch["history"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["target"].to(device)
        target_item_emb = batch.get("target_item_emb")
        target_item_emb = (
            target_item_emb.to(device) if target_item_emb is not None else None
        )

        optimizer.zero_grad()
        if model.__class__.__name__ == "CausalTIGER":
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                target_item_emb=target_item_emb,
            )
        else:
            loss, _ = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
        loss.backward()
        optimizer.step()

        loss_dict = getattr(model, "last_loss_dict", None)
        totals["total"] += loss.item()
        totals["ce"] += (
            loss_dict["ce"].item()
            if loss_dict is not None and "ce" in loss_dict
            else loss.item()
        )
        totals["align"] += (
            loss_dict["align"].item()
            if loss_dict is not None and "align" in loss_dict
            else 0.0
        )
        hidden_states = getattr(model, "last_target_hidden_states", None)
        if hidden_states is not None:
            geometry = compute_geometry_metrics(hidden_states)
            for key, value in geometry.items():
                totals[key] += value
    return {k: v / len(train_loader) for k, v in totals.items()}


def validate(model, valid_loader, device):
    model.eval()
    losses = {"total": 0.0, "ce": 0.0, "align": 0.0}
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            target_item_emb = batch.get("target_item_emb")
            target_item_emb = (
                target_item_emb.to(device) if target_item_emb is not None else None
            )
            if model.__class__.__name__ == "CausalTIGER":
                loss, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    target_item_emb=target_item_emb,
                )
            else:
                loss, _ = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
            loss_dict = getattr(model, "last_loss_dict", None)
            losses["total"] += loss.item()
            losses["ce"] += (
                loss_dict["ce"].item()
                if loss_dict is not None and "ce" in loss_dict
                else loss.item()
            )
            losses["align"] += (
                loss_dict["align"].item()
                if loss_dict is not None and "align" in loss_dict
                else 0.0
            )
    return {k: v / len(valid_loader) for k, v in losses.items()}


def evaluate(model, eval_loader, topk_list, beam_size, device, trie=None):
    model.eval()
    recalls = {"Recall@" + str(k): [] for k in topk_list}
    ndcgs = {"NDCG@" + str(k): [] for k in topk_list}
    constraint_fn = prefix_allowed_tokens_fn(trie) if trie is not None else None
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["history"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)

            is_causal = model.__class__.__name__ == "CausalTIGER"
            generate_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "num_beams": beam_size,
                "prefix_allowed_tokens_fn": constraint_fn,
            }
            if is_causal:
                generate_kwargs["max_length"] = labels.size(1)

            preds = model.generate(**generate_kwargs)
            if not is_causal:
                preds = preds[:, 1:]  # Exclude T5 decoder start token.
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)
            assert preds.size(2) == labels.size(1), (
                f"Generated sequence length {preds.size(2)} does not match "
                f"label length {labels.size(1)} for "
                f'{"CausalTIGER" if is_causal else "T5"} decoding.'
            )
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            # print(f"pos_index shape: {pos_index.shape}, pos_index: {pos_index}")
            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls["Recall@" + str(k)].append(recall)
                ndcgs["NDCG@" + str(k)].append(ndcg)
                # Calculate average recalls and ndcgs
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs


def set_seed(seed):
    """Set random seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIGER configuration")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--infer_size",
        type=int,
        default=96,
        help="Inference size for generating recommendations",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=200, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to run the model on (e.g., "cuda" or "cpu")',
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of layers in the model"
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=4,
        help="Number of decoder layers in the model",
    )
    parser.add_argument(
        "--d_model", type=int, default=128, help="Dimension of the model"
    )
    parser.add_argument(
        "--d_ff", type=int, default=1024, help="Dimension of the feed-forward layer"
    )
    parser.add_argument(
        "--num_heads", type=int, default=6, help="Number of attention heads"
    )
    parser.add_argument(
        "--d_kv", type=int, default=64, help="Dimension of key and value vectors"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--vocab_size", type=int, default=1025, help="Vocabulary size")
    parser.add_argument("--pad_token_id", type=int, default=0, help="Padding token ID")
    parser.add_argument(
        "--eos_token_id", type=int, default=0, help="End of sequence token ID"
    )
    parser.add_argument(
        "--feed_forward_proj",
        type=str,
        default="relu",
        help="Feed forward projection type",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=20,
        help="Maximum length for padding or truncation",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="../data/Beauty", help="Path to the dataset"
    )
    parser.add_argument(
        "--code_path",
        type=str,
        default="../data/Beauty/Beauty_t5_rqvae.npy",
        help="Path to the item-to-code mapping file",
    )
    parser.add_argument(
        "--item_emb_path",
        type=str,
        default=None,
        help="Path to item embedding parquet for auxiliary MSE loss",
    )
    parser.add_argument(
        "--item_emb_dim",
        type=int,
        default=0,
        help="Item embedding dimension for auxiliary MSE loss",
    )
    parser.add_argument(
        "--mse_loss_weight",
        type=float,
        default=0.0,
        help="Weight of auxiliary item embedding MSE loss",
    )
    parser.add_argument(
        "--mse_loss_mode",
        type=str,
        default="token",
        choices=["token", "mean"],
        help="Auxiliary MSE mode: per-token or mean-pooled target hidden states",
    )
    parser.add_argument(
        "--align_loss_type",
        type=str,
        default="mse",
        choices=["mse", "cos"],
        help="Auxiliary alignment loss type",
    )
    parser.add_argument(
        "--align_target",
        type=str,
        default="item",
        choices=["item", "latent", "quantized", "codebook", "shallow"],
        help="Auxiliary alignment target source",
    )
    parser.add_argument(
        "--align_item",
        type=str,
        default="next",
        choices=["pre", "next"],
        help="Item embedding to align: previous history item or next target item",
    )
    parser.add_argument(
        "--shallow_layer",
        type=int,
        default=1,
        help="Shallow layer index used when align_target=shallow",
    )
    parser.add_argument(
        "--lm_head",
        type=str,
        default="emb",
        choices=["emb", "linear"],
        help="LM head type: tied embedding or independent linear layer",
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="total",
        choices=["total", "ce"],
        help="Validation loss used for early stopping",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "evaluation"],
        help="Mode of operation",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="t5",
        choices=["t5", "causal", "gpt2"],
        help="Model type to train",
    )
    parser.add_argument(
        "--log_path", type=str, default="./logs/tiger.log", help="Path to the log file"
    )
    parser.add_argument(
        "--tensorboard_dir", type=str, default=None, help="TensorBoard log directory"
    )
    parser.add_argument(
        "--seed", type=int, default=2025, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./ckpt/tiger.pth",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--early_stop", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--topk_list",
        type=list,
        default=[5, 10, 20],
        help="List of top-k values for evaluation metrics",
    )
    parser.add_argument(
        "--beam_size", type=int, default=30, help="Beam size for generation"
    )
    config = vars(parser.parse_args())
    if config["item_emb_path"] in {"", "None"}:
        config["item_emb_path"] = None
        # Set up logging
    logging.basicConfig(
        filename=config["log_path"],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Configuration: {config}")
    tb_dir = config["tensorboard_dir"]
    if tb_dir is None or tb_dir == "None":
        tb_dir = os.path.join(
            "./runs", os.path.splitext(os.path.basename(config["log_path"]))[0]
        )
    writer = SummaryWriter(tb_dir)
    logging.info(f"TensorBoard dir: {tb_dir}")
    set_seed(config["seed"])
    # Initialize model
    if config["model_type"] == "causal":
        model = CausalTIGER(config)
    else:
        model = TIGER(config)
    print(model.n_parameters)
    logging.info(model.n_parameters)
    # Check if the device is available
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    train_dataset = GenRecDataset(
        dataset_path=config["dataset_path"] + "/train.parquet",
        code_path=config["code_path"],
        mode="train",
        max_len=config["max_len"],
        item_emb_path=(
            config["item_emb_path"] if config["mse_loss_weight"] > 0 else None
        ),
        align_item=config["align_item"],
    )
    validation_dataset = GenRecDataset(
        dataset_path=config["dataset_path"] + "/valid.parquet",
        code_path=config["code_path"],
        mode="evaluation",
        max_len=config["max_len"],
        item_emb_path=(
            config["item_emb_path"] if config["mse_loss_weight"] > 0 else None
        ),
        align_item=config["align_item"],
    )
    test_dataset = GenRecDataset(
        dataset_path=config["dataset_path"] + "/test.parquet",
        code_path=config["code_path"],
        mode="evaluation",
        max_len=config["max_len"],
    )

    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(config["seed"])
    train_dataloader = GenRecDataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=dataloader_generator,
        worker_init_fn=seed_worker,
    )
    validation_dataloader = GenRecDataLoader(
        validation_dataset,
        batch_size=config["infer_size"],
        shuffle=False,
        worker_init_fn=seed_worker,
    )
    test_dataloader = GenRecDataLoader(
        test_dataset,
        batch_size=config["infer_size"],
        shuffle=False,
        worker_init_fn=seed_worker,
    )

    print("Building Trie...")
    trie_sequences = []
    for code in list(test_dataset.item_to_code.values()):
        code_list = list(code) if isinstance(code, (np.ndarray, list)) else list(code)
        if config["model_type"] == "causal":
            trie_sequences.append(code_list + [config["eos_token_id"]])
        else:
            trie_sequences.append(
                [config["pad_token_id"]] + code_list + [config["eos_token_id"]]
            )
    item_trie = Trie(trie_sequences)
    print("Trie built.")

    model.to(device)
    if config["mode"] == "evaluation":
        logging.info(f"Loading model from {config['save_path']} for testing...")
        model.load_state_dict(torch.load(config["save_path"], map_location=device))
        test_avg_recalls, test_avg_ndcgs = evaluate(
            model,
            test_dataloader,
            config["topk_list"],
            config["beam_size"],
            device,
            trie=item_trie,
        )
        logging.info(f"Test Recalls: {test_avg_recalls}")
        logging.info(f"Test NDCGs: {test_avg_ndcgs}")
        print(f"Test Recalls: {test_avg_recalls}")
        print(f"Test NDCGs: {test_avg_ndcgs}")
        raise SystemExit
        # print(f"Train dataset size: {len(train_dataset)}")
        # print(f"Validation dataset size: {len(validation_dataset)}")
        # print(f"Test dataset size: {len(test_dataset)}")
        # for batch in train_dataloader:
        #     print(f"Batch size: {len(batch['history'])}")
        #     print(f"the first batch history:{batch['history'][0]}")
        #     print(f"the first batch target:{batch['target'][0]}")
        #     print(f"the first batch attention mask:{batch['attention_mask'][0]}")
        #     break

        # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Train the model
    best_loss = 10000.0
    early_stop_counter = 0
    best_epoch = 0
    for epoch in tqdm(range(config["num_epochs"])):
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_losses = train(model, train_dataloader, optimizer, device)
        logging.info(f"Training total loss: {train_losses['total']}")
        logging.info(f"Training CE loss: {train_losses['ce']}")
        logging.info(
            f"Training {config['align_loss_type'].upper()} align loss: {train_losses['align']}"
        )
        logging.info(
            f"Training geometry effective rank: {train_losses['effective_rank']}"
        )
        logging.info(f"Training geometry avg cosine: {train_losses['avg_cosine']}")
        for token_idx in range(1, 5):
            logging.info(
                f"Training geometry effective rank token{token_idx}: {train_losses[f'effective_rank_token{token_idx}']}"
            )
            logging.info(
                f"Training geometry avg cosine token{token_idx}: {train_losses[f'avg_cosine_token{token_idx}']}"
            )
        writer.add_scalar("train/loss_total", train_losses["total"], epoch + 1)
        writer.add_scalar("train/loss_ce", train_losses["ce"], epoch + 1)
        writer.add_scalar("train/loss_align", train_losses["align"], epoch + 1)
        writer.add_scalar(
            "train/effective_rank_token_avg", train_losses["effective_rank"], epoch + 1
        )
        writer.add_scalar(
            "train/avg_cosine_token_avg", train_losses["avg_cosine"], epoch + 1
        )
        for token_idx in range(1, 5):
            writer.add_scalar(
                f"train/effective_rank_token{token_idx}",
                train_losses[f"effective_rank_token{token_idx}"],
                epoch + 1,
            )
            writer.add_scalar(
                f"train/avg_cosine_token{token_idx}",
                train_losses[f"avg_cosine_token{token_idx}"],
                epoch + 1,
            )
        valid_losses = validate(model, validation_dataloader, device)
        valid_loss = valid_losses[config["early_stop_metric"]]
        logging.info(f"Validation total loss: {valid_losses['total']}")
        logging.info(f"Validation CE loss: {valid_losses['ce']}")
        logging.info(
            f"Validation {config['align_loss_type'].upper()} align loss: {valid_losses['align']}"
        )
        logging.info(f"Early stop metric ({config['early_stop_metric']}): {valid_loss}")
        writer.add_scalar("valid/loss_total", valid_losses["total"], epoch + 1)
        writer.add_scalar("valid/loss_ce", valid_losses["ce"], epoch + 1)
        writer.add_scalar("valid/loss_align", valid_losses["align"], epoch + 1)
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            early_stop_counter = 0  # Reset early stop counter
            # Save the best model
            torch.save(model.state_dict(), config["save_path"])
            logging.info(
                f"Best validation {config['early_stop_metric']} loss: {best_loss}"
            )
            logging.info(f"Best model saved to {config['save_path']}")
        else:
            early_stop_counter += 1
            logging.info(
                f"No improvement in validation loss. Early stop counter: {early_stop_counter}"
            )
            if early_stop_counter >= config["early_stop"]:
                logging.info("Early stopping triggered.")
                break

    writer.close()
    logging.info("Loading best model for final testing...")
    model.load_state_dict(torch.load(config["save_path"], map_location=device))
    test_avg_recalls, test_avg_ndcgs = evaluate(
        model,
        test_dataloader,
        config["topk_list"],
        config["beam_size"],
        device,
        trie=item_trie,
    )
    logging.info(f"Final Epoch: {best_epoch + 1}")
    logging.info(f"Final Test Recalls: {test_avg_recalls}")
    logging.info(f"Final Test NDCGs: {test_avg_ndcgs}")
    print(f"Final Epoch: {best_epoch + 1}")
    print(f"Final Test Recalls: {test_avg_recalls}")
    print(f"Final Test NDCGs: {test_avg_ndcgs}")
