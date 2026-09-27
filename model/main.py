import torch
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
import os
import random
import pandas as pd
from tqdm import tqdm
import logging
import torch.nn.functional as F
from dataset import GenRecDataset
from dataloader import GenRecDataLoader
from generation_trie import Trie, prefix_allowed_tokens_fn

class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
        num_layers=config['num_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        d_kv=config['d_kv'],
        dropout_rate=config['dropout_rate'],
        vocab_size=config['vocab_size'],
        pad_token_id=config['pad_token_id'],
        eos_token_id=config['eos_token_id'],
        decoder_start_token_id=config['pad_token_id'],
        feed_forward_proj=config['feed_forward_proj'],
    )
        # Initialize T5 model with the specified configuration
        self.model = T5ForConditionalGeneration(t5config)
        self.item_emb_dim = config.get('item_emb_dim', 0)
        self.mse_loss_weight = config.get('mse_loss_weight', 0.0)
        self.mse_loss_mode = config.get('mse_loss_mode', 'mean')
        self.align_loss_type = config.get('align_loss_type', 'mse')
        self.align_target = config.get('align_target', 'quantized')
        self.align_item = config.get('align_item', 'current')
        if self.align_item == 'pre':
            self.align_item = 'current'
        if self.mse_loss_mode != 'mean':
            raise ValueError("TIGER/model only supports mse_loss_mode='mean'")
        if self.align_loss_type not in {'mse', 'cos'}:
            raise ValueError("align_loss_type must be 'mse' or 'cos'")
        if self.align_target not in {'item', 'latent', 'quantized'}:
            raise ValueError("align_target must be 'item', 'latent' or 'quantized'")
        if self.align_item not in {'current', 'next'}:
            raise ValueError("align_item must be 'current', 'pre' or 'next'")
        self.hidden_to_item_emb = (
            nn.Sequential(
                nn.Linear(config['d_model'], config['d_model']),
                nn.GELU(),
                nn.Linear(config['d_model'], self.item_emb_dim, bias=False),
            )
            if self.item_emb_dim > 0
            else None
        )
    
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
          f'#Embedding parameters: {emb_params}\n'
          f'#Non-embedding parameters: {total_params - emb_params}\n'
          f'#Total trainable parameters: {total_params}\n'
      )

    def _align_loss_from_hidden(
        self,
        hidden_states: torch.Tensor,
        target_item_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.hidden_to_item_emb is None:
            return hidden_states.new_zeros(())
        if target_item_emb.dim() == 3:
            target_item_emb = target_item_emb.mean(dim=1)
        pred_item_emb = self.hidden_to_item_emb(hidden_states.mean(dim=1))
        target_item_emb = target_item_emb.to(pred_item_emb.dtype)
        if self.align_loss_type == 'cos':
            return 1.0 - F.cosine_similarity(
                pred_item_emb,
                target_item_emb,
                dim=-1,
            ).mean()
        return F.mse_loss(pred_item_emb, target_item_emb, reduction='mean')

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        target_item_emb: Optional[torch.Tensor] = None,
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
      target_hidden_states = outputs.decoder_hidden_states[-1]
      self.last_target_hidden_states = target_hidden_states.detach()
      align_loss = target_hidden_states.new_zeros(())
      if (
          target_item_emb is not None
          and self.hidden_to_item_emb is not None
          and self.mse_loss_weight > 0
      ):
          align_loss = self._align_loss_from_hidden(
              target_hidden_states,
              target_item_emb,
          )
      loss = outputs.loss + self.mse_loss_weight * align_loss
      self.last_loss_dict = {
          'total': loss.detach(),
          'ce': outputs.loss.detach(),
          'align': align_loss.detach(),
      }
      return loss, outputs.logits
    
    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,  num_beams: int = 20, **kwargs):
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
            **kwargs
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
    assert (
        preds.shape[1] == maxk
    ), f'preds.shape[1] = {preds.shape[1]} != {maxk}'

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
  dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
  return dcg[:, :k].sum(dim=1).cpu().float()

def train(model, train_loader, optimizer, device):
    model.train()
    totals = {'total': 0.0, 'ce': 0.0, 'align': 0.0}
    for batch in train_loader:
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)
        target_item_emb = batch.get('target_item_emb')
        target_item_emb = target_item_emb.to(device) if target_item_emb is not None else None

        optimizer.zero_grad()
        loss, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            target_item_emb=target_item_emb,
        )
        loss.backward()
        optimizer.step()

        loss_dict = getattr(model, 'last_loss_dict', None)
        totals['total'] += loss.item()
        totals['ce'] += (
            loss_dict['ce'].item()
            if loss_dict is not None and 'ce' in loss_dict
            else loss.item()
        )
        totals['align'] += (
            loss_dict['align'].item()
            if loss_dict is not None and 'align' in loss_dict
            else 0.0
        )
        
    return {k: v / len(train_loader) for k, v in totals.items()}

def validate(model, valid_loader, device):
    model.eval()
    totals = {'total': 0.0, 'ce': 0.0, 'align': 0.0}
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)
            target_item_emb = batch.get('target_item_emb')
            target_item_emb = target_item_emb.to(device) if target_item_emb is not None else None
            loss, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                target_item_emb=target_item_emb,
            )
            loss_dict = getattr(model, 'last_loss_dict', None)
            totals['total'] += loss.item()
            totals['ce'] += (
                loss_dict['ce'].item()
                if loss_dict is not None and 'ce' in loss_dict
                else loss.item()
            )
            totals['align'] += (
                loss_dict['align'].item()
                if loss_dict is not None and 'align' in loss_dict
                else 0.0
            )
    return {k: v / len(valid_loader) for k, v in totals.items()}

def evaluate(model, eval_loader, topk_list, beam_size, device, trie=None):
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    
    constraint_fn = prefix_allowed_tokens_fn(trie) if trie is not None else None
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=beam_size,
                prefix_allowed_tokens_fn=constraint_fn
            )
            preds = preds[:, 1:]  # Exclude the start token
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)  # Reshape to (batch_size, beam_size, seq_len)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            # print(f"pos_index shape: {pos_index.shape}, pos_index: {pos_index}")
            for k in topk_list:
                recall = recall_at_k(pos_index, k).mean().item()
                ndcg = ndcg_at_k(pos_index, k).mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)
    # Calculate average recalls and ndcgs
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    return avg_recalls, avg_ndcgs

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIGER configuration")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--infer_size', type=int, default=96, help='Inference size for generating recommendations')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers in the model')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--d_ff', type=int, default=1024, help='Dimension of the feed-forward layer')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--d_kv', type=int, default=64, help='Dimension of key and value vectors')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--vocab_size', type=int, default=1025, help='Vocabulary size')
    parser.add_argument('--pad_token_id', type=int, default=0, help='Padding token ID')
    parser.add_argument('--eos_token_id', type=int, default=0, help='End of sequence token ID')
    parser.add_argument('--feed_forward_proj', type=str, default='relu', help='Feed forward projection type')
    parser.add_argument('--max_len', type=int, default=20, help='Maximum length for padding or truncation')
    parser.add_argument('--dataset_path', type=str, default='../data/Beauty', help='Path to the dataset')
    parser.add_argument('--code_path', type=str, default='../data/Beauty/Beauty_t5_rqvae.npy', help='Path to the item-to-code mapping file')
    parser.add_argument('--item_emb_path', type=str, default=None, help='Path to current/next item embedding parquet for auxiliary alignment loss')
    parser.add_argument('--item_emb_dim', type=int, default=0, help='Embedding dim for auxiliary alignment target')
    parser.add_argument('--mse_loss_weight', type=float, default=0.0, help='Weight of auxiliary representation alignment loss')
    parser.add_argument('--mse_loss_mode', type=str, default='mean', choices=['mean'], help='Mean-pool decoder final hidden states before alignment')
    parser.add_argument('--align_loss_type', type=str, default='mse', choices=['mse', 'cos'], help='Auxiliary alignment loss type')
    parser.add_argument('--align_target', type=str, default='quantized', choices=['item', 'latent', 'quantized'], help='Auxiliary alignment target source')
    parser.add_argument('--align_item', type=str, default='current', choices=['current', 'pre', 'next'], help='Use current/pre item or next target item embedding for alignment')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluation'], help='Mode of operation')
    parser.add_argument('--log_path', type=str, default='./logs/tiger.log', help='Path to the log file')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./ckpt/tiger.pth', help='Path to save the trained model')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--topk_list', type=list, default=[5,10,20], help='List of top-k values for evaluation metrics')
    parser.add_argument('--beam_size', type=int, default=30, help='Beam size for generation')
    config = vars(parser.parse_args())
    if config['item_emb_path'] in {'', 'None'}:
        config['item_emb_path'] = None
    if config['mse_loss_weight'] > 0 and config['item_emb_path'] is None:
        raise ValueError('mse_loss_weight > 0 requires --item_emb_path')
    if config['mse_loss_weight'] > 0 and config['item_emb_dim'] <= 0:
        raise ValueError('mse_loss_weight > 0 requires --item_emb_dim > 0')
    # Set up logging
    logging.basicConfig(
        filename=config['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration: {config}")
    
    # Initialize model
    model = TIGER(config)
    print(model.n_parameters)
    logging.info(model.n_parameters)

    # Set random seed for reproducibility
    set_seed(config['seed'])
    # Check if the device is available
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    needs_item_emb = config['mse_loss_weight'] > 0
    
    train_dataset = GenRecDataset(
        dataset_path=config['dataset_path']+ '/train.parquet',
        code_path=config['code_path'],
        mode='train',
        max_len=config['max_len'],
        item_emb_path=(config['item_emb_path'] if needs_item_emb else None),
        align_item=config['align_item'],
    )
    validation_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/valid.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len'],
        item_emb_path=(config['item_emb_path'] if needs_item_emb else None),
        align_item=config['align_item'],
    )
    test_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/test.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len']
    )

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=config['infer_size'], shuffle=False)
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)

    print("Building Trie...")
    trie_sequences = []
    for code in list(test_dataset.item_to_code.values()):
        code_list = list(code) if isinstance(code, (np.ndarray, list)) else list(code)
        trie_sequences.append([config['pad_token_id']] + code_list + [config['eos_token_id']])
    item_trie = Trie(trie_sequences)
    print("Trie built.")
    
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
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Train the model
    model.to(device)
    best_loss = 10000.0
    early_stop_counter = 0
    best_epoch = 0
    
    for epoch in tqdm(range(config['num_epochs'])):
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_loss = train(model, train_dataloader, optimizer, device)
        logging.info(f"Training loss: {train_loss}")
        valid_loss = validate(model, validation_dataloader, device)
        logging.info(f"Validation loss: {valid_loss}")
        if valid_loss['total'] < best_loss:
            best_loss = valid_loss['total']
            best_epoch = epoch
            early_stop_counter = 0  # Reset early stop counter
            # Save the best model
            torch.save(model.state_dict(), config['save_path'])
            logging.info(f"Best validation loss: {best_loss}")
            logging.info(f"Best model saved to {config['save_path']}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in validation loss. Early stop counter: {early_stop_counter}")
            if early_stop_counter >= config['early_stop']:
                logging.info("Early stopping triggered.")
                break

    logging.info("Loading best model for final testing...")
    model.load_state_dict(torch.load(config['save_path'], map_location=device))
    test_avg_recalls, test_avg_ndcgs = evaluate(model, test_dataloader, config['topk_list'], config['beam_size'], device, trie=item_trie)
    logging.info(f"Final Epoch: {best_epoch + 1}")
    logging.info(f"Final Test Recalls: {test_avg_recalls}")
    logging.info(f"Final Test NDCGs: {test_avg_ndcgs}")
    print(f"Final Epoch: {best_epoch + 1}")
    print(f"Final Test Recalls: {test_avg_recalls}")
    print(f"Final Test NDCGs: {test_avg_ndcgs}")
        
