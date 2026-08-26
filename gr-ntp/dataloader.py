import numpy as np
import torch
from torch.utils.data import DataLoader

class GenRecDataLoader(DataLoader):
    """
    GenRecDataLoader for Generative Recommendation tasks.
    
    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): Number of subprocesses to use for data loading.
        collate_fn (callable, optional): Function to merge a list of samples to form a mini-batch.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=None, generator=None, worker_init_fn=None):
        self.max_len = dataset.max_len
        collate_fn = self.collate_fn
        super(GenRecDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, collate_fn=collate_fn,
                                               generator=generator, worker_init_fn=worker_init_fn)
    
            
    def collate_fn(self, batch, pad_token=0):
        """
        crate attention mask for input sequence.
        
        Args:
            batch (list): List of samples from the dataset.
        
        Returns:
            dict: Batched data with padded sequences.
        """
        if 'sequence' in batch[0]:
            return self.collate_ntp_fn(batch, pad_token)

        # Assuming each item in batch is a dictionary with 'history' and 'target'
        histories = [item['history'] for item in batch]
        targets = [item['target'] for item in batch]

        # Flatten histories and targets
        flattened_histories = torch.stack(
            [torch.tensor([elem for sublist in history for elem in sublist], dtype=torch.int64) for history in histories]
        )
        flattened_targets = torch.stack(
            [torch.tensor(target, dtype=torch.int64) for target in targets]
        )

        # Create attention masks for flattened histories
        attention_masks = torch.stack(
            [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_histories]
        )

        return {'history': flattened_histories, 'target': flattened_targets, 'attention_mask': attention_masks}

    def collate_ntp_fn(self, batch, pad_token=0):
        sequences = [item['sequence'] for item in batch]
        item_loss_masks = [item['item_loss_mask'] for item in batch]
        code_len = len(sequences[0][0])
        max_items = max(len(sequence) for sequence in sequences)
        max_tokens = max_items * code_len

        input_ids = []
        token_attention_masks = []
        loss_masks = []
        attention_masks = []
        position_ids = []
        window_tokens = self.max_len * code_len

        for sequence, item_loss_mask in zip(sequences, item_loss_masks):
            item_count = len(sequence)
            flat_sequence = [elem for code in sequence for elem in code]
            pad_size = max_tokens - len(flat_sequence)
            input_ids.append(torch.tensor(flat_sequence + [pad_token] * pad_size, dtype=torch.int64))

            token_mask = torch.zeros(max_tokens, dtype=torch.bool)
            token_mask[:len(flat_sequence)] = True
            token_attention_masks.append(token_mask.to(torch.int64))
            position_ids.append(
                torch.arange(max_tokens, dtype=torch.int64) % window_tokens
            )

            flat_loss_mask = []
            for use_loss in item_loss_mask:
                flat_loss_mask.extend([use_loss] * code_len)
            loss_mask = torch.zeros(max_tokens, dtype=torch.bool)
            loss_mask[:len(flat_loss_mask)] = torch.tensor(flat_loss_mask, dtype=torch.bool)
            loss_masks.append(loss_mask)

            window_mask = torch.zeros((max_tokens, max_tokens), dtype=torch.bool)
            for query_pos in range(len(flat_sequence)):
                target_pos = min(query_pos + 1, len(flat_sequence) - 1)
                target_item_idx = target_pos // code_len
                context_items = max(self.max_len - 1, 1)
                min_item_idx = max(0, target_item_idx - context_items)
                key_limit = query_pos + 1
                for key_pos in range(key_limit):
                    key_item_idx = key_pos // code_len
                    if key_item_idx >= min_item_idx:
                        window_mask[query_pos, key_pos] = True
            attention_masks.append(window_mask)

        input_ids = torch.stack(input_ids)
        token_attention_masks = torch.stack(token_attention_masks)
        attention_masks = torch.stack(attention_masks)
        attention_masks = attention_masks & token_attention_masks[:, None, :].to(torch.bool)

        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'loss_mask': torch.stack(loss_masks),
            'attention_mask': attention_masks,
            'token_attention_mask': token_attention_masks,
            'position_ids': torch.stack(position_ids),
        }
