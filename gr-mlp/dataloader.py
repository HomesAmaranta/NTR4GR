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
        parallel = getattr(dataset, 'mode', None) == 'train_parallel'
        collate_fn = self.collate_fn_parallel if parallel else self.collate_fn
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
        # Assuming each item in batch is a dictionary with 'history' and 'target'
        histories = [item['history'] for item in batch]
        targets = [item['target'] for item in batch]
        target_item_embs = [item.get('target_item_emb') for item in batch]

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

        output = {
            'history': flattened_histories,
            'target': flattened_targets,
            'attention_mask': attention_masks,
        }
        if target_item_embs[0] is not None:
            target_item_embs = torch.stack(
                [
                    torch.tensor(np.asarray(emb), dtype=torch.float32)
                    for emb in target_item_embs
                ]
            )
            if target_item_embs.dim() == 2:
                target_item_embs = target_item_embs.unsqueeze(1).repeat(
                    1,
                    flattened_targets.size(1),
                    1,
                )
            output['target_item_emb'] = target_item_embs

        return output

    def collate_fn_parallel(self, batch, pad_token=0, code_per_item=4):
        """
        Build parallel next-token-prediction batches for the causal model.

        Each sample is a block of items (list of 4-token code arrays). The block
        is flattened into a token sequence; ``input_ids`` and ``labels`` are the
        sequence shifted by one token. Positions that fall inside the overlap
        (context-only) region, as well as left-padding, are set to -100 in
        ``labels`` so they are ignored by the cross-entropy loss.

        Returns a dict with 'input_ids', 'labels' and 'attention_mask', all left
        padded to the longest sequence in the batch.
        """
        token_seqs = []
        input_list = []
        label_list = []
        for sample in batch:
            block = sample['block']
            context_items = sample.get('context_items', 0)
            # Flatten items -> tokens.
            tokens = [int(t) for item_code in block for t in item_code]
            # Shift by one token for next-token prediction.
            input_ids = tokens[:-1]
            labels = tokens[1:]
            # Mask the overlap (context-only) region: any predicted token that
            # belongs to the first `context_items` items is context-only.
            # A label at index i predicts token (i + 1); mask it while that
            # target token still lives inside the context region.
            context_tokens = context_items * code_per_item
            labels = [
                (lab if (i + 1) >= context_tokens else -100)
                for i, lab in enumerate(labels)
            ]
            input_list.append(input_ids)
            label_list.append(labels)
            token_seqs.append(len(input_ids))

        max_len = max(token_seqs)
        padded_inputs = []
        padded_labels = []
        attention_masks = []
        for input_ids, labels in zip(input_list, label_list):
            pad = max_len - len(input_ids)
            # Left pad so the most recent tokens stay right-aligned.
            padded_inputs.append([pad_token] * pad + input_ids)
            padded_labels.append([-100] * pad + labels)
            attention_masks.append([0] * pad + [1] * len(input_ids))

        output = {
            'input_ids': torch.tensor(padded_inputs, dtype=torch.int64),
            'labels': torch.tensor(padded_labels, dtype=torch.int64),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.int64),
        }
        return output
