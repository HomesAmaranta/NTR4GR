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
