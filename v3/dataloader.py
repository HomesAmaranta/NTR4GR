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
        item_seqs = []
        input_list = []
        label_list = []
        phase_list = []
        group_list = []
        block_item_embs = [sample.get('block_item_emb') for sample in batch]
        has_item_emb = block_item_embs[0] is not None
        for sample in batch:
            block = sample['block']
            context_items = sample.get('context_items', 0)
            # Item-level causal prediction. One model position represents one
            # item, whose input embedding is the mean of its 4 SID token
            # embeddings. The hidden state of item k predicts all 4 SID tokens
            # of item k+1 through 4 independent heads.
            input_ids = [[int(t) for t in item_code] for item_code in block[:-1]]
            labels = [[int(t) for t in item_code] for item_code in block[1:]]
            loss_start_item = max(context_items, 1)
            masked_labels = []
            phases = []
            groups = []
            for i, lab in enumerate(labels):
                target_item_idx = i + 1
                if target_item_idx < loss_start_item:
                    masked_labels.append([-100] * code_per_item)
                    phases.extend([0] * code_per_item)
                    groups.append(-1)
                else:
                    masked_labels.append(lab)
                    phases.extend(list(range(1, code_per_item + 1)))
                    groups.append(target_item_idx)
            labels = masked_labels
            input_list.append(input_ids)
            label_list.append(labels)
            phase_list.append(phases)
            group_list.append(groups)
            item_seqs.append(len(input_ids))

        max_len = max(item_seqs)
        padded_inputs = []
        padded_labels = []
        padded_phases = []
        padded_groups = []
        attention_masks = []
        for input_ids, labels, phases, groups in zip(
            input_list, label_list, phase_list, group_list
        ):
            pad = max_len - len(input_ids)
            # Left pad so the most recent tokens stay right-aligned.
            padded_inputs.append([[pad_token] * code_per_item] * pad + input_ids)
            padded_labels.append([[-100] * code_per_item] * pad + labels)
            padded_phases.append([0] * (pad * code_per_item) + phases)
            padded_groups.append([-1] * pad + groups)
            attention_masks.append([0] * pad + [1] * len(input_ids))

        output = {
            'input_ids': torch.tensor(padded_inputs, dtype=torch.int64),
            'labels': torch.tensor(padded_labels, dtype=torch.int64),
            'code_phase': torch.tensor(padded_phases, dtype=torch.int64),
            'item_group': torch.tensor(padded_groups, dtype=torch.int64),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.int64),
        }
        if has_item_emb:
            # Pad the per-item target embeddings to [B, max_items, ...]. Item
            # groups index into this tensor (group g -> block_item_emb[g]).
            max_items = max(len(emb) for emb in block_item_embs)
            emb_shape = np.asarray(block_item_embs[0][0]).shape
            padded_embs = np.zeros(
                (len(batch), max_items, *emb_shape), dtype=np.float32
            )
            for b, embs in enumerate(block_item_embs):
                for k, emb in enumerate(embs):
                    padded_embs[b, k] = np.asarray(emb, dtype=np.float32)
            output['block_item_emb'] = torch.from_numpy(padded_embs)
        return output
