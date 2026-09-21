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

        history_token_levels = [item.get('history_token_level') for item in batch]
        use_mixed_history = history_token_levels[0] is not None
        if use_mixed_history:
            mixed_histories = []
            mixed_code_masks = []
            for history, token_levels in zip(histories, history_token_levels):
                positions = []
                code_masks = []
                for item_code, is_token_level in zip(history, token_levels):
                    item_code = [int(t) for t in item_code]
                    if all(t == pad_token for t in item_code):
                        positions.append([pad_token] * len(item_code))
                        code_masks.append([0] * len(item_code))
                    elif is_token_level:
                        for token in item_code:
                            positions.append([token] + [pad_token] * (len(item_code) - 1))
                            code_masks.append([1] + [0] * (len(item_code) - 1))
                    else:
                        positions.append(item_code)
                        code_masks.append([1] * len(item_code))
                mixed_histories.append(positions)
                mixed_code_masks.append(code_masks)
            max_positions = max(len(history) for history in mixed_histories)
            code_width = len(mixed_histories[0][0])
            flattened_histories = torch.tensor(
                [
                    [[pad_token] * code_width for _ in range(max_positions - len(history))]
                    + history
                    for history in mixed_histories
                ],
                dtype=torch.int64,
            )
            input_code_masks = torch.tensor(
                [
                    [[0] * code_width for _ in range(max_positions - len(mask))]
                    + mask
                    for mask in mixed_code_masks
                ],
                dtype=torch.int64,
            )
            attention_masks = (input_code_masks.sum(dim=2) > 0).to(torch.int64)
        else:
            # Flatten histories and targets
            flattened_histories = torch.stack(
                [torch.tensor([elem for sublist in history for elem in sublist], dtype=torch.int64) for history in histories]
            )
            input_code_masks = None
            # Create attention masks for flattened histories
            attention_masks = torch.stack(
                [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_histories]
            )
        flattened_targets = torch.stack(
            [torch.tensor(target, dtype=torch.int64) for target in targets]
        )

        output = {
            'history': flattened_histories,
            'target': flattened_targets,
            'attention_mask': attention_masks,
        }
        if input_code_masks is not None:
            output['input_code_mask'] = input_code_masks
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
        Build mixed item/token decoder batches for the causal model.

        Items whose ids are in the global eval-target set are expanded into
        4 SID-token positions and supervised with next-token decoder labels.
        Other items stay as one input-only item position whose 4 SID embeddings
        are averaged.
        """
        seq_lens = []
        input_list = []
        input_mask_list = []
        label_list = []
        phase_list = []
        group_list = []
        block_item_embs = [sample.get('block_item_emb') for sample in batch]
        has_item_emb = block_item_embs[0] is not None
        for sample in batch:
            block = sample['block']
            token_level_item = sample.get('block_token_level')
            if token_level_item is None:
                raise KeyError("train_parallel samples must include block_token_level")
            token_level_item = np.asarray(token_level_item, dtype=bool)
            if token_level_item.size != len(block):
                raise ValueError("block_token_level length must match block length")

            positions = []
            code_masks = []
            position_labels = []
            position_phases = []
            position_groups = []
            for item_idx, item_code in enumerate(block):
                item_code = [int(t) for t in item_code]
                if token_level_item[item_idx]:
                    for phase_idx, token in enumerate(item_code):
                        positions.append([token] + [pad_token] * (code_per_item - 1))
                        code_masks.append([1] + [0] * (code_per_item - 1))
                        position_labels.append(token)
                        position_phases.append(phase_idx + 1)
                        position_groups.append(item_idx)
                else:
                    positions.append(item_code)
                    code_masks.append([1] * code_per_item)
                    position_labels.append(-100)
                    position_phases.append(0)
                    position_groups.append(-1)

            labels = []
            phases = []
            groups = []
            for pos_idx in range(len(positions)):
                if pos_idx + 1 < len(positions) and position_labels[pos_idx + 1] != -100:
                    labels.append(position_labels[pos_idx + 1])
                    phases.append(position_phases[pos_idx + 1])
                    groups.append(position_groups[pos_idx + 1])
                else:
                    labels.append(-100)
                    phases.append(0)
                    groups.append(-1)

            input_ids = positions
            input_list.append(input_ids)
            input_mask_list.append(code_masks)
            label_list.append(labels)
            phase_list.append(phases)
            group_list.append(groups)
            seq_lens.append(len(input_ids))

        max_len = max(seq_lens)
        padded_inputs = []
        padded_input_masks = []
        padded_labels = []
        padded_phases = []
        padded_groups = []
        attention_masks = []
        for input_ids, input_code_mask, labels, phases, groups in zip(
            input_list, input_mask_list, label_list, phase_list, group_list
        ):
            pad = max_len - len(input_ids)
            # Left pad so the most recent tokens stay right-aligned.
            padded_inputs.append(
                [[pad_token] * code_per_item for _ in range(pad)] + input_ids
            )
            padded_input_masks.append(
                [[0] * code_per_item for _ in range(pad)] + input_code_mask
            )
            padded_labels.append([-100] * pad + labels)
            padded_phases.append([0] * pad + phases)
            padded_groups.append([-1] * pad + groups)
            attention_masks.append([0] * pad + [1] * len(input_ids))

        output = {
            'input_ids': torch.tensor(padded_inputs, dtype=torch.int64),
            'input_code_mask': torch.tensor(padded_input_masks, dtype=torch.int64),
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
