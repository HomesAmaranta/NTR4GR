import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def process_data(file_path, mode, max_len, PAD_TOKEN=0,
                 block_items=40, stride_items=20):
    """
    Process parquet data based on mode ('train', 'train_parallel' or 'evaluation').

    Args:
        file_path (str): Path to the parquet file.
        mode (str): Mode of operation ('train', 'train_parallel' or 'evaluation').
        max_len (int): Maximum length for padding or truncation.
        block_items (int): Block size in items for 'train_parallel' mode.
        stride_items (int): Sliding stride in items for 'train_parallel' mode.

    Returns:
        list: Processed data.
    """
    # Load parquet data
    data = pd.read_parquet(file_path)

    # Combine "history" and "target" columns into a single sequence
    # Important: Ensure 'history' is a list and 'target' is appended correctly
    data['sequence'] = data['history'].apply(lambda x: list(x)) + data['target'].apply(lambda x: [x])

    if mode == 'train':
        # Sliding window processing
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            for i in range(1, len(sequence)):
                processed_data.append({
                    'history': sequence[:i],
                    'target': sequence[i],
                    'pre_item': sequence[i - 1],
                })
    elif mode == 'train_parallel':
        # Cut each user sequence into overlapping big blocks. Each block keeps
        # the whole item sequence so the causal model can predict every
        # next token in a single forward pass. The first `context_items` items
        # of a block (the part overlapping the previous block) only serve as
        # context and are excluded from the loss.
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            n = len(sequence)
            if n < 2:
                continue
            start = 0
            while True:
                block = sequence[start:start + block_items]
                # Overlap items (kept for context only) = items before `start`
                # that still live inside this block window. For the first
                # block it is 0; for later blocks it is block_items - stride.
                context_items = 0 if start == 0 else (block_items - stride_items)
                context_items = min(context_items, len(block) - 1)
                processed_data.append({
                    'block': block,
                    'context_items': context_items,
                })
                if start + block_items >= n:
                    break
                start += stride_items
    elif mode == 'evaluation':
        # Use the last item as target and the rest as history
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            history = sequence[:-1]
            processed_data.append({
                'history': history,
                'target': sequence[-1],
                'pre_item': history[-1] if history else sequence[-1],
            })
    else:
        raise ValueError("Mode must be 'train', 'train_parallel' or 'evaluation'.")

    # Apply padding or truncation (only for modes that use fixed-length history;
    # train_parallel keeps variable-length blocks and pads at the token level
    # inside collate_fn).
    if mode != 'train_parallel':
        for item in processed_data:
            item['history'] = pad_or_truncate(item['history'], max_len)

    return processed_data

def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    """
    Pad or truncate a sequence to a specified maximum length.

    Args:
        sequence (list): Input sequence.
        max_len (int): Maximum length for the sequence.

    Returns:
        list: Padded or truncated sequence.
    """
    if len(sequence) > max_len:
        # Truncate sequence
        return sequence[-max_len:]
    else:
        # Left pad sequence with PAD_TOKEN
        return [PAD_TOKEN] * (max_len - len(sequence)) + sequence
    
def item2code(code_path, codebook_size=256):
    """
    Convert itemID to code
    :param code_path: npy file path to store rqvae codes
    :return: dict item_to_code, code_to_item
    """
    data = np.load(code_path, allow_pickle=True)
    item_to_code = {}
    code_to_item = {}
    
    # for index, code in enumerate(data):
    #     item_to_code[index + 1] = code
    #     code_to_item[tuple(code)] = index + 1
    for index, code in enumerate(data):
        offsets = [c + i * codebook_size + 1 for i,c in enumerate(code)]
        item_to_code[index + 1] = offsets
        code_to_item[tuple(offsets)] = index + 1

    return item_to_code, code_to_item

def load_item_embeddings(item_emb_path):
    data = pd.read_parquet(item_emb_path)
    def to_array(embedding):
        arr = np.asarray(embedding)
        if arr.dtype == object:
            arr = np.stack(embedding)
        return arr.astype(np.float32)
    return {
        int(row.ItemID): to_array(row.embedding)
        for row in data.itertuples(index=False)
    }

class GenRecDataset(Dataset):
    def __init__(self, dataset_path, code_path, mode, max_len, PAD_TOKEN=0, item_emb_path=None, align_item='next',
                 block_items=40, stride_items=20):
        """
        Initialize the GenRecDataset.
        Args:
            dataset_path (str): Path to the dataset file.
            code_path (str): Path to the item-to-code mapping file.
            mode (str): Mode of operation ('train', 'train_parallel' or 'evaluation').
            max_len (int): Maximum length for padding or truncation.
            PAD_TOKEN (int, optional): Token used for padding. Defaults to 0.
            block_items (int): Block size in items for 'train_parallel' mode.
            stride_items (int): Sliding stride in items for 'train_parallel' mode.
        """
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.mode = mode
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.block_items = block_items
        self.stride_items = stride_items
        if align_item not in {'pre', 'next', 'near'}:
            raise ValueError(f"Unsupported align_item: {align_item}")
        self.align_item = align_item
        # Load item-to-code mapping
        self.item_to_code, self.code_to_item = item2code(code_path)
        self.item_embeddings = load_item_embeddings(item_emb_path) if item_emb_path else None
        # Process the dataset
        self.data = self._prepare_data()
        
    def _prepare_data(self):
        """
        Process the dataset and convert items to codes.
        Returns:
            list: Processed data with items converted to codes.
        """
        # Process the data using the process_data function
        processed_data = process_data(
            self.dataset_path, self.mode, self.max_len, self.PAD_TOKEN,
            block_items=self.block_items, stride_items=self.stride_items,
        )
        if self.mode == 'train_parallel':
            # Convert each block's items into their 4-token codes. The block
            # stays as a list of per-item code arrays; flattening / label
            # shifting happens in the dataloader collate_fn. When item
            # embeddings are provided (for the auxiliary alignment loss), also
            # attach the per-item target embedding. In next-token parallel
            # training the position that predicts item k is aligned with the
            # embedding of item k itself (align_item='next'); with
            # align_item='pre' it is aligned with the previous item (k-1).
            for item in processed_data:
                item_ids = list(item['block'])
                item['block'] = [
                    self.item_to_code.get(x, np.array([self.PAD_TOKEN] * 4))
                    for x in item_ids
                ]
                if self.item_embeddings is not None:
                    block_item_emb = []
                    for k, x in enumerate(item_ids):
                        if self.align_item == 'pre':
                            src = item_ids[k - 1] if k > 0 else x
                        else:
                            src = x
                        src = int(src)
                        if src not in self.item_embeddings:
                            raise KeyError(
                                f"Missing item embedding for align item id: {src}"
                            )
                        block_item_emb.append(self.item_embeddings[src])
                    item['block_item_emb'] = block_item_emb
            return processed_data
        # Convert items to codes
        for item in processed_data:
            target_item = item['target']
            align_item = item['pre_item'] if self.align_item == 'pre' else target_item
            item['history'] = [self.item_to_code.get(x, np.array([self.PAD_TOKEN]*4)) for x in item['history']]
            item['target'] = self.item_to_code.get(item['target'], np.array([self.PAD_TOKEN]*4))
            if self.item_embeddings is not None:
                align_item = int(align_item)
                if align_item not in self.item_embeddings:
                    raise KeyError(f"Missing item embedding for align item id: {align_item}")
                item['target_item_emb'] = self.item_embeddings[align_item]
        return processed_data
    
    def __getitem__(self, index):
        """
        Get a single data item by index.
        Args:
            index (int): Index of the data item.
        Returns:
            dict: A dictionary containing 'history' and 'target'.
        """
        return self.data[index]
    
    def __len__(self):
        """
        Get the total number of data.
        Returns:
            int: Total number of data.
        """
        return len(self.data)
    
if __name__ == "__main__":
    # Example usage
    dataset_path = '../data/Beauty/train.parquet'
    code_path = '../data/Beauty/Beauty_t5_rqvae.npy'
    mode = 'train'  # or 'train'
    max_len = 20

    dataset = GenRecDataset(dataset_path, code_path, mode, max_len)
    print("Number of items in dataset:", len(dataset))

    print("First five items in dataset:", [dataset[i] for i in range(5)])
    
