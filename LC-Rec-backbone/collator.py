import torch
import copy
import argparse
from dataclasses import dataclass

import transformers
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, T5Tokenizer, T5Config, T5ForConditionalGeneration


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # print(self.tokenizer.model_max_length)

    def __call__(self, batch):
        
        input_texts = [d["input_ids"] for d in batch]
        label_texts = [d["labels"] for d in batch]

        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)

        labels = self.tokenizer(label_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=self.tokenizer.model_max_length,
                                truncation=True,
                                return_attention_mask=True)
        inputs['labels'] = labels['input_ids']
        inputs['labels'][inputs['labels'] == self.tokenizer.pad_token_id] = -100

        return inputs

class Collator_DecoderOnly_manual(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        print("*** only train response:",  self.only_train_response)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id


    def __call__(self, batch):

        if not self.only_train_response:

            full_texts = [d["input_ids"] + d["labels"] + self.tokenizer.eos_token for d in batch]
            inputs = self.tokenizer(
                text = full_texts,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True,
            )
            labels = copy.deepcopy(inputs["input_ids"])
            inputs["labels"] = labels
        else:

            input_texts = [d["input_ids"] for d in batch]
            full_texts = [d["input_ids"] + d["labels"] + self.tokenizer.eos_token for d in batch]

            inputs = self.tokenizer(
                text = full_texts,
                text_target = input_texts,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_attention_mask=True,
            )
            labels = copy.deepcopy(inputs["input_ids"])


            labels[labels == self.tokenizer.pad_token_id] = -100

            labels[torch.where(inputs["labels"] != self.tokenizer.pad_token_id)] = -100


            inputs["labels"] = labels
            
        return inputs

class AlignmentCollator(Collator_DecoderOnly_manual):
    """Keep the original LM batch and add four causal prediction positions."""

    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        separator = tokenizer.encode(args.special_token_for_answer, add_special_tokens=False)
        if len(separator) != 1:
            raise ValueError("Alignment requires the answer separator to be registered as one token")
        self.separator_id = separator[0]

    def __call__(self, batch):
        inputs = super().__call__(batch)
        positions = []
        for row, sample in enumerate(batch):
            token_ids = inputs["input_ids"][row]
            mask = inputs["attention_mask"][row]
            separator_positions = torch.where((token_ids == self.separator_id) & mask.bool())[0]
            if len(separator_positions) != 1:
                raise ValueError("Expected one answer separator; check prompt formatting and model_max_length")
            separator = separator_positions.item()
            target_ids = self.tokenizer.encode(sample["labels"], add_special_tokens=False)
            if len(target_ids) != 4:
                raise ValueError("Alignment requires exactly four target SID tokens")
            start, end = separator + 1, separator + 5
            if (
                end > token_ids.shape[0]
                or token_ids[start:end].tolist() != target_ids
                or not mask[start:end].bool().all()
            ):
                raise ValueError("The complete target SID was truncated; increase model_max_length")
            # Target positions [c1,c2,c3,c4] minus one => [separator,c1,c2,c3].
            positions.append(list(range(separator, separator + 4)))
        inputs["alignment_positions"] = torch.tensor(positions, dtype=torch.long)
        inputs["target_item_emb"] = torch.stack([
            torch.as_tensor(sample["target_item_emb"], dtype=torch.float32) for sample in batch
        ])
        return inputs


class Collator_DecoderOnly(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.only_train_response = args.only_train_response
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

            
    def __call__(self, batch):

        input_texts = [d["input_ids"] + d["labels"]+ self.tokenizer.eos_token for d in batch]

        inputs = self.tokenizer(input_texts,
                                return_tensors="pt",
                                padding="longest",
                                max_length=512,
                                truncation=True,
                                return_attention_mask=True)

        inputs['labels'] = copy.deepcopy(inputs["input_ids"])
        
        return inputs

class TestCollator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0

    def __call__(self, batch):

        input_texts = [d["input_ids"] for d in batch]
        targets = [d["labels"] for d in batch]

        inputs = self.tokenizer(
            text=input_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return (inputs, targets)
