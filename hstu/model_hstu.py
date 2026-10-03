import math

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100


class RelativePositionalBias(nn.Module):
    """Learned relative-position bias without timestamp features."""

    def __init__(self, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.weight = nn.Parameter(torch.empty(2 * max_seq_len - 1))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, seq_len):
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured maximum "
                f"{self.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=self.weight.device)
        offsets = positions[:, None] - positions[None, :]
        indices = offsets + self.max_seq_len - 1
        return self.weight[indices]


class HSTUBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dqk,
        dv,
        dropout_rate,
        max_seq_len,
        code_length,
        max_history_items,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dqk = dqk
        self.dv = dv
        self.code_length = code_length
        self.max_history_items = max_history_items
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.uvqk = nn.Linear(
            embedding_dim,
            num_heads * (2 * dv + 2 * dqk),
            bias=False,
        )
        self.relative_bias = RelativePositionalBias(max_seq_len)
        self.out = nn.Linear(num_heads * dv, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.normal_(self.uvqk.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x, attention_mask):
        _, seq_len, _ = x.shape
        uvqk = F.silu(self.uvqk(self.norm(x)))
        u, v, q, k = torch.split(
            uvqk,
            [
                self.num_heads * self.dv,
                self.num_heads * self.dv,
                self.num_heads * self.dqk,
                self.num_heads * self.dqk,
            ],
            dim=-1,
        )
        u = u.view(x.size(0), seq_len, self.num_heads, self.dv)
        v = v.view(x.size(0), seq_len, self.num_heads, self.dv)
        q = q.view(x.size(0), seq_len, self.num_heads, self.dqk)
        k = k.view(x.size(0), seq_len, self.num_heads, self.dqk)

        scores = torch.einsum("bnhd,bmhd->bhnm", q, k)
        scores = scores + self.relative_bias(seq_len)[None, None, :, :]
        scores = F.silu(scores) / seq_len

        positions = torch.arange(seq_len, device=x.device)
        item_positions = positions // self.code_length
        item_distances = item_positions[:, None] - item_positions[None, :]
        causal_mask = (
            positions[:, None] >= positions[None, :]
        ) & (
            item_distances < self.max_history_items
        )
        valid_mask = (
            causal_mask[None, None, :, :]
            & attention_mask[:, None, None, :].bool()
        )
        scores = scores * valid_mask.to(scores.dtype)

        attention_output = torch.einsum("bhnm,bmhd->bnhd", scores, v)
        attention_output = F.layer_norm(
            attention_output.reshape(
                x.size(0), seq_len, self.num_heads * self.dv
            ),
            [self.num_heads * self.dv],
            eps=1e-6,
        )
        gated = (
            u.reshape(x.size(0), seq_len, self.num_heads * self.dv)
            * attention_output
        )
        output = x + self.out(self.dropout(gated))
        return output.masked_fill(attention_mask.unsqueeze(-1) == 0, 0.0)


class HSTURec(nn.Module):
    """Causal HSTU language model over SID tokens."""

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        code_length=4,
        max_history_items=20,
        embedding_dim=64,
        num_blocks=2,
        num_heads=4,
        dqk=32,
        dv=32,
        dropout_rate=0.1,
        pad_token_id=0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.code_length = code_length
        self.max_history_items = max_history_items
        self.pad_token_id = pad_token_id
        self.sid_emb = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_token_id
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [
                HSTUBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dqk=dqk,
                    dv=dv,
                    dropout_rate=dropout_rate,
                    max_seq_len=max_seq_len,
                    code_length=code_length,
                    max_history_items=max_history_items,
                )
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.sid_emb.weight

    @property
    def n_parameters(self):
        count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"#Total params: {count}"

    def encode(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id).long()
        x = self.sid_emb(input_ids) * math.sqrt(self.sid_emb.embedding_dim)
        x = self.dropout(x)
        x = x.masked_fill(attention_mask.unsqueeze(-1) == 0, 0.0)
        for block in self.blocks:
            x = block(x, attention_mask)
        return self.norm(x).masked_fill(
            attention_mask.unsqueeze(-1) == 0, 0.0
        )

    def logits_from_hidden(self, hidden):
        logits = self.lm_head(hidden)
        logits[..., self.pad_token_id] = torch.finfo(logits.dtype).min
        return logits

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        hidden = self.encode(input_ids, attention_mask)
        logits = self.logits_from_hidden(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return loss, logits

    def next_token_logits(self, input_ids, attention_mask):
        hidden = self.encode(input_ids, attention_mask)
        lengths = attention_mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("Cannot generate from an empty SID sequence")
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        return self.logits_from_hidden(hidden[rows, lengths - 1])

    @staticmethod
    def _append_prefixes(input_ids, attention_mask, prefixes):
        context_lengths = attention_mask.sum(dim=1).tolist()
        prefix_len = prefixes.size(1)
        max_length = max(context_lengths) + prefix_len
        combined = input_ids.new_zeros((input_ids.size(0), max_length))
        combined_mask = attention_mask.new_zeros(
            (input_ids.size(0), max_length)
        )
        for row, context_length in enumerate(context_lengths):
            context_length = int(context_length)
            combined[row, :context_length] = input_ids[row, :context_length]
            combined_mask[row, :context_length] = 1
            if prefix_len:
                combined[
                    row, context_length : context_length + prefix_len
                ] = prefixes[row]
                combined_mask[
                    row, context_length : context_length + prefix_len
                ] = 1
        return combined, combined_mask

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        prefix_allowed_tokens_fn,
        num_beams=20,
    ):
        """Generate one complete SID with trie-constrained beam search."""
        if prefix_allowed_tokens_fn is None:
            raise ValueError("prefix_allowed_tokens_fn is required")
        batch_size = input_ids.size(0)
        prefixes = input_ids.new_empty((batch_size, 1, 0))
        beam_scores = torch.zeros(
            batch_size, 1, device=input_ids.device, dtype=torch.float32
        )

        for step in range(self.code_length):
            beam_count = prefixes.size(1)
            repeated_ids = input_ids.repeat_interleave(beam_count, dim=0)
            repeated_mask = attention_mask.repeat_interleave(
                beam_count, dim=0
            )
            flat_prefixes = prefixes.reshape(
                batch_size * beam_count, step
            )
            model_ids, model_mask = self._append_prefixes(
                repeated_ids, repeated_mask, flat_prefixes
            )
            log_probs = F.log_softmax(
                self.next_token_logits(model_ids, model_mask).float(), dim=-1
            )
            allowed_mask = torch.zeros_like(log_probs, dtype=torch.bool)
            for row, prefix in enumerate(flat_prefixes.tolist()):
                allowed = prefix_allowed_tokens_fn(
                    row // beam_count, flat_prefixes[row]
                )
                if not allowed:
                    raise RuntimeError(f"No valid SID continuation for {prefix}")
                allowed_mask[row, allowed] = True
            log_probs = log_probs.masked_fill(~allowed_mask, -torch.inf)
            candidate_scores = (
                beam_scores.reshape(-1, 1) + log_probs
            ).view(batch_size, -1)

            valid_expansions = allowed_mask.view(
                batch_size, beam_count, self.vocab_size
            ).sum(dim=(1, 2))
            next_beam_count = min(
                num_beams, int(valid_expansions.min().item())
            )
            beam_scores, flat_indices = torch.topk(
                candidate_scores, k=next_beam_count, dim=-1
            )
            parent_indices = flat_indices // self.vocab_size
            next_tokens = flat_indices % self.vocab_size
            if step:
                parent_prefixes = torch.gather(
                    prefixes,
                    1,
                    parent_indices.unsqueeze(-1).expand(-1, -1, step),
                )
            else:
                parent_prefixes = prefixes.expand(
                    -1, next_beam_count, -1
                )
            prefixes = torch.cat(
                [parent_prefixes, next_tokens.unsqueeze(-1)], dim=-1
            )

        return prefixes, beam_scores
