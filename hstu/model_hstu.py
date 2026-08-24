import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSTUBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dqk, dv, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.dqk = dqk
        self.dv = dv
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.uvqk = nn.Linear(embedding_dim, num_heads * (2 * dv + 2 * dqk), bias=False)
        self.out = nn.Linear(num_heads * dv, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None):
        bsz, seq_len, _ = x.shape
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
        u = u.view(bsz, seq_len, self.num_heads, self.dv)
        v = v.view(bsz, seq_len, self.num_heads, self.dv)
        q = q.view(bsz, seq_len, self.num_heads, self.dqk)
        k = k.view(bsz, seq_len, self.num_heads, self.dqk)

        scores = torch.einsum("bnhd,bmhd->bhnm", q, k)
        scores = F.silu(scores) / seq_len
        causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=scores.dtype))
        scores = scores * causal.view(1, 1, seq_len, seq_len)
        if attention_mask is not None:
            scores = scores * attention_mask[:, None, None, :].to(scores.dtype)

        attn_out = torch.einsum("bhnm,bmhd->bnhd", scores, v)
        attn_out = F.layer_norm(
            attn_out.reshape(bsz, seq_len, self.num_heads * self.dv),
            [self.num_heads * self.dv],
            eps=1e-6,
        )
        gated = u.reshape(bsz, seq_len, self.num_heads * self.dv) * attn_out
        return x + self.out(self.dropout(gated))


class HSTURec(nn.Module):
    def __init__(self, item_num, max_len=200, embedding_dim=50, num_blocks=2, num_heads=1, dqk=50, dv=50, dropout_rate=0.2):
        super().__init__()
        self.item_num = item_num
        self.item_emb = nn.Embedding(item_num + 1, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([
            HSTUBlock(embedding_dim, num_heads, dqk, dv, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6)

    @property
    def n_parameters(self):
        return f"#Total params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"

    def get_item_embeddings(self, ids):
        return self.item_emb(ids)

    def encode(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        pos = torch.arange(1, seq_len + 1, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        pos = pos * (input_ids != 0)
        x = self.item_emb(input_ids) * math.sqrt(self.item_emb.embedding_dim) + self.pos_emb(pos)
        x = self.dropout(x)
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        x = x.masked_fill(attention_mask.unsqueeze(-1) == 0, 0.0)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
            x = x.masked_fill(attention_mask.unsqueeze(-1) == 0, 0.0)
        return self.norm(x)

    def score_all(self, input_ids, attention_mask=None):
        hidden = self.encode(input_ids, attention_mask)[:, -1, :]
        return hidden @ self.item_emb.weight[1:].t()
