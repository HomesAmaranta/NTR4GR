import torch
import torch.nn as nn
import torch.nn.functional as F


class SampledSoftmaxLoss(nn.Module):
    def __init__(self, num_negatives=128, temperature=0.05, item_l2_norm=True, eps=1e-6):
        super().__init__()
        self.num_negatives = num_negatives
        self.temperature = temperature
        self.item_l2_norm = item_l2_norm
        self.eps = eps

    def _norm(self, x):
        if not self.item_l2_norm:
            return x
        return x / torch.clamp(torch.linalg.norm(x, ord=2, dim=-1, keepdim=True), min=self.eps)

    def forward(self, output_embeddings, supervision_ids, supervision_weights, item_emb):
        valid = supervision_weights > 0
        if valid.sum() == 0:
            return output_embeddings.sum() * 0.0
        query = output_embeddings[valid]
        pos_ids = supervision_ids[valid]
        pos_emb = self._norm(item_emb(pos_ids))
        sampled_ids = torch.randint(
            low=1,
            high=item_emb.num_embeddings,
            size=(query.size(0), self.num_negatives),
            dtype=pos_ids.dtype,
            device=pos_ids.device,
        )
        neg_emb = self._norm(item_emb(sampled_ids))
        pos_logits = (query * pos_emb).sum(dim=-1, keepdim=True) / self.temperature
        neg_logits = torch.einsum("bd,bnd->bn", query, neg_emb) / self.temperature
        neg_logits = torch.where(sampled_ids == pos_ids.unsqueeze(1), -5e4, neg_logits)
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        return F.cross_entropy(logits, labels)
