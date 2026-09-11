import math
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class T5LayerNorm(nn.Module):
    """T5-style RMS layer norm without bias."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(self.weight.dtype)


class T5RelativePositionBias(nn.Module):
    """Relative position bias compatible with T5's parameter shape."""

    def __init__(
        self,
        num_buckets: int,
        max_distance: int,
        num_heads: int,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool,
        num_buckets: int,
        max_distance: int,
    ) -> torch.Tensor:
        relative_buckets = torch.zeros_like(relative_position, dtype=torch.long)
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position,
                torch.zeros_like(relative_position),
            )

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact + 1e-6)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small,
            relative_position,
            relative_position_if_large,
        )
        return relative_buckets

    def forward(
        self, query_length: int, key_length: int, device: torch.device
    ) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position=relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        return values.permute(2, 0, 1).unsqueeze(0)


class CausalMultiHeadAttention(nn.Module):
    """Hand-written multi-head self-attention with a causal mask."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_kv: int,
        dropout_rate: float,
        has_relative_attention_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        attention_window: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = num_heads * d_kv
        self.num_heads = num_heads
        self.d_kv = d_kv
        self.attention_window = attention_window
        self.q = nn.Linear(d_model, inner_dim, bias=False)
        self.k = nn.Linear(d_model, inner_dim, bias=False)
        self.v = nn.Linear(d_model, inner_dim, bias=False)
        self.o = nn.Linear(inner_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.position_bias = (
            T5RelativePositionBias(
                relative_attention_num_buckets,
                relative_attention_max_distance,
                num_heads,
                bidirectional=False,
            )
            if has_relative_attention_bias
            else None
        )

    def _shape(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        states = states.view(batch_size, seq_len, self.num_heads, self.d_kv)
        return states.transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self._shape(self.q(hidden_states))
        key_states = self._shape(self.k(hidden_states))
        value_states = self._shape(self.v(hidden_states))

        scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        if self.position_bias is not None:
            scores = scores + self.position_bias(seq_len, seq_len, hidden_states.device)

        causal_mask = torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=hidden_states.device,
        ).tril()
        if self.attention_window is not None:
            # Sliding-window causal mask: each query may attend to at most
            # `attention_window` most-recent tokens (itself included), so the
            # visible history stays bounded regardless of the full block length.
            positions = torch.arange(seq_len, device=hidden_states.device)
            within_window = (
                positions[:, None] - positions[None, :]
            ) < self.attention_window
            causal_mask = causal_mask & within_window
        scores = scores.masked_fill(
            ~causal_mask.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min
        )

        if attention_mask is not None:
            key_mask = attention_mask.to(torch.bool).view(batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.num_heads * self.d_kv,
            )
        )
        return self.o(attn_output)


class T5FeedForward(nn.Module):
    """T5 feed-forward block; relu mode matches the current main.py default."""

    def __init__(
        self, d_model: int, d_ff: int, dropout_rate: float, feed_forward_proj: str
    ):
        super().__init__()
        self.feed_forward_proj = feed_forward_proj
        self.dropout = nn.Dropout(dropout_rate)
        if feed_forward_proj.startswith("gated"):
            self.wi_0 = nn.Linear(d_model, d_ff, bias=False)
            self.wi_1 = nn.Linear(d_model, d_ff, bias=False)
        else:
            self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.feed_forward_proj.startswith("gated"):
            hidden_gelu = F.gelu(self.wi_0(hidden_states))
            hidden_linear = self.wi_1(hidden_states)
            hidden_states = hidden_gelu * hidden_linear
        else:
            hidden_states = F.relu(self.wi(hidden_states))
        hidden_states = self.dropout(hidden_states)
        return self.wo(hidden_states)


class CausalBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        d_kv: int,
        dropout_rate: float,
        feed_forward_proj: str,
        has_relative_attention_bias: bool = False,
        extra_attention: bool = False,
        attention_window: Optional[int] = None,
    ):
        super().__init__()
        self.self_attn_norm = T5LayerNorm(d_model)
        self.self_attn = CausalMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            dropout_rate=dropout_rate,
            has_relative_attention_bias=has_relative_attention_bias,
            attention_window=attention_window,
        )
        self.extra_attention = extra_attention
        if extra_attention:
            self.extra_attn_norm = T5LayerNorm(d_model)
            self.extra_attn = CausalMultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                d_kv=d_kv,
                dropout_rate=dropout_rate,
                has_relative_attention_bias=False,
                attention_window=attention_window,
            )
        self.ffn_norm = T5LayerNorm(d_model)
        self.ffn = T5FeedForward(d_model, d_ff, dropout_rate, feed_forward_proj)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_output = self.self_attn(self.self_attn_norm(hidden_states), attention_mask)
        hidden_states = hidden_states + self.dropout(attn_output)
        if self.extra_attention:
            extra_output = self.extra_attn(
                self.extra_attn_norm(hidden_states), attention_mask
            )
            hidden_states = hidden_states + self.dropout(extra_output)
        ffn_output = self.ffn(self.ffn_norm(hidden_states))
        return hidden_states + self.dropout(ffn_output)


class CausalTIGER(nn.Module):
    """
    Decoder-only TIGER with hand-written causal MHA.

    The module keeps the same trainable parameter count as the T5 model in
    model/main.py by preserving T5's shared embedding, block widths, FFN shapes,
    relative attention-bias tensors, and the decoder block's second attention
    sublayer. The behavioral difference is that every attention sublayer uses a
    causal mask.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.d_model = config["d_model"]
        self.pad_token_id = config["pad_token_id"]
        self.eos_token_id = config["eos_token_id"]
        self.decoder_start_token_id = config.get(
            "decoder_start_token_id", self.pad_token_id
        )
        self.item_emb_dim = config.get("item_emb_dim", 0)
        self.mse_loss_weight = config.get("mse_loss_weight", 0.0)
        self.mse_loss_mode = config.get("mse_loss_mode", "token")
        self.align_loss_type = config.get("align_loss_type", "mse")
        self.align_target = config.get("align_target", "item")
        self.align_item = config.get("align_item", "next")
        if self.align_target == "shallow" and self.item_emb_dim <= 0:
            self.item_emb_dim = self.d_model
        self.shallow_layer = config.get("shallow_layer", 1)
        self.lm_head_type = config.get("lm_head", "emb")
        if self.mse_loss_mode not in {"token", "mean"}:
            raise ValueError("mse_loss_mode must be 'token' or 'mean'")
        if self.align_item not in {"pre", "next", "near"}:
            raise ValueError("align_item must be 'pre', 'next' or 'near'")
        if self.align_item == "near" and self.align_target != "shallow":
            raise ValueError("align_item='near' is only supported for shallow align")
        if self.align_loss_type not in {"mse", "cos"}:
            raise ValueError("align_loss_type must be 'mse' or 'cos'")
        if self.lm_head_type not in {"emb", "linear", "mlp", "mlp-emb"}:
            raise ValueError("lm_head must be 'emb', 'linear', 'mlp' or 'mlp-emb'")
        if self.shallow_layer < 1 or self.shallow_layer > config["num_layers"]:
            raise ValueError(f"shallow_layer must be in [1, {config['num_layers']}]")
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.attention_window = config.get("attention_window", None)
        self.shared = nn.Embedding(config["vocab_size"], config["d_model"])
        # For 'mlp-emb', hidden states pass through a dimension-preserving MLP
        # before being multiplied with the (tied) embedding table, like 'emb'.
        self.lm_head_mlp = None
        if self.lm_head_type == "linear":
            self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        elif self.lm_head_type == "mlp":
            self.lm_head = nn.Sequential(
                nn.Linear(config["d_model"], config["d_model"]),
                nn.GELU(),
                nn.Linear(config["d_model"], config["vocab_size"], bias=False),
            )
        else:
            self.lm_head = None
            if self.lm_head_type == "mlp-emb":
                self.lm_head_mlp = nn.Sequential(
                    nn.Linear(config["d_model"], config["d_model"]),
                    nn.GELU(),
                    nn.Linear(config["d_model"], config["d_model"]),
                )
        self.hidden_to_item_emb = (
            nn.Sequential(
                nn.Linear(config["d_model"], config["d_model"]),
                nn.GELU(),
                nn.Linear(config["d_model"], self.item_emb_dim, bias=False),
            )
            if self.item_emb_dim > 0
            else None
        )
        self.sid_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config["d_model"], config["d_model"]),
                    nn.GELU(),
                    nn.Linear(config["d_model"], config["vocab_size"], bias=False),
                )
                for _ in range(4)
            ]
        )

        self.context_blocks = nn.ModuleList(
            [
                CausalBlock(
                    d_model=config["d_model"],
                    d_ff=config["d_ff"],
                    num_heads=config["num_heads"],
                    d_kv=config["d_kv"],
                    dropout_rate=config["dropout_rate"],
                    feed_forward_proj=config["feed_forward_proj"],
                    has_relative_attention_bias=(layer_idx == 0),
                    extra_attention=False,
                    attention_window=self.attention_window,
                )
                for layer_idx in range(config["num_layers"])
            ]
        )
        self.context_final_layer_norm = T5LayerNorm(config["d_model"])
        self.decoder_blocks = nn.ModuleList(
            [
                CausalBlock(
                    d_model=config["d_model"],
                    d_ff=config["d_ff"],
                    num_heads=config["num_heads"],
                    d_kv=config["d_kv"],
                    dropout_rate=config["dropout_rate"],
                    feed_forward_proj=config["feed_forward_proj"],
                    has_relative_attention_bias=(layer_idx == 0),
                    extra_attention=True,
                    attention_window=self.attention_window,
                )
                for layer_idx in range(config["num_decoder_layers"])
            ]
        )
        self.decoder_final_layer_norm = T5LayerNorm(config["d_model"])

    @property
    def n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.get_input_embeddings().parameters())
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.shared

    def _encode_tokens(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_shallow_layer: Optional[int] = None,
    ) -> torch.Tensor:
        hidden_states = self.dropout(self.shared(token_ids))
        shallow_states = None
        for layer_idx, block in enumerate(self.context_blocks, start=1):
            hidden_states = block(hidden_states, attention_mask)
            if return_shallow_layer == layer_idx:
                shallow_states = hidden_states
        hidden_states = self.dropout(self.context_final_layer_norm(hidden_states))
        for block in self.decoder_blocks:
            hidden_states = block(hidden_states, attention_mask)
            hidden_states = self.dropout(self.decoder_final_layer_norm(hidden_states))
        if return_shallow_layer is not None:
            if shallow_states is None:
                raise ValueError(f"Unsupported shallow_layer: {return_shallow_layer}")
            return hidden_states, shallow_states
        return hidden_states

    def _mean_item_embeddings(self, item_code_ids: torch.Tensor) -> torch.Tensor:
        # item_code_ids: [B, items, 4]. Padding items are masked outside this
        # function; averaging their pad embeddings is harmless.
        return self.shared(item_code_ids).mean(dim=2)

    def _encode_item_codes(
        self,
        item_code_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_shallow_layer: Optional[int] = None,
    ) -> torch.Tensor:
        hidden_states = self.dropout(self._mean_item_embeddings(item_code_ids))
        shallow_states = None
        for layer_idx, block in enumerate(self.context_blocks, start=1):
            hidden_states = block(hidden_states, attention_mask)
            if return_shallow_layer == layer_idx:
                shallow_states = hidden_states
        hidden_states = self.dropout(self.context_final_layer_norm(hidden_states))
        for block in self.decoder_blocks:
            hidden_states = block(hidden_states, attention_mask)
            hidden_states = self.dropout(self.decoder_final_layer_norm(hidden_states))
        if return_shallow_layer is not None:
            if shallow_states is None:
                raise ValueError(f"Unsupported shallow_layer: {return_shallow_layer}")
            return hidden_states, shallow_states
        return hidden_states

    def _sid_logits(self, item_hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [head(item_hidden_states) for head in self.sid_heads],
            dim=-2,
        )

    def _last_item_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1, :]
        lengths = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        gather_index = lengths.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
        return hidden_states.gather(1, gather_index).squeeze(1)

    def _lm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        if self.lm_head_mlp is not None:
            hidden_states = self.lm_head_mlp(hidden_states)
        hidden_states = hidden_states * (self.d_model**-0.5)
        return torch.matmul(hidden_states, self.shared.weight.transpose(0, 1))

    def _parallel_align_loss(
        self,
        hidden_states: torch.Tensor,
        target_item_emb: torch.Tensor,
        item_group: torch.Tensor,
        code_phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Per-item mean-pool alignment for parallel training. Every supervised
        # position carries the block-item index it predicts (item_group >= 0);
        # ignored positions are -1. We average the hidden states of the code
        # positions belonging to the same predicted item, project them through
        # hidden_to_item_emb, and align with that item's target embedding
        # (target_item_emb is [B, max_items, dim], or [B, max_items, 4, dim]
        # for codebook targets).
        batch_size, _, d_model = hidden_states.shape
        max_items = target_item_emb.size(1)
        device = hidden_states.device

        if item_group.dim() == 2:
            valid = item_group >= 0
            if not valid.any():
                return hidden_states.new_zeros(())
            batch_idx = (
                torch.arange(batch_size, device=device)
                .unsqueeze(1)
                .expand_as(item_group)
            )
            flat_slot = (
                batch_idx * max_items + item_group.clamp_min(0)
            ).reshape(-1)[valid.reshape(-1)]
            flat_hidden = hidden_states.reshape(-1, d_model)[valid.reshape(-1)]
            pred_item_emb = self.hidden_to_item_emb(flat_hidden)
            if self.align_target == "codebook":
                target_emb = target_item_emb.reshape(
                    batch_size * max_items,
                    target_item_emb.size(-2),
                    target_item_emb.size(-1),
                )[flat_slot, :3].mean(dim=1)
            else:
                target_emb = target_item_emb.reshape(batch_size * max_items, -1)[
                    flat_slot
                ]
            target_emb = target_emb.to(pred_item_emb.dtype)
            if self.align_loss_type == "cos":
                return 1.0 - F.cosine_similarity(
                    pred_item_emb, target_emb, dim=-1
                ).mean()
            return F.mse_loss(pred_item_emb, target_emb, reduction="mean")

        valid = item_group >= 0  # [B, seq]
        if self.align_target == "codebook":
            if code_phase is None:
                raise ValueError("code_phase is required for parallel codebook align")
            # The fourth RQ layer has no codebook vector, so only align target
            # code phases 1/2/3.
            valid = valid & (code_phase >= 1) & (code_phase <= 3)
        if not valid.any():
            return hidden_states.new_zeros(())

        # Unique slot id per (batch, item) pair so we can scatter-mean.
        num_slots = batch_size * max_items
        batch_idx = (
            torch.arange(batch_size, device=device)
            .unsqueeze(1)
            .expand_as(item_group)
        )
        slot = batch_idx * max_items + item_group.clamp_min(0)  # [B, seq]
        flat_slot = slot.reshape(-1)[valid.reshape(-1)]
        flat_hidden = hidden_states.reshape(-1, d_model)[valid.reshape(-1)]
        if self.align_target == "codebook":
            flat_phase = code_phase.reshape(-1)[valid.reshape(-1)] - 1
            flat_target_emb = target_item_emb.reshape(
                num_slots, target_item_emb.size(-2), target_item_emb.size(-1)
            )[flat_slot, flat_phase].to(flat_hidden.dtype)
        else:
            flat_target_emb = target_item_emb.reshape(num_slots, -1)[flat_slot].to(
                flat_hidden.dtype
            )

        if self.mse_loss_mode == "token":
            pred_item_emb = self.hidden_to_item_emb(flat_hidden)
            if self.align_loss_type == "cos":
                return 1.0 - F.cosine_similarity(
                    pred_item_emb, flat_target_emb, dim=-1
                ).mean()
            return F.mse_loss(pred_item_emb, flat_target_emb, reduction="mean")

        sum_hidden = hidden_states.new_zeros((num_slots, d_model))
        sum_hidden.index_add_(0, flat_slot, flat_hidden)
        counts = hidden_states.new_zeros((num_slots,))
        counts.index_add_(0, flat_slot, torch.ones_like(flat_slot, dtype=counts.dtype))

        active = counts > 0
        pooled_hidden = sum_hidden[active] / counts[active].unsqueeze(-1)
        pred_item_emb = self.hidden_to_item_emb(pooled_hidden)
        if self.align_target == "codebook":
            target_emb = target_item_emb.reshape(
                num_slots, target_item_emb.size(-2), target_item_emb.size(-1)
            )[active, :3].mean(dim=1).to(pred_item_emb.dtype)
        else:
            target_emb = target_item_emb.reshape(num_slots, -1)[active].to(
                pred_item_emb.dtype
            )

        if self.align_loss_type == "cos":
            return 1.0 - F.cosine_similarity(pred_item_emb, target_emb, dim=-1).mean()
        return F.mse_loss(pred_item_emb, target_emb, reduction="mean")

    def _owner_group_and_phase(
        self,
        attention_mask: torch.Tensor,
        code_per_item: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_index = attention_mask.to(torch.long).cumsum(dim=1) - 1
        owner_group = token_index // code_per_item
        owner_phase = (token_index % code_per_item) + 1
        owner_group = owner_group.masked_fill(attention_mask == 0, -1)
        owner_phase = owner_phase.masked_fill(attention_mask == 0, 0)
        return owner_group, owner_phase

    def _parallel_shallow_align_loss(
        self,
        hidden_states: torch.Tensor,
        shallow_states: torch.Tensor,
        item_group: torch.Tensor,
        code_phase: torch.Tensor,
        full_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, d_model = hidden_states.shape
        device = hidden_states.device
        if self.align_item == "near":
            valid_near = (full_attention_mask[:, :-1] > 0) & (
                full_attention_mask[:, 1:] > 0
            )
            if not valid_near.any():
                return hidden_states.new_zeros(())
            flat_hidden = hidden_states[valid_near]
            target_hidden = shallow_states[:, 1:][valid_near].detach()
            pred_hidden = self.hidden_to_item_emb(flat_hidden)
            if self.align_loss_type == "cos":
                return 1.0 - F.cosine_similarity(
                    pred_hidden, target_hidden.to(pred_hidden.dtype), dim=-1
                ).mean()
            return F.mse_loss(
                pred_hidden, target_hidden.to(pred_hidden.dtype), reduction="mean"
            )
        owner_group, owner_phase = self._owner_group_and_phase(full_attention_mask)
        full_item_group = F.pad(item_group, (0, 1), value=-1)
        full_code_phase = F.pad(code_phase, (0, 1), value=0)
        if self.align_item == "pre":
            teacher_group = full_item_group
            teacher_phase = full_code_phase
        else:
            teacher_group = owner_group
            teacher_phase = owner_phase
        max_group = torch.cat(
            [
                item_group[item_group >= 0],
                teacher_group[teacher_group >= 0],
            ]
        )
        if max_group.numel() == 0:
            return hidden_states.new_zeros(())
        max_items = int(max_group.max().item()) + 1
        num_slots = batch_size * max_items

        batch_idx = torch.arange(batch_size, device=device)
        student_batch = batch_idx.unsqueeze(1).expand_as(item_group)
        valid_student = (item_group >= 0) & (code_phase >= 1) & (code_phase <= 4)
        next_group = item_group + 1
        valid_student = valid_student & (next_group < max_items)
        if not valid_student.any():
            return hidden_states.new_zeros(())

        if self.mse_loss_mode == "token":
            teacher_batch = batch_idx.unsqueeze(1).expand_as(teacher_group)
            valid_teacher = (
                (teacher_group >= 0) & (teacher_phase >= 1) & (teacher_phase <= 4)
            )
            teacher_key = (
                teacher_batch * max_items * 4
                + teacher_group.clamp_min(0) * 4
                + (teacher_phase.clamp_min(1) - 1)
            )
            teacher_values = shallow_states.reshape(-1, d_model)[valid_teacher.reshape(-1)]
            flat_teacher_key = teacher_key.reshape(-1)[valid_teacher.reshape(-1)]
            num_keys = num_slots * 4
            sum_teacher = shallow_states.new_zeros((num_keys, d_model))
            sum_teacher.index_add_(0, flat_teacher_key, teacher_values)
            teacher_counts = shallow_states.new_zeros((num_keys,))
            teacher_counts.index_add_(
                0,
                flat_teacher_key,
                torch.ones_like(flat_teacher_key, dtype=teacher_counts.dtype),
            )

            student_key = (
                student_batch * max_items * 4
                + next_group.clamp_min(0) * 4
                + (code_phase.clamp_min(1) - 1)
            )
            flat_student_key = student_key.reshape(-1)[valid_student.reshape(-1)]
            flat_hidden = hidden_states.reshape(-1, d_model)[valid_student.reshape(-1)]
            has_teacher = teacher_counts[flat_student_key] > 0
            if not has_teacher.any():
                return hidden_states.new_zeros(())
            flat_hidden = flat_hidden[has_teacher]
            target_hidden = (
                sum_teacher[flat_student_key[has_teacher]]
                / teacher_counts[flat_student_key[has_teacher]].unsqueeze(-1)
            ).detach()
            pred_hidden = self.hidden_to_item_emb(flat_hidden)
            if self.align_loss_type == "cos":
                return 1.0 - F.cosine_similarity(
                    pred_hidden, target_hidden.to(pred_hidden.dtype), dim=-1
                ).mean()
            return F.mse_loss(pred_hidden, target_hidden.to(pred_hidden.dtype), reduction="mean")

        student_slot = student_batch * max_items + item_group.clamp_min(0)
        flat_student_slot = student_slot.reshape(-1)[valid_student.reshape(-1)]
        flat_hidden = hidden_states.reshape(-1, d_model)[valid_student.reshape(-1)]
        sum_student = hidden_states.new_zeros((num_slots, d_model))
        sum_student.index_add_(0, flat_student_slot, flat_hidden)
        student_counts = hidden_states.new_zeros((num_slots,))
        student_counts.index_add_(
            0,
            flat_student_slot,
            torch.ones_like(flat_student_slot, dtype=student_counts.dtype),
        )

        teacher_batch = batch_idx.unsqueeze(1).expand_as(teacher_group)
        valid_teacher = teacher_group >= 0
        teacher_slot = teacher_batch * max_items + teacher_group.clamp_min(0)
        flat_teacher_slot = teacher_slot.reshape(-1)[valid_teacher.reshape(-1)]
        flat_teacher = shallow_states.reshape(-1, d_model)[valid_teacher.reshape(-1)]
        sum_teacher = shallow_states.new_zeros((num_slots, d_model))
        sum_teacher.index_add_(0, flat_teacher_slot, flat_teacher)
        teacher_counts = shallow_states.new_zeros((num_slots,))
        teacher_counts.index_add_(
            0,
            flat_teacher_slot,
            torch.ones_like(flat_teacher_slot, dtype=teacher_counts.dtype),
        )

        active_slots = torch.nonzero(student_counts > 0, as_tuple=False).flatten()
        active_group = active_slots % max_items
        active_batch = active_slots // max_items
        has_next_group = active_group + 1 < max_items
        target_slots = active_batch * max_items + active_group + 1
        safe_target_slots = target_slots.clamp_max(num_slots - 1)
        valid_pair = has_next_group & (teacher_counts[safe_target_slots] > 0)
        if not valid_pair.any():
            return hidden_states.new_zeros(())
        active_slots = active_slots[valid_pair]
        target_slots = target_slots[valid_pair]
        pooled_student = sum_student[active_slots] / student_counts[active_slots].unsqueeze(-1)
        pooled_teacher = (
            sum_teacher[target_slots] / teacher_counts[target_slots].unsqueeze(-1)
        ).detach()
        pred_hidden = self.hidden_to_item_emb(pooled_student)
        if self.align_loss_type == "cos":
            return 1.0 - F.cosine_similarity(
                pred_hidden, pooled_teacher.to(pred_hidden.dtype), dim=-1
            ).mean()
        return F.mse_loss(pred_hidden, pooled_teacher.to(pred_hidden.dtype), reduction="mean")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        target_item_emb: Optional[torch.Tensor] = None,
        item_group: Optional[torch.Tensor] = None,
        code_phase: Optional[torch.Tensor] = None,
    ):
        if labels is None:
            if attention_mask is None:
                attention_mask = (input_ids != self.pad_token_id).any(dim=-1).long()
            hidden_states = self._encode_item_codes(input_ids, attention_mask)
            return None, self._sid_logits(hidden_states)

        if input_ids.dim() == 3:
            if attention_mask is None:
                attention_mask = (input_ids != self.pad_token_id).any(dim=-1).long()
            shallow_states = None
            if self.align_target == "shallow" and self.mse_loss_weight > 0:
                hidden_states, shallow_states = self._encode_item_codes(
                    input_ids, attention_mask, return_shallow_layer=self.shallow_layer
                )
            else:
                hidden_states = self._encode_item_codes(input_ids, attention_mask)
            logits = self._sid_logits(hidden_states)
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            self.last_target_hidden_states = (
                hidden_states.unsqueeze(2)
                .expand(-1, -1, 4, -1)
                .reshape(hidden_states.size(0), -1, hidden_states.size(-1))
                .detach()
            )
            align_loss = torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
            if (
                self.align_target == "shallow"
                and shallow_states is not None
                and item_group is not None
                and self.hidden_to_item_emb is not None
                and self.mse_loss_weight > 0
            ):
                valid = item_group >= 0
                if valid.any():
                    pred_hidden = self.hidden_to_item_emb(hidden_states[valid])
                    target_hidden = shallow_states[valid].detach()
                    if self.align_loss_type == "cos":
                        align_loss = 1.0 - F.cosine_similarity(
                            pred_hidden, target_hidden.to(pred_hidden.dtype), dim=-1
                        ).mean()
                    else:
                        align_loss = F.mse_loss(
                            pred_hidden,
                            target_hidden.to(pred_hidden.dtype),
                            reduction="mean",
                        )
            elif (
                target_item_emb is not None
                and item_group is not None
                and self.hidden_to_item_emb is not None
                and self.mse_loss_weight > 0
            ):
                align_loss = self._parallel_align_loss(
                    hidden_states, target_item_emb, item_group, code_phase
                )
            loss = ce_loss + self.mse_loss_weight * align_loss
            self.last_loss_dict = {
                "total": loss.detach(),
                "ce": ce_loss.detach(),
                "align": align_loss.detach(),
            }
            return loss, logits

        batch_size = input_ids.size(0)
        item_input_ids = input_ids.view(batch_size, -1, 4)
        if attention_mask is None:
            item_attention_mask = (item_input_ids != self.pad_token_id).any(
                dim=-1
            ).long()
        else:
            item_attention_mask = attention_mask.view(batch_size, -1, 4).any(
                dim=-1
            ).long()
        hidden_states = self._encode_item_codes(item_input_ids, item_attention_mask)
        target_hidden = self._last_item_hidden(hidden_states, item_attention_mask)
        self.last_target_hidden_states = (
            target_hidden.unsqueeze(1).expand(-1, labels.size(1), -1).detach()
        )
        logits = self._sid_logits(target_hidden).view(
            batch_size, labels.size(1), self.vocab_size
        )
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        align_loss = torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
        if (
            target_item_emb is not None
            and self.hidden_to_item_emb is not None
            and self.mse_loss_weight > 0
        ):
            pred_item_emb = self.hidden_to_item_emb(target_hidden)
            if target_item_emb.dim() == 3:
                valid_emb = target_item_emb.abs().sum(dim=-1).gt(0)
                denom = valid_emb.sum(dim=1, keepdim=True).clamp_min(1)
                target_emb = (
                    target_item_emb * valid_emb.unsqueeze(-1)
                ).sum(dim=1) / denom
                target_emb = target_emb.to(pred_item_emb.dtype)
            else:
                target_emb = target_item_emb.to(pred_item_emb.dtype)
            if self.align_loss_type == "cos":
                align_loss = 1.0 - F.cosine_similarity(
                    pred_item_emb, target_emb, dim=-1
                ).mean()
            else:
                align_loss = F.mse_loss(pred_item_emb, target_emb, reduction="mean")
        loss = ce_loss + self.mse_loss_weight * align_loss
        self.last_loss_dict = {
            "total": loss.detach(),
            "ce": ce_loss.detach(),
            "align": align_loss.detach(),
        }
        return loss, logits

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 5,
        num_beams: int = 20,
        num_return_sequences: Optional[int] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if num_return_sequences is None:
            num_return_sequences = num_beams
        if num_return_sequences > num_beams:
            raise ValueError("num_return_sequences must be <= num_beams")

        batch_size = input_ids.size(0)
        vocab_size = self.vocab_size
        device = input_ids.device
        item_input_ids = input_ids.view(batch_size, -1, 4)
        if attention_mask is None:
            item_attention_mask = (item_input_ids != self.pad_token_id).any(
                dim=-1
            ).long()
        else:
            item_attention_mask = attention_mask.view(batch_size, -1, 4).any(
                dim=-1
            ).long()
        hidden_states = self._encode_item_codes(item_input_ids, item_attention_mask)
        last_hidden = self._last_item_hidden(hidden_states, item_attention_mask)
        step_log_probs = F.log_softmax(self._sid_logits(last_hidden).float(), dim=-1)
        max_length = min(max_length, step_log_probs.size(1))
        all_outputs = []
        for batch_idx in range(batch_size):
            beams = [([], step_log_probs.new_tensor(0.0))]
            for step in range(max_length):
                candidates = []
                for prefix, score in beams:
                    if prefix_allowed_tokens_fn is None:
                        token_scores, token_ids = torch.topk(
                            step_log_probs[batch_idx, step], num_beams
                        )
                        allowed_pairs = zip(token_ids.tolist(), token_scores)
                    else:
                        prefix_tensor = torch.tensor(
                            prefix, dtype=input_ids.dtype, device=device
                        )
                        allowed_tokens = prefix_allowed_tokens_fn(
                            batch_idx, prefix_tensor
                        )
                        if not allowed_tokens:
                            continue
                        token_scores = step_log_probs[
                            batch_idx,
                            step,
                            torch.tensor(allowed_tokens, dtype=torch.long, device=device),
                        ]
                        allowed_pairs = zip(allowed_tokens, token_scores)
                    for token_id, token_score in allowed_pairs:
                        candidates.append((prefix + [int(token_id)], score + token_score))
                if not candidates:
                    candidates = beams
                candidates.sort(key=lambda x: float(x[1]), reverse=True)
                beams = candidates[:num_beams]
            while len(beams) < num_return_sequences:
                beams.append(beams[-1])
            all_outputs.extend([seq for seq, _ in beams[:num_return_sequences]])
        return torch.tensor(all_outputs, dtype=input_ids.dtype, device=device)
