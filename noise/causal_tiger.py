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
        self.nexcur_current_weight = float(config.get("nexcur_current_weight", 0.5))
        self.code_per_item = 4
        self.hidden_layer = config.get("hidden_layer", -1)
        self.embedding_noise_mode = config.get("embedding_noise_mode", "add")
        self.embedding_noise_mode = str(self.embedding_noise_mode).lower()
        self.embedding_noise_std = config.get("embedding_noise_std", 0.0)
        self.embedding_noise_prob = config.get("embedding_noise_prob", 1.0)
        self.eval_embedding_noise = config.get("eval_embedding_noise", False)
        self.align_item_emb_to_hidden_mode = config.get(
            "align_item_emb_to_hidden_mode",
            "add" if config.get("add_align_item_emb_to_hidden", False) else "none",
        )
        if isinstance(self.align_item_emb_to_hidden_mode, bool):
            self.align_item_emb_to_hidden_mode = (
                "add" if self.align_item_emb_to_hidden_mode else "none"
            )
        self.align_item_emb_to_hidden_mode = str(
            self.align_item_emb_to_hidden_mode
        ).lower()
        self.add_align_item_emb_to_hidden = (
            self.align_item_emb_to_hidden_mode != "none"
        )
        if self.hidden_layer != -1 and self.mse_loss_weight > 0:
            self.mse_loss_mode = "only-hidden"
        if self.align_target in {"shallow", "vocab"} and self.item_emb_dim <= 0:
            self.item_emb_dim = self.d_model
        self.shallow_layer = config.get("shallow_layer", 1)
        self.lm_head_type = config.get("lm_head", "emb")
        if self.mse_loss_mode not in {"token", "mean", "mean-bar", "only-hidden", "pre-first"}:
            raise ValueError(
                "mse_loss_mode must be 'token', 'mean', 'mean-bar', 'only-hidden' or 'pre-first'"
            )
        if self.mse_loss_mode == "mean-bar" and self.align_target != "shallow":
            raise ValueError("mse_loss_mode='mean-bar' only supports shallow align")
        if self.mse_loss_mode == "only-hidden" and self.align_target == "codebook":
            raise ValueError("mse_loss_mode='only-hidden' does not support codebook align")
        if self.align_item == "pre":
            self.align_item = "current"
        if self.align_item not in {"current", "next", "near", "nexcur"}:
            raise ValueError("align_item must be 'current', 'next', 'near' or 'nexcur'")
        if not 0.0 <= self.nexcur_current_weight <= 1.0:
            raise ValueError("nexcur_current_weight must be in [0, 1]")
        if self.align_item == "near" and self.align_target != "shallow":
            raise ValueError("align_item='near' is only supported for shallow align")
        if self.align_item == "nexcur" and self.align_target in {"shallow", "vocab"}:
            raise ValueError("align_item='nexcur' only supports external item embedding targets")
        if self.align_target == "vocab" and self.align_item != "next":
            raise ValueError("align_target='vocab' requires align_item='next'")
        if self.mse_loss_mode == "pre-first" and self.align_item != "pre":
            raise ValueError("mse_loss_mode='pre-first' requires align_item='pre'")
        if self.align_loss_type not in {"mse", "cos"}:
            raise ValueError("align_loss_type must be 'mse' or 'cos'")
        if self.lm_head_type not in {"emb", "mlp"}:
            raise ValueError("lm_head must be 'emb' or 'mlp'")
        if self.embedding_noise_mode not in {
            "add",
            "fusion",
            "replace",
            "gaussian_replace",
        }:
            raise ValueError(
                "embedding_noise_mode must be 'add', 'fusion', 'replace', "
                "or 'gaussian_replace'"
            )
        if self.embedding_noise_std < 0.0:
            raise ValueError("embedding_noise_std must be non-negative")
        if self.embedding_noise_mode in {"fusion", "replace", "gaussian_replace"} and self.embedding_noise_std > 1.0:
            raise ValueError(
                "embedding_noise_std must be in [0, 1] for fusion/replace/gaussian_replace"
            )
        if self.embedding_noise_prob < 0.0 or self.embedding_noise_prob > 1.0:
            raise ValueError("embedding_noise_prob must be in [0, 1]")
        if self.align_item_emb_to_hidden_mode not in {"none", "add", "concat"}:
            raise ValueError(
                "align_item_emb_to_hidden_mode must be 'none', 'add' or 'concat'"
            )
        if self.shallow_layer < 1 or self.shallow_layer > config["num_layers"]:
            raise ValueError(f"shallow_layer must be in [1, {config['num_layers']}]")
        self.selected_hidden_layer = self._resolve_hidden_layer(
            self.hidden_layer,
            config["num_layers"],
        )
        self.dropout = nn.Dropout(config["dropout_rate"])
        # Sliding-window attention span in tokens (None = full causal). Used by
        # the parallel training mode to keep the visible history bounded.
        self.attention_window = config.get("attention_window", None)
        self.shared = nn.Embedding(config["vocab_size"], config["d_model"])
        if self.lm_head_type == "mlp":
            self.lm_head = nn.Sequential(
                nn.Linear(config["d_model"], config["d_model"]),
                nn.GELU(),
                nn.Linear(config["d_model"], config["vocab_size"], bias=False),
            )
        else:
            self.lm_head = None
        self.item_output_mlp = None
        self.hidden_to_item_emb = (
            nn.Sequential(
                nn.Linear(config["d_model"], config["d_model"]),
                nn.GELU(),
                nn.Linear(config["d_model"], self.item_emb_dim, bias=False),
            )
            if self.item_emb_dim > 0
            else None
        )
        self.align_item_emb_to_hidden = (
            nn.Linear(self.item_emb_dim, config["d_model"], bias=False)
            if self.add_align_item_emb_to_hidden and self.item_emb_dim > 0
            else None
        )
        self.align_hidden_concat_proj = (
            nn.Linear(config["d_model"] * 2, config["d_model"], bias=False)
            if self.align_item_emb_to_hidden_mode == "concat"
            else None
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

    def _resolve_hidden_layer(self, hidden_layer: int, num_layers: int) -> Optional[int]:
        if hidden_layer == -1:
            return None
        if hidden_layer == 0:
            raise ValueError("hidden_layer must be -1, a negative layer index, or a 1-based layer index")
        if hidden_layer > 0:
            resolved = hidden_layer
        else:
            resolved = num_layers + hidden_layer + 1
        if resolved < 1 or resolved > num_layers:
            raise ValueError(
                f"hidden_layer={hidden_layer} resolves to {resolved}, "
                f"but must be within [1, {num_layers}]"
            )
        return resolved

    def _add_embedding_noise(
        self,
        embeddings: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            not (self.training or self.eval_embedding_noise)
            or self.embedding_noise_std <= 0.0
        ):
            return embeddings

        position_shape = embeddings.shape[:-1]
        if self.embedding_noise_mode == "add":
            mask_prob = self.embedding_noise_prob
        elif self.embedding_noise_mode == "fusion":
            mask_prob = 1.0
        else:
            mask_prob = self.embedding_noise_std
        if mask_prob <= 0.0:
            return embeddings

        valid_position_mask = torch.ones(
            position_shape,
            device=embeddings.device,
            dtype=torch.bool,
        )
        if mask_prob >= 1.0:
            noise_mask = torch.ones(
                position_shape,
                device=embeddings.device,
                dtype=torch.bool,
            )
        else:
            noise_mask = torch.rand(position_shape, device=embeddings.device) < mask_prob
        if valid_mask is not None:
            valid_position_mask = valid_mask.to(
                device=embeddings.device,
                dtype=torch.bool,
            )
            noise_mask = noise_mask & valid_position_mask
        if not noise_mask.any():
            return embeddings

        if self.embedding_noise_mode == "add":
            noise = torch.randn_like(embeddings) * self.embedding_noise_std
            return embeddings + noise * noise_mask.unsqueeze(-1).to(embeddings.dtype)

        if self.embedding_noise_mode == "fusion":
            alpha = self.embedding_noise_std
            noise = torch.randn_like(embeddings)
            fused = (1.0 - alpha) * embeddings + alpha * noise
            return torch.where(noise_mask.unsqueeze(-1), fused, embeddings)

        if self.embedding_noise_mode == "gaussian_replace":
            noise = torch.randn_like(embeddings)
            return torch.where(noise_mask.unsqueeze(-1), noise, embeddings)

        return self._replace_embedding_positions(
            embeddings,
            noise_mask,
            valid_position_mask,
        )

    def _replace_embedding_positions(
        self,
        embeddings: torch.Tensor,
        noise_mask: torch.Tensor,
        valid_position_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_dim = embeddings.size(-1)
        flat_embeddings = embeddings.reshape(-1, hidden_dim)
        flat_noise_mask = noise_mask.reshape(-1)
        valid_indices = valid_position_mask.reshape(-1).nonzero(
            as_tuple=False
        ).squeeze(-1)
        selected_indices = flat_noise_mask.nonzero(as_tuple=False).squeeze(-1)
        if selected_indices.numel() == 0 or valid_indices.numel() <= 1:
            return embeddings

        rank = torch.empty(
            flat_noise_mask.numel(),
            device=embeddings.device,
            dtype=torch.long,
        )
        rank[valid_indices] = torch.arange(
            valid_indices.numel(),
            device=embeddings.device,
        )
        selected_rank = rank[selected_indices]
        replacement_rank = torch.randint(
            0,
            valid_indices.numel() - 1,
            (selected_indices.numel(),),
            device=embeddings.device,
        )
        replacement_rank = replacement_rank + (replacement_rank >= selected_rank).long()
        replacement_indices = valid_indices[replacement_rank]

        output = flat_embeddings.clone()
        output[selected_indices] = flat_embeddings[replacement_indices]
        return output.view_as(embeddings)

    def _add_parallel_align_item_emb_to_hidden(
        self,
        hidden_states: torch.Tensor,
        target_item_emb: Optional[torch.Tensor],
        item_group: Optional[torch.Tensor],
        code_phase: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.add_align_item_emb_to_hidden:
            return hidden_states
        if self.align_item_emb_to_hidden is None:
            raise ValueError("add_align_item_emb_to_hidden requires item_emb_dim > 0")
        if item_group is None:
            raise ValueError(
                "add_align_item_emb_to_hidden requires item_group"
            )

        batch_size, _, _ = hidden_states.shape
        if self.align_target == "vocab":
            if labels is None:
                raise ValueError(
                    "add_align_item_emb_to_hidden requires labels for vocab target"
                )
            max_items = int(item_group.clamp_min(0).max().item()) + 1
        else:
            if target_item_emb is None:
                raise ValueError(
                    "add_align_item_emb_to_hidden requires target_item_emb"
                )
            max_items = target_item_emb.size(1)

        target_group = item_group if self.align_item == "next" else item_group - 1
        valid = (item_group >= 0) & (target_group >= 0) & (target_group < max_items)
        if self.align_target == "codebook":
            if code_phase is None:
                raise ValueError(
                    "add_align_item_emb_to_hidden requires code_phase for codebook target"
                )
            valid = valid & (code_phase >= 1) & (code_phase <= 3)
        if self.align_target == "vocab":
            valid = valid & labels.ge(0)
        if not valid.any():
            return hidden_states

        batch_idx = (
            torch.arange(batch_size, device=hidden_states.device)
            .unsqueeze(1)
            .expand_as(item_group)
        )
        safe_group = target_group.clamp(0, max_items - 1)
        if self.align_target == "vocab":
            item_emb = self.shared(labels.clamp_min(0)).detach().to(hidden_states.dtype)
        elif self.align_target == "codebook":
            if target_item_emb.dim() != 4:
                raise ValueError(
                    "codebook target_item_emb must have shape "
                    "[batch, max_items, code_per_item, item_emb_dim]"
                )
            safe_phase = (code_phase - 1).clamp(0, target_item_emb.size(-2) - 1)
            item_emb = target_item_emb[batch_idx, safe_group, safe_phase].to(
                hidden_states.dtype
            )
        else:
            if target_item_emb.dim() != 3:
                raise ValueError(
                    "target_item_emb must have shape [batch, max_items, item_emb_dim]"
                )
            item_emb = target_item_emb[batch_idx, safe_group].to(hidden_states.dtype)
        item_hidden = self.align_item_emb_to_hidden(item_emb)
        return self._fuse_align_item_hidden(hidden_states, item_hidden, valid)

    def _fuse_align_item_hidden(
        self,
        hidden_states: torch.Tensor,
        item_hidden: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.align_item_emb_to_hidden_mode == "add":
            fused = hidden_states + item_hidden
        elif self.align_item_emb_to_hidden_mode == "concat":
            if self.align_hidden_concat_proj is None:
                raise ValueError(
                    "align_item_emb_to_hidden_mode='concat' requires concat projection"
                )
            fused = self.align_hidden_concat_proj(
                torch.cat([hidden_states, item_hidden], dim=-1)
            )
        else:
            return hidden_states
        if valid_mask is None:
            return fused
        valid_mask = valid_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states * (1.0 - valid_mask) + fused * valid_mask

    def _select_eval_align_item_emb(
        self,
        target_item_emb: Optional[torch.Tensor],
        phase_idx: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if target_item_emb is None:
            return None
        if target_item_emb.dim() == 2:
            return target_item_emb
        if target_item_emb.dim() == 3:
            if phase_idx is None:
                return target_item_emb
            safe_phase = min(max(int(phase_idx), 0), target_item_emb.size(1) - 1)
            return target_item_emb[:, safe_phase, :]
        raise ValueError("target_item_emb must have shape [batch, dim] or [batch, 4, dim]")

    def _add_eval_align_item_emb_to_hidden(
        self,
        hidden_states: torch.Tensor,
        target_item_emb: Optional[torch.Tensor],
        target_start: Optional[int] = None,
        phase_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.add_align_item_emb_to_hidden:
            return hidden_states
        if self.align_item_emb_to_hidden is None:
            raise ValueError("add_align_item_emb_to_hidden requires item_emb_dim > 0")
        item_emb = self._select_eval_align_item_emb(target_item_emb, phase_idx)
        if item_emb is None:
            raise ValueError("add_align_item_emb_to_hidden requires target_item_emb")
        item_hidden = self.align_item_emb_to_hidden(item_emb.to(hidden_states.dtype))
        if target_start is None:
            hidden_states = hidden_states.clone()
            hidden_states[:, -1:, :] = self._fuse_align_item_hidden(
                hidden_states[:, -1:, :],
                item_hidden.unsqueeze(1),
            )
            return hidden_states
        hidden_states = hidden_states.clone()
        item_hidden = (
            item_hidden
            if item_hidden.dim() == 3
            else item_hidden.unsqueeze(1)
        )
        hidden_states[
            :, target_start : target_start + self.code_per_item, :
        ] = self._fuse_align_item_hidden(
            hidden_states[:, target_start : target_start + self.code_per_item, :],
            item_hidden,
        )
        return hidden_states

    def _encode_tokens(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_code_mask: Optional[torch.Tensor] = None,
        return_shallow_layer: Optional[int] = None,
    ) -> torch.Tensor:
        code_per_item = self.code_per_item
        if token_ids.dim() == 3:
            batch_size, item_len, code_width = token_ids.shape
            if code_width != code_per_item:
                raise ValueError("mixed token input must have width 4")
            token_embeds = self.shared(token_ids)
            if input_code_mask is None:
                input_code_mask = torch.ones_like(token_ids)
            input_code_mask = input_code_mask.to(token_embeds.dtype).unsqueeze(-1)
            denom = input_code_mask.sum(dim=2).clamp_min(1.0)
            hidden_states = (token_embeds * input_code_mask).sum(dim=2) / denom
            valid_mask = attention_mask if attention_mask is not None else denom.squeeze(-1) > 0
            hidden_states = self._add_embedding_noise(hidden_states, valid_mask)
            hidden_states = self.dropout(hidden_states)
        elif token_ids.dim() == 2:
            batch_size, seq_len = token_ids.shape
            if seq_len % code_per_item != 0:
                print(
                    "CausalTIGER _encode_tokens invalid sequence length: "
                    f"batch_size={batch_size}, seq_len={seq_len}, "
                    f"code_per_item={code_per_item}"
                )
                if attention_mask is not None:
                    print(
                        "CausalTIGER _encode_tokens attention lengths: "
                        f"{attention_mask.sum(dim=1).detach().cpu().tolist()}"
                    )
                raise ValueError("token sequence length must be divisible by 4")
            item_len = seq_len // code_per_item
            token_embeds = self.shared(token_ids).view(
                batch_size,
                item_len,
                code_per_item,
                self.d_model,
            )
            hidden_states = token_embeds.mean(dim=2)
            if attention_mask is not None:
                attention_mask = attention_mask.view(batch_size, item_len, code_per_item)
                item_attention_mask = attention_mask[:, :, 0]
                if not torch.equal(
                    attention_mask,
                    item_attention_mask.unsqueeze(-1).expand_as(attention_mask),
                ):
                    raise ValueError("attention_mask must be identical within each 4-token item")
                attention_mask = item_attention_mask
            hidden_states = self._add_embedding_noise(hidden_states, attention_mask)
            hidden_states = self.dropout(hidden_states)
        else:
            raise ValueError("token_ids must be a 2D or 3D tensor")
        if attention_mask is not None and attention_mask.size(1) != item_len:
            raise ValueError(
                "attention_mask length must match encoded sequence length"
            )
        shallow_states = None
        selected_hidden_states = None
        for layer_idx, block in enumerate(self.context_blocks, start=1):
            hidden_states = block(hidden_states, attention_mask)
            if return_shallow_layer == layer_idx:
                shallow_states = hidden_states
            if self.selected_hidden_layer == layer_idx:
                selected_hidden_states = hidden_states
        if selected_hidden_states is not None:
            hidden_states = selected_hidden_states
        hidden_states = self.dropout(self.context_final_layer_norm(hidden_states))
        if selected_hidden_states is None:
            for block in self.decoder_blocks:
                hidden_states = block(hidden_states, attention_mask)
                hidden_states = self.dropout(self.decoder_final_layer_norm(hidden_states))
        if return_shallow_layer is not None:
            if shallow_states is None:
                raise ValueError(f"Unsupported shallow_layer: {return_shallow_layer}")
            return hidden_states, shallow_states
        return hidden_states

    def _lm_logits(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_phase_hidden: bool = False,
    ) -> torch.Tensor:
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            hidden_states = hidden_states * (self.d_model**-0.5)
            logits = torch.matmul(hidden_states, self.shared.weight.transpose(0, 1))
        if labels is not None and labels.size(1) != logits.size(1):
            raise ValueError("labels length must match decoder sequence length")
        if return_phase_hidden:
            return logits, hidden_states
        return logits

    def _teacher_forced_item_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        input_code_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch_size = input_ids.size(0)
        if labels.size(1) != self.code_per_item:
            raise ValueError("item-level labels must contain 4 SID tokens")
        if input_ids.dim() == 3:
            history_ids = input_ids
            item_len = input_ids.size(1)
            if input_ids.size(2) != self.code_per_item:
                raise ValueError("mixed history input must have width 4")
            history_code_mask = (
                input_code_mask if input_code_mask is not None else torch.ones_like(input_ids)
            )
            history_attention = (
                attention_mask
                if attention_mask is not None
                else (history_code_mask.sum(dim=2) > 0).to(input_ids.dtype)
            )
        else:
            seq_len = input_ids.size(1)
            if seq_len % self.code_per_item != 0:
                raise ValueError("history length must be divisible by 4")
            item_len = seq_len // self.code_per_item
            history_ids = input_ids.view(batch_size, item_len, self.code_per_item)
            history_code_mask = torch.ones_like(history_ids)
            if attention_mask is None:
                history_attention = torch.ones(
                    (batch_size, item_len),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
            else:
                history_attention = attention_mask.view(
                    batch_size, item_len, self.code_per_item
                )[:, :, 0]

        prefix_ids = input_ids.new_full(
            (batch_size, self.code_per_item - 1, self.code_per_item),
            self.pad_token_id,
        )
        prefix_ids[:, :, 0] = labels[:, :-1]
        prefix_code_mask = input_ids.new_zeros(
            (batch_size, self.code_per_item - 1, self.code_per_item)
        )
        prefix_code_mask[:, :, 0] = 1
        prefix_attention = input_ids.new_ones(
            (batch_size, self.code_per_item - 1)
        )

        mixed_ids = torch.cat([history_ids, prefix_ids], dim=1)
        mixed_code_mask = torch.cat([history_code_mask, prefix_code_mask], dim=1)
        mixed_attention = torch.cat([history_attention, prefix_attention], dim=1)
        lm_labels = labels.new_full(
            (batch_size, item_len + self.code_per_item - 1), -100
        )
        start = item_len - 1
        lm_labels[:, start : start + self.code_per_item] = labels
        return mixed_ids, mixed_code_mask, mixed_attention, lm_labels, start

    def _align_loss_from_hidden(
        self,
        pred_emb: torch.Tensor,
        target_emb: torch.Tensor,
    ) -> torch.Tensor:
        target_emb = target_emb.to(pred_emb.dtype)
        if self.align_loss_type == "cos":
            return 1.0 - F.cosine_similarity(pred_emb, target_emb, dim=-1).mean()
        return F.mse_loss(pred_emb, target_emb, reduction="mean")

    def _only_hidden_align_loss(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.hidden_to_item_emb is None:
            return hidden_states.new_zeros(())
        batch_size, item_len, _ = hidden_states.shape
        if attention_mask is None:
            valid_item = torch.ones(
                (batch_size, item_len),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        else:
            if attention_mask.size(1) == item_len * self.code_per_item:
                valid_item = attention_mask.view(
                    batch_size,
                    item_len,
                    self.code_per_item,
                )[:, :, 0].bool()
            elif attention_mask.size(1) == item_len:
                valid_item = attention_mask.bool()
            else:
                raise ValueError("attention_mask length does not match hidden_states")
        if self.align_item == "next":
            if item_len <= 1:
                return hidden_states.new_zeros(())
            pred_hidden = hidden_states[:, :-1]
            target_hidden = hidden_states[:, 1:].detach()
            valid_item = valid_item[:, :-1] & valid_item[:, 1:]
        else:
            pred_hidden = hidden_states
            target_hidden = hidden_states.detach()
        if not valid_item.any():
            return hidden_states.new_zeros(())
        pred_emb = self.hidden_to_item_emb(pred_hidden[valid_item])
        return self._align_loss_from_hidden(pred_emb, target_hidden[valid_item])

    def _item_level_align_loss(
        self,
        phase_hidden_states: torch.Tensor,
        target_item_emb: torch.Tensor,
        item_group: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.align_target not in {"item", "latent", "quantized", "codebook", "shallow"}:
            raise ValueError(
                "item-level align only supports item/latent/quantized/codebook/shallow"
            )
        if self.mse_loss_mode not in {"token", "mean", "mean-bar", "only-hidden"}:
            raise ValueError(
                "item-level align only supports token, mean, mean-bar or only-hidden mode"
            )
        if self.mse_loss_mode == "mean-bar" and self.align_target != "shallow":
            raise ValueError("mean-bar align only supports shallow target")

        batch_size, item_len, _, _ = phase_hidden_states.shape
        if item_group is None:
            if self.align_target == "codebook":
                if target_item_emb.dim() == 3:
                    target_item_emb = target_item_emb.unsqueeze(1)
            else:
                if target_item_emb.dim() == 2:
                    target_item_emb = target_item_emb.unsqueeze(1)
                elif target_item_emb.dim() == 3 and target_item_emb.size(1) != item_len:
                    target_item_emb = target_item_emb.mean(dim=1).unsqueeze(1)
            valid_item = phase_hidden_states.new_ones(
                (batch_size, target_item_emb.size(1)), dtype=torch.bool
            )
            item_hidden = phase_hidden_states[:, : target_item_emb.size(1)]
            item_target = target_item_emb
        else:
            group_by_item = item_group.view(batch_size, item_len, self.code_per_item)[
                :, :, 0
            ]
            target_group = group_by_item if self.align_item == "next" else group_by_item - 1
            max_items = target_item_emb.size(1)
            valid_item = (target_group >= 0) & (target_group < max_items)
            safe_group = target_group.clamp(0, max_items - 1)
            batch_idx = torch.arange(
                batch_size, device=phase_hidden_states.device
            ).unsqueeze(1)
            item_hidden = phase_hidden_states
            item_target = target_item_emb[batch_idx, safe_group]

        if self.align_target == "codebook":
            phase_hidden = item_hidden[:, :, :3]
            phase_target = item_target[:, :, :3]
            valid_phase = valid_item.unsqueeze(-1) & phase_target.abs().sum(dim=-1).gt(0)
        else:
            phase_hidden = item_hidden
            phase_target = item_target.unsqueeze(2).expand(
                -1, -1, self.code_per_item, -1
            )
            valid_phase = valid_item.unsqueeze(-1).expand(
                -1, -1, self.code_per_item
            )

        if self.mse_loss_mode == "token":
            if not valid_phase.any():
                print(
                    "CausalTIGER align warning: no valid token phases "
                    f"target={self.align_target}, item={self.align_item}, "
                    f"phase_hidden_shape={tuple(phase_hidden.shape)}, "
                    f"phase_target_shape={tuple(phase_target.shape)}"
                )
                return phase_hidden_states.new_zeros(())
            pred_emb = self.hidden_to_item_emb(phase_hidden[valid_phase])
            return self._align_loss_from_hidden(pred_emb, phase_target[valid_phase])

        if not valid_item.any():
            return phase_hidden_states.new_zeros(())
        if self.mse_loss_mode in {"mean-bar", "only-hidden"}:
            pooled_hidden = phase_hidden[:, :, 0]
            pooled_target = item_target
        elif self.align_target == "codebook":
            counts = valid_phase.to(phase_hidden.dtype).sum(dim=2).clamp_min(1.0)
            pooled_hidden = (
                phase_hidden * valid_phase.unsqueeze(-1).to(phase_hidden.dtype)
            ).sum(dim=2) / counts.unsqueeze(-1)
            pooled_target = (
                phase_target * valid_phase.unsqueeze(-1).to(phase_target.dtype)
            ).sum(dim=2) / counts.unsqueeze(-1)
        else:
            pooled_hidden = phase_hidden.mean(dim=2)
            pooled_target = item_target
        pred_emb = self.hidden_to_item_emb(pooled_hidden[valid_item])
        return self._align_loss_from_hidden(pred_emb, pooled_target[valid_item])

    def _shallow_align_targets(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)
        if self.align_item == "next":
            pad_item = input_ids.new_full(
                (batch_size, self.code_per_item),
                self.pad_token_id,
            )
            pad_attention = torch.ones(
                (batch_size, self.code_per_item),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            shallow_input_ids = torch.cat([input_ids, pad_item], dim=1)
            shallow_attention_mask = torch.cat([attention_mask, pad_attention], dim=1)
            _, shallow_states = self._encode_tokens(
                shallow_input_ids,
                shallow_attention_mask,
                return_shallow_layer=self.shallow_layer,
            )
            return shallow_states[:, 1:].detach()

        _, shallow_states = self._encode_tokens(
            input_ids,
            attention_mask,
            return_shallow_layer=self.shallow_layer,
        )
        return shallow_states.detach()

    def _parallel_align_loss(
        self,
        hidden_states: torch.Tensor,
        target_item_emb: Optional[torch.Tensor],
        item_group: torch.Tensor,
        code_phase: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        align_item_override: Optional[str] = None,
    ) -> torch.Tensor:
        # Per-item mean-pool alignment for parallel training. Every supervised
        # position carries the block-item index it predicts (item_group >= 0);
        # ignored positions are -1. We average the hidden states of the code
        # positions belonging to the same predicted item, project them through
        # hidden_to_item_emb, and align with that item's target embedding
        # (target_item_emb is [B, max_items, dim], [B, max_items, 4, dim] for
        # codebook, or derived from vocab embeddings when align_target='vocab').
        effective_align_item = align_item_override or self.align_item
        if effective_align_item == "nexcur":
            next_loss = self._parallel_align_loss(
                hidden_states,
                target_item_emb,
                item_group,
                code_phase=code_phase,
                labels=labels,
                align_item_override="next",
            )
            current_loss = self._parallel_align_loss(
                hidden_states,
                target_item_emb,
                item_group,
                code_phase=code_phase,
                labels=labels,
                align_item_override="current",
            )
            current_weight = self.nexcur_current_weight
            next_weight = 1.0 - current_weight
            total_loss = next_weight * next_loss + current_weight * current_loss
            self.last_align_loss_tensors = {
                "total": total_loss,
                "next": next_loss,
                "current": current_loss,
            }
            return total_loss

        batch_size, _, d_model = hidden_states.shape
        if self.align_target == "vocab":
            if labels is None:
                raise ValueError("labels are required for parallel vocab align")
            max_items = int(item_group.clamp_min(0).max().item()) + 1
        else:
            max_items = target_item_emb.size(1)
        device = hidden_states.device

        target_group = item_group if effective_align_item == "next" else item_group - 1
        valid = (item_group >= 0) & (target_group >= 0) & (target_group < max_items)
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
        target_slot = batch_idx * max_items + target_group.clamp(0, max_items - 1)
        flat_slot = slot.reshape(-1)[valid.reshape(-1)]
        flat_target_slot = target_slot.reshape(-1)[valid.reshape(-1)]
        flat_hidden = hidden_states.reshape(-1, d_model)[valid.reshape(-1)]
        if self.align_target == "vocab":
            flat_labels = labels.reshape(-1)[valid.reshape(-1)].clamp_min(0)
            flat_target_emb = self.shared(flat_labels).detach().to(flat_hidden.dtype)
        elif self.align_target == "codebook":
            flat_phase = code_phase.reshape(-1)[valid.reshape(-1)] - 1
            flat_target_emb = target_item_emb.reshape(
                num_slots, target_item_emb.size(-2), target_item_emb.size(-1)
            )[flat_target_slot, flat_phase].to(flat_hidden.dtype)
        else:
            flat_target_emb = target_item_emb.reshape(num_slots, -1)[
                flat_target_slot
            ].to(flat_hidden.dtype)

        if self.mse_loss_mode == "token":
            pred_item_emb = self.hidden_to_item_emb(flat_hidden)
            if self.align_loss_type == "cos":
                return 1.0 - F.cosine_similarity(
                    pred_item_emb, flat_target_emb, dim=-1
                ).mean()
            return F.mse_loss(pred_item_emb, flat_target_emb, reduction="mean")

        if self.mse_loss_mode == "pre-first":
            if code_phase is None:
                raise ValueError("code_phase is required for pre-first align")
            first_valid = valid & (code_phase == 1)
            if not first_valid.any():
                return hidden_states.new_zeros(())
            flat_first_slot = slot.reshape(-1)[first_valid.reshape(-1)]
            first_hidden = hidden_states.reshape(-1, d_model)[first_valid.reshape(-1)]
            pred_item_emb = self.hidden_to_item_emb(first_hidden)
            if self.align_target == "codebook":
                target_emb = target_item_emb.reshape(
                    num_slots, target_item_emb.size(-2), target_item_emb.size(-1)
                )[flat_first_slot, 0].to(pred_item_emb.dtype)
            else:
                target_emb = target_item_emb.reshape(num_slots, -1)[
                    flat_first_slot
                ].to(pred_item_emb.dtype)
            if self.align_loss_type == "cos":
                return 1.0 - F.cosine_similarity(
                    pred_item_emb, target_emb, dim=-1
                ).mean()
            return F.mse_loss(pred_item_emb, target_emb, reduction="mean")

        sum_hidden = hidden_states.new_zeros((num_slots, d_model))
        sum_hidden.index_add_(0, flat_slot, flat_hidden)
        counts = hidden_states.new_zeros((num_slots,))
        counts.index_add_(0, flat_slot, torch.ones_like(flat_slot, dtype=counts.dtype))

        active = counts > 0
        pooled_hidden = sum_hidden[active] / counts[active].unsqueeze(-1)
        pred_item_emb = self.hidden_to_item_emb(pooled_hidden)
        active_slot = torch.arange(num_slots, device=device)[active]
        active_batch = active_slot // max_items
        active_group = active_slot % max_items
        active_target_group = (
            active_group if effective_align_item == "next" else active_group - 1
        )
        active_target_slot = active_batch * max_items + active_target_group
        if self.align_target == "vocab":
            sum_target = hidden_states.new_zeros((num_slots, self.d_model))
            sum_target.index_add_(0, flat_slot, flat_target_emb)
            target_emb = sum_target[active] / counts[active].unsqueeze(-1)
            target_emb = target_emb.to(pred_item_emb.dtype)
        elif self.align_target == "codebook":
            target_emb = target_item_emb.reshape(
                num_slots, target_item_emb.size(-2), target_item_emb.size(-1)
            )[active_target_slot, :3].mean(dim=1).to(pred_item_emb.dtype)
        else:
            target_emb = target_item_emb.reshape(num_slots, -1)[
                active_target_slot
            ].to(pred_item_emb.dtype)

        if self.align_loss_type == "cos":
            return 1.0 - F.cosine_similarity(pred_item_emb, target_emb, dim=-1).mean()
        return F.mse_loss(pred_item_emb, target_emb, reduction="mean")

    def _parallel_shallow_token_align_loss(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_code_mask: Optional[torch.Tensor],
        item_group: Optional[torch.Tensor],
        code_phase: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if item_group is None or code_phase is None:
            return hidden_states.new_zeros(())
        _, shallow_states = self._encode_tokens(
            input_ids,
            attention_mask,
            input_code_mask=input_code_mask,
            return_shallow_layer=self.shallow_layer,
        )
        if hidden_states.size(1) <= 1:
            return hidden_states.new_zeros(())
        valid = (item_group[:, :-1] >= 0) & (code_phase[:, :-1] >= 1)
        if self.mse_loss_mode == "pre-first":
            valid = valid & (code_phase[:, :-1] == 1)
        if not valid.any():
            return hidden_states.new_zeros(())

        student_hidden = hidden_states[:, :-1]
        teacher_hidden = shallow_states[:, 1:].detach()
        if self.mse_loss_mode in {"token", "pre-first"}:
            pred_hidden = self.hidden_to_item_emb(student_hidden[valid])
            target_hidden = teacher_hidden[valid].to(pred_hidden.dtype)
            if self.align_loss_type == "cos":
                return 1.0 - F.cosine_similarity(
                    pred_hidden, target_hidden, dim=-1
                ).mean()
            return F.mse_loss(pred_hidden, target_hidden, reduction="mean")

        batch_size, _, d_model = student_hidden.shape
        max_items = int(item_group.clamp_min(0).max().item()) + 1
        num_slots = batch_size * max_items
        batch_idx = (
            torch.arange(batch_size, device=hidden_states.device)
            .unsqueeze(1)
            .expand_as(item_group[:, :-1])
        )
        slot = batch_idx * max_items + item_group[:, :-1].clamp_min(0)
        flat_slot = slot.reshape(-1)[valid.reshape(-1)]
        flat_student = student_hidden.reshape(-1, d_model)[valid.reshape(-1)]
        flat_teacher = teacher_hidden.reshape(-1, d_model)[valid.reshape(-1)]
        sum_student = hidden_states.new_zeros((num_slots, d_model))
        sum_teacher = hidden_states.new_zeros((num_slots, d_model))
        counts = hidden_states.new_zeros((num_slots,))
        sum_student.index_add_(0, flat_slot, flat_student)
        sum_teacher.index_add_(0, flat_slot, flat_teacher)
        counts.index_add_(0, flat_slot, torch.ones_like(flat_slot, dtype=counts.dtype))
        active = counts > 0
        pred_hidden = self.hidden_to_item_emb(
            sum_student[active] / counts[active].unsqueeze(-1)
        )
        target_hidden = (
            sum_teacher[active] / counts[active].unsqueeze(-1)
        ).to(pred_hidden.dtype)
        if self.align_loss_type == "cos":
            return 1.0 - F.cosine_similarity(
                pred_hidden, target_hidden, dim=-1
            ).mean()
        return F.mse_loss(pred_hidden, target_hidden, reduction="mean")

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
        input_code_mask: Optional[torch.Tensor] = None,
        parallel: bool = False,
    ):
        self.last_loss_tensors = None
        self.last_align_loss_tensors = {}
        if labels is None:
            hidden_states = self._encode_tokens(input_ids, attention_mask)
            return None, self._lm_logits(hidden_states)

        if parallel:
            # Parallel next-token training: input_ids and labels are already
            # token-aligned (labels = input shifted by one token, with -100 on
            # context/pad positions). A single forward computes CE over every
            # supervised position at once.
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            hidden_states = self._encode_tokens(
                input_ids,
                attention_mask,
                input_code_mask=input_code_mask,
            )
            hidden_states = self._add_parallel_align_item_emb_to_hidden(
                hidden_states,
                target_item_emb,
                item_group,
                code_phase=code_phase,
                labels=labels,
            )
            self.last_target_hidden_states = hidden_states.detach()
            logits, token_hidden_states = self._lm_logits(
                hidden_states, labels, return_phase_hidden=True
            )
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            align_loss = torch.zeros(
                (), dtype=ce_loss.dtype, device=ce_loss.device
            )
            if (
                self.align_target == "shallow"
                and self.hidden_to_item_emb is not None
                and self.mse_loss_weight > 0
            ):
                align_loss = self._parallel_shallow_token_align_loss(
                    token_hidden_states,
                    input_ids,
                    attention_mask,
                    input_code_mask,
                    item_group,
                    code_phase,
                )
            elif (
                target_item_emb is not None
                and self.align_target in {"item", "latent", "quantized", "codebook"}
                and item_group is not None
                and self.hidden_to_item_emb is not None
                and self.mse_loss_weight > 0
            ):
                align_loss = self._parallel_align_loss(
                    token_hidden_states,
                    target_item_emb,
                    item_group,
                    code_phase,
                    labels,
                )
            loss = ce_loss + self.mse_loss_weight * align_loss
            align_loss_tensors = dict(getattr(self, "last_align_loss_tensors", {}))
            align_loss_tensors.setdefault("total", align_loss)
            self.last_align_loss_tensors = align_loss_tensors
            self.last_loss_tensors = {
                "total": loss,
                "ce": ce_loss,
                "align": align_loss,
                "align_total": align_loss_tensors["total"],
            }
            if "next" in align_loss_tensors:
                self.last_loss_tensors["align_next"] = align_loss_tensors["next"]
            if "current" in align_loss_tensors:
                self.last_loss_tensors["align_current"] = align_loss_tensors["current"]
            self.last_loss_dict = {
                "total": loss.detach(),
                "ce": ce_loss.detach(),
                "align": align_loss.detach(),
            }
            return loss, logits


        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        (
            mixed_ids,
            mixed_code_mask,
            mixed_attention,
            lm_labels,
            target_start,
        ) = self._teacher_forced_item_inputs(
            input_ids,
            attention_mask,
            labels,
            input_code_mask=input_code_mask,
        )
        hidden_states = self._encode_tokens(
            mixed_ids,
            mixed_attention,
            input_code_mask=mixed_code_mask,
        )
        hidden_states = self._add_eval_align_item_emb_to_hidden(
            hidden_states,
            target_item_emb,
            target_start=target_start,
        )
        target_hidden_states = hidden_states[
            :, target_start : target_start + self.code_per_item, :
        ]
        self.last_target_hidden_states = target_hidden_states.detach()
        logits, token_hidden_states = self._lm_logits(
            hidden_states, lm_labels, return_phase_hidden=True
        )
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            lm_labels.reshape(-1),
            ignore_index=-100,
        )
        mse_loss = torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
        if (
            self.align_target == "shallow"
            and self.hidden_to_item_emb is not None
            and self.mse_loss_weight > 0
        ):
            shallow_targets = self._shallow_align_targets(input_ids, attention_mask)[:, -1:]
            phase_hidden_states = target_hidden_states.unsqueeze(1)
            align_loss = self._item_level_align_loss(
                phase_hidden_states,
                shallow_targets,
                item_group=None,
            )
        elif (
            target_item_emb is not None
            and self.align_target in {"item", "latent", "quantized", "codebook"}
            and self.hidden_to_item_emb is not None
            and self.mse_loss_weight > 0
        ):
            phase_hidden_states = target_hidden_states.unsqueeze(1)
            align_loss = self._item_level_align_loss(
                phase_hidden_states,
                target_item_emb,
                item_group=None,
            )
        else:
            align_loss = mse_loss
        loss = ce_loss + self.mse_loss_weight * align_loss
        self.last_align_loss_tensors = {"total": align_loss}
        self.last_loss_tensors = {
            "total": loss,
            "ce": ce_loss,
            "align": align_loss,
            "align_total": align_loss,
        }
        self.last_loss_dict = {
            "total": loss.detach(),
            "ce": ce_loss.detach(),
            "align": align_loss.detach(),
        }
        return loss, logits

    def _next_token_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generated_ids: torch.Tensor,
        input_code_mask: Optional[torch.Tensor] = None,
        target_item_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)
        if input_ids.dim() == 3:
            history_ids = input_ids
            item_len = input_ids.size(1)
            if input_ids.size(2) != self.code_per_item:
                raise ValueError("mixed history input must have width 4")
            history_code_mask = (
                input_code_mask if input_code_mask is not None else torch.ones_like(input_ids)
            )
            history_attention = (
                attention_mask
                if attention_mask is not None
                else (history_code_mask.sum(dim=2) > 0).to(input_ids.dtype)
            )
        else:
            _, seq_len = input_ids.shape
            if seq_len % self.code_per_item != 0:
                raise ValueError("history length must be divisible by 4")
            item_len = seq_len // self.code_per_item
            history_ids = input_ids.view(batch_size, item_len, self.code_per_item)
            history_code_mask = torch.ones_like(history_ids)
            if attention_mask is None:
                history_attention = torch.ones(
                    (batch_size, item_len),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
            else:
                history_attention = attention_mask.view(
                    batch_size, item_len, self.code_per_item
                )[:, :, 0]

        if generated_ids.size(1) > 0:
            generated_positions = input_ids.new_full(
                (batch_size, generated_ids.size(1), self.code_per_item),
                self.pad_token_id,
            )
            generated_positions[:, :, 0] = generated_ids
            generated_code_mask = input_ids.new_zeros(
                (batch_size, generated_ids.size(1), self.code_per_item)
            )
            generated_code_mask[:, :, 0] = 1
            model_input_ids = torch.cat([history_ids, generated_positions], dim=1)
            input_code_mask = torch.cat([history_code_mask, generated_code_mask], dim=1)
            generated_attention = input_ids.new_ones(
                (batch_size, generated_ids.size(1))
            )
            model_attention_mask = torch.cat(
                [history_attention, generated_attention], dim=1
            )
        else:
            model_input_ids = history_ids
            input_code_mask = history_code_mask
            model_attention_mask = history_attention
        hidden_states = self._encode_tokens(
            model_input_ids,
            model_attention_mask,
            input_code_mask=input_code_mask,
        )
        hidden_states = self._add_eval_align_item_emb_to_hidden(
            hidden_states,
            target_item_emb,
            phase_idx=generated_ids.size(1),
        )
        return self._lm_logits(hidden_states)[:, -1, :]

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_code_mask: Optional[torch.Tensor] = None,
        target_item_emb: Optional[torch.Tensor] = None,
        max_length: int = 4,
        num_beams: int = 20,
        num_return_sequences: Optional[int] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if max_length != self.code_per_item:
            raise ValueError("max_length must be 4 for item-level SID generation")
        if num_return_sequences is None:
            num_return_sequences = num_beams
        if num_return_sequences > num_beams:
            raise ValueError("num_return_sequences must be <= num_beams")

        batch_size = input_ids.size(0)
        vocab_size = self.vocab_size
        device = input_ids.device
        beam_input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        beam_attention_mask = (
            attention_mask.repeat_interleave(num_beams, dim=0)
            if attention_mask is not None
            else None
        )
        beam_input_code_mask = (
            input_code_mask.repeat_interleave(num_beams, dim=0)
            if input_code_mask is not None
            else None
        )
        beam_target_item_emb = (
            target_item_emb.repeat_interleave(num_beams, dim=0)
            if target_item_emb is not None
            else None
        )
        generated = torch.empty(
            (batch_size * num_beams, 0),
            dtype=input_ids.dtype,
            device=device,
        )
        beam_scores = torch.full(
            (batch_size, num_beams),
            torch.finfo(torch.float32).min,
            device=device,
        )
        beam_scores[:, 0] = 0.0

        for phase_idx in range(max_length):
            logits = self._next_token_logits(
                beam_input_ids,
                beam_attention_mask,
                generated,
                input_code_mask=beam_input_code_mask,
                target_item_emb=beam_target_item_emb,
            )
            log_probs = F.log_softmax(logits.float(), dim=-1)

            if prefix_allowed_tokens_fn is not None:
                allowed_mask = torch.zeros_like(log_probs, dtype=torch.bool)
                for flat_idx in range(generated.size(0)):
                    batch_idx = flat_idx // num_beams
                    allowed_tokens = prefix_allowed_tokens_fn(
                        batch_idx, generated[flat_idx]
                    )
                    if allowed_tokens:
                        allowed_mask[flat_idx, allowed_tokens] = True
                log_probs = log_probs.masked_fill(
                    ~allowed_mask, torch.finfo(log_probs.dtype).min
                )

            next_scores = log_probs + beam_scores.view(-1, 1)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            top_scores, top_indices = torch.topk(next_scores, num_beams, dim=1)
            next_beam_indices = top_indices // vocab_size
            next_tokens = top_indices % vocab_size

            batch_offsets = (
                torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            )
            gather_indices = (batch_offsets + next_beam_indices).reshape(-1)
            generated = torch.cat(
                [
                    generated[gather_indices],
                    next_tokens.reshape(-1, 1).to(input_ids.dtype),
                ],
                dim=1,
            )
            beam_input_ids = beam_input_ids[gather_indices]
            if beam_attention_mask is not None:
                beam_attention_mask = beam_attention_mask[gather_indices]
            if beam_input_code_mask is not None:
                beam_input_code_mask = beam_input_code_mask[gather_indices]
            if beam_target_item_emb is not None:
                beam_target_item_emb = beam_target_item_emb[gather_indices]
            beam_scores = top_scores

        generated = generated.view(batch_size, num_beams, max_length)
        return generated[:, :num_return_sequences, :].reshape(
            batch_size * num_return_sequences,
            max_length,
        )
