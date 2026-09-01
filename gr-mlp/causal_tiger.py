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
        if self.align_target == "shallow" and self.item_emb_dim <= 0:
            self.item_emb_dim = self.d_model
        self.shallow_layer = config.get("shallow_layer", 1)
        self.lm_head_type = config.get("lm_head", "emb")
        if self.mse_loss_mode not in {"token", "mean"}:
            raise ValueError("mse_loss_mode must be 'token' or 'mean'")
        if self.align_loss_type not in {"mse", "cos"}:
            raise ValueError("align_loss_type must be 'mse' or 'cos'")
        if self.lm_head_type not in {"emb", "linear"}:
            raise ValueError("lm_head must be 'emb' or 'linear'")
        if self.shallow_layer < 1 or self.shallow_layer > config["num_layers"]:
            raise ValueError(f"shallow_layer must be in [1, {config['num_layers']}]")
        self.dropout = nn.Dropout(config["dropout_rate"])
        # Sliding-window attention span in tokens (None = full causal). Used by
        # the parallel training mode to keep the visible history bounded.
        self.attention_window = config.get("attention_window", None)
        self.shared = nn.Embedding(config["vocab_size"], config["d_model"])
        self.lm_head = (
            nn.Linear(config["d_model"], config["vocab_size"], bias=False)
            if self.lm_head_type == "linear"
            else None
        )
        self.hidden_to_item_emb = (
            nn.Linear(config["d_model"], self.item_emb_dim, bias=False)
            if self.item_emb_dim > 0
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

    def _lm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        hidden_states = hidden_states * (self.d_model**-0.5)
        return torch.matmul(hidden_states, self.shared.weight.transpose(0, 1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        target_item_emb: Optional[torch.Tensor] = None,
        parallel: bool = False,
    ):
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
            hidden_states = self._encode_tokens(input_ids, attention_mask)
            self.last_target_hidden_states = hidden_states.detach()
            logits = self._lm_logits(hidden_states)
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            self.last_loss_dict = {
                "total": ce_loss.detach(),
                "ce": ce_loss.detach(),
                "align": torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device),
            }
            return ce_loss, logits

        decoder_input_ids = labels[:, :-1]
        model_input_ids = torch.cat([input_ids, decoder_input_ids], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        model_attention_mask = torch.cat(
            [attention_mask, decoder_attention_mask], dim=1
        )

        hidden_states = self._encode_tokens(model_input_ids, model_attention_mask)
        target_hidden_states = hidden_states[:, -labels.size(1) :, :]
        self.last_target_hidden_states = target_hidden_states.detach()
        logits = self._lm_logits(target_hidden_states)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        mse_loss = torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
        if (
            (target_item_emb is not None or self.align_target == "shallow")
            and self.hidden_to_item_emb is not None
            and self.mse_loss_weight > 0
        ):
            if self.align_target == "shallow":
                shallow_input_ids = torch.cat([input_ids, labels], dim=1)
                shallow_attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(labels)],
                    dim=1,
                )
                max_shallow_len = input_ids.size(1)
                if shallow_input_ids.size(1) > max_shallow_len:
                    shallow_input_ids = shallow_input_ids[:, -max_shallow_len:]
                    shallow_attention_mask = shallow_attention_mask[
                        :, -max_shallow_len:
                    ]
                _, shallow_states = self._encode_tokens(
                    shallow_input_ids,
                    shallow_attention_mask,
                    return_shallow_layer=self.shallow_layer,
                )
                target_item_emb = shallow_states[:, -labels.size(1) :, :].detach()

            if self.mse_loss_mode == "mean":
                if self.align_target == "codebook":
                    valid_codebook = (
                        target_item_emb.abs()
                        .sum(dim=-1)
                        .gt(0)
                        .to(target_hidden_states.dtype)
                    )
                    denom = valid_codebook.sum(dim=1, keepdim=True).clamp_min(1.0)
                    pooled_hidden = (
                        target_hidden_states * valid_codebook.unsqueeze(-1)
                    ).sum(dim=1) / denom
                    target_emb = (target_item_emb * valid_codebook.unsqueeze(-1)).sum(
                        dim=1
                    ) / denom
                    target_emb = target_emb.to(pooled_hidden.dtype)
                else:
                    pooled_hidden = target_hidden_states.mean(dim=1)
                    target_emb = target_item_emb.mean(dim=1).to(pooled_hidden.dtype)
                pred_item_emb = self.hidden_to_item_emb(pooled_hidden)
                if self.align_loss_type == "cos":
                    align_loss = (
                        1.0
                        - F.cosine_similarity(pred_item_emb, target_emb, dim=-1).mean()
                    )
                else:
                    align_loss = F.mse_loss(pred_item_emb, target_emb, reduction="mean")
            else:
                pred_item_emb = self.hidden_to_item_emb(target_hidden_states)
                target_emb = target_item_emb.to(pred_item_emb.dtype)
                if self.align_target == "codebook":
                    valid_codebook = target_item_emb.abs().sum(dim=-1).gt(0)
                    if self.align_loss_type == "cos":
                        token_loss = 1.0 - F.cosine_similarity(
                            pred_item_emb, target_emb, dim=-1
                        )
                    else:
                        token_loss = F.mse_loss(
                            pred_item_emb, target_emb, reduction="none"
                        ).mean(dim=-1)
                    align_loss = (
                        token_loss[valid_codebook].mean()
                        if valid_codebook.any()
                        else mse_loss
                    )
                elif self.align_loss_type == "cos":
                    align_loss = (
                        1.0
                        - F.cosine_similarity(pred_item_emb, target_emb, dim=-1).mean()
                    )
                else:
                    align_loss = F.mse_loss(pred_item_emb, target_emb, reduction="mean")
        else:
            align_loss = mse_loss
        loss = ce_loss + self.mse_loss_weight * align_loss
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
    ) -> torch.Tensor:
        model_input_ids = torch.cat([input_ids, generated_ids], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        generated_mask = torch.ones_like(generated_ids)
        model_attention_mask = torch.cat([attention_mask, generated_mask], dim=1)
        hidden_states = self._encode_tokens(model_input_ids, model_attention_mask)
        return self._lm_logits(hidden_states[:, -1, :])

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
        beam_input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        beam_attention_mask = (
            attention_mask.repeat_interleave(num_beams, dim=0)
            if attention_mask is not None
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

        for _ in range(max_length):
            logits = self._next_token_logits(
                beam_input_ids, beam_attention_mask, generated
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
            beam_scores = top_scores

        generated = generated.view(batch_size, num_beams, max_length)
        return generated[:, :num_return_sequences, :].reshape(
            batch_size * num_return_sequences,
            max_length,
        )
