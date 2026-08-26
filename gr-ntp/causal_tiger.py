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

    def forward(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
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
    ):
        super().__init__()
        inner_dim = num_heads * d_kv
        self.num_heads = num_heads
        self.d_kv = d_kv
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
        scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                key_mask = attention_mask.to(torch.bool).view(batch_size, 1, seq_len, seq_len)
            else:
                key_mask = attention_mask.to(torch.bool).view(batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        attn_weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size,
            seq_len,
            self.num_heads * self.d_kv,
        )
        return self.o(attn_output)


class T5FeedForward(nn.Module):
    """T5 feed-forward block; relu mode matches the current main.py default."""

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float, feed_forward_proj: str):
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
    ):
        super().__init__()
        self.self_attn_norm = T5LayerNorm(d_model)
        self.self_attn = CausalMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_kv=d_kv,
            dropout_rate=dropout_rate,
            has_relative_attention_bias=has_relative_attention_bias,
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
            extra_output = self.extra_attn(self.extra_attn_norm(hidden_states), attention_mask)
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
        self.decoder_start_token_id = config.get("decoder_start_token_id", self.pad_token_id)
        self.dropout = nn.Dropout(config["dropout_rate"])
        self.shared = nn.Embedding(config["vocab_size"], config["d_model"])

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
    ) -> torch.Tensor:
        hidden_states = self.dropout(self.shared(token_ids))
        for block in self.context_blocks:
            hidden_states = block(hidden_states, attention_mask)
        hidden_states = self.dropout(self.context_final_layer_norm(hidden_states))
        for block in self.decoder_blocks:
            hidden_states = block(hidden_states, attention_mask)
        hidden_states = self.dropout(self.decoder_final_layer_norm(hidden_states))
        return hidden_states

    def _lm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states * (self.d_model ** -0.5)
        return torch.matmul(hidden_states, self.shared.weight.transpose(0, 1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        if labels is None:
            hidden_states = self._encode_tokens(input_ids, attention_mask)
            return None, self._lm_logits(hidden_states)

        if labels.shape == input_ids.shape:
            hidden_states = self._encode_tokens(input_ids, attention_mask)
            logits = self._lm_logits(hidden_states)
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            if loss_mask is None:
                shift_labels = shift_labels.masked_fill(shift_labels == self.pad_token_id, -100)
            else:
                shift_labels = shift_labels.masked_fill(~loss_mask[:, 1:].to(torch.bool), -100)
            token_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(shift_labels.size())
            valid_mask = shift_labels.ne(-100)
            loss = (token_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1)
            return loss, shift_logits

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        model_input_ids = torch.cat([input_ids, labels[:, :-1]], dim=1)
        label_attention_mask = torch.ones_like(labels[:, :-1])
        model_attention_mask = torch.cat([attention_mask, label_attention_mask], dim=1)
        hidden_states = self._encode_tokens(model_input_ids, model_attention_mask)
        history_last_pos = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        first_hidden = hidden_states[
            torch.arange(input_ids.size(0), device=input_ids.device),
            history_last_pos,
        ].unsqueeze(1)
        rest_hidden = hidden_states[:, input_ids.size(1) :, :]
        pred_hidden = torch.cat([first_hidden, rest_hidden], dim=1)
        logits = self._lm_logits(pred_hidden)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
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
        trie_prefix = torch.full(
            (batch_size * num_beams, 1),
            self.decoder_start_token_id,
            dtype=input_ids.dtype,
            device=device,
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

        steps = max_length - 1
        for _ in range(steps):
            logits = self._next_token_logits(beam_input_ids, beam_attention_mask, generated)
            log_probs = F.log_softmax(logits.float(), dim=-1)

            if prefix_allowed_tokens_fn is not None:
                allowed_mask = torch.zeros_like(log_probs, dtype=torch.bool)
                for flat_idx in range(generated.size(0)):
                    batch_idx = flat_idx // num_beams
                    allowed_tokens = prefix_allowed_tokens_fn(batch_idx, trie_prefix[flat_idx])
                    if allowed_tokens:
                        allowed_mask[flat_idx, allowed_tokens] = True
                log_probs = log_probs.masked_fill(~allowed_mask, torch.finfo(log_probs.dtype).min)

            next_scores = log_probs + beam_scores.view(-1, 1)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            top_scores, top_indices = torch.topk(next_scores, num_beams, dim=1)
            next_beam_indices = top_indices // vocab_size
            next_tokens = top_indices % vocab_size

            batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            gather_indices = (batch_offsets + next_beam_indices).reshape(-1)
            generated = torch.cat(
                [generated[gather_indices], next_tokens.reshape(-1, 1).to(input_ids.dtype)],
                dim=1,
            )
            trie_prefix = torch.cat(
                [trie_prefix[gather_indices], next_tokens.reshape(-1, 1).to(input_ids.dtype)],
                dim=1,
            )
            beam_input_ids = beam_input_ids[gather_indices]
            if beam_attention_mask is not None:
                beam_attention_mask = beam_attention_mask[gather_indices]
            beam_scores = top_scores

        output = torch.cat([trie_prefix[:, :1], generated], dim=1)
        output = output.view(batch_size, num_beams, max_length)
        return output[:, :num_return_sequences, :].reshape(
            batch_size * num_return_sequences,
            max_length,
        )
