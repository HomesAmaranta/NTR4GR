from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel


class GPT2TIGER(nn.Module):
    """GPT2 causal-LM wrapper with the same train/evaluate interface as TIGER."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        if config["d_model"] % config["num_heads"] != 0:
            raise ValueError(
                "GPT2 requires d_model to be divisible by num_heads; "
                f"got d_model={config['d_model']} and num_heads={config['num_heads']}."
            )

        self.pad_token_id = config["pad_token_id"]
        self.eos_token_id = config["eos_token_id"]
        self.decoder_start_token_id = config.get("decoder_start_token_id", self.pad_token_id)

        max_positions = config.get("max_seq_len", config["max_len"]) * 4 + 8
        gpt2_config = GPT2Config(
            vocab_size=config["vocab_size"],
            n_positions=max_positions,
            n_ctx=max_positions,
            n_embd=config["d_model"],
            n_layer=config["num_decoder_layers"],
            n_head=config["num_heads"],
            resid_pdrop=config["dropout_rate"],
            embd_pdrop=config["dropout_rate"],
            attn_pdrop=config["dropout_rate"],
            bos_token_id=self.decoder_start_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )
        self.model = GPT2LMHeadModel(gpt2_config)

    @property
    def n_parameters(self) -> str:
        num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
        total_params = num_params(self.parameters())
        emb_params = num_params(self.model.get_input_embeddings().parameters())
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if labels is None:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            return None, outputs.logits

        if labels.shape == input_ids.shape:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:].clone()
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

        batch_size = input_ids.size(0)
        start_tokens = torch.full(
            (batch_size, 1),
            self.decoder_start_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        model_input_ids = torch.cat([input_ids, start_tokens, labels], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        target_attention_mask = torch.ones(
            (batch_size, labels.size(1) + 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        model_attention_mask = torch.cat([attention_mask, target_attention_mask], dim=1)

        ignored_prefix = torch.full(
            (batch_size, input_ids.size(1) + 1),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        lm_labels = torch.cat([ignored_prefix, labels], dim=1)
        outputs = self.model(
            input_ids=model_input_ids,
            attention_mask=model_attention_mask,
            position_ids=position_ids,
            labels=lm_labels,
        )
        target_logits = outputs.logits[:, -(labels.size(1) + 1) : -1, :]
        return outputs.loss, target_logits

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

        batch_size = input_ids.size(0)
        start_tokens = torch.full(
            (batch_size, 1),
            self.decoder_start_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        prompt_input_ids = torch.cat([input_ids, start_tokens], dim=1)
        prompt_length = prompt_input_ids.size(1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        prompt_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )

        adjusted_prefix_allowed_tokens_fn = None
        if prefix_allowed_tokens_fn is not None:
            trie_prefix_start = prompt_length - 1

            def adjusted_prefix_allowed_tokens(batch_id, sentence):
                trie_sentence = sentence[trie_prefix_start:]
                return prefix_allowed_tokens_fn(batch_id, trie_sentence)

            adjusted_prefix_allowed_tokens_fn = adjusted_prefix_allowed_tokens

        outputs = self.model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_length - 1,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            prefix_allowed_tokens_fn=adjusted_prefix_allowed_tokens_fn,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            **kwargs,
        )

        generated = outputs[:, input_ids.size(1) :]
        if generated.size(1) < max_length:
            pad = torch.full(
                (generated.size(0), max_length - generated.size(1)),
                self.pad_token_id,
                dtype=generated.dtype,
                device=generated.device,
            )
            generated = torch.cat([generated, pad], dim=1)
        return generated[:, :max_length]
