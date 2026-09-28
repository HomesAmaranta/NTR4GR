"""Optional representation alignment for the Qwen3 training model."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class AlignedQwen3ForCausalLM(Qwen3ForCausalLM):
    # The auxiliary loss is a batch mean. Let Trainer scale the combined loss
    # for gradient accumulation rather than supplying a token-count denominator.
    accepts_loss_kwargs = False

    def initialize_alignment(self, item_emb_dim, loss_weight):
        """Call after k-bit preparation, before PEFT wraps modules_to_save."""
        if item_emb_dim <= 0 or loss_weight <= 0:
            raise ValueError("Alignment requires positive item_emb_dim and loss_weight")
        self.alignment_loss_weight = loss_weight
        self.hidden_to_item_emb = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, item_emb_dim, bias=False),
        ).to(device=self.get_input_embeddings().weight.device, dtype=torch.float32)

    def alignment_loss(self, hidden_states, alignment_positions, target_item_emb):
        if alignment_positions.shape != (hidden_states.shape[0], 4):
            raise ValueError("Expected four prediction positions per example")
        positions = alignment_positions.to(device=hidden_states.device, dtype=torch.long)
        if torch.any(positions < 0) or torch.any(positions >= hidden_states.shape[1]):
            raise ValueError("Alignment positions are outside the input sequence")
        selected = hidden_states.gather(
            1, positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        )
        head_dtype = next(self.hidden_to_item_emb.parameters()).dtype
        prediction = self.hidden_to_item_emb(selected.mean(dim=1).to(head_dtype))
        target = target_item_emb.detach().to(device=prediction.device, dtype=prediction.dtype)
        if target.ndim == 3:
            target = target.mean(dim=1)
        if target.shape != prediction.shape:
            raise ValueError(f"Alignment shape mismatch: {prediction.shape} vs {target.shape}")
        return 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        target_item_emb=None,
        alignment_positions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Alignment is a training-only objective. Trainer switches the model to
        # eval mode for validation, where checkpoint selection must use CE only.
        if not self.training or (
            target_item_emb is None and alignment_positions is None
        ):
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs,
            )
        if target_item_emb is None or alignment_positions is None or labels is None:
            raise ValueError("Alignment requires labels, target_item_emb and alignment_positions")
        if not hasattr(self, "hidden_to_item_emb"):
            raise ValueError("Call initialize_alignment before aligned training")

        outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            output_hidden_states=True, return_dict=True, **kwargs,
        )
        ce_loss = outputs.loss
        align_loss = self.alignment_loss(
            outputs.hidden_states[-1], alignment_positions, target_item_emb
        )
        outputs.loss = ce_loss + self.alignment_loss_weight * align_loss
        self.last_loss_dict = {
            "ce": ce_loss.detach(),
            "align": align_loss.detach(),
            "total": outputs.loss.detach(),
        }
        # Do not retain every layer in Trainer's prediction output.
        if not (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states):
            outputs = CausalLMOutputWithPast(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                attentions=outputs.attentions,
            )
        use_dict = self.config.use_return_dict if return_dict is None else return_dict
        return outputs if use_dict else outputs.to_tuple()
