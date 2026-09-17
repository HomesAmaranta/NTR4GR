import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SourceToNextSidModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        mlp_dim=512,
        vocab_size=1025,
        codebook_size=256,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.codebook_size = codebook_size
        self.hidden = MLP(input_dim, mlp_dim, hidden_dim, dropout=dropout)
        self.sid_proj = nn.Embedding(vocab_size, hidden_dim)
        self.heads = nn.ModuleList(
            [MLP(hidden_dim, mlp_dim, vocab_size, dropout=dropout) for _ in range(4)]
        )

    @property
    def n_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"#Total trainable parameters: {total}\n"

    def _phase_masked_logits(self, logits, phase):
        start = phase * self.codebook_size + 1
        end = (phase + 1) * self.codebook_size + 1
        masked = logits.new_full(logits.shape, torch.finfo(logits.dtype).min)
        masked[:, start:end] = logits[:, start:end]
        return masked

    def forward(self, source_emb, target_code=None):
        hidden = self.hidden(source_emb)
        state = hidden
        logits_by_phase = []
        loss = None

        for phase, head in enumerate(self.heads):
            logits = self._phase_masked_logits(head(state), phase)
            logits_by_phase.append(logits)
            if target_code is not None and phase < 3:
                state = state + self.sid_proj(target_code[:, phase])

        if target_code is not None:
            losses = [
                F.cross_entropy(logits_by_phase[phase], target_code[:, phase])
                for phase in range(4)
            ]
            loss = sum(losses) / len(losses)

        return loss, logits_by_phase

    @torch.no_grad()
    def generate(self, source_emb, beam_size=30):
        hidden = self.hidden(source_emb)
        batch_sequences = []

        for row_hidden in hidden:
            beams = [(0.0, [], row_hidden)]
            for phase, head in enumerate(self.heads):
                candidates = []
                for score, seq, state in beams:
                    logits = self._phase_masked_logits(head(state.unsqueeze(0)), phase)
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                    top_scores, top_tokens = torch.topk(log_probs, beam_size)
                    for token_score, token in zip(top_scores.tolist(), top_tokens.tolist()):
                        next_state = state
                        if phase < 3:
                            token_tensor = torch.tensor(
                                token, dtype=torch.long, device=source_emb.device
                            )
                            next_state = state + self.sid_proj(token_tensor)
                        candidates.append((score + token_score, seq + [token], next_state))
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_size]
            batch_sequences.append([seq for _, seq, _ in beams])

        return torch.tensor(batch_sequences, dtype=torch.long, device=source_emb.device)
