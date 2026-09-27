# Optional cosine alignment

`--mse_loss_weight` defaults to `0`. At zero weight, training uses the original
AutoModel/Trainer/collator path, without loading item embeddings, creating a
projection head, or requesting hidden states.

Enable from the repository root:

```bash
MSE_LOSS_WEIGHT=1 ALIGN_TARGET=latent ALIGN_ITEM=current \
bash LC-Rec-backbone/scripts_qwen/train.sh
```

The Python entry point accepts:

```text
--mse_loss_weight 1
--align_loss_type cos
--align_target latent       # latent or quantized
--align_item current        # current or next
--item_emb_dim 32
--item_emb_path PATH        # optional override
```

Default embedding files under `data_path/dataset/`:

- `latent`: `item_emb_rqvae_encoder_latent.parquet`
- `quantized`: `item_emb_rqvae_quantized_latent.parquet`

Both use `ItemID` and `embedding` columns, as in `model/dataset.py`. Multiple
vectors for one item are mean-pooled. Embeddings are fixed targets.

For history `[a, b]`, target `c`, current selects item `b`; next selects item `c`.
Selection happens separately for each expanded training sample. An empty
validation history falls back to the target item, matching `model/dataset.py`.
Validation uses the same weighted objective; early stopping therefore monitors
the combined validation loss. Test/generation does not load item embeddings.

The final-layer hidden states used to predict `c1,c2,c3,c4` are gathered at their
positions minus one: `[answer_separator,c1,c2,c3]`. They are mean-pooled and passed
through `Linear(H,H) -> GELU -> Linear(H,D,bias=False)`.

`loss = original_lm_loss + weight * mean(1 - cosine(projected_hidden, item_embedding))`.

The collator preserves the original CE labels, including EOS masking. It checks
that the answer separator is one token and all four target SID tokens survive
tokenization/truncation. Invalid target boundaries raise an error.

`hidden_to_item_emb` is included in PEFT `modules_to_save` and saved with the
adapter. Resume aligned training with the same alignment options and embedding
dimension. The existing inference model can load the recommendation weights
without constructing the training-only projection head.

CPU regression checks (PyTorch, Transformers with Qwen3 support, PEFT, pandas,
PyArrow and tokenizer dependencies required):

```bash
python -m unittest discover -s LC-Rec-backbone/tests -v
```
