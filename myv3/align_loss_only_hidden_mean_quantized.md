# only-hidden + mean + quantized Align Loss 说明

本文说明当前代码中 `align_target=quantized`、`mse_loss_mode=only-hidden` 时 auxiliary align loss 的计算方式。

这里的 `mean` 指最终 loss 使用 `mean` reduction，即对有效样本和 embedding 维度求平均。如果 `mean` 指的是 `mse_loss_mode=mean`，则它和 `only-hidden` 是两种不同的 hidden 取法，区别见文末。

## 配置含义

典型配置如下：

```bash
--align_target quantized
--mse_loss_mode only-hidden
--align_loss_type mse
--align_item current
--align_current_k 1
--item_emb_path ../data/${dataset}/item_emb_rqvae_quantized_latent.parquet
```

各参数含义：

- `align_target=quantized`：对齐目标来自 RQ-VAE 的 quantized latent 表征。
- `mse_loss_mode=only-hidden`：每个 item 只取该 item 第 1 个 code 位置的 hidden state 做对齐。
- `align_loss_type=mse`：使用 MSE align loss；如果配置为 `cos`，则使用 cosine distance。
- `align_item=current`：hidden 对齐当前输入侧 item，而不是 CE 要预测的 next item。
- `align_current_k=1`：取当前 item；如果为 `2/3/...`，则取更前面的第 k 个 current-side item。

## 训练中的序列关系

并行训练时，每个 item 有 `code_per_item=4` 个 code token。训练数据会整体 shift 一个 item：

```text
输入 item:     a      b      c      d
CE 预测目标:   b      c      d      e
item_group:   b      c      d      e
```

`item_group` 在代码中标的是 CE 的预测目标，也就是 next item。

当 `align_item=current` 且 `align_current_k=1` 时：

```text
align target index = item_group - 1

输入 hidden:   a      b      c      d
对齐目标:      a      b      c      d
```

当 `align_current_k=2` 时：

```text
align target index = item_group - 2

输入 hidden:   a      b      c      d
对齐目标:      x      a      b      c
```

其中 `x` 表示没有物品，这个位置不计算 align loss，但 CE loss 仍然正常计算。

## Hidden 的取法

模型内部会先拿到每个 item 的 4 个 code hidden：

```text
phase_hidden_states shape = [B, item_len, 4, d_model]
```

在 `mse_loss_mode=only-hidden` 下，不会对 4 个 code hidden 求平均，而是只取第 0 个 code hidden：

```python
pooled_hidden = phase_hidden[:, :, 0]
```

因此每个 item 参与 align 的 hidden 是：

```text
h_i = hidden_states[item_i, code_0]
```

不是：

```text
mean(hidden_states[item_i, code_0:code_3])
```

## Quantized Target 的取法

`align_target=quantized` 时，脚本会使用：

```bash
../data/${dataset}/item_emb_rqvae_quantized_latent.parquet
```

dataset 会根据 item id 取出对应的 quantized latent：

```text
q_i = quantized_embedding(item_i)
```

在 parallel 模式中，block 内每个 item 都保存一份 quantized embedding。模型再根据 `item_group` 和 `align_item/current_k` 选择最终对齐目标。

## Projection Head

模型 hidden 维度通常是 `d_model`，而 quantized latent 维度是 `item_emb_dim`，当前常见为 `32`。

因此 align loss 不是直接比较 hidden 和 quantized latent，而是先经过一个投影层：

```python
pred_emb = hidden_to_item_emb(pooled_hidden)
```

形状为：

```text
pooled_hidden: [N, d_model]
pred_emb:      [N, item_emb_dim]
target_emb:    [N, item_emb_dim]
```

其中 `N` 是 batch 内所有有效 item 位置的数量。

## Loss 计算

如果 `align_loss_type=mse`，则：

```python
align_loss = F.mse_loss(pred_emb, target_emb, reduction="mean")
```

也就是：

```text
L_align = mean((W h_i - q_i)^2)
```

这里的 `mean` 会对所有有效 item 位置和 embedding 维度一起求平均。

如果 `align_loss_type=cos`，则：

```python
align_loss = 1 - cosine_similarity(pred_emb, target_emb).mean()
```

也就是：

```text
L_align = 1 - mean(cos(W h_i, q_i))
```

最终训练总 loss 是：

```text
L_total = L_ce + mse_loss_weight * L_align
```

## 有效位置 Mask

不是所有位置都会计算 align loss。

会计算 align loss 的位置需要满足：

- 该 item 位置不是 padding。
- 如果是 overlap block 的 context-only 区域，该位置需要有有效 `item_group`。
- `align_item=current` 时，`item_group - align_current_k` 不能小于 0。
- `align_item=current` 时，`item_group - align_current_k` 不能超过 block 中保存的 item embedding 数量。

例如：

```text
输入 hidden:      a      b      c      d
align_current_k=2
对齐目标:         x      a      b      c
是否算 align:     否     是     是     是
```

`a` 的 align 目标是 `x`，所以不算 align loss；但 `a -> b` 的 CE loss 仍然计算。

## 和 mse_loss_mode=mean 的区别

如果 `mse_loss_mode=mean`，hidden 取法会变成：

```python
pooled_hidden = phase_hidden.mean(dim=2)
```

也就是每个 item 的 4 个 code hidden 先求平均：

```text
h_i = mean(code_0_hidden, code_1_hidden, code_2_hidden, code_3_hidden)
```

然后再：

```text
L_align = mean((W h_i - q_i)^2)
```

所以：

- `only-hidden`：只取 item 的第 0 个 code hidden。
- `mean`：取 item 的 4 个 code hidden 的平均。
- 两者最终的 MSE 都是 `reduction="mean"`。

## 对应代码位置

- `main.py`：解析 `mse_loss_mode`、`align_loss_type`、`align_target`、`align_item`、`align_current_k`。
- `dataset.py`：加载 `item_emb_rqvae_quantized_latent.parquet`，并为 block 内 item 保存 quantized target。
- `causal_tiger.py`：在 `_item_level_align_loss` 中选择有效 item、取 hidden、投影并计算 align loss。
