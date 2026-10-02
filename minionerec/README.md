# 在 NTR4GR Beauty 数据上运行 MiniOneRec

该目录是一个独立的 MiniOneRec 适配版本，直接使用 NTR4GR 的 Beauty
数据划分、ItemID 映射、商品属性和四层 RQ-VAE SID，不会修改原始
`MiniOneRec` 仓库。

## 目录结构

- `sft/`：监督微调及约束解码评测
- `rl/`：面向推荐任务的强化学习
- `scripts/prepare_beauty.py`：确定性数据转换脚本
- `data/Beauty/`：生成的 MiniOneRec 输入数据

转换脚本读取 `../../data/Beauty`，并生成以下文件：

- `data/Beauty/train/Beauty.csv`
- `data/Beauty/valid/Beauty.csv`
- `data/Beauty/test/Beauty.csv`
- `data/Beauty/index/Beauty.index.json`
- `data/Beauty/index/Beauty.item.json`
- `data/Beauty/info/Beauty.txt`

每行四层码字按照以下规则转换：

```text
[c0, c1, c2, c3] -> <a_c0><b_c1><c_c2><d_c3>
```

训练数据按照 NTR4GR 的方式进行前缀滑窗展开，并只保留最近 20 个历史物品。
验证集和测试集保持原有的 leave-one-out 目标不变。

## 准备数据

```bash
python scripts/prepare_beauty.py
```

默认会对训练序列进行滑窗展开。只有明确需要“每个用户仅保留一条训练样本”时，
才使用 `--no-expand-train`：

```bash
python scripts/prepare_beauty.py --no-expand-train
```

## 监督微调

```bash
bash sft.sh
```

Python 环境、基础模型路径、GPU 数量和输出目录直接配置在根目录的 `sft.sh` 中，
无需通过命令行传入模型路径。

## 约束解码评测

```bash
bash eval.sh
```

待评估模型路径、Python 环境和 GPU 列表直接配置在根目录的 `eval.sh` 中。
当前默认评估 RL 输出的 `final_checkpoint`，并使用 GPU `0,1,2,3`。

## 强化学习

```bash
bash rl.sh
```

RL 使用的 SFT checkpoint 路径、Python 环境、GPU 数量和输出目录直接配置在
根目录的 `rl.sh` 中。
