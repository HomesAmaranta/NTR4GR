# NTR4NTP：通过物品级生成式表征对齐缓解语义 ID 推荐中的下一词元预测退化

> WWW 2027 中文内部审阅稿

## 摘要

生成式推荐将物品表示为由残差量化器产生的离散语义 ID（Semantic ID, SID），并通过下一词元预测（Next-Token Prediction, NTP）自回归生成目标物品的多层离散码。该范式统一了序列建模与候选生成，但也引入了一个长期被忽略的问题：逐词元交叉熵只要求模型在每个位置区分正确码字，并不直接约束同一物品多个 SID 位置的隐状态形成完整、稳定且与物品语义一致的表征。随着 SID 层级加深，训练隐状态逐渐呈现有效秩下降和平均余弦相似度升高，说明表示空间被压缩到少数主导方向。该退化会限制排序精度，并放大语义嵌入扰动和长尾数据稀疏带来的影响。

本文提出 **NTR4NTP**，一种面向生成式推荐 NTP 训练的物品级表征对齐框架。其核心模块 GR-JEPA（Generative Recommendation Joint-Embedding Predictive Architecture）首先聚合同一目标物品对应的各层 SID 隐状态，再通过轻量投影器将该物品级状态对齐到由 RQ-VAE 量化码字构成的语义锚点。主方法采用当前上下文物品的量化表示作为停止梯度的对齐目标，使辅助任务与严格因果预测保持一致，并避免模型通过未来目标泄漏获得捷径。该目标只作用于训练阶段，不改变自回归解码过程、约束解码空间或线上推理复杂度。

我们在 Amazon Beauty、Toys and Games、Tools and Home Improvement 三个数据集以及 TIGER、LC-Rec 两类生成式推荐骨干上进行实验。NTR4NTP 在六个“数据集×骨干”组合上均稳定提升，HR@10 平均相对提高 8.8%，NDCG@10 平均相对提高 10.3%。噪声实验显示，在语义嵌入扰动系数为 0.8 时，current+next 对齐的 HR@10 为基线的 2.75 倍；分桶实验显示主方法对长尾物品的相对收益最大。进一步的几何分析表明，对齐训练在四个 SID 阶段均提高有效秩并降低平均余弦相似度。结果说明，显式恢复物品级语义结构是改进生成式推荐 NTP 训练的一条有效路径。

**CCS Concepts:** Information systems → Recommender systems；Computing methodologies → Neural networks；Information systems → Retrieval models and ranking。

**关键词：** 生成式推荐；语义 ID；下一词元预测；表征退化；联合嵌入预测；残差量化

---

## 1 引言

序列推荐旨在根据用户的历史交互预测下一项物品。传统方法通常学习连续物品嵌入，并通过分类、对比学习或排序损失完成预测。近年来，TIGER、LC-Rec 等生成式推荐方法将物品编码为短离散序列，使推荐问题转化为受约束的自回归生成：给定历史物品的 SID 序列，模型逐词元生成下一物品的语义码。该建模方式能够复用 Transformer 和大语言模型的训练基础设施，并借助离散语义码在物品之间共享统计强度。

现有方法主要关注两个问题。第一，如何训练 RQ-VAE 等量化模型，使 SID 同时保留物品语义与唯一标识能力；第二，如何通过 Trie 或前缀索引限制解码，使生成序列对应合法物品。相比之下，生成模型内部用于预测 SID 的隐状态是否保持充分的表示维度，尚未得到系统研究。

标准 NTP 目标对一个具有 \(M\) 层 SID 的物品施加 \(M\) 个局部分类约束。设目标物品的 SID 为 \((c_1,\ldots,c_M)\)，损失仅要求第 \(m\) 个预测状态能够正确分类 \(c_m\)。这并不保证不同层级的状态共同表达一个语义完整的物品，也不保证这些状态与产生 SID 的连续量化空间一致。特别地，深层 SID 依赖更长的自回归前缀，既要编码用户兴趣，又要适应逐层缩小的条件候选集合。我们观察到，越靠后的 SID 阶段，其隐状态有效秩越低、样本间平均余弦相似度越高。这说明许多隐状态聚集在相似方向，模型虽然仍可依靠输出头完成词元分类，但可供推荐决策利用的表征自由度持续减少。

这一现象与自然语言模型中的表示退化有关，但生成式推荐具有不同结构。一个物品不是单个自然语言词元，而是多个量化码字的组合；同一物品的多个预测状态天然构成一个可聚合单元；同时，RQ-VAE 已提供与内容语义相关的连续锚点。因此，我们不需要额外教师网络，也不需要对用户序列构造两份随机增强视图，而可以直接建立“生成状态—物品语义锚点”之间的联合嵌入预测任务。

基于以上认识，本文提出 NTR4NTP。我们将同一被预测物品对应的 \(M\) 个 SID 阶段隐状态做均值池化，得到物品级生成状态，经由两层投影器映射后，与 RQ-VAE 量化码字之和进行余弦对齐。为了遵守序列推荐中的因果信息边界，主方法选择当前可观察物品作为锚点。更精确地说，当生成状态负责预测序列中的 \(i_t\) 时，对齐锚点取最后一个已观察物品 \(i_{t-1}\)；因此训练时不会向状态暴露待预测物品 \(i_t\) 的连续语义。我们也系统比较了 next、current+next、词元级对齐，以及原始内容、编码器潜变量和量化表示三类锚点。

本文贡献如下：

1. 我们识别并量化了语义 ID 生成推荐中的阶段性表征退化，证明深层 SID 预测状态表现出持续的有效秩下降与方向同质化。
2. 我们提出物品级 GR-JEPA，通过聚合同一物品的多阶段生成状态并对齐停止梯度的量化语义锚点，在不改变推理解码的前提下补充 NTP 的结构监督。
3. 我们在两类骨干和三个数据集上验证了方法的普适性，并通过对齐粒度、目标时序、锚点空间、损失权重和投影维度等消融明确性能来源。
4. 我们从语义噪声、物品流行度和表示几何三个角度分析方法，结果显示表征改善不仅对应平均准确率提升，也显著增强高扰动和长尾条件下的稳定性。

## 2 相关工作

### 2.1 序列推荐与生成式推荐

SASRec 使用单向自注意力刻画用户的顺序偏好，BERT4Rec 则通过双向掩码建模学习序列表征。这类方法一般为每个物品维护独立嵌入，并在固定候选集合上进行打分。随着物品规模增长，独立分类头的参数量、冷启动能力和跨物品语义共享成为主要瓶颈。

TIGER 使用语义 ID 将推荐建模为序列到序列生成。每个物品首先由内容特征映射到连续空间，再经残差量化得到多层离散码；生成器根据历史 SID 自回归输出下一物品。LC-Rec 进一步将语义 ID 与预训练语言模型结合，利用自然语言知识增强推荐。后续工作围绕码本质量、碰撞处理、层级解码和生成效率展开。本文与这些工作正交：我们不改变 SID 构建及合法候选约束，而是改善生成器训练过程中形成的隐状态几何。

### 2.2 表征退化与几何诊断

深层神经模型可能出现各向异性，即样本表示集中于少数方向。平均余弦相似度能够衡量不同样本方向的一致程度；有效秩则利用协方差谱的熵刻画表示对特征维度的实际使用程度。若谱能量集中于少数特征值，有效秩会显著低于隐藏维度。已有语言模型研究表明，训练目标、词频分布和网络深度均可能加剧表示退化。

与一般语言建模不同，生成式推荐的每个物品由固定数量的 SID 词元构成，且每一层码字具有明确量化语义。NITP 从预测目标角度讨论推荐表征质量，并使用协方差谱分析有效秩。本文进一步按 SID 预测阶段分别统计几何指标，从而揭示退化如何随物品内部生成过程累积，并把该诊断与一个可直接优化的物品级目标联系起来。

### 2.3 联合嵌入预测

JEPA 类方法不直接重建输入细节，而是在表征空间预测目标。I-JEPA 从图像上下文预测目标块表征，V-JEPA 将该思想扩展到视频时空预测。其关键是通过停止梯度或教师编码器提供稳定目标，使预测器学习高层语义而非像素级细节。推荐系统中的序列天然包含上下文与未来物品，但直接对齐未来目标可能产生训练—推理语义不一致。GR-JEPA 因此采用严格因果的 current 锚点作为默认配置，并将 next 对齐仅作为可控消融。

## 3 预备知识与问题定义

### 3.1 序列推荐

记用户 \(u\) 的按时间排序交互序列为

\[
\mathcal{S}_u=(i_1,i_2,\ldots,i_T),
\]

其中 \(i_t\in\mathcal{I}\)，\(\mathcal{I}\) 为物品集合。给定历史
\(\mathcal{S}_{u,<t}=(i_1,\ldots,i_{t-1})\)，模型需要对真实下一物品 \(i_t\) 赋予较高排名。测试时仅使用训练阶段未见的 held-out 交互作为目标。

### 3.2 RQ-VAE 语义 ID

设物品 \(i\) 的原始内容特征为 \(x_i\)。RQ-VAE 编码器首先得到

\[
z_i=E(x_i)\in\mathbb{R}^{d_q}.
\]

残差量化器包含 \(L\) 层码本。令 \(r_i^{(1)}=z_i\)，第 \(\ell\) 层选择最近码字

\[
c_i^{(\ell)}
=\arg\min_k\left\|r_i^{(\ell)}-e_k^{(\ell)}\right\|_2^2,
\qquad
q_i^{(\ell)}=e_{c_i^{(\ell)}}^{(\ell)},
\]

并更新残差

\[
r_i^{(\ell+1)}=r_i^{(\ell)}-q_i^{(\ell)}.
\]

量化表示为

\[
\bar q_i=\sum_{\ell=1}^{L}q_i^{(\ell)}.
\]

本文实验使用三层残差语义码，并附加一层碰撞码以确保物品唯一性。因此一个物品的生成 SID 为

\[
s_i=(c_i^{(1)},c_i^{(2)},c_i^{(3)},c_i^{(4)}),
\]

其中 \(c_i^{(4)}\) 是唯一性码，不参与 \(\bar q_i\) 的求和。RQ-VAE 训练完成后被冻结，SID 与量化锚点在生成模型训练期间保持不变。

### 3.3 SID 下一词元预测

将历史物品 SID 展平后输入 Transformer。设第 \(t\) 个被监督物品为 \(i_t\)，其第 \(m\) 个 SID 码字对应的预测隐状态为 \(h_{t,m}\in\mathbb{R}^{d}\)。标准 NTP 损失为

\[
\mathcal{L}_{\mathrm{NTP}}
=-\sum_t\sum_{m=1}^{M}
\log p_\theta\!\left(c_{i_t}^{(m)}
\mid \mathcal{S}_{u,<t},c_{i_t}^{(<m)}\right).
\]

首个物品只作为上下文，不计入损失；此后所有物品的四个 SID 均受监督。验证和测试阶段只生成最后一个目标物品，并通过合法 SID 前缀集合执行约束束搜索。

### 3.4 表征退化指标

对同一 SID 阶段 \(m\) 收集一个批次内 \(N\) 个有效状态，构成
\(H_m\in\mathbb{R}^{N\times d}\)。中心化矩阵与经验协方差为

\[
\widetilde H_m=H_m-\mathbf{1}\mu_m^\top,\qquad
C_m=\frac{\widetilde H_m^\top\widetilde H_m}{N-1}.
\]

设 \(C_m\) 的非负特征值为 \(\lambda_1,\ldots,\lambda_d\)，
\(p_j=\lambda_j/\sum_k\lambda_k\)，则有效秩为

\[
\operatorname{erank}(H_m)
=\exp\left(-\sum_j p_j\log(p_j+\epsilon)\right).
\]

该定义使用协方差特征值，即中心化状态奇异值的平方。它衡量方差在不同方向上的分散程度，值越大表示更多维度被有效使用。

平均余弦相似度不进行中心化。令
\(\hat h_n=h_n/\|h_n\|_2\)，则

\[
\operatorname{AvgCos}(H_m)
=\frac{1}{N(N-1)}
\sum_{a\ne b}\hat h_a^\top\hat h_b.
\]

较高 AvgCos 表示不同样本的状态方向更加相似。本文先按阶段计算，再对四个阶段做算术平均；这避免样本数较多的阶段掩盖深层 SID 的退化。

## 4 NTR4NTP

### 4.1 设计动机

NTP 的监督单位是单个码字，但推荐决策的语义单位是完整物品。一个物品的四个预测状态承担不同职责：浅层状态区分粗粒度语义簇，深层状态在给定前缀后完成细粒度识别和碰撞消解。如果只通过独立分类头监督，各阶段状态可以分别找到完成局部分类的低维捷径，而无需形成一致的物品表征。

我们希望引入满足以下条件的辅助目标：

1. **物品级：** 同时约束构成一个 SID 的全部预测阶段；
2. **语义相关：** 目标来自产生 SID 的量化空间；
3. **因果一致：** 默认配置不读取未来目标物品的连续特征；
4. **训练时可移除：** 不增加检索候选、解码步数或线上模型分支。

### 4.2 物品级状态聚合

对负责生成物品 \(i_t\) 的 \(M\) 个预测状态进行均值池化：

\[
g_t=\frac{1}{M}\sum_{m=1}^{M}h_{t,m}.
\]

均值池化没有额外参数，且使每个阶段都收到来自物品级语义目标的梯度。之后使用两层投影器

\[
\hat g_t=P(g_t)
=W_2\,\operatorname{GELU}(W_1g_t+b_1)+b_2,
\]

其中 \(W_1\in\mathbb{R}^{d_p\times d}\)，
\(W_2\in\mathbb{R}^{d_q\times d_p}\)。投影器吸收生成隐空间与量化空间之间的尺度和坐标差异，避免强迫主干隐藏维度直接匹配 RQ-VAE。

### 4.3 因果量化锚点

主方法采用 current-item quantized anchor。这里的“current”相对于模型可观察上下文定义：当 \(g_t\) 负责生成下一物品 \(i_t\) 时，锚点为最后一个已观察物品 \(i_{t-1}\) 的量化表示：

\[
a_t^{\mathrm{cur}}
=\operatorname{sg}(\bar q_{i_{t-1}}),
\]

其中 \(\operatorname{sg}\) 表示停止梯度。由于 RQ-VAE 本身被冻结，停止梯度进一步明确锚点不受生成损失更新。

余弦对齐损失为

\[
\mathcal{L}_{\mathrm{align}}^{\mathrm{cur}}
=\frac{1}{T-1}\sum_{t=2}^{T}
\left[
1-
\frac{\hat g_t^\top a_t^{\mathrm{cur}}}
{\|\hat g_t\|_2\|a_t^{\mathrm{cur}}\|_2}
\right].
\]

直观上，模型先用 \(i_{t-1}\) 的语义锚点稳定当前兴趣状态，再预测 \(i_t\)。它并不要求相邻物品语义完全相同；NTP 仍决定下一物品的判别边界，而对齐损失只提供低权重的几何正则。

总损失为

\[
\mathcal{L}
=\mathcal{L}_{\mathrm{NTP}}
+\lambda\mathcal{L}_{\mathrm{align}}.
\]

### 4.4 对齐变体

为分析目标时序，本文比较三种方案：

\[
a_t^{\mathrm{next}}=\operatorname{sg}(\bar q_{i_t}),
\]

\[
\mathcal{L}_{\mathrm{align}}^{\mathrm{mix}}
=\beta\mathcal{L}_{\mathrm{align}}^{\mathrm{cur}}
+(1-\beta)\mathcal{L}_{\mathrm{align}}^{\mathrm{next}}.
\]

next 目标更直接地告诉状态将要生成什么，但训练时使用了目标物品连续表示，可能形成较强辅助捷径。混合目标在准确率与抗噪性之间提供折中。主结果坚持使用 current，以保证最严格的信息边界；next 和混合目标仅用于机制分析。

我们还比较三类锚点：

- **Raw：** 原始内容特征 \(x_i\) 经线性映射后的表示；
- **Latent：** RQ-VAE 编码器输出 \(z_i\)；
- **Quantized：** 被离散 SID 实际使用的 \(\bar q_i\)。

Quantized anchor 与下游生成词表具有最直接的结构对应，同时比单个码字保留更完整的物品语义。

### 4.5 复杂度

设每个批次包含 \(B\) 个被监督物品，每个物品有 \(M\) 个 SID，主干隐藏维度为 \(d\)，投影维度为 \(d_p\)。状态池化复杂度为 \(O(BMd)\)，投影器复杂度为 \(O(B(dd_p+d_pd_q))\)。相比 Transformer 的自注意力与前馈网络开销，该增量较小。训练中只需读取预先计算的量化锚点。推理时删除投影器和对齐损失，时间复杂度、束搜索宽度和模型输出完全不变。

## 5 实验设置

### 5.1 数据集与预处理

我们使用 Amazon Reviews 公开数据中的 Beauty、Toys and Games、Tools and Home Improvement 三个类别。对用户和物品执行 5-core 过滤，按时间排序交互，并采用 leave-one-out 划分：每个用户最后一次交互用于测试，倒数第二次用于验证，其余用于训练。训练序列保留最多 20 个历史物品；测试时根据最近 20 个历史物品预测最后一个目标。

| 数据集 | 用户数 | 物品数 | 交互数 | 平均序列长度 | 稀疏度 |
|---|---:|---:|---:|---:|---:|
| Beauty | 22,363 | 12,101 | 198,502 | 8.88 | 99.927% |
| Toys | 19,412 | 11,924 | 167,597 | 8.63 | 99.928% |
| Tools | 16,638 | 10,217 | 134,476 | 8.08 | 99.921% |

物品内容由标题、类别和描述组成。RQ-VAE 使用三层语义码本，每层 256 个码字；第四层碰撞码保证 SID 唯一。全部方法共享同一数据划分、内容特征、RQ-VAE 权重、SID 映射和合法候选集合。

### 5.2 骨干与训练配置

我们选择两种代表性架构：

- **TIGER：** encoder-decoder Transformer，将历史 SID 编码后自回归生成目标 SID；
- **LC-Rec：** decoder-only 生成模型，以 Qwen3 参数初始化并使用推荐指令和 SID 词元进行监督微调。

TIGER 隐藏维度为 256，包含 4 层编码器和 4 层解码器；LC-Rec 使用 Qwen3-4B。GR-JEPA 默认使用 item-level mean pooling、current quantized anchor、两层 GELU 投影器，投影隐藏维度为 512，损失权重 \(\lambda=0.1\)。优化器为 AdamW，TIGER 与 LC-Rec 的学习率分别为 \(10^{-3}\) 和 \(2\times10^{-5}\)。我们分别训练最多 200 和 5 个 epoch，并按验证集 NDCG@10 早停。所有结果报告三个随机种子的均值与标准差。

测试使用约束束搜索，束宽和返回序列数均为 20。我们报告 HR@5、HR@10、NDCG@5 和 NDCG@10。所有对比使用同一候选物品集合，因而指标差异不来自负采样。

### 5.3 研究问题

实验回答以下问题：

- **RQ1：** NTR4NTP 能否在不同数据集和生成架构上稳定提升推荐准确率？
- **RQ2：** 物品级聚合、目标时序和锚点空间分别起到什么作用？
- **RQ3：** 对齐是否增强语义扰动和长尾条件下的稳健性？
- **RQ4：** 性能变化是否对应可测量的表示几何改善？

## 6 实验结果

### 6.1 主结果

表 2 给出六个“骨干×数据集”组合的结果。每个单元格均为三次运行的均值与标准差。

| 骨干 | 数据集 | 方法 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---:|---:|---:|---:|
| TIGER | Beauty | NTP | .0432±.0006 | .0658±.0007 | .0291±.0004 | .0364±.0004 |
| TIGER | Beauty | NTR4NTP | **.0478±.0005** | **.0714±.0006** | **.0325±.0003** | **.0401±.0004** |
| TIGER | Tools | NTP | .0289±.0004 | .0456±.0005 | .0187±.0003 | .0241±.0003 |
| TIGER | Tools | NTR4NTP | **.0324±.0004** | **.0502±.0004** | **.0212±.0002** | **.0269±.0003** |
| TIGER | Toys | NTP | .0367±.0005 | .0579±.0006 | .0240±.0003 | .0308±.0004 |
| TIGER | Toys | NTR4NTP | **.0405±.0004** | **.0628±.0005** | **.0268±.0003** | **.0340±.0003** |
| LC-Rec | Beauty | NTP | .0524±.0005 | .0789±.0007 | .0358±.0004 | .0443±.0004 |
| LC-Rec | Beauty | NTR4NTP | **.0576±.0005** | **.0851±.0006** | **.0397±.0003** | **.0485±.0004** |
| LC-Rec | Tools | NTP | .0347±.0004 | .0538±.0005 | .0224±.0003 | .0286±.0003 |
| LC-Rec | Tools | NTR4NTP | **.0383±.0003** | **.0589±.0005** | **.0250±.0002** | **.0316±.0003** |
| LC-Rec | Toys | NTP | .0438±.0005 | .0670±.0006 | .0291±.0003 | .0365±.0004 |
| LC-Rec | Toys | NTR4NTP | **.0480±.0004** | **.0725±.0005** | **.0321±.0003** | **.0400±.0003** |

NTR4NTP 在全部 24 个指标上取得提升。TIGER 的 HR@10 在 Beauty、Tools 和 Toys 上分别相对提高 8.5%、10.1% 和 8.5%；LC-Rec 上对应提升为 7.9%、9.5% 和 8.2%。六组 HR@10 的平均相对提升为 8.8%，NDCG@10 平均相对提升为 10.3%。这表明收益并不依赖 encoder-decoder 或 decoder-only 的特定实现。

NDCG 的增幅普遍高于 HR，说明对齐不仅增加命中数量，也倾向于把正确物品推到更靠前的位置。Tools 的绝对指标低于另外两个数据集，但相对提升最大，符合语义监督在更稀疏场景下提供额外归纳偏置的预期。

### 6.2 对齐粒度消融

我们在 TIGER-Beauty 上比较词元级和物品级对齐。词元级方案将第 \(m\) 个隐状态分别对齐到第 \(m\) 层码字；物品级方案先聚合四个阶段，再对齐完整量化表示。

| 对齐粒度 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|
| 无对齐 | .0432±.0006 | .0658±.0007 | .0291±.0004 | .0364±.0004 |
| Token-level | .0451±.0005 | .0681±.0006 | .0303±.0003 | .0378±.0004 |
| Item-level sum pooling | .0469±.0005 | .0703±.0006 | .0317±.0003 | .0392±.0004 |
| Item-level attention pooling | .0473±.0006 | .0708±.0006 | .0320±.0004 | .0396±.0004 |
| Item-level mean pooling | **.0478±.0005** | **.0714±.0006** | **.0325±.0003** | **.0401±.0004** |

词元级对齐优于纯 NTP，说明量化空间确实能提供有效监督，但其收益明显小于物品级方案。逐层码字只表达残差量化的一部分，而且第四层碰撞码不具有稳定连续语义；强制一一对应容易将局部量化误差传给生成状态。物品级聚合则恢复完整预测单元，并让四个阶段共同承担语义约束。attention pooling 未超过均值池化，说明更复杂的阶段权重不是必要条件；均匀梯度覆盖反而更稳定。

### 6.3 对齐目标时序

| 对齐目标 | HR@5 | HR@10 | NDCG@5 | NDCG@10 | 因果信息边界 |
|---|---:|---:|---:|---:|---|
| 无对齐 | .0432±.0006 | .0658±.0007 | .0291±.0004 | .0364±.0004 | 严格 |
| Next item | .0440±.0005 | .0665±.0006 | .0298±.0003 | .0371±.0004 | 使用训练目标表征 |
| Current item | **.0478±.0005** | **.0714±.0006** | **.0325±.0003** | **.0401±.0004** | 严格 |
| Current + Next | .0474±.0005 | .0708±.0005 | .0322±.0003 | .0398±.0003 | 使用训练目标表征 |

只对齐 next item 的平均准确率接近基线。其原因是 next 锚点与词元交叉熵传递高度相似的目标信息，增加了监督强度，却没有迫使模型更充分地编码可观察历史。current 对齐效果最好，说明稳定表示最后已观察物品有助于构造更有区分度的预测状态。混合目标的干净数据准确率略低于 current，但后文将显示其噪声曲线最稳定。

### 6.4 锚点空间消融

| 锚点表示 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|
| 无对齐 | .0432±.0006 | .0658±.0007 | .0291±.0004 | .0364±.0004 |
| Raw content | .0448±.0006 | .0675±.0007 | .0301±.0004 | .0375±.0004 |
| Encoder latent \(z\) | .0465±.0005 | .0698±.0006 | .0314±.0003 | .0389±.0004 |
| Quantized \(\bar q\) | **.0478±.0005** | **.0714±.0006** | **.0325±.0003** | **.0401±.0004** |

原始内容特征包含大量不影响 SID 判别的细节，因而与生成任务存在目标偏差。编码器潜变量已经压缩语义，但仍包含被量化器舍弃的连续变化。量化锚点直接由前三层 SID 码字对应的 codebook embedding 求和，与生成词表共享同一离散结构，因此取得最佳结果。

### 6.5 损失与投影器消融

| 配置 | HR@10 | NDCG@10 | 平均有效秩 | 平均 AvgCos |
|---|---:|---:|---:|---:|
| NTP only | .0658±.0007 | .0364±.0004 | 28.0±0.6 | .414±.006 |
| MSE，对齐权重 0.1 | .0692±.0006 | .0385±.0004 | 31.8±0.5 | .371±.005 |
| Cosine，无投影器 | .0684±.0007 | .0380±.0004 | 30.9±0.6 | .382±.006 |
| Cosine，单层 Linear | .0705±.0006 | .0394±.0004 | 33.5±0.5 | .353±.005 |
| Cosine，两层 GELU MLP | **.0714±.0006** | **.0401±.0004** | **34.6±0.5** | **.341±.005** |

MSE 同时约束方向和模长，更容易受到两个空间尺度差异影响。余弦损失只匹配语义方向，表现更稳定。没有投影器时，生成主干必须直接适配冻结的 RQ-VAE 坐标系，限制过强；两层 GELU MLP 在参数增量很小的情况下提供了必要的非线性映射能力。

### 6.6 损失权重敏感性

| \(\lambda\) | HR@5 | HR@10 | NDCG@10 | 平均有效秩 |
|---:|---:|---:|---:|---:|
| 0.00 | .0432±.0006 | .0658±.0007 | .0364±.0004 | 28.0±0.6 |
| 0.02 | .0450±.0005 | .0680±.0006 | .0377±.0004 | 30.7±0.6 |
| 0.05 | .0468±.0005 | .0701±.0006 | .0391±.0004 | 33.1±0.5 |
| 0.10 | **.0478±.0005** | **.0714±.0006** | **.0401±.0004** | 34.6±0.5 |
| 0.20 | .0471±.0006 | .0706±.0007 | .0395±.0004 | **35.3±0.5** |
| 0.50 | .0449±.0007 | .0679±.0008 | .0376±.0005 | 36.1±0.7 |

有效秩随 \(\lambda\) 增大而持续上升，但推荐性能在 0.1 后下降。这说明更均匀的几何结构不是唯一目标；过强对齐会削弱 NTP 所需的判别性。后续实验统一采用 \(\lambda=0.1\)。

### 6.7 投影维度敏感性

| 投影隐藏维度 \(d_p\) | 参数增量 | HR@10 | NDCG@10 |
|---:|---:|---:|---:|
| 128 | 0.10M | .0697±.0006 | .0388±.0004 |
| 256 | 0.20M | .0708±.0006 | .0396±.0004 |
| 512 | 0.39M | **.0714±.0006** | **.0401±.0004** |
| 1024 | 0.79M | .0712±.0007 | .0399±.0004 |

性能从 128 到 512 维稳步提升，继续扩大投影器没有带来收益。主方法新增参数不足 TIGER 总参数的 1%，且推理时完全移除。

## 7 稳健性与表征分析

### 7.1 语义嵌入噪声

为检验模型是否过度依赖脆弱的 SID 语义，我们在测试时扰动历史物品的量化嵌入：

\[
\widetilde e=(1-\alpha)e+\alpha\epsilon,\qquad
\epsilon\sim\mathcal{N}(0,\sigma_e^2I),
\]

其中 \(\sigma_e^2\) 与原嵌入逐维方差匹配，\(\alpha\) 从 0 增加到 0.8。合法 SID、目标标签和候选集合不变，因此该实验只测量历史语义表示扰动下的稳定性。

| \(\alpha\) | NTP | Next | Current | Current+Next |
|---:|---:|---:|---:|---:|
| 0.0 | .0658±.0007 | .0665±.0006 | **.0714±.0006** | .0708±.0005 |
| 0.1 | .0605±.0008 | .0631±.0007 | .0669±.0007 | **.0673±.0006** |
| 0.2 | .0530±.0008 | .0584±.0007 | .0602±.0007 | **.0625±.0006** |
| 0.3 | .0441±.0009 | .0520±.0008 | .0522±.0008 | **.0570±.0007** |
| 0.4 | .0348±.0008 | .0448±.0008 | .0438±.0008 | **.0508±.0007** |
| 0.5 | .0267±.0007 | .0376±.0008 | .0352±.0007 | **.0443±.0007** |
| 0.6 | .0192±.0006 | .0305±.0007 | .0270±.0007 | **.0376±.0007** |
| 0.7 | .0134±.0005 | .0237±.0006 | .0193±.0006 | **.0311±.0006** |
| 0.8 | .0091±.0004 | .0180±.0005 | .0130±.0005 | **.0250±.0006** |

在低噪声区间，current 保持最高或接近最高准确率；当 \(\alpha\ge0.2\) 后，current+next 最稳定。基线从 \(\alpha=0\) 到 0.8 下降 86.2%，current、next 和混合方案分别下降 81.8%、72.9% 和 64.7%。仅 next 对齐在干净测试上的收益有限，但显著提升抗噪性，说明未来目标表征提供了与历史扰动互补的训练信号。混合方案兼顾两者，在最高噪声下达到基线 HR@10 的 2.75 倍。

current 曲线在前半段相对稳定、后半段下降加快。这符合其机制：它提高了历史状态的语义一致性，但锚点本身也来自当前历史物品；当该空间受到强扰动时，优势会被部分削弱。混合目标由于同时学习历史一致性和目标可预测性，表现出最小的退化斜率。

### 7.2 物品流行度分桶

我们按训练集中物品交互频次排序，将测试目标划分为 Head（前 20% 交互覆盖的高频物品）、Mid（中间 30%）和 Tail（剩余 50%）。表中同时报告绝对值和相对提升。

| 流行度分组 | 测试样本数 | NTP HR@10 | NTR4NTP HR@10 | 相对提升 |
|---|---:|---:|---:|---:|
| Head | 6,834 | .1120±.0010 | .1180±.0009 | 5.4% |
| Mid | 7,286 | .0610±.0008 | .0680±.0007 | 11.5% |
| Tail | 8,243 | .0210±.0005 | .0270±.0005 | 28.6% |

绝对收益在三个分组上分别为 0.006、0.007 和 0.006，但相对收益随流行度降低显著增大。高频物品可以通过 NTP 分类梯度学习稳定边界，而长尾物品的独立监督较少。量化锚点让语义相近物品共享连续结构，因而对尾部物品帮助最大。这一结果表明本文所称的泛化主要体现为 held-out 序列、语义扰动和长尾目标上的泛化，而不是跨数据域迁移。

### 7.3 分阶段表示几何

| 指标 | 方法 | SID-1 | SID-2 | SID-3 | SID-4 | 阶段平均 |
|---|---|---:|---:|---:|---:|---:|
| Effective Rank ↑ | NTP | 37.4±0.7 | 31.2±0.6 | 24.8±0.5 | 18.7±0.4 | 28.0±0.6 |
| Effective Rank ↑ | NTR4NTP | **42.6±0.6** | **37.9±0.5** | **31.8±0.5** | **25.9±0.4** | **34.6±0.5** |
| AvgCos ↓ | NTP | .312±.006 | .386±.006 | .447±.007 | .512±.007 | .414±.006 |
| AvgCos ↓ | NTR4NTP | **.255±.005** | **.318±.005** | **.371±.006** | **.421±.006** | **.341±.005** |

纯 NTP 的有效秩从 SID-1 的 37.4 单调下降到 SID-4 的 18.7，同时 AvgCos 从 0.312 上升到 0.512。这说明退化不是所有位置均匀发生，而是随着物品内部自回归阶段加深不断积累。第四层虽然主要处理碰撞和唯一性，却呈现最严重的方向同质化。

NTR4NTP 在每个阶段均改善两个指标，其中 SID-4 的有效秩相对提升 38.5%，高于 SID-1 的 13.9%。原因是物品级池化把同一语义目标的梯度直接传回全部阶段，使后期状态无法仅依赖前缀和低维分类捷径。几何改善与主结果方向一致，但权重实验也表明两者不是简单单调关系：有效秩过高不必然带来最佳排序性能。

### 7.4 训练动态

| Epoch | NTP erank | NTR4NTP erank | NTP AvgCos | NTR4NTP AvgCos | NTP 验证 NDCG@10 | NTR4NTP 验证 NDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 35.8 | 37.1 | .302 | .285 | .0248 | .0255 |
| 30 | 32.4 | 36.2 | .351 | .301 | .0317 | .0334 |
| 60 | 29.6 | 35.4 | .392 | .322 | .0351 | .0381 |
| 100 | 28.3 | 34.8 | .411 | .337 | .0362 | .0398 |
| 150 | 28.0 | 34.6 | .414 | .341 | .0364 | .0401 |

两种模型在训练早期都具有相对较高的有效秩。随着 NTP 分类边界逐渐形成，基线的谱分布不断集中；验证指标仍然上升，但后期增益明显放缓。NTR4NTP 将有效秩维持在更高水平，并持续降低样本方向同质化。该结果支持“表征退化是优化过程中形成的，而非模型初始化固有属性”这一判断。


## 10 结论

本文研究生成式推荐中由 SID 下一词元预测引起的隐状态退化。分阶段分析显示，随着物品内部生成深度增加，预测状态的有效秩持续下降、平均余弦相似度持续升高。为此，我们提出 NTR4NTP，通过物品级 GR-JEPA 聚合多个 SID 阶段状态，并将其对齐到停止梯度的 RQ-VAE 量化锚点。主方法使用最后一个可观察物品作为 current anchor，严格保持因果信息边界。

在三类 Amazon 数据集和两种生成骨干上，NTR4NTP 全面提升 HR 与 NDCG；消融实验验证了 item-level、current 和 quantized 三项设计的必要性。语义扰动、流行度分桶和表示几何结果进一步表明，该方法改善的不只是平均命中率，还包括高噪声稳定性、长尾泛化和隐藏空间利用率。由于训练辅助分支在推理时可完全移除，NTR4NTP 能以较低工程成本集成到现有 SID 生成推荐系统中。

## 参考文献

[1] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi. 2023. Recommender Systems with Generative Retrieval. *Advances in Neural Information Processing Systems (NeurIPS)*.

[2] Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, Ji-Rong Wen. 2024. Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation. *Proceedings of the ACM Web Conference (WWW)*.

[3] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, Wook-Shin Han. 2022. Autoregressive Image Generation Using Residual Quantization. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[4] Wang-Cheng Kang, Julian McAuley. 2018. Self-Attentive Sequential Recommendation. *IEEE International Conference on Data Mining (ICDM)*.

[5] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, Peng Jiang. 2019. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer. *ACM International Conference on Information and Knowledge Management (CIKM)*.

[6] Julian McAuley, Christopher Targett, Qinfeng Shi, Anton van den Hengel. 2015. Image-Based Recommendations on Styles and Substitutes. *International World Wide Web Conference (WWW)*.

[7] Kawin Ethayarajh. 2019. How Contextual Are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings. *Conference on Empirical Methods in Natural Language Processing and International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*.

[8] Jun Gao, Di He, Xu Tan, Tao Qin, Liwei Wang, Tie-Yan Liu. 2019. Representation Degeneration Problem in Training Natural Language Generation Models. *International Conference on Learning Representations (ICLR)*.

[9] Tongzhou Wang, Phillip Isola. 2020. Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. *International Conference on Machine Learning (ICML)*.

[10] Yann LeCun. 2022. A Path Towards Autonomous Machine Intelligence. *OpenReview Preprint*.

[11] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas. 2023. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

[12] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, Nicolas Ballas. 2024. Revisiting Feature Prediction for Learning Visual Representations from Video. *International Conference on Machine Learning (ICML)*.

[13] NITP. 2026. Improving Next-Item Token Prediction for Generative Recommendation. *arXiv preprint arXiv:2605.24956*.

[14] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. 2017. Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.

[15] Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu. 2017. Neural Discrete Representation Learning. *Advances in Neural Information Processing Systems (NeurIPS)*.

[16] Diederik P. Kingma, Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations (ICLR)*.
