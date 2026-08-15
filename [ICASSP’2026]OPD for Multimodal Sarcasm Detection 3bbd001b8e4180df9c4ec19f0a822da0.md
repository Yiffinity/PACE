# [ICASSP’2026]OPD for Multimodal Sarcasm Detection

Status: In progress

**Keeping PACE with Evolving Sarcasm: Progressive Adaptation through Curated Experience for Multimodal Sarcasm Detection**

PACE: Progressive Adaptation through Curated Experience

- Experience Curation and Evolution
    - Emerging Sarcasm Experience Construction
    - Consolidated Sarcasm Experience Construction
- On-Policy Experience Distillation

**待看论文**

- LEAF: Towards Lightweight Explainable Hateful Video Detection via Self-Grounding CoT Guided Stage-Wise Distillation
- lang alarm

# 故事

本文将每批已标注数据视为可持续复用的经验来源，而非一次性监督数据，通过跨样本验证沉淀可靠的讽刺判断知识，再利用 on-policy distillation 将其内化到轻量模型中，使历史标注产生的知识能够持续服务于后续尚未完成标注的数据批次，从而缓解实际社交媒体场景中数据快速产生与高质量人工标注滞后之间的矛盾。

# 方法

```python
Labeled Historical Samples
          ↓
 Experience Extraction
          ↓
   Emerging Sarcasm Experience e
          ↓
 ┌─────────────────────────┐
 │ Experience Utility Test │
 │                         │
 │ Relevant Bank: 64       │
 │ Global Bank:   64       │
 └────────────┬────────────┘
              ↓
 S_relevant > 0
       AND
 S_global ≥ 0 ?
       /       \
     Yes       No
      ↓         ↓
    Keep      Discard
      ↓
Filtered Experience Memory
      ↓
 Experience Accumulation
      ↓
   OPD Consolidation
```

## 模块 A：Experience Curation and Evolution

冻结 MLLM 参数，维护Emerging Sarcasm Experience和Consolidated Sarcasm Experience。模型根据带标签数据流产生新 experience，并首先存入待选经验池。每条候选 experience 在独立的 held-out validation samples 上进行效用验证，只有能够提高模型正确标签偏好、且不损害整体预测性能的 experience 才转移到持久经验池。

### 问题定义

给定图文样本：$x_i=(I_i,T_i)$,

以及标签：$y_i\in\{\text{sarcastic},\text{non-sarcastic}\}$.

教师 MLLM 记为：$\pi_T$.

学生多模态模型记为：$\pi_S$.

定义两个经验池：$\mathcal P_c:\text{Emerging Sarcasm Experience }$,

$\mathcal P_p:\text{Consolidated Sarcasm Experience}$.

其中：

$\mathcal P_c$存储刚刚生成、尚未经过独立效用验证的 Emerging Sarcasm Experience；

$\mathcal P_p$存储已经在来源样本之外表现出正向迁移作用的 validated experience。

experience 只能从：$\mathcal P_c\rightarrow\mathcal P_p$单向转移。

持久经验一旦通过验证进入 $\mathcal P_p$，即作为后续正式推理和 OPED 使用的固定经验，不再参与候选经验筛选过程。

### Experience Item

experience 内容只保留一段短自然语言：$e=s_e=<适用现象> + <判断启示>$

推荐形式：

> <适用现象> + <判断启示>
> 

讽刺经验示例：

> 当文本使用明显的正面评价描述图像中的失败结果，并形成真实态度反转时，该表达倾向于构成讽刺。
> 

非讽刺经验示例：

> 图像与文本包含不同信息并不必然构成讽刺；如果不存在评价反转、期待违背或明确嘲讽对象，不应只凭图文差异判断为讽刺。
> 

### 教师输出格式（json形式）

无论经验生成还是后续蒸馏，教师和学生均使用同一简洁格式：

> <视觉证据> 与判断直接相关的图像事实
> 
> 
> <文本证据> 与判断直接相关的原始文本表达
> 
> <解释> 两类证据如何共同支持讽刺或非讽刺
> 
> <判断> 讽刺 / 非讽刺
> 

例如：

> “视觉证据”：这个样本的图片展示了…，（对xx进行了讽刺，如果没有就不讲）
> 
> 
> “文本证据“：这个样本的文本展示了…，（对xx进行了讽刺，如果没有就不讲）
> 
> “解释”：结合图片中的xx和文本中的xx，共同支持讽刺或非讽刺
> 
> “判断“： 讽刺 / 非讽刺
> 

### 推理与experience构成

**推理：**对于当前样本 ($x_i$)，将持久池中的所有经验$E$作为teacher model的上下文来进行推理判断当前样本是否讽刺。

**experience构成：**对于当前样本 ($x_i$)，将持久池中的所有经验$E$作为teacher model的上下文来进行推理判断当前样本是否讽刺$o_i=\pi_T(I_i,T_i,E)$。在预测被记录后揭示标签 ($y_i$)。当模型预测错误，或当前输出暴露出已有持久经验尚未覆盖的讽刺判断规律时，教师根据：

- 当前图像和文本；
- 模型原始输出；
- gold label；
- 检索到的持久经验；
- 当前预测错误的原因；

生成至多一条简短 Emerging Sarcasm Experience：$e_i^{c}=\operatorname{Extract}(x_i,o_i,y_i,E_i)$.

生成 prompt 要求：

- 不复述当前样本中的具体人名、品牌或无关细节；
- 提取能够迁移到其他样本的判断规律；
- 明确讽刺成立或不成立的关键条件；
- 一次只输出一条核心 experience；
- 不建立复杂的人工讽刺类型体系。

生成后的 experience 首先进入：$e_i^c\rightarrow\mathcal P_c.$

### Emerging Sarcasm Experience 的独立验证

Emerging Sarcasm Experience 不需要积累到固定数量后再统一评估。

对于每一条新生成的：$e\in\mathcal P_c$, 均独立执行一次 Experience Utility Test。

在实际实现中，可以将多条 Emerging Sarcasm Experience 组成 batch 并行计算，但每条 experience 的效用分数独立计算。

因此经验验证遵循：$\text{Generate }e \rightarrow \text{Evaluate }e \rightarrow \text{Consolidated Sarcasm Experience or Reject}.$

### Experience Validation Set

为避免使用产生该 experience 的原始样本验证自身，每条 Emerging Sarcasm Experience 均在独立的 held-out samples 上进行评估。我们构建一个 Experience Validation Set，由两部分组成。

#### Global Validation Bank

从训练数据中预先划出固定的类别平衡样本,其中讽刺样本与非讽刺样本数量相等。该集合对所有 Emerging Sarcasm Experience 保持固定，用于判断某条经验是否会损害整体讽刺检测能力。

这些样本：

- 不参与 experience extraction；
- 不参与 OPED training；
- 不使用最终测试集。

#### Relevant Validation Bank

由于单条 sarcasm experience 通常只适用于部分语义或讽刺现象，仅使用全局随机样本可能稀释其真实作用。

因此，对于产生 Emerging Sarcasm Experience (e) 的来源样本 ($x_e$)，从独立 validation samples 中检索与 ($x_e$) 最相关的 (M) 个样本：$\operatorname{Retrieve}(x_e,\mathcal V),$

检索可直接使用冻结的 CLIP ，不引入新的可训练模块。

因此每条 experience 实际在$\text{ global}+\text{ relevant}$  held-out samples 上接受验证。

对于 validation sample$(x_j,y_j),$分别计算不加入 Emerging Sarcasm Experience 时的教师预测：$\pi_T(y_j\mid x_j,E)$,以及加入 Emerging Sarcasm Experience 后的预测：$\pi_T(y_j\mid x_j,E\cup{e})$.这里关注的不是预测标签是否发生离散翻转，而是 Emerging Sarcasm Experience 是否提高模型对 gold label 的偏好。

### Experience Utility

对于 validation sample：$(x_j,y_j)$,分别计算不加入 candidate experience 时的教师预测：$\pi_T(y_j\mid x_j,E)$,以及加入 candidate experience 后的预测：$\pi_T(y_j\mid x_j,E\cup{e})$.这里关注的不是预测标签是否发生离散翻转，而是 candidate experience 是否提高模型对 gold label 的偏好。

定义单样本效用：$p_j^{-e}=P_{\pi_T}(y_j\mid x_j,E)$, $p_j^{+e}=P_{\pi_T}(y_j\mid x_j,E\cup\{e\})$,

$\boxed{
\Delta_j(e)=\log\frac{p_j^{+e}}{p_j^{-e}}
}$

如果：$\Delta_j(e)>0$,说明加入 experience 后，模型对正确标签赋予了更高概率；

如果：$\Delta_j(e)<0$,说明该 experience 降低了模型对正确标签的偏好。

分别定义 relevant utility：$S_{\mathrm{rel}}(e)=\frac{1}{|\mathcal V_r(e)|}\sum_{j\in\mathcal V_r(e)}\Delta_j(e),$

以及 global utility：$S_{\mathrm{global}}(e)=\frac{1}{|\mathcal V_g|}\sum_{j\in\mathcal V_g}\Delta_j(e)$.

Emerging Sarcasm Experience 只有同时满足：

$S_{\mathrm{rel}}(e)>0$以及$S_{\mathrm{global}}(e)\geq0$时，才被认为具有稳定的正向迁移作用，并从 Emerging Sarcasm Experience 转移到 Consolidated Sarcasm Experience  Pool：

该规则不引入：

- 最低验证次数；
- 人工正确率阈值；
- confidence threshold；
- 正负效用计数；
- experience 生命周期管理超参数。

零点直接由“是否产生正向效用”这一语义自然确定。

### 完整经验积累算法

对于经验生成阶段的带标签数据流：

**输入：** 当前 MLLM、Emerging Sarcasm Experience ($\mathcal P_c$)、Consolidated Sarcasm Experience($\mathcal P_p$)、带标签训练数据流。

**输出：** 更新后的 ($\mathcal P_p$)。

对于每个样本：

1. 从 $(\mathcal P_p)$  检索相关经验；
2. MLLM 输出视觉证据、文本证据、解释和标签；
3. 记录预测后揭示 gold label；
4. 根据当前样本、原始输出、gold label 和已有经验生成至多一条 transferable Emerging Sarcasm Experience；
5. 将 Emerging Sarcasm Experience 暂存至 ($\mathcal P_c$)；
6. 在来源样本之外构建对应的 validation bank；
7. 分别计算有无该 experience 时的 gold-label probability；
8. 计算：[$S_{\mathrm{rel}}(e),S_{\mathrm{global}}(e)$;]
9. 若：[$S_{\mathrm{rel}}(e)>0 and S_{\mathrm{global}}(e)\geq0$,]则将该 experience 转移至 ($\mathcal P_p$)；
10. 否则该 experience 不进入持久经验池。

重复上述过程即可逐渐积累经过经验效用验证的 Consolidated Sarcasm Experience。

## 模块 B：On-Policy Experience Distillation

持久经验池增强教师 MLLM。学生不读取经验池，在自己的生成轨迹上接受教师 token-level 分布监督，将外部 experience 内化到参数中。

#### 教师输入

教师读取：$(I,T,E), \qquad E=\mathcal P_p$.

教师分布为：$\pi_T(\cdot\mid I,T,E,y_{<t})$.

#### 学生输入

学生只读取：$(I,T)$.

学生首先生成自己的四字段轨迹：$Y\sim\pi_S(\cdot\mid I,T)$.

#### 标准 On-Policy Experience Distillation 目标

在学生自己的 $prefix (y_{<t})$ 上，最小化学生和经验增强教师之间的 reverse KL：

$\mathbb E_{Y\sim\pi_S}\left[\sum_tD_{\mathrm{KL}}\left(\pi_S(\cdot\mid I,T,y_{<t})\parallel\pi_T(\cdot\mid I,T,E,y_{<t})\right)\right]$.

核心只有：

- 学生生成自身轨迹，因此属于 on-policy；
- 教师读取经过验证的持久经验；
- 学生不读取经验；
- 学生在自身实际访问的 token prefix 上学习经验增强教师的分布；
- 最终将外部经验知识内化到学生参数中。

### 4.14 训练流程

建议使用两阶段学生训练。

#### Stage 1：Format and Task SFT

使用少量高质量数据，使学生先学会：

> <视觉证据>
> 
> 
> <文本证据>
> 
> <解释>
> 
> <判断>
> 

并具备基本讽刺分类能力。

该阶段只负责冷启动，避免学生初始轨迹与教师分布差异过大。

#### Stage 2：On-Policy Experience Distillation

学生生成自身轨迹；

经过持久经验增强的冻结教师在学生 prefix 上提供 token-level distribution；

使用 reverse KL 更新学生参数。

如果实验发现最终标签格式不稳定，可以将 gold-label cross-entropy 作为工程增强项进行消融，但不将其作为核心创新。
