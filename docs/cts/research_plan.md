# CTS 研究方案（含实现对照）

本文件 = 研究方案原文（第 1–8 节，2026-09-29 定稿）+ 本节"实现对照"。实验怎么跑见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 实现对照（代码与方案不一致之处，以代码为准）

| 方案中的写法 | 代码中的实现 | 位置 | 影响 |
| --- | --- | --- | --- |
| C1 查询内 `v = λ_c·u + I` | IF 充电 `v = u + g·I`：查询内无泄漏，g = sigmoid(decay_logit) 作输入增益，沿用 base 的参数名与初始化；减法复位，残差钳位在 [−θ, θ] 防 wind-up | `turbovla/cts/neuron.py` `CTSNeuron` | λ 只出现在查询之间（λ^ctrl），查询内是纯积分器 |
| λ^ctrl 双峰可学习；E2-b "全慢" | 逐通道可学习，统一初始化 0.9；E2-b 用"关（no-carry）/ 固定 0.9 / 可学习"三档 | `CTSConfig.carry_retention_*` | 双峰初始化未做；若 E2-b 显示保留率收敛成双峰，可在论文中作为观察报告 |
| C2 树突迹 ρ 双峰初始化；胞体用 `x_k + d_k` 发放 | ρ_c = exp(−1/τ_c)，τ 在 [1, 16] 次查询间对数均匀分布；迹是连续量，以 `gain·m_k` 加到指令流残差上（gain 逐通道、初始 1e-4，训练开始时≈base），由下一层脉冲神经元发放 | `DendriticMemory`，`fusion.py` `BiAttentionBlock` | 论文方法节需如实写明迹是连续树突变量 |
| C2 写入电流 x_k | x_k = 该层 vision→language cross 写入指令 token 的增量 `γ_l·Δl` | 同上 | — |
| E2-c "仅动作头" | 未实现；有"无 / 仅指令流 / 指令流+视觉流"三档 | `recipes.json` E2c-* | 如需要再加 |
| C3 K ∈ {2,3,4} | K ∈ {2,4}：每步 256 个监督查询不能被 3 整除 | `recipes.json` notes | — |
| C3 状态 dropout | 窗口内每个查询边界，每个样本以概率 p 冷启动（膜与迹一并清零，等价于回合开头） | `TurboVLA.forward_window` | — |
| C3 较早的帧 `no_grad` | burn-in 查询整段 `no_grad`（默认 2 个），损失查询保留梯度；截断 BPTT 在窗口内 | 同上 | — |
| 4.3 用 10 个 wait 步做状态热身 | 未实现：wait 步不调用策略，训练也没有热身帧；第一个查询冷启动，与训练的回合开头一致 | `rollout.py` | 训练/部署一致，但少了一个潜在增益 |
| 文本分支每回合只算一次 | CTS：`carry["instruction"]` 缓存冻结 BERT 输出；base 评估仍每次查询重算。能耗/延迟统计（E5）对两者都缓存并单列 | `forward_cts`，`efficiency.py` | E5 口径公平 |
| E0b 对各模块输入施加 1e-4–1e-2 相对扰动 | 四种扰动源：像素噪声（`--epsilon`）、TF32、int-k 权重、bf16 carry；逐层统计翻转率与动作变化 | `diagnose.py flips` | 模块级相对扰动可由像素噪声的 ε 扫描近似 |
| E0c 记录物体位姿，判定抓取失败/放置偏差 | 用 BDDL 目标谓词逐查询记录，分类 none / stalled / regressed | `rollout.py` trace，`diagnose.py failures` | 抓取失败与放置偏差不细分；G0 判据见 EXPERIMENTS.md |
| E2-f "不刷新 EMA BN" | base 的评估守卫拒绝未刷新的 EMA 权重（设计如此）；此档改为"原始权重 + 训练 running stats、不做刷新"：`evaluate.py --weights raw` | `evaluate.py` | 回答"CTS 部署是否还需要刷新步骤" |
| E2-g 更多脉冲化的动作头 | 未实现（可选） | — | — |
| E3 ANN 头 + 2 帧堆叠、TTF-VLA 式时序融合 | 实现为"显式记忆基线"：上一次查询的视觉 token 拼到动作头 memory，base 神经元、无 carry（E3-history-tokens）；帧堆叠与 TTF 未实现 | `CTSConfig.history_tokens` | 论文需如实写对照的形式 |
| E4-1 BF16 膜电位 | 只把跨查询 carry 存成 bf16/fp16（`--cts_carry_precision`），查询内计算仍 fp32 | `policy.py` | 对应"状态存储精度"，不是全 bf16 推理 |
| E1 MT50 | 仓库不含 MT50 适配器（README 已说明），需另行接入 | — | 主表先报 LIBERO |

另有两处实现细节需在论文中说明：

- 逐时间步 BN（`time_index`）在跨查询 carry 下像一个隐藏时钟；E2f-shared-norm 用一组统计量覆盖所有时间步。
- CTS 默认 T_q=2，对应配方传 `--cross_timesteps 2`；base T=4。carry 规模见 E5：视觉 token 的膜电位占大头（6 层 × 392 token × 1024 通道级别），bf16 存储约减半。

---

# 让脉冲 VLA 活在机器人时间上：面向 spike-driven-vla 的问题诊断与改进方案

**英文工作标题**：*Spiking on Robot Time: Control-Time Neuronal Dynamics for Efficient, Long-Horizon Spiking VLAs*（方法暂名 **CTS**，Control-Time Spiking）

> 日期：2026-09-29 ｜ 目标：CVPR 2027（注册 11-10 AoE，投稿 11-16 AoE，补充材料 11-23）｜ 起点：`wxqnl/spike-turbovla@v1`（commit `fc64ba0`）的 spike-driven-vla base
> 本文中"已测"指仓库已有结果或本次 CPU 复现，"待测"均已写成第 5 节的实验。

---

## 摘要草稿

> 方括号 `[ ]` 为待实验填入的占位符。

Spiking vision-language-action (VLA) models promise low-power robot control, yet current designs treat time as a nuisance. At every control query, each spiking module restarts from a zero membrane, re-encodes the same static input for T synthetic steps, and averages the result away at its output. We show that this wastes the temporal dimension. Under a static input, the standard leaky neuron's resolution barely improves with T (decoding error falls 3% from T=4 to T=16). All neuronal state is discarded 14 times per query and again between queries, although consecutive robot observations are highly correlated. The cost shows up where temporal context matters. On LIBERO-Long, a strong spiking VLA trails its ANN teacher by 13.2 points, and its weakest tasks are two-stage tasks whose completion cues are subtle. We propose Control-Time Spiking (CTS), which aligns neuronal time with control time. CTS spends a small per-query spike budget on error-feedback neurons whose residual is carried across queries. The instruction-token stream keeps a persistent dendritic state, so the instruction itself tracks task progress. Training uses windows of consecutive queries at matched compute. With [2×] fewer spiking timesteps per query, CTS raises LIBERO-Long from 82.0% to [X]% and the four-suite average from 92.05% to [Y]%. It also reduces the success drop under arithmetic-precision perturbation from [Z] to [W] points.

### 中文摘要

脉冲 VLA 有望实现低功耗机器人控制，但现有设计把时间当成累赘：每次控制查询时，每个脉冲模块都从零膜电位重启，把同一个静态输入重复编码 T 个合成步，再在出口取平均。我们发现这浪费了时间维度：在静态输入下，常用的泄漏神经元的分辨率几乎不随 T 提升（T 从 4 到 16，解码误差仅降 3%）；每次查询内神经元状态被丢弃 14 次，查询之间再丢弃一次，而相邻机器人观测高度相关。代价集中出现在需要时间上下文的地方：在 LIBERO-Long 上，一个强的脉冲 VLA 比其 ANN 教师低 13.2 个百分点，最差的任务都是完成线索不明显的两阶段任务。我们提出 CTS，让神经元时间与控制时间对齐：每次查询只用少量脉冲步，神经元采用误差反馈，残差跨查询保留；指令 token 流带持久的树突状态，使指令本身跟踪任务进度；训练时使用连续查询窗口，计算量对齐。在每次查询脉冲步数减少 [2 倍] 的同时，CTS 把 LIBERO-Long 从 82.0% 提升到 [X]%，四套平均从 92.05% 提升到 [Y]%，并把算术精度扰动下的成功率下降从 [Z] 降到 [W] 个百分点。

---

## 1. base 是什么（与问题相关的部分）

| 模块 | 实现 | 时间处理（每次控制查询） |
| --- | --- | --- |
| 视觉 | SDT-V3 19M，224px，输出 360×14×14 连续残差图 | 单次前向 |
| 文本 | 冻结 SmoothSpike BERT，T=4 | 每次查询重算（指令在整个回合内不变） |
| 投影 | BipolarPLIF + 逐时间步 BN，T=4 | 零状态起步，出口取均值 |
| 融合 | 6 层旧 SDSA Cross + 6 层 Top-3 WTA 文本块 | 每层各自零状态跑 T=4，出口取均值 |
| 动作头 | 8 个逐坐标连续状态 token（GELU）、3 层解码器、连续注意力输入与连续 V、脉冲 Q/K + Shiftmax、脉冲 FFN 与读出 | 条件在 T=4 内作常量电流，`tanh(mean_T)` 读出 |

代码依据：`wta_operators.py:56`（`BipolarPLIF` 硬复位）、`spike_fusion.py:43-45`（PLIF 初值 decay=σ(0)=0.5、θ=0.5）、`spike_text_only.py:40-43`（每层新建状态并 `mean(0)`）、`spike_fusion.py:95,127`（Cross 同样）、`spike_action_head.py:237-247,280-282`（动作头把"不变的条件作常量电流"积分，状态每层清空）、`turbovla.py:288,363,382`（强制 B=1 推理、投影零状态起步并取均值）。

成绩：LIBERO 四套 **92.05%**（Spatial 90.4 / Object 98.6 / Goal 97.2 / Long 82.0），公共 MT50 **75.88%**。

---

## 2. 发现的问题

### 2.1 与 ANN 的差距集中在 Long 和 Spatial，而且来自上游，不在动作头

| 模型 | Spatial | Object | Goal | Long | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboVLA（ANN 教师，DINOv3 ViT-B，论文 Tab.1） | 97.0 | 100.0 | 98.2 | 95.2 | 97.6 |
| spike-driven-vla base | 90.4 | 98.6 | 97.2 | **82.0** | 92.05 |
| 差距 | −6.6 | −1.4 | −1.0 | **−13.2** | −5.55 |
| 新 A 80k（同一脉冲上游 + ANN 动作头） | 95.0 | 95.4 | 96.6 | 82.4 | 92.35 |
| 仅 Vision spike 70k（SDT-V3 + 其余全 ANN） | 90.6 | 93.8 | 92.4 | 78.6 | 88.85 |

- 换成 ANN 动作头后 Long 仍是 82.4，动作头不是 Long 缺口的来源；把视觉换成 SDT-V3 时 Long 就降到 78.6。
- 逐任务看（新 A 80k，逐任务数据在 `libero_20260913.json`；base 的逐任务尚未发布），最差的三个 Long 任务都是**两阶段、完成线索不明显**的任务：

| Long 任务 | 指令 | 新 A 80k | TurboVLA |
| --- | --- | ---: | ---: |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 31/50 | 92% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 30/50 | 100% |
| 9 | put the yellow and white mug in the microwave and close it | 30/50 | 98% |

第一阶段完成后，能证明"已完成"的只有篮子里的小物体或被微波炉门挡住的杯子。14×14 的脉冲视觉特征很难每帧都可靠地看出这些线索，而策略又没有任何跨帧记忆。ANN 教师同样没有记忆却能做到 95.2%，说明问题出在弱感知与无记忆叠加：**脉冲模型需要跨帧积累证据，ANN 靠单帧就够了。** 失败类型分解见 E0c。

### 2.2 时间维度被浪费：T 是静态重复，神经元形式使 T 几乎不增加信息

每次查询，投影、6 层 Cross、6 层文本块和动作头共 **14 个模块级 T 循环**，各自从零膜电位开始、输入是常量、出口取均值。本次 CPU 复现（`experiments/cts/neuron_probe.py`，不加载权重）：

- `BipolarPLIF` 的充电式是 `v = u + a(I − u)`：膜电位是向 I 收敛的指数滑动平均，**不累加证据**；a→1 时退化为无记忆（v=I）。
- 初值（a=0.5，θ=0.5）下，T=4 只有 **7 个读出电平**（`{0,±¼,±½,±1}`，缺 ±¾），死区 |I|<0.533，**40.5%** 的归一化输入被直接置零。
- 选最优阈值后的解码 RMSE（越低越好）：

| 神经元 | T=2 | T=4 | T=8 | T=16 |
| --- | ---: | ---: | ---: | ---: |
| repo PLIF，a=0.5（初值），硬复位 | 0.362 | 0.328 | 0.323 | 0.317 |
| repo PLIF，a=0.1，硬复位 | 0.323 | 0.244 | 0.214 | 0.188 |
| IF（累加），硬复位 | 0.316 | 0.232 | 0.203 | 0.144 |
| **IF（累加），减法复位** | **0.316** | **0.209** | **0.129** | **0.076** |

结论：初值下 T 从 4 到 16 只改善 3%，a 越大越差；饱和的主因是泄漏（τ=2 短于合成窗口），硬复位是次因。**T=2 的 IF+减法复位已不差于 T=4 的 repo 神经元**（a≥0.5 时）。训练后的 a/θ 需从 checkpoint 读出（E0a）。另外，所有状态在查询之间清零；而 LIBERO 每 12 步才查询一次，相邻两次查询的画面大部分相同，却每次都从头编码。

### 2.3 动作通路的"精度墙"：越脉冲化越差，base 靠连续通路回到 92%

| 动作头 | Avg | 说明 |
| --- | ---: | --- |
| ANN 头（新 A 80k） | 92.35 | |
| 全脉冲 SDSA 头（AA） | 69.20 | 18,750 步、global 1024 |
| AA + 状态修复 | 73.70 | |
| ① 入口 PLIF + 脉冲 FFN | 83.65 | 脉冲状态编码 |
| ② 连续 attention 入口 + 脉冲 FFN | 77.45 | |
| ③ 连续 attention 入口 + ANN FFN | 82.95 | |
| base（连续状态 GELU + 连续入口 + 连续 V + 轻增强） | 92.05 | |

崩溃集中在需要精细接触或两阶段的任务（AA 对新 A）：goal-3 "open the top drawer and put the bowl inside" **0/50 对 48/50**，goal-5 "push the plate to the front of the stove" **7/50 对 50/50**，long-0 **1/50 对 45/50**，long-8 **3/50 对 42/50**。base 为了回到 92% 保留了连续状态 MLP、连续注意力入口（Q/K/V 投影是 MAC）、连续 V 和 LayerNorm，所以动作头的相当一部分不是 spike-driven。评审会直接问"到底哪里是脉冲的"。

### 2.4 数值脆弱：一次 TF32 差异让闭环成功率掉 30.7 个百分点

`docs/precision_fix_20260908.md` 已记录：

- 同一旧 Cross 50k EMA，训练用 `high`（TF32）、评估用 `highest`：**90.90% → 60.20%**；逐回合配对只在 high 下成功 695 次，只在 highest 下成功 81 次。
- 冻结 SmoothSpike 在两种精度下的有效 token 相对 MSE 为 **0.172**。一个 1e-3 量级的舍入差异被 12 层 T=4 的脉冲 BERT 放大到 17%。
- 修复后，文本特征在 B=128 与 B=1 下的相对 MSE 仍为 0.4%（余弦 0.998）；代码注释写明"硬阈值会放大与 batch 形状相关的 GEMM 舍入"，因此推理强制 B=1。EMA 必须用训练数据刷新 BN 才能评估。

硬阈值附近的膜电位被舍入翻转，翻转沿层级传播，再被闭环控制累积。神经形态芯片本来就用定点运算，这个问题会阻碍实际部署。同类现象在分类任务上已有人研究（Burst SNN, 2607.11914）；**闭环 VLA 上尚无系统研究**。本方案把它作为评估维度（E4），不作为主要方法贡献。

### 2.5 现有实验会被评审攻击的地方

1. 全部单训练 seed；变体之间的 batch、步数、warmup、重计算设置不同，大多不是单变量消融。
2. 缺少同配方的 DINOv3 ANN 上界（README 明确"主表没有为它补造全量成绩"）。
3. 能耗没有测量；base 保留大量连续通路。
4. 文本分支在回合内每次查询重算 12 层 T=4，而指令是常量，这是免费的效率损失（缓存对所有模型都成立，只作工程项，不算贡献）。

---

## 3. 相关工作定位

| 工作 | 脉冲 | 机器人 / VLA | 跨控制步保留神经元状态 | 与本方案的关系 |
| --- | --- | --- | --- | --- |
| SpikeVLA (arXiv 2606.27807) | V/L/A 全脉冲 | VLN 导航 + Go2 控制，无 LIBERO | **否**，差分神经元在单帧的虚拟时间步内累积，逐帧重新初始化 | 最直接的竞争者；时间轴仍是合成的 |
| SDP (2409.11195)、STMDP (2411.09953)、L-SDPPO (2606.06049) | 脉冲扩散策略 | 操作 | 否 | 逐观测独立 |
| Li et al. (2601.09031) | 递归脉冲网络 | 人形操作，ManiSkill | 摘要写"recursive spiking for spatiotemporal consistency"，**需读全文确认**是否跨控制步保留膜电位 | 这一维度上最接近的前作，必须精读并对比 |
| SpikeMLLM (2604.18610) | 脉冲 MLLM | 否 | 否 | 已提出把 T 从 L−1 压缩到 log₂L−1：**二进制加权 / 对数时间编码已被占** |
| Burst SNN (2607.11914) | 分级 burst 神经元 | 否 | 否 | 已研究翻转鲁棒性（分类）：**只改神经元的鲁棒性方案已被占** |
| I-LIF / SpikeYOLO (ECCV 2024)、Radix encoding (TNNLS 2022) | 整数 / 基数脉冲 | 否 | 否 | 多电平脉冲已是成熟方案，只能当组件或基线 |
| TTF-VLA (2508.19257)、MemoryVLA (ICLR 2026)、TempoFit (2603.07647) | ANN | VLA | ANN 历史 / 记忆 | 说明时序上下文对 VLA 有用；本方案的对应是"时序上下文长在神经元状态里" |
| TurboVLA (2607.27205) | ANN | VLA，LIBERO 97.6 | 否 | 教师与上界 |

**空位**：还没有脉冲 VLA 让神经元时间等于控制时间。只做更精细神经元（量化电平、burst、对数编码）的方向已被 Guoqi Li 组的 SpikeMLLM、Burst SNN、SDT-V3 等覆盖，作为 CVPR 主创新风险高。

**与你们 Fire Without Forgetting（FwF）的关系**：两者共享"持久态优于逐步复位"的主张。本方案的 VLA 部分另有三点独立贡献：把合成 T 预算重新分配到控制时间上、以指令 token 流作进度记忆、语言条件操作中的闭环鲁棒性。投稿时应互相引用，并在方法描述上明确区分，避免被认为是同一工作的两次投稿。

---

## 4. 方法：Control-Time Spiking（CTS）

### 4.1 核心思想

把"每次查询从零开始、重复静态输入 T=4 步、出口平均"改成：**每次查询只跑 T_q 步（默认 2），神经元状态在一个回合内跨查询延续，只在回合开始时清零。** 相邻查询的画面高度相关，所以上一次查询留下的膜电位就是下一次的热启动，时间轴真正对应机器人时间。

### 4.2 三个组件

**C1 误差反馈神经元 + 跨查询残差（提供效率和精度）**
- 查询内：把 `v = u + a(I−u)` 换成累加式 `v = λ_c·u + I`，采用减法复位 `u ← v − θ·s`，每次查询跑 T_q 步。
- 查询间：残差 u 按可学习的逐通道保留率 λ_c^ctrl 延续，不再清零。λ^ctrl=0 时退化为逐查询复位，所以 base 是 CTS 的特例，可直接用于初始化和消融。
- 原理：减法复位保留量化误差，下一步会补偿，所以分辨率随累计步数约按 1/T 提高（2.2 表）；残差跨查询延续后，静态部分的精度随观测时长累积，而不是只靠 T_q 个合成步。
- 副作用（待验证）：零状态起步的充电瞬态只出现在回合开头，逐时间步 BN 统计及"评估前必须刷新 EMA BN"可能不再需要（E2-f）。

**C2 进度感知的指令状态（针对 Long）**
- 在融合模块的**指令 token 流**（通常 10–20 个 token）上加一个不复位的树突迹：`d_k = ρ_c·d_{k−1} + (1−ρ_c)·x_k`，其中 x_k 是第 k 次查询 vision→language cross 写入指令 token 的电流，ρ_c 为可学习的逐通道保留率，双峰初始化（一半接近 0，一半接近 0.95）；胞体用 `x_k + d_k` 发放，只有胞体复位。
- 依据：指令在回合内不变，是记录"哪个子目标已完成"的天然载体。树突迹跨查询积累视觉证据（如"汤罐已在篮子里"），一次查询的证据弱时也能逐步达到发放阈值；微波炉门关上把杯子挡住后，迹仍能保留一段时间。
- 代价：只多了指令 token 数 × 通道数个状态，对视觉 token 不加记忆。

**C3 计算量对齐的序列训练**
- 训练样本从单帧改为同一回合内 K 个连续查询（间隔 12 步，与部署一致），对 K 个查询都算动作损失，截断 BPTT。
- 例如 K=2、T_q=2，投影/融合/动作头的神经元步数（2×2=4）与 base（1×4）相同；视觉编码器成本 ×K，较早的帧可在 `no_grad` 下编码以控制显存。
- 状态 dropout：以概率 p（如 0.1）在窗口内随机清零状态，防止模型走历史捷径（copycat；Wen et al., NeurIPS 2020），也覆盖冷启动。
- 本体状态 token 只用当前帧，不接入历史，从结构上切断"从自身历史动作复制"的通道。

### 4.3 推理

策略封装类在回合开始时清零状态，之后每次查询传入并返回状态。可选：利用评估协议开头的 10 个 wait 步处理画面做状态热身（不增加动作延迟）。文本分支每个回合只算一次，base 也同样缓存。

### 4.4 CTS 对应的问题

| 问题 | CTS 的作用 | 验证实验 |
| --- | --- | --- |
| 2.2 T 被浪费 | T_q 4→2，精度靠误差反馈和跨查询积累补回 | E0d、E1、E5 |
| 2.1 Long 两阶段缺口 | C2 让指令状态积累子目标完成证据 | E0c、E1、E2 |
| 2.4 数值脆弱 | 慢通道对单次查询的翻转做时间平均；误差反馈使翻转可被后续步骤补偿 | E4 |
| 2.3 精度墙 | C1 每次脉冲携带的信息更多，给更多脉冲化的动作头留出余量 | E2-g（可选） |

---

## 5. 实验计划

### E0 诊断（第 1 周，决定主线）

| 编号 | 做什么 | 成本 | 判定 |
| --- | --- | --- | --- |
| E0a | 读 base checkpoint 各脉冲节点学到的 a、θ；在 2k 训练帧上统计各接口的死区比例、实际电平数、发放率 | 1 GPU 小时 | 若学到的 a 普遍 ≥0.3，2.2 的饱和在训练后仍成立 |
| E0b | 翻转增益：对各模块输入施加 1e-4–1e-2 相对扰动，逐层统计翻转数和输出相对变化；同样测 ANN 头版本 | 数 GPU 小时 | 验证 2.4 的级联放大 |
| E0c | Long 失败分类：base 跑 500 个 Long 回合并记录仿真物体位姿，自动判定"子目标 1 是否完成 / 是否重复抓取已放好的物体 / 抓取失败 / 放置偏差" | ~1 天 | **门控 G0**：阶段混淆占 Long 失败 ≥30% 时，C2 为主线；否则主线转为 C1 + 更高频重规划（见 7.2） |
| E0d | base 以 T=2、T=1 重训（不跨查询保留状态） | 2 次完整训练 | **门控 G1**：若 T=2 与 T=4 相差 ≤1 个点，2.2 的浪费在系统层面成立，效率主张站得住 |
| E0e | 同配方 DINOv3 ANN 上界（`--preset ann`，按 base 的 40k/256/轻增强训练） | 1 次完整训练 | 主表必需 |

### E1 主结果

LIBERO 四套 2,000 回合，base 与 CTS（C1+C2+C3）各跑 **3 个训练 seed**，报告均值 ± 标准差和逐套 Wilson 置信区间；再补 MT50（已有单视角适配）。

目标：Long ≥ [88]%（base 82.0），四套平均 ≥ [93.5]%，每次查询的脉冲步数减半。

### E2 消融（单变量，统一配方）

| 编号 | 变量 | 设置 |
| --- | --- | --- |
| a | 神经元形式 | repo PLIF 硬复位 / IF 硬复位 / IF 减法复位（逐查询复位，T_q=2） |
| b | 跨查询保留 | 关（λ^ctrl=0）/ 全慢 / 双峰可学习 |
| c | C2 位置 | 无 / 仅指令流 / 指令流+视觉流 / 仅动作头 |
| d | T_q | 1 / 2 / 4 |
| e | 窗口 K 与状态 dropout | K∈{2,3,4}，p∈{0,0.1,0.3} |
| f | 归一化 | 逐时间步 BN / 共享 BN / 不刷新 EMA BN |
| g（可选） | 更多脉冲化的动作头 | CTS 下重跑 ②（连续入口改脉冲入口），看精度墙是否后移 |

### E3 ANN 时序对照

ANN 动作头 + 2 帧历史堆叠，以及 TTF-VLA 式时序 token 融合，接在同一脉冲上游上。用来证明增益来自"神经元状态本身就是记忆"，而不只是看到了更多帧；同时比较记忆的额外开销（CTS 只多了指令 token 的状态）。

### E4 鲁棒性（同一 checkpoint 换评估条件，base 与 CTS 对比成功率下降）

1. 算术精度：训练 highest、评估 high（TF32）；BF16 膜电位；INT8 权重伪量化。
2. 相机噪声：高斯噪声 σ∈{2,4,8}/255，以及亮度偏移。
3. 回合中途强制清零状态：CTS 应能恢复，用来量化对记忆的依赖。
4. 可选：LIBERO-PRO（开发机上已有 LIBERO-PRO 仓库）的位置/物体扰动。

### E5 效率

报告每次查询的脉冲步数、逐层发放率与 SOPs、45nm 能耗估算（明确标注为理论值），以及 RTX 4090 或 H100 上 B=1 的实测延迟。文本缓存对所有模型同样开启，单独列出。

### E6 可选：把省下的计算量换成更频繁的重规划

CTS 每次查询更便宜，在相同总计算量下把 open-loop 从 12 缩到 6。针对 2.3 的接触类任务（goal-5 推盘子、goal-3 抽屉）检验是否提升精度。

---

## 6. 时间表（今天 9-29 → 注册 11-10，投稿 11-16）

| 周 | 日期 | 内容 |
| --- | --- | --- |
| 1 | 9-29 – 10-05 | E0a/b/c 分析；启动 E0d、E0e；实现序列 dataloader（`libero_lerobot.py` 返回 K 个查询）、状态透传（`encode_condition`、动作头、`evaluation/policy.py` 回合级状态）；过 G0/G1 |
| 2 | 10-06 – 10-12 | CTS 试跑：C1 单独、C1+C2，先只评 Long 500 回合；定 T_q、K |
| 3–4 | 10-13 – 10-26 | E1 三个 seed；E2 消融在两台 8 卡节点上并行 |
| 5 | 10-27 – 11-02 | E3、E4、E5、MT50 |
| 6 | 11-03 – 11-10 | 成文、作图；11-10 前注册 |
| — | 11-10 – 11-16 | 定稿投稿；补充材料 11-23 前（rollout 视频、逐任务表） |

关键路径是训练时长：CTS 的视觉编码器成本约 K 倍。需要先测一次 base 40k 训练的实际耗时，再定 K 和 seed 数。

---

## 7. 风险与备选

### 7.1 主要风险

| 风险 | 应对 |
| --- | --- |
| T_q=2 精度不够，平均分下降 | C1 的 λ^ctrl=0、T_q=4 退化为只换神经元的版本，可作保底；E2-d 给出精度与步数的完整曲线 |
| 历史捷径（copycat）导致闭环漂移 | 本体状态不接历史；状态 dropout；E4-3 的中途清零测试 |
| 训练-部署分布偏移（训练用专家轨迹上的状态，部署时状态由策略自身产生） | 窗口短（K≤4）；状态噪声增强；必要时加少量 DAgger 式在线回放 |
| 与 FwF 重叠 | 见第 3 节，两文互相引用，贡献分开写 |
| 2601.09031 已跨控制步保留状态 | 第 1 周精读；若确实如此，把差异落在语言条件 VLA、合成 T 的重新分配、指令进度记忆和 LIBERO 结果上 |

### 7.2 备选主线（G0 判定 Long 失败以精度或抓取为主时）

**"查询预算重分配"**：在 C1 的基础上，把省下的步数用于更频繁的重规划（E6），或按阶段分配给接触阶段。这个版本的叙事是"脉冲 VLA 把计算花在重复编码上，而不是花在需要精度的时刻"，C1、E0d、E6 可以直接复用。

---

## 8. 贡献草稿

1. **诊断**：我们发现脉冲 VLA 没有用上时间。T 只是静态重复；常用泄漏神经元的分辨率随 T 饱和；每次查询内神经元状态被清空 14 次，查询之间再清空一次。我们用神经元级分析、系统级 T 扫描和 Long 失败分解，把这一现象与两阶段长程任务上的缺口联系起来。
2. **方法**：CTS 让神经元时间与控制时间对齐。它由误差反馈神经元与跨查询残差、进度感知的指令树突状态、计算量对齐的序列训练组成；base 是它在 λ^ctrl=0 时的特例。
3. **结果**：在 LIBERO 和 MT50 上，CTS 以一半的每查询脉冲步数把 Long 提升 [X] 点，四套平均提升 [Y] 点，并在算术精度与相机噪声扰动下成功率下降更小；我们还给出首个闭环脉冲 VLA 的数值鲁棒性评测。

---

## 参考

- spike-turbovla v1：https://github.com/wxqnl/spike-turbovla/tree/v1（`docs/spike_driven.md`、`docs/results.md`、`docs/precision_fix_20260908.md`、`docs/results/*.json`）
- TurboVLA：Xie et al., arXiv 2607.27205，https://arxiv.org/abs/2607.27205
- SpikeVLA：Song et al., arXiv 2606.27807，https://arxiv.org/html/2606.27807v1
- SpikeMLLM：Xu et al., arXiv 2604.18610，https://arxiv.org/abs/2604.18610
- Burst Spiking Neural Networks：Zhang et al., arXiv 2607.11914，https://arxiv.org/abs/2607.11914
- SpikeYOLO / I-LIF：Luo et al., ECCV 2024，https://arxiv.org/abs/2407.20708
- Radix encoding：Wang et al., IEEE TNNLS 2022，https://arxiv.org/abs/2105.06943
- SDP：arXiv 2409.11195，https://arxiv.org/pdf/2409.11195；STMDP：arXiv 2411.09953，https://arxiv.org/pdf/2411.09953；L-SDPPO：arXiv 2606.06049，https://arxiv.org/pdf/2606.06049
- Li et al., Generalizable Geometric Prior and Recurrent Spiking Feature Learning for Humanoid Robot Manipulation，arXiv 2601.09031，https://arxiv.org/abs/2601.09031
- TTF-VLA：arXiv 2508.19257，https://arxiv.org/pdf/2508.19257；TempoFit：arXiv 2603.07647，https://arxiv.org/pdf/2603.07647；MemoryVLA：https://shihao1895.github.io/MemoryVLA/
- Wen et al., Fighting Copycat Agents in Behavioral Cloning from Observation Histories, NeurIPS 2020
- CVPR 2027 日期：https://cvpr.thecvf.com/Conferences/2027/Dates
- 神经元探针：`experiments/cts/neuron_probe.py`（CPU 运行约 1 分钟）
