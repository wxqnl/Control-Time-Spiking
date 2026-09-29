# CTS 实验手册

在 8 卡训练机上完成 CTS（Control-Time Spiking）全部训练与评估的操作说明。方法与动机见 [research_plan.md](research_plan.md)（开头的"实现对照"列出代码与方案的差异）。本文的命令默认在仓库根目录、bash、`conda activate cts` 下执行。

| 入口 | 作用 |
| --- | --- |
| `experiments/cts/train.py` | 按 `configs/recipes.json` 的命名配方启动训练（base / CTS / ANN），保证每步 256 个监督查询 × 40k 步 |
| `experiments/cts/evaluate.py` | 标准协议评估四套 LIBERO（自动刷新 EMA 归一化），可透传扰动/压力参数 |
| `experiments/cts/aggregate.py` | 汇总：逐套成功率、Wilson CI、seed 均值±标准差、配对 McNemar |
| `experiments/cts/diagnose.py` | E0a 神经元统计、E0b 翻转、E0c 失败分类 |
| `experiments/cts/efficiency.py` | E5：每查询步数、发放率、AC/MAC、理论能耗、B=1 延迟、carry 大小 |
| `experiments/cts/refresh_norms.py` | CTS 的 EMA 归一化刷新（evaluate.py 自动调用） |
| `experiments/cts/neuron_probe.py` | 方案 2.2 节的神经元分辨率探针（CPU，无权重） |
| `scripts/cts/setup_env.sh`、`download_assets.sh` | 环境与资产 |

---

## 0. 准备（新机器上一次性完成）

### 0.1 环境

```bash
cp .env.cts.example .env        # 至少改 SPIKE_WORK_ROOT
set -a; source .env; set +a
bash scripts/cts/setup_env.sh   # conda env "cts"、torch 2.7.1 cu128、仓库 [libero,dev]、LIBERO 代码与 config.yaml
conda activate cts
```

系统需有 OSMesa（`apt install libosmesa6-dev`），评估用 `MUJOCO_GL=osmesa`。已知坑：

- `PYTHONNOUSERSITE=1` 必须生效（setup 脚本已写入 env 变量），否则 `~/.local` 里的包会混进来。
- robosuite 1.4.0 只能配 MuJoCo 3.1.x（`pyproject.toml` 已锁 3.1.6），bddl 需要 `future`。
- 不要设置 `SPIKE_CUDA_DEVICES`：它属于旧的三卡流水线，`experiments/libero/settings.py` 要求恰好 3 张卡。
- 训练与评估都是 fp32 + `highest` matmul；不要打开 TF32（E4 的 TF32 压力测试除外，它在评估进程内单独设置）。

### 0.2 权重与数据

```bash
bash scripts/cts/download_assets.sh check          # 只检查，不联网
hf auth login                                      # DINOv3 是 gated 模型，先在网页上接受条款
bash scripts/cts/download_assets.sh download all   # 或逐项：bert dino teacher grounding sdtv3 smoothspike data
```

| 资产 | 用途 |
| --- | --- |
| SDT-V3 19M `V3_19.0M_1x4.pth` | 视觉编码器初始化 |
| SmoothSpike BERT（融合 H2/H3 后） | 冻结文本骨干 |
| BERT-base-uncased、GroundingDINO Swin-T | 文本投影校正参考、交互层初始化 |
| DINOv3 ViT-B/16 | 图像预处理配置、TurboVLA 教师输入、E0e ANN 学生 |
| TurboVLA LIBERO 教师 | 训练期 KD（特征 MSE + 动作 MSE，各 0.1） |
| LIBERO 四套 LeRobot v2.1 no-noops | 训练与归一化刷新；episode 数 432 / 454 / 428 / 379（spatial / object / goal / 10） |

数据的 Hugging Face 仓库（默认 `IPEC-COMMUNITY/<suite>_no_noops_1.0.0_lerobot`）与 revision 记录在 `download_assets.sh`；下载后 `check` 会核对 episode 数。

### 0.3 文本投影校正（SmoothSpike → BERT 投影接口，一次）

```bash
mkdir -p "$CTS_ROOT"
CUDA_VISIBLE_DEVICES=0 python experiments/libero/prepare_cross_joint.py --mode calibrate \
  --output_dir "$CTS_ROOT/text_init"            # 写出 text_projection.pth 与 result.json
```

所有 spike 配方（base 与 CTS）都用这个文件作 `--text_projection_init`，E0e ANN 不需要。

### 0.4 自检

```bash
CTS_TEST_LIBERO_DATA="$SPIKE_DATA_ROOT" CTS_TEST_BERT="$SPIKE_BERT_PATH" \
  CUDA_VISIBLE_DEVICES="" python -m pytest tests/cts -q      # CPU；含 2 步训练冒烟，约 1–2 分钟
python experiments/cts/train.py --list
```

测试覆盖：plif_hard 与 base 神经元逐位一致；CTS 全关时整模型与 base 逐位一致；回合重置等价于冷启动；窗口前向所有 CTS 参数有梯度；每个配方都解析到 256 个查询/步并通过真实参数解析器；2 步真实数据训练、EMA 保存。

---

## 1. 训练

### 1.1 命令

```bash
set -a; source .env; set +a
mkdir -p "$CTS_RUNS" "$CTS_EVAL" "$CTS_ROOT/analysis"
COMMON=(--vision_checkpoint "$CTS_VISION" --text_projection_init "$CTS_TEXT_PROJECTION")

# 先看展开后的完整参数（不加载模型、不占 GPU）
python experiments/cts/train.py --recipe E1-cts --run_dir "$CTS_RUNS/E1-cts-s42" "${COMMON[@]}" \
  --dry-run --world-size 8

train() {   # train RECIPE SEED [launcher options] [-- raw trainer flags]
  local recipe=$1 seed=$2; shift 2
  torchrun --standalone --nproc_per_node 8 experiments/cts/train.py --recipe "$recipe" --seed "$seed" \
    --run_dir "$CTS_RUNS/$recipe-s$seed" "${COMMON[@]}" --approve_training "$@" \
    2>&1 | tee -a "$CTS_RUNS/$recipe-s$seed.log"
}
train E0-base 42
train E1-cts 42
```

- 输出：`$CTS_RUNS/<recipe>-s<seed>/checkpoints/<prefix>_<step>.pth`，prefix 是配方名小写、`-` 换成 `_`（如 `e1_cts_40000.pth`）；`launcher.json` 记录配方、布局和全部参数。
- 中断后续训：同一命令加 `--resume_mode all`（CTS 配置必须一致，否则拒绝）。
- 显存不够：`--windows_per_gpu 8`，梯度累积自动变为 2，每步监督查询数不变。SDT-V3 含 BN，改 microbatch 不是严格数值等价，同一组对比内保持一致。
- 磁盘：每个 checkpoint 约 1 GB（含 EMA 与优化器），默认每 1000 步存一次、不轮转，一次 40k 训练约 40 GB。消融可加 `-- --save_steps 5000`。
- 日志：每 `log_freq` 步一行 JSON，含 `loss`、`act`、`act_kd`、逐查询 `act_q{k}`、学习率、保留率均值/p10、记忆 τ 中位数与 gain、`sec_per_step`。第 1 步检查所有 CTS 参数都有梯度，否则报错退出。

### 1.2 预算对齐

每个配方每步都看到 **256 个带损失的控制查询**，共 40k 步，与 base（8 卡 × 32）相同：

| 配方类型 | 每卡 | K（损失查询/窗口） | burn-in | 梯度累积（8 卡） |
| --- | --- | --- | --- | --- |
| base / ANN | 32 帧 | 1 | 0 | 1 |
| CTS 默认 | 16 窗口 | 2 | 2（no_grad） | 1 |
| E2e-K4 | 8 窗口 | 4 | 2 | 1 |
| E2a 单查询 | 32 窗口 | 1 | 0 | 1 |

CTS 的 base 相关超参（优化器、KD、轻增强、warmup、EMA、精度）全部继承 base 启动器；CTS 参数（保留率、记忆 ρ 与 gain）进入单独的 no-decay 组，学习率 = head_lr × 10。

**计算量估计**：每个损失查询多出一次 burn-in 前向（视觉编码器 + T_q=2 的融合，无反向），融合与动作头的步数减半。视觉编码器前向多约 1/3，融合与头约少 1/3，预计每次训练 1.0–1.4× base 的时间。以前 200 步 `sec_per_step` 实测为准，据此定排期。

### 1.3 配方一览（`python experiments/cts/train.py --list`）

T_q 为每次查询的脉冲步数（`--cross_timesteps`）。

| 组 | 配方 | T_q | 变量 | seed | 优先级 |
| --- | --- | --- | --- | --- | --- |
| E0 | E0-base | 4 | base 复现（同机器、同代码） | 42, 43, 44 | A（42）/ B |
| E0d | E0d-base-T2 | 2 | base 降 T | 42 | A（门控 G1） |
| E0d | E0d-base-T1 | 1 | base 降 T | 42 | B |
| E0e | E0e-ann-upper | — | DINOv3 + BERT + ANN 融合/动作头，同预算同 KD | 42 | B |
| E1 | E1-cts | 2 | 主方法：IF-soft + carry（投影/融合/头）+ 指令记忆，K=2 | 42, 43, 44 | A（42）/ B |
| E1 | E1-cts-T4 | 4 | T_q=2 掉点时的保底 | 42 | C（条件） |
| E2a | E2a-if-soft-only | 2 | 只换神经元，逐查询复位、无记忆、单查询训练 | 42 | B |
| E2a | E2a-if-hard-only | 2 | IF + 硬复位 | 42 | C |
| E2a | E2a-plif-carry | 2 | base 神经元 + carry + 记忆 | 42 | C |
| E2b | E2b-no-carry | 2 | 无膜电位 carry（λ^ctrl=0），保留记忆 | 42 | B |
| E2b | E2b-no-head-carry | 2 | 动作头不 carry | 42 | C |
| E2b | E2b-fixed-retention | 2 | 保留率固定 0.9 | 42 | C |
| E2c | E2c-no-memory | 2 | 无指令记忆 | 42 | B |
| E2c | E2c-memory-vision | 2 | 指令流 + 视觉流记忆 | 42 | C |
| E2d | E2d-cts-T1 | 1 | T_q 曲线（与 E1 的 2、E1-cts-T4 的 4 合成） | 42 | B |
| E2e | E2e-K4 | 2 | K=4 | 42 | C |
| E2e | E2e-no-burnin | 2 | 每个窗口冷启动 | 42 | B |
| E2e | E2e-no-dropout / E2e-dropout0.3 | 2 | 状态 dropout 0 / 0.3（默认 0.1） | 42 | C |
| E2f | E2f-shared-norm | 2 | 所有时间步共用一组 BN 统计 | 42 | C |
| E3 | E3-history-tokens | 4 | 显式记忆基线：上一查询视觉 token 拼入动作头 memory | 42 | C |
| E6 | E6-cts-stride6 | 2 | 按 6 步重规划训练（评估加 `--num_open_loop_steps 6`） | 42 | 可选 |
| P0 | P0-pilot-finetune | 4 | 从训练好的 base EMA 微调 5k 步（非论文数字，只作早期信号） | 42 | A |

P0 需要训练好的 base：`train P0-pilot-finetune 42 --init_checkpoint "$CTS_BASE_CHECKPOINT"`（或 E0-base-s42 的最终 checkpoint）。它保持 base 的 T=4，从 base 权重加载，只有 CTS 参数新建。

A 档合计约 3 次完整训练 + P0；B 档 11 次；C 档 11 次（含条件项）；全部约 25 次完整训练。

---

## 2. 评估

### 2.1 标准协议

```bash
final() { ls -1v "$CTS_RUNS/$1"/checkpoints/*.pth | tail -1; }
evaluate() {   # evaluate RUN OUT GPUS [rollout flags...]
  local run=$1 out=$2 gpus=$3; shift 3
  python experiments/cts/evaluate.py --checkpoint "$(final "$run")" --out_dir "$CTS_EVAL/$out" \
    --gpus "$gpus" -- --trace_dir "$CTS_EVAL/$out/traces" "$@"
}
evaluate E0-base-s42 E0-base-s42 0,1,2,3
evaluate E1-cts-s42  E1-cts-s42  4,5,6,7
```

- 协议与 base 相同：EMA 权重（先用训练数据刷新归一化统计，CTS 用 4 查询窗口），每任务 50 回合（初态 0–49）、seed 7、10 个 wait 步、12 步 open loop、256px、fp32/highest。第 i 个回合用第 i 个初态，所以任意两个模型可逐回合配对。
- 四套在给定 GPU 上并行，每卡一个进程。8 卡可同时跑两个模型（如上）。
- 输出：`ema_refreshed.pth`（刷新后的导出，之后的压力测试直接复用）、`<suite>.result.json`、`summary.json`、`traces/<suite>/taskXX_epYYY.json`（逐查询的目标谓词）。
- 输出目录必须是新的；重复评估换 `OUT` 名。
- ANN（E0e）没有 WTA 归一化，直接评估其 EMA 权重。

### 2.2 汇总

```bash
python experiments/cts/aggregate.py \
  base="$CTS_EVAL/E0-base-s42,$CTS_EVAL/E0-base-s43,$CTS_EVAL/E0-base-s44" \
  cts="$CTS_EVAL/E1-cts-s42,$CTS_EVAL/E1-cts-s43,$CTS_EVAL/E1-cts-s44" \
  ann="$CTS_EVAL/E0e-ann-upper-s42" base_T2="$CTS_EVAL/E0d-base-T2-s42" \
  --reference base --markdown results/main.md --json results/main.json
```

输出逐套与四套平均：seed 均值 ± 标准差、合并回合的 Wilson 95% CI，以及相对 `--reference` 的精确 McNemar（按 seed 序号、套件、任务、回合配对）。E2 消融表用 `--reference cts`，与 E1-cts-s42 配对。

---

## 3. 诊断与门控（E0）

E0 可以先在已有的 base checkpoint 上跑（`$CTS_BASE_CHECKPOINT` 先经 `evaluate.py` 得到 `ema_refreshed.pth`），不必等 E0-base 重训。`diagnose.py` 从 checkpoint 记录的参数里读数据路径；换机器后若路径不同，先把数据放到相同路径或重新导出。

```bash
BASE_EMA="$CTS_EVAL/E0-base-s42/ema_refreshed.pth"
CTS_EMA="$CTS_EVAL/E1-cts-s42/ema_refreshed.pth"

# E0a：学到的 g/θ、逐层发放率、静默比例、实际电平数 + 常量输入分辨率探针
python experiments/cts/diagnose.py neurons --checkpoint "$BASE_EMA" --batches 64 \
  --out "$CTS_ROOT/analysis/E0a-base.json"

# E0b：扰动引起的脉冲翻转（逐层、逐查询）与动作变化
for eps in 1e-4 1e-3 1e-2; do
  python experiments/cts/diagnose.py flips --checkpoint "$BASE_EMA" --mode pixel --epsilon $eps \
    --out "$CTS_ROOT/analysis/E0b-base-pixel-$eps.json"
done
python experiments/cts/diagnose.py flips --checkpoint "$BASE_EMA" --mode tf32 --out "$CTS_ROOT/analysis/E0b-base-tf32.json"
python experiments/cts/diagnose.py flips --checkpoint "$BASE_EMA" --mode weights --bits 8 \
  --out "$CTS_ROOT/analysis/E0b-base-int8.json"
# CTS 训练后同样跑一遍，并加 --mode bf16_carry

# E0c：失败分类（来自评估轨迹）
python experiments/cts/diagnose.py failures --trace_dir "$CTS_EVAL/E0-base-s42/traces" \
  --out "$CTS_ROOT/analysis/E0c-base.json"
```

失败分类：`none` 是从未满足任何目标谓词，`stalled` 是完成部分子目标后停住，`regressed` 是已满足的谓词后来被破坏且最终失败。

| 门控 | 判据 | 通过 | 不通过 |
| --- | --- | --- | --- |
| G0（E0c） | libero_10 失败中 `stalled + regressed` 的占比 ≥ 30%。这是"完成至少一个子目标后失败"，即阶段缺口的上界；`regressed` 是阶段混淆的直接证据。抽 20 个回合视频人工核对（`--save_video`） | C2（指令记忆）作为主线之一 | 主线转为 C1 + 更频繁重规划（E6，见方案 7.2） |
| G1（E0d） | E0d-base-T2 的四套平均与 E0-base 相差 ≤ 1 个点 | "T=4 被浪费"成立，效率主张站得住 | 效率叙事改为"CTS 以 T_q=2 补回 base T=2 的损失"，E1-cts-T4 升为 B 档 |

早期信号：P0 微调后只评 libero_10（`--suites libero_10`，500 回合），与 base 在同一 500 回合上配对。

---

## 4. 鲁棒性（E4）

同一个刷新后的导出，只改评估条件。base 与 CTS（至少 s42，时间允许时三个 seed）各跑一遍，报告相对干净评估的成功率下降，并做配对 McNemar。

```bash
stress() {   # stress MODEL_EVAL_DIR OUT GPUS rollout-flags...
  local src=$1 out=$2 gpus=$3; shift 3
  python experiments/cts/evaluate.py --checkpoint "$CTS_EVAL/$src/ema_refreshed.pth" \
    --out_dir "$CTS_EVAL/$out" --gpus "$gpus" -- --trace_dir "$CTS_EVAL/$out/traces" "$@"
}
stress E1-cts-s42 E4-cts-tf32 0,1,2,3 --numerics_stress tf32
```

| 编号 | 条件 | rollout 参数 | 模型 |
| --- | --- | --- | --- |
| E4-1a | TF32 matmul/cuDNN（训练为 highest） | `--numerics_stress tf32` | 全部 |
| E4-1b | int8 / int6 / int4 权重伪量化（逐输出通道对称，冻结文本骨干除外） | `--weight_bits 8`（6、4） | 全部 |
| E4-1c | carry 以 bf16 / fp16 存储 | `--cts_carry_precision bf16` | CTS |
| E4-2a | 双视角高斯噪声 σ = 2 / 4 / 8（0–255 像素单位），全程 | `--perturb_mode noise --perturb_noise_std 4 --perturb_views both --perturb_start_query 0 --perturb_queries 100000` | 全部 |
| E4-2b | 亮度 ×1.3 / ×0.7，全程 | `--perturb_mode brightness --perturb_brightness 1.3 --perturb_views both --perturb_start_query 0 --perturb_queries 100000` | 全部 |
| E4-2c | 主视角瞬时遮挡：第 4、5 次查询黑屏 | `--perturb_mode blackout --perturb_start_query 4 --perturb_queries 2` | 全部 |
| E4-3a | 回合中途清零状态一次 | `--cts_reset_at_query 10` | CTS |
| E4-3b | 每 N 次查询清零（状态依赖的时间尺度） | `--cts_reset_every 4`（2、8） | CTS |
| E2f | 不做 EMA 刷新：原始权重 + 训练 running stats | `evaluate.py --weights raw --checkpoint <训练 checkpoint>` | base、E1、E2f-shared-norm |
| E6 | 6 步重规划 | `--num_open_loop_steps 6` | E6-cts-stride6、E1、base |

扰动在策略看到图像之前施加，每个回合的随机数由 (seed, task, episode) 决定，两个模型看到的噪声相同。轨迹中每个查询标注 `perturbed`，失败分类可直接比较扰动后的恢复情况。LIBERO-PRO（方案 E4-4）未接入。

---

## 5. 效率（E5）

```bash
python experiments/cts/efficiency.py \
  base="$CTS_EVAL/E0-base-s42/ema_refreshed.pth" \
  base_T2="$CTS_EVAL/E0d-base-T2-s42/ema_refreshed.pth" \
  cts="$CTS_EVAL/E1-cts-s42/ema_refreshed.pth" \
  cts_T1="$CTS_EVAL/E2d-cts-T1-s42/ema_refreshed.pth" \
  --device cuda:0 --count_episodes 16 --latency_queries 200 \
  --out results/efficiency.json --markdown results/efficiency.md
# ANN 上界：训练 checkpoint 没有刷新导出，单独跑并加 --weights ema
python experiments/cts/efficiency.py ann="$(final E0e-ann-upper-s42)" --weights ema --device cuda:0 \
  --out results/efficiency_ann.json
```

每个模型在真实 LIBERO 回合的连续查询上（B=1，与部署一致）报告：

- 每查询脉冲步数、逐组发放率与静默比例、神经元更新数；
- 每个 matmul / 卷积（aten 级，含注意力乘积）的运算数，按模块（视觉编码器 / 投影 / 融合 / 动作头）分：有脉冲取值操作数（二值、三值或 SDT-V3 整数脉冲）的算 AC = MAC × 平均脉冲数，其余算 MAC；
- 理论能耗 4.6 pJ × MAC + 0.9 pJ × AC（45 nm，Horowitz 2014），论文中标注为理论值，不含访存；
- 冻结文本骨干每回合一次，对所有模型都缓存，单列；
- CTS carry 的元素数与 fp32/bf16 字节（按投影 / 融合 / 动作头拆分）。视觉 token 膜电位占大头，E2b 消融对应"哪部分 carry 值得保留"；
- B=1 延迟（中位数 / p90，预热 20 次）与峰值显存，在 4090 或 H100 上测，写明型号。

---

## 6. 排期（今天 9-29 → 注册 11-10，投稿 11-16）

| 周 | 日期 | 训练（两台 8 卡节点，A/B 各一条队列） | 评估与分析 |
| --- | --- | --- | --- |
| 1 | 9-29 – 10-05 | 节点 1：E0-base s42 → E0d-base-T2；节点 2：P0（先跑）→ E1-cts s42 | 已有 base checkpoint 上跑 E0a/b/c；前 200 步测 `sec_per_step`；过 G0 |
| 2 | 10-06 – 10-12 | E1-cts s43/s44、E0-base s43/s44 | E1-s42 完整评估；过 G1；定 T_q、K（必要时启动 E1-cts-T4） |
| 3–4 | 10-13 – 10-26 | B 档：E0e、E0d-T1、E2a-if-soft-only、E2b-no-carry、E2c-no-memory、E2d-cts-T1、E2e-no-burnin；之后 C 档 | 主表（三 seed）、E2 表 |
| 5 | 10-27 – 11-02 | C 档剩余、E3、可选 E6 | E4 全部条件、E5、失败分类图 |
| 6 | 11-03 – 11-10 | 补跑 | 成文、作图；11-10 前注册 |

实际节奏取决于 base 一次 40k 训练的时长 X。若 X > 2 天，C 档只保留 E2a-plif-carry、E2b-fixed-retention、E2c-memory-vision、E3，其余放到补充材料或 rebuttal。

---

## 7. 结果记录约定

- 运行目录 `$CTS_RUNS/<recipe>-s<seed>`，评估目录 `$CTS_EVAL/<recipe>-s<seed>[-<条件>]`，分析 `$CTS_ROOT/analysis/`。
- 论文中的每个数字对应一个 `summary.json`，该文件记录 checkpoint、评估权重、seed、每任务 50 回合、额外 rollout 参数和数值设置。汇总表由 `aggregate.py` 生成，不手填。
- base 的已发表结果（92.05%：90.4 / 98.6 / 97.2 / 82.0）只作参考；主表用同机器重训的 E0-base 三个 seed。
