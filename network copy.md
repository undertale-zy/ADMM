# ADMM 网络无文件复刻交接说明

Last updated: 2026-08-26

本文档面向另一个无法直接访问当前仓库、checkpoint、服务器 NAS 和历史实验产物的对话。
目标不是继续设计网络，而是让另一个对话在只能读取文字规范的条件下，准确完成当前
ADMM 神经替代网络的结构级复刻，并在获得算力后完成训练级复刻。

---

## 0. 当前任务定义

当前任务是维护一份可以脱离原始代码和文件使用的网络复刻规范。

“复刻”必须区分三个等级：

### A 级：结构级复刻

要求另一个对话能够根据 Markdown 重新实现：

- 所有模型类和模块层级；
- 输入输出 tensor shape；
- 复数双通道转换；
- normalized partial-DFT forward / adjoint；
- ADMM 每层的更新顺序；
- CNN、Transformer、Support、Guide 和 Round 32–36 分支；
- 参数初始化、softplus 正值参数化和 frozen buffer；
- Point、Structured-v1、Dense-v2 数据生成逻辑；
- RNG 顺序和数据混合概率；
- base、support、noise-aware、Round 32–36 loss；
- checkpoint selector；
- 评测指标和 checkpoint schema。

当前 A 级网络实现已经完成。当前剩余工作主要是自动化验收证据，而不是重新设计
或补写网络主体。

### B 级：训练级复刻

使用相同或等价的：

- 训练配置；
- 随机种子；
- 数据生成器；
- workers；
- PyTorch/CUDA/GPU 环境；
- DDP 和 AMP 配置；
- checkpoint selector；

重新训练后，得到趋势一致、指标处于历史合理范围内的新 checkpoint。

B 级不要求新权重与历史权重逐参数相同。

### C 级：历史权重级复刻

要求恢复服务器历史 checkpoint 的具体权重，并实现逐参数或逐像素输出一致。

没有原始 checkpoint 或完整权重文本时，C 级无法完成。结果表、参数量、SHA256
和指标都不能反推出 CNN/Transformer 权重。

---
## 1. 当前真实版本锚点

当前实际工作区：

```text
/aiot001/zy/ADMM
```

当前 Git 分支：

```text
main
```

当前 Git HEAD：

```text
e6b6d204f5bda1c3145eea80fa181672dd27a9d7
```

最近关键提交：

```text
e6b6d20 Document workspace paths and evaluation artifact levels
348c294 Document exact evaluation and runtime protocols
9823d75 Fill replication spec evaluation gaps
2e5f337 Expand replication spec for all 36 training rounds
9030a7d Add complete network replication specification
6b8c629 Add ADMM unfolding research experiments and Round 32-36 models
2ec77a0 Implement ADMM unfolding network and training pipeline
```

其中：

- `2ec77a0`：最初的 Stage 1/2/3 网络和训练管线；
- `6b8c629`：加入 Support、Guide、Structured/Dense 数据集和 Round 32–36 网络；
- `e6b6d20`：当前文档、评测协议和 artifact 说明的最新版本。

注意：`NETWORK_REPLICATION_SPEC.md` 中部分章节仍将 `6b8c629` 称为当前迭代版本，
这是历史版本锚点，不应覆盖本文件记录的真实 HEAD。

---

## 2. 文档权威顺序

如果不同 Markdown 之间出现冲突，按以下顺序判断：

```text
1. 当前代码在 e6b6d20 HEAD 中的实际行为；
2. NETWORK_REPLICATION_SPEC.md；
3. memory_2DADMM.md 第 12 节以后；
4. reviews/review_018 至 review_023；
5. admm_unrolled_network_plan.md；
6. 更早的历史状态描述。
```

特别注意：

- `admm_unrolled_network_plan.md` 是早期计划，里面的“待实现”不能覆盖当前完成状态；
- `memory_2DADMM.md` 第 1–11 节包含历史记录，第 12 节以后才是服务器端权威状态；
- 早期的“31 轮”是 Round 32–36 完成前的状态；当前完整训练轮次是 36；
- 早期的“10 轮”是代表方案口径，不是完整训练任务轮次。

---
## 3. 文件和职责映射

```text
admm_unrolled.py
    complex conversion、FFT forward/adjoint、Stage 1、CNN proximal、Transformer proximal

admm_support_fusion.py
    zero-preserving SupportFusion、support-gated CNN/Transformer

admm_guide_support_fusion.py
    Stage-1 GuideSupportFusion、firm support、masked DC

admm_deep_scalar.py
    Deep Scalar 12/16/24/32 层模型

admm_rounds32_36.py
    Round 32、33、34、35、36 的特殊 forward

synthetic_isar_dataset.py
    Point-only sparse synthetic generator

structured_isar_dataset.py
    Structured-aircraft v1 generator

dense_aircraft_isar_dataset.py
    Dense-aircraft v2 generator

admm_2d.py
    传统 explicit / Fast 2D-ADMM 和 entropy

yak42_admm_demo.py
    Yak-42 加载、预处理和传统 Fast-ADMM demo

train_admm_unrolled.py
    R1–R3 和基础 scalar 训练器

train_support_fusion.py
    SupportFusion 训练器及其派生历史训练行为

train_balanced_scalar.py
    R16–R19

train_noise_aware_support.py
    R20–R23 的 clean-echo / discrepancy loss

train_structured_noiseaware.py
    R24–R27

train_dense_aircraft_noiseaware.py
    R28–R31

train_rounds32_36.py
    R32–R36 训练器

evaluate_reconstruction.py
    通用 Yak-42 评测

evaluate_structured_generalization.py
    Structured 固定域和 Yak 评测

evaluate_rounds32_36.py
    R32–R36 固定 point/dense/Yak 评测

generate_all_complete_training_visualizations.py
    R1–R31 统一可视化
```

---
## 4. 统一物理模型

默认正式尺寸：

```text
image_shape       = (512, 128)
measurement_shape = (256, 64)
```

图像和观测都是复数矩阵：

```text
X ∈ C^(512×128)
S ∈ C^(256×64)
```

PyTorch 输入输出使用两个实数通道：

```text
measurement input = [B, 2, 256, 64]
image output      = [B, 2, 512, 128]
channel 0         = real
channel 1         = imaginary
```

复数转换：

```python
z = real + 1j * imag
```

训练时网络内部可以使用 bfloat16，但进入 `torch.complex` 和 FFT 前，必须提升到
float32，因此物理算子使用 complex64。

### 4.1 Forward operator

记 `A(X)` 为 normalized partial-DFT forward：

```python
range_domain = ifft(X, n=P, dim=-2) * sqrt(P)
range_domain = range_domain[..., :M, :]
echo = fft(range_domain, n=Q, dim=-1) / sqrt(Q)
echo = echo[..., :, :N]
```

其中：

```text
P=512, Q=128, M=256, N=64
```

### 4.2 Adjoint operator

记 `Aᴴ(S)` 为 forward 的伴随：

```python
image = fft(S, n=P, dim=-2) / sqrt(P)
image = ifft(image, n=Q, dim=-1) * sqrt(Q)
```

这套 forward/adjoint 只适用于当前 normalized partial-Fourier 结构，不能描述成任意
sensing matrix 的通用求解器。

不要在物理 forward/adjoint 内部加入 `fftshift`。`fftshift` 只用于显示。

---
## 5. 基础 ADMM 展开

状态变量：

```text
X：数据一致性图像状态
Z：负责稀疏性的辅助变量
U：scaled dual variable
```

第 k 层：

```text
D_k       = Z_k - U_k
R_k       = A(D_k) - S
X_(k+1)   = D_k - c_k Aᴴ(R_k)
Z_(k+1)   = soft(X_(k+1)+U_k, tau_k)
U_(k+1)   = U_k + beta_k (X_(k+1)-Z_(k+1))
```

每层参数：

```text
c_k      > 0：data-consistency step
τ_k      ≥ 0：complex soft-threshold
β_k      > 0：dual update coefficient
```

实际保存 raw 参数，并使用 softplus：

```python
c[k]   = softplus(raw_c[k])
tau[k] = softplus(raw_tau[k])
beta[k]= softplus(raw_beta[k])
```

Stage 1 初始化：

```text
c=0.5
tau=0.0065
beta=1.0
```

复数 soft-threshold：

```text
soft(z, τ) = z * max(|z|-τ, 0) / (|z|+1e-12)
```

重要历史行为：基础 Stage 1/2/3 最终返回 `X`，不是 `Z`。不要因为 `Z` 更稀疏而
擅自修改所有模型的返回值。

---

## 6. 模型族与 Round 映射

```text
R1       PhysicsUnrolledADMM，scalar-8
R2       PhysicsUnrolledADMM + shared CNN proximal
R3       PhysicsUnrolledADMM + shared Transformer proximal
R4–R7    zero-preserving SupportFusionADMM
R8–R11   Stage-1 Guide + Firm Support
R12–R15  Deep Scalar 12/16/24/32
R16–R19  Balanced Scalar echo sweep
R20–R23  noise-aware SupportFusion
R24–R27  Structured-aircraft v1
R28–R31  Dense-aircraft v2
R32      Guide + 8-step masked DC
R33      Guide + three-band confidence + 4-step DC
R34      Stage-1 guide + residual CNN
R35      Stage-1 support + explicit physics solve
R36      Round 33 architecture + dense-heavy mixed training
```
### 6.1 Guide 和 Support 的关键行为

Guide branch：

- 使用 Stage 1 的 `raw_c/raw_tau/raw_beta`；
- 读取 checkpoint 时只复制这三个 key；
- 注册为 frozen buffer；
- 不计入 trainable parameter；
- 仍然会保存在新模型 state dict 中。

Firm support：

```text
low_db  = -65 + 20*sigmoid(raw_support_db)
high_db = low_db + support_width_db
```

peak 必须逐样本计算：

```python
peak = amax(abs(guide_image), dim=(-2,-1), keepdim=True).clamp_min(1e-12)
```

support 外必须严格归零。

Round 33 的 confidence：

```text
db >= -40       → 1
-60 < db < -40   → (db+60)/20
 db <= -60       → 0
```

### 6.2 Round 32–36 特殊要求

```text
R32：8 次 support-masked DC，当前最佳综合候选
R33：three-band confidence，4 次 masked DC
R34：guide_image + 连续 mask × residual CNN，无最终显式 DC
R35：guide support 内 8 次 explicit physics solve，support 外严格零
R36：forward 与 R33 相同，仅 dense probability 从 0.50 改为 0.75
```

R36 没有独立 distillation head。它的 support distillation 只是 loss 中的 support-energy
hinge，不能擅自添加新的输出头。

R34 的连续 mask 会导致 residual leakage，这是历史失败机制的一部分，不能在做历史
复刻时偷偷改成硬 mask。

---

## 7. 数据和 RNG

Point-only：

- 3–30 个离散复散射点；
- 随机位置；
- 幅度约 0.5–1.0，峰值归一化；
- 独立随机复相位；
- SNR 均匀覆盖 `[-10,30] dB`；
- clean target 通过同一 `A` 生成 clean echo；
- 加 circular complex Gaussian noise。

Structured-v1：

- 机身主轴；
- 主翼；
- 尾翼；
- 发动机/强散射簇；
- 随机缺失、坐标抖动和随机相位。
Dense-v2：

- 机身、主翼、尾翼使用连续有厚度散射带；
- 截断 Gaussian splat；
- 结构附近多个非零像素；
- global phase、缓慢 phase slope 和小扰动；
- 保留机头、尾部、发动机和翼尖强散射中心。

训练数据 seed 规则：

```text
train seed = seed
validation seed = seed + 1,000,000
test seed = seed + 2,000,000
```

Round 32–36 默认：

```text
seed = 3200
train/val/test = 4000/500/500
batch_size = 2
epochs = 20
```

R36：

```text
structured_probability = 0.75
```

注意：历史变量名 `structured_probability` 在 Dense-v2 相关流程中实际表示 dense/structured
数据注入概率，不要根据变量名擅自改义。

---

## 8. Loss 和指标口径

必须区分：

```text
image NMSE
clean-target echo NMSE
observed noisy residual
true noise residual
discrepancy
background
entropy
```

### 8.1 Clean echo 和 observed residual

```text
S_clean = A(X_true)
S_pred  = A(X_pred)
```

```text
image NMSE
= ||X_pred-X_true||² / (||X_true||²+ε)

clean echo NMSE
= ||S_pred-S_clean||² / (||S_clean||²+ε)

observed residual
= ||S_pred-S_observed||² / (||S_observed||²+ε)
```

均为先逐样本求空间和，再对 batch 求平均。

### 8.2 Discrepancy

目标定义：

```text
true_noise_ratio
= ||S_observed-S_clean||² / ||S_observed||²

predicted_residual_ratio
= ||S_pred-S_observed||² / ||S_observed||²

discrepancy
= (predicted_residual_ratio-true_noise_ratio)²
```

必须：

- 每个样本独立计算；
- 分母使用该样本 observed echo 能量；
- 先算两个 ratio，再做平方差；
- 最后才进行 batch/dataset 平均。

训练和评测应共同调用一个公共 discrepancy 函数。当前评测代码的 discrepancy 统计仍需
专门审计和修正，不能仅凭“公式看起来类似”就视为已完成。
### 8.3 Background

- target 的非零支撑阈值为 `abs(target)>1e-6`；
- 用 3×3 max-pool 膨胀保护真实支撑邻域；
- 其余区域为 background；
- background 同时使用 normalized L2 和 normalized L1。

Round 32–36 总 loss：

```text
L = image
  + 0.5 × clean_echo
  + 0.1 × discrepancy
  + 0.2 × background
  + 0.1 × support_loss
```

其中 R32/R34/R35 的 `support_loss=0`，R33/R36 使用：

```text
support_energy = Σ(|prediction|² × protected) / target_energy
support_loss = mean(relu(0.05-support_energy))
```

不能把 observed noisy residual 当成唯一 checkpoint 选择标准。

---

## 9. 训练配置和 selector

Round 32–36 正式配置：

```text
optimizer = AdamW
learning_rate = 1e-3
weight_decay = 1e-5
gradient_clip_norm = 10
AMP = bfloat16 on CUDA
GradScaler = torch.amp.GradScaler("cuda")
DDP find_unused_parameters = True
```

历史正式 workers 应固定记录，不要继续使用依赖 `os.cpu_count()` 的动态值。建议统一写入
spec 和启动命令：

```text
train workers = 2
validation workers = 1
test workers = 1
evaluation workers = 1
```

如果为了复刻某个旧任务而使用不同 workers，必须在该轮的 metadata 中明确记录。

Round 32–36 selector：

```text
score = validation_total + 0.5 × validation_background
```

每轮：

1. 保存 `checkpoint-best.pt`；
2. 始终保存 `checkpoint-last.pt`；
3. 训练结束后重新加载 `checkpoint-best.pt`；
4. 用 reload 后的模型计算 test metrics。

历史 base trainer 的 selector 不能简单套用：R2/R3、R12–R15 存在训练和 validation
权重不一致的历史行为。复刻历史时必须保留，修正版只能作为新实验。

---
## 10. 当前 36 轮参考结果

当前综合候选排序：

```text
R32 Guide + 8-step DC
    > R36 Multi-domain confidence
    > R33 Soft confidence band
    > R1 Stage-1 Scalar-8
    > R35 Support physics
    > R34 Residual CNN
```

Round 32–36 固定 point-only / dense-only 评测参考：

| Round | Point image | Point clean echo | Dense image | Dense clean echo | Yak band | Yak support | Params |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.006898 | 0.006564 | 0.171632 | 0.039110 | 0.8200% | 1.5981% | 19,718 |
| 33 | 0.004284 | 0.004056 | 0.231572 | 0.065353 | 0.9875% | 1.5318% | 19,714 |
| 34 | 0.012340 | 0.017301 | 0.398229 | 0.087337 | 31.7529% | 32.2110% | 30,149 |
| 35 | 0.001279 | 0.001126 | 0.690358 | 0.325899 | 0.1025% | 0.2556% | 19,720 |
| 36 | 0.006309 | 0.005929 | 0.186350 | 0.066985 | 1.0870% | 1.5651% | 19,714 |

这些数字是重训验收参考，不是新训练必须逐位相等的目标。

结构性错误的典型信号：

```text
R1 Yak band 变成 50% 以上
R2/R3 比 R1 更干净
R8–R11 变成全图稠密
R12–R15 Yak band 不再极高
R32 Yak band 明显超过 5%
R34 不再有约 30% 的 residual leakage
R35 dense/Yak 不再严重过稀疏
R36 被实现成有额外 distillation head
```

出现这些情况时，优先检查：

```text
real/imag channel 顺序
→ FFT axis 和 normalization
→ A/Aᴴ
→ X/Z 返回值
→ peak 是否逐样本计算
→ support mask 乘法位置
→ clean/observed echo 目标
→ selector
→ dB 显示窗口
```

---
## 11. A 级自动化验收缺口

A 级实现已经完成，但验收证据仍需闭环。

### 11.1 数据 golden anchors

为 Point、Structured-v1、Dense-v2 固定：

```text
seed
shape
sample index
dtype
real/imag channel order
```

并测试：

- target/measurement 数值摘要；
- target/measurement energy；
- clean echo；
- 有效支撑比例；
- scatterer 数量或结构统计；
- 数组 SHA256。

当前测试只证明确定性，不足以证明与服务器 RNG 顺序完全一致。

### 11.2 Scalar schedule golden tests

为以下轮次增加自动化测试：

```text
R1、R12、R13、R14、R15、R16、R17、R18、R19
```

检查：

- layer count；
- c/tau/beta schedule；
- proximal 配置；
- echo weight；
- 输入固定时的输出摘要。

### 11.3 全部 forward 分支 golden tests

不需要写 36 份重复代码，但应通过 Round registry 或参数化测试覆盖 R1–R36：

- shape；
- finite；
- parameter count；
- frozen/trainable 参数；
- support 外严格零；
- DC 步数；
- 最终返回变量；
- 固定输入输出摘要。
### 11.4 Loss 和 selector tests

需要覆盖：

```text
base loss
support loss
noise-aware loss
Round 32–36 loss
best selector
score selector
constrained selector
历史 validation 默认权重行为
```

每个测试应使用小的手工构造复数张量，验证每一项的精确数值和组合权重。

### 11.5 完整 36 轮参数量测试

逐轮从 registry 构建模型，并断言实际 trainable parameter count 等于文档记录。

必须重点检查：

- Guide buffer 不计入 trainable parameter；
- R34 residual CNN 参数量；
- R32–R35 的 DC scalar 数量；
- R36 没有额外 head；
- 历史未使用参数没有被擅自删除。

---

## 12. 完整工程验收缺口

以下项目不影响“网络架构 A 级”，但影响可复现实验的完整性：

1. 统一并修正 evaluator 中的 discrepancy 实现；
2. Structured evaluator 正式输出 Yak seed 0–9 每 seed 结果、mean、std；
3. Structured evaluator 正式记录 latency；
4. 固定 train/validation/test/evaluation workers；
5. 统一 latency 的 warmup、repeat 和 CUDA synchronize；
6. 区分历史兼容评测和修正后的正式评测；
7. 对不同 checkpoint schema 做 strict reload 测试；
8. 记录完整环境、实际启动命令和 selected epoch。

当前本地代码中，Structured evaluator 已有 Yak seed 和 latency 参数入口，但需要形成稳定、
可归档的正式 artifact，而不是只依赖手动命令输出。

---
## 13. 上卡训练建议

### 13.1 最小验证路径

不要一开始提交全部 36 轮。先执行：

```text
本地 smoke
    ↓
R1 小规模训练
    ↓
R1 checkpoint reload
    ↓
R1 full training
    ↓
R32 training
    ↓
point/dense/Yak seed 0–9 evaluation
```

R1 是后续 Guide 模型的 teacher，必须先得到可加载 checkpoint。

### 13.2 当前最有价值的训练

如果目标是论文主线，优先：

```text
R1 → R32 → R33/R36
```

R32 是当前最佳综合候选；R33 和 R36 是对 confidence、弱结构和多域泛化的后续验证。

如果目标是完整历史实验包，再训练：

```text
R1
→ R2–R3
→ R4–R11
→ R12–R19
→ R20–R23
→ R24–R31
→ R32–R36
```
### 13.3 建议服务器配置

历史 Round 32–36 参考：

```text
GPU：4 × GPU-class GPU per job
POOL_IDX：16
IMAGE_IDX：-1
CONDA_ENV：None
Python：/opt/conda/bin/python
DDP：find_unused_parameters=True
```

提交形式：

```bash
cd <SERVER_WORKSPACE>
python3 scripts/submit_train.py scripts/specs/<spec>.py 0
```

实际 `<spec>`、输出目录和命令必须记录到该轮 metadata。

不要使用没有权限的 `POOL_IDX=10`。

---

## 14. 每轮必须保存的文本信息

即使 checkpoint 文件无法传回，也应从服务器复制以下终端文本：

### 环境

```text
GPU model:
GPU count:
Python version:
PyTorch version:
CUDA version:
cuDNN version:
NumPy/SciPy versions:
OMP_NUM_THREADS:
```
### 训练

```text
round:
actual launch command:
output directory:
train/validation/test sample count:
batch size:
epochs:
learning rate:
workers:
AMP dtype:
DDP world size:
```

### 结果

```text
selected checkpoint filename:
selected epoch:
checkpoint SHA256:
metrics.json:
Yak seed 0–9 per-seed metrics:
Yak mean/std:
latency protocol:
```

SHA256 只用于确认 checkpoint 身份，不能恢复权重。

---

## 15. 如果可以通过终端文本转移权重

如果文件不能通过 SCP 下载，但服务器允许复制终端文本，可以只导出
`model_state`，压缩后 Base64 分段传回。

### 只做推理所需

```text
model_state
model_config
```

### 继续训练所需

```text
model_state
model_config
optimizer_state
epoch
history
scaler_state（若对应训练器保存了它）
```

推荐流程：

```text
服务器：
model_state → CPU → gzip/zip → Base64 → 固定长度分段输出 → SHA256

本地：
合并分段 → Base64 解码 → 解压 → strict=True 加载 → SHA256 校验
```

R32 这类约 2 万参数模型适合这种方式；Transformer 权重约 35 万参数，通常也可以
分段传输，但需要控制终端输出长度和分段编号。

如果任何形式的权重都无法取出，则只能完成结构级和训练级复刻，不能称为历史 checkpoint
逐参数恢复。

---
## 16. 最终验收定义

### A 级通过

```text
所有模型分支可构建
shape 正确
FFT forward/adjoint 正确
ADMM 顺序正确
support/guide/R32–R36 行为正确
数据生成和 RNG 规则明确
loss 和 selector 明确
36 轮参数量可验证
golden tests 通过
```

### B 级通过

```text
R1 可以重新训练并 reload
R32 可以加载 R1 teacher 并重新训练
point/dense/Yak 评测流程完整
指标趋势和历史合理范围一致
没有 NaN/Inf
checkpoint artifact 完整
```

### C 级通过

```text
拿到目标历史 checkpoint 或等价完整权重内容
strict=True 加载成功
state_dict 身份经过 SHA256 确认
固定输入输出达到逐参数/逐像素验收要求
```

---

## 17. 不应做出的表述

在没有原始 checkpoint 时，不要说：

```text
已经恢复了历史模型权重
已经逐像素复刻了服务器结果
仅凭指标表反推出了 CNN/Transformer 参数
```

正确表述是：

```text
网络结构已完成 A 级复刻；
当前需要通过 golden tests 完成 A 级自动化验收；
获得算力后可以完成 B 级训练复刻；
没有原 checkpoint 时无法完成 C 级历史权重复刻。
```

---
## 18. 给另一个对话的最短执行指令

```text
请以当前 Git HEAD e6b6d20 为版本锚点，以
NETWORK_REPLICATION_SPEC.md 和本 REPLICATION_HANDOFF.md 为主要规范。
不要重新设计网络，不要擅自修复历史行为，不要覆盖旧实验定义。
先完成 A 级 golden/conformance tests 和评测公式审计，再训练 R1。
R1 产出的 checkpoint 作为 R32–R36 的 Guide teacher。
优先验证 R1 → R32；若指标和结构范围正常，再训练 R33/R36 或完整 R2–R31。
没有原始 checkpoint 时，将结果称为训练级复刻，不称为历史权重逐像素恢复。
```