# ADMM 展开网络与服务器迁移计划

Last updated: 2026-08-23

这份文档是服务器新对话的执行说明。开始实现前，先阅读本文件、
`memory_2DADMM.md` 和 `readpaper.md`。

## 1. 任务目标

实现一个轻量、可解释的 ADMM 展开网络（Unrolled ADMM / ADMM-Net），验证
神经网络是否能够用更少的展开层数近似或改进固定参数的 Fast 2D-ADMM。

第一版的明确范围：

- 使用作者的二维 ISAR 观测模型和 ADMM 结构；
- 将 ADMM 展开为 8 层网络；
- 每层学习数据一致性步长、稀疏阈值和对偶更新系数；
- 使用合成稀疏 ISAR 数据训练；
- 使用 Yak-42 实测数据做最终测试；
- 不运行或移植 2D-FFT、2D-SL0、2D-GP-SOONE 对照方法；
- 第一版不使用黑盒 U-Net、Transformer、扩散模型或 CNN 近端模块；
- 第一版不处理自动聚焦、相位误差和未补偿运动。

目标不是一开始做完整的大规模识别系统，而是先确认“物理模型展开网络”
能够稳定训练并完成稀疏 ISAR 重建。

## 2. 当前机器资源评估

当前本机：

- Apple M4 MacBook Air；
- 10 核 CPU；
- 8 核集成 GPU；
- 16 GB 统一内存；
- 可用磁盘约 77 GiB；
- Conda 环境：`usual`；
- Python：3.12.13；
- NumPy：2.5.1；
- SciPy：1.18.0；
- Matplotlib：3.11.1；
- PyTorch：2.13.0；
- pytest：9.1.1。

PyTorch 状态：

- `torch.backends.mps.is_built() = True`；
- `torch.backends.mps.is_available() = False`；
- CUDA 不可用；
- 当前训练按 CPU 规划。

资源结论：

- 小尺寸标量参数展开网络可以在本机完成；
- 本机先做数据管线、网络调试、小规模训练和单元测试；
- 当前不需要 A100，也不需要立即迁移服务器；
- 如果加入 CNN 近端模块、扩大数据集或进行大量消融实验，再迁移 GPU；
- 服务器优先选择单卡 8--16 GB GPU；24 GB 以上用于扩展实验，不默认使用
  A100。

## 3. 网络接口与结构

PyTorch 网络接口固定为：

```text
输入：y，形状 [batch, 2, M, N]
输出：x，形状 [batch, 2, P, Q]
```

两个通道分别是复数数据的实部和虚部。网络内部可以组合成：

```text
z = real_channel + j * imag_channel
```

第一阶段尺寸：

```text
观测矩阵：64 × 32
目标图像：128 × 64
展开层数：8
batch size：8
训练轮数：20
```

第二阶段尺寸：

```text
观测矩阵：256 × 64
目标图像：512 × 128
```

每层执行：

```text
D(k) = B(k) − V(k)

X(k+1) = D(k) − c(k) · Fᴴ(F(D(k)) − S)

B(k+1) = soft_threshold(X(k+1) + V(k), τ(k))

V(k+1) = V(k) + β(k) · [X(k+1) − B(k+1)]
```

变量角色：

- `X`：负责数据一致性的图像；
- `B`：负责稀疏化的辅助图像；
- `V`：累计 `X` 与 `B` 不一致的缩放对偶变量；
- `D`：`B − V` 的临时量，没有独立物理意义；
- `F`：二维傅里叶正向算子；
- `Fᴴ`：二维傅里叶反向/共轭转置算子；
- `c(k)`：第 k 层可学习的数据一致性步长；
- `τ(k)`：第 k 层可学习的软阈值；
- `β(k)`：第 k 层可学习的对偶更新系数。

参数约束：

- `c(k) > 0`；
- `τ(k) >= 0`；
- `β(k) > 0`；
- 使用 softplus 或等价正值参数化保证训练稳定。

初始化：

```text
X(0) = Fᴴ(S)
B(0) = 0
V(0) = 0
```

第一版保持软阈值形式，不加入 CNN 近端模块。

## 4. 合成训练数据

从现有 `simulation_2D_ADMM.m` 的物理模型迁移数据生成逻辑：

```text
X_true → F(X_true) → S_clean → 加复高斯噪声 → S_noisy
```

每个样本随机生成：

- 3--30 个散射点；
- 随机距离位置；
- 随机方位位置；
- 随机散射幅度；
- 随机复数相位；
- SNR 从 −10 dB 到 30 dB；
- 第一版使用完整采样；
- 不使用 Yak-42 作为训练目标。

第一阶段数据规模：

```text
训练集：1024 个样本
验证集：128 个样本
测试集：128 个样本
```

训练、验证和测试必须使用不同的随机场景，不能只改变噪声种子。数据可以
动态生成或轻量缓存，避免一次性占用大量磁盘。

## 5. 两阶段训练流程

### 阶段一：小尺寸原型

使用 64×32 观测和 128×64 图像，在本机 CPU 上完成：

- 数据生成器；
- 网络前向和反向；
- 梯度检查；
- loss 下降检查；
- 小数据集过拟合检查；
- 与固定参数 Fast 2D-ADMM 的内部比较。

### 阶段二：全尺寸测试

切换到 Yak-42 对应的 256×64 观测和 512×128 图像，使用阶段一参数初始化，
再用合成数据完成全尺寸适配，最后只在 Yak-42 上测试。

Yak-42 流程必须保持已有 demo：

1. 读取 `Yak42.mat`；
2. MATLAB 列 `129:192` 对应 Python 切片 `128:192`；
3. 按最大幅度归一化；
4. 加入固定种子的复高斯噪声；
5. 沿第一个轴执行 IFFT；
6. 输入展开网络；
7. 输出 `512×128` 图像；
8. 保存图像并计算 SNR、熵、耗时和 NaN 状态。

Yak-42 不参与第一阶段训练，避免单个实测目标被网络记忆。

## 6. 损失函数

使用归一化的图像、回波和稀疏损失：

```text
L_image = ||X_pred − X_true||²

L_echo = ||F(X_pred) − S_noisy||²

L_sparse = mean(|X_pred|)

L = L_image + 0.1 · L_echo + 0.0001 · L_sparse
```

含义：

- `L_image` 是主要监督目标；
- `L_echo` 保证网络输出符合雷达物理模型；
- `L_sparse` 延续论文的稀疏先验；
- 各项按样本能量归一化，避免不同散射强度造成梯度尺度差异。

## 7. 实现状态与代码文件

已新增并验证：

```text
2D_ADMM/admm_unrolled.py
2D_ADMM/synthetic_isar_dataset.py
2D_ADMM/train_admm_unrolled.py
2D_ADMM/tests/test_admm_unrolled.py
```

实现细节：

- `admm_unrolled.py` 使用 PyTorch 复数 FFT，公共接口是实部/虚部双通道；
  每层只有 `c`、`tau`、`beta` 三个标量，均通过 softplus 参数化；
- `synthetic_isar_dataset.py` 生成 3--30 个随机复散射点、随机 SNR 的复回波，
  支持缓存或按索引动态生成，并确保不同数据划分使用不同随机种子；
- `train_admm_unrolled.py` 提供归一化图像/回波/稀疏损失、CPU/GPU 训练、
  checkpoint、训练历史、每层参数 CSV，以及 Yak-42 推理入口；
- `tests/test_admm_unrolled.py` 覆盖形状、FFT 伴随关系、梯度、参数约束、
  零输入稳定性和数据集可复现性。

现有文件保持不变：

```text
2D_ADMM/admm_2d.py
2D_ADMM/yak42_admm_demo.py
2D_ADMM/tests/test_admm_2d.py
```

现有固定 Fast 2D-ADMM 作为内部基线，不加入传统对照方法。

阶段一已在本机完成一次默认训练：20 轮 CPU 训练约 50 秒，验证损失从
`0.784680` 降到 `0.109051`，测试总损失为 `0.083766`。同一测试集上的固定
40 次 Fast 2D-ADMM 平均图像 NMSE 为 `0.547422`、平均图像熵为 `4.349631`。
两者损失定义和输出尺度不同，这个结果只说明原型训练管线可运行，不能直接
解释为论文级优越性；尚未完成全尺寸合成数据适配或论文级消融实验。

## 8. 测试与验收标准

必须测试：

- 实部/虚部输入输出形状；
- FFT 正向模型与现有 Fast 2D-ADMM 方向一致；
- 前向输出没有 NaN 或 Inf；
- 所有学习参数范围合法；
- 所有可学习参数具有有限梯度；
- 空输入稳定；
- 小数据集过拟合；
- 训练 loss 下降；
- 与固定 40 次 Fast 2D-ADMM 比较；
- Yak-42 输出尺寸为 512×128；
- Yak-42 图像熵、推理耗时和结果图可以保存。

第一阶段成功标准：

- 网络可以完成训练；
- 验证损失下降；
- 输出稳定且无 NaN；
- 8 层网络结果不明显差于固定参数 40 次 ADMM；
- 稳定时争取取得更低 NMSE 或图像熵。

## 9. 服务器迁移说明

当前不需要立即迁移。需要迁移时，服务器新对话应先读取：

```text
2D_ADMM/memory_2DADMM.md
2D_ADMM/admm_unrolled_network_plan.md
2D_ADMM/readpaper.md
```

服务器环境建议：

- Python 3.12；
- PyTorch 与服务器 CUDA 驱动匹配；
- NumPy、SciPy、Matplotlib、pytest；
- 单卡 8--16 GB 显存即可开始；
- 24 GB 显存用于 CNN 近端模块或更大 batch；
- 只有大规模消融或更复杂网络才考虑 A100。

迁移后先做以下检查：

```text
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
pytest 2D_ADMM/tests
```

然后先运行小尺寸训练，再运行全尺寸训练。不要直接从服务器上开始大规模训练。

## 10. 当前状态

已完成：

- MATLAB 论文方法阅读；
- 普通 2D-ADMM Python 移植；
- Fast 2D-ADMM Python 移植；
- Yak-42 demo；
- 8 项基础测试；
- 固定 Fast 2D-ADMM Yak-42 验证。

已验证的固定 ADMM 基线：

```text
measurement_shape：256×64
image_shape：512×128
SNR：10.027920 dB
图像熵：4.276910
迭代次数：40
```

待实现：

- ADMM 展开网络；
- 合成 ISAR 数据集生成器；
- 网络训练脚本；
- 展开网络测试；
- 小尺寸和全尺寸训练结果。

尚未完成：

- 网络训练；
- 网络 Yak-42 结果；
- 服务器实验；
- CNN 近端扩展；
- 欠采样网络实验。
