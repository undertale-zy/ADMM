# ADMM 神经替代网络复刻报告

日期：2026-08-27

## 1. 结论

已按 `ADMM 神经替代网络完整复刻规范.txt` 完成 R1--R36 的架构级复刻：网络
forward、物理算子、三类数据、loss、selector、checkpoint schema、统一训练、
固定评测和历史总览图入口均已实现。当前本机是 CPU 环境，只完成单元测试、
小规模训练 smoke 和一次全尺寸评测链路 smoke，没有重新执行服务器正式训练。

因此本次交付能做到：重建同样的实验体系并在服务器重新训练。不能做到：在没有
原始 `.pt` checkpoint 的情况下逐权重恢复 R2--R36 的历史 CNN/Transformer。

## 2. Round 映射

| Round | 模型族 | 数据与主要变量 |
| --- | --- | --- |
| R1 | Scalar-8 | point，24 个 scalar |
| R2--R3 | CNN / Transformer proximal | point |
| R4--R7 | SupportFusion | proximal × background weight |
| R8--R11 | Stage-1 GuideSupportFusion | proximal × background weight，DC2 |
| R12--R15 | Deep Scalar | 12/16/24/32 层 |
| R16--R19 | Balanced Scalar | 12 层，observed echo weight sweep |
| R20--R23 | noise-aware SupportFusion | proximal × clean echo weight |
| R24--R27 | Structured-v1 | mixed probability .50/.25/.50/.75 |
| R28--R31 | Dense-v2 | CNN/Transformer × probability .25/.50 |
| R32 | Guide + CNN + DC8 | Dense-v2 p=.50 |
| R33 | confidence band + DC4 | Dense-v2 p=.50 |
| R34 | Stage-1 residual CNN | Dense-v2 p=.50 |
| R35 | guide support + physics DC8 | Dense-v2 p=.50 |
| R36 | 与 R33 相同的网络 | Dense-v2 p=.75 + support hinge |

精确逐轮配置由不可变 `ROUND_REGISTRY` 保存，`build_model()`、
`build_dataset()` 和 `compute_round_loss()` 以 Round 编号统一派发。

## 3. 参数量验收

| 模型 | 参数量 |
| --- | ---: |
| R1 Scalar-8 | 24 |
| R2 CNN | 19,707 |
| R3 Transformer | 346,683 |
| Support CNN / Transformer | 19,711 / 346,687 |
| Guide CNN / Transformer | 19,712 / 346,688 |
| R32 | 19,718 |
| R33 | 19,714 |
| R34 | 30,149 |
| R35 | 19,720 |
| R36 | 19,714 |

这些数值已由自动测试在默认 512x128 图像尺寸下逐项断言。

## 4. 历史兼容性

默认模式是 `historical`，明确保留：

- R2/R3、R12--R15 validation 使用默认 observed echo weight 0.1；
- R16--R19 忽略历史命令中的 CNN 参数，保持纯 scalar；
- R20--R23 constrained selector 检查 clean echo；
- R28--R36 参数名 `structured_probability` 实际表示 dense probability；
- R34 的连续 mask leakage；
- R35 父类未使用参数仍注册；
- R36 没有额外 distillation head；
- R32--R36 历史双栏图保留 Fast baseline measurement seed 错位。

`corrected` 只用于新实验：它修正 validation weight 或公平 baseline，并在 JSON
写入 protocol metadata。它不会改变 Round 的网络身份。

## 5. 本机验证

使用环境：Python 3.12.13、PyTorch 2.13.0、无 CUDA、无可用 MPS。

已完成：

- 全仓库 compileall；
- `52 passed`；
- 15 个代表 Round 的 forward、对应 loss、backward 与有限梯度检查；
- 同一批代表 Round 的 1-epoch CPU trainer smoke；
- 三类 checkpoint 严格 reload；
- R32 全尺寸 512x128 的 point/dense/Yak seed 0/1 evaluator smoke；
- JSON protocol、参数量、延迟和 2268x936 双栏 PNG 检查。

Fast-ADMM Yak-42 seed-0 联合锚点：

| 指标 | 本机值 |
| --- | ---: |
| entropy | 4.2769098825 |
| observed residual | 0.1875038803 |
| -40~-60 dB pixel fraction | 0.0006256103515625 |
| >-40 dB support | 0.005218505859375 |
| >-60 dB support | 0.0058441162109375 |
| iterations | 40 |

## 6. 服务器参考与限制

规范记录的 R32 正式参考为：point image NMSE `0.006898`、point clean echo
`0.006564`、dense image NMSE `0.171632`、dense clean echo `0.039110`、Yak band
`0.8200%`、Yak support `1.5981%`、entropy `2.689`。

这些是服务器历史锚点，不是本机 smoke 的结果。本机生成的未充分训练 checkpoint
只用于验证工程链路，不应写入模型效果表，也不能据此和 Fast-ADMM 比优劣。

正式复现实验按 `SERVER_RUNBOOK.md` 执行：先训练 R1 teacher，再训练依赖 guide
的 Round，最后使用 500 个 point、500 个 dense 和 Yak seed 0--9 做固定评测。
