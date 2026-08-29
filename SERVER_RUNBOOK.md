# R1--R36 服务器运行手册

这份手册只负责执行顺序。网络、数据、loss 和历史例外的完整定义在
`ADMM 神经替代网络完整复刻规范.txt`。

## 1. 环境与路径

推荐目录：

```text
<WORKSPACE_ROOT>/
  ADMM/       # 本仓库
  runs/       # checkpoint、metrics、figure
```

进入服务器后设置：

```bash
export REPO_ROOT=/path/to/workspace/ADMM
export RUN_ROOT=/path/to/workspace/runs
export ADMM_WORKSPACE=/path/to/workspace
export GUIDE_CHECKPOINT="$RUN_ROOT/admm_stage1_full_8gpu/checkpoint-last.pt"
export OMP_NUM_THREADS=1
cd "$REPO_ROOT"
```

环境自检：

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -m pytest tests -q
python train_round.py --round 32 --smoke --device cpu --output-dir /tmp/admm-r32-smoke
```

必须先看到全部测试通过和 smoke 产物，再启动正式训练。

## 2. 训练顺序

R1 是 R8--R11、R32--R36 的 frozen teacher，必须先完成：

```bash
torchrun --standalone --nproc-per-node=8 train_round.py \
  --round 1 --device cuda \
  --output-dir "$RUN_ROOT/admm_stage1_full_8gpu"

test -f "$GUIDE_CHECKPOINT"
```

R2--R3 使用 8 卡，可在不同节点并行：

```bash
torchrun --standalone --nproc-per-node=8 train_round.py --round 2 --device cuda \
  --output-dir "$RUN_ROOT/admm_stage2_cnnprox8"

torchrun --standalone --nproc-per-node=8 train_round.py --round 3 --device cuda \
  --output-dir "$RUN_ROOT/admm_stage3_transformer8"
```

R4--R7、R12--R31 使用 4 卡。下面的循环是串行运行；要并行时给每个任务分配
独立节点或互不重叠的 GPU：

```bash
for round_id in {4..7} {12..31}; do
  torchrun --standalone --nproc-per-node=4 train_round.py \
    --round "$round_id" --device cuda
done
```

R8--R11 和 R32--R36 显式加载 R1 teacher：

```bash
for round_id in {8..11} {32..36}; do
  torchrun --standalone --nproc-per-node=4 train_round.py \
    --round "$round_id" --device cuda \
    --guide-checkpoint "$GUIDE_CHECKPOINT"
done
```

未指定 `--output-dir` 时，统一 trainer 会按 registry 写入
`<WORKSPACE_ROOT>/runs/<run_directory>`。默认 `--compatibility historical`；
不要在历史目录上运行 `corrected`。

## 3. R32--R36 正式评测

以 R32 为例：

```bash
python evaluate_rounds32_36.py \
  --round 32 \
  --checkpoint "$RUN_ROOT/admm_round32_guide_dc/checkpoint-best.pt" \
  --guide-checkpoint "$GUIDE_CHECKPOINT" \
  --samples 500 \
  --yak-seeds 0 1 2 3 4 5 6 7 8 9 \
  --device cuda \
  --compatibility historical \
  --output-json "$RUN_ROOT/admm_round32_guide_dc/eval_round32_full.json" \
  --output-figure "$RUN_ROOT/admm_round32_guide_dc/figures/yak42_seed0_full.png"
```

R33--R36 更换 Round、目录和输出文件名。论文正式公平图应另用
`--compatibility corrected` 写入新文件，不能覆盖 historical artifact。

## 4. R1--R31 总览

```bash
python generate_all_complete_training_visualizations.py \
  --run-root "$RUN_ROOT" \
  --data "$REPO_ROOT/Yak42.mat" \
  --device cuda \
  --output-dir "$RUN_ROOT/all_complete_trainings"
```

输出包括 31 张单图、8 张 group 图、31-panel master 和 metrics JSON。

## 5. 验收边界

- smoke 只验证代码链路，不进入论文结果表。
- 正式 fixed evaluation 必须是 point 500、dense 500、Yak seed 0--9。
- 参数量与趋势先验见 `ADMM_NETWORK_REPLICATION.md`。
- 没有历史 CNN/Transformer checkpoint 时只能重训得到同类结果，不能逐 bit
  恢复服务器权重。
- 保存 PyTorch、CUDA、cuDNN、GPU 型号和命令；DDP、bf16 和 FFT kernel 会造成
  小数级训练差异。
