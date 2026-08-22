# 2D_ADMM Project Memory

Last updated: 2026-08-23

This file is the handoff memory for future conversations working on
`/Users/undertale/Work/EMW_py/2D_ADMM`. Read it before making changes so the
project context, prior decisions, and verified results do not need to be
rediscovered.

## 1. User Goal And Communication Preferences

The user is studying the paper and code without prior ISAR imaging knowledge.
Explanations must start from the physical meaning and proceed in a continuous
logical chain. Do not jump directly from an optimization objective to an ADMM
update without defining every variable and explaining why each transformation
is valid.

Formula preference:

- Show formulas directly with readable Unicode symbols.
- Do not present LaTeX source code or fenced LaTeX blocks.
- Define `X_hat`, `arg min`, norms, superscripts, and intermediate variables
  before using them.
- Distinguish what is a physical quantity, an estimated quantity, an auxiliary
  variable, and a temporary abbreviation.

Implementation preference established in this project:

- Focus on the method proposed by the author: ordinary 2D-ADMM and Fast
  2D-ADMM.
- Do not run or port 2D-FFT, 2D-SL0, or 2D-GP-SOONE unless the user explicitly
  asks for comparison methods later.
- Use the existing Conda environment named `usual`.

## 2. Paper And MATLAB Project

Paper:

- Title: *Sparsity-Driven ISAR Imaging Based on Two-Dimensional ADMM*
- Author: Hamid Reza Hashempour
- Journal: IEEE Sensors Journal, 2020
- DOI: `10.1109/JSEN.2020.3006105`
- Local PDF: `10.1109jsen.2020.3006105.pdf`

The paper reconstructs a sparse ISAR image directly as a two-dimensional
matrix. Its central observation model is:

`S = F_a * X * F_r^T + Z`

Meaning:

- `S`: measured radar echo matrix.
- `X`: unknown two-dimensional ISAR reflectivity image.
- `F_a`: azimuth Fourier operator.
- `F_r`: range Fourier operator.
- `Z`: measurement noise.

The sparse reconstruction objective is:

`X_hat = arg min_X [ 1/2 * ||S - F_a * X * F_r^T||_F^2 + lambda * ||X||_1 ]`

Interpretation:

- `X_hat` is the estimated image, not the unknown true image.
- The first term requires the image to explain the measured echo.
- The second term encourages most pixels to become zero.
- `lambda` controls data fitting versus sparsity.

The main innovation has three layers:

1. Replace vectorized recovery and the huge Kronecker dictionary
   `Phi = F_r kron F_a` with direct two-dimensional matrix recovery using the
   two small separable dictionaries.
2. Replace the large inverse `(Phi^H Phi + delta I)^-1` with a closed-form
   residual backprojection. This is valid because normalized partial Fourier
   matrices have orthonormal rows, so `Phi Phi^H = I`, and Woodbury's identity
   applies.
3. Replace explicit Fourier matrix products with FFT/IFFT in the fast version.
   This is exact for the normalized DFT convention used by the paper; it is not
   valid for arbitrary sensing matrices.

Vector and matrix forms are exactly equivalent because:

`vec(F_a * X * F_r^T) = (F_r kron F_a) * vec(X)`

ADMM introduces:

- `X`: the data-consistent image.
- `B`: an auxiliary copy responsible for sparsity.
- `V`: the scaled dual variable that accumulates disagreement between `X` and
  `B`.
- `D(k) = B(k) - V(k)`: only a temporary abbreviation used by the X update; it
  is not another physical image.

One iteration is:

1. `D(k) = B(k) - V(k)`
2. `X(k+1) = D(k) - 1/(1+delta) * F_a^H * [F_a * D(k) * F_r^T - S] * F_r*`
3. `B(k+1) = soft(X(k+1) + V(k), lambda/delta)`
4. `V(k+1) = V(k) + X(k+1) - B(k+1)`

Important boundary: the paper assumes translational motion compensation and
MTRC correction are already complete, assumes approximately uniform rotation
and a small rotation angle, and does not solve phase errors or autofocus.

Original MATLAB files:

- `Yak42_ADMM_2D_example.m`: real-data example and original main entry point.
- `simulation_2D_ADMM.m`: synthetic LFM echo simulation.
- `admm_2D.m`: explicit matrix implementation.
- `admm_2D_fast.m`: FFT/IFFT implementation.
- `SL0_2D.m` and `GP_SOONE.m`: comparison methods, not part of the requested
  Python demo.
- `Entropy_img.m`: image entropy.
- `Yak42.mat`: MATLAB v5 file containing complex variable `y` with shape
  `256 x 256` and dtype equivalent to complex128.

## 3. Known MATLAB Issues And Chosen Behavior

The original `admm_2D.m` and `admm_2D_fast.m` have two stop-condition issues:

1. The loop condition uses hard-coded `cgtol=1e-4`, while the example passes
   `e=1e-5`. The loop can stop at `1e-4` before the requested tolerance is met.
2. When `ERROR <= e`, MATLAB executes `break` before assigning `im=im_k1`, so
   it may return the previous iterate instead of the newest acceptable one.

The user chose the corrected behavior for Python:

- One `tol` parameter controls relative image change.
- Always store and return the newest iterate before checking convergence.
- Preserve `max_iterations=40` as the default hard limit.

This means the Python code follows the paper's stated stop rule but is not
expected to match the MATLAB script pixel-for-pixel at its old stopping point.

Other MATLAB caveats found during review:

- `simulation_2D_ADMM.m` contains manual-toggle expressions such as
  `iter=1;200;` and `SNR_db=10;-10:5:30;`. MATLAB only assigns the first value,
  so the script does not directly perform the intended sweep.
- `Entropy_img.m` can evaluate `0*log(0)` and produce NaN.
- `GP_SOONE.m` can divide by zero at zero-valued pixels.
- The current MATLAB package does not contain a complete implementation of the
  paper's two-dimensional undersampling update with `Psi_a` and `Psi_r`.

## 4. Python Port

Created files:

- `admm_2d.py`
  - `ADMMResult`
  - `soft_threshold()`
  - `image_entropy()`
  - `admm_2d()` for explicit compatible Fourier dictionaries
  - `admm_2d_fast()` for the paper's normalized partial-DFT structure
- `yak42_admm_demo.py`
  - Loads `Yak42.mat`
  - Runs only Fast 2D-ADMM
  - Does not run any traditional comparison method
  - Saves `yak42_admm_fast.png`
- `tests/conftest.py`
- `tests/test_admm_2d.py`
- `requirements-python.txt`
- `PYTHON_README.md`

Fast API design decision:

`admm_2d_fast(measurements, image_shape, ...)` accepts the measurements and
output shape rather than `Fr` and `Fa`. The original MATLAB fast function only
uses the sizes of `Fr` and `Fa`, not their values. The Python signature makes
that restriction explicit instead of implying support for arbitrary matrices.

Yak-42 demo preprocessing:

1. Load complex `y`, shape `256 x 256`.
2. MATLAB columns `129:192` correspond to Python slice `128:192`, producing
   shape `256 x 64`.
3. Normalize by maximum magnitude.
4. Add deterministic complex Gaussian noise with seed `0` and
   `noise_std = 0.505 * 0.03 * sqrt(2)` per real/imaginary component.
5. Apply IFFT along axis 0.
6. Reconstruct a `512 x 128` image.
7. Use defaults `alpha=0.0065`, `tol=1e-5`, `delta=1`, and
   `max_iterations=40`.
8. Apply `fftshift` on image axis 1 and save a Doppler-versus-range contour.

## 5. Environment

Conda installation:

- Conda root: `/Users/undertale/miniforge3`
- Environment: `/Users/undertale/miniforge3/envs/usual`
- Python: `3.12.13`

Verified packages in `usual`:

- NumPy `2.5.1`
- SciPy `1.18.0`
- Matplotlib `3.11.1`
- pytest `9.1.1`
- PyTorch `2.13.0`

Missing packages: none.

The Codex shell may not expose `conda` on `PATH`. Use the explicit executable
when necessary:

`/Users/undertale/miniforge3/bin/conda run -n usual ...`

## 6. Verified Tests And Demo Result

The original baseline tests were run successfully on 2026-08-03:

`/Users/undertale/miniforge3/bin/conda run -n usual pytest 2D_ADMM/tests -q`

Result at that time:

- `8 passed in 7.37s`
- Includes explicit-versus-fast equivalence on a small DFT model, complex soft
  thresholding, zero measurement stability, corrected stop behavior, invalid
  dimension checks, and Yak-42 preprocessing dimensions.

After adding the unfolding network and its training smoke test, the current
full suite is `16 passed` (2026-08-23).

## 7. ADMM 展开网络实现状态（2026-08-23）

本轮已按 `admm_unrolled_network_plan.md` 实现第一版轻量物理模型引导网络，
新增文件：

- `admm_unrolled.py`：PyTorch 版 8 层 ADMM 展开；实部/虚部双通道输入输出；
  固定归一化部分 Fourier 正向/伴随算子；每层学习正的 `c`、非负的 `tau`、
  正的 `beta`，通过 softplus 参数化；不含 CNN、U-Net、Transformer 或扩散模块。
- `synthetic_isar_dataset.py`：随机 3--30 个复散射点、随机位置/幅度/相位、
  SNR -10 至 30 dB 的合成数据；训练、验证、测试通过不同 seed 生成独立场景。
- `train_admm_unrolled.py`：归一化图像/回波/稀疏损失，训练、验证、测试、
  checkpoint、历史记录、每层参数 CSV，以及 Yak-42 推理和 PNG 保存。
- `tests/test_admm_unrolled.py`：网络和数据管线测试。

网络每层实际执行：

`D = B - V`

`X_next = D - c * F^H(F(D) - S)`

`B_next = complex_soft_threshold(X_next + V, tau)`

`V_next = V + beta * (X_next - B_next)`

当前阶段一默认配置：观测 `64x32`、图像 `128x64`、8 层、训练/验证/测试
`1024/128/128`、batch 8、20 epochs。已在本机 `usual` 环境 CPU 跑通：每轮约
2.5 秒；验证总损失从 `0.784680` 降至 `0.109051`，测试总损失 `0.083766`。
产物位于：

`2D_ADMM/outputs/admm_unrolled_stage1/`

其中包括 `admm_unrolled_checkpoint.pt`、`history.json`、
`learned_parameters.csv` 和 `yak42_unrolled.png`。训练脚本还会在
`history.json` 的最后一条记录中保存固定 Fast 2D-ADMM 的同测试集基线。

Yak-42 推理输出已验证为 `512x128`，SNR `10.027920 dB`，无 NaN，图像熵
`2.339475`。这里的 checkpoint 只在小尺寸合成数据上训练，直接用于 Yak-42
只是验证算子、尺寸和保存流程，不能声称已经完成全尺寸适配，也不能声称优于
固定 40 次 Fast 2D-ADMM（现有 Yak-42 基线熵约 `4.276910`）。在同一阶段一
合成测试集上，固定基线平均图像 NMSE `0.547422`、平均熵 `4.349631`，而网络
测试图像损失 `0.063644`；损失定义不同，不能把它们当成同一指标直接比较。
下一步是阶段二全尺寸
合成数据适配，再做同场景 NMSE/熵/耗时的严格对比。

服务器新对话继续工作时，先读本文件、`admm_unrolled_network_plan.md`、
`readpaper.md`，然后从阶段二开始，不要重复声称网络已有论文级效果。

Demo command:

`/Users/undertale/miniforge3/bin/conda run -n usual python 2D_ADMM/yak42_admm_demo.py`

Verified demo output:

- Measurement shape: `256 x 64`
- Reconstruction shape: `512 x 128`
- SNR: `10.027920 dB`
- Fast 2D-ADMM elapsed time: `0.060880 s`
- Iterations: `40`
- Final relative change: `5.633205e-03`
- Converged to `tol=1e-5`: `False` because the 40-iteration hard limit was
  reached first
- Image entropy: `4.276910`
- Paper's approximately 10 dB Yak-42 entropy for 2D-ADMM: about `4.27`
- Output image: `yak42_admm_fast.png`, `1512 x 1260`, about 101 KiB

The `converged=False` result is not a runtime failure. The reconstruction is
valid at the author's 40-iteration limit, and its entropy closely matches the
paper. Increasing `--max-iterations` would test convergence to the stricter
corrected tolerance but would no longer use the author's default limit.

During the sandboxed run, Matplotlib could not write
`/Users/undertale/.matplotlib` and used a temporary cache. This warning did not
affect the reconstruction or image.

## 8. Reports And Visual Assets

Existing reports:

- `article_report.html`: detailed Chinese technical paper report.
- `isar_beginner_guide.html`: beginner-oriented visual explanation.
- `report-assets/paper-method-page.png`
- `report-assets/paper-simulation-page.png`
- `report-assets/paper-yak42-page.png`

The HTML reports open directly in a browser and do not need a server.

## 9. Recommended Continuation Checklist

When a future conversation resumes this project:

1. Read this file, `PYTHON_README.md`, and the relevant Python module before
   changing behavior.
2. Keep the Fast DFT restriction explicit; do not describe it as a general
   sensing-matrix solver.
3. Do not reintroduce traditional methods unless the user asks.
4. If algorithm code changes, run the eight tests and then rerun the Yak-42
   demo, comparing entropy and dimensions against the verified values above.
5. If explaining the math, build the chain in this order:
   measured echo -> unknown image -> ill-posed inverse problem -> sparsity
   prior -> objective -> auxiliary B -> dual V -> X subproblem -> large inverse
   -> Fourier row orthogonality -> Woodbury simplification -> FFT fast path.
6. Use directly readable formulas rather than LaTeX source.

## 10. ADMM 展开网络与服务器迁移计划

The requested lightweight, physics-guided ADMM unrolled network is now
implemented. The staged scope and migration criteria remain in:

- `admm_unrolled_network_plan.md`

Read that file before implementing the network. It is the source of truth for
the staged scope, interfaces, data generation, loss, tests, and migration
criteria.

Current objective:

- Validate whether an 8-layer ADMM unrolled network can learn a faster or
  better reconstruction than fixed-parameter Fast 2D-ADMM.
- Train on synthetic sparse ISAR scenes with known ground truth.
- Use Yak-42 only as a final real-data test; do not train or fine-tune on it in
  the first version.

Current resource decision:

- First prototype runs on the local Apple M4 MacBook Air CPU.
- The machine has 10 CPU cores, an 8-core Apple GPU, and 16 GB unified memory.
- PyTorch 2.13 reports MPS built but unavailable and CUDA unavailable, so the
  current environment is CPU-only.
- No A100 is needed for the scalar-parameter 8-layer prototype.
- If a GPU server is later used, start with one 8--16 GB GPU; use 24 GB or more
  only for CNN proximal modules, larger batches, or extensive ablations.

Locked first-version choices:

- Stage one uses synthetic observations of 64×32 and target images of 128×64.
- Stage two uses Yak-42 dimensions 256×64 to 512×128.
- The first network has 8 unrolled ADMM layers.
- Each layer learns a positive data step `c(k)`, nonnegative threshold `tau(k)`,
  and positive dual coefficient `beta(k)`.
- The first network keeps soft-thresholding and does not add a CNN proximal
  module.
- First training uses complete sampling, not undersampling masks.
- The loss is normalized image error plus `0.1` echo-consistency loss plus
  `0.0001` sparsity loss.
- Existing fixed Fast 2D-ADMM is the internal baseline.
- 2D-FFT, 2D-SL0, and 2D-GP-SOONE are excluded unless explicitly requested.

Data plan:

- Generate random scenes with 3--30 scatterers, random positions, amplitudes,
  phases, and SNR from −10 to 30 dB.
- Use 1024 training, 128 validation, and 128 test scenes for the first small
  experiment.
- Keep train, validation, and test scenes distinct; changing only noise seeds
  is not an acceptable split.

Implemented files:

- `admm_unrolled.py`
- `synthetic_isar_dataset.py`
- `train_admm_unrolled.py`
- `tests/test_admm_unrolled.py`

State distinction:

- Completed: MATLAB method reading, Python ordinary/Fast ADMM, Yak-42 demo,
  baseline tests, fixed-ADMM Yak-42 validation, unrolled network, synthetic
  dataset generator, training script, and network tests.
- Completed prototype: stage-one 1024/128/128 training on CPU, checkpoint and
  fixed Fast 2D-ADMM same-test-set comparison. The validation total loss fell
  from `0.784680` to `0.109051`; fixed baseline mean image NMSE was `0.547422`.
- Pending: stage-two full-size synthetic adaptation, strict equal-metric
  comparisons, ablations, and any server runs. Do not present the stage-one
  prototype or direct Yak-42 transfer as a paper-level improvement.

When explaining this work, preserve the user's preferred continuous logic:
measured echo -> unknown image -> ill-posed inverse problem -> sparsity prior
-> objective -> auxiliary B -> dual V -> X subproblem -> large inverse ->
Fourier row orthogonality -> Woodbury simplification -> FFT fast path ->
unrolled learnable parameters. Use directly readable formulas, not raw LaTeX.

## 11. Broader Recognition Project And 2026 Plan

The 2D-ADMM work supports a broader project for recognizing aircraft and ship
targets from ISAR images under both interference-free and interference
conditions.

Current project situation as of 2026-08-07:

- Measured data from another group is still pending.
- The near-term direction is a learned surrogate for fast ISAR simulation.
- Generated ISAR images will be used to iterate the existing recognition
  network and later supplement the limited measured-image set.
- The intended loop is physical/baseline simulation -> learned fast generation
  -> recognition training and evaluation -> targeted hard-sample generation ->
  measured-data calibration when the external data arrives.
- The network-based simulation itself has not yet been implemented. In status
  reports it may be described accurately as being under investigation, design,
  and early-stage development, but completed quantitative results should not be
  invented.
- When writing management-facing progress, do not mention the source paper.
  Describe the direction as the result of extensive technical research.

Planning document:

- `recent_progress_and_weekly_plan_2026.md`
- It contains a Chinese progress summary and 21 rolling seven-day work periods
  from 2026-08-07 through 2026-12-31 inclusive.
