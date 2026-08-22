# Python Fast 2D-ADMM Demo

This port implements only the method proposed by the paper:

- explicit matrix 2D-ADMM from `admm_2D.m`;
- FFT-accelerated 2D-ADMM from `admm_2D_fast.m`;
- the relevant Yak-42 preprocessing and image-entropy calculation.

The 2D-FFT, 2D-SL0, and 2D-GP-SOONE comparison methods are intentionally
excluded from the Python demo.

## Environment

The existing Conda environment `usual` already contains all required packages:

| Package | Required version | Installed in `usual` |
| --- | ---: | ---: |
| Python | 3.12 | 3.12.13 |
| NumPy | 2.5.1 | 2.5.1 |
| SciPy | 1.18.0 | 1.18.0 |
| Matplotlib | 3.11.1 | 3.11.1 |
| pytest | 9.1.1 | 9.1.1 |

Missing packages: **none**.

No package installation is needed. `requirements-python.txt` records the
versions used by the environment for reproducibility.

## Run Later

Run the unit tests:

```bash
conda run -n usual pytest 2D_ADMM/tests
```

Run the Yak-42 demo from the repository root:

```bash
conda run -n usual python 2D_ADMM/yak42_admm_demo.py
```

The demo only runs Fast 2D-ADMM. By default it reads `Yak42.mat`, uses a fixed
noise seed, prints reconstruction metrics, and saves:

```text
2D_ADMM/yak42_admm_fast.png
```

Use custom paths or parameters when needed:

```bash
conda run -n usual python 2D_ADMM/yak42_admm_demo.py \
  --data 2D_ADMM/Yak42.mat \
  --output 2D_ADMM/yak42_custom.png \
  --seed 0 \
  --alpha 0.0065 \
  --tol 1e-5 \
  --max-iterations 40
```

## ADMM 展开网络

`admm_unrolled.py` implements the first physics-guided unfolding model. It
keeps the FFT forward model and its adjoint fixed, while learning three
positive scalar parameters at each of eight layers: the data-consistency step
`c`, soft threshold `tau`, and scaled-dual step `beta`. Inputs and outputs use
two real channels for the real and imaginary parts of a complex echo/image.

Run the planned stage-one experiment (64x32 measurements, 128x64 image,
1024/128/128 samples, 20 epochs) on the current CPU environment with:

```bash
/Users/undertale/miniforge3/envs/usual/bin/python \
  2D_ADMM/train_admm_unrolled.py \
  --device cpu \
  --output-dir 2D_ADMM/outputs/admm_unrolled_stage1
```

The script writes `admm_unrolled_checkpoint.pt`, `history.json`, and
`learned_parameters.csv`. A small smoke run can be made by overriding the
sample counts, dimensions, layers, and epochs. To evaluate a checkpoint on
the bundled Yak-42 data:

```bash
/Users/undertale/miniforge3/envs/usual/bin/python \
  2D_ADMM/train_admm_unrolled.py \
  --evaluate-yak42 2D_ADMM/outputs/admm_unrolled_stage1/admm_unrolled_checkpoint.pt \
  --device cpu \
  --yak42-output 2D_ADMM/outputs/admm_unrolled_stage1/yak42_unrolled.png
```

The checkpoint produced by the stage-one run is a scalar-parameter prototype.
Applying it to Yak-42 verifies the full-size operator and output path, but is
not a full-size adaptation experiment and must not be reported as a measured
network improvement over the fixed 40-iteration Fast 2D-ADMM baseline.

The default run also evaluates the fixed Fast 2D-ADMM on the same held-out
synthetic test scenes. In the recorded run its mean image NMSE was `0.547422`
and mean image entropy was `4.349631`; the unfolded network's normalized test
image loss was `0.063644`. These objectives are useful diagnostics but are not
directly interchangeable with the paper's Yak-42 entropy result.

## Intentional MATLAB Fixes

The MATLAB functions combine a hard-coded `cgtol=1e-4` loop condition with a
separate input tolerance. As a result, the example's `e=1e-5` does not control
the actual stop point. They can also `break` before assigning the newest image.

The Python implementation follows the paper's stated rule instead:

- one `tol` value controls relative image change;
- the newest image is stored before convergence is evaluated;
- `max_iterations` remains the hard iteration limit.

Consequently, the Python result is algorithmically faithful to the paper but is
not expected to match the MATLAB script pixel-for-pixel at its old stop point.

## Fast-Path Constraint

`admm_2d_fast()` is not a general replacement for arbitrary sensing matrices.
It implements the normalized partial-DFT convention used by the bundled
Yak-42 example. Use `admm_2d()` when explicit compatible dictionaries are
available.
