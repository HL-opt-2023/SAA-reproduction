# SAA Numerical Experiments — Reproducibility Code

This repository reproduces the two numerical experiments in

> Liu, H. and Tong, J. (2026).
> *Metric entropy-free sample complexity bounds for sample average
> approximation in convex stochastic programming*.
> Mathematical Programming.
> https://doi.org/10.1007/s10107-026-02335-3

> **Disclaimer.** The repository layout, helper scripts, READMEs, and
> SLURM batch files were reorganized by [Claude Code](https://claude.com/claude-code)
> for clarity and reproducibility.  The numerical algorithms, problem
> formulation, and experimental design follow the paper.

A brief description of each experiment:

* **Experiment 1** ([`exp1_linreg/`](exp1_linreg/)) — light-tailed
  stochastic linear regression (paper Section 4.1, Tables S1–S3, Fig 2/3).
  Regressor `a ∼ 𝒩(0, Σ)` with `Σᵢⱼ = 0.5^|i−j|`; response
  `b = a'·x* + w` with `w ∼ 𝒩(0,1)`.
* **Experiment 2** ([`exp2_utility/`](exp2_utility/)) — heavy-tailed
  stochastic utility problem (paper Section 4.2, Tables S4–S8,
  Fig 4/4b).  Subgradient admits a bounded second moment but is
  heavy-tailed.

## Repo layout

```
.
├── README.md                  ← this file
├── data/                      ← problem coefficients (loaded once, fixed)
│   └── exp2/
│       ├── piecewise_parameter.mat   (v, s for ϕ(t) = max_k(v_k + s_k·t))
│       └── x_init.mat                (per-d initial points for SAA-L*)
├── exp1_linreg/
│   └── run_exp1.m                    main driver (single-cell + aggregator)
├── exp2_utility/
│   └── run_exp2.m                    main driver (single-cell + aggregator)
├── slurm/                            HPC batch scripts
│   ├── run_exp1.slurm                aggregator job
│   ├── sweep_exp1.slurm              13 d × 3 N = 39-task array (parallel)
│   ├── run_exp2.slurm                aggregator job
│   ├── sweep_exp2.slurm              39-task array (parallel)
│   ├── precompute_xref.slurm         x_ref reference solve (Exp 2 bootstrap)
│   └── deploy.sh                     rsync local → cluster + sbatch
└── results/                          generated outputs
    ├── exp1/                         exp1_results.mat + Fig 2/3 PNGs
    └── exp2/                         exp2_results.mat + Fig 2/3/4 PNGs
```

## Reproducing the experiments

### Prerequisites

* MATLAB 2024a or newer (uses `containers.Map`, `tiledlayout`).
* For the parallel SLURM runs: a SLURM cluster with a MATLAB module.
  The `slurm/*.slurm` files contain placeholders `--account=YOUR_ACCOUNT`
  and `--qos=YOUR_QOS` that you must edit for your cluster.  Output paths
  are written under `./logs/` relative to the submission directory.

### One-shot local run

```matlab
% Experiment 1
cd exp1_linreg
run_exp1                              % takes a few hours serially

% Experiment 2
cd ../exp2_utility
run_exp2                              % takes ~10 hours serially
```

The data files in `data/exp2/` are loaded automatically by
`run_exp2.m`; `run_exp1.m` generates problem instances on-the-fly with
seeded RNG.

### Parallel SLURM run (13 d × 3 N = 39-cell sweep)

```bash
# 1.  Push code + data to the cluster.
bash slurm/deploy.sh

# 2.  (Exp 2 only.)  Compute the high-fidelity reference x_ref{d}.
ssh CLUSTER "cd $SAA_ROOT/slurm && sbatch precompute_xref.slurm"

# 3.  Submit the parallel cell sweep (one task per (N, d)).
ssh CLUSTER "cd $SAA_ROOT/slurm && sbatch sweep_exp1.slurm"
ssh CLUSTER "cd $SAA_ROOT/slurm && sbatch sweep_exp2.slurm"

# 4.  Once all 39 cells per experiment are on disk, run the aggregator.
ssh CLUSTER "cd $SAA_ROOT/slurm && sbatch run_exp1.slurm"
ssh CLUSTER "cd $SAA_ROOT/slurm && sbatch run_exp2.slurm"
```

The sweep array writes per-cell `.mat` files to
`<exp_dir>/cache/sweep/iN<i>_id<j>.mat`; the aggregator merges them
into the final `<exp_dir>/results/exp{1,2}_results.mat` and produces
the figures.

## Solver design choices (matching the paper)

* **SAA-L_q'** and **LASSO**: subgradient (resp. proximal-gradient)
  with **fixed step `1e-4`**, no descent gate, terminate when
  `max|Δ obj| / max(1, mean|obj|) < 10⁻⁹` over a 10-iter window or
  `maxit = 500 000` reached.
* **SAA_r / SAA_0**: same scheme, λ = 0.
* **SMD-L1**: entropic mirror descent on the simplex reformulation of
  `‖x‖_1 ≤ R_ℓ1` (Nemirovski et al. 2009 Eq. (93)).
* **SMD-L2**: robust SA in the 2-norm setting with Euclidean projection
  on `‖x‖_2 ≤ R_ℓ2`.
* **Cross-validation**:
  * lambda cross-validation on (d=1000, N=200) over the paper's grid `{0.01, 0.05, …,
    0.5}`; in our experiments the grid endpoint `0.5` was always
    selected, so the published `run_exp{1,2}.m` hard-codes `0.5`.
  * theta cross-validation (Appendix E) for SMD step size on (d=1000, N=600) over
    `{a·b : a∈1..9, b∈{0.1, 1, 10, 100, 1000}}`.  Selected values
    under `data/exp2/piecewise_parameter.mat`:
    `θ_SMD-L1 = 6`, `θ_SMD-L2 = 0.6`.
* **High-fidelity reference solution**: per-`d`, an unregularized SAA
  solve with `N_ref = 5000` cached in `cache/x_ref_d<d>.mat`.

## Coefficient regeneration

The deterministic problem coefficients `(v_k, s_k)` are bundled in
`data/exp2/piecewise_parameter.mat` and reused across every solver,
cross-validation, and replication.  **Do not regenerate them between experiments**
— the same `(v, s)` must be applied uniformly.

To regenerate them (e.g., to test seed sensitivity), run

```matlab
rng(1);  v = randn(10, 1);  s = randn(10, 1);
save('data/exp2/piecewise_parameter.mat', 'v', 's');
```

and rerun the entire pipeline (re-bootstrap → re-run-cross-validation → re-sweep).

## License

This code is released under the MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the paper that introduces the
numerical experiments.  The version of record appears in
*Mathematical Programming*:

```bibtex
@article{LiuTong2026SAA,
  author  = {Liu, Hongcheng and Tong, Jindong},
  title   = {Metric entropy-free sample complexity bounds for
             sample average approximation in convex stochastic
             programming},
  journal = {Mathematical Programming},
  year    = {2026},
  doi     = {10.1007/s10107-026-02335-3}
}
```

## Contact

Code maintained by HL-opt-2023 (https://github.com/HL-opt-2023).
