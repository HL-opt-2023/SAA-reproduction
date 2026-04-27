# Sample average approximation for convex stochastic programming

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

## Problem instances

### Experiment 1 — light-tailed stochastic quadratic program (Section 4.1)

A population-level linear regression instance, following Fan et al.
(2014) and Liu et al. (2019):

```
min_{x ∈ ℝ^d}  E_{(a,b)} [ (a'x - b)^2 ]
```

* `a ∈ ℝ^d` is a centered Gaussian random vector with covariance
  `Σ_{i,j} = 0.5^|i-j|`.
* `b = a'·x* + w`, where `w ~ 𝒩(0, 1)` is independent of `a`.
* The optimal solution is `x*`, generated once per instance:
  pick `r = min{d, 200}` indices uniformly at random as the support
  set `S`; let `x*_S = 1.5 u / ‖u‖_{1.8}` and `x*_{S^c} = 1.5 v / ‖v‖_{1.8}`,
  with `u, v` i.i.d. standard Gaussian.  This construction keeps
  `‖x*‖_{1.8}` small for all dimensions while leaving `x*` non-sparse.
* For each `(d, N)` we draw an i.i.d. sample `(a_j, b_j)`, `j=1..N`.

Reported in paper: Tables S1–S3, Figures 2 and 3.

### Experiment 2 — heavy-tailed stochastic utility problem (Section 4.2)

Adapted from Nemirovski et al. (2009):

```
min_{x ∈ ℝ^d}  E[ f(x, ξ) ],

f(x, ξ) = ϕ( Σ_{i=1}^d (i/d + r_i) x_i )
        + (M/2) Σ_i (x_i - 1)_+^2  +  (M/2) Σ_i (-x_i - 1)_+^2
```

* `ϕ(t) = max_{k=1..10} (v_k + s_k t)` is piecewise affine, with
  `v_k, s_k ~ 𝒩(0, 1)` drawn **once** at the start of the experiment
  and fixed thereafter (bundled in `data/exp2/piecewise_parameter.mat`).
* The two `(M/2)·(·)_+^2` terms (with `M = 1000`) are quadratic
  penalties enforcing the box `{-1 ≤ x_i ≤ 1}`.
* Heavy-tailed noise: `r_i = ν_i - E[ν_i]`, with `ν_i` i.i.d. drawn
  from a Pareto Type I distribution (shape 3.01, scale 1).  This
  yields a subgradient with bounded second moment but heavy tails.
* No closed-form `x*` — a high-fidelity reference is computed by
  solving the unregularized Sample Average Approximation (SAA) at
  `N_ref = 5000` for each `d`, cached in `cache/x_ref_d<d>.mat`.

Reported in paper: Tables S4–S8, Figures 4 and 4b.

### Sweep grid (both experiments)

* `d ∈ {100, 200, …, 900, 1000, 1500, 2000, 5000}` (13 dimensions)
* `N ∈ {200, 400, 600}`
* 5 independent replications per `(N, d)`

## Algorithms and methods involved in the experiments

Both experiments share the SAA-family methods (A)–(D) below; Experiment
2 additionally includes the two stochastic mirror descent variants (E)
and (F).

(A) **SAA_r** — vanilla SAA solving the empirical objective
    `(1/N) Σ_j ℓ(x; ξ_j)`, started from `x_0 ~ Uniform(-0.5, 0.5)^d`.

(B) **SAA_0** — same SAA formulation, started from `x_0 = 0`.

(C) **SAA-L_{q'}** — Tikhonov-regularized SAA
    `min_x  (1/N) Σ_j ℓ(x; ξ_j)  +  (λ_0 / 2) ‖x‖_{q'}^2`
    for `q' ∈ {1.01, 1.5, 2}`, started from the same `x_0` as SAA_r.

(D) **LASSO** — Tibshirani (1996) formulation
    `min_x  (1/N) Σ_j ℓ(x; ξ_j)  +  λ_0 ‖x‖_1`,
    solved by iterative soft-thresholding (Chambolle et al. 1998),
    started from the same `x_0` as SAA_r.

(E) **SMD-L1** — entropic stochastic mirror descent (Nemirovski et al.
    2009), applied to the simplex reformulation of `‖x‖_1 ≤ R_{ℓ1}`
    given by Eq. (93) of the paper, with `R_{ℓ1} = 2.5 · ‖x*‖_1`.
    Step size `γ = θ · √(2 ln(2d+1)) / (M̃_∞ √N)`, where `θ` is
    selected by cross-validation per Appendix E.  *(Experiment 2 only.)*

(F) **SMD-L2** — robust stochastic approximation (Nemirovski et al.
    2009), also based on the reformulation in Eq. (93) of the paper,
    here in the 2-norm-constrained form
    `min { F(x) : ‖x‖_2 ≤ R_{ℓ2} }` with `R_{ℓ2} = 2.5 · ‖x*‖_2`.
    Step size `γ = θ · ‖x*‖_2 / (M̃_2 √N)`; initial point at the
    origin; Euclidean projection onto the 2-norm ball each iteration.
    *(Experiment 2 only.)*

The pairings used in the paper's Figure 4 are (i) SAA-L_{1.01} versus
SMD-L1 (matched 1-norm geometry) and (ii) SAA-L_2 versus SMD-L2
(matched 2-norm geometry).

## Metrics

* **Approximate suboptimality gap.**
  Experiment 1 uses the population-level closed form
  `(x − x*)' Σ (x − x*)` (which equals `E[(a'x − b)^2] − E[(a'x* − b)^2]`).
  Experiment 2 uses Monte-Carlo:
  `Gap(x) = (1/n_test) Σ_j [f(x, ξ̃_j) − f(x*, ξ̃_j)]` with
  `n_test = 10⁴`.
* **ℓ₂-loss.**  `‖x − x*‖_2`.
* **Wall-clock runtime.**  Per-call solver time in seconds.

For Experiment 2 Figure 4, the paper's relative-difference metric is
also computed:

```
RelDiff(x) = 100 · ( Gap(x_SMD) − Gap(x_SAA) )
                  / ( (1/n_test) Σ_j f(x*, ξ̃_j) )    %
```

## Repo layout

```
.
├── README.md                  ← this file
├── data/                      ← problem coefficients (loaded once, fixed)
│   └── exp2/
│       └── piecewise_parameter.mat   (v, s for ϕ(t) = max_k(v_k + s_k·t))
├── exp1_linreg/
│   ├── run_exp1.m                    main driver (single-cell + aggregator)
│   └── cv_lambda.m                   cross-validate λ_0 for SAA-L_q' / LASSO
├── exp2_utility/
│   ├── run_exp2.m                    main driver (single-cell + aggregator)
│   ├── cv_lambda.m                   cross-validate λ_0 for SAA-L_q' / LASSO
│   └── cv_theta.m                    cross-validate θ for SMD-L1 / SMD-L2
├── slurm/                            HPC batch scripts
│   ├── run_exp1.slurm                aggregator job
│   ├── sweep_exp1.slurm              SLURM array job — one task per
│   │                                  (sample size N, dimension d)
│   │                                  combination (3 × 13 = 39 tasks)
│   ├── run_exp2.slurm                aggregator job
│   ├── sweep_exp2.slurm              same parallel scheme for
│   │                                  Experiment 2
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

### Parallel SLURM run

The expensive part of each experiment is the main sweep — for every
combination of sample size `N ∈ {200, 400, 600}` and dimension
`d ∈ {100, …, 5000}`, five replications are run for each algorithm.
Rather than executing all of these one-by-one in a single job, the
provided SLURM scripts launch a job array in which each task handles a
single `(N, d)` combination.  With three sample sizes and thirteen
dimensions there are 39 such tasks per experiment.

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

Each array task writes its results to a separate `.mat` file under
`<experiment>/cache/sweep/`, named after the indices of the sample
size and dimension it handled.  The aggregator job then loads every
such file, merges them into a single `<experiment>/results/exp{1,2}_results.mat`
containing the full suboptimality / loss / runtime tables, and
produces the figures from the paper.

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
* **Cross-validation** is controlled by a single flag at the top of
  `run_exp{1,2}.m`:

  ```matlab
  run_cross_validation = false;   % use hardcoded picks (default)
  run_cross_validation = true;    % invoke cv_lambda()/cv_theta() inline
  ```

  When the flag is `false` (default) the script uses the values that
  every prior cross-validation chose under the bundled coefficient
  file.  When `true`, the script calls the cross-validation routines
  (which can also be run standalone from the MATLAB prompt:
  `lambda_best = cv_lambda();`).
  * `exp{1,2}/cv_lambda.m` — λ_0 cross-validation on `(d=1000,
    N=200)` over the paper's grid `{0.01, 0.05, …, 0.5}`.
    Reports the best λ for each of `SAA-L_{q'}` and `LASSO`,
    independently.  In our experiments the grid endpoint `0.5` was
    always selected.
  * `exp2_utility/cv_theta.m` — θ cross-validation (Appendix E) for
    the SMD step size on `(d=1000, N=600)` over the 45-candidate
    grid `{a·b : a∈1..9, b∈{0.1, 1, 10, 100, 1000}}`.  Selected
    values under `data/exp2/piecewise_parameter.mat` are
    `θ_SMD-L1 = 6` and `θ_SMD-L2 = 0.6`.  Requires
    `cache/x_ref_d1000.mat` (run the bootstrap stage of
    `run_exp2.m` first).
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

## References

The papers explicitly cited above (consistent with the bibliography
in Liu & Tong 2026):

* **Liu, H. and Tong, J. (2026).** Metric entropy-free sample
  complexity bounds for sample average approximation in convex
  stochastic programming.  *Mathematical Programming.*
  https://doi.org/10.1007/s10107-026-02335-3
* **Chambolle, A., De Vore, R. A., Lee, N.-Y., and Lucier, B. J. (1998).**
  Nonlinear wavelet image processing: variational problems, compression,
  and noise removal through wavelet shrinkage.  *IEEE Transactions on
  Image Processing*, 7(3):319–335.
* **Fan, J., Xue, L., and Zou, H. (2014).**  Strong oracle optimality
  of folded concave penalized estimation.  *Annals of Statistics*,
  42(3):819–849.
* **Liu, H., Wang, X., Yao, T., Li, R., and Ye, Y. (2019).**  Sample
  average approximation with sparsity-inducing penalty for
  high-dimensional stochastic programming.  *Mathematical Programming*,
  178:69–108.
* **Nemirovski, A., Juditsky, A., Lan, G., and Shapiro, A. (2009).**
  Robust stochastic approximation approach to stochastic programming.
  *SIAM Journal on Optimization*, 19(4):1574–1609.
* **Tibshirani, R. (1996).**  Regression shrinkage and selection via
  the lasso.  *Journal of the Royal Statistical Society, Series B
  (Statistical Methodology)*, 58(1):267–288.

## Contact

Hongcheng Liu — liu.h@ufl.edu
GitHub: [HL-opt-2023](https://github.com/HL-opt-2023)
