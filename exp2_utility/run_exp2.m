% Experiment 2 (Section 4.2 of Liu & Tong, 2024):
% Heavy-tailed SP — stochastic utility problem
%   min_{x in R^d}  E[f(x, xi)],
%   f(x,xi) = phi( sum_i (i/d + r_i) x_i )
%             + (M/2) sum_i (x_i - 1)_+^2 + (M/2) sum_i (-x_i - 1)_+^2,
%   phi(t)  = max_{k=1..10} (v_k + s_k t),   v_k, s_k ~ N(0,1) (fixed),
%   r_i     = nu_i - E[nu_i],  nu_i ~ Pareto(shape=3.01, scale=1).
%
% Methods compared:
%   SAA_r, SAA_0, SAA-L_{q'} (q' in {1.01, 1.5, 2}), LASSO,
%   SMD-L1 (entropic mirror descent on simplex), SMD-L2 (robust SA).

clear; close all; rng(0);

% --------------------- Configuration (matches paper) ---------------
d_list       = [100:100:900, 1000, 1500, 2000, 5000];
N_list       = [200, 400, 600];
num_reps     = 5;
q_prime_list = [1.01, 2.0];
lambda_grid  = [0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5];
gd_max_iter  = 500000;
gd_tol       = 1e-7;
M_pen        = 1000;     % penalty coefficient for box constraint
n_test       = 1e4;      % Monte Carlo for gap evaluation
% SMD stepsize cross-validation (Appendix E of the paper):
%   theta in {a*b : a=1..9, b in {0.1,1,10,100,1000}},
%   fix d=1000, N=600, 5 reps, two i.i.d. sample sets (optimize / validate).
theta_cv_grid = reshape((1:9)' * [0.1 1 10 100 1000], [], 1);
d_smd_cv     = 1000;
N_smd_cv     = 600;
outdir       = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(outdir, 'dir'); mkdir(outdir); end

% --------------------- phi parameters from data/exp2/ ---------------
% By default we load v_k, s_k from the bundled
%     <repo>/data/exp2/piecewise_parameter.mat
% so every experiment, cross-validation, and replication uses the same
% piecewise-affine phi.  The script aborts if the file is missing — to
% regenerate the coefficients from scratch see "Coefficient regeneration"
% in the top-level README.
pp_path = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'exp2', ...
                   'piecewise_parameter.mat');
if ~exist(pp_path, 'file')
    error(['piecewise_parameter.mat not found at %s.\n' ...
           'Run the snippet in README.md (section "Coefficient ' ...
           'regeneration") to (re)create it before launching run_exp2.'], ...
          pp_path);
end
pp = load(pp_path);
vk = pp.v(:);
sk = pp.s(:);
K  = numel(vk);
fprintf('Loaded vk, sk from %s (K = %d).\n', pp_path, K);

% Theoretical lower bound m_star = min_t max_k (v_k + s_k * t).
% Used as Polyak f_star floor for the unregularized bootstrap (lam=0):
% F_N(x) >= mean_j phi(coef_j' x) >= m_star for any x with no penalty.
m_star = fminbnd(@(t) max(vk + sk*t), -1e4, 1e4);
fprintf('m_star (lower bound for phi) = %.6f\n', m_star);

% Pareto sampled as nu = U^{-1/(alpha-1)} with effective tail index (alpha-1).
% This matches the paper's "bounded 2nd moment but heavy-tailed" wording:
% 1st & 2nd moments finite, 3rd infinite (Reading B convention).
alpha  = 3.01;
nu_bar = (alpha - 1) / (alpha - 2);   % effective Pareto mean

% --------------------- Compute / load reference solution x* per d ---
% Paper uses N=50,000; we subsample for compute budget. Cached to disk so
% follow-up runs skip the (slow) high-fidelity solves. Cache key: the tuple
% (d_list, N_ref, vk, sk, M_pen, alpha) — recomputed if any of these change.
N_ref          = 5e3;
xref_maxit     = 5000;     % bootstrap line-search budget (per d)
cache_dir = fullfile(fileparts(mfilename('fullpath')), 'cache');
if ~exist(cache_dir, 'dir'); mkdir(cache_dir); end
xref_path = fullfile(cache_dir, 'x_ref.mat');

% --- Single-d mode for parallel array job ---
% If env var XREF_ONLY_D_IDX is set (e.g., by SLURM_ARRAY_TASK_ID), compute
% x_ref for that one d and save to a per-d cache file, then exit. This lets
% an sbatch array fan all 13 d's out in parallel.
single_d_env = getenv('XREF_ONLY_D_IDX');
if ~isempty(single_d_env)
    id = str2double(single_d_env);
    if isnan(id) || id < 1 || id > numel(d_list)
        error('XREF_ONLY_D_IDX out of range: %s', single_d_env);
    end
    d = d_list(id);
    per_d_path = fullfile(cache_dir, sprintf('x_ref_d%d.mat', d));
    if exist(per_d_path, 'file')
        cd_ = load(per_d_path);
        if isfield(cd_, 'x_ref_one') && cd_.N_ref == N_ref ...
                && isequal(cd_.vk, vk) && isequal(cd_.sk, sk) ...
                && cd_.M_pen == M_pen && cd_.alpha == alpha
            fprintf('d=%d already cached at %s; nothing to do.\n', d, per_d_path);
            return;
        end
    end
    fprintf('Single-d mode: computing x_ref for d=%d (idx=%d), N_ref=%d, maxit=%d.\n', ...
            d, id, round(N_ref), xref_maxit);
    [X_xi, ~] = gen_xi(N_ref, d, alpha, nu_bar, 777);
    coef = bsxfun(@plus, (1:d)/d, X_xi);
    x0 = zeros(d, 1);
    x_ref_one = saa_solve(coef, vk, sk, 0, 2.0, M_pen, x0, xref_maxit, 1e-6);
    fprintf('  d=%d  ||x*||_inf=%.3f  ||x*||_2=%.3f\n', ...
            d, max(abs(x_ref_one)), norm(x_ref_one));
    save(per_d_path, 'x_ref_one', 'd', 'N_ref', 'vk', 'sk', 'M_pen', 'alpha');
    fprintf('Saved %s\n', per_d_path);
    return;
end

% --- Normal mode: assemble x_ref from main cache + per-d caches, fill rest ---
% In sweep (single-cell) mode (SWEEP_TASK_IDX env var set) we MUST NOT write
% the consolidated main cache (write contention with parallel tasks).
in_sweep_mode = ~isempty(getenv('SWEEP_TASK_IDX'));

x_ref = cell(numel(d_list), 1);
if exist(xref_path, 'file')
    try
        cached = load(xref_path);
        if isfield(cached, 'd_list') && isfield(cached, 'N_ref') ...
                && isequal(cached.d_list, d_list) && cached.N_ref == N_ref ...
                && isequal(cached.vk, vk) && isequal(cached.sk, sk) ...
                && cached.M_pen == M_pen && cached.alpha == alpha
            x_ref = cached.x_ref;
            fprintf('Loaded main x_ref cache from %s\n', xref_path);
        else
            fprintf('x_ref main cache stale (config changed); will rebuild.\n');
        end
    catch ME
        fprintf('main cache unreadable (%s); ignoring.\n', ME.message);
    end
end
% Pull in any per-d caches written by the array job
for id = 1:numel(d_list)
    if ~isempty(x_ref{id}); continue; end
    d = d_list(id);
    per_d_path = fullfile(cache_dir, sprintf('x_ref_d%d.mat', d));
    if exist(per_d_path, 'file')
        try
            cd_ = load(per_d_path);
            if isfield(cd_, 'x_ref_one') && cd_.N_ref == N_ref ...
                    && isequal(cd_.vk, vk) && isequal(cd_.sk, sk) ...
                    && cd_.M_pen == M_pen && cd_.alpha == alpha
                x_ref{id} = cd_.x_ref_one;
                fprintf('Loaded per-d cache d=%d\n', d);
            end
        catch ME
            fprintf('per-d cache d=%d unreadable (%s); skipping.\n', d, ME.message);
        end
    end
end
n_need = sum(cellfun(@isempty, x_ref));
fprintf('Reference x* (N_ref=%d, maxit=%d): %d / %d still to compute.\n', ...
        round(N_ref), xref_maxit, n_need, numel(d_list));
for id = 1:numel(d_list)
    if ~isempty(x_ref{id}); continue; end
    d = d_list(id);
    [X_xi, ~] = gen_xi(N_ref, d, alpha, nu_bar, 777);
    coef = bsxfun(@plus, (1:d)/d, X_xi);
    x0 = zeros(d, 1);
    x_ref{id} = saa_solve(coef, vk, sk, 0, 2.0, M_pen, x0, xref_maxit, 1e-6);
    fprintf('  d=%d  ||x*||_inf=%.3f  ||x*||_2=%.3f\n', ...
            d, max(abs(x_ref{id})), norm(x_ref{id}));
    if ~in_sweep_mode
        save(xref_path, 'x_ref', 'd_list', 'N_ref', 'vk', 'sk', 'M_pen', 'alpha');
    end
end
% Consolidate the main cache only outside of sweep mode (avoid write race).
if ~in_sweep_mode
    save(xref_path, 'x_ref', 'd_list', 'N_ref', 'vk', 'sk', 'M_pen', 'alpha');
end

% --------------------- lambda_0 -----------------------------------
% Toggle: when true, run cv_lambda(); when false, use the hardcoded
% picks from prior cross-validation under the bundled coefficients.
run_cross_validation = false;

methods_reg = [arrayfun(@(q) sprintf('SAA-L%.2f', q), q_prime_list, 'uni', 0), {'LASSO'}];
if run_cross_validation
    fprintf('Running lambda cross-validation (cv_lambda)...\n');
    lambda_best = cv_lambda();
else
    lambda_best = containers.Map(methods_reg, num2cell(zeros(1, numel(methods_reg))));
    hardcoded_lam = struct('SAA_L1_01', 0.450, 'SAA_L2_00', 0.500, 'LASSO', 0.400);
    for mi = 1:numel(methods_reg)
        name = methods_reg{mi};
        if strcmp(name, 'LASSO')
            lambda_best(name) = hardcoded_lam.LASSO;
        elseif strcmp(name, 'SAA-L1.01')
            lambda_best(name) = hardcoded_lam.SAA_L1_01;
        elseif strcmp(name, 'SAA-L2.00')
            lambda_best(name) = hardcoded_lam.SAA_L2_00;
        else
            lambda_best(name) = 0.5;   % default for any other q'
        end
        fprintf('  %-10s  lambda_0 = %.3f  (hardcoded)\n', name, lambda_best(name));
    end
end

% --------------------- theta for SMD ------------------------------
if run_cross_validation
    fprintf('Running SMD theta cross-validation (cv_theta)...\n');
    [theta_smd_l1, theta_smd_l2] = cv_theta();
else
    theta_smd_l1 = 6;     % paper Appendix E selection under bundled vk, sk
    theta_smd_l2 = 0.6;
    fprintf('  theta_SMD_L1 = %g   theta_SMD_L2 = %g  (hardcoded)\n', ...
            theta_smd_l1, theta_smd_l2);
end

% --------------------- Main sweep -----------------------------------
% methods_reg = cross-validation-selected lambda; (no fixed-lambda comparison column this run)
methods_reg_fixed = {};
methods_all = [{'SAA_r','SAA_0'}, methods_reg, methods_reg_fixed, {'SMD_L1','SMD_L2'}];
nM = numel(methods_all);

subopt = NaN(numel(N_list), numel(d_list), nM, num_reps);
l2loss = NaN(numel(N_list), numel(d_list), nM, num_reps);
runtime = NaN(numel(N_list), numel(d_list), nM, num_reps);

% --- Per-cell sweep dir (each (N,d) task writes one .mat there) ---
sweep_dir = fullfile(cache_dir, 'sweep');
if ~exist(sweep_dir, 'dir'); mkdir(sweep_dir); end

% --- If SWEEP_TASK_IDX env var set: run only one (N, d) cell, save, exit ---
% Mapping: task in 1..(numel(N_list)*numel(d_list)),  i_N = ceil(t/13), i_d = mod(t-1,13)+1
sweep_idx_env = getenv('SWEEP_TASK_IDX');
if ~isempty(sweep_idx_env)
    t = str2double(sweep_idx_env);
    i_N = ceil(t / numel(d_list));
    i_d = mod(t - 1, numel(d_list)) + 1;
    N   = N_list(i_N);
    d   = d_list(i_d);
    fprintf('\n>>> SINGLE-CELL MODE  task=%d -> i_N=%d, i_d=%d, N=%d, d=%d\n', ...
            t, i_N, i_d, N, d);

    % methods + reg lambdas
    methods_reg_fixed = {};
    methods_all = [{'SAA_r','SAA_0'}, methods_reg, methods_reg_fixed, {'SMD_L1','SMD_L2'}];
    nM = numel(methods_all);

    % test sample for this d only
    [X_te, ~]   = gen_xi(n_test, d, alpha, nu_bar, 9999);
    test_coef_d = bsxfun(@plus, (1:d)/d, X_te);
    x_ref_d     = x_ref{i_d};
    Rl1 = 2.5 * norm(x_ref_d, 1);
    Rl2 = 2.5 * norm(x_ref_d, 2);

    subopt_cell  = NaN(num_reps, nM);
    l2loss_cell  = NaN(num_reps, nM);
    runtime_cell = NaN(num_reps, nM);
    for rep = 1:num_reps
        [X_xi, ~] = gen_xi(N, d, alpha, nu_bar, 20*rep + i_d);
        coef = bsxfun(@plus, (1:d)/d, X_xi);
        x_rand = 1.0*(rand(d,1) - 0.5);
        x_zero = zeros(d, 1);

        for mi = 1:nM
            name = methods_all{mi};
            tic;
            switch name
                case 'SAA_r'
                    x = saa_solve(coef, vk, sk, 0, 2.0, M_pen, x_rand, gd_max_iter, gd_tol);
                case 'SAA_0'
                    x = saa_solve(coef, vk, sk, 0, 2.0, M_pen, x_zero, gd_max_iter, gd_tol);
                case 'LASSO'
                    x = lasso_solve(coef, vk, sk, lambda_best(name), M_pen, x_rand, gd_max_iter, gd_tol);
                case 'SMD_L1'
                    x = smd_l1_solve(coef, vk, sk, Rl1, N, d, theta_smd_l1, M_pen);
                case 'SMD_L2'
                    x = smd_l2_solve(coef, vk, sk, Rl2, N, d, theta_smd_l2, M_pen);
                otherwise
                    q   = sscanf(name, 'SAA-L%f');
                    lam = lambda_best(name);
                    x = saa_solve(coef, vk, sk, lam, q, M_pen, x_rand, gd_max_iter, gd_tol);
            end
            runtime_cell(rep, mi) = toc;
            subopt_cell(rep, mi)  = approx_gap(x, x_ref_d, test_coef_d, vk, sk, M_pen);
            l2loss_cell(rep, mi)  = norm(x - x_ref_d);
        end
        fprintf('  rep %d:', rep);
        for mi = 1:nM
            fprintf('  %s=%.3g', methods_all{mi}, subopt_cell(rep, mi));
        end
        fprintf('\n');
    end

    cell_path = fullfile(sweep_dir, sprintf('iN%d_id%d.mat', i_N, i_d));
    save(cell_path, 'subopt_cell', 'l2loss_cell', 'runtime_cell', ...
                    'methods_all', 'i_N', 'i_d', 'N', 'd', ...
                    'lambda_best', 'theta_smd_l1', 'theta_smd_l2');
    fprintf('Saved %s\n', cell_path);
    return;
end

% --- Aggregator mode: assemble per-cell files into full results + plots ---
% (runs when SWEEP_TASK_IDX is unset; assumes all 39 cells are on disk).
% test_coef built only for plotting/gap-recompute path below.
fprintf('\nGenerating test samples for gap evaluation...\n');
test_coef = cell(numel(d_list), 1);
for id = 1:numel(d_list)
    [X_te, ~] = gen_xi(n_test, d_list(id), alpha, nu_bar, 9999);
    test_coef{id} = bsxfun(@plus, (1:d_list(id))/d_list(id), X_te);
end

for iN = 1:numel(N_list)
    N = N_list(iN);
    for id = 1:numel(d_list)
        d = d_list(id);
        cell_path = fullfile(sweep_dir, sprintf('iN%d_id%d.mat', iN, id));
        if exist(cell_path, 'file')
            cd_ = load(cell_path);
            for mi = 1:nM
                for rep = 1:num_reps
                    subopt(iN, id, mi, rep)  = cd_.subopt_cell(rep, mi);
                    l2loss(iN, id, mi, rep)  = cd_.l2loss_cell(rep, mi);
                    runtime(iN, id, mi, rep) = cd_.runtime_cell(rep, mi);
                end
            end
            fprintf('Loaded cell N=%d d=%d from %s\n', N, d, cell_path);
            continue;
        end

        % Fallback: if cell file missing, run the cell inline (legacy serial path)
        fprintf('\n== N=%d  d=%d (cell missing; running inline) ==\n', N, d);

        Rl1 = 2.5 * norm(x_ref{id}, 1);
        Rl2 = 2.5 * norm(x_ref{id}, 2);

        for rep = 1:num_reps
            [X_xi, ~] = gen_xi(N, d, alpha, nu_bar, 20*rep + id);
            coef = bsxfun(@plus, (1:d)/d, X_xi);
            x_rand = 1.0*(rand(d,1) - 0.5);
            x_zero = zeros(d, 1);

            for mi = 1:nM
                name = methods_all{mi};
                tic;
                switch name
                    case 'SAA_r'
                        x = saa_solve(coef, vk, sk, 0, 2.0, M_pen, x_rand, gd_max_iter, gd_tol);
                    case 'SAA_0'
                        x = saa_solve(coef, vk, sk, 0, 2.0, M_pen, x_zero, gd_max_iter, gd_tol);
                    case 'LASSO'
                        x = lasso_solve(coef, vk, sk, lambda_best(name), M_pen, x_rand, gd_max_iter, gd_tol);
                    case 'SMD_L1'
                        x = smd_l1_solve(coef, vk, sk, Rl1, N, d, theta_smd_l1, M_pen);
                    case 'SMD_L2'
                        x = smd_l2_solve(coef, vk, sk, Rl2, N, d, theta_smd_l2, M_pen);
                    otherwise
                        m_fixed = regexp(name, '^SAA-L([\d.]+)-l([\d.]+)$', 'tokens', 'once');
                        if ~isempty(m_fixed)
                            q   = str2double(m_fixed{1});
                            lam = str2double(m_fixed{2});
                        else
                            q   = sscanf(name, 'SAA-L%f');
                            lam = lambda_best(name);
                        end
                        x = saa_solve(coef, vk, sk, lam, q, M_pen, x_rand, gd_max_iter, gd_tol);
                end
                runtime(iN, id, mi, rep) = toc;
                % Approximate suboptimality gap via test sample
                gap = approx_gap(x, x_ref{id}, test_coef{id}, vk, sk, M_pen);
                subopt(iN, id, mi, rep) = gap;
                l2loss(iN, id, mi, rep) = norm(x - x_ref{id});
            end
            fprintf('  rep %d:', rep);
            for mi = 1:nM
                fprintf('  %s=%.3g', methods_all{mi}, subopt(iN,id,mi,rep));
            end
            fprintf('\n');
        end
    end
end

save(fullfile(outdir, 'exp2_results.mat'), 'subopt', 'l2loss', 'runtime', ...
    'methods_all', 'N_list', 'd_list', 'lambda_best', ...
    'theta_smd_l1', 'theta_smd_l2', 'theta_cv_grid', ...
    'x_ref', 'N_ref', 'm_star');

% --------------------- Plots: paper Figure 2 style ------------------
markers = {'+','v','x','o','s','^','d','>','<','p','h'};
colors  = lines(nM);
pretty  = @(s) strrep(s, '_', '\_');

for iNplot = 1:numel(N_list)
    Nn = N_list(iNplot);
    subopt_mean = squeeze(mean(subopt(iNplot, :, :, :), 4));   % d x M
    fig = figure('Position', [100 100 1200 900]); tiledlayout(2, 2, 'Padding', 'compact');

    % (a) All methods
    nexttile; hold on; grid on;
    for mi = 1:nM
        plot(d_list, subopt_mean(:, mi), '-', 'Marker', markers{mod(mi-1,numel(markers))+1}, ...
            'Color', colors(mi,:), 'LineWidth', 1.2, 'DisplayName', pretty(methods_all{mi}));
    end
    xlabel('Dimensionality d'); ylabel('Approx. suboptimality gap');
    title(sprintf('(a) All methods  (N = %d)', Nn));
    legend('Location','northwest');

    % (b) Exclude SAA_r
    nexttile; hold on; grid on;
    sel = ~strcmp(methods_all, 'SAA_r');
    for mi = find(sel)
        plot(d_list, subopt_mean(:, mi), '-', 'Marker', markers{mod(mi-1,numel(markers))+1}, ...
            'Color', colors(mi,:), 'LineWidth', 1.2, 'DisplayName', pretty(methods_all{mi}));
    end
    xlabel('Dimensionality d'); ylabel('Approx. suboptimality gap');
    title('(b) All except SAA\_r'); legend('Location','northwest');

    % (c) SAA variants only
    nexttile; hold on; grid on;
    sel = startsWith(methods_all, 'SAA_') | startsWith(methods_all, 'SAA-L');
    for mi = find(sel)
        plot(d_list, subopt_mean(:, mi), '-', 'Marker', markers{mod(mi-1,numel(markers))+1}, ...
            'Color', colors(mi,:), 'LineWidth', 1.2, 'DisplayName', pretty(methods_all{mi}));
    end
    xlabel('Dimensionality d'); ylabel('Approx. suboptimality gap');
    title('(c) SAA variants only'); legend('Location','northwest');

    % (d) SAA vs SMD comparisons
    nexttile; hold on; grid on;
    sel = ismember(methods_all, {'SAA-L1.01','SAA-L2.00','SMD_L1','SMD_L2'});
    for mi = find(sel)
        plot(d_list, subopt_mean(:, mi), '-', 'Marker', markers{mod(mi-1,numel(markers))+1}, ...
            'Color', colors(mi,:), 'LineWidth', 1.2, 'DisplayName', pretty(methods_all{mi}));
    end
    xlabel('Dimensionality d'); ylabel('Approx. suboptimality gap');
    title('(d) SAA vs SMD (matched norms)'); legend('Location','northwest');

    sgtitle(sprintf('Experiment 2 (Exp. 4.2): suboptimality vs d, N = %d (mean over %d reps)', Nn, num_reps));
    saveas(fig, fullfile(outdir, sprintf('exp2_fig2_suboptimality_N%d.png', Nn)));
    savefig(fig, fullfile(outdir, sprintf('exp2_fig2_suboptimality_N%d.fig', Nn)));
end

% -------- Computational time: grouped bar chart (paper Fig. 3) ------
fig = figure('Position', [100 100 1200 750]); tiledlayout(numel(N_list), 1, 'Padding','compact');
for iNplot = 1:numel(N_list)
    Nn = N_list(iNplot);
    rt_mean = squeeze(mean(runtime(iNplot, :, :, :), 4));
    nexttile;
    b = bar(rt_mean, 'grouped');
    for mi = 1:nM
        b(mi).DisplayName = pretty(methods_all{mi});
        b(mi).FaceColor   = colors(mi, :);
    end
    grid on;
    xticks(1:numel(d_list)); xticklabels(arrayfun(@num2str, d_list, 'uni', 0));
    xlabel('Dimension'); ylabel('Time (s)');
    title(sprintf('Computational time for N = %d', Nn));
    if iNplot == 1
        legend('Location', 'northwest', 'NumColumns', ceil(nM/2));
    end
end
sgtitle('Experiment 2: mean wall-clock time (5 reps)');
saveas(fig, fullfile(outdir, 'exp2_fig3_time.png'));
savefig(fig, fullfile(outdir, 'exp2_fig3_time.fig'));

% -------- SAA vs SMD: relative differences (paper Fig. 4) -----------
% Per Eq. (94) of the paper:
%   rel_diff = 100 * ( Gap(x_SMD) - Gap(x_SAA) ) / ( (1/n_test) sum f(x*, xi_j) )
% Since both gaps subtract the same x*-term, the numerator simplifies to
% f(x_SMD) - f(x_SAA). The denominator is the Monte Carlo estimate of
% F(x*) (signed, NOT |F(x*)|).
F_xstar = zeros(numel(d_list), 1);
for id = 1:numel(d_list)
    F_xstar(id) = obj_and_grad(x_ref{id}, test_coef{id}, vk, sk, M_pen);
end

pairs = { {'SAA-L1.01','SMD_L1'}, {'SAA-L2.00','SMD_L2'} };
pair_labels = {'(a) SAA-L_{1.01} vs SMD-L_1', '(b) SAA-L_2 vs SMD-L_2'};
fig = figure('Position', [100 100 1200 500]); tiledlayout(1, 2, 'Padding','compact');

iNplot = find(N_list == 600, 1);
if isempty(iNplot); iNplot = numel(N_list); end        % fallback: largest N available
N_fig4 = N_list(iNplot);
for ignore = 1                                          % single-pass wrapper
    for pp = 1:numel(pairs)
        nexttile; hold on; grid on;
        saa_name = pairs{pp}{1}; smd_name = pairs{pp}{2};
        miSAA = find(strcmp(methods_all, saa_name), 1);
        miSMD = find(strcmp(methods_all, smd_name), 1);
        if isempty(miSAA) || isempty(miSMD); continue; end

        % Per-rep relative difference (paper Eq. 94): signed F(x*) denominator.
        % subopt(:,:,:,:) stores Gap = f(x) - f(x*); difference of gaps simplifies
        % to f(x_SMD) - f(x_SAA), i.e. the difference of our stored subopt values.
        rd = zeros(numel(d_list), num_reps);
        for id = 1:numel(d_list)
            for rep = 1:num_reps
                rd(id, rep) = 100 * (subopt(iNplot,id,miSMD,rep) - ...
                                     subopt(iNplot,id,miSAA,rep)) / F_xstar(id);
            end
        end

        rd_mean = mean(rd, 2);
        rd_min  = min(rd, [], 2);
        rd_max  = max(rd, [], 2);

        % Fill min-max band, overlay mean line and individual points
        fill([d_list, fliplr(d_list)], [rd_min', fliplr(rd_max')], ...
             [0.6 0.8 1.0], 'FaceAlpha', 0.35, 'EdgeColor','none', ...
             'DisplayName', 'Min-max gap range');
        plot(d_list, rd_mean, '-o', 'Color', [0 0.45 0.74], 'LineWidth', 1.5, ...
             'DisplayName', 'Mean relative gap');
        scatter(repmat(d_list(:), 1, num_reps), rd, 12, [0.5 0.5 0.5], 'x', ...
            'DisplayName', 'Individual relative gaps');
        yline(0, 'r--', 'DisplayName', 'Zero relative gap');
        xlabel('Dimensionality d');
        ylabel('Relative differences in suboptimality gap (%)');
        title(pair_labels{pp});
        legend('Location', 'best');
    end
end
sgtitle(sprintf('Experiment 2: SMD vs SAA relative gap, N = %d', N_fig4));
saveas(fig, fullfile(outdir, sprintf('exp2_fig4_relgap_N%d.png', N_fig4)));
savefig(fig, fullfile(outdir, sprintf('exp2_fig4_relgap_N%d.fig', N_fig4)));

% -------- Runtime relative difference (paper Eq. 95) ---------------
% rel_rt = 100 * (Runtime_SMD - Runtime_SAA) / Runtime_SAA
fig = figure('Position', [100 100 1200 500]); tiledlayout(1, 2, 'Padding','compact');
for pp = 1:numel(pairs)
    nexttile; hold on; grid on;
    saa_name = pairs{pp}{1}; smd_name = pairs{pp}{2};
    miSAA = find(strcmp(methods_all, saa_name), 1);
    miSMD = find(strcmp(methods_all, smd_name), 1);
    if isempty(miSAA) || isempty(miSMD); continue; end

    rrt = zeros(numel(d_list), num_reps);
    for id = 1:numel(d_list)
        for rep = 1:num_reps
            t_saa = runtime(iNplot, id, miSAA, rep);
            t_smd = runtime(iNplot, id, miSMD, rep);
            rrt(id, rep) = 100 * (t_smd - t_saa) / max(t_saa, eps);
        end
    end
    rrt_mean = mean(rrt, 2);
    rrt_min  = min(rrt, [], 2);
    rrt_max  = max(rrt, [], 2);

    fill([d_list, fliplr(d_list)], [rrt_min', fliplr(rrt_max')], ...
         [1 0.85 0.7], 'FaceAlpha', 0.35, 'EdgeColor','none', ...
         'DisplayName', 'Min-max range');
    plot(d_list, rrt_mean, '-o', 'Color', [0.85 0.33 0.1], 'LineWidth', 1.5, ...
         'DisplayName', 'Mean rel. runtime diff');
    scatter(repmat(d_list(:), 1, num_reps), rrt, 12, [0.5 0.5 0.5], 'x', ...
        'DisplayName', 'Individual reps');
    yline(0, 'r--', 'DisplayName', 'Zero');
    xlabel('Dimensionality d');
    ylabel('Relative difference in runtime (%)');
    title(pair_labels{pp});
    legend('Location', 'best');
end
sgtitle(sprintf('Experiment 2: SMD vs SAA runtime, N = %d', N_fig4));
saveas(fig, fullfile(outdir, sprintf('exp2_fig4b_relrt_N%d.png', N_fig4)));
savefig(fig, fullfile(outdir, sprintf('exp2_fig4b_relrt_N%d.fig', N_fig4)));

% ====================================================================
% Local functions
% ====================================================================

function [X, nu] = gen_xi(N, d, alpha, nu_bar, seed)
    % Generate N i.i.d. xi = (r_1,...,r_d) with r_i = nu_i - E[nu_i].
    % Reading B convention: effective tail index = alpha - 1 (matches the
    % paper's "bounded 2nd moment but heavy-tailed" description).
    rng(seed);
    U  = rand(N, d);
    nu = U.^(-1/(alpha - 1));     % effective shape (alpha - 1)
    X  = nu - nu_bar;
end

function [loss, grad] = obj_and_grad(x, coef, vk, sk, M_pen)
    % Sample-average loss and gradient for the utility problem.
    % coef: N-by-d, row j is (i/d + r_{i,j})_{i=1..d}.
    [~, d] = size(coef);
    y = coef * x;                             % N-by-1
    L = bsxfun(@plus, y * sk', vk');          % N-by-K,  L(j,k) = v_k + s_k y_j
    [phi_val, kstar] = max(L, [], 2);         % N-by-1
    loss_phi = mean(phi_val);
    % penalty
    pos = max(x - 1, 0);
    neg = max(-x - 1, 0);
    loss_pen = (M_pen/2) * (pos' * pos + neg' * neg);
    loss = loss_phi + loss_pen;

    if nargout > 1
        s_sel = sk(kstar);                     % N-by-1
        grad_phi = (coef' * s_sel) / size(coef,1);
        grad_pen = M_pen * (pos - neg);
        grad = grad_phi + grad_pen;
    end
end

function x = saa_solve(coef, vk, sk, lam, qp, M_pen, x0, maxit, tol)
    % SAA (2) or SAA (3) via subgradient descent with FIXED step = 1e-4.
    % Every step is accepted (no descent gate).
    % Termination: maxit, OR relative-stability over a window of 10 iters
    % (max|Δobj| / max(1, mean|obj|) < 1e-9).
    step        = 1e-4;
    stop_window = 10;
    stop_tol    = 1e-9;
    obj_history = zeros(maxit, 1);
    x = x0;
    for k = 1:maxit
        [~, gf] = obj_and_grad(x, coef, vk, sk, M_pen);
        g = gf + reg_grad(x, lam, qp);
        x = x - step * g;
        of = obj_and_grad(x, coef, vk, sk, M_pen);     % obj at new x
        obj_history(k) = of + 0.5 * lam * norm(x, qp)^2;
        if k >= stop_window
            recent = obj_history(k-stop_window+1:k);
            base = max(1.0, mean(abs(recent)));
            max_delta = max(abs(diff(recent)));
            if (max_delta / base) < stop_tol
                break;
            end
        end
    end
end

function x = saa_solve_polyak(coef, vk, sk, lam, qp, M_pen, x0, f_star, maxit)
    % Subgradient descent with Polyak's step:
    %   alpha_k = (F_lam_N(x_k) - f_star) / ||g_k||^2.
    % Caller supplies f_star (a lower bound or a tight estimate). Terminate
    % on maxit, obj <= f_star, or ||g|| ~ 0.
    step_min = 1e-20;
    x = x0;
    for k = 1:maxit
        [of, gf] = obj_and_grad(x, coef, vk, sk, M_pen);
        obj  = of + 0.5 * lam * norm(x, qp)^2;
        g    = gf + reg_grad(x, lam, qp);
        g2   = g' * g;
        if g2 < 1e-30;   break; end
        diff = obj - f_star;
        if diff <= 0;    break; end
        alpha = diff / g2;
        if alpha < step_min; break; end
        x = x - alpha * g;
    end
end

function x = lasso_solve_polyak(coef, vk, sk, lam, M_pen, x0, f_star, maxit)
    % Proximal gradient with Polyak's step on the smooth part. Caller
    % supplies f_star.
    step_min = 1e-20;
    x = x0;
    for k = 1:maxit
        [of, gf] = obj_and_grad(x, coef, vk, sk, M_pen);
        obj  = of + lam * sum(abs(x));
        g2   = gf' * gf;
        if g2 < 1e-30;   break; end
        diff = obj - f_star;
        if diff <= 0;    break; end
        alpha = diff / g2;
        if alpha < step_min; break; end
        z = x - alpha * gf;
        x = sign(z) .* max(abs(z) - alpha * lam, 0);
    end
end

function g = reg_grad(x, lam, qp)
    if lam == 0
        g = zeros(size(x));
        return;
    end
    if abs(qp - 2) < 1e-12
        g = lam * x;
    else
        nxq = norm(x, qp);
        if nxq < 1e-30
            g = zeros(size(x));
        else
            g = lam * (nxq^(2 - qp)) * sign(x) .* abs(x).^(qp - 1);
        end
    end
end

function x = lasso_solve(coef, vk, sk, lam, M_pen, x0, maxit, tol)
    % Proximal (ISTA soft-threshold) with FIXED step = 1e-4.
    % Every prox-step accepted. Termination: maxit, OR relative-stability
    % over 10-iter window (max|Δobj|/max(1,mean|obj|) < 1e-9).
    step        = 1e-4;
    stop_window = 10;
    stop_tol    = 1e-9;
    obj_history = zeros(maxit, 1);
    x = x0;
    for k = 1:maxit
        [~, gf] = obj_and_grad(x, coef, vk, sk, M_pen);
        z = x - step * gf;
        x = sign(z) .* max(abs(z) - step * lam, 0);
        of = obj_and_grad(x, coef, vk, sk, M_pen);
        obj_history(k) = of + lam * sum(abs(x));
        if k >= stop_window
            recent = obj_history(k-stop_window+1:k);
            base = max(1.0, mean(abs(recent)));
            max_delta = max(abs(diff(recent)));
            if (max_delta / base) < stop_tol
                break;
            end
        end
    end
end

function x = smd_l1_solve(coef, vk, sk, Rl1, N, d, theta, M_pen)
    % Entropic SMD on the simplex reformulation:
    %   min F(Rl1 * (y_plus - y_minus))
    %   s.t. 1'*y_plus + 1'*y_minus + s = 1, y_plus, y_minus, s >= 0.
    % Variable z = [y_plus; y_minus; s] in 2d+1 simplex.
    % Step size (Nemirovski et al. 2009): theta * sqrt(2*ln(K)) / (Minf * sqrt(N)).
    % theta is selected by the cross-validation procedure in Appendix E of Liu & Tong 2024.
    K = 2*d + 1;
    z = ones(K, 1) / K;                        % uniform init on simplex
    Minf = estimate_Minf(coef, sk, Rl1, N, M_pen);
    gamma = theta * sqrt(2*log(K)) / (max(Minf,1e-6) * sqrt(N));

    sum_z = zeros(K, 1);
    for t = 1:N
        y_plus  = z(1:d);
        y_minus = z(d+1:2*d);
        x = Rl1 * (y_plus - y_minus);
        % stochastic subgradient w.r.t. x from sample t
        row = coef(t, :);
        y_val = row * x;
        [~, kstar] = max(vk + sk * y_val);
        g_x = sk(kstar) * row';                % d-vector
        % subgradient w.r.t. z  (chain rule through x = Rl1 (y+ - y-))
        g_z = [ Rl1 * g_x;  -Rl1 * g_x;  0 ];
        % mirror descent (entropic) on simplex
        w = z .* exp(-gamma * g_z);
        z = w / sum(w);
        sum_z = sum_z + z;
    end
    z_bar = sum_z / N;
    y_plus  = z_bar(1:d);
    y_minus = z_bar(d+1:2*d);
    x = Rl1 * (y_plus - y_minus);
end

function x = smd_l2_solve(coef, vk, sk, Rl2, N, d, theta, M_pen)
    % Robust stochastic approximation (2-norm).
    % Step size (Nemirovski et al. 2009): theta * Rl2 / (M2 * sqrt(N)).
    % theta is selected by the cross-validation procedure in Appendix E of Liu & Tong 2024.
    x = zeros(d, 1);
    M2 = estimate_M2(coef, sk, Rl2, N, M_pen);
    gamma = theta * Rl2 / (max(M2, 1e-6) * sqrt(N));

    sum_x = zeros(d, 1);
    for t = 1:N
        row = coef(t, :);
        y_val = row * x;
        [~, kstar] = max(vk + sk * y_val);
        g = sk(kstar) * row';
        x = x - gamma * g;
        % Euclidean projection onto {||x||_2 <= Rl2}
        nx = norm(x);
        if nx > Rl2; x = x * (Rl2 / nx); end
        sum_x = sum_x + x;
    end
    x = sum_x / N;
end

function Minf = estimate_Minf(coef, sk, Rl1, N, M_pen)
    % Paper Appendix E:  E[ sup_{||x||_1<=Rl1}  ||G(x,xi)||_inf ]  via MC.
    % For  f(x,xi) = phi(coef*x) + (M_pen/2) * sum box_penalty(x_i),
    %   G(x,xi) = s_{k*}(x,xi) * coef_row + M_pen * phi'(x_i) at each i,
    % where sup is achieved by a single-index spike  x = +/- Rl1 * e_{i*}
    % with i* = argmax |coef_row_i|  and k* chosen to take max_k |s_k|.
    nS = min(500, N);
    smax    = max(abs(sk));
    maxcoef = max(abs(coef(1:nS, :)), [], 2);      % per-sample ||a||_inf
    Minf    = smax * mean(maxcoef) + M_pen * max(Rl1 - 1, 0);
end

function M2 = estimate_M2(coef, sk, Rl2, N, M_pen)
    % Paper Appendix E:  E[ sup_{||x||_2<=Rl2} ||G(x,xi)||_2 ] via MC.
    % Upper bound by Minkowski:  |s_{k*}| ||a||_2  + M_pen (Rl2-1)_+.
    nS = min(500, N);
    smax  = max(abs(sk));
    norm2 = sqrt(sum(coef(1:nS, :).^2, 2));
    M2    = smax * mean(norm2) + M_pen * max(Rl2 - 1, 0);
end

function gap = approx_gap(x, xstar, test_coef, vk, sk, M_pen)
    % Monte Carlo approximate suboptimality gap.
    fx     = obj_and_grad(x,     test_coef, vk, sk, M_pen);
    fxstar = obj_and_grad(xstar, test_coef, vk, sk, M_pen);
    gap = fx - fxstar;
end
