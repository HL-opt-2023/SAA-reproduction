% Experiment 1 (Section 4.1 of Liu & Tong, 2024):
% Light-tailed SP — stochastic linear regression
%   min_{x in R^d}  E_{a,b}[(a'x - b)^2]
% where a ~ N(0, Sigma), Sigma_{ij} = 0.5^|i-j|, b = a'x* + w, w ~ N(0,1).
%
% Methods compared:
%   SAA_r     : SAA (2), random init
%   SAA_0     : SAA (2), zero init
%   SAA-L_q'  : SAA (3) with Tikhonov-like penalty for q' in {1.01, 1.5, 2}
%   LASSO     : L1-regularized
%
% Metrics: suboptimality gap = (x-x*)' Sigma (x-x*); L2-loss = ||x-x*||_2

clear; close all; rng(0);

% --------------------- Configuration (matches paper) ---------------
d_list       = [100:100:900, 1000, 1500, 2000, 5000];
N_list       = [200, 400, 600];
num_reps     = 5;
q_prime_list = [1.01, 1.5, 2.0];
lambda_grid  = [0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5];
gd_max_iter  = 500000;
gd_tol       = 1e-8;       % unused (relative-stability window-based termination)

% Toggle: when true, run lambda cross-validation (cv_lambda); when
% false, use the hard-coded picks (the value selected by every prior
% cross-validation under the bundled coefficients).
run_cross_validation = false;

outdir       = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(outdir, 'dir'); mkdir(outdir); end

% --------------------- lambda_0 ------------------------------------
methods_reg = [arrayfun(@(q) sprintf('SAA-L%.2f', q), q_prime_list, 'uni', 0), {'LASSO'}];
if run_cross_validation
    fprintf('Running lambda cross-validation (cv_lambda)...\n');
    lambda_best = cv_lambda();
else
    % All prior cross-validation runs picked the grid endpoint 0.5.
    lambda_best = containers.Map(methods_reg, num2cell(0.5*ones(1, numel(methods_reg))));
    for mi = 1:numel(methods_reg)
        fprintf('  %-10s  lambda_0 = %.3f  (hardcoded)\n', ...
                methods_reg{mi}, lambda_best(methods_reg{mi}));
    end
end

% --------------------- Main sweep -----------------------------------
methods_all = [{'SAA_r','SAA_0'}, methods_reg];
nM = numel(methods_all);

% --- Per-cell sweep dir (parallel array writes one .mat per (N, d) cell) ---
sweep_dir = fullfile(outdir, '..', 'cache', 'sweep');
sweep_dir = char(java.io.File(sweep_dir).getCanonicalPath());
if ~exist(sweep_dir, 'dir'); mkdir(sweep_dir); end

% --- Single-cell mode (env var SWEEP_TASK_IDX = 1..(numel(N)*numel(d))) ---
sweep_idx_env = getenv('SWEEP_TASK_IDX');
if ~isempty(sweep_idx_env)
    t = str2double(sweep_idx_env);
    i_N = ceil(t / numel(d_list));
    i_d = mod(t - 1, numel(d_list)) + 1;
    N   = N_list(i_N);
    d   = d_list(i_d);
    fprintf('\n>>> SINGLE-CELL MODE  task=%d -> i_N=%d, i_d=%d, N=%d, d=%d\n', ...
            t, i_N, i_d, N, d);

    subopt_cell  = NaN(num_reps, nM);
    l2loss_cell  = NaN(num_reps, nM);
    runtime_cell = NaN(num_reps, nM);
    for rep = 1:num_reps
        [A, b, xstar, Sigma] = gen_problem(d, N, rep + 1000);
        x_rand = 1.0 * (rand(d,1) - 0.5);
        x_zero = zeros(d,1);
        for mi = 1:nM
            name = methods_all{mi};
            tic;
            switch name
                case 'SAA_r'
                    x = solve_saa_reg(A, b, 0, 2.0, x_rand, gd_max_iter, gd_tol);
                case 'SAA_0'
                    x = solve_saa_reg(A, b, 0, 2.0, x_zero, gd_max_iter, gd_tol);
                case 'LASSO'
                    x = solve_lasso(A, b, lambda_best(name), x_rand, gd_max_iter, gd_tol);
                otherwise
                    q = sscanf(name, 'SAA-L%f');
                    x = solve_saa_reg(A, b, lambda_best(name), q, x_rand, gd_max_iter, gd_tol);
            end
            runtime_cell(rep, mi) = toc;
            e = x - xstar;
            subopt_cell(rep, mi)  = e' * Sigma * e;
            l2loss_cell(rep, mi)  = norm(e);
        end
        fprintf('  rep %d:', rep);
        for mi = 1:nM
            fprintf('  %s=%.3f', methods_all{mi}, subopt_cell(rep, mi));
        end
        fprintf('\n');
    end

    cell_path = fullfile(sweep_dir, sprintf('iN%d_id%d.mat', i_N, i_d));
    save(cell_path, 'subopt_cell', 'l2loss_cell', 'runtime_cell', ...
                    'methods_all', 'i_N', 'i_d', 'N', 'd', 'lambda_best');
    fprintf('Saved %s\n', cell_path);
    return;
end

% --- Aggregator mode: assemble per-cell files into full results ---
subopt = NaN(numel(N_list), numel(d_list), nM, num_reps);
l2loss = NaN(numel(N_list), numel(d_list), nM, num_reps);
runtime = NaN(numel(N_list), numel(d_list), nM, num_reps);

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
        % Fallback: run inline if cell missing (legacy serial path).
        fprintf('\n== N=%d  d=%d (cell missing; running inline) ==\n', N, d);
        for rep = 1:num_reps
            [A, b, xstar, Sigma] = gen_problem(d, N, rep + 1000);
            x_rand = 1.0 * (rand(d,1) - 0.5);
            x_zero = zeros(d,1);
            for mi = 1:nM
                name = methods_all{mi};
                tic;
                switch name
                    case 'SAA_r'
                        x = solve_saa_reg(A, b, 0, 2.0, x_rand, gd_max_iter, gd_tol);
                    case 'SAA_0'
                        x = solve_saa_reg(A, b, 0, 2.0, x_zero, gd_max_iter, gd_tol);
                    case 'LASSO'
                        x = solve_lasso(A, b, lambda_best(name), x_rand, gd_max_iter, gd_tol);
                    otherwise
                        q = sscanf(name, 'SAA-L%f');
                        x = solve_saa_reg(A, b, lambda_best(name), q, x_rand, gd_max_iter, gd_tol);
                end
                runtime(iN, id, mi, rep) = toc;
                e = x - xstar;
                subopt(iN, id, mi, rep) = e' * Sigma * e;
                l2loss(iN, id, mi, rep) = norm(e);
            end
        end
    end
end

save(fullfile(outdir, 'exp1_results.mat'), 'subopt', 'l2loss', 'runtime', ...
    'methods_all', 'N_list', 'd_list', 'lambda_best');

% --------------------- Plots: paper Figure 2 style ------------------
% Four-panel suboptimality plot at N=200: (a) all methods; (b)(c)(d) zoomed.
markers   = {'+','v','x','o','s','^','d','>','<'};
colors    = lines(nM);
pretty    = @(s) strrep(s, '_', '\_');

iN = find(N_list == 200, 1);
if ~isempty(iN)
    subopt_mean = squeeze(mean(subopt(iN, :, :, :), 4));   % d x M
    fig = figure('Position', [100 100 1200 900]); tiledlayout(2, 2, 'Padding', 'compact');

    % Subplot (a): all methods
    nexttile; hold on; grid on;
    for mi = 1:nM
        plot(d_list, subopt_mean(:, mi), '-', 'Marker', markers{mod(mi-1,numel(markers))+1}, ...
            'Color', colors(mi,:), 'LineWidth', 1.2, 'DisplayName', pretty(methods_all{mi}));
    end
    xlabel('Dimensionality d'); ylabel('Suboptimality');
    title(sprintf('(a) All methods  (N = 200)')); legend('Location','northwest');

    % Subplot (b): exclude SAA_r (so the others are visible)
    nexttile; hold on; grid on;
    sel_b = ~strcmp(methods_all, 'SAA_r');
    for mi = find(sel_b)
        plot(d_list, subopt_mean(:, mi), '-', 'Marker', markers{mod(mi-1,numel(markers))+1}, ...
            'Color', colors(mi,:), 'LineWidth', 1.2, 'DisplayName', pretty(methods_all{mi}));
    end
    xlabel('Dimensionality d'); ylabel('Suboptimality');
    title('(b) All except SAA\_r'); legend('Location','northwest');

    % Subplot (c): SAA variants only (no LASSO)
    nexttile; hold on; grid on;
    sel_c = ~ismember(methods_all, {'SAA_r','LASSO'});
    for mi = find(sel_c)
        plot(d_list, subopt_mean(:, mi), '-', 'Marker', markers{mod(mi-1,numel(markers))+1}, ...
            'Color', colors(mi,:), 'LineWidth', 1.2, 'DisplayName', pretty(methods_all{mi}));
    end
    xlabel('Dimensionality d'); ylabel('Suboptimality');
    title('(c) SAA variants only'); legend('Location','northwest');

    % Subplot (d): regularized SAA + LASSO (finest view)
    nexttile; hold on; grid on;
    sel_d = startsWith(methods_all, 'SAA-L') | strcmp(methods_all, 'LASSO');
    for mi = find(sel_d)
        plot(d_list, subopt_mean(:, mi), '-', 'Marker', markers{mod(mi-1,numel(markers))+1}, ...
            'Color', colors(mi,:), 'LineWidth', 1.2, 'DisplayName', pretty(methods_all{mi}));
    end
    xlabel('Dimensionality d'); ylabel('Suboptimality');
    title('(d) Regularized SAA & LASSO'); legend('Location','northwest');

    sgtitle(sprintf('Experiment 1 (Exp. 4.1): suboptimality vs d, N = 200 (mean over %d reps)', num_reps));
    saveas(fig, fullfile(outdir, 'exp1_fig2_suboptimality_N200.png'));
    savefig(fig, fullfile(outdir, 'exp1_fig2_suboptimality_N200.fig'));
end

% Also produce the N=400 and N=600 panels
for iNplot = 1:numel(N_list)
    Nn = N_list(iNplot);
    subopt_mean = squeeze(mean(subopt(iNplot, :, :, :), 4));
    fig = figure('Position', [100 100 700 500]); hold on; grid on;
    for mi = 1:nM
        plot(d_list, subopt_mean(:, mi), '-', 'Marker', markers{mod(mi-1,numel(markers))+1}, ...
            'Color', colors(mi,:), 'LineWidth', 1.2, 'DisplayName', pretty(methods_all{mi}));
    end
    xlabel('Dimensionality d'); ylabel('Suboptimality');
    title(sprintf('Experiment 1: suboptimality vs d, N = %d', Nn));
    legend('Location','northwest');
    saveas(fig, fullfile(outdir, sprintf('exp1_subopt_N%d.png', Nn)));
end

% -------- Computational time: grouped bar chart (paper Fig. 3) ------
fig = figure('Position', [100 100 1200 750]); tiledlayout(numel(N_list), 1, 'Padding','compact');
for iNplot = 1:numel(N_list)
    Nn = N_list(iNplot);
    rt_mean = squeeze(mean(runtime(iNplot, :, :, :), 4));   % d x M
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
sgtitle('Experiment 1: mean wall-clock time (5 reps)');
saveas(fig, fullfile(outdir, 'exp1_fig3_time.png'));
savefig(fig, fullfile(outdir, 'exp1_fig3_time.fig'));

% ====================================================================
% Local functions
% ====================================================================

function [A, b, xstar, Sigma] = gen_problem(d, N, seed)
    % Generate linear regression data per Section 4.1.
    rng(seed);
    % Sigma_{ij} = 0.5^|i-j|
    [I, J] = meshgrid(1:d, 1:d);
    Sigma  = 0.5.^abs(I - J);
    % Square-root via Cholesky (Sigma is Toeplitz PD)
    L = chol(Sigma + 1e-12*eye(d), 'lower');
    A = (L * randn(d, N))';      % N-by-d, rows are a_j ~ N(0, Sigma)
    % x* — kx*k_{1.8} small but x* not sparse
    r  = min(d, 200);
    S  = randperm(d, r);
    Sc = setdiff(1:d, S);
    u  = randn(r, 1);            % indices in S
    v  = randn(d - r, 1);        % indices in Sc
    xstar        = zeros(d, 1);
    xstar(S)     = 1.5 * u / norm(u, 1.8);
    xstar(Sc)    = 1.5 * v / norm(v, 1.8);
    w = randn(N, 1);
    b = A * xstar + w;
end

function x = solve_saa_reg(A, b, lam, qp, x0, maxit, tol)
    % Solve  min  (1/N) ||Ax - b||_2^2  + (lam/2) ||x||_{qp}^2
    % via gradient descent with FIXED step = 1e-4. Every step is accepted
    % (no descent gate). Termination: maxit, OR relative-stability over a
    % window of 10 iters (max|Delta obj| / max(1, mean|obj|) < 1e-9).
    N = size(A, 1);
    step        = 1e-4;
    stop_window = 10;
    stop_tol    = 1e-9;
    obj_history = zeros(maxit, 1);
    x = x0;
    for k = 1:maxit
        r = A*x - b;
        g = (2/N) * (A' * r) + reg_grad(x, lam, qp);
        x = x - step * g;
        r_new = A*x - b;
        obj_history(k) = mean(r_new.^2) + 0.5*lam*norm(x, qp)^2;
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

function g = reg_grad(x, lam, qp)
    % Gradient of (lam/2) * ||x||_{q'}^2
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

function x = solve_lasso(A, b, lam, x0, maxit, tol)
    % min (1/N) ||Ax-b||_2^2 + lam * ||x||_1   via proximal gradient
    % with FIXED step = 1e-4 (matches saa_solve scheme). Every prox-step
    % accepted. Termination: maxit OR relative-stability over 10-iter window.
    N = size(A, 1);
    step        = 1e-4;
    stop_window = 10;
    stop_tol    = 1e-9;
    obj_history = zeros(maxit, 1);
    x = x0;
    for k = 1:maxit
        g_smooth = (2/N) * (A' * (A*x - b));
        z = x - step * g_smooth;
        x = sign(z) .* max(abs(z) - step * lam, 0);
        obj_history(k) = mean((A*x - b).^2) + lam * sum(abs(x));
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

function x = solve_saa_reg_polyak(A, b, lam, qp, x0, f_star, maxit)
    % Gradient descent with Polyak's step:
    %   alpha_k = (F_lam_N(x_k) - f_star) / ||g_k||^2.
    % Caller supplies f_star.
    N = size(A, 1);
    step_min = 1e-20;
    x = x0;
    for k = 1:maxit
        r   = A*x - b;
        obj = mean(r.^2) + 0.5 * lam * norm(x, qp)^2;
        g   = (2/N) * (A' * r) + reg_grad(x, lam, qp);
        g2  = g' * g;
        if g2 < 1e-30;   break; end
        diff = obj - f_star;
        if diff <= 0;    break; end
        alpha = diff / g2;
        if alpha < step_min; break; end
        x = x - alpha * g;
    end
end
