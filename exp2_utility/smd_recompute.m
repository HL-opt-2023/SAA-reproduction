% Recompute the SMD-L1 / SMD-L2 columns of every existing per-cell sweep
% file, using the CV-selected theta values under user's vk, sk.
% SAA / LASSO entries are preserved; only SMD columns change.

clear; close all; rng(0);

% ---------- USER's phi parameters ----------
pp = load(fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'exp2', ...
                   'piecewise_parameter.mat'));
vk = pp.v(:);
sk = pp.s(:);

% ---------- Reading B Pareto ----------
alpha  = 3.01;
nu_bar = (alpha - 1) / (alpha - 2);

% ---------- problem grid ----------
M_pen    = 1000;
n_test   = 1e4;
N_list   = [200 400 600];
d_list   = [100 200 300 400 500 600 700 800 900 1000 1500 2000 5000];
num_reps = 5;

% ---------- CV-selected theta under user's vk, sk ----------
theta_smd_l1 = 6;
theta_smd_l2 = 0.6;
fprintf('Updating SMD columns with theta_L1=%g, theta_L2=%g\n', ...
        theta_smd_l1, theta_smd_l2);

% ---------- Load x_ref{d} from cache ----------
cache_dir = fullfile(fileparts(mfilename('fullpath')), 'cache');
x_ref = cell(numel(d_list), 1);
for id = 1:numel(d_list)
    p = fullfile(cache_dir, sprintf('x_ref_d%d.mat', d_list(id)));
    if ~exist(p, 'file')
        warning('Missing %s; skipping', p);
        continue;
    end
    cd_ = load(p);
    x_ref{id} = cd_.x_ref_one;
end

sweep_dir = fullfile(cache_dir, 'sweep');
n_upd = 0; n_skip = 0;
for i_N = 1:numel(N_list)
    N = N_list(i_N);
    for i_d = 1:numel(d_list)
        d = d_list(i_d);
        cell_path = fullfile(sweep_dir, sprintf('iN%d_id%d.mat', i_N, i_d));
        if ~exist(cell_path, 'file') || isempty(x_ref{i_d})
            fprintf('  SKIP  iN=%d id=%d  (cell or x_ref missing)\n', i_N, i_d);
            n_skip = n_skip + 1;
            continue;
        end
        S = load(cell_path);
        methods_all = S.methods_all;
        miL1 = find(strcmp(methods_all, 'SMD_L1'), 1);
        miL2 = find(strcmp(methods_all, 'SMD_L2'), 1);
        if isempty(miL1) || isempty(miL2)
            fprintf('  SKIP  iN=%d id=%d  (no SMD columns)\n', i_N, i_d);
            n_skip = n_skip + 1;
            continue;
        end

        % regenerate test sample for this d (matches main sweep seed 9999)
        [X_te, ~] = gen_xi(n_test, d, alpha, nu_bar, 9999);
        test_coef_d = bsxfun(@plus, (1:d)/d, X_te);
        x_ref_d = x_ref{i_d};
        Rl1 = 2.5 * norm(x_ref_d, 1);
        Rl2 = 2.5 * norm(x_ref_d, 2);

        for rep = 1:num_reps
            [X_xi, ~] = gen_xi(N, d, alpha, nu_bar, 20*rep + i_d);
            coef = bsxfun(@plus, (1:d)/d, X_xi);

            tic;
            x = smd_l1_solve(coef, vk, sk, Rl1, N, d, theta_smd_l1, M_pen);
            S.runtime_cell(rep, miL1) = toc;
            S.subopt_cell(rep, miL1)  = approx_gap(x, x_ref_d, test_coef_d, vk, sk, M_pen);
            S.l2loss_cell(rep, miL1)  = norm(x - x_ref_d);

            tic;
            x = smd_l2_solve(coef, vk, sk, Rl2, N, d, theta_smd_l2, M_pen);
            S.runtime_cell(rep, miL2) = toc;
            S.subopt_cell(rep, miL2)  = approx_gap(x, x_ref_d, test_coef_d, vk, sk, M_pen);
            S.l2loss_cell(rep, miL2)  = norm(x - x_ref_d);
        end
        S.theta_smd_l1 = theta_smd_l1;
        S.theta_smd_l2 = theta_smd_l2;
        save(cell_path, '-struct', 'S');
        fprintf('  UPD   iN=%d id=%d (N=%d, d=%d)  meanL1=%.4f  meanL2=%.4f\n', ...
                i_N, i_d, N, d, mean(S.subopt_cell(:, miL1)), mean(S.subopt_cell(:, miL2)));
        n_upd = n_upd + 1;
    end
end
fprintf('\nDone: %d cells updated, %d skipped.\n', n_upd, n_skip);

% ====================================================================
% Local helpers
% ====================================================================

function [X, nu] = gen_xi(N, d, alpha, nu_bar, seed)
    rng(seed);
    U  = rand(N, d);
    nu = U.^(-1/(alpha - 1));
    X  = nu - nu_bar;
end

function [loss, grad] = obj_and_grad(x, coef, vk, sk, M_pen)
    [~, d] = size(coef);
    y = coef * x;
    L = bsxfun(@plus, y * sk', vk');
    [phi_val, kstar] = max(L, [], 2);
    loss_phi = mean(phi_val);
    pos = max(x - 1, 0);
    neg = max(-x - 1, 0);
    loss_pen = (M_pen/2) * (pos' * pos + neg' * neg);
    loss = loss_phi + loss_pen;
    if nargout > 1
        s_sel    = sk(kstar);
        grad_phi = (coef' * s_sel) / size(coef,1);
        grad_pen = M_pen * (pos - neg);
        grad     = grad_phi + grad_pen;
    end
end

function gap = approx_gap(x, xstar, test_coef, vk, sk, M_pen)
    fx     = obj_and_grad(x,     test_coef, vk, sk, M_pen);
    fxstar = obj_and_grad(xstar, test_coef, vk, sk, M_pen);
    gap = fx - fxstar;
end

function Minf = estimate_Minf(coef, sk, Rl1, N, M_pen)
    nS = min(500, N);
    smax    = max(abs(sk));
    maxcoef = max(abs(coef(1:nS, :)), [], 2);
    Minf    = smax * mean(maxcoef) + M_pen * max(Rl1 - 1, 0);
end

function M2 = estimate_M2(coef, sk, Rl2, N, M_pen)
    nS = min(500, N);
    smax  = max(abs(sk));
    norm2 = sqrt(sum(coef(1:nS, :).^2, 2));
    M2    = smax * mean(norm2) + M_pen * max(Rl2 - 1, 0);
end

function x = smd_l1_solve(coef, vk, sk, Rl1, N, d, theta, M_pen)
    K = 2*d + 1;
    z = ones(K, 1) / K;
    Minf = estimate_Minf(coef, sk, Rl1, N, M_pen);
    gamma = theta * sqrt(2*log(K)) / (max(Minf,1e-6) * sqrt(N));
    sum_z = zeros(K, 1);
    for t = 1:N
        y_plus  = z(1:d);
        y_minus = z(d+1:2*d);
        x = Rl1 * (y_plus - y_minus);
        row = coef(t, :);
        y_val = row * x;
        [~, kstar] = max(vk + sk * y_val);
        g_x = sk(kstar) * row';
        g_z = [ Rl1 * g_x;  -Rl1 * g_x;  0 ];
        w = z .* exp(-gamma * g_z);
        z = w / sum(w);
        sum_z = sum_z + z;
    end
    z_bar = sum_z / N;
    x = Rl1 * (z_bar(1:d) - z_bar(d+1:2*d));
end

function x = smd_l2_solve(coef, vk, sk, Rl2, N, d, theta, M_pen)
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
        nx = norm(x);
        if nx > Rl2; x = x * (Rl2 / nx); end
        sum_x = sum_x + x;
    end
    x = sum_x / N;
end
