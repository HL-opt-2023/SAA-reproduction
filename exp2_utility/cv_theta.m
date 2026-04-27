function [theta_smd_l1, theta_smd_l2] = cv_theta()
% CV_THETA  Cross-validate the SMD step-size scalar theta
% (Appendix E of Liu & Tong 2026):
%   - fix d = 1000, N = 600
%   - candidate grid: {a*b : a = 1..9, b in {0.1, 1, 10, 100, 1000}}
%   - for each theta and each rep = 1..5, draw two i.i.d. samples
%     (optimization + validation), train SMD on the optimization set,
%     score the unregularized objective on the validation set.
%   - select theta minimizing mean validation cost.
% Independently for SMD-L1 (entropic) and SMD-L2 (robust SA).
%
% Returns the best (theta_smd_l1, theta_smd_l2).
% Also writes ../data/exp2/theta_smd.mat for bookkeeping.
%
% Requires that the high-fidelity reference x*(d=1000) is already
% cached at  cache/x_ref_d1000.mat  (e.g. via precompute_xref slurm).

rng(0);

% phi parameters
pp_path = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'exp2', ...
                   'piecewise_parameter.mat');
pp = load(pp_path);
vk = pp.v(:);  sk = pp.s(:);

% Pareto noise
alpha  = 3.01;
nu_bar = (alpha - 1) / (alpha - 2);

% setup
d_smd_cv = 1000;
N_smd_cv = 600;
M_pen    = 1000;
theta_grid = reshape((1:9)' * [0.1 1 10 100 1000], [], 1);

% load reference x* (= argmin F_N at N_ref=5000)
xref_path = fullfile(fileparts(mfilename('fullpath')), 'cache', ...
                     sprintf('x_ref_d%d.mat', d_smd_cv));
if ~exist(xref_path, 'file')
    error(['x_ref cache not found at %s.\nRun precompute_xref first ' ...
           '(submit slurm/precompute_xref.slurm or run the bootstrap ' ...
           'block of run_exp2.m).'], xref_path);
end
cd_ = load(xref_path);
x_star = cd_.x_ref_one;
Rl1 = 2.5 * norm(x_star, 1);
Rl2 = 2.5 * norm(x_star, 2);
fprintf('Reference x*(d=%d):  ||x||_1=%.4f, ||x||_2=%.4f\n', d_smd_cv, ...
        norm(x_star, 1), norm(x_star, 2));
fprintf('R_l1 = %.4f,  R_l2 = %.4f\n', Rl1, Rl2);

best_l1 = NaN;  best_l1_v = Inf;
best_l2 = NaN;  best_l2_v = Inf;
val_l1_curve = nan(numel(theta_grid), 1);
val_l2_curve = nan(numel(theta_grid), 1);

for ii = 1:numel(theta_grid)
    th = theta_grid(ii);
    val_l1 = zeros(5, 1);
    val_l2 = zeros(5, 1);
    for rep = 1:5
        [X_opt, ~] = gen_xi(N_smd_cv, d_smd_cv, alpha, nu_bar, 8000 + rep);
        [X_val, ~] = gen_xi(N_smd_cv, d_smd_cv, alpha, nu_bar, 9000 + rep);
        coef_opt = bsxfun(@plus, (1:d_smd_cv)/d_smd_cv, X_opt);
        coef_val = bsxfun(@plus, (1:d_smd_cv)/d_smd_cv, X_val);
        x_l1 = smd_l1_solve(coef_opt, vk, sk, Rl1, N_smd_cv, d_smd_cv, th, M_pen);
        x_l2 = smd_l2_solve(coef_opt, vk, sk, Rl2, N_smd_cv, d_smd_cv, th, M_pen);
        val_l1(rep) = obj_and_grad(x_l1, coef_val, vk, sk, M_pen);
        val_l2(rep) = obj_and_grad(x_l2, coef_val, vk, sk, M_pen);
    end
    val_l1_curve(ii) = mean(val_l1);
    val_l2_curve(ii) = mean(val_l2);
    fprintf('  theta=%-8g  L1 val=%.5f  L2 val=%.5f\n', th, ...
            val_l1_curve(ii), val_l2_curve(ii));
    if val_l1_curve(ii) < best_l1_v
        best_l1_v = val_l1_curve(ii);  best_l1 = th;
    end
    if val_l2_curve(ii) < best_l2_v
        best_l2_v = val_l2_curve(ii);  best_l2 = th;
    end
end

theta_smd_l1 = best_l1;
theta_smd_l2 = best_l2;
fprintf('\n=== best theta ===\n');
fprintf('  theta_SMD_L1 = %g  (val = %.5f)\n', theta_smd_l1, best_l1_v);
fprintf('  theta_SMD_L2 = %g  (val = %.5f)\n', theta_smd_l2, best_l2_v);

out_path = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'exp2', ...
                    'theta_smd.mat');
save(out_path, 'theta_smd_l1', 'theta_smd_l2', 'theta_grid', ...
               'val_l1_curve', 'val_l2_curve', 'd_smd_cv', 'N_smd_cv');
fprintf('\nSaved %s\n', out_path);

% ====================================================================
% Local helpers (mirror run_exp2.m).
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
    pos = max(x - 1, 0);  neg = max(-x - 1, 0);
    loss = loss_phi + (M_pen/2)*(pos'*pos + neg'*neg);
    if nargout > 1
        grad = (coef' * sk(kstar)) / size(coef, 1) + M_pen*(pos - neg);
    end
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
    gamma = theta * sqrt(2*log(K)) / (max(Minf, 1e-6) * sqrt(N));
    sum_z = zeros(K, 1);
    for t = 1:N
        y_plus  = z(1:d);
        y_minus = z(d+1:2*d);
        x = Rl1 * (y_plus - y_minus);
        row = coef(t, :);
        y_val = row * x;
        [~, kstar] = max(vk + sk * y_val);
        g_x = sk(kstar) * row';
        g_z = [ Rl1 * g_x; -Rl1 * g_x; 0 ];
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
