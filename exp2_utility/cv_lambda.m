% cv_lambda.m  --  Cross-validate the regularization weight lambda_0 for
% Experiment 2 (heavy-tailed utility problem).  Follows the procedure in
% Section 4.2 of Liu & Tong 2026:
%   - fix d = 1000 and N = 200
%   - for each candidate lambda in {0.01, 0.05, 0.10, ..., 0.50}
%     and each replication rep = 1..5, draw two independent samples of
%     xi (one for training, one for validation), solve the regularized
%     SAA on training, then score the unregularized objective on
%     validation
%   - select lambda minimizing the mean validation cost
% Independently for SAA-L1.01, SAA-L2.00, and LASSO.
%
% Output: writes ../data/exp2/lambda_best.mat with one field per method.

clear; close all; rng(0);

% --- piecewise affine phi parameters ---
pp_path = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'exp2', ...
                   'piecewise_parameter.mat');
if ~exist(pp_path, 'file')
    error('piecewise_parameter.mat not found at %s.', pp_path);
end
pp = load(pp_path);
vk = pp.v(:);  sk = pp.s(:);

% --- Pareto noise (Reading B convention: shape parameter = alpha-1) ---
alpha  = 3.01;
nu_bar = (alpha - 1) / (alpha - 2);

% --- problem / cv setup ---
d_cv        = 1000;
N_cv        = 200;
M_pen       = 1000;
gd_max_iter = 500000;
q_prime_list = [1.01, 2.00];
methods_reg  = [arrayfun(@(q) sprintf('SAA-L%.2f', q), q_prime_list, 'uni', 0), {'LASSO'}];
lambda_grid  = [0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50];

lambda_best = containers.Map('KeyType','char','ValueType','double');
for mi = 1:numel(methods_reg)
    name = methods_reg{mi};
    fprintf('\n=== %s ===\n', name);
    best_val = Inf;
    best_lam = lambda_grid(1);
    for lam = lambda_grid
        vals = zeros(5, 1);
        for rep = 1:5
            [X_tr, ~] = gen_xi(N_cv, d_cv, alpha, nu_bar, 1000*rep + 123);
            [X_va, ~] = gen_xi(N_cv, d_cv, alpha, nu_bar, 1000*rep + 456);
            coef_tr = bsxfun(@plus, (1:d_cv)/d_cv, X_tr);
            coef_va = bsxfun(@plus, (1:d_cv)/d_cv, X_va);
            x0 = 1.0 * (rand(d_cv, 1) - 0.5);
            if strcmp(name, 'LASSO')
                x = lasso_solve(coef_tr, vk, sk, lam, M_pen, x0, gd_max_iter);
            else
                q = sscanf(name, 'SAA-L%f');
                x = saa_solve(coef_tr, vk, sk, lam, q, M_pen, x0, gd_max_iter);
            end
            y_va = coef_va * x;
            vals(rep) = mean(max(bsxfun(@plus, y_va*sk', vk'), [], 2)) ...
                      + (M_pen/2)*sum(max(x-1,0).^2 + max(-x-1,0).^2);
        end
        m = mean(vals);
        fprintf('  lam=%.3f  mean_val=%.5f\n', lam, m);
        if m < best_val; best_val = m; best_lam = lam; end
    end
    fprintf('  --> best lambda for %s = %.3f\n', name, best_lam);
    lambda_best(name) = best_lam;
end

out_path = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'exp2', ...
                    'lambda_best.mat');
save(out_path, 'lambda_best', 'lambda_grid', 'd_cv', 'N_cv');
fprintf('\nSaved %s\n', out_path);

% ====================================================================
% Local helpers (kept here so cv_lambda.m is self-contained).
% Mirror the implementations in run_exp2.m verbatim.
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

function g = reg_grad(x, lam, qp)
    if lam == 0;            g = zeros(size(x));   return; end
    if abs(qp - 2) < 1e-12; g = lam * x;          return; end
    nxq = norm(x, qp);
    if nxq < 1e-30;         g = zeros(size(x));   return; end
    g = lam * (nxq^(2 - qp)) * sign(x) .* abs(x).^(qp - 1);
end

function x = saa_solve(coef, vk, sk, lam, qp, M_pen, x0, maxit)
    step = 1e-4;
    stop_window = 10;  stop_tol = 1e-9;
    obj_history = zeros(maxit, 1);
    x = x0;
    for k = 1:maxit
        [~, gf] = obj_and_grad(x, coef, vk, sk, M_pen);
        g = gf + reg_grad(x, lam, qp);
        x = x - step * g;
        of = obj_and_grad(x, coef, vk, sk, M_pen);
        obj_history(k) = of + 0.5 * lam * norm(x, qp)^2;
        if k >= stop_window
            recent = obj_history(k-stop_window+1:k);
            base = max(1.0, mean(abs(recent)));
            if max(abs(diff(recent))) / base < stop_tol; break; end
        end
    end
end

function x = lasso_solve(coef, vk, sk, lam, M_pen, x0, maxit)
    step = 1e-4;
    stop_window = 10;  stop_tol = 1e-9;
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
            if max(abs(diff(recent))) / base < stop_tol; break; end
        end
    end
end
