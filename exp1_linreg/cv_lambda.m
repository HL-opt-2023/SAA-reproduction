function lambda_best = cv_lambda()
% CV_LAMBDA  Cross-validate the regularization weight lambda_0 for
% Experiment 1 (light-tailed linear regression).  Follows Section 4.1
% of Liu & Tong 2026:
%   - fix d = 1000, N = 200
%   - candidate grid: {0.01, 0.05, 0.10, ..., 0.50}
%   - 5 replications, each with two i.i.d. (a, b) samples (training and
%     validation), both drawn through gen_problem with distinct seeds
%   - score = mean((A_va * x - b_va)^2) on the validation sample
% Independently for SAA-L_q' (q' in {1.01, 1.5, 2}) and LASSO.
%
% Returns a containers.Map keyed by method name -> selected lambda.
% Also writes ../data/exp1/lambda_best.mat for bookkeeping.
%
% Can be invoked standalone from the MATLAB prompt:
%   >> lambda_best = cv_lambda();

rng(0);

% --- problem / cv setup ---
d_cv         = 1000;
N_cv         = 200;
gd_max_iter  = 500000;
q_prime_list = [1.01, 1.50, 2.00];
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
            [A_tr, b_tr, ~, ~] = gen_problem(d_cv, N_cv, rep);
            [A_va, b_va, ~, ~] = gen_problem(d_cv, N_cv, rep + 100);
            x0 = 1.0 * (rand(d_cv, 1) - 0.5);
            if strcmp(name, 'LASSO')
                x = solve_lasso(A_tr, b_tr, lam, x0, gd_max_iter);
            else
                q = sscanf(name, 'SAA-L%f');
                x = solve_saa_reg(A_tr, b_tr, lam, q, x0, gd_max_iter);
            end
            vals(rep) = mean((A_va * x - b_va).^2);
        end
        m = mean(vals);
        fprintf('  lam=%.3f  mean_val=%.5f\n', lam, m);
        if m < best_val; best_val = m; best_lam = lam; end
    end
    fprintf('  --> best lambda for %s = %.3f\n', name, best_lam);
    lambda_best(name) = best_lam;
end

out_dir  = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'exp1');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
out_path = fullfile(out_dir, 'lambda_best.mat');
save(out_path, 'lambda_best', 'lambda_grid', 'd_cv', 'N_cv');
fprintf('\nSaved %s\n', out_path);

% ====================================================================
% Local helpers (mirror run_exp1.m).
% ====================================================================

function [A, b, xstar, Sigma] = gen_problem(d, N, seed)
    rng(seed);
    [I, J] = meshgrid(1:d, 1:d);
    Sigma  = 0.5.^abs(I - J);
    L = chol(Sigma + 1e-12*eye(d), 'lower');
    A = (L * randn(d, N))';
    r = min(d, 200);
    S  = randperm(d, r);
    Sc = setdiff(1:d, S);
    u  = randn(r, 1);
    v  = randn(d - r, 1);
    xstar = zeros(d, 1);
    xstar(S)  = 1.5 * u / norm(u, 1.8);
    xstar(Sc) = 1.5 * v / norm(v, 1.8);
    w = randn(N, 1);
    b = A * xstar + w;
end

function g = reg_grad(x, lam, qp)
    if lam == 0;            g = zeros(size(x));   return; end
    if abs(qp - 2) < 1e-12; g = lam * x;          return; end
    nxq = norm(x, qp);
    if nxq < 1e-30;         g = zeros(size(x));   return; end
    g = lam * (nxq^(2 - qp)) * sign(x) .* abs(x).^(qp - 1);
end

function x = solve_saa_reg(A, b, lam, qp, x0, maxit)
    N = size(A, 1);
    step        = 1e-4;
    stop_window = 10;  stop_tol = 1e-9;
    obj_history = zeros(maxit, 1);
    x = x0;
    for k = 1:maxit
        r = A*x - b;
        g = (2/N) * (A' * r) + reg_grad(x, lam, qp);
        x = x - step * g;
        r_new = A*x - b;
        obj_history(k) = mean(r_new.^2) + 0.5 * lam * norm(x, qp)^2;
        if k >= stop_window
            recent = obj_history(k-stop_window+1:k);
            base = max(1.0, mean(abs(recent)));
            if max(abs(diff(recent))) / base < stop_tol; break; end
        end
    end
end

function x = solve_lasso(A, b, lam, x0, maxit)
    N = size(A, 1);
    step        = 1e-4;
    stop_window = 10;  stop_tol = 1e-9;
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
            if max(abs(diff(recent))) / base < stop_tol; break; end
        end
    end
end
