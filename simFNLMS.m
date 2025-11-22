function [y, e, wgt, MSE, varargout] = simFNLMS(x, d, mu, W0, lambda, lambda_a, varargin)
% SIMFNLMS  Fast-convergence NLMS-type algorithm (Benallal & Arezki formulation, Version 2)
%
% [y,e,wgt,MSE,gamma_hist,c_tilde_norm] =
%       simFNLMS(x,d,mu,W0,lambda,lambda_a,c0,ca,E0)
%
% Inputs:
%   x,d        : 输入和期望信号
%   mu         : 步长
%   W0         : 初始权重向量
%   lambda     : 方差更新的遗忘因子 (0.95~0.999)
%   lambda_a   : 预测器的遗忘因子
%   c0,ca,E0   : 可选常量 (default = 1)
%
% Outputs:
%   y,e        : 输出信号与误差信号
%   wgt        : 权重演变矩阵 (N x L)
%   MSE        : 瞬时平方误差
%   gamma_hist : gamma_N 的历史
%   c_tilde_norm : ||c_tilde_N|| history

% ------------------------------------------------------------
%  Default parameters
% ------------------------------------------------------------
c0 = 1; ca = 1; E0 = 1;
if nargin >= 7, c0 = varargin{1}; end
if nargin >= 8, ca = varargin{2}; end
if nargin >= 9, E0 = varargin{3}; end

x = x(:); d = d(:); W0 = W0(:);
N = length(W0); L = length(x);

% ------------------------------------------------------------
%  Initialization (CN(0)=0, hN(0)=W0, YN(0)=1, r1(0)=0, α(0)=r0(0)=E0)
% ------------------------------------------------------------
W = W0;
y = zeros(L,1); e = zeros(L,1); MSE = zeros(L,1);
wgt = zeros(N,L);

c_tilde_N = zeros(N,1);    % \tilde{c}_N
gamma_N = 1;               % Y_N(0) = 1
alpha_N = E0;              % 方差 α_N(0)
r1 = 0; r0 = E0;           % 自相关初值
u_buf = zeros(N,1);

% diagnostics
gamma_hist = zeros(L,1);
c_tilde_norm = zeros(L,1);

% numeric safety constants
gamma_min = 1e-6;
gamma_max = 5;
eps_denom = 1e-12;

% ------------------------------------------------------------
%  Main iteration
% ------------------------------------------------------------
for n = 1:L
    % --- input buffer update ---
    u_buf = [x(n); u_buf(1:end-1)];
    u = u_buf;

    % --- predictor & prediction error (13a,13b,12,10) ---
    if n > 1
        r1 = lambda_a * r1 + x(n) * x(n-1);
        r0 = lambda_a * r0 + x(n)^2;
        a = r1 / (r0 + ca);
        e_pred = x(n) - a * x(n-1);
    else
        e_pred = x(n);
    end

    % --- save previous alpha (α_{n-1}) ---
    alpha_prev = alpha_N;

    % --- build \tilde{c}_N(n) (8) ---
    denom = lambda * alpha_prev + c0;
    denom = max(denom, eps_denom);
    c_tilde_prev = c_tilde_N;
    % FTF-consistent convention: first element = -c(n), rest shift
    c_tilde_N = [- e_pred / denom; c_tilde_prev(1:end-1)];

    % --- compute v(n) = \tilde{c}_N .* u ---
    v_n = c_tilde_N .* u;

    % --- update γ_N (Version 2, eq. (14)-(15)) ---
    sum_v = sum(v_n);
    gamma_N = 1 / (1 - sum_v);
    gamma_N = min(max(gamma_N, gamma_min), gamma_max);

    % --- effective gain vector c_N = γ_N * \tilde{c}_N ---
    c_N = gamma_N * c_tilde_N;

    % --- 输出和误差 (1) ---
    y(n) = W' * u;
    e(n) = d(n) - y(n);

    % --- 权重更新 (2) ---
    W = W - mu * e(n) * c_N;

    % --- update α_N (9) ---
    alpha_N = lambda * alpha_prev + e_pred^2;

    % --- record ---
    wgt(:,n) = W;
    MSE(n) = e(n)^2;
    gamma_hist(n) = gamma_N;
    c_tilde_norm(n) = norm(c_tilde_N);
end

% ------------------------------------------------------------
%  Optional outputs
% ------------------------------------------------------------
if nargout >= 5, varargout{1} = gamma_hist; end
if nargout >= 6, varargout{2} = c_tilde_norm; end
end
