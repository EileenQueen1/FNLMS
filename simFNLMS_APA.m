function [y, e, wgt, varargout] = simFNLMS_APA(x, d, mu, W0, lambda, lambda_a, c0, ca, E0, P, delta)
% simFNLMS_APA  结合 FNLMS 与 APA 的自适应滤波算法
%
% Inputs:
%   x,d        : 输入与期望信号
%   mu         : 步长
%   W0         : 初始权重
%   lambda     : 方差更新遗忘因子
%   lambda_a   : 预测器遗忘因子
%   c0,ca,E0   : 正则化常数
%   P          : APA 投影阶数
%   delta      : APA 正则化项
%
% Outputs:
%   y,e        : 输出与误差
%   wgt        : 权重演化矩阵
%   varargout{1} : γ 历史
%   varargout{2} : ||c̃|| 历史

x = x(:)'; d = d(:)'; W = W0(:);
N = length(W); L = length(x);

y = zeros(1,L); e = zeros(1,L);
wgt = zeros(N,L);

c_tilde = zeros(N,1); gamma = 1;
alpha = E0; r1 = 0; r0 = E0;

gamma_hist = zeros(1,L); c_tilde_norm = zeros(1,L);

for n = 1:L
    % --- 构造输入矩阵 U (P x N) ---
    K = min(P,n);
    U = zeros(K,N); D = zeros(K,1);
    for k = 1:K
        idx = n-k+1;
        u_k = zeros(1,N);
        for t = 1:N
            src = idx-(t-1);
            if src>=1, u_k(t)=x(src); end
        end
        U(k,:) = u_k;
        D(k) = d(idx);
    end

    % --- 预测器 ---
    if n>1
        r1 = lambda_a*r1 + x(n)*x(n-1);
        r0 = lambda_a*r0 + x(n)^2;
        a = r1/(r0+ca);
        e_pred = x(n) - a*x(n-1);
    else
        e_pred = x(n);
    end

    % --- 构造 c̃ ---
    denom = lambda*alpha + c0;
    c_tilde = [-e_pred/denom; c_tilde(1:end-1)];
    c_n = e_pred/denom;

    % --- δ 与 γ 更新 ---
    delta_val = c_n*x(max(n-N,1)) + (x(n)*e_pred)/denom;
    gamma = gamma/(1+gamma*delta_val);
    gamma = min(max(gamma,1e-6),5);

    % --- 输出与误差 ---
    y(n) = W' * U(1,:)';
    e(n) = D(1) - y(n);

    % --- APA 更新方向 ---
    E = D - U*W;
    R = U*U' + delta*eye(K);
    G = R\E;
    update = mu * U' * G;

    % --- 融合 FNLMS 增益 ---
    W = W + gamma * update;

    % --- 方差更新 ---
    alpha = lambda*alpha + e_pred^2;

    % --- 记录 ---
    wgt(:,n) = W;
    gamma_hist(n) = gamma;
    c_tilde_norm(n) = norm(c_tilde);
end

if nargout>=4, varargout{1}=gamma_hist; end
if nargout>=5, varargout{2}=c_tilde_norm; end
end
