function [y, e, w, gain_norm] = simSRRLS(d, x, lambda, delta, M)
% SIMSRRLS  Square-root RLS using cholupdate (稳定版)
Ns = length(d);
w = zeros(M, Ns);
w_k = zeros(M,1);
y = zeros(Ns,1);
e = zeros(Ns,1);
gain_norm = zeros(Ns,1);

% R 为协方差的上三角 Cholesky 因子缩放版
R = sqrt(delta) * eye(M);
xx = zeros(M,1);

for n = 1:Ns
    % 输入缓冲（列向量）
    xx = [x(n); xx(1:M-1)];

    % 先验输出与误差
    y(n) = w_k' * xx;
    e(n) = d(n) - y(n);

    % 遗忘：缩放 R
    R = sqrt(lambda) * R;

    % 增益（两次三角求解）
    g = R'\xx;     % 解 R' g = xx
    k = R\g;       % 解 R  k = g
    gain_norm(n) = norm(k);

    % 权重更新
    w_k = w_k + k * e(n);

    % 秩一更新保持三角结构（QR样式稳定化）
    alpha = sqrt(1 - lambda);
    R = cholupdate(R, alpha * xx, '+');  % 需要 MATLAB 的 cholupdate

    w(:,n) = w_k;
end
end
