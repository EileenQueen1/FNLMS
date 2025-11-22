function [y, e, w] = simRLS(d, x, lambda, delta, M)
% SIMRLS_STABLE  稳定且快速收敛的RLS实现
Ns = length(d);
P = delta * eye(M);
w1 = zeros(M,1);
y = zeros(Ns,1);
e = zeros(Ns,1);
w = zeros(M,Ns);
xx = zeros(M,1);

for n = 1:Ns
    xx = [x(n); xx(1:M-1)];
    denom = lambda + xx' * P * xx;
    k = (P * xx) / (denom + eps);
    y(n) = w1' * xx;
    e(n) = d(n) - y(n);
    w1 = w1 + k * e(n);
    P = (P - k * xx' * P) / lambda;
    w(:,n) = w1;
end
end
