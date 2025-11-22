function [y, e, wgt, Mu] = simNLMS(x, d, mu, W0)
% 经典 NLMS 实现（用于对照）
    x = x(:); d = d(:); W = W0(:);
    L = length(x); N = length(W0);

    y = zeros(L,1); e = zeros(L,1);
    wgt = zeros(N,L);
    Mu  = zeros(L,1);
    eps_denom = 1e-12;

    for n = 1:L
        % 输入缓冲 u
        u = zeros(N,1);
        for t = 1:N
            src = n - (t - 1);
            if src >= 1, u(t) = x(src); end
        end

        y(n) = W' * u;
        e(n) = d(n) - y(n);

        den = (u' * u) + eps_denom;
        W = W + (mu * e(n) / den) * u;

        wgt(:,n) = W;
        Mu(n) = mu;
    end
end
