function [y, e, wgt, varargout] = simAPA(x, d, mu, W0, P, delta)
% simAPA  Affine Projection Algorithm
% x,d: 输入与期望信号
% mu: 步长
% W0: 初始权重
% P: 投影阶数
% delta: 正则化项

x = x(:)'; d = d(:)'; W = W0(:);
N = length(W); L = length(x);

y = zeros(1,L); e = zeros(1,L);
wgt = zeros(N,L);
gain_norm = zeros(1,L);

for n = 1:L
    K = min(P,n); % 前期不足时用 n 行
    U = zeros(K,N); D = zeros(K,1);

    for k = 1:K
        idx = n-k+1;
        u_k = zeros(1,N);
        for t = 1:N
            src = idx-(t-1);
            if src>=1
                u_k(t) = x(src);
            else
                u_k(t) = 0;
            end
        end
        U(k,:) = u_k;
        D(k) = d(idx);
    end

    y(n) = W' * U(1,:)';
    e(n) = D(1) - y(n);

    E = D - U*W;
    R = U*U' + delta*eye(K);
    G = R\E;
    update = mu * U' * G;
    W = W + update;

    wgt(:,n) = W;
    gain_norm(n) = norm(update);
end

if nargout>=4
    varargout{1} = gain_norm;
end
end
