function [y, e, wgt, MSE, varargout] = simN_ISM_FNLMS(x, d, W0, lambda, lambda_a, beta, zeta, varargin)
% Improved Set-Membership Fast NLMS (ISM-FNLMS) with SM-APA (P=4) + Proportionate weighting (P-ISM)

    % ----- 参数与初始化 -----
    c0 = 1; ca = 1; E0 = 1;
    if nargin >= 8 && ~isempty(varargin{1}), c0 = varargin{1}; end
    if nargin >= 9 && ~isempty(varargin{2}), ca = varargin{2}; end
    if nargin >= 10 && ~isempty(varargin{3}), E0 = varargin{3}; end

    x = x(:); d = d(:); W = W0(:);
    M = length(W); L = length(x);

    % 访问 x(n-M)：前置零填充
    x_pad = [zeros(M,1); x];

    y = zeros(L,1); e = zeros(L,1); MSE = zeros(L,1);
    wgt = zeros(M,L);

    % FNLMS状态
    c_tilde = zeros(M,1);
    y_like  = 1;
    alpha_v = E0;
    r1 = 0; r0 = E0;

    % 诊断
    y_hist   = zeros(L,1);
    mu_hist  = zeros(L,1);
    est_abs_hist = zeros(L,1);

    abs_err_est = 0; % 递归绝对误差估计初值

    % 数值安全
    eps_denom = 1e-12;
    y_min = 1e-6; y_max = 10;   % 放宽上限以增强瞬态
    delta_clip = 100;

    % 暖启动
    T_warm  = 200;          % 暖启动使用瞬时误差门控
    T_boost = 300;          % 启动脉冲持续时长
    boost_max = 1.5;        % 初期步长提升上限（线性衰减到 1）
    nlms_fallback_mu = 0.01;% SM 关闭时的小步长回退（仅在 |e| 很大时）

    % APA 设置
    P = 4;                  % 投影维度
    delta_APA = 1e-3;       % APA 正则化

    for n = 1:L
        % 输入缓冲向量 u(n)
        u = zeros(M,1);
        for t = 1:M
            src = n - (t - 1);
            if src >= 1, u(t) = x(src); end
        end

        % 输出与误差（原始）
        y(n) = W' * u;
        e(n) = d(n) - y(n);

        % 递归绝对误差估计
        abs_err_est = beta * abs_err_est + (1 - beta) * abs(e(n));

        % 前向预测器（冻结前两点）
        if n > 2
            r1 = lambda_a * r1 + x(n) * x(n-1);
            r0 = lambda_a * r0 + x(n)^2;
            a_hat = r1 / (r0 + ca);
            e_pred = x(n) - a_hat * x(n-1);
        else
            e_pred = x(n);
        end

        % α(n-1)
        alpha_prev = alpha_v;

        % 构造 c̃ 与 c_n（保留）
        denom = max(lambda * alpha_prev + c0, eps_denom);
        c_tilde = [- e_pred / denom; c_tilde(1:end-1)];
        c_n = e_pred / denom;

        % δ(n)（保留）
        x_n_M = x_pad(n);
        delta = c_n * x_n_M + (x(n) * e_pred) / denom;
        delta = max(min(delta, delta_clip), -delta_clip);

        % y_like 更新（保留）
        y_like = y_like / (1 + y_like * delta);
        y_like = min(max(y_like, y_min), y_max);

        % --- 误差度量（暖启动与平滑过渡）---
        if n <= T_warm
            err_metric = abs(e(n));
        else
            k = min(n - T_warm, T_warm);
            w_inst = (T_warm - k) / T_warm;    % 逐步减小
            w_rec  = 1 - w_inst;
            err_metric = w_inst * abs(e(n)) + w_rec * abs_err_est;
        end

        % ------------------------------------------------------------------
        % 方向构造：SM-APA (P=4) + P-ISM（替代原 y_like * c_tilde）
        % ------------------------------------------------------------------
        K = min(P, n);
        U = zeros(K, M); E = zeros(K, 1);

        for k = 1:K
            idx = n - k + 1;
            % 构造 u_k（长度 M 的延迟输入向量）
            u_k = zeros(M,1);
            for t = 1:M
                s = idx - (t - 1);
                if s >= 1, u_k(t) = x(s); end
            end
            U(k,:) = u_k.';
            % 对应误差 E(k)
            y_k = W' * u_k;
            E(k) = d(idx) - y_k;
        end

        R = U * U.' + delta_APA * eye(K);
        Gproj = U.' * (R \ E);       % M x 1，APA 更新方向

        dir_vec_base = y_like * Gproj;

        % 比例加权（P-ISM / IPNLMS样式）
        eps_p = 1e-6;
        g_weights = abs(W) + eps_p;
        g_weights = g_weights / max(mean(g_weights), eps_denom);
        Gp = diag(g_weights);
        dir_vec = Gp * dir_vec_base;

        % ------------------------------------------------------------------
        % 几何因子与步长投影（替代原基于 c_tilde 的几何因子）
        % ------------------------------------------------------------------
        proj = abs(dir_vec' * u);
        energy_term = norm(dir_vec) * norm(u);
        weight_energy = (n <= T_boost) * 0.2 + (n > T_boost) * 0.1; % 初期更强
        geom = proj + weight_energy * energy_term;
        geom = max(geom, eps_denom);

        if err_metric > zeta
            mu_eff = (1 - (zeta / err_metric)) / geom;

            % 启动脉冲：前 T_boost 点线性提升
            if n <= T_boost
                boost = 1 + (boost_max - 1) * (1 - n / T_boost);
                mu_eff = mu_eff * boost;
            end

            mu_eff = min(max(mu_eff, 0), 5);
        else
            mu_eff = 0;
        end

        % 权重更新（替代原 W = W - mu_eff * e * (y_like * c_tilde)）
        if mu_eff > 0
            W = W - mu_eff * e(n) * dir_vec;
        else
            % 若误差仍较大且 SM 关闭，做一个小的 NLMS 回退
            if abs(e(n)) > 2 * zeta
                denom_nlms = (u' * u) + eps_denom;
                W = W + (nlms_fallback_mu * e(n) / denom_nlms) * u;
            end
        end

        % 方差更新（保留）
        alpha_v = lambda * alpha_prev + e_pred^2;

        % 记录
        wgt(:,n) = W;
        MSE(n) = e(n)^2;
        y_hist(n) = y_like;
        mu_hist(n) = mu_eff;
        est_abs_hist(n) = abs_err_est;
    end

    % 可选输出
    if nargout >= 5, varargout{1} = y_hist; end
    if nargout >= 6, varargout{2} = est_abs_hist; end
    if nargout >= 7, varargout{3} = mu_hist; end
end
