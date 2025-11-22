function [y, e, wgt, MSE, varargout] = simISM_FNLMS(x, d, W0, lambda, lambda_a, beta, zeta, varargin)
% Improved Set-Membership Fast NLMS (ISM-FNLMS) with NLMS–ISM mixing and per-sample double updates

    % ----- 参数与初始化 -----
    c0 = 1; ca = 1; E0 = 1;
    if nargin >= 8 && ~isempty(varargin{1}), c0 = varargin{1}; end
    if nargin >= 9 && ~isempty(varargin{2}), ca = varargin{2}; end
    if nargin >= 10 && ~isempty(varargin{3}), E0 = varargin{3}; end

    x = x(:); d = d(:); W = W0(:);
    N = length(W); L = length(x);

    % 访问 x(n-L)：前置零填充
    x_pad = [zeros(N,1); x];

    y = zeros(L,1); e = zeros(L,1); MSE = zeros(L,1);
    wgt = zeros(N,L);

    % FNLMS状态
    c_tilde = zeros(N,1);
    y_like  = 1;
    alpha_v = E0;
    r1 = 0; r0 = E0;

    % 诊断
    y_hist   = zeros(L,1);
    mu_hist  = zeros(L,1);
    est_abs_hist = zeros(L,1);
    mix_p_hist   = zeros(L,1);

    abs_err_est = 0; % 递归绝对误差估计初值

    % 数值安全
    eps_denom = 1e-12;
    y_min = 1e-6; y_max = 10;   % 放宽上限以增强瞬态
    delta_clip = 100;

    % 暖启动与提升设置（不改变外部参数）
    T_warm  = 200;          % 暖启动使用瞬时误差门控
    T_boost = 300;          % 启动脉冲持续时长
    boost_max = 1.5;        % 初期步长提升上限（线性衰减到 1）
    nlms_fallback_mu = 0.01;% SM 关闭时的小步长回退（仅在 |e| 很大时）
    K_inner = 2;            % 每样本双更新

    % NLMS混合的误差能量平滑
    sigma_inst = 0; sigma_ism = 0;

    for n = 1:L
        % 输入缓冲
        u = zeros(N,1);
        for t = 1:N
            src = n - (t - 1);
            if src >= 1, u(t) = x(src); end
        end

        % 输出（用当前 W、不混合）
        y_raw = W' * u;
        e_raw = d(n) - y_raw;

        % 递归绝对误差估计
        abs_err_est = beta * abs_err_est + (1 - beta) * abs(e_raw);

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

        % 构造 c̃ 与 c_n
        denom = max(lambda * alpha_prev + c0, eps_denom);
        c_tilde = [- e_pred / denom; c_tilde(1:end-1)];
        c_n = e_pred / denom;

        % δ(n)
        x_n_L = x_pad(n);
        delta = c_n * x_n_L + (x(n) * e_pred) / denom;
        delta = max(min(delta, delta_clip), -delta_clip);

        % y_like 更新
        y_like = y_like / (1 + y_like * delta);
        y_like = min(max(y_like, y_min), y_max);

        % --- NLMS分支（为混合提供输出，不改变主权重）---
        den_nlms = (u' * u) + eps_denom;
        y_nlms = y_raw;                 % 与当前 W 同步的输出
        e_nlms = d(n) - y_nlms;

        % --- ISM分支方向（使用 y_like * c_tilde）---
        dir_vec_base = y_like * c_tilde;

        % --- 误差度量（暖启动与平滑过渡）---
        if n <= T_warm
            err_metric = abs(e_raw);
        else
            k = min(n - T_warm, T_warm);
            w_inst = (T_warm - k) / T_warm;    % 逐步减小
            w_rec  = 1 - w_inst;
            err_metric = w_inst * abs(e_raw) + w_rec * abs_err_est;
        end

        % --- NLMS–ISM 自适应混合权重 p(n)（误差驱动）---
        sigma_inst = 0.99 * sigma_inst + 0.01 * abs(e_nlms);
        sigma_ism  = 0.99 * sigma_ism  + 0.01 * abs(e_raw);
        p = sigma_inst / max(sigma_inst + sigma_ism, eps_denom);
        p = min(max(p, 0.05), 0.95);  % 防止极端
        mix_p_hist(n) = p;

        % --- 每样本双更新（K_inner 次小步）---
        mu_eff_total = 0;
        for k_inner = 1:K_inner
            % 几何因子（投影项 + 能量项，初期能量占比更大）
            proj = abs(dir_vec_base' * u);
            energy_term = norm(dir_vec_base) * norm(u);
            weight_energy = (n <= T_boost) * 0.2 + (n > T_boost) * 0.1;
            geom = proj + weight_energy * energy_term;
            geom = max(geom, eps_denom);

            % 集合边界投影步长
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

            % 权重更新：SM 或 NLMS 回退（分摊到双更新）
            if mu_eff > 0
                W = W - (mu_eff / K_inner) * e_raw * dir_vec_base;
                mu_eff_total = mu_eff_total + mu_eff;
            else
                if abs(e_raw) > 2 * zeta
                    W = W + (nlms_fallback_mu / K_inner) * (e_raw / den_nlms) * u;
                end
            end

            % 更新后刷新输出与误差（用于第二次小步）
            y_raw = W' * u;
            e_raw = d(n) - y_raw;
        end

        % 混合输出与误差供外部观察（不改变内部更新）
        y(n) = p * y_nlms + (1 - p) * y_raw;
        e(n) = d(n) - y(n);

        % 方差更新
        alpha_v = lambda * alpha_prev + e_pred^2;

        % 记录
        wgt(:,n) = W;
        MSE(n) = e(n)^2;
        y_hist(n) = y_like;
        mu_hist(n) = mu_eff_total / max(K_inner,1);  % 平均小步步长
        est_abs_hist(n) = abs_err_est;
    end

    % 可选输出
    if nargout >= 5, varargout{1} = y_hist; end
    if nargout >= 6, varargout{2} = est_abs_hist; end
    if nargout >= 7, varargout{3} = mu_hist; end
    if nargout >= 8, varargout{4} = mix_p_hist; end
end
