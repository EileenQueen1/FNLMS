%% NLMS / FNLMS 对比测试（贴近论文设置）
clear; close all; clc;

% 1 参数设置
N = 2e5;               % 输入信号长度
M = 256;               % 滤波器阶数
num_runs = 2;          % 运行次数
W0 = zeros(M, 1);      % 初始权重向量

% NLMS参数
mu_NLMS = 1.0;

% FNLMS参数
mu_FNLMS = 0.6;
lambda   = 0.99;
lambda_a = 0.9985;

% NMSE平滑窗口
ws = 1024;

% 存储统计结果
convergence_stats = zeros(2, num_runs); % [NLMS; FNLMS]
final_nmse_stats  = zeros(2, num_runs);

% 用于绘图的最后一次运行数据
last_x = []; last_d = [];
last_y_NLMS = []; last_e_NLMS = []; last_W_NLMS = [];
last_y_FNLMS = []; last_e_FNLMS = []; last_W_FNLMS = [];
last_h = [];
last_nmse_nlms_db = []; last_nmse_fnlms_db = [];

fprintf('开始 %d 次运行统计...\n\n', num_runs);

for run_idx = 1:num_runs
    fprintf('运行 %d/%d...\n', run_idx, num_runs);

    %% 2 输入信号：WGN-AR(20)
    p = 20;
    a = [0.92, -0.85, 0.72, -0.58, 0.47, -0.39, 0.31, -0.26, 0.21, -0.17, ...
         0.14, -0.12, 0.10, -0.085, 0.072, -0.060, 0.050, -0.042, 0.035, -0.030];
    if any(abs(roots([1, -a])) >= 1), a = 0.95 * a; end
    sigma_w = 0.1;
    w = randn(N + 1000, 1) * sigma_w;
    x_full = filter(1, [1, -a], w);
    x = x_full(1001:end);
    x = x / max(abs(x));

    %% 3 系统与期望信号
    h = randn(M,1);
    h = filter([1 0.5 -0.3], [1 -0.2 0.1], h);
    h = h .* hamming(M);
    h = h / norm(h);
    d_system = filter(h, 1, x);

    SNR_dB = 30;
    signal_power = mean(d_system.^2);
    noise_power  = signal_power / (10^(SNR_dB/10));
    v = sqrt(noise_power) * randn(N, 1);
    d = d_system + v;

    %% 4 两种算法
    fprintf('运行NLMS...\n');
    [y_NLMS, e_NLMS, W_NLMS, ~] = simNLMS(x, d, mu_NLMS, W0);

    fprintf('运行FNLMS...\n');
    [y_FNLMS, e_FNLMS, W_FNLMS, ~, ~, ~] = simFNLMS(x, d, mu_FNLMS, W0, lambda, lambda_a);

    %% 5 NMSE计算（滑动平均）
    nmse_nlms  = movmean(e_NLMS.^2,  ws) ./ (movmean(d.^2, ws) + eps);
    nmse_fnlms = movmean(e_FNLMS.^2, ws) ./ (movmean(d.^2, ws) + eps);

    nmse_nlms_db  = 10*log10(nmse_nlms);
    nmse_fnlms_db = 10*log10(nmse_fnlms);

    %% 6 收敛统计
    target_db = -20;
    conv_nlms  = find(nmse_nlms_db  <= target_db, 1); if isempty(conv_nlms),  conv_nlms  = N; end
    conv_fnlms = find(nmse_fnlms_db <= target_db, 1); if isempty(conv_fnlms), conv_fnlms = N; end

    convergence_stats(:, run_idx) = [conv_nlms; conv_fnlms];
    final_nmse_stats(:, run_idx)  = [nmse_nlms_db(end); nmse_fnlms_db(end)];

    %% 7 保存最后一次运行数据
    if run_idx == num_runs
        last_x = x; last_d = d; last_h = h;
        last_y_NLMS = y_NLMS; last_e_NLMS = e_NLMS; last_W_NLMS = W_NLMS;
        last_y_FNLMS = y_FNLMS; last_e_FNLMS = e_FNLMS; last_W_FNLMS = W_FNLMS;
        last_nmse_nlms_db = nmse_nlms_db;
        last_nmse_fnlms_db = nmse_fnlms_db;
    end
end

%% 8 绘图
figure('Position',[100,100,1200,600]);

subplot(2,2,1);
plot(last_d,'k'); hold on;
plot(last_y_NLMS,'b'); plot(last_e_NLMS,'r');
title('NLMS: 期望 vs 输出'); legend('d','y','e'); grid on;

subplot(2,2,2);
plot(last_d,'k'); hold on;
plot(last_y_FNLMS,'b'); plot(last_e_FNLMS,'r');
title('FNLMS: 期望 vs 输出'); legend('d','y','e'); grid on;

subplot(2,2,3);
semilogy(vecnorm(last_W_NLMS - last_h),'b','LineWidth',1.3); hold on;
semilogy(vecnorm(last_W_FNLMS - last_h),'r','LineWidth',1.3);
legend('NLMS','FNLMS','Location','best');
title('系统识别误差 ||W - h||'); xlabel('迭代次数'); grid on;

subplot(2,2,4);
plot(last_nmse_nlms_db,  'b', 'LineWidth', 1.5); hold on;
plot(last_nmse_fnlms_db, 'r', 'LineWidth', 1.5);
xlabel('迭代次数'); ylabel('NMSE (dB)');
legend('NLMS','FNLMS','Location','best');
title(sprintf('NMSE 收敛曲线对比（ws=%d, SNR=%ddB）', ws, SNR_dB));
grid on;

%% 9 性能统计
fprintf('\n=== 算法性能统计 (%d 次运行平均) ===\n', num_runs);

fprintf('\n最终 NMSE (dB)：\n');
fprintf('NLMS:      %.2f ± %.2f\n', mean(final_nmse_stats(1,:)), std(final_nmse_stats(1,:)));
fprintf('FNLMS:     %.2f ± %.2f\n', mean(final_nmse_stats(2,:)), std(final_nmse_stats(2,:)));

fprintf('\n收敛到 %.1f dB 所需迭代次数：\n', target_db);
fprintf('NLMS:      %.0f ± %.1f (范围: %d–%d)\n', ...
    mean(convergence_stats(1,:)), std(convergence_stats(1,:)), ...
    min(convergence_stats(1,:)), max(convergence_stats(1,:)));
fprintf('FNLMS:     %.0f ± %.1f (范围: %d–%d)\n', ...
    mean(convergence_stats(2,:)), std(convergence_stats(2,:)), ...
    min(convergence_stats(2,:)), max(convergence_stats(2,:)));

% 收敛速度提升倍数（相对 NLMS）
speedup_fnlms = mean(convergence_stats(1,:)) / mean(convergence_stats(2,:));
fprintf('\n相对 NLMS 的收敛速度提升：\n');
fprintf('FNLMS:     %.2f 倍\n', speedup_fnlms);
