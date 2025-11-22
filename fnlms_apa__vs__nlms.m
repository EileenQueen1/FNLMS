%% NLMS 与 FNLMS-APA 对比测试
clear; close all; clc;

% 1 参数设置
N = 5000;              % 输入信号长度
M = 256;               % 滤波器阶数
num_runs = 5;          % 运行次数
W0 = zeros(M, 1);      % 初始权重向量

% NLMS参数
mu_NLMS = 1;

% FNLMS-APA参数
mu_FNLMS_APA = 1;
lambda = 0.9985;
lambda_a = 0.99;
c0 = 0.1; ca = 0.1; E0 = 0.1;
P = 4;                 % APA投影阶数
delta_APA = 0.01;      % APA正则化项

% 存储统计结果
convergence_stats = zeros(2, num_runs); % [NLMS; FNLMS-APA]
final_nmse_stats = zeros(2, num_runs);

% 用于绘图的最后一次运行数据
last_x = []; last_d = [];
last_y_NLMS = []; last_e_NLMS = []; last_W_NLMS = [];
last_y_FNLMS_APA = []; last_e_FNLMS_APA = []; last_W_FNLMS_APA = [];
last_h = []; last_gamma_history = []; last_Mu = [];

fprintf('开始 %d 次运行统计...\n\n', num_runs);

for run_idx = 1:num_runs
    fprintf('运行 %d/%d...\n', run_idx, num_runs);

    % 2 生成AR(20)输入信号
    p = 20; N_ar = N + 1000; sigma_w = 0.1;
    while true
        a = 0.1 * randn(1, p);
        poles = roots([1, -a]);
        if all(abs(poles) < 1), break; end
    end
    model = arima('AR', a, 'Constant', 0, 'Variance', sigma_w^2);
    x_full = simulate(model, N_ar); x_full = x_full';
    x = x_full(1001:end);
    x = 0.15 * x / std(x);

    % 3 生成期望信号 d
    h_realistic = [0.9, -0.7, 0.5, -0.3, 0.2, 0.1, 0.05, zeros(1, M-7)]';
    h_realistic = h_realistic / norm(h_realistic);
    d_system = filter(h_realistic, 1, x);
    SNR_dB = 30;
    signal_power = mean(d_system.^2);
    noise_power = signal_power / (10^(SNR_dB/10));
    v = sqrt(noise_power) * randn(1, length(x));
    d = d_system + v;
    h = h_realistic;

    % 4 两种算法
    fprintf('运行NLMS...\n');
    [y_NLMS, e_NLMS, W_NLMS, Mu] = simNLMS(x, d, mu_NLMS, W0);

    fprintf('运行FNLMS-APA...\n');
    [y_FNLMS_APA, e_FNLMS_APA, W_FNLMS_APA, gamma_history] = ...
        simFNLMS_APA(x, d, mu_FNLMS_APA, W0, lambda, lambda_a, c0, ca, E0, P, delta_APA);

    % 5 性能指标
    NMSE_NLMS = e_NLMS.^2 / var(d);
    NMSE_FNLMS_APA = e_FNLMS_APA.^2 / var(d);

    target_db = -25;
    nlms_conv = find(10*log10(NMSE_NLMS) <= target_db, 1);
    fnlms_apa_conv = find(10*log10(NMSE_FNLMS_APA) <= target_db, 1);

    if isempty(nlms_conv)
        convergence_stats(1, run_idx) = N;
    else
        convergence_stats(1, run_idx) = nlms_conv;
    end
    if isempty(fnlms_apa_conv)
        convergence_stats(2, run_idx) = N;
    else
        convergence_stats(2, run_idx) = fnlms_apa_conv;
    end

    final_nmse_stats(1, run_idx) = 10*log10(NMSE_NLMS(end));
    final_nmse_stats(2, run_idx) = 10*log10(NMSE_FNLMS_APA(end));

    % 保存最后一次运行数据
    if run_idx == num_runs
        last_x = x; last_d = d;
        last_y_NLMS = y_NLMS; last_e_NLMS = e_NLMS; last_W_NLMS = W_NLMS;
        last_y_FNLMS_APA = y_FNLMS_APA; last_e_FNLMS_APA = e_FNLMS_APA; last_W_FNLMS_APA = W_FNLMS_APA;
        last_h = h; last_gamma_history = gamma_history; last_Mu = Mu;
    end
end

%% 绘图
figure('Position',[100,100,1200,800]);

% NLMS
subplot(2,1,1);
plot(last_d,'k'); hold on;
plot(last_y_NLMS,'b'); plot(last_e_NLMS,'r');
title('NLMS: 期望 vs 输出'); legend('期望信号 d','滤波器输出 y','误差信号 d'); grid on;

% FNLMS-APA
subplot(2,1,2);
plot(last_d,'k'); hold on;
plot(last_y_FNLMS_APA,'b'); plot(last_e_FNLMS_APA,'r');
title('FNLMS-APA: 期望 vs 输出'); legend('期望信号 d','滤波器输出 y','误差信号 d'); grid on;

% 系统误差对比
figure('Position',[100,100,900,500]);
sys_error_NLMS = vecnorm(last_W_NLMS - last_h);
sys_error_FNLMS_APA = vecnorm(last_W_FNLMS_APA - last_h);
semilogy(sys_error_NLMS,'b','LineWidth',1.3); hold on;
semilogy(sys_error_FNLMS_APA,'r','LineWidth',1.3);
legend('NLMS','FNLMS-APA','Location','best');
title('系统识别误差'); xlabel('迭代次数'); ylabel('||W - h||'); grid on;

% γ 历史
figure('Position',[100,100,900,400]);
plot(last_gamma_history,'g'); title('FNLMS-APA γ'); grid on;

% 6 性能统计
fprintf('\n=== 算法性能统计 (%d次运行平均) ===\n', num_runs);
fprintf('最终NMSE - NLMS:  %.2f ± %.2f dB\n', mean(final_nmse_stats(1, :)), std(final_nmse_stats(1, :)));
fprintf('最终NMSE - FNLMS: %.2f ± %.2f dB\n', mean(final_nmse_stats(2, :)), std(final_nmse_stats(2, :)));

fprintf('\n收敛到-25dB迭代次数:\n');
fprintf('NLMS:  %.0f ± %.1f (范围: %d-%d)\n', ...
    mean(convergence_stats(1, :)), std(convergence_stats(1, :)), ...
    min(convergence_stats(1, :)), max(convergence_stats(1, :)));
fprintf('FNLMS-APA: %.0f ± %.1f (范围: %d-%d)\n', ...
    mean(convergence_stats(2, :)), std(convergence_stats(2, :)), ...
    min(convergence_stats(2, :)), max(convergence_stats(2, :)));

speedup = mean(convergence_stats(1, :)) / mean(convergence_stats(2, :));
fprintf('\nFNLMS-APA收敛速度提升: %.2f倍\n', speedup);
