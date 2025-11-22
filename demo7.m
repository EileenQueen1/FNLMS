%% NLMS 与 ISM-FNLMS 对比测试
clear; close all; clc;

% 1 参数设置
N = 5e4;              % 输入信号长度
M = 256;               % 滤波器阶数
num_runs = 2;          % 运行次数
W0 = zeros(M, 1);      % 初始权重向量

% NLMS参数
mu_NLMS = 1;

% ISM-FNLMS 参数（论文建议）
lambda   = 0.99;       % 方差递推遗忘因子
lambda_a = 0.9975;     % 预测器遗忘因子（约25ms窗口）
beta     = 0.9975;     % |e|递归估计遗忘因子（与lambda_a同窗）
% zeta 在有噪声时设为噪声标准差量级；无噪声可设为 2e-5
zeta_no_noise = 2e-5;  % 无噪声阈值
use_noise_bound = true; % 有噪声：true；无噪声：false

% 存储统计结果
convergence_stats = zeros(2, num_runs); % [NLMS; ISM-FNLMS]
final_nmse_stats  = zeros(2, num_runs);

% 用于绘图的最后一次运行数据
last_x = []; last_d = [];
last_y_NLMS = []; last_e_NLMS = []; last_W_NLMS = [];
last_y_ISM   = []; last_e_ISM   = []; last_W_ISM   = [];
last_h = []; last_y_hist = []; last_mu_hist = []; last_abs_est_hist = []; last_Mu = [];

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

    % 3 生成期望信号 d（非最小相位系统 + 测量噪声）
    h_realistic = [0.9, -0.7, 0.5, -0.3, 0.2, 0.1, 0.05, zeros(1, M-7)]';
    h_realistic = h_realistic / norm(h_realistic);
    d_system = filter(h_realistic, 1, x);

    SNR_dB = 30;
    signal_power = mean(d_system.^2);
    noise_power = signal_power / (10^(SNR_dB/10));
    v = sqrt(noise_power) * randn(1, length(x));

    d = d_system + v;
    h = h_realistic;

    % ISM-FNLMS 的阈值选择
    if use_noise_bound
        zeta = std(v);         % 噪声标准差为阈值量级
    else
        zeta = zeta_no_noise;  % 无噪声场景固定小阈值
    end

    % 4 两种算法
    fprintf('运行NLMS...\n');
    [y_NLMS, e_NLMS, W_NLMS, Mu] = simNLMS(x, d, mu_NLMS, W0);

    fprintf('运行ISM-FNLMS...\n');
    % 可选常数 c0, ca, E0 默认都为 1，不传即可
    [y_ISM, e_ISM, W_ISM, MSE_ISM, y_hist, abs_est_hist, mu_hist] = ...
        simISM_FNLMS(x, d, W0, lambda, lambda_a, beta, zeta);

    % 5 性能指标
    NMSE_NLMS = e_NLMS.^2 / var(d);
    NMSE_ISM  = e_ISM.^2  / var(d);

    target_db = -25;
    nlms_conv = find(10*log10(NMSE_NLMS) <= target_db, 1);
    ism_conv  = find(10*log10(NMSE_ISM)  <= target_db, 1);

    if isempty(nlms_conv)
        convergence_stats(1, run_idx) = N;
    else
        convergence_stats(1, run_idx) = nlms_conv;
    end
    if isempty(ism_conv)
        convergence_stats(2, run_idx) = N;
    else
        convergence_stats(2, run_idx) = ism_conv;
    end

    final_nmse_stats(1, run_idx) = 10*log10(NMSE_NLMS(end));
    final_nmse_stats(2, run_idx) = 10*log10(NMSE_ISM(end));

    % 保存最后一次运行数据
    if run_idx == num_runs
        last_x = x; last_d = d;
        last_y_NLMS = y_NLMS; last_e_NLMS = e_NLMS; last_W_NLMS = W_NLMS;
        last_y_ISM   = y_ISM;  last_e_ISM   = e_ISM;  last_W_ISM   = W_ISM;
        last_h = h; last_y_hist = y_hist; last_mu_hist = mu_hist; last_abs_est_hist = abs_est_hist; last_Mu = Mu;
    end
end

%% 绘图
figure('Position',[100,100,1200,800]);

% NLMS
subplot(2,2,1);
plot(last_d,'k'); hold on;
plot(last_y_NLMS,'b'); plot(last_e_NLMS,'r');
title('NLMS: 期望 vs 输出'); legend('d','y','e'); grid on;

% ISM-FNLMS
subplot(2,2,2);
plot(last_d,'k'); hold on;
plot(last_y_ISM,'b'); plot(last_e_ISM,'r');
title('ISM-FNLMS: 期望 vs 输出'); legend('d','y','e'); grid on;

% 系统误差对比
subplot(2,2,3);
sys_error_NLMS = vecnorm(last_W_NLMS - last_h);
sys_error_ISM  = vecnorm(last_W_ISM  - last_h);
semilogy(sys_error_NLMS,'b','LineWidth',1.3); hold on;
semilogy(sys_error_ISM,'r','LineWidth',1.3);
legend('NLMS','ISM-FNLMS','Location','best');
title('系统识别误差'); xlabel('迭代次数'); ylabel('||W - h||'); grid on;

% NMSE 对比曲线
subplot(2,2,4);
nmse_nlms = 10 * log10(last_e_NLMS.^2 / var(last_d));
nmse_ism  = 10 * log10(last_e_ISM.^2  / var(last_d));
plot(nmse_nlms, 'b', 'LineWidth', 1.5); hold on;
plot(nmse_ism,  'r', 'LineWidth', 1.5);
xlabel('迭代次数'); ylabel('NMSE (dB)');
legend('NLMS','ISM-FNLMS','Location','best');
title('NMSE 收敛曲线对比');
grid on;


%% 性能统计
fprintf('\n=== 算法性能统计 (%d次运行平均) ===\n', num_runs);
fprintf('最终NMSE - NLMS:      %.2f ± %.2f dB\n', mean(final_nmse_stats(1, :)), std(final_nmse_stats(1, :)));
fprintf('最终NMSE - ISM-FNLMS: %.2f ± %.2f dB\n', mean(final_nmse_stats(2, :)), std(final_nmse_stats(2, :)));

fprintf('\n收敛到-25dB迭代次数:\n');
fprintf('NLMS:      %.0f ± %.1f (范围: %d-%d)\n', ...
    mean(convergence_stats(1, :)), std(convergence_stats(1, :)), ...
    min(convergence_stats(1, :)), max(convergence_stats(1, :)));
fprintf('ISM-FNLMS: %.0f ± %.1f (范围: %d-%d)\n', ...
    mean(convergence_stats(2, :)), std(convergence_stats(2, :)), ...
    min(convergence_stats(2, :)), max(convergence_stats(2, :)));

speedup = mean(convergence_stats(1, :)) / mean(convergence_stats(2, :));
fprintf('\nISM-FNLMS收敛速度提升: %.2f倍\n', speedup);
