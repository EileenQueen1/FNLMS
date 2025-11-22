%% SR-RLS 与 NLMS 收敛性能对比
clear; close all; clc;

% ---------------- 参数设置 ----------------
N = 5000;                 % 输入信号长度
M = 256;                  % 滤波器阶数
W0 = zeros(M,1);          % 初始权重
mu_NLMS = 1.0;            % NLMS步长
lambda_SRRLS = 0.9999;    % SR-RLS遗忘因子（高阶滤波器建议更接近1）
delta_SRRLS  = 1e6;       % SR-RLS初始正则（增强初始增益）
ws = 64;                  % NMSE平滑窗口
target_db = -25;          % 收敛阈值
SNR_dB = 30;              % 期望信号信噪比

% ---------------- 生成AR(20)输入 ----------------
p = 20;
while true
    a = 0.1 * randn(1, p);
    if all(abs(roots([1, -a])) < 1), break; end
end
sigma_w = 0.1;
x_full = filter(1, [1, -a], randn(N + 1000, 1) * sigma_w);
x = x_full(1001:end);
x = x / std(x);           % 保持方差约为1，更适配RLS增益计算

% ---------------- 构造非最小相位系统 ----------------
h = [0.9, -0.7, 0.5, -0.3, 0.2, 0.1, 0.05, zeros(1, M-7)]';
h = h / norm(h);
d_system = filter(h, 1, x);

% 添加噪声
signal_power = mean(d_system.^2);
noise_power  = signal_power / (10^(SNR_dB/10));
v = sqrt(noise_power) * randn(N, 1);
d = d_system + v;

% ---------------- NLMS 仿真 ----------------
[y_NLMS, e_NLMS, W_NLMS] = simNLMS(x, d, mu_NLMS, W0);

% ---------------- SR-RLS 仿真（cholupdate 版） ----------------
[y_SRRLS, e_SRRLS, W_SRRLS, gain_norm] = simSRRLS(d, x, lambda_SRRLS, delta_SRRLS, M);

% ---------------- NMSE 计算 ----------------
nmse_nlms   = movmean(e_NLMS.^2, ws) ./ (movmean(d.^2, ws) + eps);
nmse_srrls  = movmean(e_SRRLS.^2, ws) ./ (movmean(d.^2, ws) + eps);
nmse_nlms_db  = 10 * log10(nmse_nlms);
nmse_srrls_db = 10 * log10(nmse_srrls);

% 收敛统计
conv_nlms  = find(nmse_nlms_db  <= target_db, 1); if isempty(conv_nlms),  conv_nlms  = N; end
conv_srrls = find(nmse_srrls_db <= target_db, 1); if isempty(conv_srrls), conv_srrls = N; end
speedup = conv_nlms / conv_srrls;

% ---------------- 绘图 ----------------
figure('Position', [100, 80, 1300, 720]);

subplot(2,3,1);
plot(d, 'k'); hold on;
plot(y_NLMS, 'b'); plot(e_NLMS, 'r');
title('NLMS: 期望 vs 输出'); legend('d','y','e'); grid on;

subplot(2,3,2);
plot(d, 'k'); hold on;
plot(y_SRRLS, 'b'); plot(e_SRRLS, 'r');
title('SR-RLS: 期望 vs 输出'); legend('d','y','e'); grid on;

subplot(2,3,3);
plot(nmse_nlms_db, 'b', 'LineWidth', 1.5); hold on;
plot(nmse_srrls_db, 'r', 'LineWidth', 1.5);
legend('NLMS','SR-RLS'); title(sprintf('NMSE 收敛曲线 (ws=%d, SNR=%ddB)', ws, SNR_dB));
xlabel('迭代次数'); ylabel('NMSE (dB)'); grid on;

subplot(2,3,4);
semilogy(vecnorm(W_NLMS - h), 'b'); hold on;
semilogy(vecnorm(W_SRRLS - h), 'r');
legend('NLMS','SR-RLS'); title('系统识别误差 ||W - h||'); xlabel('迭代次数'); grid on;

subplot(2,3,5);
plot(gain_norm, 'm'); title('SR-RLS 增益范数 ||k||'); xlabel('迭代次数'); grid on;

subplot(2,3,6);
plot(d_system, 'k'); title('无噪期望信号 d_{system}'); grid on;

% 输出统计
fprintf('\n=== 收敛统计 ===\n');
fprintf('NLMS  收敛到 %.1f dB 所需迭代: %d\n', target_db, conv_nlms);
fprintf('SR-RLS收敛到 %.1f dB 所需迭代: %d\n', target_db, conv_srrls);
fprintf('SR-RLS 收敛速度提升: %.2f 倍\n', speedup);
