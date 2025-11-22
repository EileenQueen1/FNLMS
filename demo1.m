% main_complete.m
% 完整版：包含所有函数的FNLMS和NLMS性能对比

clear; clc; close all;

%% 参数设置 - 优化参数
N = 32;             % 滤波器长度
mu_NLMS = 0.1;      % NLMS步长
mu_FNLMS = 0.05;    % FNLMS步长
lambda = 0.98;      % 遗忘因子
lambda_a = 0.95;    % 自相关估计遗忘因子
num_iter = 3000;    
SNR = 40;

% FNLMS正则化参数
c0 = 0.1;          
ca = 0.1;          
E0 = 0.1;          

%% 生成信号
fprintf('生成信号...\n');

% 使用高度相关信号
order_AR = 1;
AR_coeffs = [1, -0.95];
x_clean = filter(1, AR_coeffs, randn(num_iter+200, 1));
x_clean = x_clean(201:end);
x_clean = x_clean / std(x_clean);
x = x_clean';

% 未知系统
unknown_system = randn(N, 1);
unknown_system = unknown_system / norm(unknown_system);

% 期望信号
d_clean = zeros(size(x));
u_buffer = zeros(1, N);
for i = 1:length(x)
    u_buffer = [x(i), u_buffer(1:N-1)];
    d_clean(i) = unknown_system' * u_buffer';
end

% 添加噪声
noise_var = var(d_clean) / (10^(SNR/10));
d_noisy = d_clean + sqrt(noise_var) * randn(size(d_clean));

%% 运行NLMS算法
fprintf('运行NLMS算法...\n');
W0 = zeros(N, 1);
[y_NLMS, e_NLMS, wgt_NLMS, Mu_history] = simNLMS_complete(x, d_noisy, mu_NLMS, W0);
MSE_NLMS = e_NLMS.^2;

%% 运行FNLMS算法
fprintf('运行FNLMS算法...\n');
[y_FNLMS, e_FNLMS, wgt_FNLMS, MSE_FNLMS, gamma_history] = simFNLMS_complete(x, d_noisy, mu_FNLMS, W0, lambda, lambda_a, c0, ca, E0);

%% 性能对比
figure('Position', [100, 100, 1200, 400]);

% MSE对比
subplot(1,3,1);
MSE_FNLMS_dB = 10*log10(MSE_FNLMS + 1e-15);
MSE_NLMS_dB = 10*log10(MSE_NLMS + 1e-15);

plot(1:num_iter, MSE_FNLMS_dB, 'r-', 'LineWidth', 2); hold on;
plot(1:num_iter, MSE_NLMS_dB, 'b--', 'LineWidth', 2);
xlabel('迭代次数');
ylabel('MSE (dB)');
legend('FNLMS', 'NLMS', 'Location', 'northeast');
title('MSE收敛性能对比');
grid on;

% 误差曲线
subplot(1,3,2);
plot(e_FNLMS, 'r', 'LineWidth', 1); hold on;
plot(e_NLMS, 'b', 'LineWidth', 1);
xlabel('迭代次数');
ylabel('误差');
legend('FNLMS误差', 'NLMS误差');
title('误差曲线');
grid on;

% Gamma变化
subplot(1,3,3);
plot(gamma_history, 'LineWidth', 1);
xlabel('迭代次数');
ylabel('\gamma_N');
title('似然变量变化');
grid on;

%% 性能统计
final_segment = floor(0.7*num_iter):num_iter;
final_MSE_FNLMS = mean(MSE_FNLMS(final_segment));
final_MSE_NLMS = mean(MSE_NLMS(final_segment));

fprintf('\n=== 性能统计 ===\n');
fprintf('FNLMS 最终 MSE: %.6f (%.2f dB)\n', final_MSE_FNLMS, 10*log10(final_MSE_FNLMS + 1e-15));
fprintf('NLMS  最终 MSE: %.6f (%.2f dB)\n', final_MSE_NLMS, 10*log10(final_MSE_NLMS + 1e-15));

if final_MSE_FNLMS < final_MSE_NLMS
    improvement = (final_MSE_NLMS - final_MSE_FNLMS) / final_MSE_NLMS * 100;
    fprintf('✓ FNLMS性能提升: %.2f%%\n', improvement);
else
    degradation = (final_MSE_FNLMS - final_MSE_NLMS) / final_MSE_NLMS * 100;
    fprintf('✗ FNLMS性能下降: %.2f%%\n', degradation);
    fprintf('需要进一步优化参数...\n');
end

%% 内嵌函数定义

function [y,e,wgt,varargout] = simNLMS_complete(x,d,mu,W0,varargin)
% 完整的NLMS算法实现
    if nargin < 4
        error('至少需要4个输入参数: x, d, mu, W0');
    end
    
    x = x(:)';
    d = d(:)';
    W0 = W0(:);
    
    W = W0;
    N = length(W);
    L = length(x);
    
    y = zeros(1, L);
    e = zeros(1, L);
    u = zeros(1, N);
    Mu = zeros(1, L);
    wgt = zeros(N, L);
    
    for i = 1:L
        u = [x(i), u(1:N-1)];
        y(i) = W' * u';
        e(i) = d(i) - y(i);
        Mu(i) = mu / (eps + u * u');
        W = W + (Mu(i) * e(i)) * u';
        wgt(:, i) = W;
    end
    
    if nargout >= 4
        varargout{1} = Mu;
    end
end

function [y, e, wgt, MSE, varargout] = simFNLMS_complete(x, d, mu, W0, lambda, lambda_a, varargin)
% 完整的FNLMS算法实现 - 使用论文Version 2计算gamma
    if nargin < 6
        error('至少需要6个输入参数: x, d, mu, W0, lambda, lambda_a');
    end
    
    x = x(:)';
    d = d(:)';
    W0 = W0(:);
    
    % 默认参数
    c0 = 1; ca = 1; E0 = 1;
    if nargin >= 7, c0 = varargin{1}; end
    if nargin >= 8, ca = varargin{2}; end
    if nargin >= 9, E0 = varargin{3}; end
    
    W = W0;
    N = length(W);
    L = length(x);
    
    y = zeros(1, L);
    e = zeros(1, L);
    MSE = zeros(1, L);
    wgt = zeros(N, L);
    
    % 初始化变量
    c_bar_N = zeros(N, 1);
    gamma_N = 1;
    alpha = E0;
    r1 = 0;
    r0 = E0;
    
    u_buffer = zeros(1, N);
    v_history = zeros(1, N);  % 用于Version 2计算gamma
    
    if nargout >= 5
        gamma_history = zeros(1, L);
    end
    
    for i = 1:L
        % 更新输入向量
        u_buffer = [x(i), u_buffer(1:end-1)];
        u = u_buffer;
        
        % 预测误差计算
        if i > 1
            r1 = lambda_a * r1 + x(i) * x(i-1);
            r0 = lambda_a * r0 + x(i)^2;
            a = r1 / (r0 + ca);
            e_pred = x(i) - a * x(i-1);
        else
            e_pred = x(i);
        end
        
        % 更新前向预测误差方差
        alpha = lambda * alpha + e_pred^2;
        
        % 更新对偶Kalman增益
        denom = lambda * alpha + c0;
        if denom < 1e-10, denom = 1e-10; end
        
        c_bar_N = [-e_pred / denom; c_bar_N(1:end-1)];
        
        % 关键改进：使用论文Version 2计算gamma (公式14-15)
        v = c_bar_N(1) * x(i);  % v(n) = c_bar_N^1(n) * x(n)
        v_history = [v, v_history(1:end-1)];
        
        % gamma_N = 1 / (1 - sum(v_history))
        gamma_N = 1 / (1 - sum(v_history) + 1e-10);
        
        % 限制gamma范围
        gamma_N = max(min(gamma_N, 10), 0.1);
        
        % 滤波输出和误差
        y(i) = W' * u';
        e(i) = d(i) - y(i);
        
        % 更新滤波器系数
        W = W - mu * e(i) * gamma_N * c_bar_N;
        
        % 存储结果
        wgt(:, i) = W;
        MSE(i) = e(i)^2;
        
        if nargout >= 5
            gamma_history(i) = gamma_N;
        end
    end
    
    if nargout >= 5
        varargout{1} = gamma_history;
    end
end