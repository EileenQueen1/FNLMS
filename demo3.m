%% （三）LMS算法与VSLMS算法对比测试
clear; close all; clc;
% 1 参数设置
N = 1000;              % 输入信号长度
M = 4;                 % 滤波器阶数
W0 = zeros(1, M);      % 初始权重向量
mu_LMS = 0.01;         % LMS算法固定步长参数
mu_max = 0.5;
mu_min = 5e-5;
% 2 生成输入信号 (正弦信号加噪声)
f = 0.05;              % 正弦信号的频率 (归一化)
x = sin(2 * pi * f * (0:N-1)) + 0.4 * randn(1, N);  % 正弦输入信号加噪声
% x = sin(2 * pi * f * (0:N-1));
% 3 生成期望信号 d（假设目标是通过一个FIR滤波器生成）
h = [0.3, 0.5, -0.2, 0.1]; % FIR滤波器系数
d = filter(h, 1, x)+ 0.01 * randn(1, N);  % 期望信号
% 4 分别使用simLMS函数和simVSLMS函数进行自适应滤波
[y_LMS, e_LMS, W_LMS, R, p] = simLMS(x, d, mu_LMS, W0);
[y_VSLMS, e_VSLMS, mu_VSLMS, W_VSLMS] = simVSLMS(x, d, W0,[mu_min,mu_max]);
1/max(eig(R))
MSE = zeros(1,N);
mse = zeros(1,N);
for i = 1:N
    Werr = W_LMS(:,i)-h.';
    MSE(i) = Werr.'*R*Werr;
    Werr = W_VSLMS(:,i)-h.';
    mse(i) = Werr.'*R*Werr;
end

% 绘制结果对比
figure;
subplot(2, 2, 1);
plot(1:N, d, 1:N, y_LMS, 1:N, e_LMS); title('LMS: 期望信号 vs 滤波器输出');
xlabel('样本'); ylabel('振幅'); ylim([-2 2]);
legend('期望信号 d', 'LMS 滤波器输出 y','LMS: 误差信号 e'); grid on;
subplot(2, 2, 3);
plot(1:N, d, 1:N, y_VSLMS, 1:N, e_VSLMS); title('VSLMS: 期望信号 vs 滤波器输出');
xlabel('样本'); ylabel('振幅'); legend('期望信号 d', 'VSLMS 滤波器输出 y', 'VSLMS 误差信号 e');
grid on;  ylim([-2 2]);
subplot(2,2,2);
plot(1:N,MSE,1:N,mse); title('MSE学习曲线');
xlabel('迭代次数n/samples');ylabel('MSE'); grid on;
legend('LMS 学习曲线', 'VSLMS 学习曲线');
subplot(2,2,4); plot(h); hold on; 
plot(W_LMS(:,end)); plot(W_VSLMS(:,end));
title('权重向量 W 的收敛');
xlabel('样本');
ylabel('权重值');
grid on; legend('系统FIR系数','LMS 权重向量', 'VSLMS 权重向量');
tightfig;
