%% （二）LMS算法与NLMS算法对比测试
% 需要注意的是，如果输入信号初相设置不合理，可能导致步长参数在一开始就发散
clear; close all; clc;
% 1 参数设置
N = 3000;              % 输入信号长度
M = 4;                 % 滤波器阶数
W0 = zeros(1, M);      % 初始权重向量
mu_LMS = 0.05;         % LMS算法步长参数
mu_NLMS = 0.05;         % NLMS算法步长参数 (0 < mu < 2)
% 2 生成输入信号
x = 0.2*sin(2 * pi * 0.05 * (0:N-1)+pi/2) + 0.1 * randn(1, N);  % 先是正弦信号，后是噪声信号
% 3 生成期望信号 d（假设目标是通过一个FIR滤波器生成）
h = [0.3, 0.5, -0.2, 0.1]; % FIR滤波器系数
d = filter(h, 1, x);       % 期望信号
v = 0.01 * randn(1, N);    % 噪声信号
d = d + v;
% 4 分别使用simLMS函数与simNLMS函数进行自适应滤波
[y_LMS, e_LMS, W_LMS, R, P] = simLMS(x, d, mu_LMS, W0);
[y_NLMS, e_NLMS, W_NLMS] = simNLMS(x, d, mu_NLMS, W0);
% 绘制结果对比
figure;
subplot(2, 1, 1); plot(1:N, d, 1:N, y_LMS, 1:N, e_LMS);
title('LMS: 期望信号 vs 滤波器输出');
xlabel('样本'); ylabel('振幅');
legend('期望信号 d', 'LMS 滤波器输出 y','LMS 误差信号 e');
grid on;
subplot(2, 1, 2);
plot(1:N, d, 1:N, y_NLMS, 1:N, e_NLMS); title('NLMS 期望信号 vs 滤波器输出');
xlabel('样本'); ylabel('振幅');
legend('期望信号 d', 'NLMS 滤波器输出 y','NLMS 误差信号 e');
grid on;
tightfig;
% 学习曲线
MSE = zeros(1,N);
mse = zeros(1,N);
for i = 1:N
    Werr = W_LMS(:,i)-h.';
    MSE(i) = Werr.'*R*Werr;
    Werr = W_NLMS(:,i)-h.';
    mse(i) = Werr.'*R*Werr;
end
figure;
subplot(2,1,1);plot(1:N,MSE,1:N,mse); title('MSE学习曲线');
xlabel('迭代次数n/samples');ylabel('MSE'); grid on;
legend('LMS','NLMS'); 
subplot(2,1,2); plot(h,'.-'); hold on; 
plot(inv(R)*P,'--');
plot(W_LMS(:,end),'--');plot(W_NLMS(:,end)); grid on;
legend('系统权重真值','Wiener-Hopf最优权重','LMS算法稳态权重','NLMS算法稳态权重');
tightfig;