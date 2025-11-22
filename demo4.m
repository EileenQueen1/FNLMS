clear; close all; clc;
%% 问题初始化
A = randn(2);
A = A'*A;
R = A + 0.01*eye(2)
% 设计零均值高斯随机噪声x[n],主噪声d[n]
Nstr = 1000;
k = 1:Nstr;
muGauss = [0,0]; % 与噪声d(n)的设计相关
noise = mvnrnd(muGauss, R, Nstr);
x = noise(:,1);
SNR = [0.1 1 10 50];
W = 20*pi;
d = sqrt(SNR(2))*sin(W*k) + x.';
H = [1,0]; % 上述设计中次级通路FIR滤波器的系数
% 计算 mse = w^T * R * w
[w1, w2] = meshgrid(-1.2:0.01:5, -1.2:0.01:5);
mse = R(1,1)*(w1-H(1)).^2 + (R(1,2) + R(2,1))*(w1-H(1)).*(w2-H(2)) + R(2,2)*(w2-H(2)).^2;
levels = linspace(min(mse(:)), max(mse(:)), 30);
for i = 1:2
    if i == 1
        %% LMS算法收敛
        mu = 5e-2; W0 = [-1 -1];
        [y,e,wgt,R1,P1] = simLMS(x,d,mu,W0);
    else
        %% VSLMS算法收敛
        mu_max = .1/max(eig(R));
        mu_min = 5e-5;
        [y,e,Mu,wgt] = simVSLMS(x,d,W0,[mu_min,mu_max]);
    end
    MSE = zeros(1,Nstr);
    for i = 1:Nstr
        Werr = wgt(:,i)-H';
        MSE(i) = Werr.'*R*Werr;
    end
    % 绘制结果
    figure; subplot(2,2,1:2)
    plot(d); hold on; 
    plot(y); title('期望信号 vs 滤波器输出');
    plot(e); xlabel('样本'); ylabel('振幅'); 
    legend('期望信号 d', '滤波器输出 y','误差信号e');
    grid on;
    subplot(2,2,3);plot(1:Nstr,MSE); title('MSE学习曲线');
    axis([-10 500 -1 max(MSE)]); xlabel('迭代次数n/samples');ylabel('MSE'); grid on;
    % 绘制等高线图
    subplot(2,2,4);
    contour(w1, w2, mse, levels, 'LineWidth', 2);
    xlabel('w_1'); ylabel('w_2');
    title('MSE = w^T R w 的等高线图');
    colorbar; grid on; hold on;
    plot(W0(1), W0(2), 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');
    plot(H(1), H(2), 'bo', 'MarkerSize', 5, 'MarkerFaceColor', 'b');
    text(-1.1, -0.9, 'origin', 'FontSize', 12, 'Color', 'r');
    text(H(1)+0.1, H(2)+0.1, 'end', 'FontSize', 12, 'Color', 'b');
    plot([W0(1) wgt(1, :)], [W0(2) wgt(2, :)], '-k'); axis([-1.2 5 -1.2 5]);
    tightfig;
end
