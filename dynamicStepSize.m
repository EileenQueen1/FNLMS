% 步长调整策略函数 (示例：基于误差的步长调整)
function out = dynamicStepSize(u,W,i,y,e,mu,varargin)
    % 参数设置
    if nargin > 6
        var = varargin{1};
        alpha = var(1);  % 步长调整参数
        beta = var(2);    % 步长调整速率
    else
        alpha = 0.9;  % 步长调整参数
        beta = 0.9;    % 步长调整速率
    end
    if i == 1
        mu(i) = alpha;%why?
    else
        mu(i) = alpha*mu(i-1) + beta * e(i)*e(i-1)*u(1)*u(2);
    end
    mu(i) = max(mu(i), 0.0001); % 防止步长过小
    out = mu(i);
end