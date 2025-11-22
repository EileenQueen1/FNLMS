function [y,e,Mu,wgt] = simVSLMS(x,d,W0,mu0)
%     error(nargchk(3,5,nargin));
%     if nargin > 3 % 处理参数个数较多的情况
%         if strcmp(varargin{1}, 'USERPAR')
%             userpar = varargin{2};
%         else
%             disp('输入参数可能存在错误');
%         end
%     end
    a = 2;
    m0 = 2;
    m1 = 3;
    L = length(W0);
    W = reshape(W0,[L,1]);
    mu_min = mu0(1);
    mu_max = mu0(2);
    y = zeros(1,length(x));
    e = zeros(1,length(x));
    u = zeros(L,1); % input_sav vector
    wgt = zeros(L,length(x));         % 初始化权重记录
    Mu = diag(mu_max.*ones(L,1));
    gradSign = 2*ones(L,m1+1);
    for k = 1 : length(x)
        u = [x(k)'; u(1:L-1)];
        y(k) = W' * u;
        e(k) = d(k) - y(k);
        gradSign = [sign(-2*e(k)*u) gradSign(:,1:end-1)];
        cmpSign = gradSign + gradSign(:,1);
        Mu = variableStepSize(Mu,cmpSign,m0,m1,a); % 步长参数迭代
        Mu(Mu>mu_max) = mu_max; % 步长参数饱和截止
        Mu(Mu<mu_min) = mu_min;
        W = W + 2*e(k)*Mu*u;
        wgt(:,k) = W;
    end
end