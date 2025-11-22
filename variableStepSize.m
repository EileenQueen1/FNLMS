% 步长调整策略函数 (示例：基于误差的步长调整)
function Mu = variableStepSize(Mu,cmpSign,m0,m1,a)
    % 查找各行最大不变号的个数
    L = size(Mu,1);
    idx = zeros(L,1); 
    for i = 1:L
        if isempty(find(cmpSign(i,:)==0, 1))
            idx(i) = m1 + 1; % 为什么要+1？
        else
            idx(i) = find(cmpSign(i,:)==0, 1)-1;
        end
    end
    % 根据最大不变号个数来迭代步长
    p = -1*(idx<=m0-1)+(idx>=m1);
    Mu = diag(Mu*a.^p);
end