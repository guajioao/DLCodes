import torch
from torch import nn
from torch.nn import functional as F

########################################### 自定义层
class CenteredLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return X - X.mean()
    
layer = CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))
# 将层作为组件参加网络
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(Y.mean())

# 带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
# 实例化
linear = MyLinear(5, 3)
linear.weight
# 使用自定义层直接执行前向传播计算
linear(torch.rand(2, 5))
# 使用自定义层构建模型
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))