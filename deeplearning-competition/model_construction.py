import torch
from torch import nn
from torch.nn import functional as F

X = torch.rand(2, 20)

# 使用封装类
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)

# 自定义块
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20.256)
        self.out = nn.Linear(256, 10)

    # 定义模型的前向计算
    def forward(self, X):
        # 1.将输入进入隐藏层中得到输出
        # 2.调用relu函数激活
        # 3.最后传入输出层
        return self.out(F.relu(self.hidden(X)))

# 自定义顺序块
class MySequential(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
    
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

# 使用自定义块
net = MLP()
# 使用自定义顺序块
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# 使用封装类
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

net(X)

# 自定义隐藏层
'''
    在该隐藏层中：
    1.将权重随机初始化，之后该矩阵参数不更新，保持不变
    2.？
    3.运行while循环，如果L1范数（绝对值之和）大于1，则除以2，直到输出向量X满足条件
    4.输出X所有项之和

    问题：代码中两个linear的作用是？
'''
class FixedHiddenMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 不计算梯度的随机权重参数，在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), required_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self, X):
        X = self.linear(X)
        # 手写X与随机权重参数做矩阵乘法并加一，即wx+b
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 再次调用linear函数->?
        # 复用全连接层。这相当于两个全连接层共享参数（？）
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

