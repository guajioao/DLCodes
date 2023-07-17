"""丢弃法（在层之间加入噪音）
    目前最主流的对于多层感知机的控制方法
"""
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 所有元素都被保留
    if dropout == 0:
        return X
    # randn正态分布，所以可能出现>1的情况
    # rand均匀分布
    # mask = (torch.randn(X.shape) > dropout).float()
    mask = (torch.rand(X.shape) > dropout).float()
    # 之所以不用X[mask] = 0 是因为计算乘法比选择元素更快
    return mask * X / (1.0 - dropout)

X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X) # 原数组
print(dropout_layer(X, 0.)) # 完全不丢
print(dropout_layer(X, 0.5)) # 随机
print(dropout_layer(X, 1.)) # 全丢

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5

# 从零实现
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training == True: # 是在训练的话要dropout
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
# 训练
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


# 简洁实现
net = nn.Sequential(
    nn.Flatten(),nn.Linear(784, 256), nn.ReLU(),
    # 在第一个全连接层后添加一个dropout层
    nn.Dropout(dropout1), nn.Linear(256,256), nn.ReLU(),
    # 在第一个全连接层后添加一个dropout层
    nn.Dropout(dropout1), nn.Linear(256,256), nn.ReLU(),
    # 在第二个全连接层后添加一个dropout层
    nn.Dropout(dropout2), nn.Linear(256,10)
)
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weight)
# 训练
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)