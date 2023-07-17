"""softmax回归的简洁实现"""
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
batch_size = 256
# Fashion-MNIST数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) # 生成训练集和测试集的迭代器

# 定义输出层（是一个全连接层）
# Flatten():把任何维度的tensor变成2维的，其中第0维保留，其余全部展成一个向量
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))# 再Sequential中添加一个带有10个输出的全连接层

def init_weights(m): # m为当前layer
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)# 将该层初始化为一个方差为0.01的随机值

net.apply(init_weights) # 在每一层做一次init操作

# 定义损失函数
loss = nn.CrossEntropyLoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()