"""多层感知机从零开始实现"""
import torch
from torch import nn
from d2l import torch as d2l

# 继续使用Fashion-MNIST图像分类数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
# n,m,k(28*28的图片，10个分类，隐藏层大小)
num_inputs, num_outputs, num_hiddens = 784, 10, 256

#对每一层记录一个权重矩阵和偏置向量
# nn.Parameter:可以不加，是一个声明
# w设置为随机，可以尝试全0、全1的效果
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01) # n*k大小
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True)) # k
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)# k*m
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True)) # m

params = [W1,b1,W2,b2]

def relu(X):
    a = torch.zeros_like(X) # 数据类型和形状都一样但元素为0的矩阵
    return torch.max(X,a) # 小于0的数全为0，大于0则为其本身 max(X,0)

def net(X):
    X = X.reshape((-1,num_inputs))
    H = relu(X @ W1 + b1) # @矩阵乘法
    return (H @ W2 + b2)

loss = nn.CrossEntropyLoss()

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.predict_ch3(net, test_iter)







