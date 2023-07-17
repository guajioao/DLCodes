"""线性回归简洁实现"""
import random
import torch
from torch.utils import data
from d2l import torch as d2l


# 生成数据集
def synthetic_data(w, b, num_examples):
    ### 生成人造数据集 y=Xw+b+噪声
    X = torch.normal(0, 1, (num_examples, len(w))) # 均值，方差，（样本个数n，列数）
    y = torch.matmul(X,w) + b # x*w+b
    y += torch.normal(0, 0.01, y.shape) # 均值为0，方差为0.01，形状与y相同的噪声
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

# 使用框架预定义好的层
from torch import nn
"""
Sequential: layer容器
Linear: 线性层(全连接层) 输入为2,输出为1
"""
net = nn.Sequential(nn.Linear(2,1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01) # 使用正态分布替换模型的值
net[0].bias.data.fill_(0)

# 定义损失函数 计算均方误差使用的是MSELoss类，也称为平方𝐿2范数
# 默认情况下，它返回所有样本损失的平均值
# loss = nn.HuberLoss()
loss = nn.MSELoss()
# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward() # pytorch自动做sum
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

