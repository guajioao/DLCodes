"""线性回归从零开始"""
import random
import torch
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
# 输出数据样本，标签
print('features:', features[0],'\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)


# 读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 生成随机读取的顺序列表
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size): # 从前到后，每次跳batchsize大小
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 初始化模型
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1,requires_grad=True)

# 定义模型
def linreg(X, w, b):
    """线性回归模型 y=wx+b"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失 L=(y_hat - y)平方 / 2"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 # ** 2


# 定义优化算法
def sgd(params, lr, batch_size): # 参数，学习率
    """小批量随机梯度下降"""
    with torch.no_grad():# 更新时不参与梯度计算
        for param in params:
            param -= lr * param.grad / batch_size # 损失函数没求均值，故在此/batch_size
            param.grad.zero_() # 梯度设置为0，防止下一次计算与上一次相关


# 训练过程
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        # net(X, w, b)做预测 -> y_hat
        # loss(y_hat, y) 
        l = loss(net(X, w, b), y) 
        # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起，并以此计算关于[w,b]的梯度
        l.sum().backward() # 损失求和后再算梯度
        sgd([w,b], lr, batch_size) # 用优化算法更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')













