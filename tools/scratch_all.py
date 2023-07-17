"""所有从零开始"""
import random
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt
from IPython import display

# 定义模型
def linreg(X, w, b):
    """线性回归模型 y=wx+b"""
    return torch.matmul(X, w) + b

def softmax(X):
    X_exp = torch.exp(X)
    # tensor([
    # [0.8520, 1.5417, 1.9524, 0.2051, 0.5953],
    # [0.2025, 0.3661, 0.8373, 0.2364, 1.4421]])
    partition = X_exp.sum(1, keepdim=True)
    # tensor([[5.1465], [3.0844]])
    return X_exp / partition # 广播


# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失 L=(y_hat - y)平方 / 2"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 # ** 2

def cross_entropy(y_hat, y):
    get_indics = range(len(y_hat)) # y_hat的下标顺序存入list中
    return - torch.log(y_hat[get_indics, y]) # 求ln再取负


# 定义优化算法
def sgd(params, lr, batch_size): # 参数，学习率
    """小批量随机梯度下降"""
    with torch.no_grad():# 更新时不参与梯度计算
        for param in params:
            param -= lr * param.grad / batch_size # 损失函数没求均值，故在此/batch_size
            param.grad.zero_() # 梯度设置为0，防止下一次计算与上一次相关


# 准确度计算
# 分类精度计算
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: # 如果是二维以上的矩阵
        y_hat = y_hat.argmax(axis=1) # 求每一行最大值的下标
    # cmp为bool值的tensor
    cmp = y_hat.type(y.dtype) == y # 将y_hat转成与y同形状后再与y比较
    return float(cmp.type(y.dtype).sum())
def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    # 在累加器中创建了两个变量，分别存储正确预测的数量和总数量
    # 遍历数据集时，两者都将随着时间的推移⽽累加
    metric = Accumulator(2) 
    for X, y in data_iter:
        y_hat = net(X) # 在网络中算出评测值
        # accuracy(y_hat,y)：预估正确的样本数
        # y.numel(): 样本总数
        metric.add(accuracy(y_hat,y), y.numel())
    return metric[0] / metric[1] # 预估正确的样本数/样本总数


# 训练
def train_epoch_ch3(net, train_iter, loss, updater): # updater优化器，如sgd
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X) # 用网络计算预测值
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer): # 如果是封装的优化器
            updater.zero_grad() # 先把梯度设置为0
            l.backward() # 计算梯度
            updater.step() # 对参数进行一次更新
            metric.add(
                float(l) * len(y), accuracy(y_hat, y), y.size.numel()
            )
        else: # 自定义的优化器
            l.sum().backward() # l先求和再求梯度（因为l为向量）
            updater(X.shape[0]) # 根据批量大小放入updater更新
            metric.add(
                float(l.sum()), accuracy(y_hat, y), y.numel()
            )
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics + (test_acc, ))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    plt.show()

# 工具类

class Accumulator: # 累加器
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Animator: # 图像绘制
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, 
                 xlim=None, ylim=None, xscale='linear', yscale='linear', 
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplot(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使⽤lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, Y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)





