"""softmax回归的从零开始实现"""
import torch
from IPython import display
from d2l import torch as d2l
import matplotlib.pyplot as plt

batch_size = 256
# Fashion-MNIST数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) # 生成训练集和测试集的迭代器

# 图像大小为28*28，展平后视为长度为784的向量
# 有十个分类，故输出结果数量为10
num_inputs = 784
num_outputs = 10

# 权重为784*10的矩阵，初始化为正态分布
# 偏置为1*10的行向量，初始化为0
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 定义softmax操作
X = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

# 按0维求和即对列求和，与按1维求和即对行求和
# keepdim:保持维度，求和完还是二维矩阵
print(X.sum(0, keepdim=True))
# 输出：[[5, 7, 9]]
print(X.sum(1, keepdim=True))
# 输出：[[6], [15]]


"""
    实现softmax:
    1.对每一项求幂(exp)
    2.对每一行求和
    3.每一行除以所有项求幂之和，保证结果的和为1

    缺少防止上溢和下溢的措施
"""
def softmax(X):
    X_exp = torch.exp(X)
    # tensor([
    # [0.8520, 1.5417, 1.9524, 0.2051, 0.5953],
    # [0.2025, 0.3661, 0.8373, 0.2364, 1.4421]])
    partition = X_exp.sum(1, keepdim=True)
    # tensor([[5.1465], [3.0844]])
    return X_exp / partition # 广播

X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob) 
print(X_prob.sum(1))
# tensor([
#     [0.2968, 0.4115, 0.0945, 0.1603, 0.0368],
#     [0.2128, 0.5422, 0.0865, 0.1104, 0.0481]
# ]),
# tensor([1.0000, 1.0000])

# softmax回归模型
def net(X):
    # y = x*w + b
    # -1表示重新计算实际该批数据的大小，实际上除最后一批外=batch_size
    # 
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 交叉熵损失函数
# python的特殊用法 索引
y = torch.tensor([0,2])
y_hat = torch.tensor([
    [0.1,0.3,0.6],
    [0.3,0.2,0.5]
])
# 对于第0个样本(即第0行数据[0.1,0.3,0.6])，拿出下标为y[0]=0的预测值，即0.1
# 对于第1个样本(即第1行数据[0.3,0.2,0.5])，拿出下标为y[1]=2的预测值，即0.5
print(y_hat[[0, 1], y]) # = tensor([0.1000, 0.5000])

# 将上述用法用于损失函数中
def cross_entropy(y_hat, y):
    get_indics = range(len(y_hat)) # y_hat的下标顺序存入list中
    return - torch.log(y_hat[get_indics, y]) # 求ln再取负
print(cross_entropy(y_hat, y)) # = tensor([2.3026, 0.6931])

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

class Accumulator: # 累加器
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

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

# 定义⼀个在动画中绘制数据的实⽤程序类Animator
class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
            ylim=None, xscale='linear', yscale='linear',
            fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
            figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使⽤lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
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

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # 训练一圈
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # 测试误差
        test_acc = evaluate_accuracy(net, test_iter)
        # 画图
        animator.add(epoch+1, train_metrics + (test_acc, ))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    plt.show() # 加上图才会显示

def updater(batch_size):
    # 使用之前自定义的sgd方法（见scratch_all）
    return d2l.sgd([W, b], lr, batch_size) 

# 训练10个迭代周期
lr = 0.1
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
# plt.close()
def predict_ch3(net, test_iter, n=6): #@save
    """预测标签（定义⻅第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
    X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    predict_ch3(net, test_iter)
    # plt.show() # 加上图才会显示
 
predict_ch3(net, test_iter)