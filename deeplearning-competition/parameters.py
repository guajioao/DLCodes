import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

print(net[2].state_dict())

########################################## 初始化参数函数
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01) # 对weight做均值为0，方差为0.01的初始化
        nn.init.zeros_(m.bias) # 将bias赋0

net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])
#####
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1) # 把所有weight固定为1
        nn.init.zeros_(m.bias)

net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])
#######
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
#####
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0]) #先输出初始化对象
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5 #保留绝对值大于5的权重，其他设成0

net.apply(my_init)
net[0].weight[:2]
######直接设置参数
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
################################################ 参数绑定
# 给共享层一个名称，以便可以引用它的参数
# net中两个shared共享权重，一个改变另一个也会改变
# 可用于不同网络之间共享参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])

