import torch
from torch import nn
from torch.nn import functional as F

# 存入张量
x = torch.arange(4)
torch.save(x, 'x-file')
# 读出张量
x2 = torch.load('x-file')
print(x2)
# 存读一个张量列表
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print((x2, y2))
# 存读一个字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

####################################### 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
# 存储模型
torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())
# 验证两模型输入参数相同时计算结果相同
Y_clone = clone(X)
print(Y_clone == Y)
