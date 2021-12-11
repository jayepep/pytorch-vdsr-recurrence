import torch
import torch.nn as nn
from math import sqrt

# 神经网络结构块
class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        # stride步长为1 padding填充为1 bias不添加偏置参数作为可学习参数
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        # 对从上层网络Conv2d中传递下来的tensor直接进行修改，inplace变量替换能够节省运算内存，不用多存储其他变量
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

# 主要网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # Conv2d中参数的初始化 normal高斯
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 3 3 64  最后一次3 3 1
                # print(m.kernel_size[0], m.kernel_size[1], m.out_channels)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        # Sequential一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out

if __name__ == '__main__':
    modeltest = Net()
    print(modeltest)