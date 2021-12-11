import torch
"""
在这里测试了一下我一开始训练的模型和原坐着给的模型是否一样，这样跑的话没有区别，因为都只是输出了一下每一层是什么及参数是什么
"""
model_last = torch.load("model_last/model_epoch_50.pth")
print(model_last)

model_new = torch.load("model/model_epoch_50.pth")
print(model_new)
