import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr import Net
# 数据加载代码 DatasetFromHdf5中
from dataset import DatasetFromHdf5
import time

"""
data/train.h5文件是示例文件，数据集需要自己生成 data/dataset_grnerate中有.m文件
model_last存放的是原作者的示例模型
model里存放的是我训练的模型，分别为50、53、59轮，测试效果差别不大
"""


# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
# 多了会更稳定 要求显存 batch_size一次处理的数量
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
# 梯度下降的轮次 迭代次数 epoch
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
# 学习率从大到小下降
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
# parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--cuda", default=True, help="Use cuda?")

parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
# SGD 动量
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
# 约束
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='none', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
"""
先定义一堆训练中需要的工具
"""
def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)
    cuda = opt.cuda
    if cuda:
        print("cuda")
        print("=> use gpu id: '{}'".format(opt.gpus))
        # 设定使用哪个GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    # Sets the seed for generating random numbers.
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
# 参数配置
# 让内置的cudnn的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率
    cudnn.benchmark = True
# 模型的学习基于模型与数据的交互
    print("===> Loading datasets")
    # 导入数据集
    train_set = DatasetFromHdf5("data/train.h5")
    """
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    nunum_workers=opt.thread 表示在加载训练集时使用了多线程 train的时候shuffle要打开
    """
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    # 输入数据到网络中
    model = Net()

    # 损失函数对结果进行度量 MSE求和
    criterion = nn.MSELoss(reduction='sum')

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint  从预训练模型中加载参数
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    else:
        print("opt.resume == none")
    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    # param_groups的各项参数：[{'params','lr', 'momentum', 'dampening', 'weight_decay', 'nesterov'},{……}]
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    # 接下来是训练部分 model.train()保证Batch Normalization层用每一批数据的均值和方差，对于Droupout:model.train()是随机取一部分网络连接来训练更新参数
    # model.eval()是保证Batch Normalization用全部训练数据的均值和方差，对于Droupout:model.eval()是利用到了所有网络连接
    model.train()
    # 遍历training_data_loader，从下标1开始
    for iteration, batch in enumerate(training_data_loader, 1):
        # running_loss = 0.0
        # Variable是一种可以不断变化的变量，符合反向传播，参数更新的属性
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model(input), target)

        if iteration % 100 == 0:
            output = "===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item())
            with open("loss.txt", "a+") as f:
                f.write(output + '\n')
                f.close

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        optimizer.step()

        if iteration%100 == 0:
          print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item()))

        # running_loss = running_loss + loss.item()
    # print(running_loss)

# 对模型参数保存 记录epoch 模型等信息
def save_checkpoint(model, epoch):
    model_out_path = "checkpoint/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()

