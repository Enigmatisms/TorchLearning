#-*-coding:utf-8-*-
# CNN 应用简单复习 cifar10数据集的分类

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable as Var
from torch.nn.modules import Softmax
from torchvision import datasets
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.transforms.transforms import ColorJitter, ToTensor

"""
    简单的CNN网络只需要网络结构和前向传递函数
    注意，CIFAR10与MNIST的区别，首先是数据是3通道的，必须利用颜色信息
    卷积时，需要设置三通道卷积
    1.16 初步正确率只有61%，是网络结构设计的问题
    1.17 做了数据增强，调整了网络结构（浅的类ResNet），73.27%
        非常奇异的是，训练集上的结果只有72.9%，测试集比训练集还高
"""
class CNN(nn.Module):       # 注意继承的是什么
    def __init__(self):
        super().__init__()
        """
            关于感受野，此处需要说明的是：
                第一层卷积输出的特征图，特征图上每一个点都是完全由本层大小的滤波器核产生
                第二层卷积输出的特征图：相当于对第一层结果进行卷积，由级联效应，等效kernel size变大
                ...
                第n层卷积输出：kernel size被进一步放大
                对于stride而言也有类似的结果，也就是越浅的层输出的特征越原始，越局部
                而越深的层，输出的是越具有全局性的特征（因为感受野大（级联效应吧））
            感觉这根本谈不上什么设计
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 12, 3, 1, padding = 1),        # 先卷积，再对卷积结果进行非线性激活，最后降采样
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 36, 5, 1, padding = 2),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.conv3 = nn.Sequential(                     # 本层将输出到全连接
            nn.Conv2d(36, 72, 3, 1, padding = 1),
            nn.BatchNorm2d(72),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(                     # 本层将输出到全连接
            nn.Conv2d(72, 48, 5, 1, padding = 2),
            nn.BatchNorm2d(48),
            nn.LeakyReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(48, 36, 5, 1, padding = 2),
            nn.BatchNorm2d(36),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(                     # 本层将输出到全连接
            nn.Conv2d(36, 18, 3, 1, padding = 1),
            nn.BatchNorm2d(18),
            nn.LeakyReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(18 * 8 * 8, 72),
            nn.BatchNorm1d(72),
            nn.ReLU(),
            nn.Linear(72, 10)                           # 从96 到 10，完成最后的映射
        )

        self.inner1 = nn.Sequential(
            nn.Conv2d(3, 72, 3, 2, padding = 1),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.inner2 = nn.Sequential(
            nn.Conv2d(72, 18, 3, 1, padding = 1),
            nn.BatchNorm2d(18),
            nn.ReLU()
        )
    def forward(self, x):
        tmp = self.inner1(x)                    # 一种浅层的类ResNet结构，有相对直接的数据通路
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += tmp
        tmp = self.inner2(x)
        x = self.conv4(x)                         # 跨层的传输
        x = self.conv5(x)
        x = self.conv6(x)
        x += tmp
        x = x.view(x.size(0), -1)                       # 将多维数组进行ravel
        return self.out(x)

def makeTransforms():
    return ([
        transforms.Compose([
            transforms.RandomHorizontalFlip(1),
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.RandomVerticalFlip(1),
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.ColorJitter(brightness = 0.5),
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.ColorJitter(brightness = 1.5),
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.ColorJitter(contrast = 0.5),
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.ColorJitter(contrast = 1.5),
            transforms.ToTensor()
        ])
    ])


if __name__ == "__main__":
    load = True
    train_test = True
    tfs = makeTransforms()
    lrate = 3e-3
    batch_size = 50
    test_raw = datasets.CIFAR10(root = '..\\data', train = False,
            download = False, transform =  transforms.ToTensor())
    test_set = DataLoader(test_raw, batch_size = batch_size, shuffle = True)
    if torch.cuda.is_available() == False:
        raise RuntimeError("CUDA is not available.")
    if load == True:        # 从已经训练好的模型进行加载
        cnn = torch.load("..\\models\\cifar10.pkl")
    else:
        train_raw = datasets.CIFAR10(root = '..\\data', train = True,
                download = False, transform = transforms.ToTensor())
        for tf in tfs:
            tmp = datasets.CIFAR10(root = '..\\data', train = True, download = False, transform = tf)
            train_raw += tmp
        train_set = DataLoader(train_raw, batch_size = batch_size, shuffle = True)

        cnn = CNN()
        cnn.cuda()
        opt = optim.Adam(cnn.parameters(), lr = lrate)
        loss_func = nn.CrossEntropyLoss()

        for k, (bx, by) in enumerate(train_set):
            _bx = bx.cuda()
            _by = by.cuda()
            out = cnn(_bx)
            loss = loss_func(out, _by)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if k % batch_size == 0:             # mini_batch 每一个batch训练结束，进行一次输出
                print("Epoch: %d, loss: %f"%(k, loss.data.item()))
            
        torch.save(cnn, "..\\models\\cifar10.pkl")

    if train_test == True:
        test_raw = datasets.CIFAR10(root = '..\\data', train = True,
                download = False, transform = transforms.ToTensor())
        for tf in tfs:
            tmp = datasets.CIFAR10(root = '..\\data', train = True, download = False, transform = tf)
            test_raw += tmp
        test_set = DataLoader(test_raw, batch_size = batch_size, shuffle = True)
    
    total_rights = 0
    for (bx, by) in test_set:       # 注意，test_set也是分batch的
        _bx = bx.cuda()
        _by = by.cuda()
        out = cnn(_bx)

        _, pred = torch.max(out, 1)
        rights = (pred == _by).sum()
        total_rights += rights.item()

    total_samples = len(test_set) * batch_size
    print("Total %d samples."%(total_samples))
    print("Correction ratio: %f"%(total_rights / total_samples))










        


