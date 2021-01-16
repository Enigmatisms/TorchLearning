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

"""
    简单的CNN网络只需要网络结构和前向传递函数
    注意，CIFAR10与MNIST的区别，首先是数据是3通道的，必须利用颜色信息
    卷积时，需要设置三通道卷积
    初步正确率只有61%，是网络结构设计的问题
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
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 36, 5, 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(                     # 本层将输出到全连接
            nn.Conv2d(36, 18, 5, 1, padding = 2),
            nn.BatchNorm2d(18),
            nn.LeakyReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(18 * 8 * 8, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Linear(96, 10)                           # 从96 到 10，完成最后的映射
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)                       # 将多维数组进行ravel
        return self.out(x)

if __name__ == "__main__":
    load = False
    tf = transforms.ToTensor()
    lrate = 1e-3
    batch_size = 50
    test_raw = datasets.CIFAR10(root = '..\\data', train = False, download = False, transform = tf)
    test_set = DataLoader(test_raw, batch_size = batch_size, shuffle = True)
    if torch.cuda.is_available() == False:
        raise RuntimeError("CUDA is not available.")
    if load == True:        # 从已经训练好的模型进行加载
        cnn = torch.load("..\\models\\cifar10.pkl")
    else:
        train_raw = datasets.CIFAR10(root = '..\\data', train = True, download = False, transform = tf)
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










        


