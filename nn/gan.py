#-*-coding:utf-8-*-
"""
    生成对抗网络实践
    由于自动编码器，比如VAE，是输入一个编码，输出对应编码的生成图片
    而VAE比普通编码器的优势在于，其有随机性
    那么如何才能知道自己生成的数据足够好呢？
    再构建一个二分类网络，用于判定数据是真实的还是生成的，通过此网络来指导编码器的优化
        使用MNIST进行手写数字风格生成（但是可能很难知道哪些是生成的，哪些是真实的）
"""

from ctypes import get_errno
from numpy.core.numeric import ones_like
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable as Var
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision import transforms

class Disc(nn.Module):
    """
        二分类判别器
        二分类判别器需要使用什么样的激活函数，网络结构需要是什么样的？
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, padding = 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 1, padding = 2),
            nn.LeakyReLU(),
            nn.AvgPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, padding = 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(8 * 7 * 7, 32),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

class Gen(nn.Module):
    """
        图片向量生成器
        forward的输入数据为一个正态分布的单个向量（较大的一维向量）
        forward输出是生成的数据
        in_size: 输入特征向量的维度
        out_size: 输出图片的宽度（默认方形图片）
        nchan: 第一层卷积原始通道数（二的幂次）
    """
    def __init__(self, in_size = 1024, out_size = 28, in_chan = 3, out_chan = 1):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.out_size = out_size
        self.linMap = nn.Sequential(
            nn.Linear(in_size, 2 * (out_size ** 2) * in_chan),
            nn.LeakyReLU()
        )
        self.convs = self.makeConvs()

    def makeConvs(self):
        chan = self.out_chan * (2 ** self.in_chan)
        convs = ([
            nn.Sequential(
                nn.Conv2d(int(chan), int(chan / 2), 3, 1, padding = 1),
                nn.LeakyReLU(),
                nn.AvgPool2d(2)
            )
        ])
        chan /= 2
        while chan > self.out_chan:
            convs.append(
                nn.Sequential(
                    nn.Conv2d(int(chan), int(chan / 2), 3, 1, padding = 1),
                    nn.BatchNorm2d(int(chan / 2)),
                    nn.LeakyReLU()
                )
            )
            chan /= 2
        return convs
           
    def forward(self, x):
        x = self.linMap(x)      # 线性映射
        x = x.view(x.size(0), 1, 2 * self.out_size, 2 * self.out_size)
        for conv in self.convs:
            x = conv(x)
        return x

if __name__ == "__main__":
    if torch.cuda.is_available() == False:
        raise RuntimeError("Cuda is not available.")
    tf = transforms.ToTensor()
    batch_sz = 50               # batch大小
    in_size = 1024              # 生成网络的输入向量大小
    in_chan = 3                 # 生成网络层数幂次
    train_iter = 3              # 训练迭代次数
    origin = datasets.MNIST("..\\data\\", True, transform = tf)
    origin_set = DataLoader(origin, batch_sz, shuffle = True)
    disc = Disc()
    generator = Gen(in_size, in_chan = in_chan)
    disc.cuda()
    generator.cuda()
    dopt = optim.Adam(disc.parameters(), lr = 1e-3)
    gopt = optim.Adam(generator.parameters(), lr = 1e-3)
    loss_func = nn.BCELoss()
    for k, (bx, _) in enumerate(origin_set):
        bx = bx.cuda()
        gen_loss = 0
        dis_loss = 0
        for i in range(train_iter):                         # 对每一个batch都进行train_iter次训练
            out = disc(bx)                                  # 分类器输出
            loss = loss_func(out, torch.ones_like(out).cuda())     # 真实数据loss计算
            batch_sz = len(bx)
            false_batch = torch.randn(batch_sz, in_size).cuda()    # 生成batch_size个复合标准正态分布的向量
            # 判别器训练 (清楚地让分类器知道什么是真的，什么是假的)
            false_in = generator(false_batch)               # 使用生成器进行生成判别
            false_out = disc(false_in)                      # 生成数据判别
            loss += loss_func(false_out, torch.zeros_like(false_out).cuda())   # 综合两方面的loss
            if i == train_iter - 1:
                dis_loss = loss.data.item()
            dopt.zero_grad()
            loss.backwards()
            dopt.step()

            # 生成器训练（生成器需要让分类器的结果尽可能为1）
            false_out = disc(false_in)          # 根据训练后的生成器重新计算真假分类结果
            loss = loss_func(false_out, torch.ones_like(false_out).cuda()) # 1是真实数据，生成网络需要调整至尽可能让分类器混淆
            if i ==  train_iter - 1:
                gen_loss = loss.data.item()
            gopt.zero_grad()
            loss.backwards()
            gopt.step()
        
        print("Training batch number %d. Generator loss: %f, discriminator loss: %f", k, gen_loss, dis_loss)
    print("Training completed.")
    torch.save("..\\models\\gan_dis.pkl", disc)
    torch.save("..\\models\\gan_gen.pkl", generator)