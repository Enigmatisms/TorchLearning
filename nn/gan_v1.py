#-*-coding:utf-8-*-
"""
    生成对抗网络实践
    由于自动编码器，比如VAE，是输入一个编码，输出对应编码的生成图片
    而VAE比普通编码器的优势在于，其有随机性
    那么如何才能知道自己生成的数据足够好呢？
    再构建一个二分类网络，用于判定数据是真实的还是生成的，通过此网络来指导编码器的优化
        使用MNIST进行手写数字风格生成（但是可能很难知道哪些是生成的，哪些是真实的）
    version 1 多层感知机作为网络生成器
    TODO: 有个奇怪的问题 batch 训练时的channels设置问题
"""

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
            nn.Conv2d(1, 8, 3, stride = 2, padding = 1),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 1, padding = 2),
            nn.LeakyReLU(),
            nn.AvgPool2d(2)
        )
        self.out = nn.Sequential(
            nn.Linear(16 * 7 * 7, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

class Gen(nn.Module):
    """
        图片向量生成器
        输入一个in_sz大小的一维正态分布随机数向量
        多层感知机生成
        输出一个大小为28 * 28的二维矩阵
    """
    def __init__(self, in_sz = 128, out_sz = 28):
        super().__init__()
        self.linMap = nn.Sequential(
            *Gen.block(in_size, 1024, False),
            *Gen.block(1024, 4096)
        )
        # linMap之后，view为 64 * 64
        self.downConv1 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 2, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        self.downConv2 = nn.Sequential(
            nn.Conv2d(1, 1, 5, 2, 2),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        self.downConv = nn.Sequential(      # padding 0, 32 -> 28
            nn.Conv2d(1, 1, 5, 1, 0), 
            nn.BatchNorm2d(1),
            nn.Tanh()      
        )

    @staticmethod
    def block(in_chans, out_chans, do_norm = True):
        bloc = [nn.Linear(in_chans, out_chans)]
        if do_norm == True:
            bloc.append(nn.BatchNorm1d(out_chans))
        bloc.append(nn.LeakyReLU(0.02))
        return bloc
    
    def forward(self, x):
        # print(x.shape)
        x = self.linMap(x)
        x = x.view(x.size(0), 1, 64, 64)
        # print(x.shape)
        x1 = self.downConv1(x)
        x2 = self.downConv2(x)
        x = x1 + x2
        x = self.downConv(x)
        return x

if __name__ == "__main__":
    if torch.cuda.is_available() == False:
        raise RuntimeError("Cuda is not available.")
    tf = transforms.ToTensor()
    batch_sz = 40               # batch大小
    in_size = 256              # 生成网络的输入向量大小
    train_iter = 12              # 训练迭代次数
    origin = datasets.MNIST("..\\data\\", True, transform = tf)
    origin_set = DataLoader(origin, batch_sz, shuffle = True)
    disc = Disc()
    generator = Gen(in_size, out_sz = 28)
    disc.cuda()
    generator.cuda()
    dopt = optim.Adam(disc.parameters(), lr = 1e-4)
    gopt = optim.Adam(generator.parameters(), lr = 2e-2)
    loss_func = nn.BCELoss()
    real_labels = Var(torch.ones((batch_sz, 1))).cuda()
    fake_labels = Var(torch.zeros((batch_sz, 1))).cuda()
    for k, (bx, _) in enumerate(origin_set):
        bx = bx.cuda()
        gen_loss = 0
        dis_loss = 0
        out = disc(bx)                                  # 分类器输出
        loss = loss_func(out, real_labels)     # 真实数据loss计算
        batch_sz = len(bx)
        false_batch = Var(torch.randn(batch_sz, in_size)).cuda()    # 生成batch_size个复合标准正态分布的向量
        # 判别器训练 (清楚地让分类器知道什么是真的，什么是假的)
        false_in = generator(false_batch)               # 使用生成器进行生成判别
        false_out = disc(false_in)                      # 生成数据判别
        # print("Shape of false in:", false_in.shape, ", shape of false out:", false_out.shape)
        loss += loss_func(false_out, fake_labels)   # 综合两方面的loss
        dis_loss = loss.data.item()
        dopt.zero_grad()
        loss.backward()
        dopt.step()
        for i in range(train_iter):                         # 对每一个batch都进行train_iter次训练
            # 生成器训练（生成器需要让分类器的结果尽可能为1）\
            false_batch = Var(torch.randn(batch_sz, in_size)).cuda()
            false_in = generator(false_batch)
            false_out = disc(false_in)          # 根据训练后的生成器重新计算真假分类结果
            loss = loss_func(false_out, real_labels) # 1是真实数据，生成网络需要调整至尽可能让分类器混淆
            if i ==  train_iter - 1:
                gen_loss = loss.data.item()
            gopt.zero_grad()
            loss.backward()
            gopt.step()
        
        print("Training batch number %d. Generator loss: %f, discriminator loss: %f"%(k, gen_loss, dis_loss))
    print("Training completed.")
    torch.save(disc, "..\\models\\gan_dis.pkl", )
    torch.save(generator, "..\\models\\gan_gen.pkl", )