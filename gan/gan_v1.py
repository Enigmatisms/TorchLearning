#-*-coding:utf-8-*-
"""
    生成对抗网络实践
    第一版本完全死亡，生成出来的就不是数字，就是噪声，很漂亮的雪花噪声
        GAN v1：雪花噪声生成网络
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
        输出一个大小为28 * 28的二维矩阵
    """
    def __init__(self, in_sz = 128, out_sz = 28):
        super().__init__()
        self.linMap = nn.Sequential(
            *Gen.block(in_size, 1024, False),
            *Gen.block(1024, 4096)
        )
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
        x = self.linMap(x)
        x = x.view(x.size(0), 1, 64, 64)
        x1 = self.downConv1(x)
        x2 = self.downConv2(x)
        x = x1 + x2
        x = self.downConv(x)
        return x

if __name__ == "__main__":
    if torch.cuda.is_available() == False:
        raise RuntimeError("Cuda is not available.")
    tf = transforms.ToTensor()
    batch_sz = 50                    # batch大小
    in_size = 128                   # 生成网络的输入向量大小
    train_iter = 12                 # 生成器训练迭代次数
    origin = datasets.MNIST("..\\data\\", True, transform = tf)
    origin_set = DataLoader(origin, batch_sz, shuffle = True)
    disc = Disc()
    generator = Gen(in_size, out_sz = 28)
    disc.cuda()
    generator.cuda()
    dopt = optim.Adam(disc.parameters(), lr = 2e-5)
    gopt = optim.Adam(generator.parameters(), lr = 2e-2)
    loss_func = nn.BCELoss()
    real_labels = Var(torch.ones((batch_sz, 1))).cuda()
    fake_labels = Var(torch.zeros((batch_sz, 1))).cuda()
    for k, (bx, _) in enumerate(origin_set):
        bx = bx.cuda()
        gen_loss = 0
        dis_loss = 0
        out = disc(bx)
        loss = loss_func(out, real_labels)
        batch_sz = len(bx)
        false_batch = Var(torch.randn(batch_sz, in_size)).cuda()
        # ================= 判别器训练 =====================
        false_in = generator(false_batch)
        false_out = disc(false_in)
        loss += loss_func(false_out, fake_labels)
        dis_loss = loss.data.item()
        dopt.zero_grad()
        loss.backward()
        dopt.step()
        # ================= 生成器训练 =====================
        for i in range(train_iter):
            false_batch = Var(torch.randn(batch_sz, in_size)).cuda()
            false_in = generator(false_batch)
            false_out = disc(false_in)
            loss = loss_func(false_out, real_labels)
            if i ==  train_iter - 1:
                gen_loss = loss.data.item()
            gopt.zero_grad()
            loss.backward()
            gopt.step()
        
        print("Training batch number %d. Generator loss: %f, discriminator loss: %f"%(k, gen_loss, dis_loss))
    print("Training completed.")
    torch.save(disc, "..\\models\\gan_dis.pkl", )
    torch.save(generator, "..\\models\\gan_gen.pkl", )