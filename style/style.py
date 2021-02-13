"""
    CycleGAN 复现，只测试简单的 橙子 - 苹果转换
    论文中使用了一些trciks，可能不会在此处被实现，本实现将在：
        - 实现CycleGAN基本思想
        - 加一些必要的tricks的情况上
    尽可能多花一下
    需要实现：
    - 两个结构等同的discriminator
    - 两个结构等同的generator
    - GAN Loss 基于
        - 最小二乘的对抗误差（非log，更稳定）
        - 循环一致误差
    CycleGAN -v1 致命的缺陷：训练极其慢 （1s才能处理完一张图像）
"""
from numpy.core.shape_base import hstack
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable as Var
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

tf = transforms.ToTensor()

def getLoader(_tf, path = "trainA\\", shuffle = False):
    folder = datasets.ImageFolder("..\\data\\apple2orange\\" + path, transform = _tf)
    loader = DataLoader(folder, shuffle = shuffle)
    return loader

# 卷积层输出
def makeConv(ic, oc, kernel, stride = 1, norm = 'ins', pad = 'ref', act = 'leaky'):
    padd = (int((kernel - 1) / 2), ) * 4
    net = []
    if pad == 'ref':
        net.append(nn.ReflectionPad2d(padd))
    else:
        net.append(nn.ZeroPad2d(padd))
    net.append(nn.Conv2d(ic, oc, kernel, stride = stride, padding = 0))
    
    if norm == 'ins':
        net.append(nn.InstanceNorm2d(oc))
    elif norm == 'batch':
        net.append(nn.BatchNorm2d(oc))
    if act == 'leaky':
        net.append(nn.LeakyReLU(0.15))
    elif act == 'relu':
        net.append(nn.ReLU())
    elif act == 'tanh':
        net.append(nn.Tanh())
    else:
        net.append(nn.Sigmoid())
    return net

def makeResBlock(in_chan):
    net = [nn.Conv2d(in_chan, in_chan, 3, stride = 1, padding = 1)]
    net.append(nn.InstanceNorm2d(in_chan))
    net.append(nn.LeakyReLU(0.2, True))
    net.append(nn.Conv2d(in_chan, in_chan, 3, stride = 1, padding = 1))
    net.append(nn.InstanceNorm2d(in_chan))
    net.append(nn.LeakyReLU(0.2, True))
    return net

"""
    分类器 PatchGANs PatchGAN分类器就是输出一个
    n * n 的Patch，论文中用了70 * 70 (再对输出取平均) 此处直接使用两次降采样 5 * 5 kernel
"""
class Discriminator(nn.Module):
    def __init__(self, sz = 256):
        super().__init__()
        self.size = sz

        # (256, 256, 3) -> (128, 128, 8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(8),
        )

        # (128, 128, 8) -> (64, 64, 16)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        
        # (64, 64, 16) -> (32, 32, 32)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 4, stride = 2, padding = 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, True)
        )

        # （32, 32, 32) -> (16, 16, 1)
        self.output = nn.Conv2d(32, 1, 4, stride = 2, padding = 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.output(x)

"""
    生成器如何进行设计？看了CycleGAN的网络架构，大概明白了
    对Apple以及Orange的图像不进行增强，对原始的网络进行一定的简化（MX150带不动）
    多个残差块可以进行简化：那么第一版的架构大概是这样的：
    - 7 * 7 卷积 + 步长1 reflect padding 输出32 channel
    - 3 * 3 步长2 reflect padding，变为 128 * 128 * 64
    - 3 * 3 步长2 reflect padding，变为64 * 64 * 128
    - 残差块1，3 * 3 zero padding 与原输入合并
    - 3 * 3 反卷积1，64 * 64 * 128 —> 128 * 128 * 64 (V1.0)
        - （v1.1）改为了upsample + 卷积
    - 3 * 3 反卷积2，128 * 128 * 64 -> 256 * 256 * 16
        - （v1.1）改为了upsample + 卷积，删除了一个res block
    - 7 * 7 卷积，通道为16 -> 3
    个人觉得这必定极其慢
"""
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(*makeConv(3, 32, 7, 1))
        self.conv1 = nn.Sequential(*makeConv(32, 64, 3, 2))
        self.conv2 = nn.Sequential(*makeConv(64, 128, 3, 2))
        self.res1 = nn.Sequential(*makeResBlock(128))
        self.res2 = nn.Sequential(*makeResBlock(128))
        self.inv1 = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(128, 64, 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        self.inv2 = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(64, 16, 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(16),
            nn.ReLU()
        )
        self.output = nn.Sequential(*makeConv(16, 3, 5, 1, norm = 'none', act = 'tanh'))
    
    def forward(self, x):
        x = self.input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.res1(x)
        x = x + self.res2(x)
        x = self.inv1(x)
        x = self.inv2(x)
        return self.output(x)

# Cycle Consistancy Loss
# difference between origin and cycle should be small
# 其实torch实现了这个，叫做L1Loss
def makeCycleLoss(origin, cycle):
    return torch.norm(origin.view(1, -1) - cycle.view(1, -1), 'fro', dim = 1)[0]

"""
    训练流程？对于G/F Dx，Dy的训练过程应该是什么样的？
    1. 高斯初始化网络参数
    2. 尝试与原始GAN使用类似的策略 训练n次分类器后训练一次生成器
    3. 由于有两个生成器和分类器（实际上是一样的，可以并行化（但是Python不支持并行））
        分类器训练时，Dx 用于判别 x 与 G(x)，Dy用于判定 y 与生成器F(y)
        生成器训练的目的是让分类器混淆，G(x)判定结果要接近1，同理F(y)也是
        但是CCL（Cycle Consistancy Loss）不知道要加在什么位置。可以在G/D两个训练时均加入
        注意：
            1. 使用最小二乘误差
            2. 使用history buffer的训练方式
                history buffer 先不考虑
            3. 可能要训练100个epoch，前60个恒定学习率，后40个线性下降至0
            4. batch size 为0
"""
if __name__ == "__main__":
    if torch.cuda.is_available() == False:
        raise RuntimeError("Cuda is not available")
    torch.autograd.set_detect_anomaly(True)
    tf = transforms.ToTensor()
    load = True
    g_lr = 1e-3
    d_lr = 1e-3
    cyc_cf = 10
    epoches = 2
    sign_epoch = 60     # significant epoch
    n_crit = 3          # train Generator every n_crit
    n_save_time = 20    # save imgs every n_save_time

    loadx = getLoader(tf)
    loady = getLoader(tf, "trainB\\")

    loader = zip(loadx, loady)
    G = None
    F = None
    Dx = None
    Dy = None
    if load == True:
        Dx = torch.load("..\\models\\Cycle_Dx_v1_2.pkl")
        Dy = torch.load("..\\models\\Cycle_Dy_v1_2.pkl")
        G = torch.load("..\\models\\Cycle_G_v1_2.pkl")
        F = torch.load("..\\models\\Cycle_F_v1_2.pkl")
        print("Model loaded.")
    else:
        G = Generator().cuda()     # 苹果->橙子
        F = Generator().cuda()     # 橙子->苹果
        Dx = Discriminator().cuda()    # 苹果判定器
        Dy = Discriminator().cuda()    # 橙子判定器


    g_opt = optim.Adam(G.parameters(), lr = g_lr)
    f_opt = optim.Adam(F.parameters(), lr = g_lr)
    dx_opt = optim.Adam(Dx.parameters(), lr = d_lr)
    dy_opt = optim.Adam(Dy.parameters(), lr = d_lr)
    
    loss_func_base = nn.MSELoss()
    loss_func_same = nn.L1Loss()        # 苹果经过F之后应该映射到原来的苹果，橙子同理
    loss_func_cyc = nn.L1Loss()
    real_lable = Var(torch.ones(1, 1, 16, 16)).cuda()
    fake_lable = Var(torch.zeros(1, 1, 16, 16)).cuda()

    print("Start training...")
    for epoch in range(epoches):
        # 重新shuffle训练
        loadx = getLoader(tf)
        loady = getLoader(tf, "trainB\\")
        loader = zip(loadx, loady)
        print("Epoch(%d/%d) Loader zipped."%(epoch, epoches))
        for i, raw in enumerate(loader):
            bx = Var(raw[0][0]).cuda()
            by = Var(raw[1][0]).cuda()
            # ============= 判别器训练 =================
            dx_out = Dx(bx)
            dy_out = Dy(by)

            G_out = G(bx)       # bx(apple)->G_out(orange)
            F_out = F(by)       # by(orange)->F_out(apple)

            G_res = Dy(G_out)   # Dy is orange D
            F_res = Dx(F_out)   # Dx is apple D

            G_cyc = F(G_out)
            F_cyc = G(F_out)
            
            dx_base_loss = loss_func_base(dx_out, real_lable)
            dx_base_loss += loss_func_base(G_res, fake_lable)

            dy_base_loss = loss_func_base(dy_out, real_lable)
            dy_base_loss += loss_func_base(F_res, fake_lable)

            dx_opt.zero_grad()
            dy_opt.zero_grad()

            dx_base_loss.backward()
            dy_base_loss.backward()

            dx_opt.step()
            dy_opt.step()
            # ============= 生成器训练 =================
            if i % n_crit == 0:
                G_out = G(bx)       # bx(apple)->G_out(orange)
                F_out = F(by)       # by(orange)->F_out(apple)
                G_res = Dy(G_out)   # Dy is orange D
                F_res = Dx(F_out)   # Dx is apple D
                G_cyc = F(G_out)
                F_cyc = G(F_out)
                g_loss = loss_func_base(G_res, real_lable)
                f_loss = loss_func_base(F_res, real_lable)
                cyc_loss = (loss_func_cyc(bx, G_cyc) + loss_func_cyc(by, F_cyc)) * cyc_cf
                g_loss += cyc_loss
                f_loss += cyc_loss
                g_loss += loss_func_same(F(bx), bx)
                f_loss += loss_func_same(G(by), by)

                g_opt.zero_grad()
                f_opt.zero_grad()

                g_loss.backward(retain_graph = True)
                f_loss.backward()

                g_opt.step()
                f_opt.step()
                print("Epoch: %d/%d, batch: %d/%d:"%(epoch, epoches, i, 995), end = '')
                print("Dx loss: %f, Dy loss: %f, G loss: %f, F loss: %f"%(
                    dx_base_loss.data.item(), dy_base_loss.data.item(),
                    g_loss.data.item(), f_loss.data.item()
                ))
            if i % n_save_time == 0:
                number = epoch * 995 + i
                g_to_save = torch.vstack((bx, G_out))
                f_to_save = torch.vstack((by, F_out))

                save_image(g_to_save, ".\\imgs\\G_%d.jpg"%(number), 1)
                save_image(f_to_save, ".\\imgs\\F_%d.jpg"%(number), 1)
    torch.save(Dx, "..\\models\\Cycle_Dx_v1_2.pkl", )
    torch.save(Dy, "..\\models\\Cycle_Dy_v1_2.pkl", )
    torch.save(G, "..\\models\\Cycle_G_v1_2.pkl", )
    torch.save(F, "..\\models\\Cycle_F_v1_2.pkl", )


    


