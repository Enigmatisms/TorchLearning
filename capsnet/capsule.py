#-*-coding:utf-8-*-

"""
    这篇论文复现不太成功。对torch的运用不够熟练，主要问题在：
    - 如果只是逻辑性算法的话，实现起来还是较为简单的，但是涉及到矩阵运算，需要充分利用大矩阵运算比分块小矩阵运算快的特点
    - 不知道如何利用torch变换shape，高维矩阵以及高维矩阵的运算把我搞晕了
    胶囊网络论文：*Dynamic Routing Between Capsules* 复现的尝试（v1版本）（失败，需要深入学习）
    MNIST单数字 / 双重叠数字分类，几个问题：
        1. 动态路由的实现（方法偏数学，不偏逻辑）
        2. 网络结构的复现（论文里的胶囊网络结构不算复杂，也不深）
        3. 两个loss的定义
        4. 向量输出的卷积？
            此处的实现应该是：使用list保存多个Conv2d（每个Conv2d输出capsule向量的其中一个维度）
            实现还是有一定难度的
        5. DigitCaps，decoder对应的reconstruction loss
        6. 从PrimaryCaps到digitCap的映射W如何实现？
"""

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable as Var
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torch.nn import functional as F
from sys import argv

default_kernel_size = 9
discriminative_number = 10

class Capsule(nn.Module):
    """
        胶囊网络的胶囊层定义
        ---
        - `in_chan` 输入通道数
        - `out_chan` 输出通道数
        - `ndims` 向量神经网络输出的向量维数
        - `stride` 步长
        - `caps_below` 上级胶囊层的输出？如果没有上一级胶囊层（比如Conv1到PrimaryCaps），就为None，此时不需要进行routing
        ---
        困扰我的主要是$W_{ij}$和动态路由的实现，个人理解的动态路由算法流程大概是这样的：
        1. 初始化$b_{ij}$（所有的logits）
        2. 由于第k层capsules会输出一个向量（这就是本层的输出向量）$u_{ij}$
        3. k+1层的胶囊输入为$W_{ij}\times u_{ij}$（输出左乘一个weight matrix）
        4. 使用logit 的$b_{ij}$ softmax转化为概率后($c_{ij}$)再加权得到k+1层某个胶囊的输入向量
        5. 使用squash函数归一化这个向量
        6. 更新每个$b_{ij}$（使用点积）
        7. 论文中的primary capsule中cap数量为32 * 36
        8. 动态路由是可导的，设计出的计算逻辑最后需要综合reconstruction error进行反向传播
    """
    def __init__(self, in_chan, out_chan, ndims = 8, stride = 1, padding = 0, caps_below = None):
        super().__init__()
  
        # 卷积神经网络的组合，注意传统CNN是依靠一些非线性activation函数，而胶囊网络依靠动态路由
        # 此处构建了一个ModuleList，保存了ndims个卷积filter（论文中，PrimaryCaps ndims = 8）
        self.caps_below = caps_below
        self.layers = nn.ModuleList([
            nn.Conv2d(in_chan, out_chan, default_kernel_size, stride = stride, padding = padding) for _ in range(ndims)
        ])
        if not caps_below is None:
            # 由于上层PrimaryCaps输出的是(n, 1152, 8)，digitCaps需要输入(n, 1152)个channel8的向量
            self.weight_matrix = nn.Parameter((discriminative_number, caps_below, in_chan, out_chan))
            self.softmax = nn.Softmax()

    # 这是完全不知道为什么要这样写的一个函数
    @staticmethod
    def softmax(input, dim=1):
        # from gram-ai/capsule-networks
        transposed_input = input.transpose(dim, len(input.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

    """
        此处实现的时候没有经验
    """
    @staticmethod
    def squash(x):
        length2 = (x ** 2).sum(dim = -1, keepdim = True)    # 输入一个(n, 1152, 8)，输出(n, 1152, 1)
        return length2 / (1 + length2) * x / torch.sqrt(length2)

    # 对Capsule的
    """
        此处应该实现dynamic routing，应该是这样的：
        - 存在dynamic routing的Capsule层，self.caps_below必然是存在的（非None）
            - 此情况下就需要使用Logit按dynamic routing算法进行计算（代替激活函数）
        - 如果caps_below = None，比如说PrimaryCaps（没有dynamic routing）
            - 直接过网络生成向量输出即可
    """
    def forward(self, x):
        if not self.caps_below is None:     # 是否是连续的capsule层
            # 动态路由在此处实现，如果不是None，caps_below是上一层的输出大小，应该为32 * 36
            # 上一层胶囊的输出全部需要输入到本层的每个胶囊
            # 比如上一层输出：(n, 1152, 8) 到本层应该变为（n, 10, 16)
            # weight_matrix本身的维度应该是：(10, 32 * 36, 8, 16), x的输入维度为（n, 32 * 36, 8）
            # 所以x需要加两个维度(10, n, 32 * 36, 1, 8)，weight_matrix需要加1个维度(10, n, 32 * 36, 8, 16)
            # ，则先验输出为(10, n, 32 * 36, 1, 16)
            prior = x[None, :, :, None, :] @ self.weight_matrix[:, None, :, :, :]
            bijs = Var(torch.zeros_like(prior)).cuda()      # logit
            for i in range(3):                              # 3 logit迭代
                cijs = Capsule.softmax(bijs, dim = 2)       # 对 32 * 36 求softmax
                # 输出是(10, n, 1, 1, 16)
                sjs = (cijs * prior).sum(dim = 2, keepdim = True)   # 对所有32 * 36进行求和
                out = Capsule.squash(sjs)
                if not i == 2:
                    bijs += (prior * out).sum(dim = -1, keepdim = True) # 16维的digitCap向量输出并不是内积结果，需要求和
        else:
            out = [cap(x).view(x.size(0), -1, 1) for cap in self.layers]
            out = torch.cat(out, dim = -1)
            out = self.squash(out)
        return out

class CapsNet(nn.Module):

    def __init__(self, size = 28):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, 9, stride = 1, padding = 0),
            nn.ReLU()
        )                       # 低级特征提取层

        self.pri_caps = Capsule(256, 32, ndims = 8, stride = 2)              # 低级特征组合胶囊
        self.digit_caps = Capsule(8, 16, ndims = 16, caps_below = 32 * 36)   # 高级特征合成数字

        self.decoder = nn.Sequential(
            nn.Linear(160, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y = None):
        x = self.conv1(x)
        x = self.pri_caps(x)
        x = self.digit_caps(x).squeeze().transpose(0, 1)
        # 输出的x shape应该是 (n, 10, 16)
        # 根据capsuleNet的网络原理，x的长度是分类概率
        possi = torch.sqrt((x ** 2).sum(dim = -1))      # 输出为(n, 10, 1)
        possi = F.softmax(possi, dim = -1)              # 输出为(n, 10)
        # 需要取出batch中每个元素softmax最大对应的向量进行reconstruction
        if y is None:       # 提供了label (训练阶段）
            _, indices = possi.max(dim = 1)                 # 找batch中最大概率的一个
            y = Var(torch.eye(10).cuda().index_select(dim = 0, index = indices.data))
        tmp = x * y[:, :, None].view(x.size(0), -1)
        reconstructions = self.decoder(tmp)
        return possi, reconstructions


if __name__ == '__main__':
    pass