#-*-coding:utf-8-*-
"""
    使用WGAN的损失函数，由于WGAN_GP（梯度惩罚 WGAN）会存在梯度爆炸的问题，只简单尝试WGAN
"""

import time
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable as Var
from torch.nn.modules import loss
from torch.nn.modules.dropout import Dropout
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

# 生成器使用卷积上采样网络（下采样也可以尝试，但可能训练更慢）
# 7 * 7 * 4 = 196 （4通道 7 * 7 输入，上采样到 28 * 28 * 1)
class Generator(nn.Module):
    def __init__(self, in_feat = 196):
        super().__init__()
        
        def linBlock(in_dim, out_dim, norm = True, inplace = True):
            block = [nn.Linear(in_dim, out_dim)]
            if norm == True:
                block.append(nn.BatchNorm1d(out_dim))
            block.append(nn.LeakyReLU(0.1, inplace))
            return block
        # 输出 392 * 2 = 7 * 7 * 16
        self.linUpSamp = nn.Sequential(
            *linBlock(in_feat, 2 * in_feat, False),
            *linBlock(2 * in_feat, 4 * in_feat),
        )
        # 输出：7 * 7 * 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 8, 3, stride = 1, padding = 1),
            nn.LayerNorm([7, 7]),
            nn.LeakyReLU(0.1, True)
        )
        # 输出 14 * 14 * 4
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 4, 3, stride = 1, padding = 1),
            nn.LayerNorm([14, 14]),
            nn.LeakyReLU(0.1, True)
        )
        # 输出 28 * 28 * 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, 2, 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(2),
            nn.LeakyReLU(0.1, True)
        )
        # 输出 56 * 56 * 1
        self.conv4 = nn.Sequential(
            nn.Conv2d(2, 1, 3, stride = 1, padding = 1),
            nn.BatchNorm2d(1),
        )
        self.sampler = nn.Upsample(scale_factor = 2)
        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.linUpSamp(x)
        x = x.view(x.size(0), 16, 7, 7)
        x = self.conv1(x)
        x = self.sampler(x)
        x = self.conv2(x)
        x = self.sampler(x)
        x = self.conv3(x)
        x = self.sampler(x)
        x = self.conv4(x)
        return self.Tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 2, 3, stride = 1, padding = 1),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(0.1)
        )
        # 输入 28 * 28 * 2
        self.lin1 = nn.Sequential(
            nn.Linear(28 * 28 * 8, 28 * 28 * 2),
            nn.BatchNorm1d(28 * 28 * 2),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(0.1)
        )
        self.lin2 = nn.Sequential(
            nn.Linear(28 * 28 * 2, 49),
            nn.BatchNorm1d(49),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.1),
            nn.Linear(49, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        return self.lin2(x)

# 相比于普通GAN， WGAN只需要修改的是loss的计算以及优化器的优化策略（RMSProp）
if __name__ == "__main__":
    if torch.cuda.is_available() == False:
        raise RuntimeError("Cuda is not available.")
    batch_size = 50
    train_epoch = 60
    input_size = 196
    sample_time = 600
    # ============ WGAN 相关优化参数 =================
    n_critic = 3            # 每四次判别器优化后进行一次生成器优化
    crop = 0.02
    data_set = DataLoader(
        datasets.MNIST(
            "..\\data\\",
            train = True,
            download = False,
            transform = transforms.Compose(
                    [transforms.Resize((56, 56)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
        ),
        batch_size = batch_size,
        shuffle = True,
    )
    batch_number = data_set.__len__()
    G = Generator(input_size).cuda()
    D = Discriminator().cuda()
    gopt = optim.RMSprop(G.parameters(), lr = 5e-4)
    dopt = optim.RMSprop(D.parameters(), lr = 5e-4)
    real_labels = Var(torch.ones((batch_size, 1))).cuda()
    fake_labels = Var(torch.zeros((batch_size, 1))).cuda()
    train_cnt = 0
    last_time = 0
    this_time = 0
    time_init = False
    for epoch in range(train_epoch):
        for k, (bx, _) in enumerate(data_set):
            dopt.zero_grad()
            bx = Var(bx).cuda()
            # =============== 判别器训练 =================
            real_out = D(bx)

            fake_in = Var(torch.randn(batch_size, input_size)).cuda()
            fake_imgs = G(fake_in)
            fake_out = D(fake_imgs)
            d_loss = - torch.mean(real_out) + torch.mean(fake_out)
            d_loss.backward()
            dopt.step()

            for p in D.parameters():        # 截断处理
                p.data.clamp_(- crop, crop)
            # ================ 生成器 ===================
            if train_cnt % n_critic == 0:
                gopt.zero_grad()
                fake_in = Var(torch.randn(batch_size, input_size)).cuda()
                fake_imgs = G(fake_in)
                fake_out = D(fake_imgs)
                g_loss = - torch.mean(fake_out)
                g_loss.backward()
                gopt.step()
                print("Epoch %d/%d, batch %d/%d, [D loss: %f], [G loss: %f]"%(
                        epoch, train_epoch, k, batch_number,
                        d_loss.data.item(),
                        g_loss.data.item()
                    )
                )
            if k % sample_time == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" %(k + batch_number * epoch),
                     nrow = 5, normalize = True)
                if time_init == False:
                    time_init = True
                    last_time = time.time()
                else:
                    this_time = time.time()
                    interval = this_time - last_time
                    last_time = this_time
                    eta = (
                        batch_number / sample_time * (train_epoch - epoch) +    # 剩余epoch时间
                        (batch_number - k) / sample_time                        # 本epoch剩余时间
                    ) * interval
                    print("Time for %d batches: %f, ETA: %f s"%(
                        sample_time, interval,  eta
                    ))
            train_cnt += 1
    print("training completed.")
    torch.save(D, "..\\models\\WGAN_D_v3.pkl", )
    torch.save(G, "..\\models\\WGAN_G_v3.pkl", )



    
