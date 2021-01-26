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
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(0.1, True)
        )
        # 输出 14 * 14 * 4
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 4, 3, stride = 1, padding = 1),
            nn.InstanceNorm2d(4),
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
        self.conv5 = nn.Sequential(
            nn.Conv2d(2, 1, 5, stride = 1, padding = 2),
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
        x = self.conv4(x) + self.conv5(x)
        return self.Tanh(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 2, 3, stride = 2, padding = 1),
            nn.LeakyReLU(0.1, True),
            nn.Dropout2d(0.25)
        )
        # 输入 28 * 28 * 2
        self.lin1 = nn.Sequential(
            nn.Linear(28 * 28 * 2, 49),
            nn.BatchNorm1d(49),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.3)
        )
        self.lin2 = nn.Sequential(
            nn.Linear(49, 1),
            nn.Softmax()
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        return self.lin2(x)

if __name__ == "__main__":
    if torch.cuda.is_available() == False:
        raise RuntimeError("Cuda is not available.")
    batch_size = 100
    train_epoch = 30
    input_size = 196
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
    gopt = optim.Adam(G.parameters(), lr = 4e-3)
    dopt = optim.Adam(D.parameters(), lr = 5e-4)
    lossFunc = nn.BCELoss()
    real_labels = Var(torch.ones((batch_size, 1))).cuda()
    fake_labels = Var(torch.zeros((batch_size, 1))).cuda()

    for epoch in range(train_epoch):
        for k, (bx, _) in enumerate(data_set):
            dopt.zero_grad()
            bx = bx.cuda()
            # =============== 判别器训练 =================
            real_out = D(bx)
            d_loss_real = lossFunc(real_out, real_labels)

            fake_in = Var(torch.randn(batch_size, input_size)).cuda()
            fake_imgs = G(fake_in)
            fake_out = D(fake_imgs)
            d_loss_fake = lossFunc(fake_out, fake_labels)
            d_loss = d_loss_fake + d_loss_real
            d_loss.backward()
            dopt.step()
            # =============== 判别器acc计算 ==============
            pred_real = torch.round(real_out)
            acc_cnt = (pred_real == real_labels).sum()

            pred_fake = torch.round(fake_out)
            acc_cnt += (pred_fake == fake_labels).sum()
            # ================ 生成器 ===================
            gopt.zero_grad()
            fake_in = Var(torch.randn(batch_size, input_size)).cuda()
            fake_imgs = G(fake_in)
            fake_out = D(fake_imgs)
            g_loss = lossFunc(fake_out, real_labels)
            g_loss.backward()
            gopt.step()
            print("Epoch %d/%d, batch %d/%d, [D loss: %f, acc: %f], [G loss: %f]"%(
                    epoch, train_epoch, k, batch_number,
                    d_loss.data.item(),
                    acc_cnt / (2 * batch_size),
                    g_loss.data.item()
                )
            )
            if k % 200 == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" %(k + batch_number * epoch), nrow = 5, normalize = True)
    print("training completed.")
    torch.save(D, "..\\models\\GAN_D_v2.pkl", )
    torch.save(G, "..\\models\\GAN_G_v2.pkl", )



    
