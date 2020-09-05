#!/usr/bin/env python3
#-*-coding:utf-8-*-
# optimizer 测试
# GPU 占用率为 0.3Gb

import numpy as np
import torch as t
from torch.optim import Optimizer
from torch.autograd import Variable as Var
from torch import nn
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

t.set_default_tensor_type(t.FloatTensor)

class Polyfit(nn.Module):
    """
        模型参数被隐式地保存在了func中
    """
    def __init__(self):
        super(Polyfit, self).__init__()
        self.func = nn.Linear(3, 1)

    # 前向传播（直接计算值）
    def forward(self, x):
        return self.func(x)

# 需要拟合的函数确定
def funcToFit(x):
    if type(x) == t.Tensor:
        coeffs = t.FloatTensor([1.2, 3.4, -0.5]).unsqueeze(1)           # 维度转换：行向量变列向量
        intersect = t.FloatTensor([2.3])
    else:
        coeffs = np.array([1.2, 3.4, -0.5]).reshape(-1, 1)
        intersect = 2.3
    return intersect + x @ coeffs                                       # 最后输出的是一个列向量

def wrapUp(x):
    _x = x.unsqueeze(1)     # 转置操作
    return t.cat([_x**(4 - i) for i in range(1, 4)], 1)    # 拼接成 n * 3 矩阵

# 为了送入线性模型中，矩阵一行为三个变量的线性组合，列为不同的x数据点
# 可能intersect被隐式地包含在了线性模型中吧
def getSamples(size = 200):
    x_data = t.linspace(-5, 5, size)
    noise = t.randn(size)       
    x = Var(wrapUp(x_data))                            # 创建变量x， x是三部分组成的（x, x^2, x^3）
    # print(x)
    y_truth = funcToFit(x)            
    y = Var(y_truth + 2 * noise.unsqueeze(1))           # 统一一下维度 列向量 加 行向量会成矩阵的
    return x.cuda(), y.cuda(), x_data                    # 返回x , y (含噪声的数据), x_data (真实值)

fit = Polyfit().cuda()
crit = nn.MSELoss()                                                 # 均方误差
opt = t.optim.SGD(fit.parameters(), lr = 0.0003, momentum = 0.8)     # 随机梯度下降

if __name__ == "__main__":
    # 一次拟合 得到参数
    xs, ys, truth = getSamples()
    truth = truth.numpy()
    ys_truth = funcToFit(xs.cpu()).data.numpy()

    max_iter = 4000
    minimum = 3.5
    old_loss = 0
    for i in range(max_iter):
        out = fit.forward(xs)       # 使用更新后的参数进行前向传播以计算损失

        loss = crit(out, ys)        # 损失函数创建

        opt.zero_grad()             # 梯度清零
        loss.backward()             # 反向传播，计算梯度
        opt.step()                  # 随机梯度下降更新参数

        d_loss = loss.data.item() - old_loss
        print("Iter (%d / %d), Loss: %f"%(i, max_iter, loss.data.item()))
        # 这倒是挺奇怪的，loss 这个 nn.MSELoss() 是如何与模型练习在一起的？
        # 个人的理解是 forward 由于其中存在一个Linear操作，估计是将Linear内部参数与xs相乘取出了，loss在backward处对参数的梯度进行更新
        # 而opt 这个Optimizer 开始就接受了fit.parameters 已经包含了与模型参数的联系
        old_loss = loss.data.item()
        if loss.data.item() < minimum or abs(d_loss) < 10e-4:
            print("Convergence before max iter (%d / %d) with delta_loss %f"%(i, max_iter, abs(d_loss)))
            break
    params = list(fit.parameters())

    x_coeffs = params[0].data.cpu().numpy()[0]      # 取出 data 转到host上 转为numpy 结构后取出第一位（为什么这么麻烦啊？）
    inter = params[1].data.cpu().numpy()

    result = np.concatenate((x_coeffs, inter))

    func = np.poly1d(result)
    ys_pre = func(truth)

    plt.plot(truth, ys_truth, c = "blue", label = "ground truth")
    plt.scatter(truth, ys.data.cpu().numpy(), c = "green", label = "data scatters")
    plt.plot(truth, ys_pre, c = "red", label = "Linear regression model")

    plt.legend()
    plt.show()


    





