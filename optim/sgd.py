#-*-coding:utf-8-*-

import sys

from torch.optim import optimizer
sys.path.append("..")
import torch
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mp3
from torch import optim
from torch.autograd import Variable, grad
from utils import rosenBrock, himmelblau, draw2D

optims = (optim.Adam, optim.Adagrad, optim.Adadelta, optim.SGD)
colors = ("green", "orange", "purple", "pink")
lrs = (2e-1, 1, 20, 1e-3)

if __name__ == "__main__":
    """
        Adadelta            # 下降出奇的慢
        SGD                 # SGD很容易在学习率很大时直接优化出nan
        Adagrad             # 下降第二慢（主要看学习率）
        Adam                # 还可以
        学习率需要整定
    """
    max_iter = 300
    func_to_use = himmelblau
    cmap = plt.get_cmap('bwr')
    fig = plt.figure()
    ax = mp3.Axes3D(fig)
    for i, opt in enumerate(optims):
        x = Variable(torch.normal(
            torch.zeros(2), torch.FloatTensor([1, 1])) + 
            torch.FloatTensor([9, 9]), requires_grad = True
        )
        optimizer = opt([x, ], lr = lrs[i])
        x_start = x.data.clone()
        pos = []
        for j in range(max_iter):
            point = x.data.clone()
            pos.append(point.numpy())
            y = func_to_use(x)
            y.backward(retain_graph = True)
            optimizer.step()            # 在不传入closure(某个可以自动绑定外部变量的函数)时只更新一步，传入closure更新多步
            # 如果没有closure，由于backward已经求出了x的梯度，则可以使用这个梯度进行线搜索得到步长，并移动x（更新）        
            optimizer.zero_grad()       # 一般的习惯是，先进行zero_grad，再重新开始计算
            # zero_grad的作用是，将之前的grad清零（否则会累加）
            # step的作用：

            print("Optimizer %s, iteration %d / %d, x = "%(opt.__name__, j, max_iter), x.data, "y = ", float(y.data))

        ## 绘图
        pos = np.array(pos)
        
        curve_vals = np.array([func_to_use(p) for p in pos])
        # print(pos)
        xs = pos[:, 0]
        ys = pos[:, 1]
        # print(xs.shape, ys.shape, curve_vals.shape)
        ax.plot3D(xs, ys, curve_vals, c = colors[i], label = opt.__name__)
        ax.scatter3D(xs, ys, curve_vals, c = colors[i], s = 10)
    xx, yy = np.meshgrid(np.linspace(0, 11, 60), np.linspace(0, 11, 60))
    dots = np.c_[xx.ravel(), yy.ravel()]
    vals = np.array([func_to_use(dot) for dot in dots])
    vals = vals.reshape(xx.shape)
    ax.plot_surface(xx, yy, vals, cmap = cmap, alpha = 0.4)
    plt.legend()
    plt.show()

    