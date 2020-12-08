#-*-coding:utf-8-*-

import sys

from torch.optim import optimizer
sys.path.append("..")
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mp3
from torch.autograd import Variable, grad
from utils import rosenBrock, draw2D

if __name__ == "__main__":
    use_sch = False
    x = Variable(torch.normal(
        torch.zeros(2), torch.FloatTensor([1, 1])) + 
        torch.FloatTensor([9, 9]), requires_grad = True
    )
    max_iter = 6000
    func_to_use = rosenBrock
    if use_sch:
        fig = plt.figure()
        ax = mp3.Axes3D(fig)
        cmap = plt.get_cmap('bwr')
        pos = []
        opt = optim.Adam([x, ], lr = 3e-1)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, 24, 1e-4)
        for i in range(max_iter):
            point = x.data.clone()
            pos.append(point.numpy())
            print("Iter %d with x = ", x.data)
            opt.zero_grad()
            y = func_to_use(x)
            y.backward(retain_graph = True)
            opt.step()
            sch.step()
        pos = np.array(pos)
        
        curve_vals = np.array([func_to_use(p) for p in pos])
        # print(pos)
        xs = pos[:, 0]
        ys = pos[:, 1]
        # print(xs.shape, ys.shape, curve_vals.shape)
        ax.plot3D(xs, ys, curve_vals, c = 'k', label = "Adam")
        ax.scatter3D(xs, ys, curve_vals, c = 'k', s = 10)
        xx, yy = np.meshgrid(np.linspace(0, 11, 60), np.linspace(0, 11, 60))
        dots = np.c_[xx.ravel(), yy.ravel()]
        vals = np.array([func_to_use(dot) for dot in dots])
        vals = vals.reshape(xx.shape)
        ax.plot_surface(xx, yy, vals, cmap = cmap, alpha = 0.4)
        plt.legend()
        plt.show()
    else:
        opt = optim.LBFGS([x, ], max_iter = 3000)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, verbose = True)
        # 如何输出过程？
        # 在闭包里写全局变量就行了，带了closure的传递都会导致一次step完成优化，之后就没有可以优化的东西了
        cnt = 0
        def closure():
            global cnt
            opt.zero_grad()
            y = func_to_use(x)
            y.backward(retain_graph = True)
            sch.step(y)         # 可能确实可以？
            print(sch._reduce_lr(cnt))
            cnt += 1
            return y
        print("Before iterations: x = ", x.data)
        opt.step(closure)
        print("After iterations: x = ", x.data)

