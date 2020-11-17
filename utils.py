#-*-coding:utf-8-*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mp3

def rosenBrock(x, a = 10):
    return (1 - x[0]) ** 2 + a * (x[1] - x[0] ** 2) ** 2

def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

def draw2D(func, pos):
    xy = np.array(pos)
    # print(xy)
    xs = xy[:, 0]
    ys = xy[:, 1]
    xlb = min(xs)
    xub = max(xs)
    ylb = min(ys)
    yub = max(ys)
    x, y = pos[-1]
    x_range = np.linspace(min(int(x) - 5, xlb - 1), max(int(x) + 6, xub + 1), 200)
    xx, yy = np.meshgrid(x_range, np.linspace(min(int(y) - 5, ylb - 1), max(int(y) + 6, yub + 1), 200))
    dots = np.c_[xx.ravel(), yy.ravel()]
    dot_num = dots.shape[0]
    res = np.array([func(dots[i, :]) for i in range(dot_num)])
    res = res.reshape(xx.shape)
    cmap = plt.get_cmap('bwr')
    plt.contourf(xx, yy, res, cmap = cmap)
    plt.plot(xs, ys, c = 'k')
    plt.scatter(xs, ys, c = 'k', s = 7)
    plt.show()

    """
        绘制三维曲线
    """
def draw3D(func, pos):
    pass