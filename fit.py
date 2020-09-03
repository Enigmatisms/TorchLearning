#!/usr/bin/env python3
#-*-coding:utf-8-*-
# 直线拟合 (Pytorch项目)

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable as Var

def generateSamples(a = 3, b = 1):
    xs_raw = np.linspace(-5, 5, 100)
    noise = np.random.normal(0, 2, (1, 100))
    xs = xs_raw + noise
    ys = a * xs + b
    return xs_raw, ys[0], a * xs_raw + b

def generateParabola(a = 1, b = 2, c = 1):
    xs_raw = np.linspace(-3, 3, 100)
    noise = np.random.normal(0, 2, (1, 100))
    ys = a * xs_raw * xs_raw + b * xs_raw + c
    temp = ys + noise
    return xs_raw, temp[0], ys

def costFunction(coeffs, xs, ys, order):
    z = Var(torch.FloatTensor([0]))
    for i in range(len(xs)):
        temp = ys[i]
        for j in range(order + 1):
            temp -= coeffs[j] * xs[i] ** (order - j)
        z += temp ** 2
    return z

# 二阶牛顿法 (好鸡儿快啊 一次迭代就收敛了 我怀疑这是精确搜索)
def torchFit(xs, ys, order = 2, max_iter = 1000, error = 10e-4):
    coe = Var(torch.ones(order + 1), requires_grad = True)
    func = costFunction(coe, xs, ys, order)
    for i in range(max_iter):
        df = torch.autograd.grad(func, coe, create_graph = True)
        ddf = torch.autograd.grad(df, coe, torch.ones(order + 1))
        temp = df[0].data / ddf[0]
        if temp.norm() < error:
            print("Convergence before max iter(%d / %d)"%(i, max_iter))
            break
        coe.data -= temp
        func = costFunction(coe, xs, ys, order)
    return coe.data.numpy()
    

if __name__ == "__main__":
    xs, ys, truth = generateSamples()
    xsp, ysp, truthp = generateParabola()
    coeffs = torchFit(xs, ys, 1, 1000, 10e-5)
    csp = torchFit(xsp, ysp, 2, 1000, 10e-5)
    
    f_lin = np.poly1d(coeffs)
    f_para = np.poly1d(csp)
    f_npfit = np.poly1d(np.polyfit(xsp, ysp, 2))

    ys_lin = f_lin(xs)
    ys_para = f_para(xsp)
    ys_npfit = f_npfit(xsp)

    plt.figure(1)
    plt.scatter(xs, ys, c = "red", label = "data")
    plt.plot(xs, truth, c = "blue", label = "truth")
    plt.plot(xs, ys_lin, c = "green", label = "torch fit")
    plt.title("Pytorch Polyfit result (Linear)")
    plt.legend()

    plt.figure(2)
    plt.scatter(xsp, ysp, c = "red", label = "data")
    plt.plot(xsp, truthp, c = "blue", label = "truth")
    plt.plot(xsp, ys_para, c = "green", label = "torch fit")
    plt.plot(xsp, ys_npfit, c = "orange", label = "numpy fit")
    plt.title("Pytorch Polyfit result (Parabola)")
    plt.legend()

    plt.show()

