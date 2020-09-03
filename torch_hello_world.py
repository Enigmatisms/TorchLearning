#!/usr/bin/env python3
#-*-coding:utf-8-*-
# Torch 的第一个代码

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


def function(x:torch.tensor):
    return torch.sin(x)

# grad 在不同情况下的叠加性
def gradTest():
    print("Order 1 gradient test:")
    x = Variable(torch.FloatTensor([1, 2, 3]), requires_grad = True)
    y = x * x
    z = 3 * x
    f = x ** 3

    y.backward(torch.ones(3))       # 此步的结果 x.grad = [2, 4, 6]
    z.backward(torch.ones(3))       # 此步的结果 x.grad = [2, 4, 6] + [3, 3, 3] = [5, 7, 9]
    print(x.grad)
    f.backward(torch.ones(3))       # 此步的结果 x.grad = [5, 10, 15] + [3, 12, 27] = [8, 19, 36]
    print(x.grad)

    # 实验结果：正确 grad 就是叠加的

# 二阶导测试
# 这是典型的错误，二阶导就不是这样求的
# retain graph 保存的是当前的计算图 以便再一次对z求导，而不是说保存求导的结果(dz/dx)
def grad2Test():
    print("Order 2 gradient test: (False)")
    x = Variable(torch.FloatTensor([1, 2, 3]).cuda(), requires_grad = True)
    z = 1 / 3 * x ** 3

    z.backward(torch.ones(3).cuda(), retain_graph = True)     
    print(x.grad)
    z.backward(torch.ones(3).cuda(), retain_graph = True)      
    print(x.grad)
    z.backward(torch.ones(3).cuda(), retain_graph = True)     
    print(x.grad)

# 二阶导求法
def order2():
    print("Order 2 gradient test: (True)")
    x = Variable(torch.FloatTensor([1, 2, 3]), requires_grad = True)
    z = 1 / 3 * x ** 3

    # 此时新建了一个计算图 这个计算图将返回z的一阶导吗
    # dz = z.backward(torch.ones(3), retain_graph = True, create_graph = True)
    # 以上写法是不会返回值的 (backward 没有返回值)，这样无法获得二阶导

    # 使用 以下写法会返回一个tensor 元组，保存了一阶导的结果
    # 原来的z已经被销毁了（没有retain_graph）
    dz = torch.autograd.grad(z, x, torch.FloatTensor([1, 1, 1]), create_graph = True)
    print(dz)

    # 回顾一下高等数学求二阶导，[df/dx, df/dy, df/dz] 二阶导求出来则是一个矩阵
    # 我们需要对其中每一个元素(一阶导的每一个元素对x求导) 求导
    grad = dz[0]
    for element in grad:
        print(torch.autograd.grad(element, x, retain_graph = True, create_graph = True))

    # 注意!!! @ 的用法 @ 是矩阵乘法 * 则是对应元素相乘
    # 也就是说，z实际上是[1 / 3 * x1 ^ 3, 1 / 3 * x2 ^ 3, 1 / 3 * x3 ^ 3]
    # 求导以后为[x1 ^ 2, x2 ^ 2, x3 ^ 2] 那么二阶导自然会是一个对角阵了
    print(x.grad)       # x.grad在进行torch.autograd.grad时貌似不会被计算 （貌似作为返回值）

def testAutoGrad():
    print("Test auto grad:")
    x = Variable(torch.FloatTensor([1, 2, 3]), requires_grad = True)
    z = 1 / 3 * x ** 3

    print(torch.autograd.grad(z, x, torch.FloatTensor([1, 1, 1])))
    

if __name__ == "__main__":
    data = torch.tensor([1., 2., 3.])     # 这是普通的标量tensor
    x = Variable(data, requires_grad = True)
    y = x ** 3

    print("Grad before backward:", x.grad)

    # FloatTensor 此处的作用不是表示在哪一点求梯度，输入的tensor是对backward的梯度乘以放大系数
    # backward 求出的直接是位于 x.data 的导数
    y.backward(torch.FloatTensor([1, 2, 4]), retain_graph = True)
    print("Grad after backward:", x.grad)

    """
        Returns a new Tensor, detached from the current graph.
        The result will never require gradient.
        需要求导的Variable是无法直接转化为 numpy 结构的 其中存在了自动求导需要的函数结构，而不是完全的data
    """
    xs = x.detach().numpy()
    ys = x.grad

    gradTest()
    grad2Test()
    order2()
    testAutoGrad()