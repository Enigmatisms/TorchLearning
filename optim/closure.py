#-*-coding:utf-8-*-

import sys

from torch.optim import optimizer
sys.path.append("..")
import torch
from torch import optim
from torch.autograd import Variable, grad
from utils import rosenBrock, draw2D

if __name__ == "__main__":
    x = Variable(torch.normal(
        torch.zeros(2), torch.FloatTensor([1, 1])) + 
        torch.FloatTensor([9, 9]), requires_grad = True
    )
    func_to_use = rosenBrock
    opt = optim.LBFGS([x, ], lr = 1, max_iter = 3000)
    # 如何输出过程？
    def closure():
        opt.zero_grad()
        y = func_to_use(x)
        y.backward(retain_graph = True)
        return y
    print("Before iterations: x = ", x.data)
    opt.step(closure)
    print("After iterations: x = ", x.data)

    ## TODO：使用scheduler进行优化器参数动态调整

