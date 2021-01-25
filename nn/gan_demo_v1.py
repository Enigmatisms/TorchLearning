#-*-coding:utf-8-*-
"""
    gan_v1.py 的结果可视化
"""

import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
from gan_v1 import Gen

if __name__ == "__main__":
    gan = torch.load("..\\models\\gan_gen.pkl")
    x = torch.autograd.Variable(torch.randn(50, 128)).cuda()
    output = gan(x).cpu()
    print(output.shape)
    for i in range(5):
        for j in range(10):
            plt.subplot(5, 10, i * 10 + j + 1)
            img = output[i * 10 + j].data[0]
            img = img.clamp(0, 1).cpu()
            plt.imshow(img, cmap = 'gray')
    plt.show()

