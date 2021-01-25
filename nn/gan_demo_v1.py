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
    cnt = 0
    cnt += 1
    x = torch.autograd.Variable(torch.randn(40, 256)).cuda()
    output = gan(x).cpu()
    print(output.shape)
    for i in range(4):
        for j in range(10):
            plt.subplot(4, 10, i * 10 + j + 1)
            img = output[i * 10 + j].data.numpy()[0]
            img = (img * 255).astype(np.uint8)
            plt.imshow(img, cmap = 'gray')
    plt.show()

