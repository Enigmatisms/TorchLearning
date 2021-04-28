#-*-coding:utf-8-*-
"""
    多分类SVM one-versus-one形式的naive实现
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mp3
from svm import SVM

colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'purple']
labels = ["between red & blue", "between red & green", "between green & blue"]
pairs = ((0, 1), (0, 2), (1, 2))

def plotResult(X, y, clfs, plot_plains = True):
    fig = plt.figure()
    ax = mp3.Axes3D(fig)
    if plot_plains:
        for i in range(3):
            clfs[i].plotOnlySuperPlain(ax, colors[i + 3], labels[i], 0.3)
    for i in range(3):
        Xs = X[i]
        ax.scatter3D(Xs[0, :], Xs[1, :], Xs[2, :], color = colors[i])
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    clfs = [SVM() for i in range(3)] # c12, c13, c23
    pts = (50, 50, 50)
    xs = np.random.randn(3, 3, pts[0])
    xs[0] += np.array([[2], [1], [1]])
    xs[1] -= np.array([[1], [1], [2]])
    xs[2] -= np.array([[1], [2], [-4]])
    ys = np.array([np.zeros(pts[0]), np.ones(pts[1]), 2 * np.ones(pts[2])])
    for i in range(3):
        a, b = pairs[i]
        X = np.hstack((xs[a], xs[b]))
        y = np.hstack((ys[a], ys[b]))
        clfs[i].fit(X, y)
    plotResult(xs, ys, clfs)
    