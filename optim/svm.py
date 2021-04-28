#-*-coding:utf-8-*-
"""
    SVM numpy 实现
    基于论文SMO: *Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines*
"""

import numpy as np
from random import randint
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as mp3

# 方便可视化，这个SVM是 二分类SVM，维数可变
class SVM:
    # `feat_dim` 特征空间的维度
    def __init__(self, feat_dim = 3, C = 0.002, eps = 1e-5, max_iter = 1000):
        self.fdim = feat_dim
        self.w = np.zeros((feat_dim, 1), dtype = float)
        self.b = 0.0
        self.C = C
        self.eps = eps
        self.max_iter = 1000
        self.alphas = None

    @staticmethod
    def transform(y):
        y = y.astype(np.float64)
        mini = min(y)
        maxi = max(y)
        if maxi == mini:
            raise ValueError("Training set has only one class.")
        if maxi == 1 and mini == -1: return y
        y -= (maxi + mini) / 2
        return np.sign(y)

    # 输入的X 是行列(fdim * n)的  y是任意一维向量（两类标签）
    def fit(self, X, y):
        if not X.shape[0] == self.fdim:
            raise ValueError("Rows of X(%d) is not fdim(%d)"%(X.shape[0], self.fdim))
        sample_num = X.shape[1]
        self.alphas = np.zeros((sample_num, 1))
        y = SVM.transform(y)
        old_norm = 0.0
        iter_num = 0
        print("Start random selection: ", sample_num)
        while True:
            for j in range(sample_num):
                i = randint(0, sample_num - 1)
                while i == j:       # 不允许相同
                    i = randint(0, sample_num - 1)
                self.takeStep(X, y, i, j)
            norm = np.linalg.norm(self.alphas)
            if abs(norm - old_norm) < self.eps: break
            old_norm = norm
            iter_num += 1
            if iter_num > self.max_iter:
                print("Possible non-convergence, for iteration counter exceeds max iteration.")
                break
        print("Training completed. (%d / %d)"%(iter_num, self.max_iter))
        self.w = X @ (self.alphas * y.reshape(-1, 1))
        self.b = np.mean(y - self.w.T @ X)
        print()
        # print("W: ", self.w)
        # print("b: ", self.b)

    def takeStep(self, X, y, i1, i2):
        if i1 == i2: 
            # print("i1 == i2, break.")
            return False
        alph1 = self.alphas[i1]
        alph2 = self.alphas[i2]
        y1 = y[i1]
        y2 = y[i2]
        x1 = X[:, i1].reshape(-1, 1)
        x2 = X[:, i2].reshape(-1, 1)
        E1 = self.predict(x1) - y1
        E2 = self.predict(x2) - y2
        s = y1 * y2
        is_equal = (y1 == y2)
        L = self.calcL(alph1, alph2, is_equal)
        H = self.calcH(alph1, alph2, is_equal)
        if L == H: 
            # print("L == H, break")
            return False
        k11 = SVM.kernel(x1, x1)
        k22 = SVM.kernel(x2, x2)
        k12 = SVM.kernel(x1, x2)
        eta = k11 + k22 - 2 * k12
        if eta > 0:
            a2 = alph2 + float(y2 * (E1 - E2)) / eta
            if a2 < L: a2 = L       # clipping
            elif a2 > H: a2 = H
        else:
            print("Eta <= 0")
            Lobj = self.calcObj(X, y, i2, L)
            Hobj = self.calcObj(X, y, i2, H)
            if Lobj < Hobj - self.eps: a2 = L
            elif Hobj > Hobj + self.eps: a2 = H
            else: a2 = alph2
        # if abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
        #     # print("Something being too small, break")
        #     return False
        a1 = alph1 + s * (alph2 - a2)
        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + self.b
        self.b = (b1 + b2) / 2
        self.w += y1 * (a1 - alph1) * x1 + y2 * (a2 - alph2) * x2
        self.alphas[i1] = a1
        self.alphas[i2] = a2
        return True

    def predict(self, X):
        return np.sign(self.w.T @ X + self.b)

    def plotSuperPlain3D(self, X, y, plot_plain = False):
        y = SVM.transform(y)
        X1 = X[:, y == 1]
        X2 = X[:, y == -1]
        fig = plt.figure()
        ax = mp3.Axes3D(fig)
        ax.scatter3D(X1[0, :], X1[1, :], X1[2, :], color = 'red')
        ax.scatter3D(X2[0, :], X2[1, :], X2[2, :], color = 'blue')
        if plot_plain == True:
            xx, yy = np.meshgrid(np.linspace(-3, 3), np.linspace(-3, 3))
            zz = - self.w[0] / self.w[2] * xx - self.w[1] / self.w[2] * yy - self.b
            ax.plot_surface(xx, yy, zz, color = 'g', alpha = 0.2)
    
    def plotOnlySuperPlain(self, ax, c, label, alpha = 0.2):
        xx, yy = np.meshgrid(np.linspace(-3, 3), np.linspace(-3, 3))
        zz = - self.w[0] / self.w[2] * xx - self.w[1] / self.w[2] * yy - self.b
        surf = ax.plot_surface(xx, yy, zz, color = c, alpha = alpha, label = label)
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d

    def calcL(self, alph1, alph2, equal = False):
        if equal == True:
            return max(0, alph1 + alph2 - self.C)
        else:
            return max(0, alph2 - alph1)

    def calcH(self, alph1, alph2, equal = False):
        if equal == True:
            return min(self.C, alph2 + alph1)
        else:
            return min(self.C, alph2 - alph1 + self.C)

    @staticmethod
    def kernel(x1, x2):
        return x1.T @ x2

    # 简化的objective function 由于计算Lobj 与 Hobj时 只有i2对应的alpha（a2）不相同，其他均一致
    # 所以只需要计算 i2不同造成的差异即可 返回负值 TODO: check again
    def calcObj(self, X, y, index, a2):
        res = 0
        xi = X[:, index]
        yi = y[index]
        for i in range(self.fdim):
            if i != index:
                res += 2 * yi * y[i] * self.alphas[i] * a2 * SVM.kernel(xi, X[:, i])
            else:
                res += ((yi * a2) ** 2) * SVM.kernel(xi, xi)
        return -res

if __name__ == "__main__":
    pts = (40, 44)
    clf = SVM()
    xs = np.random.randn(3, pts[0] + pts[1])
    xs[:, :pts[0]] += np.array([[0.5], [0.5], [0.5]])
    xs[:, pts[0]:] -= np.array([[0.5], [0.5], [0.5]])
    ys = np.hstack((np.ones(pts[0]), -np.ones(pts[1])))
    clf.fit(xs, ys)
    clf.plotSuperPlain3D(xs, ys, True)
    plt.show()
