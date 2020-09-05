#!/usr/bin/env python3
#-*-coding:utf-8-*-
# RANSAC 对图像匹配结果进行优化
# 非线性最小二乘 优化所得的透视投影
# Optimizer 学的没有那么深，还不会用optim来解决非线性优化问题

"""
    龟速外点消除 真的巨慢
    而且结果还并不好看，但我觉得我的思路是对的
"""

import cv2 as cv
import numpy as np
import torch
import random as rd
import matplotlib.pyplot as plt
from torch.autograd import Variable as Var

# 相机内参
K = np.array([[1776.67168581218, 0, 720],
    [0, 1778.59375346543, 540],
    [0, 0, 1]])

invK = np.array([[ 5.62850192e-04,  0.00000000e+00, -4.05252138e-01],
       [ 0.00000000e+00,  5.62241939e-04, -3.03610647e-01],
       [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

pp = np.array([[734.801179192106, 623.419725108874]])
focal = np.array([[1622.54547967434, 1626.12207046718]])
dist = np.array([-0.388384764711487, 0.213883897041784, -0.0104616918755179, 0.00171604199932318])

def roughMatching(pic1, pic2, draw_result = False):
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(pic1, None)
    kp2, des2 = orb.detectAndCompute(pic2, None)

    matcher = cv.BFMatcher(normType = cv.NORM_HAMMING, crossCheck = True)
    matches = matcher.knnMatch(des1, des2, k = 1)

    if draw_result:
        img = cv.drawMatchesKnn(pic1, kp1, pic2, kp2, matches, pic2)
        cv.imshow("disp", img)
        cv.waitKey(0)
    else:
        temp = []
        for match in matches:
            if match: temp.append(match[0])
        return kp1, kp2, des1, des2, temp

def ransac(kp1, kp2, matches, max_iter = 1500):
    whole = list(range(len(matches)))
    chosen = set()
    best_inlier = set()
    total_dist = float("inf")
    for i in range(max_iter):
        samp = tuple(sorted(rd.sample(whole, k = 4)))      # 随机取四个点进行透视变换矩阵求解 (不这样实现就只能设计哈希函数)
        while samp in chosen:
            samp = tuple(sorted(rd.sample(whole, k = 4)))  
        points1 = np.array([kp1[matches[pos].queryIdx].pt for pos in samp], dtype = "float32")
        points2 = np.array([kp2[matches[pos].trainIdx].pt for pos in samp], dtype = "float32")
        chosen.add(samp)
        inlier = set()
        H = cv.findHomography(points1, points2)[0]
        # print(H)
        
        for mt in matches:
            src = np.ones(3)      # 齐次坐标
            dst = np.ones(3)
            pt1 = kp1[mt.queryIdx].pt
            pt2 = kp2[mt.trainIdx].pt

            src[:2] = pt1
            dst[:2] = pt2

            dist = costFunction(src, dst, H)
            # print(dist)
            if dist < 32:                              # 内点阈值参数
                inlier.add(mt)
        if len(inlier) < 6:
            print("Not the best inlier set or too few inliers (%d / %d)"%(len(inlier), len(best_inlier)))
            continue
        else:
            # 在计算完最小二乘估计的变换之后，不进行重新估计
            pts1 = np.array([kp1[mt.queryIdx].pt for mt in inlier])
            pts2 = np.array([kp2[mt.trainIdx].pt for mt in inlier])
            pts1 = np.hstack((pts1, np.ones( (len(pts1), 1) )))     # 转换为齐次坐标
            pts2 = np.hstack((pts2, np.ones( (len(pts2), 1) )))
            cost = homoOptimization(torch.FloatTensor(pts1), torch.FloatTensor(pts2))
            print("Iter (%d / %d) with cost %f"%(i, max_iter, cost))
            if len(inlier) > len(best_inlier):
                total_dist = cost
                best_inlier = inlier
    return list(best_inlier)
            # 以下需要使用梯度下降

def homoOptimization(pts1, pts2, lr = 1, max_iter = 200):
    M = Var(torch.ones((3, 3)), True)
    for i in range(max_iter):
        cost = Var(torch.FloatTensor([0]))
        for j in range(len(pts1)):
            cost += costFunction(pts1[j], pts2[j], M)
        cost.backward()
        M.data -= M.grad * lr
        norm = M.grad.norm()
        # print(">>>>>> Iter (%d / %d) with norm %f"%(i, max_iter, norm))
        if M.grad.norm() < 0.01:        # 梯度下降
            break
        M.grad.zero_()
    return cost.data

def costFunction(p1, p2, M):
    if type(M) == torch.Tensor:     # 返回变量式（计算图）
        return (
            (p1 @ M[0].unsqueeze(1) / (p1 @ M[2].unsqueeze(1)) - p2[0]) ** 2 +
            (p1 @ M[1].unsqueeze(1) / (p1 @ M[2].unsqueeze(1)) - p2[1]) ** 2
        )
    else:       # 返回数值
        # print(p1, "\n")
        # print(M[0].reshape(-1, 1), "\n")
        # print(p2)
        return np.sqrt(
            (p1 @ M[0].reshape(-1, 1) / (p1 @ M[2].reshape(-1, 1)) - p2[0]) ** 2 +
            (p1 @ M[1].reshape(-1, 1) / (p1 @ M[2].reshape(-1, 1)) - p2[1]) ** 2
        )

if __name__ == "__main__":
    size = (1200, 900)
    pic1 = cv.imread(".\\data\\p1.png")
    pic2 = cv.imread(".\\data\\p2.png")

    pic1 = cv.undistort(pic1, K, dist)
    pic2 = cv.undistort(pic2, K, dist)

    pic1 = cv.resize(pic1, (1200, 900))
    pic2 = cv.resize(pic2, (1200, 900))

    kp1, kp2, des1, des2, matches = roughMatching(pic1, pic2)
    best_match = ransac(kp1, kp2, matches, 100)
    
    
    img = cv.drawMatches(pic1, kp1, pic2, kp2, best_match, pic2)
    plt.imshow(img)
    plt.show()