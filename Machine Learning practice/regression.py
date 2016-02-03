#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-01-19 10:57:07
# @Author  : ganzhiruyi (ganzhiruyi0@gmail.com)
# @Link    : https://github.com/ganzhiruyi
# @Version : $1.0$

from numpy import *
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filePath):
    # 导入数据，数据每行的最后一个是Y值
    X = loadtxt(filePath, delimiter='\t')
    Y = X[:, -1]
    X = X[:, :-1]  # 第一个一直是1,表示常数项
    return X, Y


def standRegress(X, Y):
    # 最小二乘法，返回回归系数
    X = mat(X)
    Y = mat(Y).transpose()
    if linalg.det(X.T * X) == 0.0:
        raise ValueError('The matrix is singular, cannot inverse.')
    w = (X.T * X).I * (X.T * Y)
    return w


def plot2DRegress(X, Y, w):
    plt.scatter(X[:, 1], Y)  # 画图要把X只取x变量部分
    y = X * w
    plt.plot(X[:, 1], y)
    plt.show()


def lwlr(x, X, Y, k):
    # 根据每个点x和整个数据集X的差距，计算一个对应的y值
    X = mat(X)
    Y = mat(Y).transpose()
    m, n = X.shape
    weights = mat(eye(m))
    for j in range(m):
        diffMat = x - X[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = X.T * (weights * X)
    if linalg.det(xTx) == 0.0:
        raise ValueError('The matrix is singular, cannot inverse.')
    w = xTx.I * (X.T * weights * Y)
    return x * w


def lwlrTest(testArr, X, Y, k=1.0):
    # 得到所有testArr关于X,Y的预测值
    m, n = testArr.shape
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], X, Y, k)
    return yHat


def plot2Dlwlr(X, Y, yHat):
    sortedIdices = X[:, 1].argsort()
    xSort = X[sortedIdices]
    ySort = yHat[sortedIdices]
    plt.scatter(X[:, 1], Y)
    plt.plot(xSort[:, 1], ySort, c='r')
    plt.show()


def regularize(X):
    # 按列进行正则化
    retX = X.copy()
    xMean = mean(retX, axis=0)
    invarX = var(retX, axis=0)
    retX = (retX - xMean) / invarX
    return retX


def rssError(yTrue, yPred):
    # 统计均方误差
    return ((yTrue - yPred)**2).sum()


def stagewise(X, Y, eps=0.01, numIters=200):
    # 前向逐步线性回归
    X = mat(X)
    m, n = X.shape
    Y = mat(Y).transpose()
    yMean = mean(Y)
    Y = Y - yMean
    X = regularize(X)  # 这里的这个处理对于第一列全为1就会出错
    # from sklearn import preprocessing
    # X = preprocessing.normalize(X)
    w = zeros((n, 1))
    wTest = w.copy()
    wBest = w.copy()
    returnMat = mat(zeros((numIters, n)))
    for i in range(numIters):
        minRssError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wTest = w.copy()
                wTest[j] += sign * eps
                yTest = X * wTest
                error = rssError(Y.A, yTest.A)
                if error < minRssError:
                    minRssError = error
                    wBest = wTest
        w = wBest.copy()
        returnMat[i, :] = w.T
    return returnMat

def plotWs(ws):
    # 画出相关系数随迭代次数的变化规律
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ws)
    plt.show()

# X, Y = loadDataSet('data/8-2.txt')
X, Y = loadDataSet('data/8-abalone.txt')
print X
# 测试standRegress
# w = standRegress(X, Y)
# plot2DRegress(X, Y, w)

# 测试lwlr
# yHat = lwlrTest(X,X,Y,k=0.01)
# plot2Dlwlr(X, Y, yHat)

# 测试前向逐步回归 stagewise
ws = stagewise(X, Y, eps=0.005, numIters=1000)
plotWs(ws)
