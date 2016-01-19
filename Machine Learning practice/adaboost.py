#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-01-15 15:47:15
# @Author  : ganzhiruyi (ganzhiruyi0@gmail.com)
# @Link    : https://github.com/ganzhiruyi
# @Version : $1.0$

import numpy as np
from numpy import zeros,ones,mat,multiply,exp,log,inf,sign
from sklearn import metrics, datasets, cross_validation
import matplotlib.pyplot as plt
def stumpClassify(X,dimen,threshVal,threshIneq):
    # X:特征矩阵,dimen:特征的维度,threshVal:阀值,threshIneq:大于阀值是-1还是不大于阀值是-1
    m,n = X.shape
    retArr = mat(ones((m,1)))
    if threshIneq == 'lt':
        retArr[X[:,dimen] <= threshVal] = -1.0
    else:
        retArr[X[:,dimen] > threshVal] = -1.0
    return retArr

def buildStump(X,Y,D):
    # 建立弱分类器，决策树，D表示权值矩阵
    m,n = X.shape
    numStep = 10.0 # 指定枚举的特征个数
    bestStump = {} # 表示最好的决策树对应的字典，包括dimen,threshVal,threshIneq
    bestClassEst = mat(zeros((m,1))) # 最好的决策树对应的预测分类
    minError = inf
    for i in range(n):
        rangeMin = X[:,i].min();rangeMax = X[:,i].max()
        stepSize = (rangeMax-rangeMin)/numStep
        for j in range(-1,int(numStep)+1):
            threshVal = rangeMin + stepSize*j
            for ineq in ['lt','gt']:
                errorArr = mat(ones((m,1)))
                predictY = stumpClassify(X, i, threshVal, ineq)
                errorArr[predictY == Y] = 0 # 预测值等于真实值的错误数为0
                error = errorArr.T*D # 出错的实例乘以权值并相加
                if error < minError:
                    minError = error
                    bestClassEst = predictY.copy() # 这里用predict的拷贝
                    bestStump['dimen'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['threshIneq'] = ineq
    return minError,bestStump,bestClassEst

def adaBoostTrainDS(X,Y,numIter=40):
    weakClassifiers = []
    X = mat(X)
    Y = mat(Y).transpose()
    m,n = X.shape
    D = mat(ones((m,1))*1.0/m) # 初始化权值矩阵
    aggClassEst = mat(ones((m,1))) # 表示每次迭代后累加分类器得到的预测分类
    for i in range(numIter):
        error,bestStump,classEst = buildStump(X, Y, D)
        alpha = float(0.5*log((1.0-error)/max(error,1e-12))) # 避免除0溢出
        # 这里加float是为了把矩阵alpha转化为数字
        tmp = multiply(-alpha*Y, classEst)
        D = multiply(D, exp(tmp))
        D = D/D.sum()
        bestStump['alpha'] = alpha
        weakClassifiers.append(bestStump)
        aggClassEst = aggClassEst + alpha*classEst
        aggError = (sign(aggClassEst) != Y).sum()
        aggErrorRate = aggError*1.0/m
        print 'Iter %d, total error: %.4f' % (i,aggErrorRate)
        if aggErrorRate == 0.0:
            break
    return weakClassifiers

def loadIrisData():
    # 读取iris的数据,用于线性划分
    iris = datasets.load_iris()
    dataX = iris.get('data')[:,:2] # 只取前两个特征,方便后面画图
    dataY = iris.get('target')
    dataY = [1 if y == 2 else -1 for y in dataY]
    return dataX,dataY

def predict(X,Y,weakClassifiers):
    X = mat(X)
    Y = mat(Y).transpose()
    m,n = X.shape
    aggClassEst = mat(ones((m,1)))
    for clf in weakClassifiers:
        alpha = clf['alpha']
        classEst = stumpClassify(X, clf['dimen'], clf['threshVal'], clf['threshIneq'])
        aggClassEst = aggClassEst + alpha*classEst
    print metrics.classification_report(Y.A, sign(aggClassEst).A)
    return aggClassEst

def plotROC(predStrengths,Y):
    # 根据预测值以及真实的标签画出ROC曲线
    cur = (1.0,1.0)
    ySum = 0.0
    m = len(Y)
    cntPosclass = 0
    for y in Y:
        cntPosclass += (1 if y == 1 else 0)
    print cntPosclass
    yStep = 1.0/cntPosclass
    xStep = 1.0/(m-cntPosclass)
    sortedIndices = predStrengths.T.A[0].argsort()
    for idx in sortedIndices:
        if Y[idx] == 1.0:
            deltax = 0
            deltay = yStep
        else:
            deltax = xStep
            deltay = 0
            ySum += cur[1]
        plt.plot([cur[0],cur[0]-deltax],[cur[1],cur[1]-deltay])
        cur = (cur[0]-deltax,cur[1]-deltay)
    plt.show()
    print 'The Area unser the curve is: ',ySum*xStep

 
X,Y = loadIrisData()
train_x,test_x,train_y,test_y = cross_validation.train_test_split(X,Y,test_size=0.2)
weakclfs = adaBoostTrainDS(train_x, train_y)
aggClassEst = predict(test_x, test_y, weakclfs)
plotROC(aggClassEst, test_y)



