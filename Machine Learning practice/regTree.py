#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-03 21:35:49
# @Author  : ganruyi (ganzhiruyi0@gmail.com)
# @Link    : https://github.com/ganzhiruyi
# @Version : 1.0

import os
from numpy import *

def loadDataSet(file_path):
    dataSet = loadtxt(file_path, delimiter='\t')
    return mat(dataSet)

def binSplitDataSet(dataSet, feature, val):
    # 注意这里的左儿子和右儿子都只保留第一维的特征值，为了显示数据其实也只有一维
    lson = dataSet[nonzero(dataSet[:,feature] > val)[0]]
    rson = dataSet[nonzero(dataSet[:,feature] <= val)[0]]
    return lson,rson

def regLeaf(dataSet):
    return mean(dataSet[:,-1])

def regErr(dataSet):
    # 返回所有Y值对应的平均差的总和（可以理解为熵），即方差*数据的个数
    return var(dataSet[:,-1])*dataSet.shape[0]

def chooseBestSplit(dataSet,leafType,errType,ops):
    tolS,tolN = ops # 其中tolS表示子树应该获得的提升的下界,tolN表示子树应该包含的个数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet) # 如果只有一个值
    m,n = dataSet.shape
    bestS = inf
    bestFeature = 0
    bestSplitVal = 0
    S = errType(dataSet)
    for feature in range(n-1):
        for splitVal in set(dataSet[:,feature].T.tolist()[0]):
            lson,rson = binSplitDataSet(dataSet, feature, splitVal)
            if (lson.shape[0] < tolN) or (rson.shape[0] < tolN):
                continue
            newS = errType(lson) + errType(rson)
            if newS < bestS:
                bestS = newS
                bestFeature = feature
                bestSplitVal = splitVal
    if S - bestS < tolS: # 如果误差减少不多于tolS,舍弃划分
        return None,leafType(dataSet)
    lson,rson = binSplitDataSet(dataSet, bestFeature, bestSplitVal)
    if (lson.shape[0] < tolN) or (rson.shape[0] < tolN): # 如果子树少于tolN,舍弃划分
        return None,leafType(dataSet)
    return bestFeature,bestSplitVal

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feature,val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feature == None:
        return val
    retTree = {}
    retTree['spInd'] = feature
    retTree['spVal'] = val
    lson, rson = binSplitDataSet(dataSet, feature, val)
    retTree['left'] = createTree(lson)
    retTree['right'] = createTree(rson)
    return retTree

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    # 对树进行塌陷处理，合并叶子节点
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right'])/2.0

def prune(tree,testData):
    # 利用测试数据对之前生成的树tree进行后剪枝
    if testData.shape[0] == 0:
        return getMean(tree) # 没有测试数据直接塌陷
    if isTree(tree['left']) or isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet,rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1]-tree['left'],2)) + sum(power(rSet[:,-1]-tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(treeMean-testData[:,-1],2))
        if errorMerge < errorNoMerge:
            print 'merge'
            return treeMean
        else:
            return tree
    else:
        return tree

def linearSolve(dataSet):
    m,n = dataSet.shape
    X = mat(ones((m,n)))
    Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, try increasing the second value of ops')
    ws = xTx.I*(X.T*Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yPre = X*ws
    return sum(power(Y-yPre,2))


'''
X = loadDataSet('data/9-2.txt')
tree = createTree(X)
print tree
print 'start prune....'
testX = loadDataSet('data/9-2-test.txt')
treeAfterPrune = prune(tree,testX)
print treeAfterPrune
'''
X = loadDataSet('data/9-3.txt')
print X
tree = createTree(X, leafType=modelLeaf, errType=modelErr, ops=(1,10))
print tree
