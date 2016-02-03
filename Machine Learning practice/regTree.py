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

X = loadDataSet('data/9-2.txt')
print createTree(X)
