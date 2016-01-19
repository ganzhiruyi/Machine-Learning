import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris = datasets.load_iris()
dataX = iris.get('data')[:,:2] # 只取前两个特征,方便后面画图
dataY = iris.get('target')
dataY = [1 if y == 2 else 0 for y in dataY]
clf = LogisticRegression()
clf.fit(dataX, dataY)
y = clf.predict(dataX)
print 'sklearn LR:'
print metrics.classification_report(dataY, y)
# sklearn的LR

# 机器学习上自己实现LR随机下降算法
def sigmoid(X):
    return 1.0/(1+np.exp(-X))

def stocGradAscent(X,Y,iter_num=100):
    m,n = X.shape
    thetas = np.ones(n)
    for i in range(iter_num):
        data_index = range(m)
        for j in range(m):
            alpha = 4.0/(i+j+1)+0.01 # 在每次循环训练实例时不断减小alpha的值
            rand_index = np.random.randint(0,len(data_index))
            idx = data_index[rand_index]
            h = sigmoid(np.sum(X[idx]*thetas))
            error = Y[idx]-h
            thetas = thetas + alpha*error*X[idx] 
            del data_index[rand_index]
    return thetas

def m_fit(X,Y):
    # 添加常数项
    X = np.array(X)
    m,n = X.shape
    extra = np.ones((m,1))
    X = np.concatenate((X, extra), axis=1)
    thetas = stocGradAscent(X, Y)
    return thetas

def m_predict(X,thetas):
    X = np.array(X)
    m,n = X.shape
    extra = np.ones((m,1))
    X = np.concatenate((X, extra), axis=1)
    # 计算sigmoid函数值
    thetas = np.mat(thetas).transpose()
    h = sigmoid(X*thetas)
    return [1 if y >= 0.5 else 0 for y in h]

def plotDecesion(X,Y,thetas):
    x_min = np.min(X[:,0].ravel())
    x_max = np.max(X[:,0].ravel())
    x = np.arange(x_min,x_max,0.1)
    y = -(thetas[0]*x+thetas[2])/thetas[1]
    plt.plot(x,y)
    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []
    for i,y in enumerate(Y):
        if y == 0:
            neg_x.append(X[i][0])
            neg_y.append(X[i][1])
        else:
            pos_x.append(X[i][0])
            pos_y.append(X[i][1])
    plt.scatter(neg_x, neg_y, c='r', marker='o')
    plt.scatter(pos_x, pos_y, c='g', marker='+')
    plt.show()

thetas = m_fit(dataX, dataY)
y = m_predict(dataX, thetas)
print 'my own LR:'
print metrics.classification_report(dataY, y)
plotDecesion(dataX, dataY, thetas)