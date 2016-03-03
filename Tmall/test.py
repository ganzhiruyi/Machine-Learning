# -*- coding:utf-8 -*-
__author__ = 'Ganruyi'
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.feature_selection import SelectPercentile, f_classif
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from numpy import vstack
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_user_types_index(pre_y,types):
    # 这是从前面的预测结果pre_y中找出types分类对应的index
    all_idx = []
    for type in types:
        idx = []
        for i,y in enumerate(pre_y):
            if y == type:
                idx.append(i)
        all_idx.extend(idx)
    return all_idx

def get_train_user_types_index(Y, types):
    # 这是从训练数据Y中找出types对应的index
    train_user_type_index = {}
    n = Y.shape[0]
    for type in types:
        train_user_type_index[type] = []
        for i in range(n):
            if Y.irow(i) == type:
                train_user_type_index[type].append(i)
    return train_user_type_index

def combine_all_features(behavior,is_train = True):
    if is_train:
        prefix = 'train'
    else:
        prefix = 'test'
    # 将三个训练特征文件组合成一个
    cat_x = pd.read_csv('data/%s_feature_cat_id_%d.csv' % (prefix,behavior),index_col=0)
    seller_x = pd.read_csv('data/%s_feature_seller_id_%d.csv' % (prefix,behavior),index_col=0)
    brand_x = pd.read_csv('data/%s_feature_brand_id_%d.csv' % (prefix,behavior),index_col=0)
    # time_x = pd.read_csv('data/%s_feature_time_stamp.csv' % prefix)
    x = pd.concat([cat_x,brand_x,seller_x], axis=1)
    print '%s behavior %d complete...' % (prefix,behavior)
    return x

def combine_all_behavior(is_train = True):
    x0 = combine_all_features(0,is_train)
    x2 = combine_all_features(2,is_train)
    x3 = combine_all_features(3,is_train)
    x = pd.concat([x0,x2,x3], axis=1)
    return x.fillna(0).sort_index()
# combine_all_behavior(is_train=False)

def get_train_and_test_spaese_matrix():
    Y = pd.read_csv('data/train_Y.csv', index_col='user_id')['type']
    train_X = combine_all_behavior()
    # dump_svmlight_file(train_X,Y,'data/train_metrix')
    dump_svmlight_file(train_X,Y,'data/train_metrix_3')
    test_X = combine_all_behavior(is_train=False)
    test_Y = [0]*(test_X.shape[0])
    # dump_svmlight_file(test_X,test_Y,'data/test_metrix')
    dump_svmlight_file(test_X,test_Y,'data/test_metrix_3')

def nn_classify():
    # train_X,Y = load_svmlight_file('data/train_metrix')
    # rows = pd.read_csv('data/log_test2.csv',index_col=0).sort_index().index.unique()
    # train_X = pd.read_csv('data/train_tfidf.csv',index_col=0)
    # test_X = pd.read_csv('data/test_tfidf.csv',index_col=0)
    # select = SelectPercentile(f_classif, percentile=50)
    # select.fit(train_X,Y)
    # train_X = select.transform(train_X)
    # test_X = select.transform(test_X)
    # print 'dump train...'
    # dump_svmlight_file(train_X,Y,'data/train_last')
    # test_Y = [0]*(test_X.shape[0])
    # print 'dump test...'
    # dump_svmlight_file(test_X,test_Y,'data/test_last')

    train_X,Y = load_svmlight_file('data/train_last')
    test_X,test_Y = load_svmlight_file('data/test_last')
    train_X = train_X.toarray()
    test_X = test_X.toarray()
    Y = [int(y)-1 for y in Y]
    print 'Y:',len(Y)
    rows = pd.read_csv('data/log_test2.csv',index_col=0).sort_index().index.unique()
    train_n = train_X.shape[0]
    m = train_X.shape[1]
    test_n = test_X.shape[0]
    print train_n,m,#test_n
    train_data = ClassificationDataSet(m,1,nb_classes=12)
    test_data = ClassificationDataSet(m,1,nb_classes=12)
    # test_data = ClassificationDataSet(test_n,m,nb_classes=12)
    for i in range(train_n):
        train_data.addSample(np.ravel(train_X[i]),Y[i])
    for i in range(test_n):
        test_data.addSample(test_X[i],Y[i])
    trndata = train_data
    # tstdata = train_data

    trndata._convertToOneOfMany()
    # tstdata._convertToOneOfMany()
    test_data._convertToOneOfMany()

     # 先用训练集训练出所有的分类器
    print 'train classify...'
    fnn = buildNetwork( trndata.indim, 400 , trndata.outdim, outclass=SoftmaxLayer )
    trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0.01)
    trainer.trainEpochs(3)
    # print 'Percent Error on Test dataset: ' , percentError( trainer.testOnClassData (
    #            dataset=tstdata )
    #            , )
    print 'end train classify'
    pre_y = trainer.testOnClassData(dataset=trndata)
    print metrics.classification_report(Y,pre_y)
    pre_y = trainer.testOnClassData(dataset=test_data)
    print 'write result...'
    print 'before:',pre_y[:100]
    pre_y = [int(y)+1 for y in pre_y]
    print 'after:',pre_y[:100]
    DataFrame(pre_y,index=rows).to_csv('data/info_test2.csv', header=False)
    print 'end...'
    # print 'start classify....'
    # 第一个分类结果
    # predict_Y = clf.predict(x_train)
    # predict_Y = clf.predict(train_X)
    # print 'classify result:'
    # print metrics.classification_report(y_train,predict_Y)
    # print metrics.classification_report(Y,predict_Y)
    # print predict_Y,len(predict_Y)
    # print 'end classify...'
    # predict_Y = clf.predict(X[cnt_train:]) # 训练注释这一行,输出测试集打开这一行，注释之后的print metric
    # predict_Y = clf.predict(test_X) # 训练注释这一行,输出测试集打开这一行，注释之后的print metric
    # DataFrame(predict_Y,index=rows).to_csv('data/info_test2.csv', header=False)

# nn_classify()

def classify():
    train_X,Y = load_svmlight_file('data/train_last')
    test_X,test_Y = load_svmlight_file('data/test_last')
    train_X = train_X.toarray()
    test_X = test_X.toarray()
    Y = [int(y) for y in Y]
    # print 'Y:',len(Y)
    rows = pd.read_csv('data/log_test2.csv',index_col=0).sort_index().index.unique()
    train_n = train_X.shape[0]
    m = train_X.shape[1]
    test_n = test_X.shape[0]
    print train_n,m,#test_n
     # 先用训练集训练出所有的分类器
    print 'train classify...'
    clf1 = LinearDiscriminantAnalysis()
    clf2 = GaussianNB()
    clf3 = LogisticRegression()
    clf4 = RandomForestClassifier()
    clf5 = KNeighborsClassifier(n_neighbors=12)
    clf6 = AdaBoostClassifier()
    # x_train,x_test,y_train,y_test = train_test_split(train_X,Y,test_size=0.2) # 对训练集进行划分

    # print x_train.shape
    # print x_test.shape
    # clf.fit(train_X,Y)
    clf = VotingClassifier(estimators=[('la',clf1),('nb',clf2),('lr',clf3),('rf',clf4),('nn',clf5),('ac',clf6)], voting='soft', weights=[1.5,1,1,1,1,1])
    # clf1.fit(x_train,y_train)
    # clf2.fit(x_train,y_train)
    # clf3.fit(x_train,y_train)
    # clf4.fit(x_train,y_train)
    clf.fit(train_X,Y)
    print 'end train classify'

    print 'start classify....'
    # print metrics.classification_report(Y,predict_Y)
    # clf2.fit(train_X,Y)
    # print 'clf2 fited...'
    # clf3.fit(train_X,Y)
    # print 'clf3 fited...'
    # clf4.fit(train_X,Y)
    # print 'clf4 fited...'
    # clf1.fit(train_X,Y)
    # print 'clf1 fited...'
    # 第一个分类结果
    predict_Y = clf.predict(train_X)
    # predict_Y = clf.predict(train_X)
    print 'classify result:'
    print metrics.classification_report(Y,predict_Y)

    predict_Y = clf.predict(test_X)
    # print predict_Y,len(predict_Y)
    print 'end classify...'
    # predict_Y = clf.predict(X[cnt_train:]) # 训练注释这一行,输出测试集打开这一行，注释之后的print metric
    # predict_Y = clf.predict(test_X) # 训练注释这一行,输出测试集打开这一行，注释之后的print metric
    DataFrame(predict_Y,index=rows).to_csv('data/info_test2.csv', header=False)

classify()

# ans = {}
# def test(behavior):
#     Y = pd.read_csv('data/train_Y_%d.csv'%behavior, index_col='user_id')['type']
#     X_train = combine_all_features(behavior)
#     X_test = combine_all_features(behavior,is_train=False)
#     # rows = X_test.index.sort_values().unique()
#     # rows = X_train.index.sort_values().unique()
#     rows = Y.index
#     cnt_train = X_train.shape[0]
#
#     print 'X_train:', X_train.shape
#     print 'X_test:', X_test.shape
#
#     X = X_train.append(X_test)
#     print 'X: ',X.shape
#     X = X_train
#     del X_train
#     del X_test
#     X = TfidfTransformer().fit_transform(X).todense()
#
#     # 先用训练集训练出所有的分类器
#     print 'train behavior %d classify...' % behavior
#     clf = LinearDiscriminantAnalysis()
#     # clf = RandomForestClassifier()
#     # clf.fit(X[:cnt_train],Y)
#
#     x_train,x_test,y_train,y_test = train_test_split(X[:cnt_train],Y,test_size=0.2)
#     clf.fit(x_train,y_train)
#     print x_train.shape
#     print x_test.shape
#     print 'end train classify'
#
#     print 'start classify....'
#     # 第一个分类结果
#     # predict_Y = clf.predict(X[:cnt_train])
#     predict_Y = clf.predict(x_test)
#     # predict_p = clf.predict_proba(x_test)
#     # print predict_p,len(predict_p)
#     print 'behavior %d classify result:' % behavior
#     print metrics.classification_report(y_test,predict_Y)
#     # print predict_Y,len(predict_Y)
#     print 'end classify...'
#     # predict_Y = clf.predict(X[cnt_train:]) # 训练注释这一行,输出测试集打开这一行，注释之后的print metric
#
#     # DataFrame(predict_Y,index=rows).to_csv('data/info_test1.csv', header=False)
#     # for i,y in enumerate(predict_Y[cnt_train:]):
#     for i,y in enumerate(predict_Y):
#         # print i,y
#         key = rows[i]
#         # p = predict_p[i][y-1]
#         if ans.has_key(key):
#             if ans[key].has_key(y):
#                 ans[key][y] += 1
#             else:
#                 ans[key][y] = 1
#         else:
#             ans[key] = {}
#             ans[key][y] = 1
#     # print 'ans:',ans.keys()
# def compare_classify():
#     test(0)
#     test(2)
#     test(3)
#     # 统计最大的概率的类
#     for user_id,d in ans.items():
#         max_y = 0
#         max_p = 0
#         for y in range(12):
#             y += 1
#             if (d.has_key(y)) and (d[y] > max_p):
#                 max_p = d[y]
#                 max_y = y
#         ans[user_id] = max_y
#     Y = pd.read_csv('data/train_Y.csv',index_col='user_id')['type']
#     # print ans.keys()
#     print Y.index
#     Y = Y[ans.keys()]
#     Y = Y.sort_index()
#     print 'after:',Y.index
#     print Y.shape
#     # print ans
#     predict_Y = Series(ans)
#     predict_Y = predict_Y.sort_index()
#     print 'at end classify result:'
#     print metrics.classification_report(Y,predict_Y)
#     print len(predict_Y)
# compare_classify()
# behaviors = [0,2,3]
# for b in behaviors:
#     test(b)
# DataFrame(ans).to_csv('data/info_test1.csv', header=False)



# train_X,Y = load_svmlight_file('data/train_metrix')
# test_X,test_Y = load_svmlight_file('data/test_metrix')
# rows = pd.read_csv('data/log_test2.csv',index_col=0).sort_index().index.unique()
# cnt_train = train_X.shape[0]