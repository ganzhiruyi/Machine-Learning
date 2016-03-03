# -*- coding:utf-8 -*-
__author__ = 'Ganruyi'
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model, metrics
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import SelectPercentile, f_classif
data_dir = 'data/'

def merge_train_data(train_file_name,y_file_name):
    train_file_path = data_dir + train_file_name
    y_file_path = data_dir + y_file_name
    df = pd.read_csv(train_file_path)
    Y = pd.read_csv(y_file_path, names=['user_id','type'])
    df = pd.merge(df,Y,on='user_id')
    df.to_csv('data/merge_train_data.csv')

# merge_train_data('log_train.csv','info_train.csv')
def split_train_data(behavior,is_train=True):
    # behavior 0,2,3,根据action_type把数据分成3类
    if is_train:
        x = pd.read_csv('data/log_train.csv',index_col='user_id')
    else:
        x = pd.read_csv('data/log_test2.csv',index_col='user_id')
    X = x[x['action_type']==behavior]
    idx = X.index.unique()
    idx.sort()
    print 'idx:',idx,len(idx)
    if is_train:
        Y = pd.read_csv('data/train_Y.csv',index_col='user_id')
        Y = Y.ix[idx]
        Y.to_csv('data/train_Y_%d.csv'%behavior)
    if is_train:
        X.to_csv('data/train_%d.csv'%behavior)
    else:
        X.to_csv('data/test2_%d.csv'%behavior)


def get_features(feature_type,behavior,is_train=True):
    # 得到对应的训练/测试集的各种特征
    print 'start get %s features X of behavior %d' % (feature_type,behavior)
    if feature_type == 'seller_id':
        feature = pd.read_csv('data/valid_seller_ids.csv', header=None, index_col=0)[1]
    elif feature_type == 'cat_id':
        feature = pd.read_csv('data/valid_cat_ids.csv', header=None, index_col=0)[1]
    elif feature_type == 'brand_id':
        feature = pd.read_csv('data/valid_brand_ids.csv', header=None, index_col=0)[1]
    else:
        feature = pd.read_csv('data/valid_time_stamp.csv', header=None, index_col=0)[1]
    mp_feature = {}
    for idx,cat in enumerate(feature):
        mp_feature[cat] = idx
    cnt_feature = feature.shape[0]
    if is_train:
        origin_file_name = 'data/train_%d.csv'%behavior
    else:
        origin_file_name = 'data/test2_%d.csv'%behavior
    X = []
    group_by_user_id = pd.read_csv(origin_file_name)[['user_id',feature_type]].groupby(['user_id'])
    features_name = [('%c%d' % (feature_type[0],f)) for f in feature]
    columns = pd.Index(features_name)
    print columns
    user_ids = []
    for user_id,group in group_by_user_id:
        # print 'user_id:',user_id
        x = [0]*cnt_feature
        user_ids.append(user_id)
        features = group[feature_type]
        for i in features.index:
            f = features[i]
            if mp_feature.has_key(f):
                x[mp_feature[f]] += 1
        X.append(x)
    X = DataFrame(X,columns=columns,index=user_ids, dtype='int')
    print 'end get %s features X of behavior %d' % (feature_type,behavior)
    return X
def get_user_feature(feature_type,behavior,num_feature=800):
    X_train = get_features(feature_type,behavior)
    index = X_train.index
    # 对X进行降维
    Y = pd.read_csv('data/train_Y_%d.csv'%behavior, index_col='user_id')['type']
    print 'start selectKbest...'
    # select = SelectKBest(chi2,k=min(num_feature,X_train.shape[1]))
    percent = 0
    if feature_type == 'cat_id':
        percent = 60
    elif feature_type == 'brand_id':
        percent = 15
    elif feature_type == 'seller_id':
        percent = 20
    select = SelectPercentile(f_classif, percentile=percent)
    select.fit(X_train,Y)
    X_train = select.transform(X_train)

    print 'end select...'
    print 'write %s features to train file' % feature_type
    train_feature_file_name = 'data/train_feature_%s_%d.csv' % (feature_type,behavior)
    DataFrame(X_train,index=index).to_csv(train_feature_file_name)

    # 用同样的列降维对应的测试集数据
    X_test = get_features(feature_type,behavior,is_train=False)
    index = X_test.index
    X_test = select.transform(X_test)
    # 写入文件
    print 'write %s features to test file' % feature_type
    test_feature_file_name = 'data/test_feature_%s_%d.csv' % (feature_type,behavior)
    DataFrame(X_test,index=index).to_csv(test_feature_file_name)
    print 'end....'

def train():
    # 得到训练集和测试集对应的三种行为文件
    # split_train_data(0,is_train=True)
    # split_train_data(2,is_train=True)
    # split_train_data(3,is_train=True)
    # split_train_data(0,is_train=False)
    # split_train_data(2,is_train=False)
    # split_train_data(3,is_train=False)
    behaviors = [0,2,3]
    # 得到三种行为关于商品种类的特征矩阵，包括训练集和测试集，且降维
    for b in behaviors:
        get_user_feature('cat_id', b)
    # 得到三种行为关于商品品牌的特征矩阵，包括训练集和测试集，且降维
    # for b in behaviors:
    #     get_user_feature('brand_id', b)
    # # 得到三种行为关于商品卖家的特征矩阵，包括训练集和测试集，且降维
    # for b in behaviors:
    #     get_user_feature('seller_id', b)
train()