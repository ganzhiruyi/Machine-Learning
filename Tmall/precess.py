# -*- coding:utf-8 -*-
__author__ = 'Ganruyi'
import pandas as pd
from pandas import DataFrame,Series
def erase_invalid_user(is_train = True):
    if is_train:
        file_name = 'data/log_train.csv'
        index_set_name = 'data/train_valid_index_set.csv'
    else:
        file_name = 'data/log_test2.csv'
        index_set_name = 'data/test_valid_index_set.csv'
    x = pd.read_csv(file_name)
    x = x.groupby(['user_id','action_type']).size()
    x = x.unstack().fillna(0)

    index_set = x.index
    index_set = index_set.difference(x[x[0]*2<=x[3]].index)
    index_set = index_set.difference(x[x[0]*2<=x[2]].index)
    index_set = index_set.difference(x[x[0]<10].index)
    index_set = index_set.difference(x[x[0]>800].index)

    pd.DataFrame(index_set, index=range(index_set.size)).to_csv(index_set_name, header=None)

def get_sex_type():
    file_name = 'data/info_train.csv'
    y = pd.read_csv(file_name,header=None,index_col=0)
    male_id = y[y[1]<7].index
    m = DataFrame([0]*male_id.size,index=male_id,columns=['sex'])
    female_id = y[y[1]>6].index
    f = DataFrame([1]*female_id.size,index=female_id,columns=['sex'])
    m.append(f).to_csv('data/train_sex.csv')

def erase_invalid_time_stamp():
    train_file_name = 'data/log_train.csv'
    test_file_name = 'data/log_test2.csv'
    valid_cat_name = 'data/valid_time_stamp.csv'
    x = pd.read_csv(train_file_name).groupby('time_stamp').size()
    y = pd.read_csv(test_file_name).groupby('time_stamp').size()
    print x.shape
    print y.shape
    x = x.index.union(y.index)
    print 'time_stamp:',x.shape
    DataFrame(x).to_csv(valid_cat_name,header=None)

def erase_invalid_brand():
    train_file_name = 'data/log_train.csv'
    test_file_name = 'data/log_test2.csv'
    valid_brand_name = 'data/valid_brand_ids.csv'
    x = pd.read_csv(train_file_name).groupby('brand_id').size()
    # y = pd.read_csv(test_file_name).groupby('brand_id').size()
    # print x.shape,y.shape
    x = x[x>50].index
    # y = y[y>150].index
    # print x
    # print y
    # x = x.union(y)
    # x = x.index.union(y.index)
    print 'brand:',x.shape
    DataFrame(x).to_csv(valid_brand_name,header=None)

def erase_invalid_category():
    train_file_name = 'data/log_train.csv'
    test_file_name = 'data/log_test2.csv'
    valid_cat_name = 'data/valid_cat_ids.csv'
    x = pd.read_csv(train_file_name).groupby('cat_id').size()
    # y = pd.read_csv(test_file_name).groupby('cat_id').size()
    x = x[x>10].index
    # y = y[y>10].index
    # print x
    # print y
    # x = x.union(y)
    # x = x.index.union(y.index)
    print 'category:',x.shape
    DataFrame(x).to_csv(valid_cat_name,header=None)

def erase_invalid_seller():
    train_file_name = 'data/log_train.csv'
    test_file_name = 'data/log_test2.csv'
    valid_cat_name = 'data/valid_seller_ids.csv'
    x = pd.read_csv(train_file_name).groupby('seller_id').size()
    # y = pd.read_csv(test_file_name).groupby('seller_id').size()
    # print x.shape
    # print y.shape
    x = x[x>10].index
    # y = y[y>10].index
    # print x
    # print y
    # x = x.union(y)
    # x = x.index.union(y.index)
    print 'seller:',x.shape
    DataFrame(x).to_csv(valid_cat_name,header=None)


erase_invalid_seller()
erase_invalid_brand()
erase_invalid_category()
# erase_invalid_time_stamp()