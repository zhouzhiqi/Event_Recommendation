import numpy as np
import pandas as pd

import scipy.sparse as ss
from scipy.spatial.distance import jaccard, cosine

from sklearn.externals.joblib import dump, load

# 数据路径
dpath = '../data/'
# 缓存数据路径
tmp_dpath = '../tmp_data/'
# 数据类型
data_types = {'user':np.int64,'event':np.int64,'invited':np.int8,'interested':np.int8,'not_interested':np.int8,}


def get_distance(i1, i2, number, category):
    """计算距离
    
    分别计算i1,i2之间的cosine和jaccard距离

    i1,i2:  索引ID
    number: DataFrame, 数值型数据, 且经过归一化
    category: DataFrame, 类别数据, 且经过LabeEncoder

    return: 加权后的距离
    """
    # 计算数值型的cosine距离
    cos_sim = cosine(number.loc[i1,:], number.loc[i2,:])
    # 计算类别型的jaccard距离
    jac_sim = jaccard(category.loc[i1,:], category.loc[i2,:])
    return cos_sim*0.6 + jac_sim*0.4

def user_event_and(i1, i2, opt, all_u_e):
    """判断交集
    
    opt=='e': 返回参加两个event的共同的users
    opt=='u': 返回两个user共同参加的event

    i1, i2: 索引值
    opt: {'e','u'}
    all_u_e: user <-> event 稀疏矩阵
    """
    if opt == 'u': # u->e
        data = all_u_e
    elif opt == 'e': # e->u
        data = all_u_e.to
    # 各自的集合
    sam1 = set(data[i1,:].getrows())
    sam2 = set(data[i2,:].getrows())
    # 取交集
    all_atd = sam1 & sam2
    return all_atd

def normalization(data):
    """归一化 pd.Series"""
    min_ = data.min()
    max_ = data.max()
    return (data - min_) / (max_ - min_)

def label_encoder(data):
    """LabelEncoder pd.Series"""
    return data.astype('category').values.codes
