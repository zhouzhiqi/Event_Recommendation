import numpy as np
import pandas as pd

import scipy.sparse as ss
from scipy.spatial.distance import jaccard, cosine

from sklearn.externals.joblib import dump, load

import utils

# 数据路径
dpath = utils.dpath
# 数据类型
data_types = utils.data_types
# 缓存数据路径
tmp_dpath = utils.tmp_dpath

# 读入训练数据
train = pd.read_csv(dpath+'train.csv',dtype=data_types, index_col=['timestamp'])
# 以时间类型数据为index
train.index = train.index.astype(np.datetime64)
# 重新定义index为int型(下面必需用int型)
train.index = np.arange(train.shape[0])
# 读入测试数据
test = pd.read_csv(dpath+'test.csv',dtype=data_types, index_col=['timestamp'])
# 以时间类型数据为index
test.index = test.index.astype(np.datetime64)

# train和test中所有的users
users_trte = set(train.user) | set(test.user)
# train和test中所有的events
events_trte = set(train.event) | set(test.event)
num_users = len(users_trte)
num_events = len(events_trte)

# 生成users索引
users_index = {u:i for u,i in zip(users_trte, range(num_users))}
# 生成events索引
events_index = {e:i for e,i in zip(events_trte, range(num_events))}

# 保存users索引
dump(users_index, tmp_dpath+'users_index.joblib.gz', compress=('gzip',3))
# 保存events索引
dump(events_index, tmp_dpath+'events_index.joblib.gz', compress=('gzip',3))

# 生成users对events的打分{-1,0,1}
train['scores'] = train.loc[:,'interested'] - train.loc[:,'not_interested']
# 生成全是0的稀疏矩阵
user_event_scores = ss.lil_matrix((num_users, num_events), dtype=np.int8)
# 把打分数据填入稀疏矩阵
for i in train.index:
    u = users_index[train.loc[i,'user']]
    e = events_index[train.loc[i,'event']]
    user_event_scores[u,e] = train.loc[i,'scores']

# 保存打分矩阵
dump(user_event_scores, tmp_dpath+'user_event_scores.joblib.gz', compress=('gzip',3))


