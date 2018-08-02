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
# 距离计算公式
get_distance = utils.get_distance

# 导入users和events的index索引, 以及相关信息
users_index = load(tmp_dpath+'users_index.joblib.gz')
events_index = load(tmp_dpath+'events_index.joblib.gz')
all_user = set(users_index.keys())
all_event = set(events_index.keys())
num_users = len(users_index)
num_events = len(events_index)

print('getting events information')
# 取出在train/test中出现的所有events信息
with open(dpath+'events.csv') as events:
    # 读入列名信息
    columns = events.readline().split(',')
    event_train_test_df=[]
    for line in events:
        # 若读入的event_id在train/test中出现, 
        # 添加入event_train_test_df
        cols = line.split(',') 
        if int(cols[0]) in all_event:  
            event_train_test_df.append(cols)

print('to DataFrame')
# 把生成好的event_train_test_df 转成DataFrame
event_train_test = pd.DataFrame(event_train_test_df,columns=columns)
# 把空缺值替换为np.nan
event_train_test.replace('',np.nan,inplace=True)
# 把event_id转换成index
event_train_test.index = event_train_test.pop('event_id').apply(lambda x:events_index[int(x)])
# 把user_id转换成int64
event_train_test['user_id'] = event_train_test['user_id'].astype(np.int64)
# 把'start_time'转换成日期类型
event_train_test['start_time'] = event_train_test['start_time'].astype(np.datetime64).dt.date
# 把'c_other\n'中所有的'\n'去掉
event_train_test['c_other'] = event_train_test.pop('c_other\n').apply(lambda x:x[:-1]).values

print('Encoding . . . ')
# 对类别特征进行LabelEncoder
for c in ['city', 'state', 'country', 'lat', 'lng']:
    event_train_test[c] = event_train_test[c].astype('category').values.codes
# 对'c_1', 'c_2', 'c_100', ***, 'c_other'进行归一化处理
for c in event_train_test.columns[-101:]:
    event_train_test[c] = event_train_test[c].astype(np.float64)
    min_ = event_train_test[c].min()
    max_ = event_train_test[c].max()
    length_ = max_ - min_
    event_train_test[c] = (event_train_test[c] - min_) / length_

print('getting the distance')
# 生成数值型和类别型数据
cat = event_train_test.loc[:, ['city', 'state', 'country', 'lat', 'lng']]
num = event_train_test.loc[:, 'c_1':]
# 生成空的distance, 默认为1
event_distance = np.ones((num_events, num_events), dtype=np.float64)
# 生成event_index
e_index = event_train_test.index
# 生成1/10的界限, 用于输出进度
num_events010 = num_events//10
for i,e1 in enumerate(e_index):
    # 显示进度
    if i < num_events010: print(i)
    # 对角线距离为0
    event_distance[e1,e1] = 0
    for e2 in e_index[i+1:]:
        # 计算距离, 并对称赋值
        dis = get_distance(e1,e2,num,cat)
        event_distance[e1,e2] = dis
        event_distance[e2,e1] = dis
        #print(dis)

print('saving . . . ')
# 保存处理好的event_distance
dump(event_distance, tmp_dpath+'event_distance.joblib.gz', compress=('gzip',3))



