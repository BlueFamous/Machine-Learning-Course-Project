# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 20:02:35 2020

@author: cw817615
"""

import pandas as pd

'''读取数据'''
io = r'C:\Users\HP\Desktop\551.xlsx'
data = pd.read_excel(io,sheet_name=0,header=None)


import numpy as np
#

X = data.ix[:,0:39].values     # 自变量
y = data.ix[:,39].values     # 因变量
##'''SMOTE的改进：Borderline-SMOTE处理过采样'''
#from imblearn.under_sampling import ClusterCentroids
#cc = ClusterCentroids(random_state=0)
#X_resampled, y_resampled = cc.fit_sample(X, y)

#from imblearn.over_sampling import RandomOverSampler

#ros = RandomOverSampler(random_state=0)
#X_resampled, y_resampled = ros.fit_sample(X, y)

from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
print(X_resampled)
print(y_resampled)
# 合并数据
data_resampled = np.zeros([len(X_resampled[:,0]),40])
data_resampled[:,:40] = X_resampled
data_resampled[:,39] = y_resampled

data_resampled2 = pd.DataFrame(data_resampled)
writer = pd.ExcelWriter(r'C:\Users\HP\Desktop\999.xlsx')#创建数据存放路径
data_resampled2.to_excel(writer)
writer.save()#文件保存
writer.close()#文件关闭
