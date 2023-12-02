# -*- coding: utf-8 -*-
"""
使用平台及版本：
    two-stage-Tradaboost: Python 3.8.12 64-bit | Qt 5.9.7 | PyQt5 5.9.2 | Windows 10 
    012预测: Python 3.7.6| Windows 10
    合并属性: Python 3.7.6| Windows 10
    合并文件: Python 3.7.6| Windows 10
    简单的机器学习: Python 3.7.6| Windows 10
    添加label: Python 3.7.6| Windows 10
    lasso降维 R4.0.4 64-bit | Windows 10
    过采样:Python 3.9.5 64-bit | Qt 5.9.6 | PyQt5 5.9.2 | Windows 10
深度学习框架：ELM，极限学习机
"""
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import tree
import os
import math

from sklearn import svm

path = "D:\\学习之都\\大三上\\机器学习与大数据\\大作业\\小测试\\少维度" # 设置工作目录


# H 测试样本分类结果
# TrainS 原训练样本 np数组,行数为m
# TrainA 辅助训练样本,行数为n
# LabelS 原训练样本标签
# LabelA 辅助训练样本标签
# Test  测试样本
# N 迭代次数

def tradaboost(trans_S, label_S, test, N):
    #trans_label = np.concatenate((label_A, label_S), axis=0)
    

    # calculate number of rows
    row_S = trans_S.shape[0]
    row_T = test.shape[0]
    #print(trans_data.shape,test.shape)

    #test_data = np.concatenate((trans_data, test), axis=0)
    '''
    # 初始化权重
    weights_A = np.ones([row_A, 1]) / row_A #row number is A, col number is 1
    weights_S = np.ones([row_S, 1]) / row_S
    weights = np.concatenate((weights_A, weights_S), axis=0)

    #bata
    bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))

    # 存储每次迭代的标签和bata值？
    bata_T = np.zeros([1, N])
    result_label = np.ones([row_A + row_S + row_T, N])
    '''
    predict = np.zeros([row_T])

    # initial finished.
    # row array
    #trans_data = np.asarray(trans_data, order='C')
    #trans_label = np.asarray(trans_label, order='C')
    #test_data = np.asarray(test_data, order='C')
    for h in range(0,len(list_csv)):
        #if h == 5:
            #break
        train_a = pd.read_csv(list_csv[h],sep=',')
        transa = np.asarray(train_a.iloc[:1000000,0:39],order = 'C')
        labela = np.asarray(train_a.iloc[:1000000,[-1]],order = 'C')
        trans_data = np.concatenate((transa, trans_S), axis=0)
        trans_label = np.concatenate((labela, label_S), axis=0)
        row_A = transa.shape[0]
        print(row_A)
        print(h)

        test_data = np.concatenate((trans_data, test), axis=0)
        if h == 0:
            # 初始化权重
            weights_A = np.ones([row_A, 1]) / row_A #row number is A, col number is 1
            weights_S = np.ones([row_S, 1]) / row_S
            weights = np.concatenate((weights_A, weights_S), axis=0)
            #bata
            #bata = 1 / (1 + np.sqrt(2 * np.log(row_A / N)))
            # 存储每次迭代的标签和bata值？
            bata_T = np.zeros([1, N])
            result_label = np.ones([row_A + row_S + row_T, N])
        # initial finished.
        # row array
        trans_data = np.asarray(trans_data, order='C')
        trans_label = np.asarray(trans_label, order='C')
        test_data = np.asarray(test_data, order='C')
        
        if trans_data.shape[0] != weights.shape[0]:
            continue
        for i in range(N):
            P = calculate_P(weights, trans_label)
    
            result_label[:, i] = train_classify(trans_data, trans_label,
                                                test_data, P)
            #print(result_label[row_A:row_A + row_S, i])
            #print(label_S)
    
            g_mean = calculate_g_mean(label_S, result_label[row_A:row_A + row_S, i])
                                              
            print('g_mean:', g_mean)
            error_rate = 1 - g_mean
            if error_rate > 0.5:
                error_rate = 0.5
            if error_rate == 0:
                N = i
                break  # 防止过拟合
                # error_rate = 0.001
    
            bata_T[0, i] = error_rate / (1 - error_rate)
    
            # 调整源域样本权重
            Zt = row_S / (row_S + row_T) + (i + 1) / (N - 1) * (1 - row_S / (row_S + row_T))
            for j in range(row_S):
                weights[row_A + j] = weights[row_A + j] / Zt
    
            # 调整辅域样本权重
            for j in range(row_A):
                weights[j] = weights[j] * np.power(bata_T[0, i], error_rate)/ Zt
            
    # print bata_T
    sum1 = 0
    sum0 = 0
    for i in range(row_T):
        # 跳过训练数据的标签
        left = np.sum(
            result_label[row_A + row_S + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
        right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))

        if left >= right:
            predict[i] = 1
            sum1 = sum1 + 1 
        else:
            predict[i] = 0
            sum0 = sum0 + 1
    print("predict",predict)
    print(sum0,sum1)
    predict = np.asarray(predict)
    np.savetxt('predict.csv', predict, delimiter = ',')
    return predict


def calculate_P(weights, label):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


def train_classify(trans_data, trans_label, test_data, P):
    clf = tree.DecisionTreeClassifier(criterion="gini", splitter="random", max_depth = 500) #防止过拟合
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    return clf.predict(test_data)    
    


def calculate_g_mean(label_R, label_H):
    FN = 0
    FP = 0
    TP = 0
    tr = np.transpose(label_R)    
    
    for z in range(tr.shape[1]):
        if (tr[0,z] == 1) & (label_H[z] == 1):
            TP = TP + 1
        if (tr[0,z] == 1) & (label_H[z] == 0):
            FN = FN + 1
        if (tr[0,z] == 0) & (label_H[z] == 1):
            FP = FP + 1
           
    #total = np.sum(weight)

    #print(weight[:, 0] / total)
    #print(label_R)
    #print(label_H)
    #print(np.abs(np.transpose(label_R) - label_H))
    #return np.sum(weight[:, 0] / total * np.abs(np.transpose(label_R) - label_H))
    return math.sqrt(TP ** 2 / (TP + FN + 0.0000000001) / (TP + FP + 0.0000000001)) # avoid equiling 0


def list_dir(file_dir):
    list_csv = []
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = os.path.join(file_dir,cur_file)
        #判断是文件夹还是文件
        if os.path.isfile(path):
            # print("{0} : is file!".format(cur_file))
            dir_files = os.path.join(file_dir, cur_file)
        #判断是否存在.csv文件，如果存在则获取路径信息写入到list_csv列表中
        if os.path.splitext(path)[1] == '.csv':
            csv_file = os.path.join(file_dir, cur_file)
            # print(os.path.join(file_dir, cur_file))
            # print(csv_file)
            list_csv.append(csv_file)
        if os.path.isdir(path):
            # print("{0} : is dir".format(cur_file))
            # print(os.path.join(file_dir, cur_file))
            list_dir(path)
    return list_csv
   
list_csv = list_dir(path)
list_csv = np.array(list_csv)


#train_a = pd.read_csv(r'D:\学习之都\大三上\机器学习与大数据\大作业\数据\tradaboost\others.csv',sep=',')
train_s = pd.read_csv(r'D:\学习之都\大三上\机器学习与大数据\大作业\小测试\012_new.csv',sep=',')
Test = pd.read_csv(r'D:\学习之都\大三上\机器学习与大数据\大作业\小测试\012no_new.csv',sep=',')
transs = np.asarray(train_s.iloc[:,0:39],order = 'C')
labels = np.asarray(train_s.iloc[:,[-1]],order = 'C')
#transa = np.asarray(train_a.iloc[:,0:75],order = 'C')
#labela = np.asarray(train_a.iloc[:,[-1]],order = 'C')
test_ = np.asarray(Test.iloc[:,0:39],order = 'C')
num = 3
tradaboost(transs, labels, test_, num)