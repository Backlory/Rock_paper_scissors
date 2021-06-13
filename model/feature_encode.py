#特征编码（特征vec->分类vec，结构规整)
import math
from os import replace
from sys import api_version
import cv2
import random
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
        
import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time

#@fun_run_time
def Featurencoder(datas_list, labels, mode = 0, onehot=False, display=True):
    '''
    输入：
    datas=N个元素的特征列表，每个元素代表一幅图的特征值矩阵
    labels=N个元素的标签矩阵,numpy，1维
    mode= normal
    输出：X_dataset,  Y_dataset，代表训练集向量，N个*m维特征矩阵，N个*K类的二维独热编码
    '''
    if display:
        print(colorstr('='*50, 'red'))
        print(colorstr('Feature encoding...'))
    #
    N = len(datas_list)
    assert(N == len(labels))

    #X_dataset
    if mode == 'normal':
        #直接输出
        X_dataset = np.array(datas_list)
        if len(X_dataset.shape) == 1:
            X_dataset = X_dataset[:, np.newaxis]
    elif mode=='bagofword':
        #词袋模型
        word_num = 500
        #生成词袋
        word_bag = datas_list[0]   #视觉词袋，m*36
        for data in datas_list[1:]:
            word_bag = np.concatenate((word_bag, data), axis=0) 
        #训练词典
        word_dict = KMeans(n_clusters=word_num,verbose=1) #视觉词典，容量500
        word_dict.fit(word_bag)
        #编码转化，视觉词统计
        X_dataset = np.zeros((N, word_num))
        for idx, data in enumerate(datas_list):
            words = word_dict.predict(data)
            for word in words:
                X_dataset[idx, word] += 1
        X_dataset = np.array(X_dataset)
    #Y_dataset
    if onehot:
        ohe = OneHotEncoder()
        labels = labels[:, np.newaxis]
        ohe.fit(labels)
        Y_dataset = ohe.transform(labels).toarray()
    else:
        Y_dataset = labels

    #处理结束，得到二维特征矩阵，每行一个图，每列一个特征
    assert(len(X_dataset.shape) == 2)
    return X_dataset,  Y_dataset


















# TODO:
# 生成词袋字典K=300
# 词袋特征映射

# 离群值剔除（方差）

# PCA降维

# 独热编码转换(不一定用)

# 自编码器？

# 归一化(不改变分布)
# 标准化(改变分布)