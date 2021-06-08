#特征编码（特征vec->分类vec，结构规整)
import math
from os import replace
from sys import api_version
import cv2
import random
import numpy as np

from sklearn.preprocessing import OneHotEncoder

import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time

@fun_run_time
def Featurencoder(datas, labels, mode = 0, onehot=False):
    '''
    输入：
    datas=N个元素的特征列表，每个元素代表一幅图的特征值矩阵
    labels=N个元素的标签矩阵,numpy，1维

    输出：X_dataset,  Y_dataset，代表训练集向量，N个*m维特征矩阵，N个*K类的二维独热编码
    '''
    print(colorstr('='*50, 'red'))
    print(colorstr('Feature encoding...'))
    #
    N = len(datas)
    assert(N == len(labels))

    #X_dataset
    X_dataset=0
    if mode == 0:
        #直接输出
        X_dataset = np.array(datas)
    elif mode==1:
        #
        pass

    #Y_dataset
    if onehot:
        ohe = OneHotEncoder()
        labels = labels[:, np.newaxis]
        ohe.fit(labels)
        Y_dataset = ohe.transform(labels).toarray()
    else:
        Y_dataset = labels

    #处理结束
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