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
def Featurencoder(datas, labels, mode = 0):
    '''
    输入：
    datas=特征列表，列表内num个元素，每个元素代表一幅图的特征值矩阵
    labels=标签numpy矩阵

    输出：X_dataset,  Y_dataset，代表训练集向量，N个*m维特征矩阵，N个*K类独热编码
    '''
    print(colorstr('='*50, 'red'))
    print(colorstr('Feature encoding...'))
    #
    num = len(datas)
    assert(num == len(labels))

    #X_dataset
    X_dataset=0
    if mode == 0:
        #圆形度
        pass
    elif mode==1:
        #Hu不变矩
        pass

    #Y_dataset
    ohe = OneHotEncoder()
    ohe.fit(labels)
    Y_dataset = ohe.transform(labels)
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