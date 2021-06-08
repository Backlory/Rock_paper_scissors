# 特征提取。（pic1->vec_list)


# ROI区域提取。（pic3->pic1黑白)
# CV图片通道在第四位，平时numpy都放在第二位的
# 预处理部分。（pic3->pic3)
import math
from os import replace
from sys import api_version
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import median, place

from sklearn.preprocessing import scale

import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time


@fun_run_time
def Featurextractor(PSR_Dataset_img, mode = 0):
    '''
    输入：被剪除mask部分的4d图片集，(num, c, h, w),RGB

    输出：特征列表，列表内每个元素都是矩阵。
    '''
    print(colorstr('='*50, 'red'))
    print(colorstr('Feature extracting...'))
    #
    num, c, h, w = PSR_Dataset_img.shape

    #特征获取
    if mode == 0:
        #圆形度
        PSR_Dataset_Vectors = get_Vectors(PSR_Dataset_img, fea_circularity)
    elif mode==1:
        #Hu不变矩
        PSR_Dataset_Vectors = get_Vectors(PSR_Dataset_img, fea_hu_moments)
    #处理结束
    return PSR_Dataset_Vectors


def get_Vectors(imgs, func, **kwargs):
    '''
    输入图片组和函数、参数字典，输出函数结果
    '''
    t=tic()
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    img_num = len(imgs)
    #
    result = []
    for idx, img in enumerate(imgs):
        temp = func(img, **kwargs)
        result.append(temp)
        if idx % int(img_num/10) == int(img_num/10)-1:
            print(f'\t----{idx+1}/{img_num} has been preprocessed...')
    #
    #result = np.array(result) #对于同长度向量而言可以转化为array，对于sift等视觉词特征则不行
    toc(t, func.__name__, img_num)
    return result


# TODO: 

# 圆形度计算
def fea_circularity(img_cv):
    '''
    计算圆形度。输入cv图片，RGB
    '''
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    img_cv[img_cv>0]=255
    contours, _ = cv2.findContours(img_cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a = cv2.contourArea(contours[0]) * 4 * math.pi      #面积
    b = math.pow(cv2.arcLength(contours[0], True), 2)   #周长
    try:
        Vectors = a / b
    except:
        Vectors = 0
    return Vectors

#Hu不变矩
def fea_hu_moments(img_cv):
    '''
    计算Hu不变矩。输入cv图片，RGB
    '''
    #
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(img_cv)   #支持自动转换，非零像素默认为1，计算图像的三阶以内的矩
    humoments = cv2.HuMoments(moments) #计算Hu不变矩
    humoments = np.log10(np.abs(humoments))
    return humoments

# 质心提取
# 欧氏距离探测算子
# 归一化

# 边缘提取
# 链码提取
# 傅里叶描述子

# 细化骨架提取
# 图论统计量