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
from sklearn.metrics.pairwise import linear_kernel

from sklearn.preprocessing import scale

import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time


@fun_run_time
def Featurextractor(PSR_Dataset_img, mode = 0):
    '''
    输入：被剪除mask部分的4d图片集，(num, c, h, w),RGB\n
    mode\n
    1=圆形度\n
    2=Hu不变矩\n
    3=欧氏距离探测算子\n
    \n
    输出：特征列表，列表内每个元素都是矩阵。\n
    '''
    print(colorstr('='*50, 'red'))
    print(colorstr('Feature extracting...'))
    #
    num, c, h, w = PSR_Dataset_img.shape

    #特征获取
    if mode == 1:
        #圆形度
        PSR_Dataset_Vectors = get_Vectors(PSR_Dataset_img, fea_circularity)
    elif mode==2:
        #Hu不变矩
        PSR_Dataset_Vectors = get_Vectors(PSR_Dataset_img, fea_hu_moments)
    elif mode==3:
        PSR_Dataset_Vectors = get_Vectors(PSR_Dataset_img, fea_distence_detector, 36)

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
            print(f'\t----{idx+1}/{img_num} has been extracted...')
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
    #
    contours, _ = cv2.findContours(img_cv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    a = cv2.contourArea(contours[0]) * 4 * math.pi      #面积
    b = math.pow(cv2.arcLength(contours[0], True), 2)   #周长
    try:
        Vectors = a / b
    except:
        Vectors = 0
    return np.array(Vectors)

#Hu不变矩
def fea_hu_moments(img_cv):
    '''
    计算Hu不变矩的负对数。输入cv图片，RGB
    '''
    #
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    img_cv[img_cv>0]=255
    #
    moments = cv2.moments(img_cv)   #支持自动转换，非零像素默认为1，计算图像的三阶以内的矩
    humoments = cv2.HuMoments(moments) #计算Hu不变矩
    humoments = humoments[:,0]
    humoments = -np.log10(np.abs(humoments))
    return humoments

# 质心提取
# 欧氏距离探测算子
def fea_distence_detector(img_cv, direct_number = 36):
    '''
    计算质心。
    \n输入cv图片，RGB
    '''
    #
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    img_cv[img_cv>0]=255
    #获取质心
    m = cv2.moments(img_cv)   #支持自动转换，非零像素默认为1，计算图像的三阶以内的矩
    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])

    #计算最大面积区域作为目标区域
    image, contours, hier = cv2.findContours(img_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]   #取得二维最大连通域
    rect = cv2.minAreaRect(contour)                                             #计算区域最小矩形
    box_ = cv2.boxPoints(rect)                                                  #获取坐标
    h = abs(box_[3, 1] - box_[1, 1])
    w = abs(box_[3, 0] - box_[1, 0])
    s_h = h/10  #挪动的次数为10，计算挪动步长
    s_w = w/10
    
    #在目标区域框中遍历81个位置，获取特征向量(1, direct_number)
    Vectors = []
    for i in range(9):              #
        for j in range(9):
            h_ = int(cy + s_h*(i-4))
            w_ = int(cx + s_w*(i-4))
            if img_cv[h_, w_] != 0: #仅在区域内点处计算
                
                Vector = np.zeros((direct_number))
                # 获取各个方向值。
                for idx, theta in enumerate(range(direct_number)):  
                    #在原图划线=0，两图片相减，得到该角度的线
                    theta = theta / direct_number * 2 * math.pi
                    r = 300
                    tempx = int(r * math.cos(theta)) + 300
                    tempy = int(r * math.sin(theta)) + 300
                    
                    ptEnd = 
                    
                    img_cv_lined = cv2.line(CIB_mask, (300, 300), ptEnd, 0, thickness=1, lineType=8)
                    line = img_cv - img_cv_lined        #只有一条线的图片
                    Vector[idx] = np.sum(line)
                
                #处理特征向量
                Vectors.append(Vector)


    #获取同心等距放射线模板(Concentric Isometric Beam),401*401
    #for theta in range(direct_number):
    #    theta = theta / direct_number * 2 * math.pi
    #    r = 300
    #    tempx = int(r * math.cos(theta)) + 300
    #    tempy = int(r * math.sin(theta)) + 300
    #    ptEnd = (tempx, tempy)
    #    CIB_mask = cv2.line(CIB_mask, (300, 300), ptEnd, 255, thickness=1, lineType=8)
    #CIB_mask = CIB_mask[100:501, 100:501]
    #u_idsip.show_pic(CIB_mask)
    
    return np.array(Vectors)
# 归一化

# 边缘提取
# 链码提取
# 傅里叶描述子

# 细化骨架提取
# 图论统计量