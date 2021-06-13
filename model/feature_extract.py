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


#@fun_run_time
def Featurextractor(PSR_Dataset_img, mode = 0, display=True):
    '''
    输入：被剪除mask部分的4d图片集，(num, c, h, w),RGB\n
    mode\n
    Round =圆形度\n
    Hu =Hu不变矩\n
    distence_detector =欧氏距离探测算子\n
    \n
    输出：特征列表，列表内每个元素都是矩阵。\n
    '''
    if display:
        print(colorstr('='*50, 'red'))
        print(colorstr('Feature extracting...'))
    #
    num, c, h, w = PSR_Dataset_img.shape

    #特征获取
    if mode == 'Round':
        #圆形度
        PSR_Dataset_Vectors = get_Vectors(PSR_Dataset_img, fea_circularity)
    elif mode=='Hu':
        #Hu不变矩
        PSR_Dataset_Vectors = get_Vectors(PSR_Dataset_img, fea_hu_moments)
    elif mode == 'Round_Hu':
        temp1 = get_Vectors(PSR_Dataset_img, fea_circularity)
        temp1 = np.array(temp1)
        temp1 = temp1[:, np.newaxis]
        #
        temp2 = get_Vectors(PSR_Dataset_img, fea_hu_moments)
        temp2 = np.array(temp2)
        #
        PSR_Dataset_Vectors = np.concatenate((temp1, temp2), axis=1)
        assert(PSR_Dataset_Vectors.shape[1]==8)
        PSR_Dataset_Vectors = PSR_Dataset_Vectors.tolist()
    elif mode=='distence_detector':
        direct_number = 36
        CIB_masks = get_CIB_masks(direct_number)
        PSR_Dataset_Vectors = get_Vectors(  PSR_Dataset_img,
                                            fea_distence_detector,
                                            direct_number = direct_number,
                                            CIB_masks=CIB_masks)
    elif mode=='fourier':
        PSR_Dataset_Vectors = get_Vectors(PSR_Dataset_img, fea_fourier)
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
        if img_num>10:
            if idx % int(img_num/10) == int(img_num/10)-1:
                print(f'\t----{idx+1}/{img_num} has been extracted...')
    #
    #result = np.array(result) #对于同长度向量而言可以转化为array，对于sift等视觉词特征则不行
    #toc(t, func.__name__, img_num)
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
    contours = sorted(contours, key = cv2.contourArea, reverse=True)#按面积排序
    a = cv2.contourArea(contours[0]) * 4 * math.pi      #面积
    b = math.pow(cv2.arcLength(contours[0], True), 2)   #周长
    try:
        Vector = a / b
    except:
        Vector = 0
    return np.array(Vector)

#Hu不变矩
def fea_hu_moments(img_cv):
    '''
    计算Hu不变矩的负对数。输入cv图片，RGB
    '''
    #
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    #img_cv[img_cv>0]=255
    #
    moments = cv2.moments(img_cv)   #支持自动转换，非零像素默认为1，计算图像的三阶以内的矩
    humoments = cv2.HuMoments(moments) #计算Hu不变矩
    humoments = humoments[:,0]
    humoments = -np.log10(np.abs(humoments))
    return humoments


#获取同心等距放射线模板组(Concentric Isometric Beam),direct_number个，每个为601*601
def get_CIB_masks(direct_number):
    CIB_masks = []
    for theta in range(direct_number):
        CIB_mask = np.zeros((601,601), dtype=np.uint8)
        theta = theta / direct_number * 2 * math.pi
        r = 600
        tempx = int(r * math.cos(theta)) + 300
        tempy = int(r * math.sin(theta)) + 300
        ptEnd = (tempx, tempy)
        CIB_mask = cv2.line(CIB_mask, (300, 300), ptEnd, 255, thickness=1, lineType=8)
        CIB_masks.append(CIB_mask)  #[100:501, 100:501]
    return CIB_masks

# 欧氏距离探测算子
def fea_distence_detector(img_cv, direct_number = 36, CIB_masks = None):
    '''
    计算质心。
    \n输入三通道cv图片，RGB
    '''
    #
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)       #3d-->2d
    img_cv[img_cv>0]=255
    ori_h, ori_w = img_cv.shape

    #获取质心
    m = cv2.moments(img_cv)   #支持自动转换，非零像素默认为1，计算图像的三阶以内的矩
    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])

    #计算最大面积区域作为目标区域
    contours, hier = cv2.findContours(img_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]   #取得二维最大连通域
    x_leftop, y_leftop, area_w, area_h = cv2.boundingRect(contour)                         #计算区域最小正矩形（左上角点坐标，宽，高）
    s_h = (area_h/10)-1  #挪动的次数为10，计算挪动步长
    s_w = (area_w/10)-1
    
    #在目标区域框中遍历81个位置，获取特征向量(1, direct_number)
    Vectors = []
    for i in range(9):              #
        for j in range(9):
            h_ = int(cy + s_h*(i-4))
            w_ = int(cx + s_w*(i-4))
            if img_cv[h_, w_] != 0: #仅在区域内点处计算

                # 利用获取各个方向值。
                Vector = np.zeros((direct_number))
                for idx, theta in enumerate(range(direct_number)):  
                    
                    CIB_mask = CIB_masks[idx] #以300,300为中心，变为以
                    CIB_mask = CIB_mask[300-h_:300-h_+ori_h, 300-w_:300-w_+ori_w]
                    lined = CIB_mask * img_cv
                    
                    ##获得目标方向的终点坐标
                    #theta = theta / direct_number * 2 * math.pi
                    #r = 300
                    #tempx = int(r * math.cos(theta)) + w_
                    #tempy = int(r * math.sin(theta)) + h_
                    ##划线，作差得到，得到目标方向的相应
                    #img_cv_lined = img_cv.copy()
                    #img_cv_lined = cv2.line(img_cv_lined, (w_, h_), (tempx, tempy), 0, thickness=1, lineType=8)
                    #lined = cv2.subtract(img_cv, img_cv_lined)        #只有一条线的图片

                    #加和得到特征的值
                    Vector[idx] = np.sum(lined/255.0)   #
                #print(Vector)
                #处理特征向量
                
                Vectors.append(Vector)
    
    #对特征处理，以矩阵形式
    Vectors = np.array(Vectors) #n*36
    Vectors_diff = Vectors - Vectors[:, [x-1 for x in range(direct_number)]] #循环一阶后向差分

    Vectors_2 = Vectors + Vectors[:,[x-1 for x in range(direct_number)]] \
                        + Vectors[:,[x-2 for x in range(direct_number)]] \
                        + Vectors[:,[x-3 for x in range(direct_number)]] \
                        + Vectors[:,[x-4 for x in range(direct_number)]] #移动累加，仅用于获取起点
    idxs = np.argmax(Vectors_2, axis=1)
    for i, idx in enumerate(idxs):
        temp = Vectors[i]
        temp = temp[[x-idx for x in range(direct_number)]]  #转顺序
        temp = (temp-np.min(temp)) / (np.max(temp) - np.min(temp))  #归一化
        Vectors[i] = temp

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
def fea_fourier(img_cv, MIN_DESCRIPTOR = 32):
    '''
    计算f傅里叶描述子。输入cv图片，RGB
    输出长度为MIN_DESCRIPTOR的傅里叶描述子向量(如32*1)
    '''

    def truncate_descriptor(fourier_result, MIN_DESCRIPTOR=32):
        '''
        截断傅里叶描述子。对于图像，shift后，取中间的MIN_DESCRIPTOR 项描述子，再变回来\n
        输入：傅里叶描述子\n
        输出：截断保留中间后的傅里叶描述子\n
        '''
        descriptors_fftshift = np.fft.fftshift(fourier_result)
        temp = int(len(descriptors_fftshift) / 2)
        low, high = temp - int(MIN_DESCRIPTOR / 2), temp + int(MIN_DESCRIPTOR / 2)
        descriptors_fftshift = descriptors_fftshift[low:high]
        descriptors_fftshift = np.fft.ifftshift(descriptors_fftshift)
        return descriptors_fftshift

    def reconstruct(img, descirptor):
        '''
        由傅里叶描述子重建轮廓图\n
        输入：图像(只用了size)和傅里叶描述子\n
        输出：重建好的图片\n
        '''
        #descirptor = truncate_descriptor(fourier_result, degree)
        #descirptor = np.fft.ifftshift(fourier_result)
        #descirptor = truncate_descriptor(fourier_result)
        #print(descirptor)
        contour_reconstruct = np.fft.ifft(descirptor)
        contour_reconstruct = np.array([contour_reconstruct.real,       #虚部实部映射到图片中
                                        contour_reconstruct.imag])
        contour_reconstruct = np.transpose(contour_reconstruct)     #转置
        contour_reconstruct = np.expand_dims(contour_reconstruct, axis = 1) #添加轴
        #归一化？
        if contour_reconstruct.min() < 0:
            contour_reconstruct -= contour_reconstruct.min()
        contour_reconstruct *= img.shape[0] / contour_reconstruct.max()
        contour_reconstruct = contour_reconstruct.astype(np.int32, copy = False)
        #根据归一化结果绘制边框
        black_bg = np.ones(img.shape, np.uint8) #创建黑色幕布
        black = cv2.drawContours(black_bg,contour_reconstruct,-1,(255,255,255),1) #绘制白色轮廓
        cv2.imshow("contour_reconstruct", black)
        return black

    #获取二值化图像
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    img_cv[img_cv>0]=255
    
    #轮廓获取
    contours, hier = cv2.findContours(img_cv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #寻找轮廓
    contours = sorted(contours, key = cv2.contourArea, reverse=True)#对一系列轮廓点坐标按它们围成的区域面积进行排序
    if contours!= []:
        contour_array = contours[0][:, 0, :]
    else:
        contour_array = np.array([[40,40],[40,41]])
    #画个样子出来
    #ret = np.ones(img_cv.shape, np.uint8)
    #ret = cv2.drawContours(ret,contours[0],-1,(255,255,255),1) #以最大的轮廓，绘制白色轮廓
    #傅里叶
    contours_complex = np.empty(contour_array.shape[:-1], dtype=np.complex)
    contours_complex.real = contour_array[:,0]#横坐标作为实数部分
    contours_complex.imag = contour_array[:,1]#纵坐标作为虚数部分
    fourier_result = np.fft.fft(contours_complex)#进行傅里叶变换
    
    #截短傅里叶描述子
    Vector = truncate_descriptor(fourier_result)
    #reconstruct(ret, descirptor_in_use)
    Vector = np.abs(Vector)
    
    return np.array(Vector)


# 细化骨架提取
# 图论统计量