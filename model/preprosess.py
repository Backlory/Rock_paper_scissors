# 预处理部分。（pic3->pic3)
import math
import cv2
import random
import numpy as np

import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc   
from utils.tools import fun_run_time


@fun_run_time
def Preprosessing(PSR_Dataset, readlist = [], funclist = [], savesample=False, timenow='', disp_sample_list=[]):
    '''
    输入：数据集对象，一系列预处理函数

    输出：按顺序预处理好的数据集与标签，图像是np+rgb
    '''
    print(colorstr('='*50, 'red'))
    print(colorstr('Preprocess...'))
    if savesample and ( timenow=='' or disp_sample_list==[]):
        raise(ValueError('timenow and disp_sample_list not given.'))
    #
    #加载数据
    t=tic()
    temp, _ = PSR_Dataset[0]
    h_std,w_std,_=temp.shape
    PSR_Dataset_img = []
    PSR_Dataset_label = []
    readlist_len = len(readlist)
    for i,readidx in enumerate(readlist):
        img, label = PSR_Dataset[readidx]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        if h != h_std or w != w_std:
            img = cv2.resize(img, (h_std,w_std))
        
        img = u_st.cv2numpy(img)    #channal, height, width，RGB
        PSR_Dataset_img.append(img)
        PSR_Dataset_label.append(label)
        if readlist_len >20:
            if i % int(readlist_len/10) == int(readlist_len/10)-1:
                print(f'\t{i+1}/{readlist_len} has been preprocessed...')
    toc(t,'data load', readlist_len)
    #
    #转成四维张量
    t=tic()
    PSR_Dataset_img = np.array(PSR_Dataset_img)
    PSR_Dataset_label = np.array(PSR_Dataset_label)
    print('shapes of images and label:')
    print(PSR_Dataset_img.shape)
    print(PSR_Dataset_label.shape)
    toc(t,'list2numpy')
    if savesample:
        temp = PSR_Dataset_img[disp_sample_list, :, :, :]
        temp = u_idsip.img_square(temp)
        u_idsip.save_pic(temp, '01_00_original_image', 'experiment/'+ timenow +'/')
    #
    #按顺序做预处理
    if funclist != []:
        for idx, funcinfo in enumerate(funclist):
            func = funcinfo[0]
            args = funcinfo[1]

            PSR_Dataset_img = func(PSR_Dataset_img, args)
            if savesample:
                temp = u_idsip.img_square(PSR_Dataset_img[disp_sample_list, :, :, :])
                u_idsip.save_pic(temp, '01_'+str(idx)+'_'+str(func.__name__), 'experiment/'+ timenow +'/')

    #处理结束
    return PSR_Dataset_img, PSR_Dataset_label


# 中值滤波     MidBlur
#@fun_run_time
def median_blur(imgs, arg, size=3):    
    ''' 
    中值模糊  对椒盐噪声有去燥效果。输入numpy图片
    '''
    size=arg[0]
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    imgs_new = np.zeros_like(imgs,  dtype=np.uint8)
    for idx, img in enumerate(imgs):
        dst = cv2.medianBlur(img, size)
        imgs_new[idx, :, :, :] = dst
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new

# 缩放
#@fun_run_time
def resize(imgs, arg=[], size=(300,300)):    
    ''' 
    图像缩放。输入numpy图片
    '''
    size=arg[0]
    #
    u_st._check_imgs(imgs)
    num, c, h, w = imgs.shape
    h2, w2 = size
    if h==h2 and w==w2:
        return imgs
    else:
        imgs = u_st.numpy2cv(imgs)
        imgs_new = np.zeros((num,h2, w2, c),  dtype=np.uint8)
        for idx, img in enumerate(imgs):
            dst = cv2.resize(img, size)
            imgs_new[idx, :, :, :] = dst
        #
        imgs_new = u_st.cv2numpy(imgs_new)
        u_st._check_imgs(imgs_new)
        return imgs_new


# TODO: 
# 高斯低通滤波 GaussianBlur(img, ksize=5, sigma=0)
# 非线性变换
# 灰度世界算法
# HE
@fun_run_time
def equalizeHist(imgs, arg=[],type='CLAHE'):    
    ''' 
    转到YUV空间，在Y通道均衡，
    '''
    type = arg[0]
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    imgs_new = np.zeros_like(imgs,  dtype=np.uint8)
    for idx, img in enumerate(imgs):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if type == 'HE':
            img_hsv[:, :, 0] = cv2.equalizeHist(img_hsv[:, :, 0])
        elif type == 'CLAHE':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_hsv[:, :, 0] = clahe.apply(img_hsv[:, :, 0])
        dst = cv2.cvtColor(img_hsv, cv2.COLOR_YUV2RGB)
        
        imgs_new[idx, :, :, :] = dst
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new

#@fun_run_time
def ad_exp_trans(imgs, arg=[]):
    ''' 
    基于高斯模糊的自适应指数变换，消除光照不均衡阴影
    输入图片numpy图片,rgb
    '''
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    imgs_new = np.zeros_like(imgs,  dtype=np.uint8)
    for idx, img in enumerate(imgs):
        #
        temp = cv2.resize(img, (128,128))       #纹理复杂度计算
        temp = cv2.GaussianBlur(temp, (5, 5), 100)
        #u_idsip.show_pic(u_st.cv2numpy(temp))
        temp = cv2.Canny(temp, 10, 200)
        #u_idsip.show_pic(temp)
        temp = np.mean(temp.flatten())/2.55
        temp = 0.2983*math.log(temp)+0.302    #拟合结果
        alpha_base = np.clip(temp, 0.5, 0.99)
        #print(alpha_base)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)#【】【】【】【】【】【】【】【】【】【】【】【】【】【】【】【】【】【】
        #cv2.imshow('tm', img)
        #cv2.imshow('tm', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #cv2.waitKey(0)
        h, s, v = cv2.split(hsv)
        v = v / 255.0
        for i in range(3):
            gaus = cv2.GaussianBlur(v, (21, 21), 100, 21)*1.0
            gaus_median = np.mean(gaus)
            alpha = np.power(alpha_base, (1 - gaus/gaus_median)) # 进行指数变换
            v = np.power(v, alpha)

            # 对亮度v通道进行不同参数的高斯滤波
            '''
            HSIZE = 25
            q = 2**0.5
            F1 = cv2.GaussianBlur(v, (HSIZE, HSIZE), 15 / q, HSIZE)
            F2 = cv2.GaussianBlur(v, (HSIZE, HSIZE), 80 / q, HSIZE)
            F3 = cv2.GaussianBlur(v, (HSIZE, HSIZE), 250 / q, HSIZE)
            gaus = (F1 + F2 + F3) / 3
            m = np.mean(gaus)
            w, height = np.shape(v)
            out = np.zeros(np.size(v))
            gama = np.power(0.5, ((m - gaus) / m))
            v = (np.power(v, gama))
            '''

        v = np.array(v*255, dtype=np.uint8)
        hsv = cv2.merge([h, s, v])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        imgs_new[idx, :, :, :] = rgb
        #
        if len(imgs)>10:
            if idx % int(len(imgs)/10) == int(len(imgs)/10)-1:
                print(f'\t{idx+1}/{len(imgs)} has been processed...')
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new 


