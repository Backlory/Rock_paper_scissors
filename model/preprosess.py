# 预处理部分。（pic3->pic3)
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

    输出：按顺序预处理好的数据集与标签，numpy矩阵
    '''
    print(colorstr('='*50, 'red'))
    print(colorstr('Preprocess...'))
    if savesample and ( timenow=='' or disp_sample_list==[]):
        raise(ValueError('timenow and disp_sample_list not given.'))
    #
    #加载数据
    t=tic()
    PSR_Dataset_img = []
    PSR_Dataset_label = []
    #readlist = list(range(0, 120)) + list(range(840, 960)) + list(range(1680, 1800))
    #disp_sample_list = random.sample(range(len(readlist)), 9)
    
    readlist_len = len(readlist)
    for i,readidx in enumerate(readlist):
        img, label = PSR_Dataset[readidx]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = u_st.cv2numpy(img)    #channal, height, width
        PSR_Dataset_img.append(img)
        PSR_Dataset_label.append(label)
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
        u_idsip.save_pic(temp, 'original_image', 'experiment/'+ timenow +'/')
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
@fun_run_time
def median_blur(imgs, arg, size=3):    
    ''' 
    中值模糊  对椒盐噪声有去燥效果
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
@fun_run_time
def resize(imgs, arg=[], size=(300,300)):    
    ''' 
    图像缩放。
    '''
    size=arg[0]
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    num, h, w, c = imgs.shape
    h2, w2= size
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
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        if type == 'HE':
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        elif type == 'CLAHE':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            dst = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        imgs_new[idx, :, :, :] = dst
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new

#二维伽马函数的光照不均匀的图像自适应校正

@fun_run_time
def gamma2D(imgs, arg=[],):    
    ''' 
    转到YUV空间，在Y通道均衡，
    '''
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    HSIZE = 25
    q = 2**0.5
    SIGMA1 = 15
    SIGMA2 = 80
    SIGMA3 = 250
    imgs_new = np.zeros_like(imgs,  dtype=np.uint8)
    for idx, img in enumerate(imgs):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        F1 = cv2.GaussianBlur(v, (HSIZE, HSIZE), SIGMA1 / q, HSIZE)*1.0 # 对亮度v通道进行不同参数的高斯滤波
        F2 = cv2.GaussianBlur(v, (HSIZE, HSIZE), SIGMA2 / q, HSIZE)*1.0
        F3 = cv2.GaussianBlur(v, (HSIZE, HSIZE), SIGMA3 / q, HSIZE)*1.0
        gaus = (F1 + F2 + F3) / 3 # 取每个滤波后的平均值
        gaus = np.array(gaus, dtype=np.uint8)

        cv2.imshow('gaus.jpg', gaus)
        cv2.waitKey(0)
        
        m = np.mean(gaus) # 获取亮度通道的平均值，便于跟原亮度进行比较
        w, height = np.shape(v) # 获取亮度通道的宽和高
        out = np.zeros(np.size(v))
        
        gama = np.power(0.5, ((m - gaus) / m)) # 进行gama变换
        out = (np.power(v, gama))
        
        cv2.imshow('out.jpg', out)
        cv2.waitKey(0)
        
        hsv = cv2.merge([h, s, out])
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('output.jpg', rgb)
        cv2.waitKey(0)
        
        rgb = rgb / rgb.max()
        rgb = 255 * rgb
        rgb = rgb.astype(np.uint8)
        cv2.imshow('output2.jpg', rgb)
        cv2.waitKey(0)


        imgs_new[idx, :, :, :] = rgb
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new 


