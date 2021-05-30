# 预处理部分。（pic3->pic3)
import cv2
import random
import numpy as np

import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc   
from utils.tools import fun_run_time


@fun_run_time
def Preprosessing(PSR_Dataset, funclist = [], savesample=False, timenow='', disp_sample_list=[]):
    '''
    输入：数据集对象，一系列预处理函数

    输出：按顺序预处理好的数据集与标签，numpy矩阵
    '''
    print(colorstr('='*50, 'red'))
    print(colorstr('Preprocess begin.'))
    if savesample and ( timenow=='' or disp_sample_list==[]):
        raise(ValueError('timenow and disp_sample_list not given.'))
    #
    #加载数据
    t=tic()
    PSR_Dataset_img = []
    PSR_Dataset_label = []
    #readlist = range(len(PSR_Dataset))

    readlist = list(range(0, 120)) + list(range(840, 960)) + list(range(1680, 1800))
    disp_sample_list = random.sample(range(len(readlist)), 9)
    
    readlist_len = len(readlist)
    for i,readidx in enumerate(readlist):
        img, label = PSR_Dataset[readidx]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = u_st.cv2numpy(img)    #channal, height, width
        PSR_Dataset_img.append(img)
        PSR_Dataset_label.append(label)
        if i % int(readlist_len/10) == int(readlist_len/10)-1:
            print(f'\t{i+1}/{readlist_len} has been preprocessed...')
    toc(t,'data load')
    #
    #转成四维张量
    t=tic()
    PSR_Dataset_img = np.array(PSR_Dataset_img)
    PSR_Dataset_label = np.array(PSR_Dataset_label)
    print('shapes of images and label:')
    print(PSR_Dataset_img.shape)
    print(PSR_Dataset_label.shape)
    if savesample:
        temp = u_idsip.img_square(PSR_Dataset_img[disp_sample_list, :, :, :])
        u_idsip.save_pic(temp, 'original_image', 'experiment/'+ timenow +'/')
    toc(t,'list2numpy')
    #
    #按顺序做预处理
    if funclist != []:
        for idx, funcinfo in enumerate(funclist):
            func = funcinfo[0]
            args = funcinfo[1]

            PSR_Dataset_img = func(PSR_Dataset_img, args)
            if savesample:
                temp = u_idsip.img_square(PSR_Dataset_img[disp_sample_list, :, :, :])
                u_idsip.save_pic(temp, str(idx)+'_'+str(func.__name__), 'experiment/'+ timenow +'/')

    #处理结束
    print(colorstr('Preprocess finish.'))
    return PSR_Dataset_img, PSR_Dataset_label


# 中值滤波     MidBlur
@fun_run_time
def median_blur(imgs, arg):    
    ''' 
    中值模糊  对椒盐噪声有去燥效果
    '''
    size=arg[0]
    #
    imgs = u_st.numpy2cv(imgs)
    imgs_processed = np.zeros_like(imgs)
    for idx, img in enumerate(imgs):
        dst = cv2.medianBlur(img, size)
        imgs_processed[idx, :, :, :] = dst
    imgs_processed = u_st.cv2numpy(imgs_processed)
    return imgs_processed

# TODO: 
# 高斯低通滤波 GaussianBlur(img, ksize=5, sigma=0)
# 非线性变换
# 灰度世界算法
# HE
# CLAHE