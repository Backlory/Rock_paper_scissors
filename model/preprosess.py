# 预处理部分。（pic3->pic3)

# TODO: 
# 高斯低通滤波 GaussianBlur(img, ksize=5, sigma=0)
# 中值滤波     MidBlur
# 非线性变换   
# 灰度世界算法
# HE
# CLAHE
import random
import numpy as np

import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr
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
    PSR_Dataset_img = []
    PSR_Dataset_label = []
    #readlist = list(range(0, 120)) + list(range(840, 960)) + list(range(1680, 1800))
    readlist = range(len(PSR_Dataset))
    readlist_len = len(readlist)
    for i,readidx in enumerate(readlist):
        img, label = PSR_Dataset[readidx]
        img = u_st.cv2numpy(img)    #channal, height, width
        PSR_Dataset_img.append(img)
        PSR_Dataset_label.append(label)
        if i % int(readlist_len/10) == int(readlist_len/10)-1:
            print(f'\t{i+1}/{readlist_len} has been preprocessed...')
    #
    #转成四维张量
    PSR_Dataset_img = np.array(PSR_Dataset_img)
    PSR_Dataset_label = np.array(PSR_Dataset_label)
    print('shapes of images and label:')
    print(PSR_Dataset_img.shape)
    print(PSR_Dataset_label.shape)
    if savesample:
        temp = u_idsip.img_square(PSR_Dataset_img[disp_sample_list, :, :, :])
        u_idsip.save_pic(temp, 'original_image', 'experiment/'+ timenow +'/')
    #
    #按顺序做预处理
    if funclist != []:
        for func in funclist:
            PSR_Dataset_img = func(PSR_Dataset_img)
            if savesample:
                temp = u_idsip.img_square(PSR_Dataset_img[disp_sample_list, :, :, :])
                u_idsip.save_pic(temp, str(func.__name__), 'experiment/'+ timenow +'/')

    #处理结束
    print(colorstr('Preprocess finish.'))
    return PSR_Dataset_img, PSR_Dataset_label