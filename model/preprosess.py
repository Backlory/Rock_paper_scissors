# 预处理部分。（pic3->pic3)

# TODO: 
# 高斯低通滤波 GaussianBlur(img, ksize=5, sigma=0)
# 中值滤波     MidBlur
# 非线性变换   
# 灰度世界算法
# HE
# CLAHE
import utils.structure_trans as u_st
import numpy as np
from utils.tools import colorstr

def Preprosessing(PSR_Dataset, funclist = []):
    '''
    输入：数据集对象，一系列预处理函数

    输出：按顺序预处理好的数据集与标签，numpy矩阵
    '''
    print(colorstr('='*50, 'red'))
    print(colorstr('Preprocess begin.'))
    #
    PSR_Dataset_img = []
    PSR_Dataset_label = []
    #
    readlist = list(range(0, 120)) + list(range(840, 960)) + list(range(1680, 1800))
    readlist_len = len(readlist)
    for i,readidx in enumerate(readlist):
        img, label = PSR_Dataset[readidx]
        img = u_st.cv2numpy(img)    #channal, height, width
        #
        if funclist is not []:
            for func in funclist:
                img = func(img)
        #
        PSR_Dataset_img.append(img)
        PSR_Dataset_label.append(label)
        if i % int(readlist_len/10) == int(readlist_len/10)-1:
            print(f'\t{i+1}/{readlist_len} has been preprocessed...')
    #
    #PSR_Dataset_img = PSR_Dataset_img[0:120] + PSR_Dataset_img[840:960] + PSR_Dataset_img[1680:1800]
    #PSR_Dataset_label = PSR_Dataset_label[0:120] + PSR_Dataset_label[840:960] + PSR_Dataset_label[1680:1800]
    PSR_Dataset_img = np.array(PSR_Dataset_img)
    PSR_Dataset_label = np.array(PSR_Dataset_label)
    #
    print('shapes of images and label:')
    print(PSR_Dataset_img.shape)
    print(PSR_Dataset_label.shape)
    print(colorstr('Preprocess finish.'))
    return PSR_Dataset_img, PSR_Dataset_label