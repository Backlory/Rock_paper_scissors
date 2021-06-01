# ROI区域提取。（pic3->pic1黑白)

# 预处理部分。（pic3->pic3)
import cv2
import random
import numpy as np
from numpy.testing._private.utils import tempdir

import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc   
from utils.tools import fun_run_time


@fun_run_time
def ROIextractor(PSR_Dataset_img, funclist = [], savesample=False, timenow='', disp_sample_list=[]):
    '''
    输入：4d图片集，(num, c, h, w)

    输出：被剪除mask部分的4d图片集，(num, c, h, w)
    '''
    print(colorstr('='*50, 'red'))
    print(colorstr('ROI extracting...'))
    if savesample and ( timenow=='' or disp_sample_list==[]):
        raise(ValueError('timenow and disp_sample_list not given.'))
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
    return PSR_Dataset_img


@fun_run_time
def rgb2gray(imgs, arg=[]):
    '''矩阵降维，由多通道变为单通道'''
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    num, h, w, c = imgs.shape
    imgs_new = np.zeros((num, h, w, 1), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        imgs_new[idx, :, :, 0] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new 


# 大津阈值法
@fun_run_time
def threshold_expend(imgs, arg=[], thres=127):    
    ''' 
    先对原图固定阈值分割，得到遮罩，然后
    '''
    thres=arg[0]
    u_st._check_imgs(imgs)
    masks = rgb2gray(imgs)  #n 1 h w
    imgs = u_st.numpy2cv(imgs)
    masks = u_st.numpy2cv(masks)
    #
    imgs_new = np.zeros_like(imgs, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        _, temp = cv2.threshold(masks[idx], thres, 255, cv2.THRESH_BINARY) 
        imgs_new[idx, :, :, 0] = np.where(temp==255, 0, img[:,:,0]) #255白
        imgs_new[idx, :, :, 1] = np.where(temp==255, 0, img[:,:,1])
        imgs_new[idx, :, :, 2] = np.where(temp==255, 0, img[:,:,2])
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new



# TODO: 


# RGB2HSV
# RGB2YCrCb
# 色彩空间滤波肤色模型

# 形态学处理：腐蚀膨胀
# 面积丢弃

#复杂背景：主动轮廓模型snake
#复杂背景：梯度矢量流主动轮廓模型GVF-snake
#复杂背景：超像素生长法


'''
@fun_run_time
def XXX(imgs, arg=[], k=5, r=6):    
    ''' 
    # 说明
'''
    k = arg[0]
    r = arg[1]
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    imgs_processed = np.zeros_like(imgs)
    for idx, img in enumerate(imgs):
        dst = 
        imgs_processed[idx, :, :, :] = dst
    #
    imgs_processed = u_st.cv2numpy(imgs_processed)
    u_st._check_imgs(imgs)
    return imgs_processed
'''