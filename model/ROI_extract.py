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
def ROIextractor(PSR_Dataset_img, mode = 0, savesample=False, timenow='', disp_sample_list=[]):
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
    PSR_Dataset_img_pred = PSR_Dataset_img.copy()
    filedir = 'experiment/'+ timenow +'/'
    if mode == 0:
        #三通道转HSV，取V通道后OTSU
        mask = rgb2HSV(PSR_Dataset_img)
        mask = mask[:,2,:,:]
        mask = mask[:,np.newaxis,:,:]        
        if savesample: u_idsip.save_pic(u_idsip.img_square(mask[disp_sample_list, :, :, :]), '01_rgb2HSV', filedir)

        mask = threshold_OTSU(mask) 
        PSR_Dataset_img_pred[:,0,:,:] = np.where(mask==255, 0, PSR_Dataset_img[:,0,:,:])
        PSR_Dataset_img_pred[:,1,:,:] = np.where(mask==255, 0, PSR_Dataset_img[:,1,:,:])
        PSR_Dataset_img_pred[:,2,:,:] = np.where(mask==255, 0, PSR_Dataset_img[:,2,:,:])

        if savesample:
            temp = u_idsip.img_square(PSR_Dataset_img_pred[disp_sample_list, :, :, :])
            u_idsip.save_pic(temp, '01_rgb2HSV', 'experiment/'+ timenow +'/')

    #处理结束
    return PSR_Dataset_img_pred



@fun_run_time
def rgb2HSV(imgs):
    '''rgb转HSV'''
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    num, h, w, c = imgs.shape
    imgs_new = np.zeros((num, h, w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        imgs_new[idx, :, :, :] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new

@fun_run_time
def three2one(imgs, channal=0):
    '''三通道转单通道'''
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    imgs = imgs[:,:,:,channal]
    imgs_new = imgs[:,:,:,np.newaxis]
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new

#V通道大津阈值法
@fun_run_time
def threshold_OTSU(imgs):
    ''' 
    对单通道原图大津阈值分割
    '''
    u_st._check_imgs(imgs) #[num, c, h, w]
    imgs = u_st.numpy2cv(imgs)
    #
    imgs_new = np.zeros_like(imgs, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        _, dst = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        imgs_new[idx,0,:,:] = dst
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