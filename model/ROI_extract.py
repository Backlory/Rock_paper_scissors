# ROI区域提取。（pic3->pic1黑白)
# CV图片通道在第四位，平时numpy都放在第二位的
# 预处理部分。（pic3->pic3)
import math
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.index_tricks import ndenumerate


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
        #基于椭圆肤色模型
        masks = segskin_ellipse_mask(PSR_Dataset_img)
        if savesample: u_idsip.save_pic(u_idsip.img_square(masks[disp_sample_list, :, :, :]), '02_01_mask', filedir)
        
        #形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
        for idx, mask in enumerate(masks):
            dst = mask[0,:,:]
            dst = cv2.dilate(dst,kernel=kernel)   #白色区域膨胀
            dst = baweraopen_adapt(dst, intensity = 0.2, alpha = 0.01)
            masks[idx,0,:,:] = dst
        if savesample: u_idsip.save_pic(u_idsip.img_square(masks[disp_sample_list, :, :, :]), '02_02_mask_xtx', filedir)


    for idx, mask in enumerate(masks):
        PSR_Dataset_img_pred[idx, 0, :, :] = np.where(mask==255, PSR_Dataset_img[idx, 0, :, :], 0)
        PSR_Dataset_img_pred[idx, 1, :, :] = np.where(mask==255, PSR_Dataset_img[idx, 1, :, :], 0)
        PSR_Dataset_img_pred[idx, 2, :, :] = np.where(mask==255, PSR_Dataset_img[idx, 2, :, :], 0)
    if savesample: u_idsip.save_pic(u_idsip.img_square(PSR_Dataset_img_pred[disp_sample_list, :, :, :]), '02_03_maskminus', filedir)

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
def rgb2YCrCb(imgs):
    '''rgb转HSV'''
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    num, h, w, c = imgs.shape
    imgs_new = np.zeros((num, h, w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        imgs_new[idx, :, :, :] = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
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
        imgs_new[idx,:,:,0] = dst
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new

@fun_run_time
def segskin_ellipse_mask(imgs):
    '''基于YCRCB的椭圆肤色模型.
    输入一组图像，输出masks。
    '''
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    num, h, w, c = imgs.shape

    Kl, Kh=125, 128
    Ymin, Ymax = 16, 235
    Wcb,WLcb, WHcb = 46.97, 23, 14
    Wcr,WLcr, WHcr = 38.76, 20, 10
    
    theta = 145/180*3.14 #新坐标系倾角
    cx = 145                #新坐标中心在原坐标系的坐标
    cy = 120
    ecx = -5
    ecy = -2
    a = ((13-ecx)**2+(-2-ecy)**2)**0.5   #肤色模型椭圆中心与轴长，在新坐标系
    b = ((-5-ecx)**2+(7 -ecy)**2)**0.5

    color = ['red','blue','yellow','green','orange','purple','black','gray']
    #plt.figure()
    
    masks = np.zeros((num, h, w, 1), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        Y, Cr, Cb = cv2.split(img)
        #
        '''
        cb1 = 108 + (Kl-Y) * 10/(Kl-Ymin)
        cr1 = 154 + (Kl-Y) * 10/(Kl-Ymin)
        wcbY = WLcb + (Y-Ymin) * (Wcb-WLcb)/(Kl-Ymin)
        wcrY = WLcr + (Y-Ymin) * (Wcr-WLcr)/(Kl-Ymin);
        Cb_ = np.where(Y<Kl, (Cb - cb1) * Wcb / wcbY + cb1, Cb)
        Cr_ = np.where(Y<Kl, (Cr - cr1) * Wcr / wcrY + cr1, Cr)
        #
        cb1 = 108 + (Y-Kh) * 10 / (Ymax-Kh);
        cr1 = 154 + (Y-Kh) * 22 / (Ymax-Kh);
        wcbY = WHcb + (Ymax-Y) * (Wcb-WHcb) / (Ymax-Kh);
        wcrY = WHcr + (Ymax-Y) * (Wcr-WHcr) / (Ymax-Kh);
        Cb_ = np.where(Y>Kh, (Cb - cb1) * Wcb / wcbY + cb1, Cb_)
        Cr_ = np.where(Y>Kh, (Cr - cr1) * Wcr / wcrY + cr1, Cr_)
        '''
        Cb_= np.array(Cb, dtype=np.float)
        Cr_= np.array(Cr, dtype=np.float)
        #
        c_tha = math.cos(theta)
        s_tha = math.sin(theta)
        x1 = c_tha*(Cb_-cx) + s_tha*(Cr_-cy)
        y1 = -s_tha*(Cb_-cx) + c_tha*(Cr_-cy)
        #
        #plt.figure()
        #plt.axis([-255,255,-255,255])
        #plt.scatter(Cb_.flatten(),Cr_.flatten(), s=20,c=color[idx], alpha=0.01)
        #plt.axis([-40,40,-40,40])
        #plt.scatter(x1.flatten(),y1.flatten(), s=20,c=color[idx], alpha=0.01)
        #plt.grid()
        #plt.show()
        #
        distense = np.where(1, ((x1/a)**2+(y1/b)**2), 0)
        mask = np.where(distense <= 1, 255, 0)
        #u_idsip.show_pic(mask,'ori',showtype='freedom')
        if np.mean(mask)/255<0.2:
            adapt_x = 13
            adapt_y = -18
            x1 = c_tha*(Cb_-cx) + s_tha*(Cr_-cy) - adapt_x
            y1 = -s_tha*(Cb_-cx) + c_tha*(Cr_-cy) -adapt_y
            distense = np.where(1, ((x1/a)**2+(y1/b)**2), 0)
            mask = np.where(distense <= 1, 255, 0)
            #u_idsip.show_pic(mask,'after')
        mask = np.array(mask, dtype=np.uint8)
        masks[idx, :, :, 0] = mask
    #
    #plt.ioff()
    #plt.show()

    masks = u_st.cv2numpy(masks)
    u_st._check_imgs(masks)
    return masks


# TODO: 


# RGB2HSV
# RGB2YCrCb
# 色彩空间滤波肤色模型

# 形态学处理：腐蚀膨胀

# 单张面积丢弃
#@fun_run_time
def baweraopen_adapt(img, intensity = 0.2, alpha = 0.001):
    '''
    自适应面积丢弃(黑底上的白区域)
    二值化后，统计白色区域的总面积，并去除掉面积低于白色总面积20%的白色小区域。
    img:单通道二值图，数据类型uint8
    intensity:相对面积阈值
    alpha:绝对面积阈值
    eg.
    im2=baweraopen_adapt(im1,0.2, 0.001)去除面积低于20%或面积低于总0.1%
    '''
    img_h, img_w = img.shape
    _, output = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)    #二值化处理
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(output)
    total_region_size = np.sum(stats[1:nlabels,4])
    for i in range(1, nlabels):
        regions_size = stats[i,4]
        if regions_size < total_region_size * intensity or  regions_size < img_h * img_w * alpha:
            x0 = stats[i,0]
            y0 = stats[i,1]
            x1 = stats[i,0]+stats[i,2]
            y1 = stats[i,1]+stats[i,3]
            # output[labels[y0:y1, x0:x1] == i] = 0
            for row in range(y0,y1):
                for col in range(x0,x1):
                    if labels[row, col]==i:
                        output[row, col]=0
    return output

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