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
    print(colorstr('Preprocess...'))
    if savesample and ( timenow=='' or disp_sample_list==[]):
        raise(ValueError('timenow and disp_sample_list not given.'))
    #
    #加载数据
    t=tic()
    PSR_Dataset_img = []
    PSR_Dataset_label = []
    readlist = range(len(PSR_Dataset))
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
    imgs_new = np.zeros_like(imgs)
    for idx, img in enumerate(imgs):
        dst = cv2.medianBlur(img, size)
        imgs_new[idx, :, :, :] = dst
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new


# 邻域投票降噪【】【】【】【】【】【】【】【】【】【】【】【】【】【】【】【】【】】【】【】【】【】【】【】【】【】【】【】
@fun_run_time
def noise_remove_Rnearby(imgs, arg, k = 3, r = 4):
    '''24邻域降噪,输入四维图片、阈值\搜索半径，输出降噪结果'''
    u_st._check_imgs(imgs)
    #
    k = arg[0]
    r = arg[1]

    def calculate_noise_count(img, w, h):
        '''计算w,h处点的8邻域中，白色点的个数'''
        count = 0
        width, height = img.shape
        for _w_ in [w - r, w, w + r]:
            for _h_ in [h - r, h, h + r]:
                if _w_ > width - 1 or _h_ > height - 1 or (_w_ == w and _h_ == h):
                    continue
                if img[_w_, _h_] > 15:  #如果某个邻域点大于15说明是亮的
                    count += 1
        return count
    #
    imgs_number,imgs_Channel,w,h= imgs.shape
    imgs_processed = imgs.copy()
    for i in range(imgs_number):#所有图片
        for _w in range(w):
            for _h in range(h):#所有点
                if _w == 0 or _h == 0:
                    imgs_processed[i,:,_w, _h] = 255
                    continue
                # 计算邻域pixel值小于255的个数
                pixel = imgs_processed[i,:,_w, _h] #获取该点的像素值
                if pixel < 15:
                    continue
                elif calculate_noise_count(imgs[i,0,:,:], _w, _h) < k: #若周围的亮点个数小于某个阈值则过滤掉
                    imgs_processed[i,:,_w, _h] = 0
    #
    u_st._check_imgs(imgs)
    return imgs_processed

# TODO: 
# 高斯低通滤波 GaussianBlur(img, ksize=5, sigma=0)
# 非线性变换
# 灰度世界算法
# HE
# CLAHE