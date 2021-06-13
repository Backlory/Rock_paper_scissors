# 训练有关函数
import numpy as np
import random
from datetime import datetime
import cv2
from numpy.core.fromnumeric import size
import utils.img_display as u_idsip
import utils.structure_trans as u_st
from utils.tools import tic, toc


import model.preprosess as m_pp
import model.ROI_extract as m_Re


seconds = 1200    #120秒                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
mps = 10        #每秒10图
capnum = 0



if __name__ =='__main__':
    # 变量准备
    timenow = datetime.now().strftime('%Y%m%d-%H_%M_%S')
    
    try:
        #cap = cv2.VideoCapture('test5.mp4')
        cap = cv2.VideoCapture(capnum)
    except:
        cap = cv2.VideoCapture(1-capnum)
    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', 600, 300)

    obj_h,obj_w = 300, 300 #目标宽度高度
    for i in range(int(seconds * mps)):
        ret, img = cap.read()
        h,w,c = img.shape
        img = img[:, int(w/2):, :]
        img = cv2.flip(img, 1)
        cv2.imwrite(f'data\data_my\{i}_.jpg', img)
        #
        img_new_ = u_st.cv2numpy(img)
        img_new_ = img_new_[np.newaxis,:,:,:]
        #
        #fx,fy,fw,fh = cv2.getWindowImageRect('main')
        #img = cv2.rectangle(img, (fx+int(fw*0.6),int(fh*0.1)),(fx+int(fw*0.8),int(fh*0.3)),(0,0,255),thickness=5) #目标区域
        #pt1 = (0,0)
        #pt2 = (int(fw/2*0.5), int(fh*0.5))
        #img = cv2.rectangle(img, pt1, pt2,(0,0,255),thickness=5) #目标区域
        #
        img_new_ = m_pp.resize(img_new_, [(obj_h,obj_w)])
        #img_new_ = m_pp.median_blur(img_new_, [3])
        #img_new_ = m_pp.ad_exp_trans(img_new_)


        #
        img_new = img_new_[0,:,:,:]
        img_new = u_st.numpy2cv(img_new)
        img = cv2.resize(img, (obj_w,obj_h))
        temp = np.concatenate((img, img_new), axis = 1)          #, temp), axis = 1)
        cv2.imshow('main', temp)
        cv2.waitKey(int(1000/mps))





    # ROI提取
    mode=3
    PSR_Dataset_img = m_Re.ROIextractor(PSR_Dataset_img,
                                        mode,
                                        savesample = True, 
                                        timenow = timenow, 
                                        disp_sample_list = disp_sample_list)
    # 特征提取
    # 特征编码
    # 训练集分割
    # 模型初始化
    # 模型训练
    # 权重文件保存


#TODO: 下边这段代码是视频形式显示删了，不需要了
'''
    for i in range(120*3):
        #
        img = PSR_Dataset_img[i]
        label = PSR_Dataset_label[i]
        #
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        #img = cv2.Canny(img, 60,200)

        #contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        #img = cv2.drawContours(img, contours,-1,(255,255,255),thickness=-1)  #边缘框

        cv2.imshow('a', img)
        cv2.waitKey(10)
        '''