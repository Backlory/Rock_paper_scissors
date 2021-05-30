# 训练有关函数
import numpy as np
import random
import data.data_loading
from datetime import datetime

import utils.img_display as u_idsip
from utils.tools import tic, toc

import model.preprosess as m_pp


if __name__ =='__main__':
    # 变量准备
    timenow = datetime.now().strftime('%Y%m%d-%H_%M_%S')
    
    # 数据加载
    PSR_Dataset = data.data_loading.PSR_Dataset('data')
    disp_sample_list = random.sample(range(len(PSR_Dataset)), 64)
    
    # 数据预处理
    PSR_Dataset_img, PSR_Dataset_label = m_pp.Preprosessing(PSR_Dataset,
                                                        [m_pp.median_blur], 
                                                        savesample = True, 
                                                        timenow = timenow, 
                                                        disp_sample_list = disp_sample_list)

    # 保存样例图片

    # ROI提取
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