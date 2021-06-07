# 训练有关函数
import numpy as np
import random
import data.data_loading
from datetime import datetime

import utils.img_display as u_idsip
from utils.tools import tic, toc


import model.preprosess as m_pp
import model.ROI_extract as m_Re


if __name__ =='__main__':
    # 变量准备
    timenow = datetime.now().strftime('%Y%m%d-%H_%M_%S')
    
    # 数据加载
    PSR_Dataset = data.data_loading.PSR_Dataset('data/data_my') # data_origin,  data_my, data_test
    
    np.random.seed(777)
    #readlist = list(range(0, 120)) + list(range(840, 960)) + list(range(1680, 1800))
    readlist = np.random.choice(range(len(PSR_Dataset)), 100).tolist()
    #readlist = list(range(len(PSR_Dataset)))
    
    disp_sample_list = random.sample(range(len(readlist)), 16) #9,16,64
    #disp_sample_list = range(16)
    
    
    # 数据预处理
    funlist=[]
    funlist.append([m_pp.resize, [(300,300)]])
    funlist.append([m_pp.median_blur, [3]])
    funlist.append([m_pp.ad_exp_trans, []])
    PSR_Dataset_img, PSR_Dataset_label = m_pp.Preprosessing(PSR_Dataset,
                                                            readlist,
                                                            funlist, 
                                                            savesample = True, 
                                                            timenow = timenow, 
                                                            disp_sample_list = disp_sample_list)
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