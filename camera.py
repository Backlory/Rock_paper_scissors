# 训练有关函数
# 训练有关函数
import os, sys
import cv2
import numpy as np
import random

from numpy.lib import utils
import data.data_loading
from datetime import datetime
#
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold          
from sklearn.metrics import classification_report                         
from sklearn.metrics import confusion_matrix, cohen_kappa_score                           
from matplotlib import pyplot as plt 
#
import utils.img_display as u_idsip
import utils.structure_trans as u_st
from utils.tools import colorstr, tic, toc
from weights.weightio import save_obj, load_obj
#
import model.preprosess as m_pp
import model.ROI_extract as m_Re
import model.feature_extract as m_fet
import model.feature_encode as m_fed
import model.train_strategy as m_ts
from utils.img_display import prepare_path, save_pic, img_square


seconds = 1200    #120秒                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
mps = 10        #每秒10图
capnum = 0



if __name__ =='__main__':
    # 变量准备
    timenow = datetime.now().strftime('%Y%m%d-%H_%M_%S')
    num_dict = {0:'paper',
                1:'scissors',
                2:'rock'}
    obj_h,obj_w = 300, 300 #目标宽度高度
    current_out = sys.stdout
    
    #数据加载
    classifier = load_obj('weights\\svc_classifier.joblib')
    scaler = load_obj('weights\\scaler.joblib')
    
    # 打开摄像头
    try:
        #cap = cv2.VideoCapture('test5.mp4')
        cap = cv2.VideoCapture(capnum)
    except:
        cap = cv2.VideoCapture(1-capnum)
    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('main', 600, 300)

    #开始
    for i in range(int(seconds * mps)):
        ret, img = cap.read()
        h,w,c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:, int(w/2 ):, :]
        img = cv2.flip(img, 1)
        img_new_ = u_st.cv2numpy(img)
        PSR_Dataset_imgs = img_new_[np.newaxis,:,:,:]
        try:
            # 数据预处理
            PSR_Dataset_imgs = m_pp.resize(PSR_Dataset_imgs, [(obj_h,obj_w)])
            #PSR_Dataset_imgs = m_pp.ad_exp_trans(PSR_Dataset_imgs, [])
            PSR_Dataset_imgs = m_pp.bilateralfilter(PSR_Dataset_imgs, [])
            PSR_Dataset_imgs = m_pp.median_blur(PSR_Dataset_imgs, [5])

            # ROI提取
            temp = PSR_Dataset_imgs.copy()
            temp = u_st.numpy2cv(temp)
            temp[0,:,:,:] = cv2.cvtColor(temp[0,:,:,:], cv2.COLOR_RGB2BGR)
            temp = u_st.cv2numpy(temp)
            masks = m_Re.segskin_ellipse_mask(  temp, 
                                                theta = -50/180*3.14, 
                                                cx = 120, 
                                                cy = 147, 
                                                ecx = 38.5, 
                                                ecy = 2.3, 
                                                a = 15, b = 8)
                                                
            masks[0,0,:,:] = m_Re.Morphological_processing(masks[0,0,:,:])
            masks = np.concatenate((masks, masks, masks), axis = 1)
            mask = masks[0,:,:,:]
            mask = u_st.numpy2cv(mask)
            #
            mode_ROI = 1  #-1
            PSR_Dataset_imgs = m_Re.ROIextractor(PSR_Dataset_imgs,
                                                mode_ROI,
                                                savesample = False, 
                                                display = False
                                                )

            # 特征提取
            mode_fet = 'fourier'           #Round_Hu, Round, Hu, distence_detector, fourier
            PSR_Dataset_Vectors_list = m_fet.Featurextractor(   PSR_Dataset_imgs,
                                                                mode_fet, 
                                                                display = False
                                                                )

            # 特征编码
            mode_encode = 'normal'          #bagofword, normal
            X_dataset,  Y_dataset= m_fed.Featurencoder(     PSR_Dataset_Vectors_list,
                                                            [0],
                                                            mode_encode, 
                                                            display = False
                                                            )
        
            #scaler = StandardScaler().fit(X_dataset)                     #标准化
            x_train = scaler.transform(X_dataset)
        
            #分类器分类
            y_pred = classifier.predict(x_train)
            print(num_dict[int(y_pred)])

            img = cv2.resize(img, (obj_w,obj_h))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            temp = np.concatenate((img, mask), axis = 1)          #, temp), axis = 1)
            cv2.imshow('main', temp)
            cv2.waitKey(int(1000/mps))
        except:
            print('no target')