# 答辩时测试用
import cv2
import numpy as np
import random
import os

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


if __name__ =='__main__':
    #参数
    timenow = datetime.now().strftime('%Y%m%d-%H_%M_%S')
    dect_num = {'paper':0,
                'scissors':1,
                'rock':2
                }

    # 数据加载
    PSR_Dataset_imgs = []
    PSR_Dataset_labels = []
    for dirpath, dirnames, filenames in os.walk("data\\test_set", topdown=False):
        for filename in filenames:
            temp = cv2.imread(dirpath+'\\'+filename)
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            PSR_Dataset_imgs.append(temp)
            PSR_Dataset_labels.append(dect_num[str(filename[:-5])])
    PSR_Dataset_imgs = np.array(PSR_Dataset_imgs)
    PSR_Dataset_imgs = u_st.cv2numpy(PSR_Dataset_imgs)
    classifier = load_obj('weights\\svc_classifier.joblib')
    

    # 数据预处理
    PSR_Dataset_imgs = m_pp.resize(PSR_Dataset_imgs, [(300,300)])
    #PSR_Dataset_imgs = m_pp.ad_exp_trans(PSR_Dataset_imgs, [])
    PSR_Dataset_imgs = m_pp.bilateralfilter(PSR_Dataset_imgs, [])
    PSR_Dataset_imgs = m_pp.median_blur(PSR_Dataset_imgs, [5])

    # ROI提取
    mode_ROI = 1  #-1
    PSR_Dataset_imgs = m_Re.ROIextractor(PSR_Dataset_imgs,
                                        mode_ROI,
                                        savesample = False
                                        )


    #============================================================   
    # 特征提取
    mode_fet = 'fourier'           #Round_Hu, Round, Hu, distence_detector, fourier
    PSR_Dataset_Vectors_list = m_fet.Featurextractor(   PSR_Dataset_imgs,
                                                        mode_fet
                                                        )

    # 特征编码
    mode_encode = 'normal'          #bagofword, normal
    X_dataset,  Y_dataset= m_fed.Featurencoder(     PSR_Dataset_Vectors_list,
                                                    PSR_Dataset_labels,
                                                    mode_encode
                                                    )
    
    #获取数据
    x_train, y_groundtruth = X_dataset, Y_dataset

    scaler = StandardScaler().fit(x_train)                     #标准化
    x_train = scaler.transform(x_train)
    
    #分类器分类
    y_pred = classifier.predict(x_train)

    print(confusion_matrix(y_groundtruth, y_pred))
    print(classification_report(y_groundtruth, y_pred, zero_division=1, digits=4, output_dict=False))

        
        




    print()
