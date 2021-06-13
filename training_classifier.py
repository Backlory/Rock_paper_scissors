# 训练最终分类器用的。生成scaler和classifier
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


if __name__ =='__main__':
    # 变量准备
    timenow = datetime.now().strftime('%Y%m%d-%H_%M_%S')
    experiment_data = 'data_my'   #data_origin,  data_my, data_test, data_my_add
    experiment_dir = 'experiment/'+ timenow +'/'
    prepare_path(experiment_dir)
    
    # 数据加载
    PSR_Dataset = data.data_loading.PSR_Dataset('data/'+str(experiment_data)) #
    
    # 获取展示列表
    np.random.seed(777)
    readlist = list(range(len(PSR_Dataset)))
    disp_sample_list = random.sample(range(len(readlist)), 64) #9,16,64
    #============================================================          

    # 数据预处理
    funlist=[]
    funlist.append([m_pp.resize, [(300,300)]])
    funlist.append([m_pp.bilateralfilter, []])
    funlist.append([m_pp.median_blur, [5]])
    PSR_Dataset_imgs, PSR_Dataset_labels = m_pp.Preprosessing(PSR_Dataset,
                                                            readlist,
                                                            funlist, 
                                                            savesample = True, 
                                                            timenow = timenow, 
                                                            disp_sample_list = disp_sample_list)

    # ROI提取
    mode_ROI = 1  #-1
    PSR_Dataset_imgs = m_Re.ROIextractor(PSR_Dataset_imgs,
                                        mode_ROI,
                                        savesample = True, 
                                        timenow = timenow, 
                                        disp_sample_list = disp_sample_list
                                        )
    for idx, img in enumerate(PSR_Dataset_imgs):
        if np.max(img) == 0:
            PSR_Dataset_imgs[idx] = PSR_Dataset_imgs[idx-1]
            PSR_Dataset_labels[idx] = PSR_Dataset_labels[idx-1]
            print(idx)
    
    # 扩增
    PSR_Dataset_imgs = u_st.numpy2cv(PSR_Dataset_imgs)
    P1 = PSR_Dataset_imgs
    P2 = [cv2.flip(img, 1 ) for img in PSR_Dataset_imgs]
    P3 = [cv2.flip(img, 2 ) for img in PSR_Dataset_imgs]
    P4 = [cv2.flip(img, 3 ) for img in PSR_Dataset_imgs]
    PSR_Dataset_imgs = np.concatenate((P1, P2,P3, P4), axis=0)
    PSR_Dataset_imgs = u_st.cv2numpy(PSR_Dataset_imgs)

    P1 = PSR_Dataset_labels
    PSR_Dataset_labels = np.concatenate((P1, P1,P1, P1), axis=0)

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
    
    # 训练集分割

    # 模型训练
    print(colorstr('='*50, 'red'))
    print(colorstr('Training...'))
    #获取数据
    x_train, y_train = X_dataset, Y_dataset

    scaler = StandardScaler().fit(x_train)                     #标准化
    x_train = scaler.transform(x_train)
    
    #分类器训练
    classifiers = m_ts.fit_classifiers(x_train, y_train, classifier = 'SVC', mode = 1) #ALL_classifier, svc
    classifier = classifiers[0]

    print(colorstr('Testing...'))
    y_pred = classifier.predict(x_train)

    print(confusion_matrix(y_train, y_pred))
    print(classification_report(y_train, y_pred, zero_division=1, digits=4, output_dict=False))

        
    # 模型文件保存
    save_obj(classifier, 'weights\\svc_classifier.joblib')
    save_obj(scaler, 'weights\\scaler.joblib')
        
        




    print()
