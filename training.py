# 训练有关函数
import numpy as np
import random
import data.data_loading
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import utils.img_display as u_idsip
from utils.tools import tic, toc
from weights.weightio import save_obj
#
import model.preprosess as m_pp
import model.ROI_extract as m_Re
import model.feature_extract as m_fet
import model.feature_encode as m_fed
import model.train_strategy as m_ts


if __name__ =='__main__':
    # 变量准备
    timenow = datetime.now().strftime('%Y%m%d-%H_%M_%S')
    
    # 数据加载
    PSR_Dataset = data.data_loading.PSR_Dataset('data/data_origin') # data_origin,  data_my, data_test
    
    np.random.seed(777)
    #readlist = list(range(0, 120)) + list(range(840, 960)) + list(range(1680, 1800))
    readlist = np.random.choice(range(len(PSR_Dataset)), 10).tolist()
    #readlist = list(range(len(PSR_Dataset)))
    
    disp_sample_list = random.sample(range(len(readlist)), 9) #9,16,64
    
    
    # 数据预处理
    funlist=[]
    funlist.append([m_pp.resize, [(300,300)]])
    funlist.append([m_pp.ad_exp_trans, []])
    funlist.append([m_pp.bilateralfilter, []])
    funlist.append([m_pp.median_blur, [9]])
    PSR_Dataset_imgs, PSR_Dataset_labels = m_pp.Preprosessing(PSR_Dataset,
                                                            readlist,
                                                            funlist, 
                                                            savesample = True, 
                                                            timenow = timenow, 
                                                            disp_sample_list = disp_sample_list)
    # ROI提取
    mode=-1
    PSR_Dataset_imgs = m_Re.ROIextractor(PSR_Dataset_imgs,
                                        mode,
                                        savesample = True, 
                                        timenow = timenow, 
                                        disp_sample_list = disp_sample_list
                                        )
    # 特征提取
    mode = 1
    PSR_Dataset_Vectors_list = m_fet.Featurextractor(   PSR_Dataset_imgs,
                                                        mode
                                                        )
    
    # 特征编码
    mode = 1
    X_dataset,  Y_dataset= m_fed.Featurencoder(     PSR_Dataset_Vectors_list,
                                                    PSR_Dataset_labels,
                                                    mode
                                                    )
    
    # 训练集分割
    x_train, x_test, y_train, y_test = train_test_split(    X_dataset,
                                                            Y_dataset,
                                                            test_size=0.3,
                                                            random_state=999
                                                            )
    scaler = StandardScaler().fit(x_train)                     #标准化
    #scaler = MinMaxScaler().fit(x_train)                       #归一化
    x_train = scaler.transform(x_train)

    # 模型训练
    classifier = m_ts.fit(x_train, y_train, mode = 1)
    # 权重文件保存
    save_obj(classifier, 'weights\\mode_1.joblib')

