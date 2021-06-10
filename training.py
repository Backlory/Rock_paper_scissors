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
from weights.weightio import save_obj, load_obj
#
import model.preprosess as m_pp
import model.ROI_extract as m_Re
import model.feature_extract as m_fet
import model.feature_encode as m_fed
import model.train_strategy as m_ts


if __name__ =='__main__':
    # 变量准备
    timenow = datetime.now().strftime('%Y%m%d-%H_%M_%S')
    experiment_data = 'data_my'   #data_origin,  data_my, data_test
    
    # 数据加载
    PSR_Dataset = data.data_loading.PSR_Dataset('data/'+str(experiment_data)) #
    
    # 获取展示列表
    np.random.seed(777)
    #readlist = list(range(0, 120)) + list(range(840, 960)) + list(range(1680, 1800))
    #readlist = np.random.choice(range(len(PSR_Dataset)), 100).tolist()
    readlist = list(range(len(PSR_Dataset)))
    disp_sample_list = random.sample(range(len(readlist)), 64) #9,16,64
    #============================================================          
    try:
        PSR_Dataset_imgs = load_obj('data\\Dataset_imgs_'+ experiment_data +'.joblib')
        PSR_Dataset_labels = load_obj('data\\Dataset_labels_'+ experiment_data +'.joblib')
    except:
        # 数据预处理
        funlist=[]
        funlist.append([m_pp.resize, [(128,128)]])
        funlist.append([m_pp.ad_exp_trans, []])
        funlist.append([m_pp.bilateralfilter, []])
        funlist.append([m_pp.median_blur, [5]])
        PSR_Dataset_imgs, PSR_Dataset_labels = m_pp.Preprosessing(PSR_Dataset,
                                                                readlist,
                                                                funlist, 
                                                                savesample = True, 
                                                                timenow = timenow, 
                                                                disp_sample_list = disp_sample_list)
        # ROI提取
        mode_ROI=-1  #-1
        PSR_Dataset_imgs = m_Re.ROIextractor(PSR_Dataset_imgs,
                                            mode_ROI,
                                            savesample = True, 
                                            timenow = timenow, 
                                            disp_sample_list = disp_sample_list
                                            )
        #保存处理数据
        save_obj(PSR_Dataset_imgs, 'data\\Dataset_imgs_'+ experiment_data +'.joblib')
        save_obj(PSR_Dataset_labels, 'data\\Dataset_labels_'+ experiment_data +'.joblib')

    #============================================================          
    # 特征提取
    mode = 0
    PSR_Dataset_Vectors_list = m_fet.Featurextractor(   PSR_Dataset_imgs,
                                                        mode
                                                        )
    
    # 特征编码
    mode = 0
    X_dataset,  Y_dataset= m_fed.Featurencoder(     PSR_Dataset_Vectors_list,
                                                    PSR_Dataset_labels,
                                                    mode
                                                    )
    
    # 训练集分割
    from sklearn.model_selection import StratifiedKFold          
    from sklearn.metrics import classification_report                         
    from sklearn.metrics import confusion_matrix                         
    skf = StratifiedKFold(n_splits=10, shuffle = True,random_state=999) #交叉验证，分层抽样
    for train_index, test_index in skf.split(X_dataset, Y_dataset):
        print('='*50)
        print('K-fold cross validation')
        #获取数据
        x_train, y_train = X_dataset[train_index], Y_dataset[train_index]
        x_test, y_test = X_dataset[test_index], Y_dataset[test_index]
        
        #处理标准化
        scaler = StandardScaler().fit(x_train)                     #标准化
        #scaler = MinMaxScaler().fit(x_train)                       #归一化
        x_train = scaler.transform(x_train)
        
        #分类器训练
        
        classifiers = m_ts.fit_classifiers(x_train, y_train, classifier = 'ALL', mode = 1)
        #分类器预测
        
        #print('train accuracy:')
        #y_pred = classifier.predict(x_train)
        #print(classification_report(y_train, y_pred, zero_division=1))
        #print(confusion_matrix(y_train, y_pred))
        
        print('test accuracy:')
        x_test = scaler.transform(x_test)
        for classifier in classifiers:
            y_pred = classifier.predict(x_test)
            print(confusion_matrix(y_test, y_pred))
            print(classification_report(y_test, y_pred, zero_division=1))
            print('='*20)
        
    save_obj(classifier, 'weights\\classifier.joblib')
        
        

    # 权重文件保存



    print()
