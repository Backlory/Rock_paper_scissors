# 训练有关函数
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
    #readlist = list(range(0, 120)) + list(range(840, 960)) + list(range(1680, 1800))
    #readlist = np.random.choice(range(len(PSR_Dataset)), 100).tolist()
    readlist = list(range(len(PSR_Dataset)))
    try:
        disp_sample_list = random.sample(range(len(readlist)), 64) #9,16,64
    except:
        disp_sample_list = random.sample(range(len(readlist)), 16) #9,16,64
    #============================================================          
    try:
        PSR_Dataset_imgs = load_obj('data\\Dataset_imgs_'+ experiment_data +'.joblib')
        PSR_Dataset_labels = load_obj('data\\Dataset_labels_'+ experiment_data +'.joblib')
    except:
        # 数据预处理
        funlist=[]
        funlist.append([m_pp.resize, [(300,300)]])
        #funlist.append([m_pp.ad_exp_trans, []])
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
        #保存处理数据
        save_obj(PSR_Dataset_imgs, 'data\\Dataset_imgs_'+ experiment_data +'.joblib')
        save_obj(PSR_Dataset_labels, 'data\\Dataset_labels_'+ experiment_data +'.joblib')

    #============================================================   
    # 特征提取
    mode_fet = 'fourier'           #Round_Hu, Round, Hu, distence_detector, fourier
    try:
        PSR_Dataset_Vectors_list = load_obj('data\\Dataset_vectors_'+mode_fet+'_'+ experiment_data +'.joblib')
    except:
        save_pic(img_square(PSR_Dataset_imgs[disp_sample_list, :, :, :]), 'dataset', experiment_dir)
        PSR_Dataset_Vectors_list = m_fet.Featurextractor(   PSR_Dataset_imgs,
                                                            mode_fet
                                                            )
        save_obj(PSR_Dataset_Vectors_list, 'data\\Dataset_vectors_'+mode_fet+'_'+ experiment_data +'.joblib')
    
    # 特征编码
    mode_encode = 'normal'          #bagofword, normal
    try:
        if mode_fet != 'distence_detector' and mode_encode == 'bagofword':
            assert(0)
        else:
            X_dataset,  Y_dataset = load_obj('data\\Dataset_encode_'+mode_encode+'_'+ experiment_data +'.joblib')

    except:
        X_dataset,  Y_dataset= m_fed.Featurencoder(     PSR_Dataset_Vectors_list,
                                                        PSR_Dataset_labels,
                                                        mode_encode
                                                        )
        save_obj((X_dataset,  Y_dataset), 'data\\Dataset_encode_'+mode_encode+'_'+ experiment_data +'.joblib')
    
    # 训练集分割
    for K_fold_size in range(2,11):
        skf = StratifiedKFold(n_splits=K_fold_size, shuffle = True,random_state=999) #交叉验证，分层抽样
        
        # 模型训练
        print(colorstr('='*50, 'red'))
        print(colorstr('Training...'))
        y_test_list, y_pred_list = [], []
        for idx, (train_index, test_index) in enumerate(skf.split(X_dataset, Y_dataset)):
            print(f'K = {idx+1} / {skf.n_splits}')
            
            #获取数据
            x_train, y_train = X_dataset[train_index], Y_dataset[train_index]
            x_test, y_test = X_dataset[test_index], Y_dataset[test_index]
            
            #处理标准化
            scaler = StandardScaler().fit(x_train)                     #标准化
            #scaler = MinMaxScaler().fit(x_train)                       #归一化
            x_train = scaler.transform(x_train)
            
            #分类器训练
            classifiers = m_ts.fit_classifiers(x_train, y_train, classifier = 'SVC', mode = 1) #ALL_classifier, svc

            #分类器预测
            #print('train accuracy:')
            #y_pred = classifier.predict(x_train)
            #print(classification_report(y_train, y_pred, zero_division=1))
            #print(confusion_matrix(y_train, y_pred))
            

            x_test = scaler.transform(x_test)
            for idx, classifier in enumerate(classifiers):
                y_pred = classifier.predict(x_test)
                try:
                    y_test_list[idx] = np.concatenate((y_test_list[idx], y_test), axis = 0)
                    y_pred_list[idx] = np.concatenate((y_pred_list[idx], y_pred), axis = 0)
                except:
                    y_test_list.append(y_test)
                    y_pred_list.append(y_pred)
        
        #模型评估，对第idx个分类器作出评估
        print(colorstr('='*50, 'red'))
        print(colorstr('Evaluating...'))
        classifier_names = []
        conf_mats = []
        classification_reports = []
        classification_reports_dict = []
        kappas = []
        for idx, (y_test, y_pred) in enumerate(zip(y_test_list, y_pred_list)):
            #分类器
            print('-'*20)
            print(f'No.{idx+1} : {classifiers[idx]}')
            classifier_names.append(str(classifiers[idx]))
            
            #混淆矩阵图
            conf_mat = confusion_matrix(y_test, y_pred)
            conf_mats.append(conf_mat)
            print(conf_mat)
            #plt.matshow(conf_mat, cmap='viridis')
            #plt.colorbar()
            #for x in range(len(conf_mat)):
            #    for y in range(len(conf_mat)):
            #        plt.annotate(conf_mat[x,y], xy=(x,y), horizontalalignment='center', verticalalignment='center')
            #plt.show()
            
            #评估报告
            temp = classification_report(y_test, y_pred, zero_division=1, digits=4, output_dict=False)
            print(temp)
            classification_reports.append(temp)
            temp = classification_report(y_test, y_pred, zero_division=1, digits=4, output_dict=True)
            classification_reports_dict.append(temp)


            kappa = cohen_kappa_score(y_test, y_pred)
            kappas.append(kappa)
            print(kappa)
        
        #评估报告打印
        performence_report = ''
        performence_report += '\n' + str(timenow)
        performence_report += '\n' + f'feasure extract mode = {mode_fet}, encode mode = {mode_encode}, K_fold_size={K_fold_size}.'
        performence_report += '\n'
        performence_report += '\n' + '='*50
        performence_report += '\n'
        #
        performence_report += '\n' + f' {classifier_names}:'
        performence_report += '\n accuracy, '   + str( np.round([x['accuracy'] for x in classification_reports_dict], 4))
        performence_report += '\n precision, '  + str( np.round([x['weighted avg']['precision'] for x in classification_reports_dict], 4))
        performence_report += '\n recall, '     + str( np.round([x['weighted avg']['recall'] for x in classification_reports_dict], 4))
        performence_report += '\n f1-score, '   + str( np.round([x['weighted avg']['f1-score'] for x in classification_reports_dict], 4))
        performence_report += '\n kappas, '     + str(np.round(kappas, 4))
        performence_report = performence_report.replace('[','')
        performence_report = performence_report.replace(']','')
        performence_report += '\n'
        performence_report += '\n' + '='*50
        performence_report += '\n'
        for i in range(len(classifier_names)):
            performence_report += '\n' + '-'*20
            performence_report += '\n' + f' {classifier_names[i]}'
            performence_report+='\n'+ f' {conf_mats[i]}'
            performence_report+='\n'+ f' {classification_reports[i]}'
            performence_report+='\n'+ f' kappas = {kappas[i]}'
        #
        with open(experiment_dir+str(K_fold_size)+'_performence.txt', 'w', encoding='utf-8') as f:
            f.write(performence_report)
            f.close()

    # 模型文件保存
    save_obj(classifiers, 'weights\\svc_classifier.joblib')
        
        




    print()
