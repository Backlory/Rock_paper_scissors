# 分类器训练(分类vec，结构规整->独热编码)
# 
import numpy as np
import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time

@fun_run_time
def fit(x_train, y_train, mode = 1):
    print(colorstr('='*50, 'red'))
    print(colorstr('Training...'))
    if mode == 1:
        #输入SVM分类
        from sklearn.svm import SVC
        classifier = SVC(C=0.2, kernel='rbf', gamma='scale', probability=True, verbose=2)
        classifier = classifier.fit(x_train, y_train)
    
    elif mode == 2:
        from sklearn.svm import SVR
        classifier = SVR(kernel='rbf', verbose=2)
        classifier = classifier.fit(x_train, y_train)
    return classifier

# print(a.time)
# 
# print(a.name)

















# TODO:

# 核kmeans聚类模型

# GMM-EM

# 核化软间隔svm
# 对抗投票输出