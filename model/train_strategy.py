# 分类器训练(分类vec，结构规整->独热编码)
# 
from model.ROI_extract import classifier_mask
import numpy as np
import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time

@fun_run_time
def fit_classifiers(x_train, y_train, classifier = None, mode = 1):
    '''
    classfier = {SVC, SVR, DT, RFC, RFR, NB, KNN, LR, GBDT, ALL_classifier}
    '''
    #分类器
    classifiers = []
    if classifier == 'SVC':
        #支持向量机分类
        from sklearn.svm import SVC
        classifier = SVC(C=1, kernel='rbf', gamma='scale', probability=True, verbose=0)
        classifiers.append(classifier)
    elif classifier == 'SVR':
        #支持向量机回归
        from sklearn.svm import SVR
        classifier = SVR(kernel='rbf', verbose=0)
        classifiers.append(classifier)
    elif classifier == 'DT':
        #决策树
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(    criterion="gini",
                                                splitter="best",
                                                max_depth=None)
        classifiers.append(classifier)
    elif classifier == 'RFC':
        #随机森林
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(    n_estimators=100,
                                                criterion="gini", 
                                                max_depth=None)
        classifiers.append(classifier)
    elif classifier == 'RFR':
        from sklearn.ensemble import RandomForestRegressor 
        classifier = RandomForestRegressor(    n_estimators=100,
                                                criterion="gini", 
                                                max_depth=None)
        classifiers.append(classifier)
    elif classifier == 'NB':
        #朴素贝叶斯多项式
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifiers.append(classifier)
    elif classifier == 'KNN':
        #K最近邻分类器
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier()
        classifiers.append(classifier)
    elif classifier == 'LR':
        #逻辑斯蒂回归
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(penalty='l2')
        classifiers.append(classifier)
    elif classifier == 'GBDT':
        from sklearn.ensemble import GradientBoostingClassifier
        classifier = GradientBoostingClassifier(n_estimators=200)
        classifiers.append(classifier)
    elif classifier == 'ALL_classifier':
        #鱼 龙 混 杂 
        from sklearn.svm import SVC
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor 
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        #
        classifiers.append( SVC(C=1, kernel='rbf', gamma='scale', probability=True, verbose=0)  )
        #classifiers.append( SVR(kernel='rbf', verbose=0)                                        )
        classifiers.append( DecisionTreeClassifier()                                            )
        classifiers.append( RandomForestClassifier(n_estimators=100)                            )
        #classifiers.append( RandomForestRegressor(n_estimators=100)                             )
        classifiers.append( GaussianNB()                                           )
        classifiers.append( KNeighborsClassifier()                                              )
        #classifiers.append( LogisticRegression(penalty='l2')                                    )
        classifiers.append( GradientBoostingClassifier(n_estimators=100)                        )
    
    #训练
    if mode == 1:
        #直接fit
        for classifier in classifiers:
            print('fitting ', classifier, '...')
            classifier = classifier.fit(x_train, y_train)
    elif mode == 2:
        pass
    #结束
    return classifiers

# print(a.time)
# 
# print(a.name)

















# TODO:

# 核kmeans聚类模型

# GMM-EM

# 核化软间隔svm
# 对抗投票输出