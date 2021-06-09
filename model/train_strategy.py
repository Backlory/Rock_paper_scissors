# 分类器训练(分类vec，结构规整->独热编码)
# 
import numpy as np
import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time

@fun_run_time
def fit(x_train, y_train, classifier = None, mode = 1):
    '''
    classfier = {SVC, SVR, DT, RF, NB, KNN, LR, GBDT}
    '''
    print(colorstr('='*50, 'red'))
    print(colorstr('Training...'))
    #分类器
    if classifier == 'SVC':
        #支持向量机分类
        from sklearn.svm import SVC
        classifier = SVC(C=1, kernel='rbf', gamma='scale', probability=True, verbose=0)
    elif classifier == 'SVR':
        #支持向量机回归
        from sklearn.svm import SVR
        classifier = SVR(kernel='rbf', verbose=0)
    elif classifier == 'DT':
        #决策树
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(    criterion="gini",
                                                splitter="best",
                                                max_depth=None)
    elif classifier == 'RF':
        #随机森林
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(    n_estimators=100,
                                                criterion="gini", 
                                                max_depth=None)
    elif classifier == 'NB':
        #朴素贝叶斯多项式
        from sklearn.naive_bayes import MultinomialNB
        classifier = MultinomialNB(alpha=0.01)
    elif classifier == 'KNN':
        #K最近邻分类器
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier()
    elif classifier == 'LR':
        #逻辑斯蒂回归
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(penalty='l2')
    elif classifier == 'GBDT':
        from sklearn.ensemble import GradientBoostingClassifier
        classifier = GradientBoostingClassifier(n_estimators=200)
    #训练
    if mode == 1:
        #输入SVM分类
        classifier = classifier.fit(x_train, y_train)
    elif mode == 2:
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