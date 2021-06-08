# 分类器训练(分类vec，结构规整->独热编码)
# 


def fit(classifier, x_train, y_train, mode = 1):
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