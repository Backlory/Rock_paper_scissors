# 预处理部分。（pic3->pic3)

# TODO: 
# 高斯低通滤波 GaussianBlur(img, ksize=5, sigma=0)
# 中值滤波     MidBlur
# 非线性变换   
# 灰度世界算法
# HE
# CLAHE
import utils.structure_trans as u_st
import numpy as np

def Preprosessing(PSR_Dataset, func = None):
    '''
    输入数据集对象，
    输出预处理好的数据集与标签，numpy矩阵
    '''
    PSR_Dataset_img = []
    PSR_Dataset_label = []
    for idx, item in enumerate(PSR_Dataset):
        img, label = item
        img = u_st.cv2numpy(img)    #channal, height, width
        #
        PSR_Dataset_img.append(img)
        PSR_Dataset_label.append(label)
    
    PSR_Dataset_img = PSR_Dataset_img[0:120] + PSR_Dataset_img[840:960] + PSR_Dataset_img[1680:1800]
    PSR_Dataset_img = np.array(PSR_Dataset_img)
    PSR_Dataset_label = PSR_Dataset_label[0:120] + PSR_Dataset_label[840:960] + PSR_Dataset_label[1680:1800]
    PSR_Dataset_label = np.array(PSR_Dataset_label)

    return PSR_Dataset_img, PSR_Dataset_label