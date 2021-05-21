import numpy as np
import cv2

def gray2ggg(img):
    '''
    灰度转换为3d
    input:单通道灰度图
    output:三通道灰度图
    '''
    return np.concatenate((img, img, img), axis = 0)

def cv2numpy(img):
    '''
    输入（32，32，3），输出（3，32，32）
    输入（1500, 32，32，3），输出（1500, 3，32，32）
    '''
    if len(img.shape) == 3:
        assert(img.shape[2] == 1 or img.shape[2] == 3)
        return np.transpose(img,(2,0,1))
    else:
        assert(img.shape[3] == 1 or img.shape[3] == 3)
        return np.transpose(img,(0,3,1,2))