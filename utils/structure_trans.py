import numpy as np
import cv2

def gray_to_3d(img):
    '''
    灰度转换为3d
    input:单通道灰度图
    output:三通道灰度图
    '''
    return np.concatenate((img, img, img), axis = 0)

    