import numpy as np
import cv2
import data.data_loading
import time

if __name__ =='__main__':
    PSR_Dataset = data.data_loading.PSR_Dataset('data')
    for idx, item in enumerate(PSR_Dataset):
        img, label = item
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        #img = cv2.Canny(img, 60,200)

        #contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        #img = cv2.drawContours(img, contours,-1,(255,255,255),thickness=-1)  #边缘框

        cv2.imshow('a', img)
        cv2.waitKey(50)