# 子实验：数据采集
# 对数据进行采集。

import cv2
from utils.img_display import save_pic
from utils.structure_trans import cv2numpy

if __name__=="__main__":
    seconds = 10000
    mps = 5
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    for i in range(int(seconds * mps)):
        #
        ret, frame = cap.read()
        framesave = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        framesave = cv2.resize(framesave, (256, 256))
        framesave = cv2numpy(framesave)
        #
        save_pic(framesave, 'shitou_'+str(i), 'data/data_bg/')
        cv2.imshow('210530', frame)
        cv2.waitKey(int(1000/mps))