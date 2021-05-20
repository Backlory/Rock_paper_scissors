import numpy as np
from PIL import Image
import cv2
import os,sys


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



def prepare_path(name_dir):
    '''
    生成路径。
    '''
    name_dir = name_dir.replace('/','\\')
    if os.path.exists(name_dir):
        pass
    elif sys.platform=='win32':
        os.system('mkdir ' + name_dir)
    else:
        os.system('mkdir -p ' +name_dir)
    return 1



def save_pic(data,filename,filedir = ''):
    '''
    对三维data,保存png图片。
    save_pic(temp,'1','test/test1')，
    '''
    assert (len(data.shape)==3) #channels, height, width
    assert (data.shape[0] ==1 or data.shape[0] == 3)
    
    if np.max(data)<=1: data=data*255.
    data=np.uint8(data)
    
    if data.shape[0]==1:
        img = Image.fromarray(data[0,:])  #黑白图片
    else:
        img1= Image.fromarray(data[0,:])
        img2= Image.fromarray(data[1,:])
        img3= Image.fromarray(data[2,:])
        img = Image.merge('RGB', (img1, img2, img3))
    
    try:  
        prepare_path(filedir)
        img.save( './'+ filedir +'/' + filename + '.png')  #【】【】【】】【】】【】】【】】【】】【】】【】】【】】【】】【】】【】】【】
    except:
        print('file dir error')



def show_pic(data,windowname = 'default',showtype='freeze'):
    '''
    展示三维矩阵图片。
    show_pic(temp,"r","freeze") 冻结型显示
    show_pic(temp,"r","freedom")自由型显示
    '''
    assert (len(data.shape)==3) 
    assert (data.shape[0] ==1 or data.shape[0] == 3)    #通道数，高，宽
    if np.max(data)<=1: data=data*255.
    data=np.uint8(data)
    #
    if data.shape[0]==1:
        img = Image.fromarray(data[0,:])                  #黑白图片,直接展示
    else:
        img1= Image.fromarray(data[0,:])
        img2= Image.fromarray(data[1,:])
        img3= Image.fromarray(data[2,:])
        img = Image.merge('RGB', (img1, img2, img3))

    img=np.asarray(img)
    if data.shape[0]==3: img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)#如果是三通道的话，需要RGB转BGR
    
    cv2.namedWindow(windowname,0)
    cv2.resizeWindow(windowname, 640, 480)
    cv2.imshow(windowname, img)

    if showtype=='freeze':
        cv2.waitKey(0)
    else:
        cv2.waitKey(1000*60*2)



    