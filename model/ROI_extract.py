# ROI区域提取。（pic3->pic1黑白)
# CV图片通道在第四位，平时numpy都放在第二位的
# 预处理部分。（pic3->pic3)
import math
from os import replace
from sys import api_version
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import median, place


from sklearn.preprocessing import scale

import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc
from utils.tools import fun_run_time


@fun_run_time
def ROIextractor(PSR_Dataset_img, mode = 0, savesample=False, timenow='', disp_sample_list=[]):
    '''
    输入：4d图片集，(num, c, h, w).rgb图片。\n
    mode\n
    1=椭圆肤色模型\n
    2=canny边缘模型\n
    3=v通道otsu阈值模型\n
    4=基于外围框检测的otsu模型\n
    5=主动网格区域分类器模型\n
    6=固定阈值分割\n
    -1=联合1、2、3、4的分割\n
    输出：被剪除mask部分的4d图片集，(num, c, h, w)
    '''
    print(colorstr('='*50, 'red'))
    print(colorstr('ROI extracting...'))
    if savesample and ( timenow=='' or disp_sample_list==[]):
        raise(ValueError('timenow and disp_sample_list not given.'))
    #
    #cv2.waitKey(0)
    PSR_Dataset_img_pred = PSR_Dataset_img.copy()
    filedir = 'experiment/'+ timenow +'/'
    
    #mask获取
    if mode == 1:
        #基于椭圆肤色模型
        masks = segskin_ellipse_mask(PSR_Dataset_img)
    elif mode==2:
        #canny边缘模型
        masks = canny_expend_mask(PSR_Dataset_img)
    elif mode==3:
        #V通道otsu阈值模型
        masks = threshold_OTSU_mask(PSR_Dataset_img)
    elif mode==4:
        #基于外围框的固定阈值模型,V通道
        masks = threshold_bg_mask(PSR_Dataset_img)
    elif mode==-1:
        #缝合怪模型，多个mask取平均
        PSR_Dataset_img_64 = graylevel_down(PSR_Dataset_img, 4)
        masks1 = segskin_ellipse_mask(PSR_Dataset_img_64)
        masks2 = canny_expend_mask(PSR_Dataset_img_64)
        masks3 = threshold_OTSU_mask(PSR_Dataset_img_64)
        masks4 = threshold_bg_mask(PSR_Dataset_img_64, 0.2)
        masks7 = slic_masks(PSR_Dataset_img_64)
        #
        #masks_muti = cv2.addWeighted(masks1,0.5,masks2,0.5,0)
        #masks_muti = cv2.addWeighted(masks_muti,2/3,masks3,1/3,0)
        #masks_muti = cv2.addWeighted(masks_muti,3/4,masks4,1/4,0)
        masks_muti = masks1*0.25 + masks2*0.2 + masks3*0.2 + masks4*0.1 + masks7*0.25
        masks_muti = masks_muti.astype(np.uint8)
        #
        masks = np.where(masks_muti>=255*(0.2), 255, 0) #置信度阈值0.2
        for idx, mask in enumerate(masks):
            temp = Morphological_processing(mask[0,:,:])
            temp = cv2.dilate(temp, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7)))   #白色区域膨胀
            temp = cv2.erode(temp, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7)))   #白色区域收缩
            temp[0:2,:] = 0
            temp[-3:,:] = 0
            temp[:, 0:2] = 0
            temp[:, -3:] = 0
            masks[idx,0,:,:] =temp
            
    
        #
        if savesample: u_idsip.save_pic(u_idsip.img_square(masks1[disp_sample_list, :, :, :]), '02_01_masks1', filedir)
        if savesample: u_idsip.save_pic(u_idsip.img_square(masks2[disp_sample_list, :, :, :]), '02_01_masks2', filedir)
        if savesample: u_idsip.save_pic(u_idsip.img_square(masks3[disp_sample_list, :, :, :]), '02_01_masks3', filedir)
        if savesample: u_idsip.save_pic(u_idsip.img_square(masks4[disp_sample_list, :, :, :]), '02_01_masks4', filedir)
        if savesample: u_idsip.save_pic(u_idsip.img_square(masks7[disp_sample_list, :, :, :]), '02_01_masks7', filedir)
        if savesample: u_idsip.save_pic(u_idsip.img_square(masks_muti[disp_sample_list, :, :, :]), '02_01_mutil_masks', filedir)

        #u_idsip.show_pic(masks[0,:,:,:])
    elif mode==5:
        # 基于像素值分类器的主动网格背景模型。有缺陷，不能用
        len(PSR_Dataset_img)
        region_roi, region_fg, region_bg = get_area_by_mouse(PSR_Dataset_img[0])
        model = classifier_trained_by_img(PSR_Dataset_img[0], region_roi, region_fg, region_bg)
        masks = classifier_mask(PSR_Dataset_img, model)
        

    elif mode==6:
        # 固定阈值分割，230
        imgs = u_st.numpy2cv(PSR_Dataset_img)
        num, h, w, c = imgs.shape
        masks = np.zeros((num, h,w,1), dtype=np.uint8)
        for idx, img in enumerate(imgs):
            mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = np.where(mask>230, 0, 255)
            mask = Morphological_processing(mask)
            masks[idx,:,:,0] = mask
        masks = u_st.cv2numpy(masks)
    elif mode ==7:
        masks = slic_masks(PSR_Dataset_img)


        
    if savesample: u_idsip.save_pic(u_idsip.img_square(masks[disp_sample_list, :, :, :]), '02_02_mask', filedir)
    
    #mask剪除
    for idx, mask in enumerate(masks):
        PSR_Dataset_img_pred[idx, 0, :, :] = np.where(mask==255, PSR_Dataset_img[idx, 0, :, :], 0)
        PSR_Dataset_img_pred[idx, 1, :, :] = np.where(mask==255, PSR_Dataset_img[idx, 1, :, :], 0)
        PSR_Dataset_img_pred[idx, 2, :, :] = np.where(mask==255, PSR_Dataset_img[idx, 2, :, :], 0)
        
        
    if savesample: u_idsip.save_pic(u_idsip.img_square(PSR_Dataset_img_pred[disp_sample_list, :, :, :]), '02_03_maskminus', filedir)
    
    print('\tshapes of images:')
    print('\t',PSR_Dataset_img.shape)
    #处理结束
    return PSR_Dataset_img_pred

def slic_masks(imgs):
    ''' 
    GMM聚类
    '''
    u_st._check_imgs(imgs) #[num, c, h, w]
    imgs = u_st.numpy2cv(imgs)
    #
    from sklearn.mixture import GaussianMixture as GMM
    num, h, w, c = imgs.shape
    #
    masks = np.zeros((num, h,w,1), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        #
        temp = img
        temp.resize((h*w, c))
        gmm = GMM(n_components=2).fit(temp)
        dst = gmm.predict(temp)*255
        dst.resize((h, w))
        # 图像最外围检测
        temp  = np.mean(dst[0, :])/255.0
        temp += np.mean(dst[-1,:])/255.0
        temp += np.mean(dst[:, 0])/255.0
        temp += np.mean(dst[:,-1])/255.0
        temp /= 4
        if temp > 0.5:
            dst = 255- dst
        dst = dst.astype(np.uint8)
        dst = cv2.erode(dst, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        dst = baweraopen_adapt(dst, 0.2, 0.001)
        dst = cv2.dilate(dst, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
            
        masks[idx,:,:,0] = dst
    #
    masks = u_st.cv2numpy(masks)
    u_st._check_imgs(masks)
    return masks

#@fun_run_time
def classifier_mask(imgs, model):
    ''' 
    输入图像和基于像素的分类器，采用二分类器处理图像获取mask。
    '''
    u_st._check_imgs(imgs) #[num, c, h, w]
    imgs = u_st.numpy2cv(imgs)
    imgs = graylevel_down(imgs, 16)
    #
    num, h, w, c = imgs.shape
    masks = np.zeros((num, h,w,1), dtype=np.uint8)
    obj_h,obj_w = 64,64
    for idx, img in enumerate(imgs):
        u_idsip.show_pic(u_st.cv2numpy(img))

        img = cv2.resize(img, (obj_h,obj_w))
        u_idsip.show_pic(u_st.cv2numpy(img))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        img = img[:,:,1:]
        
        x_test = img.reshape((obj_h*obj_w, -1))
        x_test = scale(x_test)
        y_pred = model.predict(x_test)
        dst = y_pred.reshape(obj_h,obj_w)*255
        dst = dst.astype(np.uint8)
        u_idsip.show_pic(dst)
        
        dst = cv2.resize(dst, (h,w))
        u_idsip.show_pic(dst)

        #形态学处理
        dst = Morphological_processing(dst)
        u_idsip.show_pic(dst)
        
        dst = dst.astype(np.uint8)
        masks[idx,:,:,0] = dst
    #
    masks = u_st.cv2numpy(masks)
    u_st._check_imgs(masks)
    return masks


def classifier_trained_by_img(img, region_roi, region_fg, region_bg):
    '''
    输入numpy图像，roi区域，前景，背景
    输出：SVM分类器。
    '''
    y0, y1, x0, x1 = region_roi
    roi_y0, roi_y1, roi_x0, roi_x1 = region_fg
    bg_y0, bg_y1, bg_x0, bg_x1 = region_bg
    
    # 转HSV训练SVM模型
    from sklearn.model_selection import StratifiedKFold #交叉验证
    from sklearn.model_selection import GridSearchCV #网格搜索
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    
    # data
    img_cv = u_st.numpy2cv(img)                                 #转cv格式

    img_cv = graylevel_down(img_cv, 16)

    img_ycrcb_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2YCR_CB)   #转ycrcb
    img_ycrcb_cv = img_ycrcb_cv[:,:,1:]                         #取crcb
    h,w,c = img_ycrcb_cv.shape 
    img_ycrcb_cv = np.reshape(img_ycrcb_cv, (h*w,c))
    img_ycrcb_cv = scale(img_ycrcb_cv)                          #归一化
    img_ycrcb_cv = np.reshape(img_ycrcb_cv, (h,w,c))

    region_fg = img_ycrcb_cv[roi_y0:roi_y1, roi_x0:roi_x1, :]
    x_train1 = region_fg.reshape(((roi_y1-roi_y0)*(roi_x1-roi_x0),-1))
    region_bg = img_ycrcb_cv[bg_y0:bg_y1, bg_x0:bg_x1, :]
    x_train2 = region_bg.reshape(((bg_y1-bg_y0)*(bg_x1-bg_x0),-1))
    
    min_len = min((len(x_train1), len(x_train2)))
    x_train1 = x_train1[np.random.choice(range(len(x_train1)), size=min_len,replace=False)]
    x_train2 = x_train2[np.random.choice(range(len(x_train2)), size=min_len,replace=False)]
    X_dataset = np.concatenate((x_train1[:min_len],x_train2[:min_len]), axis=0)
    Y_dataset = np.array([1]*min_len+[0]*min_len)   #1为前景，0为背景
    #X_train, X_test, Y_train, Y_test = train_test_split(X_dataset, Y_dataset, test_size=0.3, random_state=777)
    
    #model
    from sklearn.svm import SVC
    classifier = SVC(C=0.01, kernel='rbf',gamma='scale',probability=True,verbose=2)
    #from sklearn.ensemble import RandomForestClassifier
    #classifier = RandomForestClassifier(n_estimators=50, criterion='gini')

    classifier.fit(X_dataset, Y_dataset)
    return classifier

def graylevel_down(img, fac=16):
    '''
    输入图像，输出灰度级降低过的图像。
    目前是fac=16.
    256级灰度降低为256/fac级灰度。
    '''
    temp = img*1.0/fac
    temp = temp.astype(np.uint8)
    temp = temp * fac
    temp = temp.astype(np.uint8)
    return temp

g_mouse_img_cv=None #原始图像
g_mouse_point1=None
g_mouse_point2=None
def get_area_by_mouse(img, title=''):
    '''
    点击鼠标获取区域.
    输入，numpy图像
    返回:目标大致区域，目标颜色区域，背景颜色区域。
    (y0, y1, x0, x1), (roi_y0, roi_y1, roi_x0, roi_x1), (bg_y0, bg_y1, bg_x0, bg_x1)= get_area_by_mouse(img)
    '''
    def on_mouse(event, x, y, flags, param):
        global g_mouse_img_cv, g_mouse_point1, g_mouse_point2
        if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
            g_mouse_point1 = (x,y)
            temp = g_mouse_img_cv.copy()
            temp = cv2.circle(temp, g_mouse_point1, 7, (0,255,0), 3)
            cv2.imshow('image', temp)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
            temp = g_mouse_img_cv.copy()
            temp = cv2.circle(temp, g_mouse_point1, 7, (0,255,0), 3)
            temp = cv2.circle(temp, (x,y), 7, (0,255,0), 3)
            temp = cv2.rectangle(temp, g_mouse_point1, (x,y), (255,0,0), 2)
            cv2.imshow('image', temp)
        elif event == cv2.EVENT_LBUTTONUP:         #左键释放
            g_mouse_point2 = (x,y)
            temp = g_mouse_img_cv.copy()
            temp = cv2.circle(temp, g_mouse_point1, 7, (0,255,0), 3)
            temp = cv2.circle(temp, g_mouse_point2, 7, (0,255,0), 3)
            temp = cv2.rectangle(temp, g_mouse_point1, g_mouse_point2, (255,0,0), 2)
            cv2.imshow('image', temp)
            '''
            min_x = min(g_mouse_point1[0],g_mouse_point2[0])     
            min_y = min(g_mouse_point1[1],g_mouse_point2[1])
            width = abs(g_mouse_point1[0] - g_mouse_point2[0])
            height = abs(g_mouse_point1[1] -g_mouse_point2[1])
            
            '''
    def getarea():
        '''
        抓取全局变量转化为区域。
        y0, y1, x0, x1 = getarea()
        img_cut = img[y0:y1, x0:x1,:]
        '''
        global  g_mouse_point1, g_mouse_point2
        min_x = min(g_mouse_point1[0],g_mouse_point2[0])
        min_y = min(g_mouse_point1[1],g_mouse_point2[1])
        width = abs(g_mouse_point1[0] - g_mouse_point2[0])
        height = abs(g_mouse_point1[1] -g_mouse_point2[1])
        return (min_y, min_y+height, min_x, min_x+width)
    
    global g_mouse_img_cv, g_mouse_point1, g_mouse_point2
    img = graylevel_down(img, 16)
    
    #区域划定
    g_mouse_img_cv = u_st.numpy2cv(img)
    print(colorstr(f'\n\tArea sampling...'))
    cv2.namedWindow('image',0)
    cv2.resizeWindow('image', 640, 480)
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', g_mouse_img_cv)
    cv2.waitKey(0)
    y0, y1, x0, x1 = getarea()
    print(colorstr(f'\tArea sampled on [{y0}:{y1}, {x0}:{x1}].'))

    # 目标颜色选取
    g_mouse_img_cv = u_st.numpy2cv(img[:, y0:y1, x0:x1])
    print(colorstr(f'\tROI sampling...'))
    cv2.namedWindow('image',0)
    cv2.resizeWindow('image', 640, 480)
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', g_mouse_img_cv)
    cv2.waitKey(0)
    roi_y0, roi_y1, roi_x0, roi_x1 = getarea()
    print(colorstr(f'\tROI:[{roi_y0}:{roi_y1}, {roi_x0}:{roi_x1}].'))

    # 背景颜色选取
    print(colorstr(f'\tBackground sampling...'))
    cv2.namedWindow('image',0)
    cv2.resizeWindow('image', 640, 480)
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', g_mouse_img_cv)
    cv2.waitKey(0)
    bg_y0, bg_y1, bg_x0, bg_x1 = getarea()
    print(colorstr(f'\tBackground:[{bg_y0}:{bg_y1}, {bg_x0}:{bg_x1}].'))
    
    return (y0, y1, x0, x1), (roi_y0, roi_y1, roi_x0, roi_x1), (bg_y0, bg_y1, bg_x0, bg_x1)

@fun_run_time
def rgb2HSV(imgs):
    '''rgb转HSV'''
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    num, h, w, c = imgs.shape
    imgs_new = np.zeros((num, h, w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        imgs_new[idx, :, :, :] = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new

@fun_run_time
def rgb2YCrCb(imgs):
    '''rgb转HSV'''
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    num, h, w, c = imgs.shape
    imgs_new = np.zeros((num, h, w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        imgs_new[idx, :, :, :] = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new

@fun_run_time
def three2one(imgs, channal=0):
    '''三通道转单通道'''
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    imgs = imgs[:,:,:,channal]
    imgs_new = imgs[:,:,:,np.newaxis]
    #
    imgs_new = u_st.cv2numpy(imgs_new)
    u_st._check_imgs(imgs_new)
    return imgs_new

# 小面积丢弃
#@fun_run_time
def baweraopen_adapt(img, intensity = 0.2, alpha = 0.001):
    '''
    自适应面积丢弃(黑底上的白区域)
    二值化后，统计白色区域的总面积，并去除掉面积低于白色总面积20%的白色小区域。
    img:单通道二值图，数据类型uint8
    intensity:相对面积阈值
    alpha:绝对面积阈值
    eg.
    im2=baweraopen_adapt(im1,0.2, 0.001)去除面积低于20%或面积低于总0.1%
    '''
    img_h, img_w = img.shape
    _, output = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)    #二值化处理
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(output)
    total_region_size = np.sum(stats[1:nlabels,4])
    for i in range(1, nlabels):
        regions_size = stats[i,4]
        if regions_size < total_region_size * intensity or  regions_size < img_h * img_w * alpha:
            x0 = stats[i,0]
            y0 = stats[i,1]
            x1 = stats[i,0]+stats[i,2]
            y1 = stats[i,1]+stats[i,3]
            # output[labels[y0:y1, x0:x1] == i] = 0
            for row in range(y0,y1):
                for col in range(x0,x1):
                    if labels[row, col]==i:
                        output[row, col]=0
    return output

#ycrcb椭圆肤色模型
@fun_run_time
def segskin_ellipse_mask(imgs):
    '''基于YCRCB的椭圆肤色模型.
    输入一组图像，输出masks。
    '''
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    num, h, w, c = imgs.shape

    Kl, Kh=125, 128
    Ymin, Ymax = 16, 235
    Wcb,WLcb, WHcb = 46.97, 23, 14
    Wcr,WLcr, WHcr = 38.76, 20, 10
    
    theta = -50/180*3.14 #新坐标系倾角，正为顺时针转
    cx = 114                #新坐标中心在原坐标系的坐标
    cy = 147
    ecx = -5 #椭圆偏移坐标
    ecy = -2
    a = 18   #肤色模型椭圆中心与轴长，在新坐标系
    b = 7

    color = ['red','blue','yellow','green','orange','purple','black','gray','pink']
    #plt.figure()
    
    masks = np.zeros((num, h, w, 1), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        cv2.waitKey(0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        Y, Cr, Cb = cv2.split(img)
        #u_idsip.show_pic(Y,'Y')
        #u_idsip.show_pic(Cr,'Cr')
        #u_idsip.show_pic(Cb,'Cb')
        #
        #'''
        cb1 = 108 + (Kl-Y) * 10/(Kl-Ymin)
        cr1 = 154 + (Kl-Y) * 10/(Kl-Ymin)
        wcbY = WLcb + (Y-Ymin) * (Wcb-WLcb)/(Kl-Ymin)
        wcrY = WLcr + (Y-Ymin) * (Wcr-WLcr)/(Kl-Ymin);
        Cb_ = np.where(Y<Kl, (Cb - cb1) * Wcb / wcbY + cb1, Cb)
        Cr_ = np.where(Y<Kl, (Cr - cr1) * Wcr / wcrY + cr1, Cr)
        #
        cb1 = 108 + (Y-Kh) * 10 / (Ymax-Kh);
        cr1 = 154 + (Y-Kh) * 22 / (Ymax-Kh);
        wcbY = WHcb + (Ymax-Y) * (Wcb-WHcb) / (Ymax-Kh);
        wcrY = WHcr + (Ymax-Y) * (Wcr-WHcr) / (Ymax-Kh);
        Cb_ = np.where(Y>Kh, (Cb - cb1) * Wcb / wcbY + cb1, Cb_)
        Cr_ = np.where(Y>Kh, (Cr - cr1) * Wcr / wcrY + cr1, Cr_)
        #'''
        Cb_= np.array(Cb, dtype=np.float)
        Cr_= np.array(Cr, dtype=np.float)
        #
        c_tha = math.cos(theta)
        s_tha = math.sin(theta)
        x1 = c_tha*(Cb_-cx) + s_tha*(Cr_-cy)
        y1 = -s_tha*(Cb_-cx) + c_tha*(Cr_-cy)
        #
        #plt.figure()
        #plt.axis([-255,255,-255,255])
        #plt.scatter(Cb_.flatten(),Cr_.flatten(), s=20,c=color[idx], alpha=0.01)
        #plt.axis([-40,40,-40,40])
        #plt.scatter(x1.flatten(),y1.flatten(), s=20,c=color[idx], alpha=0.01)
        #plt.grid()
        #plt.show()
        #
        distense = np.where(1, ((x1/a)**2+(y1/b)**2), 0.001)
        distense[distense==0]=0.0001
        mask = np.where(distense <= 1, 255, 127/distense)
        #u_idsip.show_pic(mask,'ori',showtype='freedom')
        mask=Morphological_processing(mask)

        #
        '''
        m = cv2.moments(mask)
        if m['m00'] != 0:
            x = m['m10']/m['m00']
            y = m['m01']/m['m00']
        else:
            x = 0
            y = 0
        print('x=',x, 'y=',y)
        temp = cv2.circle(mask, (int(x),int(y)), 5, 127, 5)
        #cv2.imshow('out',temp)
        #cv2.waitKey(0)'''

        if np.mean(mask)/255<0.1: #另一张背景
            adapt_x = 15
            adapt_y = 21
            x1 = c_tha*(Cb_-cx) + s_tha*(Cr_-cy) - adapt_x
            y1 = -s_tha*(Cb_-cx) + c_tha*(Cr_-cy) - adapt_y
            distense = np.where(1, ((x1/a)**2+(y1/b)**2), 0)
            mask = np.where(distense <= 1, 255, 127/distense)
            #u_idsip.show_pic(mask,'after')
            mask=Morphological_processing(mask)

        masks[idx, :, :, 0] = mask
    #
    #plt.ioff()
    #plt.show()

    masks = u_st.cv2numpy(masks)
    u_st._check_imgs(masks)
    return masks

def Morphological_processing(mask):
    '''
    输入：2d图片，
    输出：形态学处理好后的2d图片
    '''
    h, w = mask.shape
    temp = np.zeros((h+10, w+10), dtype=np.uint8) #扩张，防止边缘被丢弃
    bg = get_bg_bound(mask[:,:,np.newaxis])  
    bg = 0 if bg<127 else 255
    temp[:,:] =bg
    temp[5:-5, 5:-5] = mask
    mask = temp
    #
    mask = cv2.dilate(mask, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5)))   #白色区域膨胀
    mask = cv2.erode(mask, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5)))   #白色区域收缩
    mask = baweraopen_adapt(mask, intensity = 0.2, alpha = 0.01)
    mask = 255 - mask
    mask = cv2.erode(mask, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5)))   #白色区域收缩
    mask = baweraopen_adapt(mask, intensity = 0.3, alpha = 0.01)
    mask = 255 - mask
    mask = cv2.erode(mask, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5)))   #白色区域收缩
    #
    mask = mask[5:-5, 5:-5]
    mask = np.array(mask, dtype=np.uint8)
    return mask

#canny形态学算子
@fun_run_time
def canny_expend_mask(imgs):
    ''' 
    对单通道原图大津阈值分割
    '''
    u_st._check_imgs(imgs) #[num, c, h, w]
    imgs = u_st.numpy2cv(imgs)
    #
    num, h, w, c = imgs.shape
    #
    masks = np.zeros((num, h,w,1), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        #
        bg1, bg2, bg3 = get_bg_bound(img)   #扩张，防止边缘被丢弃
        temp = np.zeros((h+10, w+10, c), dtype=np.uint8)
        temp[:,:,0] = bg1
        temp[:,:,1] = bg2
        temp[:,:,2] = bg3
        temp[5:-5, 5:-5, :] = img

        dst = cv2.Canny(temp, 50, 150)
        dst = cv2.dilate(dst, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5)))
        dst = 255 - dst
        #u_idsip.show_pic(dst,'2','freedom')
        
        dst = baweraopen_adapt(dst, intensity = 0.4, alpha = 0.001) 
        #u_idsip.show_pic(dst,'3','freedom')

        dst = 255 - dst
        #dst = baweraopen_adapt(dst, intensity = 0.1, alpha = 0.001)
        #u_idsip.show_pic(dst,'4','freedom')
        dst = cv2.erode( dst, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7)))
        #u_idsip.show_pic(dst,'5','freedom')

        #dst = cv2.resize(dst, (h, w))
        
        dst = dst[5:-5, 5:-5]
        masks[idx,:,:,0] = dst
    #
    masks = u_st.cv2numpy(masks)
    u_st._check_imgs(masks)
    return masks

#V通道大津阈值法
@fun_run_time
def threshold_OTSU_mask(imgs):
    ''' 
    对V通道大津阈值分割
    '''
    u_st._check_imgs(imgs) #[num, c, h, w]
    imgs = u_st.numpy2cv(imgs)
    #
    num, h, w, c = imgs.shape
    masks = np.zeros((num, h,w,1), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h,s,v = cv2.split(img)
        _, dst = cv2.threshold(v, 0, 255, cv2.THRESH_OTSU)
        # 图像最外围检测
        temp  = np.mean(dst[0, :])/255.0
        temp += np.mean(dst[-1,:])/255.0
        temp += np.mean(dst[:, 0])/255.0
        temp += np.mean(dst[:,-1])/255.0
        temp /= 4
        if temp > 0.5:
            dst = 255- dst
        #形态学处理
        #dst = Morphological_processing(dst)

        dst = dst.astype(np.uint8)
        masks[idx,:,:,0] = dst
    #
    masks = u_st.cv2numpy(masks)
    u_st._check_imgs(masks)
    return masks

#背景边缘阈值分割
@fun_run_time
def threshold_bg_mask(imgs, alpha=0.2):
    ''' 
    取边缘区域计算背景，再阈值分割
    alpha代表容差限度
    '''
    u_st._check_imgs(imgs) #[num, c, h, w]
    imgs = u_st.numpy2cv(imgs)
    #
    num, h, w, c = imgs.shape
    masks = np.zeros((num, h,w,1), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h,s,v = cv2.split(img)

        bg1, bg2, bg3 = get_bg_bound(img)
        v_th = bg3
        
        dst = np.where(v < v_th*(1-alpha), 255, 0)
        dst = np.where(v > v_th*(1+alpha), 255, dst)
        dst = dst.astype(np.uint8)
        dst = cv2.erode(dst, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
        dst = baweraopen_adapt(dst, 0.2, 0.001)
        dst = cv2.dilate(dst, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))

        # 图像最外围检测
        temp  = np.mean(dst[0, :])/255.0
        temp += np.mean(dst[-1,:])/255.0
        temp += np.mean(dst[:, 0])/255.0
        temp += np.mean(dst[:,-1])/255.0
        temp /= 4
        if temp > 0.5:
            dst = 255- dst
        #形态学处理
        #dst = Morphological_processing(dst)

        dst = dst.astype(np.uint8)
        masks[idx,:,:,0] = dst
    #
    masks = u_st.cv2numpy(masks)
    u_st._check_imgs(masks)
    return masks



def get_bg_bound(img_cv):
    '''
    输入三维图像，输出外圈平均像素值。与通道模式无关。若图像为单通道图片，则为(h, w, 1)即可
    eg. bg1, bg2, bg3 = get_bg_bound(img)
    '''
    h, w, c = img_cv.shape
    temp1 = img_cv[0,:,:].reshape((w,-1))
    temp2 = img_cv[-1,:,:].reshape((w,-1))
    temp3 = img_cv[:,0,:].reshape((h,-1))
    temp4 = img_cv[:,-1,:].reshape((h,-1))
    temp5 = np.concatenate((temp1, temp2, temp3, temp4), axis=0)
    if temp5.shape[1]==3:
        c1,c2,c3 = temp5[:,0], temp5[:,1], temp5[:,2]
        bg1 = np.median(c1, axis=0)
        bg2 = np.median(c2, axis=0)
        bg3 = np.median(c3, axis=0)
        return bg1, bg2, bg3
    else:
        temp5 = temp5[:,0]
        bg = np.median(temp5, axis=0)
        return bg

'''
@fun_run_time
def XXX(imgs, arg=[], k=5, r=6):    
    # 说明
    k = arg[0]
    r = arg[1]
    u_st._check_imgs(imgs)
    imgs = u_st.numpy2cv(imgs)
    #
    imgs_processed = np.zeros_like(imgs)
    for idx, img in enumerate(imgs):
        dst = 
        imgs_processed[idx, :, :, :] = dst
    #
    imgs_processed = u_st.cv2numpy(imgs_processed)
    u_st._check_imgs(imgs)
    return imgs_processed
'''