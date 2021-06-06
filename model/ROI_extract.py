# ROI区域提取。（pic3->pic1黑白)
# CV图片通道在第四位，平时numpy都放在第二位的
# 预处理部分。（pic3->pic3)
import math
from os import replace
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.function_base import place


from sklearn.preprocessing import scale

import utils.structure_trans as u_st
import utils.img_display as u_idsip
from utils.tools import colorstr, tic, toc   
from utils.tools import fun_run_time


@fun_run_time
def ROIextractor(PSR_Dataset_img, mode = 0, savesample=False, timenow='', disp_sample_list=[]):
    '''
    输入：4d图片集，(num, c, h, w)

    输出：被剪除mask部分的4d图片集，(num, c, h, w)
    '''
    print(colorstr('='*50, 'red'))
    print(colorstr('ROI extracting...'))
    if savesample and ( timenow=='' or disp_sample_list==[]):
        raise(ValueError('timenow and disp_sample_list not given.'))
    #
    PSR_Dataset_img_pred = PSR_Dataset_img.copy()
    filedir = 'experiment/'+ timenow +'/'
    
    #mask获取
    if mode == 0:
        #基于椭圆肤色模型
        masks = segskin_ellipse_mask(PSR_Dataset_img)
    elif mode==1:
        #canny边缘模型
        masks = canny_expend_mask(PSR_Dataset_img)
    elif mode==2:
        #HSV通道otsu阈值模型
        masks = threshold_OTSU_mask(PSR_Dataset_img)
    elif mode==3:
        #缝合怪模型，多个mask取平均
        masks1 = segskin_ellipse_mask(PSR_Dataset_img)
        #u_idsip.show_pic(masks1[0,:,:,:])
        masks2 = canny_expend_mask(PSR_Dataset_img)
        #u_idsip.show_pic(masks1[0,:,:,:])
        masks3 = threshold_OTSU_mask(PSR_Dataset_img)
        #u_idsip.show_pic(masks1[0,:,:,:])
        #
        masks_ = cv2.addWeighted(masks1,0.5,masks2,0.5,0)
        masks = cv2.addWeighted(masks_,0.666,masks3,0.333,0)
        #u_idsip.show_pic(masks[0,:,:,:])
    elif mode==4:
        # 蒙特卡洛采样的GMM主动网格背景模型
        # 划出目标区域，选取前景色，选取背景色，GMM生长？
        len(PSR_Dataset_img)
        region_roi, region_fg, region_bg = get_area_by_mouse(PSR_Dataset_img[0])
        model = classifier_trained_by_img(PSR_Dataset_img[0], region_roi, region_fg, region_bg)
        masks = classifier_mask(PSR_Dataset_img, model)
    elif mode==5:
        # 1、初始化背景模型
        # 在数据集中随机采样，然后双边滤波
        # 在CrCb通道，计算样本各像素点的【区域配准方差】。
        # 【区域配准方差】，a图m点与b图m点之间的方差 = a图m点3*3区域与b图m点3*3区域，两两结合后得到的最小值
        # 方差小的像素点，意味着没怎么动过，只受到光照影响。
        # 然后在yCrCb空间用阈值取形态学保留外环形成背景。
        # 然后计算背景数据在HSV通道的阈值
        # 2、利用该阈值对原始做阈值分割，把背景都割掉
        PSR_subDataset_img = PSR_Dataset_img[np.random.choice(range(len(PSR_Dataset_img)), size=100,replace=False)]
        PSR_subDataset_img_cv = u_st.numpy2cv(PSR_subDataset_img)
        for idx, img_cv in enumerate(PSR_subDataset_img_cv):
            img_cv = cv2.bilateralFilter(img_cv, 0, 100, 5)
            u_idsip.show_pic(u_st.cv2numpy(img_cv))
        pass
        
        
    if savesample: u_idsip.save_pic(u_idsip.img_square(masks[disp_sample_list, :, :, :]), '02_01_mask', filedir)
    
    #mask剪除
    for idx, mask in enumerate(masks):
        PSR_Dataset_img_pred[idx, 0, :, :] = np.where(mask==255, PSR_Dataset_img[idx, 0, :, :], 0)
        PSR_Dataset_img_pred[idx, 1, :, :] = np.where(mask==255, PSR_Dataset_img[idx, 1, :, :], 0)
        PSR_Dataset_img_pred[idx, 2, :, :] = np.where(mask==255, PSR_Dataset_img[idx, 2, :, :], 0)
    if savesample: u_idsip.save_pic(u_idsip.img_square(PSR_Dataset_img_pred[disp_sample_list, :, :, :]), '02_03_maskminus', filedir)

    #处理结束
    return PSR_Dataset_img_pred



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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
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

    img_ycrcb_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCR_CB)   #转ycrcb
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

    '''
    kflod = StratifiedKFold(n_splits=10, shuffle = True,random_state=999)
    param_grid = dict(C = [1])# 网格参数
    grid_search = GridSearchCV( svmclassifier,
                                param_grid,
                                scoring = 'neg_log_loss',
                                n_jobs = -1,    #CPU全开
                                cv = kflod,
                                verbose=1)
    grid_result = grid_search.fit(X_train, Y_train) #运行网格搜索
    print("Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean,param in zip(means,params):
        print("%f  with:   %r" % (mean,param))
    '''

    #Y_pred = svmclassifier.predict(X_dataset)
    #print(classification_report(Y_dataset, Y_pred))
    #print('='*20)
    #print(confusion_matrix(Y_test, Y_pred))
    #print('='*20)
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
        imgs_new[idx, :, :, :] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
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
        imgs_new[idx, :, :, :] = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
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


##
 

 

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
    
    theta = 145/180*3.14 #新坐标系倾角
    cx = 145                #新坐标中心在原坐标系的坐标
    cy = 120
    ecx = -5
    ecy = -2
    a = ((13-ecx)**2+(-2-ecy)**2)**0.5   #肤色模型椭圆中心与轴长，在新坐标系
    b = ((-5-ecx)**2+(7 -ecy)**2)**0.5

    color = ['red','blue','yellow','green','orange','purple','black','gray']
    #plt.figure()
    
    masks = np.zeros((num, h, w, 1), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        Y, Cr, Cb = cv2.split(img)
        #
        '''
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
        '''
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
        distense = np.where(1, ((x1/a)**2+(y1/b)**2), 0)
        mask = np.where(distense <= 1, 255, 0)
        #u_idsip.show_pic(mask,'ori',showtype='freedom')
        mask=Morphological_processing(mask)

        if np.mean(mask)/255<0.05:
            adapt_x = 13
            adapt_y = -18
            x1 = c_tha*(Cb_-cx) + s_tha*(Cr_-cy) - adapt_x
            y1 = -s_tha*(Cb_-cx) + c_tha*(Cr_-cy) -adapt_y
            distense = np.where(1, ((x1/a)**2+(y1/b)**2), 0)
            mask = np.where(distense <= 1, 255, 0)
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
    mask = np.array(mask, dtype=np.uint8)
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
    mask = np.array(mask, dtype=np.uint8)
    return mask
# TODO: 


# RGB2HSV
# RGB2YCrCb
# 色彩空间滤波肤色模型

# 形态学处理：腐蚀膨胀



#复杂背景：主动轮廓模型snake
#复杂背景：梯度矢量流主动轮廓模型GVF-snake
#复杂背景：超像素生长法


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
    masks = np.zeros((num, h,w,1), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        dst = cv2.Canny(img, 50, 150)
        dst = cv2.dilate(dst, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7)))
        dst = 255 - dst
        #u_idsip.show_pic(dst,'2','freedom')
        dst = baweraopen_adapt(dst, intensity = 0.5, alpha = 0.001)
        #u_idsip.show_pic(dst,'3','freedom')
        dst = 255 - dst
        dst = baweraopen_adapt(dst, intensity = 0.2, alpha = 0.001)
        #u_idsip.show_pic(dst,'4','freedom')
        dst = cv2.erode( dst, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7)))
        #u_idsip.show_pic(dst,'5','freedom')
        masks[idx,:,:,0] = dst
    #
    masks = u_st.cv2numpy(masks)
    u_st._check_imgs(masks)
    return masks

#V通道大津阈值法
@fun_run_time
def threshold_OTSU_mask(imgs):
    ''' 
    对单通道原图大津阈值分割
    '''
    u_st._check_imgs(imgs) #[num, c, h, w]
    imgs = u_st.numpy2cv(imgs)
    #
    num, h, w, c = imgs.shape
    masks = np.zeros((num, h,w,1), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img)
        _, dst = cv2.threshold(h, 0, 255, cv2.THRESH_OTSU)
        # 图像最外围检测
        temp  = np.mean(dst[0, :])/255.0
        temp += np.mean(dst[-1,:])/255.0
        temp += np.mean(dst[:, 0])/255.0
        temp += np.mean(dst[:,-1])/255.0
        temp /= 4
        if temp > 0.5:
            dst = 255- dst
        #形态学处理
        dst = Morphological_processing(dst)

        dst = dst.astype(np.uint8)
        masks[idx,:,:,0] = dst
    #
    masks = u_st.cv2numpy(masks)
    u_st._check_imgs(masks)
    return masks

'''
@fun_run_time
def XXX(imgs, arg=[], k=5, r=6):    
    ''' 
    # 说明
'''
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