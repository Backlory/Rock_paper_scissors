import numpy as np



def _check_image(imgs):
    temp = imgs.shape
    #
    try:
        assert(len(temp) == 4)
    except:
        raise Exception('图片格式不对，应为[num, channal, h, width]') 
    #
    try:
        assert(temp[1] == 1 or temp[1] == 3)
    except:
        raise Exception('通道数应为1或3！') 
    #
    return 1



def img_hstack(imgs):
    '''
    水平摆放图片。4d->3d
    [][][][]
    input: imgs, [num, channal, h, width]
    output: img, [channal, h, width*num]

    e.g.
    img_newline = img_hstack([patches_ori[i], patches_pre[i], patches_gt[i]])

    Warning:如果图片中有单通道也有三通道，三通道一定要放在第一个
    '''
    _check_image(imgs)

    img_out = imgs[0]
    for img in imgs[1:]:
        if img.shape[0] == 1 and img_out.shape[0] == 3:#通道扩增
            img = gray_to_3d(img)
        img_out = np.concatenate((img_out, img), axis = 2)
    return img_out



def img_vstack(imgs):
    '''
    竖直摆放图片。4d->3d
    input: imgs, [num, chan, height, width]
    output: img, [chan, height*num, width]

    e.g.
    img = img_vstack([img, img_newline])

    Warning:如果图片中有单通道也有三通道，三通道一定要放在第一个
    '''
    _check_image(imgs)
    img_out = imgs[0]
    for img in imgs[1:]:
        if img.shape[0] == 1 and img_out.shape[0] == 3:#通道扩增
            img = gray_to_3d(img)
        img_out = np.concatenate((img_out, img), axis = 1)
    return img_out




def img_square(imgs):
    '''
    将所给图片组整合为最合适的正方形。4d->3d
    input: imgs, [num, chan, height, width]
    output: img, [chan, int((height*width)**0.5), int((height*width)**0.5)]

    img = img_square(patches_pred)
    '''
    _check_image(imgs)
    num, chan, height, width = imgs.shape
    temp = int(num**0.5)
    img_out_mat = np.zeros((chan, temp * height, temp * width))

    for m in range(temp): #m行n列，m*height+n+1
        for n in range(temp):#拼接	
            img_out_mat[:, m*height:(m+1)*height, n*width:(n+1)*width] = imgs[m*temp+n,:,:,:]
    return img_out_mat



def gray_to_3d(img):
    '''
    灰度转换为3d
    input:单通道灰度图
    output:三通道灰度图
    '''
    return np.concatenate((img, img, img), axis = 0)

