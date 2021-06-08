import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class PSR_Dataset(Dataset):
    def __init__(self, dataset_path):
        self.data_paths_list = []
        self.data_labels_list = []
        self.num2label = {  0:'others',
                            1:'paper',
                            2:'scissors',
                            3:'rock'
                            }
        self.dataset_paths = [self.num2label[x] for x in range(len(self.num2label))] #按顺序获取num2label中的键值
        self.dataset_paths = [os.path.join(dataset_path, label) for label in self.dataset_paths] #将键值转成路径，因为路径是以文件夹命名
        for idx, dataset_path in enumerate(self.dataset_paths): #依次读取
            data_path_list = self.get_read_file_list(dataset_path)
            self.data_paths_list = self.data_paths_list + data_path_list
            self.data_labels_list = self.data_labels_list + [idx]*len(data_path_list)
    #
    def __getitem__(self, index):
        data_path = self.data_paths_list[index]
        data_label = self.data_labels_list[index]
        img = cv2.imread(data_path, 1)
        return img, data_label
    #
    def __len__(self):
        return len(self.data_labels_list)
    #
    def get_read_file_list(self, path):
        """eg.
            >>read_file_list('data\\rock\\')
            ['data\\rock\\rock07-k03-118.png',...]
        """
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        return file_path_list
