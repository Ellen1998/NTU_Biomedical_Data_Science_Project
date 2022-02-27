import cv2
from cv2 import transform
import numpy as np
import os
import os.path as osp
import random
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F

class MnistDataset(data.Dataset):

    def __init__(self, data_root, data_split, transforms, bag_size=100):
        self.data_root = data_root
        self.data_split = data_split
        self.bag_size = bag_size
        self.transforms = transforms
        self.images, self.labels = self.get_all_infos(self.data_root, self.data_split)
    
    def get_all_infos(self, data_root, data_split):
        label_list = os.listdir(osp.join(data_root, data_split))
        res_images_list = []
        res_labels_list = []
        for label in label_list:
            tmp_images_list = []
            tmp_labels_list = []
            for img in os.listdir(osp.join(data_root, data_split, label)):
                tmp_images_list.append(osp.join(data_root, data_split, label, img))
                tmp_labels_list.append(label)
            res_images_list.append(tmp_images_list)
            res_labels_list.append(tmp_labels_list)

        return res_images_list, res_labels_list
    
    def read_image(self, file_name):
        image = cv2.imread(file_name)
        return image
    
    def __len__(self):
        return int((len(self.images[0]) + len(self.images[1])) / self.bag_size)
       
    def __getitem__(self, index):
        # get images per patch
        imgs_0_path_per_patch, imgs_7_path_per_patch = self.images[0], self.images[1]
        # get images number per patch
        ratio = random.random()
        num_0 = int(self.bag_size * ratio)
        num_7 = self.bag_size - num_0
        imgs_path_per_patch = random.sample(imgs_0_path_per_patch, num_0) + random.sample(imgs_7_path_per_patch, num_7)
        imgs_per_batch = []
        for img_path in imgs_path_per_patch:
            img = self.read_image(img_path)
            img = img / img.max()
            imgs_per_batch.append(img)
        # convert to tensor
        imgs_per_batch = torch.from_numpy(np.array(imgs_per_batch, dtype=np.float32))
        imgs_per_batch = imgs_per_batch.permute(0,3,1,2)

        return imgs_per_batch, ratio

        



