
import torch
import torch.utils.data as tud
import os
import numpy as np
import cv2
import json
from torchvision import transforms as tvt
from PIL import Image
import matplotlib.pyplot as plt
import logging
import random
import copy
import codecs
import pandas as pd
import torchvision.transforms.functional as TF


class Pure_Dataset(tud.Dataset):
    def __init__(self, train_csv='', img_dir='', resize=(224,224), transform=None, **kargs):
        super(Pure_Dataset, self).__init__()
        self.img_dir = img_dir
        print(f'[{type(self).__name__}] img_dir: ', self.img_dir)
        
        train_df = pd.read_csv(train_csv)
        self.nclasses = 2        
        self.resize = resize
        self.transform=transform
        self.len_images = len(train_df)
        print('[{type(self).__name__}] len_images: ', self.len_images)
        
        # Feel for free to add more settings.
        self.im_load = 'cv2'
        self.resize_done = False
        self.im_type = '.jpg'
        data_opt = kargs['data_opt']
        for k, v in data_opt.items():
            print(k)
            if(k=='im_load'):
                self.im_load = v
            elif(k=='resize_done'):
                self.resize_done=v
            elif(k=='im_type'):
                self.im_type = v
            elif(k=='use_Gray'):
                self.use_Gray = v    
            elif(k=='rand_anti_white'):
                self.rand_anti_white = v
            elif(k=='use_reverse'):
                self.use_reverse = v
        print(f'data_opt:\n{data_opt}')
        
        self.samples = train_df.to_numpy()

        class_counts = train_df['label'].value_counts().sort_index().to_numpy()
        num_samples = sum(class_counts)
        assert num_samples == self.len_images
        labels=[]
        for i, cls_count in enumerate(class_counts):
            labels.extend([i]*cls_count)
        
        class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
        weights = [class_weights[labels[i]] for i in range(int(num_samples))]
        from torch.utils.data import WeightedRandomSampler
        # self.weighted_random_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples), replacement=False)
        self.weighted_random_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
        self.denormalize_image = tvt.Normalize(
            mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
            std=[1/0.5, 1/0.5, 1/0.5])
        
    def __len__(self):
        return self.len_images
    
    def read_image(self, impath):
        image = cv2.imread(impath)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def __getitem__(self, idx):
        
        pick_im, label = self.samples[idx]
        pick_im = pick_im+'.jpg'
        img_path = os.path.join(self.img_dir, pick_im)
        if self.im_load =='PIL':
            image = Image.open(img_path)
            image = image.resize(tuple(self.resize))
        else:
            image = self.read_image(img_path)
            image = cv2.resize(image, tuple(self.resize))
            image = Image.fromarray(image)
        if(self.use_Gray):
            image = image.convert('L')
            image = image.convert('RGB')


        if self.transform:
            image = self.transform(image)
            if(self.rand_anti_white):
                if(random.getrandbits(1)==0):
                    image = 255 - image

            '''For reverse image'''
            if(self.use_reverse):
                if random.random() > 0.5:
                    angle = 180
                    image = TF.rotate(image, 180)
                    label = not label



        # print(label, self.id_chara_map[str(label)])
        # self.show_image(image)
        return image, label

    def denormalize_image(self, image):
        return self.denormalize_image(image)
    
    def show_image(self, image):
        image_denorm = self.denormalize_image(image)
        plt.imshow(image_denorm.numpy().transpose(1,2,0))
        plt.show()

if __name__ == '__main__':
    jsonpath = 'get_classlist_train21_test19_unland.json'
    rootpath = '/home/solomon/public/Datasets/landmark-recognition-21_19'
    resize = (224,224)
    transform = tvt.Compose([
        # tvt.CenterCrop(224),
        # tvt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        tvt.ToTensor(),
        tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_opt = {
        'im_load':'cv2',
        'resize_done':False,
        'im_type':'.jpg',
    }

    cldataset = Pure_Dataset(train_csv=r'E:\Research_code\myProject\Competition\Chinese_OCR\train\train_cls.csv',
                             label_map=r'E:\Research_code\myProject\Competition\Chinese_OCR\train\cls_label_map.json',
                             img_dir=r'E:\Research_code\myProject\Competition\Chinese_OCR\train\train_cls_dataset',
                             resize=resize,
                             transform=transform,
                             data_opt=data_opt)

            
    # # TEST SANITY DATALOADER
    dataloader = tud.DataLoader(cldataset, batch_size=12, shuffle=True, num_workers=0)

    import time
    s_time, tmp_time  = time.time(), time.time()
    for i, (images, labels, infos) in enumerate(dataloader):
        # print(labels)
        # print(infos)
        pass

    
    