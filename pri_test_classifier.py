#%%
import numpy as np
import cv2, os
import torch
import codecs
import timm
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import yaml
import torch.utils.data as tud
from collections import Counter
import json
import time
import tqdm

def get_model(ckpt, model_name, nclass=2):
    model_cls = timm.create_model(model_name, pretrained=True)

    if(model_name == 'tf_efficientnetv2_m_in21k' or model_name == 'tf_efficientnetv2_s_in21k' or model_name == 'tf_efficientnetv2_l_in21k'):
        num_ftrs = 1280
        model_cls.classifier = torch.nn.Sequential(torch.nn.Dropout(0.001), torch.nn.Linear(num_ftrs, nclass))
    elif (model_name == 'vit_base_patch32_224_in21k' or model_name == 'vit_base_r50_s16_224_in21k'):
        num_ftrs = 768
        model_cls.head = torch.nn.Sequential(torch.nn.Dropout(0.001), torch.nn.Linear(num_ftrs, nclass))
    elif(model_name == 'resnest101e'):
        num_ftrs = 2048
        model_cls.fc = torch.nn.Sequential(torch.nn.Dropout(0.001), torch.nn.Linear(num_ftrs, nclass))
    else:
        raise "Error"

    model_cls = model_cls.cuda()
    model_cls.load_state_dict(torch.load(ckpt))
    model_cls.eval()
    return model_cls


class Classifier_Dataset(tud.Dataset):
    def __init__(self, img_dir, transforms, resizes):
        super(Classifier_Dataset, self).__init__()
        self.img_dir = img_dir
        self.img_names = [file for file in os.listdir(img_dir) if file.endswith(".jpg")]
        self.resizes = resizes
        self.transforms = transforms
        assert len(self.transforms) == len(self.resizes)
        self.len_images = len(self.img_names)
        print(f'[{type(self).__name__}] img_dir: ', self.img_dir)
        print('[{type(self).__name__}] len_images: ', self.len_images)
        
    def __len__(self):
        return self.len_images
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        imgs, img_names=[],[]
        for resize, transform in zip(self.resizes, self.transforms):
            img_single = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_single = cv2.resize(img_single, resize)
            img_single = transform(img_single)
            imgs.append(img_single)
            img_names.append(img_name)
        return imgs, img_names


if __name__ == '__main__':
    ROOT=os.getcwd()
    # path = '/home/solomon/public/Pawn/Others/othercom/CL_OCR/predict_detection'
    path = os.path.join(ROOT, 'private_detection_final')
    # path = '/home/solomon/public/Pawn/Others/othercom/CL_OCR/data_detection_val'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_imagenet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    configs =[
    os.path.join(ROOT, 'configs/cls_L_300.yml'),
    # os.path.join(ROOT, 'configs/cls_L_384.yml'),
    # os.path.join(ROOT, 'configs/cls_M_224.yml'),
    # os.path.join(ROOT, 'configs/cls_M_300.yml'),
    # os.path.join(ROOT, 'configs/cls_M_384.yml'),
    # os.path.join(ROOT, 'configs/cls_S_224.yml'),
    # os.path.join(ROOT, 'configs/cls_S_300.yml'),
    # os.path.join(ROOT, 'configs/cls_Vit_224.yml'),
    # os.path.join(ROOT, 'configs/cls_Res_256.yml'), 
    ]

    '''Prepare arguments'''
    models, transforms, resizes=[],[],[]
    for i, config in enumerate(configs):
        with open(config) as f:
            hyp = yaml.safe_load(f) 
        ckpt = os.path.join(ROOT, 'run', os.path.basename(config), 'weight', 'model_cls_final.pth')
        models.append(get_model(ckpt, hyp['model_name']))
        resizes.append((hyp['im_size'], hyp['im_size']))
        if(hyp.get('mean_std') == 'imagenet'):
            transforms.append(transform_imagenet)
        else:
            transforms.append(transform)

    '''Dataset loader'''
    test_Dataset = Classifier_Dataset(path, transforms=transforms, resizes=resizes)
    test_loader = tud.DataLoader(test_Dataset, batch_size=60, shuffle=False, num_workers=12, pin_memory=True)
    print(f'loader size : {len(test_loader)}')

    '''All models inference'''
    outputs=[[] for _ in range(len(models))]
    img_list=[]
    s_time = time.time()
    for itv, (imgs, img_names) in tqdm.tqdm(enumerate(test_loader)):
        assert len(models)==len(transforms)==len(resizes)
        num_infer=len(models)
        for i in range(num_infer):
            model = models[i]
            model.eval()
            with torch.no_grad():
                img_batches = imgs[i].cuda()
                out = model(img_batches)
                y_score, y_pred = torch.max(F.softmax(out,1), 1)
                output = y_pred.cpu().tolist()
            outputs[i].extend(output)
        img_list.extend(img_names[0])

    import copy
    outputs_ = copy.deepcopy(outputs)
    outputs_.insert(0,img_list)
    outputs_ = pd.DataFrame(outputs_)
    outputs_.to_csv(os.path.join(ROOT, 'private_classifier_final_ori.csv'), index=False)

    outputs = pd.DataFrame(outputs)
    vote_output=[outputs[i].value_counts()[:1].index.tolist()[0] for i in range(len(outputs.columns))]

    output_cls = pd.DataFrame({'filename':img_list,
                                'normal': vote_output
    })
    # output_cls.to_csv('/home/solomon/public/Pawn/Others/othercom/CL_OCR/predict_classifier.csv', index=False)
    # output_cls.to_csv('/home/solomon/public/Pawn/Others/othercom/CL_OCR/predict_classifier_voting7.csv', index=False)
    output_cls.to_csv(os.path.join(ROOT, 'private_classifier_final.csv'), index=False)
    print(time.time()-s_time)
    out_path_cls = os.path.join(ROOT, 'private_classifier_final')
    os.makedirs(out_path_cls, exist_ok=True)
    for file, normal in tqdm.tqdm(zip(img_list, vote_output)):
        img = cv2.imread(os.path.join(path, file))
        if(normal==0):
            h, w = img.shape[:2]
            center = (w / 2, h / 2)
            scale = 1.0
            M = cv2.getRotationMatrix2D(center, 180, scale)
            img = cv2.warpAffine(img, M, (w, h))
        cv2.imwrite(os.path.join(out_path_cls, file), img)



