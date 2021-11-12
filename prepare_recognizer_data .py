import numpy as np
import os
import pandas as pd
import cv2
import copy
import tqdm

root_path = r'/home/solomon/public/Pawn/Others/othercom/CL_OCR' 
data_path = os.path.join(root_path, 'public_training_data', 'public_training_data')
# train_df = pd.read_csv(os.path.join(root_path, 'Training Label', 'public_total_data_final.csv'))
train_df = pd.read_csv(os.path.join(root_path, 'Training Label', 'final_data_mix.csv'))
#%% Create annotation csv
train_df_normal = train_df[train_df['normal']]
train_df_reverse = train_df[~train_df['normal']]

img_normal_list = train_df_normal['filename'].tolist()
label_normal_list = train_df_normal['label'].tolist()
img_reverse_list = train_df_reverse['filename'].tolist()
label_reverse_list = train_df_reverse['label'].tolist()

train_img_list = img_normal_list + img_reverse_list
val_img_list = img_normal_list[len(img_normal_list)-1000:] + img_reverse_list[len(img_reverse_list)-500:]
train_label_list = label_normal_list + label_reverse_list
val_label_list = label_normal_list[len(label_normal_list)-1000:] + label_reverse_list[len(label_reverse_list)-500:]

with open(os.path.join(root_path, 'Training Label', 'train_recognizer_final_mix.txt'), 'w') as f:
    assert len(train_img_list) == len(train_label_list)
    for i in range(len(train_img_list)):
        f.write(train_img_list[i]+'.jpg'+' '+train_label_list[i]+'\n')

with open(os.path.join(root_path, 'Training Label', 'val_recognizer_final_mix.txt'), 'w') as f:
    assert len(val_img_list) == len(val_label_list)
    for i in range(len(val_img_list)):
        f.write(val_img_list[i]+'.jpg'+' '+val_label_list[i]+'\n')
#%% Create crop data folder
out_path = os.path.join(root_path, 'public_training_data', 'recognizer_data_final_mix')
os.makedirs(out_path, exist_ok=True)

for data in tqdm.tqdm(train_df.values):
    img_name, pts = data[0]+'.jpg', data[2:10]
    is_normal = data[-1]
    
    pts = pts.reshape(4,2).astype(np.int32)
    img = cv2.imread(os.path.join(data_path, img_name))
    pts[:,0] = np.clip(pts[:,0], a_min=0, a_max=img.shape[1])
    pts[:,1] = np.clip(pts[:,1], a_min=0, a_max=img.shape[0])
    xmin, ymin, xmax, ymax = np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])
    img_crop = img.copy()[ymin:ymax, xmin:xmax]
    if(not is_normal):
        h, w = img_crop.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, 180, scale)
        img_crop = cv2.warpAffine(img_crop, M, (w, h))
    cv2.imwrite(os.path.join(out_path, img_name), img_crop)