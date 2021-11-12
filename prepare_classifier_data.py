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
train_df['normal'] = train_df['normal'].astype(int)
#%% Create annotation csv
train_df_normal = train_df[train_df['normal']==1]
train_df_reverse = train_df[train_df['normal']==0]

img_normal_list = train_df_normal['filename'].tolist()
label_normal_list = train_df_normal['normal'].tolist()
img_reverse_list = train_df_reverse['filename'].tolist()
label_reverse_list = train_df_reverse['normal'].tolist()

train_img_list = img_normal_list[:len(img_normal_list)-1000] + img_reverse_list[:len(img_reverse_list)-500]
val_img_list = img_normal_list[len(img_normal_list)-1000:] + img_reverse_list[len(img_reverse_list)-500:]
train_label_list = label_normal_list[:len(label_normal_list)-1000] + label_reverse_list[:len(label_reverse_list)-500]
val_label_list = label_normal_list[len(label_normal_list)-1000:] + label_reverse_list[len(label_reverse_list)-500:]


train_classifier = pd.DataFrame({'filename':train_img_list, 'label':train_label_list})
val_classifier = pd.DataFrame({'filename':val_img_list, 'label':val_label_list})
train_classifier.to_csv(os.path.join(root_path, 'Training Label', 'train_classifier_final_mix.csv'), index=False)
val_classifier.to_csv(os.path.join(root_path, 'Training Label', 'val_classifier_final_mix.csv'), index=False)
#%% Create crop data folder
out_path = os.path.join(root_path, 'public_training_data', 'classifier_data_final_mix')
os.makedirs(out_path, exist_ok=True)

for data in tqdm.tqdm(train_df.values):
    img_name, pts = data[0]+'.jpg', data[2:10]
    pts = pts.reshape(4,2).astype(np.int32)
    img = cv2.imread(os.path.join(data_path, img_name))
    pts[:,0] = np.clip(pts[:,0], a_min=0, a_max=img.shape[1])
    pts[:,1] = np.clip(pts[:,1], a_min=0, a_max=img.shape[0])
    xmin, ymin, xmax, ymax = np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])
    img_crop = img.copy()[ymin:ymax, xmin:xmax]
    cv2.imwrite(os.path.join(out_path, img_name), img_crop)











