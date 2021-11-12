import numpy as np
import os
import pandas as pd
import cv2
import copy
import json
import tqdm

root_path = r'/home/solomon/public/Pawn/Others/othercom/CL_OCR' 
data_path = os.path.join(root_path, 'Images')
# train_df = pd.read_csv(os.path.join(root_path, 'Training Label', 'public_total_data_final.csv'))
train_df = pd.read_csv(os.path.join(root_path, 'Training Label', 'final_data_mix.csv'))
#%% Create instances_training.json

segmentation_id=0
CATEGORIES=[{'id':1, 'name':'text'}]
coco_output = {"categories": CATEGORIES,"images": [],"annotations": []}
image_id=0

for data in tqdm.tqdm(train_df.values):
    img_name, pts = data[0]+'.jpg', data[2:10]
    img = cv2.imread(os.path.join(data_path, img_name))
    image_info = {
        "id": image_id,
        "file_name": img_name,
        "width": img.shape[1],
        "height": img.shape[0],
        }
    coco_output["images"].append(image_info)
    
    pts = pts.reshape(4,2).astype(np.int32)
    pts[:,0] = np.clip(pts[:,0], a_min=0, a_max=img.shape[1])
    pts[:,1] = np.clip(pts[:,1], a_min=0, a_max=img.shape[0])
    xmin, ymin, xmax, ymax = np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])
    box = [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)]
            
    seg = [[box[0], 
            box[1],
            box[0]+box[2], 
            box[1],
            box[0]+box[2], 
            box[1]+box[3],
            box[0], 
            box[1]+box[3],
            ]]
    annotation_info = {
        "iscrowd": 0,
        "category_id": 1,
        "bbox": box,
        "area": box[2]*box[3],
        'segmentation':seg,
        "image_id": image_id,
        "id": segmentation_id,
        "angle":-1
    }
    coco_output["annotations"].append(annotation_info)
    image_id+=1
    segmentation_id+=1
with open(os.path.join(root_path, 'Training Label', 'instances_training.json'), 'w') as f:
    json.dump(coco_output, f)
with open(os.path.join(root_path, 'Images', 'trainval.json'), 'w') as f:
    json.dump(coco_output, f)

#%% Create instances_test.json
segmentation_id=0
CATEGORIES=[{'id':1, 'name':'text'}]
coco_output = {"categories": CATEGORIES,"images": [],"annotations": []}
image_id=0

for i, data in tqdm.tqdm(enumerate(train_df.values)):
    img_name, pts = data[0]+'.jpg', data[2:10]
    img = cv2.imread(os.path.join(data_path, img_name))
    image_info = {
        "id": image_id,
        "file_name": img_name,
        "width": img.shape[1],
        "height": img.shape[0],
        }
    coco_output["images"].append(image_info)
    
    pts = pts.reshape(4,2).astype(np.int32)
    pts[:,0] = np.clip(pts[:,0], a_min=0, a_max=img.shape[1])
    pts[:,1] = np.clip(pts[:,1], a_min=0, a_max=img.shape[0])
    xmin, ymin, xmax, ymax = np.min(pts[:,0]), np.min(pts[:,1]), np.max(pts[:,0]), np.max(pts[:,1])
    box = [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)]
            
    seg = [[box[0], 
            box[1],
            box[0]+box[2], 
            box[1],
            box[0]+box[2], 
            box[1]+box[3],
            box[0], 
            box[1]+box[3],
            ]]
    annotation_info = {
        "iscrowd": 0,
        "category_id": 1,
        "bbox": box,
        "area": box[2]*box[3],
        'segmentation':seg,
        "image_id": image_id,
        "id": segmentation_id,
        "angle":-1
    }
    coco_output["annotations"].append(annotation_info)
    image_id+=1
    segmentation_id+=1
    if(i==10):break
with open(os.path.join(root_path, 'Training Label', 'instances_test.json'), 'w') as f:
    json.dump(coco_output, f)
