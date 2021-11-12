import datetime
import json
import os

import numpy as np
import cv2
from PIL import Image



INFO = {
    "description": "SOLOMON Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "trungpham2606, Brilian",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

def create_annotation_info(boundingbox_coor, image_id, category_info, image_size, annotation_id, gen_angle=None):
    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": category_info["iscrowd"],
        "bbox": boundingbox_coor.tolist(),
        "segmentation": None,
    }
    if gen_angle is not None:
        annotation_info['angle'] = gen_angle
    return annotation_info


def create_image_info(image_id, 
                      file_name, 
                      image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      er_id=None):
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
    }
    if er_id is not None:
        image_info['NTask'] = 'Task_{}'.format(er_id)  if type(er_id) == int else er_id

    return image_info

from time import time
def convert_coco_format(ROOT_DIR, name_classes, ndds=False, smallest_obj=30, autogen_angle=False, 
                        use_keypoint=False, use_existing_anno=[False, '', '', 0, 200], json_name='annotations.json'):
    start = time()
    image_id = 0
    annotation_id = 1
    folder_ntask = ''
    print('[INFO] name_classes: ', name_classes, )#flush=True)

    IMAGE_DIR = ROOT_DIR
    ANNO_DIR = os.path.join(ROOT_DIR, "Annotation", "maskAnot.txt")
    er_id = None
    ntask = ''

    if not use_existing_anno[0] or not os.path.exists(use_existing_anno[1]):
        CATEGORIES = []
        for i, n in enumerate(name_classes):
            info_dict = dict()
            info_dict['id'] = i+1
            info_dict['name'] = n
            info_dict['supercategory'] = 'SOLOMON'
            CATEGORIES.append(info_dict)
        coco_output = {
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }
        if use_existing_anno[0]:
            coco_output["images_rootdir"] = []
            
        if use_existing_anno[0] and len(use_existing_anno) == 2:    
            with open(os.path.join(IMAGE_DIR, json_name), 'r') as uea:
                coco_output = json.load(uea)
            annotation_id = coco_output['annotations'][-1]['id']+1
            image_id = len(coco_output['images'])
            
        with open(ANNO_DIR, 'r') as anno_file:
            lines = anno_file.readlines()
            num_images = len([1 for l in lines if '# ' in l])
    else:
        with open(use_existing_anno[1], 'r') as uea:
            coco_output = json.load(uea)
            
    if coco_output.get('NTask', False) != False:
        coco_output['NTask'] =  coco_output['NTask'] + 1 if type(use_existing_anno[3]) == int else use_existing_anno[3]
        er_id = coco_output['NTask'] if type(use_existing_anno[3]) == int else use_existing_anno[3]
        ntask = 'Task_{}'.format(er_id) if type(use_existing_anno[3]) == int else use_existing_anno[3]
    if use_existing_anno[0] and len(use_existing_anno) > 2:
        if coco_output.get('NTask', False) == False and use_existing_anno[3] != 0:
            er_id = use_existing_anno[3]
            ntask = 'Task_{}'.format(er_id)  if type(er_id) == int else er_id
            coco_output['NTask'] = 1
        base_path = os.path.dirname( use_existing_anno[1] )
        IMAGE_DIR = base_path if os.path.exists(os.path.join(base_path, ntask)) else IMAGE_DIR
        lines = use_existing_anno[2]
        num_images = len([1 for l in lines if '# ' in l])
        image_id = coco_output['images'][-1]['id'] if len(coco_output['images']) > 0 else  image_id
        annotation_id = coco_output['annotations'][-1]['id'] + 1 if len(coco_output['annotations']) > 0 else annotation_id
        folder_ntask = os.path.join(IMAGE_DIR, ntask)
        coco_output["images_rootdir"].append({'NTask':ntask, 
                                               'rootdir': IMAGE_DIR if not os.path.exists(folder_ntask) else folder_ntask,
                                               'num_images': num_images,
                                               'image_id': []})
    total_img = num_images + len(coco_output['images'])
    
    for i in range(len(lines)):
        last_im_id = len(coco_output['images'])+1
        if (annotation_id == len(lines)-image_id-1) or '#' in lines[i] and last_im_id % 10 == 0:
            print('Converting [{:6d}/{:6d}] - anno: {}...'.format(last_im_id, total_img, annotation_id+1), )#flush=True)
        if '#' in lines[i]:
            image_id += 1
            imgfile = lines[i].split(' ')[1].split('\n')[0]
#            image_name = os.path.join(IMAGE_DIR, ntask, imgfile)
            image_name = os.path.join(IMAGE_DIR if not os.path.exists(folder_ntask) else folder_ntask, imgfile)
            image = Image.open(image_name)
            if image is not None:
                try:
                    h, w = image.shape[:2]
                except:
                    w, h = image.size
                image_info = create_image_info(image_id, imgfile, (w, h), er_id=er_id)
                coco_output["images"].append(image_info)
                if er_id is not None:
                    coco_output["images_rootdir"][-1]['image_id'] += [last_im_id]
            else:
                continue

        else:
            angle = -1
            try:
                anno_line = lines[i].split(' ')[0] 
                obj_class_name = lines[i].split(' ')[1]
                angle = int( eval(lines[i].split(' ')[-1]) )
            except:
                continue
            
            anno_file = os.path.join(ROOT_DIR, 'SegmentImgs', anno_line)
            if not os.path.exists(anno_file):
                anno_line_temp = anno_line.split('_')
                anno_line_temp[-1] = ('0'*(4-len(anno_line_temp[-1].split('.')[0]))) + anno_line_temp[-1] 
                anno_file = os.path.join(ROOT_DIR, 'SegmentImgs', '_'.join(anno_line_temp))
            if image is None or not os.path.exists(anno_file):
                continue
            else:
                with open(anno_file, 'r') as ano:
                    ls = ano.readlines()
                    all_poitns = []
                    for l in ls:
                        if '#' not in l:
                            nl = l.split(',')
                            x = float(nl[0])
                            y = float(nl[1])
                            all_poitns.append([x, y])

                    segmentation = all_poitns.copy()
                    segmentation = np.reshape(segmentation, (-1,)).tolist()
                    all_poitns = np.asarray(all_poitns, dtype=np.int32)
                    all_poitns = np.reshape(all_poitns, (1, -1, 2))
                    x, y, w, h = cv2.boundingRect(all_poitns)
                    boundingbox_coor = np.array([x, y, w, h])
                        
                    try:
                        class_id = [x['id'] for x in CATEGORIES if x['name'] in obj_class_name][0]
                    except:
                        class_id = obj_class_name
                    category_info = {'id': class_id, 'iscrowd': 0}

                    annotation_info = create_annotation_info(boundingbox_coor, image_id, category_info, (w, h),
                                                             annotation_id)
                    annotation_info['segmentation'] = [segmentation]
                    
                    if annotation_info is not None:
                        if angle != -1:
                            annotation_info['angle'] = angle
                        if autogen_angle:
                            # add 90 because solvision will start 0 degree at 12 o'clock.
                            annotation_info['angle'] = (360 + cv2.minAreaRect(all_poitns)[-1] - 90) % 360
                        coco_output["annotations"].append(annotation_info)
                    annotation_id += 1
                    
    if use_existing_anno[0] and len(use_existing_anno) > 2:
        IMAGE_DIR = base_path
    print('[INFO] Start converting {}...'.format(json_name), )#flush=True)
    print('[INFO] Path: {}'.format(IMAGE_DIR), )#flush=True)
    with open(os.path.join(IMAGE_DIR, json_name), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    end = time()
    print('[INFO] converting time from SOLVISION to COCO format ({}): {:.3f}s'.format(json_name, end-start), )#flush=True)
    
if __name__ == "__main__":
    convert_coco_format(ROOT_DIR = r'D:\dataset\unknown_segmentation_dataset\unknown_son3', name_classes= ['Object'],
                        use_existing_anno=[True, ''])
    pass