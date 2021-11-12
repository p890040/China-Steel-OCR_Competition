# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:28:01 2020

@author: user
"""

import json
import os
import numpy as np
import shutil as sh
from detectron2.utils.convert_mask_dataset import convert_coco_format
from time import strftime, localtime

class MemoryGather:
    def __init__(self, base_dir='', tool_name='Unknown'):
        self.base_dir = base_dir
        self.current_alloc_pertask = []
        self.image_id_pertask = []
        self.num_images = []
        self.num_images_total = 0
        self.json_name = 'annotations.json' if tool_name.lower() =='unknown' else 'trainval.json'
        self.get_coco_anno = False
        self.coco_output = None
        
        self.moveimg2onefolder = False

    @staticmethod
    def get_maskanot_id_index(maskanot_txt):
        maskanot_ctr = []
        for idx, mat in enumerate(maskanot_txt):
            if '# ' in mat:
                maskanot_ctr.append(idx)
        return maskanot_ctr
    
    @staticmethod
    def generate_maskAnot_from_datasets(ld):
        with open(os.path.join(ld, 'Annotation', 'datasets.txt'),'r') as datfile:
                datasets_txt = datfile.read().split('\n')
        mask_from_dataset_txt = []
        ctr = 0
        while ctr < len(datasets_txt) - 1:
            if '# ' in datasets_txt[ctr]:
                if ctr %10 == 0: print('Progress [{}/{}]'.format(ctr, len(datasets_txt)), )#flush=True)
                ctr += 1
                mask_from_dataset_txt.append('# {}'.format(datasets_txt[ctr]))
                ctr += 1
                n_anno_txt = eval(datasets_txt[ctr])
                for _ in range(n_anno_txt):
                    ctr+= 1
                    class_split = 1
                    try:
                        test_split = datasets_txt[ctr].split('\t')
                        if len(test_split) < 3:
                            raise
                        test_split = test_split[-1]
                        class_split = datasets_txt[ctr].split('\t')[0]
                        angle_split = datasets_txt[ctr].split('\t')[-2]
                    except:
                        test_split = datasets_txt[ctr].split(' ')[-1]
                        class_split = datasets_txt[ctr].split(' ')[0]
                        angle_split = datasets_txt[ctr].split(' ')[-2]
                    mask_from_dataset_txt.append('{} {} {}'.format(test_split.split('.')[0]+'.txt', class_split, angle_split))
                ctr += 1
        with open(os.path.join(ld, 'Annotation', 'maskAnot.txt'),'w') as matfile:
            matfile.write('\n'.join(mask_from_dataset_txt))
        return mask_from_dataset_txt 
    
    @staticmethod
    def get_sample_maskanot(maskanot_txt, limit=0, maskanot_ctr=[], prefix_name=''):
        maskanot_txt2 = []
        # maskanot_ctr = get_maskanot_id_index(maskanot_txt)
        pick_ctr = np.random.permutation(len(maskanot_ctr))[:limit]
        pick_ctr = np.array(maskanot_ctr)[pick_ctr]
        pick_mask = []
        for idx, pc in enumerate(pick_ctr):
            getmask = maskanot_txt[pc]
            if idx % 10 == 0:
                print('[{:3d}/{}]. getmask: {}'.format(idx, limit, getmask), )#flush=True)
            pick_mask.append(getmask)
            maskanot_txt2.append('# {}{}'.format(prefix_name, getmask.split(' ')[-1]) )
            
            ctr = 1
            while '# ' not in maskanot_txt[pc+ctr]:
                maskanot_txt2.append(maskanot_txt[pc+ctr])
                ctr += 1
                if pc+ctr >= len(maskanot_txt):
                    break
        return maskanot_txt2, pick_mask
    
    # ref: http://10.1.30.16:3000/AI-Team/knowledge-distillation/src/branch/feature/ER_with_GEM/ER.py
    def limit_images_per_task(self, anno, limit_all_task=1000):
        with open(anno, 'r') as uea:
            coco_output = json.load(uea)
        
        images_rootdir = coco_output['images_rootdir']
        
        num_current_all_task = np.sum([ir['num_images'] for ir in images_rootdir])
        
        print('Number of images from all tasks: {}, Limit: {}'.format(num_current_all_task, limit_all_task), )#flush=True)
        if num_current_all_task <= limit_all_task:
            return
        
        curr_task_ratio = limit_all_task/num_current_all_task
        current_alloc_pertask = [int(round(ir['num_images'] * curr_task_ratio)) \
                                 if int(round(ir['num_images'] * curr_task_ratio))>= 10 \
                                 else ir['num_images'] for ir in images_rootdir]
        
        # check leftover data, add to the first n tasks
        leftover_alloc_pertask = limit_all_task - np.sum(current_alloc_pertask)
        if leftover_alloc_pertask < 0:
            current_alloc_pertask[-1] += leftover_alloc_pertask
        else:
            for i in range(len(current_alloc_pertask)):
                if leftover_alloc_pertask == 0:
                    break
                current_alloc_pertask[i] += 1
                leftover_alloc_pertask -= 1
        self.current_alloc_pertask = current_alloc_pertask
        print('\nleftover_alloc_pertask: {} {}\n'.format(leftover_alloc_pertask, current_alloc_pertask), )#flush=True)        
        
        annotations = coco_output['annotations']
        
        img_to_pop = []
        print('='*50, )#flush=True)
        for idx, ir in enumerate(images_rootdir):
            samples_current_id = np.array(ir['image_id'])
            ndata = np.random.permutation(ir['num_images'])
            nthrow = ndata[current_alloc_pertask[idx]:]
            sample_to_throw = samples_current_id[nthrow].tolist()
            print('{}- Num images: {}, Remove {} images from anno...'.format(ir['NTask'],
                                                              ir['num_images'],
                                                              len(sample_to_throw)), )#flush=True )
            images_rootdir[idx]['num_images'] = 0
            images_rootdir[idx]['image_id'] = []
            for im_ctr, im_id in enumerate(sample_to_throw):
                im_info = coco_output['images'][im_id-1]
                img_to_pop.append(im_id)
                if self.moveimg2onefolder:
                    im_to_remove = os.path.join(images_rootdir[idx]['rootdir'], im_info['file_name'])
                    os.remove(im_to_remove )
                anno_idx = 0
                while anno_idx < len(annotations):
                    if im_info['id'] == annotations[anno_idx]['image_id']:
                        annotations.pop(anno_idx)
                    else:
                        anno_idx += 1        
            
        # keep the images outside of the img_to_pop
        coco_output['images'] = [im for idx,im in enumerate(coco_output['images']) if idx+1 not in img_to_pop]
        coco_output['annotations'] = annotations
        print('*'*20, )#flush=True)
        # update image id again, because the position will keep changing after filtering
        for idx_co, co in enumerate(coco_output['images']):
            for idx_ir, ir in enumerate(images_rootdir):
                if co['NTask'] == ir['NTask']:
                    images_rootdir[idx_ir]['image_id'].append(idx_co+1)
                    images_rootdir[idx_ir]['num_images'] += 1
        self.num_images = []
        self.image_id_pertask = []
        self.num_images_total = 0
        for idx_ir, ir in enumerate(images_rootdir):
            print('{}, Current num_images: {}'.format(ir['NTask'], ir['num_images']), )#flush=True)
            self.num_images += [ir['num_images']]
            self.image_id_pertask.append(ir['image_id'])
            self.num_images_total += ir['num_images']
        print('[INFO] Total num_images: {}'.format(self.num_images_total), )#flush=True)
        print('='*50 + '\n', )#flush=True)
        
        # re-save the json_name
        with open(anno, 'w') as output_json_file:
            json.dump(coco_output, output_json_file)
            
        if self.get_coco_anno:
            self.coco_output = coco_output
    

    def gather_memory_task(self, curr_task_dir, 
                           prev_task_dir=None, 
                           dst_dir=None,
                           limit_memory_pertask = 0.4,
                           smallest_image_total = 50,
                           limit_image_total = 200, ntask=0, limit_all_task=1000, 
                           use_date_format_folder=False, date_format_name=None):
        prev_task_dir = self.base_dir if prev_task_dir is None else prev_task_dir
        dst_dir = self.base_dir if dst_dir is None else dst_dir
        if os.path.exists( os.path.join(dst_dir, self.json_name) ):
            with open(os.path.join(dst_dir, self.json_name), 'r') as uea:
                ntask = int(json.load(uea)['NTask']) + 1
        date_now = strftime("%Y%m%d_%H%M", localtime()) if date_format_name is None else date_format_name
        ntask = ntask if not use_date_format_folder else date_now
        prefix_folder_task = 'Task_{}'.format(ntask) if self.moveimg2onefolder and not use_date_format_folder else date_now if use_date_format_folder else ''
        
        prefix_name = prefix_folder_task + '_' if self.moveimg2onefolder else ''
        os.makedirs(dst_dir, exist_ok=True)
        if self.moveimg2onefolder:
            os.makedirs(os.path.join(dst_dir, prefix_folder_task), exist_ok=True)
            os.makedirs(os.path.join(os.path.join(dst_dir, prefix_folder_task), 'Annotation'), exist_ok=True)
        if prev_task_dir != '' and prev_task_dir != dst_dir:
            if self.moveimg2onefolder:
                list_task_prev = os.listdir(prev_task_dir)
                list_task_im_prev = [im for im in list_task_prev if im.endswith('png') or im.endswith('jpg') or im.endswith('jpeg') or im.endswith('bmp')]
                print('\n[INFO] Copy prev file task from-{} to {}\n'.format(prev_task_dir, dst_dir), )#flush=True)
                
                for idx,ltip in enumerate(list_task_im_prev):
                    sh.copyfile(os.path.join(prev_task_dir, ltip), os.path.join(dst_dir, prefix_folder_task, ltip)) 
            sh.copyfile(os.path.join(prev_task_dir, self.json_name), os.path.join(dst_dir, self.json_name)) 
            print('\n[INFO] Copy prev file anno from-{} to {}\n'.format(self.json_name, dst_dir), )#flush=True)
            
        if self.moveimg2onefolder:
            # copy current task files
            sh.copyfile(os.path.join(curr_task_dir, 'Annotation', 'class_name.txt'), os.path.join(dst_dir, prefix_folder_task, 'Annotation', 'class_name.txt')) 
        
        name_classes = np.loadtxt(os.path.join(curr_task_dir, 'Annotation', 'class_name.txt'), dtype=str)[1:,0].tolist()
        
        list_ano_txt = os.listdir(os.path.join(curr_task_dir, 'Annotation'))
        if 'datasets.txt' in list_ano_txt:    
            maskanot_txt = MemoryGather.generate_maskAnot_from_datasets(curr_task_dir)
        else:
            print('[ER INFO-MEMORY GATHER] No datasets.txt found, raise exception...')
            raise 
        # define limit per memory
        maskanot_ctr = MemoryGather.get_maskanot_id_index(maskanot_txt)
        lim_mem_final = int(limit_memory_pertask*len(maskanot_ctr)) if limit_memory_pertask*len(maskanot_ctr) < limit_image_total else limit_image_total    
        lim_mem_final = lim_mem_final if lim_mem_final > smallest_image_total else len(maskanot_ctr[:smallest_image_total])
        sampled_maskanot_txt, sampled_img = MemoryGather.get_sample_maskanot(maskanot_txt, 
                                                                limit=lim_mem_final, 
                                                                maskanot_ctr=maskanot_ctr,prefix_name=prefix_name)
        
        print('\n[INFO] Copy current file task from-{} to {}\n'.format(curr_task_dir, dst_dir), )#flush=True)
        # for idx,ltip in enumerate(list_task_im):
        if self.moveimg2onefolder:
            for idx,si in enumerate(sampled_img):
                ltip = si.split(' ')[-1]
                sh.copyfile(os.path.join(curr_task_dir, ltip), os.path.join(dst_dir, prefix_folder_task, prefix_name+ltip)) 
        
        # get class name
        print('\n[INFO] Append current task to memory...\n')
        convert_coco_format(ROOT_DIR=curr_task_dir, 
                            name_classes=name_classes, 
                            use_existing_anno=[True, 
                                               os.path.join(dst_dir, self.json_name), 
                                               sampled_maskanot_txt, 
                                               ntask,
                                               limit_image_total],
                           json_name=self.json_name)
        print('\n[INFO] Append current task to memory done...\n', )#flush=True)
        
        self.limit_images_per_task(os.path.join(dst_dir, self.json_name), limit_all_task=limit_all_task)
        print('\n[INFO] Validate limit image per task done...\n', )#flush=True)
    
    def gather_memory_task_from_list(self, list_of_task, get_coco_anno=False, limit_all_task=500, smallest_image_total=50, 
                                     limit_image_total = 200, use_date_format_folder=False, date_format_name=None):
        list_of_task = list_of_task.split(',')
        list_of_task = [lot for lot in list_of_task if lot != '']
        
        for i in range(len(list_of_task)):
            if i == len(list_of_task) - 1 and get_coco_anno:
                self.get_coco_anno = get_coco_anno
            curr_task_dir = list_of_task[i]
            self.gather_memory_task(curr_task_dir, ntask = 1, 
                                    limit_all_task       = limit_all_task, 
                                    smallest_image_total = smallest_image_total,
                                    limit_image_total    = limit_image_total,
                                    use_date_format_folder = use_date_format_folder,
                                    date_format_name     = date_format_name)
if __name__ == '__main__':
    prev_task_dir = r'D:\GitSource\Project_bank\ER\ContinualImages'
#    dst_dir = r'D:\GitSource\FamilyMart\memory_samples'
#    curr_task_dir = r'D:\GitSource\FamilyMart\Release\Projects\Instance Segmentation4 Tool1\Images'
    memgather = MemoryGather(base_dir=prev_task_dir, tool_name='Unknown')
    curr_task_dir = [r'D:\GitSource\Project_KD10T\Task7_Familymart\Images',
                         r'D:\GitSource\Project_KD4T\Task3_Carton\Images',
                         r'D:\GitSource\Project_bank\fedex_case2\Unknown Segmentation Tool3\Images',
                         r'D:\GitSource\Component\AI_Vision_WPF_3.2\TaskProcess\bin\x64\Release\Projects_KD\Unknown Segmentation Tool2\Images',
                         r'D:\GitSource\Project_KD10T\Task8_K2ABCDE\Images']
# =============================================================================
#     METHOD 1, USING LIST
# =============================================================================
    memgather.moveimg2onefolder = True
    import time
    temp_date = ''
    for i in range(7):
        pick_ctd = np.random.choice(curr_task_dir)
        curr_task_dir_now = pick_ctd
        date_now = strftime("%Y%m%d_%H%M", localtime())
        print('\v>>> Date: {}, {}'.format(temp_date, date_now))
        if temp_date != date_now:
            temp_date = date_now
        else:
            time.sleep(60)
            date_now = strftime("%Y%m%d_%H%M", localtime())
        memgather.gather_memory_task(curr_task_dir_now, ntask=1, limit_all_task=400, use_date_format_folder=True, date_format_name=date_now)
        
# =============================================================================
#         METHOD 2, USING STRING FOR SOLVISION
# =============================================================================
    # curr_task_dir_str = ','.join(curr_task_dir)
    # memgather.moveimg2onefolder = True
    # memgather.gather_memory_task_from_list(curr_task_dir_str, limit_all_task=500, use_date_format_folder=True)
    
        
        