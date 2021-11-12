import os
ROOT=os.getcwd()
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--root', default = ROOT, help='Code root path.')
parser.add_argument('--pretrain', default =os.path.join(ROOT, 'res50.pkl'), help='Pretrain model path.')
parser.add_argument('--project', default = ROOT, help='Dataset path as solvision format.')
parser.add_argument('--batch', default=1)
parser.add_argument('--mode', default='training_FeatureDetect')
parser.add_argument('--worker', default=12)
args = parser.parse_args()
print(args)
import numpy as np
import cv2
import sys

ROOT= os.getcwd()

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'D2'))
print('system path :')
print(sys.path)

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS=None

import detectron2
#from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor, DefaultPredictor_Batch
#from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances, register_semantic
from detectron2.modeling.postprocessing import PYTHON_INFERENCE
import time
import json
import copy
from detectron2.utils.logger import setup_logger
import detectron2.data.detection_utils as utils
import ctypes
import torch

from skimage import measure
import skimage

from detectron2.utils.memory_gather import MemoryGather
import tqdm
# torch.backends.cudnn.benchmark = True

exec_environment = 0
root_model = ""
model_type = "default"
#setup_logger()
def read_classname_file(arg_path):
    arg_path = os.path.join(os.path.dirname(os.path.abspath(arg_path)), 'class_name.txt')
    class_name, is_use_angle, color = [], [], []
    with open(arg_path) as f:
        text = [s.split() for s in f.readlines()]
        if(text[0][0] != 'BackGround'): 
            print('FORMAT ERROR!')
            raise 'FORMAT ERROR!'
        for name in text[1:]:
            class_name.append(name[0])
            is_use_angle.append(name[1])
            color.append(name[3])
    return class_name, is_use_angle, color

class ObjectConfig():
    def __init__(self, path):
        self.NAME = None        
        self.vars= dict()
        self.lst_int= ["min_dimension", "max_dimension", "max_iter", "save_by_second", "is_use_data_aug", "max_detections", "batch_size", "rpn_post_nms_top_n", "relative_bg", "pr_on", "use_er", "use_er_limit", 'use_er_smallest_limit_pertask', 'use_er_largest_limit_pertask']
        self.lst_float= ["test_score_thresh", "gpu_limit", "base_lr", "weight_decay", "relative_bg_value", "angle_loss_weight", "roi_filter_ratio"]
        self.lst_str= ["dataset_folder", "annotation_file", "fine_tune_checkpoint", "docolorjitter", "dohistogram", "doresizelow", "doflip", "dorotate", "donoise", "doshift", "dozoom", "use_er_dir", 'ac_size', 'ac_ratios', 'iou_thr', 'iou_label', "pre_topk_test", "post_topk_test", 'pre_topk_train', 'post_topk_train', 'd_batch', 'positive_fraction', 'nms_thr', 'lr_decrease']
        self.loadConfigFromFile(path)
        self.root_path= os.path.dirname(os.path.abspath(path))
        self.model_path= self.root_path
        
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def loadConfigFromFile(self, path):     
        self.vars["docolorjitter"] = "False,0.5,0.5,0.5,0.2,0.5"
        self.vars["dohistogram"] = "False,clahe,0.0"
        self.vars["doresizelow"] = "False,0.4,1.0"
        self.vars["doflip"] = "False,both,0.7"
        self.vars["dorotate"] = "False,180,0.7"
        
        with open(path) as json_data:            
            configParam = json.load(json_data)
        for key in list(configParam.keys()):
            if key in self.lst_int:        
                self.vars[key]= int(configParam[key])
            elif key in self.lst_float:
                self.vars[key]= float(configParam[key])
            elif key in self.lst_str:
                self.vars[key]= configParam[key]            

        self.vars["dataset_folder"] = os.path.join(args.project, 'Images')
        self.vars["annotation_file"] = os.path.join(args.project, 'Images', 'Annotation', 'datasets.txt')

        self.root_path= os.path.dirname(os.path.abspath(path))
        self.model_path= self.root_path
        self.class_name, self.is_use_angle, _ =  read_classname_file(self.vars['annotation_file'])
        self.vars['fine_tune_checkpoint'] = os.path.join(self.model_path, 'weights_predict.caffemodel')
        self.display()
        
    def changeConfig(self, key, value):
        if key in self.lst_int:        
            self.vars[key]= int(value)
        elif key in self.lst_float:
            self.vars[key]= float(value)
        elif key in self.lst_str:
            self.vars[key]= value
        else:
            print("Key [{0}] does not exit in Config".format(key))

class MaskRCNN_PYTORCH():
    
    def __init__(self, arg_mode, arg_config_path, arg_environ_path, gpu_device=''):
        torch.set_num_threads(1)
        self.environ_path = arg_environ_path
        self.ConfigPath= arg_config_path 
        self.config = ObjectConfig(arg_config_path)
#        self.init_model()
        self.isNeedLoadModel = True
        self.datasets_register_name = "links_"+time.ctime()
        self.arg_mode = arg_mode
        self.training = False if('detecting' in arg_mode) else True
        
        self.mode = 'mask'
        if('FeatureDetect' in arg_mode):
            self.mode = 'feature'
            
        print(f'[Logging] mode : {self.mode}')
        print(f'[Logging] root_model : {root_model}')
        print(f'[Logging] model_type : {model_type}')
        self.gpu_device = str(gpu_device)
        print(f'[Logging] gpu_device : {gpu_device}')
            
    def initail_cfg(self):
        self.cfg = get_cfg()
        if(self.mode == 'mask' or self.mode=='keypoint'):
            if(model_type == 'default'):
                cfg_path = os.path.join(self.environ_path, "D2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        elif(self.mode == 'feature'):
            if(model_type == 'default'):
                cfg_path = os.path.join(self.environ_path, "D2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.cfg.merge_from_file(cfg_path)

        self.cfg.OUTPUT_DIR = self.config.model_path
        self.cfg.DATASETS.TRAIN = (self.datasets_register_name,)
        self.cfg.DATASETS.TEST = (self.datasets_register_name, )  
        self.cfg.DATALOADER.NUM_WORKERS =int(args.worker)
    
        self.cfg.INPUT.MIN_SIZE_TRAIN = (self.config.vars['min_dimension'],)
        self.cfg.INPUT.MAX_SIZE_TRAIN = self.config.vars['max_dimension']
        self.cfg.INPUT.MIN_SIZE_TEST = self.config.vars['min_dimension']
        self.cfg.INPUT.MAX_SIZE_TEST = self.config.vars['max_dimension']
    
        self.cfg.SOLVER.MAX_ITER = (self.config.vars["max_iter"])  
        self.cfg.SOLVER.IMS_PER_BATCH = int(args.batch)
        self.cfg.SOLVER.BASE_LR = self.config.vars.get('base_lr')
        self.cfg.SOLVER.WEIGHT_DECAY = self.config.vars.get('weight_decay')
        
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.config.class_name)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.vars['test_score_thresh']
        self.cfg.MODEL.ANGLE_ON = True if('True' in self.config.is_use_angle) else False
        self.cfg.MODEL.RPN.relative_bg = False if(self.config.vars['relative_bg'] == 0) else True
        self.cfg.MODEL.ROI_HEADS.relative_bg = False if(self.config.vars['relative_bg'] == 0) else True
        self.cfg.MODEL.RPN.relative_bg_value =  self.config.vars.get('relative_bg_value')
        self.cfg.MODEL.ROI_HEADS.relative_bg_value = self.config.vars.get('relative_bg_value')
        self.cfg.MODEL.WEIGHTS = os.path.join(self.config.model_path, 'model_final.pth') if(os.path.exists(os.path.join(self.config.model_path, 'model_final.pth'))) else root_model

        self.cfg.TEST.DETECTIONS_PER_IMAGE = self.config.vars.get('max_detections')
        
        self.cfg.MODEL.MASKPOINT_ON = False
        self.cfg.MODEL.ROI_MASKPOINT_HEAD.NUM_KEYPOINTS = 50
        self.cfg.MODEL.BMASK_ON = False
        if(self.cfg.MODEL.BMASK_ON):self.cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = True
        self.cfg.MODEL.MASKIOU_ON = False    
        self.cfg.MODEL.ITERDET_ON = False
    

        if(self.config.vars.get('pre_topk_test') is not None and self.config.vars.get('pre_topk_test') != ""):
            self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = int(self.config.vars.get('pre_topk_test'))
        if(self.config.vars.get('post_topk_test') is not None and self.config.vars.get('post_topk_test') != ""):
            self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = int(self.config.vars.get('post_topk_test'))
        if(self.config.vars.get('pre_topk_train') is not None and self.config.vars.get('pre_topk_train') != ""):
            self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = int(self.config.vars.get('pre_topk_train'))
        if(self.config.vars.get('post_topk_train') is not None and self.config.vars.get('post_topk_train') != ""):
            self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = int(self.config.vars.get('post_topk_train'))
        if(self.config.vars.get('ac_size') is not None and self.config.vars.get('ac_size') != ""):
            self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[int(o) for o in self.config.vars.get('ac_size').split(',')]]
        if(self.config.vars.get('ac_ratios') is not None and self.config.vars.get('ac_ratios') != ""):
            self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[float(o) for o in self.config.vars.get('ac_ratios').split(',')]]
        if(self.config.vars.get('iou_thr') is not None and self.config.vars.get('iou_thr') != ""):
            self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [float(o) for o in self.config.vars.get('iou_thr').split(',')]
        if(self.config.vars.get('iou_label') is not None and self.config.vars.get('iou_label') != ""):
            self.cfg.MODEL.ROI_HEADS.IOU_LABELS = [int(o) for o in self.config.vars.get('iou_label').split(',')]
        if(self.config.vars.get('d_batch') is not None and self.config.vars.get('d_batch') != ""):
            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = int(self.config.vars.get('d_batch'))
        if(self.config.vars.get('positive_fraction') is not None and self.config.vars.get('positive_fraction') != ""):
            self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = float(self.config.vars.get('positive_fraction'))
        if(self.config.vars.get('nms_thr') is not None and self.config.vars.get('nms_thr') != ""):
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = float(self.config.vars.get('nms_thr'))
        if(self.config.vars.get('lr_decrease') is not None and self.config.vars.get('lr_decrease') != ""):
            self.cfg.SOLVER.STEPS = tuple([int(o) for o in self.config.vars.get('lr_decrease').split(',')])
        
        self.cfg.MODEL.DEVICE = 'cuda' if self.gpu_device=='' else 'cuda'+':'+ self.gpu_device
        
        self.display_key_params()
        
    def display_key_params(self):
        print("[Architecture config] :")
        print(f'MASK : {self.cfg.MODEL.MASK_ON}')
        print(f'ANGLE : {self.cfg.MODEL.ANGLE_ON}')
        print(f'batch size : {self.cfg.SOLVER.IMS_PER_BATCH}')
        print(f'base_lr : {self.cfg.SOLVER.BASE_LR}')
        print(f'weight_decay : {self.cfg.SOLVER.WEIGHT_DECAY}')
        print(f'lr_decrease : {self.cfg.SOLVER.STEPS}')
        print(f'pre_topk_test : {self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST}')
        print(f'post_topk_test : {self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST}')
        print(f'pre_topk_train : {self.cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN}')
        print(f'post_topk_train : {self.cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN}')
        print(f'ac_size : {self.cfg.MODEL.ANCHOR_GENERATOR.SIZES}')
        print(f'ac_ratios : {self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS}')
        print(f'iou_thr : {self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS}')
        print(f'iou_label : {self.cfg.MODEL.ROI_HEADS.IOU_LABELS}')
        print(f'd_batch : {self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}')
        print(f'positive_fraction : {self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION}')
        print(f'nms_thr : {self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST}')    
        print(f'gpu_device : {self.cfg.MODEL.DEVICE}')
        pass
                
    def train(self, use_retrain=False):
        DatasetCatalog.clear()
        metadata={}
        register_coco_instances(self.datasets_register_name, metadata, os.path.join(self.config.vars['dataset_folder'], 'trainval.json'), self.config.vars['dataset_folder'])
        self.initail_cfg()    
        self.trainer = DefaultTrainer(self.cfg)
        self.trainer.resume_or_load(resume=True)
        self.trainer.train()
        self.isNeedLoadModel = True


if __name__ == "__main__":

    train_time = time.time()

    root_model = args.pretrain
    model_type = "default"
    #model_type= "pro"
    path = os.path.join(args.project, 'voc_config.json')
    mrcnn_tool = MaskRCNN_PYTORCH(args.mode, path, args.root)
    mrcnn_tool.train(True)    



