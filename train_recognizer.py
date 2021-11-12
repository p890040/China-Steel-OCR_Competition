#%%
import os, sys
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, 'mmocr'))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
args = parser.parse_args()
import mmcv
from mmcv import Config
experiment = args.cfg
cfg = Config.fromfile(os.path.join(ROOT, 'configs', experiment+'.py'))
from mmdet.apis import set_random_seed

# Set up working dir to save files and logs.
cfg.work_dir = os.path.join(ROOT, 'run', experiment)

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
# if(experiment == 'text_crnn_academic_dataset' or experiment=='text_crnn_tps_academic_dataset'):
    # pass
# else:
    # cfg.optimizer.lr = 0.001 / 8
    # cfg.lr_config.warmup = None
# Choose to log training results every 40 images to reduce the size of log file. 
cfg.log_config.interval = 40

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')
#%%
from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.apis import train_detector
import os.path as osp

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)

# %%
# import matplotlib.pyplot as plt 
# from mmocr.apis import init_detector, model_inference

# img = '/home/solomon/public/Pawn/Others/othercom/CL_OCR/mmocr/tests/data/ocr_toy_dataset/imgs/1036169.jpg'
# checkpoint = "/home/solomon/public/Pawn/Others/othercom/CL_OCR/mmocr/demo/tutorial_exps/epoch_5.pth"
# out_file = 'outputs/1036169.jpg'

# model = init_detector(cfg, checkpoint, device="cuda:0")
# if model.cfg.data.test['type'] == 'ConcatDataset':
#     model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][0].pipeline


# result = model_inference(model, img)
# print(f'result: {result}')

# img = model.show_result(
#         img, result, out_file=out_file, show=False)

# mmcv.imwrite(img, out_file)

# # Visualize the results
# predicted_img = mmcv.imread('/home/solomon/public/Pawn/Others/othercom/CL_OCR/mmocr/tests/data/ocr_toy_dataset/imgs/1036169.jpg')
# plt.figure(figsize=(4, 4))
# plt.imshow(mmcv.bgr2rgb(predicted_img))
# plt.show()
# # %%
