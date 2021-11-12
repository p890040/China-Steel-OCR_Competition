import os, sys
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, 'mmocr'))
import mmcv
from mmcv import Config
import matplotlib.pyplot as plt 
from mmocr.apis import init_detector, model_inference
from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.apis import train_detector
import numpy as np
import pandas as pd
import time
import tqdm
# experiment = 'cfg_recognizer_SAR_90_pre'
experiments =[]
for folder in os.listdir(os.path.join(ROOT, 'run')):
    if(folder[:3]=='reg'):
        experiments.append([folder, 'epoch_2'])
        experiments.append([folder, 'epoch_4'])
        # experiments.append([folder, 'epoch_6'])
        # experiments.append([folder, 'latest'])
    # if(folder[:4]=='text'):
    #     experiments.append([folder, 'epoch_4'])

# experiments = [
# ['text_sar_r31_parallel_decoder_academic_36', 'epoch_1'],
# ]

def simple_predict(experiment, ckpt):
    cfg = Config.fromfile(os.path.join(ROOT, 'configs', experiment+'.py'))
    checkpoint = f"{ROOT}/run/{experiment}/{ckpt}.pth"
    # out_file = 'outputs/1036169.jpg'
    model = init_detector(cfg, checkpoint, device="cuda:0")


    path = os.path.join(ROOT, 'private_classifier_final')

    img_lists = [os.path.join(path, file) for file in os.listdir(path)]
    img_lists = np.array(img_lists)
    img_lists = img_lists.reshape(-1, 661).tolist()
    # img_lists = img_lists.reshape(-1, 248).tolist()
    s_time = time.time()
    id, text = [], []
    for imgs_path in tqdm.tqdm(img_lists):
        result = model_inference(model, imgs_path)
        files = [os.path.basename(img_path).replace('.jpg','') for img_path in imgs_path]
        texts = [res['text']for res in result]
        id.extend(files)
        text.extend(texts)
    output = pd.DataFrame({'id':id, 'text':text})
    output.to_csv(f'{ROOT}/submissionFinalPrivate/submission_{experiment}_{ckpt}.csv', index=False)
    print(time.time()-s_time)

for exp in (experiments):
    experiment, ckpt = exp
    print(experiment, ckpt)
    simple_predict(experiment, ckpt)
    

