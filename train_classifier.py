import argparse

from cv2 import normalize
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
args = parser.parse_args()
from sys import platform
import os
import timm
from timm.models.nfnet import GammaAct
from timm import loss
import torch
import torch.nn.functional as F
from torch import nn
from dataset import os, np, tvt, tud, Pure_Dataset
from torch.utils.tensorboard import SummaryWriter
# from time import time, ctime
import time
import datetime
import yaml
import prettytable as pt
# from utils.general import colorstr
import logging

from model import get_model, Focal_loss

t_localtime = time.localtime()
os.makedirs(r'run', exist_ok=True)
FORMAT = '%(asctime)s %(levelname)s %(process)s: %(message)s'
config= args.cfg

job_name = os.path.basename(config)
job_folder = os.path.join('run', job_name)
os.makedirs(job_folder, exist_ok=True)
os.makedirs(os.path.join(job_folder, 'weight'), exist_ok=True)
log_file_name = os.path.join('run', job_name, 'training.Log') 
logging.basicConfig(level=logging.DEBUG, filename=log_file_name, filemode='w', format=FORMAT)
current_path = os.path.dirname(os.path.abspath(__file__))

def Log(s):
    logging.info(s)
    print(s)

def Initial_info():
    # wandb.login()
    writer = SummaryWriter(job_folder) # TENSORBOARD
    Log('>>> INITIALIZE HYPER-PARAMS:\n')

    hyp = config
    Log(f"hyperparameter file: {hyp} \n")
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict

    tb1 = pt.PrettyTable()
    tb1.field_names = ['Title', 'Value']
    for i in hyp: tb1.add_row([i, hyp[i]])
    Log(tb1)

    return hyp, writer

if __name__ == '__main__':
    '''INITIALIZE HYPER-PARAMS'''
    hyp, writer = Initial_info()

    data_opt = {
        'im_load': hyp['im_load'],
        'resize_done':hyp['resize_done'],
        'im_type':hyp['im_type'],
        'use_Gray':hyp['use_Gray'],
        'rand_anti_white':hyp['rand_anti_white'],
        'use_reverse':hyp['use_reverse']
        }
    val_data_opt = {
        'im_load': hyp['im_load'],
        'resize_done':hyp['resize_done'],
        'im_type':hyp['im_type'],
        'use_Gray':hyp['use_Gray'],
        'rand_anti_white':False,
        'use_reverse':False
        }

    #=========================================================================
    if(hyp['use_Gray']):
        hyp['saturation']=0.0
        hyp['hue']=0.0

    '''Dataset & Dataloader'''
    if(hyp.get('mean_std') == 'imagenet'):
        normalize_fun =  tvt.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        Log('imagenet normalize')
    else:
        normalize_fun =  tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = tvt.Compose([
        # tvt.CenterCrop(224),
        tvt.Pad(hyp['Pad_size'], padding_mode="symmetric"),
        tvt.RandomCrop(hyp['im_size']),
        tvt.ColorJitter(brightness=hyp['brightness'], contrast=hyp['contrast'], saturation=hyp['saturation'], hue=hyp['hue']),
        tvt.ToTensor(),
        normalize_fun,
    ])
    val_transform = tvt.Compose([
        # tvt.CenterCrop(224),
        tvt.ToTensor(),
        normalize_fun,
    ])



    if(hyp['Dataset_type'] == 'Pure_Dataset'): 
        Dataset = Pure_Dataset
    train_dataset = Dataset(train_csv=hyp['train_csv'],
                            img_dir=hyp['Dataset_path'],
                            resize=(hyp['im_size'], hyp['im_size']),
                            transform=transform,
                            data_opt=data_opt)
    val_dataset = Pure_Dataset(train_csv=hyp['val_csv'],
                               img_dir=hyp['Dataset_path'],
                               resize=(hyp['im_size'], hyp['im_size']),
                               transform=val_transform,
                               data_opt=val_data_opt)

    if(hyp['Data_sampler'] == 'WeightedRandomSampler'):
        weighted_random_sampler = train_dataset.weighted_random_sampler
        dataloader = tud.DataLoader(train_dataset, batch_size=hyp['batch_size'], num_workers=hyp['num_workers'], sampler=weighted_random_sampler, pin_memory=True)
    else:
        dataloader = tud.DataLoader(train_dataset, batch_size=hyp['batch_size'], shuffle=True, num_workers=hyp['num_workers'], pin_memory=True)
    val_dataloader = tud.DataLoader(val_dataset, batch_size=hyp['batch_size'], shuffle=False, num_workers=hyp['num_workers'], pin_memory=True)
    nclass = train_dataset.nclasses
    assert nclass == 2
    max_iters = len(dataloader)
    max_iters_val = len(val_dataloader)
    print(f'max_iters : {max_iters}')
    print(f'max_iters_val : {max_iters_val}')
    #=========================================================================

    '''Building model, opt, and loss function.'''
    model = get_model(hyp['model_name'], dropout=hyp['dropout'], nclass = nclass)
    if(hyp['opt']=='Adam'): optimizer = torch.optim.Adam(model.parameters(), lr=hyp['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max_iters*hyp['lr_schedule1'], max_iters*hyp['lr_schedule2'], max_iters*hyp['lr_schedule3']])
    if(hyp['loss']=='CE'): criterion = nn.CrossEntropyLoss()
    elif(hyp['loss']=='Focal_loss'): criterion = Focal_loss(alpha=0.25, gamma=2.0)

    model.cuda()

    ''' TRAINING PROGRESS '''
    s_time, start = time.time(), time.time()
    val_acc_max = 0.0
    step=0
    for epoch in range(hyp['epochs']):
        writer.add_scalar('EPOCH', epoch, step)
        for it, (data, target) in enumerate(dataloader):
            step = it+max_iters*epoch
            # print(target)
            optimizer.zero_grad()
            data, target = data.cuda(), target.cuda()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if it % 10 == 0:
                with torch.no_grad():
                    y_pred = F.softmax(out,1).argmax(1)
                    train_acc = torch.sum(y_pred == target) / data.shape[0]

                    writer.add_scalar('TRAINING_LOSS', loss.item(), step)
                    writer.add_scalar('TRAINING_ACC', train_acc.item(), step)
                    writer.add_scalar('LR', scheduler.get_last_lr()[0], step)

                Log(f"[TRAIN {it}/{max_iters}/{epoch}] Train Loss:{loss.item():.3f}, time(s/10itr): {time.time()-start:.3f}, total time: {str(datetime.timedelta(seconds=time.time()-s_time))}, training acc: {train_acc.item():.3f}")
                start = time.time()
        # model_path = os.path.join(job_folder, 'weight', f'Macc_{epoch}_{it}.pth')
        # logging.info(f"Model save to : {model_path}")
        # torch.save(model.state_dict(), model_path)

        print(f'>>>START VALIDATION... ep:{epoch}')
        model.eval()
        val_correct = 0
        with torch.no_grad():
            val_time = time.time()
            for itv, (data, target) in enumerate(val_dataloader):
                data, target = data.cuda(), target.cuda()
                out = model(data)
                y_pred = F.softmax(out,1).argmax(1)
                val_correct+=torch.sum(y_pred == target).item()
                if(itv % 200==0):
                    print(f'[VALIDATE {itv}/{max_iters_val}] time(s/200itr): {time.time()-val_time:.3f}')
                    val_time = time.time()
            val_time = time.time()
   
            val_acc = (val_correct)/(len(val_dataset))
            print(f'val_acc : {val_acc:.5f}')
            writer.add_scalar('VALIDATION_ACCURACY', val_acc, step)
            
            if val_acc >= val_acc_max:
                val_acc_max = val_acc
                with open(os.path.join(job_folder, 'weight', f'Macc_{epoch}_{it}_{val_acc:.3f}.txt'), 'w') as f:
                    pass
                model_path = os.path.join(job_folder, 'weight', 'model_cls_final.pth')
                logging.info(f"Model save to : {model_path}")
                torch.save(model.state_dict(), model_path)

            del data, target, out, y_pred, val_acc
            model.train()


    
    # final_model_path = os.path.join(job_folder, 'weight', 'model_final.pth')
    # logging.info(f"Final model save to : {final_model_path}")
    # torch.save(model.state_dict(), final_model_path)
    writer.close()
    # wandb.finish()