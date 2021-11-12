import numpy as np
import os
import pandas as pd
import cv2
import copy
from collections import Counter
import tqdm
root_path = '/home/solomon/public/Pawn/Others/othercom/CL_OCR/submissionFinalPublic' 

preds =[]
for folder in os.listdir('/home/solomon/public/Pawn/Others/othercom/CL_OCR/run'):
    if(folder[:3]=='reg'):
        preds.append('submission_'+folder+'_'+'epoch_2.csv')
        preds.append('submission_'+folder+'_'+'epoch_4.csv')
        # preds.append('submission_'+folder+'_'+'epoch_6.csv')
        preds.append('submission_'+folder+'_'+'latest.csv')
    if(folder[:4]=='text'):
        preds.append('submission_'+folder+'_'+'epoch_4.csv')
print(preds)
pred_dfs = [pd.read_csv(os.path.join(root_path, file)) for file in preds]
for i in range(len(pred_dfs)-1): assert pred_dfs[i]['id'].equals(pred_dfs[i+1]['id'])

# #%%
# root_path_ = '/home/solomon/public/Pawn/Others/othercom/CL_OCR/submissionF/' 
# preds_ =[]
# for folder in os.listdir('/home/solomon/public/Pawn/Others/othercom/CL_OCR/run'):
#     if folder =='text_satrn_academic_aug':continue
#     if(folder[:4]=='text'):
#         preds_.append('submission_'+folder+'_'+'epoch_2.csv')
#         preds_.append('submission_'+folder+'_'+'epoch_4.csv')
#         preds_.append('submission_'+folder+'_'+'epoch_6.csv')
#         # preds.append('submission_'+folder+'_'+'latest.csv')
# print(preds_)
# pred_dfs_ = [pd.read_csv(os.path.join(root_path_, file)) for file in preds_]
# for i in range(len(pred_dfs_)-1): assert pred_dfs_[i]['id'].equals(pred_dfs_[i+1]['id'])
# pred_dfs = pred_dfs + pred_dfs_
# #%%

text_voting=[]
for i in tqdm.tqdm(range(len(pred_dfs[0]))):
    outputs= []
    for pred_df in pred_dfs:
        output = pred_df.iloc[i]['text']
        outputs.append(output)
    counts = Counter(outputs)
    max_output, times = counts.most_common(1)[0]
    if(times==1):
        print(max_output)
        assert outputs.index(max_output)==0
    text_voting.append(max_output)

new_df = copy.deepcopy(pred_dfs[0])
new_df['text'] = text_voting

new_df.to_csv(os.path.join(root_path, f'submission_public_{len(pred_dfs)}.csv'), index=False)

