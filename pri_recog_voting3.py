import numpy as np
import os
ROOT = os.getcwd()
import pandas as pd
import cv2
import copy
from collections import Counter
import tqdm
root_path = os.path.join(ROOT, 'submissionFinalPrivate')

preds =[]
for folder in os.listdir(os.path.join(ROOT, 'run')):
    if(folder[:3]=='reg'):
        preds.append('submission_'+folder+'_'+'epoch_2.csv')
        preds.append('submission_'+folder+'_'+'epoch_4.csv')
        # preds.append('submission_'+folder+'_'+'epoch_6.csv')
        preds.append('submission_'+folder+'_'+'latest.csv')
    # if(folder[:4]=='text'):
    #     preds.append('submission_'+folder+'_'+'epoch_4.csv')
print(preds)
pred_dfs = [pd.read_csv(os.path.join(root_path, file)) for file in preds]
for i in range(len(pred_dfs)):
    pred_dfs[i] = pred_dfs[i].sort_values(by='id').reset_index(drop=True)
for i in range(len(pred_dfs)-1): assert pred_dfs[i]['id'].equals(pred_dfs[i+1]['id'])


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

new_df.to_csv(os.path.join(root_path, f'submission_private_{len(pred_dfs)}_.csv'), index=False)

