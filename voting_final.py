import numpy as np
import os
import pandas as pd
import cv2
import copy
import tqdm
import matplotlib.pyplot as plt
from collections import Counter

#%%
root_path = r'E:\Research_code\myProject\Competition\CL_OCR'
base_df = pd.read_csv(os.path.join(root_path, 'submission_private_57_1.csv'))
def check(pred_dfs, check_path):
    pred_dfs = [pd.read_csv(os.path.join(root_path, pred_df)) for pred_df in pred_dfs]
    for i in range(len(pred_dfs)-1): assert pred_dfs[i]['id'].equals(pred_dfs[i+1]['id'])
    singles=[]
    text_voting=[]
    for i in range(len(pred_dfs[0])):
        outputs= []
        for pred_df in pred_dfs:
            output = pred_df.iloc[i]['text']
            outputs.append(output)
        counts = Counter(outputs)
        max_output, times = counts.most_common(1)[0]
        if(times==1):
            singles.append([pred_df.iloc[i]['id']]+outputs)
            print(pred_df.iloc[i]['id'], max_output)
            assert outputs.index(max_output)==0
        text_voting.append(max_output)
        
    final_submission =pd.DataFrame({'id':pred_dfs[0].id,'text':text_voting})
    final_submission.to_csv(os.path.join(root_path, 'submission_private_final.csv'), index=False)
    
    idx = final_submission[final_submission['text']!=base_df['text']].index
    os.makedirs(check_path, exist_ok=True)
    for img_name, final_out, out in zip(final_submission['id'].iloc[idx], final_submission['text'].iloc[idx], base_df['text'].iloc[idx]):
        img = cv2.imread(os.path.join(root_path, 'private_data_v2', img_name+'.jpg'))
        cv2.putText(img, f'Last:{final_out}', (10, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,255), 2, cv2.LINE_AA)
        cv2.putText(img, f'Pres:{out}', (10, 200), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(check_path, img_name+'.jpg'), img)
    
    print(f"{os.path.basename(check_path)} : {len(singles)}")


# pred_dfs = ['submission_private_57_1.csv',
#             'submission_private_57_2.csv',
#             'submission_private_57_3.csv',
#             'submission_private_57_4.csv',
#             'submission_private_57_5.csv',
#             'submission_private_57_6.csv',]
# check(pred_dfs, os.path.join(root_path, 'check1'))

# pred_dfs = ['submission_private_57_1.csv',
#             'submission_private_57_3.csv',
#             'submission_private_57_4.csv',
#             'submission_private_57_6.csv',]
# check(pred_dfs, os.path.join(root_path, 'check7'))
# pred_dfs = ['submission_private_57_3.csv',
#             'submission_private_57_4.csv',
#             'submission_private_57_6.csv',
#             'submission_private_57_1.csv',]
# check(pred_dfs, os.path.join(root_path, 'check8'))
# pred_dfs = ['submission_private_57_4.csv',
#             'submission_private_57_6.csv',
#             'submission_private_57_1.csv',
#             'submission_private_57_3.csv',]
# check(pred_dfs, os.path.join(root_path, 'check9'))
# pred_dfs = ['submission_private_57_6.csv',
#             'submission_private_57_1.csv',
#             'submission_private_57_3.csv',
#             'submission_private_57_4.csv',]
# check(pred_dfs, os.path.join(root_path, 'check10'))

# pred_dfs = ['submission_private_57_6.csv',
#             'submission_private_57_2.csv',
#             'submission_private_57_5.csv',
#             'submission_private_57_4.csv',]
# check(pred_dfs, os.path.join(root_path, 'check11'))

pred_dfs = ['submission_private_57_6.csv',
            'submission_private_57_1.csv',
            'submission_private_57_3.csv',
            'submission_private_57_4.csv',
            'submission_private_57_7.csv',
            'submission_private_57_8.csv',]
check(pred_dfs, os.path.join(root_path, 'check12'))