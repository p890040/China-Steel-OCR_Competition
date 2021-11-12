import numpy as np
import os
import pandas as pd
import cv2
import copy
import tqdm
import matplotlib.pyplot as plt

vocab = []
for i in range(ord('0'),ord('0')+10):vocab.append(chr(i))
for i in range(ord('A'),ord('A')+26):vocab.append(chr(i))
vocab.append('@')


root_path = r'E:\Research_code\myProject\Competition\CL_OCR' 
label_files = [file for file in os.listdir(os.path.join(root_path, 'public_gt')) if file.endswith(".txt")]
labels = []
normals = []
for label_file in tqdm.tqdm(label_files):
    with open(os.path.join(root_path, 'public_gt', label_file)) as f:
        label = f.readlines()[0].strip()
        assert len(label)>4
        for c in label: assert c in vocab
        
        if('@' in label): is_normal=0
        else: is_normal=1
        label = label.split('@')[0]
        labels.append(label)
        normals.append(is_normal)
public_test_df = pd.DataFrame({'id':label_files,
                                'text':labels,
                                'normal':normals})
# pd.DataFrame({'id':label_files, 'text':labels,}).to_csv(os.path.join(root_path, 'submission_m.csv'), index=False)
public_test_df['id'] = public_test_df['id'].str.replace('.txt','', regex=False)
# best_predict = pd.read_csv(os.path.join(root_path, 'submission_mix_42.csv'))
# class_predict = pd.read_csv(os.path.join(root_path, 'predict_classifier.csv'))
# class_predict['filename'] = class_predict['filename'].str.replace('.jpg','', regex=False)

public_test_df = public_test_df.sort_values(by='id').reset_index(drop=True)
# best_predict = best_predict.sort_values(by='id').reset_index(drop=True)
# class_predict = class_predict.sort_values(by='filename').reset_index(drop=True)
# assert public_test_df['id'].equals(best_predict['id'])
# assert public_test_df['id'].equals(class_predict['filename'])
# best_predict['normal'] = class_predict['normal']

# #%% 

# diff_nomal = public_test_df[public_test_df['normal']!=best_predict['normal']]
# idx_text = public_test_df[public_test_df['text']!=best_predict['text']].index
# diff_text = pd.concat([public_test_df['id'].iloc[idx_text], 
#                         public_test_df['text'].iloc[idx_text], 
#                         best_predict['text'].iloc[idx_text]], axis=1)
# diff_text.to_csv(os.path.join(root_path, 'diff.csv'), index=False)
#%%

# def fix_label(file, new_label):
#     assert os.path.exists(file)
#     with open(file, 'w') as f:
#         f.write(new_label)

# diff_text = pd.read_csv(os.path.join(root_path, 'diff.csv'))
# for i in range(len(diff_text)):
#     file, label, predict = diff_text.iloc[i]
#     img_name = file+'.jpg'
#     img = cv2.imread(os.path.join(root_path, 'public_testing_data', img_name))
#     cv2.putText(img, f'LABEL:{label}', (10, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,255), 2, cv2.LINE_AA)
#     cv2.putText(img, f'PREDS:{predict}', (10, 200), cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 2, cv2.LINE_AA)
#     plt.figure(dpi=300)
#     plt.axis('off')
#     plt.imshow(img)
#     plt.show()
#     _INPUT = str(input())
#     if(_INPUT=='R'):
#         img = cv2.imread(os.path.join(root_path, 'public_testing_data', img_name))
#         h, w = img.shape[:2]
#         center = (w / 2, h / 2)
#         scale = 1.0
#         M = cv2.getRotationMatrix2D(center, 180, scale)
#         img_rot = cv2.warpAffine(img, M, (w, h))
#         cv2.putText(img_rot, f'LABEL:{label}', (10, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,255), 2, cv2.LINE_AA)
#         cv2.putText(img_rot, f'PREDS:{predict}', (10, 200), cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 2, cv2.LINE_AA)
#         plt.figure(dpi=300)
#         plt.axis('off')
#         plt.imshow(img_rot)
#         plt.show()
#         _INPUT = str(input())+'@R'
    
#     if(len(_INPUT)>4):
#         file_name = os.path.join(root_path, 'public_gt', file+'.txt')
#         print(f'file_name:{os.path.basename(file_name)}')
#         with open(file_name) as f:
#             old_label = f.readlines()[0]
#         print(f'old_label:{old_label}')
#         fix_label(file_name, _INPUT)
#         with open(file_name) as f:
#             new_label = f.readlines()[0]
#         print(f'new_label:{new_label}')
#%%
# diff_text = pd.read_csv(os.path.join(root_path, 'diff.csv'))
# for i in range(len(diff_text)):
#     file, label, predict = diff_text.iloc[i]
#     img_name = file+'.jpg'
#     img = cv2.imread(os.path.join(root_path, 'public_testing_data', img_name))
#     img_crop = cv2.imread(os.path.join(root_path, 'predict_classify', img_name))
#     cv2.putText(img, f'LABEL:{label}', (10, 150), cv2.FONT_HERSHEY_PLAIN, 4, (255,0,255), 2, cv2.LINE_AA)
#     cv2.putText(img, f'PREDS:{predict}', (10, 200), cv2.FONT_HERSHEY_PLAIN, 4, (0,255,0), 2, cv2.LINE_AA)

#     f, axarr = plt.subplots(1,2, dpi=300)
#     axarr[0].imshow(img)
#     axarr[1].imshow(img_crop)
#     axarr[0].axis('off')
#     axarr[1].axis('off')
#     plt.show()
#     input()
#%%
test_df = pd.read_csv(os.path.join(root_path, 'submission_m.csv'))
test_df = test_df.sort_values(by='id').reset_index(drop=True)
test_detection = pd.read_csv(os.path.join(root_path, 'predict_detection.csv'))
test_detection['filename'] = test_detection['filename'].str.replace('.jpg','', regex=False)
test_detection = test_detection.sort_values(by='filename').reset_index(drop=True)
assert test_df['id'].equals(test_detection['filename'])
assert test_df['id'].equals(public_test_df['id'])
assert test_df['text'].equals(public_test_df['text'])


train_df = pd.read_csv(os.path.join(root_path, 'Training Label', 'public_total_data_finalv3.csv'))
test_df = pd.DataFrame({'filename':public_test_df['id'],
                        'label':public_test_df['text'],
                        'x1':test_detection['x1'],
                        'y1':test_detection['y1'],
                        'x2':test_detection['x2'],
                        'y2':test_detection['y2'],
                        'normal':public_test_df['normal'],   
    })



filenames, labels, p1, p2, p3, p4, p5, p6, p7, p8, normals=[],[],[],[],[],[],[],[],[],[],[]
for data in test_df.values:
    filenames.append(data[0])
    labels.append(data[1])
    box = data[2:6].astype(np.int32)
    assert box[0]<=box[2] and box[1]<=box[3]
    seg = [box[2], box[1],
           box[2], box[3],
           box[0], box[3],
           box[0], box[1],]
    p1.append(seg[0])
    p2.append(seg[1])
    p3.append(seg[2])
    p4.append(seg[3])
    p5.append(seg[4])
    p6.append(seg[5])
    p7.append(seg[6])
    p8.append(seg[7])
    normals.append(data[6]==1)
test_df_final = pd.DataFrame({'filename':filenames,
                              'label':labels,
                              'top right x':p1,
                              'top right y':p2, 
                              'bottom right x':p3,
                              'bottom right y':p4, 
                              'bottom left x':p5, 
                              'bottom left y':p6, 
                              'top left x':p7,
                              'top left y':p8, 
                              'normal':normals
                              })
test_df_final.to_csv(os.path.join(root_path, 'Training Label', 'public_testing_data.csv'), index=False)

final_data_mix = pd.concat([train_df, test_df_final], ignore_index=True)

final_data_mix.to_csv(os.path.join(root_path, 'Training Label', 'final_data_mix.csv'), index=False)


