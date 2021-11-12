# China-Steel-OCR_Competition
中鋼人工智慧挑戰賽-字元辨識 2nd Solution ([https://tbrain.trendmicro.com.tw/Competitions/Details/17](https://tbrain.trendmicro.com.tw/Competitions/Details/17))

# Requirement
### Text Deteciton - Detectron2 (Object Detection)
> **Already wrapped in this git, the following is officail installation.**
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you may need to prepend the above commands with a few environment variables:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...
```
### Classification - timm (pytorch-image-models)
```
pip install timm
```

### MMOCR - OCR recognition opensource toolbox
> **Already wrapped in this git, the following is requirements for mmocr**
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet
```

### Other packages
```
pip install torch==1.9.0+cu111
pip install opencv-python==4.1.2.30
pip install pandas
pip install numpy
pip install shapely
```
# Run
### Train
+ Train text detection model
```
python train_detection.py  
```
+ Train classifer model
```
python train_classifier.py --cfg configs/cls_L_300.yml &&
python train_classifier.py --cfg configs/cls_L_384.yml &&
python train_classifier.py --cfg configs/cls_M_224.yml &&
python train_classifier.py --cfg configs/cls_M_300.yml &&
python train_classifier.py --cfg configs/cls_M_384.yml &&
python train_classifier.py --cfg configs/cls_S_224.yml &&
python train_classifier.py --cfg configs/cls_S_300.yml &&
python train_classifier.py --cfg configs/cls_Res_256.yml &&
python train_classifier.py --cfg configs/cls_Vit_224.yml
```
+ Train OCR recognizer model
```
python train_recognizer.py --cfg reg_crnn_tps &&
python train_recognizer.py --cfg reg_crnn_tps_aug &&
python train_recognizer.py --cfg reg_crnn_tps_nokeep &&
python train_recognizer.py --cfg reg_nrtr_r31_1by16_1by8 &&
python train_recognizer.py --cfg reg_nrtr_r31_1by16_1by8_aug &&
python train_recognizer.py --cfg reg_nrtr_r31_1by16_1by8_aug_lowlr &&
python train_recognizer.py --cfg reg_nrtr_r31_1by16_1by8_w &&
python train_recognizer.py --cfg reg_robustscanner_r31 &&
python train_recognizer.py --cfg reg_robustscanner_r31_aug &&
python train_recognizer.py --cfg reg_robustscanner_r31_aug_lowlr &&
python train_recognizer.py --cfg reg_robustscanner_r31_w &&
python train_recognizer.py --cfg reg_sar_r31_parallel_decoder &&
python train_recognizer.py --cfg reg_sar_r31_parallel_decoder_36 &&
python train_recognizer.py --cfg reg_sar_r31_parallel_decoder_aug &&
python train_recognizer.py --cfg reg_sar_r31_parallel_decoder_aug_lowlr &&
python train_recognizer.py --cfg reg_sar_r31_parallel_decoder_w &&
python train_recognizer.py --cfg reg_satrn &&
python train_recognizer.py --cfg reg_satrnr_aug &&
python train_recognizer.py --cfg reg_satrn_w
```

### Inference
+ Run text detection model
```
python pri_test_detection.py
```
+ Run classifer model
```
python pri_test_classifier.py
```
+ Run OCR recognizer model
```
 python pri_test_recognizer3.py
```
+ Run ensemble function
```
python pri_recog_voting3.py 
```