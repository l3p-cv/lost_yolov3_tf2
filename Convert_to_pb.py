#================================================================
#
#   File name   : Convert_to_pb.py
#   Author      : PyLessons
#   Created date: 2020-08-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : used to freeze tf model to .pb model
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

foldername = os.path.basename(os.getcwd())
if foldername == "tools":
    os.chdir("..")
sys.path.insert(1, os.getcwd())

from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights
from yolov3.configs import *

if YOLO_TYPE == "yolov4":
    Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
if YOLO_TYPE == "yolov3":
    Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

if YOLO_CUSTOM_WEIGHTS == False:
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
    load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
else:
    checkpoint = f"./checkpoints/{TRAIN_MODEL_NAME}"
    if TRAIN_YOLO_TINY:
        checkpoint += "_Tiny"
    print("Loading custom weights from:", checkpoint)
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    yolo.load_weights(checkpoint)  # use custom weights

yolo.summary()

versions = []
for _, dirs, _ in os.walk(MODEL_PATH):
    for dir in dirs:
        try:
            versions.append(int(dir))
        except:
            continue

versions.sort()

if not versions:
    highest_version = 1
else:
    highest_version = versions[-1] + 1
    
model_path=os.path.join(MODEL_PATH, str(highest_version), 'model.savedmodel')
print(model_path)
yolo.save(model_path)

