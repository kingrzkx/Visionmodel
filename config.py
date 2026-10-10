# config.py 全局配置
import os

# 模型配置
MODEL_WEIGHT_PATH = "yolov8s.pt"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# COCO中需要识别的动物类别ID与名称
ANIMAL_CLS = {
    1: "bird",
    14: "cat",
    15: "dog",
    16: "horse",
    17: "sheep",
    18: "cow",
    19: "elephant",
    20: "bear",
    21: "zebra",
    22: "giraffe"
}
ANIMAL_IDS = list(ANIMAL_CLS.keys())

# 摄像头
CAMERA_ID = 0

# 保存输出目录
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
