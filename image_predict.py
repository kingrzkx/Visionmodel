# image_predict.py 图片推理模块：单图 / 批量文件夹
import cv2
import os
from model import AnimalDetector
from utils import save_detect_image


def single_image_detect(detector: AnimalDetector, img_path: str, conf=None):
    img = cv2.imread(img_path)
    if img is None:
        return None, "图片读取失败"
    results = detector.predict(img, conf=conf)
    draw_img = detector.draw_result(img, results)
    draw_img = draw_img.copy() # 额外拷贝
    return draw_img, results


def batch_folder_detect(detector: AnimalDetector, folder_path: str, conf=None):
    """批量检测文件夹内图片，保存结果"""
    supported_suffix = (".jpg", ".jpeg", ".png", ".bmp")
    log_list = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_suffix):
            full_path = os.path.join(folder_path, filename)
            img, res = single_image_detect(detector, full_path, conf=conf)
            if img is not None:
                save_path = save_detect_image(img)
                log_list.append(f"{filename} -> saved: {save_path}")
            else:
                log_list.append(f"{filename} -> 读取失败")
    return "\n".join(log_list)
