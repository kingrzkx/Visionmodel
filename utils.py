# utils.py 工具函数
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from config import OUTPUT_DIR
import os
from datetime import datetime


def cv2qt(cv_img: np.ndarray):
    """opencv BGR图像转Qt显示QPixmap"""
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_img.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def save_detect_image(img: np.ndarray):
    """保存检测图片到output文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(OUTPUT_DIR, f"detect_{timestamp}.jpg")
    cv2.imwrite(save_path, img)
    return save_path


def filter_animal_results(results, animal_ids):
    """过滤YOLO结果，只保留指定动物类别"""
    new_boxes = []
    for box in results[0].boxes:
        cls_id = int(box.cls.cpu().item())
        if cls_id in animal_ids:
            new_boxes.append(box)
    results[0].boxes = new_boxes
    return results
