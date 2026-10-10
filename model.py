# model.py YOLO模型封装
import os
os.environ['CURL_CA_BUNDLE'] = ''
from ultralytics import YOLO
import torch
import cv2
from config import MODEL_WEIGHT_PATH, CONF_THRESHOLD, IOU_THRESHOLD, ANIMAL_IDS, ANIMAL_CLS
from utils import filter_animal_results


class AnimalDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        self.model = YOLO(MODEL_WEIGHT_PATH).to(self.device)
        self.conf = CONF_THRESHOLD
        self.iou = IOU_THRESHOLD

    def predict(self, image, conf=None, iou=None):
        """
        image: cv2 BGR图像
        return: YOLO原始result，只保留动物类别
        """
        if conf is None:
            conf = self.conf
        if iou is None:
            iou = self.iou
        results = self.model(image, conf=conf, iou=iou)
        results = filter_animal_results(results, ANIMAL_IDS)
        return results

    def draw_result(self, img, results):
        """手动OpenCV绘制框，抛弃results.plot()，避免内存崩溃"""
        draw_img = img.copy()
        for box in results[0].boxes:
            # 坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls.cpu().item())
            conf = float(box.conf.cpu().item())
            cls_name = ANIMAL_CLS[cls_id]
            # 画框
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 文字
            text = f"{cls_name} {conf:.2f}"
            cv2.putText(draw_img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # 强制clone，独立内存，解决Qt内存越界崩溃
        draw_img = draw_img.copy()
        return draw_img
