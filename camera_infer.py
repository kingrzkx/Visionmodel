# camera_infer.py 摄像头子线程 QThread
import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from model import AnimalDetector
from config import CAMERA_ID


class CameraThread(QThread):
    frame_signal = pyqtSignal(object)
    log_signal = pyqtSignal(str)

    def __init__(self, detector: AnimalDetector, conf):
        super().__init__()
        self.detector = detector
        self.conf = conf
        self._run_flag = False

    def run(self):
        self._run_flag = True
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            self.log_signal.emit("摄像头打开失败，请检查设备")
            return
        self.log_signal.emit("摄像头启动成功")
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                self.log_signal.emit("读取画面失败")
                break
            results = self.detector.predict(frame, conf=self.conf)
            draw_frame = self.detector.draw_result(frame, results)
            self.frame_signal.emit(draw_frame)
        cap.release()
        self.log_signal.emit("摄像头已释放")

    def stop(self):
        self._run_flag = False
        self.wait()
