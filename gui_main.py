# gui_main.py PyQt5图形界面【修复导入顺序版本】
# 关键：先加载torch，再加载PyQt，解决WinError1114 c10.dll冲突
import sys
# ========== 强制先导入torch，再导入PyQt ==========
import torch
from model import AnimalDetector
from camera_infer import CameraThread
from image_predict import single_image_detect, batch_folder_detect
from utils import cv2qt, save_detect_image

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QFileDialog, QTextEdit, QGroupBox)
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("动物检测系统 YOLOv8")
        self.resize(1100,750)
        self.detector = AnimalDetector()
        self.camera_thread = None
        self.current_img = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        left_panel = QVBoxLayout()
        # 模式组
        mode_group = QGroupBox("功能模式")
        mode_layout = QVBoxLayout(mode_group)
        self.btn_single_img = QPushButton("单张图片检测")
        self.btn_batch = QPushButton("文件夹批量检测")
        self.btn_camera = QPushButton("开启摄像头")
        self.btn_stop_camera = QPushButton("关闭摄像头")
        self.btn_save = QPushButton("保存当前结果图")
        mode_layout.addWidget(self.btn_single_img)
        mode_layout.addWidget(self.btn_batch)
        mode_layout.addWidget(self.btn_camera)
        mode_layout.addWidget(self.btn_stop_camera)
        mode_layout.addWidget(self.btn_save)

        # 置信度滑块
        conf_group = QGroupBox("置信度阈值")
        conf_layout = QVBoxLayout(conf_group)
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(0,100)
        self.slider_conf.setValue(50)
        self.label_conf = QLabel("当前:0.50")
        conf_layout.addWidget(self.slider_conf)
        conf_layout.addWidget(self.label_conf)

        left_panel.addWidget(mode_group)
        left_panel.addWidget(conf_group)
        left_panel.addStretch()

        # 右侧：图片显示 + 日志
        right_layout = QVBoxLayout()
        self.label_img = QLabel("图像显示区域")
        self.label_img.setFixedSize(800,550)
        self.label_img.setStyleSheet("border:1px solid #999;")
        self.log_text = QTextEdit()
        self.log_text.setFixedHeight(150)
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.label_img)
        right_layout.addWidget(self.log_text)

        main_layout.addLayout(left_panel, stretch=1)
        main_layout.addLayout(right_layout, stretch=4)

        # 绑定事件
        self.btn_single_img.clicked.connect(self.on_single_image)
        self.btn_batch.clicked.connect(self.on_batch)
        self.btn_camera.clicked.connect(self.on_start_camera)
        self.btn_stop_camera.clicked.connect(self.on_stop_camera)
        self.btn_save.clicked.connect(self.on_save_img)
        self.slider_conf.valueChanged.connect(self.on_conf_change)

    def get_conf(self):
        return self.slider_conf.value() / 100.0

    def on_conf_change(self):
        val = self.get_conf()
        self.label_conf.setText(f"当前:{val:.2f}")

    def log(self, msg):
        self.log_text.append(msg)

    def on_single_image(self):
        file_path,_ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png *.jpeg *.bmp)")
        if not file_path:
            return
        img, res = single_image_detect(self.detector, file_path, conf=self.get_conf())
        if img is None:
            self.log(f"错误：{res}")
            return
        self.current_img = img
        pix = cv2qt(img)
        self.label_img.setPixmap(pix.scaled(self.label_img.size(), Qt.KeepAspectRatio))
        self.log(f"单图检测完成：{file_path}")

    def on_batch(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder:
            return
        log_str = batch_folder_detect(self.detector, folder, conf=self.get_conf())
        self.log(log_str)

    def on_start_camera(self):
        if self.camera_thread is not None and self.camera_thread.isRunning():
            return
        self.camera_thread = CameraThread(self.detector, conf=self.get_conf())
        self.camera_thread.frame_signal.connect(self.update_camera_frame)
        self.camera_thread.log_signal.connect(self.log)
        self.camera_thread.start()

    def on_stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread = None

    def update_camera_frame(self, frame):
        self.current_img = frame
        pix = cv2qt(frame)
        self.label_img.setPixmap(pix.scaled(self.label_img.size(), Qt.KeepAspectRatio))

    def on_save_img(self):
        if self.current_img is None:
            self.log("没有图片可以保存")
            return
        save_path = save_detect_image(self.current_img)
        self.log(f"图片已保存至: {save_path}")

    def closeEvent(self, event):
        # 关闭窗口时停止摄像头
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
