# main.py 命令行入口
import argparse
from model import AnimalDetector
from image_predict import single_image_detect
from camera_infer import CameraThread
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["img", "camera"], required=True)
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    detector = AnimalDetector()
    if args.mode == "img":
        img, res = single_image_detect(detector, args.path, conf=args.conf)
        if img is not None:
            print("图片检测完成")
        else:
            print("图片读取失败")
    elif args.mode == "camera":
        cam = CameraThread(detector, conf=args.conf)
        cam.start()
        try:
            while cam.isRunning():
                time.sleep(0.1)
        except KeyboardInterrupt:
            cam.stop()
            print("摄像头停止")

if __name__ == "__main__":
    main()
