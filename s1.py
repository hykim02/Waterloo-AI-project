# Receives images and runs them through ML algorithm

import socket
import struct
import cv2
import pickle
from time import sleep
# for demonstration
from PIL import Image
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# YOLOv5 모델 로드
import sys
sys.path.append('C:/Users/User/Desktop/Waterloo-project/yolov5-master/')

from pathlib import Path

HOST = '0.0.0.0'
PORT = 5555

flag = True
i = 0

# model = torch.load("C:\\Users\\User\\Desktop\\Waterloo-project\\best_v2\\best.pt")
# 모델 경로 설정
model_path = 'C:/Users/User/Desktop/Waterloo-project/best_v2/best.pt'

# 디바이스 설정
device = torch.device('cpu')

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_path, force_reload=True)  # 모델 파일 경로 조정

model.to(device)  # 모델을 디바이스로 이동
model.conf = 0.4
model.eval()

image = cv2.imread('C:\\Users\\User\\Desktop\\Waterloo-project\\2024-08-07-14-35-24.jpg')

if __name__ == '__main__':
    result = model(image)
    result1 = result.pandas().xyxy[0].to_json()
    print(result)
    print(result1)
    