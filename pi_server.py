# Receives images and runs them through ML algorithm

import socket
import struct
import cv2
import pickle
from time import sleep
# for demonstration
import matplotlib.pyplot as plt
import torch

HOST = '0.0.0.0'
PORT = 5555

flag = True
i = 0

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


# 모델 경로 설정
model_path = 'C:/Users/User/Desktop/Waterloo-project/best_v2/best.pt'

# 디바이스 설정
device = torch.device('cpu')

# 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_path, force_reload=True)  # 모델 파일 경로 조정

model.to(device)  # 모델을 디바이스로 이동
model.conf = 0.4
model.eval()

image_real = None


def show_image(frame):
    global i
    global image_real
    path  = "C:/Users/User/Desktop/waterloo-project/img/" + str(i) + ".jpg"

    with open(path, 'wb') as f:
        f.write(frame)
    print('save')

    image_real = cv2.imread(path)

    i += 1
    i = i % 50
    print(f'i = {i}')

def send_data(socket, data):

    if len(data) == 0:
        pass

    elif len(data) == 1:
        if data[0] == 'turn_right':
            socket.send('1'.encode())
        elif data[0] == 'turn_left':
            socket.send('2'.encode())
        elif data[0] == 'stop':
            socket.send('3'.encode())
    
    else:
        for _ in data:
            if _ == 'turn_right':
                socket.send('4'.encode())
            elif _ == 'turn_left':
                socket.send('5'.encode())
            else:
                continue

    print(data)

    

def main():
    global flag
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)

    print("Server started")
    while True:
        client_socket, addr = s.accept()
        print("Connected by", addr)

        try: 
            while True:
                result = ''
                print('server 1')
                frame_size_bytes = client_socket.recv(4)
                frame_size = int.from_bytes(frame_size_bytes, byteorder='big')

                print(frame_size)

                print('server 2')
                if frame_size == 0:
                    continue
                
                data = b''
                while len(data) < frame_size:
                    recv = client_socket.recv(4096)
                    if not recv:
                        break
                    data += recv

                print('server 3')

                frame = pickle.loads(data)

                print(frame)

                print('server 4')

                show_image(frame)

                result = model(image_real)

                labels = result.names  # 모델의 클래스 라벨 리스트
                print(labels)
                detected_labels = result.pred[0][:, -1].cpu().numpy().astype(int)  # 예측된 객체의 인덱스 (라벨 번호)
                detected_labels = [labels[i] for i in detected_labels]  # 라벨 번호를 실제 라벨 이름으로 변환


                print('server 5')
                send_data(client_socket, detected_labels)

                frame_size = 0

        except Exception as e:
            print(e)
        finally:
            client_socket.close()

if __name__ == "__main__":
    main()