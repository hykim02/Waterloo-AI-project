# struct 모듈의 함수와 객체는 두 가지 크게 다른 애플리케이션, 외부 소스(파일 또는 네트워크 연결)와의 
# 데이터 교환 또는 Python 애플리케이션과 C 계층 간의 데이터 전송에 사용
import struct
import socket

import pickle
import cv2
import matplotlib.pyplot as plt

def send_data(image, socket):
    # Send the image to the server
    print("Sending image")
    # pickle our image
    image = pickle.dumps(image)  # to bytes

    # pack the image and send it with its size
    image_size = struct.pack("L", len(image))
    socket.sendall(image_size + image)

    data = b""
    payload_size = struct.calcsize("L")
    while len(data) < payload_size:
        data += socket.recv(4096)  # 데이터 수신

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    while len(data) < msg_size:
        data += socket.recv(4096)

    result_data = data[:msg_size]
    result = pickle.loads(result_data)

    print("Image sent")
    return result

def main():
    HOST = '127.0.0.1' # This is IP of your server
    PORT = 1234

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

    # Let's load a bunch of images and send them to the server
    for image in ["0000001.jpg", "0000002.jpg", "0000003.jpg", "0000004.jpg", "0000005.jpg"]:
        # load image, it's in RGB format
        image = cv2.imread("images/" + image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # show image for demonstration
        # plt.imshow(image)

        result = send_data(image, s)
        print("Prediction: " + result)

    s.close()

if __name__ == "__main__":
    main()