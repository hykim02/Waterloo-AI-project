# Receives images and runs them through ML algorithm

import socket
import struct
import pickle
import torch
import torchvision.models as models

# for demonstration
import matplotlib.pyplot as plt

HOST = '0.0.0.0'
PORT = 1234

def load_model():
    # load ai model

    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval() // 예측 모드
    pass

# img 모델에 맞게 전처리
def process_image(image):
    model = load_model()
    # you would do your processing here and return some results in whatever format you need
    print("Processing image")
    # plot the image for demonstration
    plt.imshow(image)

     # 예: result = model.predict(image)
    result = model.predict(image)  # 이 부분을 실제 모델 예측 결과로 교체
    
    # 모델의 예측값은 3가지 -> 경우의 수에 따라 예측값과 정수값을 어떻게 매핑할지 ?
    # 결과값을 정수로 변환
    result_map = {
        "stop": 0,
        "turn-right": 1,
        "turn-left": 2,
        "straight": 3,
        "stop & turn-right": 4,
        "stop & turn-left": 5,
        "stop & straight": 6
    }

    int_result = result_map.get(result, 3)  # 결과값이 맵에 없으면 3으로 설정
    return int_result
    

def main():
    # load model
    model = load_model()
    
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)

    print("Server started")
    while True:
        client_socket, addr = s.accept()
        print("Connected by", addr)

        try: 
            while True:            
                data = b""
                payload_size = struct.calcsize("L")
                while len(data) < payload_size:
                    data += client_socket.recv(4096)
                    if not data:
                        raise ConnectionResetError("Client disconnected")

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]

                while len(data) < msg_size:
                    data += client_socket.recv(4096)

                frame_data = data[:msg_size]
                
                # handle conversion of image data
                frame_data = pickle.loads(frame_data)
                
                # process the image
                predictions = process_image(frame_data)


                # stop: 0, turn-right: 1, turn-left: 2, straight: 3, 
                # stop & turn-right: 4, stop & turn-left: 5, stop & straight: 6
                # send the results back
                predictions = pickle.dumps(predictions)
                client_socket.sendall(struct.pack("L", len(predictions)) + predictions)
                print("Returned")
                
        except Exception as e:
            print(e)
        finally:
            client_socket.close()

if __name__ == "__main__":
    main()