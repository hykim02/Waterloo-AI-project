# Receives images and runs them through ML algorithm

import socket
import struct

import pickle

# for demonstration
import matplotlib.pyplot as plt

HOST = '0.0.0.0'
PORT = 1234

def process_image(image):
    # you would do your processing here and return some results in whatever format you need
    print("Processing image")
    # plot the image for demonstration
    # plt.imshow(image)
    return "testresult"

def main():
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