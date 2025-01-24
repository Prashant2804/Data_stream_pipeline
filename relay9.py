import socket
import struct
import zlib
import cv2
import numpy as np
from ultralytics import YOLO

# Server Configuration
HOST = '0.0.0.0'
PORT = 8000

# Load YOLO models for flame and smoke detection
print("Loading YOLO models...")
model_flame = YOLO("Weights/best (1).pt")
model_smoke = YOLO("Weights/smoke1.pt")
print("YOLO models loaded successfully.")

def detect_and_draw(frame):
    """
    Detect objects and annotate the frame using flame and smoke YOLO models.
    """

    # YOLO flame detection
    results_flame = model_flame(frame)
    for box in results_flame.boxes:  # Iterate through detected boxes
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class ID
        label = f"{model_flame.names[cls]}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # YOLO smoke detection
    results_smoke = model_smoke(frame)
    for box in results_smoke.boxes:  # Iterate through detected boxes
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        cls = int(box.cls[0].item())  # Class ID
        label = f"{model_smoke.names[cls]}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

# Start server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)
print("Server running on {}:{}".format(HOST, PORT))

while True:
    client_socket, addr = server_socket.accept()
    print("Client connected: {}".format(addr))

    try:
        while True:
            # Receive frame size
            header = client_socket.recv(4)
            if not header:
                print("Client disconnected.")
                break

            data_size = struct.unpack("!I", header)[0]

            # Receive compressed frame data
            compressed_data = b""
            while len(compressed_data) < data_size:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                compressed_data += chunk

            # Decompress the frame
            try:
                frame_data = zlib.decompress(compressed_data)
                np_arr = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print("Error during frame decoding:", e)
                break

            if frame is None:
                print("Received empty frame.")
                continue

            # Run dual YOLO inference and annotate the frame
            print("Running flame and smoke detection...")
            annotated_frame = detect_and_draw(frame)

            # Encode and send processed frame back
            _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            compressed_response = zlib.compress(buffer.tobytes())

            client_socket.sendall(struct.pack("!I", len(compressed_response)))
            client_socket.sendall(compressed_response)
            print("Processed frame sent back to client.")

    except Exception as e:
        print("Error:", e)

    finally:
        client_socket.close()
        print("Client connection closed.")
