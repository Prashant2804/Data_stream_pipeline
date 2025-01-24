import socket
import struct
import zlib
import cv2
import numpy as np
from ultralytics import YOLO

# Server IP and Port
SERVER_IP = "0.0.0.0"  # Bind to all interfaces
SERVER_PORT = 8000

# Load YOLO models for flame and smoke detection
print("Loading YOLO models...")
model_flame = YOLO("Weights/best (1).pt")
model_smoke = YOLO("Weights/smoke1.pt")
print("YOLO models loaded successfully.")

# Function to process frame
def detect_and_draw(frame):
    # Detect objects and annotate the frame using YOLO models
    results_flame = model_flame(frame)
    for box in results_flame.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = f"{model_flame.names[cls]}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    results_smoke = model_smoke(frame)
    for box in results_smoke.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = f"{model_smoke.names[cls]}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return annotated_frame

# Set up server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_IP, SERVER_PORT))
server_socket.listen(5)

print(f"Server listening on {SERVER_IP}:{SERVER_PORT}")

try:
    while True:
        client_socket, addr = server_socket.accept()
        print(f"Client connected: {addr}")

        try:
            while True:
                # Receive frame size
                header = client_socket.recv(4)
                if not header:
                    print("No header received; client disconnected.")
                    break

                data_size = struct.unpack("!I", header)[0]
                print(f"Expecting frame of size: {data_size}")

                # Receive frame data
                compressed_data = b""
                while len(compressed_data) < data_size:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        print("Incomplete frame received.")
                        break
                    compressed_data += chunk

                print(f"Received frame of size: {len(compressed_data)}")

                # Decompress and decode frame
                frame_data = zlib.decompress(compressed_data)
                np_arr = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    print("Decoded frame is None.")
                    continue

                # Process the frame
                annotated_frame = detect_and_draw(frame)

                # Compress and send the processed frame
                _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                compressed_response = zlib.compress(buffer.tobytes())
                client_socket.sendall(struct.pack("!I", len(compressed_response)))
                client_socket.sendall(compressed_response)
                print("Processed frame sent back to client.")

        except Exception as e:
            print("Error during frame processing:", e)
        finally:
            client_socket.close()
            print("Client connection closed.")

except Exception as e:
    print("Server error:", e)
finally:
    server_socket.close()
