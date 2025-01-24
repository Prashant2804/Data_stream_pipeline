import cv2
import socket
import numpy as np
from ultralytics import YOLO

# Load YOLO models for flame and smoke detection
model_flame = YOLO("Weights/best (1).pt")  # Use GPU if available
model_smoke = YOLO("Weights/smoke1.pt")

# UDP server configuration
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind(('0.0.0.0', 8000))  # Listen on all interfaces at port 8000

client_address = None  # Store the client address to send processed frames back

def detect_and_draw(frame):
    """
    Detect objects and annotate the frame using flame and smoke YOLO models.
    """
    # YOLO flame detection
    results_flame = model_flame(frame)
    for result in results_flame:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model_flame.names[cls]}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # YOLO smoke detection
    results_smoke = model_smoke(frame)
    for result in results_smoke:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model_smoke.names[cls]}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame

print("Relay server is running...")

try:
    while True:
        # Receive data from the source
        data, addr = server_socket.recvfrom(65536)  # Maximum UDP packet size
        if client_address is None:
            client_address = addr  # Store the client's address
        
        # Decode the received frame
        frame = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        
        # Perform YOLO detections and annotate frame
        processed_frame = detect_and_draw(frame)
        
        # Encode the processed frame back to JPEG
        _, encoded_frame = cv2.imencode('.jpg', processed_frame)
        
        # Send the processed frame to the client
        server_socket.sendto(encoded_frame.tobytes(), client_address)

except KeyboardInterrupt:
    print("\nServer shutting down...")
finally:
    server_socket.close()
