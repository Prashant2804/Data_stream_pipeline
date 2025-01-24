import socket
import struct
import zlib
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import queue
import time

class DetectionServer:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        
        # Load YOLO models
        print("Loading YOLO models...")
        self.model_flame = YOLO("Weights/best (1).pt")
        self.model_smoke = YOLO("Weights/smoke1.pt")
        print("YOLO models loaded successfully.")

    def detect_and_draw(self, frame):
        """Process frame with YOLO models and draw detections."""
        # Create a copy of the frame for annotations
        annotated_frame = frame.copy()
        
        # Detect flames
        results_flame = self.model_flame(frame)
        for r in results_flame:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{self.model_flame.names[cls]}: {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Detect smoke
        results_smoke = self.model_smoke(frame)
        for r in results_smoke:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{self.model_smoke.names[cls]}: {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return annotated_frame

    def handle_client(self, client_socket, addr):
        """Handle individual client connection."""
        print(f"Handling client: {addr}")
        try:
            while self.running:
                # Receive frame size
                header = client_socket.recv(4)
                if not header:
                    print(f"Client {addr} disconnected.")
                    break
                
                data_size = struct.unpack("!I", header)[0]
                print(f"Expecting frame of size: {data_size} from {addr}")

                # Receive frame data with timeout
                client_socket.settimeout(5.0)  # 5 second timeout
                compressed_data = b""
                remaining = data_size
                
                while remaining > 0:
                    chunk = client_socket.recv(min(remaining, 8192))
                    if not chunk:
                        raise ConnectionError("Connection broken while receiving frame")
                    compressed_data += chunk
                    remaining -= len(chunk)

                # Process frame
                frame_data = zlib.decompress(compressed_data)
                np_arr = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    raise ValueError("Failed to decode frame")

                # Process frame and send back
                annotated_frame = self.detect_and_draw(frame)
                _, buffer = cv2.imencode('.jpg', annotated_frame, 
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                compressed_response = zlib.compress(buffer.tobytes())
                
                # Send response size first
                response_size = len(compressed_response)
                client_socket.sendall(struct.pack("!I", response_size))
                
                # Send response data
                client_socket.sendall(compressed_response)
                print(f"Processed frame sent to {addr}")

        except socket.timeout:
            print(f"Timeout while communicating with {addr}")
        except ConnectionError as e:
            print(f"Connection error with {addr}: {e}")
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()
            print(f"Closed connection with {addr}")

    def start(self):
        """Start the server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True
        
        print(f"Server listening on {self.host}:{self.port}")
        
        try:
            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, addr)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except Exception as e:
                    print(f"Error accepting connection: {e}")
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            self.stop()

    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

if __name__ == "__main__":
    server = DetectionServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        server.stop()
