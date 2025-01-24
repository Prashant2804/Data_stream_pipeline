import cv2
import socket
import numpy as np
from ultralytics import YOLO
import zlib  # For compression
from cryptography.fernet import Fernet  # For encryption
import struct  # For packing metadata

# Load YOLO models for flame and smoke detection
model_flame = YOLO("Weights/best (1).pt")
model_smoke = YOLO("Weights/smoke1.pt")

# TCP server configuration
SERVER_IP = '0.0.0.0'
SERVER_PORT = 8000
BUFFER_SIZE = 65535
KEY = Fernet.generate_key()  # Generate encryption key
cipher = Fernet(KEY)

# Initialize TCP server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((SERVER_IP, SERVER_PORT))
server_socket.listen(1)
print(f"Relay server is running on {SERVER_IP}:{SERVER_PORT}")

client_socket = None
client_address = None


def detect_and_draw(frame):
    """Detect objects and annotate the frame."""
    # YOLO flame detection
    results_flame = model_flame(frame)
    for result in results_flame:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            label = f"Flame: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # YOLO smoke detection
    results_smoke = model_smoke(frame)
    for result in results_smoke:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            label = f"Smoke: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame


def receive_complete_data(sock):
    """Receive the complete encrypted and compressed frame data."""
    data = b""
    try:
        # Receive chunk size first
        chunk_size_data = sock.recv(4)  # 4 bytes for struct.pack('I', chunk_size)
        if not chunk_size_data:
            return None
        chunk_size = struct.unpack('I', chunk_size_data)[0]

        # Receive chunk
        while len(data) < chunk_size:
            packet = sock.recv(min(chunk_size - len(data), BUFFER_SIZE))
            if not packet:
                break
            data += packet
    except Exception as e:
        print(f"Error receiving data: {e}")
        return None
    return data


try:
    print("Waiting for a client to connect...")
    client_socket, client_address = server_socket.accept()
    print(f"Client connected: {client_address}")

    while True:
        print("Waiting to receive a new frame...")
        encrypted_data = receive_complete_data(client_socket)
        if not encrypted_data:
            print("No data received. Closing connection.")
            break

        try:
            # Decrypt the data
            compressed_data = cipher.decrypt(encrypted_data)
            print(f"Data decrypted. Size: {len(compressed_data)} bytes.")

            # Decompress the data
            frame_data = zlib.decompress(compressed_data)
            print(f"Data decompressed. Size: {len(frame_data)} bytes.")

            # Decode the frame
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            if frame is None:
                print("Failed to decode frame.")
                client_socket.sendall(b"ACK: Frame decoding failed")
                continue

            print("Frame decoded successfully.")

            # Perform YOLO detections and annotate frame
            processed_frame = detect_and_draw(frame)

            # Display the processed frame (optional)
            cv2.imshow("Processed Frame", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Send acknowledgment to sender
            ack_message = "ACK: Frame received and processed"
            client_socket.sendall(ack_message.encode())
            print("Acknowledgment sent.")

        except Exception as e:
            print(f"Error during frame processing: {e}")
            client_socket.sendall(f"ACK: Error processing frame: {e}".encode())

except KeyboardInterrupt:
    print("\nServer shutting down...")

finally:
    if client_socket:
        client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()
    print("Resources released. Server stopped.")
