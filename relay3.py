import cv2

import socket

import numpy as np

from ultralytics import YOLO



# Load YOLO models for flame and smoke detection

model_flame = YOLO("Weights/best (1).pt")  # Use GPU if available

model_smoke = YOLO("Weights/smoke1.pt")



# UDP server configuration

SERVER_IP = '0.0.0.0'

SERVER_PORT = 8000

BUFFER_SIZE = 65535  # UDP buffer size



# Initialize UDP server socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_socket.bind((SERVER_IP, SERVER_PORT))

print(f"Relay server is running on {SERVER_IP}:{SERVER_PORT}")



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



try:

    while True:

        # Receive data from the source

        data, addr = server_socket.recvfrom(BUFFER_SIZE)



        # Log received frame details

        print(f"Received data from {addr}. Buffer size: {len(data)} bytes.")



        # If this is the first frame, store the client's address

        if client_address is None:

            client_address = addr

            print(f"Client connected: {client_address}")



        # Decode the received frame

        try:

            frame = np.frombuffer(data, dtype=np.uint8)

            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)



            if frame is None:

                print("Failed to decode frame. Sending acknowledgment.")

                server_socket.sendto(b"ACK: Frame decoding failed", addr)

                continue



            print("Frame decoded successfully.")



            # Perform YOLO detections and annotate frame

            processed_frame = detect_and_draw(frame)

            print("Frame processed successfully with YOLO models.")



            # Encode the processed frame back to JPEG

            _, encoded_frame = cv2.imencode('.jpg', processed_frame)

            print("Frame encoded successfully.")



            # Send the processed frame to the client

            server_socket.sendto(encoded_frame.tobytes(), client_address)

            print(f"Processed frame sent to {client_address}.")



            # Send acknowledgment to the sender

            ack_message = "ACK: Frame processed and sent"

            server_socket.sendto(ack_message.encode(), addr)

            print(f"Acknowledgment sent to {addr}: {ack_message}")



        except Exception as e:

            error_message = f"Error processing frame: {e}"

            print(error_message)

            server_socket.sendto(f"ACK: {error_message}".encode(), addr)

            continue



except KeyboardInterrupt:

    print("\nServer shutting down...")



finally:

    server_socket.close()

    print("Resources released. Server stopped.")
