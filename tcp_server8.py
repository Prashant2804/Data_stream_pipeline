#senders code
import socket
import struct
import zlib
import cv2
import time
import numpy as np

# Server IP and Port
SERVER_IP = "10.147.20.85"  # Replace with your server's IP
SERVER_PORT = 8001

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Initialize video capture
pipeline = gstreamer_pipeline(flip_method=0)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()


try:
    while True:
        # Create a socket connection
        print("Connecting to server...")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_IP, SERVER_PORT))
        client_socket.settimeout(10)  # Set a timeout for operations
        print("Connected to server.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Compress frame
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            compressed_data = zlib.compress(buffer.tobytes())
            frame_size = len(compressed_data)

            # Send frame size and data
            print("Sending frame of size: 456")
            client_socket.sendall(struct.pack("!I", frame_size))
            client_socket.sendall(compressed_data)

            # Receive response
            response_header = client_socket.recv(4)
            if not response_header:
                print("No response header received; server closed connection.")
                break

            response_size = struct.unpack("!I", response_header)[0]
            response_data = b""
            while len(response_data) < response_size:
                chunk = client_socket.recv(4096)
                if not chunk:
                    print("Incomplete response received.")
                    break
                response_data += chunk

            print("Received response of size: 123")

            # Decompress and display the response frame
            decompressed_response = zlib.decompress(response_data)
            np_arr = np.frombuffer(decompressed_response, dtype=np.uint8)
            annotated_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if annotated_frame is not None:
                cv2.imshow("Annotated Frame", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Error: Received frame is None.")
        
        client_socket.close()
        print("Connection closed. Retrying in 5 seconds...")
        time.sleep(5)

except Exception as e:
    print("Error:", e)
finally:
    cap.release()
    cv2.destroyAllWindows()

