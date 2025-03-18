import grpc
import cv2
import time
import numpy as np
import sys
sys.path.append("generated")
from generated import streaming_pb2, streaming_pb2_grpc

# Kết nối đến server
channel = grpc.insecure_channel("localhost:5000")
stub = streaming_pb2_grpc.VideoStreamingServiceStub(channel)

def generate_frames(cap):
    """Generator gửi frame liên tục trong một stream."""
    frame_id = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        _, img_encoded = cv2.imencode(".jpg", frame)
        frame_data = streaming_pb2.Frame(frame_id=frame_id, image_data=img_encoded.tobytes())
        
        print(f"Client: Sending Frame ID: {frame_id}")  # Ghi log frame_id
        yield frame_data  # Duy trì stream liên tục

        frame_id += 1
        # time.sleep(0.03)  # Giữ tốc độ gửi ổn định

# Đọc video đầu vào
video_path = './input_test/video/1.mp4'
cap = cv2.VideoCapture(video_path)

# Gửi toàn bộ frames trong một stream duy nhất
start_time = time.time()
responses = stub.StreamVideo(generate_frames(cap))  # Giữ stream liên tục

# Xử lý phản hồi từ server
for response in responses:
    print(f"🔹 Server đã xử lý xong Frame ID: {response.frame_id}")

# Tính tổng thời gian xử lý
total_time = time.time() - start_time
print(f"⏱️ Tổng thời gian gửi & xử lý: {total_time:.4f} giây")
cap.release()
