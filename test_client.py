import grpc
import cv2
import time
import numpy as np
import sys
sys.path.append("generated")
from generated import streaming_pb2, streaming_pb2_grpc

# Kết nối đến server
channel = grpc.insecure_channel("localhost:5002")
stub = streaming_pb2_grpc.PlateDetectionServiceStub(channel)

def send_frame(image):
    """Gửi một frame đến server và đo thời gian phản hồi."""
    _, img_encoded = cv2.imencode(".jpg", image)
    frame = streaming_pb2.Frame(frame_id=int(time.time()), image_data=img_encoded.tobytes())

    start_time = time.time()
    responses = stub.DetectPlates(iter([frame]))  # Đây là một stream
    # Duyệt qua phản hồi từ server
    for response in responses:
        print(f"Frame ID: {response.frame_id}")
        for plate in response.plates:
            print(f"  - Biển số phát hiện: {plate.bbox}, Conf: {plate.confidence}")
            
    inference_time = time.time() - start_time
    print(f"⏱️ Thời gian phản hồi xử lý biển số: {inference_time:.4f} giây")
    

video_path = './input_test/video/1.mp4'
# Mở video hoặc camera (0 là webcam)
cap = cv2.VideoCapture(video_path)  # Hoặc thay bằng đường dẫn video

start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    send_frame(frame)
inference_time = time.time() - start_time
print(f"⏱️ Tổng thời gian phản hồi xử lý biển số: {inference_time:.4f} giây")
cap.release()
