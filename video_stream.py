import grpc
import cv2
import time
import threading
import queue
import numpy as np
import sys
import signal

# Thêm đường dẫn tới mã gRPC
sys.path.append("generated")
from generated import streaming_pb2, streaming_pb2_grpc

# Kết nối đến gateway_server
channel = grpc.insecure_channel("localhost:5000")
stub = streaming_pb2_grpc.VideoStreamingServiceStub(channel)

frame_queue = queue.Queue()
stats = {"sent": 0, "received": 0}
stop_event = threading.Event()

# 🟡 Địa chỉ stream từ VLC hoặc camera (có thể dùng rtsp, http, udp,...)
stream_url = "rtsp://localhost:8554/stream"  # Ví dụ RTSP từ VLC
cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

target_send_fps = 10
frame_id = 1

def signal_handler(sig, frame):
    print("\n🛑 Nhận tín hiệu dừng! Đang thoát...")
    stop_event.set()
    frame_queue.put(None)

signal.signal(signal.SIGINT, signal_handler)

def read_video_thread():
    """Đọc từ VLC stream theo nhịp FPS mục tiêu."""
    global frame_id
    while not stop_event.is_set() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Không đọc được frame từ VLC, đợi 0.5s rồi thử lại...")
            time.sleep(0.5)
            continue

        if frame_id % (30 // target_send_fps) == 0:  # Giảm tốc độ gửi nếu cần
            _, img_encoded = cv2.imencode(".jpg", frame)
            frame_data = streaming_pb2.Frame(
                frame_id=frame_id,
                image_data=img_encoded.tobytes()
            )
            frame_queue.put(frame_data)
            print(f"📤 Đưa vào hàng đợi Frame ID {frame_id}")
            stats["sent"] += 1

        frame_id += 1
        time.sleep(1.0 / 30)  # Giữ nhịp (giả định 30 FPS)

    frame_queue.put(None)

def send_stream():
    """Gửi frame qua gRPC Streaming."""
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        frame.timestamp = int(time.time() * 1000)
        print(f"Gửi Frame ID {frame.frame_id} tới server lúc {frame.timestamp}")
        yield frame

def receive_response_thread(responses):
    try:
        for response in responses:
            print(f"✅ Đã nhận phản hồi cho Frame ID: {response.vehicle_detection.frame_id}")
            stats["received"] += 1
    except grpc.RpcError as e:
        print(f"❌ Lỗi gRPC: {e}")

# Bắt đầu
video_thread = threading.Thread(target=read_video_thread)
video_thread.start()

start_time = time.time()
responses = stub.StreamVideo(send_stream())
receive_thread = threading.Thread(target=receive_response_thread, args=(responses,))
receive_thread.start()

# Đợi kết thúc
video_thread.join()
cap.release()

# # Hiệu suất
# total_time = time.time() - start_time
# fps_sent = stats["sent"] / total_time if total_time > 0 else 0

# print("\n🎯 **Tóm tắt hiệu suất truyền VLC stream**")
# print(f"🎥 Số frame đã gửi: {stats['sent']}")
# print(f"⏱️ Tổng thời gian gửi: {total_time:.2f} giây")
# print(f"⚡ Tốc độ gửi thực tế: {fps_sent:.2f} FPS")
