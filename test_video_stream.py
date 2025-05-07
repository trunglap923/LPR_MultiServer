import grpc
import cv2
import time
import threading
import queue
import numpy as np
import sys
import signal

# Thêm đường dẫn đến mã gRPC đã sinh
sys.path.append("generated")
from generated import streaming_pb2, streaming_pb2_grpc

vehicle_dict = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

channel = grpc.insecure_channel("localhost:5000")
stub = streaming_pb2_grpc.VideoStreamingServiceStub(channel)

frame_queue = queue.Queue()
stats = {"sent": 0, "received": 0}
stop_event = threading.Event()  # ⛔ Biến để dừng an toàn

# Đường dẫn video
video_path = "./input_test/video/1.mp4"
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_interval = 1.0 / frame_rate
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def signal_handler(sig, frame):
    print("\n🛑 Nhận tín hiệu dừng! Đang thoát...")
    stop_event.set()  # Bật cờ dừng
    frame_queue.put(None)  # Đảm bảo unblock nếu đang chờ lấy từ hàng đợi

# Đăng ký Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def read_video_thread():
    """Luồng đọc frame từ video và đưa vào queue theo FPS gốc."""
    frame_id = 1
    while not stop_event.is_set() and cap.isOpened():
        # start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        _, img_encoded = cv2.imencode(".jpg", frame)
        frame_data = streaming_pb2.Frame(
            frame_id=frame_id,
            image_data=img_encoded.tobytes()
        )
        frame_queue.put(frame_data)
        print(f"📤 Đã đưa vào hàng đợi Frame ID {frame_id}")
        frame_id += 1
        stats["sent"] += 1
        
        # user_input = input("⏸️ Nhấn Enter để gửi frame tiếp theo hoặc Ctrl+C để dừng: ")
        # if stop_event.is_set():
        #     break

        # elapsed = time.time() - start_time
        # time.sleep(max(0, frame_interval - elapsed))

    frame_queue.put(None)  # Gửi tín hiệu kết thúc

def send_stream():
    """Generator gửi frame từ queue sang server."""
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        yield frame

def receive_response_thread(responses):
    """Luồng nhận phản hồi từ server."""
    try:
        for response in responses:
            print(f"✅ Server đã xử lý xong Frame ID: {response.vehicle_detection.frame_id}")
            stats["received"] += 1
    except grpc.RpcError as e:
        print(f"❌ Lỗi gRPC: {e}")

# Khởi động
video_thread = threading.Thread(target=read_video_thread)
video_thread.start()

start_time = time.time()
responses = stub.StreamVideo(send_stream())
receive_thread = threading.Thread(target=receive_response_thread, args=(responses,))
receive_thread.start()

# Đợi luồng hoàn tất
video_thread.join()
receive_thread.join()
cap.release()

# Hiệu suất
total_time = time.time() - start_time
fps_sent = stats["sent"] / total_time if total_time > 0 else 0

print("\n🎯 **Tóm tắt hiệu suất gửi video**")
print(f"📽️ Video FPS gốc: {frame_rate:.2f} FPS")
print(f"🎥 Số frame đã gửi: {stats['sent']}/{frame_count}")
print(f"⏱️ Tổng thời gian gửi: {total_time:.2f} giây")
print(f"⚡ Tốc độ gửi thực tế: {fps_sent:.2f} FPS")
