import grpc
from concurrent import futures
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import sys
sys.path.append("generated")
from generated import streaming_pb2, streaming_pb2_grpc

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Đang sử dụng: {device}")

vehicle_dict = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

def resize_and_pad(image, target_size=640):
    orig_h, orig_w = image.shape[:2]
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Đảm bảo chiều mới không vượt quá target_size
    new_w = min(new_w, target_size)
    new_h = min(new_h, target_size)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2

    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y, orig_w, orig_h

class VehicleDetectionServicer(streaming_pb2_grpc.VehicleDetectionServiceServicer):
    def __init__(self):
        self.model = YOLO("../weights/yolov8n.pt").to(device)
        self.tracking_channel = grpc.insecure_channel("localhost:5004")
        self.tracking_stub = streaming_pb2_grpc.TrackingServiceStub(self.tracking_channel)
        self.frame_count = 0
        self.last_frame_time = time.time()

    def DetectVehicles(self, request_iterator, context):
        for frame in request_iterator:
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_frame_time > 10:
                print("⚠️ Không nhận được frame trong 10 giây, reset frame_count về 0.")
                self.frame_count = 1

            self.last_frame_time = current_time

            image = np.frombuffer(frame.image_data, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized, scale, pad_x, pad_y, orig_w, orig_h = resize_and_pad(image_rgb)
            image_resized = image_resized.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).unsqueeze(0).to(device)
            if device == "cuda":
                image_tensor = image_tensor.half()

            start_time = time.time()
            with torch.no_grad():
                results = self.model(image_tensor, verbose=False)
            inference_time = time.time() - start_time
            print(f"⏱️ Thời gian YOLO xử lý phương tiện: {inference_time:.4f} giây")

            detected_vehicles = []
            for obj in results[0].boxes.data.tolist():
                x1, y1, x2, y2, conf, class_id = obj
                if int(class_id) in vehicle_dict:
                    x1 = int((x1 - pad_x) / scale)
                    x2 = int((x2 - pad_x) / scale)
                    y1 = int((y1 - pad_y) / scale)
                    y2 = int((y2 - pad_y) / scale)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)
                    detected_vehicles.append(streaming_pb2.DetectedVehicle(
                        id=-1,
                        bbox=[x1, y1, x2, y2],
                        class_id=int(class_id),
                        confidence=round(conf, 3)
                    ))

            vehicle_detection = streaming_pb2.VehicleDetection(
                frame_id=frame.frame_id,
                image_data=frame.image_data,
                vehicles=detected_vehicles
            )

            print(f"🔹 Gửi Frame ID {vehicle_detection.frame_id} đến Tracking Server...")

            # self.tracking_stub.TrackObjects(iter([vehicle_detection]))
            
            success = False
            while not success:
                try:
                    response_stream = self.tracking_stub.TrackObjects(iter([vehicle_detection]))
                    for response in response_stream:
                        if response.success:
                            print(f"✅ Tracking Server xác nhận Frame ID {vehicle_detection.frame_id}")
                            success = True
                            break  # Thoát khỏi vòng lặp phản hồi
                    if not success:
                        print(f"⚠️ Chưa nhận được phản hồi thành công cho Frame ID {vehicle_detection.frame_id}, thử lại...")
                except grpc.RpcError as e:
                    print(f"⚠️ Lỗi gửi Frame ID {vehicle_detection.frame_id}, thử lại... ({e})")
                # time.sleep(0.1)

            yield vehicle_detection

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    streaming_pb2_grpc.add_VehicleDetectionServiceServicer_to_server(VehicleDetectionServicer(), server)
    server.add_insecure_port("[::]:5001")
    server.start()
    print("🚀 Vehicle Detection Server đang chạy trên cổng 5001...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
