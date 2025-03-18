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

def resize_and_pad(image, target_size=640):
    """Resize ảnh về (target_size, target_size) nhưng giữ nguyên tỉ lệ, thêm padding nếu cần."""
    orig_h, orig_w = image.shape[:2]

    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas, scale, pad_x, pad_y, orig_w, orig_h

class PlateDetectionServicer(streaming_pb2_grpc.PlateDetectionServiceServicer):
    def __init__(self):
        self.model = YOLO("../weights/best.pt").to(device)
        self.ocr_channel = grpc.insecure_channel("localhost:5003")
        self.ocr_stub = streaming_pb2_grpc.OCRServiceStub(self.ocr_channel)
        self.frame_count = 0  # Biến đếm số lượng frame đã nhận
        self.last_frame_time = time.time()  # Thời gian nhận frame cuối cùng
    
    def DetectPlates(self, request_iterator, context):
        for frame in request_iterator:
            # Tăng bộ đếm frame
            self.frame_count += 1
            # Kiểm tra thời gian từ frame cuối cùng
            current_time = time.time()
            if current_time - self.last_frame_time > 10:  # Nếu quá 10 giây
                print("⚠️ Không nhận được frame trong 10 giây, reset frame_count về 0.")
                self.frame_count = 1  # Reset frame_count
            
            # Cập nhật thời gian nhận frame cuối cùng
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
            print(f"⏱️ Thời gian YOLO xử lý biển số: {inference_time:.4f} giây")
            
            detected_plates = []
            for obj in results[0].boxes.data.tolist():
                x1, y1, x2, y2, conf, class_id = obj
                x1, x2 = int((x1 - pad_x) / scale), int((x2 - pad_x) / scale)
                y1, y2 = int((y1 - pad_y) / scale), int((y2 - pad_y) / scale)

                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)

                cropped_plate = image[y1:y2, x1:x2]
                _, cropped_encoded = cv2.imencode(".jpg", cropped_plate)
                cropped_bytes = cropped_encoded.tobytes()

                detected_plates.append(streaming_pb2.DetectedPlate(
                    bbox=[x1, y1, x2, y2],
                    class_id=int(class_id),
                    confidence=round(conf, 3),
                    cropped_plate=cropped_bytes
                ))
            
            plate_detection = streaming_pb2.PlateDetection(
                frame_id=frame.frame_id,
                plates=detected_plates,
                success=True
            )
            
            # Hiển thị frame_id, số lượng biển số phát hiện và tổng số frame đã nhận
            print(f"Frame ID: {plate_detection.frame_id}")
            print(f"Số lượng biển số phát hiện: {len(detected_plates)}")
            print(f"Tổng số frame đã nhận: {self.frame_count}")
            
            print(f"🔹 Gửi Frame ID {plate_detection.frame_id} đến OCR Server...")
            
            success = False
            while not success:
                try:
                    response_stream = self.ocr_stub.RecognizePlate(iter([plate_detection]))
                    for response in response_stream:
                        if response.success:
                            print(f"✅ Tracking Server xác nhận Frame ID {plate_detection.frame_id}")
                            success = True
                            break  # Thoát khỏi vòng lặp phản hồi
                    if not success:
                        print(f"⚠️ Chưa nhận được phản hồi thành công cho Frame ID {plate_detection.frame_id}, thử lại...")
                except grpc.RpcError as e:
                    print(f"⚠️ Lỗi gửi Frame ID {plate_detection.frame_id}, thử lại... ({e})")
                # time.sleep(0.1) 
            
            # self.ocr_stub.RecognizePlate(iter([plate_detection]))
            yield plate_detection

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    streaming_pb2_grpc.add_PlateDetectionServiceServicer_to_server(PlateDetectionServicer(), server)
    server.add_insecure_port("[::]:5002")
    server.start()
    print("🚀 gRPC Plate Detection Server đang chạy trên cổng 5002...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
