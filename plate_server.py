import grpc
import asyncio
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

class PlateDetectionServicer(streaming_pb2_grpc.PlateDetectionServiceServicer):
    def __init__(self, num_workers=2):
        self.model = YOLO("./weights/best.engine", task='detect')
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_image_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
        print("🔥 Warmup YOLO model...")
        _ = self.model(dummy_image_rgb)
        print("✅ Warmup hoàn tất.")
        
        self.queue = asyncio.Queue()
        self.send_queue = asyncio.Queue()
        
        self.ocr_channel = grpc.aio.insecure_channel("localhost:5003")
        self.ocr_stub = streaming_pb2_grpc.OCRServiceStub(self.ocr_channel)
        
        self.num_workers = num_workers
  
    async def DetectPlates(self, request_iterator, context):
        async for frame in request_iterator:
            await self.queue.put(frame)
            print(f"📥 Nhận Frame ID: {frame.frame_id} lúc {time.time()}")
            print(f"📥 Kích thước hàng đợi: {self.queue.qsize()}")
            yield streaming_pb2.Response(frame_id=frame.frame_id, status="OK")

    async def worker(self):
        while True:
            frame = await self.queue.get()
            
            start_time = time.time()
            
            # Giải mã hình ảnh
            image = np.frombuffer(frame.image_data, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Nhận diện phương tiện bằng YOLO
            results = await asyncio.to_thread(self.model, image_rgb, conf=0.5, iou=0.4, verbose=False)            
            boxes = results[0].boxes.data.tolist()
            
            detected_plates = []
            for obj in boxes:
                x1, y1, x2, y2, conf, class_id = obj
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bbox = streaming_pb2.BBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2
                )
                cropped_plate = image[y1:y2, x1:x2]
                _, cropped_encoded = cv2.imencode(".jpg", cropped_plate)
                cropped_bytes = cropped_encoded.tobytes()
                detected_plates.append(streaming_pb2.DetectedPlate(
                    bbox=bbox,
                    class_id=int(class_id),
                    confidence=round(conf, 3),
                    cropped_plate=cropped_bytes
                ))

            print(f"✅ Frame {frame.frame_id}: {len(detected_plates)} biển số - ⏱ {time.time() - start_time:.4f}s")

            plate_detection = streaming_pb2.PlateDetection(
                frame_id=frame.frame_id,
                plates=detected_plates
            )
            
            try:
                self.send_queue.put_nowait(plate_detection)
                print(f"📤 Đưa vào hàng đợi gửi đi frame ID {plate_detection.frame_id}")
            except asyncio.QueueFull:
                print(f"❌ Hàng đợi gửi đi đầy, bỏ qua frame ID {plate_detection.frame_id}")
            
            self.queue.task_done()

    async def ocr_sender(self):
        async def request_generator():
            while True:
                plate_detection = await self.send_queue.get()
                try:
                    print(f"📤 Gửi frame ID {plate_detection.frame_id} đến OCR service")
                    t0 = time.time()
                    yield plate_detection
                    elapsed = time.time() - t0
                    if elapsed > 0.1:
                        print(f"⚠️ Gửi frame ID {plate_detection.frame_id} chậm {elapsed:.3f} giây")
                except Exception as e:
                    print(f"❌ Lỗi gửi frame ID {plate_detection.frame_id}: {e}")
                finally:
                    self.send_queue.task_done()
                
        async def response_handler(call):
            try:
                async for response in call:
                    pass
            except grpc.aio.AioRpcError as e:
                print(f"❌ Lỗi từ OCR service: {e.code()} - {e.details()}")
        
        while True:
            try:
                print("🔁 Kết nối tới OCR service...")
                call = self.ocr_stub.RecognizePlate(request_generator())
                await response_handler(call)
            except grpc.aio.AioRpcError as e:
                print(f"❌ Lỗi kết nối tới OCR service: {e.code()} - {e.details()}")
            except Exception as e:
                print(f"❌ Lỗi trong OCR sender: {e}")
            print("🕒 Đợi 3 giây trước khi thử kết nối lại...")
            await asyncio.sleep(3)
    
    def start_workers(self):
        for _ in range(self.num_workers):
            asyncio.create_task(self.worker())

        asyncio.create_task(self.ocr_sender())
    
async def serve():
    server = grpc.aio.server()
    servicer = PlateDetectionServicer(num_workers=4)  # Tạo server với 4 worker
    servicer.start_workers()
    
    streaming_pb2_grpc.add_PlateDetectionServiceServicer_to_server(servicer, server)
    server.add_insecure_port("[::]:5002")

    await server.start()
    print("🚀 Async Vehicle Detection Server đang chạy trên cổng 5002...")
    try:
        await server.wait_for_termination()
    finally:
        await servicer.ocr_channel.close()

if __name__ == "__main__":
    asyncio.run(serve())
