import asyncio
import grpc
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import sys

sys.path.append("generated")
from generated import streaming_pb2, streaming_pb2_grpc

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Đang sử dụng thiết bị: {device}")

vehicle_dict = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

class VehicleDetectionServicer(streaming_pb2_grpc.VehicleDetectionServiceServicer):
    def __init__(self, num_workers=2):
        self.model = YOLO("./weights/yolov8n.engine", task='detect')
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_image_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
        print("🔥 Warmup YOLO model...")
        _ = self.model(dummy_image_rgb)
        print("✅ Warmup hoàn tất.")
        
        self.queue = asyncio.Queue()
        self.send_queue = asyncio.Queue()

        self.tracking_channel = grpc.aio.insecure_channel("localhost:5004")
        self.tracking_stub = streaming_pb2_grpc.TrackingServiceStub(self.tracking_channel)

        self.num_workers = num_workers

    async def DetectVehicles(self, request_iterator, context):
        async for frame in request_iterator:
            await self.queue.put(frame)
            print(f"📥 Nhận Frame ID: {frame.frame_id} lúc {time.time()}")
            print(f"📥 Kích thước hàng đợi: {self.queue.qsize()}")
            yield streaming_pb2.Response(frame_id=frame.frame_id, status="OK")

    async def worker(self):
        while True:
            frame = await self.queue.get()
            
            start_time = time.time()

            # Decode ảnh từ bytes
            image = np.frombuffer(frame.image_data, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Nhận diện bằng YOLO
            results = await asyncio.to_thread(self.model, image_rgb, conf=0.4, iou=0.5, verbose=False)
            boxes = results[0].boxes.data.tolist()

            detected_vehicles = []
            for obj in boxes:
                x1, y1, x2, y2, conf, class_id = obj
                bbox = streaming_pb2.BBox(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2)
                )
                if int(class_id) in vehicle_dict:
                    detected_vehicles.append(
                        streaming_pb2.DetectedVehicle(
                            id=-1,
                            bbox=bbox,
                            class_id=int(class_id),
                            confidence=round(conf, 3)
                        )
                    )

            print(f"✅ Frame {frame.frame_id}: {len(detected_vehicles)} phương tiện - ⏱ {time.time() - start_time:.3f}s")

            vehicle_detection_result = streaming_pb2.VehicleDetection(
                frame_id=frame.frame_id,
                image_data=frame.image_data,
                vehicles=detected_vehicles,
            )

            # Đưa vào hàng đợi gửi gRPC, nếu đầy thì bỏ frame
            try:
                self.send_queue.put_nowait(vehicle_detection_result)
                print(f"📤 Đưa vào hàng đợi gửi đi Frame ID {vehicle_detection_result.frame_id} lúc {time.time()}")
            except asyncio.QueueFull:
                print(f"⚠️ Bỏ qua frame {frame.frame_id} do send_queue đầy")

            self.queue.task_done()

    async def track_sender(self):
        async def request_generator():
            while True:
                vehicle_detection_result = await self.send_queue.get()
                try:
                    print(f"📤 Gửi frame ID {vehicle_detection_result.frame_id} lúc {time.time()}")
                    t0 = time.time()
                    yield vehicle_detection_result
                    elapsed = time.time() - t0
                    if elapsed > 0.1:
                        print(f"⚠️ Gửi frame {vehicle_detection_result.frame_id} chậm: {elapsed:.3f}s")
                except Exception as e:
                    print(f"❌ Lỗi khi yield frame: {e}")
                finally:
                    self.send_queue.task_done()

        async def response_handler(call):
            try:
                async for response in call:
                    # Nếu bạn muốn xử lý kết quả từ tracking server
                    pass
            except grpc.aio.AioRpcError as e:
                print(f"❌ Lỗi phản hồi từ TrackingServer: {e.code()} - {e.details()}")

        while True:
            try:
                print("🔁 Kết nối tới TrackingServer...")
                call = self.tracking_stub.TrackObjects(request_generator())
                await response_handler(call)
            except grpc.aio.AioRpcError as e:
                print(f"❌ Mất kết nối tới TrackingServer: {e.code()} - {e.details()}")
            except Exception as e:
                print(f"❌ Lỗi không xác định khi gửi stream: {e}")
            print("🕒 Đợi 3 giây trước khi thử kết nối lại...")
            await asyncio.sleep(3)

    def start_workers(self):
        for _ in range(self.num_workers):
            asyncio.create_task(self.worker())

        asyncio.create_task(self.track_sender())

async def serve():
    server = grpc.aio.server()
    servicer = VehicleDetectionServicer(num_workers=4)
    servicer.start_workers()

    streaming_pb2_grpc.add_VehicleDetectionServiceServicer_to_server(servicer, server)
    server.add_insecure_port("[::]:5001")
    await server.start()
    print("🚀 VehicleDetectionService đang chạy tại cổng 5001...")

    try:
        await server.wait_for_termination()
    finally:
        await servicer.tracking_channel.close()

if __name__ == "__main__":
    asyncio.run(serve())
