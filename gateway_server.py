import grpc
import asyncio
import sys
import time
import websockets
import json
import base64
import cv2
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
import os

sys.path.append("generated")
from generated import streaming_pb2, streaming_pb2_grpc

def decode_image(image_data: bytes):
    if not image_data:
        raise ValueError("❌ Dữ liệu ảnh trống - image_data is empty.")
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("❌ Không giải mã được ảnh - imdecode failed.")
    return image

def draw_objects(image, objects):
    for obj in objects:
        x1, y1, x2, y2 = obj.vehicle_bbox.x1, obj.vehicle_bbox.y1, obj.vehicle_bbox.x2, obj.vehicle_bbox.y2
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        label = f"ID:{obj.tracking_id} - {obj.vehicle_class}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if obj.plate_bbox:
            px1, py1, px2, py2 = obj.plate_bbox.x1, obj.plate_bbox.y1, obj.plate_bbox.x2, obj.plate_bbox.y2
            px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
            cv2.rectangle(image, (px1, py1), (px2, py2), (255, 0, 0), 4)
            cv2.putText(image, obj.plate_number, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

class VideoStreamingServicer(streaming_pb2_grpc.VideoStreamingServiceServicer):
    def __init__(self):
        self.vehicle_queue = asyncio.Queue(maxsize=100)
        self.plate_queue = asyncio.Queue(maxsize=100)
        self.main_queue = asyncio.Queue(maxsize=100)

        self.vehicle_channel = grpc.aio.insecure_channel("localhost:5001")
        self.plate_channel = grpc.aio.insecure_channel("localhost:5002")
        self.main_channel = grpc.aio.insecure_channel("localhost:5005")

        self.vehicle_stub = streaming_pb2_grpc.VehicleDetectionServiceStub(self.vehicle_channel)
        self.plate_stub = streaming_pb2_grpc.PlateDetectionServiceStub(self.plate_channel)
        self.main_stub = streaming_pb2_grpc.AggregatedServiceStub(self.main_channel)

        self.vehicle_stream_task = asyncio.create_task(self.vehicle_stream_sender())
        self.plate_stream_task = asyncio.create_task(self.plate_stream_sender())
        self.main_stream_task = asyncio.create_task(self.main_stream_sender())
        
        self.websocket_uri = "ws://localhost:8000/ws/gateway"
        self.websocket_queue = asyncio.Queue(maxsize=300)
        self.websocket_sender_task = asyncio.create_task(self.websocket_sender())
        
        self.mongo_client = AsyncIOMotorClient("mongodb+srv://lap:12345@cluster0.89utstf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        self.collection = self.mongo_client["vehicle_tracking_db"]["detection_logs"]
        self.db_queue = asyncio.Queue(maxsize=500)
        self.db_writer_task = asyncio.create_task(self.db_writer())

    async def StreamVideo(self, request_iterator, context):
        async for frame in request_iterator:
            frame_id = frame.frame_id
            print(f"📥 Nhận frame ID: {frame_id} lúc {time.time():.2f}")

            await self.vehicle_queue.put(frame)
            await self.plate_queue.put(frame)
            await self.main_queue.put(frame)

        return streaming_pb2.Response(frame_id=frame_id, status="OK")

    async def vehicle_stream_sender(self):

        async def request_generator():
            while True:
                frame = await self.vehicle_queue.get()
                send_time = time.time()
                print(f"🚗 Gửi frame ID {frame.frame_id} lúc {send_time:.2f}")
                yield frame
                
        async def response_handler(call):
            try:
                async for response in call:
                    # Nếu bạn muốn xử lý kết quả từ VehicleDetection
                    pass
            except grpc.aio.AioRpcError as e:
                print(f"❌ Lỗi phản hồi từ VehicleDetection: {e.code()} - {e.details()}")
                
        while True:
            try:
                print("🔁 Kết nối tới VehicleDetection...")
                call = self.vehicle_stub.DetectVehicles(request_generator())
                await response_handler(call)
            except grpc.aio.AioRpcError as e:
                print(f"❌ Mất kết nối tới VehicleDetection: {e.code()} - {e.details()}")
            except Exception as e:
                print(f"❌ Lỗi không xác định khi gửi stream: {e}")
            print("🕒 Đợi 3 giây trước khi thử kết nối lại...")
            await asyncio.sleep(3)

    async def plate_stream_sender(self):
        
        async def request_generator():
            while True:
                frame = await self.plate_queue.get()
                send_time = time.time()
                print(f"🔳 Gửi frame ID {frame.frame_id} lúc {send_time:.2f}")
                yield frame
                
        async def response_handler(call):
            try:
                async for response in call:
                    # Nếu bạn muốn xử lý kết quả từ PlateDetection
                    pass
            except grpc.aio.AioRpcError as e:
                print(f"❌ Lỗi phản hồi từ PlateDetection: {e.code()} - {e.details()}")
                
        while True:
            try:
                print("🔁 Kết nối tới PlateDetection...")
                call = self.plate_stub.DetectPlates(request_generator())
                await response_handler(call)
            except grpc.aio.AioRpcError as e:
                print(f"❌ Mất kết nối tới PlateDetection: {e.code()} - {e.details()}")
            except Exception as e:
                print(f"❌ Lỗi không xác định khi gửi stream: {e}")
            print("🕒 Đợi 3 giây trước khi thử kết nối lại...")
            await asyncio.sleep(3)

    async def main_stream_sender(self):

        async def request_generator():
            while True:
                frame = await self.main_queue.get()
                if frame is None:
                    print("⚠️ Nhận được frame None trong main_queue.")
                    continue
                send_time = time.time()
                print(f"🔄 Gửi frame ID {frame.frame_id} đến MainServer lúc {send_time:.2f}")
                yield frame
            
        async def response_handler(call):
            last_sent_ts = 0  # Timestamp của frame cuối cùng đã gửi
            last_sent_id = 0  # ID của frame cuối cùng đã gửi
            try:
                async for response in call:
                    ts = response.timestamp
                    id = response.frame_id
                    print(f"📥 Nhận phản hồi từ MainServer - ID {id} lúc {time.time():.2f}")

                    # Lưu vào MongoDB
                    try:
                        self.db_queue.put_nowait(response)
                    except asyncio.QueueFull:
                        print(f"⚠️ Queue lưu MongoDB đầy, bỏ qua frame ID {id}.")
                    
                    if ts > last_sent_ts:
                        try:
                            await self.websocket_queue.put(response)
                            last_sent_ts = ts
                            last_sent_id = id
                        except asyncio.QueueFull:
                            print("⚠️ WebSocket queue đầy, không gửi được.")
                    else:
                        print(f"⏩ Bỏ qua frame ID {id} - timestamp {ts} vì nhỏ hơn hoặc bằng frame cuối đã gửi ({last_sent_id} - {last_sent_ts})")

            except grpc.aio.AioRpcError as e:
                print(f"❌ Lỗi phản hồi từ MainServer: {e.code()} - {e.details()}")
            except Exception as ex:
                print(f"❌ Lỗi xử lý ảnh: {ex}")
        
        while True:
            try:
                print("🔁 Kết nối tới MainServer...")
                call = self.main_stub.ProcessVideo(request_generator())
                await response_handler(call)
            except grpc.aio.AioRpcError as e:
                print(f"❌ Mất kết nối tới MainServer: {e.code()} - {e.details()}")
            except Exception as e:
                print(f"❌ Lỗi không xác định khi gửi stream: {e}")
            print("🕒 Đợi 3 giây trước khi thử kết nối lại...")
            await asyncio.sleep(3)

    def save_image_to_disk_cv2(self, image, frame_id, timestamp, output_dir="/tmp/saved_frames"):
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"frame_{frame_id}_{timestamp}.jpg")
        success = cv2.imwrite(filepath, image)
        if success:
            print(f"💾 Đã lưu frame ID {frame_id} tại {filepath}")
        else:
            print(f"❌ Lỗi khi lưu ảnh frame ID {frame_id}")

    def response_to_doc(self, response):
        try:
            image = decode_image(response.image_data)
            image = draw_objects(image, response.objects)
            self.save_image_to_disk_cv2(image, response.frame_id, response.timestamp)
        except Exception as e:
            print(f"❌ Lỗi xử lý ảnh lưu vào MongoDB: {e}")
        data = {
            "frame_id": response.frame_id,
            "image_link": f"frame_{response.frame_id}_{response.timestamp}.jpg",
            "timestamp": response.timestamp,
            "objects": [
                {
                    "tracking_id": obj.tracking_id,
                    "vehicle_class": obj.vehicle_class,
                    "vehicle_bbox": {
                        "x1": obj.vehicle_bbox.x1,
                        "y1": obj.vehicle_bbox.y1,
                        "x2": obj.vehicle_bbox.x2,
                        "y2": obj.vehicle_bbox.y2
                    },
                    "plate_bbox": {
                        "x1": obj.plate_bbox.x1,
                        "y1": obj.plate_bbox.y1,
                        "x2": obj.plate_bbox.x2,
                        "y2": obj.plate_bbox.y2
                    } if obj.plate_bbox else None,
                    "plate_number": obj.plate_number
                }
                for obj in response.objects
            ]
        }
        return data
    
    async def db_writer(self):
        while True:
            response = await self.db_queue.get()
            
            document = self.response_to_doc(response)
            try:
                await self.collection.insert_one(document)
                print(f"📝 Đã lưu frame ID {document['frame_id']} vào MongoDB.")
            except Exception as e:
                print(f"❌ Lỗi lưu MongoDB: {e}")
            finally:
                self.db_queue.task_done()
    
    async def send_via_websocket(self, response, websocket):
        try:
            image = decode_image(response.image_data)
            image = draw_objects(image, response.objects)
            _, img_encoded = cv2.imencode(".jpg", image)
            image_bytes = img_encoded.tobytes()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            print(f"❌ Lỗi mã hóa ảnh gửi đến WS_Server: {e}")
            image_base64 = None

        data = {
            "frame_id": response.frame_id,
            "image_data": image_base64,
            "timestamp": response.timestamp,
            "objects": [
                {
                    "tracking_id": obj.tracking_id,
                    "vehicle_class": obj.vehicle_class,
                    "vehicle_bbox": {
                        "x1": obj.vehicle_bbox.x1,
                        "y1": obj.vehicle_bbox.y1,
                        "x2": obj.vehicle_bbox.x2,
                        "y2": obj.vehicle_bbox.y2
                    },
                    "plate_bbox": {
                        "x1": obj.plate_bbox.x1,
                        "y1": obj.plate_bbox.y1,
                        "x2": obj.plate_bbox.x2,
                        "y2": obj.plate_bbox.y2
                    } if obj.plate_bbox else None,
                    "plate_number": obj.plate_number
                }
                for obj in response.objects
            ]
        }
        try:
            await websocket.send(json.dumps(data))
            print(f"📤 Gửi frame ID {response.frame_id} tới ws_server lúc {time.time():.2f}")
        except websockets.exceptions.ConnectionClosed:
            print("⚠️ WebSocket đã bị đóng. Không thể gửi dữ liệu.")
    
    async def websocket_sender(self):
        while True:
            try:
                print("🔗 Kết nối tới websocket server...")
                async with websockets.connect(self.websocket_uri) as websocket:
                    print("✅ Đã kết nối tới websocket server.")
                    while True:
                        response = await self.websocket_queue.get()
                        await self.send_via_websocket(response, websocket)

            except websockets.exceptions.ConnectionClosedError as e:
                print(f"❌ WebSocket bị đóng: {e.code} - {e.reason}")
                if e.code == 1012:
                    print("🔁 Server đang restart. Đợi trước khi reconnect...")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"❌ Lỗi WebSocketSender: {e}")
                await asyncio.sleep(3)

    

async def serve():
    server = grpc.aio.server()
    servicer = VideoStreamingServicer()
    streaming_pb2_grpc.add_VideoStreamingServiceServicer_to_server(
        servicer, server
    )
    server.add_insecure_port("[::]:5000")
    await server.start()
    print("🚀 VideoStreamingService đang chạy tại cổng 5000")
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        print("🛑 Đang tắt server...")
    
    await servicer.vehicle_channel.close()
    await servicer.plate_channel.close()
    await servicer.main_channel.close()
    print("🔌 Đã đóng tất cả các kênh kết nối.")

if __name__ == "__main__":
    asyncio.run(serve())
