import grpc
import asyncio
import sys
import time
from google.protobuf import empty_pb2
import cv2
import numpy as np
import collections

sys.path.append("generated")
from generated import streaming_pb2, streaming_pb2_grpc

def decode_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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
            buffer = {}
            next_display_id = 1
            max_wait_time = 0.15
            frame_timestamps = {}
            try:
                async for response in call:
                    frame_id = response.frame_id
                    buffer[frame_id] = response
                    frame_timestamps[frame_id] = time.time()
                    
                    while True:
                        if next_display_id in buffer:
                            response = buffer.pop(next_display_id)
                            frame_timestamps.pop(next_display_id, None)
                            next_display_id += 1
                    
                            image = decode_image(response.image_data)
                            image = draw_objects(image, response.objects)
                            
                            # 👇 Resize ảnh nếu quá to
                            screen_res = 1920, 1080  # Hoặc bạn có thể dùng pyautogui.size() để lấy độ phân giải thật
                            scale_width = screen_res[0] / image.shape[1]
                            scale_height = screen_res[1] / image.shape[0]
                            scale = min(scale_width, scale_height)
                            window_width = int(image.shape[1] * scale)
                            window_height = int(image.shape[0] * scale)

                            resized_image = cv2.resize(image, (window_width, window_height))
                            
                            cv2.imshow("📺 Kết quả sau xử lý", resized_image)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            print(f"✅ Hiển thị frame ID {response.frame_id} lúc {time.time():.2f}")
                        else:
                            if next_display_id not in frame_timestamps:
                                frame_timestamps[next_display_id] = time.time()
                                
                            elapsed_time = time.time() - frame_timestamps[next_display_id]
                            if elapsed_time > max_wait_time:
                                print(f"⚠️ Bỏ qua frame ID {next_display_id} do quá thời gian chờ")
                                frame_timestamps.pop(next_display_id, None)
                                next_display_id += 1
                                continue
                            break
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

if __name__ == "__main__":
    asyncio.run(serve())
