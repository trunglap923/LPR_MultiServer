import grpc
import asyncio
import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys
import time

sys.path.append("generated")
from generated import streaming_pb2, streaming_pb2_grpc

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Đang sử dụng: {device}")

class TrackingServicer(streaming_pb2_grpc.TrackingServiceServicer):
    def __init__(self, num_workers=2):
        self.tracker = DeepSort(
            max_age=10,
            embedder="mobilenet",
            embedder_gpu=(device == "cuda")
        )
        self.tracker_lock = asyncio.Lock()
        self.queue = asyncio.Queue()
        self.send_queue = asyncio.Queue()
        
        self.main_channel = grpc.aio.insecure_channel("localhost:5005")
        self.main_stub = streaming_pb2_grpc.AggregatedServiceStub(self.main_channel)
        
        self.num_workers = num_workers
        self.response_queue = {}
        self.workers_started = False
        
        
    
    async def stream_sender(self):
        async def request_generator():
            while True:
                result = await self.send_queue.get()
                try:
                    send_time = time.time()
                    print(f"📤 Gửi frame ID {result.frame_id} lúc {send_time:.2f}")
                    yield result
                except Exception as e:
                    print(f"❌ Lỗi khi gửi frame ID {result.frame_id}: {e}")
                finally:
                    self.send_queue.task_done()
                    
        async def response_handler(call):
            try:
                async for response in call:
                    pass
            except grpc.aio.AioRpcError as e:
                print(f"❌ Lỗi từ AggregatedService: {e.code()} - {e.details()}")
                
        while True:
            try:
                print("🔁 Kết nối tới AggregatedService...")
                call = self.main_stub.ProcessVehicles(request_generator())
                await response_handler(call)
            except grpc.aio.AioRpcError as e:
                print(f"❌ Mất kết nối tới AggregatedService: {e.code()} - {e.details()}")
            except Exception as e:
                print(f"❌ Lỗi không xác định khi gửi stream: {e}")
            print("🕒 Đợi 3 giây trước khi thử kết nối lại...")
            await asyncio.sleep(3)

    def start_workers(self):
        if not self.workers_started:
            for i in range(self.num_workers):
                asyncio.create_task(self.worker(i))
            self.workers_started = True
            
        asyncio.create_task(self.stream_sender())

    async def TrackObjects(self, request_iterator, context):    
        self.start_workers()

        async def receiver():
            print("🔄 Bắt đầu nhận frame từ VehicleServer...")
            async for vehicleDetection in request_iterator:
                response_queue = asyncio.Queue()
                self.response_queue[vehicleDetection.frame_id] = response_queue
                await self.queue.put((vehicleDetection, response_queue))

        receiver_task = asyncio.create_task(receiver())
        
        try:
            while True:
                for frame_id in list(self.response_queue.keys()):
                    q = self.response_queue[frame_id]
                    if not q.empty():
                        result = await q.get()
                        del self.response_queue[frame_id]
                        yield streaming_pb2.Response(frame_id=frame_id, status="OK")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            print("🔚 Client ngắt kết nối.")
        finally:
            receiver_task.cancel()

    async def worker(self, worker_id):
        while True:
            try:
                vehicleDetection, response_queue = await self.queue.get()
                frame_id = vehicleDetection.frame_id
                
                print(f"Thời điểm worker-{worker_id} nhận frame {frame_id}:", time.time())
                
                try:
                    image = np.frombuffer(vehicleDetection.image_data, np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("❌ Ảnh decode bị lỗi.")
                except Exception as e:
                    print(f"❌ Lỗi khi giải mã ảnh frame {frame_id}: {e}")
                    self.queue.task_done()
                    continue
                vehicles = vehicleDetection.vehicles
                
                formatted_vehicles = [
                    [[v.bbox.x1, v.bbox.y1, v.bbox.x2 - v.bbox.x1, v.bbox.y2 - v.bbox.y1], v.confidence, v.class_id]
                    for v in vehicles
                ]

                start_time = asyncio.get_event_loop().time()
                try:
                    await asyncio.wait_for(self.tracker_lock.acquire(), timeout=0.4)
                    try:
                        tracks = await asyncio.to_thread(self.tracker.update_tracks, formatted_vehicles, frame=image)
                    finally:
                        self.tracker_lock.release()
                except asyncio.TimeoutError:
                    print(f"⚠️ Worker-{worker_id} timeout khi chờ lock tracker")
                inference_time = asyncio.get_event_loop().time() - start_time

                tracked_objects = [
                    streaming_pb2.TrackedObject(
                        id=int(track.track_id),
                        bbox=streaming_pb2.BBox(
                            x1=int(track.to_ltrb()[0]),
                            y1=int(track.to_ltrb()[1]),
                            x2=int(track.to_ltrb()[2]),
                            y2=int(track.to_ltrb()[3])
                        ),
                        class_id=track.get_det_class(),
                        score=track.get_det_conf() or 0
                    )
                    for track in tracks if track.is_confirmed()
                ]
                # print(tracked_objects)
                result = streaming_pb2.TrackingResult(
                    frame_id=frame_id,
                    tracks=tracked_objects
                )
                await response_queue.put(result)
                await self.send_queue.put(result)
                self.queue.task_done()
                print(f"✅ Worker-{worker_id} hoàn tất tracking Frame {frame_id} | {len(tracked_objects)} object | ⏱ {inference_time:.3f}s")
                print(f"Thời điểm worker-{worker_id} hoàn tất:", time.time())
            except asyncio.QueueEmpty:
                print(f"⚠️ Worker-{worker_id} không có frame để xử lý.")
            except Exception as e:
                print(f"❌ Worker-{worker_id} gặp lỗi: {e}")
                await asyncio.sleep(1)  # Chờ 1 giây trước khi thử lại

async def serve():
    server = grpc.aio.server()
    servicer = TrackingServicer(num_workers=4)

    streaming_pb2_grpc.add_TrackingServiceServicer_to_server(servicer, server)
    server.add_insecure_port("[::]:5004")
    await server.start()
    print("🚀 Tracking Server đang chạy trên cổng 5004...")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
